"""Summoner's Rift geometry.

Every spatial feature in Stage 2 -- where a wave is, which camp a jungler just
cleared, whether a death happened in enemy territory -- reduces to arithmetic
on the ``(x, y)`` positions Riot puts in timeline frames and events. This
module holds that arithmetic in one place.

Coordinates
-----------
Riot's map spans roughly ``x, y in [0, 15000]`` with blue team's base at the
bottom-left and red team's at the top-right. The constants below are the
community-standard positions used by every replay tool; they are also
**self-correcting**: :func:`refine_turrets_from_events` re-derives turret
positions from observed ``BUILDING_KILL`` and ``TURRET_PLATE_DESTROYED``
events, so a map change or a wrong constant is fixed by the data itself once
enough matches are ingested.

Lane axis
---------
The key primitive is :meth:`LaneGeometry.project`, which maps a point onto the
line between the two outer turrets of a lane and returns a scalar in
``[-1, +1]``: ``-1`` at your own outer turret, ``0`` at the lane midpoint,
``+1`` at the enemy's. Wave state, lane priority and rotation detection are
all expressed in that one coordinate, which makes them symmetric between
sides for free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

MAP_MIN = 0.0
MAP_MAX = 15000.0

BLUE, RED = 100, 200
LANES = ("TOP", "MIDDLE", "BOTTOM")

Point = tuple[float, float]


def dist(a: Point, b: Point) -> float:
    """Euclidean distance in map units."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

#: Outer ("tier 1") turret positions, keyed by ``(team_id, lane)``.
OUTER_TURRETS: dict[tuple[int, str], Point] = {
    (BLUE, "TOP"): (981.0, 10441.0),
    (BLUE, "MIDDLE"): (5846.0, 6396.0),
    (BLUE, "BOTTOM"): (10504.0, 1029.0),
    (RED, "TOP"): (4318.0, 13875.0),
    (RED, "MIDDLE"): (8955.0, 8510.0),
    (RED, "BOTTOM"): (13866.0, 4505.0),
}

#: Inner ("tier 2") turrets, used to judge how deep a dive or crash went.
INNER_TURRETS: dict[tuple[int, str], Point] = {
    (BLUE, "TOP"): (1512.0, 6699.0),
    (BLUE, "MIDDLE"): (5048.0, 4812.0),
    (BLUE, "BOTTOM"): (6919.0, 1483.0),
    (RED, "TOP"): (7943.0, 13411.0),
    (RED, "MIDDLE"): (9767.0, 10113.0),
    (RED, "BOTTOM"): (13327.0, 8226.0),
}

NEXUS: dict[int, Point] = {BLUE: (1748.0, 1807.0), RED: (12611.0, 12665.0)}

#: Neutral objective pits.
DRAGON_PIT: Point = (9866.0, 4414.0)
BARON_PIT: Point = (4950.0, 10400.0)
#: Atakhan spawns in whichever half took more damage; the two candidate pits
#: are the two river entrances, so the midpoint is used for zoning purposes.
ATAKHAN_PIT: Point = (7500.0, 7500.0)

#: Jungle camps, keyed by the half of the map they belong to.
JUNGLE_CAMPS: dict[str, Point] = {
    "BLUE_BLUEBUFF": (3800.0, 7900.0),
    "BLUE_GROMP": (2100.0, 8450.0),
    "BLUE_WOLVES": (3700.0, 6500.0),
    "BLUE_RAPTORS": (7000.0, 5500.0),
    "BLUE_REDBUFF": (7800.0, 4000.0),
    "BLUE_KRUGS": (8400.0, 2800.0),
    "RED_BLUEBUFF": (11000.0, 6900.0),
    "RED_GROMP": (12600.0, 6300.0),
    "RED_WOLVES": (10950.0, 8400.0),
    "RED_RAPTORS": (7600.0, 9400.0),
    "RED_REDBUFF": (7000.0, 10800.0),
    "RED_KRUGS": (6300.0, 12000.0),
    "SCUTTLE_TOP": (4400.0, 9600.0),
    "SCUTTLE_BOT": (10500.0, 5200.0),
}

#: Which team's jungle a camp sits in (scuttles are neutral river camps).
CAMP_SIDE: dict[str, int | None] = {
    **{k: BLUE for k in JUNGLE_CAMPS if k.startswith("BLUE_")},
    **{k: RED for k in JUNGLE_CAMPS if k.startswith("RED_")},
    "SCUTTLE_TOP": None,
    "SCUTTLE_BOT": None,
}


def nearest_camp(pos: Point, max_dist: float = 1600.0) -> tuple[str | None, float]:
    """Closest jungle camp to ``pos``, or ``(None, inf)`` if nothing is close."""
    best: str | None = None
    best_d = float("inf")
    for name, cpos in JUNGLE_CAMPS.items():
        d = dist(pos, cpos)
        if d < best_d:
            best, best_d = name, d
    if best_d > max_dist:
        return None, best_d
    return best, best_d


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

def zone_of(pos: Point) -> str:
    """Coarse label for a position, used for rotations and death context.

    The map is split by the two diagonals: the main diagonal separates top
    side from bottom side, and the anti-diagonal separates blue's half from
    red's half. Lane corridors are detected first because they are narrow.
    """
    x, y = pos
    for lane in LANES:
        if lane_corridor_distance(pos, lane) < 1400.0:
            return f"{lane}_LANE"
    if dist(pos, DRAGON_PIT) < 2200.0:
        return "DRAGON_PIT"
    if dist(pos, BARON_PIT) < 2200.0:
        return "BARON_PIT"
    # River runs along the anti-diagonal x + y ~= 15000.
    if abs((x + y) - 15000.0) < 1500.0:
        return "RIVER"
    above_main = y > x
    blue_half = (x + y) < 15000.0
    if blue_half:
        return "BLUE_TOP_JUNGLE" if above_main else "BLUE_BOT_JUNGLE"
    return "RED_TOP_JUNGLE" if above_main else "RED_BOT_JUNGLE"


def quadrant_of(pos: Point) -> str:
    """Which quarter of the jungle a position falls in."""
    z = zone_of(pos)
    if z.endswith("_JUNGLE"):
        return z
    if z == "RIVER":
        return "RIVER"
    if z == "DRAGON_PIT":
        return "RED_BOT_JUNGLE" if pos[0] > 9000 else "BLUE_BOT_JUNGLE"
    if z == "BARON_PIT":
        return "BLUE_TOP_JUNGLE" if pos[0] < 6000 else "RED_TOP_JUNGLE"
    return z


#: A champion this close to a nexus is standing on the fountain.
FOUNTAIN_RADIUS = 2200.0


def at_fountain(pos: Point, team: int | None = None) -> bool:
    """True when a position is on a spawn platform.

    This matters more than it looks: both bases sit inside the top and bottom
    lane corridors, so without this check a player who has just recalled is
    counted as standing in lane and drags the wave estimate to the midpoint.
    Passing ``team`` restricts the test to that team's own fountain.
    """
    if team is not None:
        return dist(pos, NEXUS[team]) < FOUNTAIN_RADIUS
    return any(dist(pos, base) < FOUNTAIN_RADIUS for base in NEXUS.values())


def in_enemy_half(pos: Point, team: int) -> bool:
    """True when ``pos`` is on the far side of the river from ``team``."""
    blue_half = (pos[0] + pos[1]) < 15000.0
    return blue_half if team == RED else not blue_half


# ---------------------------------------------------------------------------
# Lane geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaneGeometry:
    """Projection helper for one lane, from one team's point of view."""

    lane: str
    team: int
    own_turret: Point
    enemy_turret: Point

    @property
    def midpoint(self) -> Point:
        return ((self.own_turret[0] + self.enemy_turret[0]) / 2.0,
                (self.own_turret[1] + self.enemy_turret[1]) / 2.0)

    @property
    def length(self) -> float:
        return dist(self.own_turret, self.enemy_turret)

    def project(self, pos: Point) -> float:
        """Position along the lane axis, ``-1`` own turret .. ``+1`` enemy turret.

        Values are clamped slightly beyond the turrets so that a dive under
        the enemy tower still reads as "past +1" rather than saturating
        exactly at the turret.
        """
        ax, ay = self.own_turret
        bx, by = self.enemy_turret
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom == 0:
            return 0.0
        t = ((pos[0] - ax) * dx + (pos[1] - ay) * dy) / denom
        return max(-1.4, min(1.4, (t * 2.0) - 1.0))

    def offset(self, pos: Point) -> float:
        """Perpendicular distance from the lane axis, in map units."""
        return lane_corridor_distance(pos, self.lane)

    def point_at(self, p: float) -> Point:
        """Inverse of :meth:`project`: the map position for a lane coordinate."""
        t = (p + 1.0) / 2.0
        return (self.own_turret[0] + (self.enemy_turret[0] - self.own_turret[0]) * t,
                self.own_turret[1] + (self.enemy_turret[1] - self.own_turret[1]) * t)


def lane_geometry(lane: str, team: int) -> LaneGeometry:
    """Build the projection helper for ``lane`` from ``team``'s perspective."""
    enemy = RED if team == BLUE else BLUE
    return LaneGeometry(lane=lane, team=team,
                        own_turret=OUTER_TURRETS[(team, lane)],
                        enemy_turret=OUTER_TURRETS[(enemy, lane)])


#: Waypoints describing each lane's actual path, which is an L-shape for the
#: side lanes rather than a straight line between turrets. Distance to a lane
#: is measured against these polylines.
LANE_PATHS: dict[str, tuple[Point, ...]] = {
    "TOP": ((1000.0, 1800.0), (700.0, 6000.0), (900.0, 11000.0),
            (2500.0, 13200.0), (7000.0, 13800.0), (11000.0, 13600.0),
            (13000.0, 13000.0)),
    "MIDDLE": ((1800.0, 1800.0), (4200.0, 4200.0), (7400.0, 7400.0),
               (10500.0, 10500.0), (13000.0, 13000.0)),
    "BOTTOM": ((1800.0, 1000.0), (6000.0, 700.0), (11000.0, 900.0),
               (13200.0, 2500.0), (13800.0, 7000.0), (13600.0, 11000.0),
               (13000.0, 13000.0)),
}


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Distance from ``p`` to segment ``ab``."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0:
        return dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom))
    return dist(p, (ax + t * dx, ay + t * dy))


def lane_corridor_distance(pos: Point, lane: str) -> float:
    """Shortest distance from ``pos`` to the lane's actual path."""
    path = LANE_PATHS[lane]
    return min(_point_segment_distance(pos, path[i], path[i + 1])
               for i in range(len(path) - 1))


def assign_lane(pos: Point, threshold: float = 1500.0) -> str | None:
    """Which lane a position is standing in, or ``None`` for jungle/river."""
    best, best_d = None, float("inf")
    for lane in LANES:
        d = lane_corridor_distance(pos, lane)
        if d < best_d:
            best, best_d = lane, d
    return best if best_d <= threshold else None


# ---------------------------------------------------------------------------
# Self-correction from observed events
# ---------------------------------------------------------------------------

def refine_turrets_from_events(events: Iterable[dict]) -> dict[tuple[int, str], Point]:
    """Re-derive outer turret positions from observed destruction events.

    ``BUILDING_KILL`` and ``TURRET_PLATE_DESTROYED`` events carry the exact
    position of the structure along with its lane and owning team, so a large
    enough corpus pins every turret precisely. The median is used rather than
    the mean because plate events occasionally record the attacker's position
    on some patches, and a median ignores those outliers.

    Returns only the turrets it could determine; callers merge the result over
    the defaults.
    """
    buckets: dict[tuple[int, str], list[Point]] = {}
    for ev in events:
        etype = ev.get("type")
        if etype not in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
            continue
        if etype == "BUILDING_KILL" and ev.get("towerType") != "OUTER_TURRET":
            continue
        lane = _normalise_lane(ev.get("laneType"))
        team = ev.get("teamId")
        pos = ev.get("position") or {}
        if lane is None or team not in (BLUE, RED):
            continue
        x, y = pos.get("x"), pos.get("y")
        if x is None or y is None:
            continue
        buckets.setdefault((team, lane), []).append((float(x), float(y)))

    refined: dict[tuple[int, str], Point] = {}
    for key, pts in buckets.items():
        if len(pts) < 5:
            continue
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        mid = len(pts) // 2
        refined[key] = (xs[mid], ys[mid])
    return refined


def _normalise_lane(lane_type: str | None) -> str | None:
    """Riot uses ``TOP_LANE``/``MID_LANE``/``BOT_LANE`` in events."""
    if not lane_type:
        return None
    mapping = {"TOP_LANE": "TOP", "MID_LANE": "MIDDLE", "BOT_LANE": "BOTTOM",
               "TOP": "TOP", "MIDDLE": "MIDDLE", "BOTTOM": "BOTTOM",
               "MID": "MIDDLE", "BOT": "BOTTOM"}
    return mapping.get(str(lane_type).upper())


def apply_refined_turrets(refined: dict[tuple[int, str], Point]) -> int:
    """Install refined turret positions as the module defaults.

    Returns the number of turrets updated. Called once per ingest run after
    the event table is populated.
    """
    count = 0
    for key, pos in refined.items():
        if key in OUTER_TURRETS and dist(OUTER_TURRETS[key], pos) > 200.0:
            OUTER_TURRETS[key] = pos
            count += 1
    return count
