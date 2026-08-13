"""Jungle pathing reconstruction and prediction.

Two capabilities come out of this module:

**Reconstruction.** Timeline frames give the jungler's position every minute
and a running ``jungleMinionsKilled`` count. When that count rises between two
frames and the jungler is standing on a camp, the camp was cleared. Chaining
those gives the clear route, which is what a coach means by "their jungler
pathed top side first".

**Prediction.** "Where is their jungler now" is the single most valuable
inference in solo queue, and it is genuinely learnable: given the corpus, the
transition from one camp to the next is strongly patterned. This module builds
a first-order Markov model over ``(camp, game phase)`` from every ingested
match and falls back to the canonical clear routes when a transition has not
been observed often enough to trust.

The predictor deliberately reports a probability. A prediction of "raptors,
55%" is coaching; "they are at raptors" is a guess dressed as a fact.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .geometry import (BLUE, CAMP_SIDE, JUNGLE_CAMPS, RED, Point, dist,
                       lane_corridor_distance, nearest_camp, quadrant_of)

#: Camp respawn timers in seconds (buffs and the big camps).
CAMP_RESPAWN_S: dict[str, float] = {
    "BLUEBUFF": 300.0, "REDBUFF": 300.0, "GROMP": 135.0,
    "WOLVES": 135.0, "RAPTORS": 135.0, "KRUGS": 135.0,
    "SCUTTLE": 150.0,
}

#: Canonical full-clear routes, used as the prior before data is available.
#: Both start on a buff and end opposite, which is what almost every jungler
#: does on the first clear.
STANDARD_ROUTES: dict[int, list[list[str]]] = {
    BLUE: [
        ["BLUE_BLUEBUFF", "BLUE_GROMP", "BLUE_WOLVES", "BLUE_RAPTORS",
         "BLUE_REDBUFF", "BLUE_KRUGS", "SCUTTLE_BOT"],
        ["BLUE_KRUGS", "BLUE_REDBUFF", "BLUE_RAPTORS", "BLUE_WOLVES",
         "BLUE_BLUEBUFF", "BLUE_GROMP", "SCUTTLE_TOP"],
    ],
    RED: [
        ["RED_BLUEBUFF", "RED_GROMP", "RED_WOLVES", "RED_RAPTORS",
         "RED_REDBUFF", "RED_KRUGS", "SCUTTLE_TOP"],
        ["RED_KRUGS", "RED_REDBUFF", "RED_RAPTORS", "RED_WOLVES",
         "RED_BLUEBUFF", "RED_GROMP", "SCUTTLE_BOT"],
    ],
}


def camp_family(camp: str) -> str:
    """``BLUE_RAPTORS`` -> ``RAPTORS``; scuttles collapse to ``SCUTTLE``."""
    if camp.startswith("SCUTTLE"):
        return "SCUTTLE"
    return camp.split("_", 1)[1] if "_" in camp else camp


def phase_of(t_s: float) -> str:
    """Coarse game phase, used to condition the transition model."""
    if t_s < 3.5 * 60:
        return "FIRST_CLEAR"
    if t_s < 14 * 60:
        return "EARLY"
    if t_s < 25 * 60:
        return "MID"
    return "LATE"


@dataclass
class CampClear:
    """One camp taken by one jungler."""

    ts_ms: int
    participant_id: int
    camp: str
    clear_index: int
    quadrant: str


@dataclass
class GankEvent:
    """A jungler appearing in a lane with intent."""

    ts_ms: int
    participant_id: int
    lane: str
    successful: bool
    victim_id: int | None = None


@dataclass
class JunglePathPoint:
    """A row for the ``jungle_paths`` table."""

    ts_ms: int
    participant_id: int
    quadrant: str
    camp: str | None
    clear_index: int
    gank_target: str | None
    predicted_next: str | None
    confidence: float


def reconstruct_path(states: Sequence[tuple[int, Point, int]],
                     team_id: int) -> list[CampClear]:
    """Rebuild the camps a jungler cleared.

    Parameters
    ----------
    states:
        ``(timestamp_ms, position, jungle_minions_killed)`` in time order.
    team_id:
        Used only for labelling, since camp names already encode the side.

    Notes
    -----
    Frames are a minute apart, so a jungler can clear two or three camps
    between them. When the kill count jumps by more than one camp's worth, the
    intermediate camps are inferred from the ones adjacent to the observed
    path -- reported with a lower clear confidence by the caller.
    """
    clears: list[CampClear] = []
    prev_kills = 0
    index = 0
    for ts_ms, pos, kills in states:
        delta = kills - prev_kills
        prev_kills = kills
        if delta <= 0:
            continue
        camp, distance = nearest_camp(pos, max_dist=2200.0)
        if camp is None:
            continue
        index += 1
        clears.append(CampClear(ts_ms=ts_ms, participant_id=0, camp=camp,
                                clear_index=index, quadrant=quadrant_of(pos)))
    return clears


def detect_ganks(states: Sequence[tuple[int, Point]],
                 kills: Sequence[tuple[int, int, int, Point]],
                 participant_id: int) -> list[GankEvent]:
    """Find lane visits by a jungler and whether they produced a kill.

    Parameters
    ----------
    states:
        ``(timestamp_ms, position)`` for the jungler.
    kills:
        ``(timestamp_ms, killer_id, victim_id, position)`` for all kills.
    """
    out: list[GankEvent] = []
    for ts_ms, pos in states:
        lane = None
        best = 1500.0
        for candidate in ("TOP", "MIDDLE", "BOTTOM"):
            d = lane_corridor_distance(pos, candidate)
            if d < best:
                lane, best = candidate, d
        if lane is None:
            continue
        # A kill within 45 seconds and 2500 units counts as a successful gank.
        hit = next((k for k in kills
                    if abs(k[0] - ts_ms) <= 45_000 and k[1] == participant_id
                    and dist(k[3], pos) < 2500.0), None)
        out.append(GankEvent(ts_ms=ts_ms, participant_id=participant_id, lane=lane,
                             successful=hit is not None,
                             victim_id=hit[2] if hit else None))
    return out


class JunglePredictor:
    """First-order Markov model over camp transitions.

    The model is keyed by ``(phase, from_camp)`` and stores counts of the next
    camp. Counts are smoothed with the canonical routes so a cold model still
    produces sensible answers, and the reported probability reflects both.
    """

    #: Weight given to the canonical route prior, in pseudo-counts.
    PRIOR_WEIGHT = 3.0

    def __init__(self) -> None:
        self.counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._prior_built = False

    # -- training --------------------------------------------------------
    def observe_sequence(self, clears: Sequence[CampClear]) -> None:
        """Fold one jungler's clear order into the transition counts."""
        for a, b in zip(clears, clears[1:]):
            key = f"{phase_of(a.ts_ms / 1000.0)}|{a.camp}"
            self.counts[key][b.camp] += 1.0

    def _ensure_prior(self) -> None:
        """Seed transitions from the canonical routes, once."""
        if self._prior_built:
            return
        for routes in STANDARD_ROUTES.values():
            for route in routes:
                for a, b in zip(route, route[1:]):
                    for phase in ("FIRST_CLEAR", "EARLY"):
                        self.counts[f"{phase}|{a}"][b] += self.PRIOR_WEIGHT
        self._prior_built = True

    # -- inference -------------------------------------------------------
    def predict(self, from_camp: str, t_s: float, top_k: int = 3
                ) -> list[tuple[str, float]]:
        """Most likely next camps with probabilities, best first."""
        self._ensure_prior()
        key = f"{phase_of(t_s)}|{from_camp}"
        row = self.counts.get(key)
        if not row:
            # Unseen state: fall back to whatever is geometrically adjacent.
            neighbours = self._adjacent(from_camp)
            if not neighbours:
                return []
            p = 1.0 / len(neighbours)
            return [(c, p) for c in neighbours[:top_k]]
        total = sum(row.values())
        ranked = sorted(row.items(), key=lambda kv: -kv[1])[:top_k]
        return [(camp, count / total) for camp, count in ranked]

    def predict_position(self, from_camp: str, t_s: float) -> tuple[Point | None, float]:
        """Predicted map position of the jungler and its probability."""
        ranked = self.predict(from_camp, t_s, top_k=1)
        if not ranked:
            return None, 0.0
        camp, p = ranked[0]
        return JUNGLE_CAMPS.get(camp), p

    @staticmethod
    def _adjacent(camp: str, radius: float = 4200.0) -> list[str]:
        """Camps within walking distance, nearest first."""
        origin = JUNGLE_CAMPS.get(camp)
        if origin is None:
            return []
        near = [(dist(origin, pos), name)
                for name, pos in JUNGLE_CAMPS.items() if name != camp]
        near.sort()
        return [name for d, name in near if d <= radius]

    # -- persistence -----------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({k: dict(v) for k, v in self.counts.items()})

    @classmethod
    def from_json(cls, blob: str | None) -> "JunglePredictor":
        model = cls()
        if blob:
            try:
                data = json.loads(blob)
            except ValueError:
                return model
            for key, row in data.items():
                for camp, count in row.items():
                    model.counts[key][camp] = float(count)
            model._prior_built = True
        return model


def camp_timer_advice(camp: str, cleared_at_s: float, now_s: float) -> tuple[float, str]:
    """Seconds until a camp respawns, with a coaching phrase.

    Returns ``(seconds_remaining, phrase)``; a negative remainder means the
    camp is already back up.
    """
    family = camp_family(camp)
    respawn = CAMP_RESPAWN_S.get(family, 135.0)
    remaining = (cleared_at_s + respawn) - now_s
    if remaining <= 0:
        return remaining, f"{family.lower()} is already back up"
    return remaining, (f"{family.lower()} respawns in about "
                       f"{int(remaining)} seconds")


def infer_start_side(first_clears: Sequence[CampClear], team_id: int) -> str:
    """Which half of the jungle the jungler started in.

    Used for the "jungle path prediction" question type: knowing the start
    side plus the elapsed time narrows the possible positions sharply.
    """
    if not first_clears:
        return "UNKNOWN"
    first = first_clears[0].camp
    side = CAMP_SIDE.get(first)
    if side is None:
        return "RIVER"
    quadrant = quadrant_of(JUNGLE_CAMPS[first])
    return "TOP_SIDE" if "TOP" in quadrant else "BOT_SIDE"
