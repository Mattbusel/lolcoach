"""Macro features: deaths, teamfights, rotations and positioning.

These are the features behind the "midgame rotations", "teamfight positioning"
and "every death" capabilities.

Deaths
------
Each ``CHAMPION_KILL`` is annotated with *why* it probably happened, by
checking the conditions a coach checks: was the victim outnumbered, were they
in enemy territory, did their team have vision there, were they at low health
already, and were they alone. The cause is a classification over those
conditions, not a guess -- and where the conditions are ambiguous the cause is
reported as ``UNCLEAR`` rather than fabricated.

Teamfights
----------
Kills are clustered in time and space; a cluster with at least three deaths or
three participants per side is a teamfight. The gold swing across the fight
plus who died first gives the outcome and a usable "what went wrong" signal.

Rotations
---------
Movement between zones across consecutive frames becomes a rotation when it
crosses a meaningful boundary. Each rotation is given a purpose by looking at
what happened next: an objective died, a fight started, a lane got shoved, or
the player went home.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .geometry import (BARON_PIT, BLUE, DRAGON_PIT, RED, Point, dist,
                       in_enemy_half, lane_corridor_distance, zone_of)

#: Champions within this range are "together" for outnumbering purposes.
GROUP_RADIUS = 2200.0
#: Kills within this window and radius belong to the same fight.
FIGHT_WINDOW_MS = 20_000
FIGHT_RADIUS = 4000.0
#: Wards within this range of a death count as covering it.
VISION_RADIUS = 1400.0
#: How long a ward is assumed to last for coverage purposes (control wards
#: are permanent but get cleared; this is a deliberate underestimate).
WARD_LIFETIME_MS = 150_000

# Death causes.
OVEREXTENDED = "OVEREXTENDED"
CAUGHT_ROTATING = "CAUGHT_ROTATING"
LOST_TRADE = "LOST_TRADE"
OUTNUMBERED = "OUTNUMBERED"
DIVED = "DIVED"
NO_VISION = "NO_VISION"
FIGHT_DEATH = "FIGHT_DEATH"
UNCLEAR = "UNCLEAR"


@dataclass
class DeathAnalysis:
    """One death with its context and probable cause."""

    ts_ms: int
    victim_id: int
    killer_id: int | None
    position: Point
    zone: str
    assisters: int
    outnumbered: int          # enemies minus allies nearby
    vision_nearby: bool
    hp_before: int
    cause: str
    cost_gold: int
    reasons: list[str] = field(default_factory=list)


def analyse_death(ts_ms: int, victim_id: int, killer_id: int | None,
                  position: Point, victim_team: int, assisters: int,
                  positions: Mapping[int, Point], teams: Mapping[int, int],
                  wards: Sequence[tuple[int, int, Point]],
                  hp_before: int, hp_max: int, bounty: int,
                  in_fight: bool) -> DeathAnalysis:
    """Explain one death.

    Parameters
    ----------
    positions:
        ``participant_id -> position`` at the nearest frame before the death.
    teams:
        ``participant_id -> team_id``.
    wards:
        ``(placed_ms, team_id, position)`` for wards placed before the death.
    """
    enemy_team = RED if victim_team == BLUE else BLUE
    allies = sum(1 for pid, p in positions.items()
                 if pid != victim_id and teams.get(pid) == victim_team
                 and dist(p, position) < GROUP_RADIUS)
    enemies = sum(1 for pid, p in positions.items()
                  if teams.get(pid) == enemy_team and dist(p, position) < GROUP_RADIUS)
    outnumbered = enemies - allies - 1

    covered = any(
        team == victim_team and (ts_ms - placed) < WARD_LIFETIME_MS
        and dist(wpos, position) < VISION_RADIUS
        for placed, team, wpos in wards)

    deep = in_enemy_half(position, victim_team)
    zone = zone_of(position)
    hp_pct = (hp_before / hp_max) if hp_max else 1.0

    reasons: list[str] = []
    if in_fight:
        cause = FIGHT_DEATH
        reasons.append("this death was part of a teamfight rather than a solo mistake")
    elif outnumbered >= 2:
        cause = OUTNUMBERED
        reasons.append(
            f"you were {enemies}v{allies + 1} here, so the fight was lost before it started")
    elif deep and not covered:
        cause = NO_VISION
        reasons.append(
            "no active friendly ward placement was recorded near this enemy-side "
            "position; timeline ward data is incomplete, so enemy vision cannot be inferred")
    elif deep:
        cause = OVEREXTENDED
        reasons.append("you were pushed past the river with no way out")
    elif zone.endswith("_LANE") and outnumbered <= 0 and hp_pct < 0.5:
        cause = LOST_TRADE
        reasons.append(
            f"you took the fight at {int(hp_pct * 100)}% health, which is why "
            f"the all-in went against you")
    elif zone.endswith("_LANE") and not deep:
        cause = DIVED
        reasons.append("they committed to a dive on your side of the lane")
    elif zone in ("RIVER", "DRAGON_PIT", "BARON_PIT") or "JUNGLE" in zone:
        cause = CAUGHT_ROTATING
        reasons.append(f"you were caught moving through {zone.replace('_', ' ').lower()}")
    else:
        cause = UNCLEAR
        reasons.append("there is not enough information in the timeline to be sure why")

    if not covered and cause != NO_VISION:
        reasons.append(
            "no active friendly ward placement was recorded near this spot "
            "(this does not establish enemy vision)")

    return DeathAnalysis(
        ts_ms=ts_ms, victim_id=victim_id, killer_id=killer_id, position=position,
        zone=zone, assisters=assisters, outnumbered=outnumbered,
        vision_nearby=covered, hp_before=hp_before, cause=cause,
        cost_gold=int(bounty or 0), reasons=reasons)


@dataclass
class Fight:
    """A clustered teamfight or skirmish."""

    fight_id: int
    start_ms: int
    end_ms: int
    zone: str
    team_a_count: int
    team_b_count: int
    kills_a: int
    kills_b: int
    winner_team: int | None
    gold_swing: int
    trigger: str
    positioning: dict[str, str] = field(default_factory=dict)


def cluster_fights(kills: Sequence[tuple[int, int, int, Point]],
                   teams: Mapping[int, int],
                   positions_at: Mapping[int, Mapping[int, Point]],
                   bounties: Mapping[int, int] | None = None) -> list[Fight]:
    """Group kills into fights.

    Parameters
    ----------
    kills:
        ``(timestamp_ms, killer_id, victim_id, position)`` in time order.
    positions_at:
        ``frame_timestamp_ms -> {participant_id: position}``, used to count how
        many champions were actually present.
    """
    fights: list[Fight] = []
    cluster: list[tuple[int, int, int, Point]] = []

    def flush() -> None:
        if not cluster:
            return
        start, end = cluster[0][0], cluster[-1][0]
        centre = (sum(k[3][0] for k in cluster) / len(cluster),
                  sum(k[3][1] for k in cluster) / len(cluster))
        kills_by_team: dict[int, int] = defaultdict(int)
        for _, killer, victim, _pos in cluster:
            victim_team = teams.get(victim)
            killer_team = teams.get(killer) if killer else None
            if killer_team is None and victim_team is not None:
                killer_team = RED if victim_team == BLUE else BLUE
            if killer_team is not None:
                kills_by_team[killer_team] += 1

        frame = _nearest_frame(positions_at, start)
        present = positions_at.get(frame, {}) if frame is not None else {}
        a_count = sum(1 for pid, p in present.items()
                      if teams.get(pid) == BLUE and dist(p, centre) < FIGHT_RADIUS)
        b_count = sum(1 for pid, p in present.items()
                      if teams.get(pid) == RED and dist(p, centre) < FIGHT_RADIUS)

        ka, kb = kills_by_team.get(BLUE, 0), kills_by_team.get(RED, 0)
        winner = BLUE if ka > kb else (RED if kb > ka else None)
        swing = 0
        if bounties:
            for _, _killer, victim, _pos in cluster:
                swing += bounties.get(victim, 300)
        fights.append(Fight(
            fight_id=len(fights) + 1, start_ms=start, end_ms=end,
            zone=zone_of(centre), team_a_count=a_count, team_b_count=b_count,
            kills_a=ka, kills_b=kb, winner_team=winner, gold_swing=swing,
            trigger=_infer_trigger(centre, start)))
        cluster.clear()

    for k in kills:
        if not cluster:
            cluster.append(k)
            continue
        near_time = (k[0] - cluster[-1][0]) <= FIGHT_WINDOW_MS
        near_place = dist(k[3], cluster[-1][3]) <= FIGHT_RADIUS
        if near_time and near_place:
            cluster.append(k)
        else:
            flush()
            cluster.append(k)
    flush()
    return [f for f in fights if (f.kills_a + f.kills_b) >= 2]


def _nearest_frame(positions_at: Mapping[int, Mapping[int, Point]],
                   ts_ms: int) -> int | None:
    """Frame timestamp closest to but not after ``ts_ms``."""
    best = None
    for frame_ts in positions_at:
        if frame_ts <= ts_ms and (best is None or frame_ts > best):
            best = frame_ts
    return best


def _infer_trigger(centre: Point, ts_ms: int) -> str:
    """What the fight was probably over."""
    if dist(centre, DRAGON_PIT) < 3000:
        return "DRAGON"
    if dist(centre, BARON_PIT) < 3000:
        return "BARON_OR_HERALD"
    zone = zone_of(centre)
    if zone.endswith("_LANE"):
        return "LANE_SKIRMISH"
    if zone == "RIVER":
        return "RIVER_SKIRMISH"
    return "JUNGLE_SKIRMISH"


def positioning_notes(fight: Fight, positions: Mapping[int, Point],
                      teams: Mapping[int, int], roles: Mapping[int, str],
                      deaths: Sequence[int]) -> dict[str, str]:
    """Per-player positioning assessment for one fight.

    The check that generalises across champions is *distance from your own
    team's centre of mass relative to your role*: carries that are further
    forward than their front line died for a reason, and front-liners standing
    behind their carries never engaged.
    """
    notes: dict[str, str] = {}
    centres: dict[int, Point] = {}
    for team in (BLUE, RED):
        members = [p for pid, p in positions.items() if teams.get(pid) == team]
        if members:
            centres[team] = (sum(m[0] for m in members) / len(members),
                             sum(m[1] for m in members) / len(members))

    enemy_centre = {BLUE: centres.get(RED), RED: centres.get(BLUE)}
    for pid, pos in positions.items():
        team = teams.get(pid)
        if team is None or enemy_centre.get(team) is None:
            continue
        role = (roles.get(pid) or "").upper()
        d_enemy = dist(pos, enemy_centre[team])
        own_centre = centres.get(team)
        d_own = dist(pos, own_centre) if own_centre else 0.0

        if role in ("BOTTOM", "MIDDLE", "UTILITY"):
            if d_enemy < 1200:
                notes[str(pid)] = ("stood inside the enemy team as a backline "
                                   "champion, which is why they were focused")
            elif d_own > 2600:
                notes[str(pid)] = "was separated from their team when the fight started"
            elif pid in deaths:
                notes[str(pid)] = "died despite reasonable spacing, so this was a focus issue"
            else:
                notes[str(pid)] = "held usable spacing behind the front line"
        elif role in ("TOP", "JUNGLE"):
            if d_enemy > 3000:
                notes[str(pid)] = ("never reached the enemy team, so their front "
                                   "line contributed nothing to this fight")
            else:
                notes[str(pid)] = "engaged from the front, which is correct for this role"
    return notes


@dataclass
class Rotation:
    """One player's movement between zones."""

    ts_ms: int
    participant_id: int
    from_zone: str
    to_zone: str
    purpose: str
    tempo_gain: float
    was_correct: bool | None


def detect_rotations(states: Sequence[tuple[int, Point]], participant_id: int,
                     objective_kills: Sequence[tuple[int, str, int]],
                     fights: Sequence[Fight], team_id: int) -> list[Rotation]:
    """Turn a movement trace into labelled rotations.

    A rotation is scored by what followed it within 90 seconds: an objective
    the player's team took, a fight their team won, or nothing at all. That
    "what happened next" framing is exactly how a coach reviews a rotation.
    """
    out: list[Rotation] = []
    for (t0, p0), (t1, p1) in zip(states, states[1:]):
        z0, z1 = zone_of(p0), zone_of(p1)
        if z0 == z1:
            continue

        purpose = "SIDE_LANE"
        tempo = 0.0
        correct: bool | None = None

        obj = next((o for o in objective_kills if 0 <= o[0] - t1 <= 90_000), None)
        fight = next((f for f in fights if 0 <= f.start_ms - t1 <= 90_000), None)

        if z1 in ("DRAGON_PIT", "BARON_PIT", "RIVER"):
            purpose = "OBJECTIVE_SETUP"
        elif z1.endswith("_LANE") and z0.endswith("_LANE"):
            purpose = "ROAM"
        elif "JUNGLE" in z1:
            purpose = "COLLAPSE" if fight else "SIDE_LANE"

        if obj is not None:
            tempo += 1.0 if obj[2] == team_id else -0.5
            correct = obj[2] == team_id
            purpose = "OBJECTIVE_SETUP"
        if fight is not None:
            won = fight.winner_team == team_id
            tempo += 0.8 if won else -0.8
            correct = won if correct is None else (correct and won)

        out.append(Rotation(ts_ms=t1, participant_id=participant_id,
                            from_zone=z0, to_zone=z1, purpose=purpose,
                            tempo_gain=round(tempo, 2), was_correct=correct))
    return out
