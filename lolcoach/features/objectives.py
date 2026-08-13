"""Objective timing and control.

Covers the "objective timing" and "macro decisions" capabilities: when the
next dragon, herald, grub set, baron or atakhan is available, whether a team
had the map set up to take it, and whether the take was contested.

Timers
------
Spawn and respawn timings change between patches, so they live in one table
keyed by patch with a ``DEFAULT`` fallback, and :func:`observed_first_spawn`
can re-derive the first-spawn time empirically from ingested matches. That
matters because a coaching model that confidently states a stale baron timer
is worse than one that says it is unsure.

Control
-------
``setup_score`` summarises how well a team had prepared, in the 60 seconds
before an objective died: how many of its members were near the pit, how many
enemies were, and how much vision it had in the area. It is a number between
0 and 1 that makes "you took that dragon for free" and "you took that dragon
on a coinflip" separable in the training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .geometry import BARON_PIT, DRAGON_PIT, BLUE, RED, Point, dist

#: Objective names as they appear in Riot timeline events.
DRAGON = "DRAGON"
BARON = "BARON_NASHOR"
HERALD = "RIFTHERALD"
GRUBS = "HORDE"
ATAKHAN = "ATAKHAN"

#: Spawn and respawn schedule in seconds.
#:
#: ``first`` is when the objective first becomes available, ``respawn`` is the
#: delay after a kill, and ``despawn`` is when it stops spawning at all.
#: Values are per patch family; ``DEFAULT`` applies when the patch is unknown.
TIMERS: dict[str, dict[str, dict[str, float | None]]] = {
    "DEFAULT": {
        DRAGON:  {"first": 300.0,  "respawn": 300.0, "despawn": None},
        GRUBS:   {"first": 360.0,  "respawn": 240.0, "despawn": 840.0},
        HERALD:  {"first": 840.0,  "respawn": None,  "despawn": 1500.0},
        BARON:   {"first": 1500.0, "respawn": 360.0, "despawn": None},
        ATAKHAN: {"first": 1200.0, "respawn": None,  "despawn": None},
    },
}

#: Radius around a pit that counts as "at the objective".
PIT_RADIUS = 2600.0
#: Window before a kill used to judge setup.
SETUP_WINDOW_MS = 60_000


def pit_of(objective: str) -> Point:
    """Map position of an objective's pit."""
    if objective in (BARON, HERALD, GRUBS):
        return BARON_PIT
    return DRAGON_PIT


def timers_for(patch: str | None) -> dict[str, dict[str, float | None]]:
    """Timer table for a patch, falling back to the default set."""
    if patch and patch in TIMERS:
        return TIMERS[patch]
    return TIMERS["DEFAULT"]


def next_spawn_ms(objective: str, killed_at_ms: int, patch: str | None = None
                  ) -> int | None:
    """When this objective comes back after being killed at ``killed_at_ms``.

    Returns ``None`` for objectives that do not respawn (herald once taken,
    atakhan), which the dataset renders as "does not come back this game"
    rather than inventing a timer.
    """
    table = timers_for(patch).get(objective)
    if not table:
        return None
    respawn = table.get("respawn")
    if respawn is None:
        return None
    spawn = killed_at_ms + int(respawn * 1000)
    despawn = table.get("despawn")
    if despawn is not None and spawn > despawn * 1000:
        return None
    return spawn


def seconds_to_next_objective(now_ms: int, kills: Sequence[tuple[int, str]],
                              patch: str | None = None
                              ) -> tuple[float, str]:
    """Time until the soonest objective spawns, and which one it is.

    Parameters
    ----------
    kills:
        ``(timestamp_ms, objective)`` pairs already taken this game.
    """
    table = timers_for(patch)
    best_t = float("inf")
    best_name = ""
    taken: dict[str, int] = {}
    for ts, obj in kills:
        taken[obj] = max(taken.get(obj, 0), ts)

    for obj, spec in table.items():
        if obj in taken:
            nxt = next_spawn_ms(obj, taken[obj], patch)
            if nxt is None:
                continue
            delta = (nxt - now_ms) / 1000.0
        else:
            first = spec.get("first")
            if first is None:
                continue
            delta = first - now_ms / 1000.0
        if 0 <= delta < best_t:
            best_t, best_name = delta, obj
    if best_name == "":
        return float("inf"), ""
    return best_t, pretty_name(best_name)


def pretty_name(objective: str) -> str:
    """Readable objective name for prose."""
    return {
        DRAGON: "dragon", BARON: "baron", HERALD: "rift herald",
        GRUBS: "void grubs", ATAKHAN: "atakhan",
    }.get(objective, objective.lower().replace("_", " "))


@dataclass
class ObjectiveTake:
    """One elite monster kill with its context."""

    ts_ms: int
    objective: str
    subtype: str | None
    team_id: int
    contested: bool
    setup_score: float
    prior_kills: int
    vision_edge: float
    next_spawn_ms: int | None
    reasons: list[str] = field(default_factory=list)


def evaluate_take(ts_ms: int, objective: str, subtype: str | None, team_id: int,
                  positions_before: Mapping[int, list[Point]],
                  wards_before: Mapping[int, int],
                  kills_before: Sequence[tuple[int, int]],
                  patch: str | None = None) -> ObjectiveTake:
    """Score how well a team set up an objective they took.

    Parameters
    ----------
    positions_before:
        ``team_id -> champion positions`` from the frame closest to one minute
        before the kill.
    wards_before:
        ``team_id -> ward count placed near the pit`` in the same window.
    kills_before:
        ``(timestamp_ms, team_id)`` champion kills in the window, used to
        detect that the objective was taken off the back of a fight.
    """
    pit = pit_of(objective)
    enemy = RED if team_id == BLUE else BLUE

    own_near = sum(1 for p in positions_before.get(team_id, [])
                   if dist(p, pit) < PIT_RADIUS * 1.6)
    enemy_near = sum(1 for p in positions_before.get(enemy, [])
                     if dist(p, pit) < PIT_RADIUS * 1.6)

    own_wards = wards_before.get(team_id, 0)
    enemy_wards = wards_before.get(enemy, 0)
    vision_edge = float(own_wards - enemy_wards)

    prior_own = sum(1 for _, t in kills_before if t == team_id)
    prior_enemy = sum(1 for _, t in kills_before if t == enemy)

    # Setup is presence advantage, plus vision, plus recent kill advantage.
    presence = (own_near - enemy_near) / 5.0
    score = 0.5 + 0.28 * presence + 0.07 * max(-3.0, min(3.0, vision_edge)) \
        + 0.09 * max(-3, min(3, prior_own - prior_enemy))
    score = max(0.0, min(1.0, score))

    reasons: list[str] = []
    if own_near >= 3 and enemy_near <= 1:
        reasons.append(
            f"{own_near} of your team were at the pit against {enemy_near} of theirs, "
            f"so it was uncontested")
    elif enemy_near >= own_near:
        reasons.append(
            f"the enemy had {enemy_near} players nearby against your {own_near}, "
            f"so this was a contested take")
    if vision_edge >= 2:
        reasons.append("you had the vision advantage around the pit going into it")
    elif vision_edge <= -2:
        reasons.append("they had more vision than you around the pit")
    if prior_own - prior_enemy >= 2:
        reasons.append("you took it off the back of a won fight, which is the free way to get it")

    return ObjectiveTake(
        ts_ms=ts_ms, objective=objective, subtype=subtype, team_id=team_id,
        contested=enemy_near >= max(1, own_near - 1),
        setup_score=round(score, 3), prior_kills=prior_own + prior_enemy,
        vision_edge=vision_edge,
        next_spawn_ms=next_spawn_ms(objective, ts_ms, patch), reasons=reasons)


def observed_first_spawn(kills: Iterable[tuple[int, str]]) -> dict[str, float]:
    """Earliest observed kill time per objective, in seconds.

    A corpus-wide minimum is a lower bound on the spawn time and a good
    sanity check on the hard-coded table: if a patch moved dragon to 6:00,
    this shows up as no dragon ever dying before 6:00.
    """
    best: dict[str, float] = {}
    for ts_ms, obj in kills:
        t = ts_ms / 1000.0
        if obj not in best or t < best[obj]:
            best[obj] = t
    return best


def dragon_soul_progress(takes: Sequence[ObjectiveTake], team_id: int) -> tuple[int, str]:
    """Dragons taken by a team and how close they are to soul."""
    drakes = [t for t in takes if t.objective == DRAGON and t.team_id == team_id]
    n = len(drakes)
    if n >= 4:
        return n, "this team already has dragon soul"
    if n == 3:
        return n, "this team takes soul on the next dragon, so it must be contested"
    return n, f"this team has {n} dragon{'s' if n != 1 else ''}"
