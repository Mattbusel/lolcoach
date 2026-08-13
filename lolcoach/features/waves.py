"""Minion wave modelling and wave-state inference.

This module answers the questions the brief leads with -- *should I crash or
freeze here*, *is this wave slow pushing* -- from Riot match timelines.

What the data actually gives us
-------------------------------
A timeline has one participant frame per **60 seconds**: position, gold, xp,
level, cs, and champion stats. It does not contain minions. So wave state
cannot be read off; it has to be inferred. Being honest about that is the
difference between a model that coaches and a model that bluffs, which is why
every row this module produces carries a ``confidence`` and why the dataset
stage drops low-confidence rows instead of turning them into confident advice.

The inference combines three independent signals:

1. **Where the laners are standing.** Players last-hit from just behind their
   own minions, so the midpoint between the two laners is a good estimate of
   where the waves are colliding. Projected onto the lane axis this gives a
   number in ``[-1, +1]`` (own turret to enemy turret).
2. **How the estimate is moving.** The derivative of that position across
   frames separates a freeze (flat, on your half) from a slow push (drifting
   out) from a crash (moving fast into the enemy turret).
3. **Minion accounting.** The spawn schedule is deterministic, so the number
   of minions that *should* exist by time ``t`` is known exactly. Subtracting
   what both laners actually killed leaves the minions that died to turrets,
   were taken by someone else, or are **still alive** -- which is what a big
   slow-pushing wave looks like numerically.

Where the signals disagree, confidence falls. Where a lane is empty, the
estimate decays toward the previous value rather than inventing one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .geometry import (BLUE, RED, LaneGeometry, Point, at_fountain,
                       lane_corridor_distance, lane_geometry)

# ---------------------------------------------------------------------------
# Spawn schedule
# ---------------------------------------------------------------------------

#: The first wave leaves the spawn platform at 1:05.
FIRST_SPAWN_S = 65.0
#: Subsequent waves every 30 seconds.
SPAWN_INTERVAL_S = 30.0
#: Melee and caster minions per wave (before super minions).
MELEE_PER_WAVE = 3
CASTER_PER_WAVE = 3
#: Cannon cadence changes twice during a game.
CANNON_EVERY_3_UNTIL_S = 15 * 60.0
CANNON_EVERY_2_UNTIL_S = 25 * 60.0
#: Rough travel time from spawn to the lane meeting point.
TRAVEL_S = {"MIDDLE": 22.0, "TOP": 38.0, "BOTTOM": 38.0}


def wave_index_at(t_s: float) -> int:
    """Number of waves that have spawned by ``t_s`` seconds."""
    if t_s < FIRST_SPAWN_S:
        return 0
    return int((t_s - FIRST_SPAWN_S) // SPAWN_INTERVAL_S) + 1


def wave_spawn_time(index: int) -> float:
    """Spawn time of the ``index``-th wave (1-based)."""
    return FIRST_SPAWN_S + SPAWN_INTERVAL_S * (index - 1)


def wave_has_cannon(index: int) -> bool:
    """Whether wave ``index`` contains a siege (cannon) minion.

    Cadence: every third wave until 15:00, every second wave until 25:00,
    then every wave. The cadence is evaluated at the wave's spawn time, which
    is what the game does -- a wave that spawns at 14:55 follows the early
    rule even though it arrives after 15:00.
    """
    if index < 1:
        return False
    t = wave_spawn_time(index)
    if t < CANNON_EVERY_3_UNTIL_S:
        return index % 3 == 0
    if t < CANNON_EVERY_2_UNTIL_S:
        return index % 2 == 0
    return True


def minions_spawned_by(t_s: float) -> int:
    """Total minions sent down **one** lane by time ``t_s``."""
    n = wave_index_at(t_s)
    if n <= 0:
        return 0
    base = n * (MELEE_PER_WAVE + CASTER_PER_WAVE)
    cannons = sum(1 for i in range(1, n + 1) if wave_has_cannon(i))
    return base + cannons


def current_wave_in_lane(t_s: float, lane: str) -> tuple[int, bool]:
    """Index of the wave meeting in ``lane`` at ``t_s``, and whether it has a cannon.

    A wave is "in lane" once it has had time to walk there, so the arrival
    time is the spawn time plus the lane's travel time.
    """
    travel = TRAVEL_S.get(lane, 30.0)
    arrived = wave_index_at(t_s - travel)
    if arrived <= 0:
        return 0, False
    return arrived, wave_has_cannon(arrived)


def expected_cs_by(t_s: float, lane: str, solo: bool = True) -> float:
    """CS a lane's minions offer by ``t_s``, per player.

    In a solo lane one player can take everything; in the bot lane the marksman
    takes the large majority, so the support's share is modelled separately by
    callers. This is the denominator for "is this player farming normally".
    """
    total = float(minions_spawned_by(t_s))
    return total if solo else total * 0.85


# ---------------------------------------------------------------------------
# Wave state
# ---------------------------------------------------------------------------

#: Canonical wave states. These are the labels the model is trained to use.
FREEZE = "FREEZE"
SLOW_PUSH = "SLOW_PUSH"
BIG_WAVE = "BIG_WAVE"
CRASHING = "CRASHING"
BOUNCING = "BOUNCING"
NEUTRAL = "NEUTRAL"

#: Recommended actions.
CRASH = "CRASH"
HOLD_FREEZE = "FREEZE"
BUILD_SLOW_PUSH = "SLOW_PUSH"
SHOVE_AND_ROAM = "SHOVE_AND_ROAM"
RESET = "RESET"


@dataclass
class WaveObservation:
    """Everything known about one lane at one timestamp, one team's view."""

    ts_ms: int
    lane: str
    team: int
    position: float          # -1 own turret .. +1 enemy turret
    velocity: float          # change per minute
    state: str
    size_diff: float         # estimated surplus minions in the wave
    cannon: bool
    confidence: float
    #: Free-form factors that justified the state, rendered into prose later.
    factors: list[str] = field(default_factory=list)
    recommended: str = NEUTRAL
    reasons: list[str] = field(default_factory=list)


@dataclass
class LaneFrameInput:
    """Per-frame inputs the estimator needs for one lane."""

    ts_ms: int
    own_positions: list[Point]
    enemy_positions: list[Point]
    own_cs: int
    enemy_cs: int
    #: Turret plates taken in this lane by each side, cumulative.
    own_plates: int = 0
    enemy_plates: int = 0
    #: Whether each side's outer turret in this lane still stands.
    own_turret_alive: bool = True
    enemy_turret_alive: bool = True


class WaveEstimator:
    """Sequential estimator of wave position and state for one lane and team.

    Feed it frames in chronological order with :meth:`update`. It keeps the
    previous estimate so it can compute velocity and decay gracefully through
    frames where nobody is in the lane.
    """

    #: How far in front of a player their minions typically are.
    MINION_LEAD = 0.12
    #: Below this offset from the lane path a champion counts as "in lane".
    IN_LANE_OFFSET = 1800.0

    def __init__(self, lane: str, team: int) -> None:
        self.lane = lane
        self.team = team
        self.geo: LaneGeometry = lane_geometry(lane, team)
        self._prev: WaveObservation | None = None
        self._flat_frames = 0
        self._peak_position = -1.0

    # -- main entry point ------------------------------------------------
    def update(self, f: LaneFrameInput) -> WaveObservation:
        """Fold one frame into the estimate and return the new observation."""
        t_s = f.ts_ms / 1000.0
        # Both nexuses sit inside the top and bottom lane corridors, so a
        # player who has just recalled would otherwise be counted as being in
        # lane and would drag the estimate toward the midpoint.
        own_in = [p for p in f.own_positions
                  if lane_corridor_distance(p, self.lane) < self.IN_LANE_OFFSET
                  and not at_fountain(p)]
        enemy_in = [p for p in f.enemy_positions
                    if lane_corridor_distance(p, self.lane) < self.IN_LANE_OFFSET
                    and not at_fountain(p)]

        position, confidence, factors = self._estimate_position(own_in, enemy_in)

        prev = self._prev
        if prev is None:
            velocity = 0.0
        else:
            dt_min = max(1e-6, (f.ts_ms - prev.ts_ms) / 60000.0)
            velocity = (position - prev.position) / dt_min

        size_diff, size_conf, size_factor = self._estimate_size(t_s, f)
        if size_factor:
            factors.append(size_factor)
        confidence = min(1.0, confidence * 0.75 + size_conf * 0.25)

        _, cannon = current_wave_in_lane(t_s, self.lane)
        state = self._classify(position, velocity, size_diff, f)

        obs = WaveObservation(
            ts_ms=f.ts_ms, lane=self.lane, team=self.team, position=position,
            velocity=velocity, state=state, size_diff=size_diff, cannon=cannon,
            confidence=round(confidence, 3), factors=factors)
        self._prev = obs
        self._peak_position = max(self._peak_position, position)
        return obs

    # -- position --------------------------------------------------------
    def _estimate_position(self, own_in: list[Point], enemy_in: list[Point]
                           ) -> tuple[float, float, list[str]]:
        """Blend champion positions into a wave-position estimate."""
        factors: list[str] = []
        own_p = [self.geo.project(p) for p in own_in]
        enemy_p = [self.geo.project(p) for p in enemy_in]

        if own_p and enemy_p:
            # Both laners present: the minions are colliding between them.
            position = (max(own_p) + min(enemy_p)) / 2.0
            confidence = 0.78
            factors.append("both laners in lane")
        elif own_p:
            position = min(1.3, max(own_p) + self.MINION_LEAD)
            confidence = 0.55
            factors.append("only your laner is in lane")
        elif enemy_p:
            position = max(-1.3, min(enemy_p) - self.MINION_LEAD)
            confidence = 0.50
            factors.append("only the enemy laner is in lane")
        elif self._prev is not None:
            # Nobody home: waves keep walking toward whoever left last, so the
            # estimate drifts along its current velocity but loses confidence.
            drift = 0.35 if self._prev.velocity >= 0 else -0.35
            position = max(-1.2, min(1.2, self._prev.position + drift))
            confidence = max(0.2, self._prev.confidence * 0.55)
            factors.append("lane is empty, position extrapolated")
        else:
            position, confidence = 0.0, 0.2
            factors.append("no information yet")
        return position, confidence, factors

    # -- size ------------------------------------------------------------
    def _estimate_size(self, t_s: float, f: LaneFrameInput
                       ) -> tuple[float, float, str]:
        """Estimate surplus minions from the spawn-versus-killed difference.

        ``spawned - killed`` counts minions that died to a turret, were taken
        by a third party, or are still walking around. Once the lane has been
        running for a few minutes the last of those dominates, which is
        exactly the quantity that distinguishes a slow push from a reset lane.
        """
        spawned = minions_spawned_by(t_s)
        if spawned <= 0:
            return 0.0, 0.3, ""
        killed = max(0, f.own_cs) + max(0, f.enemy_cs)
        surplus = float(spawned - killed)

        # Turret-killed minions are not surplus; each destroyed plate implies
        # a wave crashed and died to the tower.
        plate_absorbed = (f.own_plates + f.enemy_plates) * 3.0
        surplus = max(0.0, surplus - plate_absorbed)

        # Early game the numbers are small and noisy; late game jungle camps
        # and roaming laners pollute the CS counts badly.
        if t_s < 180:
            conf = 0.35
        elif t_s < 900:
            conf = 0.7
        else:
            conf = 0.4
        factor = ""
        if surplus >= 8:
            factor = f"about {int(surplus)} minions unaccounted for, so the wave is large"
        elif surplus <= 2:
            factor = "minion counts are even, so no wave has built up"
        return surplus, conf, factor

    # -- classification ---------------------------------------------------
    def _classify(self, position: float, velocity: float, size_diff: float,
                  f: LaneFrameInput) -> str:
        """Map the numeric estimate onto a named wave state."""
        # Track how long the wave has been stationary for freeze detection.
        if abs(velocity) < 0.08:
            self._flat_frames += 1
        else:
            self._flat_frames = 0

        if position > 0.55 and velocity > 0.15:
            return CRASHING
        if not f.enemy_turret_alive and position > 0.5:
            return CRASHING
        if self._peak_position > 0.5 and velocity < -0.2:
            return BOUNCING
        if self._flat_frames >= 2 and -0.85 < position < -0.1:
            return FREEZE
        if size_diff >= 9:
            return BIG_WAVE
        if 0.05 < velocity <= 0.25 and size_diff >= 4:
            return SLOW_PUSH
        if -0.25 <= velocity < -0.05 and size_diff >= 4:
            return SLOW_PUSH   # a slow push building toward you
        return NEUTRAL


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

@dataclass
class DecisionContext:
    """Non-wave inputs that change what the right wave decision is."""

    hp_pct: float = 1.0
    enemy_hp_pct: float = 1.0
    gold_available: int = 0
    #: Gold value of the item the player is saving for, if known.
    next_item_cost: int = 0
    level: int = 1
    enemy_level: int = 1
    enemy_laner_alive: bool = True
    enemy_laner_in_lane: bool = True
    #: Distance in map units from the nearest enemy jungler, if visible.
    enemy_jungler_distance: float = 99999.0
    #: Seconds until the next major objective spawns (dragon/baron/herald).
    seconds_to_objective: float = 9999.0
    objective_name: str = ""
    #: Turret plates still available in this lane (they expire at 14:00).
    plates_remaining: int = 0
    game_time_s: float = 0.0
    #: True when the player is the one with a meaningful roam target.
    has_roam_target: bool = False


def recommend(obs: WaveObservation, ctx: DecisionContext) -> WaveObservation:
    """Attach a recommended action and its reasoning to an observation.

    The rules encode standard wave-management theory, ordered so that the
    highest-stakes consideration wins:

    1. **Safety.** Low health next to a pushed wave is how players die; reset.
    2. **Tempo.** An objective inside the next minute makes lane priority
       worth more than the wave itself.
    3. **Denial.** A dead or absent enemy laner turns a slow push into free
       plates or a free crash.
    4. **Economy.** Enough gold for a real item plus a crashing wave is the
       textbook reset window.
    5. **Default.** Otherwise hold the state that keeps you safest given the
       matchup pressure.

    The reasons list is what the dataset stage turns into prose, so each entry
    is written as a complete, human-readable clause.
    """
    reasons: list[str] = []
    action = NEUTRAL

    danger = (ctx.enemy_jungler_distance < 3500.0)
    deep = obs.position > 0.35

    # 1. Safety first.
    if ctx.hp_pct < 0.35 and (deep or danger):
        action = RESET
        reasons.append(
            f"you are at {int(ctx.hp_pct * 100)}% health with the wave "
            f"{_position_phrase(obs.position)}, which is how you get killed here")
        if danger:
            reasons.append("the enemy jungler is close enough to collapse")

    # 2. Objective tempo.
    elif ctx.seconds_to_objective <= 60 and ctx.objective_name:
        action = CRASH
        reasons.append(
            f"{ctx.objective_name} spawns in about {int(ctx.seconds_to_objective)} "
            f"seconds, so you want the wave crashed and lane priority before it")
        if obs.state in (SLOW_PUSH, BIG_WAVE):
            reasons.append("the wave is already big enough to crash on its own")

    # 3. Enemy laner unavailable.
    elif not ctx.enemy_laner_alive:
        if ctx.plates_remaining > 0 and ctx.game_time_s < 14 * 60:
            action = BUILD_SLOW_PUSH
            reasons.append(
                f"the enemy laner is dead and {ctx.plates_remaining} plates are "
                f"still up, so build the wave and take plates rather than crashing now")
        else:
            action = CRASH
            reasons.append("the enemy laner is dead, so crash the wave for a free reset")

    # 4. Economy: crash and reset on a real item.
    elif (obs.state in (CRASHING, BIG_WAVE) and ctx.next_item_cost
          and ctx.gold_available >= ctx.next_item_cost):
        action = RESET
        reasons.append(
            f"you have {ctx.gold_available} gold, enough for your next item, "
            f"and the wave is {_state_phrase(obs.state)}, so crash it and back")

    # 5. Freeze when you are behind or the matchup is unfavourable.
    elif obs.state == FREEZE or (obs.position < -0.2 and ctx.hp_pct < 0.6):
        action = HOLD_FREEZE
        reasons.append(
            "the wave is on your half and static, so holding the freeze keeps "
            "you safe from ganks and forces the enemy to overextend to farm")
        if ctx.level < ctx.enemy_level:
            reasons.append(
                f"you are level {ctx.level} into level {ctx.enemy_level}, "
                f"so you lose an even fight and should not break it")

    # 6. Roam windows.
    elif ctx.has_roam_target and obs.state in (SLOW_PUSH, BIG_WAVE, CRASHING):
        action = SHOVE_AND_ROAM
        reasons.append(
            "the wave is heading into the enemy turret anyway, so shove it in "
            "and use the window to roam rather than sitting in lane")

    # 7. Default by state.
    else:
        if obs.state == SLOW_PUSH:
            action = BUILD_SLOW_PUSH
            reasons.append(
                "keep the slow push building so the crash lines up with your "
                "next back or the next objective")
        elif obs.state == CRASHING:
            action = CRASH
            reasons.append("the wave is already crashing, so follow it in and reset after")
        elif obs.state == BOUNCING:
            action = HOLD_FREEZE
            reasons.append(
                "the wave is bouncing back to you, so meet it near your turret "
                "and set up a freeze instead of walking up")
        else:
            action = NEUTRAL
            reasons.append("the wave is even, so match the enemy and look for a trade window")

    if obs.cannon and action in (CRASH, RESET):
        reasons.append("the cannon minion is in this wave, so do not leave it behind")
    if danger and action in (CRASH, SHOVE_AND_ROAM):
        reasons.append(
            "watch for the enemy jungler while you push, since you will be "
            "the furthest forward")

    obs.recommended = action
    obs.reasons = reasons
    return obs


def _position_phrase(p: float) -> str:
    """Human phrasing for a lane coordinate."""
    if p > 0.75:
        return "under the enemy turret"
    if p > 0.35:
        return "pushed into the enemy half"
    if p > 0.1:
        return "slightly past the middle of the lane"
    if p > -0.1:
        return "in the middle of the lane"
    if p > -0.4:
        return "on your side of the lane"
    if p > -0.75:
        return "close to your turret"
    return "under your turret"


def _state_phrase(state: str) -> str:
    return {
        FREEZE: "frozen near your turret",
        SLOW_PUSH: "slow pushing toward the enemy",
        BIG_WAVE: "a large stacked wave",
        CRASHING: "crashing into the enemy turret",
        BOUNCING: "bouncing back toward you",
        NEUTRAL: "even",
    }.get(state, state.lower())


def describe(obs: WaveObservation) -> str:
    """One-sentence description of the wave, used in prompt contexts."""
    parts = [f"{_state_phrase(obs.state)}", f"{_position_phrase(obs.position)}"]
    if obs.size_diff >= 6:
        parts.append(f"roughly {int(obs.size_diff)} extra minions stacked")
    if obs.cannon:
        parts.append("cannon minion present")
    return ", ".join(parts)
