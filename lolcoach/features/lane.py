"""Laning features: priority, trading windows, and recall timing.

Riot's participant frames carry a ``championStats`` block with the champion's
live attack damage, ability power, armour, magic resist, health and resource
at every minute. That is enough to answer "who wins this trade right now"
numerically rather than by vibes, and it is the basis of everything here.

Three families of feature come out of this module:

``lane_states``
    Per player, per minute: gold/xp/cs differentials against the direct
    opponent, health, lane priority, whether the trade or the all-in is
    currently winning, and whether backing is worth more than staying.

``recalls``
    Detected recall windows with the gold spent and the wave state that was
    left behind, plus a label for whether the recall actually worked out.

Trading model
-------------
Effective combat power blends offence and survivability::

    offence      = attack_damage + 0.65 * ability_power
    effective_hp = health * (1 + armour/100) and (1 + magic_resist/100), averaged
    power        = offence * sqrt(effective_hp)

This is deliberately simple and deliberately documented as an approximation:
it ignores ability cooldowns, ranges and champion kits, which the frames do
not expose. It is used only to rank *relative* strength between two laners at
the same instant, where those omissions largely cancel, and every derived row
carries a confidence that reflects the crudeness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .geometry import NEXUS, Point, at_fountain, dist, lane_corridor_distance

# Lane priority labels.
HARD_PRIO = "HARD_PRIO"
PRIO = "PRIO"
EVEN_PRIO = "EVEN"
NO_PRIO = "NO_PRIO"

# Trade window labels.
FAVOURABLE = "FAVOURABLE"
EVEN_TRADE = "EVEN"
UNFAVOURABLE = "UNFAVOURABLE"

#: A champion within this distance of their own nexus is at the fountain.
FOUNTAIN_RADIUS = 2000.0
#: Turret plates stop being available at 14 minutes.
PLATE_DEADLINE_S = 14 * 60.0


@dataclass
class ChampionState:
    """The subset of a participant frame this module reasons about."""

    participant_id: int
    team_id: int
    position: Point
    level: int
    xp: int
    total_gold: int
    current_gold: int
    cs: int
    health: int
    health_max: int
    ad: int = 0
    ap: int = 0
    armor: int = 0
    mr: int = 0
    #: Frame timestamp this state came from, in milliseconds.
    ts_ms: int = 0

    @property
    def hp_pct(self) -> float:
        return (self.health / self.health_max) if self.health_max else 1.0

    @property
    def offence(self) -> float:
        """Blended damage output proxy."""
        return float(self.ad) + 0.65 * float(self.ap)

    @property
    def effective_hp(self) -> float:
        """Health averaged over physical and magic mitigation."""
        h = float(max(self.health, 1))
        phys = h * (1.0 + max(0, self.armor) / 100.0)
        magic = h * (1.0 + max(0, self.mr) / 100.0)
        return (phys + magic) / 2.0

    @property
    def power(self) -> float:
        """Instantaneous combat power used for trade comparisons."""
        return self.offence * math.sqrt(max(1.0, self.effective_hp))


@dataclass
class LaneState:
    """One player's laning situation at one timestamp."""

    ts_ms: int
    participant_id: int
    lane: str | None
    cs_diff: int
    gold_diff: int
    xp_diff: int
    level_diff: int
    hp_pct: float
    opp_hp_pct: float
    plates: int
    priority: str
    trade_window: str
    all_in: bool
    recall_value: float
    should_recall: bool
    confidence: float
    reasons: list[str] = field(default_factory=list)


def trade_window(own: ChampionState, opp: ChampionState) -> tuple[str, float]:
    """Classify the current trade and return ``(label, ratio)``.

    The ratio is own power over opponent power at full observed health; the
    label additionally accounts for the current health difference, because a
    stronger champion at 30% health does not win the trade.
    """
    own_power = own.power
    opp_power = opp.power
    if opp_power <= 0:
        return EVEN_TRADE, 1.0
    ratio = own_power / opp_power
    hp_ratio = (own.hp_pct + 0.35) / (opp.hp_pct + 0.35)
    effective = ratio * math.sqrt(hp_ratio)
    if effective >= 1.15:
        return FAVOURABLE, effective
    if effective <= 0.87:
        return UNFAVOURABLE, effective
    return EVEN_TRADE, effective


def lane_priority(own_positions: Sequence[Point], opp_positions: Sequence[Point],
                  lane: str, wave_position: float, gold_diff: int) -> str:
    """Estimate which side can leave the lane first.

    Priority is about who can walk away without losing something. That is
    driven by presence (are they even in lane), the wave (is it heading into
    their turret) and raw strength (can they contest a shove at all).
    """
    own_in = sum(1 for p in own_positions
                 if lane_corridor_distance(p, lane) < 1800 and not at_fountain(p))
    opp_in = sum(1 for p in opp_positions
                 if lane_corridor_distance(p, lane) < 1800 and not at_fountain(p))

    score = 0.0
    score += (own_in - opp_in) * 1.2
    score += wave_position * 1.5          # wave heading at them is prio for you
    score += max(-1.0, min(1.0, gold_diff / 1500.0))

    if score >= 1.8:
        return HARD_PRIO
    if score >= 0.6:
        return PRIO
    if score <= -1.8:
        return NO_PRIO
    if score <= -0.6:
        return NO_PRIO if score < -1.0 else EVEN_PRIO
    return EVEN_PRIO


def recall_value(state: ChampionState, next_item_cost: int, wave_position: float,
                 seconds_to_objective: float) -> tuple[float, bool, list[str]]:
    """Score how good backing right now would be, in gold-equivalent terms.

    The components are the ones a coach would list: the gold you would spend,
    the health you would restore, the minions you would give up, and whether
    a fight is about to start that you need to be present for.
    """
    reasons: list[str] = []
    value = 0.0

    if next_item_cost and state.current_gold >= next_item_cost:
        value += 400.0
        reasons.append(
            f"you have {state.current_gold} gold, which buys your next item")
    elif state.current_gold >= 1100:
        value += 220.0
        reasons.append(f"you are holding {state.current_gold} gold doing nothing")

    if state.hp_pct < 0.35:
        value += 350.0
        reasons.append(f"you are at {int(state.hp_pct * 100)}% health")
    elif state.hp_pct < 0.55:
        value += 140.0

    # Backing on a wave that is walking at your turret costs you the whole wave.
    if wave_position < -0.3:
        value -= 260.0
        reasons.append("the wave is heading toward your turret, so you would lose it")
    elif wave_position > 0.5:
        value += 180.0
        reasons.append("the wave is crashing into their turret, so it is a free window")

    if seconds_to_objective < 45:
        value -= 300.0
        reasons.append("an objective is about to spawn and you need to be on the map")

    return value, value >= 250.0, reasons


def detect_recalls(frames: Sequence[ChampionState], team_id: int,
                   purchases: Sequence[tuple[int, int, int]]
                   ) -> list[tuple[int, int, list[int]]]:
    """Find recall events from position and purchase history.

    A recall is inferred where a champion appears at their own fountain having
    not been there in the previous frame. Purchases in the surrounding window
    are attached, which both confirms the recall and gives the gold spent.

    Parameters
    ----------
    frames:
        One participant's states in chronological order.
    purchases:
        ``(timestamp_ms, participant_id, item_id)`` triples for this player.

    Returns
    -------
    list of ``(timestamp_ms, gold_spent, item_ids)``. ``gold_spent`` is
    returned as zero here and filled in by the caller, which holds the item
    price table needed to value the purchases.
    """
    base = NEXUS[team_id]
    out: list[tuple[int, int, list[int]]] = []
    was_home = True          # everyone starts at the fountain
    for f in frames:
        at_home = dist(f.position, base) < FOUNTAIN_RADIUS
        if at_home and not was_home:
            window = [item for ts, _, item in purchases
                      if abs(ts - f.ts_ms) <= 45_000]
            out.append((f.ts_ms, 0, window))
        was_home = at_home
    return out


def recall_was_good(before: ChampionState | None, after: ChampionState | None,
                    items_bought: int, died_within: bool) -> tuple[bool, str]:
    """Weak label for whether a recall paid off.

    A recall is judged good when the player came back with items and did not
    immediately die, and bad when they bought nothing, or died right after
    walking back to a lane they had given up. This is a heuristic label used
    for filtering training examples, never presented to the user as fact.
    """
    if died_within:
        return False, "died shortly after returning to lane"
    if items_bought == 0:
        return False, "backed without enough gold to buy anything meaningful"
    if before is not None and after is not None:
        gained = after.total_gold - before.total_gold
        if gained < 0:
            return False, "lost more gold in lane than the back was worth"
    return True, "returned with an item and kept the lane stable"


@dataclass
class ItemPlan:
    """Cheap lookup of the next item a player is saving toward."""

    #: Ordered legendary item costs seen in this player's build so far.
    costs: dict[int, int] = field(default_factory=dict)

    def next_cost(self, owned: Iterable[int], default: int = 1300) -> int:
        """Cost of the cheapest legendary the player does not yet own."""
        owned_set = set(owned)
        candidates = [c for i, c in self.costs.items()
                      if i not in owned_set and c >= 1000]
        return min(candidates) if candidates else default


def build_lane_state(ts_ms: int, own: ChampionState, opp: ChampionState | None,
                     lane: str | None, wave_position: float,
                     own_team_positions: Sequence[Point],
                     opp_team_positions: Sequence[Point],
                     plates: int, next_item_cost: int,
                     seconds_to_objective: float) -> LaneState:
    """Assemble a :class:`LaneState` for one player at one frame."""
    if opp is None:
        # No direct opponent identified (roaming support, filled lane).
        return LaneState(
            ts_ms=ts_ms, participant_id=own.participant_id, lane=lane,
            cs_diff=0, gold_diff=0, xp_diff=0, level_diff=0,
            hp_pct=own.hp_pct, opp_hp_pct=1.0, plates=plates,
            priority=EVEN_PRIO, trade_window=EVEN_TRADE, all_in=False,
            recall_value=0.0, should_recall=False, confidence=0.25,
            reasons=["no direct lane opponent was identified for this frame"])

    window, ratio = trade_window(own, opp)
    priority = lane_priority(own_team_positions, opp_team_positions,
                             lane or "MIDDLE", wave_position,
                             own.total_gold - opp.total_gold)
    value, should, reasons = recall_value(own, next_item_cost, wave_position,
                                          seconds_to_objective)
    all_in = ratio >= 1.30 and own.hp_pct >= 0.6 and opp.hp_pct <= 0.85

    if window == FAVOURABLE:
        reasons.insert(0, f"your stats beat theirs by about {int((ratio - 1) * 100)}% "
                          f"right now, so short trades favour you")
    elif window == UNFAVOURABLE:
        reasons.insert(0, f"they out-stat you by about {int((1 / max(ratio, 0.01) - 1) * 100)}%, "
                          f"so avoid extended trades")

    confidence = 0.7 if lane else 0.4
    if own.health_max <= 0:
        confidence *= 0.5

    return LaneState(
        ts_ms=ts_ms, participant_id=own.participant_id, lane=lane,
        cs_diff=own.cs - opp.cs, gold_diff=own.total_gold - opp.total_gold,
        xp_diff=own.xp - opp.xp, level_diff=own.level - opp.level,
        hp_pct=round(own.hp_pct, 3), opp_hp_pct=round(opp.hp_pct, 3),
        plates=plates, priority=priority, trade_window=window, all_in=all_in,
        recall_value=round(value, 1), should_recall=should,
        confidence=round(confidence, 3), reasons=reasons)
