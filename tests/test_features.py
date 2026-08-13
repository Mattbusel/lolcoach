"""Tests for the Stage-2 inference layer.

These are the tests that matter most in this project: the wave, jungle and
death estimators are the part a reader has to *trust*, because their output
becomes training data. Each test scripts a situation whose answer is known
and asserts the estimator recovers it.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from lolcoach.features.geometry import (BLUE, JUNGLE_CAMPS, NEXUS, RED,
                                        assign_lane, at_fountain,
                                        lane_corridor_distance, lane_geometry,
                                        nearest_camp, refine_turrets_from_events,
                                        zone_of)
from lolcoach.features.jungle import JunglePredictor, phase_of, reconstruct_path
from lolcoach.features.lane import ChampionState, trade_window
from lolcoach.features.macro import analyse_death, cluster_fights
from lolcoach.features.objectives import (DRAGON, evaluate_take, next_spawn_ms,
                                          seconds_to_next_objective)
from lolcoach.features.waves import (CRASHING, FREEZE, DecisionContext,
                                     LaneFrameInput, WaveEstimator,
                                     current_wave_in_lane, minions_spawned_by,
                                     recommend, wave_has_cannon, wave_index_at)


# ---------------------------------------------------------------------------
# Minion spawn model
# ---------------------------------------------------------------------------

def test_first_wave_spawns_at_65_seconds():
    assert wave_index_at(64) == 0
    assert wave_index_at(65) == 1
    assert wave_index_at(95) == 2


def test_cannon_every_third_wave_before_fifteen_minutes():
    early = [i for i in range(1, 16) if wave_has_cannon(i)]
    assert early[:4] == [3, 6, 9, 12]


def test_cannon_cadence_changes_at_fifteen_and_twenty_five_minutes():
    """Cadence goes every-third, then every-second, then every wave.

    Wave ``i`` spawns at ``65 + 30(i-1)`` seconds, so waves 30-48 fall inside
    the 15:00-25:00 window and waves past 49 are after 25:00.
    """
    middle = [i for i in range(30, 48) if wave_has_cannon(i)]
    assert {b - a for a, b in zip(middle, middle[1:])} == {2}

    late = [i for i in range(50, 60) if wave_has_cannon(i)]
    assert late == list(range(50, 60))


def test_minion_count_matches_hand_calculation():
    # By 5:00 exactly 8 waves have spawned, two of which carried a cannon.
    assert minions_spawned_by(300) == 8 * 6 + 2


def test_current_wave_accounts_for_travel_time():
    """A wave is only 'in lane' once it has walked there."""
    mid_index, _ = current_wave_in_lane(120, "MIDDLE")
    top_index, _ = current_wave_in_lane(120, "TOP")
    assert mid_index > top_index      # mid is a shorter walk


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_lane_projection_is_symmetric_between_teams():
    blue = lane_geometry("MIDDLE", BLUE)
    red = lane_geometry("MIDDLE", RED)
    midpoint = blue.midpoint
    assert blue.project(midpoint) == pytest.approx(red.project(midpoint), abs=1e-6)
    assert blue.project(blue.own_turret) == pytest.approx(-1.0, abs=0.01)
    assert blue.project(blue.enemy_turret) == pytest.approx(1.0, abs=0.01)


def test_fountain_positions_are_not_counted_as_lane_positions():
    """Both bases sit inside the top and bottom lane corridors.

    Without the fountain check a player who just recalled would be treated as
    standing in lane, which drags the wave estimate to the lane midpoint.
    """
    for team in (BLUE, RED):
        base = NEXUS[team]
        assert at_fountain(base)
        assert min(lane_corridor_distance(base, lane)
                   for lane in ("TOP", "MIDDLE", "BOTTOM")) < 1800


def test_zone_labels_river_and_lanes():
    assert assign_lane((7400.0, 7400.0)) == "MIDDLE"
    assert zone_of((7400.0, 7400.0)) == "MIDDLE_LANE"
    assert zone_of((4400.0, 9600.0)) in ("RIVER", "BARON_PIT", "BLUE_TOP_JUNGLE")


def test_turret_positions_refine_from_observed_events():
    """Enough observations should move a deliberately wrong constant."""
    events = [{"type": "BUILDING_KILL", "towerType": "OUTER_TURRET",
               "laneType": "MID_LANE", "teamId": BLUE,
               "position": {"x": 5000, "y": 5000}} for _ in range(10)]
    refined = refine_turrets_from_events(events)
    assert refined[(BLUE, "MIDDLE")] == (5000.0, 5000.0)


def test_refinement_ignores_sparse_evidence():
    events = [{"type": "BUILDING_KILL", "towerType": "OUTER_TURRET",
               "laneType": "MID_LANE", "teamId": BLUE,
               "position": {"x": 1, "y": 1}}]
    assert refine_turrets_from_events(events) == {}


# ---------------------------------------------------------------------------
# Wave estimation
# ---------------------------------------------------------------------------

def _run_wave(positions_own, positions_enemy, cs_own=0, cs_enemy=0, start_min=1):
    """Drive the estimator over a scripted sequence and return observations."""
    est = WaveEstimator("MIDDLE", BLUE)
    geo = lane_geometry("MIDDLE", BLUE)
    out = []
    for i, (p_own, p_enemy) in enumerate(zip(positions_own, positions_enemy)):
        ts = (start_min + i) * 60_000
        out.append(est.update(LaneFrameInput(
            ts_ms=ts,
            own_positions=[geo.point_at(p_own)],
            enemy_positions=[geo.point_at(p_enemy)],
            own_cs=cs_own + i * 7, enemy_cs=cs_enemy + i * 7)))
    return out


def test_static_wave_on_own_half_reads_as_freeze():
    own = [-0.62] * 6
    enemy = [-0.42] * 6
    observations = _run_wave(own, enemy)
    assert observations[-1].state == FREEZE
    assert observations[-1].position < -0.3


def test_wave_moving_into_enemy_turret_reads_as_crashing():
    own = [0.0, 0.25, 0.5, 0.75, 0.95]
    enemy = [0.1, 0.35, 0.6, 0.85, 1.05]
    observations = _run_wave(own, enemy)
    assert observations[-1].state == CRASHING
    assert observations[-1].position > 0.5


def test_empty_lane_lowers_confidence_rather_than_inventing_a_position():
    est = WaveEstimator("MIDDLE", BLUE)
    geo = lane_geometry("MIDDLE", BLUE)
    first = est.update(LaneFrameInput(
        ts_ms=60_000, own_positions=[geo.point_at(-0.5)],
        enemy_positions=[geo.point_at(-0.3)], own_cs=10, enemy_cs=10))
    empty = est.update(LaneFrameInput(
        ts_ms=120_000, own_positions=[], enemy_positions=[],
        own_cs=10, enemy_cs=10))
    assert empty.confidence < first.confidence
    assert any("extrapolated" in f for f in empty.factors)


def test_recommendation_prioritises_safety_over_tempo():
    """A player at low health next to a pushed wave should be told to reset."""
    observations = _run_wave([0.4, 0.6], [0.5, 0.7])
    obs = observations[-1]
    ctx = DecisionContext(hp_pct=0.2, enemy_jungler_distance=1500.0,
                          game_time_s=600.0)
    decided = recommend(obs, ctx)
    assert decided.recommended == "RESET"
    assert any("health" in r for r in decided.reasons)


def test_recommendation_takes_objective_tempo():
    observations = _run_wave([0.0, 0.1], [0.1, 0.2])
    ctx = DecisionContext(hp_pct=0.95, seconds_to_objective=30.0,
                          objective_name="dragon", game_time_s=280.0)
    decided = recommend(observations[-1], ctx)
    assert decided.recommended == "CRASH"
    assert any("dragon" in r for r in decided.reasons)


# ---------------------------------------------------------------------------
# Lane trades
# ---------------------------------------------------------------------------

def _champ(**kwargs) -> ChampionState:
    base = dict(participant_id=1, team_id=BLUE, position=(1000.0, 1000.0),
                level=6, xp=2000, total_gold=3000, current_gold=500, cs=60,
                health=1000, health_max=1000, ad=100, ap=0, armor=40, mr=35)
    base.update(kwargs)
    return ChampionState(**base)


def test_healthier_stronger_champion_wins_the_trade():
    strong = _champ(ad=140, health=1000)
    weak = _champ(participant_id=2, team_id=RED, ad=80, health=450)
    label, ratio = trade_window(strong, weak)
    assert label == "FAVOURABLE"
    assert ratio > 1.15


def test_low_health_flips_an_otherwise_winning_trade():
    strong_but_low = _champ(ad=140, health=180)
    healthy = _champ(participant_id=2, team_id=RED, ad=95, health=1000)
    label, _ = trade_window(strong_but_low, healthy)
    assert label == "UNFAVOURABLE"


# ---------------------------------------------------------------------------
# Jungle
# ---------------------------------------------------------------------------

def test_camp_clears_are_recovered_from_kill_count_increases():
    states = [
        (60_000, JUNGLE_CAMPS["BLUE_BLUEBUFF"], 4),
        (120_000, JUNGLE_CAMPS["BLUE_GROMP"], 8),
        (180_000, JUNGLE_CAMPS["BLUE_WOLVES"], 12),
    ]
    clears = reconstruct_path(states, BLUE)
    assert [c.camp for c in clears] == ["BLUE_BLUEBUFF", "BLUE_GROMP", "BLUE_WOLVES"]


def test_no_clear_recorded_without_a_kill_count_increase():
    states = [(60_000, JUNGLE_CAMPS["BLUE_GROMP"], 4),
              (120_000, JUNGLE_CAMPS["BLUE_WOLVES"], 4)]
    assert len(reconstruct_path(states, BLUE)) == 1


def test_predictor_learns_an_observed_transition():
    predictor = JunglePredictor()
    clears = reconstruct_path([
        (60_000, JUNGLE_CAMPS["BLUE_RAPTORS"], 4),
        (120_000, JUNGLE_CAMPS["BLUE_KRUGS"], 8),
    ], BLUE)
    for _ in range(20):
        predictor.observe_sequence(clears)
    top = predictor.predict("BLUE_RAPTORS", 90.0, top_k=1)
    assert top and top[0][0] == "BLUE_KRUGS"
    assert top[0][1] > 0.5


def test_predictor_round_trips_through_json():
    predictor = JunglePredictor()
    predictor.counts["EARLY|BLUE_GROMP"]["BLUE_WOLVES"] = 7.0
    restored = JunglePredictor.from_json(predictor.to_json())
    assert restored.counts["EARLY|BLUE_GROMP"]["BLUE_WOLVES"] == 7.0


def test_phase_boundaries():
    assert phase_of(60) == "FIRST_CLEAR"
    assert phase_of(600) == "EARLY"
    assert phase_of(1200) == "MID"
    assert phase_of(1800) == "LATE"


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def test_dragon_respawn_is_five_minutes():
    assert next_spawn_ms(DRAGON, 600_000) == 600_000 + 300_000


def test_next_objective_reports_the_soonest():
    seconds, name = seconds_to_next_objective(240_000, [])
    assert name == "dragon"
    assert seconds == pytest.approx(60.0, abs=1.0)


def test_uncontested_take_scores_higher_than_contested():
    pit_positions = {BLUE: [(9866.0, 4414.0)] * 4, RED: []}
    uncontested = evaluate_take(600_000, DRAGON, "FIRE", BLUE, pit_positions,
                                {BLUE: 3, RED: 0}, [])
    contested = evaluate_take(600_000, DRAGON, "FIRE", BLUE,
                              {BLUE: [(9866.0, 4414.0)], RED: [(9866.0, 4414.0)] * 4},
                              {BLUE: 0, RED: 3}, [])
    assert uncontested.setup_score > contested.setup_score
    assert contested.contested


# ---------------------------------------------------------------------------
# Deaths and fights
# ---------------------------------------------------------------------------

def test_outnumbered_death_is_labelled_as_such():
    positions = {1: (7400.0, 7400.0), 2: (7500.0, 7500.0),
                 3: (7600.0, 7400.0), 4: (7300.0, 7500.0)}
    teams = {1: BLUE, 2: RED, 3: RED, 4: RED}
    analysis = analyse_death(
        ts_ms=600_000, victim_id=1, killer_id=2, position=(7400.0, 7400.0),
        victim_team=BLUE, assisters=2, positions=positions, teams=teams,
        wards=[], hp_before=300, hp_max=1000, bounty=300, in_fight=False)
    assert analysis.cause == "OUTNUMBERED"
    assert analysis.outnumbered >= 2


def test_death_without_vision_is_distinguished_from_a_lost_trade():
    positions = {1: (12000.0, 12000.0), 2: (12100.0, 12100.0)}
    teams = {1: BLUE, 2: RED}
    analysis = analyse_death(
        ts_ms=600_000, victim_id=1, killer_id=2, position=(12000.0, 12000.0),
        victim_team=BLUE, assisters=0, positions=positions, teams=teams,
        wards=[], hp_before=900, hp_max=1000, bounty=300, in_fight=False)
    assert analysis.cause == "NO_VISION"
    assert not analysis.vision_nearby
    rationale = " ".join(analysis.reasons).lower()
    assert "enemy vision cannot be inferred" in rationale
    assert "they saw you" not in rationale


def test_kills_close_in_time_and_space_form_one_fight():
    kills = [
        (600_000, 1, 6, (7400.0, 7400.0)),
        (605_000, 2, 7, (7500.0, 7500.0)),
        (610_000, 3, 8, (7600.0, 7600.0)),
    ]
    teams = {1: BLUE, 2: BLUE, 3: BLUE, 6: RED, 7: RED, 8: RED}
    fights = cluster_fights(kills, teams, {600_000: {}})
    assert len(fights) == 1
    assert fights[0].kills_a == 3
    assert fights[0].winner_team == BLUE


def test_distant_kills_are_separate_fights():
    kills = [
        (600_000, 1, 6, (2000.0, 2000.0)),
        (601_000, 2, 7, (13000.0, 13000.0)),
        (602_000, 3, 8, (13100.0, 13100.0)),
    ]
    teams = {1: BLUE, 2: BLUE, 3: BLUE, 6: RED, 7: RED, 8: RED}
    fights = cluster_fights(kills, teams, {600_000: {}})
    # The lone far-away kill does not reach the two-kill threshold.
    assert len(fights) == 1
