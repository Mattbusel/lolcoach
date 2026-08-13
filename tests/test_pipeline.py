"""End-to-end tests: raw payload -> tables -> features -> training examples.

The fixture in :mod:`tests.fixtures` scripts a game whose wave behaviour we
already know (a freeze on top, a slow push into a crash on mid), so these
tests check that the answer survives the whole pipeline rather than just the
estimator in isolation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from fixtures import build_synthetic_match, write_synthetic

from lolcoach.dataset import generators as gen
from lolcoach.eval.scoring import (KnowledgeBase, coaching_quality, consistency,
                                   extract_claims, factual_accuracy,
                                   hallucination_risk, verify_claims)
from lolcoach.features.derive import derive_match
from lolcoach.ingest.core import _ingest_riot_match
from lolcoach.sources.ddragon import public_patch_label, strip_tags
from lolcoach.sources.oracles_elixir import discover_drive_files
from lolcoach.sources.reddit import looks_useful
from lolcoach.sources.replays import _extract_json_object, parse_rofl
from lolcoach.storage import Database


@pytest.fixture()
def loaded_db(tmp_path: Path) -> Database:
    """A database with one fully ingested synthetic match."""
    db = Database(tmp_path / "test.sqlite")
    match, timeline = build_synthetic_match("TEST_PIPE", minutes=25, seed=3)
    _ingest_riot_match(db, "TEST_PIPE", match, timeline, "memory")
    return db


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def test_ingest_populates_all_four_tables(loaded_db: Database):
    assert loaded_db.scalar("SELECT COUNT(*) FROM matches") == 1
    assert loaded_db.scalar("SELECT COUNT(*) FROM participants") == 10
    assert loaded_db.scalar("SELECT COUNT(*) FROM frames") == 26 * 10
    assert loaded_db.scalar("SELECT COUNT(*) FROM events") > 0


def test_health_is_read_from_champion_stats(loaded_db: Database):
    """Health lives in ``championStats``; reading the wrong field zeroes it.

    Every trade, recall and death feature is derived from health, so this is
    the single most damaging silent ingest bug possible.
    """
    zero_health = loaded_db.scalar(
        "SELECT COUNT(*) FROM frames WHERE health = 0 OR health IS NULL")
    assert zero_health == 0
    assert loaded_db.scalar("SELECT MAX(health_max) FROM frames") > 0


def test_elite_monster_kills_keep_their_credited_team(loaded_db: Database):
    """Riot puts the team on monster kills in ``killerTeamId``."""
    rows = loaded_db.query(
        "SELECT team_id FROM events WHERE type='ELITE_MONSTER_KILL'")
    assert rows
    assert all(row[0] in (100, 200) for row in rows)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_feature_pass_writes_every_table(loaded_db: Database):
    result = derive_match(loaded_db, "TEST_PIPE")
    assert result.matches == 1
    assert result.waves > 0
    assert result.lanes > 0
    assert result.objectives > 0
    assert result.deaths > 0


def test_scripted_freeze_is_recovered_end_to_end(loaded_db: Database):
    """The fixture freezes blue's top wave; the pipeline should say so."""
    derive_match(loaded_db, "TEST_PIPE")
    states = [row[0] for row in loaded_db.query(
        "SELECT state FROM wave_states WHERE lane='TOP' AND team_id=100 "
        "AND ts_ms BETWEEN 600000 AND 800000 ORDER BY ts_ms")]
    assert "FREEZE" in states


def test_scripted_crash_is_recovered_end_to_end(loaded_db: Database):
    """The fixture crashes blue's mid wave after minute 8."""
    derive_match(loaded_db, "TEST_PIPE")
    rows = loaded_db.query(
        "SELECT position FROM wave_states WHERE lane='MIDDLE' AND team_id=100 "
        "AND ts_ms >= 600000 ORDER BY ts_ms")
    assert rows
    assert max(row[0] for row in rows) > 0.5


def test_rationale_is_persisted_for_the_dataset_stage(loaded_db: Database):
    derive_match(loaded_db, "TEST_PIPE")
    row = loaded_db.query(
        "SELECT rationale FROM wave_states WHERE rationale IS NOT NULL LIMIT 1")
    assert row
    reasons = json.loads(row[0][0])
    assert isinstance(reasons, list) and reasons


def test_features_are_idempotent(loaded_db: Database):
    derive_match(loaded_db, "TEST_PIPE")
    first = loaded_db.scalar("SELECT COUNT(*) FROM wave_states")
    derive_match(loaded_db, "TEST_PIPE")
    assert loaded_db.scalar("SELECT COUNT(*) FROM wave_states") == first


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def test_generators_produce_grounded_examples(loaded_db: Database):
    derive_match(loaded_db, "TEST_PIPE")
    waves = list(gen.wave_examples(loaded_db, 0.0))
    assert waves
    archetype, key, user, assistant, confidence, match_id, patch = waves[0]
    assert archetype in ("wave_decision", "wave_read")
    assert "Context:" in user and "Question:" in user
    assert len(assistant) > 40
    assert 0.0 <= confidence <= 1.0


def test_wave_examples_respect_the_confidence_floor(loaded_db: Database):
    derive_match(loaded_db, "TEST_PIPE")
    strict = list(gen.wave_examples(loaded_db, 0.99))
    loose = list(gen.wave_examples(loaded_db, 0.0))
    assert len(strict) < len(loose)


def test_diagnostic_clauses_do_not_leak_into_advice(loaded_db: Database):
    """"Both laners in lane" explains the estimate; it is not coaching."""
    derive_match(loaded_db, "TEST_PIPE")
    for archetype, _key, _user, assistant, *_ in gen.wave_examples(loaded_db, 0.0):
        assert "both laners in lane" not in assistant.lower()
        assert "extrapolated" not in assistant.lower()


def test_paraphrase_selection_is_deterministic():
    options = ["a", "b", "c", "d"]
    assert gen.pick(options, "key-1") == gen.pick(options, "key-1")


def test_reddit_document_round_trip():
    text = ("Question (summonerschool, score 40): How do I freeze?\n"
            "Body text explaining the situation in detail.\n\n"
            "Answer 1 (score 55):\nFirst answer body.\n\n"
            "Answer 2 (score 12):\nSecond answer body.")
    question, answers = gen.parse_reddit_document(text)
    assert "How do I freeze?" in question
    assert [score for score, _ in answers] == [55, 12]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_wrong_objective_timer_is_caught():
    kb = KnowledgeBase()
    claims = verify_claims(
        extract_claims("Baron respawns every 2 minutes so you have time."), kb)
    assert any(c.verdict == "wrong" for c in claims)


def test_correct_objective_timer_passes():
    kb = KnowledgeBase()
    claims = verify_claims(
        extract_claims("Baron respawns every 6 minutes after it dies."), kb)
    assert any(c.verdict == "correct" for c in claims)


def test_answers_without_claims_are_scored_neutrally():
    kb = KnowledgeBase()
    accuracy, claims = factual_accuracy("Play safe and farm up.", kb)
    assert accuracy == 0.5
    assert not [c for c in claims if c.verdict != "unverified"]


def test_invented_champion_raises_hallucination_risk():
    kb = KnowledgeBase(champion_names={"jinx", "draven"}, item_names=set())
    risk = hallucination_risk("Be careful playing into Zyphrax in this lane.", kb)
    assert risk > 0.0


def test_committed_answer_scores_above_waffle():
    good = ("Crash the wave because you need priority before dragon, but that "
            "gives up the freeze so watch for their jungler on the way back.")
    bad = "It depends entirely on the situation, both are viable."
    assert coaching_quality(good) > coaching_quality(bad)


def test_consistency_compares_decisions_not_wording():
    assert consistency("Crash it now.", "You should crash the wave.") == 1.0
    assert consistency("Crash it now.", "Hold the freeze.") == 0.0
    assert consistency("Hmm.", "Well.") == 0.5


# ---------------------------------------------------------------------------
# Source parsing helpers
# ---------------------------------------------------------------------------

def test_patch_label_maps_data_dragon_to_public_numbering():
    """Data Dragon 16.16 is the patch Riot and the wiki call 26.16."""
    assert public_patch_label("16.16.1") == "26.16"
    assert public_patch_label("15.1.1") == "25.01"
    assert public_patch_label("14.19.1") == "14.19"


def test_riot_tooltip_markup_is_stripped():
    raw = "Deals <physicalDamage>60</physicalDamage> damage.<br>Then heals."
    assert strip_tags(raw) == "Deals 60 damage. Then heals."


def test_drive_folder_listing_pairs_names_with_ids():
    html = ('<div data-id="ABC123DEF456GHI789JKL"><span>'
            '2026_LoL_esports_match_data_from_OraclesElixir.csv</span></div>')
    assert discover_drive_files(html) == {
        "2026_LoL_esports_match_data_from_OraclesElixir.csv": "ABC123DEF456GHI789JKL"}


def test_reddit_quality_filter_rejects_noise_and_keeps_questions():
    useful = ("How do I know when to crash the wave instead of freezing it? " * 4)
    assert looks_useful("Wave management question", useful)
    assert not looks_useful("My fan art of Jinx", "cosplay " * 60)


def test_balanced_json_extraction_handles_nested_braces():
    blob = b'garbage{"a": {"b": "}"}, "c": 1}tail'
    extracted = _extract_json_object(blob, blob.index(b"{"))
    assert json.loads(extracted) == {"a": {"b": "}"}, "c": 1}


def test_rofl_parser_rejects_non_replay_files(tmp_path: Path):
    junk = tmp_path / "not.rofl"
    junk.write_bytes(b"nothing to see here" * 20)
    assert parse_rofl(junk) is None
