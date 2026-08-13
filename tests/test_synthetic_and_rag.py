"""Tests for Stage 3 validation and the Stage 4/5 retrieval helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lolcoach.dataset.synthetic import (SYSTEM, TEACHABLE, load_completed,
                                        validate, _action_of, _teacher_prompt)
from lolcoach.rag.index import _chunks, _document_frequencies, _index_tokens
from lolcoach.rag.retrieve import _patch_matches
from lolcoach.sources.hf_datasets import score_candidate
from lolcoach.sources.wiki import clean_html, clean_wikitext

CHAMPIONS = {"Jinx", "Draven", "Ahri", "Yorick"}


# ---------------------------------------------------------------------------
# Stage 3 validation
# ---------------------------------------------------------------------------

def test_valid_explanation_is_accepted():
    answer = ("Crash the wave here because you are ahead in health and the "
              "cannon is in this one, so you get the reset for free. The cost "
              "is that you give up the freeze and have to walk back to a "
              "pushed lane, so ward the river before you go.")
    ok, reason = validate(answer, "Context: wave crashing, Jinx", "CRASH", CHAMPIONS)
    assert ok, reason


def test_answer_contradicting_its_label_is_rejected():
    answer = ("Hold the freeze here and set up a freeze near your own turret; "
              "you should never push this wave under any circumstances at all "
              "because it gives them everything for free right now.")
    ok, reason = validate(answer, "Context: wave crashing", "CRASH", CHAMPIONS)
    assert not ok and reason == "contradicts_action"


def test_champion_absent_from_context_is_rejected():
    answer = ("You should crash the wave now because Yorick is going to walk "
              "down and collapse on you before you can get the reset done, so "
              "leaving is the only safe option here in this matchup.")
    ok, reason = validate(answer, "Context: Jinx versus Draven", "CRASH", CHAMPIONS)
    assert not ok and reason == "invented_champion"


def test_number_absent_from_context_is_rejected():
    answer = ("Crash the wave because you have 4137 gold banked and that buys "
              "the item you need before their next spike arrives on the map, "
              "so the reset is clearly worth taking right now.")
    ok, reason = validate(answer, "Context: you have 900 gold", "CRASH", CHAMPIONS)
    assert not ok and reason == "invented_number"


def test_numbers_present_in_context_are_allowed():
    answer = ("Crash the wave because you have 1450 gold banked and that buys "
              "your next item before their spike, which is worth more than the "
              "few minions you give up by leaving lane now.")
    ok, reason = validate(answer, "Context: you have 1450 gold", "CRASH", CHAMPIONS)
    assert ok, reason


def test_refusals_and_stubs_are_rejected():
    ok, reason = validate("I cannot answer that.", "Context:", "", CHAMPIONS)
    assert not ok
    ok, reason = validate("Crash.", "Context:", "CRASH", CHAMPIONS)
    assert not ok and reason == "too_short"


def test_teacher_prompt_supplies_the_decision_rather_than_asking_for_it():
    """The teacher explains a decision; it never invents the events."""
    prompt = _teacher_prompt("Context: ...\n\nQuestion: crash or freeze?",
                             "Crash it. The cannon is in this wave.")
    assert "it is correct" in prompt
    assert "Crash it." in prompt
    assert "Never invent" in SYSTEM


def test_action_is_recovered_from_the_baseline_answer():
    assert _action_of("Hold the freeze. You are two levels down.") == "FREEZE"
    assert _action_of("Crash it. The wave is already moving in.") == "CRASH"
    assert _action_of("Crash and reset. You have enough gold.") == "RESET"


def test_resume_reads_completed_keys(tmp_path: Path):
    path = tmp_path / "synthetic.jsonl"
    path.write_text(
        json.dumps({"messages": [], "metadata": {"key": "abc"}}) + "\n"
        + "not valid json\n"
        + json.dumps({"messages": [], "metadata": {"key": "def"}}) + "\n",
        encoding="utf-8")
    assert load_completed(path) == {"abc", "def"}


def test_resume_on_missing_file_is_empty(tmp_path: Path):
    assert load_completed(tmp_path / "absent.jsonl") == set()


def test_teachable_archetypes_are_timeline_derived():
    assert "wave_decision" in TEACHABLE
    assert "champion_knowledge" not in TEACHABLE


# ---------------------------------------------------------------------------
# Chunking and IDF
# ---------------------------------------------------------------------------

def test_chunks_overlap_and_cover_the_text():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = list(_chunks(text, 64, 16))
    assert len(chunks) > 1
    assert chunks[0].split()[0] == "w0"
    # Consecutive chunks must share their overlap region.
    assert set(chunks[0].split()[-16:]) & set(chunks[1].split())


def test_chunking_rejects_a_useless_window():
    with pytest.raises(ValueError):
        list(_chunks("some text", 8, 2))


def test_document_frequencies_drop_ubiquitous_tokens():
    rows = [{"text": "the wave is frozen"} for _ in range(10)]
    rows.append({"text": "draven"})
    stats = _document_frequencies(rows)
    assert stats["n_docs"] == 11
    # "the" appears in 10 of 11 chunks and is above the 60% ceiling.
    assert "the" not in stats["df"]
    assert stats["df"]["draven"] == 1


def test_index_tokeniser_lowercases_and_drops_single_characters():
    assert _index_tokens("Baron Nashor a B") == ["baron", "nashor"]


def test_patch_preference_accepts_public_and_data_dragon_labels():
    assert _patch_matches("16.16", "26.16")
    assert _patch_matches("26.16", "16.16")
    assert not _patch_matches("16.15", "26.16")


# ---------------------------------------------------------------------------
# Wiki cleaning
# ---------------------------------------------------------------------------

def test_nested_templates_are_removed():
    raw = "{{ai|Q|{{sbc|Ahri}}}} Orb of Deception deals damage."
    cleaned = clean_wikitext(raw)
    assert "{{" not in cleaned and "}}" not in cleaned
    assert "Orb of Deception" in cleaned


def test_wiki_links_keep_their_display_text():
    assert "Baron" in clean_wikitext("[[Baron Nashor|Baron]] spawns at 20 minutes.")


def test_html_tables_and_scripts_are_stripped():
    html = "<script>bad()</script><p>Dragon spawns at 5:00.</p>"
    cleaned = clean_html(html)
    assert "bad()" not in cleaned
    assert "Dragon spawns" in cleaned


# ---------------------------------------------------------------------------
# Discovery scoring
# ---------------------------------------------------------------------------

def test_image_only_dataset_is_rejected_despite_a_matching_name():
    score = score_candidate("CyberHarem/jinx_leagueoflegends",
                            ["modality:image", "format:imagefolder"], "anime art")
    assert score < 0


def test_tabular_match_dataset_scores_well():
    score = score_candidate("gptilt/lol-esports-matches",
                            ["modality:tabular", "format:parquet"],
                            "League of Legends esports match data")
    assert score > 2.5


def test_unrelated_game_is_penalised():
    assert score_candidate("someone/dota2-matches", ["modality:tabular"],
                           "Dota 2 match data") < 2.5
