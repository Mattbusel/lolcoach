"""Deterministic, offline scoring for coaching answers.

An LLM judge is useful for "is this good coaching", but it is the wrong tool
for "is this true", and making it mandatory means the benchmark cannot run
without a paid API key. This module scores what can be checked mechanically,
against the same Data Dragon and wiki data the pipeline already downloaded:

``factual_accuracy``
    Verifiable claims are extracted from the answer -- item costs, ability
    cooldowns, objective timers, champion resource types -- and checked
    against the local knowledge base. An answer that asserts "Baron respawns
    every 5 minutes" is marked wrong because the wiki says six.

``hallucination_risk``
    Counts assertions the knowledge base contradicts, plus references to
    champions and items that do not exist, normalised by answer length. A
    confident wrong number costs more than an unhedged opinion.

``coaching_quality``
    Structural: does the answer commit to a decision, give a reason, name the
    trade-off, and stay concrete? These are rubric features that correlate
    with useful coaching and can be detected without a judge.

``consistency``
    Agreement between the answers to a question and its paraphrase, measured
    as the overlap of the *decisions* they recommend rather than of their
    words. Two answers that both say "freeze" are consistent even if phrased
    differently; one that says freeze and one that says crash is not.

When an API judge *is* configured, its scores are blended with these; the
deterministic components stay in the report either way so a regression in
factuality is never hidden behind a judge's prose preference.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..storage import Database

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

#: Objective timings the model must not get wrong, in seconds. Sourced from
#: the same table the feature layer uses, so they move together.
OBJECTIVE_FACTS: dict[str, dict[str, float]] = {
    "baron": {"first": 1500.0, "respawn": 360.0},
    "dragon": {"first": 300.0, "respawn": 300.0},
    "rift herald": {"first": 840.0},
    "void grub": {"first": 360.0},
    "grubs": {"first": 360.0},
}

#: Wave mechanics facts, in the units players state them in.
WAVE_FACTS = {
    "first wave": 65.0,        # seconds
    "wave interval": 30.0,     # seconds
    "cannon every": 3.0,       # waves, before 15 minutes
}


@dataclass
class KnowledgeBase:
    """Local facts used to check answers."""

    champions: dict[str, dict[str, Any]] = field(default_factory=dict)
    items: dict[str, int] = field(default_factory=dict)
    champion_names: set[str] = field(default_factory=set)
    item_names: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, db: Database) -> "KnowledgeBase":
        champions: dict[str, dict[str, Any]] = {}
        for row in db.query("SELECT name, resource, spells, stats FROM champions"):
            name = str(row["name"] or "")
            if not name:
                continue
            try:
                spells = json.loads(row["spells"] or "[]")
            except (TypeError, ValueError):
                spells = []
            champions[name.lower()] = {"resource": row["resource"], "spells": spells}
        items: dict[str, int] = {}
        for row in db.query("SELECT name, total_gold FROM items WHERE total_gold > 0"):
            name = str(row["name"] or "")
            if name:
                items[name.lower()] = int(row["total_gold"])
        return cls(champions=champions, items=items,
                   champion_names={c for c in champions},
                   item_names={i for i in items})


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

_MINUTES = re.compile(
    r"\b(baron|dragon|rift herald|herald|void grubs?|grubs?)\b[^.?!]{0,60}?"
    r"\b(?:every|after|in|at|takes)\s+(\d+(?:\.\d+)?)\s*(second|sec|minute|min)s?\b",
    re.IGNORECASE)

_ITEM_COST = re.compile(
    r"\b([A-Z][\w' ]{3,28}?)\s+costs?\s+(\d{3,5})\s*(?:gold|g)\b", re.IGNORECASE)

_FIRST_WAVE = re.compile(
    r"\bfirst\s+(?:minion\s+)?wave\b[^.?!]{0,40}?\b(\d{1,3})\s*(second|sec)s?\b",
    re.IGNORECASE)

_CANNON = re.compile(
    r"\bcannon\b[^.?!]{0,60}?\bevery\s+(\d)(?:rd|th|nd|st)?\s+wave\b", re.IGNORECASE)


@dataclass
class Claim:
    """One checkable assertion found in an answer."""

    kind: str
    subject: str
    value: float
    unit: str
    verdict: str = "unverified"   # correct | wrong | unverified
    expected: float | None = None


def extract_claims(answer: str) -> list[Claim]:
    """Pull the mechanically checkable assertions out of an answer."""
    claims: list[Claim] = []

    for match in _MINUTES.finditer(answer):
        subject = match.group(1).lower().rstrip("s")
        value = float(match.group(2))
        unit = "minute" if match.group(3).lower().startswith("min") else "second"
        claims.append(Claim("objective_timer", subject, value, unit))

    for match in _ITEM_COST.finditer(answer):
        claims.append(Claim("item_cost", match.group(1).strip().lower(),
                            float(match.group(2)), "gold"))

    for match in _FIRST_WAVE.finditer(answer):
        claims.append(Claim("first_wave", "first wave", float(match.group(1)), "second"))

    for match in _CANNON.finditer(answer):
        claims.append(Claim("cannon_cadence", "cannon", float(match.group(1)), "wave"))

    return claims


def verify_claims(claims: list[Claim], kb: KnowledgeBase) -> list[Claim]:
    """Mark each claim correct, wrong, or unverified against local data."""
    for claim in claims:
        if claim.kind == "objective_timer":
            key = {"herald": "rift herald", "grub": "void grub"}.get(
                claim.subject, claim.subject)
            facts = OBJECTIVE_FACTS.get(key)
            if not facts:
                continue
            seconds = claim.value * (60.0 if claim.unit == "minute" else 1.0)
            # A timer statement may refer to the first spawn or the respawn;
            # accept either, since both are true statements about the object.
            candidates = [v for v in facts.values()]
            claim.expected = min(candidates, key=lambda v: abs(v - seconds))
            claim.verdict = "correct" if any(
                abs(seconds - v) <= max(15.0, v * 0.08) for v in candidates) else "wrong"

        elif claim.kind == "item_cost":
            cost = kb.items.get(claim.subject)
            if cost is None:
                continue
            claim.expected = float(cost)
            claim.verdict = "correct" if abs(claim.value - cost) <= max(
                50.0, cost * 0.05) else "wrong"

        elif claim.kind == "first_wave":
            claim.expected = WAVE_FACTS["first wave"]
            claim.verdict = "correct" if abs(claim.value - 65.0) <= 5.0 else "wrong"

        elif claim.kind == "cannon_cadence":
            claim.expected = WAVE_FACTS["cannon every"]
            claim.verdict = "correct" if claim.value == 3 else "wrong"

    return claims


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

#: Phrases showing the answer committed to a decision.
_DECISION = re.compile(
    r"\b(crash|freeze|slow ?push|shove|recall|back off|reset|take it|give it up|"
    r"contest|trade|all[- ]in|rotate|ward|hold|wait|disengage|split)\b",
    re.IGNORECASE)
#: Phrases showing the answer justified the decision.
_BECAUSE = re.compile(
    r"\b(because|since|so that|which means|otherwise|that way|as a result|"
    r"the reason|if you)\b", re.IGNORECASE)
#: Phrases naming the cost of the decision.
_TRADEOFF = re.compile(
    r"\b(but|however|the (?:trade|downside|cost)|you give up|at the cost of|"
    r"risk|instead of|rather than)\b", re.IGNORECASE)
#: Calibrated uncertainty, which should be rewarded, not punished.
_HEDGE = re.compile(
    r"\b(likely|probably|usually|generally|if you can see|check|assuming|"
    r"depends on|roughly|about|estimate)\b", re.IGNORECASE)
#: Filler that signals a non-answer.
_WAFFLE = re.compile(
    r"\b(it depends entirely|there is no right answer|both are viable|"
    r"do what feels|play to your strengths|just play better)\b", re.IGNORECASE)


def coaching_quality(answer: str) -> float:
    """Structural quality score in ``[0, 1]``.

    Rewards a committed decision, an explicit reason, a named trade-off and
    concrete references; penalises pure waffle and extreme lengths. This does
    not judge whether the advice is *right* -- that is what factual accuracy
    and the optional LLM judge are for.
    """
    text = answer.strip()
    if not text:
        return 0.0
    words = text.split()
    n = len(words)

    score = 0.0
    if _DECISION.search(text):
        score += 0.34
    if _BECAUSE.search(text):
        score += 0.24
    if _TRADEOFF.search(text):
        score += 0.18
    if _HEDGE.search(text):
        score += 0.10
    # Concrete references: numbers, timings, or named entities.
    if re.search(r"\b\d+", text):
        score += 0.14

    if _WAFFLE.search(text):
        score -= 0.30
    # Very short answers cannot contain reasoning; very long ones stop being
    # coaching and start being an essay.
    if n < 25:
        score -= 0.25
    elif n > 400:
        score -= 0.15

    return max(0.0, min(1.0, score))


def factual_accuracy(answer: str, kb: KnowledgeBase) -> tuple[float, list[Claim]]:
    """Share of checkable claims that are right.

    Answers containing no checkable claim score 0.5 -- neither credited nor
    penalised -- because an answer that avoids specifics is neither accurate
    nor inaccurate, and scoring it 1.0 would reward vagueness.
    """
    claims = verify_claims(extract_claims(answer), kb)
    checked = [c for c in claims if c.verdict in ("correct", "wrong")]
    if not checked:
        return 0.5, claims
    correct = sum(1 for c in checked if c.verdict == "correct")
    return correct / len(checked), claims


def hallucination_risk(answer: str, kb: KnowledgeBase,
                       claims: Iterable[Claim] | None = None) -> float:
    """Estimated rate of unsupported specific assertions in ``[0, 1]``.

    Two contributions: claims the knowledge base positively contradicts, and
    capitalised entity names that look like champions or items but exist in
    neither table. Hedged statements are discounted, because "dragon is
    probably up around now" is not a hallucination.
    """
    claims = list(claims) if claims is not None else \
        verify_claims(extract_claims(answer), kb)
    wrong = sum(1 for c in claims if c.verdict == "wrong")

    # Entity check: multi-word capitalised tokens that name nothing we know.
    unknown = 0
    for candidate in set(re.findall(r"\b[A-Z][a-z]{2,}(?:'[A-Za-z]+)?\b", answer)):
        lowered = candidate.lower()
        if lowered in kb.champion_names or lowered in kb.item_names:
            continue
        if lowered in _COMMON_WORDS:
            continue
        # Only count it when used like a champion, to avoid flagging prose.
        if re.search(rf"\b(?:as|into|against|versus|vs\.?)\s+{re.escape(candidate)}\b",
                     answer):
            unknown += 1

    sentences = max(1, len(re.findall(r"[.!?]", answer)))
    raw = (wrong * 1.5 + unknown) / sentences
    if _HEDGE.search(answer):
        raw *= 0.75
    return max(0.0, min(1.0, raw))


#: Capitalised words that appear mid-sentence in normal coaching prose.
_COMMON_WORDS = {
    "you", "your", "the", "this", "that", "when", "where", "what", "how",
    "should", "flash", "teleport", "smite", "ignite", "barrier", "heal",
    "cleanse", "exhaust", "ghost", "baron", "dragon", "herald", "atakhan",
    "rift", "nashor", "grubs", "elder", "soul", "top", "mid", "bot", "jungle",
    "support", "river", "lane", "wave", "tower", "turret", "inhibitor", "nexus",
    "blue", "red", "purple", "summoner", "league", "legends", "riot", "crash",
    "freeze", "push", "recall", "reset", "trade", "ward", "vision", "gold",
    "experience", "level", "minion", "cannon", "melee", "caster", "siege",
    "early", "mid-game", "late", "game", "team", "enemy", "ally", "prio",
    "priority", "tempo", "macro", "micro", "scaling", "spike", "cooldown",
}


def _decisions(answer: str) -> set[str]:
    """Normalised set of actions an answer recommends."""
    found = set()
    for match in _DECISION.finditer(answer):
        token = match.group(0).lower().replace(" ", "").replace("-", "")
        found.add({"slowpush": "slowpush", "allin": "allin",
                   "backoff": "reset", "takeit": "contest",
                   "giveitup": "concede"}.get(token, token))
    return found


def consistency(answer_a: str, answer_b: str) -> float:
    """Agreement between two answers to the same situation.

    Measured on the recommended actions rather than the wording, using
    Jaccard overlap. Two answers with no detectable decision at all score 0.5,
    since there is nothing to disagree about.
    """
    a, b = _decisions(answer_a), _decisions(answer_b)
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_answer(answer: str, kb: KnowledgeBase) -> dict[str, float]:
    """All deterministic metrics for one answer."""
    accuracy, claims = factual_accuracy(answer, kb)
    return {
        "factual_accuracy": round(accuracy, 4),
        "coaching_quality": round(coaching_quality(answer), 4),
        "hallucination_risk": round(hallucination_risk(answer, kb, claims), 4),
        "checkable_claims": float(len([c for c in claims
                                       if c.verdict in ("correct", "wrong")])),
    }
