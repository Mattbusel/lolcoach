"""Question/answer generators over the Stage-2 feature tables.

Each generator turns one feature table into instruction examples. Two rules
shape all of them:

**Answers are built from derived reasoning, not from templates.** The feature
pass stores a ``rationale`` for every wave state, lane state, objective take
and death -- the actual clauses that justified the call, referencing the real
numbers from that game. The generators assemble those into prose. A template
that says "stabilise the lane and choose the safer option" teaches a model to
waffle; "you are 2 levels down with the wave frozen near your turret, so
holding it is what stops them snowballing" teaches it to reason.

**Questions are paraphrased deterministically.** Each example picks its
phrasing from a set of natural variants, keyed by a hash of the example id.
That gives the model varied surface forms for the same intent (and gives the
Stage-7 consistency metric real paraphrase pairs to score) while keeping
dataset generation reproducible.
"""

from __future__ import annotations

import hashlib
import re
import json
from typing import Any, Iterator, Sequence

from ..storage import Database

#: Wave state to plain English.
WAVE_PHRASE = {
    "FREEZE": "frozen near your side of the lane",
    "SLOW_PUSH": "slow pushing toward the enemy",
    "BIG_WAVE": "a large stacked wave",
    "CRASHING": "crashing into the enemy turret",
    "BOUNCING": "bouncing back toward you",
    "NEUTRAL": "roughly even",
}

#: Recommended action to an opening sentence.
ACTION_PHRASE = {
    "CRASH": "Crash it.",
    "FREEZE": "Hold the freeze.",
    "SLOW_PUSH": "Keep the slow push building.",
    "SHOVE_AND_ROAM": "Shove it in and use the window.",
    "RESET": "Crash and reset.",
    "NEUTRAL": "Match the wave and look for a trade.",
}

PRIORITY_PHRASE = {
    "HARD_PRIO": "you have hard priority",
    "PRIO": "you have priority",
    "EVEN": "priority is even",
    "NO_PRIO": "you do not have priority",
}


def pick(options: Sequence[str], key: str) -> str:
    """Deterministically choose one phrasing for an example."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def reasons_of(raw: Any) -> list[str]:
    """Decode a persisted rationale column into a list of clauses."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    return []


def join_reasons(reasons: Sequence[str], limit: int = 3) -> str:
    """Turn reason clauses into a sentence or two."""
    picked = [r for r in reasons if r][:limit]
    if not picked:
        return ""
    if len(picked) == 1:
        return _sentence(picked[0])
    head = ", and ".join([", ".join(picked[:-1]), picked[-1]]) if len(picked) > 2 \
        else " and ".join(picked)
    return _sentence(head)


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def clock(ts_ms: Any) -> str:
    """``720000`` -> ``12:00``."""
    total = int(ts_ms or 0) // 1000
    return f"{total // 60}:{total % 60:02d}"


def field(row: Any, key: str, default: Any = None) -> Any:
    """Read a possibly-absent column from a ``sqlite3.Row``."""
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


#: Clauses that describe *how* an estimate was made rather than what to do.
#: They belong in the evidence, not in the advice.
_DIAGNOSTIC_MARKERS = (
    "laners in lane", "laner is in lane", "lane is empty", "extrapolated",
    "no information yet", "minions unaccounted", "minion counts are even",
    "no direct lane opponent",
)


def _is_diagnostic(clause: str) -> bool:
    """True for clauses that explain the inference rather than the decision."""
    lowered = clause.lower()
    return any(marker in lowered for marker in _DIAGNOSTIC_MARKERS)


def _position_sentence(position: float) -> str:
    """Describe where along the lane the wave sits, in words a player uses."""
    pct = abs(position) * 100
    if pct < 12:
        return "It is sitting near the middle of the lane."
    side = "toward the enemy turret" if position > 0 else "back toward your own turret"
    return f"The estimate puts it about {pct:.0f}% of the way {side}."


def confidence_caveat(confidence: float) -> str:
    """A calibrated hedge, matched to how good the estimate actually is.

    Timeline frames are one minute apart, so the wave estimate is genuinely
    uncertain. Saying so at low confidence -- and *not* saying so at high
    confidence -- is what keeps the model from either bluffing or hedging
    everything into uselessness.
    """
    if confidence >= 0.7:
        return ""
    if confidence >= 0.5:
        return " Check the actual minion count before you commit, since this read comes from one-minute snapshots."
    return " This is a low-confidence read from sparse data, so trust what you can see on screen over it."


# ---------------------------------------------------------------------------
# Wave decisions: the flagship archetype
# ---------------------------------------------------------------------------

WAVE_QUESTIONS = [
    "Should I crash this wave or freeze it here?",
    "Do I crash or hold this wave?",
    "What should I do with this wave right now?",
    "Should I be shoving this in or setting up a freeze?",
]

WAVE_READ_QUESTIONS = [
    "Is this wave slow pushing?",
    "What is this wave doing right now?",
    "How would you read this wave state?",
    "Is the wave going to push toward me or away from me?",
]


def wave_context(row: Any) -> str:
    """The structured situation block shown to the model."""
    lines = [
        "Context:",
        f"Champion: {field(row, 'champion', 'unknown')} ({field(row, 'lane', 'unknown')})",
    ]
    opponent = field(row, "opponent")
    if opponent:
        lines.append(f"Enemy laner: {opponent}")
    lines.append(f"Time: {clock(field(row, 'ts_ms', 0))}")

    gold_diff = field(row, "gold_diff")
    if gold_diff is not None:
        lines.append(f"Gold difference: {int(gold_diff):+d}")
    cs_diff = field(row, "cs_diff")
    if cs_diff is not None:
        lines.append(f"CS difference: {int(cs_diff):+d}")
    level_diff = field(row, "level_diff")
    if level_diff is not None:
        lines.append(f"Level difference: {int(level_diff):+d}")

    hp = field(row, "hp_pct")
    opp_hp = field(row, "opp_hp_pct")
    if hp is not None:
        lines.append(f"Health: {round(float(hp) * 100)}% vs {round(float(opp_hp or 1) * 100)}%")

    state = str(field(row, "wave_state", "NEUTRAL"))
    conf = float(field(row, "wave_confidence", 0.5) or 0.5)
    lines.append(f"Wave state (inferred, confidence {conf:.2f}): {WAVE_PHRASE.get(state, state.lower())}")

    size = field(row, "size_diff")
    if size is not None and float(size) >= 6:
        lines.append(f"Estimated surplus minions: {int(float(size))}")
    if field(row, "cannon"):
        lines.append("Cannon minion: present in this wave")

    lines.append(f"Lane priority: {field(row, 'priority', 'EVEN')}")
    lines.append(f"Trade window: {field(row, 'trade_window', 'EVEN')}")
    plates = field(row, "plates")
    if plates:
        lines.append(f"Turret plates taken in this lane: {int(plates)}")
    return "\n".join(lines)


def wave_examples(db: Database, min_confidence: float) -> Iterator[tuple]:
    """Yield ``(archetype, key, user, assistant, confidence, match_id, patch)``."""
    rows = db.iter_query(
        "SELECT l.*, p.champion, p.team_id, m.patch, "
        "       w.state AS wave_state, w.recommended, w.confidence AS wave_confidence, "
        "       w.rationale AS wave_rationale, w.size_diff, w.cannon, w.position, "
        "       opp.champion AS opponent "
        "FROM lane_states l "
        "JOIN participants p ON p.match_id=l.match_id AND p.participant_id=l.participant_id "
        "JOIN matches m ON m.match_id=l.match_id "
        "JOIN wave_states w ON w.match_id=l.match_id AND w.ts_ms=l.ts_ms "
        "     AND w.lane=l.lane AND w.team_id=p.team_id "
        "LEFT JOIN participants opp ON opp.match_id=l.match_id AND opp.role=p.role "
        "     AND opp.team_id<>p.team_id "
        "WHERE w.confidence>=? AND l.ts_ms>=120000 "
        "ORDER BY l.match_id, l.ts_ms", (min_confidence,), batch=2000)

    for row in rows:
        key = f"{row['match_id']}:{row['participant_id']}:{row['ts_ms']}"
        wave_conf = float(field(row, "wave_confidence", 0.5) or 0.5)
        action = str(field(row, "recommended", "NEUTRAL"))
        context = wave_context(row)

        # The feature pass records two kinds of clause: coaching reasons that
        # belong in the answer, and diagnostic factors ("both laners in lane")
        # that describe how the estimate was made. Only the former reads as
        # advice, so they are separated here rather than concatenated.
        all_reasons = reasons_of(field(row, "wave_rationale"))
        wave_reasons = [r for r in all_reasons if not _is_diagnostic(r)]
        lane_reasons = [r for r in reasons_of(field(row, "rationale"))
                        if not _is_diagnostic(r)]

        # The decision, then why, then a calibrated caveat.
        opening = ACTION_PHRASE.get(action, ACTION_PHRASE["NEUTRAL"])
        body = join_reasons(wave_reasons or lane_reasons, limit=2)
        extra = ""
        priority = str(field(row, "priority", "EVEN"))
        if priority in PRIORITY_PHRASE and priority != "EVEN":
            extra = f" Right now {PRIORITY_PHRASE[priority]}, which is what makes this the cheap option."
        answer = f"{opening} {body}{extra}{confidence_caveat(wave_conf)}".strip()

        yield ("wave_decision", key,
               f"{context}\n\nQuestion: {pick(WAVE_QUESTIONS, key)}",
               answer, min(float(field(row, "confidence", 0.5) or 0.5), wave_conf),
               row["match_id"], field(row, "patch"))

        # A second, cheaper archetype: read the wave rather than act on it.
        state = str(field(row, "wave_state", "NEUTRAL"))
        read_key = key + ":read"
        phrase = WAVE_PHRASE.get(state, state.lower())
        detail = join_reasons(wave_reasons, limit=1)
        position = float(field(row, "position", 0.0) or 0.0)
        answer_read = (
            f"It is {phrase}. {detail} {_position_sentence(position)}"
            f"{confidence_caveat(wave_conf)}").strip()
        yield ("wave_read", read_key,
               f"{context}\n\nQuestion: {pick(WAVE_READ_QUESTIONS, read_key)}",
               answer_read, wave_conf, row["match_id"], field(row, "patch"))


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

TRADE_QUESTIONS = [
    "Is this a good time to trade?",
    "Can I go for an all-in here?",
    "Should I look for a trade or back off?",
    "Do I win a fight with them right now?",
]


def trade_examples(db: Database) -> Iterator[tuple]:
    rows = db.iter_query(
        "SELECT l.*, p.champion, m.patch, opp.champion AS opponent "
        "FROM lane_states l "
        "JOIN participants p ON p.match_id=l.match_id AND p.participant_id=l.participant_id "
        "JOIN matches m ON m.match_id=l.match_id "
        "LEFT JOIN participants opp ON opp.match_id=l.match_id AND opp.role=p.role "
        "     AND opp.team_id<>p.team_id "
        "WHERE l.trade_window IS NOT NULL AND l.ts_ms>=120000 "
        "ORDER BY l.match_id, l.ts_ms", batch=2000)

    for row in rows:
        key = f"{row['match_id']}:{row['participant_id']}:{row['ts_ms']}:trade"
        window = str(field(row, "trade_window", "EVEN"))
        all_in = bool(field(row, "all_in", 0))
        reasons = reasons_of(field(row, "rationale"))

        if all_in:
            opening = "Yes, this is your all-in window."
        elif window == "FAVOURABLE":
            opening = "Short trades favour you, but do not commit to an all-in."
        elif window == "UNFAVOURABLE":
            opening = "No. Do not take an extended trade here."
        else:
            opening = "It is close to even, so only trade with a cooldown advantage."

        body = join_reasons(reasons, limit=2)
        hp = float(field(row, "hp_pct", 1.0) or 1.0)
        opp_hp = float(field(row, "opp_hp_pct", 1.0) or 1.0)
        health_note = (f" You are at {round(hp * 100)}% against their "
                       f"{round(opp_hp * 100)}%, which decides who wins the exchange "
                       f"if it goes long.")
        answer = f"{opening} {body}{health_note}".strip()

        context = wave_context(row)
        yield ("trade_window", key, f"{context}\n\nQuestion: {pick(TRADE_QUESTIONS, key)}",
               answer, float(field(row, "confidence", 0.5) or 0.5),
               row["match_id"], field(row, "patch"))


# ---------------------------------------------------------------------------
# Recalls
# ---------------------------------------------------------------------------

RECALL_QUESTIONS = [
    "Should I recall after this trade?",
    "Is this a good back timing?",
    "Do I reset now or keep farming?",
    "Should I go back here?",
]


def recall_examples(db: Database) -> Iterator[tuple]:
    rows = db.iter_query(
        "SELECT r.*, p.champion, p.role, m.patch "
        "FROM recalls r "
        "JOIN participants p ON p.match_id=r.match_id AND p.participant_id=r.participant_id "
        "JOIN matches m ON m.match_id=r.match_id "
        "ORDER BY r.match_id, r.ts_ms", batch=2000)

    for row in rows:
        key = f"{row['match_id']}:{row['participant_id']}:{row['ts_ms']}:recall"
        good = bool(field(row, "was_good", 0))
        gold = int(field(row, "gold_spent", 0) or 0)
        state = str(field(row, "wave_state", "NEUTRAL") or "NEUTRAL")
        reason = str(field(row, "reason", "")).rstrip(".")

        if good:
            opening = "Yes, this is a good back."
        else:
            opening = "This is a bad back timing."
        wave_note = f" The wave is {WAVE_PHRASE.get(state, state.lower())}"
        if state in ("CRASHING", "BOUNCING"):
            wave_note += ", so you lose nothing by leaving now"
        elif state == "FREEZE":
            wave_note += ", so leaving hands them a free reset and the freeze breaks"
        elif state in ("SLOW_PUSH", "BIG_WAVE"):
            wave_note += ", so wait for it to crash before you go or you give up the whole wave"
        wave_note += "."
        gold_note = (f" You have {gold} gold to spend, which is worth the trip."
                     if gold >= 800 else
                     f" You are only spending {gold} gold, which usually is not worth "
                     f"the walk back unless you are low.")
        answer = f"{opening}{wave_note}{gold_note} {_sentence(reason)}".strip()

        context = (f"Context:\nChampion: {field(row, 'champion')} "
                   f"({field(row, 'role')})\nTime: {clock(field(row, 'ts_ms'))}\n"
                   f"Wave state (inferred): {WAVE_PHRASE.get(state, state.lower())}\n"
                   f"Gold spent on this back: {gold}")
        yield ("recall", key, f"{context}\n\nQuestion: {pick(RECALL_QUESTIONS, key)}",
               answer, 0.58, row["match_id"], field(row, "patch"))


# ---------------------------------------------------------------------------
# Jungle
# ---------------------------------------------------------------------------

JUNGLE_QUESTIONS = [
    "Where is their jungler likely to be next?",
    "Where should I expect the enemy jungler?",
    "Can you predict their jungle path from here?",
    "Which side of the map is their jungler moving to?",
]


def jungle_examples(db: Database) -> Iterator[tuple]:
    rows = db.iter_query(
        "SELECT j.*, p.champion, p.team_id, m.patch "
        "FROM jungle_paths j "
        "JOIN participants p ON p.match_id=j.match_id AND p.participant_id=j.participant_id "
        "JOIN matches m ON m.match_id=j.match_id "
        "WHERE j.camp IS NOT NULL OR j.predicted_next IS NOT NULL "
        "ORDER BY j.match_id, j.ts_ms", batch=2000)

    for row in rows:
        key = f"{row['match_id']}:{row['participant_id']}:{row['ts_ms']}:jungle"
        camp = field(row, "camp")
        predicted = field(row, "predicted_next")
        conf = float(field(row, "confidence", 0.4) or 0.4)
        quadrant = str(field(row, "quadrant", "UNKNOWN")).replace("_", " ").lower()

        if not predicted:
            continue
        pretty = str(predicted).replace("_", " ").lower()
        opening = (f"Most likely {pretty}, at roughly {conf * 100:.0f}% confidence "
                   f"based on how junglers path from here in this corpus.")
        detail = ""
        if camp:
            detail = (f" They were last seen clearing "
                      f"{str(camp).replace('_', ' ').lower()}, and the usual "
                      f"continuation from that camp is toward {pretty}.")
        advice = (" Ward the entrance on that side before you push past the river, "
                  "and treat this as a probability rather than vision.")
        answer = (opening + detail + advice).strip()

        context = (f"Context:\nEnemy jungler: {field(row, 'champion')}\n"
                   f"Time: {clock(field(row, 'ts_ms'))}\n"
                   f"Last known area: {quadrant}\n"
                   f"Last camp seen: {str(camp).replace('_', ' ').lower() if camp else 'unknown'}")
        yield ("jungle_prediction", key,
               f"{context}\n\nQuestion: {pick(JUNGLE_QUESTIONS, key)}",
               answer, conf, row["match_id"], field(row, "patch"))


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

OBJECTIVE_QUESTIONS = [
    "Should we contest this objective?",
    "How do I play around this objective?",
    "What is the right setup for this objective?",
    "Do we take this or give it up?",
]


def objective_examples(db: Database) -> Iterator[tuple]:
    rows = db.iter_query(
        "SELECT o.*, m.patch FROM objective_events o "
        "JOIN matches m ON m.match_id=o.match_id "
        "ORDER BY o.match_id, o.ts_ms", batch=2000)

    for row in rows:
        key = f"{row['match_id']}:{row['objective']}:{row['ts_ms']}"
        name = str(field(row, "objective", "")).lower().replace("_", " ")
        setup = float(field(row, "setup_score", 0.5) or 0.5)
        contested = bool(field(row, "contested", 0))
        reasons = reasons_of(field(row, "rationale"))
        next_spawn = field(row, "next_spawn_ms")

        if setup >= 0.65:
            opening = f"Take it. The setup for this {name} is good."
        elif setup <= 0.35:
            opening = f"Give it up. You are not set up for this {name}."
        else:
            opening = f"This {name} is a coinflip as things stand."

        body = join_reasons(reasons, limit=3)
        timing = ""
        if next_spawn:
            timing = (f" It comes back around {clock(next_spawn)}, so start moving "
                      f"vision and lane priority to that side about a minute before.")
        else:
            timing = " It does not respawn, so the next one is a different objective entirely."
        contest_note = (" Because it is contested, do not start it without knowing "
                        "where their jungler and mid are."
                        if contested else "")
        answer = f"{opening} {body}{contest_note}{timing}".strip()

        context = (f"Context:\nTime: {clock(field(row, 'ts_ms'))}\n"
                   f"Objective: {name}\n"
                   f"Setup score (0-1): {setup:.2f}\n"
                   f"Contested: {'yes' if contested else 'no'}\n"
                   f"Vision advantage at the pit: {field(row, 'vision_edge', 0)}")
        yield ("objective_timing", key,
               f"{context}\n\nQuestion: {pick(OBJECTIVE_QUESTIONS, key)}",
               answer, 0.7, row["match_id"], field(row, "patch"))


# ---------------------------------------------------------------------------
# Deaths
# ---------------------------------------------------------------------------

DEATH_QUESTIONS = [
    "What caused this death and how do I avoid it?",
    "Why did I die here?",
    "What was the mistake in this death?",
    "How should I have played this differently?",
]

CAUSE_FIX = {
    "OVEREXTENDED": "Push only as far as your vision and your escape allow; if you cannot see the enemy jungler, the far half of the lane is not yours.",
    "CAUGHT_ROTATING": "Rotate through your own jungle or along a warded path, and move before the wave forces you to, not after.",
    "LOST_TRADE": "Check the health difference before you commit; you needed the trade to start from a better position than that.",
    "OUTNUMBERED": "Count the enemies you can actually see before stepping up. Missing two of them is the whole mistake.",
    "DIVED": "Once they can dive you, give up the wave and stand behind the turret rather than trying to farm through it.",
    "NO_VISION": "Review the recorded ward coverage before taking this position again. "
                 "The timeline did not record a nearby active friendly ward, but it cannot establish enemy vision.",
    "FIGHT_DEATH": "In the fight itself, hold your spacing until their engage is used, then step in.",
    "UNCLEAR": "There is not enough in the timeline to be certain here, so review the replay around this timestamp yourself.",
}


def death_examples(db: Database) -> Iterator[tuple]:
    rows = db.iter_query(
        "SELECT d.*, p.champion, m.patch FROM deaths d "
        "JOIN participants p ON p.match_id=d.match_id AND p.participant_id=d.victim_id "
        "JOIN matches m ON m.match_id=d.match_id "
        "ORDER BY d.match_id, d.ts_ms", batch=2000)

    for row in rows:
        key = f"{row['match_id']}:{row['victim_id']}:{row['ts_ms']}:death"
        cause = str(field(row, "cause", "UNCLEAR"))
        reasons = reasons_of(field(row, "rationale"))
        zone = str(field(row, "zone", "")).replace("_", " ").lower()

        body = join_reasons(reasons, limit=2)
        fix = CAUSE_FIX.get(cause, CAUSE_FIX["UNCLEAR"])
        answer = f"{body} {fix}".strip()

        outnumbered = int(field(row, "outnumbered", 0) or 0)
        context = (f"Context:\nChampion: {field(row, 'champion')}\n"
                   f"Time: {clock(field(row, 'ts_ms'))}\n"
                   f"Where: {zone}\n"
                   f"Numbers: {'outnumbered by ' + str(outnumbered) if outnumbered > 0 else 'even or ahead in numbers'}\n"
                   f"Vision covering the spot: {'yes' if field(row, 'vision_nearby') else 'no'}")
        yield ("death_review", key,
               f"{context}\n\nQuestion: {pick(DEATH_QUESTIONS, key)}",
               answer, 0.62, row["match_id"], field(row, "patch"))


# ---------------------------------------------------------------------------
# Rotations and fights
# ---------------------------------------------------------------------------

ROTATION_QUESTIONS = [
    "Where should I go after this?",
    "Is this the right rotation?",
    "Should I rotate here or keep pushing?",
    "What is the correct move on the map now?",
]

FIGHT_QUESTIONS = [
    "How should I position in this fight?",
    "Where do I stand in this teamfight?",
    "How do I play this fight from here?",
    "What is the positioning mistake to avoid here?",
]


def rotation_examples(db: Database) -> Iterator[tuple]:
    rows = db.iter_query(
        "SELECT r.*, p.champion, p.role, m.patch FROM rotations r "
        "JOIN participants p ON p.match_id=r.match_id AND p.participant_id=r.participant_id "
        "JOIN matches m ON m.match_id=r.match_id "
        "ORDER BY r.match_id, r.ts_ms", batch=2000)

    for row in rows:
        key = f"{row['match_id']}:{row['participant_id']}:{row['ts_ms']}:rot"
        purpose = str(field(row, "purpose", "SIDE_LANE")).replace("_", " ").lower()
        frm = str(field(row, "from_zone", "")).replace("_", " ").lower()
        to = str(field(row, "to_zone", "")).replace("_", " ").lower()
        correct = field(row, "was_correct")
        tempo = float(field(row, "tempo_gain", 0.0) or 0.0)

        if correct == 1:
            verdict = "This rotation worked."
        elif correct == 0:
            verdict = "This rotation did not pay off."
        else:
            verdict = "This is a neutral rotation."
        detail = (f" You moved from {frm} to {to} as a {purpose} play"
                  f"{', and your team got the next objective or fight out of it' if tempo > 0 else ''}.")
        rule = (" Before you leave, make sure your wave is either crashed or "
                "frozen; rotating off a wave that is still walking into you is "
                "how you lose plates and levels for nothing.")
        answer = (verdict + detail + rule).strip()

        context = (f"Context:\nChampion: {field(row, 'champion')} ({field(row, 'role')})\n"
                   f"Time: {clock(field(row, 'ts_ms'))}\n"
                   f"Moving: {frm} to {to}\nPurpose: {purpose}")
        yield ("rotation", key, f"{context}\n\nQuestion: {pick(ROTATION_QUESTIONS, key)}",
               answer, 0.55, row["match_id"], field(row, "patch"))


def fight_examples(db: Database) -> Iterator[tuple]:
    rows = db.iter_query(
        "SELECT f.*, m.patch FROM fights f JOIN matches m ON m.match_id=f.match_id "
        "ORDER BY f.match_id, f.fight_id", batch=2000)

    for row in rows:
        key = f"{row['match_id']}:{row['fight_id']}:fight"
        zone = str(field(row, "zone", "")).replace("_", " ").lower()
        trigger = str(field(row, "trigger", "")).replace("_", " ").lower()
        ka = int(field(row, "kills_a", 0) or 0)
        kb = int(field(row, "kills_b", 0) or 0)
        notes = field(row, "positioning")
        try:
            note_map = json.loads(notes) if notes else {}
        except (TypeError, ValueError):
            note_map = {}

        lessons = [v for v in note_map.values() if "why they were focused" in v
                   or "separated" in v or "never reached" in v]
        body = join_reasons(lessons, limit=2) or (
            "Spacing held up on both sides, so this came down to cooldowns rather "
            "than positioning.")
        answer = (f"This fight happened around {zone} over {trigger}, and it ended "
                  f"{ka}-{kb}. {body} The rule that generalises: as a backline "
                  f"champion your job is to stay at the edge of your longest range "
                  f"and only step in once their engage is spent; as a front-liner "
                  f"you have to be the one who touches them first.").strip()

        context = (f"Context:\nTime: {clock(field(row, 'start_ms'))}\n"
                   f"Where: {zone}\nWhat it was over: {trigger}\n"
                   f"Players involved: {field(row, 'team_a_count', 0)} vs {field(row, 'team_b_count', 0)}")
        yield ("teamfight", key, f"{context}\n\nQuestion: {pick(FIGHT_QUESTIONS, key)}",
               answer, 0.63, row["match_id"], field(row, "patch"))


# ---------------------------------------------------------------------------
# Matchups, aggregated across the corpus
# ---------------------------------------------------------------------------

MATCHUP_QUESTIONS = [
    "How do I play {a} into {b}?",
    "Any advice for the {a} vs {b} matchup?",
    "What should I know about {a} against {b}?",
    "Is {a} into {b} a good or bad matchup?",
]

#: A matchup needs this many observed games before it is worth an example.
MIN_MATCHUP_GAMES = 8


#: A community answer needs this much substance to be worth training on.
MIN_ANSWER_CHARS = 320
MIN_ANSWER_SCORE = 8

_ANSWER_BLOCK = re.compile(r"^Answer (\d+) \(score (-?\d+)\):\s*$", re.MULTILINE)

#: Subreddits whose threads are actually people asking for help. The general
#: r/leagueoflegends is excluded here: it is dominated by esports match
#: threads and news, which produce fluent but useless training targets.
COACHING_SUBS = ("summonerschool", "loltheorycraft", "jungle_mains",
                 "supportlol", "adcmains", "topmains", "midlaner")

#: Thread shapes that are never coaching, however upvoted.
_NOT_COACHING = re.compile(
    r"post-?match|match thread|hupu ratings|roster|signing|patch \d+\.\d+ notes|"
    r"worlds|msi \d{4}|esports world cup|group [a-d]|lck|lpl|lec|lcs\b|"
    r"vs\.?\s+\w+.*\b(spoiler|discussion)\b", re.IGNORECASE)

#: A coaching question asks something.
_QUESTION_SHAPE = re.compile(
    r"\?|^(how|why|when|what|should|is|are|can|does|do|which|where)\b",
    re.IGNORECASE)


def parse_reddit_document(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Split a stored Reddit document into its question and scored answers.

    The Reddit connector stores each post as ``Question (...)`` followed by
    ``Answer N (score M):`` blocks, so this is the inverse of that format.
    """
    blocks = list(_ANSWER_BLOCK.finditer(text))
    question = text[: blocks[0].start()].strip() if blocks else text.strip()
    answers: list[tuple[int, str]] = []
    for i, match in enumerate(blocks):
        end = blocks[i + 1].start() if i + 1 < len(blocks) else len(text)
        body = text[match.end():end].strip()
        if body:
            answers.append((int(match.group(2)), body))
    return question, answers


def reddit_qa_examples(db: Database) -> Iterator[tuple]:
    """Turn upvoted community answers into instruction pairs.

    This is the one archetype whose targets are written by humans rather than
    derived by the pipeline. A highly-upvoted r/summonerschool answer to a
    concrete question is real coaching, and the score gives a usable quality
    signal. Only the best answer per thread is used, and only when it clears
    both a score and a length bar, because the median Reddit reply is a joke.
    """
    rows = db.iter_query(
        "SELECT doc_id, title, url, text, score FROM documents "
        "WHERE source='reddit' ORDER BY score DESC", batch=500)

    for row in rows:
        doc_id = str(field(row, "doc_id", ""))
        # doc_id is ``reddit:<subreddit>:<post id>``.
        parts = doc_id.split(":")
        sub = parts[1].lower() if len(parts) > 2 else ""
        if sub not in COACHING_SUBS:
            continue

        title = str(field(row, "title", ""))
        if _NOT_COACHING.search(title):
            continue

        text = str(field(row, "text", ""))
        question_block, answers = parse_reddit_document(text)
        if not answers:
            continue
        best_score, best = max(answers, key=lambda a: a[0])
        if best_score < MIN_ANSWER_SCORE or len(best) < MIN_ANSWER_CHARS:
            continue

        # Strip the connector's own header so the model sees a plain question.
        question = re.sub(r"^Question \([^)]*\):\s*", "", question_block).strip()
        if len(question) < 40 or not _QUESTION_SHAPE.search(title):
            continue

        key = f"reddit:{field(row, 'doc_id')}"
        answer = " ".join(best.split())
        if len(answer) > 2400:
            answer = answer[:2400].rsplit(" ", 1)[0] + "..."

        yield ("community_qa", key, question, answer,
               min(0.9, 0.45 + best_score / 200.0), None, None)


#: Wiki pages that carry the mechanics a coaching model must not get wrong.
MECHANICS_QUESTIONS = {
    "Minion": "How do minion waves work?",
    "Siege minion": "How do cannon minions work and when do they spawn?",
    "Experience": "How does experience work in lane?",
    "Gold": "How does gold generation work?",
    "Turret": "How do turrets and turret plating work?",
    "Baron Nashor": "What does Baron Nashor do and when does it spawn?",
    "Rift Herald": "What does the Rift Herald do?",
    "Dragon": "How do dragons and dragon soul work?",
    "Void Grub": "What do void grubs do?",
    "Atakhan": "What is Atakhan and how does it work?",
    "Ward": "How does warding and vision work?",
    "Jungling": "How does jungle clearing and camp respawn work?",
}

#: Sentences worth keeping from a wiki page: the ones stating a rule.
_RULE_SHAPE = re.compile(
    r"\b(spawn|respawn|every|seconds|minutes|grants?|deals?|health|damage|"
    r"gold|experience|increases?|reduces?|stacks?|bonus|cannot|only)\b",
    re.IGNORECASE)


def wiki_mechanics_examples(db: Database, max_chars: int = 1400) -> Iterator[tuple]:
    """Turn the wiki's mechanics pages into grounded factual answers.

    These pages are the reference for the rules a coach reasons from -- wave
    composition, camp timers, plate gold, objective effects. The answer is
    assembled from the page's rule-bearing sentences rather than its lore, so
    the model learns numbers and conditions instead of flavour text.
    """
    titles = list(MECHANICS_QUESTIONS)
    placeholders = ",".join("?" * len(titles))
    rows = db.query(
        f"SELECT title, text, url, patch FROM documents "
        f"WHERE source='wiki' AND title IN ({placeholders})", titles)

    for row in rows:
        title = str(field(row, "title", ""))
        question = MECHANICS_QUESTIONS.get(title)
        if not question:
            continue
        text = " ".join(str(field(row, "text", "")).split())
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]
        kept: list[str] = []
        used = 0
        for sentence in sentences:
            if len(sentence) < 40 or not _RULE_SHAPE.search(sentence):
                continue
            kept.append(sentence)
            used += len(sentence)
            if used >= max_chars:
                break
        if len(kept) < 2:
            continue
        answer = " ".join(kept)
        yield ("mechanics", f"wiki-mechanics:{title}", question, answer,
               0.95, None, field(row, "patch"))


def matchup_examples(db: Database) -> Iterator[tuple]:
    """Aggregate real head-to-head outcomes into matchup advice.

    Unlike the other generators this one is corpus-wide: it counts every game
    where champion A faced champion B in the same role and reports the actual
    win rate and average gold swing. Matchups with too few games are skipped
    rather than guessed at, which is why the threshold exists.
    """
    rows = db.query(
        "SELECT a.champion AS champ_a, b.champion AS champ_b, a.role AS role, "
        "       COUNT(*) AS games, "
        "       SUM(a.win) AS wins, "
        "       AVG(a.gold_earned - b.gold_earned) AS gold_delta, "
        "       AVG(a.cs - b.cs) AS cs_delta "
        "FROM participants a "
        "JOIN participants b ON b.match_id=a.match_id AND b.role=a.role "
        "     AND b.team_id<>a.team_id "
        "WHERE a.champion IS NOT NULL AND b.champion IS NOT NULL AND a.role<>'' "
        "GROUP BY a.champion, b.champion, a.role "
        "HAVING games >= ? ORDER BY games DESC LIMIT 20000", (MIN_MATCHUP_GAMES,))

    for row in rows:
        a, b = row["champ_a"], row["champ_b"]
        role = str(row["role"] or "").upper()
        games = int(row["games"])
        wins = int(row["wins"] or 0)
        wr = wins / games if games else 0.5
        gold = float(row["gold_delta"] or 0.0)
        cs = float(row["cs_delta"] or 0.0)
        key = f"matchup:{a}:{b}:{role}"

        if wr >= 0.56:
            verdict = f"{a} is favoured into {b} here"
        elif wr <= 0.44:
            verdict = f"{a} is on the losing side of this matchup"
        else:
            verdict = f"{a} into {b} is roughly even"

        answer = (
            f"Across {games} games of {a} versus {b} in the {role.lower()} role in "
            f"this dataset, {verdict}: {wr * 100:.0f}% win rate, with an average "
            f"gold difference of {gold:+.0f} and {cs:+.1f} CS by the end of the game. "
            f"{'Play for the early lead, because the numbers say you get one and the game follows it.' if wr >= 0.56 else ''}"
            f"{'Play safe, give up the wave when it is not yours, and look to scale or to get help from your jungler rather than winning it alone.' if wr <= 0.44 else ''}"
            f"{'Neither side gets a free lane, so the matchup is decided by wave state and jungle proximity rather than by the pick.' if 0.44 < wr < 0.56 else ''}"
        ).strip()

        question = pick(MATCHUP_QUESTIONS, key).format(a=a, b=b)
        context = (f"Context:\nYour champion: {a}\nEnemy champion: {b}\n"
                   f"Role: {role}")
        yield ("matchup", key, f"{context}\n\nQuestion: {question}",
               answer, min(0.9, 0.4 + games / 100.0), None, None)
