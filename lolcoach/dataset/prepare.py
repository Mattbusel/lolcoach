"""Build deduplicated chat-format coaching examples from local evidence.

Generated labels are deliberately evidence-bound.  Each example stores a
source match/document identifier, patch, and confidence so it can be audited
or filtered later; the model is never trained to present a timeline inference
as an observed fact.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..config import Config
from ..features import derive_all
from ..ingest import ingest_all
from ..logging_utils import get_logger
from ..storage import Database
from . import generators as gen

log = get_logger("dataset")
SYSTEM = (
    "You are a League of Legends coach. Give an actionable, concise answer, "
    "distinguish known facts from uncertainty, and explain the relevant tradeoff."
)


@dataclass(frozen=True)
class TrainingExample:
    archetype: str
    key: str
    user: str
    assistant: str
    metadata: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ],
            "metadata": {"archetype": self.archetype, **self.metadata},
        }


@dataclass
class DatasetResult:
    examples: int
    train: int
    validation: int
    test: int
    skipped_duplicates: int
    by_archetype: dict[str, int]
    paths: dict[str, Path]


def prepare_dataset(
    cfg: Config,
    *,
    limit: int | None = None,
    refresh_features: bool = True,
) -> DatasetResult:
    """Run local normalization/feature derivation and emit split JSONL files.

    ``limit`` bounds source examples while retaining deterministic split
    assignment, useful for a quick smoke run.  The full cap lives in
    ``dataset.max_examples`` and is applied after duplicate suppression.
    """
    if refresh_features:
        ingest_all(cfg, limit=limit)
        derive_all(cfg, limit=limit)
    db = Database(cfg.paths.db)
    maximum = min(cfg.dataset.max_examples, limit) if limit is not None else cfg.dataset.max_examples
    stream = itertools.chain(
        _synthetic_examples(cfg),      # teacher-written answers rank first
        _all_examples(db, min_wave_confidence=cfg.dataset.min_wave_confidence),
    )
    examples = _limited_examples(
        stream, maximum, cfg.dataset.max_per_archetype, cfg.dataset.dedupe)
    result = _write_splits(cfg, examples)
    db.set_meta("dataset_latest", {
        "at": time.time(), "examples": result.examples, "by_archetype": result.by_archetype,
        "paths": {k: str(v) for k, v in result.paths.items()},
    })
    db.close()
    return result


def _all_examples(db: Database, *, min_wave_confidence: float) -> Iterator[TrainingExample]:
    """Every archetype, in a stable order.

    The timeline archetypes come from :mod:`lolcoach.dataset.generators`, which
    builds each answer out of the reasoning the feature pass recorded for that
    exact game state. ``_static_examples`` adds grounded facts from Data Dragon
    so the model also knows what items and abilities actually do.
    """
    generated = (
        gen.wave_examples(db, min_wave_confidence),
        gen.trade_examples(db),
        gen.recall_examples(db),
        gen.jungle_examples(db),
        gen.objective_examples(db),
        gen.death_examples(db),
        gen.rotation_examples(db),
        gen.fight_examples(db),
        gen.matchup_examples(db),
        gen.reddit_qa_examples(db),
        gen.wiki_mechanics_examples(db),
    )
    for stream in generated:
        for archetype, key, user, assistant, confidence, match_id, patch in stream:
            # Reason clauses are assembled from several sources, so collapse
            # any doubled spacing before the text becomes a training target.
            assistant = " ".join(str(assistant).split())
            if len(assistant) < 40:
                continue        # never emit an empty or stub answer
            yield TrainingExample(
                archetype, key, user, assistant,
                {"source": "timeline", "match_id": match_id, "patch": patch,
                 "confidence": round(float(confidence), 3)},
            )
    yield from _static_examples(db)


def _synthetic_examples(cfg: Config) -> Iterator[TrainingExample]:
    """Replay Stage-3 output into the split builder.

    Synthetic examples live in their own append-only JSONL so the expensive
    teacher pass survives dataset rebuilds. They are merged here rather than
    regenerated, and they go through the same dedupe and split logic as
    everything else.
    """
    path = cfg.paths.datasets / "synthetic.jsonl"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            messages = record.get("messages") or []
            meta = record.get("metadata") or {}
            if len(messages) < 3:
                continue
            yield TrainingExample(
                archetype=f"synthetic_{meta.get('archetype', 'general')}",
                key=str(meta.get("key") or messages[1]["content"][:80]),
                user=messages[1]["content"], assistant=messages[2]["content"],
                metadata={k: v for k, v in meta.items() if k != "archetype"},
            )


def _static_examples(db: Database) -> Iterator[TrainingExample]:
    """Turn locally stored static facts into short grounded knowledge prompts."""
    for table, noun, columns in (
        ("champions", "champion", ("name", "title", "tags", "spells", "passive", "patch")),
        ("items", "item", ("name", "description", "total_gold", "tags", "patch")),
        ("runes", "rune", ("name", "short_desc", "long_desc", "tree", "patch")),
    ):
        for row in db.iter_query(f"SELECT {','.join(columns)} FROM {table} ORDER BY name", batch=1000):
            record = dict(row)
            name = str(record["name"])
            if table == "champions":
                spells = _json(record.get("spells"))
                spell_names = ", ".join(str(s.get("name")) for s in spells[:4] if s.get("name"))
                passive = _json(record.get("passive")).get("description", "")
                answer = f"{name} is {record.get('title') or 'a champion'}. Their listed abilities are {spell_names or 'not available in the local static record'}. Passive summary: {_brief(passive)}"
            elif table == "items":
                answer = f"{name} costs {record.get('total_gold') or 0} total gold. Its local description says: {_brief(record.get('description'))}"
            else:
                answer = f"{name} belongs to the {record.get('tree') or 'unknown'} rune tree. {_brief(record.get('long_desc') or record.get('short_desc'))}"
            user = f"Question: What should I know about the {noun} {name}?"
            yield TrainingExample(f"{noun}_knowledge", f"{table}:{name}:{record.get('patch') or ''}", user, answer, {"source": "ddragon", "patch": record.get("patch"), "confidence": 1.0})


def _limited_examples(examples: Iterable[TrainingExample], maximum: int, per_archetype: int, dedupe: bool) -> tuple[list[TrainingExample], int]:
    kept: list[TrainingExample] = []
    counts: Counter[str] = Counter()
    prompts: set[str] = set()
    duplicates = 0
    for example in examples:
        if len(kept) >= maximum:
            break
        if counts[example.archetype] >= per_archetype:
            continue
        fingerprint = hashlib.sha256(_normalise(example.user).encode("utf-8")).hexdigest()
        if dedupe and fingerprint in prompts:
            duplicates += 1
            continue
        prompts.add(fingerprint)
        counts[example.archetype] += 1
        kept.append(example)
    return kept, duplicates


def _write_splits(cfg: Config, packed: tuple[list[TrainingExample], int]) -> DatasetResult:
    examples, duplicates = packed
    output = cfg.paths.datasets
    output.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for example in examples:
        buckets[_split(example.key, cfg.dataset.val_fraction, cfg.dataset.test_fraction)].append(example.as_record())
    paths = {split: output / f"{split}.jsonl" for split in buckets}
    for split, records in buckets.items():
        _atomic_jsonl(paths[split], records)
    card = output / "DATASET_CARD.md"
    _atomic_text(card, _card(cfg, buckets, duplicates))
    paths["card"] = card
    return DatasetResult(len(examples), len(buckets["train"]), len(buckets["validation"]), len(buckets["test"]), duplicates, dict(Counter(record["metadata"]["archetype"] for record in buckets["train"] + buckets["validation"] + buckets["test"])), paths)


def _field(row: Any, key: str, default: Any = None) -> Any:
    """Read a column that may be absent.

    Rows arrive as ``sqlite3.Row``, which supports indexing but not ``.get``,
    and raises rather than returning ``None`` for a missing column. Every
    optional column read goes through here.
    """
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


def _json(value: Any) -> Any:
    try:
        return json.loads(value or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _brief(value: Any, limit: int = 340) -> str:
    text = " ".join(str(value or "").split())
    return (text[:limit].rsplit(" ", 1)[0] + "…") if len(text) > limit else text or "No description is available in the local record."


def _split(key: str, validation: float, test: float) -> str:
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < test:
        return "test"
    if value < test + validation:
        return "validation"
    return "train"


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as fh:
        tmp = Path(fh.name)
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def _atomic_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as fh:
        tmp = Path(fh.name)
        fh.write(content)
    os.replace(tmp, path)


def _card(cfg: Config, buckets: Mapping[str, list[Mapping[str, Any]]], duplicates: int) -> str:
    kinds = Counter(record["metadata"]["archetype"] for split in buckets.values() for record in split)
    lines = ["# LoLCoach training dataset", "", "Generated locally from the configured sources.", "", "## Splits", ""]
    lines.extend(f"- {name}: {len(rows):,} examples" for name, rows in buckets.items())
    lines += ["", "## Archetypes", ""]
    lines.extend(f"- {name}: {count:,}" for name, count in sorted(kinds.items()))
    lines += ["", f"Near-duplicate prompts skipped: {duplicates:,}.", ""]
    lines += _provenance_section(cfg)
    lines += ["", "Timeline-derived labels are heuristic estimates and carry "
              "confidence in each record's metadata. They are inferred from "
              "one-minute timeline frames, not observed directly."]
    return "\n".join(lines) + "\n"


def _provenance_section(cfg: Config) -> list[str]:
    """Licence and attribution for every source that contributed bytes.

    Training corpora inherit their sources' obligations, so the card states
    them mechanically from the download ledger rather than relying on anyone
    to remember which source was CC BY-SA and which was not.
    """
    try:
        from ..storage import Manifest

        rows = Manifest(cfg.paths.manifest_db).licenses()
    except Exception:
        return ["## Source licences", "", "Manifest unavailable."]
    if not rows:
        return ["## Source licences", "", "No downloads recorded."]

    lines = ["## Source licences and attribution", "",
             "| Source | Files | Licence | Attribution |",
             "| --- | ---: | --- | --- |"]
    for row in sorted(rows, key=lambda r: str(r.get("source"))):
        attribution = " ".join(str(row.get("attribution") or "").split())[:160]
        lines.append(f"| {row.get('source')} | {int(row.get('files') or 0):,} | "
                     f"{row.get('license') or 'unspecified'} | {attribution} |")
    lines += ["", "Redistribution of this dataset is constrained by the most "
              "restrictive licence above. Riot API data and Oracle's Elixir data "
              "in particular are not freely redistributable."]
    return lines
