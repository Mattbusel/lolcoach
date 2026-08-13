"""Dense-plus-lexical retrieval and prompt-context construction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import Config
from .index import _embedding_model

TOKEN = re.compile(r"[a-z0-9][a-z0-9_+'-]*", re.I)
PATCH = re.compile(r"(?:patch\s*)?(\d{1,2}\.\d{1,2})", re.I)


@dataclass(frozen=True)
class RagHit:
    chunk_id: int
    score: float
    dense_score: float
    lexical_score: float
    source: str
    title: str | None
    url: str | None
    patch: str | None
    champion: str | None
    license: str | None
    text: str


class LocalRAG:
    """Read the local FAISS corpus and return attributable grounded context."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        root = cfg.paths.index
        self._config = _load_json(root / "league_index.json")
        if not self._config:
            raise RuntimeError("RAG index is absent. Run `lolcoach rag --build` first.")
        if self._config.get("model") != cfg.rag.embed_model:
            raise RuntimeError("RAG index uses a different embedding model; rebuild it with `lolcoach rag --build`.")
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS is required for retrieval. Install project dependencies first.") from exc
        self.index = faiss.read_index(str(root / "league.faiss"))
        self.records = _load_jsonl(root / "league_chunks.jsonl")
        if int(self.index.ntotal) != len(self.records):
            raise RuntimeError("RAG index and metadata have different sizes; rebuild the index.")
        self.model = _embedding_model(cfg.rag.embed_model)
        stats = _load_json(root / "league_idf.json")
        self._n_docs = max(1, int(stats.get("n_docs") or len(self.records)))
        self._df: dict[str, int] = stats.get("df") or {}

    def search(self, question: str, *, top_k: int | None = None, patch: str | None = None) -> list[RagHit]:
        """Return fused dense/lexical results with optional patch preference."""
        requested = top_k or self.cfg.rag.top_k
        if requested < 1:
            return []
        inferred_patch = patch or _infer_patch(question)
        vector = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32, copy=False)
        distances, indices = self.index.search(vector, min(len(self.records), max(requested * 5, requested)))
        query_tokens = set(_tokens(question))
        hits: list[RagHit] = []
        for dense, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            record = self.records[int(idx)]
            lexical = self._lexical(query_tokens, str(record["text"]))
            dense_scaled = max(0.0, min(1.0, (float(dense) + 1.0) / 2.0))
            score = self.cfg.rag.dense_weight * dense_scaled + (1 - self.cfg.rag.dense_weight) * lexical
            if inferred_patch and _patch_matches(str(record.get("patch") or ""), inferred_patch):
                score += self.cfg.rag.patch_boost
            hits.append(RagHit(int(record["chunk_id"]), score, float(dense), lexical, str(record["source"]), record.get("title"), record.get("url"), record.get("patch"), record.get("champion"), record.get("license"), str(record["text"])))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:requested]

    def _lexical(self, query: set[str], text: str) -> float:
        """IDF-weighted share of the query's informative tokens in a chunk.

        Falls back to an unweighted share when no IDF table is present, so an
        index built by an older version still retrieves.
        """
        if not query:
            return 0.0
        words = set(_tokens(text))
        if not words:
            return 0.0
        if not self._df:
            return sum(1 for token in query if token in words) / len(query)
        import math

        total = 0.0
        matched = 0.0
        for token in query:
            df = self._df.get(token, 0)
            # Smoothed IDF: an unseen token is maximally informative, a token
            # in every chunk contributes almost nothing.
            idf = math.log((self._n_docs + 1) / (df + 1)) + 1.0
            total += idf
            if token in words:
                matched += idf
        return matched / total if total else 0.0

    def context_for(self, question: str, *, top_k: int | None = None, patch: str | None = None, max_chars: int = 12_000) -> tuple[str, list[RagHit]]:
        """Format retrieval evidence for a generation prompt, preserving provenance."""
        hits = self.search(question, top_k=top_k, patch=patch)
        blocks: list[str] = []
        used = 0
        for number, hit in enumerate(hits, 1):
            label = " | ".join(part for part in (hit.source, hit.title, f"patch {hit.patch}" if hit.patch else None) if part)
            block = f"[{number}] {label}\n{hit.text.strip()}"
            if used + len(block) > max_chars and blocks:
                break
            blocks.append(block[:max_chars - used])
            used += len(block)
        return "\n\n".join(blocks), hits


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN.findall(text) if len(token) > 1]


def _lexical(query: set[str], text: str) -> float:
    """Fraction of query tokens present in a chunk.

    The chunk's tokens are put in a set first: with a large corpus this runs
    once per candidate chunk per query, and a list membership test would make
    retrieval quadratic in chunk length for no benefit.
    """
    if not query:
        return 0.0
    words = set(_tokens(text))
    if not words:
        return 0.0
    return sum(1 for token in query if token in words) / len(query)


def _infer_patch(question: str) -> str | None:
    match = PATCH.search(question)
    return match.group(1) if match else None


def _patch_matches(record_patch: str, requested_patch: str) -> bool:
    """Match a stored Data Dragon patch to either accepted patch label.

    Data Dragon's current ``16.xx`` label is publicly called ``26.xx`` by
    Riot and the wiki.  Retrieval is intentionally a preference rather than a
    hard exclusion (evergreen documents often have no patch), but a user who
    asks for either label must receive the same patch boost.
    """
    if record_patch == requested_patch:
        return True
    try:
        record_major, record_minor = (int(part) for part in record_patch.split(".", 1))
        requested_major, requested_minor = (int(part) for part in requested_patch.split(".", 1))
    except (TypeError, ValueError):
        return False
    return record_minor == requested_minor and abs(record_major - requested_major) == 10


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise RuntimeError("RAG metadata is absent; rebuild the index.") from exc
