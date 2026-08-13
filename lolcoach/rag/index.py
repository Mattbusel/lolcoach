"""Build a versioned FAISS index over the locally collected League corpus."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from ..storage import Database

log = get_logger("rag.index")
WORDS = re.compile(r"\S+")
#: Word tokeniser for lexical scoring, kept in sync with the retriever.
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+'-]*", re.I)


@dataclass(frozen=True)
class IndexResult:
    chunks: int
    dimensions: int
    index_path: Path
    metadata_path: Path
    model: str


def chunk_corpus(cfg: Config, db: Database | None = None) -> int:
    """Materialise static data as documents and rechunk the complete corpus.

    Rebuilding chunks instead of incrementally splicing them avoids stale
    chunks when a wiki document is updated by ``update_patch``.  Existing
    document identifiers remain stable, and the operation is atomic from the
    SQLite reader's perspective.
    """
    store = db or Database(cfg.paths.db)
    _sync_static_documents(store)
    source_rows = list(store.iter_query("SELECT doc_id,text FROM documents ORDER BY doc_id", batch=1000))
    rows: list[tuple[str, int, str, int]] = []
    for doc in source_rows:
        for ordinal, text in enumerate(_chunks(str(doc["text"]), cfg.rag.chunk_tokens, cfg.rag.chunk_overlap)):
            rows.append((str(doc["doc_id"]), ordinal, text, len(WORDS.findall(text))))
    with store.transaction() as conn:
        conn.execute("DELETE FROM chunks")
        if rows:
            conn.executemany("INSERT INTO chunks(doc_id,ord,text,n_tokens) VALUES(?,?,?,?)", rows)
    return len(rows)


def build_index(cfg: Config, *, rebuild_chunks: bool = True) -> IndexResult:
    """Embed all chunks and persist a cosine-similarity FAISS index locally."""
    db = Database(cfg.paths.db)
    if rebuild_chunks:
        chunk_corpus(cfg, db)
    rows = [dict(row) for row in db.iter_query(
        "SELECT c.chunk_id,c.ord,c.text,d.doc_id,d.source,d.kind,d.title,d.url,d.patch,d.champion,d.license "
        "FROM chunks c JOIN documents d ON d.doc_id=c.doc_id ORDER BY c.chunk_id", batch=2000)]
    if not rows:
        raise RuntimeError("No corpus chunks exist. Run download_data before building the RAG index.")
    model = _embedding_model(cfg.rag.embed_model)
    vectors = model.encode(
        [str(row["text"]) for row in rows], batch_size=cfg.rag.embed_batch_size,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    if vectors.ndim != 2 or not len(vectors):
        raise RuntimeError("Embedding model returned no usable vectors")
    try:
        import faiss
    except ImportError as exc:  # dependency lives in pyproject; explain a partial install clearly
        raise RuntimeError("FAISS is required for RAG. Install the project dependencies first.") from exc
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    root = cfg.paths.index
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "league.faiss"
    meta_path = root / "league_chunks.jsonl"
    config_path = root / "league_index.json"
    _atomic_faiss_write(faiss, index, index_path)
    _atomic_jsonl(meta_path, rows)
    _atomic_json(root / "league_idf.json", _document_frequencies(rows))
    _atomic_json(config_path, {
        "schema": 2, "created_at": time.time(), "model": cfg.rag.embed_model,
        "dimensions": int(vectors.shape[1]), "chunks": len(rows), "metric": "cosine",
    })
    with db.transaction() as conn:
        conn.execute("UPDATE chunks SET embedded=1")
    db.set_meta("rag_index", {"path": str(index_path), "model": cfg.rag.embed_model, "chunks": len(rows)})
    result = IndexResult(len(rows), int(vectors.shape[1]), index_path, meta_path, cfg.rag.embed_model)
    db.close()
    return result


def _sync_static_documents(db: Database) -> None:
    """Upsert champion/item/rune records so RAG covers patch-sensitive facts."""
    docs: list[tuple[Any, ...]] = []
    now = time.time()
    for row in db.iter_query("SELECT key,name,title,tags,stats,spells,passive,patch FROM champions ORDER BY name"):
        spells = _read_json(row["spells"])
        ability_text = "\n".join(
            f"{spell.get('name', '')}: {spell.get('description', '')}" for spell in spells if spell.get("name"))
        passive = _read_json(row["passive"])
        text = f"Champion: {row['name']}\nTitle: {row['title'] or ''}\nTags: {row['tags'] or ''}\nPassive: {passive.get('name', '')}: {passive.get('description', '')}\nAbilities:\n{ability_text}\nBase stats: {row['stats'] or '{}'}"
        docs.append((f"ddragon:champion:{row['key']}", "ddragon", "champion", row["name"], None, row["patch"], row["name"], "Riot Games API Terms of Service", 0.0, now, text))
    for row in db.iter_query("SELECT item_id,name,total_gold,tags,stats,description,patch FROM items ORDER BY name"):
        text = f"Item: {row['name']}\nTotal gold: {row['total_gold']}\nTags: {row['tags'] or ''}\nStats: {row['stats'] or '{}'}\nDescription: {row['description'] or ''}"
        docs.append((f"ddragon:item:{row['item_id']}", "ddragon", "item", row["name"], None, row["patch"], None, "Riot Games API Terms of Service", 0.0, now, text))
    for row in db.iter_query("SELECT rune_id,tree,name,short_desc,long_desc,patch FROM runes ORDER BY name"):
        text = f"Rune: {row['name']}\nTree: {row['tree']}\nShort description: {row['short_desc'] or ''}\nDetails: {row['long_desc'] or ''}"
        docs.append((f"ddragon:rune:{row['rune_id']}", "ddragon", "rune", row["name"], None, row["patch"], None, "Riot Games API Terms of Service", 0.0, now, text))
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO documents(doc_id,source,kind,title,url,patch,champion,license,score,created_at,text) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET "
            "title=excluded.title,patch=excluded.patch,created_at=excluded.created_at,text=excluded.text",
            docs,
        )


def _document_frequencies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count how many chunks contain each token, for IDF-weighted retrieval.

    Without this the lexical half of the hybrid score treats "the" and
    "Draven" as equally informative, so a chunk that merely shares the
    question's stopwords outranks the one that shares its subject. The counts
    are computed once here and reused for every query.
    """
    from collections import Counter

    frequencies: Counter[str] = Counter()
    for row in rows:
        frequencies.update(set(_index_tokens(str(row["text"]))))
    # Tokens appearing in almost every chunk carry no signal and only bloat
    # the file, so drop them; their IDF would round to zero anyway.
    ceiling = max(2, int(len(rows) * 0.6))
    kept = {token: count for token, count in frequencies.items() if count <= ceiling}
    return {"n_docs": len(rows), "df": kept}


def _index_tokens(text: str) -> list[str]:
    """Tokeniser shared by index building and querying; they must agree."""
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1]


def _chunks(text: str, size: int, overlap: int) -> Iterator[str]:
    tokens = WORDS.findall(text)
    if not tokens:
        return
    if size < 32:
        raise ValueError("rag.chunk_tokens must be at least 32")
    step = max(1, size - max(0, overlap))
    for start in range(0, len(tokens), step):
        part = tokens[start:start + size]
        if part:
            yield " ".join(part)
        if start + size >= len(tokens):
            return


def _embedding_model(name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for RAG. Install project dependencies first.") from exc
    return SentenceTransformer(name)


def _read_json(value: Any) -> Any:
    try:
        return json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _atomic_faiss_write(faiss: Any, index: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as fh:
        temporary = Path(fh.name)
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, record: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as fh:
        temporary = Path(fh.name)
        json.dump(record, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(temporary, path)
