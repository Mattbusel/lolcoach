"""Download provenance ledger.

Every byte the pipeline fetches is recorded here with its URL, on-disk path,
SHA-256, size, upstream licence and fetch time. This does three jobs:

1. **Idempotence.** A re-run skips anything already present with a matching
   digest, so ``download_data`` is safe to interrupt and restart.
2. **Attribution.** Training corpora inherit licence obligations; the wiki is
   CC-BY-SA, Oracle's Elixir asks for credit, Reddit content stays under its
   authors' rights. Keeping the licence next to the bytes means the dataset
   card can be generated mechanically rather than remembered.
3. **Auto-integration.** Datasets discovered at runtime (Stage 1 requirement)
   are registered in ``discovered_sources`` so later runs pull them without
   rediscovery, and so an operator can audit what was added.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

MANIFEST_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS downloads (
    url         TEXT NOT NULL,
    path        TEXT NOT NULL,
    source      TEXT NOT NULL,
    kind        TEXT,
    sha256      TEXT,
    bytes       INTEGER,
    license     TEXT,
    attribution TEXT,
    etag        TEXT,
    fetched_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ok',
    note        TEXT,
    PRIMARY KEY (url, path)
);
CREATE INDEX IF NOT EXISTS ix_dl_source ON downloads(source);
CREATE INDEX IF NOT EXISTS ix_dl_path ON downloads(path);

CREATE TABLE IF NOT EXISTS discovered_sources (
    key         TEXT PRIMARY KEY,   -- e.g. "hf:owner/name" or "gh:owner/repo"
    provider    TEXT NOT NULL,      -- huggingface | github | kaggle
    identifier  TEXT NOT NULL,
    title       TEXT,
    url         TEXT,
    license     TEXT,
    relevance   REAL,
    accepted    INTEGER NOT NULL DEFAULT 0,
    integrated  INTEGER NOT NULL DEFAULT 0,
    discovered_at REAL NOT NULL,
    meta        TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    stage      TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at   REAL,
    ok         INTEGER,
    detail     TEXT
);
"""


@dataclass
class DownloadRecord:
    """One fetched artefact."""

    url: str
    path: str
    source: str
    kind: str = ""
    sha256: str = ""
    bytes: int = 0
    license: str = ""
    attribution: str = ""
    etag: str = ""
    fetched_at: float = field(default_factory=time.time)
    status: str = "ok"
    note: str = ""


class Manifest:
    """Thread-safe ledger over a small SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.conn.executescript(MANIFEST_SCHEMA)

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=60.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    # -- downloads ------------------------------------------------------
    def record(self, rec: DownloadRecord) -> None:
        d = asdict(rec)
        cols = ", ".join(d)
        self.conn.execute(
            f"INSERT OR REPLACE INTO downloads ({cols}) "
            f"VALUES ({', '.join('?' * len(d))})", tuple(d.values()))

    def have(self, url: str, path: Path | str | None = None) -> bool:
        """True when this URL was fetched successfully and the file survives."""
        if path is not None:
            row = self.conn.execute(
                "SELECT path, status FROM downloads WHERE url=? AND path=?",
                (url, str(path))).fetchone()
        else:
            row = self.conn.execute(
                "SELECT path, status FROM downloads WHERE url=? AND status='ok' "
                "ORDER BY fetched_at DESC LIMIT 1", (url,)).fetchone()
        if row is None or row["status"] != "ok":
            return False
        return Path(row["path"]).exists()

    def age(self, url: str) -> float | None:
        """Seconds since this URL was last fetched, or ``None`` if never."""
        row = self.conn.execute(
            "SELECT MAX(fetched_at) AS t FROM downloads WHERE url=? AND status='ok'",
            (url,)).fetchone()
        return None if not row or row["t"] is None else time.time() - float(row["t"])

    def mark_failed(self, url: str, path: Path | str, source: str, note: str) -> None:
        self.record(DownloadRecord(url=url, path=str(path), source=source,
                                   status="failed", note=note[:2000]))

    def by_source(self, source: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM downloads WHERE source=? AND status='ok'", (source,)).fetchall()

    def summary(self) -> list[dict[str, Any]]:
        """Per-source counts and byte totals, for the CLI status table."""
        rows = self.conn.execute(
            "SELECT source, COUNT(*) AS files, COALESCE(SUM(bytes),0) AS bytes, "
            "       SUM(status='ok') AS ok, SUM(status!='ok') AS failed, "
            "       MAX(fetched_at) AS last "
            "FROM downloads GROUP BY source ORDER BY bytes DESC").fetchall()
        return [dict(r) for r in rows]

    def licenses(self) -> list[dict[str, Any]]:
        """Distinct licence/attribution pairs actually present on disk."""
        rows = self.conn.execute(
            "SELECT source, license, attribution, COUNT(*) AS files "
            "FROM downloads WHERE status='ok' GROUP BY source, license, attribution"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- discovery ------------------------------------------------------
    def add_discovered(self, key: str, provider: str, identifier: str,
                       title: str = "", url: str = "", license: str = "",
                       relevance: float = 0.0, accepted: bool = False,
                       meta: dict[str, Any] | None = None) -> bool:
        """Register a discovered dataset. Returns True when it is new."""
        exists = self.conn.execute(
            "SELECT 1 FROM discovered_sources WHERE key=?", (key,)).fetchone()
        if exists:
            self.conn.execute(
                "UPDATE discovered_sources SET relevance=?, accepted=? WHERE key=?",
                (relevance, int(accepted), key))
            return False
        self.conn.execute(
            "INSERT INTO discovered_sources "
            "(key, provider, identifier, title, url, license, relevance, accepted, "
            " integrated, discovered_at, meta) VALUES (?,?,?,?,?,?,?,?,0,?,?)",
            (key, provider, identifier, title, url, license, relevance,
             int(accepted), time.time(), json.dumps(meta or {})))
        return True

    def mark_integrated(self, key: str) -> None:
        self.conn.execute("UPDATE discovered_sources SET integrated=1 WHERE key=?", (key,))

    def pending_discoveries(self, provider: str | None = None) -> list[sqlite3.Row]:
        """Accepted-but-not-yet-downloaded discoveries."""
        if provider:
            return self.conn.execute(
                "SELECT * FROM discovered_sources WHERE accepted=1 AND integrated=0 "
                "AND provider=? ORDER BY relevance DESC", (provider,)).fetchall()
        return self.conn.execute(
            "SELECT * FROM discovered_sources WHERE accepted=1 AND integrated=0 "
            "ORDER BY relevance DESC").fetchall()

    def all_discoveries(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM discovered_sources ORDER BY relevance DESC").fetchall()

    # -- runs -----------------------------------------------------------
    def start_run(self, run_id: str, stage: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, stage, started_at) VALUES (?,?,?)",
            (run_id, stage, time.time()))

    def end_run(self, run_id: str, ok: bool, detail: str = "") -> None:
        self.conn.execute(
            "UPDATE runs SET ended_at=?, ok=?, detail=? WHERE run_id=?",
            (time.time(), int(ok), detail[:4000], run_id))

    def total_bytes(self) -> int:
        return int(self.conn.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM downloads WHERE status='ok'"
        ).fetchone()[0])

    def iter_paths(self, source: str | None = None) -> Iterable[Path]:
        sql = "SELECT path FROM downloads WHERE status='ok'"
        params: tuple[Any, ...] = ()
        if source:
            sql += " AND source=?"
            params = (source,)
        for row in self.conn.execute(sql, params):
            p = Path(row["path"])
            if p.exists():
                yield p
