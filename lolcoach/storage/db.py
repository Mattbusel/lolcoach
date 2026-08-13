"""Relational store for matches, timelines, derived features and the corpus.

SQLite is the right choice here despite the corpus size: the pipeline is
single-machine, the access pattern is "bulk insert once, scan many times",
and a single file keeps the whole dataset trivially copyable. The connection
is opened in WAL mode with a large page cache and ``synchronous=NORMAL``,
which is the standard configuration for bulk analytical loads.

Schema overview
---------------
``matches`` / ``participants``
    One row per game and per player-in-game, from Riot match-v5 or from
    Oracle's Elixir professional data (``source`` distinguishes them).
``frames`` / ``events``
    The match timeline. ``frames`` is the per-minute participant snapshot
    (position, gold, xp, cs, and the ``championStats`` block that gives us
    health and resource state); ``events`` is the discrete event stream.
``wave_states`` / ``lane_states`` / ``recalls`` / ``objective_events`` /
``jungle_paths`` / ``rotations`` / ``fights``
    Stage-2 feature tables, each carrying a confidence column because these
    are *inferred* from a one-minute sampling grid, not observed directly.
``documents`` / ``chunks``
    Text corpus for retrieval, with licence and patch provenance so the RAG
    layer can filter to the live patch and attribute every answer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..logging_utils import get_logger

log = get_logger("storage.db")

SCHEMA_VERSION = 4

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-262144;   -- 256 MB page cache
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------- matches
CREATE TABLE IF NOT EXISTS matches (
    match_id      TEXT PRIMARY KEY,
    source        TEXT NOT NULL,          -- riot | oracles_elixir | community
    platform      TEXT,
    region        TEXT,
    queue_id      INTEGER,
    patch         TEXT,
    game_version  TEXT,
    game_start_ms INTEGER,
    duration_s    INTEGER,
    tier          TEXT,                   -- CHALLENGER, pro league name, ...
    winner_team   INTEGER,                -- 100 or 200
    has_timeline  INTEGER NOT NULL DEFAULT 0,
    raw_path      TEXT,
    ingested_at   REAL
);
CREATE INDEX IF NOT EXISTS ix_matches_patch ON matches(patch);
CREATE INDEX IF NOT EXISTS ix_matches_source ON matches(source);
CREATE INDEX IF NOT EXISTS ix_matches_timeline ON matches(has_timeline);

CREATE TABLE IF NOT EXISTS participants (
    match_id       TEXT NOT NULL,
    participant_id INTEGER NOT NULL,
    puuid          TEXT,
    team_id        INTEGER,
    champion       TEXT,
    champion_id    INTEGER,
    role           TEXT,                  -- TOP JUNGLE MIDDLE BOTTOM UTILITY
    lane           TEXT,
    win            INTEGER,
    kills          INTEGER, deaths INTEGER, assists INTEGER,
    gold_earned    INTEGER,
    cs             INTEGER,
    damage_dealt   INTEGER,
    vision_score   INTEGER,
    summoner1      INTEGER, summoner2 INTEGER,
    keystone       INTEGER,
    items          TEXT,                  -- JSON list of item ids
    PRIMARY KEY (match_id, participant_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_part_champ ON participants(champion, role);
CREATE INDEX IF NOT EXISTS ix_part_match ON participants(match_id);

-- --------------------------------------------------------------- timeline
CREATE TABLE IF NOT EXISTS frames (
    match_id       TEXT NOT NULL,
    ts_ms          INTEGER NOT NULL,
    participant_id INTEGER NOT NULL,
    x              INTEGER, y INTEGER,
    total_gold     INTEGER, current_gold INTEGER,
    xp             INTEGER, level INTEGER,
    minions        INTEGER, jungle_minions INTEGER,
    health         INTEGER, health_max INTEGER,
    power          INTEGER, power_max INTEGER,
    ad             INTEGER, ap INTEGER, armor INTEGER, mr INTEGER,
    dmg_taken      INTEGER, dmg_done INTEGER,
    PRIMARY KEY (match_id, ts_ms, participant_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_frames_match ON frames(match_id, participant_id, ts_ms);

CREATE TABLE IF NOT EXISTS events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id       TEXT NOT NULL,
    ts_ms          INTEGER NOT NULL,
    type           TEXT NOT NULL,
    participant_id INTEGER,
    victim_id      INTEGER,
    killer_id      INTEGER,
    team_id        INTEGER,
    x              INTEGER, y INTEGER,
    lane           TEXT,
    monster_type   TEXT,
    monster_subtype TEXT,
    building_type  TEXT,
    tower_type     TEXT,
    item_id        INTEGER,
    skill_slot     INTEGER,
    level          INTEGER,
    ward_type      TEXT,
    bounty         INTEGER,
    payload        TEXT,                  -- full original JSON
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_events_match ON events(match_id, ts_ms);
CREATE INDEX IF NOT EXISTS ix_events_type ON events(type);

-- --------------------------------------------------------- stage 2 features
CREATE TABLE IF NOT EXISTS wave_states (
    match_id     TEXT NOT NULL,
    ts_ms        INTEGER NOT NULL,
    lane         TEXT NOT NULL,           -- TOP MIDDLE BOTTOM
    team_id      INTEGER NOT NULL,        -- perspective team
    position     REAL,                    -- -1 = own tower .. 0 mid .. +1 enemy tower
    velocity     REAL,                    -- change in position per minute
    state        TEXT,                    -- FREEZE SLOW_PUSH BIG_WAVE CRASHING BOUNCING NEUTRAL
    size_diff    REAL,                    -- estimated attacker count advantage
    cannon       INTEGER,                 -- 1 if a cannon minion is alive in the wave
    recommended  TEXT,                    -- CRASH FREEZE SLOW_PUSH SHOVE_AND_ROAM RESET
    confidence   REAL,
    rationale    TEXT,                    -- JSON list of human-readable reasons
    PRIMARY KEY (match_id, ts_ms, lane, team_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lane_states (
    match_id       TEXT NOT NULL,
    ts_ms          INTEGER NOT NULL,
    participant_id INTEGER NOT NULL,
    lane           TEXT,
    cs_diff        INTEGER,
    gold_diff      INTEGER,
    xp_diff        INTEGER,
    level_diff     INTEGER,
    hp_pct         REAL,
    opp_hp_pct     REAL,
    plates         INTEGER,
    priority       TEXT,                  -- HARD_PRIO PRIO EVEN NO_PRIO
    trade_window   TEXT,                  -- FAVOURABLE EVEN UNFAVOURABLE
    all_in         INTEGER,               -- 1 when an all-in is estimated winning
    recall_value   REAL,                  -- gold-adjusted value of recalling now
    should_recall  INTEGER,
    confidence     REAL,
    rationale      TEXT,
    PRIMARY KEY (match_id, ts_ms, participant_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recalls (
    match_id       TEXT NOT NULL,
    participant_id INTEGER NOT NULL,
    ts_ms          INTEGER NOT NULL,
    gold_spent     INTEGER,
    items          TEXT,
    wave_state     TEXT,
    was_good       INTEGER,               -- heuristic label from the outcome
    reason         TEXT,
    PRIMARY KEY (match_id, participant_id, ts_ms),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS objective_events (
    match_id     TEXT NOT NULL,
    ts_ms        INTEGER NOT NULL,
    objective    TEXT NOT NULL,           -- DRAGON BARON HERALD GRUBS TOWER INHIB ATAKHAN
    subtype      TEXT,
    team_id      INTEGER,
    contested    INTEGER,
    setup_score  REAL,                    -- pre-objective control estimate
    prior_kills  INTEGER,
    vision_edge  REAL,
    next_spawn_ms INTEGER,
    rationale    TEXT,
    PRIMARY KEY (match_id, ts_ms, objective),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jungle_paths (
    match_id       TEXT NOT NULL,
    participant_id INTEGER NOT NULL,
    ts_ms          INTEGER NOT NULL,
    quadrant       TEXT,                  -- BLUE_TOP BLUE_BOT RED_TOP RED_BOT RIVER
    camp           TEXT,
    clear_index    INTEGER,
    gank_target    TEXT,
    predicted_next TEXT,
    confidence     REAL,
    PRIMARY KEY (match_id, participant_id, ts_ms),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rotations (
    match_id       TEXT NOT NULL,
    participant_id INTEGER NOT NULL,
    ts_ms          INTEGER NOT NULL,
    from_zone      TEXT, to_zone TEXT,
    purpose        TEXT,                  -- ROAM OBJECTIVE_SETUP SIDE_LANE RESET COLLAPSE
    tempo_gain     REAL,
    was_correct    INTEGER,
    PRIMARY KEY (match_id, participant_id, ts_ms),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fights (
    match_id     TEXT NOT NULL,
    fight_id     INTEGER NOT NULL,
    start_ms     INTEGER, end_ms INTEGER,
    zone         TEXT,
    team_a_count INTEGER, team_b_count INTEGER,
    kills_a      INTEGER, kills_b INTEGER,
    winner_team  INTEGER,
    gold_swing   INTEGER,
    trigger      TEXT,
    positioning  TEXT,                    -- JSON per-participant positioning notes
    PRIMARY KEY (match_id, fight_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS deaths (
    match_id       TEXT NOT NULL,
    ts_ms          INTEGER NOT NULL,
    victim_id      INTEGER NOT NULL,
    killer_id      INTEGER,
    x INTEGER, y INTEGER,
    zone           TEXT,
    assisters      INTEGER,
    outnumbered    INTEGER,
    vision_nearby  INTEGER,
    hp_before      INTEGER,
    cause          TEXT,                  -- OVEREXTENDED, CAUGHT_ROTATING, LOST_TRADE ...
    cost_gold      INTEGER,
    rationale      TEXT,
    PRIMARY KEY (match_id, ts_ms, victim_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

-- ------------------------------------------------------------ text corpus
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    source     TEXT NOT NULL,             -- wiki | reddit | patch_notes | guide | ddragon
    kind       TEXT,                      -- champion | item | rune | matchup | discussion
    title      TEXT,
    url        TEXT,
    patch      TEXT,
    champion   TEXT,
    license    TEXT,
    score      REAL,                      -- upstream popularity signal, if any
    created_at REAL,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_docs_source ON documents(source);
CREATE INDEX IF NOT EXISTS ix_docs_champ ON documents(champion);
CREATE INDEX IF NOT EXISTS ix_docs_patch ON documents(patch);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id    TEXT NOT NULL,
    ord       INTEGER NOT NULL,
    text      TEXT NOT NULL,
    n_tokens  INTEGER,
    embedded  INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_chunks_embedded ON chunks(embedded);

-- ------------------------------------------------------ static game data
CREATE TABLE IF NOT EXISTS champions (
    champion_id INTEGER PRIMARY KEY,
    key         TEXT UNIQUE,
    name        TEXT,
    title       TEXT,
    tags        TEXT,
    resource    TEXT,
    stats       TEXT,                     -- JSON base stats + per-level growth
    spells      TEXT,                     -- JSON abilities incl. cooldowns
    passive     TEXT,
    patch       TEXT
);

CREATE TABLE IF NOT EXISTS items (
    item_id  INTEGER PRIMARY KEY,
    name     TEXT,
    gold     INTEGER,
    total_gold INTEGER,
    tags     TEXT,
    stats    TEXT,
    into_ids TEXT,
    from_ids TEXT,
    description TEXT,
    patch    TEXT
);

CREATE TABLE IF NOT EXISTS runes (
    rune_id  INTEGER PRIMARY KEY,
    tree     TEXT,
    slot     INTEGER,
    name     TEXT,
    short_desc TEXT,
    long_desc  TEXT,
    patch    TEXT
);
"""


#: Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
#: EXISTS", so each is applied only when absent from the live table.
ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "wave_states": [("rationale", "TEXT")],
    "lane_states": [("rationale", "TEXT")],
    "objective_events": [("rationale", "TEXT")],
    "deaths": [("rationale", "TEXT")],
}


class Database:
    """Thin, thread-aware wrapper around a SQLite connection.

    One connection is held per thread (SQLite objects cannot cross threads),
    which lets the download and ingest stages use worker pools freely.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        with self._init_lock:
            self._ensure_schema()

    # -- connection -----------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        """Connection for the calling thread, created on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=60.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        conn = self.conn
        conn.executescript(SCHEMA)
        cur = conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                         (str(SCHEMA_VERSION),))
        elif int(row[0]) != SCHEMA_VERSION:
            # Schema statements are all IF NOT EXISTS, so an older file gains
            # new tables automatically; record the upgrade.
            conn.execute("UPDATE meta SET value=? WHERE key='schema_version'",
                         (str(SCHEMA_VERSION),))
            log.info("database schema upgraded to v%d", SCHEMA_VERSION)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Apply additive column migrations to an existing database file."""
        conn = self.conn
        for table, columns in ADDED_COLUMNS.items():
            existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if not existing:
                continue
            for name, decl in columns:
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                    log.info("added column %s.%s", table, name)

    # -- transactions ---------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction; rolls back on exception."""
        conn = self.conn
        conn.execute("BEGIN")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # -- writes ---------------------------------------------------------
    def insert_many(self, table: str, columns: Sequence[str],
                    rows: Iterable[Sequence[Any]], replace: bool = True) -> int:
        """Bulk insert. Returns the number of rows submitted."""
        rows = list(rows)
        if not rows:
            return 0
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        sql = (f"{verb} INTO {table} ({', '.join(columns)}) "
               f"VALUES ({', '.join('?' * len(columns))})")
        with self.transaction() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None

    def iter_query(self, sql: str, params: Sequence[Any] = (),
                   batch: int = 2000) -> Iterator[sqlite3.Row]:
        """Stream a large result set without materialising it."""
        cur = self.conn.execute(sql, params)
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                return
            yield from rows

    # -- meta -----------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value) if not isinstance(value, str) else value))

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def counts(self) -> dict[str, int]:
        """Row counts for every table, for status reporting."""
        tables = [r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'").fetchall()]
        return {t: int(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                for t in sorted(tables)}

    def optimize(self) -> None:
        """Run after a bulk load: refresh planner statistics and checkpoint."""
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.execute("ANALYZE")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
