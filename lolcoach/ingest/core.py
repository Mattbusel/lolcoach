"""Idempotent local ingestion for Riot timelines and Oracle's Elixir CSVs."""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..config import Config
from ..logging_utils import get_logger
from ..sources.oracles_elixir import local_csvs
from ..sources.riot_api import iter_local_timelines, read_gz
from ..storage import Database

log = get_logger("ingest")


@dataclass
class IngestResult:
    """Counts and non-fatal input errors from one ingestion pass."""

    source: str
    matches: int = 0
    participants: int = 0
    frames: int = 0
    events: int = 0
    skipped: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def merge(self, other: "IngestResult") -> None:
        self.matches += other.matches
        self.participants += other.participants
        self.frames += other.frames
        self.events += other.events
        self.skipped += other.skipped
        self.errors.extend(other.errors or [])


def ingest_all(cfg: Config, *, limit: int | None = None, force: bool = False) -> list[IngestResult]:
    """Ingest every locally available raw match source.

    A bad or partially-written file is reported in the result and does not
    discard data already ingested from another game.  Re-ingesting a game
    replaces its dependent rows transactionally, making the operation safe to
    resume after new downloads.
    """
    db = Database(cfg.paths.db)
    riot = ingest_riot(db, cfg.paths.source("riot"), limit=limit, force=force)
    oracle = ingest_oracles_elixir(db, cfg.paths.raw, limit=limit, force=force)
    db.set_meta("last_ingest_at", time.time())
    db.set_meta("ingest_counts", {
        "riot_matches": riot.matches, "oracle_matches": oracle.matches,
        "frames": riot.frames + oracle.frames, "events": riot.events,
    })
    db.optimize()
    db.close()
    return [riot, oracle]


def ingest_riot(db: Database, raw_riot_dir: Path, *, limit: int | None = None, force: bool = False) -> IngestResult:
    """Load downloaded Match-V5 records and their timeline pairs into SQLite."""
    result = IngestResult("riot")
    # Matches already loaded with a timeline are skipped, so an incremental
    # download only pays to ingest the games it actually added.
    known: set[str] = set() if force else {
        str(row[0]) for row in db.query(
            "SELECT match_id FROM matches WHERE source='riot' AND has_timeline=1")}
    pairs = iter_local_timelines(raw_riot_dir)
    if pairs is None:
        return result
    # ``iter_local_timelines`` yields the raw files in filesystem order.  Do
    # not apply the cap to that raw position: on a refresh the first N files
    # are commonly already in SQLite, which used to make every collection
    # report zero new games while newer files were never reached.  Count only
    # genuinely new (or attempted) games against the requested limit.
    planned = 0
    for match_id, match_path, timeline_path in pairs:
        if limit is not None and planned >= limit:
            break
        if match_id in known:
            result.skipped += 1
            continue
        planned += 1
        try:
            match = read_gz(match_path)
            timeline = read_gz(timeline_path)
            counts = _ingest_riot_match(db, match_id, match, timeline, str(match_path))
        except Exception as exc:  # individual files must never block a corpus run
            result.skipped += 1
            result.errors.append(f"{match_id}: {type(exc).__name__}: {exc}")
            log.warning("cannot ingest %s: %s", match_id, exc)
            continue
        result.matches += 1
        result.participants += counts[0]
        result.frames += counts[1]
        result.events += counts[2]
    return result


def _ingest_riot_match(
    db: Database, match_id: str, match: Mapping[str, Any], timeline: Mapping[str, Any], raw_path: str,
) -> tuple[int, int, int]:
    info = match.get("info") or {}
    participants = list(info.get("participants") or [])
    if not participants:
        raise ValueError("missing info.participants")
    timeline_info = timeline.get("info") or timeline
    timeline_frames = list(timeline_info.get("frames") or [])
    winner = next((int(p.get("teamId")) for p in participants if p.get("win") is True), None)
    patch = _patch(info.get("gameVersion"))

    participant_rows = [_riot_participant_row(match_id, p) for p in participants]
    frame_rows = list(_riot_frame_rows(match_id, timeline_frames))
    event_rows = list(_riot_event_rows(match_id, timeline_frames))
    with db.transaction() as conn:
        conn.execute("DELETE FROM matches WHERE match_id=?", (match_id,))
        conn.execute(
            "INSERT INTO matches(match_id,source,platform,region,queue_id,patch,game_version,"
            "game_start_ms,duration_s,tier,winner_team,has_timeline,raw_path,ingested_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (match_id, "riot", info.get("platformId"), _region_from_match_id(match_id),
             _integer(info.get("queueId")), patch, info.get("gameVersion"),
             _integer(info.get("gameStartTimestamp") or info.get("gameCreation")), _integer(info.get("gameDuration")),
             "HIGH_ELO", winner, 1, raw_path, time.time()),
        )
        _executemany(conn, "participants", _PARTICIPANT_COLUMNS, participant_rows)
        _executemany(conn, "frames", _FRAME_COLUMNS, frame_rows)
        _executemany(conn, "events", _EVENT_COLUMNS, event_rows)
    return len(participant_rows), len(frame_rows), len(event_rows)


_PARTICIPANT_COLUMNS = (
    "match_id", "participant_id", "puuid", "team_id", "champion", "champion_id", "role", "lane", "win",
    "kills", "deaths", "assists", "gold_earned", "cs", "damage_dealt", "vision_score", "summoner1",
    "summoner2", "keystone", "items",
)
_FRAME_COLUMNS = (
    "match_id", "ts_ms", "participant_id", "x", "y", "total_gold", "current_gold", "xp", "level",
    "minions", "jungle_minions", "health", "health_max", "power", "power_max", "ad", "ap", "armor",
    "mr", "dmg_taken", "dmg_done",
)
_EVENT_COLUMNS = (
    "match_id", "ts_ms", "type", "participant_id", "victim_id", "killer_id", "team_id", "x", "y", "lane",
    "monster_type", "monster_subtype", "building_type", "tower_type", "item_id", "skill_slot", "level",
    "ward_type", "bounty", "payload",
)


def _riot_participant_row(match_id: str, p: Mapping[str, Any]) -> tuple[Any, ...]:
    role = _role(p.get("teamPosition") or p.get("individualPosition") or p.get("lane"))
    items = [p.get(f"item{i}") for i in range(7) if _integer(p.get(f"item{i}"))]
    perks = p.get("perks") or {}
    styles = perks.get("styles") or []
    keystone = None
    if styles:
        selections = styles[0].get("selections") or []
        if selections:
            keystone = _integer(selections[0].get("perk"))
    return (
        match_id, _integer(p.get("participantId")), p.get("puuid"), _integer(p.get("teamId")),
        p.get("championName"), _integer(p.get("championId")), role, p.get("lane"), int(bool(p.get("win"))),
        _integer(p.get("kills")), _integer(p.get("deaths")), _integer(p.get("assists")),
        _integer(p.get("goldEarned")), _integer(p.get("totalMinionsKilled")) + _integer(p.get("neutralMinionsKilled")),
        _integer(p.get("totalDamageDealtToChampions")), _integer(p.get("visionScore")),
        _integer(p.get("summoner1Id")), _integer(p.get("summoner2Id")), keystone, json.dumps(items),
    )


def _riot_frame_rows(match_id: str, frames: Iterable[Mapping[str, Any]]) -> Iterator[tuple[Any, ...]]:
    for index, frame in enumerate(frames):
        ts = _integer(frame.get("timestamp"), index * 60_000)
        for raw_id, p in (frame.get("participantFrames") or {}).items():
            stats = p.get("championStats") or {}
            pos = p.get("position") or {}
            damage = p.get("damageStats") or {}
            # Health and resource live inside ``championStats`` in match-v5
            # timelines. Reading them from the participant frame itself yields
            # zero for every row, which silently destroys hp_pct and every
            # trade, recall and death feature derived from it.
            yield (
                match_id, ts, _integer(p.get("participantId") or raw_id), _integer(pos.get("x")), _integer(pos.get("y")),
                _integer(p.get("totalGold")), _integer(p.get("currentGold")), _integer(p.get("xp")), _integer(p.get("level")),
                _integer(p.get("minionsKilled")), _integer(p.get("jungleMinionsKilled")),
                _integer(stats.get("health", p.get("currentHealth"))),
                _integer(stats.get("healthMax", p.get("maxHealth"))),
                _integer(stats.get("power", p.get("currentPower"))),
                _integer(stats.get("powerMax", p.get("maxPower"))),
                _integer(stats.get("attackDamage")), _integer(stats.get("abilityPower")), _integer(stats.get("armor")),
                _integer(stats.get("magicResist")), _integer(damage.get("totalDamageTaken")),
                _integer(damage.get("totalDamageDoneToChampions")),
            )


def _riot_event_rows(match_id: str, frames: Iterable[Mapping[str, Any]]) -> Iterator[tuple[Any, ...]]:
    for frame in frames:
        for event in frame.get("events") or []:
            event_type = str(event.get("type") or "UNKNOWN")
            yield (
                match_id, _integer(event.get("timestamp")), event_type,
                _integer(event.get("participantId")) or None, _integer(event.get("victimId")) or None,
                _integer(event.get("killerId")) or None,
                # Elite monster kills carry the credited team as ``killerTeamId``
                # rather than ``teamId``; reading only the latter drops every
                # dragon, herald and baron from the objective features.
                _integer(event.get("teamId") or event.get("killerTeamId")) or None,
                _integer(event.get("position", {}).get("x")) or None, _integer(event.get("position", {}).get("y")) or None,
                event.get("laneType") or event.get("lane"), event.get("monsterType"), event.get("monsterSubType"),
                event.get("buildingType"), event.get("towerType"), _integer(event.get("itemId")) or None,
                _integer(event.get("skillSlot")) or None, _integer(event.get("level")) or None,
                event.get("wardType"), _integer(event.get("bounty")) or None,
                json.dumps(event, separators=(",", ":"), ensure_ascii=False),
            )


def ingest_oracles_elixir(db: Database, raw_root: Path, *, limit: int | None = None, force: bool = False) -> IngestResult:
    """Load public pro-game CSVs, including their 10/15/20/25 minute snapshots."""
    result = IngestResult("oracles_elixir")
    seen: set[str] = set()
    # Pro CSVs are re-downloaded whole each season, so most rows in them are
    # already loaded; skipping known game ids keeps a refresh cheap.
    known: set[str] = set() if force else {
        str(row[0]) for row in db.query(
            "SELECT match_id FROM matches WHERE source='oracles_elixir'")}
    for csv_path in local_csvs(raw_root):
        try:
            groups = _oracle_groups(csv_path)
            for game_id, rows in groups.items():
                if game_id in seen:
                    continue
                if f"oe:{game_id}" in known:
                    result.skipped += 1
                    continue
                if limit is not None and result.matches >= limit:
                    return result
                seen.add(game_id)
                counts = _ingest_oracle_game(db, game_id, rows, str(csv_path))
                result.matches += 1
                result.participants += counts[0]
                result.frames += counts[1]
        except Exception as exc:
            result.errors.append(f"{csv_path.name}: {type(exc).__name__}: {exc}")
            log.warning("cannot ingest %s: %s", csv_path, exc)
    return result


def _oracle_groups(path: Path) -> dict[str, list[dict[str, str]]]:
    games: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            game_id = (row.get("gameid") or "").strip()
            # Team-total rows have no player position and cannot be paired to a lane.
            if not game_id or not (row.get("position") or "").strip():
                continue
            games.setdefault(game_id, []).append(row)
    return games


def _ingest_oracle_game(db: Database, game_id: str, rows: list[dict[str, str]], raw_path: str) -> tuple[int, int]:
    players = [r for r in rows if _role(r.get("position"))]
    if len(players) < 2:
        raise ValueError("does not contain player rows")
    match_id = f"oe:{game_id}"
    winner = _oracle_winner(players)
    patch = next((r.get("patch") for r in players if r.get("patch")), None)
    participant_rows = []
    frame_rows = []
    for index, row in enumerate(players, 1):
        pid = _integer(row.get("participantid"), index)
        team_id = 100 if (row.get("side") or "").lower() == "blue" else 200
        role = _role(row.get("position"))
        participant_rows.append((
            match_id, pid, None, team_id, row.get("champion"), None, role, role, _integer(row.get("result")),
            _integer(row.get("kills")), _integer(row.get("deaths")), _integer(row.get("assists")),
            _integer(row.get("totalgold")), _integer(row.get("total cs")) or _integer(row.get("totalcs")),
            _integer(row.get("damagetochampions")), _integer(row.get("visionscore")), None, None, None, "[]",
        ))
        for minute in (10, 15, 20, 25):
            gold = _oracle_value(row, f"goldat{minute}")
            xp = _oracle_value(row, f"xpat{minute}")
            cs = _oracle_value(row, f"csat{minute}")
            if gold is None and xp is None and cs is None:
                continue
            frame_rows.append((
                match_id, minute * 60_000, pid, None, None, gold or 0, 0, xp or 0, 0, cs or 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ))
    with db.transaction() as conn:
        conn.execute("DELETE FROM matches WHERE match_id=?", (match_id,))
        conn.execute(
            "INSERT INTO matches(match_id,source,platform,region,queue_id,patch,game_version,game_start_ms,"
            "duration_s,tier,winner_team,has_timeline,raw_path,ingested_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (match_id, "oracles_elixir", None, None, None, patch, None, None,
             _integer(players[0].get("gamelength")), players[0].get("league"), winner, 0, raw_path, time.time()),
        )
        _executemany(conn, "participants", _PARTICIPANT_COLUMNS, participant_rows)
        _executemany(conn, "frames", _FRAME_COLUMNS, frame_rows)
    return len(participant_rows), len(frame_rows)


def _executemany(conn: Any, table: str, columns: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    names = ",".join(columns)
    marks = ",".join("?" for _ in columns)
    conn.executemany(f"INSERT INTO {table}({names}) VALUES({marks})", rows)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in (None, "", "nan", "None") else default
    except (TypeError, ValueError):
        return default


def _role(value: Any) -> str | None:
    normalised = str(value or "").strip().upper()
    return {
        "TOP": "TOP", "JUNGLE": "JUNGLE", "JNG": "JUNGLE", "JGL": "JUNGLE", "MIDDLE": "MIDDLE", "MID": "MIDDLE",
        "BOTTOM": "BOTTOM", "BOT": "BOTTOM", "ADC": "BOTTOM", "UTILITY": "UTILITY",
        "SUPPORT": "UTILITY", "SUP": "UTILITY",
    }.get(normalised)


def _patch(version: Any) -> str | None:
    parts = str(version or "").split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]) else None


def _region_from_match_id(match_id: str) -> str | None:
    prefix = match_id.split("_", 1)[0].upper()
    return {"NA1": "americas", "BR1": "americas", "LA1": "americas", "LA2": "americas",
            "EUW1": "europe", "EUN1": "europe", "TR1": "europe", "RU": "europe",
            "KR": "asia", "JP1": "asia"}.get(prefix)


def _oracle_value(row: Mapping[str, str], key: str) -> int | None:
    value = row.get(key)
    if value in (None, "", "nan"):
        return None
    return _integer(value)


def _oracle_winner(rows: list[Mapping[str, str]]) -> int | None:
    for row in rows:
        if _integer(row.get("result")):
            return 100 if (row.get("side") or "").lower() == "blue" else 200
    return None
