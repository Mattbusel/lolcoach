"""Stage 2a: normalise raw Riot payloads into the relational schema.

Reads the gzipped match and timeline JSON written by the Riot connector and
loads four tables: ``matches``, ``participants``, ``frames`` and ``events``.

Two details matter for everything downstream:

* **Frames are a minute apart.** ``info.frameInterval`` is 60000 ms. Every
  feature that talks about "now" is therefore quantised to the minute, and the
  feature layer never pretends otherwise.
* **``championStats`` is the useful part.** It carries live health, attack
  damage, ability power, armour and magic resist per participant per frame,
  which is what makes trade evaluation possible at all. It is flattened into
  columns here so the feature pass is a plain SQL scan.

Ingestion is idempotent: a match already present with a timeline is skipped
unless ``force`` is set, so this can be re-run after every download.
"""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..logging_utils import get_logger
from ..storage import Database

log = get_logger("ingest.timeline")

#: Event fields promoted to columns. Everything else stays in ``payload``.
EVENT_COLUMNS = (
    "match_id", "ts_ms", "type", "participant_id", "victim_id", "killer_id",
    "team_id", "x", "y", "lane", "monster_type", "monster_subtype",
    "building_type", "tower_type", "item_id", "skill_slot", "level",
    "ward_type", "bounty", "payload",
)

FRAME_COLUMNS = (
    "match_id", "ts_ms", "participant_id", "x", "y", "total_gold",
    "current_gold", "xp", "level", "minions", "jungle_minions", "health",
    "health_max", "power", "power_max", "ad", "ap", "armor", "mr",
    "dmg_taken", "dmg_done",
)

MATCH_COLUMNS = (
    "match_id", "source", "platform", "region", "queue_id", "patch",
    "game_version", "game_start_ms", "duration_s", "tier", "winner_team",
    "has_timeline", "raw_path", "ingested_at",
)

PARTICIPANT_COLUMNS = (
    "match_id", "participant_id", "puuid", "team_id", "champion", "champion_id",
    "role", "lane", "win", "kills", "deaths", "assists", "gold_earned", "cs",
    "damage_dealt", "vision_score", "summoner1", "summoner2", "keystone", "items",
)


def read_json_gz(path: Path) -> Any:
    """Read a gzipped JSON file."""
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def patch_from_version(game_version: str) -> str:
    """``16.16.652.1234`` -> ``16.16``."""
    parts = (game_version or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else (game_version or "")


def parse_match(match: dict[str, Any], raw_path: str = "",
                source: str = "riot", tier: str | None = None
                ) -> tuple[tuple, list[tuple]]:
    """Extract the ``matches`` row and ``participants`` rows from match-v5 JSON."""
    info = match.get("info", {})
    meta = match.get("metadata", {})
    match_id = meta.get("matchId") or info.get("gameId")
    version = str(info.get("gameVersion", ""))
    platform = info.get("platformId", "")

    winner = None
    for team in info.get("teams", []):
        if team.get("win"):
            winner = team.get("teamId")
            break

    match_row = (
        str(match_id), source, platform, _region_of(platform),
        info.get("queueId"), patch_from_version(version), version,
        info.get("gameStartTimestamp") or info.get("gameCreation"),
        int(info.get("gameDuration") or 0), tier, winner, 0, raw_path, time.time(),
    )

    participants: list[tuple] = []
    for p in info.get("participants", []):
        perks = p.get("perks", {}) or {}
        keystone = None
        for style in perks.get("styles", []) or []:
            for sel in style.get("selections", []) or []:
                keystone = sel.get("perk")
                break
            if keystone:
                break
        items = [p.get(f"item{i}") for i in range(7)]
        participants.append((
            str(match_id), int(p.get("participantId", 0)), p.get("puuid"),
            p.get("teamId"), p.get("championName"), p.get("championId"),
            p.get("teamPosition") or p.get("individualPosition"), p.get("lane"),
            int(bool(p.get("win"))), p.get("kills"), p.get("deaths"),
            p.get("assists"), p.get("goldEarned"),
            int(p.get("totalMinionsKilled") or 0) + int(p.get("neutralMinionsKilled") or 0),
            p.get("totalDamageDealtToChampions"), p.get("visionScore"),
            p.get("summoner1Id"), p.get("summoner2Id"), keystone, json.dumps(items),
        ))
    return match_row, participants


def parse_timeline(match_id: str, timeline: dict[str, Any]
                   ) -> tuple[list[tuple], list[tuple]]:
    """Extract ``frames`` and ``events`` rows from a match-v5 timeline."""
    info = timeline.get("info", {})
    frames_out: list[tuple] = []
    events_out: list[tuple] = []

    for frame in info.get("frames", []) or []:
        ts = int(frame.get("timestamp", 0))
        pframes = frame.get("participantFrames", {}) or {}
        for key, pf in pframes.items():
            try:
                pid = int(pf.get("participantId", key))
            except (TypeError, ValueError):
                continue
            pos = pf.get("position") or {}
            stats = pf.get("championStats") or {}
            dmg = pf.get("damageStats") or {}
            frames_out.append((
                match_id, ts, pid, pos.get("x"), pos.get("y"),
                pf.get("totalGold"), pf.get("currentGold"), pf.get("xp"),
                pf.get("level"), pf.get("minionsKilled"),
                pf.get("jungleMinionsKilled"),
                stats.get("health"), stats.get("healthMax"),
                stats.get("power"), stats.get("powerMax"),
                stats.get("attackDamage"), stats.get("abilityPower"),
                stats.get("armor"), stats.get("magicResist"),
                dmg.get("totalDamageTaken"), dmg.get("totalDamageDoneToChampions"),
            ))

        for ev in frame.get("events", []) or []:
            pos = ev.get("position") or {}
            events_out.append((
                match_id, int(ev.get("timestamp", ts)), ev.get("type"),
                ev.get("participantId"), ev.get("victimId"), ev.get("killerId"),
                ev.get("teamId") or ev.get("killerTeamId"),
                pos.get("x"), pos.get("y"), ev.get("laneType"),
                ev.get("monsterType"), ev.get("monsterSubType"),
                ev.get("buildingType"), ev.get("towerType"), ev.get("itemId"),
                ev.get("skillSlot"), ev.get("level"), ev.get("wardType"),
                _bounty(ev), json.dumps(ev, separators=(",", ":")),
            ))
    return frames_out, events_out


def _bounty(ev: dict[str, Any]) -> int | None:
    """Total gold a kill event was worth, when the payload reports it."""
    for key in ("bounty", "shutdownBounty"):
        if ev.get(key) is not None:
            try:
                return int(ev[key])
            except (TypeError, ValueError):
                continue
    return None


def _region_of(platform: str) -> str:
    """Map a platform id onto its regional routing value."""
    platform = (platform or "").upper()
    americas = {"NA1", "BR1", "LA1", "LA2"}
    europe = {"EUW1", "EUN1", "TR1", "RU"}
    asia = {"KR", "JP1"}
    sea = {"OC1", "PH2", "SG2", "TH2", "TW2", "VN2"}
    if platform in americas:
        return "americas"
    if platform in europe:
        return "europe"
    if platform in asia:
        return "asia"
    if platform in sea:
        return "sea"
    return ""


def ingest_pair(db: Database, match_path: Path, timeline_path: Path | None,
                source: str = "riot", tier: str | None = None) -> dict[str, int]:
    """Load one match (and its timeline, if present) into the database."""
    match = read_json_gz(match_path)
    match_row, participants = parse_match(match, str(match_path), source, tier)
    match_id = match_row[0]

    frames: list[tuple] = []
    events: list[tuple] = []
    if timeline_path is not None and timeline_path.exists():
        timeline = read_json_gz(timeline_path)
        frames, events = parse_timeline(match_id, timeline)
        match_row = match_row[:11] + (1,) + match_row[12:]

    db.insert_many("matches", MATCH_COLUMNS, [match_row])
    db.insert_many("participants", PARTICIPANT_COLUMNS, participants)
    if frames:
        db.insert_many("frames", FRAME_COLUMNS, frames)
    if events:
        # Events have an autoincrement id, so re-ingesting a match would
        # duplicate them; clear this match's rows first.
        db.execute("DELETE FROM events WHERE match_id=?", (match_id,))
        db.insert_many("events", EVENT_COLUMNS, events, replace=False)

    return {"matches": 1, "participants": len(participants),
            "frames": len(frames), "events": len(events)}


def iter_match_pairs(raw_dir: Path) -> Iterator[tuple[str, Path, Path | None]]:
    """Yield ``(match_id, match_path, timeline_path_or_None)`` under ``raw_dir``."""
    if not raw_dir.exists():
        return
    for match_path in sorted(raw_dir.rglob("*.json.gz")):
        if match_path.name.endswith(".timeline.json.gz"):
            continue
        match_id = match_path.name[: -len(".json.gz")]
        tl = match_path.with_name(f"{match_id}.timeline.json.gz")
        yield match_id, match_path, (tl if tl.exists() else None)


def ingest_riot_dir(db: Database, raw_dir: Path, limit: int | None = None,
                    force: bool = False, progress: Any | None = None
                    ) -> dict[str, int]:
    """Ingest every downloaded match under ``raw_dir``.

    Already-ingested matches are skipped by consulting the database first,
    which makes this safe to run after every incremental download.
    """
    totals = {"matches": 0, "participants": 0, "frames": 0, "events": 0,
              "skipped": 0, "failed": 0}
    known: set[str] = set()
    if not force:
        known = {r[0] for r in db.query(
            "SELECT match_id FROM matches WHERE has_timeline=1")}

    done = 0
    for match_id, mpath, tpath in iter_match_pairs(raw_dir):
        if limit is not None and done >= limit:
            break
        if match_id in known:
            totals["skipped"] += 1
            continue
        try:
            counts = ingest_pair(db, mpath, tpath)
        except Exception as exc:
            log.warning("failed to ingest %s: %s", match_id, exc)
            totals["failed"] += 1
            continue
        for k, v in counts.items():
            totals[k] += v
        done += 1
        if progress is not None:
            progress(done, match_id)
    return totals


def match_summary(db: Database) -> dict[str, Any]:
    """Corpus statistics used by the CLI status output."""
    return {
        "matches": db.scalar("SELECT COUNT(*) FROM matches") or 0,
        "with_timeline": db.scalar(
            "SELECT COUNT(*) FROM matches WHERE has_timeline=1") or 0,
        "frames": db.scalar("SELECT COUNT(*) FROM frames") or 0,
        "events": db.scalar("SELECT COUNT(*) FROM events") or 0,
        "patches": [r[0] for r in db.query(
            "SELECT DISTINCT patch FROM matches WHERE patch IS NOT NULL "
            "ORDER BY patch DESC LIMIT 10")],
    }
