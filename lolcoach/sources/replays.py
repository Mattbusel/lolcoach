"""Local ``.rofl`` replay files.

What is and is not possible
---------------------------
A ``.rofl`` file has two parts: a plaintext JSON metadata block and an
**encrypted** payload of game chunks and keyframes. Riot does not publish the
per-game decryption keys, and the payload is the only place the frame-by-frame
game state lives. So no honest pipeline can turn a raw ``.rofl`` into a
timeline the way it can with a Riot API match timeline.

What this connector does, therefore, is extract the plaintext metadata block,
which is genuinely useful and genuinely available: game version, duration, and
the complete end-of-game stats line for all ten players (champion, KDA, gold,
CS, damage, vision, items, runes, and per-role identifiers). That yields
matchup outcome data for a player's own games and it links a local replay to
its Riot match id, so any timeline already downloaded for that match is joined
automatically.

It reads only from local directories the operator points it at. It never
downloads replays from third-party sites, which is both a legal and a
practical decision: those files would be equally undecryptable.

Format
------
Classic header layout, little-endian::

    0    magic "RIOT\\x00\\x00"          6 bytes
    6    signature                       256 bytes
    262  header length                   uint16
    264  file length                     uint32
    268  metadata offset                 uint32
    272  metadata length                 uint32
    276  payload header offset           uint32
    280  payload header length           uint32
    284  payload offset                  uint32

Riot has revised the container more than once, so when the header does not
parse the file is scanned for the metadata JSON directly, which has survived
every revision so far.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..logging_utils import get_logger
from ..storage import DownloadRecord
from .base import LicenseInfo, Source, SourceContext, SourceResult, register

log = get_logger("sources.replays")

MAGIC = b"RIOT\x00\x00"
HEADER_STRUCT = "<HIIIIII"      # from offset 262
HEADER_OFFSET = 262

#: Current replays carry the client build (``16.11.782.9736``) in the header
#: even though their metadata block no longer includes ``gameVersion``, so the
#: patch is recovered by scanning the header for a four-part build string.
_BUILD_RE = re.compile(rb"\b(\d{1,2}\.\d{1,2}\.\d{2,4}\.\d{2,5})\b")


@dataclass
class ReplayMetadata:
    """Plaintext portion of a replay file."""

    path: str
    game_id: str = ""
    game_version: str = ""
    game_length_ms: int = 0
    players: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def patch(self) -> str:
        parts = self.game_version.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else self.game_version


def _extract_json_object(blob: bytes, start: int) -> str | None:
    """Return the balanced JSON object beginning at ``start``.

    Brace counting is done outside of string literals so that braces inside
    values (common in the embedded ``statsJson``) do not end the object early.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(blob)):
        ch = blob[i]
        if escape:
            escape = False
            continue
        if ch == 0x5C:            # backslash
            escape = True
            continue
        if ch == 0x22:            # double quote
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == 0x7B:            # {
            depth += 1
        elif ch == 0x7D:          # }
            depth -= 1
            if depth == 0:
                return blob[start:i + 1].decode("utf-8", errors="replace")
    return None


def parse_rofl(path: Path) -> ReplayMetadata | None:
    """Parse the metadata block out of a replay file.

    Returns ``None`` when the file is not a replay or its metadata cannot be
    located; callers treat that as "skip this file", never as an error.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        log.debug("cannot read %s: %s", path, exc)
        return None
    if len(data) < 300:
        return None

    meta_text: str | None = None
    if data[:6] == MAGIC:
        try:
            (_hdr_len, _file_len, meta_off, meta_len,
             _ph_off, _ph_len, _payload_off) = struct.unpack_from(
                HEADER_STRUCT, data, HEADER_OFFSET)
            if 0 < meta_len and meta_off + meta_len <= len(data):
                candidate = data[meta_off:meta_off + meta_len]
                meta_text = candidate.decode("utf-8", errors="replace").strip("\x00 ")
        except struct.error:
            meta_text = None

    if not meta_text or not meta_text.lstrip().startswith("{"):
        # Fall back to locating the metadata object by its distinctive keys.
        for needle in (b'{"gameLength"', b'{"gameVersion"', b'"statsJson"'):
            idx = data.find(needle)
            if idx == -1:
                continue
            start = idx if needle.startswith(b"{") else data.rfind(b"{", 0, idx)
            if start == -1:
                continue
            meta_text = _extract_json_object(data, start)
            if meta_text:
                break

    if not meta_text:
        return None
    try:
        meta = json.loads(meta_text)
    except ValueError:
        return None

    players: list[dict[str, Any]] = []
    stats = meta.get("statsJson")
    if isinstance(stats, str):
        try:
            players = json.loads(stats)
        except ValueError:
            players = []
    elif isinstance(stats, list):
        players = stats

    version = str(meta.get("gameVersion") or "")
    if not version:
        build = _BUILD_RE.search(data[:4096])
        if build:
            version = build.group(1).decode("ascii")

    return ReplayMetadata(
        path=str(path),
        game_id=str(meta.get("gameId") or path.stem),
        game_version=version,
        game_length_ms=int(meta.get("gameLength") or 0),
        players=players if isinstance(players, list) else [],
        raw={k: v for k, v in meta.items() if k != "statsJson"},
    )


def iter_replays(dirs: list[str]) -> Iterator[Path]:
    """Yield ``.rofl`` files from every configured directory that exists."""
    for d in dirs:
        root = Path(d).expanduser()
        if root.is_dir():
            yield from sorted(root.rglob("*.rofl"))


@register
class ReplaySource(Source):
    """Index local replay files and store their end-of-game stats."""

    name = "replays"
    description = "Local .rofl replay metadata (payload is encrypted by Riot)"
    license = LicenseInfo(
        name="Local user replays (Riot Games client output)",
        url="https://support-leagueoflegends.riotgames.com/",
        attribution="Replay files generated locally by the League of Legends client.",
        redistributable=False)

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        dirs = list(ctx.cfg.sources.replay_dirs)
        present = [d for d in dirs if Path(d).expanduser().is_dir()]
        if not present:
            result.skipped = (f"no replay directory found (looked in: "
                              f"{', '.join(dirs) or 'nothing configured'})")
            return

        out = ctx.sub("metadata")
        parsed = 0
        failed = 0
        rows: list[tuple] = []
        part_rows: list[tuple] = []

        for rofl in iter_replays(present):
            meta = parse_rofl(rofl)
            if meta is None:
                failed += 1
                continue
            parsed += 1
            dest = out / f"{rofl.stem}.json"
            payload = {"metadata": meta.raw, "players": meta.players,
                       "source_file": meta.path}
            self.save_json(ctx, f"file://{rofl}", dest, payload, result,
                           kind="replay_metadata")

            match_id = _match_id_from_name(rofl.stem, meta.game_id)
            rows.append((match_id, "replay", None, None, None, meta.patch,
                         meta.game_version, None, meta.game_length_ms // 1000,
                         "LOCAL_REPLAY", None, 0, str(rofl), _now()))
            for idx, p in enumerate(meta.players, start=1):
                part_rows.append(_participant_row(match_id, idx, p))

            if len(rows) >= 200:
                self._flush(ctx, rows, part_rows, result)

        self._flush(ctx, rows, part_rows, result)
        result.note(f"parsed {parsed} replays"
                    + (f", {failed} unreadable" if failed else ""))
        if parsed and not any("timeline" in n for n in result.notes):
            result.note("replay payloads are encrypted by Riot, so only "
                        "end-of-game stats were extracted; timelines for these "
                        "games come from the Riot API when a key is configured")

    @staticmethod
    def _flush(ctx: SourceContext, rows: list[tuple], part_rows: list[tuple],
               result: SourceResult) -> None:
        if rows:
            result.records += ctx.db.insert_many(
                "matches",
                ["match_id", "source", "platform", "region", "queue_id", "patch",
                 "game_version", "game_start_ms", "duration_s", "tier",
                 "winner_team", "has_timeline", "raw_path", "ingested_at"], rows)
            rows.clear()
        if part_rows:
            ctx.db.insert_many(
                "participants",
                ["match_id", "participant_id", "puuid", "team_id", "champion",
                 "champion_id", "role", "lane", "win", "kills", "deaths",
                 "assists", "gold_earned", "cs", "damage_dealt", "vision_score",
                 "summoner1", "summoner2", "keystone", "items"], part_rows)
            part_rows.clear()


def _match_id_from_name(stem: str, game_id: str) -> str:
    """Replay files are named ``NA1-1234567890``; normalise to ``NA1_1234567890``."""
    if "-" in stem:
        platform, _, tail = stem.partition("-")
        if tail.isdigit():
            return f"{platform.upper()}_{tail}"
    return f"REPLAY_{game_id or stem}"


def _participant_row(match_id: str, idx: int, p: dict[str, Any]) -> tuple:
    """Map a replay stats entry onto the ``participants`` schema.

    Replay stats use string values throughout and a different key casing from
    match-v5, so every field is coerced defensively.
    """
    def num(*keys: str, default: int = 0) -> int:
        for k in keys:
            v = p.get(k)
            if v not in (None, ""):
                try:
                    return int(float(v))
                except (TypeError, ValueError):
                    continue
        return default

    team_id = num("TEAM", "teamId", default=0)
    win_raw = str(p.get("WIN", p.get("win", ""))).lower()
    win = 1 if win_raw in {"win", "true", "1"} else 0
    items = [num(f"ITEM{i}") for i in range(7)]
    return (
        match_id, idx, p.get("PUUID") or p.get("puuid"),
        team_id if team_id in (100, 200) else (100 if idx <= 5 else 200),
        p.get("SKIN") or p.get("championName"), num("CHAMPION_ID", "championId"),
        p.get("TEAM_POSITION") or p.get("INDIVIDUAL_POSITION") or p.get("teamPosition"),
        p.get("LANE") or p.get("lane"), win,
        num("CHAMPIONS_KILLED", "kills"), num("NUM_DEATHS", "deaths"),
        num("ASSISTS", "assists"), num("GOLD_EARNED", "goldEarned"),
        num("MINIONS_KILLED", "totalMinionsKilled") + num("NEUTRAL_MINIONS_KILLED"),
        num("TOTAL_DAMAGE_DEALT_TO_CHAMPIONS", "totalDamageDealtToChampions"),
        num("VISION_SCORE", "visionScore"),
        num("SUMMONER_SPELL_1", "summoner1Id"), num("SUMMONER_SPELL_2", "summoner2Id"),
        num("KEYSTONE_ID", "perk0"), json.dumps(items),
    )


def _now() -> float:
    import time
    return time.time()
