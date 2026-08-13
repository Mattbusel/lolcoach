"""Local UI assets: the real minimap and champion portraits.

The review app used to draw Summoner's Rift by hand with bezier curves and
hard-coded label positions, which looked approximate because it was. Riot
publishes the actual minimap and every champion's square portrait through
Data Dragon, free and without a key, so the app uses those instead.

Assets are cached under ``<data root>/assets/<version>/`` and served from
there. Two consequences that matter:

* the packaged desktop build works offline once assets are fetched;
* nothing is hot-linked to Riot's CDN at render time, so the UI stays fast
  and does not depend on the network mid-review.

Coordinate mapping
------------------
Timeline positions run 0..14870 on x and 0..14980 on y with the origin at
blue's bottom-left corner. The minimap image is square with the same origin,
so mapping is a straight scale plus a y-flip (screen y grows downward). The
bounds are exposed here so the map, the fog overlay and any future heatmap
all agree on one transform.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from .config import Config
from .logging_utils import get_logger
from .storage import Database

log = get_logger("assets")

DDRAGON = "https://ddragon.leagueoflegends.com"

#: Playable bounds of Summoner's Rift in timeline coordinates.
MAP_MIN_X, MAP_MAX_X = -120.0, 14870.0
MAP_MIN_Y, MAP_MAX_Y = -120.0, 14980.0

#: Riot's map id for Summoner's Rift.
SUMMONERS_RIFT_MAP_ID = 11


def assets_root(cfg: Config, version: str) -> Path:
    return cfg.paths.root / "assets" / version


def map_image_path(cfg: Config, version: str) -> Path:
    return assets_root(cfg, version) / "map.png"


def champion_icon_path(cfg: Config, version: str, key: str) -> Path:
    return assets_root(cfg, version) / "champions" / f"{key}.png"


def current_version(cfg: Config) -> str:
    """Data Dragon version recorded by the download stage, or a sane default."""
    db = Database(cfg.paths.db)
    try:
        stored = db.get_meta("ddragon_version")
    finally:
        db.close()
    if stored:
        return str(stored)
    # Fall back to the newest version directory already on disk.
    raw = cfg.paths.source("ddragon")
    if raw.is_dir():
        versions = sorted((p.name for p in raw.iterdir() if p.is_dir()),
                          key=_version_key)
        if versions:
            return versions[-1]
    return "16.16.1"


def _version_key(name: str) -> tuple[int, ...]:
    parts = []
    for chunk in name.split("."):
        parts.append(int(chunk) if chunk.isdigit() else 0)
    return tuple(parts)


def champion_keys(cfg: Config) -> list[str]:
    """Data Dragon image keys for every champion in the local database.

    The key is not always the display name -- Wukong's files are named
    ``MonkeyKing`` -- so this reads the key column rather than guessing from
    the name.
    """
    db = Database(cfg.paths.db)
    try:
        rows = db.query("SELECT key FROM champions WHERE key IS NOT NULL ORDER BY key")
    finally:
        db.close()
    return [str(row[0]) for row in rows]


def missing_assets(cfg: Config, version: str | None = None) -> dict[str, int]:
    """Count what still needs downloading, for the UI's setup panel."""
    version = version or current_version(cfg)
    keys = champion_keys(cfg)
    missing_icons = sum(
        1 for key in keys if not champion_icon_path(cfg, version, key).is_file())
    return {
        "version": version,
        "map": 0 if map_image_path(cfg, version).is_file() else 1,
        "icons_missing": missing_icons,
        "icons_total": len(keys),
    }


def ensure_assets(cfg: Config, version: str | None = None,
                  force: bool = False, workers: int = 8) -> dict[str, int]:
    """Download the minimap and champion portraits if they are not cached.

    Returns counts of what was fetched. Failures for individual icons are
    tolerated: a missing portrait degrades to an initials badge in the UI,
    which is far better than failing the whole review screen.
    """
    from .http import HttpClient

    version = version or current_version(cfg)
    root = assets_root(cfg, version)
    (root / "champions").mkdir(parents=True, exist_ok=True)

    fetched = {"map": 0, "icons": 0, "failed": 0}
    keys = champion_keys(cfg)

    with HttpClient(user_agent=cfg.user_agent, timeout=cfg.http_timeout,
                    retries=2) as http:
        map_path = map_image_path(cfg, version)
        if force or not map_path.is_file():
            try:
                data = http.get(f"{DDRAGON}/cdn/{version}/img/map/"
                                f"map{SUMMONERS_RIFT_MAP_ID}.png").content
                map_path.write_bytes(data)
                fetched["map"] = 1
            except Exception as exc:
                log.warning("could not fetch the minimap image: %s", exc)
                fetched["failed"] += 1

        def one(key: str) -> bool:
            dest = champion_icon_path(cfg, version, key)
            if dest.is_file() and not force:
                return False
            try:
                data = http.get(f"{DDRAGON}/cdn/{version}/img/champion/{key}.png").content
            except Exception:
                return False
            dest.write_bytes(data)
            return True

        if keys:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for downloaded in pool.map(one, keys):
                    if downloaded:
                        fetched["icons"] += 1

    if fetched["map"] or fetched["icons"]:
        log.info("cached UI assets for %s: map=%d icons=%d",
                 version, fetched["map"], fetched["icons"])
    return fetched


def map_bounds() -> dict[str, float]:
    """Transform constants shared with the browser."""
    return {"min_x": MAP_MIN_X, "max_x": MAP_MAX_X,
            "min_y": MAP_MIN_Y, "max_y": MAP_MAX_Y}
