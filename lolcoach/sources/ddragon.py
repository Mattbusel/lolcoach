"""Riot Data Dragon and Community Dragon static game data.

Data Dragon is Riot's own CDN of patch-versioned static data: champions with
their full ability blocks (including per-rank cooldowns and damage), items
with costs and build paths, runes, and summoner spells. It needs no API key
and no authentication, which makes it the reliable factual backbone of the
whole system:

* Stage 4 embeds it, so retrieval can answer "what does this item do".
* Stage 7 checks model answers against it, so "factual accuracy" is measured
  against ground truth rather than a judge's opinion.
* Stage 2 uses base stats and ability cooldowns to reason about trades.

Community Dragon is a community-run mirror that exposes the *game client's*
own data files. It carries fields Data Dragon omits -- notably per-rank
cooldown arrays for reworked abilities and the full perk (rune) descriptions
with their real numbers -- so both are fetched and merged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import LicenseInfo, Source, SourceContext, SourceResult, register

DDRAGON = "https://ddragon.leagueoflegends.com"
CDRAGON = "https://raw.communitydragon.org"

RIOT_STATIC_LICENSE = LicenseInfo(
    name="Riot Games static data (Data Dragon)",
    url="https://developer.riotgames.com/docs/lol#data-dragon",
    attribution="Static game data from Riot Games Data Dragon. League of Legends "
                "is a trademark of Riot Games, Inc.",
    redistributable=False,
)

#: Tag stripper for Riot's ability tooltips, which are HTML-ish with custom
#: markup such as ``<physicalDamage>`` and ``<br>``.
_TAG = re.compile(r"<[^>]+>")


def strip_tags(text: str) -> str:
    """Turn a Riot tooltip into clean prose suitable for embedding."""
    if not text:
        return ""
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = _TAG.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def latest_version(http: Any) -> str:
    """Newest Data Dragon version string, e.g. ``15.16.1``."""
    versions = http.get_json(f"{DDRAGON}/api/versions.json", use_cache=True, max_age=3600)
    if not versions:
        raise RuntimeError("Data Dragon returned an empty version list")
    return str(versions[0])


def patch_of_version(version: str) -> str:
    """``16.16.1`` -> ``16.16`` (the Data Dragon patch key)."""
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


def public_patch_label(version: str) -> str:
    """Convert a Data Dragon version to the patch name players and Riot use.

    Riot moved to year-based patch names in 2025 but left Data Dragon on the
    old season numbering, so the two disagree by exactly ten majors: Data
    Dragon ``16.16`` is the patch everyone calls **26.16**, and the wiki files
    it under ``V26.16``. Seasons before 15 used the same number in both
    places, so those pass through unchanged.

    >>> public_patch_label("16.16.1")
    '26.16'
    >>> public_patch_label("14.19.1")
    '14.19'
    """
    parts = version.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return version
    major, minor = int(parts[0]), int(parts[1])
    if major >= 15:
        return f"{major + 10}.{minor:02d}"
    return f"{major}.{minor}"


@register
class DataDragonSource(Source):
    """Champions, items, runes and summoner spells for the current patch."""

    name = "ddragon"
    description = "Riot Data Dragon static data (champions, items, runes, spells)"
    license = RIOT_STATIC_LICENSE

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        version = latest_version(ctx.http)
        patch = patch_of_version(version)
        result.note(f"Data Dragon version {version} (patch {patch})")
        ctx.db.set_meta("ddragon_version", version)
        ctx.db.set_meta("current_patch", patch)

        base = f"{DDRAGON}/cdn/{version}/data/en_US"
        out = ctx.sub(version)

        files = {
            "championFull.json": "champions",
            "item.json": "items",
            "runesReforged.json": "runes",
            "summoner.json": "summoner_spells",
            "map.json": "maps",
        }
        payloads: dict[str, Any] = {}
        for fname, kind in files.items():
            url = f"{base}/{fname}"
            dest = out / fname
            if self.should_skip(ctx, url, dest):
                payloads[kind] = json.loads(dest.read_text(encoding="utf-8"))
                continue
            data = ctx.http.get(url, use_cache=True, max_age=86400).content
            self.save_bytes(ctx, url, dest, data, result, kind=kind)
            payloads[kind] = json.loads(data.decode("utf-8"))

        # Also fetch the version list itself so update_patch can diff offline.
        vurl = f"{DDRAGON}/api/versions.json"
        self.save_json(ctx, vurl, ctx.out_dir / "versions.json",
                       ctx.http.get_json(vurl, use_cache=True, max_age=3600),
                       result, kind="versions")

        result.records += self._load_champions(ctx, payloads.get("champions"), patch)
        result.records += self._load_items(ctx, payloads.get("items"), patch)
        result.records += self._load_runes(ctx, payloads.get("runes"), patch)
        result.note(f"loaded {result.records} static-data rows into SQLite")

    # -- SQLite loading -------------------------------------------------
    def _load_champions(self, ctx: SourceContext, payload: Any, patch: str) -> int:
        """Insert champions with their stats and full ability blocks."""
        if not payload or "data" not in payload:
            return 0
        rows = []
        for key, champ in payload["data"].items():
            spells = [{
                "id": s.get("id"),
                "name": s.get("name"),
                "description": strip_tags(s.get("description", "")),
                "tooltip": strip_tags(s.get("tooltip", "")),
                "cooldown": s.get("cooldown"),
                "cost": s.get("cost"),
                "range": s.get("range"),
                "maxrank": s.get("maxrank"),
                "costType": s.get("costType"),
            } for s in champ.get("spells", [])]
            passive = {
                "name": champ.get("passive", {}).get("name"),
                "description": strip_tags(champ.get("passive", {}).get("description", "")),
            }
            rows.append((
                int(champ.get("key", 0)), key, champ.get("name"), champ.get("title"),
                json.dumps(champ.get("tags", [])), champ.get("partype"),
                json.dumps(champ.get("stats", {})), json.dumps(spells),
                json.dumps(passive), patch,
            ))
        return ctx.db.insert_many(
            "champions",
            ["champion_id", "key", "name", "title", "tags", "resource",
             "stats", "spells", "passive", "patch"], rows)

    def _load_items(self, ctx: SourceContext, payload: Any, patch: str) -> int:
        """Insert items with cost, build path and cleaned description."""
        if not payload or "data" not in payload:
            return 0
        rows = []
        for item_id, item in payload["data"].items():
            gold = item.get("gold", {}) or {}
            rows.append((
                int(item_id), item.get("name"), int(gold.get("base", 0) or 0),
                int(gold.get("total", 0) or 0), json.dumps(item.get("tags", [])),
                json.dumps(item.get("stats", {})), json.dumps(item.get("into", [])),
                json.dumps(item.get("from", [])),
                strip_tags(item.get("description", "")), patch,
            ))
        return ctx.db.insert_many(
            "items",
            ["item_id", "name", "gold", "total_gold", "tags", "stats",
             "into_ids", "from_ids", "description", "patch"], rows)

    def _load_runes(self, ctx: SourceContext, payload: Any, patch: str) -> int:
        """Flatten the rune tree into one row per rune."""
        if not payload:
            return 0
        rows = []
        for tree in payload:
            tree_name = tree.get("name", "")
            for slot_idx, slot in enumerate(tree.get("slots", [])):
                for rune in slot.get("runes", []):
                    rows.append((
                        int(rune.get("id", 0)), tree_name, slot_idx, rune.get("name"),
                        strip_tags(rune.get("shortDesc", "")),
                        strip_tags(rune.get("longDesc", "")), patch,
                    ))
        return ctx.db.insert_many(
            "runes", ["rune_id", "tree", "slot", "name", "short_desc", "long_desc", "patch"],
            rows)


@register
class CommunityDragonSource(Source):
    """Client-side game data that fills gaps in Data Dragon.

    Fetched files
    -------------
    ``items.json``
        Full item set including ornn upgrades and hidden stats.
    ``perks.json`` / ``perkstyles.json``
        Rune definitions with their real coefficients.
    ``champion-summary.json``
        Champion id/name/role mapping used to resolve Riot's numeric ids.
    ``summoner-spells.json``
        Cooldowns, needed for reasoning about Flash/TP timers.
    """

    name = "cdragon"
    description = "Community Dragon client data (items, perks, spells)"
    license = LicenseInfo(
        name="Riot game client data via Community Dragon",
        url="https://communitydragon.org/",
        attribution="Game client data mirrored by Community Dragon.",
        redistributable=False,
    )

    ENDPOINTS = {
        "items.json": "v1/items.json",
        "perks.json": "v1/perks.json",
        "perkstyles.json": "v1/perkstyles.json",
        "champion-summary.json": "v1/champion-summary.json",
        "summoner-spells.json": "v1/summoner-spells.json",
        "queues.json": "v1/queues.json",
        "maps.json": "v1/maps.json",
    }

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        root = f"{CDRAGON}/latest/plugins/rcp-be-lol-game-data/global/default"
        out = ctx.sub("latest")
        for fname, suffix in self.ENDPOINTS.items():
            url = f"{root}/{suffix}"
            dest = out / fname
            if self.should_skip(ctx, url, dest, max_age_s=86400):
                continue
            try:
                data = ctx.http.get(url, use_cache=True, max_age=86400).content
            except Exception as exc:
                result.note(f"skipped {fname}: {exc}")
                continue
            self.save_bytes(ctx, url, dest, data, result, kind=fname.replace(".json", ""))
        result.note(f"{result.files} Community Dragon files")


def load_static_index(paths_root: Path) -> dict[str, Any]:
    """Load the newest Data Dragon payloads straight from disk.

    Used by stages that need champion/item facts without touching the network
    (benchmark scoring, dataset rendering). Returns a dict with ``champions``,
    ``items`` and ``runes`` keys; missing files yield empty dicts.
    """
    root = paths_root / "raw" / "ddragon"
    if not root.exists():
        return {"champions": {}, "items": {}, "runes": []}
    versions = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: [int(x) if x.isdigit() else 0 for x in p.name.split(".")],
    )
    if not versions:
        return {"champions": {}, "items": {}, "runes": []}
    newest = versions[-1]

    def read(name: str) -> Any:
        f = newest / name
        if not f.exists():
            return {}
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    champs = read("championFull.json").get("data", {})
    items = read("item.json").get("data", {})
    runes = read("runesReforged.json") or []
    return {"champions": champs, "items": items, "runes": runes, "version": newest.name}
