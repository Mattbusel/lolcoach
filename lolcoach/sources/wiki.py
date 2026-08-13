"""League of Legends Wiki (MediaWiki) crawler.

The wiki is the densest freely-licensed source of the mechanical knowledge a
coaching model needs: minion wave composition and spawn cadence, turret plate
rules, jungle camp respawn timers, dragon soul effects, champion ability
interactions, and per-patch change history. It is CC BY-SA 4.0, so it can be
used and redistributed with attribution -- which the manifest records.

Crawl strategy
--------------
Rather than spidering HTML, this uses the MediaWiki API:

1. ``list=categorymembers`` expands a curated set of categories (champions,
   items, runes, gameplay elements) into page titles.
2. A fixed list of high-value gameplay pages is added explicitly, because the
   mechanics pages a coach needs are not all in one tidy category.
3. Page text is fetched in batches. ``prop=extracts&explaintext`` is tried
   first because it returns clean prose; wikis without the TextExtracts
   extension fall back to ``action=parse`` HTML, then to raw wikitext, both of
   which are cleaned locally.

Every page becomes a row in ``documents`` so Stage 4 can chunk and embed it.
"""

from __future__ import annotations

import html as html_lib
import re
import time
from typing import Any, Iterable, Iterator

from ..logging_utils import get_logger
from .base import CC_BY_SA, Source, SourceContext, SourceResult, register

log = get_logger("sources.wiki")

API = "https://wiki.leagueoflegends.com/en-us/api.php"
PAGE_URL = "https://wiki.leagueoflegends.com/en-us/{title}"

#: Categories expanded into page lists. These are the names that actually
#: exist on wiki.leagueoflegends.com -- "Game mechanics" and "Structures",
#: the obvious guesses, are empty there, while "Gameplay elements" holds the
#: mechanics pages. The champion and item lists do not come from categories
#: at all: Data Dragon already enumerates them exactly, so the connector uses
#: that as its index and asks the wiki for those titles directly.
CATEGORIES = [
    "Gameplay elements",
    "Champions",
    "Items",
    "Runes",
    "Summoner spells",
    "Monsters",
    "Terminology",
]

#: Pages that matter for coaching but are not reliably categorised.
CORE_PAGES = [
    "Minion", "Siege minion", "Super minion", "Experience", "Gold", "Turret",
    "Inhibitor", "Nexus", "Jungling", "Jungle", "Monster", "Dragon",
    "Baron Nashor", "Rift Herald", "Void Grub", "Atakhan", "Elder Dragon",
    "Ward", "Vision", "Warding", "Summoner's Rift", "Lane", "Last hit",
    "Wave management", "Freezing", "Zoning", "Roaming", "Split pushing",
    "Teamfight", "Trading", "Recall", "Death", "Bounty", "Turret plating",
    "Champion statistic", "Attack damage", "Ability power", "Armor",
    "Magic resistance", "Attack speed", "Movement speed", "Cooldown reduction",
    "Ability haste", "Critical strike chance", "Lifesteal", "Tenacity",
]

_TAGS = re.compile(r"<[^>]+>")
_REFS = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TEMPLATES = re.compile(r"\{\{[^{}]*\}\}")
_TABLES = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_FILES = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.IGNORECASE)
_LINKS = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_HEADERS = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)


def clean_wikitext(text: str) -> str:
    """Reduce wikitext to readable prose.

    Templates are removed iteratively because the wiki nests them several
    levels deep (``{{ai|Q|{{sbc|Ahri}}}}``); a single pass would leave the
    outer braces behind.
    """
    text = _REFS.sub(" ", text)
    text = _FILES.sub(" ", text)
    text = _TABLES.sub(" ", text)
    for _ in range(6):
        new = _TEMPLATES.sub(" ", text)
        if new == text:
            break
        text = new
    text = _LINKS.sub(r"\1", text)
    text = _HEADERS.sub(r"\n\1:\n", text)
    text = _TAGS.sub(" ", text)
    text = html_lib.unescape(text)
    text = text.replace("'''", "").replace("''", "")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip()


def clean_html(text: str) -> str:
    """Strip MediaWiki-rendered HTML down to prose."""
    text = re.sub(r"<(script|style|table)[^>]*>.*?</\1>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", r"\n\n\2:\n", text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?(p|li|div|br)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = _TAGS.sub(" ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip()


class MediaWikiClient:
    """Thin MediaWiki API client with continuation handling."""

    def __init__(self, http: Any, api: str = API) -> None:
        self.http = http
        self.api = api
        self._has_extracts: bool | None = None

    def query(self, **params: Any) -> Iterator[dict[str, Any]]:
        """Run an API query, following ``continue`` tokens to exhaustion."""
        params = {"action": "query", "format": "json", "formatversion": "2", **params}
        while True:
            data = self.http.get_json(self.api, params=params, use_cache=True,
                                      max_age=86400)
            if "query" in data:
                yield data["query"]
            cont = data.get("continue")
            if not cont:
                return
            params.update(cont)

    def category_members(self, category: str, limit: int = 5000) -> list[str]:
        """Page titles inside ``Category:<category>`` (namespace 0 only)."""
        titles: list[str] = []
        for chunk in self.query(list="categorymembers",
                                cmtitle=f"Category:{category}",
                                cmlimit="500", cmnamespace="0"):
            for m in chunk.get("categorymembers", []):
                titles.append(m["title"])
                if len(titles) >= limit:
                    return titles
        return titles

    def extracts(self, titles: list[str]) -> dict[str, str]:
        """Plain-text extracts for up to 20 titles, or ``{}`` if unsupported."""
        if self._has_extracts is False:
            return {}
        try:
            out: dict[str, str] = {}
            for chunk in self.query(prop="extracts", explaintext="1",
                                    exlimit="20", titles="|".join(titles[:20])):
                for page in chunk.get("pages", []):
                    if page.get("extract"):
                        out[page["title"]] = page["extract"]
            self._has_extracts = bool(out)
            return out
        except Exception as exc:
            log.debug("extracts unavailable: %s", exc)
            self._has_extracts = False
            return {}

    def wikitext(self, titles: list[str]) -> dict[str, str]:
        """Raw wikitext for up to 50 titles."""
        out: dict[str, str] = {}
        for chunk in self.query(prop="revisions", rvprop="content", rvslots="main",
                                titles="|".join(titles[:50])):
            for page in chunk.get("pages", []):
                revs = page.get("revisions") or []
                if not revs:
                    continue
                content = revs[0].get("slots", {}).get("main", {}).get("content")
                if content:
                    out[page["title"]] = content
        return out

    def parsed_html(self, title: str) -> str:
        """Rendered HTML for one page (last-resort fallback)."""
        data = self.http.get_json(self.api, params={
            "action": "parse", "page": title, "prop": "text",
            "format": "json", "formatversion": "2",
        }, use_cache=True, max_age=86400)
        return (data.get("parse", {}) or {}).get("text", "") or ""


@register
class WikiSource(Source):
    """Fetch and store wiki pages as retrieval documents."""

    name = "wiki"
    description = "League of Legends Wiki pages (mechanics, champions, items)"
    license = CC_BY_SA

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        client = MediaWikiClient(ctx.http)
        max_pages = ctx.limit or ctx.cfg.sources.wiki_max_pages

        # Data Dragon is the authoritative index of champions, items and runes,
        # so those titles are taken from the database rather than guessed at
        # through categories (which on this wiki are incomplete).
        champion_names = {row[0] for row in ctx.db.query(
            "SELECT name FROM champions WHERE name IS NOT NULL")}
        item_names = [row[0] for row in ctx.db.query(
            "SELECT name FROM items WHERE name IS NOT NULL AND total_gold >= 800 "
            "ORDER BY total_gold DESC")]
        rune_names = [row[0] for row in ctx.db.query(
            "SELECT name FROM runes WHERE name IS NOT NULL")]

        titles: list[str] = list(CORE_PAGES)
        titles.extend(sorted(champion_names))
        titles.extend(rune_names)
        titles.extend(item_names)
        if champion_names:
            result.note(f"indexed {len(champion_names)} champions, "
                        f"{len(item_names)} items, {len(rune_names)} runes "
                        f"from Data Dragon")
        else:
            result.note("no Data Dragon data yet; run the ddragon source first "
                        "for full champion and item coverage")

        for cat in CATEGORIES:
            try:
                members = client.category_members(cat)
                titles.extend(members)
                if members:
                    result.note(f"category {cat}: {len(members)} pages")
            except Exception as exc:
                result.note(f"category {cat} failed: {exc}")

        # Deduplicate while preserving the curated pages' priority.
        seen: set[str] = set()
        ordered: list[str] = []
        for t in titles:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        ordered = ordered[:max_pages]
        result.note(f"fetching {len(ordered)} unique pages")
        patch = ctx.db.get_meta("current_patch", "")
        rows: list[tuple] = []
        raw_dir = ctx.sub("pages")

        for batch in _batched(ordered, 20):
            texts = client.extracts(batch)
            missing = [t for t in batch if t not in texts or len(texts[t]) < 200]
            if missing:
                for title, wt in client.wikitext(missing).items():
                    cleaned = clean_wikitext(wt)
                    if len(cleaned) > len(texts.get(title, "")):
                        texts[title] = cleaned
            still_missing = [t for t in batch if not texts.get(t)]
            for title in still_missing[:5]:      # HTML fallback is expensive
                try:
                    texts[title] = clean_html(client.parsed_html(title))
                except Exception:
                    continue

            for title, text in texts.items():
                if not text or len(text) < 120:
                    continue
                url = PAGE_URL.format(title=title.replace(" ", "_"))
                doc_id = f"wiki:{title}"
                champion = title if title in champion_names else None
                rows.append((doc_id, "wiki", _kind_of(title, champion_names), title,
                             url, patch, champion, self.license.name, 0.0,
                             time.time(), text))
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", title)[:120]
                self.save_bytes(ctx, url, raw_dir / f"{safe}.txt",
                                text.encode("utf-8"), result, kind="wiki_page")

            if len(rows) >= 200:
                result.records += _flush(ctx, rows)
                rows.clear()

        result.records += _flush(ctx, rows)
        result.note(f"stored {result.records} wiki documents")


def _kind_of(title: str, champion_names: set[str]) -> str:
    if title in champion_names:
        return "champion"
    lowered = title.lower()
    for needle, kind in (("rune", "rune"), ("item", "item"), ("dragon", "objective"),
                         ("baron", "objective"), ("herald", "objective"),
                         ("minion", "mechanics"), ("turret", "objective")):
        if needle in lowered:
            return kind
    return "mechanics"


def _flush(ctx: SourceContext, rows: list[tuple]) -> int:
    if not rows:
        return 0
    return ctx.db.insert_many(
        "documents",
        ["doc_id", "source", "kind", "title", "url", "patch", "champion",
         "license", "score", "created_at", "text"], rows)


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
