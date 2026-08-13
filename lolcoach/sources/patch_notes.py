"""Riot patch notes.

Patch notes are what make a coaching model *current*. A recommendation that
was right two patches ago ("build this item first", "this jungle clear is
fastest") can be wrong today, so the retrieval corpus needs the change history
and Stage 8's ``update_patch`` needs somewhere to pull it from.

Acquisition
-----------
Riot's patch-notes index is a client-rendered page, so instead of scraping the
index this connector *derives* article URLs from the Data Dragon version list,
which is authoritative and needs no key. Patch ``15.16`` maps to
``/en-us/news/game-updates/patch-15-16-notes/``. The article HTML is then
reduced to prose.

The League wiki keeps a mirror of every patch at ``V15.16`` under a CC BY-SA
licence, so any patch whose Riot article cannot be retrieved falls back to the
wiki copy rather than being dropped.
"""

from __future__ import annotations

import re
import time
from typing import Iterable

from ..logging_utils import get_logger
from .base import CC_BY_SA, LicenseInfo, Source, SourceContext, SourceResult, register
from .ddragon import public_patch_label
from .wiki import API as WIKI_API, MediaWikiClient, clean_wikitext

log = get_logger("sources.patch_notes")

ARTICLE_URL = "https://www.leagueoflegends.com/en-us/news/game-updates/patch-{slug}-notes/"
VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"

RIOT_EDITORIAL = LicenseInfo(
    name="Riot Games editorial content",
    url="https://www.riotgames.com/en/terms-of-service",
    attribution="Patch notes (c) Riot Games, Inc. Used for research; not redistributed.",
    redistributable=False,
)

_SCRIPTS = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")


def html_to_text(html: str) -> str:
    """Reduce a patch-notes article to readable prose.

    Riot marks up changes as nested lists inside ``<div>`` blocks; converting
    block-level tags to newlines before stripping the rest keeps each buff or
    nerf on its own line, which matters because chunks are later embedded and
    a run-together wall of text retrieves poorly.
    """
    html = _SCRIPTS.sub(" ", html)
    html = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", r"\n\n\2\n", html,
                  flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"</?(p|li|div|tr|br|section)[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</t[dh]>", " | ", html, flags=re.IGNORECASE)
    text = _TAGS.sub(" ", html)
    import html as html_lib
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def patch_slug(patch: str) -> str:
    """``26.16`` -> ``26-16`` for the article URL."""
    return patch.replace(".", "-")


def recent_patches(http, limit: int = 40) -> list[tuple[str, str]]:
    """Return ``(ddragon_patch, public_label)`` pairs, newest first.

    Both forms are needed: the Data Dragon key ties a document to the static
    data it was collected with, while the public label is what the wiki files
    the notes under and what a user will type in a question.
    """
    versions = http.get_json(VERSIONS_URL, use_cache=True, max_age=3600)
    seen: list[tuple[str, str]] = []
    keys: set[str] = set()
    for v in versions:
        parts = str(v).split(".")
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        key = f"{parts[0]}.{parts[1]}"
        if key in keys:
            continue
        keys.add(key)
        seen.append((key, public_patch_label(str(v))))
        if len(seen) >= limit:
            break
    return seen


@register
class PatchNotesSource(Source):
    """Fetch the last N patches as retrieval documents."""

    name = "patch_notes"
    description = "Riot patch notes (with League wiki fallback)"
    license = RIOT_EDITORIAL

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        limit = ctx.limit or ctx.cfg.sources.patch_notes_limit
        patches = recent_patches(ctx.http, limit)
        if not patches:
            result.ok = False
            result.error = "Data Dragon returned no versions"
            return
        result.note(f"targeting {len(patches)} patches: "
                    f"{patches[0][1]} .. {patches[-1][1]}")

        wiki = MediaWikiClient(ctx.http, WIKI_API)
        out_dir = ctx.sub("articles")
        rows: list[tuple] = []
        riot_ok = 0

        # The wiki is fetched in batches of 50, which is one request per 50
        # patches instead of one per patch.
        titles = [f"V{label}" for _, label in patches]
        mirrors: dict[str, str] = {}
        for i in range(0, len(titles), 50):
            try:
                mirrors.update(wiki.wikitext(titles[i:i + 50]))
            except Exception as exc:
                log.debug("wiki batch %d failed: %s", i, exc)

        for ddragon_patch, label in patches:
            text, source_url, licence = "", "", self.license

            mirror_raw = mirrors.get(f"V{label}", "")
            if mirror_raw:
                text = clean_wikitext(mirror_raw)
                source_url = f"https://wiki.leagueoflegends.com/en-us/V{label}"
                licence = CC_BY_SA

            # Riot's own article is preferred when reachable, but its slugs are
            # not derivable from the patch number any more, so this is a
            # best-effort attempt on the newest patch only rather than a
            # guaranteed 40 wasted requests.
            if riot_ok == 0 and label == patches[0][1]:
                url = ARTICLE_URL.format(slug=patch_slug(label))
                try:
                    resp = ctx.http.get(url, use_cache=True, max_age=7 * 86400,
                                        raise_for_status=False)
                    if resp.status_code == 200:
                        official = html_to_text(resp.text)
                        if len(official) > len(text):
                            text, source_url, licence = official, url, self.license
                            riot_ok += 1
                except Exception as exc:
                    log.debug("riot article for %s unavailable: %s", label, exc)

            if len(text) < 400:
                result.note(f"patch {label}: no usable text")
                continue

            self.save_bytes(ctx, source_url, out_dir / f"patch-{label}.txt",
                            text.encode("utf-8"), result, kind="patch_notes")
            rows.append((f"patch:{label}", "patch_notes", "patch",
                         f"Patch {label} notes", source_url, label, None,
                         licence.name, 0.0, time.time(), text))

        if rows:
            result.records = ctx.db.insert_many(
                "documents",
                ["doc_id", "source", "kind", "title", "url", "patch", "champion",
                 "license", "score", "created_at", "text"], rows)
        result.note(f"stored {result.records} patch documents")


def stored_patches(db) -> list[str]:
    """Patch labels already present in the corpus, newest first."""
    rows = db.query("SELECT patch FROM documents WHERE source='patch_notes' "
                    "AND patch IS NOT NULL")
    def key(p: str) -> tuple[int, int]:
        parts = p.split(".")
        try:
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return (0, 0)
    return sorted({r[0] for r in rows}, key=key, reverse=True)
