"""Community discussion mining from Reddit.

r/summonerschool in particular is a large corpus of exactly the question form
this project targets: a player describes a situation and stronger players
explain the correct decision. That maps directly onto instruction data, and
the top-voted answer is a usable weak label for "good coaching".

Access
------
Reddit's own JSON endpoints now reject unauthenticated clients (HTTP 403), and
working around that would mean evading an access control. Instead this reads
the **Arctic Shift** public archive, a research mirror of Reddit's public data
that exists to serve bulk analysis and imposes no auth. Content remains the
property of its authors, which the manifest records; the pipeline uses it to
learn from, and the RAG layer always cites the original permalink.

Quality filtering
-----------------
Raw Reddit is mostly noise for this purpose, so posts are kept only when they
have a real body, clear the score threshold, and read like a question or an
explanation rather than a meme. For kept posts, the top comments are pulled as
the candidate answers.
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterator

from ..logging_utils import get_logger
from .base import REDDIT_UGC, Source, SourceContext, SourceResult, register

log = get_logger("sources.reddit")

BASE = "https://arctic-shift.photon-reddit.com/api"
POSTS = f"{BASE}/posts/search"
COMMENTS = f"{BASE}/comments/search"

#: Posts shorter than this are almost always link dumps or memes.
MIN_BODY_CHARS = 220

#: Signals that a post is a coaching question or explanation.
USEFUL_PATTERNS = re.compile(
    r"\b(how do i|how should i|when should i|should i|why did|why do|what do i do|"
    r"wave|freeze|crash|slow ?push|prio|priority|recall|back timing|reset|trade|"
    r"trading|matchup|jungle path|pathing|gank|track|counter|rotate|rotation|"
    r"objective|dragon|baron|herald|grub|vision|ward|tempo|tilt|macro|micro|"
    r"positioning|teamfight|split ?push|side ?lane|lane assignment|build|runes)\b",
    re.IGNORECASE)

#: Content that is not coaching, however popular.
NOISE_PATTERNS = re.compile(
    r"\b(meme|fan ?art|cosplay|skin concept|giveaway|looking for duo|lf duo|"
    r"add me|my rank up post|clip of|montage)\b", re.IGNORECASE)


def looks_useful(title: str, body: str) -> bool:
    """Cheap relevance gate applied before anything is stored."""
    blob = f"{title}\n{body}"
    if NOISE_PATTERNS.search(blob):
        return False
    if len(body) < MIN_BODY_CHARS:
        return False
    return bool(USEFUL_PATTERNS.search(blob))


class ArcticShiftClient:
    """Paged reader over the Arctic Shift archive."""

    def __init__(self, http: Any) -> None:
        self.http = http

    def posts(self, subreddit: str, limit: int, min_score: int = 0,
              page_size: int = 100) -> Iterator[dict[str, Any]]:
        """Yield posts newest-first, walking backwards through time.

        The archive caps a single response at 100 items, so paging uses the
        oldest ``created_utc`` seen so far as the next ``before`` cursor.
        """
        before: int | None = None
        fetched = 0
        stalls = 0
        while fetched < limit and stalls < 3:
            params: dict[str, Any] = {
                "subreddit": subreddit,
                "limit": min(page_size, limit - fetched),
                "sort": "desc",
            }
            if before is not None:
                params["before"] = before
            try:
                payload = self.http.get_json(POSTS, params=params, use_cache=True,
                                             max_age=7 * 86400)
            except Exception as exc:
                log.warning("arctic shift posts failed for r/%s: %s", subreddit, exc)
                return
            items = payload.get("data") if isinstance(payload, dict) else payload
            if not items:
                return
            oldest = before
            for item in items:
                created = int(item.get("created_utc") or 0)
                if oldest is None or (created and created < oldest):
                    oldest = created
                if int(item.get("score") or 0) < min_score:
                    continue
                yield item
                fetched += 1
                if fetched >= limit:
                    return
            if oldest == before or oldest is None:
                stalls += 1
            else:
                stalls = 0
            before = oldest

    def comments_for(self, link_id: str, limit: int = 25) -> list[dict[str, Any]]:
        """Top-level comments for a post id, best first."""
        try:
            payload = self.http.get_json(COMMENTS, params={
                "link_id": link_id, "limit": limit,
            }, use_cache=True, max_age=7 * 86400)
        except Exception as exc:
            log.debug("comments failed for %s: %s", link_id, exc)
            return []
        items = payload.get("data") if isinstance(payload, dict) else payload
        items = items or []
        items.sort(key=lambda c: -int(c.get("score") or 0))
        return items


@register
class RedditSource(Source):
    """Collect coaching questions and their best answers."""

    name = "reddit"
    description = "Reddit coaching discussions via the Arctic Shift archive"
    license = REDDIT_UGC

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        cfg = ctx.cfg.sources
        client = ArcticShiftClient(ctx.http)
        per_sub = ctx.limit or cfg.reddit_posts_per_sub
        raw_dir = ctx.sub("posts")
        rows: list[tuple] = []
        kept_total = 0

        for sub in cfg.reddit_subreddits:
            kept = 0
            scanned = 0
            for post in client.posts(sub, per_sub, min_score=cfg.reddit_min_score):
                scanned += 1
                title = (post.get("title") or "").strip()
                body = (post.get("selftext") or "").strip()
                if body in {"[removed]", "[deleted]"}:
                    body = ""
                if not looks_useful(title, body):
                    continue

                post_id = post.get("id") or ""
                permalink = post.get("permalink") or f"/r/{sub}/comments/{post_id}/"
                url = f"https://www.reddit.com{permalink}"
                score = float(post.get("score") or 0)

                answers = []
                if post_id:
                    for c in client.comments_for(f"t3_{post_id}", limit=12):
                        cbody = (c.get("body") or "").strip()
                        if len(cbody) < 160 or cbody in {"[removed]", "[deleted]"}:
                            continue
                        answers.append({"score": int(c.get("score") or 0), "body": cbody})
                        if len(answers) >= 5:
                            break

                text_parts = [f"Question ({sub}, score {int(score)}): {title}", "", body]
                for i, a in enumerate(answers, 1):
                    text_parts += ["", f"Answer {i} (score {a['score']}):", a["body"]]
                text = "\n".join(text_parts).strip()
                if len(text) < 300:
                    continue

                doc_id = f"reddit:{sub}:{post_id}"
                rows.append((doc_id, "reddit", "discussion", title, url,
                             None, None, self.license.name, score,
                             float(post.get("created_utc") or time.time()), text))
                self.save_bytes(ctx, url, raw_dir / sub / f"{post_id}.txt",
                                text.encode("utf-8"), result, kind="reddit_post")
                kept += 1
                if len(rows) >= 200:
                    result.records += self._flush(ctx, rows)
                    rows.clear()

            kept_total += kept
            result.note(f"r/{sub}: kept {kept} of {scanned} scanned")

        result.records += self._flush(ctx, rows)
        result.note(f"stored {result.records} discussion documents "
                    f"({kept_total} posts passed filtering)")

    @staticmethod
    def _flush(ctx: SourceContext, rows: list[tuple]) -> int:
        if not rows:
            return 0
        return ctx.db.insert_many(
            "documents",
            ["doc_id", "source", "kind", "title", "url", "patch", "champion",
             "license", "score", "created_at", "text"], rows)
