"""GitHub dataset discovery.

A lot of League data lives in GitHub repositories rather than dataset hubs:
scraped matchup spreadsheets, community-maintained item and rune tables, jungle
clear timings, and pre-parsed timeline dumps from research projects.

This connector searches the GitHub API, filters to repositories that both look
relevant and carry a licence permitting reuse, then downloads only their data
files (CSV/JSON/TSV) under a size cap rather than cloning whole repositories.

Licence handling is deliberately strict: repositories with **no** licence are
recorded as discovered but not downloaded, because "no licence" means no
grant of rights, not public domain. Their metadata is still logged so an
operator can review and whitelist them by hand.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from ..storage import DownloadRecord
from .base import LicenseInfo, Source, SourceContext, SourceResult, register

log = get_logger("sources.github")

SEARCH_REPOS = "https://api.github.com/search/repositories"
CONTENTS = "https://api.github.com/repos/{repo}/contents/{path}"
TREES = "https://api.github.com/repos/{repo}/git/trees/{sha}"

#: SPDX ids that permit redistribution in a derived dataset.
PERMISSIVE = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc0-1.0",
    "cc-by-4.0", "cc-by-sa-4.0", "unlicense", "isc", "mpl-2.0",
    "gpl-3.0", "agpl-3.0", "lgpl-3.0", "odbl-1.0",
}

DATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".yaml", ".yml"}

POSITIVE = ("league", "lol", "riot", "summoner", "champion", "esports",
            "matchup", "timeline", "jungle", "rune")
NEGATIVE = ("dota", "valorant", "overwatch", "csgo", "minecraft", "pokemon")

#: Do not pull single files larger than this.
MAX_FILE_BYTES = 64 * 1024 * 1024
#: Or more than this per repository.
MAX_REPO_BYTES = 256 * 1024 * 1024


def repo_relevance(full_name: str, description: str, topics: list[str]) -> float:
    """Score a repository on how likely it is to hold League data."""
    blob = f"{full_name} {description} {' '.join(topics)}".lower()
    score = 0.0
    for term in POSITIVE:
        if term in blob:
            score += 1.0
    for term in NEGATIVE:
        if term in blob:
            score -= 1.5
    for word in ("dataset", "data", "stats", "analytics", "scraper"):
        if word in blob:
            score += 0.7
    return score


@register
class GitHubDatasetSource(Source):
    """Search GitHub for League datasets and pull their data files."""

    name = "github"
    description = "GitHub repositories containing League datasets"
    license = LicenseInfo(
        name="per-repository (recorded individually)",
        url="https://github.com",
        attribution="See the licence recorded for each downloaded repository.")
    requires_secret = ()   # works unauthenticated, but rate-limited to 10 req/min

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        token = ctx.cfg.secret("GITHUB_TOKEN", "GH_TOKEN")
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            result.note("no GITHUB_TOKEN: search is limited to 10 requests/minute")

        seen: dict[str, dict[str, Any]] = {}
        for query in ctx.cfg.sources.github_queries:
            try:
                payload = ctx.http.get_json(
                    SEARCH_REPOS,
                    params={"q": f"{query} in:name,description,readme",
                            "sort": "stars", "order": "desc", "per_page": 40},
                    headers=headers, use_cache=True, max_age=86400)
            except Exception as exc:
                result.note(f"search '{query}' failed: {exc}")
                continue
            for item in (payload or {}).get("items", []):
                seen.setdefault(item["full_name"], item)

        result.note(f"{len(seen)} unique repositories found")
        accepted: list[tuple[float, dict[str, Any]]] = []
        for full_name, item in seen.items():
            topics = [str(t) for t in item.get("topics", [])]
            score = repo_relevance(full_name, item.get("description") or "", topics)
            lic = ((item.get("license") or {}).get("spdx_id") or "").lower()
            permitted = lic in PERMISSIVE
            ctx.manifest.add_discovered(
                key=f"gh:{full_name}", provider="github", identifier=full_name,
                title=(item.get("description") or "")[:200],
                url=item.get("html_url", ""), license=lic or "none",
                relevance=score, accepted=bool(score >= 2.0 and permitted),
                meta={"stars": item.get("stargazers_count"), "topics": topics[:20]})
            if score >= 2.0 and permitted:
                accepted.append((score, item))
            elif score >= 2.0 and not permitted:
                result.note(f"{full_name}: relevant but licence '{lic or 'none'}' "
                            f"does not grant reuse; recorded, not downloaded")

        accepted.sort(key=lambda x: -x[0])
        for _, item in accepted[: (ctx.limit or 15)]:
            self._download_repo(ctx, result, item, headers)

    def _download_repo(self, ctx: SourceContext, result: SourceResult,
                       item: dict[str, Any], headers: dict[str, str]) -> None:
        """Fetch the data files from one repository via the git tree API."""
        repo = item["full_name"]
        branch = item.get("default_branch") or "main"
        licence = ((item.get("license") or {}).get("spdx_id") or "unknown")
        try:
            tree = ctx.http.get_json(
                TREES.format(repo=repo, sha=branch), params={"recursive": "1"},
                headers=headers, use_cache=True, max_age=86400)
        except Exception as exc:
            result.note(f"{repo}: tree listing failed ({exc})")
            return

        out = ctx.out_dir / repo.replace("/", "__")
        total = 0
        files = 0
        for node in (tree or {}).get("tree", []):
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            size = int(node.get("size") or 0)
            if Path(path).suffix.lower() not in DATA_SUFFIXES:
                continue
            if size > MAX_FILE_BYTES or total + size > MAX_REPO_BYTES:
                continue
            raw = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
            dest = out / path
            if self.should_skip(ctx, raw, dest):
                continue
            try:
                data = ctx.http.get(raw).content
            except Exception:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            import hashlib
            ctx.manifest.record(DownloadRecord(
                url=raw, path=str(dest), source=self.name, kind="repo_data",
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                license=licence, attribution=f"GitHub repository {repo} ({licence})"))
            total += len(data)
            files += 1

        if files:
            ctx.manifest.mark_integrated(f"gh:{repo}")
            result.files += files
            result.bytes += total
            result.records += 1
            result.note(f"{repo}: {files} data files, {total / 1e6:.1f} MB ({licence})")
