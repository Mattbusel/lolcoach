"""Kaggle dataset discovery and download.

Kaggle hosts several of the largest public League datasets: multi-million-row
ranked match dumps, challenger game collections, and per-minute timeline
exports that are otherwise only obtainable by crawling the Riot API for weeks.

Kaggle's REST API accepts HTTP Basic authentication with a username and API
key, so this connector talks to it directly rather than depending on the
``kaggle`` package (which insists on a config file at import time and calls
``sys.exit`` on failure -- unacceptable inside a long pipeline run).

Credentials are read from ``KAGGLE_USERNAME``/``KAGGLE_KEY`` or from
``~/.kaggle/kaggle.json``. Without them the connector reports a clean skip.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import httpx

from ..logging_utils import get_logger
from ..storage import DownloadRecord
from .base import LicenseInfo, Source, SourceContext, SourceResult, register

log = get_logger("sources.kaggle")

API = "https://www.kaggle.com/api/v1"

POSITIVE = ("league", "lol", "riot", "summoner", "champion", "esports",
            "ranked", "challenger", "timeline", "matchup")
NEGATIVE = ("dota", "valorant", "overwatch", "csgo", "fifa", "nba")

#: Licences on Kaggle that permit reuse for model training.
OPEN_LICENSES = {
    "cc0-1.0", "cc-by-4.0", "cc-by-sa-4.0", "cc-by-nc-sa-4.0", "odbl-1.0",
    "pddl", "gpl-2.0", "other", "unknown", "world bank dataset terms of use",
}


def kaggle_credentials(cfg: Any) -> tuple[str, str] | None:
    """Resolve ``(username, key)`` from the environment or ``kaggle.json``."""
    user = cfg.secret("KAGGLE_USERNAME")
    key = cfg.secret("KAGGLE_KEY")
    if user and key:
        return user, key
    cfg_file = Path.home() / ".kaggle" / "kaggle.json"
    if cfg_file.is_file():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            if data.get("username") and data.get("key"):
                return str(data["username"]), str(data["key"])
        except (OSError, ValueError):
            pass
    return None


def dataset_relevance(ref: str, title: str, subtitle: str) -> float:
    blob = f"{ref} {title} {subtitle}".lower()
    score = sum(1.0 for term in POSITIVE if term in blob)
    score -= sum(1.5 for term in NEGATIVE if term in blob)
    return score


@register
class KaggleSource(Source):
    """Search Kaggle and download the most relevant League datasets."""

    name = "kaggle"
    description = "Kaggle public League of Legends datasets"
    license = LicenseInfo(
        name="per-dataset (recorded individually)",
        url="https://www.kaggle.com/datasets",
        attribution="See the licence recorded for each downloaded dataset.")
    requires_secret = ("KAGGLE_KEY",)

    MAX_BYTES = 4 * 1024 ** 3

    def missing_secret(self, cfg: Any) -> str | None:
        """Kaggle also accepts ``~/.kaggle/kaggle.json``, so check both."""
        return None if kaggle_credentials(cfg) else "KAGGLE_KEY (or ~/.kaggle/kaggle.json)"

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        creds = kaggle_credentials(ctx.cfg)
        if not creds:
            result.skipped = "no Kaggle credentials"
            return
        auth = httpx.BasicAuth(*creds)

        found: dict[str, dict[str, Any]] = {}
        for query in ctx.cfg.sources.kaggle_queries:
            for page in (1, 2):
                try:
                    resp = ctx.http._client.get(          # noqa: SLF001 - needs auth
                        f"{API}/datasets/list",
                        params={"search": query, "page": page, "sortBy": "votes"},
                        auth=auth, timeout=60.0)
                    resp.raise_for_status()
                    items = resp.json()
                except Exception as exc:
                    result.note(f"search '{query}' p{page} failed: {exc}")
                    break
                if not items:
                    break
                for item in items:
                    found.setdefault(item.get("ref", ""), item)

        result.note(f"{len(found)} unique Kaggle datasets found")
        accepted: list[tuple[float, dict[str, Any]]] = []
        for ref, item in found.items():
            if not ref:
                continue
            score = dataset_relevance(ref, item.get("title", ""), item.get("subtitle", ""))
            licence = str(item.get("licenseName", "unknown")).lower()
            ok_license = licence in OPEN_LICENSES or "cc" in licence
            ctx.manifest.add_discovered(
                key=f"kaggle:{ref}", provider="kaggle", identifier=ref,
                title=item.get("title", "")[:200],
                url=f"https://www.kaggle.com/datasets/{ref}",
                license=licence, relevance=score,
                accepted=bool(score >= 2.0 and ok_license),
                meta={"votes": item.get("voteCount"), "size": item.get("totalBytes")})
            if score >= 2.0 and ok_license:
                accepted.append((score, item))

        accepted.sort(key=lambda x: -x[0])
        for _, item in accepted[: (ctx.limit or 10)]:
            self._download(ctx, result, item, auth)

    def _download(self, ctx: SourceContext, result: SourceResult,
                  item: dict[str, Any], auth: httpx.BasicAuth) -> None:
        """Download and unzip one dataset."""
        ref = item["ref"]
        size = int(item.get("totalBytes") or 0)
        if size and size > self.MAX_BYTES:
            result.note(f"{ref}: {size / 1e9:.1f} GB exceeds cap, skipped")
            return

        out = ctx.out_dir / ref.replace("/", "__")
        if out.exists() and any(out.iterdir()) and not ctx.force:
            ctx.manifest.mark_integrated(f"kaggle:{ref}")
            return
        out.mkdir(parents=True, exist_ok=True)
        zip_path = out.with_suffix(".zip")
        url = f"{API}/datasets/download/{ref}"

        try:
            with ctx.http._client.stream(                 # noqa: SLF001 - needs auth
                    "GET", url, auth=auth, follow_redirects=True, timeout=600.0) as resp:
                resp.raise_for_status()
                with zip_path.open("wb") as fh:
                    for chunk in resp.iter_bytes(1 << 20):
                        fh.write(chunk)
        except Exception as exc:
            result.note(f"{ref}: download failed ({exc})")
            ctx.manifest.mark_failed(url, zip_path, self.name, str(exc))
            zip_path.unlink(missing_ok=True)
            return

        extracted = 0
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for info in zf.infolist():
                    if info.is_dir() or info.file_size > self.MAX_BYTES:
                        continue
                    # Guard against path traversal in untrusted archives.
                    target = (out / info.filename).resolve()
                    if not str(target).startswith(str(out.resolve())):
                        continue
                    zf.extract(info, out)
                    extracted += 1
        except zipfile.BadZipFile:
            result.note(f"{ref}: archive was not a valid zip")
            zip_path.unlink(missing_ok=True)
            return
        zip_path.unlink(missing_ok=True)

        total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        ctx.manifest.record(DownloadRecord(
            url=f"https://www.kaggle.com/datasets/{ref}", path=str(out),
            source=self.name, kind="kaggle_dataset", bytes=total,
            license=str(item.get("licenseName", "unknown")),
            attribution=f"Kaggle dataset {ref}"))
        ctx.manifest.mark_integrated(f"kaggle:{ref}")
        result.files += extracted
        result.bytes += total
        result.records += 1
        result.note(f"{ref}: {extracted} files, {total / 1e6:.1f} MB")
