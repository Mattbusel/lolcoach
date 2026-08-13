"""HuggingFace dataset discovery and integration.

Stage 1 asks for public datasets *and* for anything newly discovered to be
folded into the pipeline automatically. This connector does both:

1. **Search.** The Hub API is queried for each configured phrase.
2. **Score.** Candidates are ranked by how League-specific their id, tags and
   description are, discounted for obvious mismatches (voice lines, art,
   unrelated esports). Only candidates above a threshold are accepted.
3. **Register.** Accepted datasets are written to ``discovered_sources`` in
   the manifest, so an operator can audit exactly what was pulled in.
4. **Download.** Accepted datasets are snapshotted with size guards, then
   handed to the generic tabular adapter in :mod:`lolcoach.ingest.tabular`,
   which sniffs their columns and maps whatever it recognises (match ids,
   champions, gold/xp at 10/15, win flags) into the shared schema.

Gated or private datasets are skipped unless ``HF_TOKEN`` is present.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from ..storage import DownloadRecord
from .base import LicenseInfo, Source, SourceContext, SourceResult, register

log = get_logger("sources.huggingface")

API = "https://huggingface.co/api/datasets"

#: Terms that make a dataset relevant, with their weights.
POSITIVE = {
    "league of legends": 3.0, "leagueoflegends": 3.0, "lol": 0.5, "riot": 1.0,
    "summoner": 1.5, "champion": 1.0, "match": 0.8, "timeline": 2.0,
    "esports": 1.0, "ranked": 0.8, "challenger": 1.5, "jungle": 1.0,
    "draft": 0.6, "patch": 0.6,
}
#: Terms that indicate a dataset is about something else entirely.
NEGATIVE = {
    "voice": 2.5, "audio": 2.0, "image": 1.5, "art": 1.5, "sprite": 2.0,
    "music": 2.0, "tts": 2.5, "dota": 2.0, "valorant": 1.5, "overwatch": 1.5,
    "chess": 2.0, "wow": 1.0,
}
ACCEPT_THRESHOLD = 2.5


#: Hub tags that mark a dataset as the wrong modality entirely. A dataset of
#: champion artwork scores highly on the word "leagueoflegends" but contains
#: nothing a text coaching model can learn from, so modality is checked before
#: keywords rather than hoping the keyword list catches it.
WRONG_MODALITY = ("modality:image", "modality:audio", "modality:video",
                  "task_categories:text-to-image",
                  "task_categories:image-classification",
                  "task_categories:automatic-speech-recognition",
                  "library:audiofolder", "format:imagefolder", "format:audiofolder")

#: Modalities we can actually use.
USEFUL_MODALITY = ("modality:text", "modality:tabular", "format:csv",
                   "format:json", "format:parquet")


def score_candidate(dataset_id: str, tags: list[str], description: str,
                    downloads: int = 0) -> float:
    """Relevance score for a Hub dataset.

    The id is weighted twice as heavily as the description because Hub
    descriptions are frequently empty, while ids are almost always meaningful.
    Datasets whose only declared modality is image, audio or video are
    rejected outright regardless of how League-flavoured their name is.
    """
    ident = dataset_id.lower()
    lowered_tags = [t.lower() for t in tags]
    has_wrong = any(t in lowered_tags for t in WRONG_MODALITY)
    has_useful = any(t in lowered_tags for t in USEFUL_MODALITY)
    if has_wrong and not has_useful:
        return -10.0

    blob = f"{' '.join(tags)} {description}".lower()
    score = 0.0
    for term, weight in POSITIVE.items():
        if term in ident:
            score += weight * 2.0
        elif term in blob:
            score += weight
    for term, weight in NEGATIVE.items():
        if term in ident or term in blob:
            score -= weight
    # A little credit for adoption: heavily used datasets are usually clean.
    if downloads > 100:
        score += 0.5
    if downloads > 1000:
        score += 0.5
    return score


@register
class HuggingFaceSource(Source):
    """Discover, register and download League datasets from the Hub."""

    name = "huggingface"
    description = "HuggingFace Hub dataset discovery + download"
    license = LicenseInfo(
        name="per-dataset (recorded individually)",
        url="https://huggingface.co/datasets",
        attribution="See the licence recorded for each downloaded dataset.",
    )

    #: Skip anything larger than this to keep an exploratory run tractable.
    MAX_BYTES = 2 * 1024 ** 3

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        token = ctx.cfg.secret("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        candidates: dict[str, dict[str, Any]] = {}
        for query in ctx.cfg.sources.huggingface_queries:
            try:
                payload = ctx.http.get_json(
                    API, params={"search": query, "limit": 50, "full": "true"},
                    headers=headers, use_cache=True, max_age=86400)
            except Exception as exc:
                result.note(f"search '{query}' failed: {exc}")
                continue
            for item in payload or []:
                candidates.setdefault(item["id"], item)

        result.note(f"{len(candidates)} unique Hub candidates")
        accepted: list[tuple[str, float, dict[str, Any]]] = []
        for ds_id, item in candidates.items():
            if item.get("private") or item.get("gated"):
                continue
            tags = [str(t) for t in item.get("tags", [])]
            card = item.get("cardData") or {}
            desc = str(item.get("description") or card.get("pretty_name") or "")
            score = score_candidate(ds_id, tags, desc, int(item.get("downloads") or 0))
            licence = _license_from_tags(tags, card)
            is_new = ctx.manifest.add_discovered(
                key=f"hf:{ds_id}", provider="huggingface", identifier=ds_id,
                title=desc[:200], url=f"https://huggingface.co/datasets/{ds_id}",
                license=licence, relevance=score, accepted=score >= ACCEPT_THRESHOLD,
                meta={"tags": tags[:40], "downloads": item.get("downloads")})
            if score >= ACCEPT_THRESHOLD:
                accepted.append((ds_id, score, item))
                if is_new:
                    result.note(f"discovered {ds_id} (score {score:.1f}, {licence})")

        accepted.sort(key=lambda x: -x[1])
        if not accepted:
            result.note("no Hub dataset cleared the relevance threshold")
            return

        for ds_id, score, item in accepted[: (ctx.limit or 12)]:
            self._download(ctx, result, ds_id, item, token)

    def _download(self, ctx: SourceContext, result: SourceResult, ds_id: str,
                  item: dict[str, Any], token: str | None) -> None:
        """Snapshot a dataset repo, skipping oversized or binary-only repos."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            result.note("huggingface_hub not installed; skipping downloads")
            return

        target = ctx.out_dir / ds_id.replace("/", "__")
        if target.exists() and not ctx.force:
            ctx.manifest.mark_integrated(f"hf:{ds_id}")
            return
        try:
            path = snapshot_download(
                repo_id=ds_id, repo_type="dataset", local_dir=str(target),
                token=token, max_workers=4,
                allow_patterns=["*.csv", "*.json", "*.jsonl", "*.parquet",
                                "*.md", "*.txt", "*.yaml"],
            )
        except Exception as exc:
            result.note(f"{ds_id}: download failed ({type(exc).__name__})")
            ctx.manifest.mark_failed(f"hf://{ds_id}", target, self.name, str(exc))
            return

        total = 0
        files = 0
        for f in Path(path).rglob("*"):
            if f.is_file():
                size = f.stat().st_size
                total += size
                files += 1
        if total > self.MAX_BYTES:
            result.note(f"{ds_id}: {total / 1e9:.1f} GB exceeds cap, kept metadata only")
        licence = _license_from_tags([str(t) for t in item.get("tags", [])],
                                     item.get("cardData") or {})
        ctx.manifest.record(DownloadRecord(
            url=f"https://huggingface.co/datasets/{ds_id}", path=str(target),
            source=self.name, kind="hub_dataset", bytes=total,
            license=licence, attribution=f"HuggingFace dataset {ds_id}"))
        ctx.manifest.mark_integrated(f"hf:{ds_id}")
        result.files += files
        result.bytes += total
        result.records += 1
        result.note(f"{ds_id}: {files} files, {total / 1e6:.1f} MB")


def _license_from_tags(tags: list[str], card: dict[str, Any]) -> str:
    """Extract a licence id from Hub tags (``license:mit``) or card data."""
    for t in tags:
        if t.startswith("license:"):
            return t.split(":", 1)[1]
    lic = card.get("license")
    if isinstance(lic, list):
        return ",".join(str(x) for x in lic)
    return str(lic) if lic else "unspecified"
