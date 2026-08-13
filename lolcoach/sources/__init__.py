"""Stage 1: data acquisition.

Importing this package registers every connector. :func:`download_all` is the
orchestrator behind ``lolcoach download_data``: it resolves the source order,
skips anything disabled or missing credentials, runs the rest, and records the
outcome in the manifest so a later run can resume rather than repeat.

Connectors never raise into the orchestrator -- a broken upstream produces a
failed :class:`~lolcoach.sources.base.SourceResult` and the run continues,
because losing Reddit should not cost you Data Dragon.
"""

from __future__ import annotations

import time
import uuid
from typing import Iterable, Sequence

from ..config import Config
from ..http import HttpClient
from ..logging_utils import get_logger
from ..storage import Database, Manifest
from .base import (LicenseInfo, Source, SourceContext, SourceResult, all_sources,
                   get_source, register, run_source, source_names)

# Importing for the registration side effect; ordering here is irrelevant
# because dependencies are resolved by ``all_sources``.
from . import ddragon           # noqa: F401  (also registers Community Dragon)
from . import oracles_elixir    # noqa: F401
from . import wiki              # noqa: F401
from . import patch_notes       # noqa: F401
from . import reddit            # noqa: F401
from . import riot_api          # noqa: F401
from . import hf_datasets       # noqa: F401
from . import github_datasets   # noqa: F401
from . import kaggle_datasets   # noqa: F401
from . import replays           # noqa: F401

log = get_logger("sources")

__all__ = [
    "Source", "SourceContext", "SourceResult", "LicenseInfo",
    "all_sources", "get_source", "register", "run_source", "source_names",
    "download_all",
]


def download_all(
    cfg: Config,
    only: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    force: bool = False,
    limit: int | None = None,
) -> list[SourceResult]:
    """Run every enabled connector and return their results.

    Parameters
    ----------
    only:
        Restrict the run to these source names.
    exclude:
        Skip these source names.
    force:
        Re-download even when the manifest says a copy already exists.
    limit:
        Per-source soft cap (matches, pages, posts...). Useful for smoke runs.
    """
    paths = cfg.paths
    db = Database(paths.db)
    manifest = Manifest(paths.manifest_db)
    run_id = uuid.uuid4().hex[:12]
    manifest.start_run(run_id, "download")

    only_set = {s.lower() for s in only} if only else None
    excl_set = {s.lower() for s in exclude} if exclude else set()

    results: list[SourceResult] = []
    started = time.time()

    with HttpClient(user_agent=cfg.user_agent, cache_dir=paths.cache,
                    timeout=cfg.http_timeout, retries=cfg.http_retries) as http:
        for src in all_sources():
            if only_set is not None and src.name not in only_set:
                continue
            if src.name in excl_set:
                continue
            if not src.enabled(cfg):
                results.append(SourceResult(name=src.name, skipped="disabled in config"))
                continue
            missing = src.missing_secret(cfg)
            if missing:
                log.info("[%s] skipped: %s not set", src.name, missing)
                results.append(SourceResult(name=src.name,
                                            skipped=f"missing credential {missing}"))
                continue

            log.info("[%s] %s", src.name, src.description)
            ctx = SourceContext(cfg=cfg, http=http, manifest=manifest, db=db,
                                out_dir=paths.source(src.name), force=force,
                                limit=limit)
            ctx.out_dir.mkdir(parents=True, exist_ok=True)
            result = run_source(src, ctx)
            if result.skipped:
                log.info("[%s] skipped: %s", src.name, result.skipped)
            elif result.ok:
                log.info("[%s] done in %.1fs: %d files, %.1f MB, %d records",
                         src.name, result.elapsed_s, result.files,
                         result.bytes / 1e6, result.records)
            else:
                log.error("[%s] failed: %s", src.name, result.error)
            results.append(result)

    db.optimize()
    ok = all(r.ok for r in results)
    detail = "; ".join(f"{r.name}={'ok' if r.ok else 'fail'}" for r in results)
    manifest.end_run(run_id, ok, detail)
    log.info("download stage finished in %.1fs (%.1f MB total on disk)",
             time.time() - started, manifest.total_bytes() / 1e6)
    return results
