"""Common scaffolding for Stage-1 data sources.

Every connector implements :class:`Source`. The orchestrator hands each one a
:class:`SourceContext` carrying the config, a rate-limited HTTP client, the
provenance ledger and the database, then records a :class:`SourceResult`.

Legal posture
-------------
The brief asks for "everything legally available", which requires being
explicit about what is *not*. Each connector declares a :class:`LicenseInfo`,
and connectors whose upstream forbids bulk collection are either omitted or
restricted to metadata:

* ``.rofl`` replay files are read **locally only**. Their payload is encrypted
  with per-game keys Riot does not publish, so the connector extracts the
  plaintext JSON metadata header and nothing else. It never downloads replays
  from third parties.
* Match VODs are not scraped. Bulk downloading YouTube/Twitch video violates
  those platforms' terms, so no VOD connector exists; the equivalent signal is
  obtained from professional match *data* via Oracle's Elixir instead.
* Reddit is read through the Arctic Shift public research archive, which
  exists to serve this use case, rather than by evading Reddit's API auth.

``requires_secret`` lets the orchestrator skip a source cleanly when its
credential is absent instead of failing the run.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import Config
from ..http import HttpClient
from ..logging_utils import get_logger
from ..storage import Database, DownloadRecord, Manifest

log = get_logger("sources")


@dataclass(frozen=True)
class LicenseInfo:
    """Licence and attribution terms attached to everything a source writes."""

    name: str
    url: str = ""
    attribution: str = ""
    #: False when the data may be used for research/personal training but not
    #: redistributed. Recorded so a dataset card can be generated correctly.
    redistributable: bool = True

    def as_dict(self) -> dict[str, str]:
        return {"license": self.name, "attribution": self.attribution or self.url}


# Licences used by more than one connector.
RIOT_TOS = LicenseInfo(
    name="Riot Games API Terms of Service",
    url="https://developer.riotgames.com/policies/general",
    attribution="League of Legends and Riot Games are trademarks of Riot Games, Inc. "
                "Data retrieved via the Riot Games API. Riot Games does not endorse this project.",
    redistributable=False,
)
CC_BY_SA = LicenseInfo(
    name="CC BY-SA 4.0",
    url="https://creativecommons.org/licenses/by-sa/4.0/",
    attribution="League of Legends Wiki (wiki.leagueoflegends.com), CC BY-SA 4.0",
)
ORACLES_ELIXIR = LicenseInfo(
    name="Oracle's Elixir public match data",
    url="https://oracleselixir.com/",
    attribution="Data provided by Oracle's Elixir (Tim Sevenhuysen). Free for "
                "non-commercial use with credit.",
    redistributable=False,
)
REDDIT_UGC = LicenseInfo(
    name="Reddit User Agreement (user-generated content)",
    url="https://www.redditinc.com/policies/user-agreement",
    attribution="Reddit comments and posts remain the property of their authors; "
                "collected via the Arctic Shift research archive.",
    redistributable=False,
)


@dataclass
class SourceContext:
    """Everything a connector needs to do its job."""

    cfg: Config
    http: HttpClient
    manifest: Manifest
    db: Database
    #: Set by the orchestrator; connectors write beneath it.
    out_dir: Path
    #: When True, re-fetch even if the manifest says we already have it.
    force: bool = False
    #: Soft cap so ``--limit`` on the CLI can shrink an exploratory run.
    limit: int | None = None

    def sub(self, *parts: str) -> Path:
        p = self.out_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass
class SourceResult:
    """What a connector achieved, for the run summary."""

    name: str
    ok: bool = True
    files: int = 0
    bytes: int = 0
    records: int = 0
    skipped: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def note(self, msg: str) -> None:
        self.notes.append(msg)
        log.info("  %s", msg)


class Source(ABC):
    """Base class for every Stage-1 connector."""

    #: Short identifier, also the raw sub-directory name.
    name: str = "source"
    #: Human description shown by ``lolcoach download_data --list``.
    description: str = ""
    #: Licence attached to everything this source writes.
    license: LicenseInfo = LicenseInfo(name="unknown")
    #: Credential names; the source is skipped when none is present.
    requires_secret: tuple[str, ...] = ()
    #: Sources that must run first (e.g. ingest needs Data Dragon champions).
    depends_on: tuple[str, ...] = ()

    def enabled(self, cfg: Config) -> bool:
        """Whether the config switches this source on."""
        return bool(getattr(cfg.sources, self.name, True))

    def missing_secret(self, cfg: Config) -> str | None:
        """Return the credential name to blame when the source cannot run."""
        if not self.requires_secret:
            return None
        return None if cfg.secret(*self.requires_secret) else self.requires_secret[0]

    @abstractmethod
    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        """Download this source's data, updating ``result`` as it goes."""

    # -- helpers shared by connectors -----------------------------------
    def save_bytes(self, ctx: SourceContext, url: str, dest: Path, data: bytes,
                   result: SourceResult, kind: str = "") -> None:
        """Write a payload and record it in the manifest."""
        import hashlib

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        ctx.manifest.record(DownloadRecord(
            url=url, path=str(dest), source=self.name, kind=kind,
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
            **self.license.as_dict(),
        ))
        result.files += 1
        result.bytes += len(data)

    def save_json(self, ctx: SourceContext, url: str, dest: Path, obj: Any,
                  result: SourceResult, kind: str = "") -> None:
        import json

        self.save_bytes(ctx, url, dest,
                        json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                        result, kind=kind)

    def should_skip(self, ctx: SourceContext, url: str, dest: Path,
                    max_age_s: float | None = None) -> bool:
        """True when the manifest shows a good, fresh copy already on disk."""
        if ctx.force or not dest.exists():
            return False
        if not ctx.manifest.have(url, dest):
            return False
        if max_age_s is None:
            return True
        age = ctx.manifest.age(url)
        return age is not None and age < max_age_s


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    """Class decorator adding a connector to the global registry."""
    _REGISTRY[cls.name] = cls
    return cls


def all_sources() -> list[Source]:
    """Instantiate every registered connector, dependency-ordered."""
    instances = {name: cls() for name, cls in _REGISTRY.items()}
    ordered: list[Source] = []
    seen: set[str] = set()

    def visit(n: str, stack: tuple[str, ...] = ()) -> None:
        if n in seen or n not in instances:
            return
        if n in stack:      # defensive: a cycle would otherwise recurse forever
            log.warning("dependency cycle involving %s; ignoring edge", n)
            return
        for dep in instances[n].depends_on:
            visit(dep, stack + (n,))
        seen.add(n)
        ordered.append(instances[n])

    for name in sorted(instances):
        visit(name)
    return ordered


def get_source(name: str) -> Source | None:
    cls = _REGISTRY.get(name)
    return cls() if cls else None


def source_names() -> list[str]:
    return sorted(_REGISTRY)


def run_source(src: Source, ctx: SourceContext) -> SourceResult:
    """Execute one connector, converting failures into a recorded result."""
    result = SourceResult(name=src.name)
    started = time.time()
    try:
        src.fetch(ctx, result)
    except Exception as exc:  # a broken upstream must not kill the run
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        log.exception("source %s failed", src.name)
    result.elapsed_s = time.time() - started
    return result


def iter_json_files(root: Path, pattern: str = "*.json") -> Iterable[Path]:
    """Recursively yield files matching ``pattern`` under ``root``."""
    if root.exists():
        yield from sorted(root.rglob(pattern))
