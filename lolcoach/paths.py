"""Canonical on-disk layout for every artefact the pipeline produces.

All paths hang off a single *data root*, resolved in this order:

1. the ``LOLCOACH_HOME`` environment variable,
2. the ``data_root`` key of the active config file,
3. ``<repo>/data``.

Keeping the layout in one module means a stage never has to guess where a
previous stage put its output, and lets the whole corpus be relocated (for
example onto a larger drive) by setting one environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def default_data_root() -> Path:
    """Return the data root implied by the environment, without creating it."""
    env = os.environ.get("LOLCOACH_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "data"


@dataclass(frozen=True)
class Paths:
    """Resolved directory tree.

    Instances are cheap and immutable; call :meth:`ensure` once at start-up to
    materialise the directories.
    """

    root: Path

    # ---- top level ----------------------------------------------------
    @property
    def raw(self) -> Path:
        """Byte-for-byte copies of what each upstream source served."""
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        """Normalised but not yet training-ready intermediates."""
        return self.root / "interim"

    @property
    def processed(self) -> Path:
        """Feature tables and derived game-state annotations."""
        return self.root / "processed"

    @property
    def datasets(self) -> Path:
        """JSONL instruction datasets ready for training."""
        return self.root / "datasets"

    @property
    def index(self) -> Path:
        """FAISS indexes and their metadata sidecars."""
        return self.root / "index"

    @property
    def models(self) -> Path:
        """Adapters, merged weights and training checkpoints."""
        return self.root / "models"

    @property
    def reports(self) -> Path:
        """Benchmark results and dataset statistics."""
        return self.root / "reports"

    @property
    def cache(self) -> Path:
        """HTTP response cache, safe to delete at any time."""
        return self.root / "cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # ---- per-source raw directories -----------------------------------
    def source(self, name: str) -> Path:
        """Raw directory for a named source (``ddragon``, ``riot`` ...)."""
        return self.raw / name

    # ---- well-known files ---------------------------------------------
    @property
    def db(self) -> Path:
        """Main relational store: matches, timelines, features, corpus."""
        return self.root / "lolcoach.sqlite"

    @property
    def manifest_db(self) -> Path:
        """Download provenance: url, sha256, licence, fetch time."""
        return self.root / "manifest.sqlite"

    def ensure(self) -> "Paths":
        """Create every directory in the layout. Idempotent."""
        for p in (
            self.root, self.raw, self.interim, self.processed, self.datasets,
            self.index, self.models, self.reports, self.cache, self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: str | os.PathLike[str] | None = None) -> Paths:
    """Build a :class:`Paths` for ``root`` (defaults to the environment)."""
    return Paths(Path(root).expanduser().resolve() if root else default_data_root())
