"""Self-diagnostics for a local LoLCoach installation.

The pipeline can have substantial local state: a corpus, a vector index, a
training dataset, an adapter, and (optionally) a CUDA runtime.  ``doctor``
turns those prerequisites into a small structured report instead of making a
player infer a failure from an import traceback.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .storage import Database


@dataclass(frozen=True)
class DoctorCheck:
    """One actionable installation check."""

    name: str
    state: str  # ok | warn | fail
    detail: str
    fix: str = ""


def run_doctor(cfg: Config) -> list[DoctorCheck]:
    """Inspect local prerequisites without downloading models or calling APIs."""
    checks = [_python_check(), _dependency_check(), _cuda_check()]
    checks.extend((_credential_check(cfg), _disk_check(cfg)))
    checks.extend((_database_check(cfg), _rag_check(cfg), _dataset_check(cfg),
                   _adapter_check(cfg)))
    return checks


def _python_check() -> DoctorCheck:
    version = sys.version_info
    if version >= (3, 10):
        return DoctorCheck("Python", "ok", f"Python {version.major}.{version.minor}.{version.micro}")
    return DoctorCheck(
        "Python", "fail", f"Python {version.major}.{version.minor} is unsupported",
        "Install Python 3.10 or newer, recreate the environment, then reinstall LoLCoach.")


def _dependency_check() -> DoctorCheck:
    required = ("sentence_transformers", "faiss", "fastapi", "uvicorn")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        return DoctorCheck("Local components", "ok", "RAG and review UI dependencies are installed")
    packages = {"sentence_transformers": "sentence-transformers", "faiss": "faiss-cpu",
                "fastapi": "fastapi", "uvicorn": "uvicorn"}
    names = ", ".join(packages[name] for name in missing)
    return DoctorCheck(
        "Local components", "fail", f"Missing: {names}",
        'Run `python -m pip install -e ".[dev]"` after updating the project dependencies.')


def _cuda_check() -> DoctorCheck:
    try:
        import torch
    except ImportError:
        return DoctorCheck("GPU training", "fail", "PyTorch is not installed",
                           "Install the project dependencies, including a CUDA PyTorch build on NVIDIA hardware.")
    if not torch.cuda.is_available():
        return DoctorCheck(
            "GPU training", "warn", "No CUDA GPU is visible; existing local adapter can still be used on CPU",
            "For QLoRA training, install a CUDA-compatible PyTorch and bitsandbytes build on an NVIDIA GPU machine.")
    props = torch.cuda.get_device_properties(0)
    memory_gib = props.total_memory / (1024 ** 3)
    if memory_gib < 10:
        return DoctorCheck(
            "GPU training", "warn", f"{props.name} ({memory_gib:.1f} GiB)",
            "Use a 1.5B/3B model or a GPU with at least 12 GiB VRAM for the configured 7B QLoRA run.")
    return DoctorCheck("GPU training", "ok", f"{props.name} ({memory_gib:.1f} GiB CUDA VRAM)")


def _credential_check(cfg: Config) -> DoctorCheck:
    if cfg.secret("RIOT_API_KEY", "RIOT_KEY", "RIOT_TOKEN"):
        return DoctorCheck("Riot timeline access", "ok", "RIOT_API_KEY is configured")
    return DoctorCheck(
        "Riot timeline access", "warn", "No RIOT_API_KEY; match timeline features cannot be collected",
        "Run `lolcoach init` with a Riot Developer Portal key, then `lolcoach download_data --only riot`.")


def _disk_check(cfg: Config) -> DoctorCheck:
    usage = shutil.disk_usage(cfg.paths.root)
    free_gib = usage.free / (1024 ** 3)
    if free_gib >= 30:
        return DoctorCheck("Disk space", "ok", f"{free_gib:.1f} GiB free at {cfg.paths.root}")
    return DoctorCheck(
        "Disk space", "warn", f"Only {free_gib:.1f} GiB free at {cfg.paths.root}",
        "Set LOLCOACH_HOME to a drive with at least 30 GiB free before a large timeline crawl.")


def _database_check(cfg: Config) -> DoctorCheck:
    if not cfg.paths.db.exists():
        return DoctorCheck("Local database", "warn", "No database has been created yet",
                           "Run `lolcoach download_data --limit 1` to initialize local data.")
    db = Database(cfg.paths.db)
    try:
        counts = db.counts()
        frames = counts.get("frames", 0)
        waves = counts.get("wave_states", 0)
        if frames and waves:
            return DoctorCheck("Timeline features", "ok", f"{frames:,} frames and {waves:,} inferred wave states")
        if frames:
            return DoctorCheck(
                "Timeline features", "warn", f"{frames:,} frames are present but no inferred wave states",
                "Run `lolcoach prepare_dataset` to derive local feature tables.")
        return DoctorCheck(
            "Timeline features", "warn", "No Riot timeline frames are stored",
            "Run `lolcoach init`, then `lolcoach download_data --only riot` and `lolcoach prepare_dataset`.")
    finally:
        db.close()


def _rag_check(cfg: Config) -> DoctorCheck:
    index = cfg.paths.index / "league.faiss"
    metadata = cfg.paths.index / "league_index.json"
    if not index.is_file() or not metadata.is_file():
        return DoctorCheck("RAG index", "warn", "No completed local FAISS index",
                           "Run `lolcoach rag --build` after downloading source data.")
    age_hours = (time.time() - index.stat().st_mtime) / 3600.0
    detail = f"FAISS index is present ({index.stat().st_size / (1024 ** 2):.1f} MiB, {age_hours:.1f} hours old)"
    if age_hours > 24 * 30:
        return DoctorCheck("RAG index", "warn", detail,
                           "Run `lolcoach update_patch` to refresh patch-sensitive retrieval evidence.")
    return DoctorCheck("RAG index", "ok", detail)


def _dataset_check(cfg: Config) -> DoctorCheck:
    train = cfg.paths.datasets / "train.jsonl"
    if not train.is_file() or not train.stat().st_size:
        return DoctorCheck("Training dataset", "warn", "No training JSONL is available",
                           "Run `lolcoach prepare_dataset` after source data is ready.")
    lines = _nonempty_lines(train)
    return DoctorCheck("Training dataset", "ok", f"{lines:,} training examples at {train}")


def _adapter_check(cfg: Config) -> DoctorCheck:
    adapter = cfg.paths.models / cfg.train.output_name / "adapter"
    if (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file():
        size_mib = (adapter / "adapter_model.safetensors").stat().st_size / (1024 ** 2)
        return DoctorCheck("Local coach adapter", "ok", f"Adapter is ready ({size_mib:.1f} MiB)")
    return DoctorCheck("Local coach adapter", "warn", "No trained local adapter",
                       "Run `lolcoach train` on a CUDA-capable NVIDIA GPU after preparing the dataset.")


def _nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
