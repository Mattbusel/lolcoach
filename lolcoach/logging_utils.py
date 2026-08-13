"""Logging setup shared by the CLI and every stage.

Two sinks are configured: a Rich-formatted console handler for humans, and a
rotating plain-text file under ``<data root>/logs`` so that a long download or
training run can be inspected after the terminal has gone.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

#: Shared console. Stages use it for progress bars so output interleaves
#: correctly with log lines.
console = Console(stderr=False, highlight=False, soft_wrap=False)

_CONFIGURED = False


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO,
                  quiet_libs: bool = True) -> logging.Logger:
    """Configure root logging once and return the package logger.

    Parameters
    ----------
    log_dir:
        Directory for ``lolcoach.log``. Rotated at 32 MB, five backups kept.
    level:
        Console level. The file handler always records ``DEBUG``.
    quiet_libs:
        Suppress the very chatty third-party loggers (httpx request lines,
        urllib3 retries, tokenizer warnings) that otherwise drown progress.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return logging.getLogger("lolcoach")

    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    rich_handler = RichHandler(
        console=console, rich_tracebacks=True, show_path=False,
        omit_repeated_times=False, markup=False,
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    root.addHandler(rich_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "lolcoach.log", maxBytes=32 * 1024 * 1024, backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"))
        root.addHandler(fh)

    if quiet_libs:
        for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec",
                      "huggingface_hub", "datasets", "sentence_transformers",
                      "transformers.tokenization_utils_base"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logging.getLogger("lolcoach")


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ``lolcoach`` namespace."""
    return logging.getLogger(f"lolcoach.{name}")
