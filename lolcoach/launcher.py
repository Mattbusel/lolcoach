"""Double-click entry point for the Windows LoLCoach desktop bundle.

The PyInstaller build intentionally packages application code only.  Models,
Riot-derived data, and collected source material stay in each player's local
data directory because they are large, update frequently, and have separate
licensing/provenance obligations.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


class _WindowedStream:
    """File-like sink for libraries that call ``isatty`` in --windowed mode."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


def _configure_windowed_streams() -> None:
    if sys.stdout is None:
        sys.stdout = _WindowedStream()  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _WindowedStream()  # type: ignore[assignment]


def _configure_data_root() -> Path:
    """Select an existing portable data folder or a per-user AppData location."""
    existing = os.environ.get("LOLCOACH_HOME")
    if existing:
        return Path(existing).expanduser().resolve()

    if getattr(sys, "frozen", False):
        # A bundle built inside this checkout should open the data the owner
        # already collected.  A copied/shared ``dist\LoLCoach`` folder has no
        # such ancestor database and falls through to an isolated per-user
        # location, so it never exposes the builder's matches or credentials.
        checkout_data = Path(sys.executable).resolve().parent.parent.parent / "data"
        portable = Path(sys.executable).resolve().parent / "data"
        if (checkout_data / "lolcoach.sqlite").is_file():
            root = checkout_data
        elif portable.exists():
            root = portable
        else:
            local_app_data = os.environ.get("LOCALAPPDATA")
            base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
            root = base / "LoLCoach" / "data"
        os.environ["LOLCOACH_HOME"] = str(root)
        return root

    # Source checkouts retain their established repository-local data path.
    from lolcoach.paths import default_data_root
    return default_data_root()


def _configure_model_cache() -> None:
    """Point a shared frozen bundle at its bundled Hugging Face cache."""
    if not getattr(sys, "frozen", False):
        return
    bundled = Path(sys.executable).resolve().parent / "models" / "huggingface"
    if (bundled / "hub").is_dir():
        os.environ.setdefault("HF_HOME", str(bundled))
        os.environ.setdefault("HF_HUB_CACHE", str(bundled / "hub"))


def _parse_args(argv: list[str] | None):
    import argparse

    from lolcoach import __version__

    parser = argparse.ArgumentParser(
        prog="LoLCoach",
        description="Open the LoLCoach match review app in your browser. It serves only on "
                    "this machine (127.0.0.1) and never reads the League client.",
        epilog="Data lives in %LOCALAPPDATA%\\LoLCoach\\data unless a data folder sits "
               "beside the executable or LOLCOACH_HOME is set.",
    )
    parser.add_argument("--port", type=int, default=8765, help="local port (default 8765)")
    parser.add_argument("--no-browser", action="store_true",
                        help="start the server without opening a browser tab")
    parser.add_argument("--version", action="version", version=f"LoLCoach {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Open the local match-review app, reporting startup errors without a console."""
    _configure_windowed_streams()
    args = _parse_args(argv)
    _configure_data_root()
    _configure_model_cache()
    # PyInstaller executes this file as the top-level entry script, so these
    # imports must be absolute rather than package-relative.
    from lolcoach.config import load_config
    from lolcoach.review import serve

    cfg = load_config(None)
    url = f"http://127.0.0.1:{args.port}"
    print(f"LoLCoach is starting at {url}")
    print(f"Your data folder: {cfg.paths.root}")
    print("Keep this window open while you use LoLCoach. Close it to quit.", flush=True)
    try:
        # The packaged build opens the browser rather than a native window.
        # pywebview drives WebView2 through pythonnet, and inside a frozen
        # bundle the .NET loader probes the executable directory instead of
        # _internal\webview\lib, so assembly resolution is unreliable. The
        # browser path is deterministic; `lolcoach ui --native` still offers
        # the window when running from source.
        serve(cfg, host="127.0.0.1", port=args.port, open_browser=not args.no_browser,
              native=False)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 10048:
            _show_error(f"LoLCoach is already running. Open {url} in your browser.")
            return
        _write_startup_error(traceback.format_exc())
        _show_error(f"LoLCoach could not start: {exc}")
    except Exception as exc:
        _write_startup_error(traceback.format_exc())
        _show_error(f"LoLCoach could not start.\n\n{exc}\n\nOpen a terminal and run `lolcoach doctor` for details.")


def _show_error(message: str) -> None:
    """Use a native dialog in windowed builds, with a terminal fallback."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, "LoLCoach", 0x10)
    except Exception:
        print(message, file=sys.stderr)


def _write_startup_error(trace: str) -> None:
    """Persist a traceback for a windowed bundle whose console is hidden."""
    try:
        root = Path(os.environ.get("LOLCOACH_HOME", Path.home() / "AppData" / "Local" / "LoLCoach" / "data"))
        path = root / "logs" / "desktop-startup-error.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(trace, encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    main()
