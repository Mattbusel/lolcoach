"""The review-only download: no PyTorch, no model, everything else works."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lolcoach import __version__, launcher
from lolcoach.config import Config


def test_launcher_version_and_help(capsys):
    with pytest.raises(SystemExit) as exit_info:
        launcher.main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out

    with pytest.raises(SystemExit):
        launcher.main(["--help"])
    out = capsys.readouterr().out
    assert "--port" in out and "--no-browser" in out


def test_coach_reports_that_the_ai_is_not_bundled(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lolcoach import review

    monkeypatch.setattr(review, "_missing_coach_runtime", lambda: ["torch"])
    client = TestClient(review.create_app(Config(data_root=str(tmp_path / "data"))))

    deadline = time.time() + 10
    snapshot = client.get("/api/coach").json()
    while snapshot["state"] == "loading" and time.time() < deadline:
        time.sleep(0.05)
        snapshot = client.get("/api/coach").json()

    assert snapshot["state"] == "unavailable"
    assert snapshot["detail"] == review.COACH_NOT_BUNDLED
    assert client.get("/").status_code == 200
