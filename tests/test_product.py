"""Tests for the local product layer: onboarding, review data, and telemetry."""

from __future__ import annotations

from pathlib import Path

import httpx
from fixtures import build_synthetic_match
from typer.testing import CliRunner

from lolcoach.cli import app
from lolcoach.config import Config
from lolcoach.doctor import run_doctor
from lolcoach.features.derive import derive_match
from lolcoach.live import format_live_snapshot, parse_live_payload
from lolcoach.onboarding import parse_riot_id, save_player_puuid, save_riot_key
from lolcoach.review import ReviewStore, format_timestamp
from lolcoach.sources.riot_api import RiotClient
from lolcoach.storage import Database
from lolcoach.ingest.core import _ingest_riot_match
from lolcoach import launcher


def test_cli_exposes_product_and_feature_refresh_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("download_data", "prepare_dataset", "derive_features", "train", "benchmark", "chat", "rag", "update_patch"):
        assert command in result.output


def test_frozen_launcher_uses_checkout_data_without_copying_it(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    exe = project / "dist" / "LoLCoach" / "LoLCoach.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    checkout_data = project / "data"
    checkout_data.mkdir()
    (checkout_data / "lolcoach.sqlite").touch()
    monkeypatch.delenv("LOLCOACH_HOME", raising=False)
    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
    monkeypatch.setattr(launcher.sys, "executable", str(exe))
    assert launcher._configure_data_root() == checkout_data.resolve()


def test_onboarding_saves_key_and_optional_puuid_without_logging(tmp_path: Path):
    env_path = tmp_path / ".lolcoach.env"
    env_path.write_text("# personal credentials\nOTHER_KEY=keep\nRIOT_API_KEY=old\n", encoding="utf-8")
    validated: list[str] = []
    save_riot_key("new-secret", env_path=env_path, validate=validated.append)
    save_player_puuid("player-puuid", env_path=env_path)
    text = env_path.read_text(encoding="utf-8")
    assert validated == ["new-secret"]
    assert "OTHER_KEY=keep" in text
    assert text.count("RIOT_API_KEY=") == 1
    assert "RIOT_API_KEY=new-secret" in text
    assert "LOLCOACH_PUUID=player-puuid" in text


def test_riot_id_requires_name_and_tagline():
    assert parse_riot_id("Summoner Name#NA1") == ("Summoner Name", "NA1")
    try:
        parse_riot_id("missing-tag")
    except ValueError as exc:
        assert "GameName#TAG" in str(exc)
    else:
        raise AssertionError("A Riot ID without a tag should fail")


def test_account_lookup_uses_regional_account_route(tmp_path: Path):
    cfg = Config(data_root=str(tmp_path / "data"))
    client = RiotClient("key", cfg)
    seen: dict[str, str] = {}
    try:
        def fake_get(host: str, path: str, **params):
            seen.update(host=host, path=path)
            return {"puuid": "resolved"}
        client._get = fake_get  # type: ignore[method-assign]
        assert client.account_by_riot_id("americas", "Name With Space", "NA 1") == {"puuid": "resolved"}
    finally:
        client.close()
    assert seen == {
        "host": "americas",
        "path": "/riot/account/v1/accounts/by-riot-id/Name%20With%20Space/NA%201",
    }


def test_riot_client_keeps_the_key_out_of_urls(tmp_path: Path):
    """The key travels in the header, not the query string.

    A key in a URL is recorded everywhere URLs are: httpx logs the full request
    line at INFO, and proxies and crash reports keep query strings. This asserts
    the default path never puts the credential somewhere it can be logged.
    """
    cfg = Config(data_root=str(tmp_path / "data"))
    client = RiotClient("secret", cfg)
    seen: dict[str, object] = {}

    class FakeHttp:
        def get(self, url: str, **kwargs):
            seen["url"] = url
            seen["params"] = kwargs["params"]
            return httpx.Response(200, content=b"{}", request=httpx.Request("GET", url))

        def close(self) -> None:
            return None

    header_key = client.http._client.headers.get("X-Riot-Token")
    client.http = FakeHttp()  # type: ignore[assignment]
    try:
        assert client._get("na1", "/example", page=3) == {}
    finally:
        client.close()
    assert header_key == "secret"
    assert seen["url"] == "https://na1.api.riotgames.com/example"
    assert seen["params"] == {"page": 3}
    assert "api_key" not in seen["params"]


def test_riot_client_falls_back_to_query_auth_after_a_401(tmp_path: Path):
    """Networks that strip custom headers still work, at one wasted request.

    Some corporate and consumer filters drop ``X-Riot-Token``. Rather than pay
    the exposure on every call, the query form is used only after the header
    form has actually been rejected once.
    """
    cfg = Config(data_root=str(tmp_path / "data"))
    client = RiotClient("secret", cfg)
    calls: list[dict[str, object]] = []

    class StrippingHttp:
        def get(self, url: str, **kwargs):
            params = kwargs["params"]
            calls.append(dict(params))
            status = 200 if "api_key" in params else 401
            return httpx.Response(status, content=b"{}",
                                  request=httpx.Request("GET", url))

        def close(self) -> None:
            return None

    client.http = StrippingHttp()  # type: ignore[assignment]
    try:
        assert client._get("na1", "/example") == {}
    finally:
        client.close()
    assert len(calls) == 2, "should retry exactly once"
    assert "api_key" not in calls[0]
    assert calls[1]["api_key"] == "secret"
    assert client._use_query_key is True


def test_live_payload_is_reduced_to_personal_visible_state():
    snapshot = parse_live_payload({
        "gameData": {"gameTime": 605.5},
        "activePlayer": {
            "summonerName": "Player", "championName": "Ahri", "level": 9,
            "currentGold": 950, "championStats": {"currentHealth": 720, "maxHealth": 1200,
                                                   "resourceValue": 400, "resourceMax": 700},
        },
        "allPlayers": [{
            "summonerName": "Player", "championName": "Ahri", "level": 9,
            "scores": {"kills": 4, "deaths": 1, "assists": 6, "creepScore": 83},
            "items": [{"displayName": "Luden's Companion"}],
        }],
        "events": {"Events": [{"EventName": "GameStart"}]},
    })
    assert snapshot.champion == "Ahri"
    assert snapshot.health_pct == 0.6
    assert snapshot.items == ("Luden's Companion",)
    assert "10:05" in format_live_snapshot(snapshot)
    assert "Ahri" in format_live_snapshot(snapshot)


def test_review_store_projects_fixture_match_with_confidence(tmp_path: Path):
    cfg = Config(data_root=str(tmp_path / "data"))
    db = Database(cfg.paths.db)
    match, timeline = build_synthetic_match("REVIEW_TEST", minutes=25, seed=9)
    _ingest_riot_match(db, "REVIEW_TEST", match, timeline, "memory")
    derive_match(db, "REVIEW_TEST")
    db.close()

    store = ReviewStore(cfg)
    matches = store.list_matches()
    assert matches and matches[0]["match_id"] == "REVIEW_TEST"
    detail = store.match_detail("REVIEW_TEST")
    assert detail["frames"]
    assert detail["wave_states"]
    assert any(moment["kind"] == "wave" for moment in detail["moments"])
    context = store.moment_context("REVIEW_TEST", int(detail["frames"][0]["ts_ms"]))
    assert "REVIEW_TEST" in context
    assert format_timestamp(605000) == "10:05"


def test_doctor_returns_named_actionable_checks(tmp_path: Path):
    checks = run_doctor(Config(data_root=str(tmp_path / "data")))
    by_name = {check.name: check for check in checks}
    assert {"Python", "GPU training", "Riot timeline access", "RAG index", "Training dataset"} <= set(by_name)
    assert by_name["Riot timeline access"].state in {"ok", "warn"}
    if by_name["Riot timeline access"].state == "warn":
        assert by_name["Riot timeline access"].fix
