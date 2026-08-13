"""Safe local onboarding for Riot timeline collection.

The Riot key is deliberately kept outside the repository.  This module only
writes ``~/.lolcoach.env``, which :mod:`lolcoach.config` already loads, and it
validates a supplied key before persisting it.  No key is logged or sent
anywhere other than Riot's own API endpoint during validation.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from .config import Config
from .sources.riot_api import RiotClient

RIOT_DEVELOPER_PORTAL = "https://developer.riotgames.com/"
DEFAULT_ENV_PATH = Path.home() / ".lolcoach.env"


def validate_riot_key(cfg: Config, key: str) -> None:
    """Prove that a key works with one cheap Riot API request.

    ``RiotClient`` turns the common expired-development-key 403 into a clear
    ``PermissionError``.  Other failures are intentionally distinct: a key
    should not be saved when a network or service failure makes it impossible
    to know whether it is valid.
    """
    candidate = key.strip()
    if not candidate:
        raise ValueError("A Riot API key is required")
    client = RiotClient(candidate, cfg)
    try:
        if not client.validate():
            raise RuntimeError(
                "Riot could not validate this key. Check your connection and try again; "
                "the key was not saved.")
    finally:
        client.close()


def save_riot_key(key: str, *, env_path: Path = DEFAULT_ENV_PATH,
                  validate: Callable[[str], None] | None = None) -> Path:
    """Validate then atomically store a Riot key without exposing its value.

    ``validate`` is injectable so the persistence behaviour is testable without
    a network call.  Existing unrelated environment values and comments stay
    intact; only the ``RIOT_API_KEY`` record is replaced.
    """
    candidate = key.strip()
    if not candidate:
        raise ValueError("A Riot API key is required")
    if validate is not None:
        validate(candidate)
    env_path = Path(env_path).expanduser()
    lines = _replace_env_line(_read_lines(env_path), "RIOT_API_KEY", candidate)
    _atomic_private_write(env_path, "\n".join(lines) + "\n")
    return env_path


def save_player_puuid(puuid: str, *, env_path: Path = DEFAULT_ENV_PATH) -> Path:
    """Persist an opted-in player's PUUID for local review conveniences."""
    value = puuid.strip()
    if not value:
        raise ValueError("A player PUUID is required")
    env_path = Path(env_path).expanduser()
    lines = _replace_env_line(_read_lines(env_path), "LOLCOACH_PUUID", value)
    _atomic_private_write(env_path, "\n".join(lines) + "\n")
    return env_path


def parse_riot_id(value: str) -> tuple[str, str]:
    """Split a Riot ID written as ``Game Name#TAG`` for account-v1 lookup."""
    game_name, separator, tag_line = value.strip().partition("#")
    if not separator or not game_name.strip() or not tag_line.strip():
        raise ValueError("Riot ID must be written as GameName#TAG")
    return game_name.strip(), tag_line.strip()


#: Account-v1 is served from these routes. A Riot ID normally resolves on any
#: of them, but not always, so each is tried before giving up.
ACCOUNT_ROUTES = ("americas", "europe", "asia")


def resolve_riot_id(cfg: Config, key: str, riot_id: str) -> str:
    """Resolve an opted-in Riot ID to a PUUID through Riot account-v1.

    Previously this asked ``americas`` only, which silently failed for every
    player outside that route. Each account route is tried in turn.
    """
    game_name, tag_line = parse_riot_id(riot_id)
    client = RiotClient(key.strip(), cfg, interactive=True)
    transient: Exception | None = None
    saw_not_found = False
    try:
        for route in ACCOUNT_ROUTES:
            try:
                account = client.account_by_riot_id(route, game_name, tag_line)
            except PermissionError:
                raise
            except Exception as exc:
                transient = exc
                continue
            # ``_get`` returns None for a 404: the route answered and does not
            # know this Riot ID. That is a real answer, not a failure.
            if account is None:
                saw_not_found = True
                continue
            puuid = str((account or {}).get("puuid") or "").strip()
            if puuid:
                return puuid
    finally:
        client.close()

    # Distinguish "Riot says no such account" from "Riot did not answer".
    # Telling someone to check their spelling when Riot is down sends them
    # round in circles retyping a correct name.
    if saw_not_found and transient is None:
        raise RuntimeError(
            f"Riot has no account called {game_name}#{tag_line}. Check the name "
            f"and tag exactly as they appear in the client (the tag is the part "
            f"after the #, and it is not always your region).")
    raise RuntimeError(
        "Riot's account service did not respond just now, so the Riot ID could "
        "not be checked. Nothing was saved; try again in a moment.")


def resolve_player(cfg: Config, key: str, riot_id: str) -> tuple[str, str]:
    """Resolve a Riot ID to ``(puuid, match_region)`` and persist both.

    The match region is detected by asking each regional route for one match
    id; only the route that actually holds the player's history answers. It is
    saved so later syncs skip the probe.
    """
    from .sources.riot_api import detect_region

    puuid = resolve_riot_id(cfg, key, riot_id)

    # The account is confirmed at this point, so persist it before probing for
    # the region. Region detection touches four more routes and can fail on a
    # transient Riot error; throwing away a verified PUUID because of that
    # would make the user retype a Riot ID that was already correct.
    save_player_puuid(puuid)
    save_player_riot_id(riot_id.strip())
    cfg.creds["LOLCOACH_PUUID"] = puuid
    cfg.creds["LOLCOACH_RIOT_ID"] = riot_id.strip()

    client = RiotClient(key.strip(), cfg, interactive=True)
    try:
        region = detect_region(client, puuid)
    finally:
        client.close()

    if not region:
        raise RuntimeError(
            "Your account was found and saved, but Riot did not return any "
            "recent match history for it on any region. If you have played "
            "recently this is usually a temporary Riot issue: press Sync again "
            "in a moment and it will retry.")

    save_player_region(region)
    cfg.creds["LOLCOACH_REGION"] = region
    return puuid, region


def save_player_riot_id(riot_id: str, *, env_path: Path = DEFAULT_ENV_PATH) -> Path:
    """Remember the Riot ID so the account can be re-resolved without retyping.

    Development keys expire daily, and some Riot identifiers are encrypted per
    key. Keeping the human-readable Riot ID means recovery is automatic rather
    than another form to fill in.
    """
    value = riot_id.strip()
    if not value:
        raise ValueError("A Riot ID is required")
    env_path = Path(env_path).expanduser()
    lines = _replace_env_line(_read_lines(env_path), "LOLCOACH_RIOT_ID", value)
    _atomic_private_write(env_path, "\n".join(lines) + "\n")
    return env_path


def save_player_region(region: str, *, env_path: Path = DEFAULT_ENV_PATH) -> Path:
    """Persist the regional route that serves this player's match history."""
    value = region.strip()
    if not value:
        raise ValueError("A region is required")
    env_path = Path(env_path).expanduser()
    lines = _replace_env_line(_read_lines(env_path), "LOLCOACH_REGION", value)
    _atomic_private_write(env_path, "\n".join(lines) + "\n")
    return env_path


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


def _replace_env_line(lines: list[str], key: str, value: str) -> list[str]:
    """Replace one shell-style environment entry while preserving comments."""
    prefix = f"{key}="
    replaced = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        target = stripped[7:] if stripped.startswith("export ") else stripped
        if target.startswith(prefix):
            if not replaced:
                out.append(f"{key}={value}")
                replaced = True
            continue
        out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    return out


def _atomic_private_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=path.parent, prefix=f".{path.name}.", suffix=".tmp") as fh:
        temporary = Path(fh.name)
        fh.write(content)
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, path)
