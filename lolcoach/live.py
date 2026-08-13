"""Read-only personal telemetry from the local League Game Client API.

This module intentionally does *not* infer hidden enemy state or issue
in-game instructions.  Riot's third-party policy prohibits applications that
dictate player decisions or reveal game-session information that was not
already known to the player.  The ``live`` command is therefore a transparent
local status readout: the active player's own visible stats, inventory,
scores, and elapsed game time.

The Game Client API uses a self-signed certificate on a fixed localhost
endpoint.  Certificate verification is disabled only for that literal URL;
no user-controlled host can be passed to this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import httpx

LIVE_CLIENT_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"


@dataclass(frozen=True)
class LiveSnapshot:
    """Personal, already-visible game state returned by the Game Client API."""

    game_time_s: float
    champion: str
    summoner_name: str
    level: int
    current_gold: int
    health: float
    max_health: float
    mana: float
    max_mana: float
    kills: int
    deaths: int
    assists: int
    cs: int
    items: tuple[str, ...]
    events_seen: int

    @property
    def health_pct(self) -> float:
        return self.health / self.max_health if self.max_health > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["items"] = list(self.items)
        result["health_pct"] = round(self.health_pct, 4)
        return result


class LiveClientUnavailable(RuntimeError):
    """The League game client is not running or its local API is unavailable."""


def fetch_live_snapshot(*, timeout_s: float = 2.0) -> LiveSnapshot:
    """Fetch one snapshot from the fixed local Game Client API endpoint."""
    try:
        with httpx.Client(verify=False, timeout=timeout_s, trust_env=False) as client:
            response = client.get(LIVE_CLIENT_URL)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise LiveClientUnavailable(
            "League's local Game Client API is unavailable. Start a game, then run `lolcoach live` again.") from exc
    if not isinstance(payload, Mapping):
        raise LiveClientUnavailable("League returned an unexpected local Game Client API response.")
    return parse_live_payload(payload)


def parse_live_payload(payload: Mapping[str, Any]) -> LiveSnapshot:
    """Turn an API payload into a stable, small and non-directive snapshot."""
    active = _mapping(payload.get("activePlayer"))
    game = _mapping(payload.get("gameData"))
    stats = _mapping(active.get("championStats"))
    scores = _mapping(_active_player_row(payload, active).get("scores"))
    items = tuple(
        str(_mapping(item).get("displayName") or _mapping(item).get("itemID") or "Unknown item")
        for item in _sequence(_active_player_row(payload, active).get("items"))
    )
    events = _sequence(_mapping(payload.get("events")).get("Events"))
    return LiveSnapshot(
        game_time_s=_number(game.get("gameTime")),
        champion=str(active.get("championName") or _active_player_row(payload, active).get("championName") or "Unknown"),
        summoner_name=str(active.get("summonerName") or "Player"),
        level=_integer(active.get("level") or _active_player_row(payload, active).get("level")),
        current_gold=_integer(active.get("currentGold")),
        health=_number(stats.get("currentHealth")),
        max_health=_number(stats.get("maxHealth")),
        mana=_number(stats.get("resourceValue")),
        max_mana=_number(stats.get("resourceMax")),
        kills=_integer(scores.get("kills")),
        deaths=_integer(scores.get("deaths")),
        assists=_integer(scores.get("assists")),
        cs=_integer(scores.get("creepScore")),
        items=items,
        events_seen=len(events),
    )


def format_live_snapshot(snapshot: LiveSnapshot) -> str:
    """Render a concise status line without directing a player action."""
    minutes, seconds = divmod(max(0, int(snapshot.game_time_s)), 60)
    inventory = ", ".join(snapshot.items) if snapshot.items else "no completed items"
    return (
        f"{minutes:02d}:{seconds:02d} | {snapshot.summoner_name} on {snapshot.champion} | "
        f"level {snapshot.level} | {snapshot.health:.0f}/{snapshot.max_health:.0f} HP "
        f"({snapshot.health_pct * 100:.0f}%) | {snapshot.current_gold} gold | "
        f"{snapshot.kills}/{snapshot.deaths}/{snapshot.assists} | {snapshot.cs} CS | {inventory}")


def _active_player_row(payload: Mapping[str, Any], active: Mapping[str, Any]) -> Mapping[str, Any]:
    wanted = str(active.get("summonerName") or "").casefold()
    for row in _sequence(payload.get("allPlayers")):
        candidate = _mapping(row)
        if wanted and str(candidate.get("summonerName") or "").casefold() == wanted:
            return candidate
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _integer(value: Any) -> int:
    return int(_number(value))
