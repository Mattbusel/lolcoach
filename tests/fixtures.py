"""Synthetic match timelines with known ground truth.

These are **test fixtures**, not training data. They exist so the Stage-2
estimators can be verified without a Riot API key: each fixture scripts a
wave pattern we already know the answer to (a freeze on top, a slow push into
a crash on mid) and the tests assert that the estimator recovers it.

The output is shaped exactly like Riot's match-v5 and timeline payloads, so
the same ingest path is exercised as with real data.
"""

from __future__ import annotations

import math
import random
from typing import Any

from lolcoach.features.geometry import (BLUE, JUNGLE_CAMPS, NEXUS,
                                        OUTER_TURRETS, RED, lane_geometry)
from lolcoach.features.jungle import STANDARD_ROUTES

FRAME_MS = 60_000

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
CHAMPS_BLUE = ["Chogath", "Viego", "Zed", "Jhin", "Xerath"]
CHAMPS_RED = ["DrMundo", "Kayn", "Fizz", "Twitch", "Seraphine"]

#: Scripted wave position per lane, as a function of minute, from blue's
#: perspective (-1 blue turret .. +1 red turret). These are the ground truth
#: the tests check against.
def scripted_wave(lane: str, minute: float) -> float:
    """Wave position for a lane at a given minute (blue perspective)."""
    if lane == "TOP":
        # Pushes out briefly, then is frozen just outside blue's turret.
        if minute < 4:
            return -0.1 + 0.05 * minute
        return -0.55 + 0.01 * math.sin(minute)          # effectively static
    if lane == "MIDDLE":
        # Slow push from minute 5, crashing into red's turret around minute 9.
        if minute < 5:
            return 0.0
        if minute < 9:
            return (minute - 5) * 0.2
        return min(1.0, 0.8 + (minute - 9) * 0.05)
    # Bottom stays roughly even.
    return 0.15 * math.sin(minute / 2.0)


def _lane_position(lane: str, wave_p: float, team: int, offset: float) -> dict[str, int]:
    """Map a wave coordinate onto a champion position near the wave."""
    geo = lane_geometry(lane, BLUE)
    p = wave_p + (offset if team == BLUE else -offset)
    x, y = geo.point_at(max(-1.2, min(1.2, p)))
    return {"x": int(x), "y": int(y)}


def build_synthetic_match(match_id: str = "TEST_0001", minutes: int = 25,
                          seed: int = 7) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(match, timeline)`` payloads in Riot's format."""
    rng = random.Random(seed)

    participants = []
    for i in range(10):
        team = BLUE if i < 5 else RED
        role = ROLES[i % 5]
        champs = CHAMPS_BLUE if team == BLUE else CHAMPS_RED
        participants.append({
            "participantId": i + 1,
            "puuid": f"puuid-{i + 1:02d}",
            "teamId": team,
            "championName": champs[i % 5],
            "championId": 100 + i,
            "teamPosition": role,
            "individualPosition": role,
            "lane": {"TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MIDDLE",
                     "BOTTOM": "BOTTOM", "UTILITY": "BOTTOM"}[role],
            "win": team == BLUE,
            "kills": rng.randint(0, 12), "deaths": rng.randint(0, 9),
            "assists": rng.randint(0, 15),
            "goldEarned": rng.randint(8000, 16000),
            "totalMinionsKilled": rng.randint(100, 240),
            "neutralMinionsKilled": rng.randint(0, 120),
            "totalDamageDealtToChampions": rng.randint(8000, 30000),
            "visionScore": rng.randint(10, 60),
            "summoner1Id": 4, "summoner2Id": 14,
            "perks": {"styles": [{"selections": [{"perk": 8005}]}]},
            **{f"item{j}": 3000 + j for j in range(7)},
        })

    match = {
        "metadata": {"matchId": match_id,
                     "participants": [p["puuid"] for p in participants]},
        "info": {
            "gameId": 1234567890,
            "platformId": "NA1",
            "queueId": 420,
            "gameVersion": "16.16.652.1234",
            "gameCreation": 1_700_000_000_000,
            "gameStartTimestamp": 1_700_000_000_000,
            "gameDuration": minutes * 60,
            "participants": participants,
            "teams": [{"teamId": BLUE, "win": True}, {"teamId": RED, "win": False}],
        },
    }

    frames: list[dict[str, Any]] = []
    for minute in range(minutes + 1):
        ts = minute * FRAME_MS
        pframes: dict[str, Any] = {}
        events: list[dict[str, Any]] = []

        for p in participants:
            pid = p["participantId"]
            team = p["teamId"]
            role = p["teamPosition"]
            level = min(18, 1 + int(minute * 0.62))
            cs = _cs_for(role, minute, rng)

            if role == "JUNGLE":
                route = STANDARD_ROUTES[team][0]
                camp = route[min(len(route) - 1, minute % len(route))]
                pos = JUNGLE_CAMPS[camp]
                position = {"x": int(pos[0]), "y": int(pos[1])}
                jungle_cs = int(minute * 3.4)
            else:
                lane = {"TOP": "TOP", "MIDDLE": "MIDDLE",
                        "BOTTOM": "BOTTOM", "UTILITY": "BOTTOM"}[role]
                wave_p = scripted_wave(lane, minute)
                # Laners stand just behind their own minions.
                position = _lane_position(lane, wave_p, team, 0.14)
                jungle_cs = 0

            # Everyone starts at their fountain, and backs at minute 8.
            if minute == 0 or minute == 8:
                base = NEXUS[team]
                position = {"x": int(base[0]), "y": int(base[1])}

            health_max = 600 + level * 95
            pframes[str(pid)] = {
                "participantId": pid,
                "position": position,
                "currentGold": 200 + (minute % 8) * 180,
                "totalGold": 500 + minute * 330,
                "xp": minute * 380,
                "level": level,
                "minionsKilled": cs,
                "jungleMinionsKilled": jungle_cs,
                "championStats": {
                    "health": int(health_max * (0.55 + 0.4 * abs(math.sin(minute + pid)))),
                    "healthMax": health_max,
                    "power": 300, "powerMax": 400,
                    "attackDamage": 60 + level * 4,
                    "abilityPower": 0 if role in ("TOP", "BOTTOM") else level * 8,
                    "armor": 30 + level * 3,
                    "magicResist": 30 + level * 2,
                },
                "damageStats": {
                    "totalDamageTaken": minute * 400,
                    "totalDamageDoneToChampions": minute * 350,
                },
            }

            if minute == 8:
                events.append({"type": "ITEM_PURCHASED", "timestamp": ts + 1000,
                               "participantId": pid, "itemId": 3006})

        # Scripted objective and structure events.
        if minute == 5:
            events.append({"type": "ELITE_MONSTER_KILL", "timestamp": ts + 5000,
                           "killerId": 2, "killerTeamId": BLUE,
                           "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON",
                           "position": {"x": 9866, "y": 4414}})
        if minute == 11:
            events.append({"type": "ELITE_MONSTER_KILL", "timestamp": ts + 3000,
                           "killerId": 7, "killerTeamId": RED,
                           "monsterType": "DRAGON", "monsterSubType": "OCEAN_DRAGON",
                           "position": {"x": 9866, "y": 4414}})
        if minute == 14:
            events.append({"type": "ELITE_MONSTER_KILL", "timestamp": ts + 2000,
                           "killerId": 2, "killerTeamId": BLUE,
                           "monsterType": "RIFTHERALD",
                           "position": {"x": 4950, "y": 10400}})
        if minute in (7, 9):
            lane_tag = "MID_LANE"
            events.append({"type": "TURRET_PLATE_DESTROYED", "timestamp": ts + 8000,
                           "teamId": RED, "laneType": lane_tag,
                           "position": {"x": 8955, "y": 8510}})
        if minute == 12:
            events.append({"type": "BUILDING_KILL", "timestamp": ts + 4000,
                           "teamId": RED, "buildingType": "TOWER_BUILDING",
                           "towerType": "OUTER_TURRET", "laneType": "MID_LANE",
                           "killerId": 3, "position": {"x": 8955, "y": 8510}})
        if minute in (6, 13, 18):
            victim = 4 if minute != 13 else 9
            killer = 9 if minute != 13 else 4
            events.append({"type": "CHAMPION_KILL", "timestamp": ts + 12000,
                           "killerId": killer, "victimId": victim,
                           "bounty": 300, "assistingParticipantIds": [2],
                           "position": {"x": 11000, "y": 3000}})
        if minute in (4, 10, 16):
            events.append({"type": "WARD_PLACED", "timestamp": ts + 1000,
                           "participantId": 5, "wardType": "YELLOW_TRINKET"})

        frames.append({"timestamp": ts, "participantFrames": pframes, "events": events})

    timeline = {
        "metadata": {"matchId": match_id},
        "info": {
            "frameInterval": FRAME_MS,
            "frames": frames,
            "participants": [{"participantId": p["participantId"],
                              "puuid": p["puuid"]} for p in participants],
        },
    }
    return match, timeline


def _cs_for(role: str, minute: int, rng: random.Random) -> int:
    """Plausible cumulative CS for a role at a given minute."""
    if role == "JUNGLE":
        return int(minute * 1.2)
    if role == "UTILITY":
        return int(minute * 0.8)
    return int(minute * 7.4)


def write_synthetic(dest_dir, match_id: str = "TEST_0001", minutes: int = 25,
                    seed: int = 7):
    """Write a synthetic match pair to ``dest_dir`` in the Riot layout."""
    import gzip
    import json
    from pathlib import Path

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    match, timeline = build_synthetic_match(match_id, minutes, seed)
    for obj, name in ((match, f"{match_id}.json.gz"),
                      (timeline, f"{match_id}.timeline.json.gz")):
        with gzip.open(dest / name, "wb") as fh:
            fh.write(json.dumps(obj).encode("utf-8"))
    return dest / f"{match_id}.json.gz", dest / f"{match_id}.timeline.json.gz"
