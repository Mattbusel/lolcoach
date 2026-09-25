"""Local match-review web application.

A localhost-only viewer over data the pipeline already stored. The design
constraint that shapes every screen: inferred values are labelled and carry
their confidence, observed values are not dressed up, and nothing is invented
to fill a gap. A wave state at 0.42 confidence renders with an amber bar, not
as a fact.

The front end lives in ``lolcoach/web`` as ordinary static files rather than a
Python string, so it can be edited, diffed and reviewed like code. It uses no
bundler and no CDN, which keeps the packaged desktop build working offline.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .assets import (champion_icon_path, current_version, map_bounds,
                     map_image_path)
from .config import Config
from .storage import Database

def _web_root() -> Path:
    """Locate the static front end in both source and frozen builds.

    PyInstaller unpacks ``--add-data`` payloads under ``sys._MEIPASS``. The
    module-relative path usually resolves there too, but checking _MEIPASS
    first makes the packaged app independent of how the module was imported --
    a blank page in a shipped .exe is an expensive way to find that out.
    """
    import sys

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "lolcoach" / "web"
        if bundled.is_dir():
            return bundled
    return Path(__file__).resolve().parent / "web"


WEB_ROOT = _web_root()

#: What the app needs, split by what each tier actually unlocks. Reviewing
#: games is cheap; only the conversational coach needs a GPU, so the two are
#: listed separately rather than scaring people off with one big number.
SYSTEM_REQUIREMENTS = [
    {
        "tier": "To review your games",
        "note": "Timelines, the map, charts and coaching moments. No GPU needed.",
        "items": [
            ["Operating system", "Windows 10 or 11, 64-bit"],
            ["Memory", "8 GB RAM"],
            ["Disk", "8 GB free (grows with the games you collect)"],
            ["Internet", "Needed to sync games and fetch champion art"],
            ["Riot API key", "Free from developer.riotgames.com; a development key expires every 24 hours"],
        ],
    },
    {
        "tier": "To chat with the AI coach",
        "note": "The model runs on your PC, so once it fits, conversation is unlimited and free.",
        "items": [
            ["GPU", "NVIDIA with 8 GB VRAM or more (RTX 3060/4060 and up)"],
            ["VRAM", "8 GB runs the 7B model in 4-bit; 12 GB is comfortable"],
            ["Memory", "16 GB RAM"],
            ["Disk", "20 GB free for the bundled model"],
            ["Driver", "Recent NVIDIA driver with CUDA 12 support"],
        ],
    },
    {
        "tier": "Without a supported GPU",
        "note": "Everything except the chat still works, and that is most of the app.",
        "items": [
            ["Review", "Full match review, map, charts and moments all work"],
            ["Chat", "Answers fall back to CPU and are slow, or are unavailable"],
        ],
    },
]


@dataclass
class ReviewStore:
    """Read-only projection of the SQLite feature store for the local UI."""

    cfg: Config

    def player_puuid(self) -> str | None:
        """The opted-in player's PUUID, used to highlight their own row."""
        return self.cfg.secret("LOLCOACH_PUUID")

    def list_matches(self, limit: int = 50) -> list[dict[str, Any]]:
        """Games newest first, annotated with the viewer's own result.

        The result and champion come from the viewer's participant row when a
        PUUID is configured, so the rail reads like a match history instead of
        a list of opaque ids.
        """
        puuid = self.player_puuid()
        db = Database(self.cfg.paths.db)
        try:
            rows = db.query(
                "SELECT m.match_id, m.source, m.patch, m.duration_s, m.game_start_ms, m.tier, "
                "       m.has_timeline, "
                "       (SELECT COUNT(*) FROM frames f WHERE f.match_id=m.match_id) AS frames, "
                "       (SELECT COUNT(*) FROM wave_states w WHERE w.match_id=m.match_id) AS waves, "
                "       (SELECT COUNT(*) FROM deaths d WHERE d.match_id=m.match_id) AS deaths, "
                "       (SELECT COUNT(*) FROM objective_events o WHERE o.match_id=m.match_id) AS objectives, "
                "       (SELECT COUNT(*) FROM recalls r WHERE r.match_id=m.match_id) AS recalls "
                "FROM matches m "
                "ORDER BY COALESCE(m.game_start_ms, 0) DESC, m.match_id DESC LIMIT ?",
                (max(1, min(int(limit), 200)),))
            out: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["moments"] = (int(item.get("deaths") or 0)
                                   + int(item.get("objectives") or 0)
                                   + int(item.get("recalls") or 0))
                item.update(self._viewer_row(db, item["match_id"], puuid))
                out.append(item)
            return out
        finally:
            db.close()

    @staticmethod
    def _viewer_row(db: Database, match_id: str, puuid: str | None) -> dict[str, Any]:
        """Champion and result for the viewer, when they can be identified."""
        if puuid:
            rows = db.query(
                "SELECT champion, win FROM participants WHERE match_id=? AND puuid=?",
                (match_id, puuid))
            if rows:
                return {"champion": rows[0]["champion"], "win": rows[0]["win"]}
        return {"champion": None, "win": None}

    def match_detail(self, match_id: str) -> dict[str, Any]:
        puuid = self.player_puuid()
        db = Database(self.cfg.paths.db)
        try:
            match = db.query("SELECT * FROM matches WHERE match_id=?", (match_id,))
            if not match:
                raise LookupError(f"No local match named {match_id}")
            participants = [dict(row) for row in db.query(
                "SELECT p.participant_id,p.team_id,p.champion,p.role,p.lane,p.win,p.kills,p.deaths,p.puuid,"
                "p.assists,p.gold_earned,p.cs,p.damage_dealt,p.vision_score,c.key AS champion_key "
                "FROM participants p LEFT JOIN champions c ON c.name = p.champion "
                "WHERE p.match_id=? ORDER BY p.team_id, p.participant_id", (match_id,))]
            frames = [dict(row) for row in db.query(
                "SELECT f.ts_ms,f.participant_id,f.x,f.y,f.total_gold,f.current_gold,f.level,f.minions,"
                "f.jungle_minions,f.health,f.health_max,p.team_id,p.champion,p.role,c.key AS champion_key "
                "FROM frames f LEFT JOIN participants p ON p.match_id=f.match_id "
                "AND p.participant_id=f.participant_id "
                "LEFT JOIN champions c ON c.name = p.champion "
                "WHERE f.match_id=? ORDER BY f.ts_ms,f.participant_id", (match_id,))]
            waves = [dict(row) for row in db.query(
                "SELECT ts_ms,lane,team_id,position,velocity,state,size_diff,cannon,recommended,confidence,rationale "
                "FROM wave_states WHERE match_id=? ORDER BY ts_ms,lane,team_id", (match_id,))]
            return {
                "match": dict(match[0]),
                "participants": participants,
                "frames": frames,
                "wave_states": waves,
                "moments": self._moments(db, match_id),
                "map_bounds": map_bounds(),
                # Which participant is the viewer, so the UI can centre the
                # review on them without the client knowing their PUUID.
                "you": next((p["participant_id"] for p in participants
                             if puuid and p.get("puuid") == puuid), None),
            }
        finally:
            db.close()

    def moment_context(self, match_id: str, ts_ms: int) -> str:
        """Compact, evidence-only context for the ask-about-this-moment box."""
        detail = self.match_detail(match_id)
        moments = [m for m in detail["moments"] if abs(int(m["ts_ms"]) - ts_ms) <= 60_000]
        waves = [w for w in detail["wave_states"] if abs(int(w["ts_ms"]) - ts_ms) <= 60_000]
        frames = [f for f in detail["frames"] if int(f["ts_ms"]) == ts_ms]
        text = [f"Match {match_id}, timestamp {format_timestamp(ts_ms)}."]
        if frames:
            # Only the players who matter: the viewer plus a few others. Ten
            # full stat lines bloat the prompt without improving the answer.
            puuid = self.player_puuid()
            mine = [f for f in frames if puuid and f.get("puuid") == puuid]
            shown = (mine + [f for f in frames if f not in mine])[:6]
            roster = "; ".join(
                f"{row.get('champion') or 'Unknown'} level {row.get('level') or 0}, "
                f"{row.get('health') or 0}/{row.get('health_max') or 0} HP, "
                f"{row.get('current_gold') or 0} unspent gold"
                for row in shown)
            text.append(f"Observed player frames: {roster}.")
        if waves:
            text.append("Inferred waves: " + "; ".join(
                f"{row['lane']} team {row['team_id']}: {row['state']} "
                f"(confidence {float(row['confidence'] or 0):.2f}, recommendation {row['recommended']})"
                for row in waves))
        if moments:
            text.append("Derived moments: " + "; ".join(m["summary"] for m in moments[:6]))
        if len(text) == 1:
            text.append("No stored timeline frame or derived event is available at this timestamp.")
        return "\n".join(text)

    @staticmethod
    def _moments(db: Database, match_id: str) -> list[dict[str, Any]]:
        moments: list[dict[str, Any]] = []
        for row in db.query(
            "SELECT d.ts_ms,d.victim_id,d.killer_id,d.cause,d.cost_gold,d.x,d.y,d.zone,"
            "       d.rationale,p.champion,p.team_id "
            "FROM deaths d LEFT JOIN participants p ON p.match_id=d.match_id "
            "AND p.participant_id=d.victim_id WHERE d.match_id=? ORDER BY d.ts_ms",
            (match_id,)):
            rationale = _rationale(row["rationale"])
            champion = row["champion"] or f"Player {row['victim_id']}"
            moments.append({
                "kind": "death", "ts_ms": row["ts_ms"],
                "title": f"{champion} died: {str(row['cause'] or 'unclassified').replace('_', ' ').lower()}",
                "summary": "; ".join(rationale) or "A player death was recorded.",
                "confidence": None, "importance": 100 + min(50, int(row["cost_gold"] or 0) // 100),
                # Coordinates drive the death heatmap overlay on the minimap.
                "details": {"victim_id": row["victim_id"], "cost_gold": row["cost_gold"],
                            "x": row["x"], "y": row["y"], "zone": row["zone"],
                            "team_id": row["team_id"], "champion": row["champion"]},
            })
        for row in db.query(
            "SELECT ts_ms,participant_id,gold_spent,wave_state,was_good,reason FROM recalls WHERE match_id=? ORDER BY ts_ms",
            (match_id,)):
            good = "good" if row["was_good"] else "poor" if row["was_good"] is not None else "unscored"
            moments.append({
                "kind": "recall", "ts_ms": row["ts_ms"], "title": f"Recall: {good}",
                "summary": str(row["reason"] or f"Spent {row['gold_spent'] or 0} gold with {row['wave_state'] or 'unknown'} wave state."),
                "confidence": None,
                "importance": 85 if row["was_good"] == 0 else 50,
                "details": {"participant_id": row["participant_id"], "gold_spent": row["gold_spent"]},
            })
        for row in db.query(
            "SELECT ts_ms,objective,subtype,team_id,contested,setup_score,vision_edge,rationale "
            "FROM objective_events WHERE match_id=? ORDER BY ts_ms", (match_id,)):
            rationale = _rationale(row["rationale"])
            moments.append({
                "kind": "objective", "ts_ms": row["ts_ms"],
                "title": f"Objective: {str(row['objective']).replace('_', ' ').lower()}",
                "summary": "; ".join(rationale) or f"Team {row['team_id'] or '?'} took {row['objective']}.",
                "confidence": row["setup_score"],
                "importance": 80 if row["contested"] else 55,
                "details": {"contested": row["contested"], "vision_edge": row["vision_edge"]},
            })
        for row in db.query(
            "SELECT ts_ms,participant_id,from_zone,to_zone,purpose,tempo_gain,was_correct "
            "FROM rotations WHERE match_id=? ORDER BY ts_ms", (match_id,)):
            verdict = "worked" if row["was_correct"] else "did not pay off" if row["was_correct"] is not None else "has no outcome label"
            moments.append({
                "kind": "rotation", "ts_ms": row["ts_ms"],
                "title": f"Rotation: {str(row['purpose'] or 'movement').replace('_', ' ').lower()}",
                "summary": f"Player {row['participant_id']} moved {str(row['from_zone']).replace('_', ' ').lower()} "
                           f"to {str(row['to_zone']).replace('_', ' ').lower()}; it {verdict}.",
                "confidence": None,
                # The first two minutes contain spawn movements, not decisions.
                "importance": 65 if row["was_correct"] == 0 and int(row["ts_ms"]) >= 120_000 else 15,
                "details": {"tempo_gain": row["tempo_gain"]},
            })
        for row in db.query(
            "SELECT ts_ms,lane,team_id,state,recommended,confidence,rationale FROM wave_states "
            "WHERE match_id=? AND recommended IS NOT NULL AND recommended != 'NEUTRAL' ORDER BY ts_ms,lane,team_id",
            (match_id,)):
            moments.append({
                "kind": "wave", "ts_ms": row["ts_ms"],
                "title": f"{str(row['lane']).title()} wave: {str(row['state']).replace('_', ' ').lower()}",
                "summary": "; ".join(_rationale(row["rationale"])) or f"Inferred {row['state']} wave.",
                "confidence": row["confidence"],
                "importance": 35 + int(float(row["confidence"] or 0) * 15),
                "details": {"team_id": row["team_id"], "recommendation": row["recommended"]},
            })
        return sorted(moments, key=lambda item: (-int(item["importance"]), int(item["ts_ms"]), item["kind"]))


#: Shown in the chat panel when the AI stack is not installed, which is the
#: case for the small review-only download.
COACH_NOT_BUNDLED = (
    "The AI chat is not included in this download. Everything else works: your "
    "games, the minimap, wave states, deaths and progress. To talk to the coach, "
    "get the full Windows bundle or install from source with an NVIDIA GPU; see "
    "the README at github.com/Mattbusel/lolcoach.")


def _missing_coach_runtime() -> list[str]:
    """Names of the chat's heavy dependencies that are not importable."""
    import importlib.util

    missing = []
    for name in ("torch", "transformers", "peft"):
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def create_app(cfg: Config):
    """Create the FastAPI app lazily so the core CLI stays dependency-light."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("The review UI needs FastAPI and Uvicorn. Run `python -m pip install -e \".[ui]\"`.") from exc

    store = ReviewStore(cfg)
    coach_lock = threading.Lock()
    coach: Any | None = None
    coach_error: str | None = None
    collection_lock = threading.Lock()
    collection: dict[str, Any] = {"state": "idle", "detail": "No timeline collection has been started."}
    app = FastAPI(title="LoLCoach Review", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(WEB_ROOT)), name="static")

    # The coach is a 7B model in 4-bit: loading it takes a minute or two. Doing
    # that inside the first question means the browser waits on a blank stream
    # with no explanation, which is indistinguishable from a hang. It is loaded
    # once in the background at startup instead, and its state is published so
    # the UI can say what is happening.
    coach_state: dict[str, Any] = {"state": "loading", "detail": "Waking the coach up…"}

    def _preload_coach() -> None:
        nonlocal coach, coach_error
        started = time.time()
        missing = _missing_coach_runtime()
        if missing:
            # The review-only download ships without PyTorch and the model
            # stack. Say so plainly instead of surfacing an ImportError.
            with coach_lock:
                coach_error = COACH_NOT_BUNDLED
                coach_state.update(state="unavailable", detail=COACH_NOT_BUNDLED,
                                   missing=missing)
            return
        try:
            from .chat import LocalCoach

            loaded = LocalCoach(cfg)
        except Exception as exc:
            with coach_lock:
                coach_error = f"{type(exc).__name__}: {exc}"
                coach_state.update(state="error", detail=coach_error)
            return
        with coach_lock:
            coach = loaded
            coach_state.update(
                state="ready",
                detail=f"Coach ready (loaded in {time.time() - started:.0f}s)")

    threading.Thread(target=_preload_coach, name="lolcoach-coach-preload",
                     daemon=True).start()

    def _load_coach(wait: float = 240.0) -> Any:
        """Return the coach, waiting for the background load to finish."""
        deadline = time.time() + wait
        while time.time() < deadline:
            with coach_lock:
                if coach is not None:
                    return coach
                if coach_error:
                    raise HTTPException(status_code=503, detail=coach_error)
            time.sleep(0.4)
        raise HTTPException(
            status_code=503,
            detail="The local coach is still loading. Give it a moment and ask again.")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    # ---- assets -------------------------------------------------------
    @app.get("/assets/map")
    def map_image():
        """The real Summoner's Rift minimap, cached locally by the asset layer."""
        path = map_image_path(cfg, current_version(cfg))
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Minimap image is not cached yet.")
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/assets/champion/{key}")
    def champion_icon(key: str):
        # Reject traversal explicitly: the key becomes a path component.
        if not key.isalnum():
            raise HTTPException(status_code=400, detail="Invalid champion key.")
        path = champion_icon_path(cfg, current_version(cfg), key)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Portrait is not cached yet.")
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})

    # ---- status and setup ---------------------------------------------
    @app.get("/api/status")
    def status() -> dict[str, Any]:
        db = Database(cfg.paths.db)
        try:
            counts = db.counts()
        finally:
            db.close()
        return {
            "matches": counts.get("matches", 0), "data_root": str(cfg.paths.root),
            "riot_configured": bool(cfg.secret("RIOT_API_KEY", "RIOT_KEY", "RIOT_TOKEN")),
            "player_linked": bool(cfg.secret("LOLCOACH_PUUID") and cfg.secret("LOLCOACH_REGION")),
            "region": cfg.secret("LOLCOACH_REGION"),
            "riot_id": cfg.secret("LOLCOACH_RIOT_ID"),
            "my_matches": _my_match_count(cfg),
            "timeline_frames": counts.get("frames", 0),
            "wave_states": counts.get("wave_states", 0),
        }

    @app.get("/api/collection")
    def collection_status() -> dict[str, Any]:
        with collection_lock:
            return dict(collection)

    @app.post("/api/onboard")
    def onboard(request: Mapping[str, Any]) -> dict[str, str]:
        """Validate and save a user-entered Riot key without logging it."""
        from .onboarding import resolve_player, save_riot_key, validate_riot_key

        key = str(request.get("riot_key") or "").strip()
        existing = cfg.secret("RIOT_API_KEY", "RIOT_KEY", "RIOT_TOKEN")
        # Linking a Riot ID later must not require re-entering the key, so a
        # request with no new key reuses the stored one.
        if not key and existing:
            key = existing
        if not key:
            raise HTTPException(status_code=400, detail="Enter a Riot API key.")
        try:
            if key != existing:
                validate_riot_key(cfg, key)
                save_riot_key(key)
            cfg.creds["RIOT_API_KEY"] = key
            riot_id = str(request.get("riot_id") or "").strip()
            if riot_id:
                _puuid, region = resolve_player(cfg, key, riot_id)
                return {"message": f"Linked. Riot serves your match history from "
                                   f"{region}; you can sync your games now."}
        except (PermissionError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"message": "Key saved. Add your Riot ID to sync your own games."}

    @app.post("/api/collection")
    def collect(request: Mapping[str, Any]) -> dict[str, Any]:
        """Collect a bounded Riot timeline sample in a background worker."""
        if not cfg.secret("RIOT_API_KEY", "RIOT_KEY", "RIOT_TOKEN"):
            raise HTTPException(status_code=400, detail="Add and validate a Riot key first.")
        raw_limit = request.get("limit", 25)
        try:
            limit = max(1, min(int(raw_limit), 200))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Timeline collection limit must be an integer.") from exc
        with collection_lock:
            if collection["state"] == "running":
                return dict(collection)
            collection.update(state="running", detail=f"Collecting up to {limit} Riot timelines…",
                              started_at=time.time())

        def set_progress(**fields: Any) -> None:
            with collection_lock:
                collection.update(**fields)

        def run() -> None:
            try:
                from .assets import ensure_assets
                from .features import derive_all
                from .ingest import ingest_all
                from .sources.riot_api import fetch_player_matches

                puuid = cfg.secret("LOLCOACH_PUUID")
                region = cfg.secret("LOLCOACH_REGION")
                if not puuid:
                    raise RuntimeError(
                        "No Riot ID is linked yet, so there is no way to tell which "
                        "games are yours. Open Settings and add your Riot ID as "
                        "GameName#TAG, then sync again.")
                if not region:
                    # The account is linked but region detection did not finish,
                    # usually because Riot was briefly unavailable during
                    # onboarding. Retry it here rather than making the user
                    # re-link a Riot ID that is already correct.
                    from .onboarding import save_player_region
                    from .sources.riot_api import RiotClient, detect_region

                    set_progress(detail="Finding which region serves your account…")
                    probe = RiotClient(cfg.secret("RIOT_API_KEY", "RIOT_KEY", "RIOT_TOKEN"),
                                       cfg, interactive=True)
                    try:
                        region = detect_region(probe, puuid)
                    finally:
                        probe.close()
                    if not region:
                        raise RuntimeError(
                            "Riot did not return match history for your account on "
                            "any region just now. This is usually temporary — try "
                            "Sync again in a moment.")
                    save_player_region(region)
                    cfg.creds["LOLCOACH_REGION"] = region

                known = _known_match_ids(cfg)
                set_progress(detail=f"Asking Riot for your last {limit} games…", phase="download")

                def on_progress(done: int, total: int, summary: dict[str, Any]) -> None:
                    set_progress(detail=f"Downloading your games… {done}/{total} "
                                        f"({summary['downloaded']} new, {summary['skipped']} already stored)")

                summary = fetch_player_matches(
                    cfg, puuid, region, cfg.paths.source("riot"), limit=limit,
                    known=known, progress=on_progress)

                if summary["downloaded"]:
                    set_progress(detail="Reading timelines into your library…", phase="ingest")
                    ingest_all(cfg)
                    set_progress(detail="Working out your coaching moments…", phase="derive")
                    derived = derive_all(cfg)
                    set_progress(detail="Fetching champion art…", phase="assets")
                    ensure_assets(cfg)
                else:
                    derived = None

                db = Database(cfg.paths.db)
                try:
                    counts = db.counts()
                    mine = db.scalar(
                        "SELECT COUNT(DISTINCT match_id) FROM participants WHERE puuid=?",
                        (puuid,)) or 0
                finally:
                    db.close()

                if summary["downloaded"]:
                    detail = (f"Added {summary['downloaded']} new "
                              f"{'game' if summary['downloaded'] == 1 else 'games'}"
                              + (f" and derived {derived.waves} wave states" if derived else "")
                              + f". You now have {mine} of your own games ready to review.")
                elif summary["found"]:
                    detail = (f"You are up to date — all {summary['found']} of your recent games "
                              f"were already stored. {mine} games ready to review.")
                else:
                    detail = ("Riot returned no recent games for your account on "
                              f"{region}. If you have played recently, check that your "
                              "Riot ID is correct in Settings.")
                set_progress(state="complete", detail=detail, completed_at=time.time(), phase="done")
            except PermissionError as exc:
                set_progress(state="failed", completed_at=time.time(),
                             detail=f"Riot rejected the key: {exc}")
            except Exception as exc:
                set_progress(state="failed", completed_at=time.time(),
                             detail=f"Sync failed: {type(exc).__name__}: {exc}")

        threading.Thread(target=run, name="lolcoach-timeline-collection", daemon=True).start()
        return dict(collection)

    # ---- match data ----------------------------------------------------
    @app.get("/api/matches")
    def matches(limit: int = 50, mine: int = 1) -> list[dict[str, Any]]:
        """Your games by default.

        The database also holds high-elo ladder games collected as training
        data. They are useful to the model and useless in a personal match
        history, so the rail shows only games you played unless asked
        otherwise.
        """
        rows = store.list_matches(limit if not mine else max(limit, 200))
        if mine and store.player_puuid():
            owned = [row for row in rows if row.get("champion")]
            if owned:
                return owned[:limit]
        return rows[:limit]

    @app.get("/api/matches/{match_id}")
    def match(match_id: str) -> dict[str, Any]:
        try:
            return store.match_detail(match_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/coach")
    def coach_status() -> dict[str, Any]:
        with coach_lock:
            return dict(coach_state)

    @app.get("/api/progress")
    def progress() -> dict[str, Any]:
        """Trends across the player's games, and the one thing to work on.

        A match viewer shows you a game. A coach shows you the habit. This
        aggregates the derived features across every game the player appears
        in, finds the mistake that costs them most often, and reports whether
        it is getting better or worse.
        """
        return build_progress(cfg)

    @app.get("/api/requirements")
    def requirements() -> dict[str, Any]:
        """What this machine needs, and whether it has it.

        A static requirements list tells someone what to buy; this tells them
        whether the app will work before they spend an evening finding out.
        """
        from .doctor import run_doctor

        try:
            # DoctorCheck names the field ``state``; the UI reads ``status``.
            checks = [{"name": check.name, "status": check.state,
                       "detail": check.detail, "fix": check.fix or ""}
                      for check in run_doctor(cfg)]
        except Exception as exc:
            checks = [{"name": "System check", "status": "warn",
                       "detail": f"Could not complete: {type(exc).__name__}: {exc}",
                       "fix": ""}]
        return {"checks": checks, "spec": SYSTEM_REQUIREMENTS}

    @app.get("/api/matches/{match_id}/insights")
    def insights(match_id: str) -> dict[str, Any]:
        """Instant, model-free takeaways plus questions worth asking.

        This is what makes the coach feel alive the moment a game opens. It is
        computed from the stored features, so it is immediate and costs
        nothing; the model is only spent when the player actually wants the
        reasoning behind one of them.
        """
        try:
            detail = store.match_detail(match_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return build_insights(detail, cfg.secret("LOLCOACH_PUUID"))

    @app.post("/api/matches/{match_id}/ask")
    def ask(match_id: str, request: Mapping[str, Any]) -> dict[str, str]:
        question, context = _question_and_context(store, match_id, request)
        answer = _load_coach().ask(f"{context}\n\nPlayer question: {question}")
        return {"answer": answer.answer, "context": context}

    @app.post("/api/matches/{match_id}/ask/stream")
    def ask_stream(match_id: str, request: Mapping[str, Any]) -> Any:
        """Stream the answer so the UI fills in as the model writes.

        Plain text rather than server-sent events: the browser reads the
        response body incrementally either way, and plain text avoids having
        to escape newlines that matter in coaching prose.
        """
        question, context = _question_and_context(store, match_id, request)

        def generate() -> Iterator[str]:
            # Resolve the coach *inside* the stream. Doing it before returning
            # the response means the browser receives nothing at all until a 7B
            # model has finished loading, which reads as a hang. Here the first
            # bytes arrive instantly and the wait is explained.
            try:
                with coach_lock:
                    ready = coach is not None
                    failed = coach_error
                if failed == COACH_NOT_BUNDLED:
                    yield failed
                    return
                if failed:
                    yield ("The local coach could not start on this machine.\n\n"
                           f"{failed}\n\nRun `lolcoach doctor` for the details.")
                    return
                if not ready:
                    yield "Warming up the coach (first answer after launch takes a minute)…\n\n"
                active = _load_coach()
            except HTTPException as exc:
                yield f"\n[{exc.detail}]"
                return
            try:
                yield from active.stream(f"{context}\n\nPlayer question: {question}",
                                         system=tone_prompt(request.get("tone")))
            except Exception as exc:
                yield f"\n\n[The local coach stopped early: {type(exc).__name__}: {exc}]"

        return StreamingResponse(generate(), media_type="text/plain; charset=utf-8",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})

    def _question_and_context(store: ReviewStore, match_id: str,
                              request: Mapping[str, Any]) -> tuple[str, str]:
        question = str(request.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Enter a coaching question.")
        try:
            return question, store.moment_context(match_id, int(request.get("timestamp_ms") or 0))
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def serve(cfg: Config, *, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = False, native: bool = False) -> None:
    """Start the review UI on loopback only by default.

    With ``native=True`` the app opens in its own window through the system
    WebView (Edge WebView2 on Windows, which ships with Windows 11) instead of
    a browser tab. That is the difference between feeling like a desktop app
    and feeling like a localhost page, and it costs one small dependency.
    Without pywebview installed it falls back to the browser rather than
    failing.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("The review UI needs Uvicorn. Run `python -m pip install -e \".[ui]\"`.") from exc

    # Cache the minimap and portraits before the first paint when possible, so
    # the map is not empty on a first run. Failures are non-fatal: the UI
    # degrades to initials badges.
    try:
        from .assets import ensure_assets
        ensure_assets(cfg)
    except Exception:
        pass

    url = f"http://{host}:{port}"
    app = create_app(cfg)

    def build_server() -> Any:
        # The packaged desktop launcher is built with ``--windowed``, so stdout
        # and stderr are ``None``. Uvicorn's default console formatter calls
        # ``sys.stdout.isatty()``, which would stop the local UI from starting.
        # A stopped Server cannot be restarted, so the browser fallback needs a
        # fresh instance rather than the one the native attempt shut down.
        return uvicorn.Server(uvicorn.Config(
            app, host=host, port=port, log_config=None, access_log=False))

    if native and _try_native_window(build_server(), url, host, port):
        return

    if open_browser:
        webbrowser.open(url)
    build_server().run()


def _try_native_window(server: Any, url: str, host: str, port: int) -> bool:
    """Open the UI in a native window, returning False if that is not possible.

    The native path goes through pywebview, which on Windows drives Edge
    WebView2 via pythonnet. Inside a PyInstaller bundle the .NET loader probes
    the executable's directory rather than ``_internal\\webview\\lib``, so
    ``clr.AddReference`` can fail to resolve the WebView2 assemblies even
    though they were bundled correctly.

    Every failure mode here is recoverable by falling back to the browser, so
    nothing in this function is allowed to reach the caller. An app that will
    not start is a far worse outcome than an app that opens in a browser tab.
    """
    thread: threading.Thread | None = None
    try:
        import webview

        # Give .NET a chance to find the bundled assemblies by adding their
        # directory to the DLL search path before pywebview asks for them.
        _register_webview_assemblies(webview)

        thread = threading.Thread(target=server.run, name="lolcoach-ui", daemon=True)
        thread.start()
        _wait_for_port(host, port)
        webview.create_window("LoLCoach", url, width=1440, height=920,
                              min_size=(1040, 700))
        webview.start()
        server.should_exit = True
        thread.join(timeout=5)
        return True
    except BaseException:
        # Includes ImportError, pythonnet assembly failures and any backend
        # initialisation error. Stop the worker so the browser path can bind
        # the same port cleanly.
        if thread is not None and thread.is_alive():
            server.should_exit = True
            thread.join(timeout=5)
        return False


def _register_webview_assemblies(webview_module: Any) -> None:
    """Put pywebview's bundled .NET assemblies on the DLL search path."""
    try:
        import os
        from pathlib import Path

        lib = Path(webview_module.__file__).resolve().parent / "lib"
        if not lib.is_dir():
            return
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib))
        os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"
    except Exception:
        return


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> None:
    """Block until the local server accepts connections, or give up quietly."""
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.4)
            if probe.connect_ex((host, port)) == 0:
                return
        time.sleep(0.15)


#: How the coach talks. Only the voice changes: every persona is held to the
#: same evidence rules, because a friendlier tone must not become a looser
#: relationship with the truth.
COACH_TONES: dict[str, dict[str, str]] = {
    "direct": {
        "label": "Direct",
        "blurb": "Blunt and specific. Tells you what you did wrong.",
        "prompt": "You are a blunt, experienced League of Legends coach. Lead with "
                  "the mistake. Be specific and unsentimental, but never insulting.",
    },
    "analyst": {
        "label": "Analyst",
        "blurb": "Calm and precise. Explains the reasoning behind every call.",
        "prompt": "You are a calm analytical League of Legends coach. Explain the "
                  "reasoning step by step, name the trade-off, and quantify where "
                  "the evidence allows it.",
    },
    "mentor": {
        "label": "Mentor",
        "blurb": "Encouraging. Frames mistakes as the next thing to practise.",
        "prompt": "You are an encouraging League of Legends mentor. Acknowledge what "
                  "went right, then give one clear thing to practise next. Stay warm "
                  "without softening the actual correction.",
    },
}

DEFAULT_TONE = "analyst"


def tone_prompt(tone: str | None) -> str:
    """System prompt for a chosen persona, falling back to the analyst."""
    return COACH_TONES.get((tone or "").lower(), COACH_TONES[DEFAULT_TONE])["prompt"]


def build_progress(cfg: Config) -> dict[str, Any]:
    """Aggregate the player's habits across every game they appear in."""
    puuid = cfg.secret("LOLCOACH_PUUID")
    if not puuid:
        return {"linked": False, "games": 0, "focus": None, "trends": [], "causes": []}

    db = Database(cfg.paths.db)
    try:
        games = [dict(row) for row in db.query(
            "SELECT m.match_id, m.game_start_ms, p.participant_id, p.win, p.champion "
            "FROM matches m JOIN participants p ON p.match_id=m.match_id "
            "WHERE p.puuid=? AND m.has_timeline=1 "
            "ORDER BY COALESCE(m.game_start_ms, 0) DESC LIMIT 40", (puuid,))]
        if not games:
            return {"linked": True, "games": 0, "focus": None, "trends": [], "causes": []}

        ids = [g["match_id"] for g in games]
        placeholders = ",".join("?" * len(ids))
        by_match = {g["match_id"]: g for g in games}

        rows = db.query(
            f"SELECT match_id, victim_id, cause FROM deaths "
            f"WHERE match_id IN ({placeholders})", ids)
        # Only this player's deaths, matched per game because participant ids
        # are per match rather than global.
        causes: dict[str, int] = {}
        per_game: dict[str, int] = {mid: 0 for mid in ids}
        for row in rows:
            game = by_match.get(row["match_id"])
            if not game or row["victim_id"] != game["participant_id"]:
                continue
            cause = str(row["cause"] or "UNCLEAR").replace("_", " ").lower()
            causes[cause] = causes.get(cause, 0) + 1
            per_game[row["match_id"]] += 1

        ranked = sorted(causes.items(), key=lambda kv: -kv[1])
        total_deaths = sum(causes.values())
        wins = sum(1 for g in games if g["win"])

        # Trend: recent half versus older half, oldest first.
        ordered = list(reversed(ids))
        half = max(1, len(ordered) // 2)
        older = [per_game[m] for m in ordered[:half]]
        recent = [per_game[m] for m in ordered[half:]] or older
        older_avg = sum(older) / len(older)
        recent_avg = sum(recent) / len(recent)

        focus = None
        if ranked:
            cause, count = ranked[0]
            share = count / max(1, total_deaths)
            focus = {
                "cause": cause,
                "count": count,
                "share": round(share, 2),
                "headline": f"Your most expensive habit: {cause}",
                "detail": f"{count} of your {total_deaths} deaths across "
                          f"{len(games)} games, {round(share * 100)}% of the total.",
                "question": f"Across my recent games I keep dying to {cause}. "
                            f"Give me a concrete plan to fix it.",
            }

        return {
            "linked": True,
            "games": len(games),
            "wins": wins,
            "winrate": round(wins / len(games), 3),
            "deaths_per_game": round(total_deaths / len(games), 2),
            "trend": {
                "older": round(older_avg, 2),
                "recent": round(recent_avg, 2),
                "direction": "better" if recent_avg < older_avg - 0.25
                             else "worse" if recent_avg > older_avg + 0.25 else "flat",
            },
            "causes": [{"cause": c, "count": n} for c, n in ranked[:6]],
            "focus": focus,
        }
    finally:
        db.close()


def build_insights(detail: Mapping[str, Any], puuid: str | None = None) -> dict[str, Any]:
    """Pick the few things from a game that are actually worth talking about.

    Ordered by what a coach would open with: a repeated death pattern beats a
    single death, a thrown lead beats a clean win, and a contested objective
    beats an uncontested one. Every headline carries the timestamp it refers
    to so the UI can jump there, and a question the player can ask in one
    click.
    """
    moments = list(detail.get("moments") or [])
    frames = list(detail.get("frames") or [])
    participants = list(detail.get("participants") or [])
    me = next((p for p in participants if puuid and p.get("puuid") == puuid), None)
    my_id = me.get("participant_id") if me else None

    highlights: list[dict[str, Any]] = []

    # 1. A repeated death cause is the most actionable thing in any game --
    #    but only for one player. Aggregating all ten and calling it "you"
    #    would be a fabrication, so this needs an identified player.
    if my_id:
        causes: dict[str, list[dict[str, Any]]] = {}
        for moment in moments:
            if moment.get("kind") != "death":
                continue
            if moment.get("details", {}).get("victim_id") != my_id:
                continue
            cause = str(moment.get("title", "")).split(":")[-1].strip()
            causes.setdefault(cause, []).append(moment)
        repeated = sorted((c for c in causes.items() if len(c[1]) >= 2),
                          key=lambda kv: -len(kv[1]))
        if repeated:
            cause, group = repeated[0]
            highlights.append({
                "kind": "pattern", "ts_ms": group[0]["ts_ms"],
                "headline": f"You died the same way {len(group)} times: {cause}",
                "detail": "Same mistake, more than once. That is the cheapest thing to fix.",
                "question": f"I died {len(group)} times to {cause} in this game. "
                            f"What is the pattern and how do I stop it?",
            })

    # 2. The biggest gold swing against the player's team.
    swing = _largest_swing(frames, me.get("team_id") if me else 100)
    if swing:
        side = "you" if my_id else "blue side"
        highlights.append({
            "kind": "swing", "ts_ms": swing["ts_ms"],
            "headline": f"{abs(swing['delta']):,} gold swung against "
                        f"{side} around {format_timestamp(swing['ts_ms'])}",
            "detail": "The largest single swing in the game.",
            "question": f"At {format_timestamp(swing['ts_ms'])} the game swung "
                        f"hard. What went wrong there?",
        })

    # 3. A contested objective is where games are usually decided.
    contested = [m for m in moments if m.get("kind") == "objective"
                 and (m.get("details") or {}).get("contested")]
    if contested:
        first = contested[0]
        highlights.append({
            "kind": "objective", "ts_ms": first["ts_ms"],
            "headline": f"Contested {first['title'].split(':')[-1].strip()} at "
                        f"{format_timestamp(first['ts_ms'])}",
            "detail": first.get("summary", ""),
            "question": f"How should we have set up the objective at "
                        f"{format_timestamp(first['ts_ms'])}?",
        })

    # 4. A low-confidence wave call is worth a second opinion.
    weak = [m for m in moments if m.get("kind") == "wave"
            and (m.get("confidence") or 1) < 0.55]
    if weak and len(highlights) < 4:
        pick = weak[0]
        highlights.append({
            "kind": "wave", "ts_ms": pick["ts_ms"],
            "headline": f"Unclear wave call at {format_timestamp(pick['ts_ms'])}",
            "detail": "The estimate here is weak, so it is worth checking yourself.",
            "question": f"What was the wave doing at {format_timestamp(pick['ts_ms'])} "
                        f"and what should I have done?",
        })

    starters = [
        "What are the three biggest mistakes I made in this game?",
        "What should I practise from this game?",
        "How did I do on wave management here?",
        "Where did I lose the most tempo?",
    ]
    return {"highlights": highlights[:4], "starters": starters,
            "player_identified": bool(my_id)}


def _largest_swing(frames: list[Mapping[str, Any]], team_id: int
                   ) -> dict[str, Any] | None:
    """Find the minute where the team gold difference moved most against a team."""
    totals: dict[int, dict[int, int]] = {}
    for frame in frames:
        stamp = int(frame.get("ts_ms") or 0)
        side = frame.get("team_id")
        if side not in (100, 200):
            continue
        bucket = totals.setdefault(stamp, {100: 0, 200: 0})
        bucket[side] += int(frame.get("total_gold") or 0)
    if len(totals) < 3:
        return None
    other = 200 if team_id == 100 else 100
    series = sorted((stamp, teams[team_id] - teams[other])
                    for stamp, teams in totals.items())
    worst = None
    for (_, before), (stamp, after) in zip(series, series[1:]):
        delta = after - before
        if delta < 0 and (worst is None or delta < worst["delta"]):
            worst = {"ts_ms": stamp, "delta": delta}
    # Ignore noise: a swing under a kill's worth of gold is not a story.
    if worst and abs(worst["delta"]) >= 800:
        return worst
    return None


def _my_match_count(cfg: Config) -> int:
    """How many stored games actually contain the linked player."""
    puuid = cfg.secret("LOLCOACH_PUUID")
    if not puuid:
        return 0
    db = Database(cfg.paths.db)
    try:
        return int(db.scalar(
            "SELECT COUNT(DISTINCT match_id) FROM participants WHERE puuid=?",
            (puuid,)) or 0)
    finally:
        db.close()


def _known_match_ids(cfg: Config) -> set[str]:
    """Match ids already ingested, so a sync only pays for new games."""
    db = Database(cfg.paths.db)
    try:
        return {str(row[0]) for row in db.query(
            "SELECT match_id FROM matches WHERE has_timeline=1")}
    finally:
        db.close()


def format_timestamp(ts_ms: int) -> str:
    minutes, seconds = divmod(max(0, int(ts_ms) // 1000), 60)
    return f"{minutes}:{seconds:02d}"


def _rationale(value: Any) -> list[str]:
    if not value:
        return []
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError):
        return [str(value)]
    if isinstance(decoded, list):
        return [str(item) for item in decoded if str(item).strip()]
    return [str(decoded)]
