"""Stage 2b: run every estimator over ingested matches and persist the results.

This is the orchestration layer. For each match with a timeline it:

1. loads participants, frames and events into memory (one match at a time --
   a full game is only a few thousand rows, so this stays cheap);
2. runs the wave estimator per lane per team, in chronological order;
3. builds per-player lane states, including trades, priority and recall value;
4. reconstructs jungle paths and feeds the transition model;
5. scores every objective take;
6. explains every death, clusters fights and labels rotations;
7. writes all of it back into the Stage-2 feature tables.

The jungle transition model is global rather than per match, so it is loaded
once, updated across the corpus, and persisted to ``meta`` at the end -- which
is what makes jungle *prediction* improve as more matches are ingested.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..logging_utils import get_logger
from ..storage import Database
from . import objectives as obj
from .geometry import (BLUE, RED, Point, apply_refined_turrets, assign_lane,
                       dist, quadrant_of, refine_turrets_from_events, zone_of)
from .jungle import (CampClear, JunglePredictor, detect_ganks, phase_of,
                     reconstruct_path)
from .lane import ChampionState, build_lane_state, detect_recalls, recall_was_good
from .macro import (analyse_death, cluster_fights, detect_rotations,
                    positioning_notes)
from .waves import (DecisionContext, LaneFrameInput, WaveEstimator, recommend)

log = get_logger("features.pipeline")

ROLE_TO_LANE = {"TOP": "TOP", "MIDDLE": "MIDDLE", "MID": "MIDDLE",
                "BOTTOM": "BOTTOM", "BOT": "BOTTOM", "UTILITY": "BOTTOM",
                "SUPPORT": "BOTTOM"}

WAVE_COLUMNS = ("match_id", "ts_ms", "lane", "team_id", "position", "velocity",
                "state", "size_diff", "cannon", "recommended", "confidence",
                "rationale")
LANE_COLUMNS = ("match_id", "ts_ms", "participant_id", "lane", "cs_diff",
                "gold_diff", "xp_diff", "level_diff", "hp_pct", "opp_hp_pct",
                "plates", "priority", "trade_window", "all_in", "recall_value",
                "should_recall", "confidence", "rationale")
RECALL_COLUMNS = ("match_id", "participant_id", "ts_ms", "gold_spent", "items",
                  "wave_state", "was_good", "reason")
OBJ_COLUMNS = ("match_id", "ts_ms", "objective", "subtype", "team_id",
               "contested", "setup_score", "prior_kills", "vision_edge",
               "next_spawn_ms", "rationale")
JUNGLE_COLUMNS = ("match_id", "participant_id", "ts_ms", "quadrant", "camp",
                  "clear_index", "gank_target", "predicted_next", "confidence")
ROTATION_COLUMNS = ("match_id", "participant_id", "ts_ms", "from_zone",
                    "to_zone", "purpose", "tempo_gain", "was_correct")
FIGHT_COLUMNS = ("match_id", "fight_id", "start_ms", "end_ms", "zone",
                 "team_a_count", "team_b_count", "kills_a", "kills_b",
                 "winner_team", "gold_swing", "trigger", "positioning")
DEATH_COLUMNS = ("match_id", "ts_ms", "victim_id", "killer_id", "x", "y",
                 "zone", "assisters", "outnumbered", "vision_nearby",
                 "hp_before", "cause", "cost_gold", "rationale")


class MatchData:
    """All rows for one match, indexed for the estimators."""

    def __init__(self, db: Database, match_id: str) -> None:
        self.match_id = match_id
        row = db.query("SELECT * FROM matches WHERE match_id=?", (match_id,))
        self.match = dict(row[0]) if row else {}
        self.patch = self.match.get("patch")

        self.participants: dict[int, dict[str, Any]] = {}
        for r in db.query("SELECT * FROM participants WHERE match_id=?", (match_id,)):
            d = dict(r)
            self.participants[int(d["participant_id"])] = d

        self.teams = {pid: int(p.get("team_id") or 0)
                      for pid, p in self.participants.items()}
        self.roles = {pid: (p.get("role") or "") for pid, p in self.participants.items()}

        # frames[ts][pid] -> row
        self.frames: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        for r in db.iter_query(
                "SELECT * FROM frames WHERE match_id=? ORDER BY ts_ms, participant_id",
                (match_id,)):
            d = dict(r)
            self.frames[int(d["ts_ms"])][int(d["participant_id"])] = d
        self.timestamps = sorted(self.frames)

        self.events: list[dict[str, Any]] = [
            dict(r) for r in db.iter_query(
                "SELECT * FROM events WHERE match_id=? ORDER BY ts_ms", (match_id,))]

    # -- convenience views ----------------------------------------------
    def positions_at(self, ts: int) -> dict[int, Point]:
        out: dict[int, Point] = {}
        for pid, f in self.frames.get(ts, {}).items():
            if f.get("x") is not None and f.get("y") is not None:
                out[pid] = (float(f["x"]), float(f["y"]))
        return out

    def state_at(self, ts: int, pid: int) -> ChampionState | None:
        f = self.frames.get(ts, {}).get(pid)
        if not f or f.get("x") is None:
            return None
        return ChampionState(
            participant_id=pid, team_id=self.teams.get(pid, 0),
            position=(float(f["x"]), float(f["y"])),
            level=int(f.get("level") or 1), xp=int(f.get("xp") or 0),
            total_gold=int(f.get("total_gold") or 0),
            current_gold=int(f.get("current_gold") or 0),
            cs=int(f.get("minions") or 0) + int(f.get("jungle_minions") or 0),
            health=int(f.get("health") or 0), health_max=int(f.get("health_max") or 1),
            ad=int(f.get("ad") or 0), ap=int(f.get("ap") or 0),
            armor=int(f.get("armor") or 0), mr=int(f.get("mr") or 0),
            ts_ms=ts)

    def events_of(self, *types: str) -> list[dict[str, Any]]:
        wanted = set(types)
        return [e for e in self.events if e.get("type") in wanted]

    def lane_of(self, pid: int) -> str | None:
        return ROLE_TO_LANE.get((self.roles.get(pid) or "").upper())

    def players_in_lane(self, lane: str, team: int) -> list[int]:
        return [pid for pid in self.participants
                if self.lane_of(pid) == lane and self.teams.get(pid) == team]

    def jungler_of(self, team: int) -> int | None:
        for pid, role in self.roles.items():
            if (role or "").upper() == "JUNGLE" and self.teams.get(pid) == team:
                return pid
        return None


def _opponent_of(md: MatchData, pid: int) -> int | None:
    """Direct lane opponent: same role, other team."""
    role = (md.roles.get(pid) or "").upper()
    team = md.teams.get(pid)
    for other, orole in md.roles.items():
        if other != pid and (orole or "").upper() == role and md.teams.get(other) != team:
            return other
    return None


def _item_costs(db: Database) -> dict[int, int]:
    return {int(r[0]): int(r[1] or 0)
            for r in db.query("SELECT item_id, total_gold FROM items")}


def process_match(db: Database, match_id: str, predictor: JunglePredictor,
                  item_costs: Mapping[int, int]) -> dict[str, int]:
    """Run every estimator over one match and write the feature rows."""
    md = MatchData(db, match_id)
    if not md.timestamps or not md.participants:
        return {}

    counts: dict[str, int] = defaultdict(int)
    patch = md.patch

    # ---- objective kills, needed by several estimators ------------------
    monster_kills = [
        (int(e["ts_ms"]), str(e.get("monster_type") or ""),
         int(e.get("team_id") or 0), e.get("monster_subtype"))
        for e in md.events_of("ELITE_MONSTER_KILL")]
    obj_kill_pairs = [(ts, name) for ts, name, _team, _sub in monster_kills if name]

    # ---- wards, for vision context --------------------------------------
    wards: list[tuple[int, int, Point]] = []
    for e in md.events_of("WARD_PLACED"):
        creator = e.get("participant_id") or e.get("killer_id")
        team = md.teams.get(int(creator)) if creator else None
        pos = md.positions_at(_nearest_ts(md.timestamps, int(e["ts_ms"]))).get(
            int(creator) if creator else -1)
        if team and pos:
            wards.append((int(e["ts_ms"]), team, pos))

    # ---- plates ----------------------------------------------------------
    plates: dict[tuple[int, str], int] = defaultdict(int)
    plate_events = [(int(e["ts_ms"]), e.get("lane"), int(e.get("team_id") or 0))
                    for e in md.events_of("TURRET_PLATE_DESTROYED")]

    # ---- turret status ---------------------------------------------------
    turret_down: dict[tuple[int, str], int] = {}
    for e in md.events_of("BUILDING_KILL"):
        lane = _norm_lane(e.get("lane"))
        team = int(e.get("team_id") or 0)     # team that *owned* the building
        if lane and team and e.get("tower_type") == "OUTER_TURRET":
            turret_down[(team, lane)] = int(e["ts_ms"])

    # ---- waves -----------------------------------------------------------
    wave_rows: list[tuple] = []
    wave_by_key: dict[tuple[int, str, int], Any] = {}
    estimators = {(lane, team): WaveEstimator(lane, team)
                  for lane in ("TOP", "MIDDLE", "BOTTOM") for team in (BLUE, RED)}

    for ts in md.timestamps:
        pos = md.positions_at(ts)
        for lane in ("TOP", "MIDDLE", "BOTTOM"):
            for team in (BLUE, RED):
                enemy = RED if team == BLUE else BLUE
                own_ids = md.players_in_lane(lane, team)
                enemy_ids = md.players_in_lane(lane, enemy)
                own_cs = sum(int(md.frames[ts].get(p, {}).get("minions") or 0)
                             for p in own_ids)
                enemy_cs = sum(int(md.frames[ts].get(p, {}).get("minions") or 0)
                               for p in enemy_ids)
                own_plates = sum(1 for t, l, tm in plate_events
                                 if t <= ts and _norm_lane(l) == lane and tm == team)
                enemy_plates = sum(1 for t, l, tm in plate_events
                                   if t <= ts and _norm_lane(l) == lane and tm == enemy)
                fi = LaneFrameInput(
                    ts_ms=ts,
                    own_positions=[pos[p] for p in own_ids if p in pos],
                    enemy_positions=[pos[p] for p in enemy_ids if p in pos],
                    own_cs=own_cs, enemy_cs=enemy_cs,
                    own_plates=own_plates, enemy_plates=enemy_plates,
                    own_turret_alive=(team, lane) not in turret_down
                    or turret_down[(team, lane)] > ts,
                    enemy_turret_alive=(enemy, lane) not in turret_down
                    or turret_down[(enemy, lane)] > ts)
                obs = estimators[(lane, team)].update(fi)

                # Decision context for the recommendation.
                secs_to_obj, obj_name = obj.seconds_to_next_objective(
                    ts, obj_kill_pairs, patch)
                own_state = md.state_at(ts, own_ids[0]) if own_ids else None
                enemy_state = md.state_at(ts, enemy_ids[0]) if enemy_ids else None
                jungler = md.jungler_of(enemy)
                jd = 99999.0
                if jungler is not None and jungler in pos and own_state is not None:
                    jd = dist(pos[jungler], own_state.position)
                ctx = DecisionContext(
                    hp_pct=own_state.hp_pct if own_state else 1.0,
                    enemy_hp_pct=enemy_state.hp_pct if enemy_state else 1.0,
                    gold_available=own_state.current_gold if own_state else 0,
                    next_item_cost=1300,
                    level=own_state.level if own_state else 1,
                    enemy_level=enemy_state.level if enemy_state else 1,
                    enemy_laner_alive=enemy_state is not None,
                    enemy_laner_in_lane=bool(fi.enemy_positions),
                    enemy_jungler_distance=jd,
                    seconds_to_objective=secs_to_obj,
                    objective_name=obj_name,
                    plates_remaining=max(0, 5 - enemy_plates),
                    game_time_s=ts / 1000.0)
                obs = recommend(obs, ctx)
                wave_by_key[(ts, lane, team)] = obs
                # The reasoning is what the dataset stage turns into prose,
                # so it is persisted rather than recomputed.
                wave_rows.append((
                    match_id, ts, lane, team, obs.position, obs.velocity,
                    obs.state, obs.size_diff, int(obs.cannon), obs.recommended,
                    obs.confidence, json.dumps(obs.reasons + obs.factors)))

    counts["wave_states"] = db.insert_many("wave_states", WAVE_COLUMNS, wave_rows)

    # ---- lane states ------------------------------------------------------
    lane_rows: list[tuple] = []
    for ts in md.timestamps:
        pos = md.positions_at(ts)
        for pid in md.participants:
            role = (md.roles.get(pid) or "").upper()
            if role == "JUNGLE":
                continue
            own = md.state_at(ts, pid)
            if own is None:
                continue
            opp_id = _opponent_of(md, pid)
            opp = md.state_at(ts, opp_id) if opp_id else None
            lane = md.lane_of(pid)
            team = md.teams.get(pid, BLUE)
            enemy = RED if team == BLUE else BLUE
            wave = wave_by_key.get((ts, lane, team)) if lane else None
            secs_to_obj, _ = obj.seconds_to_next_objective(ts, obj_kill_pairs, patch)
            own_plates = sum(1 for t, l, tm in plate_events
                             if t <= ts and _norm_lane(l) == lane and tm == team)
            ls = build_lane_state(
                ts, own, opp, lane, wave.position if wave else 0.0,
                [p for pid2, p in pos.items() if md.teams.get(pid2) == team],
                [p for pid2, p in pos.items() if md.teams.get(pid2) == enemy],
                own_plates, 1300, secs_to_obj)
            lane_rows.append((
                match_id, ts, pid, ls.lane, ls.cs_diff, ls.gold_diff, ls.xp_diff,
                ls.level_diff, ls.hp_pct, ls.opp_hp_pct, ls.plates, ls.priority,
                ls.trade_window, int(ls.all_in), ls.recall_value,
                int(ls.should_recall), ls.confidence, json.dumps(ls.reasons)))
    counts["lane_states"] = db.insert_many("lane_states", LANE_COLUMNS, lane_rows)

    # ---- recalls ----------------------------------------------------------
    purchases_by_pid: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for e in md.events_of("ITEM_PURCHASED"):
        pid = e.get("participant_id")
        if pid is not None:
            purchases_by_pid[int(pid)].append(
                (int(e["ts_ms"]), int(pid), int(e.get("item_id") or 0)))

    deaths_by_pid: dict[int, list[int]] = defaultdict(list)
    for e in md.events_of("CHAMPION_KILL"):
        if e.get("victim_id") is not None:
            deaths_by_pid[int(e["victim_id"])].append(int(e["ts_ms"]))

    recall_rows: list[tuple] = []
    for pid in md.participants:
        team = md.teams.get(pid, BLUE)
        states = [s for s in (md.state_at(ts, pid) for ts in md.timestamps)
                  if s is not None]
        for ts_ms, _, items in detect_recalls(
                states, team, purchases_by_pid.get(pid, [])):
            gold = sum(item_costs.get(i, 0) for i in items)
            lane = md.lane_of(pid)
            wave = wave_by_key.get((_nearest_ts(md.timestamps, ts_ms), lane, team))
            died_soon = any(0 <= d - ts_ms <= 90_000 for d in deaths_by_pid.get(pid, []))
            good, reason = recall_was_good(None, None, len(items), died_soon)
            recall_rows.append((match_id, pid, ts_ms, gold, json.dumps(items),
                                wave.state if wave else None, int(good), reason))
    counts["recalls"] = db.insert_many("recalls", RECALL_COLUMNS, recall_rows)

    # ---- objectives -------------------------------------------------------
    obj_rows: list[tuple] = []
    champ_kills = [(int(e["ts_ms"]), int(e.get("killer_id") or 0),
                    int(e.get("victim_id") or 0),
                    (float(e.get("x") or 0), float(e.get("y") or 0)))
                   for e in md.events_of("CHAMPION_KILL")]
    for ts, name, team, subtype in monster_kills:
        if not name or team not in (BLUE, RED):
            continue
        before_ts = _nearest_ts(md.timestamps, max(0, ts - obj.SETUP_WINDOW_MS))
        pos_before = md.positions_at(before_ts)
        by_team: dict[int, list[Point]] = defaultdict(list)
        for pid, p in pos_before.items():
            by_team[md.teams.get(pid, 0)].append(p)
        pit = obj.pit_of(name)
        wards_near = defaultdict(int)
        for placed, wteam, wpos in wards:
            if ts - obj.SETUP_WINDOW_MS <= placed <= ts and dist(wpos, pit) < 3000:
                wards_near[wteam] += 1
        kills_before = [(k[0], md.teams.get(k[1], 0)) for k in champ_kills
                        if ts - obj.SETUP_WINDOW_MS <= k[0] <= ts]
        take = obj.evaluate_take(ts, name, subtype, team, by_team, wards_near,
                                 kills_before, patch)
        obj_rows.append((match_id, ts, name, subtype, team, int(take.contested),
                         take.setup_score, take.prior_kills, take.vision_edge,
                         take.next_spawn_ms, json.dumps(take.reasons)))
    counts["objective_events"] = db.insert_many("objective_events", OBJ_COLUMNS, obj_rows)

    # ---- jungle -----------------------------------------------------------
    jungle_rows: list[tuple] = []
    for team in (BLUE, RED):
        jid = md.jungler_of(team)
        if jid is None:
            continue
        trace = []
        for ts in md.timestamps:
            f = md.frames[ts].get(jid)
            if not f or f.get("x") is None:
                continue
            trace.append((ts, (float(f["x"]), float(f["y"])),
                          int(f.get("jungle_minions") or 0)))
        clears = reconstruct_path(trace, team)
        for c in clears:
            c.participant_id = jid
        predictor.observe_sequence(clears)
        ganks = detect_ganks([(t, p) for t, p, _ in trace], champ_kills, jid)
        gank_at = {g.ts_ms: g for g in ganks}
        clear_at = {c.ts_ms: c for c in clears}

        last_camp: str | None = None
        for ts, p, _kills in trace:
            camp = clear_at[ts].camp if ts in clear_at else None
            if camp:
                last_camp = camp
            predicted, conf = (None, 0.0)
            if last_camp:
                ranked = predictor.predict(last_camp, ts / 1000.0, top_k=1)
                if ranked:
                    predicted, conf = ranked[0]
            g = gank_at.get(ts)
            jungle_rows.append((
                match_id, jid, ts, quadrant_of(p), camp,
                clear_at[ts].clear_index if ts in clear_at else 0,
                g.lane if g else None, predicted, round(conf, 3)))
    counts["jungle_paths"] = db.insert_many("jungle_paths", JUNGLE_COLUMNS, jungle_rows)

    # ---- fights and deaths -------------------------------------------------
    bounties = {int(e["victim_id"]): int(e.get("bounty") or 300)
                for e in md.events_of("CHAMPION_KILL") if e.get("victim_id")}
    positions_all = {ts: md.positions_at(ts) for ts in md.timestamps}
    fights = cluster_fights(champ_kills, md.teams, positions_all, bounties)

    fight_rows: list[tuple] = []
    for f in fights:
        frame_ts = _nearest_ts(md.timestamps, f.start_ms)
        notes = positioning_notes(f, positions_all.get(frame_ts, {}), md.teams,
                                  md.roles,
                                  [k[2] for k in champ_kills
                                   if f.start_ms <= k[0] <= f.end_ms])
        fight_rows.append((match_id, f.fight_id, f.start_ms, f.end_ms, f.zone,
                           f.team_a_count, f.team_b_count, f.kills_a, f.kills_b,
                           f.winner_team, f.gold_swing, f.trigger,
                           json.dumps(notes)))
    counts["fights"] = db.insert_many("fights", FIGHT_COLUMNS, fight_rows)

    death_rows: list[tuple] = []
    for e in md.events_of("CHAMPION_KILL"):
        victim = e.get("victim_id")
        if victim is None:
            continue
        victim = int(victim)
        ts = int(e["ts_ms"])
        frame_ts = _nearest_ts(md.timestamps, ts)
        payload = json.loads(e.get("payload") or "{}")
        assisters = len(payload.get("assistingParticipantIds") or [])
        f = md.frames.get(frame_ts, {}).get(victim, {})
        in_fight = any(f2.start_ms <= ts <= f2.end_ms for f2 in fights)
        da = analyse_death(
            ts_ms=ts, victim_id=victim,
            killer_id=int(e["killer_id"]) if e.get("killer_id") else None,
            position=(float(e.get("x") or 0), float(e.get("y") or 0)),
            victim_team=md.teams.get(victim, BLUE), assisters=assisters,
            positions=positions_all.get(frame_ts, {}), teams=md.teams,
            wards=wards, hp_before=int(f.get("health") or 0),
            hp_max=int(f.get("health_max") or 1),
            bounty=int(e.get("bounty") or 0), in_fight=in_fight)
        death_rows.append((match_id, ts, victim, da.killer_id, int(da.position[0]),
                           int(da.position[1]), da.zone, da.assisters,
                           da.outnumbered, int(da.vision_nearby), da.hp_before,
                           da.cause, da.cost_gold, json.dumps(da.reasons)))
    counts["deaths"] = db.insert_many("deaths", DEATH_COLUMNS, death_rows)

    # ---- rotations ---------------------------------------------------------
    rot_rows: list[tuple] = []
    obj_for_rot = [(ts, name, team) for ts, name, team, _ in monster_kills]
    for pid in md.participants:
        team = md.teams.get(pid, BLUE)
        trace = [(ts, positions_all[ts][pid]) for ts in md.timestamps
                 if pid in positions_all.get(ts, {})]
        for r in detect_rotations(trace, pid, obj_for_rot, fights, team):
            rot_rows.append((match_id, pid, r.ts_ms, r.from_zone, r.to_zone,
                             r.purpose, r.tempo_gain,
                             None if r.was_correct is None else int(r.was_correct)))
    counts["rotations"] = db.insert_many("rotations", ROTATION_COLUMNS, rot_rows)

    return dict(counts)


def _nearest_ts(timestamps: Sequence[int], ts: int) -> int:
    """Closest frame timestamp at or before ``ts`` (first frame if none)."""
    best = timestamps[0] if timestamps else 0
    for t in timestamps:
        if t <= ts:
            best = t
        else:
            break
    return best


def _norm_lane(lane: str | None) -> str | None:
    if not lane:
        return None
    return {"TOP_LANE": "TOP", "MID_LANE": "MIDDLE", "BOT_LANE": "BOTTOM",
            "TOP": "TOP", "MIDDLE": "MIDDLE", "BOTTOM": "BOTTOM"}.get(
        str(lane).upper())


def build_features(db: Database, limit: int | None = None, force: bool = False,
                   progress: Callable[[int, str], None] | None = None
                   ) -> dict[str, int]:
    """Run the feature pass over every match that has a timeline.

    Matches that already have wave states are skipped unless ``force``.
    """
    done_ids: set[str] = set()
    if not force:
        done_ids = {r[0] for r in db.query(
            "SELECT DISTINCT match_id FROM wave_states")}

    candidates = [r[0] for r in db.query(
        "SELECT match_id FROM matches WHERE has_timeline=1 ORDER BY match_id")]
    todo = [m for m in candidates if m not in done_ids]
    if limit:
        todo = todo[:limit]

    # Refine turret positions from the corpus before estimating anything.
    events = [json.loads(r[0]) for r in db.query(
        "SELECT payload FROM events WHERE type IN "
        "('BUILDING_KILL','TURRET_PLATE_DESTROYED') LIMIT 20000")]
    if events:
        refined = refine_turrets_from_events(events)
        moved = apply_refined_turrets(refined)
        if moved:
            log.info("refined %d turret positions from observed events", moved)

    predictor = JunglePredictor.from_json(db.get_meta("jungle_model"))
    item_costs = _item_costs(db)

    totals: dict[str, int] = defaultdict(int)
    for i, match_id in enumerate(todo, 1):
        try:
            counts = process_match(db, match_id, predictor, item_costs)
        except Exception as exc:
            log.warning("feature pass failed for %s: %s", match_id, exc)
            totals["failed"] += 1
            continue
        for k, v in counts.items():
            totals[k] += v
        if progress is not None:
            progress(i, match_id)

    db.set_meta("jungle_model", predictor.to_json())
    db.set_meta("features_built_at", str(time.time()))
    totals["matches"] = len(todo)
    return dict(totals)
