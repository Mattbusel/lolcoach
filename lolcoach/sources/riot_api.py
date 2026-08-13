"""Riot Games API crawler: high-elo ladder -> match history -> timelines.

This is the connector that produces the corpus everything else in Stage 2
depends on. A match *timeline* is what makes wave states, jungle paths and
recall decisions inferable at all: it carries a participant frame every 60
seconds (position, gold, xp, cs, health, and full ``championStats``) plus the
complete discrete event stream (kills, wards, buildings, elite monsters, item
purchases, level-ups).

Crawl shape
-----------
1. **Ladder.** ``league-v4`` apex queues (Challenger/Grandmaster/Master) give
   the top of each configured platform. That is the "challenger replays"
   requirement: solo-queue games played by the highest-rated accounts.
2. **Match ids.** ``match-v5`` history per player, filtered to ranked solo.
3. **Match + timeline.** Both are stored gzipped under
   ``raw/riot/<region>/<match_id>.json.gz`` and ``.timeline.json.gz``.

Resumability
------------
The crawler is interrupt-safe at every level. Discovered puuids and the set of
already-fetched match ids live in the database, so a restart re-plans from
what is on disk rather than re-downloading. Interrupting with Ctrl-C leaves a
consistent state.

Rate limits
-----------
Riot enforces two simultaneous application windows plus per-method limits, and
answers ``429`` with ``Retry-After``. :class:`~lolcoach.http.RateLimiter`
models both windows; the client additionally parks every worker for the full
``Retry-After`` when a 429 arrives, because the limit is shared across the key
rather than per-connection.
"""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx

from ..http import HttpClient, RateLimiter
from ..logging_utils import get_logger
from ..storage import DownloadRecord
from .base import RIOT_TOS, Source, SourceContext, SourceResult, register

log = get_logger("sources.riot")

APEX_QUEUES = {
    "CHALLENGER": "challengerleagues",
    "GRANDMASTER": "grandmasterleagues",
    "MASTER": "masterleagues",
}


class RiotClient:
    """Typed wrapper over the Riot REST API with per-key rate limiting.

    Parameters
    ----------
    api_key:
        A development or production key. Development keys expire every 24
        hours; an expired key surfaces as ``403`` and is reported clearly.
    """

    def __init__(self, api_key: str, cfg: Any, cache_dir: Path | None = None,
                 interactive: bool = False) -> None:
        """Create a client.

        ``interactive=True`` is for calls a person is waiting on, such as
        onboarding lookups. The background crawler retries a transient failure
        four times with exponential backoff, which is right for an unattended
        run over thousands of matches and badly wrong for a button: probing
        seven regional routes under that policy can take minutes, during which
        the UI looks frozen. Interactive clients fail fast so the user gets an
        answer and can try again.
        """
        self.api_key = api_key
        self.cfg = cfg
        # Flipped to True only if header auth is rejected once.
        self._use_query_key = False
        limiter = RateLimiter([
            (cfg.riot.rate_per_second, 1.0),
            (cfg.riot.rate_per_two_minutes, 120.0),
        ])
        self.http = HttpClient(
            user_agent=cfg.user_agent,
            cache_dir=cache_dir,
            timeout=8.0 if interactive else cfg.http_timeout,
            retries=1 if interactive else cfg.http_retries,
            limiter=limiter,
            # Documented auth method. Keeps the key out of URLs, and therefore
            # out of logs, proxies and crash reports.
            headers={"X-Riot-Token": api_key},
        )

    # -- low level ------------------------------------------------------
    def _get(self, host: str, path: str, **params: Any) -> Any:
        url = f"https://{host}.api.riotgames.com{path}"
        # Riot accepts the key either as the X-Riot-Token header or as an
        # ``api_key`` query parameter. The header is used by default because a
        # key in a URL leaks anywhere URLs are recorded: httpx logs the full
        # request line at INFO, and proxies and crash reports keep query
        # strings. Relying on a logger staying at WARNING to protect a
        # credential is not a control.
        #
        # The query form is still needed on machines whose network filters
        # strip custom headers, so it is used automatically after the header
        # form is rejected once, rather than on every request.
        request_params = dict(params)
        if self._use_query_key:
            request_params["api_key"] = self.api_key
        resp = self.http.get(url, params=request_params, raise_for_status=False)

        if resp.status_code == 401 and not self._use_query_key:
            # The header may have been stripped in transit. Retry once with the
            # query form and remember the answer for this client's lifetime.
            self._use_query_key = True
            log.info("Riot rejected the header auth; retrying with the query "
                     "parameter form (a local filter may be stripping headers)")
            return self._get(host, path, **params)
        if resp.status_code == 401:
            raise PermissionError(
                "Riot API returned 401: this key is not recognized. Generate a new key in the Riot Developer Portal "
                "and run `lolcoach init` again.")
        if resp.status_code == 403:
            raise PermissionError(
                "Riot API returned 403. A development key expires after 24 hours; "
                "regenerate it at https://developer.riotgames.com and update RIOT_API_KEY.")
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} {resp.text[:200]}", request=resp.request, response=resp)
        return json.loads(resp.content.decode("utf-8"))

    def validate(self) -> bool:
        """Cheap call that proves the key works before a long crawl."""
        # Validation is an interactive UI action. Do not inherit the crawler's
        # long timeout/retry policy or a transient Riot edge failure looks like
        # a frozen button for minutes.
        previous_retries = self.http.retries
        previous_timeout = self.http._client.timeout
        # Two quick retries, not zero. Riot's edge returns transient 503s often
        # enough that a single-shot check rejects working keys, which reads to
        # the user as "the app is broken" at the very first step. Still short
        # enough that a genuinely bad key fails fast.
        self.http.retries = 2
        self.http._client.timeout = httpx.Timeout(8.0, connect=5.0)
        try:
            self._get("na1", "/lol/platform/v3/champion-rotations")
            return True
        except PermissionError:
            raise
        except httpx.HTTPStatusError as exc:
            # A server-side failure says nothing about the key; make that
            # distinction visible rather than blaming the user's key.
            status = exc.response.status_code if exc.response is not None else 0
            if status >= 500:
                raise RuntimeError(
                    "Riot's API is temporarily unavailable (HTTP "
                    f"{status}). Your key was not saved; try again in a moment."
                ) from exc
            log.warning("Riot key validation call failed: %s", exc)
            return False
        except Exception as exc:
            log.warning("Riot key validation call failed: %s", exc)
            return False
        finally:
            self.http.retries = previous_retries
            self.http._client.timeout = previous_timeout

    # -- endpoints ------------------------------------------------------
    def apex_league(self, platform: str, tier: str, queue: str) -> dict[str, Any] | None:
        """Challenger/Grandmaster/Master league for a platform."""
        endpoint = APEX_QUEUES[tier]
        return self._get(platform, f"/lol/league/v4/{endpoint}/by-queue/{queue}")

    def league_entries(self, platform: str, tier: str, division: str,
                       queue: str, page: int = 1) -> list[dict[str, Any]]:
        """Paged ladder entries for the non-apex tiers."""
        out = self._get(platform, f"/lol/league/v4/entries/{queue}/{tier}/{division}",
                        page=page)
        return out or []

    def summoner_by_id(self, platform: str, summoner_id: str) -> dict[str, Any] | None:
        return self._get(platform, f"/lol/summoner/v4/summoners/{summoner_id}")

    def account_by_riot_id(self, region: str, game_name: str,
                           tag_line: str) -> dict[str, Any] | None:
        """Resolve an opted-in Riot ID through the regional account-v1 route."""
        from urllib.parse import quote

        return self._get(
            region,
            f"/riot/account/v1/accounts/by-riot-id/{quote(game_name, safe='')}/"
            f"{quote(tag_line, safe='')}")

    def match_ids(self, region: str, puuid: str, count: int = 20,
                  queue: int | None = None, start: int = 0,
                  match_type: str = "ranked") -> list[str]:
        params: dict[str, Any] = {"start": start, "count": min(count, 100)}
        if queue is not None:
            params["queue"] = queue
        if match_type:
            params["type"] = match_type
        return self._get(region, f"/lol/match/v5/matches/by-puuid/{puuid}/ids", **params) or []

    def match(self, region: str, match_id: str) -> dict[str, Any] | None:
        return self._get(region, f"/lol/match/v5/matches/{match_id}")

    def timeline(self, region: str, match_id: str) -> dict[str, Any] | None:
        return self._get(region, f"/lol/match/v5/matches/{match_id}/timeline")

    def close(self) -> None:
        self.http.close()


def write_gz(path: Path, obj: Any) -> int:
    """Write JSON gzipped; timelines are ~10x smaller this way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=6) as fh:
        fh.write(blob)
    return path.stat().st_size


def read_gz(path: Path) -> Any:
    """Read a gzipped JSON payload written by :func:`write_gz`."""
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


@register
class RiotApiSource(Source):
    """Crawl high-elo solo-queue matches and their timelines."""

    name = "riot"
    description = "Riot API: challenger/GM/master ladders, matches and timelines"
    license = RIOT_TOS
    requires_secret = ("RIOT_API_KEY", "RIOT_KEY", "RIOT_TOKEN")

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        cfg = ctx.cfg
        key = cfg.secret(*self.requires_secret)
        if not key:
            result.skipped = "no RIOT_API_KEY"
            return

        client = RiotClient(key, cfg, cache_dir=None)
        try:
            client.validate()
        except PermissionError as exc:
            result.ok = False
            result.error = str(exc)
            return

        target = ctx.limit or cfg.riot.max_matches
        puuids = self._collect_ladder(client, ctx, result)
        if not puuids:
            result.note("ladder returned no players; nothing to crawl")
            return

        planned = self._plan_matches(client, ctx, result, puuids, target)
        result.note(f"{len(planned)} match ids planned across "
                    f"{len({r for _, r in planned})} regions")
        self._fetch_matches(client, ctx, result, planned)
        client.close()

    # -- step 1: ladder -------------------------------------------------
    def _collect_ladder(self, client: RiotClient, ctx: SourceContext,
                        result: SourceResult) -> list[tuple[str, str]]:
        """Return ``(platform, puuid)`` pairs from the configured tiers.

        League entries have carried ``puuid`` directly since Riot's 2025
        identity migration; the summoner-v4 fallback covers older responses
        that still only expose ``summonerId``.
        """
        cfg = ctx.cfg
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        per_platform = cfg.riot.players_per_platform

        for platform in cfg.riot.platforms:
            got = 0
            for tier in cfg.riot.tiers:
                if got >= per_platform:
                    break
                try:
                    if tier in APEX_QUEUES:
                        league = client.apex_league(platform, tier, cfg.riot.queue)
                        entries = (league or {}).get("entries", [])
                    else:
                        entries = []
                        for division in cfg.riot.divisions:
                            entries.extend(client.league_entries(
                                platform, tier, division, cfg.riot.queue))
                except PermissionError:
                    raise
                except Exception as exc:
                    result.note(f"{platform}/{tier} ladder unavailable: {exc}")
                    continue

                entries.sort(key=lambda e: -int(e.get("leaguePoints", 0) or 0))
                for entry in entries:
                    if got >= per_platform:
                        break
                    puuid = entry.get("puuid")
                    if not puuid and entry.get("summonerId"):
                        try:
                            summ = client.summoner_by_id(platform, entry["summonerId"])
                            puuid = (summ or {}).get("puuid")
                        except Exception:
                            puuid = None
                    if not puuid or puuid in seen:
                        continue
                    seen.add(puuid)
                    found.append((platform, puuid))
                    got += 1
                result.note(f"{platform}/{tier}: {got} players so far")

        ctx.db.set_meta("riot_ladder_players", len(found))
        return found

    # -- step 2: match ids ----------------------------------------------
    def _plan_matches(self, client: RiotClient, ctx: SourceContext,
                      result: SourceResult, players: list[tuple[str, str]],
                      target: int) -> list[tuple[str, str]]:
        """Build the ``(match_id, region)`` work list, skipping known matches."""
        cfg = ctx.cfg
        known = {row[0] for row in ctx.db.query(
            "SELECT match_id FROM matches WHERE source='riot'")}
        planned: dict[str, str] = {}

        for platform, puuid in players:
            if len(planned) >= target:
                break
            region = cfg.riot.region_of_platform.get(platform, "americas")
            try:
                ids = client.match_ids(region, puuid,
                                       count=cfg.riot.matches_per_player,
                                       queue=cfg.riot.queue_id)
            except PermissionError:
                raise
            except Exception as exc:
                log.debug("match ids failed for %s: %s", puuid[:8], exc)
                continue
            for mid in ids:
                if mid in known or mid in planned:
                    continue
                planned[mid] = region
                if len(planned) >= target:
                    break
        return list(planned.items())

    # -- step 3: match + timeline ---------------------------------------
    def _fetch_matches(self, client: RiotClient, ctx: SourceContext,
                       result: SourceResult, planned: list[tuple[str, str]]) -> None:
        """Download match and timeline pairs concurrently.

        Concurrency is safe because the shared rate limiter admits requests
        globally; workers mostly sit blocked in ``acquire`` and the wall-clock
        throughput is set by the key's limit, not by thread count.
        """
        out_root = ctx.out_dir
        done = 0
        started = time.time()

        def one(match_id: str, region: str) -> tuple[str, int, bool]:
            mdir = out_root / region
            mpath = mdir / f"{match_id}.json.gz"
            tpath = mdir / f"{match_id}.timeline.json.gz"
            written = 0
            have_timeline = False

            if not mpath.exists() or ctx.force:
                match = client.match(region, match_id)
                if match is None:
                    return match_id, 0, False
                written += write_gz(mpath, match)
            if not tpath.exists() or ctx.force:
                tl = client.timeline(region, match_id)
                if tl is not None:
                    written += write_gz(tpath, tl)
                    have_timeline = True
            else:
                have_timeline = True
            return match_id, written, have_timeline

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(one, mid, reg): (mid, reg) for mid, reg in planned}
            for fut in as_completed(futures):
                mid, region = futures[fut]
                try:
                    match_id, written, have_tl = fut.result()
                except PermissionError as exc:
                    result.ok = False
                    result.error = str(exc)
                    for f in futures:
                        f.cancel()
                    break
                except Exception as exc:
                    log.debug("match %s failed: %s", mid, exc)
                    ctx.manifest.mark_failed(
                        f"riot://{region}/{mid}", out_root / region / f"{mid}.json.gz",
                        self.name, str(exc))
                    continue
                if written:
                    ctx.manifest.record(DownloadRecord(
                        url=f"riot://{region}/{match_id}",
                        path=str(out_root / region / f"{match_id}.json.gz"),
                        source=self.name, kind="match",
                        bytes=written, **self.license.as_dict()))
                    result.files += 1 + int(have_tl)
                    result.bytes += written
                done += 1
                if done % 200 == 0:
                    rate = done / max(1e-6, time.time() - started)
                    result.note(f"{done}/{len(planned)} matches ({rate:.1f}/s)")

        result.records = done
        result.note(f"fetched {done} matches into {out_root}")


#: Regional routing values that serve match-v5. A PUUID resolves on exactly
#: one of them, and which one is not derivable from the account lookup, so it
#: is detected once and cached.
MATCH_ROUTES = ("americas", "europe", "asia", "sea")


def detect_region(client: RiotClient, puuid: str) -> str | None:
    """Find which regional route holds this player's match history.

    Riot's account service resolves a Riot ID from any regional route, but
    match history lives on exactly one. Asking each route for a single match
    id is cheap and unambiguous: only the correct one returns anything.
    """
    for region in MATCH_ROUTES:
        try:
            if client.match_ids(region, puuid, count=1, match_type=""):
                return region
        except PermissionError:
            raise
        except Exception as exc:
            log.debug("region probe %s failed: %s", region, exc)
    return None


def fetch_player_matches(cfg: Any, puuid: str, region: str, out_dir: Path,
                         limit: int = 25, queue: int | None = None,
                         known: set[str] | None = None,
                         progress: Any | None = None) -> dict[str, Any]:
    """Download one player's own recent matches and timelines.

    This is the path behind "collect my games". It is deliberately separate
    from :class:`RiotApiSource`, which crawls the high-elo ladder to build a
    *training* corpus: that one collects strangers' games on purpose, and
    running it for a player who asked for their own history is the wrong
    answer to the wrong question.

    Already-downloaded matches are skipped, so syncing after each play session
    only costs the new games.
    """
    key = cfg.secret("RIOT_API_KEY", "RIOT_KEY", "RIOT_TOKEN")
    if not key:
        raise RuntimeError("No Riot API key is configured.")

    client = RiotClient(key, cfg)
    known = known or set()
    summary = {"requested": limit, "found": 0, "downloaded": 0,
               "skipped": 0, "failed": 0, "region": region}
    try:
        ids: list[str] = []
        # match-v5 pages at 100 ids per call.
        while len(ids) < limit:
            page = client.match_ids(region, puuid, count=min(100, limit - len(ids)),
                                    queue=queue, start=len(ids))
            if not page:
                break
            ids.extend(page)
        ids = ids[:limit]
        summary["found"] = len(ids)

        for position, match_id in enumerate(ids, 1):
            if match_id in known:
                summary["skipped"] += 1
                if progress:
                    progress(position, len(ids), summary)
                continue
            match_path = out_dir / region / f"{match_id}.json.gz"
            timeline_path = out_dir / region / f"{match_id}.timeline.json.gz"
            try:
                if not match_path.exists():
                    match = client.match(region, match_id)
                    if match is None:
                        summary["failed"] += 1
                        continue
                    write_gz(match_path, match)
                if not timeline_path.exists():
                    timeline = client.timeline(region, match_id)
                    if timeline is not None:
                        write_gz(timeline_path, timeline)
                summary["downloaded"] += 1
            except PermissionError:
                raise
            except Exception as exc:
                log.warning("could not fetch %s: %s", match_id, exc)
                summary["failed"] += 1
            if progress:
                progress(position, len(ids), summary)
    finally:
        client.close()
    return summary


def iter_local_timelines(raw_riot_dir: Path) -> Iterator[tuple[str, Path, Path]]:
    """Yield ``(match_id, match_path, timeline_path)`` for locally stored games.

    Only pairs where both files exist are returned, since Stage 2 needs the
    match record (for champions and roles) and the timeline (for frames).
    """
    if not raw_riot_dir.exists():
        return
    for tl in sorted(raw_riot_dir.rglob("*.timeline.json.gz")):
        match_id = tl.name.replace(".timeline.json.gz", "")
        mpath = tl.with_name(f"{match_id}.json.gz")
        if mpath.exists():
            yield match_id, mpath, tl


def summarise_local(raw_riot_dir: Path) -> dict[str, int]:
    """Count downloaded matches/timelines without opening them."""
    if not raw_riot_dir.exists():
        return {"matches": 0, "timelines": 0}
    matches = sum(1 for _ in raw_riot_dir.rglob("*.json.gz")
                  if not _.name.endswith(".timeline.json.gz"))
    timelines = sum(1 for _ in raw_riot_dir.rglob("*.timeline.json.gz"))
    return {"matches": matches, "timelines": timelines}
