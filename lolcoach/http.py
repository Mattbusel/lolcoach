"""HTTP layer shared by every Stage-1 connector.

Provides three things the connectors would otherwise each reinvent:

``RateLimiter``
    A multi-window token bucket. Riot enforces *two* simultaneous windows
    (per second and per two minutes) and returns ``429`` with a
    ``Retry-After`` header when either is breached, so the limiter supports an
    arbitrary list of ``(count, seconds)`` windows plus an externally
    triggered penalty box.

``HttpClient``
    A retrying wrapper over :mod:`httpx` with exponential backoff and jitter,
    honouring ``Retry-After``, plus an on-disk response cache with
    ``ETag``/``Last-Modified`` revalidation. Revalidation matters because the
    corpus is refreshed every patch: unchanged pages cost one conditional
    request instead of a full download.

``download_file``
    Streams a large file to disk, resuming partial transfers via HTTP range
    requests and returning the SHA-256 of the finished file.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from .logging_utils import get_logger

log = get_logger("http")

#: Status codes worth retrying: transient server faults and rate limits.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Headers that describe the *transfer* rather than the entity. They are
#: dropped before caching because the cached body is already decoded.
_VOLATILE_HEADERS = frozenset({"content-encoding", "content-length",
                               "transfer-encoding", "connection"})


class RateLimiter:
    """Thread-safe token bucket over one or more sliding windows.

    Parameters
    ----------
    windows:
        Sequence of ``(max_requests, window_seconds)`` pairs. Every window
        must have capacity before a request is allowed.

    Notes
    -----
    ``penalise`` is called when a server replies ``429``; it blocks all
    callers for the requested cool-down, which is the only correct response
    to a shared limit being exhausted by a sibling process.
    """

    def __init__(self, windows: Sequence[tuple[float, float]]) -> None:
        if not windows:
            windows = [(float("inf"), 1.0)]
        self._windows = [(float(n), float(s)) for n, s in windows]
        self._hits: list[deque[float]] = [deque() for _ in self._windows]
        self._lock = threading.Lock()
        self._blocked_until = 0.0

    def acquire(self) -> None:
        """Block until a request slot is free in every window."""
        while True:
            with self._lock:
                now = time.monotonic()
                wait = max(0.0, self._blocked_until - now)
                if wait <= 0:
                    for (limit, span), hits in zip(self._windows, self._hits):
                        while hits and now - hits[0] >= span:
                            hits.popleft()
                        if len(hits) >= limit:
                            wait = max(wait, span - (now - hits[0]) + 0.001)
                    if wait <= 0:
                        for hits in self._hits:
                            hits.append(now)
                        return
            time.sleep(min(wait, 5.0))

    def penalise(self, seconds: float) -> None:
        """Block every caller for ``seconds`` (used on ``429``)."""
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)


@dataclass
class CachedResponse:
    """Minimal response record persisted alongside a cached body."""

    url: str
    status: int
    headers: dict[str, str]
    fetched_at: float


class HttpClient:
    """Retrying HTTP client with an optional revalidating disk cache.

    The client is safe to share across threads: :mod:`httpx` clients are
    thread-safe and the rate limiter serialises admission.
    """

    def __init__(
        self,
        user_agent: str = "lolcoach/1.0",
        cache_dir: Path | None = None,
        timeout: float = 60.0,
        retries: int = 4,
        limiter: RateLimiter | None = None,
        headers: Mapping[str, str] | None = None,
        follow_redirects: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.limiter = limiter
        base_headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        if headers:
            base_headers.update(headers)
        self._client = httpx.Client(
            headers=base_headers,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 20.0)),
            follow_redirects=follow_redirects,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    # -- cache helpers --------------------------------------------------
    def _cache_paths(self, url: str, params: Mapping[str, Any] | None) -> tuple[Path, Path] | None:
        if self.cache_dir is None:
            return None
        key = url + ("?" + json.dumps(dict(params), sort_keys=True) if params else "")
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        sub = self.cache_dir / digest[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{digest}.body", sub / f"{digest}.meta.json"

    # -- core request ---------------------------------------------------
    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Any | None = None,
        use_cache: bool = False,
        max_age: float | None = None,
        raise_for_status: bool = True,
    ) -> httpx.Response:
        """Perform a request with retries, rate limiting and optional caching.

        ``max_age`` short-circuits the network entirely when a cached body is
        younger than the given number of seconds. Otherwise a cached entry is
        revalidated with ``If-None-Match`` / ``If-Modified-Since``; a ``304``
        response is transparently turned back into the cached body.
        """
        cache = self._cache_paths(url, params) if use_cache and method.upper() == "GET" else None
        req_headers = dict(headers or {})
        cached_meta: CachedResponse | None = None

        if cache:
            body_path, meta_path = cache
            if body_path.exists() and meta_path.exists():
                try:
                    raw = json.loads(meta_path.read_text(encoding="utf-8"))
                    cached_meta = CachedResponse(**raw)
                except (OSError, ValueError, TypeError):
                    cached_meta = None
            if cached_meta is not None:
                fresh = max_age is not None and (time.time() - cached_meta.fetched_at) < max_age
                if fresh:
                    return self._response_from_cache(url, cached_meta, body_path)
                etag = cached_meta.headers.get("etag")
                lastmod = cached_meta.headers.get("last-modified")
                if etag:
                    req_headers.setdefault("If-None-Match", etag)
                if lastmod:
                    req_headers.setdefault("If-Modified-Since", lastmod)

        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.limiter is not None:
                self.limiter.acquire()
            try:
                resp = self._client.request(
                    method, url, params=params, headers=req_headers, json=json_body)
            except (httpx.TransportError, httpx.HTTPError) as exc:
                last_exc = exc
                if attempt >= self.retries:
                    raise
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 304 and cached_meta is not None and cache:
                log.debug("304 not modified: %s", url)
                return self._response_from_cache(url, cached_meta, cache[0])

            if resp.status_code in RETRY_STATUS and attempt < self.retries:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                if resp.status_code == 429:
                    delay = retry_after if retry_after is not None else 10.0
                    if self.limiter is not None:
                        self.limiter.penalise(delay)
                    log.warning("429 from %s; backing off %.1fs", _host(url), delay)
                    time.sleep(delay)
                else:
                    self._sleep_backoff(attempt, retry_after)
                continue

            if raise_for_status and resp.status_code >= 400:
                resp.raise_for_status()

            if cache and resp.status_code == 200:
                self._store_cache(cache, url, resp)
            return resp

        if last_exc:
            raise last_exc
        raise httpx.HTTPError(f"exhausted retries for {url}")

    def _sleep_backoff(self, attempt: int, retry_after: float | None = None) -> None:
        delay = retry_after if retry_after is not None else min(2.0 ** attempt, 30.0)
        time.sleep(delay + random.uniform(0, 0.5))

    @staticmethod
    def _response_from_cache(url: str, meta: CachedResponse, body_path: Path) -> httpx.Response:
        resp = httpx.Response(
            status_code=meta.status,
            headers=meta.headers,
            content=body_path.read_bytes(),
            request=httpx.Request("GET", url),
        )
        return resp

    @staticmethod
    def _store_cache(cache: tuple[Path, Path], url: str, resp: httpx.Response) -> None:
        body_path, meta_path = cache
        try:
            body_path.write_bytes(resp.content)
            # ``resp.content`` is already decoded, so the transfer-encoding
            # headers must not be replayed or httpx would try to decompress
            # the plain bytes a second time.
            headers = {k.lower(): v for k, v in resp.headers.items()
                       if k.lower() not in _VOLATILE_HEADERS}
            meta = CachedResponse(
                url=url, status=resp.status_code, headers=headers,
                fetched_at=time.time(),
            )
            meta_path.write_text(json.dumps(meta.__dict__), encoding="utf-8")
        except OSError as exc:
            log.debug("cache write failed for %s: %s", url, exc)

    # -- conveniences ---------------------------------------------------
    def get(self, url: str, **kw: Any) -> httpx.Response:
        return self.request("GET", url, **kw)

    def get_json(self, url: str, **kw: Any) -> Any:
        """GET and parse JSON, tolerating servers that mislabel content type."""
        resp = self.get(url, **kw)
        return json.loads(resp.content.decode("utf-8", errors="replace"))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _host(url: str) -> str:
    try:
        return httpx.URL(url).host or url
    except Exception:
        return url


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header expressed in seconds."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_file(
    client: HttpClient,
    url: str,
    dest: Path,
    *,
    resume: bool = True,
    expected_size: int | None = None,
    progress_cb: Any | None = None,
) -> tuple[Path, str, int]:
    """Stream ``url`` to ``dest``, resuming a partial file when possible.

    Returns ``(path, sha256, bytes_total)``. A ``.part`` file is used until
    the transfer completes so an interrupted download can never be mistaken
    for a finished one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if (resume and part.exists()) else 0
    headers = {"Range": f"bytes={start}-"} if start else {}

    if client.limiter is not None:
        client.limiter.acquire()

    mode = "ab" if start else "wb"
    with client._client.stream("GET", url, headers=headers) as resp:  # noqa: SLF001
        if start and resp.status_code == 200:
            # Server ignored the range request: restart from scratch.
            start, mode = 0, "wb"
        elif resp.status_code not in (200, 206):
            resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) + start
        written = start
        with part.open(mode) as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if progress_cb is not None:
                    progress_cb(written, total or expected_size or 0)

    part.replace(dest)
    return dest, sha256_file(dest), dest.stat().st_size
