"""Oracle's Elixir professional match data.

Oracle's Elixir publishes the canonical public dataset of professional League
matches: one row per player per game plus one row per team, with gold/xp/cs
splits at 10, 15, 20 and 25 minutes, objective control, and full draft
information, going back to 2014.

It is the best available proxy for "professional games" and "high-elo VODs":
the decisions a coach cares about (draft, objective trades, side-lane
assignment, tempo around spawn timers) are all recoverable from these rows,
without scraping video from platforms whose terms forbid it.

Distribution
------------
The files moved off S3 to a public Google Drive folder linked from the site.
This connector reads the folder listing, resolves each year's file id, and
streams the CSV through Drive's download endpoint. Discovering the ids at run
time means the connector keeps working when a new season file appears, which
is exactly the auto-integration behaviour Stage 1 asks for.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

from ..http import download_file
from ..logging_utils import get_logger
from ..storage import DownloadRecord
from .base import ORACLES_ELIXIR, Source, SourceContext, SourceResult, register

log = get_logger("sources.oracles_elixir")

#: Public folder advertised on oracleselixir.com/tools/downloads.
DRIVE_FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download"

#: Files are named ``<year>_LoL_esports_match_data_from_OraclesElixir.csv``.
FILE_RE = re.compile(r"(\d{4})_LoL_esports_match_data_from_OraclesElixir[^\"\\<]*?\.csv")
DATA_ID_RE = re.compile(r'data-id="([-\w]{20,50})"')


class DriveQuotaExceeded(RuntimeError):
    """Raised when Drive refuses a public file because its quota is spent."""


def discover_drive_files(html: str) -> dict[str, str]:
    """Map ``filename -> drive file id`` from a public folder listing.

    Drive renders the folder as a flat sequence of ``div`` blocks that open
    with ``data-id="<file id>"`` and contain the file name shortly after, so
    splitting on the id attribute and searching each block forward is a stable
    way to pair them without a headless browser.
    """
    pairs: dict[str, str] = {}
    for block in html.split('data-id="')[1:]:
        file_id = block.split('"', 1)[0]
        m = FILE_RE.search(block[:4000])
        if m:
            pairs.setdefault(m.group(0), file_id)
    return pairs


@register
class OraclesElixirSource(Source):
    """Download the per-year professional match CSVs."""

    name = "oracles_elixir"
    description = "Oracle's Elixir professional match data (2014-present)"
    license = ORACLES_ELIXIR

    def fetch(self, ctx: SourceContext, result: SourceResult) -> None:
        wanted_years = {int(y) for y in ctx.cfg.sources.oracles_elixir_years}
        html = ctx.http.get(FOLDER_URL, use_cache=True, max_age=3600).text
        available = discover_drive_files(html)
        if not available:
            result.ok = False
            result.error = ("Google Drive folder listing returned no CSV entries; "
                            "the folder layout may have changed")
            return

        result.note(f"{len(available)} files listed in the public folder")
        out = ctx.out_dir
        out.mkdir(parents=True, exist_ok=True)

        for fname, file_id in sorted(available.items()):
            year_match = re.match(r"(\d{4})", fname)
            if not year_match:
                continue
            year = int(year_match.group(1))
            if wanted_years and year not in wanted_years:
                continue

            url = f"{DOWNLOAD_URL}?id={file_id}&export=download&confirm=t"
            dest = out / fname
            # Season files are appended to all year; refresh the current year
            # daily and treat older seasons as immutable.
            max_age = 86400 if year >= max(wanted_years or {year}) else None
            if self.should_skip(ctx, url, dest, max_age_s=max_age):
                result.note(f"{fname}: up to date")
                continue

            try:
                path, digest, size = self._download_drive_file(ctx, url, dest)
            except DriveQuotaExceeded:
                result.note(
                    f"{fname}: Google Drive reports this public file has exceeded "
                    f"its download quota. This is an upstream limit shared by all "
                    f"downloaders, not a local failure; it clears on its own, so "
                    f"re-run the download later.")
                ctx.manifest.mark_failed(url, dest, self.name, "drive quota exceeded")
                continue
            except Exception as exc:
                result.note(f"{fname}: download failed ({exc})")
                ctx.manifest.mark_failed(url, dest, self.name, str(exc))
                continue

            ctx.manifest.record(DownloadRecord(
                url=url, path=str(path), source=self.name, kind="pro_matches",
                sha256=digest, bytes=size, **self.license.as_dict()))
            result.files += 1
            result.bytes += size
            result.note(f"{fname}: {size / 1e6:.1f} MB")

        if result.files == 0 and not result.notes:
            result.note("no matching years found")

    def _download_drive_file(self, ctx: SourceContext, url: str, dest: Path
                             ) -> tuple[Path, str, int]:
        """Fetch a Drive file, clearing the large-file confirmation if needed.

        Drive answers a large public download with an HTML page rather than the
        bytes. There are two distinct pages and they need different handling:

        * a **confirmation form**, which must be resubmitted with its hidden
          fields (including a per-request ``uuid``) to get the real download;
        * a **quota page**, which means the file has served too much traffic
          recently. No amount of retrying helps until the quota resets, so it
          is raised as a distinct error and reported as such.
        """
        probe = ctx.http.get(url, raise_for_status=False)
        content_type = probe.headers.get("Content-Type", "")

        if "text/html" not in content_type.lower():
            return download_file(ctx.http, url, dest)

        html = probe.text
        if "quota" in html.lower():
            raise DriveQuotaExceeded(dest.name)

        fields = _hidden_inputs(html)
        if not fields:
            raise RuntimeError("Drive returned an unrecognised HTML page")
        action = re.search(r'<form[^>]*action="([^"]+)"', html, flags=re.IGNORECASE)
        endpoint = action.group(1).replace("&amp;", "&") if action else DOWNLOAD_URL
        query = "&".join(f"{k}={quote(v)}" for k, v in fields.items() if v)
        confirmed = f"{endpoint}?{query}" if query else endpoint
        return download_file(ctx.http, confirmed, dest)


def _hidden_inputs(html: str) -> dict[str, str]:
    """Extract ``name -> value`` from the hidden inputs of Drive's confirm form."""
    fields: dict[str, str] = {}
    for tag in re.findall(r"<input\b[^>]*>", html, flags=re.IGNORECASE):
        name = re.search(r'name="([^"]+)"', tag)
        value = re.search(r'value="([^"]*)"', tag)
        if name:
            fields[name.group(1)] = value.group(1) if value else ""
    return fields


def local_csvs(raw_root: Path) -> list[Path]:
    """Return downloaded Oracle's Elixir CSVs, newest season first."""
    d = raw_root / "oracles_elixir"
    if not d.exists():
        return []
    return sorted(d.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"), reverse=True)
