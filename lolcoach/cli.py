"""Operator CLI for the complete local LoL coaching pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .config import load_config

app = typer.Typer(help="Build and run a local League of Legends coaching model.", no_args_is_help=True, add_completion=False)


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug-level progress.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only show warnings and errors.")] = False,
) -> None:
    """Configure logging before any command runs.

    Without this the stages' progress messages are discarded, because Python's
    root logger defaults to WARNING and every stage logs at INFO.
    """
    import logging

    from .logging_utils import setup_logging

    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    setup_logging(load_config(None).paths.logs, level=level)


def _config(path: Path | None):
    return load_config(path)


@app.command(name="init")
def init(
    config: Annotated[Path | None, typer.Option("--config", help="YAML configuration file.")] = None,
    riot_key: Annotated[str | None, typer.Option("--riot-key", help="Riot API key; omit to enter it without echoing.")] = None,
    riot_id: Annotated[str | None, typer.Option("--riot-id", help="Optional opted-in Riot ID, written as GameName#TAG.")] = None,
    env_path: Annotated[Path | None, typer.Option("--env-path", help="Credential file; defaults to ~/.lolcoach.env.")] = None,
    open_portal: Annotated[bool, typer.Option("--open-portal/--no-open-portal", help="Open Riot's Developer Portal before prompting for a key.")] = True,
) -> None:
    """Validate a Riot key and save it outside the repository for timeline collection."""
    from .onboarding import (DEFAULT_ENV_PATH, RIOT_DEVELOPER_PORTAL, resolve_riot_id,
                             save_player_puuid, save_riot_key, validate_riot_key)
    import webbrowser

    cfg = _config(config)
    if riot_key is None:
        if open_portal:
            webbrowser.open(RIOT_DEVELOPER_PORTAL)
        riot_key = typer.prompt("Riot API key", hide_input=True, confirmation_prompt=True)
    destination = env_path or DEFAULT_ENV_PATH
    try:
        validate_riot_key(cfg, riot_key)
        save_riot_key(riot_key, env_path=destination)
        if riot_id:
            puuid = resolve_riot_id(cfg, riot_key, riot_id)
            save_player_puuid(puuid, env_path=destination)
    except (PermissionError, RuntimeError, ValueError) as exc:
        typer.echo(f"Riot onboarding did not complete: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Riot API key saved to {destination}. Timeline collection is ready.")
    if riot_id:
        typer.echo("Your opted-in player identity was saved for local review commands.")
    typer.echo("Next: lolcoach download_data --only riot --limit 25, then lolcoach prepare_dataset")


@app.command(name="doctor")
def doctor(
    config: Annotated[Path | None, typer.Option("--config", help="YAML configuration file.")] = None,
    strict: Annotated[bool, typer.Option("--strict", help="Return a non-zero exit code when a required component is absent.")] = False,
) -> None:
    """Check local installation, data readiness, and exact next actions."""
    from .doctor import run_doctor

    checks = run_doctor(_config(config))
    labels = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    for check in checks:
        typer.echo(f"[{labels.get(check.state, check.state.upper()):4}] {check.name}: {check.detail}")
        if check.fix:
            typer.echo(f"       Fix: {check.fix}")
    if strict and any(check.state == "fail" for check in checks):
        raise typer.Exit(1)


@app.command(name="live")
def live(
    watch: Annotated[bool, typer.Option("--watch", help="Refresh personal telemetry until Ctrl+C.")] = False,
    interval: Annotated[float, typer.Option(min=1.0, help="Seconds between updates in watch mode.")] = 3.0,
) -> None:
    """Show your own visible Game Client data; it never gives real-time instructions."""
    import time

    from .live import LiveClientUnavailable, fetch_live_snapshot, format_live_snapshot

    try:
        while True:
            typer.echo(format_live_snapshot(fetch_live_snapshot()))
            if not watch:
                return
            time.sleep(interval)
    except LiveClientUnavailable as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        typer.echo("Stopped local live telemetry.")


@app.command(name="ui")
def ui(
    config: Annotated[Path | None, typer.Option("--config", help="YAML configuration file.")] = None,
    port: Annotated[int, typer.Option(min=1024, max=65535, help="Loopback port for the review site.")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open the local review site in your browser.")] = True,
    native: Annotated[bool, typer.Option("--native/--browser", help="Open in a native app window (needs pywebview) instead of a browser tab.")] = False,
) -> None:
    """Launch the local match-review UI at http://127.0.0.1:<port>."""
    from .review import serve

    serve(_config(config), port=port, open_browser=open_browser and not native, native=native)


@app.command(name="review")
def review(
    target: Annotated[str, typer.Argument(help="Use 'last' for the newest local match or provide a match ID.")] = "last",
    config: Annotated[Path | None, typer.Option("--config", help="YAML configuration file.")] = None,
) -> None:
    """Print the five most useful stored coaching moments from a local match."""
    from .review import ReviewStore, format_timestamp

    store = ReviewStore(_config(config))
    if target == "last":
        matches = store.list_matches(1)
        if not matches:
            typer.echo("No local matches are available. Collect Riot timelines before reviewing a game.", err=True)
            raise typer.Exit(1)
        target = str(matches[0]["match_id"])
    try:
        moments = store.match_detail(target)["moments"]
    except LookupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if not moments:
        typer.echo(f"{target} has no derived coaching moments yet. Run `lolcoach prepare_dataset` after ingesting timelines.")
        return
    typer.echo(f"{target}: five coaching moments")
    for moment in moments[:5]:
        confidence = "" if moment["confidence"] is None else f" (confidence {float(moment['confidence']):.2f})"
        typer.echo(f"- {format_timestamp(int(moment['ts_ms']))} {moment['title']}{confidence}: {moment['summary']}")


@app.command(name="download_data")
def download_data(
    config: Annotated[Path | None, typer.Option("--config", help="YAML configuration file.")] = None,
    only: Annotated[list[str] | None, typer.Option("--only", help="Only these source names; may be repeated.")] = None,
    exclude: Annotated[list[str] | None, typer.Option("--exclude", help="Skip these source names; may be repeated.")] = None,
    force: Annotated[bool, typer.Option(help="Refresh files even when the manifest has a valid copy.")] = False,
    limit: Annotated[int | None, typer.Option(min=1, help="Per-source cap for a smoke run.")] = None,
) -> None:
    """Download every configured legal source and record provenance."""
    from .sources import download_all
    results = download_all(_config(config), only=only, exclude=exclude, force=force, limit=limit)
    for result in results:
        state = f"skipped ({result.skipped})" if result.skipped else "ok" if result.ok else f"failed ({result.error})"
        typer.echo(f"{result.name}: {state}; {result.files} files, {result.records} records")
    if any(not result.ok for result in results):
        raise typer.Exit(1)


@app.command(name="prepare_dataset")
def prepare_dataset(
    config: Annotated[Path | None, typer.Option("--config")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Maximum examples for a smoke run.")] = None,
    skip_refresh: Annotated[bool, typer.Option(help="Use existing normalized feature tables without re-ingesting.")] = False,
    synthetic: Annotated[bool, typer.Option("--synthetic", help="Run Stage 3 teacher-model explanations before building splits.")] = False,
    synthetic_limit: Annotated[int | None, typer.Option(min=1, help="Cap on newly generated synthetic examples.")] = None,
) -> None:
    """Ingest raw data, infer coaching features, and write train/validation/test JSONL."""
    from .dataset import generate_synthetic, prepare_dataset as build
    cfg = _config(config)
    if synthetic:
        synth = generate_synthetic(cfg, limit=synthetic_limit)
        if synth.skipped:
            typer.echo(f"Synthetic generation skipped: {synth.skipped}")
        else:
            typer.echo(
                f"Synthetic: {synth.generated:,} kept, {synth.rejected:,} rejected, "
                f"{synth.failed:,} failed via {synth.backend}:{synth.model}")
            if synth.reject_reasons:
                typer.echo("  rejections: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(synth.reject_reasons.items())))
    result = build(cfg, limit=limit, refresh_features=not skip_refresh)
    typer.echo(f"Wrote {result.examples:,} examples: train={result.train:,}, validation={result.validation:,}, test={result.test:,}.")
    for name, count in sorted(result.by_archetype.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {name}: {count:,}")
    typer.echo(f"Dataset card: {result.paths['card']}")


@app.command(name="derive_features")
def derive_features(
    config: Annotated[Path | None, typer.Option("--config")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Maximum timeline matches to derive.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Recompute features already stored for a match.")] = False,
) -> None:
    """Derive inspectable Stage 2 features without rewriting the training JSONL."""
    from .features import derive_all

    result = derive_all(_config(config), limit=limit, force=force)
    typer.echo(
        f"Derived {result.total_rows:,} rows across {result.matches:,} matches: "
        f"waves={result.waves:,}, lanes={result.lanes:,}, recalls={result.recalls:,}, "
        f"objectives={result.objectives:,}, rotations={result.rotations:,}, deaths={result.deaths:,}.")
    if result.errors:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(1)


@app.command(name="rag")
def rag(
    question: Annotated[str | None, typer.Argument(help="Question to retrieve context for.")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    build: Annotated[bool, typer.Option("--build", help="Rebuild corpus chunks, embeddings, and the FAISS index.")] = False,
    patch: Annotated[str | None, typer.Option(help="Prefer evidence from this patch, e.g. 16.16.")] = None,
    top_k: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Build the vector index or retrieve grounded League context."""
    from .rag import LocalRAG, build_index
    cfg = _config(config)
    if build:
        result = build_index(cfg)
        typer.echo(f"Built {result.chunks:,}-chunk FAISS index ({result.dimensions} dimensions) at {result.index_path}")
        return
    if not question:
        raise typer.BadParameter("Provide a question or use --build.")
    context, hits = LocalRAG(cfg).context_for(question, top_k=top_k, patch=patch)
    typer.echo(context)
    typer.echo("\nSources:")
    for hit in hits:
        typer.echo(f"- {hit.source}: {hit.title or 'untitled'} (score {hit.score:.3f})" + (f": {hit.url}" if hit.url else ""))


@app.command(name="train")
def train(
    config: Annotated[Path | None, typer.Option("--config")] = None,
    resume_from: Annotated[str | None, typer.Option(help="Specific checkpoint directory; defaults to the newest local checkpoint.")] = None,
) -> None:
    """Train a resumable LoRA/QLoRA adapter for the configured Qwen base model."""
    from .train import train as run
    result = run(_config(config), resume_from=resume_from)
    typer.echo(f"Saved adapter to {result.output_dir / 'adapter'} after {result.examples:,} examples.")


@app.command(name="benchmark")
def benchmark(
    config: Annotated[Path | None, typer.Option("--config")] = None,
    competitor: Annotated[list[str] | None, typer.Option("--competitor", help="local, anthropic, openai, or deepseek; may be repeated.")] = None,
    count: Annotated[int | None, typer.Option(min=1, help="Question count; defaults to eval.n_questions (1,000). ")] = None,
) -> None:
    """Compare coaching answers and write factuality/quality/hallucination metrics."""
    from .eval import run_benchmark
    result = run_benchmark(_config(config), competitors=competitor, count=count)
    typer.echo(f"Benchmark report: {result.report_path}")
    for model, metrics in result.summary.items():
        typer.echo(f"{model}: " + ", ".join(f"{name}={value:.3f}" for name, value in sorted(metrics.items())))


@app.command(name="chat")
def chat(
    question: Annotated[str, typer.Argument(help="League coaching question.")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
    patch: Annotated[str | None, typer.Option(help="Prefer a specific patch.")] = None,
    top_k: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Answer with the local Qwen adapter, grounded in local RAG evidence."""
    from .chat import LocalCoach
    answer = LocalCoach(_config(config)).ask(question, patch=patch, top_k=top_k)
    typer.echo(answer.answer)
    typer.echo("\nSources:")
    for hit in answer.sources:
        typer.echo(f"- {hit.source}: {hit.title or 'untitled'}" + (f": {hit.url}" if hit.url else ""))


@app.command(name="update_patch")
def update_patch(
    config: Annotated[Path | None, typer.Option("--config")] = None,
    rebuild_index: Annotated[bool, typer.Option("--rebuild-index/--no-rebuild-index", help="Refresh embeddings after static/patch data changes.")] = True,
) -> None:
    """Refresh Riot static data and patch notes, then update the local RAG corpus."""
    from .rag import build_index, chunk_corpus
    from .sources import download_all
    cfg = _config(config)
    results = download_all(cfg, only=["ddragon", "cdragon", "patch_notes"])
    if any(not result.ok for result in results):
        raise typer.Exit(1)
    if rebuild_index:
        indexed = build_index(cfg)
        typer.echo(f"Patch data refreshed and {indexed.chunks:,} RAG chunks re-indexed.")
    else:
        chunks = chunk_corpus(cfg)
        typer.echo(f"Patch data refreshed; {chunks:,} chunks are ready for a later `rag --build`.")


@app.command(name="status")
def status(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    """Show local database counts and the current data root."""
    from .storage import Database
    cfg = _config(config)
    typer.echo(f"Data root: {cfg.paths.root}")
    for table, count in Database(cfg.paths.db).counts().items():
        typer.echo(f"{table}: {count:,}")


if __name__ == "__main__":
    app()
