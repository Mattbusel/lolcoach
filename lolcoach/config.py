"""Configuration and credential resolution.

Design goals
------------
* **One file to edit.** ``configs/default.yaml`` holds every tunable; the CLI
  accepts ``--config`` to point at an override.
* **Never prompt for secrets.** Credentials are discovered from the process
  environment and from a search path of ``.env``-style files, so an operator
  can drop a key file next to the project (or on their Desktop) and every
  stage picks it up.
* **Absent credentials are not fatal.** Each source declares which key it
  needs; the downloader skips sources whose credentials are missing and says
  so plainly instead of failing the whole run.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .paths import REPO_ROOT, Paths, get_paths

# ---------------------------------------------------------------------------
# .env discovery
# ---------------------------------------------------------------------------

#: Files scanned for credentials, in ascending order of precedence. Values
#: already present in ``os.environ`` always win over file contents.
ENV_SEARCH_PATH: tuple[Path, ...] = (
    Path.home() / "Desktop" / "keys.env",
    Path.home() / "Desktop" / "credentials.env",
    Path.home() / ".lolcoach.env",
    REPO_ROOT / ".env",
)

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` file, tolerating quotes, ``export`` and comments."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load_credentials(extra_paths: Iterable[Path] = ()) -> dict[str, str]:
    """Merge credentials from the search path, with ``os.environ`` on top."""
    merged: dict[str, str] = {}
    for path in (*ENV_SEARCH_PATH, *extra_paths):
        if path.is_file():
            merged.update(parse_env_file(path))
    merged.update({k: v for k, v in os.environ.items() if v})
    return merged


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RiotConfig:
    """Riot Games API access.

    ``platforms`` are platform routing values (league / summoner endpoints);
    ``region_of_platform`` maps them onto the regional routing values that
    match-v5 requires. ``tiers`` controls which slice of the ladder is
    sampled -- the default is the top three tiers, which is what "challenger
    replays" means in practice for a solo-queue coaching corpus.
    """

    api_key_env: str = "RIOT_API_KEY"
    platforms: list[str] = field(default_factory=lambda: ["na1", "euw1", "kr"])
    region_of_platform: dict[str, str] = field(default_factory=lambda: {
        "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
        "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
        "kr": "asia", "jp1": "asia",
        "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea",
    })
    tiers: list[str] = field(default_factory=lambda: ["CHALLENGER", "GRANDMASTER", "MASTER"])
    divisions: list[str] = field(default_factory=lambda: ["I"])
    queue: str = "RANKED_SOLO_5x5"
    #: Ranked solo queue id, used to filter match history.
    queue_id: int = 420
    players_per_platform: int = 200
    matches_per_player: int = 20
    max_matches: int = 20000
    #: Requests/second and requests/2min for a *development* key. Production
    #: keys should raise these; the limiter also self-corrects from response
    #: headers, so a wrong value here is corrected by Riot's own reply.
    rate_per_second: float = 20.0
    rate_per_two_minutes: float = 100.0


@dataclass
class SourcesConfig:
    """Which Stage-1 sources are enabled and how much each one pulls."""

    ddragon: bool = True
    cdragon: bool = True
    oracles_elixir: bool = True
    oracles_elixir_years: list[int] = field(default_factory=lambda: [2024, 2025, 2026])
    wiki: bool = True
    wiki_max_pages: int = 1200
    patch_notes: bool = True
    patch_notes_limit: int = 40
    reddit: bool = True
    reddit_subreddits: list[str] = field(default_factory=lambda: [
        "summonerschool", "leagueoflegends", "LoLTheoryCraft",
        "Jungle_Mains", "supportlol", "ADCMains", "TopMains",
    ])
    reddit_posts_per_sub: int = 1500
    reddit_min_score: int = 25
    huggingface: bool = True
    huggingface_queries: list[str] = field(default_factory=lambda: [
        "league of legends", "league of legends match", "lol esports",
    ])
    github: bool = True
    github_queries: list[str] = field(default_factory=lambda: [
        "league of legends dataset", "lol match timeline dataset",
        "league of legends matchup data",
    ])
    kaggle: bool = True
    kaggle_queries: list[str] = field(default_factory=lambda: [
        "league of legends", "lol ranked games", "league of legends challenger",
    ])
    riot: bool = True
    replays: bool = True
    #: Where the League client writes ``.rofl`` files. Scanned if present.
    replay_dirs: list[str] = field(default_factory=lambda: [
        str(Path.home() / "Documents" / "League of Legends" / "Replays"),
    ])
    #: Re-run dataset discovery on every download (Stage 1 auto-integration).
    auto_discover: bool = True


@dataclass
class DatasetConfig:
    """Stage 2/3 example generation controls."""

    max_examples: int = 400000
    #: Minimum estimator confidence for a wave-state example to be emitted.
    min_wave_confidence: float = 0.45
    val_fraction: float = 0.02
    test_fraction: float = 0.01
    #: Near-duplicate suppression on the normalised prompt hash.
    dedupe: bool = True
    #: Cap per question archetype so one template cannot dominate the mix.
    max_per_archetype: int = 60000
    synthetic_backend: str = "anthropic"   # anthropic | openai | deepseek | local | none
    synthetic_model: str = "claude-sonnet-5"
    synthetic_max_examples: int = 20000
    synthetic_concurrency: int = 4


@dataclass
class RagConfig:
    """Stage 4/5 retrieval controls."""

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_batch_size: int = 128
    chunk_tokens: int = 320
    chunk_overlap: int = 64
    top_k: int = 8
    #: Weight of the dense score when fusing with BM25 (0 = lexical only).
    dense_weight: float = 0.65
    #: Extra score for chunks whose patch matches the query's patch.
    patch_boost: float = 0.12


@dataclass
class TrainConfig:
    """Stage 6 fine-tuning controls.

    Any numeric field left at ``0`` is chosen automatically from the detected
    GPU (see :mod:`lolcoach.train.hparams`), which is what makes the same
    config work on a 12 GB 4070 and a 24 GB 4090.
    """

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_name: str = "qwen-lolcoach"
    use_unsloth: bool = True          # falls back automatically when unusable
    load_in_4bit: bool = True         # QLoRA
    lora_r: int = 0
    lora_alpha: int = 0
    lora_dropout: float = 0.05
    max_seq_len: int = 0
    epochs: float = 3.0
    lr: float = 0.0
    batch_size: int = 0
    grad_accum: int = 0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    seed: int = 3407
    gradient_checkpointing: bool = True
    save_every_epoch: bool = True
    resume: bool = True


@dataclass
class EvalConfig:
    """Stage 7 benchmark controls."""

    n_questions: int = 1000
    judge_backend: str = "anthropic"
    judge_model: str = "claude-sonnet-5"
    competitors: list[str] = field(default_factory=lambda: ["anthropic", "openai", "deepseek"])
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o-mini"
    deepseek_model: str = "deepseek-chat"
    max_new_tokens: int = 512
    #: Number of paraphrase pairs used for the consistency metric.
    consistency_pairs: int = 100


@dataclass
class Config:
    """Root configuration object handed to every stage."""

    data_root: str | None = None
    user_agent: str = "lolcoach/1.0 (research pipeline; contact: local operator)"
    http_timeout: float = 60.0
    http_retries: int = 4
    riot: RiotConfig = field(default_factory=RiotConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    #: Populated in ``__post_init__``; never read from YAML.
    creds: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.creds:
            self.creds = load_credentials()

    @property
    def paths(self) -> Paths:
        """Materialised data layout (directories are created on access)."""
        return get_paths(self.data_root).ensure()

    def secret(self, *names: str) -> str | None:
        """Return the first non-empty credential among ``names``."""
        for n in names:
            v = self.creds.get(n)
            if v:
                return v.strip()
        return None


_SECTIONS: dict[str, type] = {
    "riot": RiotConfig, "sources": SourcesConfig, "dataset": DatasetConfig,
    "rag": RagConfig, "train": TrainConfig, "eval": EvalConfig,
}


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load configuration, defaulting to ``configs/default.yaml``.

    Unknown keys are ignored rather than raising, so a config written for a
    newer version of the package still loads.
    """
    cfg_path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    kwargs: dict[str, Any] = {}
    root_fields = {f.name for f in fields(Config)}
    for key, value in data.items():
        if key in _SECTIONS and isinstance(value, Mapping):
            allowed = {f.name for f in fields(_SECTIONS[key])}
            kwargs[key] = _SECTIONS[key](**{k: v for k, v in value.items() if k in allowed})
        elif key in root_fields and key != "creds":
            kwargs[key] = value

    cfg = Config(**kwargs)
    if os.environ.get("LOLCOACH_HOME"):
        cfg.data_root = os.environ["LOLCOACH_HOME"]
    return cfg
