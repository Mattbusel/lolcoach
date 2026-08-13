# LoLCoach

LoLCoach is a local, resumable League of Legends coaching pipeline. It collects permitted source data with a provenance ledger, normalizes Riot timelines and Oracle's Elixir data into SQLite, derives cautious coaching features, writes chat JSONL, builds a local FAISS retrieval index, and can train a Qwen LoRA/QLoRA adapter on an NVIDIA GPU.

The pipeline distinguishes observed data from inferred coaching state. Timeline-derived wave, recall, jungle, rotation, fight, and death labels carry confidence and should be treated as heuristic training signals, not ground truth.

## Quick start (Windows PowerShell)

```powershell
cd C:\Users\Matthew\lol-coach
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui]"
```

PyPI's default Windows PyTorch wheel is CPU-only. Review, ingestion, dataset preparation, and RAG work with it, but QLoRA training and usable local 7B chat require an NVIDIA CUDA build in this same virtual environment. On the verified RTX 4070 setup, install the matching build after the command above:

```powershell
python -m pip install --upgrade --force-reinstall torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')"
```

If your NVIDIA driver requires a different CUDA build, use the corresponding official PyTorch wheel index and require that the final command prints `True` before running `lolcoach train` or local `chat`.

The editable install includes the RAG dependencies (`sentence-transformers` and `faiss-cpu`) as well as PyTorch, Transformers, PEFT, TRL, datasets, and the CLI. If the Hugging Face cache under your user profile is not writable, keep model downloads in the project instead:

```powershell
$env:HF_HOME = "$PWD\.hf-cache"
```

Use the included defaults or copy them before making local changes:

```powershell
Copy-Item configs\default.yaml configs\local.yaml
lolcoach status --config configs\local.yaml
```

`LOLCOACH_HOME` overrides both the config and the repository `data/` directory. Point it to a drive with adequate capacity before a full collection:

```powershell
$env:LOLCOACH_HOME = "D:\LoLCoachData"
```

## Credentials

The CLI reads environment variables first, then (in increasing fallback order) `Desktop\keys.env`, `Desktop\credentials.env`, `~\.lolcoach.env`, and the repository `.env`. Do not commit credentials.

| Capability | Variables | Required? |
| --- | --- | --- |
| High-elo Riot match/timeline collection | `RIOT_API_KEY` (or `RIOT_KEY`, `RIOT_TOKEN`) | Yes, only for the `riot` source |
| Higher Hugging Face Hub limits | `HF_TOKEN` | Optional |
| Kaggle discovery/download | `KAGGLE_USERNAME` and `KAGGLE_KEY` | Optional |
| More GitHub API quota | `GITHUB_TOKEN` or `GH_TOKEN` | Optional |
| Opt-in teacher examples / Anthropic judging | `ANTHROPIC_API_KEY` | Optional |
| Opt-in teacher examples / OpenAI judging | `OPENAI_API_KEY` | Optional |
| Opt-in DeepSeek teacher/competitor | `DEEPSEEK_API_KEY` | Optional |

For the Riot API, create a key at the Riot Developer Portal, then set it only for the current session:

```powershell
$env:RIOT_API_KEY = "your-riot-development-or-production-key"
```

A development key expires regularly; the collector reports a Riot 403 instead of retrying forever. Sources without required credentials are skipped and reported; all other configured sources continue.

For the guided local setup, use the command instead of placing a secret in a repository file. It opens the Riot Developer Portal when it needs a key, validates the key with Riot, and saves it only to `~\.lolcoach.env`:

```powershell
lolcoach init
lolcoach init --riot-id "Your Game Name#TAG"
lolcoach doctor
```

`doctor` is read-only. It reports the exact state of CUDA/VRAM, package extras, free disk space, credentials, the RAG index, dataset, adapter, and whether real Riot timeline features exist.

## What runs without any credentials

Without keys you can collect Data Dragon, Community Dragon, the League wiki, patch notes, Reddit coaching threads via Arctic Shift, and HuggingFace/GitHub dataset discovery. You can then prepare static-only datasets and build or query the local RAG index. Training additionally requires a compatible NVIDIA GPU; a fully local benchmark additionally requires a completed local adapter.

What a `RIOT_API_KEY` unlocks is the part that matters most: **match timelines**. Every wave, jungle-path, recall, objective-control, death-cause and rotation feature is derived from timeline frames, so without a key those archetypes are empty and the dataset falls back to static knowledge, wiki mechanics and community Q&A. A free development key takes a minute to create and immediately turns on the rest.

### Patch numbering

Riot moved to year-based patch names in 2025 but left Data Dragon on the old season numbering, so the two disagree by ten majors: Data Dragon `16.16` is the patch players and the wiki call **26.16**. `public_patch_label()` does the conversion, patch-notes documents are filed under the public label, and static data keeps the Data Dragon key. Use the public number (`--patch 26.16`) when querying.

### Oracle's Elixir

The professional-match CSVs moved off S3 to a public Google Drive folder. The connector enumerates the folder at run time and resolves each season's file id, so a new season file is picked up automatically. Drive enforces a per-file download quota shared across all users; when it is exhausted the connector reports that specific condition and leaves the file for a later run rather than saving the HTML error page as a `.csv`.

## End-to-end commands

Start with a small, safe collection run. `--limit` is a per-source cap for downloading and a maximum-example cap for dataset preparation.

```powershell
lolcoach download_data --config configs\local.yaml --limit 25
lolcoach prepare_dataset --config configs\local.yaml --limit 250
lolcoach derive_features --config configs\local.yaml --force
lolcoach rag --config configs\local.yaml --build
lolcoach rag "How should I set up dragon on patch 26.16?" --config configs\local.yaml --patch 26.16
```

For a full local corpus, remove the smoke-run limits:

```powershell
lolcoach download_data --config configs\local.yaml
lolcoach prepare_dataset --config configs\local.yaml
lolcoach rag --config configs\local.yaml --build
```

All source bytes are retained below `raw/`, download metadata and hashes live in `manifest.sqlite`, normalized data and provenance live in `lolcoach.sqlite`, and writes use atomic replacement where a final file is produced. Re-running ingestion skips matches that already have a complete timeline; re-running feature derivation skips matches with feature rows. Use `download_data --force` only when intentionally refreshing an upstream copy.

Refresh patch-sensitive factual data and rebuild retrieval after a new Riot patch:

```powershell
lolcoach update_patch --config configs\local.yaml
```

To refresh source files now but build the index later:

```powershell
lolcoach update_patch --config configs\local.yaml --no-rebuild-index
lolcoach rag --config configs\local.yaml --build
```

### Optional Stage 3 teacher examples

Teacher-generated examples are opt-in. Set `dataset.synthetic_backend` in `configs\local.yaml` to `anthropic`, `openai`, `deepseek`, or `local`, configure that backend's credential if needed, then run:

```powershell
lolcoach prepare_dataset --config configs\local.yaml --synthetic --synthetic-limit 1000
```

Generated examples are kept in a separate JSONL file and still pass local validation, deduplication, split assignment, and provenance handling.

## Local coaching app

Launch the match-review interface with one command:

```powershell
lolcoach ui --open
```

It binds only to `127.0.0.1`, opens a browser, and provides a recent-game picker, timeline scrubber, position map, and confidence-labelled wave, recall, death, objective, and rotation cards. "Ask about this moment" scopes the local RAG/model prompt to the stored evidence for the selected timestamp. The page has a key onboarding form and a bounded Riot-timeline collection action; neither sends data to a LoLCoach service. Position markers show only timeline positions; when players overlap, they are grouped rather than hidden.

For a quick terminal review:

```powershell
lolcoach review last
lolcoach derive_features --force
```

### Live telemetry boundaries

`lolcoach live` reads the Game Client API's fixed localhost endpoint and prints the player's already-visible game time, champion, health, gold, level, KDA, CS, and inventory. It disables TLS verification only for that self-signed `127.0.0.1:2999` endpoint. It does not inspect hidden enemy state and it deliberately does not issue "go here / do this now" instructions.

Riot’s policy prohibits apps that dictate player decisions or reveal game-session information the player did not already know. The product therefore focuses on pre-game static information, transparent personal telemetry, and post-game review. Review Riot’s current [League developer policy and Game Client API documentation](https://developer.riotgames.com/docs/lol) before distributing a product using client APIs.

## Windows desktop bundle

Build a double-clickable local application from a Windows machine with the full runtime installed:

```powershell
python -m pip install -e ".[dev,ui,bundle]"
.\scripts\build_windows.ps1
```

The result is `dist\LoLCoach\LoLCoach.exe`. Double-click it to open the local review app at `http://127.0.0.1:8765`. It stores each player's data under `%LOCALAPPDATA%\LoLCoach\data`; setting `LOLCOACH_HOME` or placing a `data` folder beside the executable creates a portable data location.

The full portable bundle includes the CUDA/PyTorch runtime, Qwen base model, embedding model, trained adapter, local RAG index, source provenance, and the current local match database. It is about 22.5 GiB unpacked. To share it, zip and send the entire `dist\LoLCoach` folder (or the generated ZIP); `LoLCoach.exe` cannot run by itself without its `_internal`, `data`, and `models` folders. Riot credentials are deliberately excluded.

The recipient workflow is: unzip the folder, double-click `LoLCoach.exe`, paste their own Riot developer key in the first-time setup panel, click **Collect my games**, choose a game from **Your games**, and click any **Review moment** card to jump to that timestamp. The app keeps the downloaded games and coaching history locally. Use **Collect latest games** after future matches.

The bundle ships application code and runtime libraries, not Qwen weights, Riot data, League Wiki data, or collected games. That keeps source licences and player data local. The generated executable is unsigned; code-sign it before public distribution to avoid Windows SmartScreen warnings.

## GPU training and chat

QLoRA training requires a CUDA-visible NVIDIA GPU and a CUDA-compatible PyTorch/bitsandbytes installation. The defaults select conservative settings for a 12 GB RTX 4070-class card and larger settings for 16 GB or 24 GB cards. A 24 GB RTX 4090-class GPU is preferred for the 7B model and long contexts. Verify the intended GPU before training:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')"
```

Do not run `train` when the last value is `False`; the command intentionally exits before loading model weights. On the intended GPU:

```powershell
lolcoach train --config configs\local.yaml
lolcoach train --config configs\local.yaml --resume-from data\models\qwen-lolcoach\checkpoint-1000
lolcoach chat "I am 600 gold ahead in mid with dragon in 50 seconds. What is my plan?" --config configs\local.yaml --patch 26.16
```

The adapter, tokenizer, checkpoints, and `training_manifest.json` are saved under `models/<output_name>/`. `chat` requires that local adapter and a built RAG index; it will not silently fall back to a remote model.

## Benchmarking

The benchmark creates 1,000 fixed-seed questions by default and writes both the individual-answer JSONL and summary JSON under `reports/`. It scores factual and structural signals locally; optional Anthropic/OpenAI judging is blended only when explicitly configured. Consistency re-asks a real paraphrase, compares the original and paraphrased answers' recommended decisions, and retains both answers in the record.

```powershell
lolcoach benchmark --config configs\local.yaml
lolcoach benchmark --config configs\local.yaml --count 25 --competitor local
```

A local adapter must exist for the `local` competitor. Remote competitors are skipped when their credentials are absent. Run the short command first when enabling any paid provider.

## Data, licensing, and provenance

The MIT license applies to this repository's code only. It does not grant rights to redistribute models, source data, generated datasets, or game assets. Every downloaded file has a URL, hash, source, license/terms label, and retrieval documents retain source URL, patch, and license metadata.

| Source | Use in this pipeline | Key restriction |
| --- | --- | --- |
| Riot Data Dragon and Riot API | Static factual data; your authorized match/timeline collection | Riot terms apply; no Riot endorsement; do not redistribute raw data unless permitted |
| Community Dragon | Client-data mirror used as supplemental static context | Respect its upstream/Riot terms and attribution |
| League of Legends Wiki | Mechanics and patch-history retrieval documents | CC BY-SA 4.0; preserve attribution and share-alike obligations when redistributing derivatives |
| Oracle's Elixir | Public professional match data | Non-commercial use with credit; do not assume redistribution rights |
| Reddit / Arctic Shift | Public coaching discussion retrieval data | User-generated content; preserve author/platform rights and do not redistribute blindly |
| GitHub, Hugging Face, Kaggle discoveries | Discovery only; approved files are recorded with their own license | Inspect each dataset/repository license before use or redistribution |
| Local `.rofl` replays | Local metadata only | Never downloads or decrypts third-party replay payloads |

The project deliberately does not scrape or bulk-download VODs from YouTube or Twitch, does not bypass platform authentication, and does not use unauthorized replay decryption. Before sharing a trained adapter or a dataset, audit the manifest and document-level provenance for every contributing source, comply with the most restrictive source terms, and obtain any needed rights.

## Verified local environment

This working copy was verified on Windows with Python 3.12, PyTorch 2.11.0+cu128, Transformers 5.5.0, PEFT 0.18.1, TRL 0.24.0, datasets 4.3.0, sentence-transformers 5.7.0, FAISS CPU 1.15.0, FastAPI 0.128.5, and an RTX 4070 (12 GB). The 86-test suite passes, as do fixture-based Riot/Oracle ingestion, forced feature re-derivation, JSONL creation, local embedding/index creation, patch-filtered retrieval, CLI command surfaces, review-UI API routes, tokenization/collation, and a packaged Windows-app launch check. A 7B QLoRA adapter was trained for three epochs and benchmarked locally; the benchmark currently identifies vagueness rather than hallucination as the quality bottleneck, so a larger, balanced set of real timeline examples is the next material quality step.

Transformers 5 changed several APIs this project depends on, and the code targets the new behaviour:

- `apply_chat_template` returns a dict unless `return_dict=False`, so the tokenizer calls pass it explicitly. Without this the loss-masking slices operate on dict keys.
- `torch_dtype` was renamed to `dtype` in `from_pretrained`.
- `TrainingArguments.overwrite_output_dir` was removed.

### Unsloth

`train.use_unsloth` defaults to true, but Unsloth is used only when it can actually train. Its fused kernels are compiled by Triton at the first backward pass, which needs a C compiler; on a machine without one, loading succeeds and training then fails minutes later. A preflight check tests for `unsloth_zoo` import compatibility and a C compiler on `PATH`, and falls back to standard PEFT + bitsandbytes QLoRA with a logged reason. The current CPU-only environment does not attempt either GPU training path; use the CUDA preflight above before starting a training run.

Run the project checks after changing dependencies or configuration:

```powershell
python -m compileall -q lolcoach tests
python -m pytest
lolcoach --help
```
