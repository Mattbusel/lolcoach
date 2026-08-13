# LoLCoach — handoff prompt for the next agent

Paste this whole file as your starting context. The project is at
`C:\Users\Matthew\lol-coach`. It works today; your job is to make it a tool
someone would actually choose to use.

---

## 1. Your mission

An eight-stage local League of Legends coaching pipeline already exists and
runs end to end: data collection → timeline feature inference → dataset →
RAG → QLoRA fine-tune → benchmark → CLI. It is correct but it is a
*pipeline*, not a *product*. There is no interface, no onboarding, and the
model is under-trained because the highest-value data source is not connected
yet.

Deliver, in priority order:

1. **Unlock the data** (Riot API key onboarding + live game integration).
2. **Build the interface** (match review UI + in-game overlay).
3. **Make it installable and self-diagnosing** for a non-engineer.
4. **Raise model quality** using the outcome labels already in the database.

Do not rewrite what works. Read section 4 before touching anything.

---

## 2. Verified current state

Measured on this machine, not assumed:

| Thing | State |
| --- | --- |
| Code | 47 modules, ~12,200 lines, 76 tests passing (`python -m pytest`) |
| Corpus | 2,966 documents; 173 champions, 868 items, 62 runes (Data Dragon patch 16.16 = public patch **26.16**) |
| Downloaded | ~1.75 GB across wiki (741 pages), Reddit (1,085 threads), patch notes (37 patches), Data Dragon, Community Dragon, HuggingFace, GitHub |
| RAG | FAISS index, 5,695 chunks, 384-dim MiniLM, hybrid dense + IDF-weighted lexical, patch-boosted |
| Dataset | 806 examples (782 train / 15 val / 9 test) |
| Trained model | Qwen2.5-7B-Instruct QLoRA adapter, 3 epochs, train loss 0.85, eval loss 1.45, checkpoints at 49/98/147, resume verified by killing and restarting mid-run |
| Benchmark | 40 questions, local model: coaching_quality 0.441, consistency 0.406, factual_accuracy 0.50, hallucination_risk 0.00, **checkable_claims 0.0** |
| Timeline features | **All zero.** `wave_states`, `lane_states`, `recalls`, `jungle_paths`, `deaths`, `fights`, `rotations`, `objective_events` are empty |

Two numbers tell you what to fix:

- **`checkable_claims 0.0`** — across 40 answers the model never made a single
  verifiable claim. It is fluent and vague. `hallucination_risk 0.00` looks
  great but only because it asserts nothing. Vagueness is the current failure
  mode, not hallucination.
- **Timeline features all zero** — the entire Stage-2 inference layer (the
  interesting part: wave state, jungle prediction, death causes) is built,
  unit-tested against scripted fixtures, and receiving no input, because
  there is no `RIOT_API_KEY`.

---

## 3. Repo map

```
lolcoach/
  config.py          Config + credential discovery (reads Desktop/keys.env etc.)
  http.py            Retrying client, dual-window rate limiter, disk cache
  paths.py           Canonical data layout
  storage/           SQLite schema (v4) + download provenance ledger
  sources/           Stage 1 connectors, one file per source
    riot_api.py      Ladder → match ids → match + timeline. COMPLETE, UNUSED (no key)
    ddragon.py       Static data + public_patch_label() (16.16 → 26.16)
    wiki.py          MediaWiki crawl, indexed from Data Dragon not categories
    reddit.py        Arctic Shift archive (Reddit's own API 403s unauthenticated)
    oracles_elixir.py Google Drive enumeration; currently blocked by Drive quota
    replays.py       .rofl metadata only (payload is encrypted by Riot)
  ingest/core.py     Riot + Oracle's Elixir → SQLite
  features/          Stage 2 inference — THE VALUABLE PART
    geometry.py      Map constants, lane projection, self-correcting turrets
    waves.py         Minion spawn model + wave state (crash/freeze/slow push)
    lane.py          Trades, priority, recall value
    jungle.py        Path reconstruction + Markov next-camp prediction
    objectives.py    Spawn timers, objective control scoring
    macro.py         Death causes, fight clustering, positioning, rotations
    pipeline.py      Runs all of the above, persists `rationale` per row
  dataset/
    generators.py    Feature rows → instruction examples (uses the rationale)
    prepare.py       Dedupe, splits, dataset card with licence provenance
    synthetic.py     Stage 3 teacher-model explanations + validation
  rag/               Chunking, FAISS, hybrid retrieval
  train/             VRAM-aware hparams, QLoRA, resume, Unsloth preflight
  eval/
    scoring.py       Deterministic offline metrics (no API key needed)
    benchmark.py     1,000-question bank, paraphrase consistency
  cli.py             8 commands + status
tests/               76 tests; fixtures.py builds synthetic Riot timelines
```

Run: `python -m lolcoach.cli --help`. Commands: `download_data`,
`prepare_dataset`, `train`, `benchmark`, `chat`, `rag`, `update_patch`,
`status`.

---

## 4. Traps that already cost hours — do not rediscover these

1. **Transformers 5.5 breaking changes.** `apply_chat_template` returns a dict
   unless `return_dict=False`; `torch_dtype` → `dtype`;
   `TrainingArguments.overwrite_output_dir` removed; `warmup_ratio` deprecated.
   All four are handled — do not "simplify" them back.
2. **Unsloth cannot run here.** Its Triton kernels need a C compiler, and
   `unsloth_zoo` pins `torch<2.11` (this box has 2.11). It imports and loads
   fine, then dies minutes into training. `_unsloth_is_usable()` preflights
   this and falls back to PEFT + bitsandbytes. Leave the preflight in.
3. **Windows WDDM does not OOM.** When VRAM is oversubscribed the driver
   silently pages to system RAM and step time collapses (7s → 55s) with GPU
   utilisation at ~30%. That is why the 12 GB tier uses `max_seq_len=1024`.
   If you raise it, watch step time, not just whether it crashes.
4. **Patch numbering.** Data Dragon says `16.16`; players, Riot and the wiki
   say `26.16`. `public_patch_label()` converts. Getting this wrong silently
   breaks every patch-notes fetch and patch filter.
5. **Both nexuses sit inside the top and bottom lane corridors.** Without the
   `at_fountain()` check a recalled player counts as "in lane" and drags the
   wave estimate to the lane midpoint. There is a regression test.
6. **Riot puts the objective's team in `killerTeamId`, not `teamId`.** Reading
   only `teamId` silently drops every dragon/baron/herald.
7. **Health lives in `championStats.health`,** not on the participant frame.
   Reading the wrong field zeroes every trade, recall and death feature
   without erroring.
8. **Another agent was editing this repo concurrently** and left a truncated
   statement and a duplicated file region behind. If something looks
   half-written, it probably is. `python -m compileall -q lolcoach` and the
   test suite catch it.

---

## 5. Priority 1 — unlock the data

### 5.1 Riot API onboarding (highest value per hour of work)

Everything in `features/` is dead code until a key exists. Build
`lolcoach init`:

- Open the Riot developer portal, prompt for the key, validate it with one
  live call, write it to `~/.lolcoach.env`.
- Detect the player: read `Documents/League of Legends/Replays/*.rofl` (the
  parser already extracts PUUIDs) or ask for Riot ID and resolve via
  account-v1.
- Warn that development keys expire every 24 hours and offer to re-prompt;
  detect the 403 and say so in one clear sentence (already implemented in
  `RiotClient.validate`).

Acceptance: a non-engineer goes from clone to a populated `wave_states` table
in under five minutes.

### 5.2 Live Client Data API — the actual killer feature

While a game is running, the League client serves
`https://127.0.0.1:2999/liveclientdata/allgamedata` **with no API key and no
rate limit**. It gives live champion stats, items, scores, events and game
time for the local player.

This turns a post-game analyser into a live coach. Build
`lolcoach live`:

- Poll the endpoint (self-signed cert; pin Riot's or disable verification for
  localhost only, and say which you did).
- Feed the live state into the *existing* `DecisionContext` and
  `recommend()` in `features/waves.py` — the decision engine already takes
  health, gold, level, objective timers and jungler distance.
- Surface one line of advice, updated every few seconds.

Caveat to respect: the live endpoint exposes only the local player's full
data plus public scoreboard info; it does not give enemy positions. Be honest
about what can and cannot be inferred, the way the rest of the codebase is.

### 5.3 Oracle's Elixir

Blocked by a Google Drive per-file quota, not by our code. The connector
detects the quota page specifically and leaves the file for a later run. Retry
periodically; consider a mirror.

---

## 6. Priority 2 — interface

The whole thing is a CLI today. Pick **one** of these and do it properly.

### 6.1 Match review web app (recommended)

Local FastAPI + a small React or HTMX front end, launched by
`lolcoach ui`:

- **Game picker**: last 20 games for the configured PUUID, one click to
  ingest + derive.
- **Timeline scrubber**: drag through the game; the minute-by-minute derived
  state updates live.
- **Minimap canvas**: draw champion positions, the inferred wave position on
  each lane axis, jungle camp clears, and the Markov-predicted next camp as a
  probability heat spot. All of this data already exists in `wave_states`,
  `jungle_paths` and `frames`.
- **Moment cards**: every death, objective and recall as a card with the
  stored `rationale` rendered as prose, linked to its timestamp.
- **Ask-about-this-moment**: a chat box pre-loaded with that timestamp's
  context block, answered by the fine-tuned model + RAG.
- **Confidence rendering**: every inferred value carries a confidence in the
  DB. Show it. A greyed-out or dashed wave marker at 0.4 confidence teaches
  the user to trust the tool appropriately, which is the whole design
  philosophy of the data layer.

### 6.2 In-game overlay

Higher ceiling, more work. Overwolf is the sanctioned route for League
overlays. Pair with 5.2.

### 6.3 Minimum viable alternative

If neither fits the time budget: a `rich`-based TUI with the timeline
scrubber and moment cards. Much better than nothing.

---

## 7. Priority 3 — usability

- **`lolcoach doctor`** — one command that checks Python version, CUDA and
  VRAM, installed extras, credentials present, disk space, model cache,
  index freshness vs current patch, and dataset staleness. Print a table of
  ✓/✗ with the exact fix for each ✗.
- **Streaming chat output.** `chat` currently blocks for ~25 s then dumps a
  wall of text. Use `TextIteratorStreamer`.
- **Model size auto-selection.** 12 GB fits 7B at 1024 tokens uncomfortably.
  Offer Qwen2.5-3B or 1.5B for faster iteration and 14B for 24 GB cards.
  `choose_training_plan()` already tiers by VRAM; extend it to pick the model.
- **GGUF export** so the coach runs on CPU / llama.cpp / LM Studio without a
  GPU. This is what makes it shareable.
- **`lolcoach review last`** — one command: pull my most recent game, ingest,
  derive, print the five biggest mistakes with timestamps.
- **Packaging.** `pipx install lolcoach`, or a Windows installer. League
  players are on Windows and will not create a venv.
- **Progress and ETA everywhere.** The download and feature stages can run for
  hours; `rich.progress` with per-source ETA.

---

## 8. Priority 4 — model quality

Ordered by expected gain:

1. **Get timelines** (section 5.1). Nothing else matters as much. Target
   ≥5,000 matches → roughly 200k–400k timeline-derived examples.
2. **Fix vagueness.** `checkable_claims 0.0` says the model never commits to a
   number. Add a specificity term: reward answers citing a timer, a gold
   amount or a wave count. The dataset generators already emit concrete
   numbers — the static-knowledge examples are diluting them. Rebalance the
   archetype mix once timeline data exists (currently 68% item knowledge).
3. **Preference tuning.** `recalls.was_good` and `rotations.was_correct` are
   already computed outcome labels. Build DPO pairs from them: same context,
   the decision that worked vs the one that did not. This is free preference
   data most projects would have to pay annotators for.
4. **Retrieval-augmented training.** Train with retrieved context in the
   prompt so the model learns to *use* citations rather than ignore them.
5. **Rank conditioning.** Tag examples with the tier they came from and let
   the user ask "what would a Challenger do here" versus "what should I do at
   Gold".
6. **Run Stage 3.** `prepare_dataset --synthetic` is implemented and validated
   but has never been run with a real teacher. With an Anthropic or OpenAI key
   it will materially improve answer style. The validator rejects answers that
   contradict the label, invent champions or invent numbers.

---

## 9. Ground rules

These were the standing instructions for this build and they should continue:

- **No placeholders, no TODOs, no mocked functionality.** If something cannot
  be done, say so plainly and implement the honest alternative. Example: the
  `.rofl` payload is encrypted with keys Riot does not publish, so the replay
  connector extracts the plaintext metadata and documents why that is all it
  can do — it does not pretend to parse the payload.
- **Verify against reality.** Every source connector in this repo was tested
  against the live endpoint before being trusted. Four upstreams had moved or
  changed since their last public documentation.
- **Confidence is a first-class output.** Timeline frames are 60 seconds
  apart; wave state is inferred, not observed. Every derived row carries a
  confidence and low-confidence rows are dropped rather than dressed up as
  advice. Keep this property — it is the difference between a coach and a
  bluffer.
- **Respect the licences.** The dataset card is generated mechanically from
  the download ledger. Riot API data and Oracle's Elixir data are not
  redistributable; the wiki is CC BY-SA and needs attribution. Do not add a
  source that bulk-scrapes VODs or bypasses authentication.
- **Run the tests.** `python -m pytest` (76 tests). The feature tests script
  situations with known answers — a freeze on top, a slow push into a crash on
  mid — and assert the estimator recovers them. If you change the estimators,
  those tests are how you know you did not break the thing that makes this
  project worth anything.
