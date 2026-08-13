# LoLCoach

**A League of Legends coach that runs entirely on your PC.**

It downloads your own ranked games from Riot, works out what actually happened in them (wave states, jungle paths, why each death happened, where the game swung), then lets you talk to a fine-tuned AI about it for as long as you like. No subscription, no cloud, no per-message cost. The model is on your machine.

### [Download LoLCoach for Windows](https://huggingface.co/Fungle/lolcoach-windows)

*Roughly 22.6 GB. Runs offline once downloaded.*

![The review screen](docs/screenshots/01-review.png)

---

## What makes it different

Most League tools show you statistics you already knew. LoLCoach reconstructs the decisions.

Riot's match timeline gives one snapshot per minute. From that, LoLCoach infers the things that actually decide games:

* **Wave states.** Whether a wave was freezing, slow pushing, crashing or bouncing, from champion positions, minion arithmetic and the exact spawn schedule.
* **Why you died.** Overextended, caught rotating, outnumbered, dived, no vision, or lost the trade, each with the evidence behind it.
* **Jungle paths.** Which camps were cleared and, from a Markov model built across your games, where that jungler probably went next.
* **Objective control.** Whether a dragon was taken for free or on a coinflip, scored from presence, vision and the fight that preceded it.
* **Where the game swung.** The single largest gold swing against you, and what caused it.

Every inference carries a confidence, because a one minute sampling grid genuinely cannot see everything. Low confidence renders as a low confidence bar, not as a confident sentence. A coach that bluffs is worse than no coach.

## Talk to it

The AI runs locally, so conversation is unlimited and costs nothing. Ask anything about any moment in any of your games.

![Asking the coach](docs/screenshots/02-chat.png)

When a game opens the coach already has something to say. It surfaces what is worth reviewing before you type a word, and each one is a button that asks the question for you.

## See your habits, not just your games

One game is an anecdote. LoLCoach aggregates every game you have played, finds the mistake that costs you most often, and tells you whether it is getting better or worse.

![Progress across games](docs/screenshots/03-progress.png)

## Make it yours

Three coach personalities and four accent themes. The persona changes the voice only, never the standard of evidence.

![Settings and personality](docs/screenshots/05-settings.png)

---

## Download

The full bundle contains the app, the Python runtime, the trained adapter, the retrieval index and the Qwen2.5-7B model. It is roughly 22 GB because the model ships inside it, which is what lets it work offline forever after the first launch.

**[Download from Hugging Face](https://huggingface.co/Fungle/lolcoach-windows)**, or pull it from the command line:

```bash
pip install huggingface_hub
huggingface-cli download Fungle/lolcoach-windows --local-dir LoLCoach
```

Then run `LoLCoach\LoLCoach.exe`. It ships with an empty match library, so you sync your own games on first run.

It is hosted on Hugging Face rather than a GitHub release because GitHub caps release assets at 2 GB per file. See [DISTRIBUTION.md](DISTRIBUTION.md) for the other options and for how to build a much smaller download.

**Build it yourself.** One command, needs Python 3.12 and an NVIDIA GPU:

```powershell
git clone https://github.com/Mattbusel/lolcoach.git
cd lolcoach
python -m pip install -e ".[ui,bundle]"
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

The bundle lands in `dist\LoLCoach\`. Double click `LoLCoach.exe`.

## System requirements

The app checks your own machine. Open it and click **Requirements**.

![System check](docs/screenshots/04-requirements.png)

**To review your games**, no GPU needed:

| | |
| --- | --- |
| Operating system | Windows 10 or 11, 64 bit |
| Memory | 8 GB RAM |
| Disk | 8 GB free, growing with the games you collect |
| Riot API key | Free from [developer.riotgames.com](https://developer.riotgames.com). A development key expires every 24 hours |

**To chat with the AI coach:**

| | |
| --- | --- |
| GPU | NVIDIA with 8 GB VRAM or more, RTX 3060 or 4060 and up |
| Memory | 16 GB RAM |
| Disk | 20 GB free for the bundled model |
| Driver | Recent NVIDIA driver with CUDA 12 support |

Without a supported GPU the full match review still works. Only the chat is slow or unavailable.

## Getting started

1. Launch the app. It opens at `http://127.0.0.1:8765` and serves only on your machine.
2. Paste a Riot API key. It is validated with Riot and saved only on this PC.
3. Add your Riot ID as `GameName#TAG`. LoLCoach works out which region serves your history.
4. Press **Sync my games**. Your matches download, features are derived, and the review fills in.

## How it works

```
Riot API  ->  SQLite  ->  feature inference  ->  training data  ->  QLoRA fine tune
                              |                                          |
                              +--------> review UI <---- RAG <-----------+
```

| Stage | What happens |
| --- | --- |
| Collect | Riot match timelines, Data Dragon static data, the League wiki, patch notes, community discussion |
| Infer | Wave states, lane priority, trades, recalls, jungle paths, objective control, death causes, rotations |
| Build | Instruction examples assembled from the recorded reasoning, deduplicated and split |
| Retrieve | FAISS index over the corpus, hybrid dense plus IDF weighted lexical, patch aware |
| Train | QLoRA on Qwen2.5-7B, VRAM aware hyperparameters, per epoch checkpoints, resumable |
| Review | Local web app with minimap, gold curve, wave ribbon, coaching moments and streaming chat |

Command line:

```
lolcoach download_data      Collect from every configured source
lolcoach prepare_dataset    Ingest, infer features, write train/val/test JSONL
lolcoach train              Fine tune the adapter
lolcoach benchmark          Score against GPT, Claude and DeepSeek
lolcoach chat               Ask a question from the terminal
lolcoach rag --build        Rebuild the retrieval index
lolcoach update_patch       Refresh static data after a patch
lolcoach ui                 Open the review app
lolcoach doctor             Check this machine
```

## Honest limitations

* **Live coaching is limited.** Riot's Live Client Data API does not expose champion coordinates, so wave position and jungle tracking cannot be inferred during a game. Post game review has full fidelity.
* **A development Riot key expires every 24 hours.** Production keys require an application to Riot.
* **The wave estimate is an estimate.** Timeline frames are 60 seconds apart, which is exactly why confidence is shown.
* **The model is small.** A 7B model fine tuned on your games beats frontier models on the specific decisions it was trained on and loses to them on open ended League conversation.

## Data and licensing

The MIT licence covers this repository's code only. It does not grant rights to redistribute models, collected data or game assets.

| Source | Use | Restriction |
| --- | --- | --- |
| Riot Data Dragon and Riot API | Static data, your authorised match history | Riot terms apply. Riot does not endorse this project |
| League of Legends Wiki | Mechanics and patch history | CC BY-SA 4.0, attribution preserved |
| Oracle's Elixir | Professional match data | Non commercial use with credit |
| Reddit via Arctic Shift | Community coaching discussion | Content remains its authors' property |
| Cinzel typeface | Display headings | SIL Open Font License, included in `lolcoach/web/fonts/` |

Every downloaded file is recorded with its URL, hash, source and licence. The dataset card is generated from that ledger rather than written by hand.

This project does not scrape VODs, does not bypass platform authentication and does not decrypt replay files.

## Development

```powershell
python -m pip install -e ".[ui,judges]"
python -m pytest          # 87 tests
python -m lolcoach.cli ui
```

The tests that matter most are in `tests/test_features.py`. They script game situations with known answers, a freeze on top and a slow push into a crash on mid, then assert the estimators recover them.

A deeper technical write up of the pipeline is in [docs/README-technical.md](docs/README-technical.md).

## Licence

MIT for the code. See [LICENSE](LICENSE).

League of Legends is a trademark of Riot Games, Inc. This project is not affiliated with or endorsed by Riot Games.
