# Distributing the build

The finished bundle is about 22.6 GB. This document explains why, what the
options are for hosting it, and how to make a smaller one.

## What is in the 22.6 GB

| Part | Size | Could it be downloaded later? |
| --- | --- | --- |
| `models/` (Qwen2.5-7B plus the embedding model) | 15 GB | Yes, from Hugging Face on first launch |
| `_internal/` (Python, PyTorch, CUDA libraries) | 5.5 GB | No, PyInstaller needs it present |
| `data/` (your matches, RAG index, trained adapter) | 2 GB | Partly, the adapter and index are small |
| `LoLCoach.exe` | 0.12 GB | No |

## Why not a GitHub release

GitHub Releases allow unlimited assets but cap **each file at 2 GB**. Even with
the model stripped out, the runtime alone is 5.5 GB uncompressed and roughly
2.5 GB compressed, which is still over the cap. Splitting into parts works but
asks every user to rejoin files before running anything.

Git LFS is worse for this: the same 2 GB file cap applies and the free
bandwidth quota is 1 GB per month, which one download would exhaust.

## Options that do work

**Hugging Face Hub, recommended.** Free, designed for multi gigabyte files,
CDN backed, no egress charges. It is also where the model already lives, so
the payload is in its natural home. Create a free token with write access at
`huggingface.co/settings/tokens`, then:

```powershell
python -m pip install huggingface_hub
$env:HF_TOKEN = "your-write-token"
python -c "from huggingface_hub import HfApi; HfApi().create_repo('Mattbusel/lolcoach-windows', repo_type='model', exist_ok=True)"
python -c "from huggingface_hub import HfApi; HfApi().upload_large_folder(repo_id='Mattbusel/lolcoach-windows', repo_type='model', folder_path='dist/LoLCoach')"
```

Then link that repo from the README download button.

**Cloudflare R2.** 10 GB free storage and zero egress fees, which matters if
the download gets popular. Needs a Cloudflare account with a card on file.

**Backblaze B2.** 10 GB free, very cheap egress, no card needed to start.

**Avoid Google Drive.** It enforces a per file download quota that trips
quickly on popular large files. This project already hit exactly that problem
when fetching Oracle's Elixir data.

## Making the download much smaller

Two independent reductions, worth doing before hosting anything.

**Drop the bundled model (22.6 GB to 7.6 GB).** The app already reads the
Hugging Face cache, so removing `models/` makes it download Qwen on first
launch instead. Build with `-NoModels` and delete the folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1 -Clean -NoModels
Remove-Item -Recurse -Force dist\LoLCoach\models
```

The trade off is that the first launch needs an internet connection and about
15 GB of download.

**Ship review only (7.6 GB to a few hundred MB).** Most of the remaining size
is PyTorch and CUDA, which exist solely for the chat. A build that excludes
`torch`, `transformers`, `peft` and `faiss` still gives the full match review:
minimap, gold curve, wave ribbon, coaching moments and progress. Only the
conversational coach is missing. That artifact fits in a GitHub release
comfortably and is the right default for most people, with the full bundle
offered separately for anyone who wants the AI.

This split is not implemented yet. It needs a second PyInstaller spec with
those modules excluded and a guard in the UI that hides the chat panel when
the coach is unavailable, which the app already reports through `/api/coach`.
