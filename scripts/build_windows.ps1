[CmdletBinding()]
param(
    [string]$Python = "python",
    # Rebuild the PyInstaller runtime from scratch. Without this the script
    # reuses an existing bundle and only refreshes data, models and assets.
    [switch]$Clean,
    # Skip copying the base model and embedding model (much faster iteration
    # when you only changed application code or the UI).
    [switch]$NoModels,
    # Produce dist\LoLCoach-windows-x64.zip as well as the folder.
    [switch]$Zip
)

# One script, one output: dist\LoLCoach.
#
# It produces a single self-contained bundle containing the executable, the
# Python runtime, the review UI, local match data, the RAG index, the trained
# adapter, and the base + embedding models. Riot credentials are never copied:
# they live outside the data directory and each user adds their own.
#
# This replaces the earlier split between a "base" build, a "full" packaging
# pass and a separate archive step, which produced several partial folders
# that were easy to confuse and easy to ship the wrong one of.

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$build = Join-Path $root "build"
$dist = Join-Path $root "dist"
$bundle = Join-Path $dist "LoLCoach"
$data = Join-Path $root "data"
$hfRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub"

& $Python -c "import fastapi, uvicorn, torch, transformers, sentence_transformers, faiss, PyInstaller"
if ($LASTEXITCODE -ne 0) {
    throw "The Windows bundle needs the UI, model, RAG, and PyInstaller dependencies. Run: $Python -m pip install -e '.[ui,bundle]'"
}

# ---------------------------------------------------------------- runtime --
# The model and data folders live inside the bundle but are not produced by
# PyInstaller, which clears its dist directory. Move them aside rather than
# copying: on one volume this is instant even for a 15 GB model cache.
$stash = Join-Path $dist "_stash"
$carried = @()
if (Test-Path -LiteralPath $bundle) {
    New-Item -ItemType Directory -Path $stash -Force | Out-Null
    foreach ($name in @("models", "data")) {
        $source = Join-Path $bundle $name
        if (Test-Path -LiteralPath $source) {
            Move-Item -LiteralPath $source -Destination (Join-Path $stash $name) -Force
            $carried += $name
        }
    }
}

try {
    if ($Clean) {
        Remove-Item -LiteralPath $build -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $bundle -Recurse -Force -ErrorAction SilentlyContinue
    }

    if ($Clean -or -not (Test-Path -LiteralPath (Join-Path $bundle "LoLCoach.exe"))) {
        $pyArgs = @(
            "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
            "--name", "LoLCoach", "--paths", $root,
            "--distpath", $dist, "--workpath", $build, "--specpath", $build,
            "--add-data", "$root\configs;configs",
            # The review UI is served from real static files, so they must ship
            # inside the bundle or the packaged app renders a blank page.
            "--add-data", "$root\lolcoach\web;lolcoach/web",
            "--collect-all", "fastapi", "--collect-all", "starlette", "--collect-all", "uvicorn",
            "--collect-all", "sentence_transformers", "--collect-all", "transformers",
            "--collect-all", "tokenizers", "--collect-all", "peft", "--collect-all", "torch",
            "--collect-all", "safetensors", "--collect-all", "faiss",
            # pywebview is bundled so `--native` can work, but the packaged app
            # opens a browser by default: inside a frozen build the .NET loader
            # probes the executable directory rather than _internal\webview\lib,
            # so WebView2 assembly resolution is unreliable. The app falls back
            # automatically either way.
            "--collect-all", "webview",
            # LoLCoach performs text-only PyTorch inference. Excluding
            # TensorFlow avoids carrying a second, unused 1+ GiB runtime.
            "--exclude-module", "tensorflow", "--exclude-module", "keras",
            "--exclude-module", "tensorboard",
            "--hidden-import", "lolcoach.chat", "--hidden-import", "lolcoach.rag",
            "--hidden-import", "lolcoach.sources", "--hidden-import", "lolcoach.doctor",
            "--hidden-import", "lolcoach.assets", "--hidden-import", "lolcoach.review",
            (Join-Path $root "lolcoach\launcher.py")
        )
        & $Python @pyArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Write-Host "Reusing the existing runtime. Pass -Clean to rebuild it."
    }
} finally {
    # Always put the carried folders back, even if PyInstaller failed, so an
    # aborted build never destroys a 15 GB model cache.
    foreach ($name in $carried) {
        $stashed = Join-Path $stash $name
        if (Test-Path -LiteralPath $stashed) {
            $target = Join-Path $bundle $name
            if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
            New-Item -ItemType Directory -Path $bundle -Force | Out-Null
            Move-Item -LiteralPath $stashed -Destination $target -Force
        }
    }
    Remove-Item -LiteralPath $stash -Recurse -Force -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------------- data --
if (-not (Test-Path -LiteralPath (Join-Path $data "lolcoach.sqlite"))) {
    throw "Expected local data at $data. Run a collection first."
}
Write-Host "Syncing local data into the bundle..."
# /MIR mirrors incrementally, so repeat builds copy only what changed.
# Credentials live outside the data directory and are never copied; logs and
# the HTTP cache are excluded because they are large and worthless to a user.
robocopy $data (Join-Path $bundle "data") /MIR /NFL /NDL /NJH /NJS /NP /XD logs cache | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copying local data failed (robocopy exit $LASTEXITCODE)." }

# ----------------------------------------------------------------- models --
if (-not $NoModels) {
    if (-not (Test-Path -LiteralPath $hfRoot)) { throw "Hugging Face cache is missing at $hfRoot." }
    $portableHub = Join-Path $bundle "models\huggingface\hub"
    New-Item -ItemType Directory -Path $portableHub -Force | Out-Null
    foreach ($name in @("models--Qwen--Qwen2.5-7B-Instruct", "models--sentence-transformers--all-MiniLM-L6-v2")) {
        $source = Join-Path $hfRoot $name
        if (-not (Test-Path -LiteralPath $source)) { throw "Required model cache is missing: $source" }
        Write-Host "Syncing $name..."
        robocopy $source (Join-Path $portableHub $name) /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "Copying $name failed (robocopy exit $LASTEXITCODE)." }
    }
} else {
    Write-Host "Skipping model sync (-NoModels)."
}

# ------------------------------------------------------------------ notes --
@"
LoLCoach
========

Double-click LoLCoach.exe. It opens the local review app in your browser at
http://127.0.0.1:8765 and serves only on this machine.

This folder is self-contained: executable, Python runtime, review UI, your
match data, the RAG index, the trained adapter, and the base and embedding
models. Keep the whole folder together; the .exe alone will not run.

Riot credentials are NOT included. Add your own key on first launch.

SYSTEM REQUIREMENTS
-------------------
To review your games (no GPU needed):
  Windows 10/11 64-bit, 8 GB RAM, 8 GB free disk, internet connection,
  and a free Riot API key from developer.riotgames.com.

To chat with the AI coach:
  NVIDIA GPU with 8 GB VRAM or more (RTX 3060/4060 and up), 16 GB RAM,
  20 GB free disk for the bundled model, recent driver with CUDA 12.
  The model runs on your PC, so conversation is unlimited and costs nothing.

Without a supported GPU the full match review still works; only the chat
is slow or unavailable.

The app can check your own machine: open it and click Requirements.

Data location:
  Launched from inside the project checkout, it uses the project's data folder.
  Copied elsewhere, it uses the bundled data folder beside the .exe.
  Set LOLCOACH_HOME to override either.

Windows may show a SmartScreen warning for an unsigned locally built app.
Code-sign the executable before distributing it to anyone else.
"@ | Set-Content -LiteralPath (Join-Path $bundle "README.txt") -Encoding UTF8

Remove-Item -LiteralPath (Join-Path $bundle "PORTABLE_MODEL_INCLUDED.txt") -Force -ErrorAction SilentlyContinue

# ------------------------------------------------------------------- zip ---
if ($Zip) {
    $archive = Join-Path $dist "LoLCoach-windows-x64.zip"
    Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
    Write-Host "Compressing (this bundle is large; expect a long run)..."
    Compress-Archive -Path $bundle -DestinationPath $archive -CompressionLevel Optimal
    $gb = [math]::Round((Get-Item -LiteralPath $archive).Length / 1GB, 2)
    Write-Host "Archive: $archive ($gb GB)"
}

$size = [math]::Round(((Get-ChildItem -LiteralPath $bundle -Recurse -File | Measure-Object -Property Length -Sum).Sum) / 1GB, 2)
Write-Host ""
Write-Host "Built: $(Join-Path $bundle 'LoLCoach.exe')"
Write-Host "Bundle size: $size GB"
