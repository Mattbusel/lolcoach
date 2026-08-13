LoLCoach desktop bundle
=======================

Double-click LoLCoach.exe. It opens a local-only review site at:
http://127.0.0.1:8765

The application stores user-specific data at:
%LOCALAPPDATA%\LoLCoach\data

To use a portable data folder, put a data directory beside LoLCoach.exe or
set LOLCOACH_HOME before launching it.

This full portable bundle includes the application, CUDA/PyTorch runtime,
Qwen base model, embedding model, trained adapter, RAG index, provenance,
and collected match data. Riot credentials are intentionally excluded.

To use it:
1. Unzip the complete folder.
2. Double-click LoLCoach.exe.
3. Paste your own Riot developer key in the first-time setup panel.
4. Click Collect my games.
5. Choose a game under Your games and click Review moment on any tip.

The entire folder must stay together. The bundle is local-only and stores
newly collected data beside the executable.

Windows may show a SmartScreen warning for an unsigned locally built app.
Code-sign the executable before public distribution.
