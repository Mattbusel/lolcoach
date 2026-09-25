# Changelog

## [1.1.0] - 2026-09-25

- New small download: the review app as a single-file executable for Windows (`LoLCoach.exe`), macOS (Apple Silicon and Intel) and Linux, attached to every GitHub Release. About 20 MB instead of 22 GB. It has the full match review (your games, minimap, gold curve, wave states, deaths, coaching moments, progress) and leaves out only the AI chat, which needs PyTorch and the 15 GB model. The full bundle with the chat stays on Hugging Face.
- When the AI stack is not installed, the coach says so plainly ("not in this download") instead of showing an import error.
- The launcher takes `--port`, `--no-browser`, `--version` and `--help`, and prints the local address and data folder when it starts.
- CI runs the test suite on Windows and Linux without the model stack.

## [1.0.0]

- Local match review, feature inference (waves, jungle paths, death causes, objectives), RAG, QLoRA fine-tuning and the full Windows bundle on Hugging Face.
