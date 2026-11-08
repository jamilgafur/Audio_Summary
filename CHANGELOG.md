# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-07

First public release.

### Added
- Paper → podcast pipeline: PDF/TXT/TeX/DOCX extraction, local-LLM narration (summary or verbatim mode), Kokoro TTS, live progress, crash recovery.
- Web UI: Library (upload, arXiv/URL fetch, search, tag filters, playlists, bulk actions), Voices (voice/mode defaults + preview), Feeds (base + per-tag RSS), and the Write tab — a verbatim read-aloud editor with play/pause/sentence-skip, live highlighting, and click-to-seek.
- Reader/editor tab (`/view`): follow-along sentence highlighting, script editor with TTS preview (Ctrl+Enter) and save-and-re-render.
- Per-paper voice selection (card menu + reader tab).
- First-run onboarding: model-download progress line on the Library, a one-click sample paper for the empty state, and error→hint mapping on failed cards.
- Per-piece TTS render cache: edits re-render only the changed passages; repeated listens replay instantly.
- `docker compose` deployment with a GPU (CUDA) build path and a CPU-only build path.
- Robustness & licensing: the folder watcher skips unprocessable filenames instead of stalling; BUSL-1.1 notice headers on all source files; `llama-cpp-python` pinned (0.3.36) and covered by CI's pip-audit via `requirements-audit.txt`.
- Open-source project scaffolding: BUSL-1.1 license (converts to Apache 2.0 on 2030-11-07), CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue/PR templates, Dependabot, and a CI/CD pipeline (lint, format, tests on 3.10–3.12, pip-audit, JS syntax check, CPU Docker build, GHCR release on tags).
