# Paper Reader

Drop a research paper (PDF, TXT, TeX, or Word) in — get an MP3 you can listen to, out. A local LLM (Qwen2.5-7B by default) reads the paper and writes a *spoken narration* script — ideas explained, equations/tables/citations skipped — then [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) TTS renders it to audio. Plus a read-aloud editor, follow-along highlighting, per-paper voices, and podcast RSS feeds.

Everything runs on your machine by default: **no cloud, no telemetry, no API keys.** Prefer a hosted model? Bring your own key — narrate through any OpenAI-compatible API while PDF parsing, audio rendering, and everything else stay local.

[![CI](https://github.com/jamilgafur/Audio_Summary/actions/workflows/ci.yml/badge.svg)](https://github.com/jamilgafur/Audio_Summary/actions/workflows/ci.yml)

## Features

- **Paper → podcast pipeline** — PDF / TXT / TeX / DOCX in, MP3 out, with live progress, crash recovery, and retry.
- **Two narration modes** — *summary* (LLM explainer, the default) and *verbatim* (the paper's exact words, no LLM — for reviewing your own drafts aloud).
- **Bring your own key (BYOK)** — narrate with any OpenAI-compatible chat API (OpenAI, OpenRouter, Ollama, LM Studio, …) instead of the local GGUF: base URL + model + key in the Settings tab, with a one-click connection test. The key is stored server-side and never echoed back.
- **Web UI** with in-app tabs: **📚 Library** (upload, arXiv/URL paste, search, tag filters, playlists), **✍️ Write** (a verbatim read-aloud editor — play/pause/sentence-skip with live highlighting, click-to-seek, and pause→edit→resume that re-renders only your changed sentences), **⚙️ Settings** (voice/mode defaults, BYOK engine + connection test), **📡 Feeds** (per-tag podcast RSS with copy buttons).
- **Settings that follow you** — voice, mode, and narration engine are stored on the server (`output/settings.json`), so every device that opens the page gets the same defaults — including papers dropped into `input/`.
- **Cleaner PDF text** — column-aware reading order for 2-column papers (no merged left/right lines), words rejoined across line wraps, plus table rows, page numbers, and citations stripped before narration.
- **LaTeX drafts read as speech** — the Write tab and `.tex` uploads speak the words, not the markup: `\%` reads “percent”, a bare `%` comments out the rest of the line, `\cite`/`\ref`/`\label`/`\url` and math are skipped, and `\textbf{…}` & friends unwrap to their text.
- **Reader/editor tab** — "read & listen" opens each paper in its own browser tab: the document, the audio, sentence-level follow-along, and a script editor with TTS preview (Ctrl+Enter = listen from ~2 sentences back).
- **Per-paper voice** — every paper carries its own voice; change it from the card menu or the reader tab and only the audio re-renders (a per-piece TTS cache means small edits re-render in seconds).
- **Podcast feeds** — point any podcast app (or Audiobookshelf) at the base feed or one filtered by tag.
- **Fast by construction** — every rendered sentence is disk-cached by content; repeated listens are instant, and re-renders after a one-sentence edit touch one piece, not the whole file.
- **First-run onboarding** — while models are missing, the Library shows what is downloading and how far along it is; an empty library offers a one-click sample paper that runs the whole pipeline; failed cards say what to do about the failure (scanned PDF, VRAM, network).

## How it works

```
input/ ──watcher──▶ queue ──▶ worker (single thread)
                         │
        extract ──▶ narrate ──▶ TTS ──▶ output/<slug>.mp3 + follow.json
        (pymupdf/   (GGUF child    (Kokoro,
         txt/tex/   proc, or       per-sentence
         docx)      BYOK API)      pieces)
```

- One worker thread processes jobs sequentially; status lives in `<output>/<stem>.status.json`, so a crash mid-job re-queues the paper on restart (twice at most, then it's marked failed).
- Narration runs the GGUF model in a **spawned child process** — llama.cpp can hard-abort; a crash costs one automatic CPU retry instead of killing the app. With BYOK configured, narration is a plain HTTPS call to your API instead and the child process is skipped.
- TTS renders sentence-sized pieces, caches each on disk by (model, voice, speed, text), and writes exact sentence→time marks (`<stem>.follow.json`) for follow-along.
- The UI is vanilla JS (no framework, no build step) served by FastAPI; it polls `/api/jobs` every 2 s while a job is busy.

## Prerequisites

- Docker + the Docker Compose plugin.
- **GPU (optional, recommended):** NVIDIA driver ≥ 525 and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/container-toolkit/install-guide.html). Tested on a GTX 1660 Ti (6 GB) / 48 GB RAM / Intel i5 — the LLM runs on the GPU (~20–30 tok/s, a script in a few minutes); TTS runs on CPU.
- CPU-only works too — script generation just takes ~20–30 min per paper.

## Quick start

```bash
git clone <your-repo-url> && cd <repo>
docker compose up --build   # first build: ~10 min (compiles llama.cpp for your GPU)
```

Then open <http://localhost:8123> and upload (or drag-drop) a paper. The provided `docker-compose.yml` is tuned for a GTX 1660 Ti (CUDA arch 7.5, 12 GPU layers, 2 CPU threads) — adjust the `build.args` and env block for your hardware (see below).

### CPU-only

```bash
docker build -t paper-reader --build-arg BASE=python:3.12-slim --build-arg CMAKE_ARGS="" .
docker run -p 8000:8000 \
  -v "$PWD/input:/data/input" -v "$PWD/output:/data/output" -v "$PWD/models:/data/models" \
  paper-reader
```

### GPU image

The compose default builds the CUDA image. The one hardware-specific knob is the CUDA architecture in `build.args` (compute capability per [NVIDIA's table](https://developer.nvidia.com/cuda-gpus) — e.g. 7.5 for Turing, 8.6 for Ampere, 8.9/9.0 for Ada/Hopper):

```yaml
CMAKE_ARGS: "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75"
```

First run downloads ~5 GB of models into `models/` (mounted, so it happens once).

## Configuration

Configuration is environment variables (see `docker-compose.yml` for a working annotated set) plus the **⚙️ Settings** tab, which stores its picks in `output/settings.json` — Settings-tab values override the env defaults below.

| Var | Default | Notes |
|---|---|---|
| `INPUT_DIR` / `OUTPUT_DIR` / `MODELS_DIR` | `./input` / `./output` / `./models` | Remapped to `/data/*` by the compose file. |
| `GGUF_REPO` / `GGUF_FILE` | Qwen2.5-7B-Instruct Q4_K_M | Any instruct GGUF works. A 14B fits on CPU with plenty of RAM (`N_GPU_LAYERS=0`). |
| `N_GPU_LAYERS` | `-1` (all on GPU) | Lower it if the GPU OOMs while sharing VRAM with other services; `0` = CPU only. |
| `LLM_BASE_URL` | unset | **BYOK:** an OpenAI-compatible endpoint, e.g. `https://api.openai.com/v1`. Setting it switches narration to the API engine. |
| `LLM_MODEL` | unset | **BYOK:** model id served by that endpoint (`gpt-4o-mini`, `qwen2.5:14b`, …). |
| `LLM_API_KEY` | unset | **BYOK:** bearer key for the endpoint; optional for keyless local gateways (Ollama, LM Studio). |
| `VOICE` | `af_heart` | Kokoro voices, e.g. `am_adam`, `bf_emma`, `bm_george`. Per-paper overrides win over this default. |
| `SPEED` | `1.1` | TTS speed; the web player also has its own per-job speed control. |
| `KOKORO_MODEL` | `kokoro-v1.0.onnx` | `kokoro-v1.0.fp16.onnx` halves model RAM (~177 MB vs ~325 MB). Bench here: ~5% faster — only worth it if RAM is tight. |
| `NARRATE_MOCK` | unset | `1` = skip the LLM entirely (dev/test; scripts are placeholder text). |
| `HF_HOME` | unset | Where model weights cache (the compose file points it inside the mounted models dir). |

The port (default `8123` in the compose file) is a deliberate non-default: change it to anything free on your box.

### Bring your own key (BYOK)

Prefer a hosted model (or a local gateway like Ollama / LM Studio) over the bundled GGUF? Open **⚙️ Settings → Narration engine** and pick **OpenAI-compatible API**:

- **base URL** — the endpoint root: `https://api.openai.com/v1`, `https://openrouter.ai/api/v1`, `http://ollama:11434/v1`, … The `/chat/completions` path is appended automatically (full endpoint URLs work too).
- **model** — any id that endpoint serves (`gpt-4o-mini`, `openai/gpt-4o-mini`, `qwen2.5:14b`, …).
- **API key** — sent as `Authorization: Bearer …`. Optional for keyless local gateways. Blank on save keeps the stored key; ✕ clear removes it.
- **test connection** — one tiny request to verify URL / model / key before committing a paper to it.

What stays local either way: PDF parsing, the Kokoro TTS render, follow-along marks, and the whole web UI. What leaves the machine: only the paper excerpts sent to *your* chosen endpoint for narration — roughly one chat completion per 1200 words at ~1400 output tokens. Jobs narrated this way show “· via API” in their progress line.

The key is stored in `output/settings.json` and never sent back to the browser. The same three values can come from `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` env vars (handiest for Docker); Settings-tab values win over env. Local GGUF model downloads are skipped entirely while the API engine is active.

## Troubleshooting

- **Port already in use** — change the left side of `ports:` in `docker-compose.yml`.
- **GPU OOM mid-job** — another service (ollama, docling, …) is holding VRAM. Lower `N_GPU_LAYERS` (e.g. `12` on a 6 GB card shared with other GPU work). The job auto-retries once on CPU.
- **Job stuck in "narrating"** — the LLM child process died; the crash-retry handles it, or hit ↻ retry in the UI.
- **`Not enough readable text in this file`** — scanned PDF (no text layer) or a very short file. No OCR by design.
- **`narration API error: …`** — the BYOK endpoint rejected the request (bad key, wrong model id, unreachable URL). Check the values in ⚙️ Settings and hit “test connection”; the raw provider message is in the error line.
- **Follow-along missing on an old paper** — timing marks were added after the render; re-render the paper once.
- **UI looks stale after code changes** — the compose file mounts `app/` read-only, so HTML/JS is live on refresh and `.py` changes need `docker compose restart`. A full rebuild is only needed for `requirements.txt` / `Dockerfile` changes.
- **Disk growing** — `output/samples/` (voice samples, preview cache, per-sentence render cache) grows unboundedly; wiping it is safe, everything regenerates.
- **Unusual PDF layouts jumble** — 2-column reading order, line-wrap hyphen joining, and reference-stripping are heuristics (3-column layouts and exotic journal styles can still confuse them — see the `shortcut:` comments in `extract.py`).

## Podcast feeds

- `<origin>/feed.xml` — all finished papers.
- `<origin>/feed.xml?tag=<tag>` — only papers carrying that tag (the Feeds tab lists them with copy buttons).

## Security notes

This is a **local/LAN tool by design**. Three deliberate boundaries to know about before exposing it anywhere:

- `POST /api/upload-url` fetches arbitrary URLs (arXiv, etc.) — an SSRF surface on a LAN. If you ever expose the app beyond your LAN, add IP allowlisting first (see `SECURITY.md`).
- The web UI assumes the browser is trusted (your machine). There is no authentication.
- A BYOK API key sits in `output/settings.json` in clear text. The API masks it (`GET /api/settings` only reports *whether* a key is set), but anyone who can reach the app can drive your narration engine — and the paper excerpts go to the endpoint you configured. Keep the app on your LAN, or add auth before configuring a key.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate    # Python 3.10–3.13 (kokoro-onnx requirement)
pip install -r requirements-dev.txt   # runtime deps + pytest (prod image uses requirements.txt only)
pytest tests/ -v                      # extraction + mocked-LLM → real TTS → MP3 smoke test
NARRATE_MOCK=1 uvicorn main:app --app-dir app   # run the app without the LLM
```

The GPU-fallback test skips gracefully where the CUDA-built llama.cpp won't load (no driver). If your host can't make a venv, run the suite inside the built image:

```bash
docker run --rm -v "$PWD":/w -w /w <image-name> \
  bash -c "pip install -q pytest && python -m pytest tests/ -q"
```

Lint, format check, and how to run the exact CI checks locally: see [CONTRIBUTING.md](CONTRIBUTING.md). Contributing expectations, code of conduct, and the security policy live in [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) respectively.

## Project layout

```
app/
  main.py      FastAPI routes + Pipeline (queue, status files, settings, watcher, crash recovery)
  extract.py   source → {title, authors, sections} (PDF/TXT/TeX/DOCX)
  narrate.py   narration script via local GGUF (spawned child) or a BYOK OpenAI-compatible API
  tts.py       Kokoro → MP3 + per-piece disk cache + follow-along marks
  static/      index.html (library UI), view.html (reader/editor), pico.min.css
tests/         pytest suite (heavy deps mocked except the real-TTS smoke test)
Dockerfile     CPU or CUDA image from one file (build args select the path)
docker-compose.yml
```

## Roadmap

The backlog is tracked in [issues](https://github.com/jamilgafur/Audio_Summary/issues) — the short list of things that would actually help:

- IP allowlisting on the URL-fetch endpoint (for anyone exposing the app beyond a trusted LAN — see SECURITY.md)
- word-level (vs sentence-level) follow-along — needs phoneme timing Kokoro doesn't currently expose
- OCR for scanned PDFs

## Acknowledgments

Built on [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (TTS), [llama.cpp](https://github.com/ggml-org/llama.cpp) (local LLM), [PyMuPDF](https://pymupdf.readthedocs.io/) (PDF), [FastAPI](https://fastapi.tiangolo.com/) (web layer), and [Pico.css](https://picocss.com/) (UI).

## License

This project is licensed under the [Business Source License 1.1 (BUSL-1.1)](LICENSE) — full text in `LICENSE`. In plain terms:

- **What you can do today**: everything — use it, self-host it, modify it, redistribute it, even use it commercially for your own work. Self-hosted use on your own machine is expressly permitted.
- **What you can't do**: run it as a **competing commercial "paper-to-audio service"** — i.e. a business offering where *you* run the engine on *your* infrastructure and convert *third parties'* documents into audio. That requires a commercial license from the maintainer. (The exact definition is in the Additional Use Grant in `LICENSE`.)
- **What happens on 2030-11-07**: the code automatically re-licenses to **Apache License 2.0** (with its patent grant) — the project becomes fully open source with no further action required.

Why BUSL: it keeps the project source-available and freely usable for individuals and internal use, while reserving the "managed service" niche from being undercut by a cloud provider's re-host. If you need a commercial license for a service, open an issue or contact the maintainer on GitHub.

A note on the dependency stack: the runtime dependencies are MIT/BSD-licensed, which is compatible with BUSL-1.1 and with the Apache 2.0 re-licensing.

*This is a self-applied BUSL-1.1 text (the standard MariaDB license body with project-specific parameters). I am not a lawyer — if you are relying on this for a commercial product, have counsel review the Additional Use Grant wording.*
