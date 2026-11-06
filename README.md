# Paper Reader

Drop a research paper (PDF) into `input/` → get an MP3 you can listen to in `output/` and at `http://localhost:8000`.

A local LLM (Qwen2.5-7B by default) reads the paper and writes a *spoken narration* script — ideas explained, equations/tables/citations skipped — then Kokoro TTS renders it to audio. Everything runs on your machine: nothing leaves it.

## Hardware target

Nvidia 1070 Ti (8 GB) / 48 GB RAM / Intel i5. LLM runs on the GPU (llama.cpp, ~20–30 tok/s, script in a few minutes); TTS runs on CPU. Prereqs: NVIDIA driver ≥ 525 and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Run

```bash
docker compose up --build   # first build: ~10 min (compiles llama.cpp for sm_61)
```

Then drop PDFs into `input/`, or use the web UI at <http://localhost:8000> (drag-drop upload, live progress, player with speed control).

No NVIDIA GPU? CPU-only image (script generation ~20–30 min/paper instead):

```bash
docker build -t paper-reader --build-arg CMAKE_ARGS="" .
docker run -p 8000:8000 -v "$PWD/input:/data/input" -v "$PWD/output:/data/output" -v "$PWD/models:/data/models" paper-reader
```

First run downloads ~5 GB of models into `models/` (mounted, so it happens once).

## Configuration (env, see docker-compose.yml)

| Var | Default | Notes |
|---|---|---|
| `GGUF_REPO` / `GGUF_FILE` | Qwen2.5-7B-Instruct Q4_K_M | Any instruct GGUF. A 14B fits your 48 GB RAM on CPU (`N_GPU_LAYERS=0`). |
| `N_GPU_LAYERS` | `-1` (all on GPU) | `0` = CPU only. |
| `VOICE` | `af_heart` | Kokoro voices, e.g. `am_adam`, `bf_emma`. |
| `SPEED` | `1.1` | TTS speed; the web player also has its own speed control. |

## Notes & limits

- `output/<name>.script.txt` keeps the narration script: TTS failures resume cheaply. To re-render from the cached script, use the 🔊 re-render button in the UI. Manual path: delete the `.mp3` *and* the `.status.json`, keep the `.script.txt`, then wait for the watcher (or restart).
- If the worker crashes mid-job (e.g. llama.cpp CUDA abort while another service holds VRAM), the job is re-queued automatically on restart — twice at most, then marked failed.
- Two modes, picked at upload (and on re-render): **summary** (default — LLM explainer) and **verbatim** (the paper's exact words, no LLM — for auditing your own paper aloud). Switching modes regenerates the script.
- Scanned PDFs (no text layer) are detected and fail loudly — no OCR.
- 2-column extraction + reference-stripping are heuristics; unusual layouts may jumble.
- The narration is an *explainer*, not a verbatim read — it can hallucinate. The PDF stays the source of truth.

## Develop / test (no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate    # needs Python 3.10–3.13 (kokoro-onnx requirement)
pip install -r requirements-dev.txt   # runtime deps + pytest (prod image uses requirements.txt only)
pytest tests/ -v        # extraction tests + mocked-LLM -> real TTS -> MP3 smoke test
NARRATE_MOCK=1 uvicorn main:app --app-dir app   # run the app without the LLM
```
