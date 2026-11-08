# Contributing

Thanks for taking a look. Paper Reader is a small, dependency-lean local service — keep it that way: no frameworks, no build step for the UI, no new runtime dependency without a strong reason (state the reason in the PR).

## Ground rules

- **Honor the invariants** — they exist because things broke when they were violated:
  - *Stem discipline*: a job's key derives from `_safe(filename)`; every file operation goes through the `Pipeline` path helpers (`mp3/script/follow/status_path/input_source/find_source`), never hand-built names.
  - *Atomic writes*: write `<x>.tmp`, then `os.replace` — the watcher only globs final names.
  - *409 rule*: any mutation of a job whose stage is `queued`/`extracting`/`narrating`/`rendering audio` must call `_busy_or_409` first — the worker rewrites those files constantly.
  - *New artifact type*: extend BOTH the archive candidate list and the delete list.
- Keep the diff as small as the change allows. Deletion beats addition; an existing helper beats a new one.
- **Never commit runtime data**: `input/`, `output/`, and `models/` are gitignored for a reason (user papers, renders, multi-GB weights).
- Python 3.10–3.13 is the supported range (`kokoro-onnx` sets the floor); target 3.10.
- UI text that comes from user/paper data goes through `esc()` or `textContent` — no raw string interpolation into HTML.

## Setup

```bash
git clone <repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate     # Python 3.10–3.13
pip install -r requirements-dev.txt
pip install ruff
```

Run the app without the LLM for UI work:

```bash
NARRATE_MOCK=1 uvicorn main:app --app-dir app
```

## Run the checks locally (what CI runs)

```bash
ruff check .                 # lint (rules in pyproject.toml)
ruff format --check .        # formatting
pytest tests/ -v             # suite (real Kokoro TTS; ~340 MB model download once)
# UI syntax check (any page's <script> block):
awk '/<script>$/{x=1;next}/<\/script>/{x=0}x' app/static/index.html | node --check
```

If you can't make a venv on your host, the suite also runs inside the built image:

```bash
docker run --rm -v "$PWD":/w -w /w <image-name> bash -c "pip install -q pytest && python -m pytest tests/ -q"
```

The llama.cpp GPU-fallback test **skips** where the CUDA lib won't load (driverless machines, CI) — that's expected, not a failure.

## Tests

- New non-trivial logic (a branch, a loop, a parser, anything touching money/security/data) ships with at least one test, even if it's an assert-based self-check.
- Heavy deps (LLM, TTS) are mocked except `tests/test_smoke.py`, which deliberately runs real Kokoro TTS end-to-end.
- `NARRATE_MOCK=1` is the switch that keeps the suite LLM-free.

## Pull requests

1. Small, focused PRs. One feature or one fix.
2. Fill in the PR template — "preserved behavior" is the important section.
3. CI must be green: lint, format, tests (3.10–3.12 matrix), `pip-audit`, JS syntax check, and a CPU Docker build.
4. If you touch `app/` Python and have a running container, verify against it (`docker compose restart`, then click through the affected flow).

## Branch protection (maintainers)

Recommended GitHub settings for `main`:

- Require a pull request before merging (1 approving review for non-dependabot PRs)
- Require status checks to pass: `lint`, `test` (all matrix legs), `security`, `js-check`, `docker-build`
- Do not allow force pushes or deletions on `main`
- Include administrators (no bypasses)

Releases: tag `vX.Y.Z` on `main`; the release workflow builds and pushes the CPU image to GHCR. Update `CHANGELOG.md` in the release PR.

## Code of conduct

Interactions in this project's spaces are covered by the [Contributor Covenant](CODE_OF_CONDUCT.md).
