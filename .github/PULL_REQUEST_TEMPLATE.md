## What & why

<!-- One paragraph: what this changes and the problem it solves. Link the issue if there is one. -->

## How it works

<!-- Key decisions, touched components (app/, UI, tests, compose). Note anything a reviewer must understand. -->

## Checks

Run before submitting (see CONTRIBUTING.md → "Run the checks locally"):

- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] `pytest tests/ -q` passes (in a venv or the built image)
- [ ] UI changes: extracted `<script>` block passes `node --check`
- [ ] New non-trivial logic has at least one test
- [ ] If you touched `app/` Python: verified against the running container (`docker compose restart`)

## Preserved behavior

<!-- What you confirmed still works — especially anything near the invariants in CONTRIBUTING.md. -->
