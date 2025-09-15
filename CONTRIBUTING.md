# Contributing to RobustCBRN Eval

Thanks for your interest in contributing! This project aims to keep runs reproducible, safe, and well‑documented.

## Getting started
- Python 3.10+
- `uv venv && uv pip install -r requirements.txt`
- Run `make setup` and `make sample` to verify your environment.

## Development workflow
- Lint: `.venv/bin/ruff check robustcbrn tests`
- Tests: `.venv/bin/pytest -q`
- Install hooks: `pip install pre-commit && pre-commit install`

## Pull request checklist
- Code has docstrings for public APIs and CLI entry points.
- No generated files (logs, artifacts, figs) included in commits.
- Add/update docs where needed (usage, pipeline, or architecture).
- Run `scripts/validate_release.sh` if your change touches artifacts.

## Safety & data
- Never commit raw datasets. Use `data/registry.yaml` and `scripts/fetch_data.py`.
- Public artifacts must not include raw question text or per‑item exploit labels.
- See `docs/safety/` for the full policy and checklist.
