# RobustCBRN Eval

Practical toolkit for robustifying CBRN-related AI benchmarks. Implements consensus/shortcut detection, verified cloze variants, and a statistical bias battery with deterministic, fail‑graceful execution. Full documentation lives in `docs/`.

**Docs**
- Project Brief: `docs/brief.md`
- PRD: `docs/prd/index.md`
- Architecture: `docs/architecture/index.md`
- List of Evaluations: `docs/listofevals.md`
- Release Checklist: `docs/release_checklist.md`
- Prompts Appendix: `docs/prompts_appendix.md`

**Quick Start**
- Python 3.10+
- Install uv (fast Python package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create virtual environment: `uv venv`
- Activate environment:
  - Linux/Mac: `source .venv/bin/activate`
  - Windows: `.venv\Scripts\activate`
- Install dependencies: `uv pip install -r requirements.txt`
- Load a sample dataset: `python cli.py load data/wmdp_bio_sample_100.jsonl --config configs/default.json`
  - Optional: add `--id-salt your_salt` to influence hashed IDs

**Data Format**
- Supports `JSONL` and `CSV` inputs; answers normalized to 0-based index.
- CSV mapping is configurable via `data.csv_mapping` in `configs/default.json`.

**Repo Layout**
- `src/` pipeline, data loader, logging, security utilities
- `configs/` default app configuration
- `tests/` unit tests (`python -m unittest`)

**Safety & Release**
- See `docs/prd/artifacts-release-policy.md` for anonymization and public artifact rules.
