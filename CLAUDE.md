RobustCBRN Eval — Developer Quick Context

Purpose
- Toolkit to robustify CBRN MCQA benchmarks: consensus/shortcut detection, verified cloze variants, statistical bias battery; deterministic and fail‑graceful.

Repo Map (paths)
- `cli.py` — CLI entry (`load` command)
- `configs/default.json` — logging, determinism, data mapping
- `requirements.txt` — core deps
- `src/config.py` — config dataclasses I/O
- `src/data/loader.py` — `load_dataset(...)`
- `src/data/schemas.py` — `Question` dataclass
- `src/data/validators.py` — CSV mapping, answer normalization
- `src/security/anonymizer.py` — `make_question_id(...)` (BLAKE2b)
- `src/utils/logging.py` — `setup_logging(...)`
- `src/utils/determinism.py` — `set_determinism(...)`
- `docs/brief.md` — project brief
- `docs/prd/index.md` — PRD
- `docs/architecture/index.md` — architecture TOC
- `data/wmdp_bio_sample_100.jsonl` — sample dataset
- `tests/` — unit tests and fixtures

Core APIs
- `load_dataset(path, fmt=None, csv_mapping=None, id_salt="") -> List[Question]`
- `Question(id, question, choices, answer, topic?, difficulty?, metadata?)`
- `make_question_id(question, choices, salt="") -> hex`
- `set_determinism(seed, cudnn_deterministic, cudnn_benchmark, cublas_workspace, python_hash_seed, tokenizers_parallelism)`
- `setup_logging(log_dir, filename, level) -> logger`

Setup & CLI (Using uv - Fast Python Package Manager)
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create environment: `uv venv`
- Activate: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
- Install deps: `uv pip install -r requirements.txt`
- Run: `python cli.py load data/wmdp_bio_sample_100.jsonl --config configs/default.json [--id-salt my_salt]`

Quick uv commands:
- `uv pip sync requirements.txt` — exact reproducible install
- `uv pip compile requirements.in -o requirements.txt` — lock dependencies
- `uv pip install torch --index-url https://download.pytorch.org/whl/cu118` — CUDA-specific packages

Data Format
- JSONL or CSV. Answers letters (A–F) or ints; normalized to 0‑based.
- CSV mapping via `configs/default.json` (`data.csv_mapping`); falls back to `choice_*` or `a,b,c,d` columns.
- IDs: BLAKE2b over question+choices with optional salt.

Tests
- `python -m unittest` (see `tests/test_pipeline.py`, `tests/test_cli_smoke.py`, `tests/test_config.py`).

Docs Pointers
- Architecture summary: `docs/architecture/summary.md`
- Release policy: `docs/prd/artifacts-release-policy.md`
- Next steps: `docs/prd/next-steps.md`

Extend (where to add code)
- New analysis modules under `src/` (e.g., `src/analysis/`), new CLI subcommands in `cli.py`, config fields in `src/config.py` + `configs/default.json`.
