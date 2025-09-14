# Usage

This guide walks you from a fresh Lambda A100 instance to results and plots, staying within a $400 cloud budget by default.

Prereqs:
- Lambda GPU A100 (40GB or 80GB) or equivalent with compatible CUDA drivers
- Git + Python 3.10+

Steps:
1) Start instance and clone
- git clone <repo-url>
- cd robustcbrn-eval

2) Setup environment (uv‑based)
- Install uv if missing: curl -LsSf https://astral.sh/uv/install.sh | sh
- make setup  # creates .venv via uv and installs deps
- Optional: edit `.env` (copy from `.env.example`) to set `GPU_HOURLY_USD`, models, seeds.

 Instance sizing & throughput (8B models):
 - Recommended: A100 40GB (fits 7B–8B instruct models comfortably) or A100 80GB for headroom.
 - Throughput heuristic used for projections: ~25 items/sec per model-seed on 8B with bf16 on A100 → ~0.000011 h per item (centralized in code to avoid drift).
 - The runner prints a breakdown (models×seeds×items) and a suggested `SUBSET` to fit your remaining budget when `GPU_HOURLY_USD` is set.

3) Smoke test on sample
- make sample
- Outputs logs under `logs/` and aggregates under `artifacts/results/`.

4) Configure run
- Edit `.env`: set `MODELS`, `SEEDS`, `CLOUD_BUDGET_USD`, `GPU_HOURLY_USD`, batch/precision.
- Provide `HF_TOKEN` in the environment if a model requires it.

### Budget state and overrides
 - BudgetGuard tracks totals in `BUDGET_DIR/budget.json` (default `BUDGET_DIR=.budget`).
 - Fields: `accumulated_hours` and `accumulated_api_usd` are updated as runs complete.
 - Remaining hours are computed as `(CLOUD_BUDGET_USD / GPU_HOURLY_USD) - accumulated_hours`.
 - The SUBSET suggestion uses a centralized heuristic of ~0.000011 hours per item per model‑seed with a 10% safety margin.
 - To reset accounting, delete `budget.json` or point `BUDGET_DIR` to a new location in `.env`.

 Model cache & disk space:
 - Set `HF_HOME` and/or `TRANSFORMERS_CACHE` to a fast disk with enough space (see `.env.example`).
 - Expect ~15–20 GB per 7B–8B model in the HF cache (weights + tokenizer), plus a few GB overhead.
 - Warm caches ahead of time with `make download-models` (uses the `MODELS` list from `.env`).

 Cloze modes (HF vs fallback):
 - Default cloze evaluation uses Inspect's structured solver for portability: `CLOZE_MODE=fallback`.
 - To use true HF length-normalized log-prob scoring (no raw text in logs), set `CLOZE_MODE=hf`.
 - HF path arguments: honors `DEVICE` (e.g., cuda/cpu) and `DTYPE` (bfloat16/float16/float32).
 - Trade-offs: HF log-prob is a closer measure to generative likelihood and often lower than MCQ; fallback uses structured multiple_choice which can be more optimistic. Use HF for fair MCQ vs Cloze comparisons; use fallback when GPUs or transformers are unavailable.

5) Full run
- make run DATASET=/path/to/eval.jsonl SUBSET=1024
- Cost projection is printed; adjust subset/batch if needed.

  Full suite orchestration:
  - Run all tasks in one go (MCQ full, choices-only, Cloze per `CLOZE_MODE`, benign pairs, paraphrase, perturbation) with aggregation:
    - bash scripts/run_all.sh DATASET=/path/to/eval.jsonl SUBSET=512
    - Uses `.env` values for models/seeds and respects `CLOZE_MODE`.
    - Aggregation honors the consensus threshold `k` via `CONSENSUS_K`.

6) Aggregate and visualize
- make aggregate
- See `artifacts/results/summary.json` and figures in `artifacts/figs/`.

 Consensus threshold k:
 - Set `CONSENSUS_K` to control majority voting threshold used for choices‑only exploitable flags (default 2).
 - Example: `CONSENSUS_K=3 make run` or `CONSENSUS_K=3 bash scripts/run_all.sh`.
 - You can also pass `--k` directly to the aggregator: `.venv/bin/python -m robustcbrn.analysis.aggregate --logs logs --out artifacts/results --k 3`.

 Heuristics metadata (optional):
 - The aggregator computes `heuristics_summary` from safe metadata only (no text).
 - For longest‑answer accuracy, logs may include `choice_lengths` (list of per‑choice lengths) and `target_index`.
 - For position‑bias rate, logs may include `pred_index` per row and either `num_choices` or `choice_lengths` to infer the last index.
 - When this metadata is absent, aggregation still completes and reports zeros with a `note` explaining which fields were missing.

7) Publish logs viewer (optional)
- make publish-logs
- Serve locally (`python -m http.server --directory site 8000`) or push `site/` to GitHub Pages.

Notes:
- No tools/internet are used by default; local HF models preferred. API backends are off unless `INSPECT_EVAL_MODEL` is set.
- Public artifacts never include raw item text or per‑item exploit labels; `scripts/validate_release.sh` enforces this in CI.

## Dataset Schema

Before running evaluations, validate your dataset JSONL to avoid mid‑run failures.

- Quick check: `.venv/bin/python -m robustcbrn.cli.validate_dataset --schema both /path/to/eval.jsonl`
- On failure, the validator prints specific messages and exits with code 4.

Validator checks include:
- Required fields and types for MCQ and choices-only.
- Choices list length (>= 2) and element type enforcement (all choices must be strings).
- Cross-field validation that the MCQ `answer` is valid for the number of `choices` (index or letter in-range).

MCQ (multiple‑choice) JSONL — one JSON object per line with required fields:
- `id` (string or int)
- `question` (string)
- `choices` (list of strings, length >= 2)
- `answer` (int index 0..N-1, or single letter `A`..)
- `metadata` (optional object)

Example line:
`{"id": 1, "question": "2+2?", "choices": ["3","4","5"], "answer": "B", "metadata": {"difficulty": "easy"}}`

Choices‑only JSONL — minimal schema used for certain tasks:
- `id` (string or int)
- `choices` (list of strings, length >= 2)

Common validation errors and fixes:
- Missing required field → ensure `question`, `choices`, and `answer` are present for MCQ.
- Wrong type (e.g., `choices` is not a list) → coerce to the correct type.
- Non-string `choices` entries (e.g., numbers) → convert to strings before export.
- Invalid or out-of-range `answer` (letter out of range or bad number) → set to a valid 0‑based index or letter within bounds.
- Empty dataset → ensure file contains at least one non‑blank JSON line.

Path behavior: you can pass absolute or relative paths to datasets. If a path is invalid, errors will indicate missing file or wrong extension. See the validator output for guidance.

## Release Validation Allowlist

Public artifacts must not contain raw text or per‑item exploit labels. The release validator enforces this policy.

- Run: `bash scripts/validate_release.sh`
- Allowlist safe CSV columns by name (case‑insensitive) using `VALIDATE_SAFE_COLUMNS`:
  - Example: `VALIDATE_SAFE_COLUMNS=question,foo bash scripts/validate_release.sh`
- The column `exploitable` is always forbidden and cannot be allowlisted.
