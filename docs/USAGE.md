# Usage

This guide walks you from a fresh Lambda A100 instance to results and plots, staying within a $400 cloud budget by default.

Prereqs:
- Lambda GPU A100 (or equivalent) with CUDA drivers
- Git + Python 3.10+

Steps:
1) Start instance and clone
- git clone <repo-url>
- cd robustcbrn-eval

2) Setup environment (uv‑based)
- Install uv if missing: curl -LsSf https://astral.sh/uv/install.sh | sh
- make setup  # creates .venv via uv and installs deps
- Optional: edit `.env` (copy from `.env.example`) to set `GPU_HOURLY_USD`, models, seeds.

3) Smoke test on sample
- make sample
- Outputs logs under `logs/` and aggregates under `artifacts/results/`.

4) Configure run
- Edit `.env`: set `MODELS`, `SEEDS`, `CLOUD_BUDGET_USD`, `GPU_HOURLY_USD`, batch/precision.
- Provide `HF_TOKEN` in the environment if a model requires it.

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
