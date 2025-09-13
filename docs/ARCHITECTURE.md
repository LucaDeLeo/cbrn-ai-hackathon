# Architecture

Overview:
- Data: JSONL items (`question`, `choices`, `answer`, `id`, `metadata`).
- Tasks: Inspect tasks wrap datasets with solvers + scorers.
- Runs: Inspect logs per model/task/seed under `logs/`.
- Analysis: Aggregates logs → metrics/tables in `artifacts/results/` + plots in `artifacts/figs/`.
- Budget Guard: Wraps runs; dry‑run projections and wall‑time accounting with caps.

Modules:
- `robustcbrn/tasks/common.py` — I/O, shuffling, choices‑only transform, prompt rendering.
- `robustcbrn/tasks/mcq_full.py` — Baseline MCQ (no CoT) with `multiple_choice()` + `choice()`.
- `robustcbrn/tasks/mcq_choices_only.py` — Choices‑only screen for consensus.
- `robustcbrn/tasks/cloze_full.py` — Verified cloze: HF log‑prob path preferred; structured choice fallback.
- `robustcbrn/budget_guard.py` — Context manager/CLI for cost guarding with state in `.budget/budget.json`.
- `robustcbrn/analysis/aggregate.py` — Loads logs, computes exploitable flags, heuristics, gaps, CIs.
- `robustcbrn/analysis/figs.py` — Minimal matplotlib charts.

Data flow:
- dataset → Inspect task → logs/*.json → aggregate → artifacts/results/*.json|csv → figs.

Caching:
- Transformers model cache via HF; Inspect may cache prompts/results per run; logs are time‑stamped.

Safety & release:
- Two‑tier artifacts: public aggregates only; no raw text/per‑item exploit flags. The validator script enforces this.

Budget guard integration:
- `make run` computes projected hours given models×seeds×subset and warns before running.
- `BudgetGuard` records accumulated hours and warns when exceeding `CLOUD_BUDGET_USD`.

