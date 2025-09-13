# RobustCBRN Eval

Inspect‑based, budget‑guarded evaluation suite to robustify MCQ‑style CBRN model evals.

- Quickstart: install `uv`, then `make setup` → `make sample` → `make run`
- Docs live under `docs/` (usage, architecture, cost plan, security, report template).

Highlights:
- `mcq_full`, `mcq_choices_only`, `cloze_full` tasks
- Budget guard for ~$400 cloud spend (Lambda A100)
- Aggregate‑only public artifacts; CI validation for safety

See docs/README.md for details.
