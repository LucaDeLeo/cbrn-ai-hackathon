# RobustCBRN Eval

Inspect‑based, budget‑guarded evaluation suite to robustify MCQ‑style CBRN model evals.

- Quickstart: install `uv`, then `make setup` → `make sample` → `make run`
- Docs live under `docs/` (usage, architecture, security, report template).
 - For 7B–8B local HF models, use an A100 40GB+ and set `HF_HOME`/`TRANSFORMERS_CACHE` to a disk with ~15–20 GB per model; warm caches via `make download-models`.

Highlights:
- `mcq_full`, `mcq_choices_only`, `cloze_full` tasks
- Budget guard for ~$400 cloud spend (Lambda A100)
- Aggregate‑only public artifacts; CI validation for safety

See docs/README.md for details.

## Testing

- Run all tests: `make test`
- Enable HF cloze smoke test: `RUN_HF_CLOZE_SMOKE=1 make test` (skips by default)
