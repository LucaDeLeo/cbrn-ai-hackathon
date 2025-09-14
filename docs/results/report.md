<!-- canonical path: docs/results/report.md -->
# Sprint Report (Draft)

Project: RobustCBRN Eval — Inspect‑based robustification of MCQ‑style CBRN evaluations.

Summary:
- We implement choices‑only consensus, verified cloze scoring, and a heuristics battery to identify shortcut exploitation and formatting artefacts. Public artifacts are aggregate‑only.

Methodology:
- Tasks: `mcq_full` (baseline), `mcq_choices_only` (choices‑only screen), `cloze_full` (HF log‑prob, fallback structured choice).
- Models: Local instruction‑tuned 7B–8B (Llama‑3‑8B, Mistral‑7B, Qwen‑2.5‑7B). API backends optional/off by default.
- Seeds: 2 seeds for shuffling/robustness.
 - Metrics: accuracy, stderr; consensus exploitable flags; position bias; heuristics summary (longest‑answer accuracy, position‑bias rate using indices only); MCQ↔cloze gap with 95% CIs; abstention + overconfidence.
- Budget: Guarded within $400, Lambda A100‑optimized.

Results (placeholders → auto‑filled post‑run):
- Overall accuracy: 0.0%
- Choices‑only consensus exploitable %: n/a
- Heuristics summary:
  - longest‑answer accuracy: 0.0%
  - position‑bias rate (first/last): 0.0%
- MCQ↔Cloze gap (95% CI): Δ=0.000 (95% CI: [0.000, 0.000])
- Abstention / overconfidence: abst=25.0%, overconf=75.0%
- Runtime / cost: n/a

Reproducibility:
- Hardware: A100 40/80GB (Lambda)
- Software: Python 3.10, Inspect, Transformers; see `requirements.lock.txt`.
- Commands:
  - make setup
  - make run DATASET=/path/to/eval.jsonl SUBSET=...
  - make aggregate

Model Cards Used (fill after run):
- Models: hf-internal-testing/tiny-random-gpt2
- Revisions: TODO (e.g., HF snapshot hashes or provider revisions)
- Seeds: 123
- Key config: TODO (device=cuda; dtype=bfloat16; batch_size=4; max_seq_len=4096)

Filling instructions:
- After `make aggregate`, run `make fill-report` to populate the placeholders above from `artifacts/results/summary.json`, `artifacts/results/all_results.csv`, `.budget/budget.json`, and environment/config where available.
 - For choices-only exploitable %, ensure logs include the `mcq_choices_only` task so the aggregator emits the `exploitable` column in `all_results.csv`. If absent, this field shows n/a.
 - Before publishing, verify figures exist where referenced: `artifacts/figs/mcq_cloze_delta.png`, `paraphrase_consistency.png`, `perturbation_fragility.png`.

Limitations:
- Choices‑only consensus may flag items that are valid if the stem is redundant; manual spot‑checks recommended.
- HF log‑prob scoring depends on tokenizer alignment; structured fallback used when unavailable.
- Only sanitized datasets and aggregates are public.

Security Considerations:
- See `docs/safety/security-considerations.md`. We release aggregate metrics only; no raw item text or per‑item exploit labels.

Prompts Appendix:
- See `docs/PROMPTS_APPENDIX.md` for templates and hashes.


## Formal Measurement Summary

- Confidence intervals: All rates report bootstrap 95% CIs.
- Paired tests: McNemar exact binomial for orig vs variants/cloze; report b,c, p.
- Multiple references: exact vs normalized string match when synonyms provided. Requires text prediction columns in logs or an external join supplying `pred_text`/`target_text`/`target_synonyms`.
- Multiple comparisons: Benjamini–Hochberg FDR at α=0.05.
- Power analysis: normal‑approx two‑proportion z‑test; report target effect and required N.

Tables/figures updated in `docs/results/results-template.md`.
