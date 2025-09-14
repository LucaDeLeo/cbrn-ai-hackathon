Module 2 — Ambiguity / Unanswerable Detection

Overview
- Adds a safe, metadata-only ambiguity/unanswerable screen to the existing Inspect-based pipeline. The module flags items as clean, ambiguous, or unanswerable using lightweight heuristics over choices only (no stems). It writes Inspect-like logs and extends aggregation to preserve module fields. An LLM critic is stubbed for sanitized-only use but disabled by default.

Goals
- Identify items with multiple plausible answers or no resolvable answer.
- Keep public artifacts free of hazardous plaintext; only IDs and safe codes.
- Integrate with existing logs → analysis flow with minimal, surgical changes.

Changes Made
- Added QA utility `robustcbrn/qa/ambiguity.py`:
  - Heuristics operating over `choices` only; never reads stems.
  - Labels: `clean`, `ambiguous`, `unanswerable` with `reason_codes`.
  - Helpers: `audit_dataset(dataset) -> List[AmbiguityDecision]`, `decisions_to_records()`.
  - Stub: `llm_critic_votes(...)` (disabled by default in OSS; for sanitized subsets).

- Added task/CLI `robustcbrn/tasks/ambiguity_audit.py`:
  - Function `run_ambiguity_audit(dataset_path, mode, ...) -> str` writes an Inspect-like JSON log.
  - CLI: `python -m robustcbrn.tasks.ambiguity_audit <dataset> [--mode heuristic|llm]`.
  - Uses `BudgetGuard` in dry-run mode to align with budget policy.
  - Log schema (safe fields only): `task='ambiguity_audit'`, `model`, `seed`, `samples=[{id,label,reason_codes}]`.

- Extended aggregator `robustcbrn/analysis/aggregate.py`:
  - Preserves new optional fields: `ambiguity_label` (from `label`), `reason_codes`.
  - Preps for future modules by parsing `variant`, `paraphrase_id`, `perturbation_kind`.
  - Existing metrics unchanged; CSV now includes these columns when present.

- Added documentation `docs/evaluation/annotation-guide.md`:
  - 3-annotator process on sanitized calibration items, tie-breaks, reason codes, and safety notes.

Heuristics (qa/ambiguity.py)
- Meta options → unanswerable: `all_of_the_above`, `none_of_the_above`, `both_and`, `dont_know`.
- Duplicate/near-duplicate options (Jaccard ≥ 0.9) → ambiguous.
- Contradictory options (simple negation pairs) → ambiguous.
- Numeric crowding (values within 1% proximity) → ambiguous.
- Boolean-like options (very short choices) → unanswerable.

How To Run
- Ambiguity audit (heuristics):
  - `python -m robustcbrn.tasks.ambiguity_audit data/sample_sanitized.jsonl --mode heuristic --out logs/ambiguity_audit_sample.json`
  - Output example fields (safe): `{"task": "ambiguity_audit", "model": "heuristic", "samples": [{"id": "...", "label": "ambiguous", "reason_codes": "duplicate_choices"}]}`

- Aggregate:
  - `python -m robustcbrn.analysis.aggregate --logs logs --out results`
  - New fields appear in `results/all_results.csv` when present.

Safety & Policy
- Heuristics avoid stems entirely; operate on IDs and choices only.
- LLM critic disabled by default; use only on sanitized subsets if enabled.
- Public artifact gate remains enforced by `scripts/validate_release.sh` (blocks raw `question`/`choices` and per-item exploit labels).

Budget & Runtime
- `BudgetGuard` is engaged in dry-run mode for this module (no GPU/API usage projected). This keeps cost accounting consistent without incurring charges.

Testing & Status
- `pytest` passes (`.........`), indicating no regressions in existing modules.
- Manual smoke run over `data/sample_sanitized.jsonl` produced expected JSON log.

Future Enhancements
- Optional sanitized LLM critic: integrate voting across lightweight critics and heuristics.
- Add a convenience left-join helper to merge ambiguity labels directly into main results in analysis notebooks.
- Threshold review and tuning based on human adjudication (see `docs/evaluation/annotation-guide.md`).
