# Next Steps to Complete RobustCBRN Eval

This list summarizes remaining work to reach a clean, shippable state after the docs unification and BudgetGuard QoL changes.

## Execution & Reporting
- [ ] Run a full end-to-end suite on the target dataset (MCQ full, choices-only, Cloze per `CLOZE_MODE`, robustness tasks).
- [ ] Populate `docs/results/report.md` using `scripts/fill_report.py` and verify all placeholders are filled.
- [ ] Generate and check figures under `artifacts/figs/` (paraphrase consistency, perturbation fragility, etc.).
- [ ] Validate release artifacts via `bash scripts/validate_release.sh` (no raw text or per-item exploit labels).
- [ ] Optionally publish the logs viewer bundle and confirm it renders locally.

## Stories, Gates, and File Lists
- [ ] Align story statuses with existing gates (e.g., mark stories with PASS gates as Done/Ready for Done as appropriate).
- [ ] Update File Lists in stories to include the latest changes (e.g., 8.1 added projection CLI and centralized cost helpers).

## CI and Testing
- [ ] Run full `pytest` and ensure green (HF cloze smoke remains guarded by `RUN_HF_CLOZE_SMOKE=0`).
- [ ] Optional: add a small assertion that `heuristics_summary` keys are always present in `summary.json` (defense-in-depth).
- [ ] Optional: add an integration smoke that `scripts/run_evalset.sh` final summary includes configured `k`.

## Runner & Scripts (Polish)
- [ ] Unify provider-prefix normalization across all Inspect calls in `scripts/run_all.sh` (benign pairs path follows the same normalization as others).
- [ ] Consider echoing per-phase elapsed times in `scripts/run_all.sh` for observability (consistency with `run_evalset.sh`).

## Docs Follow-ups
- [ ] Review internal links after the restructure; fix any missed references (anchors and paths).
- [ ] Keep cost/throughput heuristic single-sourced via `robustcbrn/utils/cost.py`; avoid re-stating numbers elsewhere.
- [ ] Optionally add a minimal “Docs Index” section at the top of `docs/README.md` with the new structure outline.

## Release
- [ ] Follow `docs/safety/release-checklist.md` to finalize the public artifact set.
- [ ] Tag a release once report, figures, and artifacts are validated.

