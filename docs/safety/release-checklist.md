<!-- canonical path: docs/safety/release-checklist.md -->
# Release Checklist

Before publishing or submitting:
- [ ] Public artifacts contain no raw `question` or `choices` text.
- [ ] No per‑item `exploitable` flags in artifacts.
- [ ] Seeds, model versions, and config are pinned in `docs/results/report.md`.
- [ ] CI passes: lint, tests, and `scripts/validate_release.sh`.
- [ ] Inspect logs viewer bundle reviewed (`site/logs/`).
- [ ] Results tables/figures updated (`docs/results/results-template.md`).
- [ ] Referenced figures exist under `artifacts/figs/` (e.g., `mcq_cloze_delta.png`, `paraphrase_consistency.png`, `perturbation_fragility.png`).

Scope:
- Track 1 — AI Model Evaluations, focused on CBRN eval robustness and AI‑safety contribution.
