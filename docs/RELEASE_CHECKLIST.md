# Release Checklist

Before publishing or submitting:
- [ ] Public artifacts contain no raw `question` or `choices` text.
- [ ] No per‑item `exploitable` flags in artifacts.
- [ ] Seeds, model versions, and config are pinned in `docs/REPORT.md`.
- [ ] CI passes: lint, tests, and `scripts/validate_release.sh`.
- [ ] Inspect logs viewer bundle reviewed (`site/logs/`).
- [ ] Results tables/figures updated (`docs/RESULTS_TEMPLATE.md`).
- [ ] Budget totals reviewed (`.budget/budget.json`).

Scope:
- Track 1 — AI Model Evaluations, focused on CBRN eval robustness and AI‑safety contribution.

