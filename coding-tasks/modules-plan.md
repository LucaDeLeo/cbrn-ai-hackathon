# RobustCBRN Eval — New Modules Implementation Plan

This plan adds eight capability buckets into the existing Inspect-based pipeline (datasets → tasks → logs → analysis) with minimal, surgical changes. It maps directly onto the project brief and hackathon context, and references concrete code entry points in this repo.

- Core context: `docs/overview/brief.md` (MVP features, safety-aware release, metrics), `hackathon-context/mainpage.md` (sprint scope, info-hazard policy)
- Current pipeline anchors: `robustcbrn/tasks/mcq_full.py`, `robustcbrn/tasks/mcq_choices_only.py`, `robustcbrn/tasks/cloze_full.py`, `robustcbrn/analysis/aggregate.py`, `robustcbrn/utils/*`
- Public-artifact gate: `scripts/validate_release.sh` (two-tier policy)
- Sample data: `data/sample_sanitized.jsonl`

We follow the repository layout proposed in your module list, but this document only defines the implementation plan; it does not create code yet.

## Integration Model

- New checks are implemented as either Inspect tasks under `robustcbrn/tasks/` or lightweight utilities under `robustcbrn/qa/` that tasks call into.
- Logs remain Inspect-native (JSON), keyed by `id`, with per-sample fields that our aggregator can consume. We will extend the aggregator to carry module-specific flags.
- Analysis aggregates merge by `id` and `model`, using existing helpers in `robustcbrn/analysis/aggregate.py` and new functions in `robustcbrn/analysis/robustness.py`.
- Safety: keep hazardous plaintext out of public artifacts; run any LLM-critic strictly over hashed IDs + metadata or a tiny sanitized calibration subset, per `docs/overview/brief.md` policy and `hackathon-context/mainpage.md` info-hazard guidance.

## Repository Changes (planned)

```
robustcbrn/
  qa/                       # NEW: data-quality & adversarial tools
    filters.py              # AFLite-lite, choices-only classifiers, heuristics
    ambiguity.py            # LLM-critic wrappers + voting logic
    paraphrase.py           # templatic paraphrases (safe-mode); optional LLM on sanitized subset
    perturb.py              # deterministic typos, punctuation, order, neutrals
    contamination.py        # time-split checks + paraphrase-drop proxy
    rules.py                # grammar, duplicates, label sanity, numeric checks
  tasks/
    aflite_screen.py        # Inspect task emitting predictability flags
    ambiguity_audit.py      # Inspect task for LLM-critic votes (IDs + metadata only)
    paraphrase_consistency.py
    perturbation_stability.py
  analysis/
    robustness.py           # fragility, consistency@k, McNemar, power, FDR
    figs.py                 # extend with new plots
docs/
  evaluation/robustness-playbook.md    # methods, defaults, safe-mode notes
  evaluation/annotation-guide.md       # multi-annotator protocol, adjudication
  DATA_QUALITY_CHECKS.md    # ruleset + thresholds
scripts/
  run_robustness_suite.sh   # one-shot pipeline orchestrator
  validate_release.sh       # extend to forbid per-item exploit labels & raw stems
```

## Logging & Aggregation (cross-cutting)

- Extend `robustcbrn/analysis/aggregate.py`:
  - Parse and preserve per-sample metadata fields if present (e.g., `flags`, `reasons`, `variant`, `paraphrase_id`, `perturbation_kind`).
  - Add a small helper to left-join per-item QA flags to main results by `id`.
  - Keep existing metrics (`majority_consensus`, `mcq_cloze_gap`, `abstention_overconfidence`), adding hooks for robustness metrics from `analysis/robustness.py`.

- Naming conventions in logs:
  - `task` contains one of: `mcq_full`, `mcq_choices_only`, `cloze_full`, `aflite_screen`, `ambiguity_audit`, `paraphrase_consistency`, `perturbation_stability`.
  - Optional columns: `variant` (e.g., `orig`, `para1`, `para2`, `pert:punct`), `flag_*` booleans, `reason_codes` (comma-separated safe codes), all safe for public aggregation.

## Safety (cross-cutting)

- Ensure `scripts/validate_release.sh` gates public artifacts: no stems or per-item exploit labels; only IDs + safe aggregates.
- For ambiguity critic & paraphrasing, operate on sanitized calibration items (`data/sample_sanitized.jsonl`) for any model-generated text.

---

## Module 1 — Adversarial filtering (AFLite‑lite) & bias probes

Goal
- Identify items that are too predictable from answer choices or shallow artifacts, to flag for removal in robust subsets.

Entry points
- `robustcbrn/qa/filters.py` (new)
- `robustcbrn/tasks/aflite_screen.py` (new Inspect task)
- `robustcbrn/analysis/robustness.py` (new reports; overlap with existing `majority_consensus()` in `analysis/aggregate.py`)

Implementation steps
- QA utilities (`qa/filters.py`):
  - Implement baseline heuristics: longest-answer, position-only, alphabetical preference. (Existing `analysis/aggregate.longest_answer_heuristic` can inform metrics.)
  - Implement a simple choices-only classifier:
    - Bag-of-words over choices; train a logistic regression using SciPy (`scipy.optimize`) to avoid new heavy deps. Optionally allow scikit-learn if later added.
    - Train on a subset of items with k-fold CV; output per-item predicted correctness probability.
  - Return per-item `predictability_score` and `flag_predictable` if score ≥ τ (default 0.7), plus which probe triggered.
- Task (`tasks/aflite_screen.py`):
  - Load MCQ dataset via `tasks/common.load_mcq_dataset()`.
  - Build choices-only view like `mcq_choices_only._build_choices_only_dataset()` and fit/evaluate probes.
  - Log per-item: `id`, `flag_predictable`, `predictability_score`, `probe_hit`.
- Analysis (`analysis/robustness.py`):
  - Compute fraction flagged, overlap with `choices_only` majority-exploitable from `aggregate.majority_consensus()`.
  - Provide presets: conservative (τ=0.8, require ≥2 probes), balanced (τ=0.7), aggressive (τ=0.6 or probe OR).

Testing & acceptance
- Unit-test probes on `data/sample_sanitized.jsonl` with seeded determinism.
- Acceptance: flags correlate with choices-only correctness > chance; reports render CIs via `utils.stats.bootstrap_ci`.

References
- Brief: bias-detection, Deep Ignorance §D.4; longest-answer artifacts.
- Code: `robustcbrn/tasks/mcq_choices_only.py`, `robustcbrn/analysis/aggregate.py`.

---

## Module 2 — Ambiguity / unanswerable detection (LLM‑critic + HIL)

Goal
- Flag items with multiple plausible answers or no resolvable answer, routed to human adjudication.

Entry points
- `robustcbrn/qa/ambiguity.py` (critic interface, voting rules)
- `robustcbrn/tasks/ambiguity_audit.py` (Inspect task)
- `docs/evaluation/annotation-guide.md` (mult i-annotator protocol; sanitized subset only)

Implementation steps
- QA (`qa/ambiguity.py`):
  - Define safe critic prompt operating only on item `id`, high-level category tags, and answer set (no stems) or on sanitized items.
  - Implement majority vote across multiple light critics or thresholds; combine with heuristics (contradictory options, identical distractors).
  - Output `label` in {`clean`,`ambiguous`,`unanswerable`} and `reason_codes`.
- Task (`tasks/ambiguity_audit.py`):
  - Batch items; process critics safely on sanitized items only.
  - Log per-item safe fields: `id`, `label`, `reason_codes`. Store any free-form rationale privately only.
- Docs (`docs/evaluation/annotation-guide.md`):
  - Specify 3-annotator process on 50-item sanitized calibration set; tie-break rules; schema for CSV.

Testing & acceptance
- Golden set: sanitized items with known ambiguous cases; target ≥70–80% agreement with human labels (brief’s validation metric).
- Public release includes IDs + aggregate counts only.

References
- Brief: two-tier safety policy; validation metrics; risk thresholds.
- Code: `robustcbrn/analysis/aggregate.py` join mechanics.

---

## Module 3 — Paraphrase & perturbation robustness

Goal
- Measure fragility: does model correctness flip under benign wording/format changes?

Entry points
- `robustcbrn/qa/paraphrase.py`, `robustcbrn/qa/perturb.py`
- `robustcbrn/tasks/paraphrase_consistency.py`, `robustcbrn/tasks/perturbation_stability.py`
- `robustcbrn/analysis/robustness.py` (consistency@k, flip rates, paired tests)

Implementation steps
- Paraphrase (`qa/paraphrase.py`):
  - Deterministic templatic rewrites: active↔passive voice, clause reordering, tense normalization, determiner swaps. Ensure no content addition.
  - Optional: LLM paraphrases only on sanitized calibration items.
- Perturb (`qa/perturb.py`):
  - Punctuation tweaks, whitespace normalization, option order swaps (A↔D), benign distractor preambles.
- Tasks:
  - For each original item, generate `k` variants; evaluate with `mcq_full`/`cloze_full` solver pathway (reuse `render_*` in `tasks/common.py`).
  - Log per-item, per-variant: `id`, `variant`, `pred_index`, `correct`.
- Analysis (`analysis/robustness.py`):
  - `consistency_at_k`: fraction of items with identical predictions across `k` variants.
  - `fragility_score`: mean flip rate; `delta_accuracy` orig vs mean-of-variants with bootstrap CIs and McNemar tests.

Testing & acceptance
- Seeded runs on `data/sample_sanitized.jsonl`; assert deterministic variant generation and expected small deltas.

References
- Brief: “Paraphrase & perturbation robustness (consistency metrics)”; cloze vs MC.
- Code: `robustcbrn/tasks/common.py` (rendering), `robustcbrn/tasks/cloze_full.py` (logprob path), `robustcbrn/analysis/aggregate.py`.

---

## Module 4 — Dynamic benchmark stress (benign pairs) & red‑team protocol

Goal
- Demonstrate safe dynamic stress-testing with benign paired prompts and a documented, private expert-in-the-loop pathway for sensitive content.

Entry points
- Docs only for protocol; optional tiny task to score benign policy-compliance pairs.
- `docs/evaluation/robustness-playbook.md` (process), `scripts/run_robustness_suite.sh` (driver)

Implementation steps
- Create a tiny sanitized set of benign pairs (e.g., safe disposal policies vs refusal to provide hazardous instructions) with no operational detail.
- Add an Inspect task (optional) to score refusal/compliance consistency on this toy set.
- Document the red-team protocol (roles, checklists, escalation) for private, post-sprint usage.

Testing & acceptance
- Ensure no hazardous strings exist in repo artifacts; validate via `scripts/validate_release.sh`.

References
- Hackathon context: info-hazards section; Brief: two-tier release policy.

---

## Module 5 — Automated QA & label hygiene

Goal
- Catch duplicates, grammar issues, label mismatches, numeric inconsistencies before release.

Entry points
- `robustcbrn/qa/rules.py`, `scripts/validate_release.sh`

Implementation steps
- Rules (`qa/rules.py`):
  - Duplicates: near-duplicate detection via MinHash (`mmh3`) or cosine similarity on local embeddings.
  - Label sanity: ensure target ∈ choices; numeric checks where applicable.
  - Grammar/lint proxy using a lightweight tool or heuristic scores (no web calls).
- CI integration:
  - Extend `scripts/validate_release.sh` to fail if rule violations exceed thresholds.
  - Produce a public-safe `artifacts/data_quality_report.csv` without stems.

Testing & acceptance
- CPU-only test on `data/sample_sanitized.jsonl` asserting detection of synthetic duplicates and bad labels.

References
- Brief: “Automated QA & label hygiene”; two-tier policy in `validate_release.sh`.

---

## Module 6 — Formal measurement & reporting

Goal
- Make results defensible: uncertainty, paired tests, multiple references, and error control.

Entry points
- `robustcbrn/analysis/robustness.py` (metrics) + updates to `robustcbrn/analysis/figs.py`
- Docs templates: `docs/results/report.md`, `docs/results/results-template.md`

Implementation steps
- Metrics:
  - Bootstrap CIs for all rates using `utils.stats.bootstrap_ci`.
  - McNemar tests for paired deltas (orig vs cloze/paraphrase) using SciPy.
  - Simple power analysis for proportions to justify sample sizes.
  - Multiple references: exact vs normalized match where datasets provide synonyms.
  - FDR control (Benjamini–Hochberg) for multiple comparisons.
- Reporting:
  - Extend `analysis/figs.py` with CI bar charts, fragility plots, and paired delta visuals.
  - Add pre-filled doc sections and tables in `docs/results/report.md` and `docs/results/results-template.md`.

Testing & acceptance
- Unit tests for McNemar implementation and BH-FDR control on synthetic data.

References
- Brief: “Formal measurement” section; existing `utils.stats` helpers.

---

## Orchestration & Commands

- Add `scripts/run_robustness_suite.sh` to run: AFLite-lite screen, ambiguity audit (IDs only), paraphrase/perturbation stability, aggregation, and figures.
- Update `Makefile` targets (read current `Makefile`):
  - `make run` continues to run `mcq_full`, `mcq_choices_only`, `cloze_full`.
  - New: `make robustness` → calls `scripts/run_robustness_suite.sh` with CLI args (`--paraphrases`, `--perturbations`, `--af-threshold`, `--subset`).
  - `make aggregate` already calls `analysis/aggregate.py`; extend to include `analysis/robustness.py` outputs.

Example (planned):
```
bash scripts/run_robustness_suite.sh \
  --paraphrases 2 --perturbations 2 --af-threshold 0.7 --subset 1500
```

## Figures & Outputs (public-safe)

- Waterfall: original → AFLite-lite removal → choices-only consensus removal → ambiguous removal → robust subset (95% CIs).
- Fragility plot: accuracy vs number of paraphrases; flip rates per domain.
- MCQ vs Cloze: paired deltas with McNemar p-values.
- Heuristics bars: longest-answer, position bias, hypothesis-only accuracy.
- Abstention & overconfidence: from existing confidence-aware pipeline (brief’s “Confidence-Aware Evaluation”).

All public artifacts must pass `scripts/validate_release.sh`.

## Prioritization & Timeline (maps to brief)

1) AFLite-lite & bias probes → 2) Paraphrase/Perturbation → 3) Ambiguity audit → 4) Formal measurement → 5) QA rules → 6) Contamination signals → 7) HIL workflow → 8) Dynamic stress (docs/protocol). This aligns with MVP weekend sequencing in `docs/overview/brief.md`.

## Dependencies & Assumptions

- Python 3.10+, PyTorch/Transformers per `requirements.txt`. SciPy present; scikit-learn optional (not currently pinned) — we will start with SciPy-based logistic regression to avoid new deps.
- GPU: A100 40GB target (brief), or reduced subsets locally. Use `BudgetGuard` to enforce caps and dry-run estimates.
- Inspect AI (`inspect_ai`) already pinned; we follow the pattern in existing tasks.

## Acceptance Criteria (overall)

- Reproducible: seeded runs yield <2% variance on 1,000-item sample (brief).
- Safe: no hazardous plaintext or per-item exploit labels in public artifacts; CI gate passes.
- Effective: 30–40% exploitable detection across combined screens; cloze vs MCQ deltas and fragility metrics reported with CIs.
- Documented: ROBUSTNESS_PLAYBOOK, DATA_QUALITY_CHECKS, ANNOTATION_GUIDE, and report templates added and referenced in README/USAGE.
