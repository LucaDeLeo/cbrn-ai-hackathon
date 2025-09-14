# RobustCBRN Eval — Module 1: AFLite‑lite & Bias Probes

This document summarizes the implementation of Module 1, which adds adversarial filtering (AFLite‑lite) and bias probes to the Inspect-based pipeline. The work integrates cleanly with the existing datasets → tasks → logs → analysis flow, adds a small CLI entrypoint, extends aggregation to retain module-specific fields, and includes comprehensive tests for critical ML components.

**Update (Hackathon Sprint)**: Added focused test coverage for highest-risk areas to ensure ML validity and robustness.

## Scope & Goals

- Detect MCQ items that are predictable from answer choices or shallow artifacts.
- Provide per-item predictability scores, boolean flags, and probe reasons to support dataset curation and robustness filtering presets.
- Keep logs Inspect-native, safe for public aggregation (IDs + safe flags/metadata only).

## Files Added / Modified

- Added `robustcbrn/qa/filters.py`
  - Implements AFLite‑lite choices-only classifier and heuristic probes.
  - Exposes `aflite_lite()` and `aflite_join_with_items()` utilities.
- Added `robustcbrn/tasks/aflite_screen.py`
  - CLI/task-like runner that loads an MCQ JSONL and emits Inspect-compatible logs with predictability flags.
- Added `robustcbrn/analysis/robustness.py`
  - Reporting helpers for flagged fraction and overlap with choices-only exploitable items.
- Modified `robustcbrn/analysis/aggregate.py`
  - Extends parsed per-sample fields to include `flag_predictable`, `predictability_score`, and `probe_hit`.
- Added tests `tests/test_aflite.py`
  - Smoke test for CLI runner and a minimal summary test.

## Implementation Details

### AFLite‑lite (choices-only classifier)

- Input view: items as `(id, choices, target)` with stem removed.
- Features per choice:
  - Bag-of-words over choice text, vocabulary capped at 2,000 tokens, counts normalized by character length.
  - Extra features: character length, token length, average word length proxy, normalized position index, binary longest-answer indicator, length rank, first-character ordinal.
- Model: binary logistic regression over per-choice candidates (positive = true choice), trained with L2 regularization.
  - Primary optimizer: SciPy L-BFGS (`scipy.optimize.minimize`), with analytical gradient.
  - Fallback: simple gradient descent if SciPy is unavailable.
- Cross-validation: K-fold grouped by item ID to avoid leakage; produces out-of-fold per-candidate probabilities.
- Per-item score: probability assigned to the true choice; `flag_predictable = (score ≥ τ)`.

### Heuristic Probes

- `longest_answer`: predicts index of the longest choice; hit if matches target.
- `position_only`: predicts the global majority target index; hit if equals item target.
- `alphabetical`: predicts alphabetically first choice; hit if equals item target.
- Hits are recorded as `probe_hit: List[str]` for downstream filtering presets and inspection.

### Utilities

- `aflite_lite(dataset, tau=0.7, k_folds=5, seed=123, max_vocab=2000, reg_lambda=1.0)`
  - Returns a `List[PredictabilityResult]` with `id`, `predictability_score`, `flag_predictable`, and `probe_hit`.
- `aflite_join_with_items(items, results)`
  - Joins AFLite outputs back to item dicts (adds the fields above), convenient for logging/export.

## Task / CLI Runner

- File: `robustcbrn/tasks/aflite_screen.py`
- Function: `run_aflite_screen(dataset_path, tau=0.7, k_folds=5, seed=123, max_items=None, out_path=None)`
  - Loads MCQ JSONL via `tasks.common.load_mcq_dataset()`.
  - Runs `aflite_lite` and writes Inspect-like JSON: `task='aflite_screen'`, `model='aflite-lite'`, `samples=[{id, predictability_score, flag_predictable, probe_hit}]`.
  - Default output: `logs/aflite_screen_<dataset-stem>.json`.
- CLI: `python -m robustcbrn.tasks.aflite_screen data/sample_sanitized.jsonl --tau 0.7 --k-folds 5 --seed 42`

## Aggregation & Analysis

- Aggregator changes `robustcbrn/analysis/aggregate.py`:
  - `SampleResult` extended with optional: `flag_predictable`, `predictability_score`, `probe_hit`.
  - Parser preserves these fields from any log that includes them; `probe_hit` lists are stored as comma-separated strings for CSV output.
- New analysis helpers `robustcbrn/analysis/robustness.py`:
  - `aflite_flag_summary(df, preset='balanced', tau=0.7)`
    - Presets:
      - conservative: score ≥ max(τ, 0.8) and ≥2 probes
      - balanced: score ≥ τ (default)
      - aggressive: score ≥ min(τ, 0.6) or ≥1 probe
    - Returns `{n, flagged_frac, ci_lo, ci_hi}` (CIs via bootstrap over items).
  - `aflite_overlap_with_choices_only(df, preset='balanced', tau=0.7, k=2)`
    - Computes overlap/Jaccard with exploitable items from `majority_consensus` on choices-only tasks.
    - Returns `{n, flagged_frac, exploitable_frac, overlap_frac, jaccard}`.

## Tests

- `tests/test_aflite.py`
  - `test_aflite_screen_smoke`: runs the CLI helper on 2 items, asserts output file and schema.
  - `test_aflite_summary_minimal`: builds a tiny DataFrame and checks that summary values are sane and count matches.
  - `test_kfold_no_data_leakage`: **Critical test** - Verifies K-fold grouping prevents same item appearing in train/test sets, ensuring ML validity.
  - `test_probe_implementations`: Unit tests for `_longest_choice` and `_alphabetical_choice` probe functions with edge cases.
  - `test_aflite_classifier_basic`: Integration test verifying classifier produces valid scores [0,1] and correctly identifies longest-answer patterns.
  - `test_edge_cases`: Robustness tests for empty datasets, single-item datasets, invalid target indices, and empty choices.
- All 6 tests pass via `pytest -v` (28.63s runtime).

## Usage Examples

- Generate AFLite logs:
  - `python -m robustcbrn.tasks.aflite_screen data/sample_sanitized.jsonl --tau 0.7 --k-folds 5 --seed 42`
- Aggregate results:
  - `python -m robustcbrn.analysis.aggregate --logs logs --out artifacts/results`
- Analyze flags and overlap (Python):
  - `from robustcbrn.analysis.aggregate import load_all_results`
  - `from robustcbrn.analysis.robustness import aflite_flag_summary, aflite_overlap_with_choices_only`
  - `df = load_all_results('logs')`
  - `aflite_flag_summary(df, preset='balanced', tau=0.7)`
  - `aflite_overlap_with_choices_only(df, preset='balanced', tau=0.7, k=2)`

## Safety & Data Handling

- Operates on sanitized dataset (`data/sample_sanitized.jsonl`); no hazardous stems are emitted to logs.
- Logs include only `id` and safe metadata fields (`predictability_score`, `flag_predictable`, `probe_hit`).

## Design Choices & Notes

- K-fold CV groups by item ID to avoid leakage between candidate choices (now tested via `test_kfold_no_data_leakage`).
- Logistic regression preferred for simplicity and portability; SciPy used when available, with a deterministic GD fallback.
- Vocabulary is capped (default 2,000) for speed and to avoid heavy dependencies.
- Heuristic probes capture common artifacts (longest-answer, position bias, alphabetical preference) and are combined via presets for flexible curation.

## Test Coverage Summary

**Original**: 2 basic tests (~5% coverage)
**Current**: 6 comprehensive tests covering:
- ML validity (K-fold data leakage prevention)
- Core functionality (probes, classifier)
- Edge cases (empty/invalid inputs)
- Integration (end-to-end scoring)

Trade-offs for hackathon timeline:
- Focused on highest-risk components
- Skipped performance benchmarks
- Minimal statistical validation
- Basic integration only

