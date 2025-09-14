# Module 6 — Formal Measurement & Reporting

This document details the implementation work completed for Module 6, focusing on formal statistical measurement, paired tests, multiple references, FDR control, and reporting utilities. Changes were made surgically to the existing Inspect-based pipeline without touching hazardous content.

## Scope

- Add robust statistical metrics (bootstrap CIs, paired McNemar, BH-FDR, power analysis, multi-reference matching).
- Extend plotting helpers to render CIs, paired deltas, and fragility-style visuals.
- Update report templates to include formal measurement sections and result placeholders.
- Add unit tests for McNemar and BH-FDR on synthetic data.

## Code Changes

- `robustcbrn/analysis/robustness.py`
  - New: `bh_fdr(p_values, alpha=0.05)`
    - Benjamini–Hochberg FDR control.
    - Returns `{"rejected": List[bool], "q_values": List[float], "alpha": float}`.
  - New: `required_sample_size_two_proportions(p1, p2, alpha=0.05, power=0.8, ratio=1.0)`
    - Normal-approx sample size calculation for a two-proportion z-test.
    - Returns `(n1, n2)` with `n2 = ceil(ratio * n1)`.
  - New: `power_two_proportions(p1, p2, n1, n2=None, alpha=0.05)`
    - Approximate power for two-proportion z-test using a two-sided normal approximation: `Φ(-z_{α/2}-z) + 1 - Φ(z_{α/2}-z)`.
    - Returns power in `[0,1]`; under the null (`p1==p2`) power ≈ α.
  - New: `multi_reference_match_rates(df, pred_col="pred_text", target_col="target_text", synonyms_col="target_synonyms")`
    - Exact and normalized string-match rates with optional synonyms.
    - Returns `{n, exact, exact_ci, normalized, normalized_ci}` using `utils.stats.bootstrap_ci`.
  - Kept/used: `mcnemar_orig_vs_variants(df, task_name)`
    - Computes discordant counts `(b, c)` and exact binomial p-value via SciPy.
    - Hardened type handling: coerces `correct` to numeric then boolean to safely handle string-encoded booleans.
  - Helpers: `_normalize_text` for normalized matching; imports `scipy.stats.norm` for z-based calcs.

- `robustcbrn/analysis/figs.py`
  - New: `save_bar_ci(fig_path, labels, means, ci_los, ci_his, title, ylabel="")`
    - Bar chart with asymmetric error bars using absolute CI bounds.
  - New: `save_paired_delta(fig_path, labels, deltas, ci_los=None, ci_his=None, title="Paired Δ", ylabel="Δ")`
    - For paired comparisons (e.g., MCQ vs Cloze; Orig vs Variants).
  - New: `save_fragility(fig_path, labels, flip_rates, ci_los=None, ci_his=None, title="Fragility (flip rate)", ylabel="Flip rate")`
    - Convenience wrapper that delegates to `save_bar_ci`.

- Docs
  - `docs/results/report.md`
    - Added “Formal Measurement Summary” section outlining: bootstrap CIs, McNemar, multi-reference matching, BH-FDR (α=0.05), and normal-approx power analysis.
    - Clarified multi-reference prerequisite: requires text columns (`pred_text`/`target_text`/`target_synonyms`) present in logs or joined externally.
  - `docs/results/results-template.md`
    - Added placeholders for:
      - McNemar (b, c, p) and FDR-adjusted q-value
      - Multi-reference exact/normalized rates with CIs
      - Power analysis (target Δ and required n per group)

- Tests
  - `tests/test_formal_stats.py`
    - `test_mcnemar_basic_counts_and_pvalue` verifies `(b, c)` counts and exact p-value against `scipy.stats.binomtest`.
    - `test_bh_fdr_rejections_and_qvalues` checks BH step-up rejections and q-value monotonicity under sorting.
  - New: `tests/test_power_and_plots.py`
    - Power: null-case equals α; monotonicity in N; sample-size sensitivity to δ, α, power, and allocation ratio.
    - Plots: smoke tests for `save_bar_ci`, `save_paired_delta`, and `save_fragility` (files created, non-empty).

## Usage Notes

- FDR control
  - Example:
    ```python
    from robustcbrn.analysis.robustness import bh_fdr
    out = bh_fdr([0.001, 0.01, 0.04, 0.2], alpha=0.05)
    # out["rejected"] -> [True, True, True, False]
    # out["q_values"] -> adjusted p-values (q-values)
    ```

- Power and sample size
  - Example:
    ```python
    from robustcbrn.analysis.robustness import required_sample_size_two_proportions, power_two_proportions
    n1, n2 = required_sample_size_two_proportions(0.70, 0.60, alpha=0.05, power=0.8)
    pow_est = power_two_proportions(0.70, 0.60, n1, n2)
    ```

- Multi-reference matching
  - Input DataFrame must include `pred_text`; `target_text` and `target_synonyms` are optional.
  - Example:
    ```python
    import pandas as pd
    from robustcbrn.analysis.robustness import multi_reference_match_rates

    df = pd.DataFrame({
        "pred_text": ["Fever", "fever", "Pyrexia"],
        "target_text": ["Fever", "Fever", "Fever"],
        "target_synonyms": ["Pyrexia; high temp", None, "fever"],
    })
    out = multi_reference_match_rates(df)
    # out -> {n, exact, exact_ci, normalized, normalized_ci}
    ```

- Plotting
  - CI bar chart:
    ```python
    from robustcbrn.analysis.figs import save_bar_ci
    save_bar_ci("artifacts/figs/acc_ci.png", ["A","B"], [0.7, 0.6], [0.65, 0.55], [0.75, 0.65], "Accuracy (95% CI)")
    ```
  - Paired delta chart:
    ```python
    from robustcbrn.analysis.figs import save_paired_delta
    save_paired_delta("artifacts/figs/mcq_cloze_delta.png", ["ModelX"], [-0.08], [ -0.12 ], [ -0.04 ], "MCQ − Cloze Δ")
    ```
  - Fragility (flip rate):
    ```python
    from robustcbrn.analysis.figs import save_fragility
    save_fragility("artifacts/figs/fragility.png", ["ModelX"], [0.22], [0.18], [0.26])
    ```

## Integration Points

- Aggregator wiring
  - Aggregator (`robustcbrn/analysis/aggregate.py`) updated to compute and store:
    - McNemar per task with `variant` present (orig vs non-orig), with BH-FDR across tasks.
    - Multi-reference match rates (exact/normalized) when text columns are present.
    - Power analysis for MCQ vs Cloze using observed accuracies and pair counts.
  - Figures are generated under `artifacts/figs/`:
    - `mcq_cloze_delta.png`: paired Δ with 95% CI.
    - `benign_pairs.png`: compliance/refusal/consistency with 95% CIs.

- Data expectations for robustness metrics
  - `mcnemar_orig_vs_variants`: expects rows with `task`, `variant` (`"orig"` vs non-`orig`), grouped by `(id, model)` and a boolean-like `correct` column.
  - CI-based metrics use `utils.stats.bootstrap_ci(values)`; values are cast to floats in `[0,1]`.
  - Multi-reference metrics require text fields: `pred_text` (predictions), `target_text` (primary reference), and optional `target_synonyms`. Provide via logs or by joining an external table prior to aggregation.

## Safety & Dependencies

- Safety: No hazardous content introduced; functions operate on numeric flags, IDs, and simple text normalization. Public artifacts remain ID- and aggregate-based.
- Dependencies: SciPy and Matplotlib are already declared in `requirements.txt`; no new packages added.

## Validation

- Ran `pytest -q`: all tests pass.
- Patched `SchemaValidationError` messaging in `robustcbrn/utils/validation.py` to include specific field-level details (e.g., `Missing required field: prompt_unsafe`), fixing the benign-pairs integration test.
- Added unit tests for power/sample-size and plotting; confirmed deterministic bootstrap behavior and figure generation.
- A `PytestUnknownMarkWarning` for `@pytest.mark.slow` persists but is benign.

## Next Steps (Optional)

- Add a CLI/Makefile target to regenerate figures only (without re-aggregation).
- Add config flags to toggle BH-FDR and power calculations at runtime.
- Extend figure generation: fragility and consistency plots across models/tasks.
- Include small HTML summary that embeds the generated figures and key metrics.
