from __future__ import annotations

import math

import pandas as pd
from scipy.stats import binomtest  # type: ignore


def test_mcnemar_basic_counts_and_pvalue():
    # Construct synthetic dataset with known discordant pairs
    # For 12 (id, model) groups: b=10, c=2
    rows = []
    model = "m"
    task = "mcq_full"
    # Create 10 groups where orig wrong, variants majority correct (b)
    for i in range(10):
        gid = f"g{i}"
        rows.append({"id": gid, "model": model, "task": task, "variant": "orig", "correct": False})
        rows.append({"id": gid, "model": model, "task": task, "variant": "v1", "correct": True})
        rows.append({"id": gid, "model": model, "task": task, "variant": "v2", "correct": True})
    # Create 2 groups where orig correct, variants majority wrong (c)
    for i in range(10, 12):
        gid = f"g{i}"
        rows.append({"id": gid, "model": model, "task": task, "variant": "orig", "correct": True})
        rows.append({"id": gid, "model": model, "task": task, "variant": "v1", "correct": False})
        rows.append({"id": gid, "model": model, "task": task, "variant": "v2", "correct": False})

    df = pd.DataFrame(rows)
    from robustcbrn.analysis.robustness import mcnemar_orig_vs_variants

    out = mcnemar_orig_vs_variants(df, task_name="mcq_full")
    assert out["n"] == 12
    assert out["b"] == 10
    assert out["c"] == 2
    # Compare p-value to exact binomial
    expected = float(binomtest(k=min(10, 2), n=12, p=0.5, alternative="two-sided").pvalue)
    assert math.isclose(out["p_value"], expected, rel_tol=1e-12, abs_tol=1e-12)


def test_bh_fdr_rejections_and_qvalues():
    # Example p-values
    ps = [0.001, 0.01, 0.02, 0.04, 0.2, 0.6]
    from robustcbrn.analysis.robustness import bh_fdr

    out = bh_fdr(ps, alpha=0.05)
    rejected = out["rejected"]
    q = out["q_values"]

    # Expect first three to be rejected under BH at alpha=0.05
    assert rejected == [True, True, True, False, False, False]
    # Q-values must be in [0,1]
    assert all(0.0 <= x <= 1.0 for x in q)
    # Monotonicity after sorting by p-values: q should be non-decreasing
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    q_sorted = [q[i] for i in order]
    assert all(q_sorted[i] <= q_sorted[i + 1] + 1e-12 for i in range(len(q_sorted) - 1))

