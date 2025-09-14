from __future__ import annotations

import os

from math import isclose

import numpy as np


def test_power_two_proportions_null_returns_alpha():
    from robustcbrn.analysis.robustness import power_two_proportions

    alpha = 0.05
    # Under null (p1 == p2), two-sided power equals test size alpha
    p = power_two_proportions(0.5, 0.5, n1=100, n2=100, alpha=alpha)
    assert isclose(p, alpha, rel_tol=1e-3, abs_tol=1e-3)


def test_power_two_proportions_monotonic_in_n():
    from robustcbrn.analysis.robustness import power_two_proportions

    p_small = power_two_proportions(0.70, 0.60, n1=50, n2=50, alpha=0.05)
    p_large = power_two_proportions(0.70, 0.60, n1=200, n2=200, alpha=0.05)
    assert 0.0 <= p_small <= 1.0
    assert 0.0 <= p_large <= 1.0
    assert p_large >= p_small - 1e-9


def test_required_sample_size_two_proportions_properties():
    from robustcbrn.analysis.robustness import required_sample_size_two_proportions

    n1, n2 = required_sample_size_two_proportions(0.70, 0.60, alpha=0.05, power=0.8, ratio=1.0)
    assert isinstance(n1, int) and isinstance(n2, int)
    assert n1 > 0 and n2 == n1

    # Smaller delta should require larger N
    n1_small, _ = required_sample_size_two_proportions(0.65, 0.60, alpha=0.05, power=0.8, ratio=1.0)
    assert n1_small >= n1

    # Stricter alpha increases N
    n1_alpha, _ = required_sample_size_two_proportions(0.70, 0.60, alpha=0.01, power=0.8, ratio=1.0)
    assert n1_alpha >= n1

    # Higher power target increases N
    n1_power, _ = required_sample_size_two_proportions(0.70, 0.60, alpha=0.05, power=0.9, ratio=1.0)
    assert n1_power >= n1

    # Ratio controls group 2 size
    n1_r, n2_r = required_sample_size_two_proportions(0.70, 0.60, alpha=0.05, power=0.8, ratio=2.0)
    assert n2_r >= 2 * n1_r - 1  # allow ceil


def test_figs_smoke(tmp_path):
    from robustcbrn.analysis.figs import save_bar_ci, save_paired_delta, save_fragility

    out1 = tmp_path / "bar_ci.png"
    save_bar_ci(
        fig_path=str(out1),
        labels=["A", "B", "C"],
        means=[0.6, 0.7, 0.8],
        ci_los=[0.55, 0.65, 0.75],
        ci_his=[0.65, 0.75, 0.85],
        title="Accuracy (95% CI)",
        ylabel="Acc",
    )
    assert out1.exists() and os.path.getsize(out1) > 0

    out2 = tmp_path / "paired_delta.png"
    save_paired_delta(
        fig_path=str(out2),
        labels=["ModelX"],
        deltas=[-0.08],
        ci_los=[-0.12],
        ci_his=[-0.04],
        title="MCQ − Cloze Δ",
        ylabel="Δ",
    )
    assert out2.exists() and os.path.getsize(out2) > 0

    out3 = tmp_path / "fragility.png"
    save_fragility(
        fig_path=str(out3),
        labels=["ModelX"],
        flip_rates=[0.22],
        ci_los=[0.18],
        ci_his=[0.26],
        title="Fragility",
        ylabel="Flip rate",
    )
    assert out3.exists() and os.path.getsize(out3) > 0

