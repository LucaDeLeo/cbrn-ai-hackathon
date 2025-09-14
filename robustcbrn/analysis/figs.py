from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_bar(
    fig_path: str | Path,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str = "",
) -> None:
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    x = range(len(labels))
    plt.bar(x, values, color="#4C78A8")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def save_hist(
    fig_path: str | Path,
    values: list[float],
    bins: int = 20,
    title: str = "Histogram",
) -> None:
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color="#72B7B2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def save_bar_ci(
    fig_path: str | Path,
    labels: list[str],
    means: list[float],
    ci_los: list[float],
    ci_his: list[float],
    title: str,
    ylabel: str = "",
) -> None:
    """Bar chart with 95% CI error bars.

    Expects ci_los/ci_his as absolute bounds; converts to error magnitudes.
    """
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    x = list(range(len(labels)))
    errs = [
        [max(0.0, m - lo), max(0.0, hi - m)]
        for m, lo, hi in zip(means, ci_los, ci_his, strict=False)
    ]
    # Transpose to match yerr shape [[lower...], [upper...]]
    yerr = [list(e[0] for e in errs), list(e[1] for e in errs)]
    plt.bar(x, means, color="#4C78A8")
    plt.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#333333", capsize=4)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def save_paired_delta(
    fig_path: str | Path,
    labels: list[str],
    deltas: list[float],
    ci_los: list[float] | None = None,
    ci_his: list[float] | None = None,
    title: str = "Paired Δ",
    ylabel: str = "Δ",
) -> None:
    """Bar plot for paired deltas (e.g., MCQ vs Cloze, orig vs variants)."""
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    x = list(range(len(labels)))
    plt.axhline(0.0, color="#888888", linewidth=1, linestyle="--")
    plt.bar(x, deltas, color="#72B7B2")
    if ci_los is not None and ci_his is not None:
        errs = [
            [max(0.0, m - lo), max(0.0, hi - m)]
            for m, lo, hi in zip(deltas, ci_los, ci_his, strict=False)
        ]
        yerr = [list(e[0] for e in errs), list(e[1] for e in errs)]
        plt.errorbar(x, deltas, yerr=yerr, fmt="none", ecolor="#333333", capsize=4)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def save_fragility(
    fig_path: str | Path,
    labels: list[str],
    flip_rates: list[float],
    ci_los: list[float] | None = None,
    ci_his: list[float] | None = None,
    title: str = "Fragility (flip rate)",
    ylabel: str = "Flip rate",
) -> None:
    """Bar plot of flip rates with optional CI error bars."""
    save_bar_ci(
        fig_path=fig_path,
        labels=labels,
        means=flip_rates,
        ci_los=ci_los or [0.0] * len(flip_rates),
        ci_his=ci_his or [0.0] * len(flip_rates),
        title=title,
        ylabel=ylabel,
    )
