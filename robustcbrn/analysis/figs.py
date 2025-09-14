from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

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
    yerr = [[e[0] for e in errs], [e[1] for e in errs]]
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
        yerr = [[e[0] for e in errs], [e[1] for e in errs]]
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


def save_calibration_plot(
    fig_path: str | Path,
    bin_data: list[dict],
    threshold: float = 0.0,
    title: str | None = None,
) -> None:
    """Save reliability diagram showing predicted vs actual accuracy bins.

    Args:
        fig_path: Path to save the figure
        bin_data: List of bin statistics from calibration.bin_predictions_by_confidence
        threshold: Confidence threshold used (for title)
        title: Optional custom title
    """
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)

    if not bin_data:
        return

    # Extract data for plotting
    avg_confidences = [b['avg_confidence'] for b in bin_data if b['count'] > 0]
    accuracies = [b['accuracy'] for b in bin_data if b['count'] > 0]
    counts = [b['count'] for b in bin_data if b['count'] > 0]

    if not avg_confidences:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Top plot: Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')

    # Scale point size by count
    max_count = max(counts) if counts else 1
    sizes = [100 * (c / max_count) for c in counts]

    ax1.scatter(avg_confidences, accuracies, s=sizes, alpha=0.6, color='#4C78A8')
    ax1.set_xlabel('Mean Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if title:
        ax1.set_title(title)
    else:
        ax1.set_title(f'Calibration Plot (t={threshold})')

    # Bottom plot: Sample distribution
    bin_centers = [(b['lower'] + b['upper']) / 2 for b in bin_data]
    bin_counts = [b['count'] for b in bin_data]

    ax2.bar(bin_centers, bin_counts, width=1.0/len(bin_data), color='#72B7B2', alpha=0.7)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def save_confidence_histogram(
    fig_path: str | Path,
    confidences: list[float],
    correctness: list[bool],
    threshold: float = 0.0,
    title: str | None = None,
    bins: int = 20,
) -> None:
    """Save confidence histogram with accuracy overlay.

    Args:
        fig_path: Path to save the figure
        confidences: List of confidence values
        correctness: List of boolean correctness values
        threshold: Confidence threshold (for visualization)
        title: Optional custom title
        bins: Number of histogram bins
    """
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)

    if not confidences:
        return

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create histogram
    n, bins_edges, patches = ax.hist(confidences, bins=bins, alpha=0.6, color='#72B7B2', label='All samples')

    # Overlay correct predictions
    correct_conf = confidences[correctness]
    ax.hist(correct_conf, bins=bins_edges, alpha=0.6, color='#4C78A8', label='Correct')

    # Add threshold line if non-zero
    if threshold > 0:
        ax.axvline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold={threshold}')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Confidence Distribution (t={threshold})')

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
