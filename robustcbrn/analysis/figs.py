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
