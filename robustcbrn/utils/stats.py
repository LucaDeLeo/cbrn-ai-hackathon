from __future__ import annotations

import math
import random
from collections.abc import Iterable


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    yt = list(y_true)
    yp = list(y_pred)
    assert len(yt) == len(yp)
    if not yt:
        return 0.0
    return sum(int(a == b) for a, b in zip(yt, yp, strict=False)) / len(yt)


def stderr(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(max(0.0, p * (1 - p)) / n)


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int((alpha / 2) * n_boot) - 1)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1)
    return (means[lo_idx], means[hi_idx])
