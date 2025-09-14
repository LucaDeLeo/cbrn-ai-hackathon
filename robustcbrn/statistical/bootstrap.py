# Bootstrap confidence intervals for generic statistics (NumPy only).

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


@dataclass
class BootstrapResult:
    """Results from bootstrap confidence interval calculation."""
    statistic: float
    confidence_interval: Tuple[float, float]
    bootstrap_estimates: np.ndarray
    confidence_level: float
    n_iterations: int
    method: str
    runtime_seconds: float
    converged: bool = False
    convergence_iteration: Optional[int] = None


def _check_convergence(bootstrap_stats: List[float], threshold: float) -> bool:
    """Simple CI-width convergence check using last 100 vs. previous 100 samples."""
    if len(bootstrap_stats) < 200:
        return False
    recent = np.array(bootstrap_stats[-100:])
    previous = np.array(bootstrap_stats[-200:-100])
    recent_width = np.percentile(recent, 97.5) - np.percentile(recent, 2.5)
    prev_width = np.percentile(previous, 97.5) - np.percentile(previous, 2.5)
    return recent_width / max(prev_width, 1e-12) > 0.98  # nearly stabilized


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    adaptive: bool = True,
    convergence_threshold: float = 0.01,
    min_iterations: int = 1_000,
    random_seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Generic bootstrap confidence interval for any statistic.

    Args:
        data: 1D array of samples (resampled with replacement).
        statistic_func: function mapping 1D array -> float statistic.
        n_bootstrap: number of bootstrap resamples (max).
        confidence_level: e.g., 0.95.
        method: "percentile" or "bca" (BCa here is a light, approximate variant).
        adaptive: stop early if CI stabilizes (cheap heuristic).
        convergence_threshold: unused in this light heuristic; can be tuned.
        min_iterations: don't stop earlier than this many iterations.
        random_seed: for determinism.
    Returns:
        BootstrapResult
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    start_time = time.time()

    # original statistic on given data
    original_stat = statistic_func(data)
    n = len(data)
    alpha = 1 - confidence_level

    estimates: List[float] = []
    converged = False
    convergence_iteration: Optional[int] = None

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        estimates.append(float(statistic_func(sample)))

        if adaptive and i >= min_iterations and i % 100 == 0:
            if _check_convergence(estimates, convergence_threshold):
                converged = True
                convergence_iteration = i + 1
                break

    estimates_arr = np.array(estimates if estimates else [original_stat], dtype=float)

    if method == "percentile":
        lo, hi = np.quantile(estimates_arr, [alpha / 2, 1 - alpha / 2])
    else:
        # light BCa approximation (does not compute full acceleration)
        z0 = _inv_norm_cdf(np.mean(estimates_arr < original_stat) or 1e-6)
        z_lo = z0 + _inv_norm_cdf(alpha / 2)
        z_hi = z0 + _inv_norm_cdf(1 - alpha / 2)
        lo = np.quantile(estimates_arr, _norm_cdf(z_lo))
        hi = np.quantile(estimates_arr, _norm_cdf(z_hi))

    runtime = time.time() - start_time
    return BootstrapResult(
        statistic=float(original_stat),
        confidence_interval=(float(lo), float(hi)),
        bootstrap_estimates=estimates_arr,
        confidence_level=confidence_level,
        n_iterations=len(estimates_arr),
        method=method,
        runtime_seconds=float(runtime),
        converged=converged,
        convergence_iteration=convergence_iteration,
    )


# Convenience wrappers
def bootstrap_mean_ci(data: np.ndarray, **kwargs) -> BootstrapResult:
    return bootstrap_ci(data, np.mean, **kwargs)

def bootstrap_median_ci(data: np.ndarray, **kwargs) -> BootstrapResult:
    return bootstrap_ci(data, np.median, **kwargs)

def bootstrap_proportion_ci(data: np.ndarray, **kwargs) -> BootstrapResult:
    """For 0/1 arrays."""
    return bootstrap_ci(data, lambda x: float(np.mean(x)), **kwargs)


# A helper CI for chi-square p-values, if you want it
def bootstrap_chi_square_pvalue_ci(observed: np.ndarray, expected: np.ndarray, **kwargs) -> BootstrapResult:
    """
    Bootstrap CI for the chi-square statistic using pseudo-data expanded to counts.
    (We return a CI on the statistic; callers can convert to a p-value as needed.)
    """
    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have equal length")
    if np.any(expected <= 0):
        raise ValueError("expected must be positive")

    def chi_square_stat(sample: np.ndarray) -> float:
        # sample are category indices; rebuild counts against categories present in sample
        cats = np.unique(sample)
        counts = np.array([np.sum(sample == c) for c in cats], dtype=float)
        n = counts.sum()
        exp = np.full(len(cats), n / len(cats))
        return float(np.sum((counts - exp) ** 2 / exp))

    # Expand pseudo-data according to observed counts
    pseudo = []
    for i, c in enumerate(observed):
        pseudo.extend([i] * int(c))
    pseudo = np.array(pseudo, dtype=int)
    return bootstrap_ci(pseudo, chi_square_stat, **kwargs)


# --- math helpers (normal CDF / inverse CDF) --------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))

def _inv_norm_cdf(q: float) -> float:
    # Acklam/Rational approx
    a = [0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    if q <= 0 or q >= 1:
        return float("nan")
    if q > 0.5:
        sign = 1; q = 1 - q
    else:
        sign = -1
    t = np.sqrt(-2.0 * np.log(q))
    num = a[6]
    den = 1.0
    for i in range(6):
        num = num * t + a[5 - i]
        if i < 5:
            den = den * t + b[4 - i]
    return sign * (t - num / den)

def _erf(x: float) -> float:
    # Abramowitz & Stegun
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y
