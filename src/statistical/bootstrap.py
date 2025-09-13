# src/statistical/bootstrap.py
"""Bootstrap confidence interval implementation for bias measurement uncertainty."""

from __future__ import annotations

import time
from typing import Callable, Tuple, List, Any, Optional
import numpy as np
from dataclasses import dataclass


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


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    adaptive: bool = True,
    convergence_threshold: float = 0.01,
    min_iterations: int = 1000,
    random_seed: Optional[int] = None
) -> BootstrapResult:
    """
    Calculate bootstrap confidence intervals for any statistic.

    Args:
        data: Input data array
        statistic_func: Function that takes array and returns statistic
        n_bootstrap: Maximum bootstrap iterations (default 10,000)
        confidence_level: Confidence level (default 0.95)
        method: "percentile" or "bca" (bias-corrected accelerated)
        adaptive: Enable early stopping when CI width stabilizes
        convergence_threshold: Relative change threshold for convergence
        min_iterations: Minimum iterations before checking convergence
        random_seed: Random seed for reproducibility

    Returns:
        BootstrapResult with confidence interval and metadata
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    start_time = time.time()

    # Calculate original statistic
    original_stat = statistic_func(data)
    n = len(data)
    alpha = 1 - confidence_level

    # Bootstrap sampling with adaptive stopping
    bootstrap_stats = []
    converged = False
    convergence_iteration = None

    for i in range(n_bootstrap):
        # Generate bootstrap sample
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)

        # Check convergence (adaptive mode)
        if adaptive and i >= min_iterations and i % 100 == 0:
            if _check_convergence(bootstrap_stats, convergence_threshold):
                converged = True
                convergence_iteration = i + 1
                break

    bootstrap_estimates = np.array(bootstrap_stats)
    n_iterations = len(bootstrap_estimates)

    # Calculate confidence interval
    if method == "percentile":
        ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    elif method == "bca":
        ci_lower, ci_upper = _bca_confidence_interval(
            data, statistic_func, bootstrap_estimates, original_stat, confidence_level
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime = time.time() - start_time

    return BootstrapResult(
        statistic=original_stat,
        confidence_interval=(ci_lower, ci_upper),
        bootstrap_estimates=bootstrap_estimates,
        confidence_level=confidence_level,
        n_iterations=n_iterations,
        method=method,
        runtime_seconds=runtime,
        converged=converged,
        convergence_iteration=convergence_iteration
    )


def _check_convergence(bootstrap_stats: List[float], threshold: float) -> bool:
    """Check if bootstrap CI has converged."""
    if len(bootstrap_stats) < 200:
        return False

    # Compare CI width from last 100 vs previous 100 iterations
    recent = np.array(bootstrap_stats[-100:])
    previous = np.array(bootstrap_stats[-200:-100])

    recent_width = np.percentile(recent, 97.5) - np.percentile(recent, 2.5)
    previous_width = np.percentile(previous, 97.5) - np.percentile(previous, 2.5)

    if previous_width == 0:
        return False

    relative_change = abs(recent_width - previous_width) / previous_width
    return relative_change < threshold


def _bca_confidence_interval(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    bootstrap_estimates: np.ndarray,
    original_stat: float,
    confidence_level: float
) -> Tuple[float, float]:
    """Calculate bias-corrected and accelerated (BCa) confidence interval."""
    n = len(data)
    alpha = 1 - confidence_level

    # Bias correction
    n_less = np.sum(bootstrap_estimates < original_stat)
    z0 = _norm_ppf(n_less / len(bootstrap_estimates)) if n_less > 0 else 0

    # Acceleration correction using jackknife
    jackknife_stats = []
    for i in range(n):
        jackknife_sample = np.concatenate([data[:i], data[i + 1:]])
        jackknife_stats.append(statistic_func(jackknife_sample))

    jackknife_stats = np.array(jackknife_stats)
    jackknife_mean = np.mean(jackknife_stats)

    acceleration = np.sum((jackknife_mean - jackknife_stats) ** 3)
    acceleration /= 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)

    # Calculate BCa endpoints
    z_alpha_2 = _norm_ppf(alpha / 2)
    z_1_alpha_2 = _norm_ppf(1 - alpha / 2)

    alpha_1 = _norm_cdf(z0 + (z0 + z_alpha_2) / (1 - acceleration * (z0 + z_alpha_2)))
    alpha_2 = _norm_cdf(z0 + (z0 + z_1_alpha_2) / (1 - acceleration * (z0 + z_1_alpha_2)))

    # Ensure valid percentiles
    alpha_1 = np.clip(alpha_1, 1e-6, 1 - 1e-6)
    alpha_2 = np.clip(alpha_2, 1e-6, 1 - 1e-6)

    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha_1)
    ci_upper = np.percentile(bootstrap_estimates, 100 * alpha_2)

    return ci_lower, ci_upper


def _norm_ppf(q: float) -> float:
    """Approximate normal percent point function (inverse CDF)."""
    if q <= 0:
        return -np.inf
    if q >= 1:
        return np.inf
    if q == 0.5:
        return 0.0

    # Beasley-Springer-Moro approximation
    a = [0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]

    if q > 0.5:
        q = 1 - q
        sign = 1
    else:
        sign = -1

    t = np.sqrt(-2 * np.log(q))
    numer = a[6]
    denom = 1

    for i in range(6):
        numer = numer * t + a[5 - i]
        if i < 5:
            denom = denom * t + b[4 - i]

    return sign * (t - numer / denom)


def _norm_cdf(x: float) -> float:
    """Approximate normal cumulative distribution function."""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x: float) -> float:
    """Approximate error function."""
    # Abramowitz and Stegun approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


def _approximate_normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using Abramowitz and Stegun approximation."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    sign = 1 if z >= 0 else -1
    z = abs(z)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)

    return 0.5 * (1 + sign * y)


def _approximate_chi2_pvalue(chi2_stat: float, df: int) -> float:
    """Approximate chi-square p-value using gamma function approximation."""
    if df == 1:
        z = np.sqrt(chi2_stat)
        p_value = 2 * (1 - _approximate_normal_cdf(z))
    elif df <= 30:
        p_value = np.exp(-chi2_stat / 2) * (chi2_stat / 2) ** (df / 2 - 1)
        p_value = np.clip(p_value, 0, 1)
    else:
        mean = df
        variance = 2 * df
        z = (chi2_stat - mean) / np.sqrt(variance)
        p_value = 1 - _approximate_normal_cdf(z)

    return float(np.clip(p_value, 1e-10, 1.0))


# Convenience functions for common statistics
def bootstrap_mean_ci(data: np.ndarray, **kwargs) -> BootstrapResult:
    """Bootstrap confidence interval for mean."""
    return bootstrap_ci(data, np.mean, **kwargs)


def bootstrap_median_ci(data: np.ndarray, **kwargs) -> BootstrapResult:
    """Bootstrap confidence interval for median."""
    return bootstrap_ci(data, np.median, **kwargs)


def bootstrap_proportion_ci(data: np.ndarray, **kwargs) -> BootstrapResult:
    """Bootstrap confidence interval for proportion (assumes 0/1 data)."""
    return bootstrap_ci(data, lambda x: np.mean(x), **kwargs)


def bootstrap_chi_square_pvalue_ci(observed: np.ndarray, expected: np.ndarray,
                                   **kwargs) -> BootstrapResult:
    """Bootstrap CI for chi-square test p-value (for position bias analysis)."""

    def chi_square_stat(obs_data):
        # Resample and recalculate chi-square
        n_categories = len(expected)
        resampled_counts = np.zeros(n_categories, dtype=int)

        # Multinomial resampling based on original proportions
        total = len(obs_data)
        probs = expected / np.sum(expected)
        resampled_counts = np.random.multinomial(total, probs)

        # Calculate chi-square statistic
        chi_sq = np.sum((resampled_counts - expected) ** 2 / expected)

        # Convert to approximate p-value (simplified)
        df = len(expected) - 1
        if df == 1:
            return _approximate_chi2_pvalue(chi_sq, df)
        else:
            # Rough approximation for higher df
            return max(0.001, min(0.999, np.exp(-chi_sq / (2 * df))))

    # Create pseudo-data from observed counts for resampling
    pseudo_data = []
    for i, count in enumerate(observed):
        pseudo_data.extend([i] * int(count))
    pseudo_data = np.array(pseudo_data)

    return bootstrap_ci(pseudo_data, chi_square_stat, **kwargs)
