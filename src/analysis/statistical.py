# src/analysis/statistical.py
"""Statistical analysis functions for the statistical battery."""

import numpy as np
from typing import List, Tuple, Callable, Any
import random

def chi_square_test(observed: np.ndarray, expected: np.ndarray) -> Tuple[float, float]:
    """
    Chi-square goodness-of-fit test.
    
    Args:
        observed: Observed frequencies
        expected: Expected frequencies
        
    Returns:
        Tuple of (chi-square statistic, p-value)
    """
    if len(observed) != len(expected):
        raise ValueError("Observed and expected arrays must have same length")
    
    if np.any(expected <= 0):
        raise ValueError("All expected frequencies must be positive")
    
    chi_square_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    
    # Simple p-value approximation using chi-square distribution
    # For more accurate results, you might want to use scipy.stats.chi2
    p_value = _approximate_chi2_pvalue(float(chi_square_stat), int(df))
    
    return float(chi_square_stat), float(p_value)

def _approximate_chi2_pvalue(chi2_stat: float, df: int) -> float:
    """Compute chi-square p-value using regularized upper incomplete gamma."""
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    if chi2_stat < 0:
        raise ValueError("Chi-square statistic must be non-negative")

    a = 0.5 * df
    x = 0.5 * chi2_stat

    if x == 0.0:
        return 1.0

    def _gammainc_series(a: float, x: float) -> float:
        # Regularized lower incomplete gamma P(a, x)
        import math
        eps = 1e-14
        term = 1.0 / a
        summation = term
        n = 1
        while True:
            term *= x / (a + n)
            summation += term
            if abs(term) < abs(summation) * eps or n > 100000:
                break
            n += 1
        return math.exp(-x + a * math.log(x) - math.lgamma(a)) * summation

    def _gammainc_cf(a: float, x: float) -> float:
        # Regularized upper incomplete gamma Q(a, x) via continued fraction
        import math
        eps = 1e-14
        max_iter = 100000
        tiny = 1e-300
        b = x + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, max_iter + 1):
            an = -i * (i - a)
            b = b + 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h

    if x < a + 1.0:
        p = _gammainc_series(a, x)
        q = 1.0 - p
    else:
        q = _gammainc_cf(a, x)
    return float(max(0.0, min(1.0, q)))

def calculate_bootstrap_ci(
    data: np.ndarray, 
    statistic_func: Callable[[np.ndarray], float], 
    n_iterations: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data array
        statistic_func: Function that calculates the statistic
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_stats = []
    
    for _ in range(n_iterations):
        # Bootstrap sample
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        stat_value = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat_value)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return float(lower_bound), float(upper_bound)

def detect_position_bias(questions: List[dict]) -> dict:
    """
    Detect position bias in multiple choice questions.
    
    Args:
        questions: List of question dictionaries with 'answer_index' field
        
    Returns:
        Dictionary containing bias detection results
    """
    if not questions:
        return {"error": "No questions provided"}
    
    # Extract answer positions
    positions = [q.get('answer_index', 0) for q in questions if 'answer_index' in q]
    
    if not positions:
        return {"error": "No valid answer positions found"}
    
    # Calculate frequencies
    max_position = max(positions)
    observed = np.bincount(positions, minlength=max_position + 1)
    
    # Need at least 2 positions for chi-square test
    if len(observed) < 2:
        return {
            "error": "Insufficient position diversity for chi-square test",
            "observed_frequencies": observed.tolist(),
            "expected_frequencies": [],
            "chi_square_statistic": 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "significant": False
        }
    
    expected = np.full(len(observed), len(positions) / len(observed))
    
    # Chi-square test
    chi2_stat, p_value = chi_square_test(observed, expected)
    
    # Calculate effect size (Cramer's V)
    n = len(positions)
    cramers_v = np.sqrt(chi2_stat / (n * (len(observed) - 1))) if n > 1 else 0
    
    return {
        "chi_square_statistic": chi2_stat,
        "p_value": p_value,
        "effect_size": cramers_v,
        "observed_frequencies": observed.tolist(),
        "expected_frequencies": expected.tolist(),
        "significant": p_value < 0.05
    }

def analyze_lexical_patterns(questions: List[dict]) -> dict:
    """
    Analyze lexical patterns in questions and choices.
    
    This is a placeholder implementation for future development.
    
    Args:
        questions: List of question dictionaries
        
    Returns:
        Dictionary containing pattern analysis results
    """
    # Placeholder implementation
    return {
        "patterns_detected": [],
        "effect_sizes": {},
        "message": "Lexical pattern analysis not yet implemented"
    }
