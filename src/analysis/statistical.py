# src/analysis/statistical.py
"""Statistical analysis functions for the statistical battery."""

import numpy as np
from typing import List, Tuple, Callable, Any, Optional, Dict
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
    statistic: Callable, 
    n_iterations: int = 10000, 
    confidence: float = 0.95,
    stratify_by: Optional[np.ndarray] = None,
    adaptive: bool = False,
    max_iterations: int = 50000,
    target_ci_width: Optional[float] = None,
    stability_threshold: float = 0.01
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Calculate bootstrap confidence interval with optional stratification and adaptive iterations.
    
    Args:
        data: Input data
        statistic: Function to compute statistic
        n_iterations: Initial number of bootstrap iterations
        confidence: Confidence level
        stratify_by: Array of strata labels for stratified sampling (optional)
        adaptive: Whether to adaptively increase iterations until CI stabilizes
        max_iterations: Maximum iterations if adaptive
        target_ci_width: Target CI width for adaptive stopping (optional)
        stability_threshold: Relative change threshold for stability (default: 0.01)
        
    Returns:
        Tuple of (lower_bound, upper_bound, metadata_dict)
    """
    n = len(data)
    if n == 0:
        return 0.0, 0.0, {"error": "Empty data"}
    
    # Initialize metadata
    metadata = {
        "iterations_used": 0,
        "final_ci_width": None,
        "stabilized": False,
        "stratification_used": stratify_by is not None,
        "strata_counts": None,
        "convergence_history": []
    }
    
    # If stratification is requested, validate and prepare strata
    if stratify_by is not None:
        if len(stratify_by) != n:
            raise ValueError("stratify_by must have same length as data")
        
        # Get unique strata and their counts
        strata = np.unique(stratify_by)
        strata_counts = {s: np.sum(stratify_by == s) for s in strata}
        metadata["strata_counts"] = strata_counts
        
        # Check if any stratum has too few samples
        min_stratum_size = min(strata_counts.values())
        if min_stratum_size < 5:
            metadata["warning"] = f"Small stratum size: {min_stratum_size} (minimum recommended: 5)"
    
    # Initialize bootstrap statistics storage
    bootstrap_stats = []
    ci_width_history = []
    
    # Determine initial batch size
    batch_size = min(1000, n_iterations)
    
    for iteration in range(0, max_iterations, batch_size):
        # Calculate remaining iterations in this batch
        current_batch_size = min(batch_size, n_iterations - iteration)
        
        # Generate bootstrap samples
        for _ in range(current_batch_size):
            if stratify_by is not None:
                # Stratified sampling: sample within each stratum
                sample_indices = []
                for s in strata:
                    stratum_indices = np.where(stratify_by == s)[0]
                    # Sample with replacement within stratum
                    sampled = np.random.choice(stratum_indices, size=len(stratum_indices), replace=True)
                    sample_indices.extend(sampled)
                sample = data[sample_indices]
            else:
                # Regular bootstrap: sample with replacement
                sample = np.random.choice(data, size=n, replace=True)
            
            # Calculate statistic
            bootstrap_stats.append(statistic(sample))
        
        metadata["iterations_used"] = len(bootstrap_stats)
        
        # Calculate current CI
        alpha = (1 - confidence) / 2
        current_lower = np.percentile(bootstrap_stats, 100 * alpha)
        current_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha))
        current_ci_width = current_upper - current_lower
        
        # Record convergence
        ci_width_history.append(current_ci_width)
        metadata["convergence_history"].append({
            "iteration": metadata["iterations_used"],
            "ci_width": current_ci_width
        })
        
        # Check stopping conditions if adaptive
        if adaptive and metadata["iterations_used"] >= n_iterations:
            # Check for stability
            if len(ci_width_history) >= 3:
                # Calculate relative change in CI width over last 3 batches
                recent_changes = [
                    abs(ci_width_history[i] - ci_width_history[i-1]) / ci_width_history[i-1]
                    for i in range(len(ci_width_history)-2, len(ci_width_history))
                ]
                avg_change = np.mean(recent_changes)
                
                # Check if we've reached target width or stability
                if (target_ci_width is not None and current_ci_width <= target_ci_width) or \
                   (avg_change < stability_threshold):
                    metadata["stabilized"] = True
                    metadata["final_ci_width"] = current_ci_width
                    return current_lower, current_upper, metadata
            
            # If we haven't stabilized but reached max iterations, stop
            if metadata["iterations_used"] >= max_iterations:
                metadata["final_ci_width"] = current_ci_width
                return current_lower, current_upper, metadata
    
    # If we're not adaptive, just return the result
    metadata["final_ci_width"] = current_ci_width
    return current_lower, current_upper, metadata

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
