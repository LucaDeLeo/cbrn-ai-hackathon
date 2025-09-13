# Appendix: Critical Algorithms Documentation

## Bootstrap Confidence Interval (Pure NumPy)

```python
"""
Bootstrap CI Implementation without SciPy
Teaching implementation with clear mathematical formulation
"""

def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    method: str = 'percentile'
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval using percentile method.
    
    Mathematical Foundation:
    ----------------------
    Given sample X = {x₁, ..., xₙ} and statistic θ = f(X)
    
    1. Generate B bootstrap samples by resampling with replacement:
       X*ᵦ = resample(X) for b = 1...B
    
    2. Calculate statistic for each bootstrap sample:
       θ*ᵦ = f(X*ᵦ)
    
    3. Percentile Method CI:
       CI = [θ*_{α/2}, θ*_{1-α/2}]
       where α = 1 - confidence_level
    
    Parameters:
    ----------
    data : array-like
        Original sample data
    statistic : function
        Function to calculate statistic of interest
    n_bootstrap : int
        Number of bootstrap samples (B)
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    method : str
        'percentile' or 'bca' (bias-corrected accelerated)
    
    Returns:
    -------
    (lower, upper) : tuple
        Lower and upper bounds of confidence interval
    
    Example:
    -------
    >>> data = np.random.normal(100, 15, 100)
    >>> lower, upper = bootstrap_confidence_interval(data, np.mean)
    >>> print(f"95% CI for mean: [{lower:.2f}, {upper:.2f}]")
    """
    
    n = len(data)
    
    # Generate bootstrap distribution
    bootstrap_statistics = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = data[np.random.choice(n, n, replace=True)]
        # Calculate statistic
        bootstrap_statistics[i] = statistic(bootstrap_sample)
    
    if method == 'percentile':
        # Simple percentile method
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_statistics, lower_percentile)
        upper = np.percentile(bootstrap_statistics, upper_percentile)
        
    elif method == 'bca':
        # Optional BCa method not enabled by default (requires either SciPy
        # or a custom normal quantile implementation). If not available,
        # fall back to percentile and log a warning.
        import warnings
        warnings.warn('BCa method not available without optional extras; '
                      'falling back to percentile CI.', RuntimeWarning)
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_statistics, (alpha / 2) * 100)
        upper = np.percentile(bootstrap_statistics, (1 - alpha / 2) * 100)
    
    return lower, upper
```
