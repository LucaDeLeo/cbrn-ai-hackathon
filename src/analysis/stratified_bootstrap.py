# src/analysis/stratified_bootstrap.py
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
import time

from src.analysis.statistical import calculate_bootstrap_ci

@dataclass
class StratifiedBootstrapResult:
    """Result of stratified bootstrap analysis"""
    metric_name: str
    ci_lower: float
    ci_upper: float
    ci_width: float
    iterations_used: int
    stabilized: bool
    strata_counts: Dict[str, int]
    convergence_history: List[Dict[str, Any]]
    warning: Optional[str] = None

class StratifiedBootstrapAnalyzer:
    """Analyzer for stratified bootstrap with CI width targeting"""
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 adaptive: bool = True,
                 max_iterations: int = 50000,
                 target_ci_width: Optional[float] = None,
                 stability_threshold: float = 0.01):
        """
        Initialize the analyzer
        
        Args:
            confidence_level: Confidence level for CIs
            adaptive: Whether to use adaptive iterations
            max_iterations: Maximum iterations
            target_ci_width: Target CI width for stopping
            stability_threshold: Relative change threshold for stability
        """
        self.confidence_level = confidence_level
        self.adaptive = adaptive
        self.max_iterations = max_iterations
        self.target_ci_width = target_ci_width
        self.stability_threshold = stability_threshold
    
    def analyze_metric(self,
                      data: np.ndarray,
                      statistic: callable,
                      metric_name: str,
                      stratify_by: Optional[np.ndarray] = None) -> StratifiedBootstrapResult:
        """
        Analyze a single metric with stratified bootstrap
        
        Args:
            data: Input data
            statistic: Function to compute the statistic
            metric_name: Name of the metric for reporting
            stratify_by: Array of strata labels
            
        Returns:
            StratifiedBootstrapResult
        """
        # Run stratified bootstrap
        ci_lower, ci_upper, metadata = calculate_bootstrap_ci(
            data=data,
            statistic=statistic,
            n_iterations=1000,  # Initial batch size
            confidence=self.confidence_level,
            stratify_by=stratify_by,
            adaptive=self.adaptive,
            max_iterations=self.max_iterations,
            target_ci_width=self.target_ci_width,
            stability_threshold=self.stability_threshold
        )
        
        return StratifiedBootstrapResult(
            metric_name=metric_name,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=metadata["final_ci_width"],
            iterations_used=metadata["iterations_used"],
            stabilized=metadata["stabilized"],
            strata_counts=metadata["strata_counts"],
            convergence_history=metadata["convergence_history"],
            warning=metadata.get("warning")
        )
    
    def analyze_multiple_metrics(self,
                               metrics_data: Dict[str, Tuple[np.ndarray, callable]],
                               stratify_by: Optional[np.ndarray] = None) -> Dict[str, StratifiedBootstrapResult]:
        """
        Analyze multiple metrics with stratified bootstrap
        
        Args:
            metrics_data: Dictionary of {metric_name: (data, statistic_function)}
            stratify_by: Array of strata labels
            
        Returns:
            Dictionary of metric_name -> StratifiedBootstrapResult
        """
        results = {}
        
        for metric_name, (data, statistic) in metrics_data.items():
            results[metric_name] = self.analyze_metric(
                data=data,
                statistic=statistic,
                metric_name=metric_name,
                stratify_by=stratify_by
            )
        
        return results
    
    def to_json(self, results: Dict[str, StratifiedBootstrapResult]) -> str:
        """Convert results to JSON"""
        return json.dumps({k: asdict(v) for k, v in results.items()}, indent=2)
    
    def save_json(self, results: Dict[str, StratifiedBootstrapResult], filepath: str):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json(results))
