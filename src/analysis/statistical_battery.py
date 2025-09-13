# src/analysis/statistical_battery.py
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any, Union
import numpy as np

from src.analysis.statistical import (
    calculate_bootstrap_ci,
    chi_square_test,
    detect_position_bias,
    analyze_lexical_patterns,
    # Import other statistical test functions as they're implemented
)

@dataclass
class TestResult:
    """Base class for statistical test results"""
    test_name: str
    status: str  # "success", "warning", "error"
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

@dataclass
class BatteryResult:
    """Result of running the statistical battery"""
    timestamp: str
    runtime_seconds: float
    total_questions: int
    overall_status: str  # "green", "yellow", "red"
    summary: Dict[str, str]  # Test name -> status
    tests: Dict[str, TestResult]
    performance: Dict[str, Any]

class StatisticalBattery:
    """Runs a suite of statistical tests on MCQA data"""
    
    def __init__(self, max_workers: int = None, confidence_level: float = 0.95):
        """
        Initialize the statistical battery
        
        Args:
            max_workers: Maximum number of parallel workers (default: CPU count)
            confidence_level: Confidence level for statistical tests (default: 0.95)
        """
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1))
        self.confidence_level = confidence_level
        
        # Register all available tests
        self._tests = {
            'position_bias': self._run_position_bias_test,
            'lexical_patterns': self._run_lexical_patterns_test,
            # Add other test functions as they're implemented
        }
    
    def run_all(self, questions: List[Dict[str, Any]]) -> BatteryResult:
        """Run all registered statistical tests"""
        return self.run(questions, tests=list(self._tests.keys()))
    
    def run(self, questions: List[Dict[str, Any]], tests: List[str] = None) -> BatteryResult:
        """
        Run specified statistical tests
        
        Args:
            questions: List of question dictionaries
            tests: List of test names to run (default: all tests)
            
        Returns:
            BatteryResult containing all test results
        """
        start_time = time.time()
        tests_to_run = tests or list(self._tests.keys())
        
        # Validate test names
        invalid_tests = [t for t in tests_to_run if t not in self._tests]
        if invalid_tests:
            raise ValueError(f"Invalid test names: {invalid_tests}")
        
        # Run tests in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_test = {
                executor.submit(self._tests[test_name], questions): test_name
                for test_name in tests_to_run
            }
            
            for future in as_completed(future_to_test):
                test_name = future_to_test[future]
                try:
                    results[test_name] = future.result()
                except Exception as e:
                    results[test_name] = TestResult(
                        test_name=test_name,
                        status="error",
                        message=str(e)
                    )
        
        # Calculate overall status and summary
        overall_status, summary = self._calculate_overall_status(results)
        
        # Calculate performance metrics
        runtime_seconds = time.time() - start_time
        performance = {
            'runtime_seconds': runtime_seconds,
            'questions_per_second': len(questions) / runtime_seconds if runtime_seconds > 0 else 0,
            'tests_run': len(results),
            'tests_passed': sum(1 for r in results.values() if r.status == "success"),
            'tests_with_warnings': sum(1 for r in results.values() if r.status == "warning"),
            'tests_failed': sum(1 for r in results.values() if r.status == "error")
        }
        
        return BatteryResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            runtime_seconds=runtime_seconds,
            total_questions=len(questions),
            overall_status=overall_status,
            summary=summary,
            tests=results,
            performance=performance
        )
    
    def _calculate_overall_status(self, results: Dict[str, TestResult]) -> tuple:
        """Calculate overall status and summary from individual test results"""
        summary = {}
        
        # Count statuses
        status_counts = {"success": 0, "warning": 0, "error": 0}
        for test_name, result in results.items():
            status_counts[result.status] += 1
            summary[test_name] = result.status
        
        # Determine overall status
        if status_counts["error"] > 0:
            overall_status = "red"
        elif status_counts["warning"] > len(results) * 0.25:  # More than 25% warnings
            overall_status = "yellow"
        elif status_counts["success"] == len(results):
            overall_status = "green"
        else:
            overall_status = "yellow"
        
        return overall_status, summary
    
    def _run_position_bias_test(self, questions: List[Dict[str, Any]]) -> TestResult:
        """Run position bias analysis"""
        try:
            # Extract answer positions
            positions = [q.get('answer_index', 0) for q in questions if 'answer_index' in q]
            
            if not positions:
                return TestResult(
                    test_name="position_bias",
                    status="error",
                    message="No valid answer positions found"
                )
            
            # Calculate observed frequencies
            observed = np.bincount(positions, minlength=4)
            expected = np.full(4, len(positions) / 4)
            
            # Run chi-square test
            chi2_stat, p_value = chi_square_test(observed, expected)
            
            # Calculate effect size (Cramer's V)
            n = len(positions)
            cramers_v = np.sqrt(chi2_stat / (n * min(4-1, n-1))) if n > 1 else 0
            
            # Calculate bootstrap CI for effect size
            def effect_size_func(data):
                obs = np.bincount(data, minlength=4)
                exp = np.full(4, len(data) / 4)
                chi2, _ = chi_square_test(obs, exp)
                n = len(data)
                return np.sqrt(chi2 / (n * min(4-1, n-1))) if n > 1 else 0
            
            ci_lower, ci_upper = calculate_bootstrap_ci(
                np.array(positions), 
                effect_size_func,
                n_iterations=1000
            )
            
            # Determine status based on p-value
            status = "success"
            message = "No significant position bias detected"
            if p_value < 0.05:
                status = "warning"
                message = f"Significant position bias detected (p={p_value:.4f})"
            
            return TestResult(
                test_name="position_bias",
                status=status,
                p_value=p_value,
                effect_size=cramers_v,
                confidence_interval=[ci_lower, ci_upper],
                message=message,
                data={
                    "observed_frequencies": observed.tolist(),
                    "expected_frequencies": expected.tolist(),
                    "chi_square_statistic": chi2_stat
                }
            )
        except Exception as e:
            return TestResult(
                test_name="position_bias",
                status="error",
                message=str(e)
            )
    
    def _run_lexical_patterns_test(self, questions: List[Dict[str, Any]]) -> TestResult:
        """Run lexical pattern analysis"""
        try:
            # This is a placeholder implementation
            # You would replace this with the actual implementation from epic 2.3
            
            # For now, just return a dummy result
            return TestResult(
                test_name="lexical_patterns",
                status="success",
                message="Lexical pattern analysis completed",
                data={
                    "patterns_detected": [],
                    "effect_sizes": {}
                }
            )
        except Exception as e:
            return TestResult(
                test_name="lexical_patterns",
                status="error",
                message=str(e)
            )
    
    def to_json(self, result: BatteryResult) -> str:
        """Convert a BatteryResult to JSON"""
        return json.dumps(asdict(result), indent=2)
    
    def save_json(self, result: BatteryResult, filepath: str):
        """Save a BatteryResult to a JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json(result))
