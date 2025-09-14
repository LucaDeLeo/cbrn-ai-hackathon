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
from src.analysis.heuristic_degradation import HeuristicDegradationAnalyzer, HeuristicDegradationResult
from src.analysis.stratified_bootstrap import StratifiedBootstrapAnalyzer, StratifiedBootstrapResult

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
    
    def __init__(self, 
                 max_workers: int = None, 
                 confidence_level: float = 0.95,
                 robust_questions: List[Dict[str, Any]] = None,
                 stratify_by: Optional[np.ndarray] = None,
                 adaptive_bootstrap: bool = True):
        """
        Initialize the statistical battery
        
        Args:
            max_workers: Maximum number of parallel workers (default: CPU count)
            confidence_level: Confidence level for statistical tests (default: 0.95)
            robust_questions: Optional list of robust question dictionaries for degradation analysis
            stratify_by: Optional array of strata labels for stratified bootstrap analysis
            adaptive_bootstrap: Whether to use adaptive bootstrap iterations
        """
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1))
        self.confidence_level = confidence_level
        self.robust_questions = robust_questions
        self.stratify_by = stratify_by
        self.adaptive_bootstrap = adaptive_bootstrap
        
        # Initialize stratified bootstrap analyzer
        self.stratified_analyzer = StratifiedBootstrapAnalyzer(
            confidence_level=confidence_level,
            adaptive=adaptive_bootstrap
        )
        
        # Register all available tests
        self._tests = {
            'position_bias': self._run_position_bias_test,
            'lexical_patterns': self._run_lexical_patterns_test,
            'heuristic_degradation': self._run_heuristic_degradation_test,
            'stratified_metrics': self._run_stratified_metrics_test,  # Add this
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
            
            ci_lower, ci_upper, metadata = calculate_bootstrap_ci(
                np.array(positions), 
                effect_size_func,
                n_iterations=1000,
                confidence=self.confidence_level
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
            # Import here to avoid circular imports
            from src.analysis.patterns import detect_lexical_patterns
            from src.data.schemas import Question
            
            # Convert dictionary format to Question objects
            question_objects = []
            for q_dict in questions:
                try:
                    question_obj = Question(
                        id=q_dict.get('id', 'unknown'),
                        question=q_dict.get('question', ''),
                        choices=q_dict.get('choices', []),
                        answer=q_dict.get('answer_index', q_dict.get('answer', 0)),
                        topic=q_dict.get('topic', 'unknown')
                    )
                    question_objects.append(question_obj)
                except Exception as e:
                    # Skip invalid questions
                    continue
            
            if not question_objects:
                return TestResult(
                    test_name="lexical_patterns",
                    status="error",
                    message="No valid questions found for lexical pattern analysis"
                )
            
            # Run lexical pattern detection
            pattern_report = detect_lexical_patterns(
                question_objects,
                show_progress=False,
                debug=False
            )
            
            # Extract key metrics from the pattern report
            results = pattern_report.results
            summary = results.get('summary', {})
            
            # Determine status based on detected patterns
            status = "success"
            message = "No significant lexical patterns detected"
            
            # Check for significant patterns
            significant_patterns = []
            if summary.get('significant_length_difference', False):
                significant_patterns.append("length bias")
            if summary.get('significant_technical_density_difference', False):
                significant_patterns.append("technical density bias")
            
            # Check meta-patterns
            meta_patterns = results.get('meta_patterns', {})
            for pattern_name, pattern_data in meta_patterns.items():
                if pattern_data.get('significant', False):
                    significant_patterns.append(f"{pattern_name} bias")
            
            if significant_patterns:
                status = "warning"
                message = f"Significant lexical patterns detected: {', '.join(significant_patterns)}"
            
            # Extract effect sizes and p-values
            length_analysis = results.get('length_analysis', {})
            technical_analysis = results.get('technical_density_analysis', {})
            
            effect_sizes = {}
            p_values = {}
            
            if 'cohens_d' in length_analysis:
                effect_sizes['length_bias'] = length_analysis['cohens_d']
                p_values['length_bias'] = length_analysis.get('t_pvalue')
            
            if 'cohens_d' in technical_analysis:
                effect_sizes['technical_density_bias'] = technical_analysis['cohens_d']
                p_values['technical_density_bias'] = technical_analysis.get('t_pvalue')
            
            return TestResult(
                test_name="lexical_patterns",
                status=status,
                p_value=min(p_values.values()) if p_values else None,
                effect_size=max(effect_sizes.values()) if effect_sizes else None,
                message=message,
                data={
                    "patterns_detected": significant_patterns,
                    "effect_sizes": effect_sizes,
                    "p_values": p_values,
                    "total_questions_analyzed": len(question_objects),
                    "discriminative_phrases_found": summary.get('discriminative_phrases_found', 0),
                    "meta_patterns": meta_patterns
                }
            )
        except Exception as e:
            return TestResult(
                test_name="lexical_patterns",
                status="error",
                message=str(e)
            )
    
    def _run_heuristic_degradation_test(self, questions: List[Dict[str, Any]]) -> TestResult:
        """Run heuristic degradation analysis"""
        try:
            if not self.robust_questions:
                return TestResult(
                    test_name="heuristic_degradation",
                    status="error",
                    message="No robust questions provided for degradation analysis"
                )
            
            # Run the analysis
            analyzer = HeuristicDegradationAnalyzer(
                confidence_level=self.confidence_level,
                stratify_by=self.stratify_by,
                adaptive_bootstrap=self.adaptive_bootstrap
            )
            degradation_result = analyzer.analyze(questions, self.robust_questions)
            
            # Convert to TestResult format
            # We'll store the full degradation result in the data field
            return TestResult(
                test_name="heuristic_degradation",
                status="success",
                message="Heuristic degradation analysis completed",
                data=asdict(degradation_result)
            )
        except Exception as e:
            return TestResult(
                test_name="heuristic_degradation",
                status="error",
                message=str(e)
            )
    
    def _run_stratified_metrics_test(self, questions: List[Dict[str, Any]]) -> TestResult:
        """Run stratified bootstrap analysis on key metrics"""
        try:
            # Extract data for key metrics
            positions = np.array([q.get('answer_index', 0) for q in questions if 'answer_index' in q])
            
            if len(positions) == 0:
                return TestResult(
                    test_name="stratified_metrics",
                    status="error",
                    message="No valid answer positions found"
                )
            
            # Define metrics to analyze
            metrics_data = {
                "mean_position": (positions, np.mean),
                "position_variance": (positions, np.var),
                "flagged_rate": (np.array([1 if p in [0, 3] else 0 for p in positions]), np.mean),
            }
            
            # Run stratified bootstrap analysis
            results = self.stratified_analyzer.analyze_multiple_metrics(
                metrics_data=metrics_data,
                stratify_by=self.stratify_by
            )
            
            # Calculate summary statistics
            total_iterations = sum(r.iterations_used for r in results.values())
            stabilized_count = sum(1 for r in results.values() if r.stabilized)
            
            return TestResult(
                test_name="stratified_metrics",
                status="success",
                message=f"Stratified bootstrap completed (total iterations: {total_iterations})",
                data={
                    "metrics": {k: asdict(v) for k, v in results.items()},
                    "summary": {
                        "total_iterations": total_iterations,
                        "stabilized_metrics": stabilized_count,
                        "total_metrics": len(results),
                        "stratification_used": self.stratify_by is not None
                    }
                }
            )
        except Exception as e:
            return TestResult(
                test_name="stratified_metrics",
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
