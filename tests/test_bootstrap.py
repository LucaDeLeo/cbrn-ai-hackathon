# tests/test_bootstrap.py
"""Unit tests for bootstrap confidence intervals implementation.

Tests against known statistical examples and verifies performance requirements.
"""

import unittest
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.statistical.bootstrap import (
    bootstrap_ci,
    bootstrap_mean_ci,
    bootstrap_median_ci,
    bootstrap_proportion_ci,
    BootstrapResult,
    _check_convergence,
    _norm_ppf,
    _norm_cdf
)


class TestBootstrapCI(unittest.TestCase):
    """Test bootstrap confidence interval implementation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests

        # Known statistical examples
        self.normal_data = np.random.normal(100, 15, 1000)  # mean=100, std=15
        self.uniform_data = np.random.uniform(0, 10, 1000)  # mean=5
        self.binary_data = np.random.binomial(1, 0.3, 1000)  # proportion=0.3

        # Large dataset for performance testing
        self.large_data = np.random.normal(50, 10, 3000)

    def test_bootstrap_mean_normal_distribution(self):
        """Test bootstrap CI for mean of normal distribution."""
        result = bootstrap_mean_ci(
            self.normal_data,
            n_bootstrap=5000,
            confidence_level=0.95,
            random_seed=42
        )

        # Check result structure
        self.assertIsInstance(result, BootstrapResult)
        self.assertEqual(result.confidence_level, 0.95)
        self.assertEqual(result.method, "percentile")
        self.assertTrue(result.n_iterations <= 5000)

        # Check that CI contains true mean (100) with high probability
        ci_lower, ci_upper = result.confidence_interval
        self.assertLess(ci_lower, result.statistic)
        self.assertGreater(ci_upper, result.statistic)
        self.assertLess(ci_upper - ci_lower, 5.0)  # Reasonable width

        # Theoretical 95% CI for normal mean should be approximately:
        # mean ± 1.96 * (std / sqrt(n))
        theoretical_margin = 1.96 * 15 / np.sqrt(1000)
        ci_width = ci_upper - ci_lower
        theoretical_width = 2 * theoretical_margin

        # Bootstrap CI width should be close to theoretical
        self.assertLess(abs(ci_width - theoretical_width), 1.0)

    def test_bootstrap_median_uniform_distribution(self):
        """Test bootstrap CI for median of uniform distribution."""
        result = bootstrap_median_ci(
            self.uniform_data,
            n_bootstrap=3000,
            confidence_level=0.90,
            random_seed=42
        )

        # For uniform(0, 10), median should be around 5
        self.assertAlmostEqual(result.statistic, 5.0, delta=0.5)

        ci_lower, ci_upper = result.confidence_interval
        self.assertTrue(4.0 < ci_lower < ci_upper < 6.0)
        self.assertEqual(result.confidence_level, 0.90)

    def test_bootstrap_proportion_binary_data(self):
        """Test bootstrap CI for proportion with binary data."""
        result = bootstrap_proportion_ci(
            self.binary_data,
            n_bootstrap=4000,
            confidence_level=0.95,
            random_seed=42
        )

        # Should be close to true proportion (0.3)
        self.assertAlmostEqual(result.statistic, 0.3, delta=0.05)

        ci_lower, ci_upper = result.confidence_interval

        # Theoretical 95% CI for proportion: p ± 1.96 * sqrt(p(1-p)/n)
        p = 0.3
        n = 1000
        theoretical_margin = 1.96 * np.sqrt(p * (1 - p) / n)

        # Check that bootstrap CI is reasonable
        self.assertTrue(0.2 < ci_lower < 0.35)
        self.assertTrue(0.25 < ci_upper < 0.4)
        self.assertLess(ci_upper - ci_lower, 2 * theoretical_margin * 1.2)  # Allow 20% tolerance

    def test_custom_statistic_function(self):
        """Test bootstrap with custom statistic function."""

        def coefficient_of_variation(data):
            """CV = std / mean"""
            return np.std(data) / np.mean(data) if np.mean(data) != 0 else 0

        result = bootstrap_ci(
            self.normal_data,
            coefficient_of_variation,
            n_bootstrap=2000,
            random_seed=42
        )

        # For normal(100, 15), CV should be around 15/100 = 0.15
        self.assertAlmostEqual(result.statistic, 0.15, delta=0.02)

        ci_lower, ci_upper = result.confidence_interval
        self.assertTrue(0.1 < ci_lower < ci_upper < 0.2)

    def test_bca_method(self):
        """Test bias-corrected accelerated (BCa) method."""
        result = bootstrap_ci(
            self.normal_data,
            np.mean,
            n_bootstrap=3000,
            method="bca",
            random_seed=42
        )

        self.assertEqual(result.method, "bca")

        ci_lower, ci_upper = result.confidence_interval
        self.assertLess(ci_lower, result.statistic)
        self.assertGreater(ci_upper, result.statistic)

        # BCa should generally give slightly different results than percentile
        percentile_result = bootstrap_ci(
            self.normal_data,
            np.mean,
            n_bootstrap=3000,
            method="percentile",
            random_seed=42
        )

        # Results should be similar but not identical
        bca_width = ci_upper - ci_lower
        perc_width = percentile_result.confidence_interval[1] - \
                     percentile_result.confidence_interval[0]
        self.assertLess(abs(bca_width - perc_width), 0.5)

    def test_adaptive_convergence(self):
        """Test adaptive early stopping when CI stabilizes."""
        result = bootstrap_ci(
            self.normal_data,
            np.mean,
            n_bootstrap=10000,
            adaptive=True,
            convergence_threshold=0.01,
            min_iterations=1000,
            random_seed=42
        )

        # Should converge before max iterations for well-behaved data
        self.assertLess(result.n_iterations, 10000)
        self.assertTrue(result.converged)
        self.assertIsNotNone(result.convergence_iteration)
        self.assertGreaterEqual(result.convergence_iteration, 1000)

    def test_performance_requirement(self):
        """Test performance: 10k iterations on 3k items in seconds."""
        start_time = time.time()

        result = bootstrap_ci(
            self.large_data,
            np.mean,
            n_bootstrap=10000,
            adaptive=False,  # Disable for exact iteration count
            random_seed=42
        )

        runtime = time.time() - start_time

        # Should complete in reasonable time (allowing some tolerance)
        self.assertLess(runtime, 10.0, f"Runtime {runtime:.2f}s exceeds 10s limit")
        self.assertEqual(result.n_iterations, 10000)
        self.assertLess(result.runtime_seconds, 10.0)

        print(
            f"Performance test: {result.n_iterations} iterations on {len(self.large_data)} items in {runtime:.3f}s")

    def test_confidence_levels(self):
        """Test different confidence levels."""
        confidence_levels = [0.80, 0.90, 0.95, 0.99]

        results = {}
        for level in confidence_levels:
            result = bootstrap_mean_ci(
                self.normal_data,
                confidence_level=level,
                n_bootstrap=2000,
                random_seed=42
            )
            results[level] = result.confidence_interval[1] - result.confidence_interval[0]

        # Higher confidence should give wider intervals
        self.assertLess(results[0.80], results[0.90])
        self.assertLess(results[0.90], results[0.95])
        self.assertLess(results[0.95], results[0.99])

    def test_convergence_detection(self):
        """Test convergence detection function."""
        # Stable sequence should converge
        stable_stats = [1.0] * 100 + [1.01] * 100 + [0.99] * 50
        self.assertTrue(_check_convergence(stable_stats, 0.05))

        # Unstable sequence should not converge
        unstable_stats = list(range(250))  # Always increasing
        self.assertFalse(_check_convergence(unstable_stats, 0.05))

        # Short sequence should not converge
        short_stats = [1.0] * 50
        self.assertFalse(_check_convergence(short_stats, 0.05))

    def test_normal_distribution_functions(self):
        """Test normal distribution helper functions."""
        # Test standard normal values
        self.assertAlmostEqual(_norm_ppf(0.5), 0.0, places=3)
        self.assertAlmostEqual(_norm_cdf(0.0), 0.5, places=3)

        # Test approximate 95% values
        self.assertAlmostEqual(abs(_norm_ppf(0.025)), 1.96, delta=0.1)
        self.assertAlmostEqual(_norm_cdf(1.96), 0.975, delta=0.01)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty data should raise error (fix this test)
        with self.assertRaises((ValueError, IndexError)):
            bootstrap_ci(np.array([]), np.mean)

        # Single value data
        single_value = np.array([5.0])
        result = bootstrap_ci(single_value, np.mean, n_bootstrap=100)
        self.assertEqual(result.statistic, 5.0)

        # All identical values
        identical = np.array([2.5] * 100)
        result = bootstrap_ci(identical, np.mean, n_bootstrap=1000)
        self.assertEqual(result.statistic, 2.5)
        # CI should be very narrow
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        self.assertLess(ci_width, 0.01)

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        result1 = bootstrap_mean_ci(self.normal_data, random_seed=123, n_bootstrap=1000)
        result2 = bootstrap_mean_ci(self.normal_data, random_seed=123, n_bootstrap=1000)

        self.assertEqual(result1.statistic, result2.statistic)
        self.assertEqual(result1.confidence_interval, result2.confidence_interval)
        self.assertEqual(result1.n_iterations, result2.n_iterations)

    def test_implementation_size(self):
        """Verify implementation is concise (<50 lines for main function)."""
        import inspect

        # Get source code for main bootstrap_ci function
        source = inspect.getsource(bootstrap_ci)
        lines = [line for line in source.split('\n') if
                 line.strip() and not line.strip().startswith('#')]

        # Should be under 50 lines (excluding comments and docstring)
        function_body_lines = []
        in_docstring = False
        for line in lines:
            if '"""' in line:
                in_docstring = not in_docstring
                continue
            if not in_docstring and not line.strip().startswith('def '):
                function_body_lines.append(line)

        print(f"Bootstrap CI implementation: {len(function_body_lines)} lines")
        self.assertLessEqual(len(function_body_lines), 50,
                             f"Implementation has {len(function_body_lines)} lines, should be ≤50")


class TestBootstrapIntegration(unittest.TestCase):
    """Test integration with position bias analysis."""

    def setUp(self):
        """Set up position bias test data."""
        np.random.seed(42)
        # Create biased dataset (60% A, 20% B, 10% C, 10% D)
        self.biased_answers = np.random.choice([0, 1, 2, 3], size=400, p=[0.6, 0.2, 0.1, 0.1])
        self.observed_counts = np.bincount(self.biased_answers, minlength=4)
        self.expected_counts = np.array([100, 100, 100, 100])  # Uniform expected

    def test_chi_square_bootstrap_integration(self):
        """Test bootstrap CI for chi-square test results."""
        # This would integrate with the position bias analysis
        from src.statistical.position_bias import chi_square_test_from_scratch

        # Original chi-square test
        chi2_stat, p_value = chi_square_test_from_scratch(self.observed_counts,
                                                          self.expected_counts)

        # Bootstrap CI for the chi-square statistic itself
        def chi_sq_statistic(counts):
            return np.sum((counts - self.expected_counts) ** 2 / self.expected_counts)

        result = bootstrap_ci(
            self.observed_counts.astype(float),
            chi_sq_statistic,
            n_bootstrap=1000,
            random_seed=42
        )

        # Original chi-square should be within bootstrap CI
        ci_lower, ci_upper = result.confidence_interval
        self.assertLessEqual(ci_lower, chi2_stat)
        self.assertGreaterEqual(ci_upper, chi2_stat)

        print(
            f"Chi-square bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}], original: {chi2_stat:.3f}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
