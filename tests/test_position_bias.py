"""Unit tests for position bias analysis implementation.

Tests the integrated position bias functionality in the statistical analysis module.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from src.data.schemas import Question
from src.analysis.statistical import (
    detect_position_bias,
    chi_square_test,
    calculate_bootstrap_ci,
)


class TestPositionBiasDetection(unittest.TestCase):
    """Test position bias detection functionality."""

    def test_detect_position_bias_uniform(self):
        """Test position bias detection with uniform distribution."""
        questions = [
            {'id': 'q1', 'answer_index': 0},
            {'id': 'q2', 'answer_index': 1},
            {'id': 'q3', 'answer_index': 2},
            {'id': 'q4', 'answer_index': 3},
        ]
        
        result = detect_position_bias(questions)
        
        self.assertIn('chi_square_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('effect_size', result)
        self.assertIn('significant', result)
        self.assertIn('observed_frequencies', result)
        self.assertIn('expected_frequencies', result)
        
        # With uniform distribution, should not be significant
        self.assertFalse(result['significant'])

    def test_detect_position_bias_biased(self):
        """Test position bias detection with biased distribution."""
        questions = [
            {'id': f'q{i}', 'answer_index': 0} for i in range(9)  # 9 in position 0
        ] + [
            {'id': f'q{i}', 'answer_index': 1} for i in range(9, 10)  # 1 in position 1
        ]
        
        result = detect_position_bias(questions)
        
        # With 9 vs 1 distribution, should be significant
        self.assertTrue(result['significant'])
        self.assertEqual(result['observed_frequencies'][0], 9)
        self.assertEqual(result['observed_frequencies'][1], 1)

    def test_detect_position_bias_empty(self):
        """Test position bias detection with empty input."""
        result = detect_position_bias([])
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No questions provided')

    def test_detect_position_bias_no_answer_index(self):
        """Test position bias detection with missing answer_index."""
        questions = [
            {'id': 'q1', 'question': 'Test?'},
            {'id': 'q2', 'question': 'Test?'},
        ]
        
        result = detect_position_bias(questions)
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No valid answer positions found')


class TestChiSquareTest(unittest.TestCase):
    """Test chi-square test functionality."""

    def test_chi_square_uniform_distribution(self):
        """Test chi-square test with uniform distribution."""
        observed = np.array([25, 25, 25, 25])
        expected = np.array([25, 25, 25, 25])
        
        chi2_stat, p_value = chi_square_test(observed, expected)

        self.assertAlmostEqual(chi2_stat, 0.0, places=6)
        self.assertAlmostEqual(p_value, 1.0, places=6)

    def test_chi_square_biased_distribution(self):
        """Test chi-square test with biased distribution."""
        observed = np.array([40, 20, 20, 20])
        expected = np.array([25, 25, 25, 25])
        
        chi2_stat, p_value = chi_square_test(observed, expected)

        self.assertGreater(chi2_stat, 0)
        self.assertLess(p_value, 0.05)  # Should be significant

    def test_chi_square_different_lengths(self):
        """Test chi-square test with mismatched array lengths."""
        observed = np.array([10, 20])
        expected = np.array([15, 15, 15])

        with self.assertRaises(ValueError):
            chi_square_test(observed, expected)

    def test_chi_square_zero_expected(self):
        """Test chi-square test with zero expected values."""
        observed = np.array([10, 20])
        expected = np.array([0, 30])

        with self.assertRaises(ValueError):
            chi_square_test(observed, expected)


class TestBootstrapCI(unittest.TestCase):
    """Test bootstrap confidence interval functionality."""

    def test_bootstrap_ci_mean(self):
        """Test bootstrap CI for mean calculation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        def mean_func(x):
            return np.mean(x)
        
        lower, upper = calculate_bootstrap_ci(data, mean_func, n_iterations=100)
        
        self.assertLess(lower, upper)
        self.assertGreater(lower, 0)
        self.assertLess(upper, 15)

    def test_bootstrap_ci_small_sample(self):
        """Test bootstrap CI with small sample."""
        data = np.array([1, 2, 3])
        
        def mean_func(x):
            return np.mean(x)
        
        lower, upper = calculate_bootstrap_ci(data, mean_func, n_iterations=10)
        
        self.assertLess(lower, upper)


if __name__ == '__main__':
    unittest.main()