"""Tests for confidence scoring and thresholded abstention."""

import numpy as np
import pandas as pd
import pytest

from robustcbrn.analysis.confidence_scoring import (
    apply_confidence_threshold,
    compute_confidence_aware_metrics,
    compute_proportional_penalties,
    evaluate_all_thresholds,
)


class TestApplyConfidenceThreshold:
    def test_basic_threshold(self):
        """Test basic confidence thresholding."""
        df = pd.DataFrame({
            'confidence': [0.9, 0.7, 0.4, 0.2],
            'response': ['Yes', 'No', 'Maybe', 'Yes']
        })

        result = apply_confidence_threshold(df, threshold=0.5)

        assert 'is_abstention' in result.columns
        # Confidence 0.4 and 0.2 should be abstentions
        assert result['is_abstention'].tolist() == [False, False, True, True]

    def test_explicit_abstention_phrases(self):
        """Test detection of explicit abstention phrases."""
        df = pd.DataFrame({
            'confidence': [0.9, 0.8, 0.7, 0.6],
            'response': [
                'The answer is 42',
                "I don't know",
                'Unable to answer this',
                'Not sure about this'
            ]
        })

        result = apply_confidence_threshold(df, threshold=0.5)

        # Last 3 should be abstentions due to phrases
        assert result['is_abstention'].tolist() == [False, True, True, True]

    def test_combined_abstention_criteria(self):
        """Test combination of low confidence and explicit phrases."""
        df = pd.DataFrame({
            'confidence': [0.3, 0.3, 0.9],
            'response': ['Yes', "I don't know", "I don't know"]
        })

        result = apply_confidence_threshold(df, threshold=0.5)

        # All should be abstentions for different reasons
        assert all(result['is_abstention'])

    def test_case_insensitive_matching(self):
        """Test case-insensitive phrase matching."""
        df = pd.DataFrame({
            'confidence': [0.9, 0.9, 0.9],
            'response': ["I DON'T KNOW", "I Do Not Know", "i DONT know"]
        })

        result = apply_confidence_threshold(df, threshold=0.5)

        # All should be abstentions
        assert all(result['is_abstention'])

    def test_missing_columns(self):
        """Test handling of missing columns."""
        # DataFrame without confidence column
        df1 = pd.DataFrame({'response': ['Yes', 'No']})
        result1 = apply_confidence_threshold(df1, threshold=0.5)
        assert 'is_abstention' in result1.columns

        # DataFrame without response column
        df2 = pd.DataFrame({'confidence': [0.9, 0.3]})
        result2 = apply_confidence_threshold(df2, threshold=0.5)
        assert result2['is_abstention'].tolist() == [False, True]


class TestComputeProportionalPenalties:
    def test_correct_answers(self):
        """Test that correct answers get score of 1."""
        df = pd.DataFrame({
            'confidence': [0.9, 0.8, 0.7],
            'is_correct': [True, True, True],
            'is_abstention': [False, False, False]
        })

        result = compute_proportional_penalties(df, threshold=0.5)

        assert all(result['penalty_score'] == 1.0)

    def test_incorrect_answers_penalty(self):
        """Test proportional penalty for incorrect answers."""
        df = pd.DataFrame({
            'confidence': [0.9, 0.8, 0.7],
            'is_correct': [False, False, False],
            'is_abstention': [False, False, False]
        })

        # For threshold=0.5, penalty = -0.5/(1-0.5) = -1.0
        result = compute_proportional_penalties(df, threshold=0.5)
        assert all(result['penalty_score'] == -1.0)

        # For threshold=0.75, penalty = -0.75/(1-0.75) = -3.0
        result = compute_proportional_penalties(df, threshold=0.75)
        assert all(result['penalty_score'] == -3.0)

    def test_abstention_zero_penalty(self):
        """Test that abstentions always get zero penalty."""
        df = pd.DataFrame({
            'confidence': [0.9, 0.3, 0.2],
            'is_correct': [False, False, True],
            'is_abstention': [False, True, True]
        })

        result = compute_proportional_penalties(df, threshold=0.5)

        # First gets penalty, others get 0
        assert result['penalty_score'].tolist()[0] == -1.0
        assert result['penalty_score'].tolist()[1] == 0.0
        assert result['penalty_score'].tolist()[2] == 0.0

    def test_threshold_edge_cases(self):
        """Test edge cases for threshold values."""
        df = pd.DataFrame({
            'confidence': [0.9],
            'is_correct': [False],
            'is_abstention': [False]
        })

        # Threshold = 0, penalty = 0
        result = compute_proportional_penalties(df, threshold=0.0)
        assert result['penalty_score'].iloc[0] == 0.0

        # Threshold = 0.9, penalty = -0.9/0.1 = -9.0
        result = compute_proportional_penalties(df, threshold=0.9)
        assert result['penalty_score'].iloc[0] == pytest.approx(-9.0)

    def test_auto_apply_threshold(self):
        """Test automatic application of threshold if not already done."""
        df = pd.DataFrame({
            'confidence': [0.9, 0.3],
            'is_correct': [True, False],
            'response': ['Yes', 'No']
        })

        # Without is_abstention column, should auto-apply threshold
        result = compute_proportional_penalties(df, threshold=0.5)

        assert 'is_abstention' in result.columns
        assert 'penalty_score' in result.columns
        # First is correct (score=1), second is abstention (score=0)
        assert result['penalty_score'].tolist() == [1.0, 0.0]


class TestComputeConfidenceAwareMetrics:
    def test_basic_metrics(self):
        """Test computation of basic confidence-aware metrics."""
        df = pd.DataFrame({
            'confidence': [0.9, 0.8, 0.4, 0.3],
            'is_correct': [True, False, True, False],
            'response': ['A', 'B', 'C', 'D']
        })

        metrics = compute_confidence_aware_metrics(df, threshold=0.5)

        assert metrics['threshold'] == 0.5
        assert metrics['n_total'] == 4
        assert metrics['n_abstentions'] == 2  # confidence 0.4 and 0.3
        assert metrics['n_answered'] == 2
        assert metrics['abstention_rate'] == 0.5
        assert metrics['accuracy_on_answered'] == 0.5  # 1 correct out of 2

    def test_high_threshold_abstention_rate(self):
        """Test that high thresholds produce high abstention rates."""
        np.random.seed(42)
        df = pd.DataFrame({
            'confidence': np.random.uniform(0.3, 1.0, 100),
            'is_correct': np.random.choice([True, False], 100),
            'response': ['Answer'] * 100
        })

        # Test with high threshold
        metrics = compute_confidence_aware_metrics(df, threshold=0.9)

        # Should have substantial abstention rate
        assert 0.1 <= metrics['abstention_rate'] <= 0.9
        assert metrics['n_abstentions'] > 0

    def test_perfect_confidence(self):
        """Test metrics with perfect confidence and accuracy."""
        df = pd.DataFrame({
            'confidence': [1.0] * 10,
            'is_correct': [True] * 10,
            'response': ['Correct'] * 10
        })

        metrics = compute_confidence_aware_metrics(df, threshold=0.5)

        assert metrics['abstention_rate'] == 0.0
        assert metrics['accuracy_on_answered'] == 1.0
        assert metrics['average_penalty_score'] == 1.0

    def test_all_abstentions(self):
        """Test when all samples are abstentions."""
        df = pd.DataFrame({
            'confidence': [0.2, 0.1, 0.3],
            'is_correct': [True, False, True],
            'response': ['A', 'B', 'C']
        })

        metrics = compute_confidence_aware_metrics(df, threshold=0.5)

        assert metrics['abstention_rate'] == 1.0
        assert metrics['n_answered'] == 0
        assert metrics['accuracy_on_answered'] == 0.0
        assert metrics['average_penalty_score'] == 0.0


class TestEvaluateAllThresholds:
    def test_multiple_thresholds(self):
        """Test evaluation across multiple thresholds."""
        df = pd.DataFrame({
            'confidence': np.linspace(0.1, 1.0, 20),
            'is_correct': [True] * 10 + [False] * 10,
            'response': ['Answer'] * 20
        })

        thresholds = [0.0, 0.5, 0.75, 0.9]
        results = evaluate_all_thresholds(df, thresholds)

        # Check structure
        assert len(results) == 4
        assert 't_0.0' in results
        assert 't_0.5' in results
        assert 't_0.75' in results
        assert 't_0.9' in results

        # Check monotonicity of abstention rates
        abstention_rates = [results[f't_{t}']['abstention_rate'] for t in thresholds]
        # Higher thresholds should generally lead to more abstentions
        assert abstention_rates[0] <= abstention_rates[-1]

    def test_default_thresholds(self):
        """Test with default threshold values."""
        df = pd.DataFrame({
            'confidence': [0.95, 0.6, 0.4, 0.1],
            'is_correct': [True, True, False, False],
            'response': ['A', 'B', 'C', 'D']
        })

        results = evaluate_all_thresholds(df)

        # Should use default thresholds
        assert 't_0.0' in results
        assert 't_0.5' in results
        assert 't_0.75' in results
        assert 't_0.9' in results

    def test_custom_thresholds(self):
        """Test with custom threshold values."""
        df = pd.DataFrame({
            'confidence': [0.95, 0.6, 0.4],
            'is_correct': [True, True, False],
            'response': ['A', 'B', 'C']
        })

        thresholds = [0.3, 0.6, 0.8]
        results = evaluate_all_thresholds(df, thresholds)

        assert 't_0.3' in results
        assert 't_0.6' in results
        assert 't_0.8' in results

        # Verify each has complete metrics
        for key in results:
            assert 'threshold' in results[key]
            assert 'abstention_rate' in results[key]
            assert 'accuracy_on_answered' in results[key]
            assert 'average_penalty_score' in results[key]

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({
            'confidence': [],
            'is_correct': [],
            'response': []
        })

        results = evaluate_all_thresholds(df)

        # Should handle gracefully
        assert len(results) == 4
        for key in results:
            assert results[key]['n_total'] == 0
