"""Tests for calibration metrics module."""

import numpy as np
import pandas as pd
import pytest

from robustcbrn.analysis.calibration import (
    bin_predictions_by_confidence,
    compute_brier_score,
    compute_calibration_metrics,
    compute_ece,
)


class TestBrierScore:
    def test_perfect_calibration(self):
        """Test Brier score for perfectly calibrated predictions."""
        # All predictions correct with confidence 1.0
        predictions = np.array([1, 1, 1, 0, 0])
        targets = np.array([1, 1, 1, 0, 0])
        confidences = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        brier = compute_brier_score(predictions, targets, confidences)
        assert brier == pytest.approx(0.0)

    def test_worst_calibration(self):
        """Test Brier score for worst calibration."""
        # All predictions wrong with confidence 1.0
        predictions = np.array([0, 0, 0, 1, 1])
        targets = np.array([1, 1, 1, 0, 0])
        confidences = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        brier = compute_brier_score(predictions, targets, confidences)
        assert brier == pytest.approx(1.0)

    def test_mixed_calibration(self):
        """Test Brier score for mixed calibration."""
        predictions = np.array([1, 1, 0, 0])
        targets = np.array([1, 0, 0, 1])
        confidences = np.array([0.8, 0.6, 0.7, 0.3])

        # Expected: (0.8-1)^2 + (0.6-0)^2 + (0.7-1)^2 + (0.3-0)^2 / 4
        # = 0.04 + 0.36 + 0.09 + 0.09 / 4 = 0.58 / 4 = 0.145
        brier = compute_brier_score(predictions, targets, confidences)
        assert brier == pytest.approx(0.145)

    def test_list_inputs(self):
        """Test that list inputs work correctly."""
        predictions = [1, 0]
        targets = [1, 0]
        confidences = [0.9, 0.8]

        brier = compute_brier_score(predictions, targets, confidences)
        assert isinstance(brier, float)
        assert 0 <= brier <= 1


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        """Test ECE for perfectly calibrated predictions."""
        # Create perfectly calibrated predictions
        # 100% correct with 100% confidence
        predictions = np.array([1] * 50 + [0] * 50)
        targets = np.array([1] * 50 + [0] * 50)
        confidences = np.array([1.0] * 100)

        ece = compute_ece(predictions, targets, confidences, n_bins=10)
        # Should be 0 for perfect calibration
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_overconfident_predictions(self):
        """Test ECE for overconfident predictions."""
        # 50% accuracy but 90% confidence
        predictions = np.array([1] * 50 + [0] * 50)
        targets = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        confidences = np.array([0.9] * 100)

        ece = compute_ece(predictions, targets, confidences, n_bins=10)
        # ECE should be around |0.9 - 0.5| = 0.4
        assert 0.35 < ece < 0.45

    def test_underconfident_predictions(self):
        """Test ECE for underconfident predictions."""
        # 100% accuracy but 60% confidence
        predictions = np.array([1] * 50 + [0] * 50)
        targets = np.array([1] * 50 + [0] * 50)
        confidences = np.array([0.6] * 100)

        ece = compute_ece(predictions, targets, confidences, n_bins=10)
        # ECE should be around |0.6 - 1.0| = 0.4
        assert 0.35 < ece < 0.45

    def test_varying_bins(self):
        """Test ECE with different numbers of bins."""
        predictions = np.array([1] * 50 + [0] * 50)
        targets = np.array([1] * 50 + [0] * 50)
        confidences = np.random.uniform(0.4, 0.9, 100)

        ece_5 = compute_ece(predictions, targets, confidences, n_bins=5)
        ece_20 = compute_ece(predictions, targets, confidences, n_bins=20)

        assert isinstance(ece_5, float)
        assert isinstance(ece_20, float)
        assert 0 <= ece_5 <= 1
        assert 0 <= ece_20 <= 1


class TestBinPredictions:
    def test_uniform_distribution(self):
        """Test binning with uniform confidence distribution."""
        n_samples = 100
        predictions = np.random.randint(0, 2, n_samples)
        targets = np.random.randint(0, 2, n_samples)
        confidences = np.linspace(0, 1, n_samples)

        bins = bin_predictions_by_confidence(predictions, targets, confidences, n_bins=10)

        assert len(bins) == 10
        # Each bin should have roughly 10 samples
        for bin_data in bins:
            assert 'lower' in bin_data
            assert 'upper' in bin_data
            assert 'count' in bin_data
            assert 'accuracy' in bin_data
            assert 'avg_confidence' in bin_data
            # Roughly equal distribution
            assert 5 <= bin_data['count'] <= 15

    def test_single_bin(self):
        """Test with all confidences in one bin."""
        predictions = np.array([1, 1, 0, 0])
        targets = np.array([1, 0, 0, 1])
        confidences = np.array([0.85, 0.86, 0.87, 0.88])

        bins = bin_predictions_by_confidence(predictions, targets, confidences, n_bins=10)

        # Only one bin (0.8-0.9) should have samples
        non_empty_bins = [b for b in bins if b['count'] > 0]
        assert len(non_empty_bins) == 1
        assert non_empty_bins[0]['count'] == 4

    def test_bin_boundaries(self):
        """Test that bin boundaries are correct."""
        predictions = np.array([1] * 10)
        targets = np.array([1] * 10)
        confidences = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])

        bins = bin_predictions_by_confidence(predictions, targets, confidences, n_bins=10)

        # Check boundaries
        for i, bin_data in enumerate(bins):
            assert bin_data['lower'] == pytest.approx(i * 0.1)
            assert bin_data['upper'] == pytest.approx((i + 1) * 0.1)

    def test_edge_cases(self):
        """Test edge cases like empty arrays."""
        bins = bin_predictions_by_confidence([], [], [], n_bins=10)
        assert len(bins) == 10
        assert all(b['count'] == 0 for b in bins)


class TestComputeCalibrationMetrics:
    def test_dataframe_input(self):
        """Test calibration metrics computation from DataFrame."""
        df = pd.DataFrame({
            'confidence': [0.9, 0.8, 0.7, 0.6, 0.5],
            'is_correct': [True, True, False, True, False]
        })

        metrics = compute_calibration_metrics(df)

        assert 'brier_score' in metrics
        assert 'ece' in metrics
        assert isinstance(metrics['brier_score'], float)
        assert isinstance(metrics['ece'], float)

    def test_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'confidence': [0.9, np.nan, 0.7, 0.6, None],
            'is_correct': [True, False, None, True, False]
        })

        metrics = compute_calibration_metrics(df)

        # Should compute on valid rows only
        assert 'brier_score' in metrics
        assert 'ece' in metrics

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({'confidence': [], 'is_correct': []})

        metrics = compute_calibration_metrics(df)

        assert np.isnan(metrics['brier_score'])
        assert np.isnan(metrics['ece'])

    def test_custom_columns(self):
        """Test with custom column names."""
        df = pd.DataFrame({
            'model_confidence': [0.9, 0.8, 0.7],
            'correct': [1, 1, 0]
        })

        metrics = compute_calibration_metrics(
            df,
            confidence_col='model_confidence',
            prediction_col='correct'
        )

        assert 'brier_score' in metrics
        assert 'ece' in metrics
