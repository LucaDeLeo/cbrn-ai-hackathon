"""Calibration metrics for confidence-aware evaluation.

This module provides functions to compute calibration metrics like Brier Score
and Expected Calibration Error (ECE) for evaluating model confidence calibration.
"""

import numpy as np
import pandas as pd


def compute_brier_score(
    predictions: np.ndarray | list[int],
    targets: np.ndarray | list[int],
    confidences: np.ndarray | list[float],
) -> float:
    """Compute Brier Score for binary classification.

    The Brier Score measures the mean squared difference between predicted
    probabilities and actual outcomes:
    BS = (1/N) Σ(confidence_i - correct_i)²

    Args:
        predictions: Binary predictions (0 or 1)
        targets: True binary labels (0 or 1)
        confidences: Confidence scores in [0, 1]

    Returns:
        Brier Score (lower is better, range [0, 1])
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    confidences = np.array(confidences)

    # Convert to binary correctness
    correct = (predictions == targets).astype(float)

    # Compute Brier Score
    brier_score = np.mean((confidences - correct) ** 2)

    return float(brier_score)


def compute_ece(
    predictions: np.ndarray | list[int],
    targets: np.ndarray | list[int],
    confidences: np.ndarray | list[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the expected difference between confidence and accuracy
    across confidence bins:
    ECE = Σ(n_b/N) * |acc_b - conf_b|

    where n_b is the number of samples in bin b, acc_b is the accuracy
    in bin b, and conf_b is the average confidence in bin b.

    Args:
        predictions: Binary predictions (0 or 1)
        targets: True binary labels (0 or 1)
        confidences: Confidence scores in [0, 1]
        n_bins: Number of bins for confidence discretization

    Returns:
        ECE value (lower is better, range [0, 1])
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    confidences = np.array(confidences)

    # Get binned predictions
    bin_data = bin_predictions_by_confidence(predictions, targets, confidences, n_bins)

    ece = 0.0
    total_samples = len(predictions)

    for bin_info in bin_data:
        if bin_info['count'] > 0:
            weight = bin_info['count'] / total_samples
            ece += weight * abs(bin_info['accuracy'] - bin_info['avg_confidence'])

    return float(ece)


def bin_predictions_by_confidence(
    predictions: np.ndarray | list[int],
    targets: np.ndarray | list[int],
    confidences: np.ndarray | list[float],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Bin predictions by confidence levels for calibration analysis.

    Divides predictions into equal-width confidence bins and computes
    statistics for each bin.

    Args:
        predictions: Binary predictions (0 or 1)
        targets: True binary labels (0 or 1)
        confidences: Confidence scores in [0, 1]
        n_bins: Number of bins for confidence discretization

    Returns:
        List of dictionaries containing bin statistics:
        - 'lower': Lower bound of confidence bin
        - 'upper': Upper bound of confidence bin
        - 'count': Number of samples in bin
        - 'accuracy': Accuracy of predictions in bin
        - 'avg_confidence': Average confidence in bin
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    confidences = np.array(confidences)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Find samples in this bin
        if i == n_bins - 1:  # Include upper boundary in last bin
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        bin_dict = {
            'lower': float(lower),
            'upper': float(upper),
            'count': int(np.sum(in_bin)),
            'accuracy': 0.0,
            'avg_confidence': 0.0
        }

        if bin_dict['count'] > 0:
            bin_predictions = predictions[in_bin]
            bin_targets = targets[in_bin]
            bin_confidences = confidences[in_bin]

            # Compute accuracy
            correct = bin_predictions == bin_targets
            bin_dict['accuracy'] = float(np.mean(correct))

            # Compute average confidence
            bin_dict['avg_confidence'] = float(np.mean(bin_confidences))

        bin_data.append(bin_dict)

    return bin_data


def compute_calibration_metrics(
    df: pd.DataFrame,
    confidence_col: str = "confidence",
    prediction_col: str = "is_correct",
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute all calibration metrics from a DataFrame.

    Args:
        df: DataFrame with predictions and confidences
        confidence_col: Name of confidence column
        prediction_col: Name of correctness column (binary)
        n_bins: Number of bins for ECE computation

    Returns:
        Dictionary with 'brier_score' and 'ece' metrics
    """
    # Filter out rows with missing values
    valid_mask = df[confidence_col].notna() & df[prediction_col].notna()
    valid_df = df[valid_mask]

    if len(valid_df) == 0:
        return {"brier_score": np.nan, "ece": np.nan}

    # Extract arrays
    confidences = valid_df[confidence_col].values
    # For Brier score, we need binary correctness (1 for correct, 0 for incorrect)
    correct = valid_df[prediction_col].astype(float).values

    # Create dummy predictions and targets for compatibility
    # Since we already have correctness, we can create matching predictions/targets
    predictions = np.ones(len(correct))
    targets = correct.copy()

    # Compute metrics
    brier = compute_brier_score(predictions, targets, confidences)
    ece = compute_ece(predictions, targets, confidences, n_bins)

    return {"brier_score": brier, "ece": ece}
