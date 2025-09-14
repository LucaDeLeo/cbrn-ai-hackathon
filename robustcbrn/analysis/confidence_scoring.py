"""Confidence-based scoring and thresholded abstention system.

This module provides functions for applying confidence thresholds and computing
proportional penalties for incorrect answers based on confidence levels.
"""

import pandas as pd


def apply_confidence_threshold(
    df: pd.DataFrame,
    threshold: float,
    confidence_col: str = 'confidence',
    response_col: str = 'response'
) -> pd.DataFrame:
    """Apply confidence threshold to identify abstentions.

    Marks samples as abstentions if:
    1. Confidence is below threshold, OR
    2. Response contains "I don't know" or similar abstention phrases

    Args:
        df: DataFrame with predictions and confidences
        threshold: Confidence threshold in [0, 1]
        confidence_col: Name of confidence column
        response_col: Name of response text column

    Returns:
        DataFrame with added 'is_abstention' column
    """
    df = df.copy()

    # Initialize abstention column
    df['is_abstention'] = False

    # Check for confidence below threshold
    if confidence_col in df.columns:
        low_confidence = df[confidence_col] < threshold
        df.loc[low_confidence, 'is_abstention'] = True

    # Check for explicit abstention phrases
    if response_col in df.columns and len(df) > 0:
        # Only process if column exists and has string data
        try:
            abstention_phrases = [
                "i don't know",
                "i do not know",
                "i dont know",
                "cannot determine",
                "unable to answer",
                "insufficient information",
                "unclear",
                "not sure",
                "cannot answer"
            ]

            for phrase in abstention_phrases:
                mask = df[response_col].str.lower().str.contains(phrase, na=False)
                df.loc[mask, 'is_abstention'] = True
        except AttributeError:
            # Column doesn't contain string data, skip string matching
            pass

    return df


def compute_proportional_penalties(
    df: pd.DataFrame,
    threshold: float,
    is_correct_col: str = 'is_correct',
    confidence_col: str = 'confidence'
) -> pd.DataFrame:
    """Compute proportional penalties for incorrect answers.

    Applies the penalty formula: -t/(1-t) for incorrect answers
    where t is the confidence threshold.

    Args:
        df: DataFrame with abstention markings from apply_confidence_threshold
        threshold: Confidence threshold used for abstention
        is_correct_col: Name of correctness column (binary)
        confidence_col: Name of confidence column

    Returns:
        DataFrame with added 'penalty_score' column
    """
    df = df.copy()

    # Initialize penalty score
    df['penalty_score'] = 0.0

    # Check if abstention column exists, if not create it
    if 'is_abstention' not in df.columns:
        df = apply_confidence_threshold(df, threshold, confidence_col)

    # Compute penalty for non-abstentions
    non_abstention_mask = ~df['is_abstention']

    # Correct non-abstentions get score of 1
    correct_mask = non_abstention_mask & df[is_correct_col]
    df.loc[correct_mask, 'penalty_score'] = 1.0

    # Incorrect non-abstentions get proportional penalty
    if threshold < 1.0:  # Avoid division by zero
        penalty = -threshold / (1 - threshold)
        incorrect_mask = non_abstention_mask & ~df[is_correct_col]
        df.loc[incorrect_mask, 'penalty_score'] = penalty

    # Abstentions always get score of 0 (already initialized)

    return df


def compute_confidence_aware_metrics(
    df: pd.DataFrame,
    threshold: float,
    confidence_col: str = 'confidence',
    is_correct_col: str = 'is_correct',
    response_col: str = 'response'
) -> dict:
    """Compute comprehensive confidence-aware metrics for a given threshold.

    Args:
        df: DataFrame with predictions, confidences, and correctness
        threshold: Confidence threshold for abstention
        confidence_col: Name of confidence column
        is_correct_col: Name of correctness column
        response_col: Name of response text column

    Returns:
        Dictionary with metrics:
        - 'threshold': The confidence threshold used
        - 'abstention_rate': Fraction of samples that abstained
        - 'accuracy_on_answered': Accuracy on non-abstained samples
        - 'average_penalty_score': Mean penalty score across all samples
        - 'confident_correct': Fraction of confident (non-abstained) samples that are correct
    """
    # Apply threshold and compute penalties
    df_scored = apply_confidence_threshold(df, threshold, confidence_col, response_col)
    df_scored = compute_proportional_penalties(df_scored, threshold, is_correct_col, confidence_col)

    # Compute metrics
    n_total = len(df_scored)
    n_abstentions = df_scored['is_abstention'].sum()
    abstention_rate = n_abstentions / n_total if n_total > 0 else 0.0

    # Accuracy on answered questions
    answered_mask = ~df_scored['is_abstention']
    n_answered = answered_mask.sum()

    if n_answered > 0:
        accuracy_on_answered = df_scored.loc[answered_mask, is_correct_col].mean()
        confident_correct = accuracy_on_answered  # Same as accuracy on answered
    else:
        accuracy_on_answered = 0.0
        confident_correct = 0.0

    # Average penalty score
    average_penalty = df_scored['penalty_score'].mean()

    return {
        'threshold': threshold,
        'abstention_rate': float(abstention_rate),
        'accuracy_on_answered': float(accuracy_on_answered),
        'average_penalty_score': float(average_penalty),
        'confident_correct': float(confident_correct),
        'n_total': int(n_total),
        'n_abstentions': int(n_abstentions),
        'n_answered': int(n_answered)
    }


def evaluate_all_thresholds(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    confidence_col: str = 'confidence',
    is_correct_col: str = 'is_correct',
    response_col: str = 'response'
) -> dict:
    """Evaluate confidence-aware metrics across multiple thresholds.

    Args:
        df: DataFrame with predictions and confidences
        thresholds: List of confidence thresholds to evaluate
        confidence_col: Name of confidence column
        is_correct_col: Name of correctness column
        response_col: Name of response text column

    Returns:
        Dictionary mapping threshold values to their metrics
    """
    results = {}

    if thresholds is None:
        thresholds = [0.0, 0.5, 0.75, 0.9]

    for threshold in thresholds:
        metrics = compute_confidence_aware_metrics(
            df, threshold, confidence_col, is_correct_col, response_col
        )
        results[f't_{threshold}'] = metrics

    return results
