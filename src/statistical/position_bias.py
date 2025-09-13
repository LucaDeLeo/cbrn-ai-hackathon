"""Statistical analysis module for position bias detection in MCQA benchmarks."""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import hashlib
import numpy as np
from src.data.schemas import Question


@dataclass
class PositionBiasReport:
    """Standard report structure for position bias analysis."""
    method: str = "position_bias_analysis"
    timestamp: str = ""
    dataset_info: Dict[str, Any] = None
    position_frequencies: Dict[str, int] = None
    chi_square_results: Dict[str, float] = None
    predictive_questions: List[str] = None
    position_swaps: Dict[str, List[Dict]] = None
    summary_statistics: Dict[str, Any] = None


@dataclass
class BootstrapStats:
    """Bootstrap statistics for uncertainty quantification."""
    statistic: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    n_bootstrap: int
    method: str


@dataclass
class EnhancedPositionBiasReport:
    """Enhanced report with bootstrap confidence intervals."""
    method: str = "position_bias_analysis_with_bootstrap"
    timestamp: str = ""
    dataset_info: Dict[str, Any] = None
    position_frequencies: Dict[str, int] = None
    chi_square_results: Dict[str, float] = None

    # Bootstrap fields
    bootstrap_chi_square: Optional[BootstrapStats] = None
    bootstrap_position_proportions: Optional[Dict[str, BootstrapStats]] = None
    bootstrap_performance: Dict[str, float] = None

    predictive_questions: List[str] = None
    position_swaps: Dict[str, List[Dict]] = None
    summary_statistics: Dict[str, Any] = None


def calculate_position_frequencies(questions: List[Question]) -> Dict[str, int]:
    """Calculate frequency distribution of correct answers across positions A,B,C,D."""
    if not questions:
        raise ValueError("Questions list cannot be empty")

    max_choices = max(len(q.choices) for q in questions)
    position_labels = [chr(65 + i) for i in range(max_choices)]

    frequencies = {label: 0 for label in position_labels}

    for question in questions:
        if question.answer >= len(question.choices):
            raise ValueError(f"Question {question.id} has answer index {question.answer} "
                             f"but only {len(question.choices)} choices")

        position_label = position_labels[question.answer]
        frequencies[position_label] += 1

    frequencies = {k: v for k, v in frequencies.items() if v > 0}
    return frequencies


def chi_square_test_from_scratch(observed: np.ndarray, expected: np.ndarray) -> Tuple[float, float]:
    """Chi-square test implementation using only NumPy."""
    if len(observed) != len(expected):
        raise ValueError("Observed and expected arrays must have same length")

    if np.any(expected <= 0):
        raise ValueError("All expected frequencies must be positive")

    chi_square_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    p_value = _approximate_chi2_pvalue(chi_square_stat, df)

    return float(chi_square_stat), float(p_value)


def _approximate_chi2_pvalue(chi2_stat: float, df: int) -> float:
    """Approximate chi-square p-value using gamma function approximation."""
    if df == 1:
        z = np.sqrt(chi2_stat)
        p_value = 2 * (1 - _approximate_normal_cdf(z))
    elif df <= 30:
        p_value = np.exp(-chi2_stat / 2) * (chi2_stat / 2) ** (df / 2 - 1)
        p_value = np.clip(p_value, 0, 1)
    else:
        mean = df
        variance = 2 * df
        z = (chi2_stat - mean) / np.sqrt(variance)
        p_value = 1 - _approximate_normal_cdf(z)

    return float(np.clip(p_value, 1e-10, 1.0))


def _approximate_normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using Abramowitz and Stegun approximation."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    sign = 1 if z >= 0 else -1
    z = abs(z)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)

    return 0.5 * (1 + sign * y)


def identify_predictive_questions(questions: List[Question], threshold: float = 0.05) -> List[str]:
    """Identify questions where correct answer position is highly predictive."""
    predictive_question_ids = []

    questions_by_choice_count = {}
    for question in questions:
        choice_count = len(question.choices)
        if choice_count not in questions_by_choice_count:
            questions_by_choice_count[choice_count] = []
        questions_by_choice_count[choice_count].append(question)

    for choice_count, group_questions in questions_by_choice_count.items():
        if len(group_questions) < 10:
            continue

        frequencies = calculate_position_frequencies(group_questions)
        observed = np.array(list(frequencies.values()))

        total = np.sum(observed)
        expected = np.full(len(observed), total / len(observed))

        try:
            chi2_stat, p_value = chi_square_test_from_scratch(observed, expected)

            if p_value < threshold:
                max_freq_position = max(frequencies.keys(), key=frequencies.get)
                max_position_index = ord(max_freq_position) - ord('A')

                for question in group_questions:
                    if question.answer == max_position_index:
                        predictive_question_ids.append(question.id)

        except (ValueError, ZeroDivisionError):
            continue

    return predictive_question_ids


def generate_position_swaps(question: Question) -> List[Dict[str, Any]]:
    """Generate position swap variants for a question."""
    num_choices = len(question.choices)
    if num_choices < 2:
        return []

    variants = []
    original_choices = question.choices.copy()

    swap_patterns = []

    if num_choices >= 4:
        swap_patterns.extend([
            (0, 3),  # A ↔ D
            (1, 2),  # B ↔ C
            (0, 3, 1, 2),  # A→D, B→C, C→B, D→A
        ])

    if num_choices >= 2:
        for i in range(num_choices - 1):
            swap_patterns.append((i, i + 1))

    for rotation in range(1, min(num_choices, 4)):
        pattern = [(i + rotation) % num_choices for i in range(num_choices)]
        swap_patterns.append(tuple(pattern))

    for pattern in swap_patterns:
        try:
            if len(pattern) == 2:
                i, j = pattern
                new_choices = original_choices.copy()
                new_choices[i], new_choices[j] = new_choices[j], new_choices[i]

                if question.answer == i:
                    new_answer = j
                elif question.answer == j:
                    new_answer = i
                else:
                    new_answer = question.answer

            else:
                new_choices = [original_choices[pattern[i]] for i in range(num_choices)]
                new_answer = pattern.index(
                    question.answer) if question.answer in pattern else question.answer

            variant = {
                'id': f"{question.id}_swap_{hash(pattern) % 10000:04d}",
                'question': question.question,
                'choices': new_choices,
                'answer': new_answer,
                'original_id': question.id,
                'swap_pattern': pattern,
                'checksum': _calculate_checksum(question.question, new_choices, new_answer)
            }

            variants.append(variant)

        except (IndexError, ValueError):
            continue

    return variants


def _calculate_checksum(question: str, choices: List[str], answer: int) -> str:
    """Calculate checksum for validation of option reindexing."""
    content = f"{question}|{','.join(choices)}|{answer}"
    return hashlib.md5(content.encode()).hexdigest()[:8]


def analyze_position_bias(
    questions: List[Question],
    significance_level: float = 0.05,
    save_path: Optional[Path] = None
) -> PositionBiasReport:
    """Standard position bias analysis without bootstrap."""
    if not questions:
        raise ValueError("Questions list cannot be empty")

    frequencies = calculate_position_frequencies(questions)

    observed = np.array(list(frequencies.values()))
    total = np.sum(observed)
    expected = np.full(len(observed), total / len(observed))

    chi2_stat, p_value = chi_square_test_from_scratch(observed, expected)

    predictive_questions = identify_predictive_questions(questions, significance_level)

    sample_questions = questions[:min(5, len(questions))]
    position_swaps = {}
    for question in sample_questions:
        swaps = generate_position_swaps(question)
        if swaps:
            position_swaps[question.id] = swaps

    report = PositionBiasReport(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        dataset_info={
            'total_questions': len(questions),
            'choice_counts': {str(k): v for k, v in
                              {len(q.choices): sum(
                                  1 for q in questions if len(q.choices) == len(q.choices))
                               for q in questions}.items()}
        },
        position_frequencies=frequencies,
        chi_square_results={
            'chi_square_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': len(observed) - 1,
            'observed_frequencies': observed.tolist(),
            'expected_frequencies': expected.tolist(),
            'significant': p_value < significance_level
        },
        predictive_questions=predictive_questions,
        position_swaps=position_swaps,
        summary_statistics={
            'total_variants_generated': sum(len(swaps) for swaps in position_swaps.values()),
            'bias_detected': p_value < significance_level,
            'predictive_question_count': len(predictive_questions),
            'significance_level': significance_level
        }
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)

    return report


def analyze_position_bias_with_bootstrap(
    questions: List[Question],
    significance_level: float = 0.05,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    bootstrap_method: str = "percentile",
    save_path: Optional[Path] = None
) -> EnhancedPositionBiasReport:
    """Position bias analysis with bootstrap confidence intervals."""

    if not questions:
        raise ValueError("Questions list cannot be empty")

    # Import bootstrap functionality
    from src.statistical.bootstrap import bootstrap_ci, bootstrap_proportion_ci
    import time

    bootstrap_start = time.time()

    # Original analysis
    frequencies = calculate_position_frequencies(questions)
    observed = np.array(list(frequencies.values()))
    total = np.sum(observed)
    expected = np.full(len(observed), total / len(observed))

    chi2_stat, p_value = chi_square_test_from_scratch(observed, expected)

    # Bootstrap analysis for chi-square statistic
    def chi_square_statistic(position_data):
        """Calculate chi-square statistic from position data."""
        unique, counts = np.unique(position_data, return_counts=True)

        # Pad to ensure all positions represented
        full_counts = np.zeros(len(expected), dtype=int)
        for i, count in zip(unique.astype(int), counts):
            if 0 <= i < len(full_counts):
                full_counts[i] = count

        return np.sum((full_counts - expected) ** 2 / expected)

    # Create position data for bootstrap (array of answer positions)
    position_data = np.array([q.answer for q in questions])

    bootstrap_chi2_result = bootstrap_ci(
        position_data,
        chi_square_statistic,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=bootstrap_method,
        random_seed=42  # For reproducibility
    )

    bootstrap_chi2_stats = BootstrapStats(
        statistic=bootstrap_chi2_result.statistic,
        confidence_interval=bootstrap_chi2_result.confidence_interval,
        confidence_level=confidence_level,
        n_bootstrap=bootstrap_chi2_result.n_iterations,
        method=bootstrap_method
    )

    # Bootstrap analysis for position proportions
    bootstrap_proportions = {}
    for pos_idx, (pos_label, count) in enumerate(frequencies.items()):
        # Create binary indicator for this position
        position_indicator = (position_data == pos_idx).astype(int)

        if len(position_indicator) > 0:
            prop_result = bootstrap_proportion_ci(
                position_indicator,
                n_bootstrap=min(n_bootstrap, 5000),  # Faster for proportions
                confidence_level=confidence_level,
                method=bootstrap_method,
                random_seed=42 + pos_idx
            )

            bootstrap_proportions[pos_label] = BootstrapStats(
                statistic=prop_result.statistic,
                confidence_interval=prop_result.confidence_interval,
                confidence_level=confidence_level,
                n_bootstrap=prop_result.n_iterations,
                method=bootstrap_method
            )

    # Original analysis components
    predictive_questions = identify_predictive_questions(questions, significance_level)

    sample_questions = questions[:min(5, len(questions))]
    position_swaps = {}
    for question in sample_questions:
        swaps = generate_position_swaps(question)
        if swaps:
            position_swaps[question.id] = swaps

    bootstrap_runtime = time.time() - bootstrap_start

    # Enhanced report with bootstrap results
    report = EnhancedPositionBiasReport(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        dataset_info={
            'total_questions': len(questions),
            'choice_counts': {str(k): v for k, v in
                              {len(q.choices): sum(
                                  1 for q in questions if len(q.choices) == len(q.choices))
                               for q in questions}.items()}
        },
        position_frequencies=frequencies,
        chi_square_results={
            'chi_square_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': len(observed) - 1,
            'observed_frequencies': observed.tolist(),
            'expected_frequencies': expected.tolist(),
            'significant': p_value < significance_level
        },
        bootstrap_chi_square=bootstrap_chi2_stats,
        bootstrap_position_proportions=bootstrap_proportions,
        bootstrap_performance={
            'runtime_seconds': bootstrap_runtime,
            'total_bootstrap_iterations': sum(
                [bootstrap_chi2_result.n_iterations] +
                [stats.n_bootstrap for stats in bootstrap_proportions.values()]
            ),
            'convergence_achieved': bootstrap_chi2_result.converged,
            'average_iterations_per_statistic': (
                                                    bootstrap_chi2_result.n_iterations +
                                                    sum(stats.n_bootstrap for stats in
                                                        bootstrap_proportions.values())
                                                ) / (1 + len(bootstrap_proportions))
        },
        predictive_questions=predictive_questions,
        position_swaps=position_swaps,
        summary_statistics={
            'total_variants_generated': sum(len(swaps) for swaps in position_swaps.values()),
            'bias_detected': p_value < significance_level,
            'predictive_question_count': len(predictive_questions),
            'significance_level': significance_level,
            'bootstrap_confidence_level': confidence_level,
            'bootstrap_method': bootstrap_method,
            'chi_square_ci_width': bootstrap_chi2_stats.confidence_interval[1] -
                                   bootstrap_chi2_stats.confidence_interval[0],
            'chi_square_significant_by_bootstrap': (
                bootstrap_chi2_stats.confidence_interval[0] >
                _get_critical_chi_square_value(len(observed) - 1, significance_level)
            )
        }
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            # Convert dataclass to dict, handling nested dataclasses
            report_dict = _serialize_bootstrap_report(report)
            json.dump(report_dict, f, indent=2)

    return report


def _serialize_bootstrap_report(report: EnhancedPositionBiasReport) -> Dict[str, Any]:
    """Serialize report with nested BootstrapStats dataclasses."""
    report_dict = asdict(report)

    # Convert BootstrapStats to dict
    if report_dict['bootstrap_chi_square']:
        report_dict['bootstrap_chi_square'] = asdict(report.bootstrap_chi_square)

    if report_dict['bootstrap_position_proportions']:
        bootstrap_props = {}
        for pos, stats in report.bootstrap_position_proportions.items():
            bootstrap_props[pos] = asdict(stats)
        report_dict['bootstrap_position_proportions'] = bootstrap_props

    return report_dict


def _get_critical_chi_square_value(df: int, alpha: float) -> float:
    """Get critical chi-square value for given df and alpha."""
    # Simplified critical values for common cases
    critical_values = {
        1: {0.05: 3.841, 0.01: 6.635},
        2: {0.05: 5.991, 0.01: 9.210},
        3: {0.05: 7.815, 0.01: 11.345},
        4: {0.05: 9.488, 0.01: 13.277},
    }

    if df in critical_values and alpha in critical_values[df]:
        return critical_values[df][alpha]
    else:
        # Rough approximation
        return df + 2.0 * np.sqrt(2 * df) if alpha == 0.05 else df + 3.0 * np.sqrt(2 * df)


# CLI integration functions - CLEAN and SIMPLE
def run_position_bias_analysis(questions: List[Question], **kwargs) -> Dict[str, Any]:
    """CLI integration function for position bias analysis."""
    if 'n_bootstrap' in kwargs:
        # Use bootstrap version
        report = analyze_position_bias_with_bootstrap(questions, **kwargs)
        return _serialize_bootstrap_report(report)
    else:
        # Use standard version
        standard_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ['n_bootstrap', 'confidence_level', 'bootstrap_method']}
        report = analyze_position_bias(questions, **standard_kwargs)
        return asdict(report)


def run_enhanced_position_bias_analysis(questions: List[Question], **kwargs) -> Dict[str, Any]:
    """CLI integration function for bootstrap position bias analysis."""
    report = analyze_position_bias_with_bootstrap(questions, **kwargs)
    return _serialize_bootstrap_report(report)
