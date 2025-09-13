"""Statistical analysis module for position bias detection in MCQA benchmarks."""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import hashlib
import math
import numpy as np
from src.data.schemas import Question


@dataclass
class PositionBiasReport:
    """Report structure for position bias analysis results."""
    method: str = "position_bias_analysis"
    timestamp: str = ""
    dataset_info: Dict[str, Any] = None
    position_frequencies: Dict[str, int] = None
    chi_square_results: Dict[str, float] = None
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
    """Chi-square goodness-of-fit test.

    Formula:
    - Statistic: χ² = Σ_i ((O_i - E_i)**2 / E_i)
    - Degrees of freedom: df = k - 1
    - p-value computed via chi-square survival function (regularized Γ).
    """
    if len(observed) != len(expected):
        raise ValueError("Observed and expected arrays must have same length")
    
    if np.any(expected <= 0):
        raise ValueError("All expected frequencies must be positive")
    
    chi_square_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    p_value = _approximate_chi2_pvalue(float(chi_square_stat), int(df))
    
    return float(chi_square_stat), float(p_value)


def _approximate_chi2_pvalue(chi2_stat: float, df: int) -> float:
    """Compute chi-square p-value using regularized upper incomplete gamma.

    p = Q(k/2, x/2) where Q is the regularized upper incomplete gamma.
    Uses series for P when x < a+1 and continued fraction for Q otherwise.
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    if chi2_stat < 0:
        raise ValueError("Chi-square statistic must be non-negative")

    a = 0.5 * df
    x = 0.5 * chi2_stat

    if x == 0.0:
        return 1.0

    def _gammainc_series(a: float, x: float) -> float:
        # Regularized lower incomplete gamma P(a, x)
        eps = 1e-14
        term = 1.0 / a
        summation = term
        n = 1
        while True:
            term *= x / (a + n)
            summation += term
            if abs(term) < abs(summation) * eps or n > 100000:
                break
            n += 1
        return math.exp(-x + a * math.log(x) - math.lgamma(a)) * summation

    def _gammainc_cf(a: float, x: float) -> float:
        # Regularized upper incomplete gamma Q(a, x) via continued fraction (Lentz)
        eps = 1e-14
        max_iter = 100000
        tiny = 1e-300
        b = x + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, max_iter + 1):
            an = -i * (i - a)
            b = b + 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h

    if x < a + 1.0:
        p = _gammainc_series(a, x)
        q = 1.0 - p
    else:
        q = _gammainc_cf(a, x)
    return float(max(0.0, min(1.0, q)))


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
    """Identify questions where correct answer position is highly predictive.

    Requires p-value < threshold and a unique maximum frequency position
    within each group of questions sharing the same number of choices.
    """
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
            
        # Build full label set to keep zero-count positions
        position_labels = [chr(65 + i) for i in range(choice_count)]
        frequencies = {label: 0 for label in position_labels}
        for q in group_questions:
            if 0 <= q.answer < choice_count:
                frequencies[position_labels[q.answer]] += 1
        observed = np.array([frequencies[lbl] for lbl in position_labels], dtype=float)
        
        total = np.sum(observed)
        expected = np.full(len(observed), total / len(observed))
        
        try:
            chi2_stat, p_value = chi_square_test_from_scratch(observed, expected)
            
            if p_value < threshold:
                counts = np.array(list(frequencies.values()), dtype=int)
                max_count = int(counts.max())
                if np.sum(counts == max_count) != 1:
                    continue  # Tie - not predictive
                max_index = int(np.argmax(counts))
                for question in group_questions:
                    if question.answer == max_index:
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
                new_answer = pattern.index(question.answer) if question.answer in pattern else question.answer
            
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
    """Complete position bias analysis for Epic 2, Story 2.1."""
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
    
    # Correct aggregation of choice counts
    choice_counts: Dict[str, int] = {}
    for q in questions:
        kk = str(len(q.choices))
        choice_counts[kk] = choice_counts.get(kk, 0) + 1

    report = PositionBiasReport(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        dataset_info={
            'total_questions': len(questions),
            'choice_counts': choice_counts
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


def run_position_bias_analysis(questions: List[Question], **kwargs) -> Dict[str, Any]:
    """CLI integration function for position bias analysis."""
    report = analyze_position_bias(questions, **kwargs)
    return asdict(report)
