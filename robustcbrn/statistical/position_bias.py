"""Position-bias analysis for multiple-choice QA benchmarks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---- Question schema (try real type, else a lightweight fallback) ----------
try:
    from robustcbrn.data.schemas import Question  # expected: .id, .question, .choices, .answer
except Exception:  # pragma: no cover
    from collections.abc import Sequence
    from typing import NamedTuple
    class Question(NamedTuple):
        id: str
        question: str
        choices: Sequence[str]
        answer: Any  # index | letter | choice text

# Optional bootstrap (kept optional)
try:
    from robustcbrn.statistical.bootstrap import (
        BootstrapResult,
    )
    _BOOTSTRAP_OK = True
except Exception:  # pragma: no cover
    BootstrapResult = Any  # type: ignore
    _BOOTSTRAP_OK = False

# -----------------------------------------------------------------------------

@dataclass
class PositionBiasReport:
    """Report structure for position bias analysis."""
    method: str
    timestamp: str
    dataset_info: dict[str, Any]
    position_frequencies: dict[str, int]
    chi_square_results: dict[str, Any]
    predictive_questions: list[str]
    position_swaps: dict[str, list[dict[str, Any]]]
    summary_statistics: dict[str, Any]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _letter_to_index(letter: str) -> int | None:
    letter = letter.strip().upper()
    if len(letter) == 1 and "A" <= letter <= "Z":
        return ord(letter) - ord("A")
    return None


def _find_correct_position(q: Question) -> int | None:
    """
    Return 1-based position of the correct answer in q.choices.
    Accepts answer as 0/1-based index, letter (A,B,...) or exact choice text.
    """
    ans = q.answer
    n = len(q.choices)

    # int index
    if isinstance(ans, int):
        if 0 <= ans < n:
            return ans + 1
        if 1 <= ans <= n:
            return ans
        return None

    # numeric string
    if isinstance(ans, str) and ans.strip().isdigit():
        i = int(ans.strip())
        if 0 <= i < n:
            return i + 1
        if 1 <= i <= n:
            return i
        return None

    # letter
    if isinstance(ans, str):
        li = _letter_to_index(ans)
        if li is not None and 0 <= li < n:
            return li + 1

    # exact choice text
    if isinstance(ans, str):
        try:
            idx = [c.strip() for c in q.choices].index(ans.strip())
            return idx + 1
        except ValueError:
            return None

    return None


def _regularized_gamma_p(s: float, x: float) -> float:
    """
    Regularized lower incomplete gamma P(s,x) using series/continued fraction (NR style).
    Accurate enough for chi-square CDF without SciPy.
    """
    if x < 0 or s <= 0:
        return float("nan")
    if x == 0:
        return 0.0

    # choose technique
    if x < s + 1:
        # series
        term = 1.0 / s
        summ = term
        k = 1
        while True:
            term *= x / (s + k)
            summ += term
            if abs(term) < abs(summ) * 1e-12 or k > 10_000:
                break
            k += 1
        return summ * math.exp(-x + s * math.log(x) - math.lgamma(s))
    else:
        # continued fraction for Q, return P = 1 - Q
        # Lentz's algorithm
        a0 = 1.0
        b0 = x + 1.0 - s
        f = a0 / b0
        c = 1.0 / 1e-30
        d = 1.0 / b0
        for i in range(1, 10_000):
            a = i * (s - i)
            b = b0 + 2.0 * i
            d = 1.0 / (b + a * d)
            c = b + a / c
            delta = c * d
            f *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        return 1.0 - f * math.exp(-x + s * math.log(x) - math.lgamma(s))


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) for chi-square(df) using regularized gamma."""
    s = df / 2.0
    return max(0.0, min(1.0, 1.0 - _regularized_gamma_p(s, x / 2.0)))


def _approximate_normal_cdf(x: float) -> float:
    """
    Approximate the standard normal CDF using a rational approximation.
    Based on Abramowitz and Stegun approximation.
    """
    # Constants for the approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def calculate_position_frequencies(questions: list[Question]) -> dict[str, int]:
    """Count how often each position (A, B, C, D...) contains the correct answer."""
    if not questions:
        raise ValueError("Cannot calculate frequencies for empty question list")

    counts: dict[str, int] = {}
    for q in questions:
        pos = _find_correct_position(q)
        if pos is None:
            raise ValueError(f"Invalid answer for question {q.id}: {q.answer}")

        # Convert 1-based position to letter (A=1, B=2, C=3, D=4, etc.)
        key = chr(ord('A') + pos - 1) if 1 <= pos <= 26 else str(pos)

        counts[key] = counts.get(key, 0) + 1
    return counts


def _group_by_choice_count(questions: list[Question]) -> dict[int, list[Question]]:
    groups: dict[int, list[Question]] = {}
    for q in questions:
        k = len(q.choices)
        groups.setdefault(k, []).append(q)
    return groups


def chi_square_test_from_scratch(observed: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """
    Classic Pearson chi-square and p-value (no SciPy).

    Formula: chi² = Σ (O_i - E_i)² / E_i
    where O_i is observed frequency and E_i is expected frequency.
    """
    if observed.shape != expected.shape:
        raise ValueError("Observed and expected must have same shape.")
    if np.any(expected <= 0):
        raise ValueError("Expected frequencies must be positive.")

    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    df = observed.size - 1
    p = _chi2_sf(chi2, df)
    return chi2, p


def identify_predictive_questions(questions: list[Question], threshold: float = 0.05) -> list[str]:
    """
    Heuristic: within each choice-count group, flag positions that are
    significantly over-represented (standardized residuals), then return the
    IDs of questions whose correct answer sits in those positions.
    """
    predictive_ids: list[str] = []
    groups = _group_by_choice_count(questions)

    z_cut = 1.96 if threshold >= 0.05 else 2.58  # rough

    for k, qs in groups.items():
        # counts by position 1..k
        counts = np.zeros(k, dtype=float)
        for q in qs:
            pos = _find_correct_position(q)
            if pos:
                counts[pos - 1] += 1

        n = counts.sum()
        if n < 10:
            continue
        exp = np.full(k, n / k)
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = (counts - exp) / np.sqrt(exp)
            resid[np.isnan(resid)] = 0.0

        hot_positions = {i + 1 for i, z in enumerate(resid) if z >= z_cut}
        if not hot_positions:
            continue

        for q in qs:
            pos = _find_correct_position(q)
            if pos in hot_positions:
                predictive_ids.append(q.id)

    # de-dup preserving order
    seen = set()
    out = []
    for qid in predictive_ids:
        if qid not in seen:
            out.append(qid)
            seen.add(qid)
    return out


def _calculate_checksum(question_id: str, choices: list[str], answer_repr: Any) -> str:
    h = hashlib.sha256()
    h.update(str(question_id).encode("utf-8"))
    h.update(b"\x1f")
    for c in choices:
        h.update(c.strip().encode("utf-8"))
        h.update(b"\x1f")
    h.update(str(answer_repr).strip().encode("utf-8"))
    return h.hexdigest()


def generate_position_swaps(question: Question) -> list[dict[str, Any]]:
    """Create deterministic choice permutations that move the correct answer to each position."""
    swaps: list[dict[str, Any]] = []
    n = len(question.choices)
    pos = _find_correct_position(question)
    if pos is None:
        return swaps

    correct_idx0 = pos - 1
    question.choices[correct_idx0]

    for target_pos in range(n):
        if target_pos == correct_idx0:
            continue
        new_choices = list(question.choices)
        # swap correct into target_pos
        new_choices[correct_idx0], new_choices[target_pos] = new_choices[target_pos], new_choices[correct_idx0]
        checksum = _calculate_checksum(question.question, new_choices, target_pos)
        swaps.append({
            "id": f"{question.id}_swap_{target_pos + 1}",
            "question": question.question,
            "choices": new_choices,
            "answer": target_pos,
            "original_id": question.id,
            "swap_pattern": f"A{correct_idx0 + 1}->A{target_pos + 1}",
            "checksum": checksum,
        })

    return swaps


def analyze_position_biais_core(questions: list[Question]) -> tuple[dict[str, int], float, float, int, np.ndarray, np.ndarray]:
    """
    Compute frequencies and a *grouped* chi-square (sum over each choice-count group).
    Returns: (freqs, chi2, p, df, observed_concat, expected_concat)
    """
    # Frequencies for report
    freqs = calculate_position_frequencies(questions)

    # Group-wise chi-square, then sum
    groups = _group_by_choice_count(questions)
    chi2_total = 0.0
    df_total = 0
    observed_all: list[float] = []
    expected_all: list[float] = []

    for k, qs in groups.items():
        if len(qs) == 0:
            continue
        counts = np.zeros(k, dtype=float)
        for q in qs:
            pos = _find_correct_position(q)
            if pos:
                counts[pos - 1] += 1
        n = counts.sum()
        if n < 1:
            continue
        exp = np.full(k, n / k)
        chi2_k, _ = chi_square_test_from_scratch(counts, exp)
        df_k = counts.size - 1
        chi2_total += chi2_k
        df_total += df_k
        observed_all.extend(counts.tolist())
        expected_all.extend(exp.tolist())

    p_total = _chi2_sf(chi2_total, max(df_total, 1))
    return freqs, chi2_total, p_total, df_total, np.array(observed_all), np.array(expected_all)


def analyze_position_bias(
    questions: list[Question],
    significance_level: float = 0.05,
    save_path: Path | None = None,
) -> PositionBiasReport:
    """Standard position bias analysis (no bootstrap)."""
    if not questions:
        raise ValueError("Cannot analyze position bias for empty question list")

    freqs, chi2, p, df, obs, exp = analyze_position_biais_core(questions)

    predictive_qids = identify_predictive_questions(questions, threshold=significance_level)

    # sample swaps for first few questions (useful for audit)
    position_swaps: dict[str, list[dict[str, Any]]] = {}
    for q in questions[: min(5, len(questions))]:
        sw = generate_position_swaps(q)
        if sw:
            position_swaps[q.id] = sw

    report = PositionBiasReport(
        method="position_bias_analysis",
        timestamp=_now_iso(),
        dataset_info={
            "total_questions": len(questions),
            "choice_counts": {str(k): sum(1 for q in questions if len(q.choices) == k)
                              for k in sorted({len(q.choices) for q in questions})},
        },
        position_frequencies=freqs,
        chi_square_results={
            "chi_square_statistic": chi2,
            "p_value": p,
            "degrees_of_freedom": df,
            "observed_frequencies": obs.tolist(),
            "expected_frequencies": exp.tolist(),
            "significant": bool(p < significance_level),
        },
        predictive_questions=predictive_qids,
        position_swaps=position_swaps,
        summary_statistics={
            "bias_detected": bool(p < significance_level),
            "predictive_question_count": len(predictive_qids),
            "significance_level": significance_level,
        },
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    return report


def detect_position_bias(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Simplified interface for CLI integration.
    Converts dict-based questions to Question objects and returns simplified results.
    """
    # Convert dict questions to Question objects
    question_objects = []
    for q_dict in questions:
        question_objects.append(Question(
            id=q_dict['id'],
            question=q_dict['question'],
            choices=q_dict['choices'],
            answer=q_dict['answer_index']
        ))

    # Run analysis
    freqs, chi2, p, df, obs, exp = analyze_position_biais_core(question_objects)

    # Calculate effect size (Cramer's V)
    n = obs.sum()
    effect_size = np.sqrt(chi2 / (n * (min(len(obs), len(exp)) - 1))) if n > 0 else 0.0

    return {
        'observed_frequencies': obs.tolist(),
        'expected_frequencies': exp.tolist(),
        'chi_square_statistic': chi2,
        'p_value': p,
        'effect_size': effect_size,
        'significant': p < 0.05,
        'predictive_questions': identify_predictive_questions(question_objects)
    }
