"""Heuristic-based analysis methods for CBRN MCQAs."""

import json
import time
import tracemalloc
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import numpy as np

from robustcbrn.data.schemas import Question


@dataclass
class HeuristicReport:
    """Report structure for heuristic analysis results."""
    method: str
    timestamp: str
    config_hash: Optional[str]
    dataset: dict
    results: dict
    performance: dict


class LongestAnswerHeuristic:
    """Selects the answer choice with the longest raw string length.

    Tie-breaking policy: On equal lengths, chooses the first occurrence (lowest index).
    Length measured as raw string length without normalization for determinism.
    """

    def predict(self, question: Question) -> int:
        """Predict answer based on longest choice length.

        Args:
            question: Question object with choices list

        Returns:
            Zero-based index of predicted answer
        """
        if not question.choices:
            raise ValueError(f"Question '{question.id}' has no choices to evaluate")
        max_length = -1
        predicted_idx = 0

        for idx, choice in enumerate(question.choices):
            length = len(choice)
            if length > max_length:
                max_length = length
                predicted_idx = idx

        return predicted_idx


def _get_memory_usage_mb() -> float:
    """Get peak memory usage in MB using stdlib methods.

    Returns:
        Memory usage in MB (peak RSS when available)
    """
    try:
        # Prefer resource.getrusage (Unix/macOS)
        import resource  # Local import for portability (Windows)
        import sys

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KiB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)  # bytes → MB
        else:
            return usage.ru_maxrss / 1024  # KiB → MB
    except Exception:
        # Fallback to tracemalloc peak if resource not available
        if tracemalloc.is_tracing():
            _current, peak = tracemalloc.get_traced_memory()
            return peak / (1024 * 1024)  # bytes → MB
        return 0.0


def analyze_questions(
    questions: List[Question],
    show_progress: bool = False,
    save_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    dataset_hash: Optional[str] = None,
    debug: bool = False,
    tests_to_run: Optional[List[str]] = None,
    robust_questions: Optional[List[Question]] = None,
    stratify_by: Optional[np.ndarray] = None  # Add this
) -> HeuristicReport:
    """Analyze questions using longest-answer heuristic and statistical battery.

    Args:
        questions: List of Question objects to analyze
        show_progress: Whether to show progress bar
        save_path: Optional path to save JSON results
        dataset_path: Optional path to the dataset
        dataset_hash: Optional hash of the dataset
        debug: Debug mode
        tests_to_run: Optional list of specific statistical tests to run
        robust_questions: Optional list of robust Question objects for degradation analysis
        stratify_by: Optional array of strata labels for stratified bootstrap analysis

    Returns:
        HeuristicReport with analysis results
    """
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback for environments without tqdm
        def tqdm(iterable, disable=False, desc="Processing"):
            if not disable:
                print(f"{desc}...")
            return iterable
    
    import logging

    logger = logging.getLogger(__name__)
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    # Start memory tracking if not already active
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    start_time = time.time()

    heuristic = LongestAnswerHeuristic()
    correct_predictions = 0
    total_predictions = len(questions)

    # Process questions with optional progress bar
    for question in tqdm(questions, disable=not show_progress, desc="Analyzing"):
        # Validate question fields for robustness
        if question.answer is None or question.answer < 0 or question.answer >= len(question.choices):
            raise ValueError(
                f"Question '{question.id}' has invalid answer index {question.answer} for {len(question.choices)} choices"
            )
        predicted_idx = heuristic.predict(question)
        if predicted_idx == question.answer:
            correct_predictions += 1

    # Calculate metrics
    runtime_seconds = time.time() - start_time
    memory_peak_mb = _get_memory_usage_mb()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    questions_per_second = total_predictions / runtime_seconds if runtime_seconds > 0 else 0.0

    # Run StatisticalBattery analysis if requested or if robust questions are provided
    battery_results = {}
    if tests_to_run is not None or robust_questions is not None:
        try:
            # Convert questions to dictionaries for the battery
            questions_dict = []
            for q in questions:
                q_dict = {
                    'id': q.id,
                    'question': q.question,
                    'choices': q.choices,
                    'answer_index': q.answer,
                    # Add other fields as needed
                }
                questions_dict.append(q_dict)
            
            # Convert robust questions to dictionaries if provided
            robust_questions_dict = None
            if robust_questions:
                robust_questions_dict = []
                for q in robust_questions:
                    q_dict = {
                        'id': q.id,
                        'question': q.question,
                        'choices': q.choices,
                        'answer_index': q.answer,
                    }
                    robust_questions_dict.append(q_dict)
            
            # For now, just run basic position bias analysis
            # TODO: Integrate full StatisticalBattery when available
            from robustcbrn.statistical.position_bias import detect_position_bias
            
            if 'position_bias' in (tests_to_run or []):
                position_bias_results = detect_position_bias(questions_dict)
                battery_results["position_bias"] = position_bias_results
            
            # Add heuristic degradation analysis if robust questions provided
            if robust_questions_dict:
                # Simple degradation analysis
                original_accuracy = accuracy
                robust_correct = 0
                for q in robust_questions:
                    predicted_idx = heuristic.predict(q)
                    if predicted_idx == q.answer:
                        robust_correct += 1
                robust_accuracy = robust_correct / len(robust_questions) if robust_questions else 0.0
                
                degradation = original_accuracy - robust_accuracy
                battery_results["heuristic_degradation"] = {
                    "summary": {
                        "total_heuristics": 1,
                        "significant_degradations": 1 if degradation > 0.05 else 0,
                        "average_degradation": degradation,
                        "maximum_degradation": degradation,
                        "degradation_percentage": (degradation / original_accuracy * 100) if original_accuracy > 0 else 0
                    },
                    "heuristics": {
                        "longest_answer": {
                            "original_accuracy": original_accuracy,
                            "robust_accuracy": robust_accuracy,
                            "absolute_delta": degradation,
                            "is_significant": degradation > 0.05
                        }
                    }
                }
                
        except Exception as e:
            logger.warning(f"Statistical analysis failed: {e}")
            battery_results = {"error": str(e)}

    # Build dataset metadata (provenance-aware)
    dataset_meta = {
        "path": str(dataset_path) if dataset_path else None,
        "total_questions": total_predictions,
    }
    if dataset_hash is not None:
        dataset_meta["hash"] = dataset_hash

    # Create report
    report = HeuristicReport(
        method="longest_answer",
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        config_hash=None,  # Placeholder for Story 1.5
        dataset=dataset_meta,
        results={
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "accuracy": accuracy,
            **battery_results  # Include battery results if available
        },
        performance={
            "runtime_seconds": runtime_seconds,
            "memory_peak_mb": memory_peak_mb,
            "questions_per_second": questions_per_second
        }
    )

    # Save to JSON if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)

    return report
