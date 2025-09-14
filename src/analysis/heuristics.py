"""Heuristic-based analysis methods for CBRN MCQAs."""

import json
import time
import tracemalloc
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import numpy as np

from src.data.schemas import Question


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
    from src.analysis.statistical_battery import StatisticalBattery

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
            
            # Run the statistical battery
            battery = StatisticalBattery(
                robust_questions=robust_questions_dict,
                stratify_by=stratify_by
            )
            if tests_to_run:
                battery_result = battery.run(questions_dict, tests=tests_to_run)
            else:
                # If robust questions are provided, include degradation analysis
                if robust_questions_dict:
                    battery_result = battery.run(questions_dict, tests=['heuristic_degradation'])
                else:
                    battery_result = battery.run_all(questions_dict)
            
            # Convert battery results to dict format
            battery_results = {
                "battery_status": battery_result.overall_status,
                "test_summary": battery_result.summary,
                "total_tests": len(battery_result.tests),
                "tests_passed": battery_result.performance.get("tests_passed", 0),
                "tests_with_warnings": battery_result.performance.get("tests_with_warnings", 0),
                "tests_failed": battery_result.performance.get("tests_failed", 0),
                "battery_runtime": battery_result.runtime_seconds
            }
            
            # Add individual test results
            for test_name, test_result in battery_result.tests.items():
                if test_name == "heuristic_degradation" and test_result.data:
                    # Special handling for heuristic degradation
                    degradation_data = test_result.data
                    battery_results["heuristic_degradation"] = {
                        "summary": degradation_data.get("summary", {}),
                        "heuristics": degradation_data.get("heuristics", {})
                    }
                else:
                    battery_results[f"test_{test_name}"] = {
                        "status": test_result.status,
                        "p_value": test_result.p_value,
                        "effect_size": test_result.effect_size,
                        "confidence_interval": test_result.confidence_interval,
                        "message": test_result.message
                    }
                
        except Exception as e:
            logger.warning(f"StatisticalBattery analysis failed: {e}")
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
