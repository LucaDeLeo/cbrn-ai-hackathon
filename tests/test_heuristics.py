"""Unit tests for heuristic analysis methods."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.analysis.heuristics import LongestAnswerHeuristic, analyze_questions, HeuristicReport
from src.data.schemas import Question


class TestLongestAnswerHeuristic(unittest.TestCase):
    """Tests for LongestAnswerHeuristic class."""

    def test_longest_answer_selection_and_ties(self):
        """Test longest answer selection with tie-breaking."""
        heuristic = LongestAnswerHeuristic()

        # Test basic longest selection
        q1 = Question(
            id="test1",
            question="Test question",
            choices=["short", "medium length", "longest answer here", "mid"],
            answer=2
        )
        self.assertEqual(heuristic.predict(q1), 2)

        # Test tie-breaking: first occurrence wins
        q2 = Question(
            id="test2",
            question="Test question",
            choices=["same", "same", "same", "same"],
            answer=0
        )
        self.assertEqual(heuristic.predict(q2), 0)  # First index on tie

        # Test tie with different positions
        q3 = Question(
            id="test3",
            question="Test question",
            choices=["short", "longer one", "mid", "longer one"],
            answer=1
        )
        self.assertEqual(heuristic.predict(q3), 1)  # First "longer one"

    def test_accuracy_computation_small_fixture(self):
        """Test accuracy computation on small fixture."""
        questions = [
            Question(
                id=f"q{i}",
                question=f"Question {i}",
                choices=["a", "bb", "ccc", "d"],
                answer=2  # Longest is correct
            )
            for i in range(5)
        ]
        # Add some where longest is wrong
        questions.extend([
            Question(
                id=f"q{i}",
                question=f"Question {i}",
                choices=["aaaa", "b", "c", "d"],
                answer=1  # Longest is wrong
            )
            for i in range(5, 8)
        ])

        report = analyze_questions(questions, show_progress=False)

        self.assertEqual(report.results["total_predictions"], 8)
        self.assertEqual(report.results["correct_predictions"], 5)
        self.assertAlmostEqual(report.results["accuracy"], 5/8)

    def test_processing_speed_1000_items_cpu_only_progress_off(self):
        """Test processing speed for 1000 items."""
        # Generate synthetic questions
        questions = [
            Question(
                id=f"speed_test_{i}",
                question=f"Question {i}",
                choices=[f"Choice {j}" * (j+1) for j in range(4)],
                answer=i % 4
            )
            for i in range(1000)
        ]

        start = time.time()
        report = analyze_questions(questions, show_progress=False)
        elapsed = time.time() - start

        self.assertLess(elapsed, 10.0, f"Processing took {elapsed:.2f}s, should be <10s")
        self.assertEqual(report.results["total_predictions"], 1000)
        self.assertGreater(report.performance["questions_per_second"], 100)

    def test_memory_usage_under_1gb_for_10k_items(self):
        """Test memory usage stays under 1GB for 10k items."""
        # Generate 10k synthetic questions
        questions = [
            Question(
                id=f"mem_test_{i}",
                question=f"Question text {i}" * 10,  # Moderate size
                choices=[f"Choice {j}" * 20 for j in range(4)],
                answer=i % 4
            )
            for i in range(10000)
        ]

        report = analyze_questions(questions, show_progress=False)

        # Check memory usage
        memory_mb = report.performance["memory_peak_mb"]
        self.assertLess(memory_mb, 1024, f"Memory usage {memory_mb:.2f}MB exceeds 1GB")
        self.assertEqual(report.results["total_predictions"], 10000)

    def test_results_json_schema_written(self):
        """Test JSON output schema and writing."""
        questions = [
            Question(
                id=f"json_test_{i}",
                question=f"Question {i}",
                choices=["a", "bb", "ccc", "d"],
                answer=2
            )
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "results" / "test_output.json"
            report = analyze_questions(questions, show_progress=False, save_path=save_path)

            # Check file was written
            self.assertTrue(save_path.exists())

            # Load and verify JSON structure
            with open(save_path) as f:
                data = json.load(f)

            # Verify required fields
            self.assertEqual(data["method"], "longest_answer")
            self.assertIn("timestamp", data)
            self.assertIsNone(data["config_hash"])  # Should be null for now

            # Verify dataset section
            self.assertIn("dataset", data)
            self.assertEqual(data["dataset"]["total_questions"], 3)

            # Verify results section
            self.assertIn("results", data)
            self.assertIn("correct_predictions", data["results"])
            self.assertIn("total_predictions", data["results"])
            self.assertIn("accuracy", data["results"])

            # Verify performance section
            self.assertIn("performance", data)
            self.assertIn("runtime_seconds", data["performance"])
            self.assertIn("memory_peak_mb", data["performance"])
            self.assertIn("questions_per_second", data["performance"])


if __name__ == "__main__":
    unittest.main()