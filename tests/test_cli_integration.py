"""Integration tests for CLI with actual file I/O operations."""

import json
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI commands with real file operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create a sample dataset file
        self.sample_data = [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris is the capital", "Madrid"],
                "answer": "C",
                "topic": "geography",
                "difficulty": "easy"
            },
            {
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
                "topic": "math",
                "difficulty": "easy"
            }
        ]

        self.test_dataset = self.temp_path / "test_dataset.jsonl"
        with open(self.test_dataset, 'w') as f:
            for item in self.sample_data:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analyze_with_real_file(self):
        """Test analyze command with actual file I/O."""
        output_file = self.temp_path / "analysis_results.json"

        # Run the analyze command
        cmd = [
            sys.executable, "cli.py", "analyze",
            "--input", str(self.test_dataset),
            "--output", str(output_file),
            "--max-items", "2"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

        # Check output file was created
        self.assertTrue(output_file.exists(), "Output file not created")

        # Verify output file contains valid JSON
        with open(output_file, 'r') as f:
            report = json.load(f)

        # Verify report structure
        self.assertIn("method", report)
        self.assertIn("results", report)
        self.assertIn("performance", report)
        self.assertEqual(report["method"], "longest_answer")
        self.assertEqual(report["results"]["total_predictions"], 2)

    def test_dry_run_with_real_file(self):
        """Test dry-run mode with actual file."""
        cmd = [
            sys.executable, "cli.py", "analyze",
            "--input", str(self.test_dataset),
            "--dry-run"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        self.assertEqual(result.returncode, 0)

        # Check validation message appears
        self.assertIn("Validation successful", result.stdout)
        self.assertIn("Questions loaded: 2", result.stdout)

    def test_verbose_output_with_real_file(self):
        """Test verbose mode shows progress."""
        cmd = [
            sys.executable, "cli.py", "analyze",
            "--input", str(self.test_dataset),
            "--verbose"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        self.assertEqual(result.returncode, 0)

        # Check verbose output appears
        self.assertIn("Analyzing 2 questions", result.stdout)
        self.assertIn("Analysis complete", result.stdout)

    def test_missing_file_error(self):
        """Test error handling for missing input file."""
        cmd = [
            sys.executable, "cli.py", "analyze",
            "--input", str(self.temp_path / "nonexistent.jsonl")
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command failed with correct exit code
        self.assertEqual(result.returncode, 1)

        # Check error message
        self.assertIn("not found", result.stdout)

    def test_invalid_json_handling(self):
        """Test handling of malformed JSON file."""
        bad_json_file = self.temp_path / "bad.jsonl"
        with open(bad_json_file, 'w') as f:
            f.write("{ invalid json }\n")

        cmd = [
            sys.executable, "cli.py", "analyze",
            "--input", str(bad_json_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command failed
        self.assertEqual(result.returncode, 1)

        # Check error message mentions JSON
        self.assertIn("Invalid JSON format", result.stdout)

    def test_output_directory_creation(self):
        """Test that output directories are created as needed."""
        nested_output = self.temp_path / "nested" / "dir" / "results.json"

        cmd = [
            sys.executable, "cli.py", "analyze",
            "--input", str(self.test_dataset),
            "--output", str(nested_output),
            "--max-items", "1"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        self.assertEqual(result.returncode, 0)

        # Check nested directories and file were created
        self.assertTrue(nested_output.exists())
        self.assertTrue(nested_output.parent.exists())

    def test_csv_file_support(self):
        """Test loading CSV format dataset."""
        csv_file = self.temp_path / "test.csv"
        with open(csv_file, 'w') as f:
            f.write("question,choice_a,choice_b,choice_c,choice_d,answer\n")
            f.write('"What is X?","Option A","Option B","Option C","Option D",1\n')

        cmd = [
            sys.executable, "cli.py", "analyze",
            "--input", str(csv_file),
            "--dry-run"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        self.assertEqual(result.returncode, 0)
        self.assertIn("Questions loaded: 1", result.stdout)


if __name__ == '__main__':
    unittest.main()