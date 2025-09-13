"""Unit tests for CLI analyze command functionality."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call

from src.data.schemas import Question
from src.analysis.heuristics import HeuristicReport


class TestCLIAnalyzeCommand(unittest.TestCase):
    """Test cases for the analyze CLI command."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_questions = [
            Question(
                id="q1",
                question="What is X?",
                choices=["Short", "Much longer answer", "Mid"],
                answer=1,
                topic="test",
                difficulty="easy",
                metadata={}
            ),
            Question(
                id="q2",
                question="What is Y?",
                choices=["A", "B", "C", "D"],
                answer=2,
                topic="test",
                difficulty="medium",
                metadata={}
            )
        ]

        self.sample_report = HeuristicReport(
            method="longest_answer",
            timestamp="2025-01-01T00:00:00Z",
            config_hash=None,
            dataset={"path": "test.jsonl", "total_questions": 2},
            results={"correct_predictions": 1, "total_predictions": 2, "accuracy": 0.5},
            performance={"runtime_seconds": 0.1, "memory_peak_mb": 10.0, "questions_per_second": 20.0}
        )

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl', '--output', 'results.json'])
    @patch('cli.analyze_questions')
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_analyze_command_basic(self, mock_is_file, mock_exists, mock_load, mock_analyze):
        """Test basic analyze command execution."""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.return_value = self.sample_questions
        mock_analyze.return_value = self.sample_report

        # Import and run main
        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('builtins.print') as mock_print:
                        exit_code = main()

        # Assertions
        self.assertEqual(exit_code, 0)
        mock_load.assert_called_once()
        mock_analyze.assert_called_once()

        # Check that summary was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Analysis complete" in str(call) for call in print_calls))

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl', '--dry-run'])
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_dry_run_flag(self, mock_is_file, mock_exists, mock_load):
        """Test --dry-run flag validates without processing."""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.return_value = self.sample_questions

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('cli.analyze_questions') as mock_analyze:
                        with patch('builtins.print') as mock_print:
                            exit_code = main()

        # Assertions
        self.assertEqual(exit_code, 0)
        mock_load.assert_called_once()
        mock_analyze.assert_not_called()  # Should not analyze in dry-run

        # Check validation message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Validation successful" in str(call) for call in print_calls))

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'nonexistent.jsonl'])
    @patch('pathlib.Path.exists')
    def test_file_not_found_error(self, mock_exists):
        """Test handling of missing input file."""
        mock_exists.return_value = False

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('builtins.print') as mock_print:
                        exit_code = main()

        # Should return error code
        self.assertEqual(exit_code, 1)

        # Check error message
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("not found" in str(call) for call in print_calls))

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl', '--output', '/readonly/results.json'])
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_permission_error(self, mock_is_file, mock_exists, mock_load):
        """Test handling of permission errors."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.return_value = self.sample_questions

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('pathlib.Path.mkdir') as mock_mkdir:
                        mock_mkdir.side_effect = PermissionError("Permission denied")
                        with patch('builtins.print') as mock_print:
                            exit_code = main()

        # Should return error code
        self.assertEqual(exit_code, 1)

        # Check error message
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Permission denied" in str(call) for call in print_calls))

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl', '--verbose'])
    @patch('cli.analyze_questions')
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_verbose_flag(self, mock_is_file, mock_exists, mock_load, mock_analyze):
        """Test --verbose flag enables progress display."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.return_value = self.sample_questions
        mock_analyze.return_value = self.sample_report

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('builtins.print'):
                        exit_code = main()

        # Check that analyze was called with show_progress=True
        self.assertEqual(exit_code, 0)
        mock_analyze.assert_called_once()
        call_kwargs = mock_analyze.call_args.kwargs
        self.assertTrue(call_kwargs['show_progress'])

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl', '--max-items', '1'])
    @patch('cli.analyze_questions')
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_max_items_flag(self, mock_is_file, mock_exists, mock_load, mock_analyze):
        """Test --max-items flag limits questions processed."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.return_value = self.sample_questions
        mock_analyze.return_value = self.sample_report

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('builtins.print'):
                        exit_code = main()

        # Check that analyze was called with limited questions
        self.assertEqual(exit_code, 0)
        mock_analyze.assert_called_once()
        questions_arg = mock_analyze.call_args.kwargs['questions']
        self.assertEqual(len(questions_arg), 1)  # Should be limited to 1

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl', '--models', 'gpt-4', 'claude'])
    @patch('cli.analyze_questions')
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_models_flag_placeholder(self, mock_is_file, mock_exists, mock_load, mock_analyze):
        """Test --models flag accepts values but shows not implemented."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.return_value = self.sample_questions
        mock_analyze.return_value = self.sample_report

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('builtins.print') as mock_print:
                        exit_code = main()

        # Should still succeed
        self.assertEqual(exit_code, 0)

        # Check for "not yet implemented" message
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Model selection not yet implemented" in str(call) for call in print_calls))

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl', '--time-limit', '60'])
    @patch('cli.analyze_questions')
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_time_limit_flag_placeholder(self, mock_is_file, mock_exists, mock_load, mock_analyze):
        """Test --time-limit flag accepts value but shows not implemented."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.return_value = self.sample_questions
        mock_analyze.return_value = self.sample_report

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('builtins.print') as mock_print:
                        exit_code = main()

        # Should still succeed
        self.assertEqual(exit_code, 0)

        # Check for "not yet implemented" message
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Time limit not yet implemented" in str(call) for call in print_calls))

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl'])
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_json_decode_error(self, mock_is_file, mock_exists, mock_load):
        """Test handling of invalid JSON in dataset."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.side_effect = json.JSONDecodeError("Invalid JSON", "test.jsonl", 10)

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('builtins.print') as mock_print:
                        exit_code = main()

        # Should return error code
        self.assertEqual(exit_code, 1)

        # Check error message
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Invalid JSON format" in str(call) for call in print_calls))

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl'])
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_value_error(self, mock_is_file, mock_exists, mock_load):
        """Test handling of invalid data format."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.side_effect = ValueError("Invalid answer format")

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('builtins.print') as mock_print:
                        exit_code = main()

        # Should return error code
        self.assertEqual(exit_code, 1)

        # Check error message
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Invalid data format" in str(call) for call in print_calls))

    @patch('sys.argv', ['cli.py', 'analyze', '--input', 'test.jsonl'])
    @patch('cli.analyze_questions')
    @patch('cli.load_dataset')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_keyboard_interrupt(self, mock_is_file, mock_exists, mock_load, mock_analyze):
        """Test graceful handling of keyboard interrupt."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_load.return_value = self.sample_questions
        mock_analyze.side_effect = KeyboardInterrupt()

        from cli import main

        with patch('cli.AppConfig.from_json') as mock_config:
            mock_config.return_value = MagicMock()
            with patch('cli.setup_logging') as mock_logging:
                mock_logging.return_value = MagicMock()
                with patch('cli.set_determinism'):
                    with patch('builtins.print') as mock_print:
                        exit_code = main()

        # Should return error code
        self.assertEqual(exit_code, 1)

        # Check interruption message
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("interrupted by user" in str(call) for call in print_calls))

    def test_exit_codes(self):
        """Test that correct exit codes are used."""
        test_cases = [
            # (scenario, expected_exit_code)
            ('success', 0),
            ('error', 1),
            # Exit code 2 (partial completion) is reserved for future stories
        ]

        for scenario, expected_code in test_cases:
            if scenario == 'success':
                # Already tested in test_analyze_command_basic
                pass
            elif scenario == 'error':
                # Already tested in test_file_not_found_error
                pass


if __name__ == '__main__':
    unittest.main()