"""Tests for the WMDP dataset adapter."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from robustcbrn.data.adapters import wmdp


class TestWMDPParquetConversion:
    """Test WMDP Parquet to JSONL conversion."""

    def test_convert_wmdp_parquet_basic(self, tmp_path):
        """Test basic conversion of WMDP Parquet format."""
        # Create test data
        test_data = pd.DataFrame({
            'question': [
                'What is the capital of France?',
                'Which element has atomic number 6?'
            ],
            'choices': [
                np.array(['London', 'Paris', 'Berlin', 'Madrid']),
                np.array(['Oxygen', 'Nitrogen', 'Carbon', 'Hydrogen'])
            ],
            'answer': [1, 2]  # 0-based indices
        })

        # Save as Parquet
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        parquet_file = raw_dir / "test.parquet"
        test_data.to_parquet(parquet_file)

        # Convert
        out_dir = tmp_path / "processed"
        result_path = wmdp.convert_wmdp_parquet_to_jsonl(raw_dir, out_dir)

        # Verify output
        assert result_path.exists()
        assert result_path.name == "eval.jsonl"

        # Load and check converted data
        with open(result_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Check first item
        item1 = json.loads(lines[0])
        assert item1['id'] == 'wmdp_00000'
        assert item1['question'] == 'What is the capital of France?'
        assert item1['choices'] == ['London', 'Paris', 'Berlin', 'Madrid']
        assert item1['answer'] == 'B'  # Index 1 -> B

        # Check second item
        item2 = json.loads(lines[1])
        assert item2['id'] == 'wmdp_00001'
        assert item2['question'] == 'Which element has atomic number 6?'
        assert item2['choices'] == ['Oxygen', 'Nitrogen', 'Carbon', 'Hydrogen']
        assert item2['answer'] == 'C'  # Index 2 -> C

    def test_convert_wmdp_parquet_with_metadata(self, tmp_path):
        """Test conversion with additional metadata fields."""
        # Create test data with extra columns
        test_data = pd.DataFrame({
            'question': ['Test question?'],
            'choices': [np.array(['A', 'B', 'C', 'D'])],
            'answer': [0],
            'topic': ['chemistry'],
            'difficulty': ['easy'],
            'source': ['textbook']
        })

        # Save as Parquet
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        parquet_file = raw_dir / "test.parquet"
        test_data.to_parquet(parquet_file)

        # Convert
        out_dir = tmp_path / "processed"
        result_path = wmdp.convert_wmdp_parquet_to_jsonl(raw_dir, out_dir)

        # Check metadata
        with open(result_path) as f:
            item = json.loads(f.readline())

        assert item['metadata']['topic'] == 'chemistry'
        assert item['metadata']['difficulty'] == 'easy'
        assert item['metadata']['source'] == 'textbook'

    def test_convert_no_parquet_file(self, tmp_path):
        """Test error when no Parquet file is found."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"

        with pytest.raises(ValueError, match="No Parquet files found"):
            wmdp.convert_wmdp_parquet_to_jsonl(raw_dir, out_dir)

    def test_pandas_not_installed(self, tmp_path):
        """Test error message when pandas is not installed."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"

        with patch.dict('sys.modules', {'pandas': None}), pytest.raises(ImportError, match="pandas is required"):
            wmdp.convert_wmdp_parquet_to_jsonl(raw_dir, out_dir)


class TestParseWMDPParquetRow:
    """Test parsing individual WMDP Parquet rows."""

    def test_parse_basic_row(self):
        """Test parsing a basic row."""
        row = pd.Series({
            'question': 'What is 2+2?',
            'choices': np.array(['3', '4', '5', '6']),
            'answer': 1
        })

        result = wmdp.parse_wmdp_parquet_row(row, 0)

        assert result['id'] == 'wmdp_00000'
        assert result['question'] == 'What is 2+2?'
        assert result['choices'] == ['3', '4', '5', '6']
        assert result['answer'] == 'B'

    def test_parse_row_with_numpy_array(self):
        """Test handling of numpy arrays for choices."""
        row = pd.Series({
            'question': 'Test?',
            'choices': np.array(['Option A', 'Option B']),
            'answer': 0
        })

        result = wmdp.parse_wmdp_parquet_row(row, 5)

        assert result['id'] == 'wmdp_00005'
        assert result['choices'] == ['Option A', 'Option B']
        assert result['answer'] == 'A'

    def test_parse_row_with_list(self):
        """Test handling of regular Python lists for choices."""
        row = pd.Series({
            'question': 'Test?',
            'choices': ['Option A', 'Option B', 'Option C'],
            'answer': 2
        })

        result = wmdp.parse_wmdp_parquet_row(row, 10)

        assert result['choices'] == ['Option A', 'Option B', 'Option C']
        assert result['answer'] == 'C'

    def test_parse_row_with_string_choices(self):
        """Test parsing choices from string representation."""
        row = pd.Series({
            'question': 'Test?',
            'choices': "['Option A', 'Option B']",  # String representation
            'answer': 1
        })

        result = wmdp.parse_wmdp_parquet_row(row, 0)

        assert result['choices'] == ['Option A', 'Option B']
        assert result['answer'] == 'B'

    def test_parse_row_with_pipe_separated(self):
        """Test fallback to pipe-separated choices."""
        row = pd.Series({
            'question': 'Test?',
            'choices': 'Option A|Option B|Option C',
            'answer': 0
        })

        result = wmdp.parse_wmdp_parquet_row(row, 0)

        assert result['choices'] == ['Option A', 'Option B', 'Option C']
        assert result['answer'] == 'A'

    def test_parse_row_answer_as_string(self):
        """Test handling answer as string."""
        row = pd.Series({
            'question': 'Test?',
            'choices': ['A', 'B', 'C'],
            'answer': '1'  # String instead of int
        })

        result = wmdp.parse_wmdp_parquet_row(row, 0)

        assert result['answer'] == 'B'

    def test_parse_row_large_index(self):
        """Test ID generation for large indices."""
        row = pd.Series({
            'question': 'Test?',
            'choices': ['A', 'B'],
            'answer': 0
        })

        result = wmdp.parse_wmdp_parquet_row(row, 99999)

        assert result['id'] == 'wmdp_99999'


class TestWMDPCSVConversion:
    """Test WMDP CSV to JSONL conversion."""

    def test_convert_wmdp_csv_basic(self, tmp_path):
        """Test basic CSV conversion."""
        import csv

        # Create test CSV
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        csv_file = raw_dir / "test.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['question', 'choice_a', 'choice_b', 'choice_c', 'choice_d', 'answer'])
            writer.writeheader()
            writer.writerow({
                'question': 'What is the capital?',
                'choice_a': 'London',
                'choice_b': 'Paris',
                'choice_c': 'Berlin',
                'choice_d': 'Madrid',
                'answer': 'B'
            })

        # Convert
        out_dir = tmp_path / "processed"
        result_path = wmdp.convert_wmdp_to_jsonl(raw_dir, out_dir)

        # Verify
        with open(result_path) as f:
            item = json.loads(f.readline())

        assert item['question'] == 'What is the capital?'
        assert item['choices'] == ['London', 'Paris', 'Berlin', 'Madrid']
        assert item['answer'] == 'B'

    def test_convert_no_csv_file(self, tmp_path):
        """Test error when no CSV file is found."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"

        with pytest.raises(ValueError, match="No CSV files found"):
            wmdp.convert_wmdp_to_jsonl(raw_dir, out_dir)


class TestExtractChoices:
    """Test choice extraction from various formats."""

    def test_extract_choices_separate_columns(self):
        """Test extracting from separate choice columns."""
        row = {
            'choice_a': 'Option A',
            'choice_b': 'Option B',
            'choice_c': 'Option C',
            'choice_d': 'Option D',
            'choice_e': '',  # Empty
            'choice_f': None  # None
        }

        choices = wmdp.extract_choices(row)
        assert choices == ['Option A', 'Option B', 'Option C', 'Option D']

    def test_extract_choices_json_list(self):
        """Test extracting from JSON list in choices column."""
        row = {
            'choices': '["First", "Second", "Third"]'
        }

        choices = wmdp.extract_choices(row)
        assert choices == ['First', 'Second', 'Third']

    def test_extract_choices_uppercase_columns(self):
        """Test extracting from uppercase letter columns."""
        row = {
            'A': 'Option A',
            'B': 'Option B',
            'C': 'Option C',
            'D': '',
            'question': 'Test?'  # Should be ignored
        }

        choices = wmdp.extract_choices(row)
        assert choices == ['Option A', 'Option B', 'Option C']

    def test_extract_choices_tab_separated(self):
        """Test extracting from tab-separated string."""
        row = {
            'choices': 'Option A\tOption B\tOption C'
        }

        choices = wmdp.extract_choices(row)
        assert choices == ['Option A', 'Option B', 'Option C']

    def test_extract_choices_pipe_separated(self):
        """Test extracting from pipe-separated string."""
        row = {
            'choices': 'Option A | Option B | Option C'
        }

        choices = wmdp.extract_choices(row)
        assert choices == ['Option A', 'Option B', 'Option C']

    def test_extract_choices_fallback(self):
        """Test fallback to searching for option/choice columns."""
        row = {
            'option1': 'First',
            'option2': 'Second',
            'choice_1': 'Third',
            'other': 'Ignored'
        }

        choices = wmdp.extract_choices(row)
        assert len(choices) == 3
        assert 'First' in choices
        assert 'Second' in choices
        assert 'Third' in choices

    def test_extract_choices_no_choices_found(self):
        """Test error when no choices can be extracted."""
        row = {
            'question': 'Test?',
            'answer': 'A'
        }

        with pytest.raises(ValueError, match="Could not extract choices"):
            wmdp.extract_choices(row)


class TestNormalizeAnswer:
    """Test answer normalization to letter format."""

    def test_normalize_already_uppercase(self):
        """Test when answer is already uppercase letter."""
        assert wmdp.normalize_answer('A', 4) == 'A'
        assert wmdp.normalize_answer('D', 4) == 'D'

    def test_normalize_lowercase(self):
        """Test converting lowercase to uppercase."""
        assert wmdp.normalize_answer('a', 4) == 'A'
        assert wmdp.normalize_answer('c', 4) == 'C'

    def test_normalize_zero_based_index(self):
        """Test converting 0-based index."""
        assert wmdp.normalize_answer('0', 4) == 'A'
        assert wmdp.normalize_answer('1', 4) == 'B'
        assert wmdp.normalize_answer('3', 4) == 'D'

    def test_normalize_one_based_index(self):
        """Test converting 1-based index."""
        assert wmdp.normalize_answer('1', 4) == 'B'  # 0-based: index 1 = B
        assert wmdp.normalize_answer('4', 4) == 'D'  # 1-based: 4th choice = D

    def test_normalize_integer(self):
        """Test converting integer answer."""
        assert wmdp.normalize_answer(0, 4) == 'A'
        assert wmdp.normalize_answer(2, 4) == 'C'

    def test_normalize_invalid(self):
        """Test error for invalid answer format."""
        with pytest.raises(ValueError, match="Could not normalize answer"):
            wmdp.normalize_answer('invalid', 4)

        with pytest.raises(ValueError, match="Could not normalize answer"):
            wmdp.normalize_answer('10', 4)  # Out of range


class TestParseWMDPRow:
    """Test parsing CSV rows."""

    def test_parse_csv_row_basic(self):
        """Test parsing a basic CSV row."""
        row = {
            'question': 'What is the answer?',
            'choice_a': 'Option A',
            'choice_b': 'Option B',
            'choice_c': 'Option C',
            'choice_d': 'Option D',
            'answer': 'B',
            'topic': 'science'
        }

        result = wmdp.parse_wmdp_row(row, 1)

        assert result['id'] == 'wmdp_00001'
        assert result['question'] == 'What is the answer?'
        assert result['choices'] == ['Option A', 'Option B', 'Option C', 'Option D']
        assert result['answer'] == 'B'
        assert result['metadata']['topic'] == 'science'

    def test_parse_csv_row_alternative_columns(self):
        """Test parsing with alternative column names."""
        row = {
            'prompt': 'Question text',  # Alternative to 'question'
            'A': 'First',
            'B': 'Second',
            'C': 'Third',
            'D': 'Fourth',
            'correct': '2',  # Alternative to 'answer', 0-based
            'category': 'test'
        }

        result = wmdp.parse_wmdp_row(row, 10)

        assert result['id'] == 'wmdp_00010'
        assert result['question'] == 'Question text'
        assert result['choices'] == ['First', 'Second', 'Third', 'Fourth']
        assert result['answer'] == 'C'  # Index 2 -> C
        assert result['metadata']['category'] == 'test'

    def test_parse_csv_row_minimal(self):
        """Test parsing with minimal required fields."""
        row = {
            'text': 'Question?',  # Another alternative
            'choices': '["A", "B"]',
            'answer': '0'
        }

        result = wmdp.parse_wmdp_row(row, 0)

        assert result['id'] == 'wmdp_00000'
        assert result['question'] == 'Question?'
        assert result['choices'] == ['A', 'B']
        assert result['answer'] == 'A'
        assert result['metadata'] == {}
