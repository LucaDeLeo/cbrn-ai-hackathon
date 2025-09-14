"""Integration tests for the complete data pipeline."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestDataPipelineIntegration:
    """Integration tests for the full data download and processing pipeline."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure."""
        # Create directory structure
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "raw").mkdir()
        (tmp_path / "data" / "processed").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "robustcbrn" / "data" / "adapters").mkdir(parents=True)

        # Copy fetch_data.py
        src_fetch = Path(__file__).parent.parent / "scripts" / "fetch_data.py"
        if src_fetch.exists():
            shutil.copy(src_fetch, tmp_path / "scripts" / "fetch_data.py")

        # Copy WMDP adapter
        src_adapter = Path(__file__).parent.parent / "robustcbrn" / "data" / "adapters" / "wmdp.py"
        if src_adapter.exists():
            shutil.copy(src_adapter, tmp_path / "robustcbrn" / "data" / "adapters" / "wmdp.py")
            # Also copy __init__.py
            (tmp_path / "robustcbrn" / "__init__.py").touch()
            (tmp_path / "robustcbrn" / "data" / "__init__.py").touch()
            (tmp_path / "robustcbrn" / "data" / "adapters" / "__init__.py").touch()

        return tmp_path

    def test_end_to_end_parquet_dataset(self, temp_project):
        """Test complete pipeline for a Parquet dataset."""
        # Create mock Parquet data
        test_data = pd.DataFrame({
            'question': [
                'What is Python?',
                'What is machine learning?',
                'What is data science?'
            ],
            'choices': [
                np.array(['A language', 'A snake', 'A framework', 'A database']),
                np.array(['AI subset', 'Database', 'Web framework', 'OS']),
                np.array(['Field of study', 'Programming language', 'Database', 'Tool'])
            ],
            'answer': [0, 0, 0]
        })

        # Save as Parquet
        parquet_file = temp_project / "test_data.parquet"
        test_data.to_parquet(parquet_file)

        # Create registry
        registry = temp_project / "data" / "registry.yaml"
        registry_data = {
            "datasets": {
                "test_dataset": {
                    "url": f"file://{parquet_file}",
                    "sha256": hashlib.sha256(parquet_file.read_bytes()).hexdigest(),
                    "license": "MIT",
                    "unpack": "none",
                    "process": {
                        "adapter": "robustcbrn.data.adapters.wmdp:convert_wmdp_parquet_to_jsonl"
                    },
                    "safe_to_publish": False,
                    "notes": "Test dataset"
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))

        # Run fetch_data.py
        os.chdir(temp_project)
        sys.path.insert(0, str(temp_project))

        import fetch_data
        # Patch paths
        fetch_data.ROOT = temp_project
        fetch_data.DATA = temp_project / "data"
        fetch_data.RAW = temp_project / "data" / "raw"
        fetch_data.PROC = temp_project / "data" / "processed"
        fetch_data.REGISTRY = registry

        # Mock download to copy local file
        def mock_download(url, target):
            if url.startswith("file://"):
                src = Path(url[7:])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, target)
            else:
                raise ValueError(f"Unsupported URL: {url}")

        with patch.object(fetch_data, 'download_with_progress', side_effect=mock_download):
            # Fetch and process
            raw_dir = fetch_data.fetch_dataset("test_dataset")
            processed_path = fetch_data.process_dataset("test_dataset", raw_dir)

        # Verify results
        assert processed_path is not None
        assert processed_path.exists()

        # Check JSONL output
        with open(processed_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify first item
        item = json.loads(lines[0])
        assert item['id'] == 'wmdp_00000'
        assert item['question'] == 'What is Python?'
        assert item['choices'] == ['A language', 'A snake', 'A framework', 'A database']
        assert item['answer'] == 'A'

    def test_makefile_integration(self, temp_project, monkeypatch):
        """Test Makefile targets for data management."""
        # Create a simple Makefile
        makefile_content = """
SHELL := /bin/bash

.PHONY: data data-list

data:
	python scripts/fetch_data.py $(DATASET)

data-list:
	python scripts/fetch_data.py --list
"""
        (temp_project / "Makefile").write_text(makefile_content)

        # Create registry
        registry = temp_project / "data" / "registry.yaml"
        registry_data = {
            "datasets": {
                "dataset1": {"notes": "Test dataset 1"},
                "dataset2": {"notes": "Test dataset 2"}
            }
        }
        registry.write_text(yaml.dump(registry_data))

        # Test data-list target
        os.chdir(temp_project)
        result = subprocess.run(['make', 'data-list'], capture_output=True, text=True)

        assert result.returncode == 0
        assert "dataset1" in result.stdout
        assert "dataset2" in result.stdout

    def test_schema_validation(self, temp_project):
        """Test that processed data matches expected schema."""
        from robustcbrn.data.schemas import Question

        # Create test data
        test_data = pd.DataFrame({
            'question': ['Q1', 'Q2'],
            'choices': [
                np.array(['A', 'B', 'C', 'D']),
                np.array(['W', 'X', 'Y', 'Z'])
            ],
            'answer': [0, 3]
        })

        # Process through adapter
        raw_dir = temp_project / "data" / "raw" / "test"
        raw_dir.mkdir(parents=True)
        parquet_file = raw_dir / "test.parquet"
        test_data.to_parquet(parquet_file)

        out_dir = temp_project / "data" / "processed" / "test"

        # Import and run adapter
        sys.path.insert(0, str(temp_project))
        from robustcbrn.data.adapters import wmdp

        result_path = wmdp.convert_wmdp_parquet_to_jsonl(raw_dir, out_dir)

        # Validate schema
        with open(result_path) as f:
            for line in f:
                item = json.loads(line)

                # Check required fields
                assert 'id' in item
                assert 'question' in item
                assert 'choices' in item
                assert 'answer' in item

                # Validate types
                assert isinstance(item['id'], str)
                assert isinstance(item['question'], str)
                assert isinstance(item['choices'], list)
                assert isinstance(item['answer'], str)
                assert item['answer'] in ['A', 'B', 'C', 'D', 'E', 'F']

                # Create Question object to validate schema compatibility
                q = Question(
                    id=item['id'],
                    question=item['question'],
                    choices=item['choices'],
                    answer=item['answer']
                )
                assert q.id == item['id']


class TestDataPipelineErrorHandling:
    """Test error handling in the data pipeline."""

    def test_network_error_handling(self, tmp_path, monkeypatch):
        """Test handling of network errors during download."""
        import fetch_data

        # Setup
        monkeypatch.setattr(fetch_data, "RAW", tmp_path / "raw")
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "test": {
                    "url": "http://invalid.example.com/data.csv",
                    "sha256": "abc123"
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        # Test network error
        with patch('urllib.request.urlopen') as mock_urlopen:
            import urllib.error
            mock_urlopen.side_effect = urllib.error.URLError("Network error")

            with pytest.raises(urllib.error.URLError):
                fetch_data.fetch_dataset("test")

    def test_checksum_mismatch_recovery(self, tmp_path, monkeypatch):
        """Test recovery from checksum mismatches."""
        import fetch_data

        monkeypatch.setattr(fetch_data, "RAW", tmp_path / "raw")

        # Create registry with specific checksum
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "test": {
                    "url": "http://example.com/data.csv",
                    "sha256": "expected_checksum",
                    "unpack": "none"
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        # Mock download that creates wrong data
        def mock_download(url, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wrong data")

        with patch.object(fetch_data, 'download_with_progress', side_effect=mock_download):
            with pytest.raises(SystemExit) as exc_info:
                fetch_data.fetch_dataset("test")

            assert exc_info.value.code == 1

        # Verify file was cleaned up
        assert not (tmp_path / "raw" / "test.download").exists()

    def test_adapter_failure_handling(self, tmp_path):
        """Test handling of adapter failures."""
        import fetch_data

        # Create a dataset directory
        raw_dir = tmp_path / "raw" / "test"
        raw_dir.mkdir(parents=True)
        (raw_dir / "data.csv").write_text("test")

        # Registry with failing adapter
        registry_data = {
            "datasets": {
                "test": {
                    "process": {
                        "adapter": "nonexistent:function"
                    }
                }
            }
        }

        with patch.object(fetch_data, 'load_registry', return_value=registry_data["datasets"]):
            result = fetch_data.process_dataset("test", raw_dir)

        # Should return None on adapter failure
        assert result is None


class TestCLIIntegration:
    """Test CLI integration and command-line usage."""

    def test_cli_list_datasets(self, tmp_path):
        """Test CLI list command."""
        # Create fetch_data.py in temp location
        fetch_script = tmp_path / "fetch_data.py"
        fetch_script.write_text("""
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.fetch_data import main
if __name__ == "__main__":
    main()
""")

        # Create registry
        registry = tmp_path / "data" / "registry.yaml"
        registry.mkdir(parents=True)
        registry_data = {
            "datasets": {
                "test1": {"notes": "Dataset 1"},
                "test2": {"notes": "Dataset 2"}
            }
        }
        registry.write_text(yaml.dump(registry_data))

        # Run CLI
        result = subprocess.run(
            [sys.executable, str(fetch_script), "--list"],
            cwd=tmp_path,
            capture_output=True,
            text=True
        )

        assert "test1" in result.stdout
        assert "test2" in result.stdout

    def test_cli_compute_checksum(self, tmp_path):
        """Test CLI checksum computation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Create minimal fetch_data.py
        fetch_script = tmp_path / "fetch_data.py"
        src_fetch = Path(__file__).parent.parent / "scripts" / "fetch_data.py"
        if src_fetch.exists():
            shutil.copy(src_fetch, tmp_path / "fetch_data.py")

        # Run CLI
        result = subprocess.run(
            [sys.executable, str(fetch_script), "--compute-checksum", str(test_file)],
            cwd=tmp_path,
            capture_output=True,
            text=True
        )

        expected_hash = hashlib.sha256(b"test content").hexdigest()
        assert expected_hash in result.stdout

    def test_cli_force_redownload(self, tmp_path, monkeypatch):
        """Test force re-download option."""
        import fetch_data

        # Setup
        raw_dir = tmp_path / "raw"
        dataset_dir = raw_dir / "test"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "existing.txt").write_text("old data")

        monkeypatch.setattr(fetch_data, "RAW", raw_dir)

        # Create registry
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "test": {
                    "url": "http://example.com/data.csv",
                    "sha256": hashlib.sha256(b"new data").hexdigest(),
                    "unpack": "none"
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        # Mock download
        download_called = False

        def mock_download(url, path):
            nonlocal download_called
            download_called = True
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"new data")

        with patch.object(fetch_data, 'download_with_progress', side_effect=mock_download):
            # Without force - should not re-download
            fetch_data.fetch_dataset("test", force=False)
            assert not download_called

            # With force - should re-download
            fetch_data.fetch_dataset("test", force=True)
            assert download_called


class TestDataValidation:
    """Test data validation after processing."""

    def test_validate_jsonl_format(self, tmp_path):
        """Test validation of JSONL output format."""
        # Create test JSONL
        jsonl_file = tmp_path / "test.jsonl"
        valid_data = [
            {"id": "1", "question": "Q1", "choices": ["A", "B"], "answer": "A"},
            {"id": "2", "question": "Q2", "choices": ["C", "D"], "answer": "B"}
        ]

        with open(jsonl_file, 'w') as f:
            for item in valid_data:
                f.write(json.dumps(item) + "\n")

        # Validate
        with open(jsonl_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    assert 'id' in item
                    assert 'question' in item
                    assert 'choices' in item
                    assert 'answer' in item
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON at line {line_num}: {e}")

    def test_validate_answer_format(self, tmp_path):
        """Test validation of answer format (must be letters)."""
        from robustcbrn.data.adapters import wmdp

        # Test valid answers
        assert wmdp.normalize_answer(0, 4) == 'A'
        assert wmdp.normalize_answer(1, 4) == 'B'
        assert wmdp.normalize_answer('C', 4) == 'C'

        # Test invalid answers
        with pytest.raises(ValueError):
            wmdp.normalize_answer('invalid', 4)

    def test_validate_choices_format(self, tmp_path):
        """Test validation of choices format (must be list of strings)."""
        from robustcbrn.data.adapters import wmdp

        # Valid choices
        row = {'choices': ['A', 'B', 'C']}
        choices = wmdp.extract_choices(row)
        assert isinstance(choices, list)
        assert all(isinstance(c, str) for c in choices)

        # Invalid - no choices
        row = {'question': 'Q'}
        with pytest.raises(ValueError, match="Could not extract choices"):
            wmdp.extract_choices(row)
