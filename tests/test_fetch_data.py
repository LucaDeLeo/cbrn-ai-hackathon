"""Tests for the fetch_data.py script."""

import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

# Add scripts directory to path to import fetch_data
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import fetch_data


class TestChecksumFunctions:
    """Test checksum computation and verification."""

    def test_sha256_file(self, tmp_path):
        """Test SHA256 computation for a file."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        expected_hash = hashlib.sha256(test_content).hexdigest()
        actual_hash = fetch_data.sha256_file(test_file)

        assert actual_hash == expected_hash

    def test_sha256_file_large(self, tmp_path):
        """Test SHA256 computation for a large file (chunked reading)."""
        test_file = tmp_path / "large.bin"
        # Create a 5MB file
        test_content = b"x" * (5 * 1024 * 1024)
        test_file.write_bytes(test_content)

        expected_hash = hashlib.sha256(test_content).hexdigest()
        actual_hash = fetch_data.sha256_file(test_file)

        assert actual_hash == expected_hash


class TestDownloadFunctions:
    """Test download functionality."""

    @patch('urllib.request.urlopen')
    def test_download_with_progress(self, mock_urlopen, tmp_path):
        """Test file download with progress reporting."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.side_effect = [b"test", b"data", b""]
        mock_response.headers = {"Content-Length": "8"}
        mock_urlopen.return_value.__enter__.return_value = mock_response

        target = tmp_path / "download.txt"
        fetch_data.download_with_progress("http://example.com/file", target)

        assert target.exists()
        assert target.read_bytes() == b"testdata"

    @patch('urllib.request.urlopen')
    def test_download_with_hf_token(self, mock_urlopen, tmp_path, monkeypatch):
        """Test download with HuggingFace token authentication."""
        monkeypatch.setenv("HF_TOKEN", "test_token")

        mock_response = MagicMock()
        mock_response.read.return_value = b"data"
        mock_response.headers = {}
        mock_urlopen.return_value.__enter__.return_value = mock_response

        target = tmp_path / "download.txt"
        fetch_data.download_with_progress("https://huggingface.co/test", target)

        # Check that authorization header was set
        call_args = mock_urlopen.call_args[0][0]
        assert call_args.headers.get("Authorization") == "Bearer test_token"

    @patch('urllib.request.urlopen')
    def test_download_authentication_error(self, mock_urlopen, tmp_path):
        """Test handling of authentication errors."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 401, "Unauthorized", {}, None
        )

        target = tmp_path / "download.txt"
        with pytest.raises(SystemExit) as exc_info:
            fetch_data.download_with_progress("https://huggingface.co/test", target)

        assert exc_info.value.code == 1


class TestUnpackFunctions:
    """Test archive unpacking functionality."""

    def test_unpack_tar_gz(self, tmp_path):
        """Test unpacking tar.gz archives."""
        import tarfile

        # Create a tar.gz file
        archive = tmp_path / "test.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            # Add a test file
            test_file = tmp_path / "test.txt"
            test_file.write_text("test content")
            tar.add(test_file, arcname="test.txt")

        # Unpack
        out_dir = tmp_path / "output"
        fetch_data.unpack_archive(archive, "tar.gz", out_dir)

        assert (out_dir / "test.txt").exists()
        assert (out_dir / "test.txt").read_text() == "test content"

    def test_unpack_zip(self, tmp_path):
        """Test unpacking zip archives."""
        # Create a zip file
        archive = tmp_path / "test.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("test.txt", "test content")

        # Unpack
        out_dir = tmp_path / "output"
        fetch_data.unpack_archive(archive, "zip", out_dir)

        assert (out_dir / "test.txt").exists()
        assert (out_dir / "test.txt").read_text() == "test content"

    def test_unpack_none_parquet(self, tmp_path):
        """Test handling of files that don't need unpacking (e.g., Parquet)."""
        # Create a file with .download extension
        archive = tmp_path / "dataset.download"
        archive.write_text("parquet data")

        # Unpack (should just rename and move)
        out_dir = tmp_path / "output"
        fetch_data.unpack_archive(archive, "none", out_dir)

        # Should be renamed to .parquet
        assert (out_dir / "dataset.parquet").exists()
        assert (out_dir / "dataset.parquet").read_text() == "parquet data"

    def test_unpack_unknown_format(self, tmp_path):
        """Test error handling for unknown archive formats."""
        archive = tmp_path / "test.unknown"
        archive.write_text("data")
        out_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="Unknown unpack kind"):
            fetch_data.unpack_archive(archive, "unknown", out_dir)


class TestRegistryFunctions:
    """Test registry loading and management."""

    def test_load_registry(self, tmp_path, monkeypatch):
        """Test loading dataset registry."""
        # Create a test registry
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "test_dataset": {
                    "url": "http://example.com/data.csv",
                    "sha256": "abc123",
                    "license": "MIT",
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))

        # Monkeypatch the REGISTRY path
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        datasets = fetch_data.load_registry()
        assert "test_dataset" in datasets
        assert datasets["test_dataset"]["url"] == "http://example.com/data.csv"

    def test_load_registry_missing(self, tmp_path, monkeypatch):
        """Test error when registry is missing."""
        monkeypatch.setattr(fetch_data, "REGISTRY", tmp_path / "missing.yaml")

        with pytest.raises(SystemExit) as exc_info:
            fetch_data.load_registry()

        assert exc_info.value.code == 1

    def test_compute_and_update_checksum(self, tmp_path, monkeypatch):
        """Test computing and updating placeholder checksums."""
        # Create a test file
        test_file = tmp_path / "data.parquet"
        test_file.write_bytes(b"test data")

        # Create registry with placeholder
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "test_dataset": {
                    "url": "http://example.com/data.parquet",
                    "sha256": "PLACEHOLDER_SHA256_TO_BE_COMPUTED",
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))

        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        # Compute and update
        checksum = fetch_data.compute_and_update_checksum("test_dataset", test_file)

        # Verify checksum was computed
        assert checksum == hashlib.sha256(b"test data").hexdigest()

        # Verify registry was updated
        with open(registry) as f:
            updated_data = yaml.safe_load(f)
        assert updated_data["datasets"]["test_dataset"]["sha256"] == checksum


class TestFetchDataset:
    """Test the main fetch_dataset function."""

    @patch('fetch_data.download_with_progress')
    @patch('fetch_data.unpack_archive')
    def test_fetch_dataset_success(self, mock_unpack, mock_download, tmp_path, monkeypatch):
        """Test successful dataset fetching."""
        # Setup paths
        monkeypatch.setattr(fetch_data, "RAW", tmp_path / "raw")

        # Create registry
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "test_dataset": {
                    "url": "http://example.com/data.parquet",
                    "sha256": hashlib.sha256(b"test data").hexdigest(),
                    "unpack": "none",
                    "expected_files": []
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        # Mock download to create the file
        def create_file(url, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test data")

        mock_download.side_effect = create_file

        # Fetch dataset
        result = fetch_data.fetch_dataset("test_dataset")

        assert result == tmp_path / "raw" / "test_dataset"
        mock_download.assert_called_once()
        mock_unpack.assert_called_once()

    def test_fetch_dataset_already_exists(self, tmp_path, monkeypatch):
        """Test fetching when dataset already exists."""
        # Setup
        raw_dir = tmp_path / "raw"
        dataset_dir = raw_dir / "test_dataset"
        dataset_dir.mkdir(parents=True)

        monkeypatch.setattr(fetch_data, "RAW", raw_dir)

        # Create registry
        registry = tmp_path / "registry.yaml"
        registry_data = {"datasets": {"test_dataset": {"url": "http://example.com"}}}
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        # Should not re-download
        result = fetch_data.fetch_dataset("test_dataset", force=False)
        assert result == dataset_dir

    def test_fetch_dataset_checksum_mismatch(self, tmp_path, monkeypatch):
        """Test error on checksum mismatch."""
        monkeypatch.setattr(fetch_data, "RAW", tmp_path / "raw")

        # Create registry with wrong checksum
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "test_dataset": {
                    "url": "http://example.com/data.parquet",
                    "sha256": "wrong_checksum",
                    "unpack": "none"
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        with patch('fetch_data.download_with_progress') as mock_download:
            def create_file(url, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test data")

            mock_download.side_effect = create_file

            with pytest.raises(SystemExit) as exc_info:
                fetch_data.fetch_dataset("test_dataset")

            assert exc_info.value.code == 1

    def test_fetch_dataset_missing_expected_files(self, tmp_path, monkeypatch):
        """Test error when expected files are missing."""
        raw_dir = tmp_path / "raw"
        monkeypatch.setattr(fetch_data, "RAW", raw_dir)

        # Create registry with expected files
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "test_dataset": {
                    "url": "http://example.com/data.tar.gz",
                    "sha256": hashlib.sha256(b"test").hexdigest(),
                    "unpack": "tar.gz",
                    "expected_files": ["data/file1.txt", "data/file2.txt"]
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        with patch('fetch_data.download_with_progress') as mock_download:
            with patch('fetch_data.unpack_archive') as mock_unpack:
                def create_file(url, path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"test")

                mock_download.side_effect = create_file

                # Unpack creates directory but not expected files
                def create_dir(archive, kind, out_dir):
                    out_dir.mkdir(parents=True, exist_ok=True)

                mock_unpack.side_effect = create_dir

                with pytest.raises(SystemExit) as exc_info:
                    fetch_data.fetch_dataset("test_dataset")

                assert exc_info.value.code == 1


class TestProcessDataset:
    """Test dataset processing with adapters."""

    def test_process_dataset_with_adapter(self, tmp_path, monkeypatch):
        """Test processing dataset with configured adapter."""
        # Setup paths
        proc_dir = tmp_path / "processed"
        raw_dir = tmp_path / "raw" / "test_dataset"
        raw_dir.mkdir(parents=True)

        monkeypatch.setattr(fetch_data, "PROC", proc_dir)

        # Create registry with adapter
        registry_data = {
            "datasets": {
                "test_dataset": {
                    "process": {
                        "adapter": "test_module:convert_func"
                    }
                }
            }
        }

        # Mock the adapter function
        mock_module = MagicMock()
        mock_convert = MagicMock(return_value=proc_dir / "test_dataset" / "eval.jsonl")
        mock_module.convert_func = mock_convert

        with patch('fetch_data.load_registry', return_value=registry_data["datasets"]):
            with patch('importlib.import_module', return_value=mock_module):
                result = fetch_data.process_dataset("test_dataset", raw_dir)

        assert result == proc_dir / "test_dataset" / "eval.jsonl"
        mock_convert.assert_called_once_with(raw_dir, proc_dir / "test_dataset")

    def test_process_dataset_no_adapter(self, tmp_path, monkeypatch):
        """Test processing when no adapter is configured."""
        raw_dir = tmp_path / "raw" / "test_dataset"

        registry_data = {
            "datasets": {
                "test_dataset": {
                    "url": "http://example.com"
                    # No process/adapter configured
                }
            }
        }

        with patch('fetch_data.load_registry', return_value=registry_data["datasets"]):
            result = fetch_data.process_dataset("test_dataset", raw_dir)

        assert result is None

    def test_process_dataset_adapter_import_error(self, tmp_path, monkeypatch):
        """Test handling of adapter import errors."""
        proc_dir = tmp_path / "processed"
        raw_dir = tmp_path / "raw" / "test_dataset"

        monkeypatch.setattr(fetch_data, "PROC", proc_dir)

        registry_data = {
            "datasets": {
                "test_dataset": {
                    "process": {
                        "adapter": "nonexistent_module:func"
                    }
                }
            }
        }

        with patch('fetch_data.load_registry', return_value=registry_data["datasets"]):
            result = fetch_data.process_dataset("test_dataset", raw_dir)

        # Should return None on import error
        assert result is None

    def test_process_dataset_adapter_exception(self, tmp_path, monkeypatch):
        """Test handling of adapter execution errors."""
        proc_dir = tmp_path / "processed"
        raw_dir = tmp_path / "raw" / "test_dataset"

        monkeypatch.setattr(fetch_data, "PROC", proc_dir)

        registry_data = {
            "datasets": {
                "test_dataset": {
                    "process": {
                        "adapter": "test_module:convert_func"
                    }
                }
            }
        }

        # Mock adapter that raises exception
        mock_module = MagicMock()
        mock_module.convert_func.side_effect = Exception("Processing error")

        with patch('fetch_data.load_registry', return_value=registry_data["datasets"]):
            with patch('importlib.import_module', return_value=mock_module):
                result = fetch_data.process_dataset("test_dataset", raw_dir)

        # Should return None on processing error
        assert result is None


class TestMainCLI:
    """Test the main CLI interface."""

    def test_main_list_datasets(self, capsys, tmp_path, monkeypatch):
        """Test listing available datasets."""
        # Create registry
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "dataset1": {"notes": "First dataset"},
                "dataset2": {"notes": "Second dataset"},
                "dataset3": {}  # No notes
            }
        }
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        # Test --list
        with patch('sys.argv', ['fetch_data.py', '--list']):
            fetch_data.main()

        captured = capsys.readouterr()
        assert "dataset1" in captured.out
        assert "First dataset" in captured.out
        assert "dataset2" in captured.out
        assert "dataset3" in captured.out

    def test_main_compute_checksum(self, capsys, tmp_path):
        """Test computing checksum for a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with patch('sys.argv', ['fetch_data.py', '--compute-checksum', str(test_file)]):
            fetch_data.main()

        captured = capsys.readouterr()
        expected_hash = hashlib.sha256(b"test content").hexdigest()
        assert expected_hash in captured.out

    def test_main_fetch_and_process(self, capsys, tmp_path, monkeypatch):
        """Test fetching and processing a dataset."""
        # Setup
        monkeypatch.setattr(fetch_data, "RAW", tmp_path / "raw")
        monkeypatch.setattr(fetch_data, "PROC", tmp_path / "processed")

        # Create registry
        registry = tmp_path / "registry.yaml"
        registry_data = {
            "datasets": {
                "test_dataset": {
                    "url": "http://example.com/data.csv",
                    "sha256": "abc123",
                }
            }
        }
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        with patch('fetch_data.fetch_dataset') as mock_fetch:
            with patch('fetch_data.process_dataset') as mock_process:
                mock_fetch.return_value = tmp_path / "raw" / "test_dataset"
                mock_process.return_value = tmp_path / "processed" / "test_dataset" / "eval.jsonl"

                with patch('sys.argv', ['fetch_data.py', 'test_dataset']):
                    fetch_data.main()

        captured = capsys.readouterr()
        assert "Result:" in captured.out
        assert "test_dataset" in captured.out

    def test_main_unknown_dataset(self, capsys, tmp_path, monkeypatch):
        """Test error for unknown dataset."""
        registry = tmp_path / "registry.yaml"
        registry_data = {"datasets": {}}
        registry.write_text(yaml.dump(registry_data))
        monkeypatch.setattr(fetch_data, "REGISTRY", registry)

        with patch('sys.argv', ['fetch_data.py', 'unknown']):
            with pytest.raises(SystemExit):
                fetch_data.main()

        captured = capsys.readouterr()
        assert "Unknown dataset: unknown" in captured.out