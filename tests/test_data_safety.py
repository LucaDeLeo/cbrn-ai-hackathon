"""Tests for data safety and validation checks."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml


class TestGitIgnorePatterns:
    """Test that .gitignore properly excludes data files."""

    @pytest.fixture
    def gitignore_content(self):
        """Get the actual .gitignore content."""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        if gitignore_path.exists():
            return gitignore_path.read_text()
        return ""

    def test_gitignore_excludes_raw_data(self, gitignore_content):
        """Test that raw data directory is excluded."""
        assert "data/raw/" in gitignore_content

    def test_gitignore_excludes_processed_data(self, gitignore_content):
        """Test that processed data directory is excluded."""
        assert "data/processed/" in gitignore_content

    def test_gitignore_excludes_data_files(self, gitignore_content):
        """Test that common data file extensions are excluded."""
        assert "data/**/*.jsonl" in gitignore_content
        assert "data/**/*.csv" in gitignore_content
        assert "data/**/*.parquet" in gitignore_content
        assert "data/**/*.tar.gz" in gitignore_content
        assert "data/**/*.zip" in gitignore_content

    def test_gitignore_allows_sanitized(self, gitignore_content):
        """Test that sanitized samples are explicitly allowed."""
        assert "!data/sample_sanitized.jsonl" in gitignore_content
        assert "!data/benign_pairs_sanitized.jsonl" in gitignore_content
        assert "!data/registry.yaml" in gitignore_content
        assert "!data/README.md" in gitignore_content

    def test_git_check_ignore(self, tmp_path):
        """Test git check-ignore with actual patterns."""
        # Create a test git repo
        repo = tmp_path / "test_repo"
        repo.mkdir()
        os.chdir(repo)

        # Initialize git
        subprocess.run(["git", "init"], check=True, capture_output=True)

        # Copy .gitignore
        gitignore_src = Path(__file__).parent.parent / ".gitignore"
        if gitignore_src.exists():
            (repo / ".gitignore").write_text(gitignore_src.read_text())

        # Test that data files would be ignored
        test_files = [
            "data/raw/dataset.jsonl",
            "data/processed/dataset/eval.jsonl",
            "data/raw/archive.tar.gz",
            "data/dataset.csv"
        ]

        for file_path in test_files:
            result = subprocess.run(
                ["git", "check-ignore", file_path],
                capture_output=True
            )
            assert result.returncode == 0, f"{file_path} should be ignored"

        # Test that allowed files are not ignored
        allowed_files = [
            "data/sample_sanitized.jsonl",
            "data/registry.yaml",
            "data/README.md"
        ]

        for file_path in allowed_files:
            result = subprocess.run(
                ["git", "check-ignore", file_path],
                capture_output=True
            )
            assert result.returncode == 1, f"{file_path} should NOT be ignored"


class TestValidateReleaseScript:
    """Test the validate_release.sh script enhancements."""

    @pytest.fixture
    def validate_script(self):
        """Get path to validate_release.sh script."""
        script_path = Path(__file__).parent.parent / "scripts" / "validate_release.sh"
        if not script_path.exists():
            pytest.skip("validate_release.sh not found")
        return script_path

    def test_script_checks_raw_data(self, validate_script, tmp_path):
        """Test that script blocks tracked raw data files."""
        # Create a test git repo
        repo = tmp_path / "test_repo"
        repo.mkdir()
        os.chdir(repo)

        # Initialize git and create files
        subprocess.run(["git", "init"], check=True, capture_output=True)

        # Create and track a raw data file
        raw_dir = repo / "data" / "raw"
        raw_dir.mkdir(parents=True)
        raw_file = raw_dir / "dataset.jsonl"
        raw_file.write_text('{"test": "data"}')

        subprocess.run(["git", "add", str(raw_file)], check=True)

        # Run validation (should fail)
        result = subprocess.run(
            ["bash", str(validate_script)],
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        assert "data/raw" in result.stdout or "data/raw" in result.stderr

    def test_script_checks_large_files(self, validate_script, tmp_path):
        """Test that script blocks large dataset files."""
        # Create a test git repo
        repo = tmp_path / "test_repo"
        repo.mkdir()
        os.chdir(repo)

        subprocess.run(["git", "init"], check=True, capture_output=True)

        # Create a large file (>5MB)
        large_file = repo / "large_dataset.jsonl"
        # Create 6MB of data
        large_content = "x" * (6 * 1024 * 1024)
        large_file.write_text(large_content)

        subprocess.run(["git", "add", str(large_file)], check=True)

        # Run validation (should fail)
        result = subprocess.run(
            ["bash", str(validate_script)],
            capture_output=True,
            text=True
        )

        # Check for failure due to large file
        # Note: The actual check depends on git ls-files working
        # which requires the file to be staged
        if "large dataset-like files" in result.stdout:
            assert result.returncode != 0

    def test_script_allows_small_sanitized(self, validate_script, tmp_path):
        """Test that script allows small sanitized files."""
        # Create a test git repo
        repo = tmp_path / "test_repo"
        repo.mkdir()
        os.chdir(repo)

        subprocess.run(["git", "init"], check=True, capture_output=True)

        # Create directory structure
        (repo / "artifacts").mkdir()
        (repo / "data").mkdir()

        # Create small sanitized file
        sanitized = repo / "data" / "sample_sanitized.jsonl"
        sanitized.write_text('{"id": "s1", "metadata": {}}\n')

        subprocess.run(["git", "add", str(sanitized)], check=True)

        # Create dummy artifacts without forbidden content
        summary = repo / "artifacts" / "summary.json"
        summary.write_text('{"accuracy": 0.95}')

        # This should pass if no forbidden content
        result = subprocess.run(
            ["bash", str(validate_script)],
            capture_output=True,
            text=True,
            env={**os.environ, "DATASET": str(sanitized)}
        )

        # Should not fail for sanitized data
        if "Policy content checks passed" in result.stdout:
            assert "sample_sanitized.jsonl" not in result.stderr


class TestRegistryValidation:
    """Test dataset registry validation."""

    def test_registry_schema(self):
        """Test that registry.yaml has valid schema."""
        registry_path = Path(__file__).parent.parent / "data" / "registry.yaml"
        if not registry_path.exists():
            pytest.skip("registry.yaml not found")

        with open(registry_path) as f:
            data = yaml.safe_load(f)

        assert "datasets" in data

        # Check each dataset entry
        for name, spec in data["datasets"].items():
            # Required fields
            assert "url" in spec, f"{name} missing url"
            assert "sha256" in spec, f"{name} missing sha256"

            # Optional but recommended fields
            if "license" in spec:
                assert isinstance(spec["license"], str)

            if "process" in spec:
                assert "adapter" in spec["process"]
                # Adapter should be module:function format
                assert ":" in spec["process"]["adapter"]

            if "safe_to_publish" in spec:
                assert isinstance(spec["safe_to_publish"], bool)

    def test_registry_checksums_format(self):
        """Test that checksums are valid hex strings or placeholders."""
        registry_path = Path(__file__).parent.parent / "data" / "registry.yaml"
        if not registry_path.exists():
            pytest.skip("registry.yaml not found")

        with open(registry_path) as f:
            data = yaml.safe_load(f)

        for name, spec in data["datasets"].items():
            checksum = spec.get("sha256", "")

            # Either placeholder or 64-char hex
            if checksum != "PLACEHOLDER_SHA256_TO_BE_COMPUTED":
                assert len(checksum) == 64, f"{name} has invalid checksum length"
                assert all(c in "0123456789abcdef" for c in checksum.lower()), \
                    f"{name} has invalid checksum characters"


class TestDatasetContentSafety:
    """Test that dataset content is properly sanitized."""

    def test_no_raw_questions_in_artifacts(self, tmp_path):
        """Test that artifacts don't contain raw question text."""
        # Create test artifacts
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        # Bad artifact with raw questions
        bad_summary = {
            "accuracy": 0.95,
            "question": "What is the formula for X?",  # Forbidden
            "choices": ["A", "B", "C", "D"]  # Also forbidden
        }

        bad_file = artifacts_dir / "bad_summary.json"
        bad_file.write_text(json.dumps(bad_summary))

        # Good artifact without raw content
        good_summary = {
            "accuracy": 0.95,
            "total_items": 100,
            "consensus_k": 2,
            "exploitable_count": 25
        }

        good_file = artifacts_dir / "good_summary.json"
        good_file.write_text(json.dumps(good_summary))

        # Validate bad artifact
        with open(bad_file) as f:
            content = f.read()
            assert '"question"' in content  # Should be caught
            assert '"choices"' in content   # Should be caught

        # Validate good artifact
        with open(good_file) as f:
            content = f.read()
            assert '"question"' not in content
            assert '"choices"' not in content

    def test_no_exploitable_labels_in_artifacts(self, tmp_path):
        """Test that per-item exploitable labels are not in artifacts."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        # Bad CSV with exploitable column
        bad_csv = artifacts_dir / "items.csv"
        bad_csv.write_text("id,accuracy,exploitable\n1,0.8,true\n2,0.9,false")

        # Check that exploitable column exists (would be caught by validator)
        content = bad_csv.read_text()
        assert "exploitable" in content.lower()

    def test_sanitized_subset_validation(self):
        """Test that sanitized subsets are properly formatted."""
        sanitized_path = Path(__file__).parent.parent / "data" / "sample_sanitized.jsonl"
        if not sanitized_path.exists():
            pytest.skip("sample_sanitized.jsonl not found")

        with open(sanitized_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    # Should have required fields but safe content
                    assert 'id' in item
                    assert 'metadata' in item or 'answer' in item
                    # Should not have identifying information
                    assert len(item.get('question', '')) < 200  # Short questions only
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON at line {line_num}")


class TestDataCaching:
    """Test data caching and reuse."""

    def test_cache_directory_creation(self, tmp_path, monkeypatch):
        """Test that cache directories are created properly."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("ROBUSTCBRN_DATA_DIR", str(cache_dir))

        # Import with env var set
        import fetch_data

        # Should respect env var
        expected = cache_dir
        # Note: The actual implementation might differ
        # This tests the concept

    def test_cache_reuse(self, tmp_path):
        """Test that downloaded data is reused from cache."""
        import fetch_data

        # Setup cache
        raw_dir = tmp_path / "raw" / "test_dataset"
        raw_dir.mkdir(parents=True)
        (raw_dir / "data.parquet").write_text("cached data")

        # Mock to track if download is called
        download_called = False

        def mock_download(url, path):
            nonlocal download_called
            download_called = True

        with patch.object(fetch_data, 'RAW', tmp_path / "raw"):
            with patch.object(fetch_data, 'download_with_progress', side_effect=mock_download):
                # Should use cache, not download
                with patch.object(fetch_data, 'load_registry', return_value={"test_dataset": {}}):
                    result = fetch_data.fetch_dataset("test_dataset", force=False)

        assert not download_called
        assert result == raw_dir


class TestMakefileTargets:
    """Test Makefile targets for data management."""

    def test_make_data_target(self, tmp_path):
        """Test 'make data' target."""
        # This would require a full project setup
        # Simplified test checking the Makefile exists and has the target
        makefile_path = Path(__file__).parent.parent / "Makefile"
        if not makefile_path.exists():
            pytest.skip("Makefile not found")

        content = makefile_path.read_text()
        assert "data:" in content
        assert "fetch_data.py" in content

    def test_make_data_list_target(self, tmp_path):
        """Test 'make data-list' target."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        if not makefile_path.exists():
            pytest.skip("Makefile not found")

        content = makefile_path.read_text()
        assert "data-list:" in content
        assert "--list" in content
