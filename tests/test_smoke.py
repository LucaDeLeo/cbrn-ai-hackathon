"""Smoke tests for RobustCBRN Eval.

Basic tests to verify installation and environment setup.
"""

import sys
import unittest
from pathlib import Path


class TestSmoke(unittest.TestCase):
    """Basic smoke tests for project setup."""

    def test_python_version(self):
        """Test that Python version is 3.10 or higher."""
        version_info = sys.version_info
        self.assertGreaterEqual(version_info.major, 3)
        if version_info.major == 3:
            self.assertGreaterEqual(
                version_info.minor, 10,
                f"Python 3.10+ required, found {version_info.major}.{version_info.minor}"
            )

    def test_required_directories_exist(self):
        """Test that all required project directories exist."""
        required_dirs = [
            "src",
            "tests",
            "configs",
            "data",
            "cache",
            "results",
            "logs",
            "scripts",
            "docs",
        ]

        project_root = Path(__file__).parent.parent
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            self.assertTrue(
                dir_path.exists(),
                f"Required directory '{dir_name}' does not exist"
            )
            self.assertTrue(
                dir_path.is_dir(),
                f"'{dir_name}' exists but is not a directory"
            )

    def test_cli_entry_point_exists(self):
        """Test that the CLI entry point is callable."""
        project_root = Path(__file__).parent.parent
        cli_path = project_root / "cli.py"

        self.assertTrue(cli_path.exists(), "cli.py does not exist")
        self.assertTrue(cli_path.is_file(), "cli.py is not a file")

        # Test that cli.py can be imported
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli", cli_path)
        self.assertIsNotNone(spec, "Cannot create import spec for cli.py")

        cli_module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(cli_module, "Cannot create module from cli.py")

    def test_core_imports(self):
        """Test that core modules can be imported."""
        required_imports = [
            "src",
            "src.config",
            "src.data",
            "src.data.loader",
            "src.data.schemas",
            "src.data.validators",
            "src.utils",
            "src.utils.logging",
            "src.utils.determinism",
            "src.security",
            "src.security.anonymizer",
            "src.analysis",
            "src.analysis.heuristics",
        ]

        for module_name in required_imports:
            try:
                __import__(module_name)
            except ImportError as e:
                self.fail(f"Cannot import required module '{module_name}': {e}")

    def test_configuration_files_exist(self):
        """Test that configuration files exist."""
        project_root = Path(__file__).parent.parent
        config_files = [
            "configs/default.json",
            "requirements.txt",
            "pyproject.toml",
            ".gitignore",
            ".editorconfig",
        ]

        for file_path in config_files:
            full_path = project_root / file_path
            self.assertTrue(
                full_path.exists(),
                f"Configuration file '{file_path}' does not exist"
            )
            self.assertTrue(
                full_path.is_file(),
                f"'{file_path}' exists but is not a file"
            )

    def test_documentation_files_exist(self):
        """Test that key documentation files exist."""
        project_root = Path(__file__).parent.parent
        doc_files = [
            "README.md",
            "LICENSE",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "AUTHORS.md",
        ]

        for file_name in doc_files:
            file_path = project_root / file_name
            self.assertTrue(
                file_path.exists(),
                f"Documentation file '{file_name}' does not exist"
            )

    def test_development_setup_doc_exists(self):
        """Test that development setup documentation exists (AC7)."""
        project_root = Path(__file__).parent.parent
        dev_setup = project_root / "docs" / "development-setup.md"
        self.assertTrue(dev_setup.exists(), "docs/development-setup.md does not exist")
        self.assertTrue(dev_setup.is_file(), "docs/development-setup.md is not a file")

    def test_fixture_data_exists(self):
        """Test that test fixture data exists."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        self.assertTrue(fixtures_dir.exists(), "fixtures directory does not exist")
        self.assertTrue(fixtures_dir.is_dir(), "fixtures is not a directory")

        # Check for sample data files
        sample_files = [
            "sample_questions.json",
            "sample_questions.csv",
        ]

        for file_name in sample_files:
            file_path = fixtures_dir / file_name
            self.assertTrue(
                file_path.exists(),
                f"Fixture file '{file_name}' does not exist"
            )

    def test_git_repository_initialized(self):
        """Test that git repository is initialized."""
        project_root = Path(__file__).parent.parent
        git_dir = project_root / ".git"

        self.assertTrue(
            git_dir.exists(),
            "Git repository not initialized (.git directory missing)"
        )
        self.assertTrue(
            git_dir.is_dir(),
            ".git exists but is not a directory"
        )

    def test_virtual_environment_detection(self):
        """Test virtual environment detection (warning only)."""
        import os
        venv_indicators = [
            "VIRTUAL_ENV",
            "CONDA_DEFAULT_ENV",
            "PIPENV_ACTIVE",
        ]

        venv_active = any(os.environ.get(indicator) for indicator in venv_indicators)

        if not venv_active:
            # Check if we're in a .venv directory
            if ".venv" not in sys.executable:
                import warnings
                warnings.warn(
                    "Virtual environment may not be activated. "
                    "Ensure you're running tests in a virtual environment."
                )


if __name__ == "__main__":
    unittest.main()
