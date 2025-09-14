#!/usr/bin/env python3
"""Quick feature verification and cleanup script."""

import sys
import os
from pathlib import Path

def test_core_features():
    """Test core features quickly (pytest-friendly: use asserts, no returns)."""
    print("🔍 Testing core features...")

    # Test imports
    from robustcbrn import (
        AppConfig,
        Question,
        load_dataset,  # noqa: F401 (import check)
        analyze_questions,  # noqa: F401 (import check)
        detect_position_bias,  # noqa: F401 (import check)
    )
    print("✅ Core imports work")

    # Test configuration
    config = AppConfig.from_json("configs/default.json")
    assert config is not None
    print("✅ Configuration loading works")

    # Test Question creation
    question = Question(id="test", question="Test?", choices=["A", "B"], answer=0)
    assert question.id == "test"
    assert question.answer == 0
    print("✅ Question creation works")

    # Test CLI help
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "robustcbrn.cli.main", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, f"CLI exited with {result.returncode}\nSTDERR: {result.stderr}"
    assert "RobustCBRN CLI" in result.stdout
    print("✅ CLI interface works")

def cleanup_temp_files():
    """Remove all temporary files created during testing."""
    print("\n🧹 Cleaning up temporary files...")
    
    temp_files = [
        "test_comprehensive.py",
        "INTEGRATION_SUMMARY.md",
        "temp_logs",
        "temp_delete",
        "*.tmp",
        "*.temp"
    ]
    
    cleaned = 0
    for pattern in temp_files:
        if "*" in pattern:
            # Handle glob patterns
            import glob
            for file in glob.glob(pattern):
                try:
                    if Path(file).is_file():
                        Path(file).unlink()
                        cleaned += 1
                    elif Path(file).is_dir():
                        import shutil
                        shutil.rmtree(file)
                        cleaned += 1
                except Exception:
                    pass
        else:
            # Handle specific files
            path = Path(pattern)
            try:
                if path.exists():
                    if path.is_file():
                        path.unlink()
                        cleaned += 1
                    elif path.is_dir():
                        import shutil
                        shutil.rmtree(path)
                        cleaned += 1
            except Exception:
                pass
    
    print(f"✅ Cleaned up {cleaned} temporary files")

def main():
    """Run tests and cleanup."""
    print("🚀 Quick feature verification and cleanup...")
    print("=" * 50)
    
    # Test features
    if test_core_features():
        print("\n✅ All core features working!")
    else:
        print("\n❌ Some features failed!")
        return False
    
    # Cleanup
    cleanup_temp_files()
    
    print("\n🎉 Verification complete and cleanup done!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
