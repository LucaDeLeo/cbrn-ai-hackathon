#!/usr/bin/env bash
# Install and configure pre-commit hooks for RobustCBRN Eval

set -euo pipefail

echo "🔧 Installing pre-commit hooks..."
echo "================================"

# Check if in git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi
echo "✅ Python $python_version found"

# Check if virtual environment is activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Activate with: source .venv/bin/activate"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
    echo "✅ pre-commit installed"
else
    echo "✅ pre-commit is already installed ($(pre-commit --version))"
fi

# Install the git hook scripts
echo "🪝 Installing git hooks..."
pre-commit install

# Install additional hook types
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg

# Update hooks to latest versions
echo "📥 Updating hooks to latest versions..."
pre-commit autoupdate

# Run hooks on all files (optional, can be slow)
echo ""
echo "🏃 Would you like to run pre-commit on all files now?"
echo "   This will check and fix formatting issues but may take a few minutes."
read -p "Run pre-commit on all files? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔍 Running pre-commit on all files..."
    pre-commit run --all-files || true
    echo "✅ Pre-commit run complete"
fi

echo ""
echo "✨ Pre-commit hooks installed successfully!"
echo ""
echo "📌 Hooks will now run automatically on:"
echo "   - git commit (code quality checks)"
echo "   - git push (additional validation)"
echo ""
echo "💡 Manual commands:"
echo "   - Run on all files: pre-commit run --all-files"
echo "   - Run on staged files: pre-commit run"
echo "   - Update hooks: pre-commit autoupdate"
echo "   - Skip hooks once: git commit --no-verify"
echo ""
echo "📝 Configuration: .pre-commit-config.yaml"