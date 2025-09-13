#!/usr/bin/env bash
# RobustCBRN Eval - Setup Script using uv for fast Python package management

set -euo pipefail

echo "🚀 RobustCBRN Eval Setup Script"
echo "================================"

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi
echo "✅ Python $python_version found"

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed ($(uv --version))"
fi

# Create virtual environment with uv
echo "🔧 Creating virtual environment with uv..."
uv venv --python python3.10

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with uv
echo "📚 Installing dependencies with uv (this is 10x faster than pip)..."
uv pip install -r requirements.txt

# For CUDA environments, install PyTorch with CUDA support
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 CUDA detected. Installing PyTorch with CUDA 11.8 support..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data cache results logs

echo ""
echo "✨ Setup complete! Environment is ready."
echo ""
echo "📌 Quick start commands:"
echo "  - Activate environment: source .venv/bin/activate"
echo "  - Run tests: python -m unittest"
echo "  - Run pipeline: python cli.py load data/wmdp_bio_sample_100.jsonl"
echo ""
echo "💡 Tip: uv commands for package management:"
echo "  - Install package: uv pip install <package>"
echo "  - Sync exact deps: uv pip sync requirements.txt"
echo "  - Compile deps: uv pip compile requirements.in -o requirements.txt"
