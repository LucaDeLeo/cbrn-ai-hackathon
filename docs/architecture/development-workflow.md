# Development Workflow

## Local Development Setup

    # Prerequisites
    python3.10 --version  # Verify Python 3.10+
    nvidia-smi           # Verify CUDA availability

    # Initial Setup
    git clone https://github.com/apart-research/robustcbrn-eval.git
    cd robustcbrn-eval

    # Install uv (fast Python package manager)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create virtual environment with uv
    uv venv
    source .venv/bin/activate  # Linux/Mac
    # or: .venv\\Scripts\\activate  # Windows

    # Install dependencies with uv (10x faster than pip)
    uv pip install -r requirements.txt

    # For CUDA-specific PyTorch installation:
    # uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # Download sample data
    wget https://example.com/wmdp-bio-sample.json -O data/sample.json

    # Run tests
    python -m unittest discover tests/

## Development Commands

    # Run with minimal config (CPU-only)
    python cli.py --input data/sample.json --config configs/minimal.json

    # Run with GPU
    python cli.py --input data/sample.json --config configs/full.json

    # Resume from checkpoint
    python cli.py --resume cache/checkpoint.json

    # Dry run to validate
    python cli.py --input data/sample.json --dry-run

    # Run specific components only
    python cli.py --input data/sample.json --components statistical

    # Generate report only (from cached results)
    python cli.py --report-only --cache-dir cache/

    # Enable deterministic mode and cost estimation
    python cli.py --input data/sample.json --deterministic --gpu-hourly-price 1.10

## Environment Configuration

    # Environment Variables
    export ROBUSTCBRN_CACHE_DIR=/path/to/cache
    export ROBUSTCBRN_LOG_LEVEL=DEBUG
    export ROBUSTCBRN_PROJECT_SALT=your-secret-salt
    export CUDA_VISIBLE_DEVICES=0  # GPU selection
