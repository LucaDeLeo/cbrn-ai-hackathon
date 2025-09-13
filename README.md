# RobustCBRN Eval

> Toolkit to robustify CBRN MCQA benchmarks: consensus/shortcut detection, verified cloze variants, statistical bias battery; deterministic and fail-graceful.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/apart-research/robustcbrn-eval)
[![Test Coverage](https://img.shields.io/badge/coverage-70%25-yellow)](https://github.com/apart-research/robustcbrn-eval)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

Practical toolkit for robustifying CBRN-related AI benchmarks. Implements consensus/shortcut detection, verified cloze variants, and a statistical bias battery with deterministic, fail-graceful execution. Designed to evaluate and improve the robustness of AI models on CBRN (Chemical, Biological, Radiological, Nuclear) multiple-choice question answering tasks.

## Features

- **Consensus Detection**: Identify when multiple models agree on incorrect answers
- **Shortcut Analysis**: Detect when models rely on spurious patterns
- **Cloze Variants**: Generate and verify masked question variants
- **Statistical Bias Battery**: Comprehensive suite of bias detection metrics
- **Deterministic Execution**: Reproducible results with configurable seeds
- **Fail-Graceful Design**: Robust error handling and recovery
- **Efficient Caching**: SQLite and JSON-based result caching
- **Secure Anonymization**: BLAKE2b hashing for question IDs

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA (optional, for GPU acceleration)
- Git

### Installation

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/apart-research/robustcbrn-eval.git
cd robustcbrn-eval

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt

# For CUDA-specific PyTorch installation:
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Commands

```bash
# Load and analyze a dataset
python cli.py load data/wmdp_bio_sample_100.jsonl --config configs/default.json

# Run with custom ID salt
python cli.py load data/wmdp_bio_sample_100.jsonl --id-salt your_salt

# Run analysis on loaded data
python cli.py analyze data/wmdp_bio_sample_100.jsonl --output results/analysis.json

# Run with minimal config (CPU-only)
python cli.py load data/sample.json --config configs/minimal.json

# Resume from checkpoint
python cli.py --resume cache/checkpoint.json

# Dry run to validate
python cli.py load data/sample.json --dry-run
```

### Data Format

Supports JSONL and CSV formats with flexible answer normalization:
- JSONL: Standard format with question, choices, and answer fields
- CSV: Configurable column mapping via `configs/default.json`
- Answers: Letters (A-F) or integers, normalized to 0-based index

## Development

See [Development Setup](docs/development-setup.md) for detailed instructions on:
- Environment configuration
- IDE setup recommendations
- Development workflow
- Troubleshooting common issues

### Running Tests

```bash
# Run all tests
python -m unittest

# Run specific test module
python -m unittest tests.test_pipeline

# Run with coverage
coverage run -m unittest
coverage report
```

## Documentation

- **Project Brief**: [`docs/brief.md`](docs/brief.md)
- **PRD**: [`docs/prd/index.md`](docs/prd/index.md)
- **Architecture**: [`docs/architecture/index.md`](docs/architecture/index.md)
- **Development Setup**: [`docs/development-setup.md`](docs/development-setup.md)
- **List of Evaluations**: [`docs/listofevals.md`](docs/listofevals.md)
- **Release Checklist**: [`docs/release_checklist.md`](docs/release_checklist.md)
- **Prompts Appendix**: [`docs/prompts_appendix.md`](docs/prompts_appendix.md)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code of conduct
- Development workflow
- Commit message conventions
- Pull request process
- Testing requirements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{robustcbrn-eval,
  title = {RobustCBRN Eval: Toolkit for Robustifying CBRN AI Benchmarks},
  author = {[Authors]},
  year = {2024},
  url = {https://github.com/apart-research/robustcbrn-eval}
}
```

## Safety & Release Policy

See [`docs/prd/artifacts-release-policy.md`](docs/prd/artifacts-release-policy.md) for anonymization and public artifact rules.
