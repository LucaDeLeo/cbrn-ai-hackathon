# RobustCBRN Eval

> Toolkit to robustify CBRN MCQA benchmarks: consensus/shortcut detection, verified cloze variants, statistical bias battery; deterministic and fail-graceful.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/apart-research/robustcbrn-eval)
[![Test Coverage](https://img.shields.io/badge/coverage-70%25-yellow)](https://github.com/apart-research/robustcbrn-eval)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

Practical toolkit for robustifying CBRN-related AI benchmarks. Implements consensus/shortcut detection, verified cloze variants, and a statistical bias battery with deterministic, fail-graceful execution. Designed to evaluate and improve the robustness of AI models on CBRN (Chemical, Biological, Radiological, Nuclear) multiple-choice question answering tasks.

## Features

### Core Evaluation Tasks (Harness Branch)
- **MCQ Tasks**: `mcq_full`, `mcq_choices_only` for comprehensive multiple-choice evaluation
- **Cloze Tasks**: `cloze_full`, `cloze_logprob` for masked language model evaluation
- **QA System**: Ambiguity detection, paraphrasing, perturbation analysis
- **Budget Guard**: ~$400 cloud spend protection (Lambda A100)
- **Inspect Integration**: Built for LM Harness framework

### Statistical Analysis (Main Branch Integration)
- **Consensus Detection**: Identify when multiple models agree on incorrect answers
- **Shortcut Analysis**: Detect when models rely on spurious patterns
- **Position Bias Detection**: Comprehensive statistical analysis of answer position bias
- **Bootstrap Analysis**: Confidence intervals and statistical significance testing
- **Heuristic Degradation**: Analyze performance degradation between original and robust datasets

### Data Management & Security
- **Flexible Data Loading**: Support for JSONL and CSV formats with answer normalization
- **Secure Anonymization**: BLAKE2b hashing for question IDs with configurable salt
- **Data Validation**: Comprehensive schema validation and error handling
- **Deterministic Execution**: Reproducible results with configurable seeds

### CLI Interface
- **Command-line Tools**: Complete CLI for data loading, analysis, and reporting
- **Configuration Management**: JSON-based configuration with sensible defaults
- **Progress Tracking**: Verbose output and progress bars for long-running analyses
- **Dry Run Support**: Validate inputs without processing

### Unified Pipeline System
- **Cross-Platform Support**: Works on Windows (Git Bash/Cygwin), macOS, and Linux
- **Comprehensive Error Handling**: Robust error detection and recovery
- **Modular Design**: Run individual steps or complete pipeline
- **Extensive Configuration**: Environment variables and command-line arguments
- **Comprehensive Logging**: Timestamped logs with configurable levels
- **Platform Detection**: Automatic platform detection and configuration
- **Dependency Validation**: Comprehensive dependency checking
- **Security Validation**: Built-in security checks and validation

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

### CLI Commands

```bash
# Load and preview a dataset
python -m robustcbrn.cli.main load data/wmdp_bio_sample_100.jsonl

# Analyze dataset with default settings
python -m robustcbrn.cli.main analyze data/wmdp_bio_sample_100.jsonl --output results/analysis.json

# Dry run to validate inputs without processing
python -m robustcbrn.cli.main analyze data/wmdp_bio_sample_100.jsonl --dry-run

# Analyze with verbose output and limit to 50 questions
python -m robustcbrn.cli.main analyze data/dataset.jsonl --verbose --max-items 50

# Analyze position bias in dataset
python -m robustcbrn.cli.main position-bias data/dataset.jsonl --output results/position_bias.json --verbose

# Analyze heuristic degradation between datasets
python -m robustcbrn.cli.main heuristic-degradation --original data/original.jsonl --robust data/robust.jsonl --output results/degradation.json
```

### New Unified Pipeline (Recommended)

The new pipeline provides a comprehensive, cross-platform end-to-end evaluation system:

```bash
# Run complete pipeline
make pipeline

# Run individual pipeline steps
make pipeline-validate    # Platform validation only
make pipeline-setup       # Setup environment
make pipeline-sample      # Sample evaluation
make pipeline-full        # Complete evaluation

# Run specific pipeline components
make pipeline-aggregate   # Aggregate results
make pipeline-figures     # Generate figures
make pipeline-tests       # Run tests and security checks
make pipeline-report      # Generate final report
make pipeline-verify      # Final verification
```

### Legacy Harness Integration (Still Supported)

```bash
# Run evaluation tasks
make setup
make sample
make run

# Run specific tasks
make run-mcq-full
make run-cloze-full
```

### Advanced Pipeline Usage

```bash
# Custom configuration
bash scripts/run_pipeline.sh --dataset data/custom.jsonl --subset 256

# Run specific steps only
bash scripts/run_pipeline.sh --steps setup,sample,aggregate

# Cross-platform support
bash scripts/run_pipeline.sh --steps validate,setup  # Works on Windows, macOS, Linux
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

### Pre-commit Hooks

Install git hooks to enforce code quality locally:

```bash
pip install pre-commit
pre-commit install
# or use the helper script
bash scripts/install-hooks.sh
```

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
- **Architecture**: [`docs/architecture/architecture.md`](docs/architecture/architecture.md)
- **Usage Guide**: [`docs/getting-started/usage.md`](docs/getting-started/usage.md)
- **Security**: [`docs/safety/security-considerations.md`](docs/safety/security-considerations.md)
- **Release Checklist**: [`docs/safety/release-checklist.md`](docs/safety/release-checklist.md)

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

See [`docs/safety/security-considerations.md`](docs/safety/security-considerations.md) for anonymization and public artifact rules.
