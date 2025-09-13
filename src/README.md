# Source Code Directory

This directory contains the core application code for RobustCBRN Eval.

## Module Structure

- **analysis/**: Statistical analysis and bias detection algorithms
- **cache/**: Caching mechanisms for performance optimization
- **data/**: Data loading, validation, and schema definitions
- **models/**: Model management, loading, and inference
- **reporting/**: Report generation and formatting
- **security/**: Anonymization and security features
- **utils/**: Shared utility functions

## Key Files

- `config.py`: Configuration management using dataclasses
- `pipeline.py`: Main orchestration logic for the evaluation pipeline

## Usage

All modules are designed to be imported and used by the main CLI entry point (`cli.py`) or directly in Python scripts:

```python
from src.data.loader import load_dataset
from src.analysis.statistical import calculate_statistics

# Load and analyze data
questions = load_dataset("data/sample.jsonl")
stats = calculate_statistics(questions)
```

## Development

When adding new functionality:
1. Create modules in the appropriate subdirectory
2. Add corresponding tests in `tests/`
3. Update `__init__.py` files to export public APIs
4. Document all public functions with docstrings