# Project Structure

This document describes the organization and purpose of directories and files in the RobustCBRN Eval project.

## Directory Layout

```
robustcbrn-eval/
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── pipeline.py        # Main orchestration logic
│   │
│   ├── analysis/          # Statistical and bias analysis
│   │   ├── __init__.py
│   │   ├── bootstrap.py
│   │   ├── chi_square.py
│   │   ├── heuristics.py
│   │   ├── patterns.py
│   │   └── statistical.py
│   │
│   ├── cache/             # Caching mechanisms
│   │   ├── __init__.py
│   │   ├── checkpoint.py
│   │   ├── json_cache.py
│   │   ├── manager.py
│   │   └── sqlite_cache.py
│   │
│   ├── data/              # Data loading and validation
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── schemas.py
│   │   └── validators.py
│   │
│   ├── models/            # Model management and inference
│   │   ├── __init__.py
│   │   ├── cloze.py
│   │   ├── consensus.py
│   │   ├── inference.py
│   │   └── loader.py
│   │
│   ├── reporting/         # Report generation
│   │   ├── __init__.py
│   │   ├── formatters.py
│   │   ├── generator.py
│   │   └── templates.py
│   │
│   ├── security/          # Security and anonymization
│   │   ├── __init__.py
│   │   ├── anonymizer.py
│   │   └── redactor.py
│   │
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── determinism.py
│       ├── diversity.py
│       ├── logging.py
│       ├── memory.py
│       └── parallel.py
│
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── fixtures/          # Test data
│   │   ├── sample_questions.csv
│   │   └── sample_questions.json
│   ├── test_cli_*.py     # CLI tests
│   ├── test_config.py    # Configuration tests
│   ├── test_heuristics.py # Heuristics tests
│   └── test_pipeline.py  # Integration tests
│
├── configs/               # Configuration files
│   ├── default.json      # Default configuration
│   ├── full.json         # Full feature set
│   └── minimal.json      # Minimal/CPU-only config
│
├── data/                  # Input data (gitignored)
│   └── wmdp_bio_sample_100.jsonl
│
├── docs/                  # Documentation
│   ├── architecture/      # Architecture docs
│   ├── prd/              # Product requirements
│   ├── qa/               # QA gates
│   ├── stories/          # User stories
│   ├── brief.md          # Project brief
│   ├── development-setup.md
│   └── project-structure.md (this file)
│
├── scripts/               # Setup and utility scripts
│   ├── setup.sh          # Linux/macOS setup
│   ├── setup.ps1         # Windows PowerShell setup
│   ├── setup.bat         # Windows CMD setup
│   └── validate_install.py
│
├── cache/                 # Cache directory (gitignored)
├── logs/                  # Log files (gitignored)
├── results/               # Output directory (gitignored)
│
├── cli.py                 # Main entry point
├── requirements.txt       # Core dependencies
├── requirements-dev.txt   # Development dependencies
├── pyproject.toml        # Project configuration
├── .gitignore            # Git ignore rules
├── .editorconfig         # Editor configuration
├── .python-version       # Python version specification
├── LICENSE               # MIT License
├── README.md             # Project documentation
├── CONTRIBUTING.md       # Contribution guidelines
├── CHANGELOG.md          # Version history
└── AUTHORS.md            # Contributors list
```

## Key Directories

### `/src`
Core application code organized by functionality:
- **analysis**: Statistical analysis and bias detection algorithms
- **cache**: Caching layer for performance optimization
- **data**: Data loading, validation, and schema definitions
- **models**: Model management, loading, and inference
- **reporting**: Report generation and formatting
- **security**: Anonymization and security features
- **utils**: Shared utility functions

### `/tests`
Comprehensive test suite:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Test fixtures with sample data
- Follows `test_*.py` naming convention

### `/configs`
JSON configuration files for different scenarios:
- **default.json**: Standard configuration for most use cases
- **minimal.json**: CPU-only configuration without GPU dependencies
- **full.json**: Full feature set with all optional components enabled

### `/docs`
Project documentation:
- **architecture/**: Technical architecture documentation
- **prd/**: Product requirements and specifications
- **qa/**: Quality assurance gates and testing plans
- **stories/**: User stories and development tasks

### `/scripts`
Setup and utility scripts:
- Cross-platform setup scripts (bash, PowerShell, batch)
- Validation and helper scripts

## File Naming Conventions

### Python Files
- Use snake_case: `my_module.py`
- Test files prefixed with `test_`: `test_my_module.py`
- Private modules prefixed with `_`: `_internal.py`

### Documentation
- Markdown files use lowercase with hyphens: `development-setup.md`
- Architecture docs follow topic naming: `caching-strategy.md`

### Configuration
- JSON files use lowercase: `default.json`
- YAML files use lowercase with hyphens: `github-actions.yml`

## Import Structure

### Absolute Imports
```python
from src.data.loader import load_dataset
from src.analysis.statistical import calculate_statistics
```

### Relative Imports (within same package)
```python
from .schemas import Question
from ..utils.logging import setup_logging
```

## Data Flow

1. **Input**: Raw data files (JSONL/CSV) → `data/loader.py`
2. **Validation**: Data validation → `data/validators.py`
3. **Processing**: Analysis pipeline → `pipeline.py`
4. **Caching**: Results caching → `cache/manager.py`
5. **Output**: Report generation → `reporting/generator.py`

## Environment Variables

Key environment variables used by the project:

```bash
ROBUSTCBRN_CACHE_DIR    # Cache directory location
ROBUSTCBRN_LOG_LEVEL    # Logging level (DEBUG, INFO, WARNING, ERROR)
ROBUSTCBRN_PROJECT_SALT # Salt for question ID generation
CUDA_VISIBLE_DEVICES    # GPU device selection
```

## Generated Directories

These directories are created automatically and excluded from version control:

- **cache/**: Stores cached analysis results
- **results/**: Output files from analysis runs
- **logs/**: Application and debug logs
- **.venv/**: Python virtual environment
- **__pycache__/**: Python bytecode cache
- **.mypy_cache/**: MyPy type checking cache
- **.ruff_cache/**: Ruff linting cache

## Module Responsibilities

### Core Modules

#### `cli.py`
- Command-line interface entry point
- Argument parsing and validation
- Command routing (load, analyze, etc.)

#### `src/config.py`
- Configuration dataclass definitions
- Config loading and validation
- Default value management

#### `src/pipeline.py`
- Main orchestration logic
- Component coordination
- Error handling and recovery

### Analysis Modules

#### `src/analysis/heuristics.py`
- Simple heuristic implementations
- Pattern-based analysis
- Quick bias detection

#### `src/analysis/statistical.py`
- Statistical computations
- Confidence intervals
- Distribution analysis

### Data Modules

#### `src/data/loader.py`
- File format detection
- Data loading from JSONL/CSV
- Format conversion

#### `src/data/schemas.py`
- Question dataclass definition
- Data model specifications
- Type definitions

### Utility Modules

#### `src/utils/logging.py`
- Logging configuration
- Log formatting
- File and console handlers

#### `src/utils/determinism.py`
- Seed setting for reproducibility
- CUDA determinism controls
- Hash seed management

## Adding New Features

When adding new features:

1. **Create module** in appropriate directory
2. **Add tests** in `tests/` with `test_` prefix
3. **Update imports** in `__init__.py` files
4. **Document** in module docstrings
5. **Update configs** if new settings needed
6. **Add to CLI** if user-facing

## Best Practices

1. **Separation of Concerns**: Each module has a single, clear purpose
2. **Dependency Management**: Minimize cross-module dependencies
3. **Testing**: Every module should have corresponding tests
4. **Documentation**: All public functions need docstrings
5. **Type Hints**: Use type hints for all function signatures
6. **Error Handling**: Fail gracefully with informative messages