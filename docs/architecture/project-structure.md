# Project Structure

```
robustcbrn-eval/
├── src/
│   ├── __init__.py
│   ├── pipeline.py                 # Main orchestrator
│   ├── config.py                   # Configuration dataclasses
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Dataset loading
│   │   ├── schemas.py             # Data models
│   │   └── validators.py          # Input validation
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistical.py         # Pure NumPy statistics
│   │   ├── bootstrap.py           # Bootstrap CI implementation
│   │   ├── chi_square.py          # Chi-square tests
│   │   ├── patterns.py            # Lexical pattern detection
│   │   ├── permutation.py         # Permutation sensitivity testing
│   │   └── bias_metrics.py        # Bias calculations
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── consensus.py           # Consensus detection
│   │   ├── cloze.py              # Cloze scoring
│   │   ├── loader.py             # Model loading/management
│   │   └── inference.py          # Optimized inference
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── manager.py            # Cache orchestration
│   │   ├── sqlite_cache.py       # SQLite layer
│   │   ├── json_cache.py         # JSON layer
│   │   └── checkpoint.py         # Checkpoint system
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── anonymizer.py         # Hash-based anonymization
│   │   └── redactor.py           # Output redaction
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── generator.py          # Report generation
│   │   ├── formatters.py         # Output formatters
│   │   └── templates.py          # Report templates
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py            # Logging setup
│       ├── memory.py             # Memory monitoring
│       ├── parallel.py           # Parallel execution helpers
│       └── diversity.py          # Model family detection/validation
│
├── tests/
│   ├── __init__.py
│   ├── test_statistical.py       # Statistical tests
│   ├── test_consensus.py         # Consensus tests
│   ├── test_cache.py            # Cache tests
│   ├── test_pipeline.py         # Integration tests
│   └── fixtures/                # Test data
│       └── sample_questions.json
│
├── configs/
│   ├── default.json             # Default configuration
│   ├── minimal.json             # CPU-only config
│   └── full.json               # All features enabled
│
├── scripts/
│   ├── setup.sh                # Environment setup
│   ├── download_models.py      # Model download helper
│   └── validate_install.py     # Installation check
│
├── results/                    # Output directory
├── cache/                      # Cache directory
├── logs/                       # Log files
│
├── cli.py                      # Main CLI entry point
├── requirements.txt            # Minimal dependencies
├── Dockerfile                  # Reproducible environment
├── README.md                   # Documentation
└── LICENSE                     # Open source license
```
