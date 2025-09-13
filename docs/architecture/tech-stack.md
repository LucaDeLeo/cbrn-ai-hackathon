# Tech Stack

## Core Dependencies (Minimal by Design)

| Category | Technology | Version | Purpose | Rationale |
|----------|------------|---------|---------|-----------|
| Language | Python | 3.10+ | Primary implementation | ML ecosystem standard |
| ML Framework | PyTorch | 2.0+ | Model inference | Industry standard, best GPU support |
| Model Library | Transformers | 4.36+ | Model loading/inference | Handles diverse architectures |
| Numerical | NumPy | 1.24+ | Statistical computations | Sufficient for all statistics |
| Progress | tqdm | 4.66+ | Progress bars | Lightweight, informative |
| Data Storage | SQLite3 | Built-in | Metadata caching | No external dependency |
| Data Parsing | json/csv | Built-in | Input/output handling | Sufficient for needs |
| CLI | argparse | Built-in | Command-line interface | Simple, sufficient |
| Config | dataclasses | Built-in | Configuration management | Type-safe, no deps |
| Testing | unittest | Built-in | Test framework | Adequate for needs |
| Logging | logging | Built-in | Structured logging | Standard library |
| Parallelism | concurrent.futures | Built-in | CPU parallelization | Simple, effective |
| Hashing | hashlib | Built-in | Question anonymization | Cryptographically secure |

## Optional Extras (Gated and Fallbacks Defined)

| Category | Technology | Purpose | Notes |
|----------|------------|---------|-------|
| Quantization | bitsandbytes | 8-bit loading | Optional; fallback to fp16 if unavailable |
| Attention | flash-attn | Throughput | Optional; disabled in deterministic mode |
| Logging | python-json-logger | JSON logs | Optional; stdlib logging used by default |
| Statistics | SciPy | BCa bootstrap only | Optional; default method is percentile without SciPy |
