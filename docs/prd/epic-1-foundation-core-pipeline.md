# Epic 1: Foundation & Core Pipeline

**Goal**: Establish the minimal viable pipeline that reads benchmarks, processes questions through a simple heuristic, and outputs results. This creates the skeleton that all other features plug into while delivering a working demo within 6 hours.

## Story 1.1: Project Setup & Data Pipeline

**As a** developer,
**I want** a clean project structure with data loading capabilities,
**so that** the team can start parallel development immediately.

**Acceptance Criteria:**
1. Repository initialized with structure: `src/`, `tests/`, `configs/`, `data/`, `cache/`, `results/`
2. Minimal requirements.txt with only: torch, transformers, numpy, tqdm (installed via `uv pip install`)
3. Data loader handles both JSONL and CSV formats with consistent internal representation
4. Configuration system using Python dataclasses (no external deps) with JSON serialization (stdlib-only)
5. Hash-based question ID generation working with configurable salt
6. Basic logging setup writing to both console and file
7. Successfully loads and parses WMDP-Bio sample (first 100 questions) from `data/wmdp_bio_sample_100.jsonl`
8. Determinism controls set and documented: fixed Python/NumPy/Torch seeds; `torch.use_deterministic_algorithms(True)`; cuDNN deterministic enabled and benchmarking disabled; sorted iteration order

## Story 1.2: Simple Heuristic Implementation

**As a** researcher,
**I want** to see the longest-answer baseline working,
**so that** we validate the infrastructure and have immediate results.

**Acceptance Criteria:**
1. Longest-answer selector implemented in <20 lines of readable code
2. Processes 1,000 questions in <10 seconds (no model needed)
3. Outputs accuracy score matching reported 46% baseline on WMDP-Bio
4. Results saved to JSON with timestamp and configuration
5. Progress bar shows questions processed with ETA
6. Memory usage stays under 1GB for 10,000 questions

## Story 1.3: CLI Interface Foundation

**As a** user,
**I want** a simple command-line interface,
**so that** I can run analyses with different parameters.

**Acceptance Criteria:**
1. Basic argparse CLI with --input, --output, --config flags
2. --dry-run flag that validates input without processing
3. --verbose flag that shows detailed progress
4. Graceful error handling with helpful messages
5. Exit codes: 0=success, 1=error, 2=partial completion
6. Help text includes concrete usage examples
7. Additional flags: `--models`, `--max-items`, `--time-limit`, `--budget`, `--public-report {on,off}`

## Story 1.4: Schema Validation & Stratified Sampling

**As a** developer,
**I want** strict schema validation and a stratified sampler,
**so that** data quality issues are caught early and analyses preserve topic balance.

**Acceptance Criteria:**
1. Schema validation with clear errors for missing fields or invalid types
2. Dataset hash and row count logged for audit
3. Stratified sampler supports topic/difficulty strata with fixed seeds
4. CLI `--subset` can sample stratified subsets deterministically
5. Summary of per-stratum counts in reports (no plaintext content)

## Story 1.5: Cache Integrity & Config Hashing

**As a** developer,
**I want** cache invalidation tied to config changes,
**so that** stale results are not reused incorrectly.

**Acceptance Criteria:**
1. Compute config hash (model IDs, seeds, params) and store with outputs
2. On run start, detect config mismatch and invalidate/segregate cache
3. Checkpoint recovery validated with changed vs unchanged configs
4. CLI `--resume` and `--no-resume` flags respected
5. Cache integrity verified in unit tests (happy path + mismatch)

## Story 1.6: Test Skeleton & Sample Data

**As a** developer,
**I want** a minimal test suite and synthetic sample data,
**so that** regressions are caught early and demos run without sensitive content.

**Acceptance Criteria:**
1. Unit tests for bootstrap CI, chi-square, and longest-answer selector
2. Integration test for CLI `--dry-run` and a small synthetic dataset
3. Determinism test validates reproducible results across runs
4. Sample sanitized dataset (synthetic) included for demo and CI
5. Test runtime <10 seconds locally
