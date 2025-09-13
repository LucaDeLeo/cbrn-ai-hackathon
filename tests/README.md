# Test Suite

This directory contains the test suite for RobustCBRN Eval.

## Structure

- **fixtures/**: Test data and sample files
- **test_*.py**: Test modules following unittest conventions

## Running Tests

```bash
# Run all tests
python -m unittest

# Run specific test file
python -m unittest tests.test_pipeline

# Run with verbose output
python -m unittest -v

# Run with coverage
coverage run -m unittest
coverage report
coverage html
```

## Test Files

- `test_pipeline.py`: Integration tests for the main pipeline
- `test_config.py`: Configuration loading and validation tests
- `test_cli_smoke.py`: Basic CLI functionality tests
- `test_heuristics.py`: Heuristic analysis tests
- `test_cli_analyze.py`: Analyze command tests
- `test_cli_integration.py`: End-to-end CLI integration tests

## Writing Tests

Follow these conventions:
1. Name test files with `test_` prefix
2. Use descriptive test method names
3. Include docstrings for test methods
4. Use setUp/tearDown for test fixtures
5. Mock external dependencies

## Test Coverage

We aim for >70% code coverage. Focus testing on:
- Critical business logic
- Data validation
- Error handling
- Edge cases