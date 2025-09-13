# RobustCBRN Eval Makefile
# Common development commands

.PHONY: help install install-dev test lint format clean run setup check-python

# Default target
help:
	@echo "RobustCBRN Eval - Available commands:"
	@echo ""
	@echo "  make setup        - Complete environment setup with uv"
	@echo "  make install      - Install core dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Remove build artifacts and caches"
	@echo "  make run          - Run sample analysis"
	@echo ""
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make type-check   - Run mypy type checking"
	@echo "  make security     - Run security checks"
	@echo ""

# Check Python version
check-python:
	@python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" || \
		(echo "Error: Python 3.10+ required" && exit 1)

# Complete setup
setup: check-python
	@echo "Setting up RobustCBRN Eval environment..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv is installed"; \
	else \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv venv --python python3.10
	@echo "Activate with: source .venv/bin/activate"
	. .venv/bin/activate && uv pip install -r requirements.txt
	@echo "Setup complete!"

# Install dependencies
install:
	uv pip install -r requirements.txt

install-dev: install
	uv pip install -r requirements-dev.txt

# Testing
test:
	python -m unittest discover -s tests -p "test_*.py" -v

test-unit:
	python -m unittest discover -s tests -p "test_*.py" -v --failfast

test-cov:
	coverage run -m unittest discover -s tests -p "test_*.py"
	coverage report -m
	coverage html
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality
lint:
	@echo "Running ruff..."
	ruff check src/ tests/
	@echo "Running mypy..."
	mypy src/

format:
	black src/ tests/ cli.py --line-length 100
	ruff check src/ tests/ --fix

type-check:
	mypy src/ --strict

# Security scanning
security:
	@echo "Running bandit security scan..."
	-bandit -r src/ -f json -o security_report.json
	@echo "Running safety check..."
	-safety check --json

# Clean artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "Clean complete!"

clean-cache:
	@echo "Cleaning cache directory..."
	rm -rf cache/*
	@echo "Cache cleaned!"

clean-logs:
	@echo "Cleaning logs directory..."
	rm -rf logs/*
	@echo "Logs cleaned!"

clean-results:
	@echo "Cleaning results directory..."
	rm -rf results/*
	@echo "Results cleaned!"

clean-all: clean clean-cache clean-logs clean-results
	@echo "All artifacts cleaned!"

# Run sample analysis
run:
	python cli.py load data/wmdp_bio_sample_100.jsonl --config configs/default.json

run-analyze:
	python cli.py analyze data/wmdp_bio_sample_100.jsonl --output results/analysis.json

# Development helpers
shell:
	ipython

watch:
	@echo "Watching for file changes..."
	@while true; do \
		inotifywait -e modify -r src/ tests/ 2>/dev/null && make test; \
	done

# Pre-commit checks
pre-commit: format lint test
	@echo "Pre-commit checks passed!"

# CI simulation
ci: clean install-dev lint type-check test-cov
	@echo "CI pipeline complete!"

# Docker operations (future)
docker-build:
	@echo "Docker support coming soon..."
	# docker build -t robustcbrn-eval .

docker-run:
	@echo "Docker support coming soon..."
	# docker run -it --rm robustcbrn-eval

# Documentation
docs:
	@echo "Documentation generation coming soon..."
	# sphinx-build -b html docs/ docs/_build/

# Release
version:
	@python -c "import json; print(json.load(open('pyproject.toml'))['project']['version'])" 2>/dev/null || echo "0.1.0"

release-check: clean-all install-dev ci
	@echo "Ready for release!"