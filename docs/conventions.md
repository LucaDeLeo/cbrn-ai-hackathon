# Coding Conventions and Standards

This document outlines the coding standards and conventions for the RobustCBRN Eval project.

## Python Style Guide

### General Principles

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with modifications noted below
- Write clear, self-documenting code
- Prefer readability over cleverness
- Use type hints consistently
- Maximum line length: 100 characters

### Code Formatting

#### Imports

```python
# Standard library imports first
import os
import sys
from typing import List, Optional, Dict

# Third-party imports
import numpy as np
import torch
from transformers import AutoModel

# Local imports
from src.data.loader import load_dataset
from src.utils.logging import setup_logging
```

#### Indentation and Spacing

```python
# Use 4 spaces for indentation
def calculate_metrics(
    predictions: List[int],
    labels: List[int],
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Calculate evaluation metrics.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Single blank line between logical sections
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    metrics['accuracy'] = accuracy

    # Two blank lines between functions/classes
    return metrics


class ModelEvaluator:
    """Evaluates model performance."""

    def __init__(self, model_name: str):
        self.model_name = model_name
```

### Naming Conventions

#### Variables and Functions

```python
# Use snake_case for variables and functions
user_input = get_user_input()
validated_data = validate_input(user_input)

# Use descriptive names
# Good
def calculate_bootstrap_confidence_interval():
    pass

# Bad
def calc_bci():
    pass
```

#### Classes

```python
# Use PascalCase for classes
class DataLoader:
    pass

class QuestionValidator:
    pass
```

#### Constants

```python
# Use UPPER_SNAKE_CASE for constants
MAX_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
SUPPORTED_FORMATS = ['json', 'jsonl', 'csv']
```

#### Private Members

```python
class MyClass:
    def __init__(self):
        self._internal_state = {}  # Single underscore for internal use
        self.__private_attr = None  # Double underscore for name mangling (rare)

    def _internal_method(self):
        """Internal method, not part of public API."""
        pass
```

### Type Hints

Always use type hints for function signatures:

```python
from typing import List, Optional, Dict, Union, Tuple, Any
from pathlib import Path

def load_dataset(
    file_path: Union[str, Path],
    format: Optional[str] = None,
    validate: bool = True
) -> List[Dict[str, Any]]:
    """Load dataset from file.

    Args:
        file_path: Path to the dataset file
        format: File format (auto-detected if None)
        validate: Whether to validate the data

    Returns:
        List of dictionaries containing dataset records
    """
    pass

# For complex types, use type aliases
QuestionDict = Dict[str, Union[str, List[str], int]]
ResultsDict = Dict[str, Union[float, List[float]]]

def process_questions(
    questions: List[QuestionDict]
) -> ResultsDict:
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def analyze_bias(
    questions: List[Question],
    models: List[str],
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000
) -> BiasReport:
    """Analyze bias in model responses to questions.

    Performs comprehensive bias analysis including statistical tests,
    bootstrap confidence intervals, and pattern detection.

    Args:
        questions: List of Question objects to analyze
        models: List of model names to evaluate
        confidence_level: Confidence level for statistical tests (default: 0.95)
        bootstrap_samples: Number of bootstrap samples (default: 1000)

    Returns:
        BiasReport object containing analysis results

    Raises:
        ValueError: If questions list is empty
        ModelNotFoundError: If specified model cannot be loaded

    Example:
        >>> questions = load_dataset("data/test.jsonl")
        >>> report = analyze_bias(questions, ["gpt-3.5-turbo"])
        >>> print(report.summary())
    """
    if not questions:
        raise ValueError("Questions list cannot be empty")

    # Implementation...
    pass
```

### Error Handling

```python
# Use specific exceptions
class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass

# Provide informative error messages
def validate_question(question: Dict[str, Any]) -> None:
    if 'answer' not in question:
        raise DataValidationError(
            f"Question missing 'answer' field: {question.get('id', 'unknown')}"
        )

# Use context managers for resource management
def process_file(file_path: str) -> None:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise DataValidationError(f"Cannot parse JSON: {e}")
```

### Logging

```python
import logging

# Create logger at module level
logger = logging.getLogger(__name__)

def process_data(data: List[Dict]) -> None:
    logger.info(f"Processing {len(data)} records")

    for i, record in enumerate(data):
        try:
            # Process record
            logger.debug(f"Processing record {i}: {record['id']}")
        except Exception as e:
            logger.error(f"Failed to process record {i}: {e}")
            raise

    logger.info("Processing complete")
```

## Testing Conventions

### Test File Structure

```python
# tests/test_module_name.py
import unittest
from unittest.mock import Mock, patch

from src.module_name import function_to_test


class TestFunctionName(unittest.TestCase):
    """Test cases for function_name."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {"key": "value"}

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_normal_case(self):
        """Test function with normal input."""
        result = function_to_test(self.test_data)
        self.assertEqual(result, expected_value)

    def test_edge_case(self):
        """Test function with edge case input."""
        with self.assertRaises(ValueError):
            function_to_test(None)

    @patch('src.module_name.external_function')
    def test_with_mock(self, mock_func):
        """Test function with mocked dependency."""
        mock_func.return_value = "mocked"
        result = function_to_test(self.test_data)
        mock_func.assert_called_once()
```

### Test Naming

```python
# Test method names should be descriptive
def test_load_dataset_with_valid_json_file(self):
    pass

def test_load_dataset_raises_error_for_invalid_format(self):
    pass

def test_calculate_metrics_returns_correct_accuracy(self):
    pass
```

## Git Commit Conventions

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Test additions or changes
- **build**: Build system changes
- **ci**: CI configuration changes
- **chore**: Other changes

### Examples

```bash
feat(analysis): add bootstrap confidence interval calculation

Implements BCa bootstrap method for robust confidence intervals.
Includes both percentile and BCa methods with configurable samples.

Closes #123

---

fix(loader): handle missing answer field in CSV files

CSV files without explicit answer column now fall back to checking
for 'correct' or 'label' columns before raising an error.

---

docs: update installation instructions for Windows users

Added PowerShell and CMD specific instructions with troubleshooting
section for common Windows-specific issues.
```

## File Organization

### Module Structure

```python
# src/analysis/statistical.py

"""Statistical analysis module.

This module provides statistical analysis functions for evaluating
model performance and detecting biases in CBRN benchmarks.
"""

# Imports (sorted and grouped)
import logging
from typing import List, Dict, Optional

import numpy as np
from scipy import stats

from src.data.schemas import Question
from src.utils.logging import get_logger

# Module-level constants
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_SAMPLE_SIZE = 30

# Module-level variables
logger = get_logger(__name__)

# Classes (if any)
class StatisticalAnalyzer:
    """Performs statistical analysis on evaluation results."""
    pass

# Functions
def calculate_confidence_interval(
    data: List[float],
    confidence: float = DEFAULT_CONFIDENCE_LEVEL
) -> tuple[float, float]:
    """Calculate confidence interval for given data."""
    pass

# Module initialization (if needed)
def _initialize_module():
    """Initialize module-level resources."""
    pass

# Call initialization if needed
_initialize_module()
```

## Performance Guidelines

### Optimization Principles

```python
# Profile before optimizing
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()

    # Code to profile
    expensive_operation()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

# Use appropriate data structures
# Good: O(1) lookup
data_dict = {item.id: item for item in items}
result = data_dict.get(target_id)

# Bad: O(n) lookup
for item in items:
    if item.id == target_id:
        result = item
        break

# Batch operations when possible
# Good: Vectorized operation
results = model.predict(batch_inputs)

# Bad: Individual predictions
results = [model.predict(input) for input in inputs]
```

## Security Guidelines

### Input Validation

```python
def validate_file_path(path: str) -> Path:
    """Validate and sanitize file path."""
    path_obj = Path(path).resolve()

    # Check path traversal
    if ".." in path_obj.parts:
        raise SecurityError("Path traversal detected")

    # Check allowed directories
    if not path_obj.is_relative_to(ALLOWED_DATA_DIR):
        raise SecurityError("Access to path not allowed")

    return path_obj
```

### Sensitive Data Handling

```python
# Never log sensitive data
logger.info(f"Processing user {user_id}")  # Good
logger.info(f"Processing {user_data}")  # Bad if contains PII

# Use environment variables for secrets
import os
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

# Anonymize data when possible
from src.security.anonymizer import make_question_id
question_id = make_question_id(question_text, salt=PROJECT_SALT)
```

## Documentation Standards

### README Files

Each directory should have a README.md explaining:
- Purpose of the directory
- Key modules and their responsibilities
- Usage examples
- Dependencies

### Code Comments

```python
# Use comments sparingly, prefer self-documenting code
# Good: Explain why, not what
# Compensate for floating-point precision issues
epsilon = 1e-10
if abs(value) < epsilon:
    value = 0.0

# Bad: Redundant comment
# Set count to zero
count = 0

# TODO comments should include context
# TODO(username): Optimize this loop for large datasets (>1M records)
# See issue #456 for performance requirements
```

## Review Checklist

Before submitting code:

- [ ] Code follows PEP 8 and project conventions
- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] Tests written for new functionality
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] Error messages are informative
- [ ] Logging used appropriately
- [ ] Security considerations addressed
- [ ] Performance implications considered