"""Schema validation utilities for datasets and configurations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class SchemaValidationError(ValidationError):
    """Raised when data doesn't match expected schema."""
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


@dataclass
class FieldSpec:
    """Specification for a data field."""
    name: str
    type: type
    required: bool = True
    nullable: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    choices: Optional[List[Any]] = None
    validator: Optional[callable] = None


@dataclass
class DatasetSchema:
    """Schema definition for dataset validation."""
    name: str
    fields: List[FieldSpec]
    allow_extra_fields: bool = True
    min_records: Optional[int] = None
    max_records: Optional[int] = None


# Predefined schemas for RobustCBRN datasets
BENIGN_PAIRS_SCHEMA = DatasetSchema(
    name="benign_pairs",
    fields=[
        FieldSpec(name="id", type=str, required=True, nullable=False),
        FieldSpec(name="prompt_safe", type=str, required=True, nullable=False, min_length=1),
        FieldSpec(name="prompt_unsafe", type=str, required=True, nullable=False, min_length=1),
        FieldSpec(name="metadata", type=dict, required=False, nullable=True),
    ],
    allow_extra_fields=False,
    min_records=1
)

MCQ_DATASET_SCHEMA = DatasetSchema(
    name="mcq_dataset",
    fields=[
        FieldSpec(name="id", type=(str, int), required=True),
        FieldSpec(name="question", type=str, required=True, min_length=1),
        FieldSpec(name="choices", type=list, required=True, min_length=2),
        FieldSpec(name="answer", type=(str, int), required=True),
        FieldSpec(name="metadata", type=dict, required=False, nullable=True),
    ],
    allow_extra_fields=True,
    min_records=1
)

CHOICES_ONLY_SCHEMA = DatasetSchema(
    name="choices_only",
    fields=[
        FieldSpec(name="id", type=(str, int), required=True),
        FieldSpec(name="choices", type=list, required=True, min_length=2),
    ],
    allow_extra_fields=True,
    min_records=1
)


class SchemaValidator:
    """Validator for dataset schemas."""

    def __init__(self, schema: DatasetSchema):
        """Initialize validator with schema.

        Args:
            schema: Dataset schema to validate against
        """
        self.schema = schema

    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate a single record against schema.

        Args:
            record: Record to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        for field_spec in self.schema.fields:
            if field_spec.required and field_spec.name not in record:
                errors.append(f"Missing required field: {field_spec.name}")
                continue

            if field_spec.name in record:
                value = record[field_spec.name]

                # Check nullable
                if value is None:
                    if not field_spec.nullable:
                        errors.append(f"Field {field_spec.name} cannot be null")
                    continue

                # Check type
                expected_types = field_spec.type if isinstance(field_spec.type, tuple) else (field_spec.type,)
                if not any(isinstance(value, t) for t in expected_types):
                    errors.append(
                        f"Field {field_spec.name} has wrong type: expected {field_spec.type}, "
                        f"got {type(value).__name__}"
                    )
                    continue

                # String-specific validations
                if isinstance(value, str):
                    if field_spec.min_length and len(value) < field_spec.min_length:
                        errors.append(
                            f"Field {field_spec.name} too short: minimum {field_spec.min_length} chars"
                        )
                    if field_spec.max_length and len(value) > field_spec.max_length:
                        errors.append(
                            f"Field {field_spec.name} too long: maximum {field_spec.max_length} chars"
                        )
                    if field_spec.pattern:
                        import re
                        if not re.match(field_spec.pattern, value):
                            errors.append(
                                f"Field {field_spec.name} doesn't match pattern: {field_spec.pattern}"
                            )

                # List-specific validations
                if isinstance(value, list):
                    if field_spec.min_length and len(value) < field_spec.min_length:
                        errors.append(
                            f"Field {field_spec.name} has too few items: minimum {field_spec.min_length}"
                        )
                    if field_spec.max_length and len(value) > field_spec.max_length:
                        errors.append(
                            f"Field {field_spec.name} has too many items: maximum {field_spec.max_length}"
                        )

                # Check choices
                if field_spec.choices and value not in field_spec.choices:
                    errors.append(
                        f"Field {field_spec.name} has invalid value: must be one of {field_spec.choices}"
                    )

                # Custom validator
                if field_spec.validator:
                    try:
                        if not field_spec.validator(value):
                            errors.append(f"Field {field_spec.name} failed custom validation")
                    except Exception as e:
                        errors.append(f"Field {field_spec.name} validation error: {e}")

        # Check for extra fields
        if not self.schema.allow_extra_fields:
            expected_fields = {f.name for f in self.schema.fields}
            extra_fields = set(record.keys()) - expected_fields
            if extra_fields:
                errors.append(f"Unexpected fields: {', '.join(extra_fields)}")

        return errors

    def validate_dataset(self, records: List[Dict[str, Any]]) -> None:
        """Validate entire dataset.

        Args:
            records: List of records to validate

        Raises:
            SchemaValidationError: If validation fails
        """
        all_errors = []

        # Check record count
        if self.schema.min_records and len(records) < self.schema.min_records:
            all_errors.append(f"Dataset has too few records: minimum {self.schema.min_records}")

        if self.schema.max_records and len(records) > self.schema.max_records:
            all_errors.append(f"Dataset has too many records: maximum {self.schema.max_records}")

        # Validate each record
        for i, record in enumerate(records):
            record_errors = self.validate_record(record)
            if record_errors:
                all_errors.extend([f"Record {i}: {e}" for e in record_errors])

        if all_errors:
            raise SchemaValidationError(
                f"Dataset validation failed with {len(all_errors)} errors",
                errors=all_errors
            )

    def validate_file(self, filepath: Union[str, Path]) -> None:
        """Validate a JSONL dataset file.

        Args:
            filepath: Path to JSONL file

        Raises:
            SchemaValidationError: If validation fails
            FileNotFoundError: If file doesn't exist
            JSONDecodeError: If file contains invalid JSON
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        records = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    raise SchemaValidationError(
                        f"Invalid JSON at line {line_num}: {e}"
                    )

        self.validate_dataset(records)


def validate_benign_pairs(filepath: Union[str, Path]) -> None:
    """Validate a benign pairs dataset file.

    Args:
        filepath: Path to dataset file

    Raises:
        SchemaValidationError: If validation fails
    """
    validator = SchemaValidator(BENIGN_PAIRS_SCHEMA)
    validator.validate_file(filepath)
    logger.info(f"Benign pairs dataset validation passed: {filepath}")


def validate_mcq_dataset(filepath: Union[str, Path]) -> None:
    """Validate an MCQ dataset file.

    Args:
        filepath: Path to dataset file

    Raises:
        SchemaValidationError: If validation fails
    """
    validator = SchemaValidator(MCQ_DATASET_SCHEMA)
    validator.validate_file(filepath)
    logger.info(f"MCQ dataset validation passed: {filepath}")


def validate_choices_only(filepath: Union[str, Path]) -> None:
    """Validate a choices-only dataset file.

    Args:
        filepath: Path to dataset file

    Raises:
        SchemaValidationError: If validation fails
    """
    validator = SchemaValidator(CHOICES_ONLY_SCHEMA)
    validator.validate_file(filepath)
    logger.info(f"Choices-only dataset validation passed: {filepath}")


# Custom validators for specific fields
def validate_answer_format(answer: Union[str, int], num_choices: int) -> bool:
    """Validate MCQ answer format.

    Args:
        answer: Answer value
        num_choices: Number of choices

    Returns:
        True if valid
    """
    if isinstance(answer, int):
        return 0 <= answer < num_choices
    elif isinstance(answer, str):
        # Check if it's a letter (A, B, C, etc.)
        if len(answer) == 1 and answer.isalpha():
            return ord(answer.upper()) - ord('A') < num_choices
        # Check if it's a number string
        try:
            idx = int(answer)
            return 0 <= idx < num_choices
        except ValueError:
            return False
    return False


def validate_pair_consistency(safe_prompt: str, unsafe_prompt: str) -> bool:
    """Validate that benign pair prompts are properly formed.

    Args:
        safe_prompt: Safe variant prompt
        unsafe_prompt: Unsafe variant prompt

    Returns:
        True if valid
    """
    # Basic checks
    if not safe_prompt or not unsafe_prompt:
        return False

    # Check they're different
    if safe_prompt.strip().lower() == unsafe_prompt.strip().lower():
        return False

    # Check unsafe doesn't contain specific dangerous patterns
    # (This is a simplified check - real implementation would be more sophisticated)
    dangerous_patterns = [
        "step-by-step",
        "instructions",
        "procedure",
        "method",
        "technique"
    ]

    unsafe_lower = unsafe_prompt.lower()
    if any(pattern in unsafe_lower for pattern in dangerous_patterns):
        logger.warning("Unsafe prompt may contain operational details")

    return True


# Configuration validation
@dataclass
class ConfigSchema:
    """Schema for configuration validation."""
    name: str
    required_keys: List[str] = field(default_factory=list)
    optional_keys: List[str] = field(default_factory=list)
    validators: Dict[str, callable] = field(default_factory=dict)


def validate_config(config: Dict[str, Any], schema: ConfigSchema) -> None:
    """Validate configuration against schema.

    Args:
        config: Configuration dictionary
        schema: Configuration schema

    Raises:
        ValidationError: If validation fails
    """
    errors = []

    # Check required keys
    for key in schema.required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: {key}")

    # Check for unknown keys
    all_keys = set(schema.required_keys + schema.optional_keys)
    unknown_keys = set(config.keys()) - all_keys
    if unknown_keys:
        logger.warning(f"Unknown configuration keys: {', '.join(unknown_keys)}")

    # Run validators
    for key, validator in schema.validators.items():
        if key in config:
            try:
                if not validator(config[key]):
                    errors.append(f"Configuration key {key} failed validation")
            except Exception as e:
                errors.append(f"Error validating {key}: {e}")

    if errors:
        raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")


# Integration with existing loaders
def load_and_validate_benign_pairs(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load and validate benign pairs dataset.

    Args:
        filepath: Path to dataset file

    Returns:
        List of validated records

    Raises:
        SchemaValidationError: If validation fails
    """
    filepath = Path(filepath)

    # First validate schema
    validate_benign_pairs(filepath)

    # Then load the data
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Additional semantic validation
    for record in records:
        if not validate_pair_consistency(
            record.get("prompt_safe", ""),
            record.get("prompt_unsafe", "")
        ):
            logger.warning(f"Potential issue with pair consistency in record {record.get('id')}")

    return records