from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..utils.validation import (
    SchemaValidationError,
    validate_choices_only,
    validate_mcq_dataset,
)


def _validate(path: Path, schema: str) -> None:
    if schema == "mcq":
        validate_mcq_dataset(path)
    elif schema == "choices":
        validate_choices_only(path)
    elif schema == "both":
        # Validate both variants against their respective schemas
        # Many MCQ files are valid for both checks (choices-only ignores question/answer)
        validate_mcq_dataset(path)
        validate_choices_only(path)
    else:
        raise ValueError(f"Unknown schema '{schema}' (expected: mcq, choices, both)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m robustcbrn.cli.validate_dataset",
        description=(
            "Validate MCQ/choices-only dataset JSONL files before evaluation.\n"
            "On failure, prints actionable guidance and a docs pointer."
        ),
    )
    ap.add_argument(
        "dataset",
        help="Path to JSONL dataset file",
    )
    ap.add_argument(
        "--schema",
        choices=["mcq", "choices", "both"],
        default="both",
        help="Which schema(s) to validate against (default: both)",
    )
    args = ap.parse_args(argv)

    dataset_path = Path(args.dataset)
    try:
        _validate(dataset_path, args.schema)
        print(f"[validate_dataset] OK: {dataset_path} matches {args.schema} schema")
        return 0
    except FileNotFoundError:
        print(f"[validate_dataset] Error: dataset not found: {dataset_path}")
        return 1
    except SchemaValidationError as e:
        # Dedicated exit code for schema failures to distinguish from other errors
        print("[validate_dataset] Schema validation failed.")
        print(f"[validate_dataset] {e}")
        print(
            "[validate_dataset] See docs/USAGE.md#dataset-schema for required fields and examples."
        )
        return 4
    except Exception as e:
        print(f"[validate_dataset] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
