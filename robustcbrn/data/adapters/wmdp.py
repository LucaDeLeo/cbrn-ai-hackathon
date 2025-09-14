"""Adapter for WMDP datasets."""

import json
from pathlib import Path
from typing import Any, Dict, List

# CSV support for potential future use
import csv


def convert_wmdp_parquet_to_jsonl(raw_dir: Path, out_dir: Path) -> Path:
    """
    Convert WMDP Parquet format to RobustCBRN JSONL schema.

    Expected WMDP Parquet columns (based on HuggingFace dataset):
    - question: The question text
    - choices: List of answer choices
    - answer: The correct answer index (0-based)

    Output JSONL schema:
    - id: Unique identifier
    - question: Question text
    - choices: List of choice strings
    - answer: Answer as letter (A, B, C, D)
    - metadata: Additional metadata dict
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to read Parquet files. Install with: pip install pandas pyarrow")

    # Find Parquet file in raw directory
    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No Parquet files found in {raw_dir}")

    input_parquet = parquet_files[0]
    output_jsonl = out_dir / "eval.jsonl"

    print(f"Converting {input_parquet} -> {output_jsonl}")

    # Read Parquet file
    df = pd.read_parquet(input_parquet)

    items = []
    for idx, row in df.iterrows():
        # Parse the row based on WMDP Parquet format
        item = parse_wmdp_parquet_row(row, idx)
        items.append(item)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write to JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(items)} questions")
    return output_jsonl


def parse_wmdp_parquet_row(row: Any, idx: int) -> Dict[str, Any]:
    """
    Parse a single WMDP Parquet row into RobustCBRN format.

    WMDP Parquet format (from HuggingFace):
    - question: string
    - choices: list of strings
    - answer: integer (0-based index)
    """
    # Generate unique ID
    item_id = f"wmdp_{idx:05d}"

    # Extract question
    question = str(row.get("question", ""))

    # Extract choices (already a list/array in Parquet)
    choices = row.get("choices", [])

    # Handle numpy arrays (common in pandas DataFrames)
    if hasattr(choices, 'tolist'):
        choices = choices.tolist()
    elif not isinstance(choices, list):
        # If stored as string representation, try to parse
        import ast
        try:
            choices = ast.literal_eval(str(choices))
        except:
            choices = str(choices).split("|")  # Fallback to delimiter

    # Ensure choices are strings
    choices = [str(c).strip() for c in choices]

    # Extract answer (convert 0-based index to letter)
    answer_idx = row.get("answer", 0)
    if isinstance(answer_idx, (int, float)):
        answer = chr(65 + int(answer_idx))  # 0->A, 1->B, etc.
    else:
        # Try to parse if it's a string
        try:
            answer = chr(65 + int(answer_idx))
        except:
            answer = "A"  # Fallback

    # Build metadata from other fields
    metadata = {}
    # Add any additional columns as metadata
    for key in row.index:
        if key not in ["question", "choices", "answer"] and row[key] is not None:
            metadata[key] = str(row[key])

    return {
        "id": item_id,
        "question": question.strip(),
        "choices": choices,
        "answer": answer,
        "metadata": metadata
    }


def convert_wmdp_to_jsonl(raw_dir: Path, out_dir: Path) -> Path:
    """
    Convert WMDP CSV format to RobustCBRN JSONL schema.

    Expected WMDP CSV columns:
    - question: The question text
    - choices: The answer choices (as a list or string to parse)
    - answer: The correct answer (letter or index)
    - topic/category: Metadata fields

    Output JSONL schema:
    - id: Unique identifier
    - question: Question text
    - choices: List of choice strings
    - answer: Answer as letter (A, B, C, D)
    - metadata: Additional metadata dict
    """
    # Find CSV file in raw directory
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {raw_dir}")

    input_csv = csv_files[0]
    output_jsonl = out_dir / "eval.jsonl"

    print(f"Converting {input_csv} -> {output_jsonl}")

    items = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, 1):
            # Parse the row based on WMDP format
            # This is a template - adjust based on actual WMDP CSV structure
            item = parse_wmdp_row(row, idx)
            items.append(item)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write to JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(items)} questions")
    return output_jsonl


def parse_wmdp_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Parse a single WMDP CSV row into RobustCBRN format.

    This is a template function - adjust based on actual WMDP CSV columns.
    Common WMDP formats include:
    - Separate columns for choice_a, choice_b, choice_c, choice_d
    - Combined choices column with delimiter
    - Answer as letter (A, B, C, D) or index (0, 1, 2, 3)
    """
    # Generate unique ID
    item_id = f"wmdp_{idx:05d}"

    # Extract question (adjust column name as needed)
    question = row.get("question", row.get("prompt", row.get("text", "")))

    # Extract choices - try different possible formats
    choices = extract_choices(row)

    # Extract answer - normalize to letter format
    answer = normalize_answer(row.get("answer", row.get("correct", "")), len(choices))

    # Build metadata from other fields
    metadata = {}
    for key in ["topic", "category", "subcategory", "difficulty", "domain"]:
        if key in row and row[key]:
            metadata[key] = row[key]

    return {
        "id": item_id,
        "question": question.strip(),
        "choices": choices,
        "answer": answer,
        "metadata": metadata
    }


def extract_choices(row: Dict[str, Any]) -> List[str]:
    """Extract choices from various possible formats."""
    # Format 1: Separate columns (choice_a, choice_b, etc.)
    if "choice_a" in row:
        choices = []
        for letter in ["a", "b", "c", "d", "e", "f"]:
            col = f"choice_{letter}"
            if col in row and row[col]:
                choices.append(row[col].strip())
        return choices

    # Format 2: Choices as JSON list in single column
    if "choices" in row:
        choices_raw = row["choices"]
        if choices_raw.startswith("["):
            import ast
            try:
                return ast.literal_eval(choices_raw)
            except:
                pass

    # Format 3: Numbered columns (1, 2, 3, 4 or A, B, C, D)
    if "A" in row:
        choices = []
        for letter in ["A", "B", "C", "D", "E", "F"]:
            if letter in row and row[letter]:
                choices.append(row[letter].strip())
        return choices

    # Format 4: Tab or pipe separated in choices column
    if "choices" in row:
        choices_raw = row["choices"]
        if "\t" in choices_raw:
            return [c.strip() for c in choices_raw.split("\t") if c.strip()]
        if "|" in choices_raw:
            return [c.strip() for c in choices_raw.split("|") if c.strip()]

    # Fallback: look for any column with "option" or "choice"
    choices = []
    for key, value in row.items():
        if ("option" in key.lower() or "choice" in key.lower()) and value:
            choices.append(value.strip())

    if not choices:
        raise ValueError(f"Could not extract choices from row: {row}")

    return choices


def normalize_answer(answer: str, num_choices: int) -> str:
    """Normalize answer to letter format (A, B, C, D)."""
    answer = str(answer).strip()

    # Already a letter
    if answer in ["A", "B", "C", "D", "E", "F"]:
        return answer

    # Lowercase letter
    if answer in ["a", "b", "c", "d", "e", "f"]:
        return answer.upper()

    # Numeric index (0-based)
    if answer in ["0", "1", "2", "3", "4", "5"]:
        idx = int(answer)
        if idx < num_choices:
            return chr(65 + idx)  # Convert to A, B, C, etc.

    # Numeric index (1-based)
    if answer in ["1", "2", "3", "4", "5", "6"]:
        idx = int(answer) - 1
        if idx < num_choices:
            return chr(65 + idx)

    # Try to parse as integer
    try:
        idx = int(answer)
        if 0 <= idx < num_choices:
            return chr(65 + idx)
        elif 1 <= idx <= num_choices:
            return chr(65 + idx - 1)
    except ValueError:
        pass

    raise ValueError(f"Could not normalize answer '{answer}' for {num_choices} choices")