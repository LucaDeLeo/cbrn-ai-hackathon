"""Data loading utilities for RobustCBRN."""

from pathlib import Path
from typing import List, Optional, Union

from .schemas import Question
from ..utils.io import read_jsonl, shuffle_list


def _stable_int(s: str) -> int:
    """Convert string to stable integer using hash."""
    import hashlib
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _answer_to_index(answer: Union[str, int], n_choices: int) -> int:
    """Convert answer to 0-based index."""
    if isinstance(answer, int):
        return answer
    answer = answer.strip().upper()
    # Map A/B/C/D...
    if len(answer) == 1 and "A" <= answer <= "Z":
        idx = ord(answer) - ord("A")
        if 0 <= idx < n_choices:
            return idx
    raise ValueError(f"Unrecognized answer label: {answer}")


def load_dataset(
    path: Union[str, Path],
    shuffle_seed: Optional[int] = None,
    max_items: Optional[int] = None,
) -> List[Question]:
    """Load dataset from JSONL file and return list of Question objects.
    
    Args:
        path: Path to JSONL file
        shuffle_seed: Optional seed for shuffling choices
        max_items: Optional limit on number of items to load
        
    Returns:
        List of Question objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or data is malformed
    """
    filepath = Path(path).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    if filepath.suffix not in {'.jsonl', '.json'}:
        raise ValueError(f"Expected .jsonl or .json file, got: {filepath.suffix}")

    questions = []
    try:
        for row in read_jsonl(filepath):
            # Validate row structure
            if not isinstance(row, dict):
                raise ValueError(f"Invalid row format: expected dict, got {type(row).__name__}")

            question_text = row.get("question", "").strip()
            choices: List[str] = list(row.get("choices", []))

            # Validate required fields
            if not choices:
                raise ValueError(f"Row missing choices: {row.get('id', 'unknown')}")
            if "answer" not in row:
                raise ValueError(f"Row missing answer: {row.get('id', 'unknown')}")

            # Shuffle choices if requested
            if shuffle_seed is not None:
                rid = str(row.get("id", "0"))
                choices = shuffle_list(choices, seed=(shuffle_seed + (_stable_int(rid) % 100000)))

            # Convert answer to index
            try:
                answer_index = _answer_to_index(row.get("answer"), len(choices))
            except ValueError as e:
                raise ValueError(f"Invalid answer in row {row.get('id', 'unknown')}: {e}")

            # Create Question object
            question = Question(
                id=str(row.get("id", "")),
                question=question_text,
                choices=choices,
                answer=answer_index
            )
            questions.append(question)
            
            if max_items is not None and len(questions) >= max_items:
                break
                
    except Exception as e:
        if "Row" in str(e) or "row" in str(e):
            raise  # Re-raise our validation errors
        raise ValueError(f"Error loading dataset from {filepath}: {e}")

    # Validate non-empty dataset
    if not questions:
        raise ValueError(f"Dataset is empty or no valid rows found in {filepath}")

    return questions
