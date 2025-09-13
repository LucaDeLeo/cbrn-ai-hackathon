from __future__ import annotations

from typing import Dict, Any, List, Optional


LETTER_TO_INDEX = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
    "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5,
}


def normalize_answer(raw: Any, choices_len: int) -> int:
    """Normalize various answer formats to a 0-based index.

    Rules:
      - If str letter (A/B/...), map via LETTER_TO_INDEX
      - If int and in 1..choices_len, convert to 0-based by subtracting 1
      - If int and already 0-based, accept if in range
    """
    if isinstance(raw, str):
        raw = raw.strip()
        if raw in LETTER_TO_INDEX:
            idx = LETTER_TO_INDEX[raw]
        else:
            # maybe numeric in string
            if raw.isdigit():
                idx = int(raw)
            else:
                raise ValueError(f"Unrecognized answer format: {raw}")
    else:
        idx = int(raw)

    if 1 <= idx <= choices_len:
        idx = idx - 1

    if not (0 <= idx < choices_len):
        raise ValueError(f"Answer index out of range: {idx} for {choices_len} choices")
    return idx


def validate_record(obj: Dict[str, Any]) -> None:
    required = ["question", "choices", "answer"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing required field: {k}")
    if not isinstance(obj["choices"], list) or len(obj["choices"]) < 2:
        raise ValueError("'choices' must be a list of at least 2 items")


def apply_csv_mapping(row: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Map a CSV row dict into the internal schema using a deterministic mapping.

    mapping maps internal keys -> csv column names, e.g.:
      {
        "question": "question",
        "choices": ["choice_a", "choice_b", "choice_c", "choice_d"],
        "answer": "answer"
      }
    For choices, mapping may be a list of column names or a prefix string like "choice_".
    """
    out: Dict[str, Any] = {}

    # question
    q_key = mapping.get("question", "question")
    out["question"] = row[q_key]

    # choices
    choices_map = mapping.get("choices")
    if isinstance(choices_map, list):
        choices = [row[c] for c in choices_map]
    elif isinstance(choices_map, str):
        # treat as prefix, sort deterministically by column name
        choices = [v for k, v in sorted(row.items()) if k.startswith(choices_map)]
    else:
        # default heuristic common naming
        keys = [k for k in row.keys() if k.lower().startswith("choice_")]
        choices = [row[k] for k in sorted(keys)]
        if not choices:
            # fall back to columns named a,b,c,d
            keys = [k for k in row.keys() if k.lower() in {"a", "b", "c", "d"}]
            choices = [row[k] for k in sorted(keys)]

    out["choices"] = choices

    # answer
    a_key = mapping.get("answer", "answer")
    out["answer"] = row[a_key]

    # optional fields
    if "topic" in mapping and mapping["topic"] in row:
        out["topic"] = row[mapping["topic"]]
    if "difficulty" in mapping and mapping["difficulty"] in row:
        out["difficulty"] = row[mapping["difficulty"]]

    return out

