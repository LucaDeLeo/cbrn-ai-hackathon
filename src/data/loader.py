from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import json
import csv

from .schemas import Question
from .validators import validate_record, normalize_answer, apply_csv_mapping
from ..security.anonymizer import make_question_id


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_csv(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_dataset(
    path: str | Path,
    fmt: Optional[str] = None,
    csv_mapping: Optional[Dict[str, Any]] = None,
    id_salt: str = "",
) -> List[Question]:
    """Load dataset from JSONL or CSV into internal Question objects.

    - Supports answer normalization (letter/int to 0-based index)
    - Generates stable hashed IDs with configurable salt
    - Returns items in deterministic order
    """
    p = Path(path)
    if fmt is None:
        if p.suffix.lower() == ".jsonl":
            fmt = "jsonl"
        elif p.suffix.lower() == ".csv":
            fmt = "csv"
        else:
            raise ValueError(f"Unknown dataset format for: {path}")

    records: Iterable[Dict[str, Any]]
    if fmt == "jsonl":
        records = _iter_jsonl(p)
    elif fmt == "csv":
        raw_rows = _iter_csv(p)
        if csv_mapping is None:
            # sensible default mapping
            csv_mapping = {
                "question": "question",
                "choices": ["choice_a", "choice_b", "choice_c", "choice_d"],
                "answer": "answer",
            }
        records = (apply_csv_mapping(row, csv_mapping) for row in raw_rows)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    out: List[Question] = []
    for obj in records:
        validate_record(obj)
        choices = list(obj["choices"])  # copy deterministically
        ans_idx = normalize_answer(obj["answer"], len(choices))
        qid = make_question_id(obj["question"], choices, salt=id_salt)

        out.append(
            Question(
                id=qid,
                question=str(obj["question"]),
                choices=[str(x) for x in choices],
                answer=ans_idx,
                topic=obj.get("topic"),
                difficulty=obj.get("difficulty"),
                metadata={k: v for k, v in obj.items() if k not in {"question", "choices", "answer", "topic", "difficulty"}},
            )
        )

    # Stable deterministic order by ID then question text
    out.sort(key=lambda x: (x.id, x.question))
    return out

