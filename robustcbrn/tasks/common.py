from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..utils.io import read_jsonl, shuffle_list


def _stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


CHOICES_ONLY_INSTRUCTION = (
    "Select the best answer from the options. No question context is provided."
)


def _answer_to_index(answer: str | int, n_choices: int) -> int:
    if isinstance(answer, int):
        return answer
    answer = answer.strip().upper()
    # Map A/B/C/D...
    if len(answer) == 1 and "A" <= answer <= "Z":
        idx = ord(answer) - ord("A")
        if 0 <= idx < n_choices:
            return idx
    raise ValueError(f"Unrecognized answer label: {answer}")


def _template_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_mcq_dataset(
    path: str | Path,
    shuffle_seed: int | None = None,
    max_items: int | None = None,
):
    """Load JSONL into an Inspect MemoryDataset if available, else a list of dicts.

    Each row in the JSONL must contain: question (str), choices (List[str]),
    answer (str letter or int index), id (str|int), metadata (optional dict).
    """
    # Path validation
    filepath = Path(path).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    if filepath.suffix not in {'.jsonl', '.json'}:
        raise ValueError(f"Expected .jsonl or .json file, got: {filepath.suffix}")

    # Path usage guidance: accept absolute or relative paths after resolving.
    # Avoid surprising project-bounds checks; rely on the file existence/type checks above.

    rows = []
    try:
        for row in read_jsonl(filepath):
            # Validate row structure
            if not isinstance(row, dict):
                raise ValueError(f"Invalid row format: expected dict, got {type(row).__name__}")

            q = row.get("question", "").strip()
            ch: list[str] = list(row.get("choices", []))

            # Validate required fields
            if not ch:
                raise ValueError(f"Row missing choices: {row.get('id', 'unknown')}")
            if "answer" not in row:
                raise ValueError(f"Row missing answer: {row.get('id', 'unknown')}")

            if shuffle_seed is not None:
                rid = str(row.get("id", "0"))
                ch = shuffle_list(ch, seed=(shuffle_seed + (_stable_int(rid) % 100000)))

            try:
                target = _answer_to_index(row.get("answer"), len(ch))
            except ValueError as e:
                raise ValueError(f"Invalid answer in row {row.get('id', 'unknown')}: {e}") from e

            rid = row.get("id")
            meta = row.get("metadata", {}) or {}
            rows.append({
                "input": q,
                "choices": ch,
                "target": target,
                "id": rid,
                "metadata": meta,
            })
            if max_items is not None and len(rows) >= max_items:
                break
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to parse JSONL file {filepath}: {e}") from e
    except Exception as e:
        if "Row" in str(e) or "row" in str(e):
            raise  # Re-raise our validation errors
        raise ValueError(f"Error loading dataset from {filepath}: {e}") from e

    # Validate non-empty dataset
    if not rows:
        raise ValueError(f"Dataset is empty or no valid rows found in {filepath}")

    # Try to package as Inspect dataset
    try:
        from inspect_ai.dataset import MemoryDataset, Sample  # type: ignore

        samples = [
            Sample(
                input=r["input"],
                choices=r["choices"],
                target=r["target"],
                id=str(r.get("id")),
                metadata=r.get("metadata", {}),
            )
            for r in rows
        ]
        return MemoryDataset(samples)
    except ImportError:
        # Inspect not available, return list of dicts
        return rows
    except Exception as e:
        # Log but don't fail - return raw data
        print(f"Warning: Could not create Inspect dataset: {e}")
        return rows


def to_choices_only(sample) -> dict:
    """Strip question stem; keep choices/target for choices-only evaluation.

    Returns a dict compatible with Inspect's MemoryDataset Sample constructor.
    """
    if hasattr(sample, "choices"):
        try:
            choices = list(sample.choices)
            target = int(sample.target)
            sid = getattr(sample, "id", None)
            meta = getattr(sample, "metadata", {}) or {}
        except Exception:
            # Fallback to dict-style access
            choices = list(sample["choices"])  # type: ignore[index]
            target = int(sample["target"])  # type: ignore[index]
            sid = sample.get("id")  # type: ignore[index]
            meta = sample.get("metadata", {}) or {}  # type: ignore[index]
    else:
        choices = list(sample["choices"])  # type: ignore[index]
        target = int(sample["target"])  # type: ignore[index]
        sid = sample.get("id")  # type: ignore[index]
        meta = sample.get("metadata", {}) or {}  # type: ignore[index]
    meta = {**meta, "choices_only": True}
    return {
        "input": CHOICES_ONLY_INSTRUCTION,
        "choices": choices,
        "target": target,
        "id": sid,
        "metadata": meta,
    }


def render_mcq_prompt(stem: str, choices: list[str], with_letters: bool = True) -> str:
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    options = (
        "\n".join([f"{letter}. {c}" for letter, c in zip(letters, choices, strict=False)])
        if with_letters
        else "\n".join(choices)
    )
    template = (
        "You are a careful assistant. Answer the multiple-choice question.\n"
        "Question: {{stem}}\nOptions:\n{{options}}\n"
        "Respond with the letter only."
    )
    rendered = template.replace("{{stem}}", stem).replace("{{options}}", options)
    return rendered


def render_choices_only_prompt(choices: list[str]) -> str:
    return render_mcq_prompt(CHOICES_ONLY_INSTRUCTION, choices)


def render_cloze(stem: str, choice: str) -> str:
    """Simple cloze template: prefer inline blank replacement, else QA style.

    - If the stem contains '_____' we replace it with the choice.
    - Otherwise we return a two-line prompt with explicit answer.
    """
    if "_____" in stem:
        return stem.replace("_____", choice)
    return f"{stem}\nAnswer: {choice}"


def template_fingerprint(template_text: str) -> str:
    return _template_hash(template_text)
