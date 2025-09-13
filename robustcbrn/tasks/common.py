from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

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
    shuffle_seed: Optional[int] = None,
    max_items: Optional[int] = None,
):
    """Load JSONL into an Inspect MemoryDataset if available, else a list of dicts.

    Each row in the JSONL must contain: question (str), choices (List[str]),
    answer (str letter or int index), id (str|int), metadata (optional dict).
    """
    rows = []
    for row in read_jsonl(path):
        q = row.get("question", "").strip()
        ch: List[str] = list(row.get("choices", []))
        if shuffle_seed is not None:
            rid = str(row.get("id", "0"))
            ch = shuffle_list(ch, seed=(shuffle_seed + (_stable_int(rid) % 100000)))
        target = _answer_to_index(row.get("answer"), len(ch))
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
    except Exception:
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


def render_mcq_prompt(stem: str, choices: List[str], with_letters: bool = True) -> str:
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


def render_choices_only_prompt(choices: List[str]) -> str:
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
