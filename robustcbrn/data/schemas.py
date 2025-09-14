"""Data schemas for RobustCBRN."""

from typing import Any, NamedTuple, Sequence


class Question(NamedTuple):
    """Question data structure for CBRN evaluations."""
    id: str
    question: str
    choices: Sequence[str]
    answer: Any  # index | letter | choice text
