from __future__ import annotations

"""Deterministic, safe paraphrase utilities.

This module provides templatic rewrites to generate benign paraphrases of
question stems without adding content. The rewrites are conservative and
rule-based to ensure safety and determinism.

Functions return ordered lists to keep runs reproducible under a fixed seed.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

_RE_WS = re.compile(r"\s+")


def _norm_spaces(s: str) -> str:
    return _RE_WS.sub(" ", s).strip()


def _ensure_question_mark(s: str) -> str:
    s = s.rstrip()
    if s.endswith("?"):
        return s
    if s.endswith("."):
        return s[:-1] + "?"
    return s + "?"


def _which_of_following(s: str) -> str | None:
    # Replace a very common evaluative phrase with an equivalent variant.
    pat = re.compile(r"\bWhich of the following\b", flags=re.IGNORECASE)
    if pat.search(s):
        return pat.sub("Which of these", s)
    return None


def _most_likely(s: str) -> str | None:
    pat = re.compile(r"\bmost likely\b", flags=re.IGNORECASE)
    if pat.search(s):
        return pat.sub("most probable", s)
    return None


def _least_likely(s: str) -> str | None:
    pat = re.compile(r"\bleast likely\b", flags=re.IGNORECASE)
    if pat.search(s):
        return pat.sub("least probable", s)
    return None


def _cause_passive(s: str) -> str | None:
    # Simple active -> passive for patterns like "X causes Y" or "X results in Y".
    # Keep only when a clean two-clause match is detected.
    m = re.search(r"^(.*?\b)(causes|results in|leads to)\b(.*)$", s, flags=re.IGNORECASE)
    if not m:
        return None
    subj = _norm_spaces(m.group(1))
    verb = m.group(2).lower()
    obj = _norm_spaces(m.group(3))
    if not subj or not obj:
        return None
    if verb == "causes":
        return f"{obj} is caused by {subj}"
    if verb == "results in":
        return f"{obj} results from {subj}"
    if verb == "leads to":
        # Use a more natural passive construction
        return f"{obj} results from {subj}"
    return None


def _because_reorder(s: str) -> str | None:
    # Reorder "X because Y" -> "Because Y, X" while preserving punctuation.
    m = re.search(r"^(.*?),?\s+because\s+(.*?)[\.?]?$", s, flags=re.IGNORECASE)
    if not m:
        return None
    x = _norm_spaces(m.group(1))
    y = _norm_spaces(m.group(2))
    if not x or not y:
        return None
    return f"Because {y}, {x}"


def _determiner_soften(s: str) -> str | None:
    # Conservative determiner tweak at sentence start only.
    m = re.match(r"^(The|the)\s+", s)
    if m:
        return s.replace(m.group(0), "This ", 1)
    m2 = re.match(r"^(This|this)\s+", s)
    if m2:
        return s.replace(m2.group(0), "The ", 1)
    return None


Rewriter = Callable[[str], str | None]


_REWRITERS: list[Rewriter] = [
    _which_of_following,
    _most_likely,
    _least_likely,
    _cause_passive,
    _because_reorder,
    _determiner_soften,
]


@dataclass(frozen=True)
class Paraphrase:
    variant: str
    text: str


def generate_paraphrases(stem: str, k: int = 2) -> list[Paraphrase]:
    """Generate up to k deterministic paraphrases of a question stem.

    The first variant is always the normalized original labeled 'orig'.
    Following variants apply conservative rewrites in fixed order.

    Args:
        stem: The question stem to paraphrase. Must be non-empty.
        k: Maximum number of paraphrase variants (excluding orig). Must be >= 0.

    Returns:
        List with 'orig' followed by up to k paraphrases.

    Raises:
        ValueError: If stem is empty/None or k is negative.
    """
    # Input validation
    if not stem or stem is None:
        raise ValueError("Question stem cannot be empty or None")
    if not isinstance(stem, str):
        raise ValueError(f"Question stem must be string, got {type(stem).__name__}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    out: list[Paraphrase] = []
    base = _ensure_question_mark(_norm_spaces(stem))

    # Handle edge case of whitespace-only stem
    if not base or base == "?":
        raise ValueError("Question stem cannot be whitespace-only")

    out.append(Paraphrase("orig", base))

    # Early return if no paraphrases requested
    if k == 0:
        return out

    count = 0
    for _, fn in enumerate(_REWRITERS, start=1):
        if count >= k:
            break
        v = fn(base)
        if v is None:
            continue
        v = _ensure_question_mark(_norm_spaces(v))
        if v == base:
            continue
        count += 1
        out.append(Paraphrase(f"para{count}", v))
    return out


def generate_k_paraphrases(stem: str, k: int = 2) -> list[tuple[str, str]]:
    """Compatibility wrapper returning (variant, text) tuples."""
    return [(p.variant, p.text) for p in generate_paraphrases(stem, k=k)]
