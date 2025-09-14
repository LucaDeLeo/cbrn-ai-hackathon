from __future__ import annotations

"""Deterministic perturbations for robustness checks.

Includes whitespace/punctuation tweaks, neutral preambles, and option order
swaps with target index remapping. These perturbations aim to be label-preserving
and safe, without introducing content beyond benign formatting.
"""

import re
from dataclasses import dataclass

_RE_MULTI_SPACE = re.compile(r"\s+")


def _norm_spaces(s: str) -> str:
    return _RE_MULTI_SPACE.sub(" ", s).strip()


def _toggle_trailing_punct(s: str) -> str:
    s = s.rstrip()
    if s.endswith("?"):
        return s[:-1] + "."
    if s.endswith("."):
        return s[:-1] + "?"
    return s + "."


def _add_neutral_preamble(s: str) -> str:
    s = _norm_spaces(s)
    return f"Please answer the following question. {s}"


def _add_space_after_punct(s: str) -> str:
    # Ensure a single space after commas/semicolons/colons if missing
    s = re.sub(r"([,;:])(\S)", r"\1 \2", s)
    return _norm_spaces(s)


def reorder_choices_swap_ends(choices: list[str], target_index: int) -> tuple[list[str], int]:
    """Swap first and last options; return new choices and remapped target index."""
    if not choices:
        return choices, target_index
    n = len(choices)
    if n == 1:
        return choices[:], target_index
    new = choices[:]
    new[0], new[-1] = new[-1], new[0]
    if target_index == 0:
        new_target = n - 1
    elif target_index == n - 1:
        new_target = 0
    else:
        new_target = target_index
    return new, new_target


def reorder_choices_reverse(choices: list[str], target_index: int) -> tuple[list[str], int]:
    """Reverse option order; return new choices and remapped target index."""
    n = len(choices)
    new = list(reversed(choices))
    if n == 0:
        return new, target_index
    new_target = n - 1 - target_index
    return new, new_target


@dataclass(frozen=True)
class Perturbation:
    variant: str
    stem: str
    choices: list[str]
    target_index: int
    kind: str


def generate_perturbations(
    stem: str, choices: list[str], target_index: int, k: int = 3
) -> list[Perturbation]:
    """Generate up to k deterministic, label-preserving perturbations.

    The original (unmodified) sample is included first with variant='orig'.
    Subsequent variants are applied in a fixed order from the following set:
      - 'punct': toggle trailing punctuation between '.' and '?'
      - 'space': ensure single-space after punctuation and normalize whitespace
      - 'preamble': prepend a neutral instruction sentence
      - 'order:swap_ends': swap first and last options (remap target)
      - 'order:reverse': reverse option list (remap target)

    Args:
        stem: The question stem. Must be non-empty.
        choices: List of answer choices. Must be non-empty.
        target_index: Index of correct answer. Must be valid for choices list.
        k: Maximum number of perturbation variants (excluding orig). Must be >= 0.

    Returns:
        List with 'orig' followed by up to k perturbations.

    Raises:
        ValueError: If inputs are invalid.
    """
    # Input validation
    if not stem or stem is None:
        raise ValueError("Question stem cannot be empty or None")
    if not isinstance(stem, str):
        raise ValueError(f"Question stem must be string, got {type(stem).__name__}")
    if not choices:
        raise ValueError("Choices list cannot be empty")
    if not isinstance(choices, list):
        raise ValueError(f"Choices must be a list, got {type(choices).__name__}")
    if not all(isinstance(c, str) for c in choices):
        raise ValueError("All choices must be strings")
    if target_index < 0 or target_index >= len(choices):
        raise ValueError(f"target_index {target_index} out of bounds for {len(choices)} choices")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    stem = _norm_spaces(stem)

    # Handle edge case of whitespace-only stem
    if not stem:
        raise ValueError("Question stem cannot be whitespace-only")

    out: list[Perturbation] = [
        Perturbation("orig", stem, list(choices), int(target_index), kind="orig")
    ]

    # Early return if no perturbations requested
    if k == 0:
        return out

    variants = []

    # Stem-only perturbations (do not affect choices/labels)
    variants.append(("pert:punct", _toggle_trailing_punct(stem), choices[:], target_index, "punct"))
    variants.append(("pert:space", _add_space_after_punct(stem), choices[:], target_index, "space"))
    variants.append(("pert:preamble", _add_neutral_preamble(stem), choices[:], target_index, "preamble"))

    # Choice order perturbations (affect labels)
    ch, ti = reorder_choices_swap_ends(choices, int(target_index))
    variants.append(("pert:order_swap", stem, ch, ti, "order:swap_ends"))
    ch2, ti2 = reorder_choices_reverse(choices, int(target_index))
    variants.append(("pert:order_rev", stem, ch2, ti2, "order:reverse"))

    # Deduplicate by (stem, choices) to avoid emitting no-ops
    seen = set()
    for name, st, chs, ti, kind in variants:
        key = (st, tuple(chs))
        if key in seen:
            continue
        seen.add(key)
        out.append(Perturbation(name, st, list(chs), int(ti), kind=kind))
        if len(out) - 1 >= k:  # exclude orig from k budget
            break
    return out

