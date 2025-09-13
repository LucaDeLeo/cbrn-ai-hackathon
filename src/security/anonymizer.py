from __future__ import annotations

import hashlib
from typing import Iterable


def make_question_id(question: str, choices: Iterable[str], salt: str = "") -> str:
    """Generate a stable hashed ID for a question.

    Uses BLAKE2b with a configurable salt. The input includes the question text
    and the choices in deterministic order.
    Returns a short hex digest for readability.
    """
    h = hashlib.blake2b(digest_size=16)
    if salt:
        h.update(salt.encode("utf-8"))
    h.update(question.strip().encode("utf-8"))
    for c in choices:
        h.update(b"|")
        h.update(str(c).strip().encode("utf-8"))
    return h.hexdigest()

