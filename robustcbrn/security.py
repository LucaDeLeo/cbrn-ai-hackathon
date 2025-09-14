"""Security utilities for RobustCBRN."""

import hashlib


def make_question_id(
    question_text: str,
    choices: list[str],
    salt: str | None = None
) -> str:
    """Generate a secure question ID using BLAKE2b hashing.

    Args:
        question_text: The question text
        choices: List of answer choices
        salt: Optional salt for additional security

    Returns:
        Hexadecimal hash string (32 characters)
    """
    # Normalize text for consistent hashing
    normalized_text = _normalize_text_for_hash(question_text, choices)

    # Add salt if provided
    if salt:
        normalized_text = f"{salt}:{normalized_text}"

    # Use BLAKE2b for secure hashing
    return hashlib.blake2b(
        normalized_text.encode('utf-8'),
        digest_size=16  # 16 bytes = 32 hex characters
    ).hexdigest()


def _normalize_text_for_hash(question_text: str, choices: list[str]) -> str:
    """Normalize text for consistent hashing."""
    # Convert to lowercase and strip whitespace
    normalized_question = question_text.lower().strip()
    normalized_choices = [choice.lower().strip() for choice in choices]

    # Sort choices for consistent ordering
    normalized_choices.sort()

    # Combine into single string
    return f"{normalized_question}|{'|'.join(normalized_choices)}"
