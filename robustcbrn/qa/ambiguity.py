"""Ambiguity/Unanswerable Detection Module.

This module provides safe, metadata-only heuristics to identify
ambiguous or unanswerable items based solely on their choices.
No stems are inspected to maintain safety boundaries.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from .ambiguity_config import AmbiguityConfig, DEFAULT_CONFIG

# Configure module logger
logger = logging.getLogger(__name__)

# Compiled regex patterns
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\s\.,;:!\?\-\(\)\[\]\{\}\"']+")
_NUM_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")


class AmbiguityDetectionError(Exception):
    """Base exception for ambiguity detection errors."""
    pass


class InvalidChoicesError(AmbiguityDetectionError):
    """Raised when choices are invalid or malformed."""
    pass


@dataclass
class AmbiguityDecision:
    """Decision result from ambiguity detection."""
    id: str
    label: str  # one of {clean, ambiguous, unanswerable}
    reason_codes: List[str]
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AmbiguityMetrics:
    """Performance metrics for ambiguity detection."""
    total_items: int = 0
    clean_count: int = 0
    ambiguous_count: int = 0
    unanswerable_count: int = 0
    total_processing_time_ms: float = 0.0

    @property
    def avg_processing_time_ms(self) -> float:
        """Average processing time per item."""
        return self.total_processing_time_ms / max(1, self.total_items)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_items": self.total_items,
            "clean_count": self.clean_count,
            "ambiguous_count": self.ambiguous_count,
            "unanswerable_count": self.unanswerable_count,
            "total_processing_time_ms": self.total_processing_time_ms,
            "avg_processing_time_ms": self.avg_processing_time_ms,
        }


def _norm_text(s: str) -> str:
    """Normalize text for comparison."""
    if not s:
        return ""
    s = s.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _tokens(s: str) -> List[str]:
    """Extract tokens from string."""
    return [t for t in _norm_text(s).split(" ") if t]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


def _has_meta_option(choice: str, config: AmbiguityConfig) -> Optional[str]:
    """Check if choice contains meta-option patterns."""
    s = _norm_text(choice)
    if not s:
        return None

    # Check against configured meta options
    for pattern, code in config.meta_options.items():
        if pattern in s:
            logger.debug(f"Found meta option '{pattern}' in choice: {choice[:50]}")
            return code

    return None


def _has_negation_pair(a: str, b: str, config: AmbiguityConfig) -> bool:
    """Check if two choices form a negation pair."""
    na = _norm_text(a)
    nb = _norm_text(b)

    # Check for explicit negation prefixes
    tokens_a = set(_tokens(na))
    tokens_b = set(_tokens(nb))

    # Check if one has negation prefix and other doesn't
    for prefix in config.negation_prefixes:
        if prefix in tokens_a and prefix not in tokens_b:
            # Check if rest of tokens match
            if tokens_a - {prefix} == tokens_b:
                logger.debug(f"Found negation pair: '{a}' vs '{b}'")
                return True
        if prefix in tokens_b and prefix not in tokens_a:
            if tokens_b - {prefix} == tokens_a:
                logger.debug(f"Found negation pair: '{a}' vs '{b}'")
                return True

    # Check for antonym pairs
    for word, antonym in config.negation_antonyms.items():
        if word in tokens_a and antonym in tokens_b:
            logger.debug(f"Found antonym pair: '{word}' vs '{antonym}'")
            return True
        if antonym in tokens_a and word in tokens_b:
            logger.debug(f"Found antonym pair: '{antonym}' vs '{word}'")
            return True

    # Legacy simple check for "not X" vs "X"
    if na.startswith("not ") and na[4:] == nb:
        return True
    if nb.startswith("not ") and nb[4:] == na:
        return True

    return False


def _is_numeric_like(s: str) -> bool:
    """Check if string contains numeric values."""
    return bool(_NUM_RE.search(s)) and len(_tokens(s)) <= 3


def _extract_numbers(s: str) -> List[float]:
    """Extract numeric values from string."""
    vals: List[float] = []
    for m in _NUM_RE.finditer(s):
        t = m.group(0).replace(",", ".")
        try:
            vals.append(float(t))
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse number '{t}': {e}")
            continue
    return vals


def _heuristics_for_item(
    choices: List[str],
    config: AmbiguityConfig = DEFAULT_CONFIG
) -> Tuple[str, List[str]]:
    """Return (label, reasons) using safe, metadata-only heuristics.

    This function never inspects stems; it only considers choices.

    Args:
        choices: List of choice options
        config: Configuration for detection thresholds

    Returns:
        Tuple of (label, list of reason codes)

    Raises:
        InvalidChoicesError: If choices are invalid
    """
    if not isinstance(choices, list):
        raise InvalidChoicesError(f"Choices must be a list, got {type(choices)}")

    reasons: List[str] = []

    # Handle edge cases
    if not choices:
        logger.debug("Empty choices list, marking as clean")
        return "clean", reasons

    if len(choices) == 1:
        logger.debug("Single choice, marking as clean")
        return "clean", reasons

    # 1) Meta options imply unanswerable without stem
    metas = [m for c in choices if (m := _has_meta_option(c, config))]
    if metas:
        reasons.extend(sorted(set(metas)))
        logger.info(f"Item marked unanswerable due to meta options: {reasons}")
        return "unanswerable", reasons

    # 2) Duplicate or near-duplicate choices → ambiguous
    norm = [_norm_text(c) for c in choices]
    seen: Dict[str, int] = {}
    dup = False

    # Check exact duplicates
    for i, s in enumerate(norm):
        if s in seen:
            dup = True
            reasons.append("duplicate_choices")
            logger.info(f"Found duplicate choices at indices {seen[s]} and {i}")
            break
        seen[s] = i

    # Check near-duplicates using Jaccard similarity
    if not dup:
        toks = [set(_tokens(c)) for c in choices]
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                if toks[i] and toks[j]:  # Skip empty token sets
                    similarity = _jaccard(toks[i], toks[j])
                    if similarity >= config.jaccard_threshold:
                        reasons.append("near_duplicate")
                        logger.info(f"Found near-duplicate (Jaccard={similarity:.2f}) at indices {i} and {j}")
                        dup = True
                        break
            if dup:
                break

    if dup:
        return "ambiguous", reasons

    # 3) Simple negation pairs across options → ambiguous
    for i in range(len(choices)):
        for j in range(i + 1, len(choices)):
            if _has_negation_pair(choices[i], choices[j], config):
                reasons.append("contradictory_options")
                logger.info(f"Found contradictory options at indices {i} and {j}")
                return "ambiguous", reasons

    # 4) Numeric crowding: multiple numbers very close together → ambiguous
    numeric_sets = [
        _extract_numbers(c) for c in choices if _is_numeric_like(c)
    ]
    flat = [x for xs in numeric_sets for x in xs]

    if len(flat) >= 2:
        flat_sorted = sorted(flat)
        for a, b in zip(flat_sorted, flat_sorted[1:]):
            base = max(1.0, abs(a))
            proximity = abs(a - b) / base
            if proximity <= config.numeric_proximity_threshold:
                reasons.append("numeric_too_close")
                logger.info(f"Found numeric crowding: {a} and {b} within {proximity*100:.1f}%")
                return "ambiguous", reasons

    # 5) Very short boolean-style choices → likely require stem
    token_lengths = [len(_tokens(c)) for c in choices if c.strip()]
    if token_lengths:
        short = all(length <= config.max_tokens_for_boolean for length in token_lengths)
        unique = len({s for s in norm if s})
        if short and unique >= 2:
            reasons.append("boolean_like_requires_stem")
            logger.info(f"Boolean-like choices detected (max tokens: {max(token_lengths)})")
            return "unanswerable", reasons

    logger.debug("Item passed all heuristics, marking as clean")
    return "clean", reasons


def audit_dataset(
    dataset: Iterable,
    config: AmbiguityConfig = DEFAULT_CONFIG,
    collect_metrics: bool = True
) -> Tuple[List[AmbiguityDecision], Optional[AmbiguityMetrics]]:
    """Run ambiguity heuristics on a dataset iterable.

    The dataset can be a list of dicts or Inspect MemoryDataset-like samples
    with attributes: id, choices.

    Args:
        dataset: Iterable of items with id and choices
        config: Configuration for detection
        collect_metrics: Whether to collect performance metrics

    Returns:
        Tuple of (decisions list, metrics object or None)

    Raises:
        AmbiguityDetectionError: If processing fails
    """
    decisions: List[AmbiguityDecision] = []
    metrics = AmbiguityMetrics() if collect_metrics else None

    logger.info("Starting dataset audit")
    start_time = time.time()

    for idx, s in enumerate(dataset):
        item_start = time.time()

        try:
            # Extract id and choices from various formats
            if hasattr(s, "choices"):
                sid = str(getattr(s, "id", f"item_{idx}"))
                choices = list(getattr(s, "choices", []))
            elif isinstance(s, dict):
                sid = str(s.get("id", f"item_{idx}"))
                choices = s.get("choices", [])
                if not isinstance(choices, list):
                    raise InvalidChoicesError(f"Item {sid}: choices must be a list")
            else:
                raise InvalidChoicesError(f"Invalid item format at index {idx}")

            # Run heuristics
            label, reasons = _heuristics_for_item(choices, config)

            # Calculate processing time
            processing_time_ms = (time.time() - item_start) * 1000

            # Create decision
            decision = AmbiguityDecision(
                id=sid,
                label=label,
                reason_codes=reasons,
                processing_time_ms=processing_time_ms
            )
            decisions.append(decision)

            # Update metrics
            if metrics:
                metrics.total_items += 1
                metrics.total_processing_time_ms += processing_time_ms
                if label == "clean":
                    metrics.clean_count += 1
                elif label == "ambiguous":
                    metrics.ambiguous_count += 1
                elif label == "unanswerable":
                    metrics.unanswerable_count += 1

        except Exception as e:
            logger.error(f"Failed to process item at index {idx}: {e}")
            raise AmbiguityDetectionError(f"Processing failed at item {idx}: {e}") from e

    total_time = (time.time() - start_time) * 1000
    logger.info(f"Audit complete: {len(decisions)} items in {total_time:.1f}ms")

    if metrics:
        logger.info(f"Metrics: {metrics.to_dict()}")

    return decisions, metrics


def decisions_to_records(decisions: List[AmbiguityDecision]) -> List[Dict[str, Any]]:
    """Convert decisions to record format for output."""
    rows: List[Dict[str, Any]] = []
    for d in decisions:
        record = {
            "id": d.id,
            "label": d.label,
            "reason_codes": ",".join(d.reason_codes) if d.reason_codes else "",
        }
        # Include processing time if available
        if d.processing_time_ms > 0:
            record["processing_time_ms"] = round(d.processing_time_ms, 2)
        # Include metadata if present
        if d.metadata:
            record["metadata"] = d.metadata
        rows.append(record)
    return rows


# Placeholder for optional LLM-critic integration (sanitized subset only)
def llm_critic_votes(*args, **kwargs):  # pragma: no cover - not implemented in OSS
    """LLM critic voting (disabled in OSS)."""
    raise NotImplementedError(
        "LLM critic is disabled in OSS by default. Use heuristics or provide a sanitized local critic."
    )