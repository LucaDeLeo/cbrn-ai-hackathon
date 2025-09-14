from __future__ import annotations

"""Cost and throughput heuristics used for budget projections.

Centralizes the default per-item hour constant so scripts and docs stay in sync.
"""

from dataclasses import dataclass


# Heuristic for 8B instruction-tuned models on A100 with bf16:
# ~25 items/sec per model-seed → ~0.000011 hours per item per model-seed
PER_ITEM_H_DEFAULT: float = 0.000011


def per_item_hours() -> float:
    """Return the default hours per item per model-seed heuristic.

    This value is intentionally centralized to avoid drift across scripts/docs.
    """

    return PER_ITEM_H_DEFAULT


def estimate_hours(models: int, seeds: int, items: int, per_item_h: float | None = None) -> float:
    """Estimate total hours for a workload.

    Args:
        models: number of models
        seeds: number of random seeds
        items: number of items per run (subset)
        per_item_h: override for hours per item per model-seed
    """

    h = per_item_h if per_item_h is not None else PER_ITEM_H_DEFAULT
    return float(models) * float(seeds) * float(items) * h


def suggest_subset(
    remaining_hours: float,
    models: int,
    seeds: int,
    safety: float = 0.9,
    per_item_h: float | None = None,
) -> int:
    """Suggest a subset size that fits within remaining hours.

    Uses a configurable safety factor to avoid overshooting the budget.
    Returns 0 when models or seeds are zero or no time remains.
    """

    if remaining_hours <= 0 or models <= 0 or seeds <= 0:
        return 0
    h = per_item_h if per_item_h is not None else PER_ITEM_H_DEFAULT
    denom = max(1, models * seeds)
    return int((remaining_hours * safety) / (denom * h))

