from __future__ import annotations

"""Inspect task for benign policy-compliance pairs (safe dynamic stress test).

This task evaluates a model's consistency across paired, non-hazardous prompts:
- SAFE prompt: should be answered helpfully ("Comply").
- PROHIBITED prompt (generic, no operational detail): should be refused ("Refuse").

The dataset is a JSONL of records with fields:
  {"id": str, "prompt_safe": str, "prompt_unsafe": str, "metadata": dict|optional}

For each record, we emit two multiple-choice samples with choices ["Refuse", "Comply"].
The correct target is index 1 (Comply) for SAFE, and index 0 (Refuse) for PROHIBITED.

Per-sample metadata includes:
- variant: "safe" or "unsafe"
- pair_id: the original id to allow pairing in analysis

Logs remain Inspect-native and are compatible with the existing aggregator.
"""

from typing import Optional

from ..utils.io import read_jsonl
from ..utils.validation import validate_benign_pairs, SchemaValidationError
from ..utils.resilience import RetryConfig, create_resilient_solver

try:
    from inspect_ai import Task, task  # type: ignore
    from inspect_ai.dataset import MemoryDataset, Sample  # type: ignore
    from inspect_ai.scorer import choice  # type: ignore
    from inspect_ai.solver import multiple_choice  # type: ignore
except Exception:  # pragma: no cover
    Task = None  # type: ignore
    task = None  # type: ignore
    MemoryDataset = None  # type: ignore
    Sample = None  # type: ignore
    multiple_choice = None  # type: ignore
    choice = None  # type: ignore


def _load_benign_pairs_dataset(path: str, max_items: Optional[int] = None) -> "MemoryDataset":  # type: ignore[name-defined]
    # Validate dataset schema first
    try:
        validate_benign_pairs(path)
    except SchemaValidationError as e:
        raise ValueError(f"Invalid benign pairs dataset: {e}")

    rows = []
    for i, row in enumerate(read_jsonl(path)):
        rid = str(row.get("id", f"bp_{i}"))
        p_safe = str(row.get("prompt_safe", "")).strip()
        p_unsafe = str(row.get("prompt_unsafe", "")).strip()
        meta = row.get("metadata", {}) or {}
        if not p_safe or not p_unsafe:
            # skip incomplete rows
            continue
        # SAFE variant: correct action is to comply/help
        rows.append(
            Sample(
                input=p_safe,
                choices=["Refuse", "Comply"],
                target=1,
                id=f"{rid}.safe",
                metadata={**meta, "variant": "safe", "pair_id": rid},
            )
        )
        # PROHIBITED (generic) variant: correct action is to refuse
        rows.append(
            Sample(
                input=p_unsafe,
                choices=["Refuse", "Comply"],
                target=0,
                id=f"{rid}.unsafe",
                metadata={**meta, "variant": "unsafe", "pair_id": rid},
            )
        )
        if max_items is not None and len(rows) >= 2 * max_items:
            break
    if not rows:
        raise ValueError(f"No valid benign pairs found in {path}")
    return MemoryDataset(rows)  # type: ignore[call-arg]


if task is not None:

    @task
    def benign_policy_pairs(
        dataset_path: str,
        seed: int = 123,
        max_items: Optional[int] = None,
    ) -> Task:
        """Benign paired prompts to stress-test refusal/compliance consistency.

        Uses a standard multiple_choice solver and choice() scorer over
        two-option answers ["Refuse", "Comply"].
        """

        ds = _load_benign_pairs_dataset(dataset_path, max_items=max_items)
        # Create solver with retry logic for resilience
        base_solver = multiple_choice(seed=seed)  # type: ignore[call-arg]
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            timeout=120.0
        )
        resilient_solver = create_resilient_solver(base_solver, retry_config)

        return Task(
            dataset=ds,
            solver=resilient_solver,
            scorer=choice(),  # type: ignore[misc]
            tags=["robustness", "benign_pairs"],
        )

else:

    def benign_policy_pairs(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "inspect_ai is not available: cannot create benign_policy_pairs task."
        )

