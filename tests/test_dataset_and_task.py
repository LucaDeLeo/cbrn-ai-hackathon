from __future__ import annotations

from pathlib import Path

import pytest

from robustcbrn.tasks.common import load_mcq_dataset


def test_load_mcq_dataset_sample():
    p = Path("data/sample_sanitized.jsonl")
    ds = load_mcq_dataset(p, shuffle_seed=42, max_items=3)
    # Works either as Inspect MemoryDataset or list of dicts
    items = list(ds) if hasattr(ds, "__iter__") and not isinstance(ds, list) else list(ds)  # type: ignore[arg-type]
    assert len(items) == 3
    first = items[0]
    # Inspect Sample or dict
    if hasattr(first, "input"):
        assert hasattr(first, "choices") and hasattr(first, "target")
    else:
        assert {"input", "choices", "target"}.issubset(first.keys())



def test_inspect_import_and_task_smoke():
    pytest.importorskip("inspect_ai")
    # Import task factory without executing a full eval
    from robustcbrn.tasks.mcq_full import mcq_full  # noqa: F401
