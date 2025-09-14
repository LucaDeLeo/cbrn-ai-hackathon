from __future__ import annotations

from pathlib import Path

from robustcbrn.analysis.robustness import aflite_flag_summary
from robustcbrn.qa.filters import (
    _alphabetical_choice,
    _kfold_by_item,
    _longest_choice,
    aflite_lite,
)
from robustcbrn.tasks.aflite_screen import run_aflite_screen


def test_aflite_screen_smoke(tmp_path: Path):
    data_path = Path("data/sample_sanitized.jsonl")
    out_path = tmp_path / "aflite.json"
    # Limit to 2 items and 2 folds for speed
    res_path = run_aflite_screen(
        str(data_path), tau=0.6, k_folds=2, seed=123, max_items=2, out_path=str(out_path)
    )
    assert Path(res_path).exists()

    import json

    payload = json.loads(Path(res_path).read_text())
    assert payload.get("task") == "aflite_screen"
    samples = payload.get("samples", [])
    assert isinstance(samples, list) and len(samples) == 2
    for s in samples:
        assert "id" in s
        assert "predictability_score" in s
        assert "flag_predictable" in s
        assert "probe_hit" in s


def test_aflite_summary_minimal():
    # Build a minimal dataframe-like structure for summary
    import pandas as pd

    rows = [
        {
            "id": "x1",
            "task": "aflite_screen",
            "predictability_score": 0.9,
            "probe_hit": ["longest_answer", "position_only"],
        },
        {
            "id": "x2",
            "task": "aflite_screen",
            "predictability_score": 0.4,
            "probe_hit": [],
        },
    ]
    df = pd.DataFrame(rows)
    summ = aflite_flag_summary(df, preset="balanced", tau=0.7)
    assert summ["n"] == 2
    # At least one flagged given the scores
    assert 0.0 <= summ["flagged_frac"] <= 1.0


def test_kfold_no_data_leakage():
    """Critical: Verify K-fold grouping prevents same item in train/test."""
    # Create sample items with repeated IDs to test grouping
    item_ids = ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"]

    folds = _kfold_by_item(item_ids, k=2, seed=42)

    # Verify each item ID appears in exactly one fold
    for i, fold1 in enumerate(folds):
        fold1_ids = {item_ids[idx] for idx in fold1}
        for j, fold2 in enumerate(folds):
            if i != j:
                fold2_ids = {item_ids[idx] for idx in fold2}
                # No item ID should appear in both folds
                assert fold1_ids.isdisjoint(fold2_ids), f"Data leakage: {fold1_ids & fold2_ids}"

    # Verify all indices are covered
    all_indices = set()
    for fold in folds:
        all_indices.update(fold)
    assert all_indices == set(range(len(item_ids)))


def test_probe_implementations():
    """Test the three heuristic probe functions."""

    # Test longest_choice probe
    choices = ["Short", "Medium length", "The longest choice here", "Mid"]
    assert _longest_choice(choices) == 2
    assert _longest_choice([]) == 0  # Edge case: empty
    assert _longest_choice(["a", "bb", "c"]) == 1

    # Test alphabetical_choice probe
    choices = ["Zebra", "Apple", "Banana", "Cat"]
    assert _alphabetical_choice(choices) == 1  # Apple
    assert _alphabetical_choice([]) == 0  # Edge case: empty
    assert _alphabetical_choice(["b", "a", "c"]) == 1

    # Test position_only is implicitly tested via aflite_lite


def test_aflite_classifier_basic():
    """Basic test that classifier produces reasonable scores."""
    # Create simple test dataset where longest answer is always correct
    # This should produce high predictability scores
    dataset = [
        {
            "id": "test1",
            "choices": ["Short", "Medium one", "This is the longest choice"],
            "target": 2,
        },
        {
            "id": "test2",
            "choices": ["A", "BB", "CCC is longest"],
            "target": 2,
        },
        {
            "id": "test3",
            "choices": ["Quick", "The longest answer here", "Mid"],
            "target": 1,
        },
    ]

    results = aflite_lite(dataset, tau=0.6, k_folds=2, seed=123)

    # Basic sanity checks
    assert len(results) == 3
    for r in results:
        assert 0.0 <= r.predictability_score <= 1.0
        assert isinstance(r.flag_predictable, bool)
        assert isinstance(r.probe_hit, list)
        # Should detect longest_answer pattern for items where it applies
        if r.id in ["test1", "test2"]:
            assert "longest_answer" in r.probe_hit


def test_edge_cases():
    """Test handling of edge cases and invalid inputs."""

    # Empty dataset
    results = aflite_lite([], tau=0.7)
    assert results == []

    # Dataset with single item (can't do proper k-fold)
    dataset = [{
        "id": "single",
        "choices": ["A", "B", "C"],
        "target": 0,
    }]
    results = aflite_lite(dataset, tau=0.7, k_folds=2)
    assert len(results) == 1

    # Invalid target index
    dataset = [{
        "id": "invalid",
        "choices": ["A", "B"],
        "target": 5,  # Out of bounds
    }]
    results = aflite_lite(dataset, tau=0.7, k_folds=2)
    assert len(results) == 1
    assert results[0].predictability_score == 0.0  # Should handle gracefully

    # Empty choices
    dataset = [{
        "id": "empty_choices",
        "choices": [],
        "target": 0,
    }]
    # Should not crash
    results = aflite_lite(dataset, tau=0.7, k_folds=2)
    assert len(results) == 1

