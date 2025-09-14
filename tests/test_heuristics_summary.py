from __future__ import annotations

import json
from pathlib import Path

import pytest

from robustcbrn.analysis.aggregate import (
    aggregate_main,
    longest_answer_heuristic,
    position_bias_heuristic,
)


def _write_log(path: Path, model: str, task: str, samples: list[dict]) -> None:
    data = {
        "model": model,
        "task": task,
        "seed": 123,
        "samples": samples,
    }
    path.write_text(json.dumps(data))


def test_heuristics_summary_missing_metadata_returns_zeros(tmp_path):
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    logs_dir.mkdir()
    out_dir.mkdir()

    # Create minimal log without heuristics metadata (no choice_lengths/num_choices)
    _write_log(
        logs_dir / "mcq_log.json",
        model="m1",
        task="mcq_full",
        samples=[
            {"id": "q1", "pred_index": 0, "target_index": 1, "correct": False},
            {"id": "q2", "pred_index": 2, "target_index": 2, "correct": True},
        ],
    )

    rc = aggregate_main(str(logs_dir), str(out_dir))
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert "heuristics_summary" in summary
    hs = summary["heuristics_summary"]
    # Without metadata, both metrics should be zero with a note
    assert float(hs.get("longest_answer_acc", 0.0)) == 0.0
    assert float(hs.get("position_bias_rate", 0.0)) == 0.0
    assert "note" in hs or ("note" in hs and isinstance(hs["note"], str))


def test_heuristics_functions_with_synthetic_df():
    import pandas as pd

    # Build a small DataFrame with per-item choice_lengths and indices
    rows = [
        # Longest-answer matches target for i1 (lens: [1, 3, 5] → idx 2)
        {"id": "i1", "model": "m", "choice_lengths": [1, 3, 5], "target_index": 2, "pred_index": 2, "num_choices": 3},
        # Longest-answer does not match target for i2 (lens: [10, 2] → idx 0)
        {"id": "i2", "model": "m", "choice_lengths": [10, 2], "target_index": 1, "pred_index": 0, "num_choices": 2},
        # i3: longest matches (lens [2,2,9,1] → 2), last position prediction (index 3)
        {"id": "i3", "model": "m", "choice_lengths": [2, 2, 9, 1], "target_index": 2, "pred_index": 3, "num_choices": 4},
    ]
    df = pd.DataFrame(rows)

    la = longest_answer_heuristic(df)
    assert la["n_items"] == 3
    # Two of three have longest==target
    assert pytest.approx(la["longest_answer_acc"], rel=1e-6) == 2.0 / 3.0

    pb = position_bias_heuristic(df)
    assert pb["n_rows"] >= 1
    # First rate = count pred==0 / rows_with_n (1/3), last rate = pred==n-1 (2/3) → pos rate = 2/3
    assert pytest.approx(pb["position_bias_rate"], rel=1e-6) == pytest.approx(2.0 / 3.0, rel=1e-6)
