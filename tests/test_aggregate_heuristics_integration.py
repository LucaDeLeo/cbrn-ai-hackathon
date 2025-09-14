from __future__ import annotations

import json
from pathlib import Path

import pytest

from robustcbrn.analysis.aggregate import aggregate_main, load_all_results, position_bias_heuristic


def _write_log(path: Path, model: str, task: str, samples: list[dict]) -> None:
    payload = {"model": model, "task": task, "seed": 123, "samples": samples}
    path.write_text(json.dumps(payload))


def test_aggregate_heuristics_summary_with_metadata(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    figs_dir = tmp_path / "figs"
    logs_dir.mkdir()
    out_dir.mkdir()
    figs_dir.mkdir()

    # Isolate figures and MPL config for headless CI
    monkeypatch.setenv("FIGS_DIR", str(figs_dir))
    mpl_dir = tmp_path / "mpl"
    mpl_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))

    # Create an MCQ log with safe heuristics metadata
    # i1: longest correct (idx 2), pred last (idx 2 of 3)
    # i2: longest wrong (idx 0), pred first
    # i3: longest correct (idx 2), pred neither first nor last
    samples = [
        {
            "id": "i1",
            "pred_index": 2,
            "target_index": 2,
            "correct": True,
            "choice_lengths": [1, 3, 5],
            "num_choices": 3,
        },
        {
            "id": "i2",
            "pred_index": 0,
            "target_index": 1,
            "correct": False,
            "choice_lengths": [10, 2],
            "num_choices": 2,
        },
        {
            "id": "i3",
            "pred_index": 1,
            "target_index": 2,
            "correct": False,
            "choice_lengths": [2, 2, 9, 1],
            "num_choices": 4,
        },
    ]
    _write_log(logs_dir / "mcq_with_meta.json", model="m", task="mcq_full", samples=samples)

    rc = aggregate_main(str(logs_dir), str(out_dir))
    assert rc == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    assert "heuristics_summary" in summary
    hs = summary["heuristics_summary"]

    # Longest-answer: 2/3 correct
    assert hs.get("n_items") == 3
    assert pytest.approx(hs.get("longest_answer_acc", 0.0), rel=1e-6) == 2.0 / 3.0

    # Position bias: compare to recomputed helper on the same loaded frame
    df_loaded = load_all_results(str(logs_dir))
    pb_expected = position_bias_heuristic(df_loaded)
    assert pytest.approx(hs.get("position_bias_rate", 0.0), rel=1e-6) == pytest.approx(
        pb_expected.get("position_bias_rate", 0.0), rel=1e-6
    )
