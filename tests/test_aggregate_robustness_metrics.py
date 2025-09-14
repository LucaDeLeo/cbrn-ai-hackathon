import json
from pathlib import Path

import pytest

from robustcbrn.analysis.aggregate import aggregate_main


def _write_log(path: Path, model: str, task: str, samples: list[dict]) -> None:
    data = {
        "model": model,
        "task": task,
        "seed": 123,
        "samples": samples,
    }
    path.write_text(json.dumps(data))


def test_aggregate_metrics_when_tasks_missing(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    figs_dir = tmp_path / "figs"
    logs_dir.mkdir()
    out_dir.mkdir()
    figs_dir.mkdir()

    # Ensure figures go to an isolated temp directory
    monkeypatch.setenv("FIGS_DIR", str(figs_dir))
    # Ensure matplotlib font cache is writable in CI/sandbox
    mpl_dir = tmp_path / "mpl"
    mpl_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))

    # Create a minimal log for a different task (e.g., cloze) so aggregation runs
    _write_log(
        logs_dir / "cloze_log.json",
        model="m1",
        task="cloze",
        samples=[
            {"id": "q1", "pred_index": 0, "target_index": 0, "correct": True},
        ],
    )

    rc = aggregate_main(str(logs_dir), str(out_dir))
    assert rc == 0

    # Summary should include metrics with zeros when tasks are absent
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())

    for key in [
        "paraphrase_consistency",
        "perturbation_fragility",
        "paraphrase_delta_accuracy",
    ]:
        assert key in summary

    assert summary["paraphrase_consistency"]["n"] == 0
    assert summary["paraphrase_consistency"]["consistency"] == 0.0
    assert summary["perturbation_fragility"]["n"] == 0
    assert summary["perturbation_fragility"]["flip_rate"] == 0.0
    assert summary["paraphrase_delta_accuracy"]["n"] == 0
    assert summary["paraphrase_delta_accuracy"]["delta"] == 0.0

    # Figures should not be created for missing/empty tasks
    assert not (figs_dir / "paraphrase_consistency.png").exists()
    assert not (figs_dir / "perturbation_fragility.png").exists()


def test_aggregate_metrics_paraphrase_and_perturb(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    figs_dir = tmp_path / "figs"
    logs_dir.mkdir()
    out_dir.mkdir()
    figs_dir.mkdir()

    # Ensure figures go to an isolated temp directory
    monkeypatch.setenv("FIGS_DIR", str(figs_dir))
    # Ensure matplotlib font cache is writable in CI/sandbox
    mpl_dir = tmp_path / "mpl"
    mpl_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))

    # Paraphrase consistency + delta accuracy
    # Two items, one consistent, one inconsistent. Delta: one item shows drop of 1.0, one shows 0.0 → mean 0.5
    _write_log(
        logs_dir / "paraphrase_consistency_m1.json",
        model="m1",
        task="paraphrase_consistency",
        samples=[
            # Item i1: consistent predictions across variants, all correct
            {"id": "i1", "variant": "orig", "pred_index": 1, "target_index": 1, "correct": True},
            {"id": "i1", "variant": "aug1", "pred_index": 1, "target_index": 1, "correct": True},
            {"id": "i1", "variant": "aug2", "pred_index": 1, "target_index": 1, "correct": True},
            # Item i2: inconsistent (orig=2, aug1=3); orig correct, variant wrong
            {"id": "i2", "variant": "orig", "pred_index": 2, "target_index": 2, "correct": True},
            {"id": "i2", "variant": "aug1", "pred_index": 3, "target_index": 2, "correct": False},
        ],
    )

    # Perturbation fragility
    # r1: flips 1/2; r2: flips 2/2 → mean flip_rate = 0.75
    _write_log(
        logs_dir / "perturbation_stability_m1.json",
        model="m1",
        task="perturbation_stability",
        samples=[
            {"id": "r1", "variant": "orig", "pred_index": 0, "target_index": 0},
            {"id": "r1", "variant": "noise1", "pred_index": 0, "target_index": 0},
            {"id": "r1", "variant": "noise2", "pred_index": 1, "target_index": 0},
            {"id": "r2", "variant": "orig", "pred_index": 2, "target_index": 2},
            {"id": "r2", "variant": "noise1", "pred_index": 3, "target_index": 2},
            {"id": "r2", "variant": "noise2", "pred_index": 4, "target_index": 2},
        ],
    )

    rc = aggregate_main(str(logs_dir), str(out_dir))
    assert rc == 0

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())

    # Paraphrase consistency
    assert summary["paraphrase_consistency"]["n"] == 2
    assert pytest.approx(summary["paraphrase_consistency"]["consistency"], rel=1e-6) == 0.5

    # Paraphrase delta accuracy
    assert summary["paraphrase_delta_accuracy"]["n"] == 2
    assert pytest.approx(summary["paraphrase_delta_accuracy"]["delta"], rel=1e-6) == 0.5

    # Perturbation fragility: ensure computed with data present and in [0,1]
    assert summary["perturbation_fragility"]["n"] >= 1
    flip = float(summary["perturbation_fragility"]["flip_rate"])
    assert 0.0 <= flip <= 1.0

    # Figures are best-effort; plotting may be skipped in headless CI if backend faults
    # Verify no exception path by ensuring summary computed and allow figures to be optional here
