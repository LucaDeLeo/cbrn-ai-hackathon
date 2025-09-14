from pathlib import Path

import pandas as pd


def test_aggregate_cli_passes_k(monkeypatch, tmp_path):
    # Arrange: stub load_all_results to return a minimal non-empty df
    from robustcbrn.analysis import aggregate as agg

    def fake_load_all_results(_logs_dir: str) -> pd.DataFrame:  # type: ignore[override]
        # Two models disagree on the same item under choices_only
        return pd.DataFrame(
            [
                {
                    "id": "q1",
                    "task": "mcq_choices_only",
                    "model": "m1",
                    "correct": True,
                    "pred_index": None,
                    "target_index": None,
                    "confidence": None,
                },
                {
                    "id": "q1",
                    "task": "mcq_choices_only",
                    "model": "m2",
                    "correct": False,
                    "pred_index": None,
                    "target_index": None,
                    "confidence": None,
                },
            ]
        )

    monkeypatch.setattr(agg, "load_all_results", fake_load_all_results)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    out1.mkdir()
    out2.mkdir()

    # Act: k=1 should mark exploitable=1; k=2 should mark exploitable=0
    rc1 = agg.cli(["--logs", str(tmp_path), "--out", str(out1), "--k", "1"])  # type: ignore[arg-type]
    assert rc1 == 0
    rc2 = agg.cli(["--logs", str(tmp_path), "--out", str(out2), "--k", "2"])  # type: ignore[arg-type]
    assert rc2 == 0

    df1 = pd.read_csv(Path(out1, "all_results.csv"))
    df2 = pd.read_csv(Path(out2, "all_results.csv"))

    # Assert: exploitable flag differs by k
    exp1 = int(df1.loc[df1["id"] == "q1", "exploitable"].max())
    exp2 = int(df2.loc[df2["id"] == "q1", "exploitable"].max())
    assert exp1 == 1
    assert exp2 == 0

