from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def test_fill_report_updates_placeholders_and_scopes_to_model_cards(tmp_path: Path, monkeypatch):
    # Prepare fake project structure under tmp
    root = tmp_path
    docs = root / "docs"
    artifacts = root / "artifacts" / "results"

    # Minimal REPORT content copied with exact keys used by the filler
    report_md = """
# Sprint Report (Draft)

Methodology:
- Tasks: `mcq_full`, `mcq_choices_only`, `cloze_full`
- Models: DO NOT TOUCH THIS LINE
- Seeds: DO NOT TOUCH THIS LINE

Results (placeholders → auto‑filled post‑run):
- Overall accuracy: 0.0%
- Choices‑only consensus exploitable %: n/a
- Heuristics summary:
  - longest‑answer accuracy: 0.0%
  - position‑bias rate (first/last): 0.0%
- MCQ↔Cloze gap (95% CI): Δ=0.000 (95% CI: [0.000, 0.000])
- Abstention / overconfidence: abst=0.0%, overconf=0.0%
- Runtime / cost: n/a

Model Cards Used (fill after run):
- Models: TODO
- Revisions: TODO
- Seeds: TODO
- Key config: TODO
""".lstrip()

    write(docs / "REPORT.md", report_md)

    # Summary with models and metrics
    summary = {
        "models": ["m1", "m2"],
        "mcq_vs_cloze": {"gap": 0.1, "ci_lo": 0.05, "ci_hi": 0.15},
        "abstention_overconfidence": {"abstention_rate": 0.0, "overconfidence_rate": 0.5},
        "heuristics_summary": {"longest_answer_acc": 0.25, "position_bias_rate": 0.75},
    }
    write(artifacts / "summary.json", json.dumps(summary))

    # all_results.csv with overall accuracy and exploitable field
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "task": ["mcq_choices_only", "mcq_choices_only", "cloze_full", "cloze_full"],
            "model": ["m1", "m2", "m1", "m2"],
            "correct": [True, False, True, False],
            "pred_index": [0, 1, 2, 3],
            "target_index": [0, 2, 2, 3],
            "confidence": [0.9, 0.8, 0.7, 0.6],
            "seed": [111, 222, 111, 222],
            # Majority exploitable across models for id=a only
            "exploitable": [1, 0, 0, 0],
        }
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    df.to_csv(artifacts / "all_results.csv", index=False)

    # Budget with hours and env for hourly

    # Point the module constants at our temp files
    import importlib.util
    import sys

    mod_path = Path(__file__).resolve().parents[1] / "scripts" / "fill_report.py"
    spec = importlib.util.spec_from_file_location("scripts.fill_report", mod_path)
    assert spec and spec.loader
    fill_report = importlib.util.module_from_spec(spec)
    sys.modules["scripts.fill_report"] = fill_report
    spec.loader.exec_module(fill_report)  # type: ignore[arg-type]

    # Override module paths to our temp files
    fill_report.ROOT = root
    fill_report.REPORT_PATH = docs / "REPORT.md"
    fill_report.SUMMARY_PATH = artifacts / "summary.json"
    fill_report.ALL_RESULTS_CSV = artifacts / "all_results.csv"

    # Env for key config and price
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("DTYPE", "bfloat16")
    monkeypatch.setenv("BATCH_SIZE", "4")
    monkeypatch.setenv("MAX_SEQ_LEN", "4096")

    # Run filler
    rc = fill_report.main()
    assert rc == 0

    out = (docs / "REPORT.md").read_text()
    # Check that methodology section remained unchanged
    assert "- Models: DO NOT TOUCH THIS LINE" in out
    assert "- Seeds: DO NOT TOUCH THIS LINE" in out

    # Check placeholders filled
    assert "- Overall accuracy: 50.0%" in out
    # Expect proper UTF-8 hyphen in "Choices‑only"
    assert "- Choices‑only consensus exploitable %: 50.0%" in out
    assert "- Abstention / overconfidence: abst=0.0%, overconf=50.0%" in out
    assert "- Runtime / cost: n/a" in out

    # Model Cards Used updates
    assert "- Models: m1; m2" in out
    assert "- Seeds: 111; 222" in out
    assert "- Revisions:" in out  # may still be TODO
    assert "- Key config: device=cuda; dtype=bfloat16; batch_size=4; max_seq_len=4096" in out
