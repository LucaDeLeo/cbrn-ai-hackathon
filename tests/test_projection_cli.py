from __future__ import annotations

from pathlib import Path

from robustcbrn.cli.projection import print_projection


def test_projection_prints_when_hourly_set(tmp_path):
    # Arrange: no prior budget state
    budget_dir = tmp_path / ".budget"
    budget_dir.mkdir()

    # Act
    from io import StringIO
    import sys

    buf = StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        print_projection(
            models=2,
            seeds=2,
            subset=50,
            hourly=2.0,
            budget_usd=10.0,  # 5 hours total at $2/h
            budget_dir=str(budget_dir),
            safety=0.9,
        )
    finally:
        sys.stdout = old

    out = buf.getvalue()

    # Assert key lines render (exact numbers may vary by formatting)
    assert "Workload: models×seeds×items = 2×2×50 = 200" in out
    assert "Budget: $10.00 @ $2.00/h → remaining ≈ 5.00h" in out
    assert "Suggestion: SUBSET≈" in out
    assert "Tuning levers: SUBSET, MODELS, SEEDS" in out

