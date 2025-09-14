"""Integration tests for confidence-aware aggregate pipeline."""

import json
import math
import tempfile
from pathlib import Path

import pandas as pd

from robustcbrn.analysis.aggregate import aggregate_main


def test_aggregate_with_confidence_metrics():
    """Test aggregate pipeline with confidence-aware metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock log files with confidence data
        logs_dir = Path(tmpdir) / "logs"
        logs_dir.mkdir()

        # Create a sample log file
        log_data = {
            "model": "test-model",
            "task": "test_task",
            "samples": [
                {
                    "id": "test1",
                    "pred_index": 0,
                    "target_index": 0,
                    "confidence": 0.9,
                    "seed": 42
                },
                {
                    "id": "test2",
                    "pred_index": 1,
                    "target_index": 0,
                    "confidence": 0.6,
                    "seed": 42
                },
                {
                    "id": "test3",
                    "pred_index": 0,
                    "target_index": 0,
                    "confidence": 0.3,
                    "seed": 42
                },
                {
                    "id": "test4",
                    "pred_index": 1,
                    "target_index": 1,
                    "confidence": 0.8,
                    "seed": 42
                }
            ]
        }

        log_file = logs_dir / "test_log.json"
        log_file.write_text(json.dumps(log_data))

        # Run aggregate
        out_dir = Path(tmpdir) / "results"
        result = aggregate_main(
            str(logs_dir),
            str(out_dir),
            k=1,
            confidence_thresholds=[0.0, 0.5, 0.75]
        )

        assert result == 0

        # Check that output files were created
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "all_results.csv").exists()

        # Load and verify summary
        summary = json.loads((out_dir / "summary.json").read_text())

        # Check confidence-aware metrics are present
        assert "confidence_aware_metrics" in summary
        assert "calibration" in summary

        # Check that we have metrics for each threshold
        conf_metrics = summary["confidence_aware_metrics"]
        assert "t_0.0" in conf_metrics
        assert "t_0.5" in conf_metrics
        assert "t_0.75" in conf_metrics

        # Verify structure of metrics
        for threshold_key in ["t_0.0", "t_0.5", "t_0.75"]:
            metrics = conf_metrics[threshold_key]
            assert "threshold" in metrics
            assert "abstention_rate" in metrics
            assert "accuracy_on_answered" in metrics
            assert "average_penalty_score" in metrics
            assert "n_total" in metrics
            assert "n_abstentions" in metrics
            assert "n_answered" in metrics

        # Check calibration metrics (global)
        calibration = summary["calibration"]
        assert "brier_score" in calibration
        assert "ece" in calibration

        # Check per-threshold calibration present and shaped
        cal_per_t = summary.get("calibration_per_threshold", {})
        assert "t_0.0" in cal_per_t
        assert "t_0.5" in cal_per_t
        assert "t_0.75" in cal_per_t
        for k in ["t_0.0", "t_0.5", "t_0.75"]:
            assert "brier_score" in cal_per_t[k]
            assert "ece" in cal_per_t[k]
            assert "n_answered" in cal_per_t[k]

        # Advisories should include abstention target info (non-fatal)
        advisories = summary.get("advisories", {})
        at = advisories.get("abstention_target", {})
        assert "target_range" in at
        assert isinstance(at.get("thresholds", {}), dict)

        # Verify CSV output
        df = pd.read_csv(out_dir / "all_results.csv")
        assert not df.empty
        assert "confidence" in df.columns
        assert "is_correct" in df.columns


def test_aggregate_without_confidence_data():
    """Test aggregate pipeline handles missing confidence data gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock log files without confidence data
        logs_dir = Path(tmpdir) / "logs"
        logs_dir.mkdir()

        log_data = {
            "model": "test-model",
            "task": "test_task",
            "samples": [
                {
                    "id": "test1",
                    "pred_index": 0,
                    "target_index": 0,
                    "seed": 42
                },
                {
                    "id": "test2",
                    "pred_index": 1,
                    "target_index": 0,
                    "seed": 42
                }
            ]
        }

        log_file = logs_dir / "test_log.json"
        log_file.write_text(json.dumps(log_data))

        # Run aggregate
        out_dir = Path(tmpdir) / "results"
        result = aggregate_main(str(logs_dir), str(out_dir), k=1)

        assert result == 0

        # Load summary
        summary = json.loads((out_dir / "summary.json").read_text())

        # Should have confidence metrics (even without explicit confidence data)
        assert "confidence_aware_metrics" in summary
        # When confidence is missing, it's treated as NaN and results in metrics
        # This is expected behavior for backward compatibility

        # Calibration should have NaN/None values when no confidence data
        # NaN is serialized as null in JSON, which Python loads as None
        # But if it's a float NaN, we need to check differently
        brier = summary["calibration"]["brier_score"]
        ece = summary["calibration"]["ece"]

        # Check if values are None or NaN
        assert brier is None or (isinstance(brier, float) and math.isnan(brier))
        assert ece is None or (isinstance(ece, float) and math.isnan(ece))

        # Per-threshold calibration should be empty or missing when no confidence
        cal_per_t = summary.get("calibration_per_threshold", {})
        assert isinstance(cal_per_t, dict)
