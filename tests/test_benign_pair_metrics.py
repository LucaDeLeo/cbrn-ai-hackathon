"""Unit tests for benign_pair_metrics function."""

import pandas as pd
import pytest
import numpy as np


class TestBenignPairMetrics:
    """Test the benign_pair_metrics analysis function."""

    def test_metrics_basic_calculation(self):
        """Test basic metric calculations."""
        # Import here to avoid circular dependency
        from robustcbrn.analysis.robustness import benign_pair_metrics

        # Create test dataframe
        data = [
            # Pair 1: both correct
            {"id": "bp1.safe", "task": "benign_policy_pairs", "variant": "safe",
             "correct": True, "model": "m1", "seed": 1, "pair_id": "bp1"},
            {"id": "bp1.unsafe", "task": "benign_policy_pairs", "variant": "unsafe",
             "correct": True, "model": "m1", "seed": 1, "pair_id": "bp1"},

            # Pair 2: safe wrong, unsafe correct
            {"id": "bp2.safe", "task": "benign_policy_pairs", "variant": "safe",
             "correct": False, "model": "m1", "seed": 1, "pair_id": "bp2"},
            {"id": "bp2.unsafe", "task": "benign_policy_pairs", "variant": "unsafe",
             "correct": True, "model": "m1", "seed": 1, "pair_id": "bp2"},
        ]
        df = pd.DataFrame(data)

        metrics = benign_pair_metrics(df)

        assert metrics["present"] is True
        assert metrics["compliance_safe"] == 0.5  # 1/2 safe correct
        assert metrics["refusal_unsafe"] == 1.0  # 2/2 unsafe correct
        assert metrics["pair_consistency"] == 0.5  # 1/2 pairs both correct

    def test_metrics_empty_dataframe(self):
        """Test metrics with empty dataframe."""
        from robustcbrn.analysis.robustness import benign_pair_metrics

        df = pd.DataFrame()
        metrics = benign_pair_metrics(df)

        assert metrics["present"] is False
        assert metrics["compliance_safe"] == 0.0
        assert metrics["refusal_unsafe"] == 0.0
        assert metrics["pair_consistency"] == 0.0

    def test_metrics_no_benign_pairs(self):
        """Test metrics when dataframe has no benign pairs tasks."""
        from robustcbrn.analysis.robustness import benign_pair_metrics

        data = [
            {"id": "q1", "task": "other_task", "correct": True},
            {"id": "q2", "task": "mcq_full", "correct": False},
        ]
        df = pd.DataFrame(data)

        metrics = benign_pair_metrics(df)

        assert metrics["present"] is False

    def test_metrics_pair_id_inference(self):
        """Test pair_id inference from id prefix."""
        from robustcbrn.analysis.robustness import benign_pair_metrics

        data = [
            # No explicit pair_id, should be inferred from id
            {"id": "bp1.safe", "task": "benign_policy_pairs", "variant": "safe",
             "correct": True, "model": "m1", "seed": 1},
            {"id": "bp1.unsafe", "task": "benign_policy_pairs", "variant": "unsafe",
             "correct": True, "model": "m1", "seed": 1},
        ]
        df = pd.DataFrame(data)

        metrics = benign_pair_metrics(df)

        # Should still calculate pair consistency via inference
        assert metrics["pair_consistency"] == 1.0

    def test_metrics_multiple_models_seeds(self):
        """Test metrics with multiple models and seeds."""
        from robustcbrn.analysis.robustness import benign_pair_metrics

        data = []
        for model in ["m1", "m2"]:
            for seed in [1, 2]:
                for pair in ["bp1", "bp2"]:
                    # Each combination: safe correct, unsafe correct for bp1; both wrong for bp2
                    safe_correct = (pair == "bp1")
                    unsafe_correct = (pair == "bp1")

                    data.extend([
                        {"id": f"{pair}.safe", "task": "benign_policy_pairs",
                         "variant": "safe", "correct": safe_correct,
                         "model": model, "seed": seed, "pair_id": pair},
                        {"id": f"{pair}.unsafe", "task": "benign_policy_pairs",
                         "variant": "unsafe", "correct": unsafe_correct,
                         "model": model, "seed": seed, "pair_id": pair},
                    ])

        df = pd.DataFrame(data)
        metrics = benign_pair_metrics(df)

        # 8 safe samples (4 from bp1 correct, 4 from bp2 wrong)
        assert metrics["compliance_safe"] == 0.5
        # 8 unsafe samples (4 from bp1 correct, 4 from bp2 wrong)
        assert metrics["refusal_unsafe"] == 0.5
        # 8 pairs total (4 bp1 consistent, 4 bp2 inconsistent)
        assert metrics["pair_consistency"] == 0.5

    def test_metrics_confidence_intervals(self):
        """Test that confidence intervals are computed."""
        from robustcbrn.analysis.robustness import benign_pair_metrics

        data = [
            {"id": "bp1.safe", "task": "benign_policy_pairs", "variant": "safe",
             "correct": True, "model": "m1", "seed": 1, "pair_id": "bp1"},
            {"id": "bp1.unsafe", "task": "benign_policy_pairs", "variant": "unsafe",
             "correct": False, "model": "m1", "seed": 1, "pair_id": "bp1"},
        ]
        df = pd.DataFrame(data)

        metrics = benign_pair_metrics(df)

        # Check CI fields exist and are lists of 2 elements
        assert "compliance_safe_ci" in metrics
        assert isinstance(metrics["compliance_safe_ci"], list)
        assert len(metrics["compliance_safe_ci"]) == 2

        assert "refusal_unsafe_ci" in metrics
        assert isinstance(metrics["refusal_unsafe_ci"], list)
        assert len(metrics["refusal_unsafe_ci"]) == 2

        assert "pair_consistency_ci" in metrics
        assert isinstance(metrics["pair_consistency_ci"], list)
        assert len(metrics["pair_consistency_ci"]) == 2

    def test_metrics_missing_variant_field(self):
        """Test handling when variant field is missing."""
        from robustcbrn.analysis.robustness import benign_pair_metrics

        data = [
            {"id": "bp1.safe", "task": "benign_policy_pairs",
             "correct": True, "model": "m1", "seed": 1, "pair_id": "bp1"},
        ]
        df = pd.DataFrame(data)

        metrics = benign_pair_metrics(df)

        # Should handle missing variant gracefully
        assert metrics["compliance_safe"] == 0.0
        assert metrics["refusal_unsafe"] == 0.0

    def test_metrics_nan_handling(self):
        """Test handling of NaN values in correct field."""
        from robustcbrn.analysis.robustness import benign_pair_metrics

        data = [
            {"id": "bp1.safe", "task": "benign_policy_pairs", "variant": "safe",
             "correct": True, "model": "m1", "seed": 1, "pair_id": "bp1"},
            {"id": "bp2.safe", "task": "benign_policy_pairs", "variant": "safe",
             "correct": np.nan, "model": "m1", "seed": 1, "pair_id": "bp2"},
            {"id": "bp1.unsafe", "task": "benign_policy_pairs", "variant": "unsafe",
             "correct": True, "model": "m1", "seed": 1, "pair_id": "bp1"},
        ]
        df = pd.DataFrame(data)

        metrics = benign_pair_metrics(df)

        # Should handle NaN values without crashing
        assert metrics["present"] is True
        assert 0 <= metrics["compliance_safe"] <= 1
        assert 0 <= metrics["refusal_unsafe"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])