"""Tests for the ambiguity detection module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robustcbrn.qa.ambiguity import (
    DEFAULT_CONFIG,
    AmbiguityConfig,
    AmbiguityDecision,
    AmbiguityDetectionError,
    AmbiguityMetrics,
    InvalidChoicesError,
    _extract_numbers,
    _has_meta_option,
    _has_negation_pair,
    _heuristics_for_item,
    _is_numeric_like,
    _jaccard,
    _norm_text,
    _tokens,
    audit_dataset,
    decisions_to_records,
)
from robustcbrn.tasks.ambiguity_audit import run_ambiguity_audit


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_norm_text(self):
        assert _norm_text("  HELLO World  ") == "hello world"
        assert _norm_text("Test-Case_123") == "test case 123"
        assert _norm_text("") == ""

    def test_tokens(self):
        assert _tokens("hello world") == ["hello", "world"]
        assert _tokens("test-case") == ["test", "case"]
        assert _tokens("123 456") == ["123", "456"]
        assert _tokens("") == []

    def test_jaccard(self):
        assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0
        assert _jaccard({"a", "b"}, {"b", "c"}) == 1/3
        assert _jaccard(set(), set()) == 0.0
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_has_meta_option(self):
        config = DEFAULT_CONFIG
        assert _has_meta_option("all of the above", config) == "all_of_the_above"
        assert _has_meta_option("none of the above", config) == "none_of_the_above"
        assert _has_meta_option("both a and b", config) == "both_and"
        assert _has_meta_option("I don't know", config) == "dont_know"
        assert _has_meta_option("regular option", config) is None

    def test_has_negation_pair(self):
        config = DEFAULT_CONFIG
        assert _has_negation_pair("is safe", "is not safe", config) is True
        assert _has_negation_pair("increases", "decreases", config) is True
        assert _has_negation_pair("safe", "unsafe", config) is True
        assert _has_negation_pair("hello", "world", config) is False

    def test_is_numeric_like(self):
        assert _is_numeric_like("123") is True
        assert _is_numeric_like("45.67") is True
        assert _is_numeric_like("10%") is True
        assert _is_numeric_like("$100") is True
        assert _is_numeric_like("hello") is False

    def test_extract_numbers(self):
        assert _extract_numbers("123") == [123.0]
        assert _extract_numbers("45.67") == [45.67]
        assert _extract_numbers("10% and 20%") == [10.0, 20.0]
        assert _extract_numbers("hello") == []


class TestHeuristics:
    """Test the main heuristics function."""

    def test_meta_options_unanswerable(self):
        """Meta options should be marked as unanswerable."""
        choices = ["Option A", "Option B", "All of the above"]
        label, reasons = _heuristics_for_item(choices)
        assert label == "unanswerable"
        assert "all_of_the_above" in reasons

        choices = ["Yes", "No", "I don't know"]
        label, reasons = _heuristics_for_item(choices)
        assert label == "unanswerable"
        assert "dont_know" in reasons

    def test_duplicate_choices_ambiguous(self):
        """Duplicate choices should be marked as ambiguous."""
        choices = ["Option A", "Option B", "Option A"]
        label, reasons = _heuristics_for_item(choices)
        assert label == "ambiguous"
        assert "duplicate_choices" in reasons

    def test_near_duplicate_ambiguous(self):
        """Near-duplicate choices should be marked as ambiguous."""
        choices = [
            "This is a long option with many words",
            "This is a long option with many word",  # Only 1 word different
            "Completely different"
        ]
        label, reasons = _heuristics_for_item(choices)
        assert label == "ambiguous"
        assert "near_duplicate" in reasons

    def test_negation_pairs_ambiguous(self):
        """Contradictory options should be marked as ambiguous."""
        choices = ["It is safe", "It is not safe", "Maybe safe"]
        label, reasons = _heuristics_for_item(choices)
        assert label == "ambiguous"
        assert "contradictory_options" in reasons

    def test_numeric_crowding_ambiguous(self):
        """Very close numeric values should be marked as ambiguous."""
        choices = ["10.0", "10.05", "20.0", "30.0"]  # 10.0 and 10.05 are within 1%
        label, reasons = _heuristics_for_item(choices)
        assert label == "ambiguous"
        assert "numeric_too_close" in reasons

    def test_boolean_like_unanswerable(self):
        """Very short boolean-style choices require stem."""
        choices = ["Yes", "No", "Maybe"]
        label, reasons = _heuristics_for_item(choices)
        assert label == "unanswerable"
        assert "boolean_like_requires_stem" in reasons

    def test_clean_item(self):
        """Normal items should be marked as clean."""
        choices = [
            "Implement proper safety protocols",
            "Ignore all guidelines",
            "Consult with experts first",
            "Delay the decision"
        ]
        label, reasons = _heuristics_for_item(choices)
        assert label == "clean"
        assert reasons == []

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty choices
        label, reasons = _heuristics_for_item([])
        assert label == "clean"

        # Single choice
        label, reasons = _heuristics_for_item(["Only option"])
        assert label == "clean"

        # Whitespace variations
        choices = ["  Option A  ", "Option A", " Option  A "]
        label, reasons = _heuristics_for_item(choices)
        assert label == "ambiguous"
        assert "duplicate_choices" in reasons

    def test_invalid_choices(self):
        """Test that invalid choices raise appropriate errors."""
        with pytest.raises(InvalidChoicesError):
            _heuristics_for_item("not a list")  # type: ignore

    def test_custom_config(self):
        """Test with custom configuration."""
        config = AmbiguityConfig(
            jaccard_threshold=0.8,  # Lower threshold
            numeric_proximity_threshold=0.05,  # Higher threshold (5%)
            max_tokens_for_boolean=1  # Stricter boolean detection
        )

        # Test Jaccard threshold
        choices = [
            "This is option one",
            "This is option two",  # Different enough with 0.9 but not 0.8
            "Completely different"
        ]
        label, reasons = _heuristics_for_item(choices, config)
        # With lower threshold, might catch as near duplicate

        # Test numeric threshold
        choices = ["10.0", "10.3", "20.0"]  # 3% difference
        label, reasons = _heuristics_for_item(choices, config)
        assert label == "clean"  # Should pass with 5% threshold

        # Test boolean tokens
        choices = ["Yes sir", "No sir"]  # 2 tokens each
        label, reasons = _heuristics_for_item(choices, config)
        assert label == "clean"  # Not boolean-like with max_tokens=1


class TestAuditDataset:
    """Test the audit_dataset function."""

    def test_audit_basic(self):
        """Test basic audit functionality."""
        samples = [
            {"id": "q1", "choices": ["Yes", "No", "Maybe"]},
            {"id": "q2", "choices": ["All of the above", "A", "B"]},
            {"id": "q3", "choices": ["Good option", "Bad option", "Neutral option"]},
        ]

        decisions, metrics = audit_dataset(samples)

        assert len(decisions) == 3
        assert decisions[0].id == "q1"
        assert decisions[0].label == "unanswerable"  # Boolean-like
        assert decisions[1].id == "q2"
        assert decisions[1].label == "unanswerable"  # Meta option
        assert decisions[2].id == "q3"
        assert decisions[2].label == "clean"

        # Check metrics
        assert metrics is not None
        assert metrics.total_items == 3
        assert metrics.clean_count == 1
        assert metrics.unanswerable_count == 2
        assert metrics.ambiguous_count == 0

    def test_audit_without_metrics(self):
        """Test audit without collecting metrics."""
        samples = [{"id": "q1", "choices": ["A", "B", "C"]}]
        decisions, metrics = audit_dataset(samples, collect_metrics=False)

        assert len(decisions) == 1
        assert metrics is None

    def test_audit_with_processing_time(self):
        """Test that processing time is tracked."""
        samples = [{"id": "q1", "choices": ["A", "B", "C"]}]
        decisions, metrics = audit_dataset(samples)

        assert decisions[0].processing_time_ms > 0
        assert metrics.total_processing_time_ms > 0
        assert metrics.avg_processing_time_ms > 0

    def test_decisions_to_records(self):
        """Test conversion to records."""
        decisions = [
            AmbiguityDecision("q1", "clean", [], processing_time_ms=1.5),
            AmbiguityDecision("q2", "ambiguous", ["duplicate_choices"], processing_time_ms=2.0),
            AmbiguityDecision("q3", "unanswerable", ["all_of_the_above", "none_of_the_above"], processing_time_ms=1.0),
        ]

        records = decisions_to_records(decisions)

        assert len(records) == 3
        assert records[0] == {"id": "q1", "label": "clean", "reason_codes": "", "processing_time_ms": 1.5}
        assert records[1] == {"id": "q2", "label": "ambiguous", "reason_codes": "duplicate_choices", "processing_time_ms": 2.0}
        assert records[2]["id"] == "q3"
        assert records[2]["label"] == "unanswerable"
        assert records[2]["reason_codes"] == "all_of_the_above,none_of_the_above"

    def test_audit_error_handling(self):
        """Test error handling in audit."""
        # Invalid item format
        samples = ["not a dict or object"]

        with pytest.raises(AmbiguityDetectionError):
            audit_dataset(samples)


class TestAmbiguityAuditTask:
    """Test the ambiguity audit task."""

    def test_run_ambiguity_audit_heuristic(self):
        """Test running audit in heuristic mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample dataset
            dataset_path = Path(tmpdir) / "test_dataset.jsonl"
            dataset_path.write_text(json.dumps({
                "id": "test1",
                "choices": ["Option A", "Option B", "All of the above"]
            }))

            # Mock the paths
            with patch('robustcbrn.tasks.ambiguity_audit.get_paths') as mock_paths:
                mock_paths.return_value = MagicMock(logs_dir=tmpdir)

                # Run audit
                output_path = run_ambiguity_audit(
                    str(dataset_path),
                    mode="heuristic",
                    seed=42,
                    verbose=False
                )

                # Check output
                assert Path(output_path).exists()

                with open(output_path) as f:
                    result = json.load(f)

                assert result["task"] == "ambiguity_audit"
                assert result["model"] == "heuristic"
                assert result["seed"] == 42
                assert len(result["samples"]) == 1
                assert result["samples"][0]["id"] == "test1"
                assert result["samples"][0]["label"] == "unanswerable"
                assert "all_of_the_above" in result["samples"][0]["reason_codes"]

                # Check metrics are included
                assert "metrics" in result
                assert result["metrics"]["total_items"] == 1

                # Check config is included
                assert "config" in result
                assert result["config"]["jaccard_threshold"] == 0.9

    def test_run_ambiguity_audit_with_custom_config(self):
        """Test audit with custom configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "test_dataset.jsonl"
            dataset_path.write_text(json.dumps({
                "id": "test1",
                "choices": ["10.0", "10.02", "20.0"]  # 0.2% difference
            }))

            with patch('robustcbrn.tasks.ambiguity_audit.get_paths') as mock_paths:
                mock_paths.return_value = MagicMock(logs_dir=tmpdir)

                # With default config (1% threshold), should be clean
                config = AmbiguityConfig(numeric_proximity_threshold=0.01)
                output_path = run_ambiguity_audit(
                    str(dataset_path),
                    config=config
                )

                with open(output_path) as f:
                    result = json.load(f)
                assert result["samples"][0]["label"] == "clean"

                # With stricter config (0.1% threshold), should be ambiguous
                config = AmbiguityConfig(numeric_proximity_threshold=0.001)
                output_path = run_ambiguity_audit(
                    str(dataset_path),
                    config=config
                )

                with open(output_path) as f:
                    result = json.load(f)
                assert result["samples"][0]["label"] == "ambiguous"

    def test_run_ambiguity_audit_llm_disabled(self):
        """Test that LLM mode is properly disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "test_dataset.jsonl"
            dataset_path.write_text(json.dumps({"id": "test1", "choices": ["A", "B"]}))

            with pytest.raises(RuntimeError, match="LLM critic mode is disabled"):
                run_ambiguity_audit(str(dataset_path), mode="llm")

    def test_run_ambiguity_audit_invalid_mode(self):
        """Test invalid mode handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "test_dataset.jsonl"
            dataset_path.write_text(json.dumps({"id": "test1", "choices": ["A", "B"]}))

            with patch('robustcbrn.tasks.ambiguity_audit.get_paths') as mock_paths:
                mock_paths.return_value = MagicMock(logs_dir=tmpdir)

                with pytest.raises(ValueError, match="Unknown mode"):
                    run_ambiguity_audit(str(dataset_path), mode="invalid")


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test the full pipeline from dataset to audit output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a diverse dataset
            dataset_path = Path(tmpdir) / "full_test.jsonl"
            items = [
                {"id": "clean1", "choices": ["Very different A", "Completely different B", "Unique option C"]},
                {"id": "dup1", "choices": ["Same", "Same", "Different"]},
                {"id": "meta1", "choices": ["A", "B", "None of the above"]},
                {"id": "neg1", "choices": ["Safe", "Not safe", "Maybe"]},
                {"id": "num1", "choices": ["10.0", "10.01", "20.0"]},
                {"id": "bool1", "choices": ["Yes", "No"]},
            ]

            with open(dataset_path, 'w') as f:
                for item in items:
                    f.write(json.dumps(item) + '\n')

            # Run audit
            with patch('robustcbrn.tasks.ambiguity_audit.get_paths') as mock_paths:
                mock_paths.return_value = MagicMock(logs_dir=tmpdir)

                output_path = run_ambiguity_audit(str(dataset_path))

                with open(output_path) as f:
                    result = json.load(f)

                # Check all items were processed
                assert len(result["samples"]) == 6

                # Build id to label map
                id_to_label = {s["id"]: s["label"] for s in result["samples"]}

                # Verify classifications
                assert id_to_label["clean1"] == "clean"
                assert id_to_label["dup1"] == "ambiguous"
                assert id_to_label["meta1"] == "unanswerable"
                assert id_to_label["neg1"] == "ambiguous"
                assert id_to_label["num1"] == "ambiguous"
                assert id_to_label["bool1"] == "unanswerable"

                # Check metrics
                metrics = result["metrics"]
                assert metrics["total_items"] == 6
                assert metrics["clean_count"] == 1
                assert metrics["ambiguous_count"] == 3
                assert metrics["unanswerable_count"] == 2


class TestAmbiguityConfig:
    """Test configuration management."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = AmbiguityConfig()
        assert config.jaccard_threshold == 0.9
        assert config.numeric_proximity_threshold == 0.01
        assert config.max_tokens_for_boolean == 2
        assert len(config.meta_options) > 0
        assert len(config.negation_prefixes) > 0
        assert len(config.negation_antonyms) > 0

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "jaccard_threshold": 0.85,
            "numeric_proximity_threshold": 0.02,
            "max_tokens_for_boolean": 3
        }
        config = AmbiguityConfig.from_dict(config_dict)
        assert config.jaccard_threshold == 0.85
        assert config.numeric_proximity_threshold == 0.02
        assert config.max_tokens_for_boolean == 3

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = AmbiguityConfig(jaccard_threshold=0.95)
        config_dict = config.to_dict()
        assert config_dict["jaccard_threshold"] == 0.95
        assert "meta_options" in config_dict
        assert "negation_prefixes" in config_dict


class TestMetrics:
    """Test performance metrics."""

    def test_metrics_calculation(self):
        """Test metrics calculations."""
        metrics = AmbiguityMetrics(
            total_items=10,
            clean_count=5,
            ambiguous_count=3,
            unanswerable_count=2,
            total_processing_time_ms=100.0
        )

        assert metrics.avg_processing_time_ms == 10.0

        metrics_dict = metrics.to_dict()
        assert metrics_dict["total_items"] == 10
        assert metrics_dict["avg_processing_time_ms"] == 10.0

    def test_metrics_empty(self):
        """Test metrics with no items."""
        metrics = AmbiguityMetrics()
        assert metrics.avg_processing_time_ms == 0.0  # Should handle division by zero
