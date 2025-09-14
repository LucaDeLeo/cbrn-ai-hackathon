"""Smoke tests for paraphrase and perturbation generation."""

import pytest
from robustcbrn.qa.paraphrase import generate_paraphrases
from robustcbrn.qa.perturb import generate_perturbations


class TestParaphraseGeneration:
    """Test paraphrase generation functionality."""

    def test_basic_paraphrase(self):
        """Test basic paraphrase generation."""
        result = generate_paraphrases("Which of the following is correct", k=2)
        assert len(result) >= 1  # At least 'orig'
        assert result[0].variant == "orig"
        assert "?" in result[0].text  # Should have question mark
        # Check if any paraphrases were generated
        if len(result) > 1:
            assert result[1].variant.startswith("para")
            assert result[1].text != result[0].text

    def test_empty_stem_raises(self):
        """Test that empty stem raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_paraphrases("", k=2)
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_paraphrases(None, k=2)

    def test_whitespace_only_raises(self):
        """Test that whitespace-only stem raises ValueError."""
        with pytest.raises(ValueError, match="whitespace-only"):
            generate_paraphrases("   ", k=2)

    def test_negative_k_raises(self):
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            generate_paraphrases("Test question", k=-1)

    def test_zero_k_returns_only_orig(self):
        """Test that k=0 returns only original."""
        result = generate_paraphrases("Test question", k=0)
        assert len(result) == 1
        assert result[0].variant == "orig"


class TestPerturbationGeneration:
    """Test perturbation generation functionality."""

    def test_basic_perturbation(self):
        """Test basic perturbation generation."""
        choices = ["Option A", "Option B", "Option C"]
        result = generate_perturbations("What is the answer", choices, 1, k=3)
        assert len(result) >= 1  # At least 'orig'
        assert result[0].variant == "orig"
        assert result[0].target_index == 1
        # Check if perturbations were generated
        if len(result) > 1:
            assert result[1].variant.startswith("pert:")

    def test_target_remapping(self):
        """Test that target index is correctly remapped for order changes."""
        choices = ["First", "Second", "Third"]
        result = generate_perturbations("Choose one", choices, 0, k=5)
        # Find the reverse perturbation
        for pert in result:
            if "order:reverse" in pert.kind:
                # Original target was 0 (First), reversed should be 2
                assert pert.target_index == 2
                assert pert.choices == ["Third", "Second", "First"]
                break

    def test_empty_choices_raises(self):
        """Test that empty choices list raises ValueError."""
        with pytest.raises(ValueError, match="Choices list cannot be empty"):
            generate_perturbations("Question", [], 0, k=2)

    def test_invalid_target_index_raises(self):
        """Test that out-of-bounds target index raises ValueError."""
        choices = ["A", "B"]
        with pytest.raises(ValueError, match="out of bounds"):
            generate_perturbations("Question", choices, 2, k=2)
        with pytest.raises(ValueError, match="out of bounds"):
            generate_perturbations("Question", choices, -1, k=2)

    def test_single_choice_handling(self):
        """Test handling of single-choice questions."""
        choices = ["Only option"]
        result = generate_perturbations("Question", choices, 0, k=3)
        assert result[0].target_index == 0
        # All perturbations should maintain target_index = 0
        for pert in result:
            assert pert.target_index == 0

    def test_zero_k_returns_only_orig(self):
        """Test that k=0 returns only original."""
        choices = ["A", "B"]
        result = generate_perturbations("Question", choices, 0, k=0)
        assert len(result) == 1
        assert result[0].variant == "orig"