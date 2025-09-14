"""Unit tests for lexical pattern detection module."""

import unittest
import numpy as np
from unittest.mock import patch
from pathlib import Path

from src.data.schemas import Question
from src.analysis.patterns import (
    PatternReport,
    detect_lexical_patterns,
    analyze_phrase_frequencies,
    analyze_length_distributions,
    detect_meta_patterns,
    calculate_technical_density,
    calculate_effect_size,
    _extract_phrases,
    _calculate_cohens_d,
    _calculate_cramers_v
)


class TestPatternDetection(unittest.TestCase):
    """Test cases for lexical pattern detection functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_questions = [
            Question(
                id="test1",
                question="What is the color of the sky?",
                choices=["Blue", "Green", "Red", "Yellow"],
                answer=0,
                topic="general"
            ),
            Question(
                id="test2", 
                question="2 + 2 = ?",
                choices=["1", "2", "3", "4"],
                answer=3,
                topic="math"
            ),
            Question(
                id="test3",
                question="Which protein is involved in DNA replication?",
                choices=["Hemoglobin", "DNA polymerase", "Insulin", "All of the above"],
                answer=1,
                topic="biology"
            ),
            Question(
                id="test4",
                question="What is the molecular weight of water?",
                choices=["18 g/mol", "32 g/mol", "None of the above", "Cannot be determined"],
                answer=0,
                topic="chemistry"
            )
        ]
    
    def test_extract_phrases(self):
        """Test phrase extraction functionality."""
        text = "This is a test sentence with technical terms"
        
        # Test basic phrase extraction
        phrases = _extract_phrases(text, min_length=2, max_length=3)
        
        self.assertIsInstance(phrases, list)
        self.assertGreater(len(phrases), 0)
        
        # Check that we get n-grams of the right lengths
        for phrase in phrases:
            words = phrase.split()
            self.assertGreaterEqual(len(words), 2)
            self.assertLessEqual(len(words), 3)
        
        # Test with empty text
        empty_phrases = _extract_phrases("")
        self.assertEqual(len(empty_phrases), 0)
        
        # Test with single word
        single_phrases = _extract_phrases("word", min_length=2, max_length=3)
        self.assertEqual(len(single_phrases), 0)
    
    def test_calculate_cohens_d(self):
        """Test Cohen's d calculation."""
        # Test with normal distributions
        group1 = [1, 2, 3, 4, 5]
        group2 = [6, 7, 8, 9, 10]
        
        cohens_d = _calculate_cohens_d(group1, group2)
        self.assertIsInstance(cohens_d, float)
        self.assertLess(cohens_d, 0)  # group1 < group2
        
        # Test with identical groups
        identical_d = _calculate_cohens_d(group1, group1)
        self.assertAlmostEqual(identical_d, 0.0, places=10)
        
        # Test with empty groups
        empty_d = _calculate_cohens_d([], [])
        self.assertEqual(empty_d, 0.0)
    
    def test_calculate_cramers_v(self):
        """Test Cramér's V calculation."""
        # Test with 2x2 contingency table
        contingency = np.array([[10, 5], [3, 12]])
        cramers_v = _calculate_cramers_v(contingency)
        
        self.assertIsInstance(cramers_v, float)
        self.assertGreaterEqual(cramers_v, 0.0)
        self.assertLessEqual(cramers_v, 1.0)
        
        # Test with zero contingency
        zero_contingency = np.array([[0, 0], [0, 0]])
        zero_v = _calculate_cramers_v(zero_contingency)
        self.assertEqual(zero_v, 0.0)
    
    def test_analyze_phrase_frequencies(self):
        """Test phrase frequency analysis."""
        correct_answers = [
            "DNA polymerase is a protein enzyme",
            "DNA polymerase catalyzes reactions",
            "Protein synthesis involves DNA"
        ]
        incorrect_answers = [
            "Hemoglobin carries oxygen",
            "Insulin regulates glucose",
            "Water is H2O"
        ]
        
        result = analyze_phrase_frequencies(correct_answers, incorrect_answers)
        
        self.assertIsInstance(result, dict)
        self.assertIn('discriminative_phrases', result)
        self.assertIn('total_phrases_analyzed', result)
        self.assertIn('correct_total_phrases', result)
        self.assertIn('incorrect_total_phrases', result)
        
        # Check discriminative phrases structure
        discriminative = result['discriminative_phrases']
        if discriminative:
            phrase = discriminative[0]
            self.assertIn('phrase', phrase)
            self.assertIn('correct_count', phrase)
            self.assertIn('incorrect_count', phrase)
            self.assertIn('chi2_statistic', phrase)
            self.assertIn('p_value', phrase)
            self.assertIn('cramers_v', phrase)
    
    def test_analyze_length_distributions(self):
        """Test length distribution analysis."""
        result = analyze_length_distributions(self.sample_questions)
        
        self.assertIsInstance(result, dict)
        self.assertIn('correct_lengths', result)
        self.assertIn('incorrect_lengths', result)
        
        # Check that we have statistical measures
        if 'error' not in result:
            self.assertIn('t_statistic', result)
            self.assertIn('t_pvalue', result)
            self.assertIn('cohens_d', result)
            self.assertIn('significant', result)
            
            # Check descriptive statistics structure
            correct_stats = result['correct_lengths']
            self.assertIn('mean', correct_stats)
            self.assertIn('median', correct_stats)
            self.assertIn('std', correct_stats)
            self.assertIn('count', correct_stats)
    
    def test_detect_meta_patterns(self):
        """Test meta-pattern detection."""
        result = detect_meta_patterns(self.sample_questions)
        
        self.assertIsInstance(result, dict)
        
        # Check that we detect expected patterns
        expected_patterns = ['all_of_the_above', 'none_of_the_above', 'cannot_be_determined', 'both']
        for pattern in expected_patterns:
            self.assertIn(pattern, result)
            
            pattern_data = result[pattern]
            self.assertIn('correct_matches', pattern_data)
            self.assertIn('incorrect_matches', pattern_data)
            self.assertIn('correct_frequency', pattern_data)
            self.assertIn('incorrect_frequency', pattern_data)
    
    def test_calculate_technical_density(self):
        """Test technical term density calculation."""
        # Test with technical text
        technical_text = "DNA polymerase catalyzes the synthesis of proteins in the laboratory"
        density = calculate_technical_density(technical_text)
        
        self.assertIsInstance(density, float)
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)
        
        # Test with non-technical text
        simple_text = "The sky is blue and the grass is green"
        simple_density = calculate_technical_density(simple_text)
        self.assertLess(simple_density, density)
        
        # Test with empty text
        empty_density = calculate_technical_density("")
        self.assertEqual(empty_density, 0.0)
        
        # Test with units and measurements
        measurement_text = "The concentration is 10 mg/ml at 25°C"
        measurement_density = calculate_technical_density(measurement_text)
        self.assertGreater(measurement_density, 0.0)
    
    def test_calculate_effect_size(self):
        """Test effect size calculation wrapper."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [6, 7, 8, 9, 10]
        
        effect_size = calculate_effect_size(group1, group2)
        self.assertIsInstance(effect_size, float)
        self.assertLess(effect_size, 0)  # group1 < group2
    
    def test_detect_lexical_patterns_integration(self):
        """Test the main integration function."""
        result = detect_lexical_patterns(
            self.sample_questions,
            show_progress=False,
            debug=False
        )
        
        # Check that we get a PatternReport
        self.assertIsInstance(result, PatternReport)
        self.assertEqual(result.method, "lexical_pattern_detection")
        self.assertIsInstance(result.timestamp, str)
        self.assertIsInstance(result.dataset, dict)
        self.assertIsInstance(result.results, dict)
        self.assertIsInstance(result.performance, dict)
        
        # Check results structure
        results = result.results
        self.assertIn('phrase_analysis', results)
        self.assertIn('length_analysis', results)
        self.assertIn('meta_patterns', results)
        self.assertIn('technical_density_analysis', results)
        self.assertIn('summary', results)
        
        # Check performance metrics
        performance = result.performance
        self.assertIn('runtime_seconds', performance)
        self.assertIn('memory_peak_mb', performance)
        self.assertIn('questions_per_second', performance)
        
        self.assertGreater(performance['runtime_seconds'], 0)
        self.assertGreaterEqual(performance['memory_peak_mb'], 0)
        self.assertGreater(performance['questions_per_second'], 0)
    
    def test_detect_lexical_patterns_with_save(self):
        """Test saving results to file."""
        test_path = Path("test_results.json")
        
        try:
            result = detect_lexical_patterns(
                self.sample_questions,
                save_path=test_path,
                show_progress=False
            )
            
            # Check that file was created
            self.assertTrue(test_path.exists())
            
            # Check that file contains valid JSON
            import json
            with open(test_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertIn('method', saved_data)
            self.assertIn('results', saved_data)
            self.assertIn('performance', saved_data)
            
        finally:
            # Clean up test file
            if test_path.exists():
                test_path.unlink()
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Test with empty questions list
        result = detect_lexical_patterns([])
        self.assertIsInstance(result, PatternReport)
        
        # Test with invalid question (should raise error)
        invalid_questions = [
            Question(
                id="invalid",
                question="Test",
                choices=["A", "B"],
                answer=5  # Invalid answer index
            )
        ]
        
        with self.assertRaises(ValueError):
            detect_lexical_patterns(invalid_questions)
    
    def test_statistical_significance(self):
        """Test that statistical tests are working correctly."""
        # Create questions with clear length bias
        biased_questions = [
            Question(
                id="long_correct",
                question="Which is the longest option?",
                choices=["A", "B", "This is a very long correct answer", "D"],
                answer=2
            ),
            Question(
                id="short_correct",
                question="Which is the shortest option?",
                choices=["Very long incorrect answer here", "B", "C", "D"],
                answer=1
            )
        ]
        
        result = analyze_length_distributions(biased_questions)
        
        if 'error' not in result:
            # Should detect significant difference
            self.assertIn('significant', result)
            self.assertIn('t_statistic', result)
            self.assertIn('t_pvalue', result)
    
    def test_performance_targets(self):
        """Test that performance targets are met."""
        # Create a larger dataset for performance testing
        large_questions = []
        for i in range(100):  # 100 questions should be fast
            large_questions.append(Question(
                id=f"perf_test_{i}",
                question=f"Question {i}?",
                choices=["A", "B", "C", "D"],
                answer=i % 4,
                topic="performance"
            ))
        
        result = detect_lexical_patterns(large_questions, show_progress=False)
        
        # Check performance metrics
        performance = result.performance
        self.assertLess(performance['runtime_seconds'], 30)  # Should be fast
        self.assertGreater(performance['questions_per_second'], 0)
        self.assertLess(performance['memory_peak_mb'], 1000)  # Should use <1GB


if __name__ == '__main__':
    unittest.main()
