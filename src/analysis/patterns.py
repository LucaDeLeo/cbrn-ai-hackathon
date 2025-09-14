"""Lexical pattern detection methods for CBRN MCQAs."""

import json
import re
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
from scipy import stats

from src.data.schemas import Question


@dataclass
class PatternReport:
    """Report structure for lexical pattern analysis results."""
    method: str
    timestamp: str
    config_hash: Optional[str]
    dataset: dict
    results: dict
    performance: dict


def _get_memory_usage_mb() -> float:
    """Get peak memory usage in MB using stdlib methods.
    
    Returns:
        Memory usage in MB (peak RSS when available)
    """
    try:
        # Prefer resource.getrusage (Unix/macOS)
        import resource  # Local import for portability (Windows)
        import sys
        
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KiB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)  # bytes → MB
        else:
            return usage.ru_maxrss / 1024  # KiB → MB
    except Exception:
        # Fallback to tracemalloc peak if resource not available
        if tracemalloc.is_tracing():
            _current, peak = tracemalloc.get_traced_memory()
            return peak / (1024 * 1024)  # bytes → MB
        return 0.0


def _extract_phrases(text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
    """Extract n-grams from text using pure string operations.
    
    Args:
        text: Input text to extract phrases from
        min_length: Minimum n-gram length
        max_length: Maximum n-gram length
        
    Returns:
        List of phrases (n-grams)
    """
    # Clean and normalize text
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    phrases = []
    for n in range(min_length, min(max_length + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i + n])
            phrases.append(phrase)
    
    return phrases


def _calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size.
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Cohen's d effect size
    """
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def _calculate_cramers_v(contingency_table: np.ndarray) -> float:
    """Calculate Cramér's V effect size for categorical data.
    
    Args:
        contingency_table: 2D array representing contingency table
        
    Returns:
        Cramér's V value
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = np.sum(contingency_table)
    min_dim = min(contingency_table.shape) - 1
    
    if min_dim == 0 or n == 0:
        return 0.0
    
    return np.sqrt(chi2 / (n * min_dim))


def analyze_phrase_frequencies(correct_answers: List[str], incorrect_answers: List[str]) -> Dict[str, Any]:
    """Analyze phrase frequencies in correct vs incorrect answers.
    
    Args:
        correct_answers: List of correct answer texts
        incorrect_answers: List of incorrect answer texts
        
    Returns:
        Dictionary with phrase analysis results
    """
    # Extract phrases from all answers
    correct_phrases = []
    incorrect_phrases = []
    
    for answer in correct_answers:
        correct_phrases.extend(_extract_phrases(answer))
    
    for answer in incorrect_answers:
        incorrect_phrases.extend(_extract_phrases(answer))
    
    # Count phrase frequencies
    correct_counter = Counter(correct_phrases)
    incorrect_counter = Counter(incorrect_answers)
    
    # Find discriminative phrases
    all_phrases = set(correct_phrases + incorrect_phrases)
    discriminative_phrases = []
    
    for phrase in all_phrases:
        if correct_counter[phrase] < 2 and incorrect_counter[phrase] < 2:
            continue  # Skip rare phrases
        
        # Create contingency table
        correct_count = correct_counter[phrase]
        incorrect_count = incorrect_counter[phrase]
        correct_total = len(correct_answers)
        incorrect_total = len(incorrect_answers)
        
        contingency = np.array([
            [correct_count, correct_total - correct_count],
            [incorrect_count, incorrect_total - incorrect_count]
        ])
        
        # Calculate chi-square test
        try:
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
            cramers_v = _calculate_cramers_v(contingency)
            
            # Calculate frequency ratios
            correct_freq = correct_count / correct_total if correct_total > 0 else 0
            incorrect_freq = incorrect_count / incorrect_total if incorrect_total > 0 else 0
            
            discriminative_phrases.append({
                'phrase': phrase,
                'correct_count': correct_count,
                'incorrect_count': incorrect_count,
                'correct_frequency': correct_freq,
                'incorrect_frequency': incorrect_freq,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'cramers_v': cramers_v
            })
        except Exception:
            continue  # Skip phrases that cause statistical errors
    
    # Sort by effect size (Cramér's V) and p-value
    discriminative_phrases.sort(key=lambda x: (-x['cramers_v'], x['p_value']))
    
    return {
        'total_phrases_analyzed': len(all_phrases),
        'discriminative_phrases': discriminative_phrases[:10],  # Top 10
        'correct_total_phrases': len(correct_phrases),
        'incorrect_total_phrases': len(incorrect_phrases)
    }


def analyze_length_distributions(questions: List[Question]) -> Dict[str, Any]:
    """Analyze length distributions of correct vs incorrect answers.
    
    Args:
        questions: List of Question objects
        
    Returns:
        Dictionary with length analysis results
    """
    correct_lengths = []
    incorrect_lengths = []
    
    for question in questions:
        correct_answer = question.choices[question.answer]
        correct_lengths.append(len(correct_answer))
        
        for i, choice in enumerate(question.choices):
            if i != question.answer:
                incorrect_lengths.append(len(choice))
    
    if not correct_lengths or not incorrect_lengths:
        return {
            'error': 'Insufficient data for length analysis',
            'correct_lengths': correct_lengths,
            'incorrect_lengths': incorrect_lengths
        }
    
    # Statistical tests
    try:
        t_stat, t_pvalue = stats.ttest_ind(correct_lengths, incorrect_lengths)
        cohens_d = _calculate_cohens_d(correct_lengths, incorrect_lengths)
        
        # Descriptive statistics
        correct_stats = {
            'mean': np.mean(correct_lengths),
            'median': np.median(correct_lengths),
            'std': np.std(correct_lengths),
            'min': np.min(correct_lengths),
            'max': np.max(correct_lengths),
            'count': len(correct_lengths)
        }
        
        incorrect_stats = {
            'mean': np.mean(incorrect_lengths),
            'median': np.median(incorrect_lengths),
            'std': np.std(incorrect_lengths),
            'min': np.min(incorrect_lengths),
            'max': np.max(incorrect_lengths),
            'count': len(incorrect_lengths)
        }
        
        return {
            'correct_lengths': correct_stats,
            'incorrect_lengths': incorrect_stats,
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'cohens_d': cohens_d,
            'significant': t_pvalue < 0.05
        }
        
    except Exception as e:
        return {
            'error': f'Statistical analysis failed: {str(e)}',
            'correct_lengths': correct_lengths,
            'incorrect_lengths': incorrect_lengths
        }


def detect_meta_patterns(questions: List[Question]) -> Dict[str, Any]:
    """Detect meta-patterns like 'all of the above' and 'none of the above'.
    
    Args:
        questions: List of Question objects
        
    Returns:
        Dictionary with meta-pattern detection results
    """
    meta_patterns = {
        'all_of_the_above': ['all of the above', 'all of above', 'all above', 'all options', 'all choices'],
        'none_of_the_above': ['none of the above', 'none of above', 'none above', 'none of the options', 'none'],
        'cannot_be_determined': ['cannot be determined', 'cannot determine', 'not enough information', 'insufficient data'],
        'both': ['both', 'both a and b', 'both a and c', 'both b and c', 'both a and d'],
    }
    
    pattern_results = {}
    
    for pattern_name, variations in meta_patterns.items():
        correct_matches = 0
        incorrect_matches = 0
        total_correct = 0
        total_incorrect = 0
        
        for question in questions:
            correct_answer = question.choices[question.answer].lower()
            correct_matched = any(variation in correct_answer for variation in variations)
            
            if correct_matched:
                correct_matches += 1
            total_correct += 1
            
            for i, choice in enumerate(question.choices):
                if i != question.answer:
                    choice_lower = choice.lower()
                    if any(variation in choice_lower for variation in variations):
                        incorrect_matches += 1
                    total_incorrect += 1
        
        # Calculate frequencies and statistical significance
        correct_freq = correct_matches / total_correct if total_correct > 0 else 0
        incorrect_freq = incorrect_matches / total_incorrect if total_incorrect > 0 else 0
        
        # Chi-square test
        contingency = np.array([
            [correct_matches, total_correct - correct_matches],
            [incorrect_matches, total_incorrect - incorrect_matches]
        ])
        
        try:
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
            cramers_v = _calculate_cramers_v(contingency)
            
            pattern_results[pattern_name] = {
                'correct_matches': correct_matches,
                'incorrect_matches': incorrect_matches,
                'correct_frequency': correct_freq,
                'incorrect_frequency': incorrect_freq,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'significant': p_value < 0.05
            }
        except Exception:
            pattern_results[pattern_name] = {
                'correct_matches': correct_matches,
                'incorrect_matches': incorrect_matches,
                'correct_frequency': correct_freq,
                'incorrect_frequency': incorrect_freq,
                'error': 'Statistical analysis failed'
            }
    
    return pattern_results


def calculate_technical_density(text: str) -> float:
    """Calculate technical term density in text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Technical density score (0.0 to 1.0)
    """
    # Define technical term patterns (CBRN-specific and general scientific)
    technical_patterns = [
        # Chemical terms
        r'\b\w*acid\w*\b', r'\b\w*compound\w*\b', r'\b\w*molecule\w*\b',
        r'\b\w*synthesis\w*\b', r'\b\w*catalyst\w*\b', r'\b\w*reaction\w*\b',
        
        # Biological terms
        r'\b\w*protein\w*\b', r'\b\w*enzyme\w*\b', r'\b\w*cell\w*\b',
        r'\b\w*DNA\w*\b', r'\b\w*RNA\w*\b', r'\b\w*gene\w*\b',
        r'\b\w*organism\w*\b', r'\b\w*bacteria\w*\b', r'\b\w*virus\w*\b',
        
        # Radiological/Nuclear terms
        r'\b\w*radiation\w*\b', r'\b\w*isotope\w*\b', r'\b\w*decay\w*\b',
        r'\b\w*uranium\w*\b', r'\b\w*plutonium\w*\b', r'\b\w*radioactive\w*\b',
        
        # General scientific terms
        r'\b\w*experiment\w*\b', r'\b\w*laboratory\w*\b', r'\b\w*research\w*\b',
        r'\b\w*analysis\w*\b', r'\b\w*measurement\w*\b', r'\b\w*concentration\w*\b',
        r'\b\w*temperature\w*\b', r'\b\w*pressure\w*\b', r'\b\w*volume\w*\b',
        
        # Units and measurements
        r'\b\d+\s*(mg|g|kg|ml|l|mol|M|mM|μM|nM|°C|K|Pa|atm|bar)\b',
        r'\b\w*per\s+\w+\b', r'\b\w*ratio\w*\b', r'\b\w*percentage\w*\b',
    ]
    
    text_lower = text.lower()
    total_words = len(text_lower.split())
    
    if total_words == 0:
        return 0.0
    
    technical_matches = 0
    for pattern in technical_patterns:
        matches = re.findall(pattern, text_lower)
        technical_matches += len(matches)
    
    return min(technical_matches / total_words, 1.0)


def calculate_effect_size(group1: List[float], group2: List[float]) -> float:
    """Calculate effect size between two groups.
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Cohen's d effect size
    """
    return _calculate_cohens_d(group1, group2)


def detect_lexical_patterns(
    questions: List[Question],
    show_progress: bool = False,
    save_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    dataset_hash: Optional[str] = None,
    debug: bool = False,
) -> PatternReport:
    """Detect lexical patterns in MCQA questions.
    
    Args:
        questions: List of Question objects to analyze
        show_progress: Whether to show progress bar
        save_path: Optional path to save JSON results
        dataset_path: Optional path to the dataset
        dataset_hash: Optional hash of the dataset
        debug: Whether to enable debug logging
        
    Returns:
        PatternReport with analysis results
    """
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback for environments without tqdm
        def tqdm(iterable, disable=False, desc="Processing"):
            if not disable:
                print(f"{desc}...")
            return iterable
    
    import logging
    
    logger = logging.getLogger(__name__)
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Start memory tracking if not already active
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    
    start_time = time.time()
    
    # Separate correct and incorrect answers
    correct_answers = []
    incorrect_answers = []
    
    for question in tqdm(questions, disable=not show_progress, desc="Processing questions"):
        # Validate question fields
        if question.answer is None or question.answer < 0 or question.answer >= len(question.choices):
            raise ValueError(
                f"Question '{question.id}' has invalid answer index {question.answer} for {len(question.choices)} choices"
            )
        
        correct_answers.append(question.choices[question.answer])
        for i, choice in enumerate(question.choices):
            if i != question.answer:
                incorrect_answers.append(choice)
    
    # Perform analyses
    phrase_analysis = analyze_phrase_frequencies(correct_answers, incorrect_answers)
    length_analysis = analyze_length_distributions(questions)
    meta_patterns = detect_meta_patterns(questions)
    
    # Technical term density analysis
    correct_technical_densities = [calculate_technical_density(answer) for answer in correct_answers]
    incorrect_technical_densities = [calculate_technical_density(answer) for answer in incorrect_answers]
    
    technical_density_analysis = {
        'correct_densities': {
            'mean': np.mean(correct_technical_densities),
            'median': np.median(correct_technical_densities),
            'std': np.std(correct_technical_densities),
            'count': len(correct_technical_densities)
        },
        'incorrect_densities': {
            'mean': np.mean(incorrect_technical_densities),
            'median': np.median(incorrect_technical_densities),
            'std': np.std(incorrect_technical_densities),
            'count': len(incorrect_technical_densities)
        }
    }
    
    # Statistical test for technical density
    try:
        t_stat, t_pvalue = stats.ttest_ind(correct_technical_densities, incorrect_technical_densities)
        cohens_d = _calculate_cohens_d(correct_technical_densities, incorrect_technical_densities)
        
        technical_density_analysis.update({
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'cohens_d': cohens_d,
            'significant': t_pvalue < 0.05
        })
    except Exception as e:
        technical_density_analysis['error'] = f'Statistical analysis failed: {str(e)}'
    
    # Calculate metrics
    runtime_seconds = time.time() - start_time
    memory_peak_mb = _get_memory_usage_mb()
    questions_per_second = len(questions) / runtime_seconds if runtime_seconds > 0 else 0.0
    
    # Build dataset metadata
    dataset_meta = {
        "path": str(dataset_path) if dataset_path else None,
        "total_questions": len(questions),
        "total_choices": sum(len(q.choices) for q in questions),
    }
    if dataset_hash is not None:
        dataset_meta["hash"] = dataset_hash
    
    # Create comprehensive results
    results = {
        "phrase_analysis": phrase_analysis,
        "length_analysis": length_analysis,
        "meta_patterns": meta_patterns,
        "technical_density_analysis": technical_density_analysis,
        "summary": {
            "total_questions_analyzed": len(questions),
            "total_correct_answers": len(correct_answers),
            "total_incorrect_answers": len(incorrect_answers),
            "discriminative_phrases_found": len(phrase_analysis.get('discriminative_phrases', [])),
            "significant_length_difference": length_analysis.get('significant', False),
            "significant_technical_density_difference": technical_density_analysis.get('significant', False)
        }
    }
    
    # Create report
    report = PatternReport(
        method="lexical_pattern_detection",
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        config_hash=None,  # Placeholder for future config hashing
        dataset=dataset_meta,
        results=results,
        performance={
            "runtime_seconds": runtime_seconds,
            "memory_peak_mb": memory_peak_mb,
            "questions_per_second": questions_per_second
        }
    )
    
    # Save to JSON if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        report_dict = asdict(report)
        converted_dict = convert_numpy_types(report_dict)
        
        with open(save_path, 'w') as f:
            json.dump(converted_dict, f, indent=2)
    
    return report
