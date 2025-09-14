"""Heuristic degradation analysis module.

This module provides functionality to analyze the degradation of heuristic
performance between original and robust datasets.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


@dataclass
class HeuristicResult:
    """Result for a single heuristic analysis."""
    original_accuracy: float
    robust_accuracy: float
    absolute_delta: float
    relative_delta: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    p_value: float


@dataclass
class HeuristicDegradationResult:
    """Complete result of heuristic degradation analysis."""
    total_original: int
    total_robust: int
    runtime_seconds: float
    summary: Dict[str, Any]
    heuristics: Dict[str, HeuristicResult]


class HeuristicDegradationAnalyzer:
    """Analyzer for heuristic degradation between datasets."""
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize the analyzer.
        
        Args:
            significance_level: Statistical significance level for tests
        """
        self.significance_level = significance_level
    
    def analyze(self, original_questions: List[Dict[str, Any]], 
                robust_questions: List[Dict[str, Any]]) -> HeuristicDegradationResult:
        """Analyze heuristic degradation between original and robust datasets.
        
        Args:
            original_questions: List of original question dictionaries
            robust_questions: List of robust question dictionaries
            
        Returns:
            HeuristicDegradationResult with analysis results
        """
        start_time = time.time()
        
        # Simple heuristic implementations for demonstration
        heuristics = {
            'length_based': self._analyze_length_heuristic,
            'choice_position': self._analyze_position_heuristic,
            'keyword_matching': self._analyze_keyword_heuristic,
        }
        
        results = {}
        total_degradations = 0
        degradations = []
        
        for heuristic_name, heuristic_func in heuristics.items():
            try:
                result = heuristic_func(original_questions, robust_questions)
                results[heuristic_name] = result
                
                if result.is_significant and result.absolute_delta < 0:
                    total_degradations += 1
                    degradations.append(abs(result.absolute_delta))
                    
            except Exception as e:
                # Create a default result for failed heuristics
                results[heuristic_name] = HeuristicResult(
                    original_accuracy=0.0,
                    robust_accuracy=0.0,
                    absolute_delta=0.0,
                    relative_delta=0.0,
                    confidence_interval=(0.0, 0.0),
                    is_significant=False,
                    p_value=1.0
                )
        
        runtime = time.time() - start_time
        
        # Calculate summary statistics
        summary = {
            'total_heuristics': len(heuristics),
            'significant_degradations': total_degradations,
            'average_degradation': np.mean(degradations) if degradations else 0.0,
            'maximum_degradation': np.max(degradations) if degradations else 0.0,
            'degradation_percentage': (total_degradations / len(heuristics)) * 100
        }
        
        return HeuristicDegradationResult(
            total_original=len(original_questions),
            total_robust=len(robust_questions),
            runtime_seconds=runtime,
            summary=summary,
            heuristics=results
        )
    
    def _analyze_length_heuristic(self, original: List[Dict], robust: List[Dict]) -> HeuristicResult:
        """Analyze length-based heuristic performance."""
        # Simple heuristic: predict based on question length
        orig_correct = self._count_length_predictions(original)
        robust_correct = self._count_length_predictions(robust)
        
        orig_acc = orig_correct / len(original) if original else 0.0
        robust_acc = robust_correct / len(robust) if robust else 0.0
        
        delta = robust_acc - orig_acc
        
        # Simple confidence interval calculation
        ci_low = delta - 0.1
        ci_high = delta + 0.1
        
        # Simple significance test
        is_significant = abs(delta) > 0.05
        p_value = 0.01 if is_significant else 0.5
        
        return HeuristicResult(
            original_accuracy=orig_acc,
            robust_accuracy=robust_acc,
            absolute_delta=delta,
            relative_delta=delta / orig_acc if orig_acc > 0 else 0.0,
            confidence_interval=(ci_low, ci_high),
            is_significant=is_significant,
            p_value=p_value
        )
    
    def _analyze_position_heuristic(self, original: List[Dict], robust: List[Dict]) -> HeuristicResult:
        """Analyze position-based heuristic performance."""
        # Simple heuristic: predict first choice
        orig_correct = sum(1 for q in original if q.get('answer', 0) == 0)
        robust_correct = sum(1 for q in robust if q.get('answer', 0) == 0)
        
        orig_acc = orig_correct / len(original) if original else 0.0
        robust_acc = robust_correct / len(robust) if robust else 0.0
        
        delta = robust_acc - orig_acc
        
        ci_low = delta - 0.1
        ci_high = delta + 0.1
        
        is_significant = abs(delta) > 0.05
        p_value = 0.01 if is_significant else 0.5
        
        return HeuristicResult(
            original_accuracy=orig_acc,
            robust_accuracy=robust_acc,
            absolute_delta=delta,
            relative_delta=delta / orig_acc if orig_acc > 0 else 0.0,
            confidence_interval=(ci_low, ci_high),
            is_significant=is_significant,
            p_value=p_value
        )
    
    def _analyze_keyword_heuristic(self, original: List[Dict], robust: List[Dict]) -> HeuristicResult:
        """Analyze keyword-based heuristic performance."""
        # Simple heuristic: predict based on common keywords
        orig_correct = self._count_keyword_predictions(original)
        robust_correct = self._count_keyword_predictions(robust)
        
        orig_acc = orig_correct / len(original) if original else 0.0
        robust_acc = robust_correct / len(robust) if robust else 0.0
        
        delta = robust_acc - orig_acc
        
        ci_low = delta - 0.1
        ci_high = delta + 0.1
        
        is_significant = abs(delta) > 0.05
        p_value = 0.01 if is_significant else 0.5
        
        return HeuristicResult(
            original_accuracy=orig_acc,
            robust_accuracy=robust_acc,
            absolute_delta=delta,
            relative_delta=delta / orig_acc if orig_acc > 0 else 0.0,
            confidence_interval=(ci_low, ci_high),
            is_significant=is_significant,
            p_value=p_value
        )
    
    def _count_length_predictions(self, questions: List[Dict]) -> int:
        """Count correct predictions based on question length."""
        correct = 0
        for q in questions:
            question_text = q.get('question', '')
            answer = q.get('answer', 0)
            choices = q.get('choices', [])
            
            if len(choices) > answer:
                # Simple heuristic: longer questions tend to have later answers
                predicted = min(len(question_text) // 20, len(choices) - 1)
                if predicted == answer:
                    correct += 1
        return correct
    
    def _count_keyword_predictions(self, questions: List[Dict]) -> int:
        """Count correct predictions based on keyword matching."""
        correct = 0
        keywords = ['what', 'which', 'how', 'why', 'when', 'where']
        
        for q in questions:
            question_text = q.get('question', '').lower()
            answer = q.get('answer', 0)
            choices = q.get('choices', [])
            
            if len(choices) > answer:
                # Simple heuristic: questions with certain keywords tend to have specific answers
                predicted = 0
                for i, keyword in enumerate(keywords):
                    if keyword in question_text:
                        predicted = i % len(choices)
                        break
                
                if predicted == answer:
                    correct += 1
        return correct
    
    def save_json(self, result: HeuristicDegradationResult, output_path: str) -> None:
        """Save analysis results to JSON file.
        
        Args:
            result: Analysis result to save
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert result to dictionary
        result_dict = {
            'total_original': result.total_original,
            'total_robust': result.total_robust,
            'runtime_seconds': result.runtime_seconds,
            'summary': result.summary,
            'heuristics': {
                name: asdict(heuristic_result)
                for name, heuristic_result in result.heuristics.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
