# src/analysis/heuristic_degradation.py
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import Counter

from src.analysis.statistical import calculate_bootstrap_ci
from src.analysis.stratified_bootstrap import StratifiedBootstrapAnalyzer

@dataclass
class HeuristicResult:
    """Result for a single heuristic"""
    heuristic_name: str
    original_accuracy: float
    robust_accuracy: float
    absolute_delta: float
    confidence_interval: List[float]
    is_significant: bool
    p_value: Optional[float] = None
    bootstrap_metadata: Optional[Dict[str, Any]] = None

@dataclass
class HeuristicDegradationResult:
    """Complete heuristic degradation results"""
    timestamp: str
    runtime_seconds: float
    total_original: int
    total_robust: int
    heuristics: Dict[str, HeuristicResult]
    summary: Dict[str, Any]

class HeuristicDegradationAnalyzer:
    """Analyzes heuristic performance degradation between original and robust datasets"""
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 stratify_by: Optional[np.ndarray] = None,
                 adaptive_bootstrap: bool = True):
        """
        Initialize the analyzer
        
        Args:
            confidence_level: Confidence level for statistical tests (default: 0.95)
            stratify_by: Array of strata labels for stratified bootstrap (optional)
            adaptive_bootstrap: Whether to use adaptive bootstrap iterations
        """
        self.confidence_level = confidence_level
        self.stratify_by = stratify_by
        self.adaptive_bootstrap = adaptive_bootstrap
        
        # Initialize stratified bootstrap analyzer
        self.stratified_analyzer = StratifiedBootstrapAnalyzer(
            confidence_level=confidence_level,
            adaptive=adaptive_bootstrap
        )
        
        # Register all available heuristics
        self._heuristics = {
            'longest_answer': self._longest_answer_heuristic,
            'position_bias': self._position_bias_heuristic,
            'lexical_patterns': self._lexical_patterns_heuristic,
        }
    
    def analyze(self, 
                original_questions: List[Dict[str, Any]], 
                robust_questions: List[Dict[str, Any]]) -> HeuristicDegradationResult:
        """
        Analyze heuristic degradation between original and robust datasets
        
        Args:
            original_questions: List of original question dictionaries
            robust_questions: List of robust question dictionaries
            
        Returns:
            HeuristicDegradationResult containing all analysis results
        """
        start_time = time.time()
        
        # Run all heuristics
        results = {}
        for heuristic_name, heuristic_func in self._heuristics.items():
            try:
                result = self._analyze_heuristic(
                    heuristic_name, 
                    heuristic_func, 
                    original_questions, 
                    robust_questions
                )
                results[heuristic_name] = result
            except Exception as e:
                # Create an error result if heuristic fails
                results[heuristic_name] = HeuristicResult(
                    heuristic_name=heuristic_name,
                    original_accuracy=0.0,
                    robust_accuracy=0.0,
                    absolute_delta=0.0,
                    confidence_interval=[0.0, 0.0],
                    is_significant=False,
                    p_value=None
                )
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        return HeuristicDegradationResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            runtime_seconds=time.time() - start_time,
            total_original=len(original_questions),
            total_robust=len(robust_questions),
            heuristics=results,
            summary=summary
        )
    
    def _analyze_heuristic(self,
                          heuristic_name: str,
                          heuristic_func: callable,
                          original_questions: List[Dict[str, Any]],
                          robust_questions: List[Dict[str, Any]]) -> HeuristicResult:
        """Analyze a single heuristic with stratified bootstrap"""
        # Get predictions for original and robust datasets
        original_predictions = heuristic_func(original_questions)
        robust_predictions = heuristic_func(robust_questions)
        
        # Calculate accuracies
        original_accuracy = self._calculate_accuracy(original_questions, original_predictions)
        robust_accuracy = self._calculate_accuracy(robust_questions, robust_predictions)
        
        # Calculate absolute delta
        absolute_delta = abs(original_accuracy - robust_accuracy)
        
        # Prepare data for bootstrap
        n = min(len(original_questions), len(robust_questions))
        if n == 0:
            return HeuristicResult(
                heuristic_name=heuristic_name,
                original_accuracy=original_accuracy,
                robust_accuracy=robust_accuracy,
                absolute_delta=absolute_delta,
                confidence_interval=[0.0, 0.0],
                is_significant=False
            )
        
        # Create paired data
        original_data = np.array([1 if original_questions[i].get('answer_index') == original_predictions[i] else 0 
                               for i in range(n)])
        robust_data = np.array([1 if robust_questions[i].get('answer_index') == robust_predictions[i] else 0 
                             for i in range(n)])
        
        # Calculate delta for each pair
        delta_data = np.abs(original_data - robust_data)
        
        # Use stratified bootstrap for the delta
        result = self.stratified_analyzer.analyze_metric(
            data=delta_data,
            statistic=np.mean,
            metric_name=f"{heuristic_name}_delta",
            stratify_by=self.stratify_by
        )
        
        # Check if degradation is significant (CI doesn't include 0)
        is_significant = result.ci_lower > 0
        
        return HeuristicResult(
            heuristic_name=heuristic_name,
            original_accuracy=original_accuracy,
            robust_accuracy=robust_accuracy,
            absolute_delta=absolute_delta,
            confidence_interval=[result.ci_lower, result.ci_upper],
            is_significant=is_significant,
            bootstrap_metadata=asdict(result)  # Add metadata
        )
    
    def _calculate_accuracy(self, questions: List[Dict[str, Any]], predictions: List[int]) -> float:
        """Calculate accuracy of predictions against ground truth"""
        if not questions or not predictions:
            return 0.0
        
        correct = 0
        for q, pred in zip(questions, predictions):
            if q.get('answer_index') == pred:
                correct += 1
        
        return correct / len(questions)
    
    def _longest_answer_heuristic(self, questions: List[Dict[str, Any]]) -> List[int]:
        """Longest answer heuristic: always select the longest option"""
        predictions = []
        for q in questions:
            choices = q.get('choices', [])
            if not choices:
                predictions.append(0)
                continue
            
            # Find the longest choice
            lengths = [len(choice) for choice in choices]
            max_length = max(lengths)
            # If there's a tie, pick the first one
            predictions.append(lengths.index(max_length))
        
        return predictions
    
    def _position_bias_heuristic(self, questions: List[Dict[str, Any]]) -> List[int]:
        """Position bias heuristic: always select the first option"""
        return [0] * len(questions)
    
    def _lexical_patterns_heuristic(self, questions: List[Dict[str, Any]]) -> List[int]:
        """Lexical patterns heuristic: select option with most common keywords"""
        predictions = []
        
        # Common keywords that might indicate correct answers
        common_keywords = ['correct', 'true', 'yes', 'all', 'none', 'not']
        
        for q in questions:
            choices = q.get('choices', [])
            if not choices:
                predictions.append(0)
                continue
            
            # Score each choice based on keyword presence
            scores = []
            for choice in choices:
                score = 0
                choice_lower = choice.lower()
                for keyword in common_keywords:
                    if keyword in choice_lower:
                        score += 1
                scores.append(score)
            
            # Select choice with highest score
            max_score = max(scores)
            # If there's a tie, pick the first one
            predictions.append(scores.index(max_score))
        
        return predictions
    
    def _calculate_summary(self, results: Dict[str, HeuristicResult]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        total_heuristics = len(results)
        significant_degradations = sum(1 for r in results.values() if r.is_significant)
        
        # Average degradation
        avg_delta = np.mean([r.absolute_delta for r in results.values()])
        
        # Maximum degradation
        max_delta = max([r.absolute_delta for r in results.values()])
        
        return {
            'total_heuristics': total_heuristics,
            'significant_degradations': significant_degradations,
            'average_degradation': avg_delta,
            'maximum_degradation': max_delta,
            'degradation_percentage': (significant_degradations / total_heuristics * 100) if total_heuristics > 0 else 0
        }
    
    def to_json(self, result: HeuristicDegradationResult) -> str:
        """Convert result to JSON"""
        # Convert dataclass to dict and handle numpy types
        result_dict = asdict(result)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        result_dict = convert_numpy(result_dict)
        return json.dumps(result_dict, indent=2)
    
    def save_json(self, result: HeuristicDegradationResult, filepath: str):
        """Save result to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json(result))
