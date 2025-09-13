"""Analysis module for RobustCBRN Eval."""

from .heuristics import LongestAnswerHeuristic, analyze_questions, HeuristicReport
from .statistical_battery import StatisticalBattery, TestResult, BatteryResult
from .statistical import (
    chi_square_test,
    calculate_bootstrap_ci,
    detect_position_bias,
    analyze_lexical_patterns,
)
from .patterns import (
    PatternReport,
    detect_lexical_patterns,
    analyze_phrase_frequencies,
    analyze_length_distributions,
    detect_meta_patterns,
    calculate_technical_density,
    calculate_effect_size
)

__all__ = [
    "LongestAnswerHeuristic", 
    "analyze_questions", 
    "HeuristicReport",
    "StatisticalBattery",
    "TestResult", 
    "BatteryResult",
    "chi_square_test",
    "calculate_bootstrap_ci", 
    "detect_position_bias",
    "analyze_lexical_patterns",
    "PatternReport",
    "detect_lexical_patterns",
    "analyze_phrase_frequencies", 
    "analyze_length_distributions",
    "detect_meta_patterns",
    "calculate_technical_density",
    "calculate_effect_size"
]