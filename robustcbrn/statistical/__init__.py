"""Statistical analysis modules for RobustCBRN."""

from .bootstrap import (
    BootstrapResult,
    bootstrap_chi_square_pvalue_ci,
    bootstrap_ci,
    bootstrap_mean_ci,
    bootstrap_median_ci,
    bootstrap_proportion_ci,
)
from .position_bias import (
    PositionBiasReport,
    analyze_position_bias,
    calculate_position_frequencies,
    detect_position_bias,
    generate_position_swaps,
    identify_predictive_questions,
)

__all__ = [
    # Bootstrap
    "BootstrapResult",
    "bootstrap_ci",
    "bootstrap_mean_ci",
    "bootstrap_median_ci",
    "bootstrap_proportion_ci",
    "bootstrap_chi_square_pvalue_ci",
    # Position bias
    "PositionBiasReport",
    "calculate_position_frequencies",
    "analyze_position_bias",
    "detect_position_bias",
    "identify_predictive_questions",
    "generate_position_swaps",
]
