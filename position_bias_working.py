"""Compatibility layer that re-exports the position bias API.

This avoids code duplication by pointing all imports to the canonical
implementation under src/statistical/position_bias.py.
"""

from src.statistical.position_bias import (
    PositionBiasReport,
    calculate_position_frequencies,
    chi_square_test_from_scratch,
    _approximate_chi2_pvalue,
    _approximate_normal_cdf,
    identify_predictive_questions,
    generate_position_swaps,
    _calculate_checksum,
    analyze_position_bias,
    run_position_bias_analysis,
)

