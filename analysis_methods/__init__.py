"""
Analysis methods for language efficiency testing.
"""

from analysis_methods.cross_validation import (
    perform_cross_validation,
    calculate_statistical_significance,
    validate_across_difficulty_levels
)

from analysis_methods.information_density import (
    calculate_bits_per_token,
    calculate_semantic_density,
    analyze_chinese_information_density
)

from analysis_methods.compression_metrics import (
    calculate_compression_ratio,
    analyze_token_usage_by_difficulty,
    calculate_normalized_compression_metrics
)

from analysis_methods.longcontext_analysis import (
    analyze_context_length_impact,
    create_context_length_visualizations,
    analyze_longcontext_strategic_selection,
    create_strategic_selection_visualizations
)
