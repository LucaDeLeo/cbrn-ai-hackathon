"""Quality Assurance module for RobustCBRN."""

from .ambiguity import (
    AmbiguityDecision,
    AmbiguityMetrics,
    AmbiguityDetectionError,
    InvalidChoicesError,
    audit_dataset,
    decisions_to_records,
    llm_critic_votes,
)

from .ambiguity_config import (
    AmbiguityConfig,
    DEFAULT_CONFIG,
)

__all__ = [
    # Main functions
    "audit_dataset",
    "decisions_to_records",
    "llm_critic_votes",
    # Data classes
    "AmbiguityDecision",
    "AmbiguityMetrics",
    "AmbiguityConfig",
    "DEFAULT_CONFIG",
    # Exceptions
    "AmbiguityDetectionError",
    "InvalidChoicesError",
]