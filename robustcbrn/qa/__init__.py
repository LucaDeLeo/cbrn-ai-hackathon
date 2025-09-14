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

from .paraphrase import generate_paraphrases, generate_k_paraphrases  # noqa: F401
from .perturb import generate_perturbations  # noqa: F401
from .rules import check_dataset_hygiene  # noqa: F401

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
    # Paraphrase / Perturb
    "generate_paraphrases",
    "generate_k_paraphrases",
    "generate_perturbations",
    "check_dataset_hygiene",
]
