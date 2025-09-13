"""Configuration for ambiguity detection module."""

from dataclasses import dataclass
from typing import Dict, Set


@dataclass
class AmbiguityConfig:
    """Configuration settings for ambiguity detection."""

    # Similarity thresholds
    jaccard_threshold: float = 0.9  # For near-duplicate detection
    numeric_proximity_threshold: float = 0.01  # 1% proximity for numeric crowding

    # Token limits
    max_tokens_for_boolean: int = 2  # Max tokens to consider as boolean-like

    # Meta option patterns
    meta_options: Dict[str, str] = None

    # Negation patterns
    negation_prefixes: Set[str] = None
    negation_antonyms: Dict[str, str] = None

    def __post_init__(self):
        """Initialize default patterns if not provided."""
        if self.meta_options is None:
            self.meta_options = {
                "all of the above": "all_of_the_above",
                "all the above": "all_of_the_above",
                "none of the above": "none_of_the_above",
                "none of above": "none_of_the_above",
                "both a and b": "both_and",
                "both b and c": "both_and",
                "both a & b": "both_and",
                "i don't know": "dont_know",
                "i dont know": "dont_know",
                "not sure": "dont_know",
                "cannot determine": "dont_know",
                "insufficient information": "dont_know",
            }

        if self.negation_prefixes is None:
            self.negation_prefixes = {
                "not", "no", "never", "neither", "none", "nothing",
                "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't",
                "shouldn't", "couldn't", "can't", "cannot", "don't", "doesn't", "didn't"
            }

        if self.negation_antonyms is None:
            self.negation_antonyms = {
                "safe": "unsafe",
                "secure": "insecure",
                "stable": "unstable",
                "increases": "decreases",
                "raises": "lowers",
                "improves": "worsens",
                "strengthens": "weakens",
                "accelerates": "decelerates",
                "expands": "contracts",
                "grows": "shrinks",
            }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AmbiguityConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "jaccard_threshold": self.jaccard_threshold,
            "numeric_proximity_threshold": self.numeric_proximity_threshold,
            "max_tokens_for_boolean": self.max_tokens_for_boolean,
            "meta_options": self.meta_options,
            "negation_prefixes": list(self.negation_prefixes),
            "negation_antonyms": self.negation_antonyms,
        }


# Default configuration instance
DEFAULT_CONFIG = AmbiguityConfig()