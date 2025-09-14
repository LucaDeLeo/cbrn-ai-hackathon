"""RobustCBRN Eval package.

Utilities and Inspect tasks for robustifying MCQ-style CBRN evaluations.

This package avoids hazardous content; examples are sanitized.
"""

from .analysis import analyze_questions
from .cli import main as cli_main
from .config import AppConfig, default_app_config
from .data import Question, load_dataset
from .security import make_question_id
from .statistical import detect_position_bias
from .utils import set_determinism, setup_logging

__all__ = [
    "__version__",
    "AppConfig",
    "default_app_config",
    "Question",
    "load_dataset",
    "analyze_questions",
    "detect_position_bias",
    "make_question_id",
    "setup_logging",
    "set_determinism",
    "cli_main",
]

__version__ = "0.1.0"

