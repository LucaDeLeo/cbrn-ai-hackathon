"""RobustCBRN Eval package.

Utilities and Inspect tasks for robustifying MCQ-style CBRN evaluations.

This package avoids hazardous content; examples are sanitized.
"""

from .config import AppConfig, default_app_config
from .data import Question, load_dataset
from .analysis import analyze_questions
from .statistical import detect_position_bias
from .security import make_question_id
from .utils import setup_logging, set_determinism
from .cli import main as cli_main

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

