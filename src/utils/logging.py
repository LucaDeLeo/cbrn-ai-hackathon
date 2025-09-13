from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: str = "logs", filename: str = "app.log", level: str = "INFO") -> Logger:
    """Configure dual console/file logging using stdlib logging.

    Creates the logs directory if needed and sets a consistent formatter.
    Multiple calls are safe; handlers are added only once.
    """
    logger = logging.getLogger("cbrn")
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / filename

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logger.level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger._configured = True  # type: ignore[attr-defined]
    logger.debug("Logging configured")
    return logger

