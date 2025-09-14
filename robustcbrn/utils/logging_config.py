"""Structured logging configuration for RobustCBRN."""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "task"):
            log_obj["task"] = record.task
        if hasattr(record, "model"):
            log_obj["model"] = record.model
        if hasattr(record, "seed"):
            log_obj["seed"] = record.seed
        if hasattr(record, "dataset"):
            log_obj["dataset"] = record.dataset
        if hasattr(record, "metrics"):
            log_obj["metrics"] = record.metrics
        if hasattr(record, "error_type"):
            log_obj["error_type"] = record.error_type
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_obj, default=str)


class TaskContextFilter(logging.Filter):
    """Filter to add task context to log records."""

    def __init__(self, task_name: str | None = None):
        """Initialize filter with task context.

        Args:
            task_name: Name of the current task
        """
        super().__init__()
        self.task_name = task_name

    def filter(self, record: logging.LogRecord) -> bool:
        """Add task context to record.

        Args:
            record: Log record to filter

        Returns:
            True (always passes through)
        """
        if self.task_name:
            record.task = self.task_name
        return True


class MetricsLogger:
    """Logger for metrics and performance data."""

    def __init__(self, logger_name: str = "robustcbrn.metrics"):
        """Initialize metrics logger.

        Args:
            logger_name: Name for the logger
        """
        self.logger = logging.getLogger(logger_name)

    def log_evaluation_start(
        self,
        task: str,
        model: str,
        dataset: str,
        seed: int,
        **kwargs
    ):
        """Log start of evaluation.

        Args:
            task: Task name
            model: Model name
            dataset: Dataset path
            seed: Random seed
            **kwargs: Additional metadata
        """
        self.logger.info(
            "Evaluation started",
            extra={
                "task": task,
                "model": model,
                "dataset": dataset,
                "seed": seed,
                **kwargs
            }
        )

    def log_evaluation_complete(
        self,
        task: str,
        model: str,
        duration_ms: float,
        samples_processed: int,
        accuracy: float | None = None,
        **kwargs
    ):
        """Log completion of evaluation.

        Args:
            task: Task name
            model: Model name
            duration_ms: Duration in milliseconds
            samples_processed: Number of samples processed
            accuracy: Overall accuracy if available
            **kwargs: Additional metrics
        """
        metrics = {
            "samples_processed": samples_processed,
            "duration_ms": duration_ms,
            "samples_per_second": samples_processed / (duration_ms / 1000) if duration_ms > 0 else 0
        }
        if accuracy is not None:
            metrics["accuracy"] = accuracy

        metrics.update(kwargs)

        self.logger.info(
            "Evaluation complete",
            extra={
                "task": task,
                "model": model,
                "metrics": metrics
            }
        )

    def log_api_call(
        self,
        provider: str,
        model: str,
        duration_ms: float,
        tokens_used: int | None = None,
        success: bool = True,
        error: str | None = None
    ):
        """Log API call metrics.

        Args:
            provider: API provider
            model: Model name
            duration_ms: Call duration
            tokens_used: Tokens consumed
            success: Whether call succeeded
            error: Error message if failed
        """
        level = logging.INFO if success else logging.ERROR
        metrics = {
            "provider": provider,
            "model": model,
            "duration_ms": duration_ms,
            "success": success
        }

        if tokens_used:
            metrics["tokens_used"] = tokens_used
        if error:
            metrics["error"] = error

        self.logger.log(
            level,
            "API call" if success else "API call failed",
            extra={"metrics": metrics}
        )

    def log_aggregation_metrics(
        self,
        total_samples: int,
        total_models: int,
        metrics_computed: dict[str, Any],
        duration_ms: float
    ):
        """Log aggregation metrics.

        Args:
            total_samples: Total samples aggregated
            total_models: Number of models
            metrics_computed: Dictionary of computed metrics
            duration_ms: Aggregation duration
        """
        self.logger.info(
            "Aggregation complete",
            extra={
                "metrics": {
                    "total_samples": total_samples,
                    "total_models": total_models,
                    "duration_ms": duration_ms,
                    "computed_metrics": metrics_computed
                }
            }
        )


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    structured: bool = True,
    task_name: str | None = None
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logs
        structured: Whether to use structured JSON logging
        task_name: Optional task name for context
    """
    # Parse level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatters
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Add task context filter if specified
    if task_name:
        task_filter = TaskContextFilter(task_name)
        for handler in root_logger.handlers:
            handler.addFilter(task_filter)

    # Configure specific loggers
    logging.getLogger("robustcbrn").setLevel(log_level)
    logging.getLogger("inspect_ai").setLevel(logging.WARNING)  # Less verbose for Inspect


class LogContext:
    """Context manager for temporary logging configuration."""

    def __init__(
        self,
        task: str | None = None,
        model: str | None = None,
        dataset: str | None = None,
        **kwargs
    ):
        """Initialize log context.

        Args:
            task: Task name
            model: Model name
            dataset: Dataset name
            **kwargs: Additional context
        """
        self.context = {
            "task": task,
            "model": model,
            "dataset": dataset,
            **kwargs
        }
        self.old_factory = None

    def __enter__(self):
        """Enter context and set up logging."""
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in self.context.items():
                if value is not None:
                    setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        self.old_factory = old_factory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore logging."""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


# Predefined loggers
def get_task_logger(task_name: str) -> logging.Logger:
    """Get a logger configured for a specific task.

    Args:
        task_name: Name of the task

    Returns:
        Configured logger
    """
    logger = logging.getLogger(f"robustcbrn.tasks.{task_name}")
    logger.addFilter(TaskContextFilter(task_name))
    return logger


def get_analysis_logger() -> logging.Logger:
    """Get logger for analysis operations.

    Returns:
        Configured logger
    """
    return logging.getLogger("robustcbrn.analysis")


def get_api_logger() -> logging.Logger:
    """Get logger for API operations.

    Returns:
        Configured logger
    """
    return logging.getLogger("robustcbrn.api")


# Performance tracking decorator
def log_performance(logger: logging.Logger | None = None):
    """Decorator to log function performance.

    Args:
        logger: Logger to use (defaults to module logger)

    Returns:
        Decorated function
    """
    def decorator(func):
        import time
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Function {func.__name__} completed",
                    extra={"duration_ms": duration_ms}
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Function {func.__name__} failed",
                    extra={
                        "duration_ms": duration_ms,
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
                raise

        return wrapper
    return decorator
