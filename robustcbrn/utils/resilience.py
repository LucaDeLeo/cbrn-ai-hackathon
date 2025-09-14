"""Resilience utilities for API calls and model evaluations."""

import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    exponential_base: float = 2.0
    max_delay: float = 60.0
    timeout: float | None = 120.0
    jitter: bool = True


class RetryableError(Exception):
    """Base exception for retryable errors."""
    pass


class TimeoutError(RetryableError):
    """Raised when an operation times out."""
    pass


class APIError(RetryableError):
    """Raised for API-related errors."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_retryable(self) -> bool:
        """Check if error is retryable based on status code."""
        if self.status_code is None:
            return True
        # Retry on 429 (rate limit), 500-599 (server errors), 408 (timeout)
        return self.status_code in {408, 429} or 500 <= self.status_code < 600


def exponential_backoff_with_jitter(
    attempt: int,
    initial_delay: float,
    exponential_base: float,
    max_delay: float,
    jitter: bool = True
) -> float:
    """Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential growth
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    delay = min(initial_delay * (exponential_base ** attempt), max_delay)

    if jitter:
        import random
        # Add jitter: random value between 0 and delay
        delay = random.uniform(0, delay)

    return delay


def with_retry(
    config: RetryConfig | None = None,
    retriable_exceptions: tuple = (RetryableError, ConnectionError, TimeoutError)
) -> Callable:
    """Decorator to add retry logic to functions.

    Args:
        config: Retry configuration
        retriable_exceptions: Tuple of exceptions to retry on

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    # Add timeout if specified
                    if config.timeout:
                        import signal

                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Operation timed out after {config.timeout}s")

                        # Set timeout alarm (Unix only)
                        try:
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(int(config.timeout))
                        except (AttributeError, OSError):
                            # Windows or non-Unix system, skip timeout
                            pass

                    # Execute function
                    result = func(*args, **kwargs)

                    # Cancel timeout if set
                    with contextlib.suppress(AttributeError, OSError):
                        signal.alarm(0)

                    return result

                except retriable_exceptions as e:
                    last_exception = e

                    # Check if it's an APIError that's not retryable
                    if isinstance(e, APIError) and not e.is_retryable:
                        logger.error(f"Non-retryable API error: {e}")
                        raise

                    # Calculate retry delay
                    if attempt < config.max_attempts - 1:
                        delay = exponential_backoff_with_jitter(
                            attempt,
                            config.initial_delay,
                            config.exponential_base,
                            config.max_delay,
                            config.jitter
                        )

                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed. Last error: {e}"
                        )

                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Non-retryable error: {e}")
                    raise

            # All retries exhausted
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"Failed after {config.max_attempts} attempts")

        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting to close circuit
            expected_exception: Exception type to track
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed, open, half-open

    @property
    def state(self) -> str:
        """Get current circuit state."""
        if (self._state == "open" and
            self._last_failure_time and
            time.time() - self._last_failure_time >= self.recovery_timeout):
            self._state = "half-open"
        return self._state

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            RuntimeError: If circuit is open
            Exception: If func raises exception
        """
        if self.state == "open":
            raise RuntimeError("Circuit breaker is open - service unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self._state == "half-open":
            logger.info("Circuit breaker closing after successful call")
            self._state = "closed"
        self._failure_count = 0
        self._last_failure_time = None

    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker opening after {self._failure_count} failures"
            )
            self._state = "open"
        elif self._state == "half-open":
            logger.warning("Circuit breaker reopening after failure in half-open state")
            self._state = "open"


def with_circuit_breaker(breaker: CircuitBreaker) -> Callable:
    """Decorator to add circuit breaker to functions.

    Args:
        breaker: Circuit breaker instance

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


# Example usage for Inspect tasks
def create_resilient_solver(base_solver: Any, config: RetryConfig | None = None) -> Any:
    """Wrap an Inspect solver with retry logic.

    Args:
        base_solver: Original solver
        config: Retry configuration

    Returns:
        Wrapped solver with retry capability
    """
    if config is None:
        config = RetryConfig()

    class ResilientSolver:
        def __init__(self, solver, retry_config):
            self.solver = solver
            self.retry_config = retry_config

        @with_retry(config=config)
        def __call__(self, *args, **kwargs):
            return self.solver(*args, **kwargs)

    return ResilientSolver(base_solver, config)


# Model API wrapper with built-in resilience
class ResilientModelAPI:
    """Wrapper for model APIs with retry and circuit breaker."""

    def __init__(
        self,
        base_api: Any,
        retry_config: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None
    ):
        """Initialize resilient API wrapper.

        Args:
            base_api: Original API client
            retry_config: Retry configuration
            circuit_breaker: Circuit breaker instance
        """
        self.base_api = base_api
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    @with_retry(config=RetryConfig())
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response with resilience.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        def _generate():
            try:
                return self.base_api.generate(prompt, **kwargs)
            except Exception as e:
                # Convert to retryable error if appropriate
                if "rate" in str(e).lower() or "timeout" in str(e).lower():
                    raise RetryableError(str(e)) from e
                raise

        if self.circuit_breaker:
            return self.circuit_breaker.call(_generate)
        else:
            return _generate()


# Health check utilities
class HealthChecker:
    """Simple health checker for services."""

    def __init__(self, check_interval: float = 30.0):
        """Initialize health checker.

        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        self._last_check = 0.0
        self._is_healthy = True

    def is_healthy(self, check_func: Callable[[], bool] | None = None) -> bool:
        """Check if service is healthy.

        Args:
            check_func: Optional function to check health

        Returns:
            True if healthy
        """
        now = time.time()

        if now - self._last_check >= self.check_interval:
            if check_func:
                try:
                    self._is_healthy = check_func()
                except Exception:
                    self._is_healthy = False
            self._last_check = now

        return self._is_healthy
