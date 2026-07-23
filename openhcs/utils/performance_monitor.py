"""Performance monitoring utilities for OpenHCS.

Provides decorators and context managers for timing operations and
logging performance metrics.
"""

import time
import functools
import logging
import os
from contextlib import contextmanager
from typing import Optional, Callable
from pathlib import Path

# Create performance logger
perf_logger = logging.getLogger("openhcs.performance")
perf_logger.setLevel(logging.WARNING)

# Add file handler for performance logs
perf_log_file = (
    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "openhcs"
    / "logs"
    / "performance.log"
)
perf_log_file.parent.mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler(perf_log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
perf_logger.addHandler(file_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("⏱️  %(message)s"))
perf_logger.addHandler(console_handler)


@contextmanager
def timer(
    operation_name: str, threshold_ms: float = 0.0, log_args: bool = False, **kwargs
):
    """Context manager for timing operations.

    Args:
        operation_name: Name of the operation being timed
        threshold_ms: Only log if operation takes longer than this (in milliseconds)
        log_args: Whether to log kwargs in the message
        **kwargs: Additional context to include in log message

    Example:
        with timer("Loading config", threshold_ms=10.0, config_type="GlobalPipelineConfig"):
            config = load_config()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms >= threshold_ms:
            msg = f"{operation_name}: {elapsed_ms:.2f}ms"
            if log_args and kwargs:
                args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                msg += f" ({args_str})"

            perf_logger.debug(msg)


def timed(operation_name: Optional[str] = None, threshold_ms: float = 0.0):
    """Decorator for timing function calls.

    Args:
        operation_name: Name for the operation (defaults to function name)
        threshold_ms: Only log if operation takes longer than this (in milliseconds)

    Example:
        @timed("Config loading", threshold_ms=10.0)
        def load_config():
            ...
    """

    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000

                if elapsed_ms >= threshold_ms:
                    perf_logger.debug(f"{operation_name}: {elapsed_ms:.2f}ms")

        return wrapper

    return decorator


class PerformanceMonitor:
    """Accumulates timing statistics for repeated operations.

    Example:
        monitor = PerformanceMonitor("Placeholder resolution")

        for field in fields:
            with monitor.measure():
                resolve_placeholder(field)

        monitor.report()  # Logs summary statistics
    """

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.timings = []
        self.current_start = None

    @contextmanager
    def measure(self):
        """Measure a single operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.timings.append(elapsed_ms)

    def report(self, log_individual: bool = False):
        """Log summary statistics.

        Args:
            log_individual: Whether to log each individual timing
        """
        if not self.timings:
            perf_logger.debug(f"{self.operation_name}: No measurements")
            return

        count = len(self.timings)
        total_ms = sum(self.timings)
        avg_ms = total_ms / count
        min_ms = min(self.timings)
        max_ms = max(self.timings)

        perf_logger.debug(
            f"{self.operation_name} - "
            f"Count: {count}, "
            f"Total: {total_ms:.2f}ms, "
            f"Avg: {avg_ms:.2f}ms, "
            f"Min: {min_ms:.2f}ms, "
            f"Max: {max_ms:.2f}ms"
        )

        if log_individual:
            for i, timing in enumerate(self.timings, 1):
                perf_logger.debug(f"  #{i}: {timing:.2f}ms")

    def reset(self):
        """Clear all timings."""
        self.timings.clear()


# Global monitors for common operations
_monitors = {}


def get_monitor(operation_name: str) -> PerformanceMonitor:
    """Get or create a global monitor for an operation.

    Example:
        monitor = get_monitor("Placeholder resolution")
        with monitor.measure():
            resolve_placeholder(field)
    """
    if operation_name not in _monitors:
        _monitors[operation_name] = PerformanceMonitor(operation_name)
    return _monitors[operation_name]


def report_all_monitors():
    """Report statistics for all global monitors."""
    if not _monitors:
        perf_logger.debug("No performance monitors active")
        return

    perf_logger.debug("=" * 60)
    perf_logger.debug("PERFORMANCE SUMMARY")
    perf_logger.debug("=" * 60)

    for monitor in _monitors.values():
        monitor.report()

    perf_logger.debug("=" * 60)


def reset_all_monitors():
    """Reset all global monitors."""
    for monitor in _monitors.values():
        monitor.reset()


# Convenience function to enable/disable performance logging
_enabled = True


def enable_performance_logging():
    """Enable performance logging."""
    global _enabled
    _enabled = True
    perf_logger.setLevel(logging.DEBUG)


def disable_performance_logging():
    """Disable performance logging."""
    global _enabled
    _enabled = False
    perf_logger.setLevel(logging.WARNING)


def is_performance_logging_enabled() -> bool:
    """Check if performance logging is enabled."""
    return _enabled
