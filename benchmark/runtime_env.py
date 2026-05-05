"""Shared runtime environment setup for benchmark entrypoints."""

from __future__ import annotations

import logging
import os


def configure_headless_cpu_benchmark_runtime(log_level: str) -> None:
    """Configure deterministic CPU-only benchmark runtime before OpenHCS imports."""
    normalized_log_level = _validated_log_level_name(log_level)
    _configure_native_thread_limits()
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("XDG_DATA_HOME", "/tmp/openhcs-benchmark-xdg-data")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/openhcs-benchmark-xdg-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/openhcs-benchmark-mpl")
    os.environ.setdefault("OPENHCS_CPU_ONLY", "true")
    os.environ.setdefault("OPENHCS_SUBPROCESS_NO_GPU", "1")
    os.environ.setdefault("POLYSTORE_SUBPROCESS_NO_GPU", "1")
    os.environ["OPENHCS_LOG_LEVEL"] = normalized_log_level
    _configure_opencv_threads()
    configure_benchmark_logging(normalized_log_level)


def configure_benchmark_logging(log_level: str) -> None:
    """Configure Python logging for benchmark harness and worker inheritance."""
    level = _validated_log_level(log_level)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def _configure_native_thread_limits() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


def _configure_opencv_threads() -> None:
    import cv2

    cv2.setNumThreads(1)


def _validated_log_level_name(log_level: str) -> str:
    _validated_log_level(log_level)
    return log_level.upper()


def _validated_log_level(log_level: str) -> int:
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {log_level!r}")
    return level
