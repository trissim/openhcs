"""Native library thread-pool configuration for OpenHCS execution processes."""

from __future__ import annotations

import os

from threadpoolctl import threadpool_limits


_NATIVE_THREAD_COUNT_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "NUMBA_NUM_THREADS",
)


def native_thread_count_environment_keys() -> tuple[str, ...]:
    """Return the variables owned by native-library thread configuration."""

    return _NATIVE_THREAD_COUNT_ENVIRONMENT_VARIABLES


def configure_native_thread_environment(thread_count: int) -> None:
    """Configure native runtimes imported later in this process or its children."""

    if thread_count < 1:
        raise ValueError("Native thread count must be positive.")

    value = str(thread_count)
    for variable in native_thread_count_environment_keys():
        os.environ.setdefault(variable, value)


def configure_native_thread_count(thread_count: int) -> None:
    """Configure future and already-loaded native runtimes for one process."""

    configure_native_thread_environment(thread_count)
    threadpool_limits(limits=thread_count)

    import cv2

    cv2.setNumThreads(thread_count)
