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


def configure_native_thread_count(thread_count: int) -> None:
    """Configure future and already-loaded native runtimes for one process."""

    if thread_count < 1:
        raise ValueError("Native thread count must be positive.")

    value = str(thread_count)
    for variable in _NATIVE_THREAD_COUNT_ENVIRONMENT_VARIABLES:
        os.environ.setdefault(variable, value)

    threadpool_limits(limits=thread_count)

    import cv2

    cv2.setNumThreads(thread_count)
