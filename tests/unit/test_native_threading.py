import os

import pytest

from openhcs.core import native_threading


def test_configure_native_thread_count_updates_future_and_loaded_runtimes(monkeypatch):
    observed_limits = []
    observed_opencv_limits = []

    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    monkeypatch.setattr(
        native_threading,
        "threadpool_limits",
        lambda *, limits: observed_limits.append(limits),
    )

    import cv2

    monkeypatch.setattr(cv2, "setNumThreads", observed_opencv_limits.append)

    native_threading.configure_native_thread_count(3)

    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    assert os.environ["NUMBA_NUM_THREADS"] == "3"
    assert observed_limits == [3]
    assert observed_opencv_limits == [3]


def test_configure_native_thread_count_rejects_non_positive_counts():
    with pytest.raises(ValueError, match="must be positive"):
        native_threading.configure_native_thread_count(0)
