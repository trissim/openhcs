"""Peak process-tree memory usage metric."""

import threading
import time

import psutil

from benchmark.contracts.metric import MetricCollector


class MemoryMetric(MetricCollector):
    """Samples process-tree RSS in a background thread and reports peak MB."""

    name = "peak_memory_mb"

    def __init__(self, interval_seconds: float = 0.1, include_children: bool = True):
        self.interval = interval_seconds
        self.include_children = include_children
        self._running = False
        self._peak_rss = 0
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self._started = False
        self._sampling_error: Exception | None = None

    def __enter__(self) -> "MemoryMetric":
        self._peak_rss = 0
        self._sampling_error = None
        self._running = True
        self._started = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _sample_loop(self) -> None:
        while self._running:
            rss = self._sample_process_tree_rss()
            if rss > self._peak_rss:
                self._peak_rss = rss
            time.sleep(self.interval)

    def _sample_process_tree_rss(self) -> int:
        try:
            rss = self._process.memory_info().rss
        except psutil.NoSuchProcess as exc:
            self._sampling_error = exc
            self._running = False
            return self._peak_rss
        if not self.include_children:
            return rss
        try:
            children = self._process.children(recursive=True)
        except psutil.NoSuchProcess as exc:
            self._sampling_error = exc
            self._running = False
            return rss
        for child in children:
            try:
                rss += child.memory_info().rss
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        return rss

    def get_result(self) -> float:
        if not self._started:
            raise RuntimeError("MemoryMetric not used as context manager")
        if self._peak_rss == 0:
            if self._sampling_error is not None:
                raise RuntimeError("MemoryMetric failed to sample process RSS") from (
                    self._sampling_error
                )
            raise RuntimeError("MemoryMetric recorded no samples")
        return self._peak_rss / (1024 * 1024)
