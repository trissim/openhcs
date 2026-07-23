"""Peak process-tree memory usage metric."""

from collections.abc import Callable
import _thread
import threading
import time
from typing import ClassVar

from openhcs.core.alias_property import AliasProperty
import psutil

from benchmark.contracts.metric import MetricCollector


class MemoryMetric(MetricCollector):
    """Samples process-tree RSS in a background thread and reports peak MB."""

    name = "peak_memory_mb"
    limit_exceeded: ClassVar[AliasProperty[bool]] = AliasProperty("_limit_exceeded")

    def __init__(
        self,
        interval_seconds: float = 0.1,
        include_children: bool = True,
        *,
        max_memory_mb: float | None = None,
        on_limit_exceeded: Callable[[float, tuple[psutil.Process, ...]], None] | None = None,
        limit_callback_interval_seconds: float = 0.5,
        interrupt_main_on_limit: bool = False,
    ):
        self.interval = interval_seconds
        self.include_children = include_children
        self.max_memory_bytes = (
            int(max_memory_mb * 1024 * 1024)
            if max_memory_mb is not None
            else None
        )
        self.on_limit_exceeded = on_limit_exceeded
        self.limit_callback_interval_seconds = limit_callback_interval_seconds
        self.interrupt_main_on_limit = interrupt_main_on_limit
        self._running = False
        self._peak_rss = 0
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self._started = False
        self._sampling_error: Exception | None = None
        self._limit_exceeded = False
        self._last_limit_callback_at: float | None = None

    def __enter__(self) -> "MemoryMetric":
        self._peak_rss = 0
        self._sampling_error = None
        self._limit_exceeded = False
        self._last_limit_callback_at = None
        self._running = True
        self._started = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except KeyboardInterrupt:
                if not (self.interrupt_main_on_limit and self._limit_exceeded):
                    raise

    def _sample_loop(self) -> None:
        while self._running:
            rss, children = self._sample_process_tree_rss()
            if rss > self._peak_rss:
                self._peak_rss = rss
            if self.max_memory_bytes is not None and rss > self.max_memory_bytes:
                self._enforce_limit(rss, children)
            time.sleep(self.interval)

    def _enforce_limit(self, rss: int, children: tuple[psutil.Process, ...]) -> None:
        first_exceedance = not self._limit_exceeded
        self._limit_exceeded = True
        now = time.monotonic()
        callback_due = (
            self._last_limit_callback_at is None
            or now - self._last_limit_callback_at
            >= self.limit_callback_interval_seconds
        )
        if self.on_limit_exceeded is not None and callback_due:
            self._last_limit_callback_at = now
            self.on_limit_exceeded(rss / (1024 * 1024), children)
        if self.interrupt_main_on_limit and first_exceedance:
            _thread.interrupt_main()

    def _sample_process_tree_rss(self) -> tuple[int, tuple[psutil.Process, ...]]:
        try:
            rss = self._process.memory_info().rss
        except psutil.NoSuchProcess as exc:
            self._sampling_error = exc
            self._running = False
            return self._peak_rss, ()
        if not self.include_children:
            return rss, ()
        try:
            children = tuple(self._process.children(recursive=True))
        except psutil.NoSuchProcess as exc:
            self._sampling_error = exc
            self._running = False
            return rss, ()
        for child in children:
            try:
                rss += child.memory_info().rss
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        return rss, children

    def get_result(self) -> float:
        if not self._started:
            raise RuntimeError("MemoryMetric not used as context manager")
        if self._peak_rss == 0:
            if self._sampling_error is not None:
                raise RuntimeError("MemoryMetric failed to sample process RSS") from (
                    self._sampling_error
                )
            rss, _children = self._sample_process_tree_rss()
            self._peak_rss = max(self._peak_rss, rss)
        if self._peak_rss == 0:
            raise RuntimeError("MemoryMetric recorded no samples")
        return self._peak_rss / (1024 * 1024)
