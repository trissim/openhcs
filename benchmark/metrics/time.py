"""Wall-clock timing metric."""

import time

from benchmark.contracts.metric import MetricCollector


class TimeMetric(MetricCollector):
    """Measures execution time using perf_counter."""

    name = "execution_time_seconds"

    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self) -> "TimeMetric":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()

    def get_result(self) -> float:
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("TimeMetric not used as context manager")
        return self.end_time - self.start_time
