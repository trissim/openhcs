"""Metric collector abstract base class for benchmark platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

from benchmark.contracts.values import BenchmarkMetricValue


class MetricCollector(ABC):
    """
    Abstract base class for metric collectors.

    Metrics are context managers that automatically collect data
    during tool execution.

    Adding a new metric = extending this ABC and implementing abstract methods.

    Subclasses must define class attribute:
        name: str - Metric name (e.g., 'execution_time', 'peak_memory_mb')
    """

    name: ClassVar[str]

    @abstractmethod
    def __enter__(self) -> Self:
        """Start metric collection."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop metric collection."""
        pass

    @abstractmethod
    def get_result(self) -> BenchmarkMetricValue:
        """Get collected metric value."""
        pass
