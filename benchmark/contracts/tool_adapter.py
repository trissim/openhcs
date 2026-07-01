"""Tool adapter abstract base class for benchmark platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from dataclasses import dataclass

from benchmark.contracts.metric import MetricCollector
from benchmark.contracts.values import (
    BenchmarkMetricMap,
    BenchmarkParameterMap,
    BenchmarkProvenanceMap,
)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """
    Normalized result from any tool execution.

    All tool adapters must return this structure.
    """

    tool_name: str
    dataset_id: str
    pipeline_name: str
    metrics: BenchmarkMetricMap
    output_path: Path
    success: bool
    error_message: str | None = None
    provenance: BenchmarkProvenanceMap | None = None


class ToolAdapter(ABC):
    """
    Abstract base class that all tool adapters must extend.

    Adding a new tool = extending this ABC and implementing abstract methods.

    Subclasses must define class attributes:
        name: str - Tool name (e.g., 'OpenHCS', 'CellProfiler')
        version: str - Tool version string
    """

    @abstractmethod
    def validate_installation(self) -> None:
        """
        Verify tool is installed and functional.

        Raises:
            ToolNotInstalledError: If tool is not available
            ToolVersionError: If tool version is incompatible
        """
        pass

    @abstractmethod
    def run(
        self,
        dataset_path: Path,
        pipeline_name: str,
        pipeline_params: BenchmarkParameterMap,
        metrics: Sequence[MetricCollector],
        output_dir: Path,
    ) -> BenchmarkResult:
        """
        Execute tool on dataset with specified pipeline.

        Args:
            dataset_path: Path to dataset root
            pipeline_name: Pipeline identifier (e.g., 'nuclei_segmentation')
            pipeline_params: Pipeline parameters
            metrics: Metric collectors (context managers)
            output_dir: Where to write outputs

        Returns:
            BenchmarkResult with metrics and outputs
        """
        pass


class ToolAdapterError(Exception):
    """Base exception for tool adapter errors."""

    pass


class ToolNotInstalledError(ToolAdapterError):
    """Tool not installed or not found."""

    pass


class ToolVersionError(ToolAdapterError):
    """Tool version incompatible."""

    pass


class ToolExecutionError(ToolAdapterError):
    """Tool execution failed."""

    pass
