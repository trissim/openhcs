"""Public API for the benchmark platform."""

from __future__ import annotations

import importlib
from typing import Any

import openhcs as _openhcs_dependency_bootstrap  # noqa: F401


_PUBLIC_EXPORTS: dict[str, tuple[str, str]] = {
    "DatasetSpec": ("benchmark.contracts.dataset", "DatasetSpec"),
    "AcquiredDataset": ("benchmark.contracts.dataset", "AcquiredDataset"),
    "MetricCollector": ("benchmark.contracts.metric", "MetricCollector"),
    "BenchmarkResult": ("benchmark.contracts.tool_adapter", "BenchmarkResult"),
    "ToolAdapter": ("benchmark.contracts.tool_adapter", "ToolAdapter"),
    "ToolAdapterError": ("benchmark.contracts.tool_adapter", "ToolAdapterError"),
    "ToolExecutionError": ("benchmark.contracts.tool_adapter", "ToolExecutionError"),
    "ToolNotInstalledError": ("benchmark.contracts.tool_adapter", "ToolNotInstalledError"),
    "ToolVersionError": ("benchmark.contracts.tool_adapter", "ToolVersionError"),
    "DatasetAcquisitionError": (
        "benchmark.datasets.acquire",
        "DatasetAcquisitionError",
    ),
    "acquire_dataset": ("benchmark.datasets.acquire", "acquire_dataset"),
    "BBBC021_SINGLE_PLATE": (
        "benchmark.datasets.registry",
        "BBBC021_SINGLE_PLATE",
    ),
    "DATASET_REGISTRY": ("benchmark.datasets.registry", "DATASET_REGISTRY"),
    "get_dataset_spec": ("benchmark.datasets.registry", "get_dataset_spec"),
    "PipelineSpec": ("benchmark.pipelines.registry", "PipelineSpec"),
    "NUCLEI_SEGMENTATION": (
        "benchmark.pipelines.registry",
        "NUCLEI_SEGMENTATION",
    ),
    "PIPELINE_REGISTRY": ("benchmark.pipelines.registry", "PIPELINE_REGISTRY"),
    "get_pipeline_spec": ("benchmark.pipelines.registry", "get_pipeline_spec"),
    "TimeMetric": ("benchmark.metrics.time", "TimeMetric"),
    "MemoryMetric": ("benchmark.metrics.memory", "MemoryMetric"),
    "CellProfilerAdapter": (
        "benchmark.adapters.cellprofiler",
        "CellProfilerAdapter",
    ),
    "OpenHCSAdapter": ("benchmark.adapters.openhcs", "OpenHCSAdapter"),
    "CellProfilerCompatibilityResult": (
        "benchmark.runner",
        "CellProfilerCompatibilityResult",
    ),
    "run_benchmark": ("benchmark.runner", "run_benchmark"),
    "run_cellprofiler_compatibility_benchmark": (
        "benchmark.runner",
        "run_cellprofiler_compatibility_benchmark",
    ),
}


def __getattr__(name: str) -> Any:
    """Load public benchmark symbols on demand."""
    if name not in _PUBLIC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _PUBLIC_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = tuple(_PUBLIC_EXPORTS)
