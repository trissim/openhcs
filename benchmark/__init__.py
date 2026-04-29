"""Public API for the benchmark platform."""

import openhcs as _openhcs_dependency_bootstrap

from benchmark.contracts.dataset import DatasetSpec, AcquiredDataset
from benchmark.contracts.metric import MetricCollector
from benchmark.contracts.tool_adapter import (
    BenchmarkResult,
    ToolAdapter,
    ToolAdapterError,
    ToolExecutionError,
    ToolNotInstalledError,
    ToolVersionError,
)
from benchmark.datasets.registry import BBBC021_SINGLE_PLATE, get_dataset_spec, DATASET_REGISTRY
from benchmark.datasets.acquire import acquire_dataset, DatasetAcquisitionError
from benchmark.metrics.time import TimeMetric
from benchmark.metrics.memory import MemoryMetric
from benchmark.pipelines.registry import (
    PipelineSpec,
    NUCLEI_SEGMENTATION,
    get_pipeline_spec,
    PIPELINE_REGISTRY,
)
from benchmark.adapters.openhcs import OpenHCSAdapter
from benchmark.adapters.cellprofiler import CellProfilerAdapter
from benchmark.runner import run_benchmark

__all__ = [
    # Contracts
    "DatasetSpec",
    "AcquiredDataset",
    "MetricCollector",
    "BenchmarkResult",
    "ToolAdapter",
    "ToolAdapterError",
    "ToolExecutionError",
    "ToolNotInstalledError",
    "ToolVersionError",
    # Datasets
    "DatasetAcquisitionError",
    "acquire_dataset",
    "BBBC021_SINGLE_PLATE",
    "DATASET_REGISTRY",
    "get_dataset_spec",
    # Pipelines
    "PipelineSpec",
    "NUCLEI_SEGMENTATION",
    "PIPELINE_REGISTRY",
    "get_pipeline_spec",
    # Metrics
    "TimeMetric",
    "MemoryMetric",
    # Adapters
    "CellProfilerAdapter",
    "OpenHCSAdapter",
    # Runner
    "run_benchmark",
]
