"""Public API for the benchmark platform."""

from __future__ import annotations

import openhcs as _openhcs_dependency_bootstrap  # noqa: F401

__all__ = (
    "DatasetSpec",
    "AcquiredDataset",
    "MetricCollector",
    "BenchmarkResult",
    "ToolAdapter",
    "ToolAdapterError",
    "ToolExecutionError",
    "ToolNotInstalledError",
    "ToolVersionError",
    "DatasetAcquisitionError",
    "acquire_dataset",
    "BBBC021_SINGLE_PLATE",
    "DATASET_REGISTRY",
    "get_dataset_spec",
    "PipelineSpec",
    "NUCLEI_SEGMENTATION",
    "PIPELINE_REGISTRY",
    "get_pipeline_spec",
    "TimeMetric",
    "MemoryMetric",
    "OpenHCSAxisSelection",
    "BenchmarkCaseProgress",
    "BenchmarkProgressEvent",
    "BenchmarkProgressEventKind",
    "BenchmarkProgressSnapshot",
    "iter_progress_events",
    "summarize_progress",
    "CellProfilerAdapter",
    "OpenHCSAdapter",
    "CellProfilerCompatibilityResult",
    "run_benchmark",
    "run_cellprofiler_compatibility_benchmark",
)

_EXPORT_NAMES = frozenset(__all__)
_MISSING_EXPORT = object()


def _benchmark_export_modules():
    import benchmark.contracts.dataset as dataset_contracts

    yield dataset_contracts

    import benchmark.contracts.metric as metric_contracts

    yield metric_contracts

    import benchmark.contracts.pipeline as pipeline_contracts

    yield pipeline_contracts

    import benchmark.contracts.tool_adapter as tool_adapter_contracts

    yield tool_adapter_contracts

    import benchmark.datasets as dataset_exports

    yield dataset_exports

    import benchmark.pipelines as pipeline_exports

    yield pipeline_exports

    import benchmark.metrics as metric_exports

    yield metric_exports

    import benchmark.progress as progress_exports

    yield progress_exports

    import benchmark.adapters.openhcs as openhcs_adapter

    yield openhcs_adapter

    import benchmark.adapters.cellprofiler as cellprofiler_adapter

    yield cellprofiler_adapter

    import benchmark.runner as runner_exports

    yield runner_exports


def resolve_benchmark_export(name: str):
    """Resolve one public benchmark export from its owning module."""
    if name not in _EXPORT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    existing = globals().get(name, _MISSING_EXPORT)
    if existing is not _MISSING_EXPORT:
        return existing
    for module in _benchmark_export_modules():
        namespace = vars(module)
        if name in namespace:
            value = namespace[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str):
    """Resolve public benchmark re-exports from their owning modules on demand."""
    return resolve_benchmark_export(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORT_NAMES)
