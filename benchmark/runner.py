"""Benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from benchmark.adapters.cellprofiler import CellProfilerAdapter
from benchmark.adapters.openhcs import OpenHCSAdapter
from benchmark.contracts.dataset import DatasetSpec
from benchmark.contracts.tool_adapter import BenchmarkResult, ToolAdapter
from benchmark.datasets.acquire import acquire_dataset
from benchmark.pipelines.registry import get_pipeline_spec


@dataclass(frozen=True, slots=True)
class CellProfilerCompatibilityResult:
    """Native CellProfiler reference plus equivalent OpenHCS candidate result."""

    native_cellprofiler: BenchmarkResult
    openhcs_converted: BenchmarkResult

    @property
    def is_equivalent(self) -> bool:
        """Return whether the OpenHCS run reported zero semantic differences."""
        provenance = self.openhcs_converted.provenance or {}
        return (
            self.native_cellprofiler.success
            and self.openhcs_converted.success
            and provenance.get("equivalence_difference_count") == 0
        )


def run_benchmark(
    dataset_spec: DatasetSpec,
    tool_adapters: list[ToolAdapter],
    pipeline_name: str,
    metrics: Iterable,
) -> list[BenchmarkResult]:
    """
    Run benchmark across tools.

    1. Validate all tools
    2. Acquire dataset
    3. For each tool: run with metrics
    4. Return results
    """
    # Validate tools are installed
    for adapter in tool_adapters:
        adapter.validate_installation()

    acquired = acquire_dataset(dataset_spec)
    pipeline_spec = get_pipeline_spec(pipeline_name)

    # Merge pipeline parameters with dataset-specific context
    pipeline_params = {
        **pipeline_spec.parameters,
        "dataset_id": dataset_spec.id,
        "microscope_type": acquired.microscope_type,
    }

    results: list[BenchmarkResult] = []
    output_root = Path.cwd() / "benchmark_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    for adapter in tool_adapters:
        tool_output_dir = output_root / f"{adapter.name}_{dataset_spec.id}"
        tool_result = adapter.run(
            dataset_path=acquired.path,
            pipeline_name=pipeline_spec.name,
            pipeline_params=pipeline_params,
            metrics=list(metrics),
            output_dir=tool_output_dir,
        )
        results.append(tool_result)

    return results


def run_cellprofiler_compatibility_benchmark(
    dataset_spec: DatasetSpec,
    pipeline_name: str,
    metrics: Iterable,
    *,
    cellprofiler_adapter: ToolAdapter | None = None,
    openhcs_adapter: ToolAdapter | None = None,
) -> CellProfilerCompatibilityResult:
    """Run native CellProfiler, then require OpenHCS converted output parity."""
    native_adapter = cellprofiler_adapter or CellProfilerAdapter()
    converted_adapter = openhcs_adapter or OpenHCSAdapter()
    native_adapter.validate_installation()
    converted_adapter.validate_installation()

    acquired = acquire_dataset(dataset_spec)
    pipeline_spec = get_pipeline_spec(pipeline_name)
    pipeline_params = {
        **pipeline_spec.parameters,
        "dataset_id": dataset_spec.id,
        "microscope_type": acquired.microscope_type,
    }
    output_root = Path.cwd() / "benchmark_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    metric_collectors = list(metrics)
    native_result = native_adapter.run(
        dataset_path=acquired.path,
        pipeline_name=pipeline_spec.name,
        pipeline_params=pipeline_params,
        metrics=metric_collectors,
        output_dir=output_root / f"{native_adapter.name}_{dataset_spec.id}",
    )
    converted_params = {
        **pipeline_params,
        "equivalence_reference_output_dir": str(native_result.output_path),
    }
    converted_result = converted_adapter.run(
        dataset_path=acquired.path,
        pipeline_name=pipeline_spec.name,
        pipeline_params=converted_params,
        metrics=metric_collectors,
        output_dir=output_root / f"{converted_adapter.name}_{dataset_spec.id}",
    )
    return CellProfilerCompatibilityResult(
        native_cellprofiler=native_result,
        openhcs_converted=converted_result,
    )
