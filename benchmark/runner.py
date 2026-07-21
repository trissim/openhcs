"""Benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmark.adapters.cellprofiler import (
    CellProfilerAdapter,
    native_cellprofiler_reference_provenance,
)
from benchmark.adapters.openhcs import OpenHCSAdapter
from benchmark.contracts.dataset import DatasetSpec
from benchmark.contracts.tool_adapter import BenchmarkResult, ToolAdapter
from benchmark.datasets.acquire import acquire_dataset
from benchmark.datasets.visible_source import resolve_visible_source_path
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
    equivalence_reference_output_dir: Path | None = None,
    cellprofiler_adapter: ToolAdapter | None = None,
    openhcs_adapter: ToolAdapter | None = None,
) -> CellProfilerCompatibilityResult:
    """Run native CellProfiler, then require OpenHCS converted output parity."""
    native_adapter = cellprofiler_adapter or CellProfilerAdapter()
    converted_adapter = openhcs_adapter or OpenHCSAdapter()
    if equivalence_reference_output_dir is None:
        native_adapter.validate_installation()

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
    if equivalence_reference_output_dir is None:
        native_result = native_adapter.run(
            dataset_path=acquired.path,
            pipeline_name=pipeline_spec.name,
            pipeline_params=pipeline_params,
            metrics=metric_collectors,
            output_dir=output_root / f"{native_adapter.name}_{dataset_spec.id}",
        )
    else:
        native_result = _cached_cellprofiler_reference_result(
            Path(equivalence_reference_output_dir),
            dataset_id=dataset_spec.id,
            pipeline_name=pipeline_spec.name,
            tool_name=native_adapter.name,
        )
    converted_params = {
        **pipeline_params,
        "equivalence_reference_output_dir": str(native_result.output_path),
    }
    converted_adapter.validate_installation()
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


def run_cellprofiler_cppipe_parity(
    dataset_path: Path,
    cppipe_path: Path,
    metrics: Iterable,
    *,
    dataset_id: str | None = None,
    pipeline_name: str | None = None,
    microscope_type: str | None = None,
    pipeline_params: Mapping[str, Any] | None = None,
    output_root: Path | None = None,
    equivalence_reference_output_dir: Path | None = None,
    native_cellprofiler_output_dir: Path | None = None,
    cellprofiler_adapter: ToolAdapter | None = None,
    openhcs_adapter: ToolAdapter | None = None,
) -> CellProfilerCompatibilityResult:
    """Run native CellProfiler, then require OpenHCS parity for one local .cppipe."""
    native_adapter = cellprofiler_adapter or CellProfilerAdapter()
    converted_adapter = openhcs_adapter or OpenHCSAdapter()
    if equivalence_reference_output_dir is None:
        native_adapter.validate_installation()

    resolved_dataset_path = resolve_visible_source_path(Path(dataset_path))
    resolved_cppipe_path = Path(cppipe_path)
    resolved_dataset_id = dataset_id or resolved_dataset_path.name
    resolved_pipeline_name = pipeline_name or resolved_cppipe_path.stem
    resolved_output_root = output_root or Path.cwd() / "benchmark_outputs"
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    base_params: dict[str, Any] = {
        **dict(pipeline_params or {}),
        "dataset_id": resolved_dataset_id,
        "cppipe_path": str(resolved_cppipe_path),
    }
    if microscope_type is not None:
        base_params["microscope_type"] = microscope_type

    metric_collectors = list(metrics)
    run_slug = _benchmark_path_slug(f"{resolved_dataset_id}_{resolved_pipeline_name}")
    if equivalence_reference_output_dir is None:
        native_result = native_adapter.run(
            dataset_path=resolved_dataset_path,
            pipeline_name=resolved_pipeline_name,
            pipeline_params=base_params,
            metrics=metric_collectors,
            output_dir=(
                Path(native_cellprofiler_output_dir)
                if native_cellprofiler_output_dir is not None
                else resolved_output_root / f"{native_adapter.name}_{run_slug}"
            ),
        )
    else:
        native_result = _cached_cellprofiler_reference_result(
            Path(equivalence_reference_output_dir),
            dataset_id=resolved_dataset_id,
            pipeline_name=resolved_pipeline_name,
            tool_name=native_adapter.name,
        )
    converted_adapter.validate_installation()
    converted_result = converted_adapter.run(
        dataset_path=resolved_dataset_path,
        pipeline_name=resolved_pipeline_name,
        pipeline_params={
            **base_params,
            "equivalence_reference_output_dir": str(native_result.output_path),
        },
        metrics=metric_collectors,
        output_dir=resolved_output_root / f"{converted_adapter.name}_{run_slug}",
    )
    return CellProfilerCompatibilityResult(
        native_cellprofiler=native_result,
        openhcs_converted=converted_result,
    )


def _cached_cellprofiler_reference_result(
    reference_output_dir: Path,
    *,
    dataset_id: str,
    pipeline_name: str,
    tool_name: str,
) -> BenchmarkResult:
    """Represent an already-produced native CellProfiler output as a result."""
    resolved_reference = Path(reference_output_dir)
    if not resolved_reference.exists():
        raise FileNotFoundError(
            "Cached CellProfiler reference output directory does not exist: "
            f"{resolved_reference}"
        )
    if not resolved_reference.is_dir():
        raise NotADirectoryError(
            "Cached CellProfiler reference output path is not a directory: "
            f"{resolved_reference}"
        )
    return BenchmarkResult(
        tool_name=tool_name,
        dataset_id=dataset_id,
        pipeline_name=pipeline_name,
        metrics={},
        output_path=resolved_reference,
        success=True,
        provenance={
            "pipeline_source": "native_cppipe",
            "reused_reference_output": True,
            **native_cellprofiler_reference_provenance(resolved_reference),
        },
    )


def _benchmark_path_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
