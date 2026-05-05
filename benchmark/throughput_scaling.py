"""OpenHCS cppipe throughput scaling benchmark."""

from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmark.adapters.cellprofiler import native_cellprofiler_reference_is_complete
from benchmark.adapters.openhcs import OpenHCSAdapter
from benchmark.cellprofiler_comparison import (
    CellProfilerComparisonCase,
    load_comparison_cases,
)
from benchmark.datasets.visible_source import resolve_visible_source_path
from benchmark.metrics.memory import MemoryMetric
from benchmark.metrics.time import TimeMetric


JOB_ROWS_CSV = "throughput_jobs.csv"
BATCH_ROWS_CSV = "throughput_batches.csv"
SUMMARY_ROWS_CSV = "throughput_summary.csv"


@dataclass(frozen=True, slots=True)
class ThroughputScalingCase:
    """One cppipe case used for OpenHCS throughput scaling."""

    name: str
    dataset_path: Path
    cppipe_path: Path
    dataset_id: str
    microscope_type: str | None = None
    value_only: bool = True
    equivalence_reference_output_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class ThroughputJobResult:
    """One independent OpenHCS job result."""

    case_name: str
    worker_count: int
    replica: int
    success: bool
    equivalent: bool | None
    execution_seconds: float | None
    total_phase_seconds: float | None
    peak_memory_mb: float | None
    difference_count: int | None
    axis_count: int | None
    output_path: str
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ThroughputBatchResult:
    """One batch of independent jobs at one concurrency level."""

    case_name: str
    worker_count: int
    replicas: int
    successful_jobs: int
    equivalent_jobs: int | None
    wall_seconds: float
    throughput_jobs_per_second: float
    speedup_vs_sequential: float | None
    parallel_efficiency: float | None
    peak_memory_mb: float
    mean_job_execution_seconds: float | None
    max_job_execution_seconds: float | None
    mean_job_peak_memory_mb: float | None
    min_axis_count: int | None


@dataclass(frozen=True, slots=True)
class ThroughputJobSpec:
    """Pickle-safe execution request for one independent OpenHCS job."""

    case_name: str
    dataset_path: Path
    cppipe_path: Path
    dataset_id: str
    microscope_type: str | None
    worker_count: int
    replica: int
    output_root: Path
    openhcs_timeout_seconds: float | None
    equivalence_reference_output_dir: Path | None
    compare_image_outputs: bool
    discard_outputs: bool


def load_scaling_cases(
    manifest_path: Path,
    *,
    case_names: Sequence[str] = (),
    native_reference_root: Path | None = None,
) -> tuple[ThroughputScalingCase, ...]:
    """Load throughput cases from the existing CP-vs-OpenHCS manifest."""
    selected = set(case_names)
    cases: list[ThroughputScalingCase] = []
    for comparison_case in load_comparison_cases(manifest_path):
        if selected and comparison_case.name not in selected:
            continue
        cases.append(
            ThroughputScalingCase(
                name=comparison_case.name,
                dataset_path=comparison_case.dataset_path,
                cppipe_path=comparison_case.cppipe_path,
                dataset_id=comparison_case.resolved_dataset_id,
                microscope_type=comparison_case.microscope_type,
                value_only=comparison_case.value_only,
                equivalence_reference_output_dir=_reference_output_dir(
                    comparison_case,
                    native_reference_root=native_reference_root,
                ),
            )
        )
    if selected:
        found = {case.name for case in cases}
        missing = selected - found
        if missing:
            raise ValueError(f"Manifest missing selected cases: {sorted(missing)!r}")
    return tuple(cases)


def run_throughput_scaling_suite(
    cases: Sequence[ThroughputScalingCase],
    *,
    output_root: Path,
    worker_counts: Sequence[int],
    replicas: int | Sequence[int],
    openhcs_timeout_seconds: float | None = None,
    verify_equivalence: bool = False,
    compare_image_outputs: bool = False,
    discard_outputs: bool = True,
) -> tuple[ThroughputBatchResult, ...]:
    """Run a throughput scaling matrix and write batch/job CSV outputs."""
    replica_counts = _normalized_replica_counts(replicas)
    normalized_workers = tuple(sorted(set(int(value) for value in worker_counts)))
    if not normalized_workers or any(value < 1 for value in normalized_workers):
        raise ValueError("worker_counts must contain positive integers.")
    output_root.mkdir(parents=True, exist_ok=True)

    job_results: list[ThroughputJobResult] = []
    batch_results: list[ThroughputBatchResult] = []
    for case in cases:
        for replica_count in replica_counts:
            sequential_wall_seconds: float | None = None
            for worker_count in normalized_workers:
                batch_output_root = (
                    output_root
                    / case.name
                    / f"samples_{replica_count}"
                    / f"workers_{worker_count}"
                )
                batch_jobs, batch = _run_case_worker_batch(
                    case,
                    worker_count=worker_count,
                    replicas=replica_count,
                    output_root=batch_output_root,
                    openhcs_timeout_seconds=openhcs_timeout_seconds,
                    verify_equivalence=verify_equivalence,
                    compare_image_outputs=compare_image_outputs,
                    discard_outputs=discard_outputs,
                    sequential_wall_seconds=sequential_wall_seconds,
                )
                if worker_count == 1:
                    sequential_wall_seconds = batch.wall_seconds
                    batch = _batch_with_speedup(
                        batch,
                        sequential_wall_seconds=sequential_wall_seconds,
                    )
                job_results.extend(batch_jobs)
                batch_results.append(batch)
                write_throughput_job_csv(output_root / JOB_ROWS_CSV, job_results)
                write_throughput_batch_csv(output_root / BATCH_ROWS_CSV, batch_results)
                write_throughput_summary_csv(
                    output_root / SUMMARY_ROWS_CSV,
                    batch_results,
                )
    _write_scaling_metadata(
        output_root / "throughput_scaling_metadata.json",
        cases=cases,
        worker_counts=normalized_workers,
        replica_counts=replica_counts,
        verify_equivalence=verify_equivalence,
        compare_image_outputs=compare_image_outputs,
        discard_outputs=discard_outputs,
    )
    return tuple(batch_results)


def write_throughput_job_csv(
    path: Path,
    rows: Sequence[ThroughputJobResult],
) -> None:
    """Write one row per OpenHCS throughput job."""
    _write_dataclass_csv(path, rows)


def write_throughput_batch_csv(
    path: Path,
    rows: Sequence[ThroughputBatchResult],
) -> None:
    """Write one row per case/concurrency batch."""
    _write_dataclass_csv(path, rows)


def write_throughput_summary_csv(
    path: Path,
    rows: Sequence[ThroughputBatchResult],
) -> None:
    """Write one row per sample-count/worker-count aggregated across cases."""
    grouped: dict[tuple[int, int], list[ThroughputBatchResult]] = {}
    for row in rows:
        grouped.setdefault((row.replicas, row.worker_count), []).append(row)
    summary_rows = [
        {
            "replicas": replicas,
            "worker_count": worker_count,
            "case_count": len(worker_rows),
            "mean_speedup_vs_sequential": _mean_present(
                row.speedup_vs_sequential for row in worker_rows
            ),
            "median_speedup_vs_sequential": _median_present(
                row.speedup_vs_sequential for row in worker_rows
            ),
            "mean_parallel_efficiency": _mean_present(
                row.parallel_efficiency for row in worker_rows
            ),
            "mean_throughput_jobs_per_second": _mean_present(
                row.throughput_jobs_per_second for row in worker_rows
            ),
            "mean_wall_seconds": _mean_present(row.wall_seconds for row in worker_rows),
            "mean_peak_memory_mb": _mean_present(
                row.peak_memory_mb for row in worker_rows
            ),
            "all_successful": all(
                row.successful_jobs == row.replicas for row in worker_rows
            ),
        }
        for (replicas, worker_count), worker_rows in sorted(grouped.items())
    ]
    _write_mapping_csv(path, summary_rows)


def _run_case_worker_batch(
    case: ThroughputScalingCase,
    *,
    worker_count: int,
    replicas: int,
    output_root: Path,
    openhcs_timeout_seconds: float | None,
    verify_equivalence: bool,
    compare_image_outputs: bool,
    discard_outputs: bool,
    sequential_wall_seconds: float | None,
) -> tuple[tuple[ThroughputJobResult, ...], ThroughputBatchResult]:
    output_root.mkdir(parents=True, exist_ok=True)
    job_specs = tuple(
        _job_spec(
            case,
            worker_count=worker_count,
            replica=replica,
            output_root=output_root / f"replica_{replica:03d}",
            openhcs_timeout_seconds=openhcs_timeout_seconds,
            verify_equivalence=verify_equivalence,
            compare_image_outputs=compare_image_outputs,
            discard_outputs=discard_outputs,
        )
        for replica in range(1, replicas + 1)
    )

    with MemoryMetric(interval_seconds=0.05, include_children=True) as memory_metric:
        started_at = time.perf_counter()
        if worker_count == 1:
            job_results = tuple(_run_openhcs_throughput_job(spec) for spec in job_specs)
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(_run_openhcs_throughput_job, spec)
                    for spec in job_specs
                ]
                job_results = tuple(future.result() for future in as_completed(futures))
        wall_seconds = time.perf_counter() - started_at
        peak_memory_mb = memory_metric.get_result()

    sorted_jobs = tuple(sorted(job_results, key=lambda result: result.replica))
    batch = _batch_result(
        case.name,
        worker_count=worker_count,
        replicas=replicas,
        job_results=sorted_jobs,
        wall_seconds=wall_seconds,
        peak_memory_mb=peak_memory_mb,
        sequential_wall_seconds=sequential_wall_seconds,
    )
    return sorted_jobs, batch


def _job_spec(
    case: ThroughputScalingCase,
    *,
    worker_count: int,
    replica: int,
    output_root: Path,
    openhcs_timeout_seconds: float | None,
    verify_equivalence: bool,
    compare_image_outputs: bool,
    discard_outputs: bool,
) -> ThroughputJobSpec:
    reference_output_dir = (
        case.equivalence_reference_output_dir
        if verify_equivalence and case.equivalence_reference_output_dir is not None
        else None
    )
    if verify_equivalence and reference_output_dir is None:
        raise ValueError(
            f"Case {case.name} requires an equivalence reference for verification."
        )
    return ThroughputJobSpec(
        case_name=case.name,
        dataset_path=case.dataset_path,
        cppipe_path=case.cppipe_path,
        dataset_id=case.dataset_id,
        microscope_type=case.microscope_type,
        worker_count=worker_count,
        replica=replica,
        output_root=output_root,
        openhcs_timeout_seconds=openhcs_timeout_seconds,
        equivalence_reference_output_dir=reference_output_dir,
        compare_image_outputs=compare_image_outputs,
        discard_outputs=discard_outputs,
    )


def _run_openhcs_throughput_job(spec: ThroughputJobSpec) -> ThroughputJobResult:
    _configure_worker_environment()
    output_root = spec.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    params: dict[str, Any] = {
        "dataset_id": spec.dataset_id,
        "cppipe_path": str(spec.cppipe_path),
        "compare_image_outputs": spec.compare_image_outputs,
        "raise_on_equivalence_failure": False,
        "cache_candidate_measurement_snapshot": False,
        "reuse_runtime_execution_cache": False,
    }
    if spec.microscope_type is not None:
        params["microscope_type"] = spec.microscope_type
    if spec.openhcs_timeout_seconds is not None:
        params["openhcs_timeout_seconds"] = spec.openhcs_timeout_seconds
    if spec.equivalence_reference_output_dir is not None:
        params["equivalence_reference_output_dir"] = str(
            spec.equivalence_reference_output_dir
        )

    try:
        result = OpenHCSAdapter().run(
            dataset_path=spec.dataset_path,
            pipeline_name=spec.case_name,
            pipeline_params=params,
            metrics=[TimeMetric(), MemoryMetric(interval_seconds=0.05)],
            output_dir=output_root / "openhcs",
        )
        provenance = result.provenance or {}
        phase_seconds = _phase_seconds(provenance)
        difference_count = provenance.get("equivalence_difference_count")
        equivalent = (
            int(difference_count) == 0
            if difference_count is not None
            else None
        )
        job_result = ThroughputJobResult(
            case_name=spec.case_name,
            worker_count=spec.worker_count,
            replica=spec.replica,
            success=result.success,
            equivalent=equivalent,
            execution_seconds=phase_seconds.get("EXECUTE_OPENHCS"),
            total_phase_seconds=sum(phase_seconds.values()) if phase_seconds else None,
            peak_memory_mb=_optional_float(result.metrics.get("peak_memory_mb")),
            difference_count=(
                int(difference_count) if difference_count is not None else None
            ),
            axis_count=(
                int(provenance["axis_count"])
                if provenance.get("axis_count") is not None
                else None
            ),
            output_path=str(result.output_path),
            error_message=result.error_message,
        )
    except Exception as exc:
        job_result = ThroughputJobResult(
            case_name=spec.case_name,
            worker_count=spec.worker_count,
            replica=spec.replica,
            success=False,
            equivalent=None,
            execution_seconds=None,
            total_phase_seconds=None,
            peak_memory_mb=None,
            difference_count=None,
            axis_count=None,
            output_path=str(output_root),
            error_message=f"{type(exc).__name__}: {exc}",
        )
    if spec.discard_outputs:
        _discard_job_outputs(output_root)
    return job_result


def _batch_result(
    case_name: str,
    *,
    worker_count: int,
    replicas: int,
    job_results: Sequence[ThroughputJobResult],
    wall_seconds: float,
    peak_memory_mb: float,
    sequential_wall_seconds: float | None,
) -> ThroughputBatchResult:
    successful_jobs = sum(1 for result in job_results if result.success)
    equivalent_values = [
        result.equivalent for result in job_results if result.equivalent is not None
    ]
    speedup = (
        sequential_wall_seconds / wall_seconds
        if sequential_wall_seconds is not None and wall_seconds > 0.0
        else None
    )
    return ThroughputBatchResult(
        case_name=case_name,
        worker_count=worker_count,
        replicas=replicas,
        successful_jobs=successful_jobs,
        equivalent_jobs=(
            sum(1 for value in equivalent_values if value) if equivalent_values else None
        ),
        wall_seconds=wall_seconds,
        throughput_jobs_per_second=replicas / wall_seconds if wall_seconds > 0.0 else 0.0,
        speedup_vs_sequential=speedup,
        parallel_efficiency=speedup / worker_count if speedup is not None else None,
        peak_memory_mb=peak_memory_mb,
        mean_job_execution_seconds=_mean_present(
            result.execution_seconds for result in job_results
        ),
        max_job_execution_seconds=_max_present(
            result.execution_seconds for result in job_results
        ),
        mean_job_peak_memory_mb=_mean_present(
            result.peak_memory_mb for result in job_results
        ),
        min_axis_count=_min_present(result.axis_count for result in job_results),
    )


def _batch_with_speedup(
    batch: ThroughputBatchResult,
    *,
    sequential_wall_seconds: float,
) -> ThroughputBatchResult:
    return ThroughputBatchResult(
        **{
            **asdict(batch),
            "speedup_vs_sequential": 1.0,
            "parallel_efficiency": 1.0,
            "wall_seconds": sequential_wall_seconds,
        }
    )


def _reference_output_dir(
    case: CellProfilerComparisonCase,
    *,
    native_reference_root: Path | None,
) -> Path | None:
    if case.equivalence_reference_output_dir is not None:
        return case.equivalence_reference_output_dir
    if native_reference_root is None:
        return None
    run_slug = _path_slug(f"{case.resolved_dataset_id}_{case.name}")
    resolved_dataset_path = resolve_visible_source_path(case.dataset_path)
    expected = (
        Path(native_reference_root)
        / run_slug
        / f"{resolved_dataset_path.name}_{case.name}_native_cellprofiler"
    )
    return expected if native_cellprofiler_reference_is_complete(expected) else None


def _configure_worker_environment() -> None:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("XDG_DATA_HOME", "/tmp/openhcs-benchmark-xdg-data")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/openhcs-benchmark-xdg-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/openhcs-benchmark-mpl")
    os.environ.setdefault("OPENHCS_CPU_ONLY", "true")
    os.environ.setdefault("OPENHCS_SUBPROCESS_NO_GPU", "1")
    os.environ.setdefault("POLYSTORE_SUBPROCESS_NO_GPU", "1")


def _phase_seconds(provenance: Mapping[str, Any]) -> dict[str, float]:
    phase_totals: dict[str, float] = {}
    for raw_record in provenance.get("phase_timing_records", ()):
        if not isinstance(raw_record, Mapping):
            continue
        phase = raw_record.get("phase")
        seconds = raw_record.get("seconds")
        if phase is None or seconds is None:
            continue
        phase_totals[str(phase)] = phase_totals.get(str(phase), 0.0) + float(seconds)
    return phase_totals


def _discard_job_outputs(output_root: Path) -> None:
    resolved = output_root.resolve()
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise NotADirectoryError(f"Throughput output is not a directory: {resolved}")
    if resolved == resolved.parent:
        raise ValueError(f"Refusing unsafe throughput output discard target: {resolved}")
    shutil.rmtree(resolved)


def _write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    _write_mapping_csv(path, (asdict(row) for row in rows))


def _write_mapping_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    row_tuple = tuple(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not row_tuple:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = tuple(row_tuple[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_tuple)


def _write_scaling_metadata(
    path: Path,
    *,
    cases: Sequence[ThroughputScalingCase],
    worker_counts: Sequence[int],
    replica_counts: Sequence[int],
    verify_equivalence: bool,
    compare_image_outputs: bool,
    discard_outputs: bool,
) -> None:
    payload = {
        "created_at_epoch_seconds": time.time(),
        "cases": [case.name for case in cases],
        "worker_counts": list(worker_counts),
        "replica_counts": list(replica_counts),
        "verify_equivalence": verify_equivalence,
        "compare_image_outputs": compare_image_outputs,
        "discard_outputs": discard_outputs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _mean_present(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _median_present(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def _max_present(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _min_present(values: Iterable[float | int | None]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return min(present) if present else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalized_replica_counts(replicas: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(replicas, int):
        values = (replicas,)
    else:
        values = tuple(int(value) for value in replicas)
    normalized = tuple(sorted(set(values)))
    if not normalized or any(value < 1 for value in normalized):
        raise ValueError("replicas must contain positive integers.")
    return normalized


def _path_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
