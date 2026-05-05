"""Figures for converted cppipe throughput-scaling benchmarks."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from enum import Enum, auto
from pathlib import Path

from benchmark.reports.cppipe_figures import DEFAULT_FORMATS
from benchmark.reports.cppipe_figures import DEFAULT_WRAP_AFTER
from benchmark.reports.cppipe_figures import BenchmarkMetricRow
from benchmark.reports.cppipe_figures import FigureMetricSpec
from benchmark.reports.cppipe_figures import generate_grouped_benchmark_metric_figures


CASE_NAME_FIELD = "case_name"
WORKER_COUNT_FIELD = "worker_count"
REPLICAS_FIELD = "replicas"
WALL_SECONDS_FIELD = "wall_seconds"
JOBS_PER_SECOND_FIELD = "throughput_jobs_per_second"
PEAK_MEMORY_MB_FIELD = "peak_memory_mb"
SPEEDUP_FIELD = "speedup_vs_sequential"
EFFICIENCY_FIELD = "parallel_efficiency"
NATIVE_SECONDS_FIELD = "median_native_execution_seconds"

THROUGHPUT_BASE_METRICS = (
    FigureMetricSpec(
        "raw_seconds",
        "cppipe_throughput_wall_seconds",
        "Batch wall time for independent samples",
        "Batch wall seconds",
        minimum_ylim=0.0,
        log_variant=True,
    ),
    FigureMetricSpec(
        "accuracy_fraction",
        "cppipe_throughput_efficiency",
        "Parallel efficiency versus sequential execution",
        "Efficiency (%)",
        percentage=True,
        baseline_line=100.0,
        minimum_ylim=0.0,
    ),
    FigureMetricSpec(
        "peak_memory_mb",
        "cppipe_throughput_peak_memory",
        "Batch peak process-tree memory",
        "Peak RSS (MB)",
        minimum_ylim=0.0,
        log_variant=True,
    ),
)
THROUGHPUT_CP_SPEEDUP_METRIC = FigureMetricSpec(
    "speedup",
    "cppipe_throughput_speedup_vs_native_cp",
    "Throughput speedup versus native CellProfiler",
    "Speedup (x)",
    baseline_line=1.0,
    minimum_ylim=0.0,
    log_variant=True,
)
THROUGHPUT_SELF_SPEEDUP_METRIC = FigureMetricSpec(
    "speedup",
    "cppipe_throughput_speedup_vs_openhcs_1job",
    "Throughput speedup versus OpenHCS 1-job execution",
    "Speedup (x)",
    baseline_line=1.0,
    minimum_ylim=0.0,
    log_variant=True,
)


def generate_cppipe_scaling_figures(
    summary_csv: Path,
    *,
    output_dir: Path,
    native_summary_csv: Path | None = None,
    output_formats: Sequence[str] = DEFAULT_FORMATS,
    include_average: bool = True,
    wrap_after: int = DEFAULT_WRAP_AFTER,
    replica_counts: Sequence[int] = (),
    worker_counts: Sequence[int] = (),
) -> tuple[Path, ...]:
    """Generate v7-style grouped-bar charts from ``throughput_batches.csv``."""
    table_rows = _filter_summary_rows(
        _load_summary_rows(summary_csv),
        replica_counts=replica_counts,
        worker_counts=worker_counts,
    )
    native_seconds_by_case = _load_native_seconds(native_summary_csv)
    base_rows = tuple(
        _throughput_metric_rows(
            table_rows,
            native_seconds_by_case=native_seconds_by_case,
            speedup_denominator=SpeedupDenominator.NATIVE_CP,
            include_average=include_average,
        )
    )
    native_speedup_rows = base_rows
    self_speedup_rows = tuple(
        _throughput_metric_rows(
            table_rows,
            native_seconds_by_case=native_seconds_by_case,
            speedup_denominator=SpeedupDenominator.OPENHCS_1JOB,
            include_average=include_average,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = _worker_methods(table_rows)
    pipeline_names = tuple(dict.fromkeys(row.pipeline_name for row in base_rows))
    long_csv_path = output_dir / "cppipe_throughput_metrics_long.csv"
    _write_metric_rows(
        long_csv_path,
        native_speedup_rows=native_speedup_rows,
        self_speedup_rows=self_speedup_rows,
    )
    return (
        long_csv_path,
        *generate_grouped_benchmark_metric_figures(
            base_rows,
            metrics=THROUGHPUT_BASE_METRICS,
            methods=methods,
            pipeline_names=pipeline_names,
            output_dir=output_dir,
            output_formats=output_formats,
            wrap_after=wrap_after,
        ),
        *generate_grouped_benchmark_metric_figures(
            native_speedup_rows,
            metrics=(THROUGHPUT_CP_SPEEDUP_METRIC,),
            methods=methods,
            pipeline_names=pipeline_names,
            output_dir=output_dir,
            output_formats=output_formats,
            wrap_after=wrap_after,
        ),
        *generate_grouped_benchmark_metric_figures(
            self_speedup_rows,
            metrics=(THROUGHPUT_SELF_SPEEDUP_METRIC,),
            methods=methods,
            pipeline_names=pipeline_names,
            output_dir=output_dir,
            output_formats=output_formats,
            wrap_after=wrap_after,
        ),
    )


def _load_summary_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Throughput summary CSV is empty: {path}")
    required = {
        CASE_NAME_FIELD,
        WORKER_COUNT_FIELD,
        REPLICAS_FIELD,
        WALL_SECONDS_FIELD,
        JOBS_PER_SECOND_FIELD,
        PEAK_MEMORY_MB_FIELD,
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Throughput summary CSV {path} missing columns: {sorted(missing)!r}"
        )
    return rows


def _filter_summary_rows(
    rows: Sequence[dict[str, str]],
    *,
    replica_counts: Sequence[int],
    worker_counts: Sequence[int],
) -> tuple[dict[str, str], ...]:
    selected_replicas = {int(value) for value in replica_counts}
    selected_workers = {int(value) for value in worker_counts}
    filtered = tuple(
        row
        for row in rows
        if (not selected_replicas or int(row[REPLICAS_FIELD]) in selected_replicas)
        and (not selected_workers or int(row[WORKER_COUNT_FIELD]) in selected_workers)
    )
    if not filtered:
        raise ValueError(
            "Throughput filters removed all rows: "
            f"replica_counts={sorted(selected_replicas)!r}, "
            f"worker_counts={sorted(selected_workers)!r}"
        )
    return filtered


def _throughput_metric_rows(
    rows: Sequence[dict[str, str]],
    *,
    native_seconds_by_case: dict[str, float],
    speedup_denominator: "SpeedupDenominator",
    include_average: bool,
) -> Iterable[BenchmarkMetricRow]:
    for row in rows:
        yield BenchmarkMetricRow(
            pipeline_name=row[CASE_NAME_FIELD],
            method=_worker_method(row),
            accuracy_fraction=_optional_float(row, EFFICIENCY_FIELD),
            raw_seconds=_optional_float(row, WALL_SECONDS_FIELD),
            speedup=_speedup(row, native_seconds_by_case, speedup_denominator),
            peak_memory_mb=_optional_float(row, PEAK_MEMORY_MB_FIELD),
        )
    if include_average:
        yield from _average_rows(
            _throughput_metric_rows(
                rows,
                native_seconds_by_case=native_seconds_by_case,
                speedup_denominator=speedup_denominator,
                include_average=False,
            )
        )


def _average_rows(rows: Iterable[BenchmarkMetricRow]) -> Iterable[BenchmarkMetricRow]:
    by_method: dict[str, list[BenchmarkMetricRow]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)
    for method, method_rows in by_method.items():
        yield BenchmarkMetricRow(
            pipeline_name="Average",
            method=method,
            accuracy_fraction=_mean_present(
                row.accuracy_fraction for row in method_rows
            ),
            raw_seconds=_mean_present(row.raw_seconds for row in method_rows),
            speedup=_mean_present(row.speedup for row in method_rows),
            peak_memory_mb=_mean_present(row.peak_memory_mb for row in method_rows),
        )


def _write_metric_rows(
    path: Path,
    *,
    native_speedup_rows: Sequence[BenchmarkMetricRow],
    self_speedup_rows: Sequence[BenchmarkMetricRow],
) -> None:
    fieldnames = (
        "pipeline_name",
        "method",
        "efficiency_fraction",
        "wall_seconds",
        "speedup_vs_native_cp",
        "speedup_vs_openhcs_1job",
        "peak_memory_mb",
    )
    self_speedup_index = {
        (row.pipeline_name, row.method): row.speedup for row in self_speedup_rows
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in native_speedup_rows:
            writer.writerow(
                {
                    "pipeline_name": row.pipeline_name,
                    "method": row.method,
                    "efficiency_fraction": row.accuracy_fraction,
                    "wall_seconds": row.raw_seconds,
                    "speedup_vs_native_cp": row.speedup,
                    "speedup_vs_openhcs_1job": self_speedup_index.get(
                        (row.pipeline_name, row.method)
                    ),
                    "peak_memory_mb": row.peak_memory_mb,
                }
            )


def _load_native_seconds(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Native summary CSV is empty: {path}")
    required = {CASE_NAME_FIELD, NATIVE_SECONDS_FIELD}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Native summary CSV {path} missing columns: {sorted(missing)!r}"
        )
    return {
        row[CASE_NAME_FIELD]: float(row[NATIVE_SECONDS_FIELD])
        for row in rows
        if row.get(NATIVE_SECONDS_FIELD)
    }


def _speedup(
    row: dict[str, str],
    native_seconds_by_case: dict[str, float],
    speedup_denominator: "SpeedupDenominator",
) -> float | None:
    if speedup_denominator is SpeedupDenominator.OPENHCS_1JOB:
        return _optional_float(row, SPEEDUP_FIELD)
    native_seconds = native_seconds_by_case.get(row[CASE_NAME_FIELD])
    wall_seconds = _optional_float(row, WALL_SECONDS_FIELD)
    replicas = int(row[REPLICAS_FIELD])
    if native_seconds is not None and wall_seconds is not None and wall_seconds > 0.0:
        return native_seconds * replicas / wall_seconds
    return None


def _worker_methods(rows: Sequence[dict[str, str]]) -> tuple[str, ...]:
    return tuple(
        _method_label(replicas, worker_count)
        for replicas, worker_count in sorted(
            {
                (int(row[REPLICAS_FIELD]), int(row[WORKER_COUNT_FIELD]))
                for row in rows
            }
        )
    )


def _worker_method(row: dict[str, str]) -> str:
    replicas = int(row[REPLICAS_FIELD])
    worker_count = int(row[WORKER_COUNT_FIELD])
    return _method_label(replicas, worker_count)


def _method_label(replicas: int, worker_count: int) -> str:
    sample_label = f"{replicas} sample{'s' if replicas != 1 else ''}"
    worker_label = f"{worker_count} job{'s' if worker_count != 1 else ''}"
    return f"{sample_label} / {worker_label}"


def _optional_float(row: dict[str, str], field_name: str) -> float | None:
    value = row.get(field_name)
    if value is None or value == "":
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _mean_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


class SpeedupDenominator(Enum):
    """Throughput speedup denominator."""

    NATIVE_CP = auto()
    OPENHCS_1JOB = auto()
