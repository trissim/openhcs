#!/usr/bin/env python
"""Generate lab-meeting summary distribution figures from long benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


METHOD_COLORS = {
    "CP": "#262626",
    "OH1": "#0f8b8d",
    "1 core": "#0f8b8d",
    "2 cores": "#d95f02",
    "3 cores": "#1b9e77",
    "4 cores": "#7570b3",
}
SUMMARY_COLUMNS = ("worst", "mean", "median", "best")
CASE_NAME_FIELD = "case_name"
NATIVE_SECONDS_FIELD = "median_native_execution_seconds"
REPLICAS_FIELD = "replicas"
WORKER_COUNT_FIELD = "worker_count"
WALL_SECONDS_FIELD = "wall_seconds"
WELL_COUNT_FIELD = "well_count"
TOTAL_SECONDS_FIELD = "total_seconds"
ASSAY_PRESENTATION_GROUPS = {
    "Cell morphology": "Morphology",
    "Tissue/object morphology": "Morphology",
    "Tumor morphology": "Morphology",
    "Colocalization microscopy": "Intensity + colocalization",
    "DNA damage assay": "Intensity + classification",
    "Positive-cell classification": "Intensity + classification",
    "Spot detection": "Intensity + classification",
    "Imaging flow cytometry": "Object layout",
    "Spatial organization": "Object layout",
    "Time-lapse tracking": "Movement + tracking",
    "Migration/scratch assay": "Movement + tracking",
    "Yeast colony screening": "Colony screens",
    "Illumination correction": "Illumination correction",
}
MODULE_PRESENTATION_GROUPS = {
    "Image correction": "Image cleanup",
    "Segmentation + shape measurement": "Find + measure objects",
    "Segmentation + object measurement": "Find + measure objects",
    "Segmentation + intensity measurement": "Find + measure objects",
    "Segmentation + distance measurement": "Find + measure objects",
    "Colony segmentation + measurement": "Find + measure objects",
    "Colocalization + object measurement": "Intensity + texture",
    "Thresholding + classification": "Intensity + texture",
    "Small-object detection": "Intensity + texture",
    "Texture/patch measurement": "Intensity + texture",
    "Grid layout + object measurement": "Object layout",
    "Object relationship measurement": "Object layout",
    "Tracking + object measurement": "Movement + tracking",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        choices=(
            "accuracy_fraction",
            "raw_seconds",
            "speedup",
            "speedup_vs_native_cp",
            "peak_memory_mb",
        ),
    )
    parser.add_argument(
        "--core-regression",
        action="store_true",
        help="Generate speedup-vs-core regression from projected 8-sample rows.",
    )
    parser.add_argument(
        "--category-speedup",
        action="append",
        default=None,
        choices=("assay_category", "module_category"),
        help="Generate average category speedup bars with one dot per benchmark.",
    )
    parser.add_argument(
        "--category-speedup-log",
        action="store_true",
        help="Also generate log-scale category speedup figures.",
    )
    parser.add_argument("--prefix", default="cppipe_summary")
    parser.add_argument("--format", dest="formats", action="append", default=None)
    parser.add_argument(
        "--measured-throughput-speedup",
        action="store_true",
        help="Generate measured CP-vs-OpenHCS throughput speedup from throughput_batches.csv.",
    )
    parser.add_argument(
        "--throughput-replicas",
        type=int,
        default=4,
        help="Replica/sample count to use for measured throughput speedup.",
    )
    parser.add_argument(
        "--throughput-worker-count",
        type=int,
        action="append",
        default=None,
        help="Primary worker count(s) to include from --input-csv.",
    )
    parser.add_argument(
        "--native-summary-csv",
        type=Path,
        default=None,
        help="Native CP/OpenHCS summary CSV with median_native_execution_seconds.",
    )
    parser.add_argument(
        "--preliminary-well-throughput-csv",
        type=Path,
        default=None,
        help="Optional preliminary well_throughput.csv to add as a clearly marked 3-core bar.",
    )
    parser.add_argument(
        "--comparison-throughput-csv",
        type=Path,
        default=None,
        help="Optional second throughput_batches.csv to add as a separate comparison bar.",
    )
    parser.add_argument(
        "--comparison-throughput-replicas",
        type=int,
        default=16,
        help="Replica/sample count from --comparison-throughput-csv.",
    )
    parser.add_argument(
        "--comparison-throughput-label",
        default="2 core\n16 samples\npartial",
        help="Display label for the comparison throughput bar.",
    )
    parser.add_argument(
        "--well-throughput-summary",
        action="store_true",
        help="Generate a summary chart from a measured well_throughput.csv input.",
    )
    parser.add_argument(
        "--exclude-case",
        action="append",
        default=None,
        help="Case name to exclude from summary figures.",
    )
    args = parser.parse_args()

    rows = tuple(_load_rows(args.input_csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for metric in args.metric or ():
        outputs.extend(
            generate_summary_distribution_figure(
                rows,
                metric=metric,
                output_dir=args.output_dir,
                prefix=args.prefix,
                output_formats=tuple(args.formats or ("png", "svg")),
            )
        )
    if args.core_regression:
        outputs.extend(
            generate_core_regression_figure(
                rows,
                output_dir=args.output_dir,
                prefix=args.prefix,
                output_formats=tuple(args.formats or ("png", "svg")),
            )
        )
    for category_field in args.category_speedup or ():
        outputs.extend(
            generate_category_speedup_dot_figure(
                rows,
                category_field=category_field,
                output_dir=args.output_dir,
                prefix=args.prefix,
                output_formats=tuple(args.formats or ("png", "svg")),
                log_scale=False,
            )
        )
        if args.category_speedup_log:
            outputs.extend(
                generate_category_speedup_dot_figure(
                    rows,
                    category_field=category_field,
                    output_dir=args.output_dir,
                    prefix=args.prefix,
                    output_formats=tuple(args.formats or ("png", "svg")),
                    log_scale=True,
                )
            )
    if args.measured_throughput_speedup:
        if args.native_summary_csv is None:
            raise ValueError("--native-summary-csv is required for measured throughput speedup.")
        outputs.extend(
            generate_measured_throughput_speedup_figure(
                rows,
                native_rows=tuple(_load_rows(args.native_summary_csv)),
                replicas=args.throughput_replicas,
                worker_counts=tuple(args.throughput_worker_count or (1, 2)),
                preliminary_well_rows=(
                    tuple(_load_rows(args.preliminary_well_throughput_csv))
                    if args.preliminary_well_throughput_csv
                    else ()
                ),
                comparison_rows=(
                    tuple(_load_rows(args.comparison_throughput_csv))
                    if args.comparison_throughput_csv
                    else ()
                ),
                comparison_replicas=args.comparison_throughput_replicas,
                comparison_label=args.comparison_throughput_label,
                output_dir=args.output_dir,
                prefix=args.prefix,
                output_formats=tuple(args.formats or ("png", "svg")),
            )
        )
    if args.well_throughput_summary:
        if args.native_summary_csv is None:
            raise ValueError("--native-summary-csv is required for well throughput summary.")
        outputs.extend(
            generate_well_throughput_summary_figure(
                rows,
                native_rows=tuple(_load_rows(args.native_summary_csv)),
                excluded_cases=frozenset(args.exclude_case or ()),
                output_dir=args.output_dir,
                prefix=args.prefix,
                output_formats=tuple(args.formats or ("png", "svg")),
            )
        )
    for output in outputs:
        print(output)
    return 0


def generate_summary_distribution_figure(
    rows: Sequence[dict[str, str]],
    *,
    metric: str,
    output_dir: Path,
    prefix: str,
    output_formats: Sequence[str],
) -> tuple[Path, ...]:
    values_by_method = _values_by_method(rows, metric)
    if not values_by_method:
        return ()

    spec = _metric_spec(metric)
    methods = tuple(values_by_method)
    display_methods = tuple(_display_method_label(method) for method in methods)
    summaries = {
        method: _summary(values, higher_is_better=spec["higher_is_better"])
        for method, values in values_by_method.items()
    }

    fig = plt.figure(figsize=(8.1, 3.6), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 2.15))
    table_axis = fig.add_subplot(grid[0, 0])
    plot_axis = fig.add_subplot(grid[0, 1])
    table_axis.axis("off")

    cell_text = [
        [_format_value(summaries[method][column], spec["percentage"]) for column in SUMMARY_COLUMNS]
        for method in methods
    ]
    table = table_axis.table(
        cellText=cell_text,
        rowLabels=display_methods,
        colLabels=SUMMARY_COLUMNS,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)

    x_positions = tuple(range(len(methods)))
    for index, method in enumerate(methods):
        values = values_by_method[method]
        mean_value = summaries[method]["mean"]
        plot_axis.bar(
            index,
            mean_value,
            width=0.52,
            color=METHOD_COLORS.get(method, "#7570b3"),
            alpha=0.75,
            label=method,
        )
        jittered_x = [
            index + _deterministic_jitter(point_index, len(values))
            for point_index, _ in enumerate(values)
        ]
        plot_axis.scatter(
            jittered_x,
            values,
            s=18,
            color="#111111",
            alpha=0.58,
            linewidths=0,
            zorder=3,
        )

    if metric == "speedup_vs_native_cp":
        _plot_projected_core_lines(plot_axis, rows, methods, summaries)
    if spec["baseline"] is not None:
        plot_axis.axhline(spec["baseline"], color="#333333", linewidth=0.9, alpha=0.7)
    plot_axis.set_xticks(x_positions)
    if metric == "speedup_vs_native_cp":
        plot_axis.set_xticklabels(display_methods, rotation=45, ha="right", fontsize=8)
    else:
        plot_axis.set_xticklabels(display_methods, fontsize=8)
    plot_axis.set_ylabel(spec["ylabel"])
    plot_axis.grid(axis="y", alpha=0.25)
    plot_axis.yaxis.set_major_formatter(FuncFormatter(_plain_tick_label))
    plot_axis.set_ylim(bottom=0.0)

    outputs: list[Path] = []
    stem = f"{prefix}_{metric}_average_distribution"
    for output_format in output_formats:
        path = output_dir / f"{stem}.{output_format}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return tuple(outputs)


def generate_category_speedup_dot_figure(
    rows: Sequence[dict[str, str]],
    *,
    category_field: str,
    output_dir: Path,
    prefix: str,
    output_formats: Sequence[str],
    log_scale: bool,
) -> tuple[Path, ...]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("pipeline_name") == "Average" or row.get("method") != "OH1":
            continue
        category = _presentation_category(category_field, row.get(category_field, ""))
        speedup = _optional_float(row.get("speedup"))
        if not category or speedup is None:
            continue
        grouped.setdefault(category, []).append(speedup)
    if not grouped:
        return ()

    ordered = tuple(sorted(grouped, key=lambda category: sum(grouped[category]) / len(grouped[category])))
    means = [sum(grouped[category]) / len(grouped[category]) for category in ordered]
    fig_width = max(8.1, len(ordered) * 0.48)
    fig, axis = plt.subplots(figsize=(fig_width, 3.6), layout="constrained")
    x_positions = tuple(range(len(ordered)))
    axis.bar(x_positions, means, width=0.58, color="#0f8b8d", alpha=0.75)
    for index, category in enumerate(ordered):
        values = grouped[category]
        axis.scatter(
            [index + _deterministic_jitter(point_index, len(values)) for point_index, _ in enumerate(values)],
            values,
            s=22,
            color="#111111",
            alpha=0.62,
            linewidths=0,
            zorder=3,
        )
    axis.axhline(4.0, color="#d95f02", linewidth=1.1, alpha=0.85)
    axis.set_ylabel("Speedup vs CellProfiler (x)")
    if log_scale:
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(FuncFormatter(_plain_tick_label))
    axis.set_xticks(x_positions)
    axis.set_xticklabels([_wrap_label(category) for category in ordered], rotation=45, ha="right", fontsize=6.6)
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(bottom=1.0 if log_scale else 0.0)
    axis.text(
        0.02,
        0.95,
        "bar = category average\ndot = one benchmark",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#0f8b8d"},
    )

    outputs: list[Path] = []
    stem = f"{prefix}_{category_field}_speedup_dots{'_log' if log_scale else ''}"
    for output_format in output_formats:
        path = output_dir / f"{stem}.{output_format}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return tuple(outputs)


def generate_measured_throughput_speedup_figure(
    rows: Sequence[dict[str, str]],
    *,
    native_rows: Sequence[dict[str, str]],
    replicas: int,
    worker_counts: Sequence[int],
    preliminary_well_rows: Sequence[dict[str, str]],
    comparison_rows: Sequence[dict[str, str]],
    comparison_replicas: int,
    comparison_label: str,
    output_dir: Path,
    prefix: str,
    output_formats: Sequence[str],
) -> tuple[Path, ...]:
    """Plot measured multi-sample throughput against native CellProfiler timing."""
    native_seconds_by_case = _native_seconds_by_case(native_rows)
    grouped: dict[str, list[float]] = {
        _sample_condition_label(worker_count, replicas): [] for worker_count in worker_counts
    }
    for row in rows:
        if int(row.get(REPLICAS_FIELD, "0")) != replicas:
            continue
        worker_count = int(row.get(WORKER_COUNT_FIELD, "0"))
        if worker_count not in worker_counts:
            continue
        case_name = row.get(CASE_NAME_FIELD, "")
        native_seconds = native_seconds_by_case.get(case_name)
        wall_seconds = _optional_float(row.get(WALL_SECONDS_FIELD))
        if native_seconds is None or wall_seconds is None or wall_seconds <= 0.0:
            continue
        grouped[_sample_condition_label(worker_count, replicas)].append(
            native_seconds * replicas / wall_seconds
        )

    comparison_values = _batch_throughput_speedups(
        comparison_rows,
        native_seconds_by_case=native_seconds_by_case,
        replicas=comparison_replicas,
        worker_count=2,
    )
    if comparison_values:
        grouped[comparison_label] = comparison_values

    prelim_values = _preliminary_well_speedups(
        preliminary_well_rows,
        native_seconds_by_case=native_seconds_by_case,
    )
    if prelim_values:
        grouped["3 core prelim"] = prelim_values

    grouped = {method: values for method, values in grouped.items() if values}
    if not grouped:
        return ()

    methods = tuple(grouped)
    means = [sum(grouped[method]) / len(grouped[method]) for method in methods]
    fig, axis = plt.subplots(figsize=(8.1, 3.6), layout="constrained")
    x_positions = tuple(range(len(methods)))
    colors = ("#0f8b8d", "#d95f02", "#1b9e77")
    for index, (method, mean_value) in enumerate(zip(methods, means, strict=True)):
        axis.bar(
            index,
            mean_value,
            width=0.52,
            color=colors[index % len(colors)],
            alpha=0.78,
        )
        values = grouped[method]
        axis.scatter(
            [index + _deterministic_jitter(point_index, len(values)) for point_index, _ in enumerate(values)],
            values,
            s=18,
            color="#111111",
            alpha=0.58,
            linewidths=0,
            zorder=3,
        )

    axis.axhline(1.0, color="#333333", linewidth=0.9, alpha=0.7)
    axis.axhline(4.0, color="#333333", linewidth=0.9, alpha=0.35, linestyle="--")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Speedup vs CellProfiler (x)")
    axis.grid(axis="y", alpha=0.25)
    axis.yaxis.set_major_formatter(FuncFormatter(_plain_tick_label))
    axis.set_ylim(bottom=0.0)
    axis.text(
        0.02,
        0.95,
        f"Left: measured {replicas}-sample runs across available benchmark cases\n"
        f"Right: measured {comparison_replicas}-sample 2-core partial run",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#0f8b8d"},
    )

    outputs: list[Path] = []
    stem = f"{prefix}_measured_throughput_speedup"
    for output_format in output_formats:
        path = output_dir / f"{stem}.{output_format}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return tuple(outputs)


def _sample_condition_label(worker_count: int, replicas: int) -> str:
    return (
        f"{worker_count} core\n"
        f"{replicas} sample{'s' if replicas != 1 else ''}"
    )


def _batch_throughput_speedups(
    rows: Sequence[dict[str, str]],
    *,
    native_seconds_by_case: dict[str, float],
    replicas: int,
    worker_count: int,
) -> list[float]:
    speedups: list[float] = []
    for row in rows:
        if int(row.get(REPLICAS_FIELD, "0")) != replicas:
            continue
        if int(row.get(WORKER_COUNT_FIELD, "0")) != worker_count:
            continue
        case_name = row.get(CASE_NAME_FIELD, "")
        native_seconds = native_seconds_by_case.get(case_name)
        wall_seconds = _optional_float(row.get(WALL_SECONDS_FIELD))
        if native_seconds is None or wall_seconds is None or wall_seconds <= 0.0:
            continue
        speedups.append(native_seconds * replicas / wall_seconds)
    return speedups


def _native_seconds_by_case(rows: Sequence[dict[str, str]]) -> dict[str, float]:
    seconds_by_case: dict[str, float] = {}
    for row in rows:
        case_name = row.get(CASE_NAME_FIELD)
        native_seconds = _optional_float(row.get(NATIVE_SECONDS_FIELD))
        if case_name and native_seconds is not None:
            seconds_by_case[case_name] = native_seconds
    return seconds_by_case


def _preliminary_well_speedups(
    rows: Sequence[dict[str, str]],
    *,
    native_seconds_by_case: dict[str, float],
) -> list[float]:
    speedups: list[float] = []
    for row in rows:
        if int(row.get(WORKER_COUNT_FIELD, "0")) != 3:
            continue
        case_name = row.get(CASE_NAME_FIELD, "")
        native_seconds = native_seconds_by_case.get(case_name)
        well_count = _optional_float(row.get(WELL_COUNT_FIELD))
        total_seconds = _optional_float(row.get(TOTAL_SECONDS_FIELD))
        if native_seconds is None or well_count is None or total_seconds is None or total_seconds <= 0.0:
            continue
        speedups.append(native_seconds * well_count / total_seconds)
    return speedups


def generate_well_throughput_summary_figure(
    rows: Sequence[dict[str, str]],
    *,
    native_rows: Sequence[dict[str, str]],
    excluded_cases: frozenset[str],
    output_dir: Path,
    prefix: str,
    output_formats: Sequence[str],
) -> tuple[Path, ...]:
    native_seconds_by_case = _native_seconds_by_case(native_rows)
    points: list[tuple[str, float, float]] = []
    for row in rows:
        case_name = row.get(CASE_NAME_FIELD, "")
        if case_name in excluded_cases:
            continue
        native_seconds = native_seconds_by_case.get(case_name)
        well_count = _optional_float(row.get(WELL_COUNT_FIELD))
        total_seconds = _optional_float(row.get(TOTAL_SECONDS_FIELD))
        execute_seconds = _optional_float(row.get("execute_seconds"))
        if (
            native_seconds is None
            or well_count is None
            or total_seconds is None
            or execute_seconds is None
            or total_seconds <= 0.0
            or execute_seconds <= 0.0
        ):
            continue
        cp_total = native_seconds * well_count
        points.append((case_name, cp_total / total_seconds, cp_total / execute_seconds))
    if not points:
        return ()

    total_values = [point[1] for point in points]
    execute_values = [point[2] for point in points]
    series = (
        ("Whole run", total_values, "#0f8b8d"),
        ("Processing only", execute_values, "#d95f02"),
    )
    summaries = (
        ("Mean", [sum(values) / len(values) for _name, values, _color in series]),
        ("Median", [_median(values) for _name, values, _color in series]),
    )

    fig, axis = plt.subplots(figsize=(8.1, 3.6), layout="constrained")
    group_positions = (0, 1)
    for series_index, (series_name, values, color) in enumerate(series):
        offset = -0.17 if series_index == 0 else 0.17
        axis.bar(
            [position + offset for position in group_positions],
            [summary_values[series_index] for _summary_name, summary_values in summaries],
            width=0.32,
            color=color,
            alpha=0.78,
            label=series_name,
        )
        for group_index, (_summary_name, _summary_values) in enumerate(summaries):
            axis.scatter(
                [
                    group_positions[group_index]
                    + offset
                    + _deterministic_jitter(point_index, len(values)) * 0.55
                    for point_index, _ in enumerate(values)
                ],
                values,
                s=16,
                color="#111111",
                alpha=0.48,
                linewidths=0,
                zorder=3,
            )

    worker_count = int(float(rows[0][WORKER_COUNT_FIELD]))
    well_count = int(float(rows[0][WELL_COUNT_FIELD]))
    axis.axhline(1.0, color="#333333", linewidth=0.9, alpha=0.7)
    axis.axhline(4.0, color="#333333", linewidth=0.9, alpha=0.35, linestyle="--")
    axis.set_xticks(group_positions)
    axis.set_xticklabels([summary_name for summary_name, _values in summaries], fontsize=9)
    axis.set_ylabel("Speedup vs CellProfiler (x)")
    axis.grid(axis="y", alpha=0.25)
    axis.yaxis.set_major_formatter(FuncFormatter(_plain_tick_label))
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.text(
        0.98,
        0.95,
        f"Measured {well_count} wells, {worker_count} cores\n"
        f"{len(points)} pipelines; dots = pipelines"
        + (f"\nExcluded: {len(excluded_cases)} failed/frozen run" if excluded_cases else ""),
        transform=axis.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#0f8b8d"},
    )

    outputs: list[Path] = []
    stem = f"{prefix}_well_throughput_summary"
    for output_format in output_formats:
        path = output_dir / f"{stem}.{output_format}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return tuple(outputs)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _wrap_label(label: str, *, threshold: int = 24) -> str:
    if len(label) <= threshold or " " not in label:
        return label
    words = label.split()
    split_at = min(
        range(1, len(words)),
        key=lambda index: _label_split_cost(words, index),
    )
    return f"{' '.join(words[:split_at])}\n{' '.join(words[split_at:])}"


def _presentation_category(category_field: str, category: str) -> str:
    if category_field == "assay_category":
        return ASSAY_PRESENTATION_GROUPS.get(category, category)
    if category_field == "module_category":
        return MODULE_PRESENTATION_GROUPS.get(category, category)
    return category


def _label_split_cost(words: Sequence[str], index: int) -> tuple[int, int]:
    left = len(" ".join(words[:index]))
    right = len(" ".join(words[index:]))
    return max(left, right), abs(left - right)


def _plot_projected_core_lines(
    axis,
    rows: Sequence[dict[str, str]],
    methods: Sequence[str],
    summaries: dict[str, dict[str, float]],
) -> None:
    method_positions = {method: index for index, method in enumerate(methods)}
    core_methods = {
        method: _method_core_count(method)
        for method in methods
        if _method_core_count(method) is not None
    }
    if len(core_methods) < 2:
        return

    values_by_pipeline: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("pipeline_name") == "Average":
            continue
        method = row.get("method", "")
        if method not in core_methods:
            continue
        speedup = _optional_float(row.get("speedup_vs_native_cp"))
        if speedup is None:
            continue
        values_by_pipeline.setdefault(row["pipeline_name"], {})[method] = speedup

    for pipeline_values in values_by_pipeline.values():
        ordered_methods = [
            method for method in methods if method in pipeline_values
        ]
        if len(ordered_methods) < 2:
            continue
        axis.plot(
            [method_positions[method] for method in ordered_methods],
            [pipeline_values[method] for method in ordered_methods],
            color="#111111",
            alpha=0.18,
            linewidth=0.8,
            zorder=2,
        )

    xs = [method_positions[method] for method in methods if method in core_methods]
    ys = [summaries[method]["mean"] for method in methods if method in core_methods]
    slope, intercept = _linear_regression(xs, ys)
    fit_y = [intercept + slope * x for x in xs]
    axis.plot(xs, fit_y, color="#d95f02", linewidth=2.2, zorder=4)
    axis.text(
        0.02,
        0.96,
        f"Average trend: +{slope:.2f}x per core",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d95f02"},
    )


def _method_core_count(method: str) -> int | None:
    core_text = method.split(" ", 1)[0]
    return int(core_text) if core_text.isdigit() else None


def _display_method_label(method: str) -> str:
    core_count = _method_core_count(method)
    if core_count is not None:
        return f"{core_count} core"
    return method


def _load_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield _normalize_row(row)


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    if "pipeline_name" not in normalized and "case_name" in normalized:
        normalized["pipeline_name"] = normalized["case_name"]
    if "method" not in normalized and {"replicas", "worker_count"} <= normalized.keys():
        replicas = int(normalized["replicas"])
        worker_count = int(normalized["worker_count"])
        sample_label = f"{replicas} sample{'s' if replicas != 1 else ''}"
        worker_label = f"{worker_count} job{'s' if worker_count != 1 else ''}"
        normalized["method"] = f"{sample_label} / {worker_label}"
    return normalized


def _values_by_method(
    rows: Sequence[dict[str, str]],
    metric: str,
) -> dict[str, tuple[float, ...]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("pipeline_name") == "Average":
            continue
        if metric == "peak_memory_mb" and _is_overprovisioned_worker_row(row):
            continue
        value = _optional_float(row.get(metric))
        if value is None:
            continue
        if metric == "accuracy_fraction":
            value *= 100.0
        grouped.setdefault(row["method"], []).append(value)
    return {method: tuple(values) for method, values in grouped.items() if values}


def _is_overprovisioned_worker_row(row: dict[str, str]) -> bool:
    if "replicas" not in row or "worker_count" not in row:
        return False
    return int(row["worker_count"]) > int(row["replicas"])


def _summary(values: Sequence[float], *, higher_is_better: bool) -> dict[str, float]:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "worst": min(values) if higher_is_better else max(values),
        "mean": sum(values) / len(values),
        "median": median,
        "best": max(values) if higher_is_better else min(values),
    }


def _metric_spec(metric: str) -> dict[str, object]:
    if metric == "accuracy_fraction":
        return {
            "ylabel": "Parity accuracy (%)",
            "percentage": True,
            "higher_is_better": True,
            "baseline": 100.0,
        }
    if metric == "raw_seconds":
        return {
            "ylabel": "Raw execution seconds",
            "percentage": False,
            "higher_is_better": False,
            "baseline": None,
        }
    if metric == "peak_memory_mb":
        return {
            "ylabel": "Peak RSS (MB)",
            "percentage": False,
            "higher_is_better": False,
            "baseline": None,
        }
    return {
        "ylabel": "Speedup vs native CP (x)",
        "percentage": False,
        "higher_is_better": True,
        "baseline": 1.0,
    }


def generate_core_regression_figure(
    rows: Sequence[dict[str, str]],
    *,
    output_dir: Path,
    prefix: str,
    output_formats: Sequence[str],
) -> tuple[Path, ...]:
    points = tuple(_core_regression_points(rows))
    if not points:
        return ()
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    slope, intercept = _linear_regression(xs, ys)
    r_value = _pearson_r(xs, ys)
    x_min, x_max = min(xs), max(xs)
    line_x = [x_min, x_max]
    line_y = [intercept + slope * x_min, intercept + slope * x_max]

    fig, axis = plt.subplots(figsize=(8.1, 3.6), layout="constrained")
    for core_count in sorted(set(xs)):
        core_values = [
            y for x, y, _pipeline_name in points if x == core_count
        ]
        jittered_x = [
            core_count + _deterministic_jitter(index, len(core_values)) * 0.9
            for index, _ in enumerate(core_values)
        ]
        axis.scatter(
            jittered_x,
            core_values,
            s=18,
            alpha=0.58,
            color=METHOD_COLORS.get(f"{core_count} cores", "#111111"),
            linewidths=0,
            label=f"{core_count} core{'s' if core_count != 1 else ''}",
        )
    axis.plot(line_x, line_y, color="#d95f02", linewidth=2.0)
    axis.set_xlabel("OpenHCS cores")
    axis.set_ylabel("Projected speedup vs native CP (x)")
    axis.set_xticks(sorted(set(xs)))
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.02,
        0.96,
        f"OLS slope: {slope:.2f}x/core\nPearson r: {r_value:.3f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#0f8b8d"},
    )

    outputs: list[Path] = []
    for output_format in output_formats:
        path = output_dir / f"{prefix}_core_count_regression.{output_format}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return tuple(outputs)


def _core_regression_points(rows: Sequence[dict[str, str]]) -> Iterable[tuple[int, float, str]]:
    for row in rows:
        if row.get("pipeline_name") == "Average":
            continue
        method = row.get("method", "")
        core_text = method.split(" ", 1)[0]
        if not core_text.isdigit():
            continue
        speedup = _optional_float(row.get("speedup_vs_native_cp"))
        if speedup is None:
            continue
        yield int(core_text), speedup, row["pipeline_name"]


def _linear_regression(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0, mean_y
    slope = numerator / denominator
    return slope, mean_y - slope * mean_x


def _pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    x_var = sum((x - mean_x) ** 2 for x in xs)
    y_var = sum((y - mean_y) ** 2 for y in ys)
    if x_var == 0.0 or y_var == 0.0:
        return 0.0
    return numerator / math.sqrt(x_var * y_var)


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _format_value(value: float, percentage: bool) -> str:
    if percentage:
        return f"{value:.3f}%"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _plain_tick_label(value: float, position: int) -> str:
    del position
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _deterministic_jitter(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    spread = 0.34
    return -spread / 2 + spread * index / (count - 1)


if __name__ == "__main__":
    raise SystemExit(main())
