"""CellProfiler cppipe benchmark figures."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import LogLocator
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import NullLocator


CASE_NAME_FIELD = "case_name"
NATIVE_SECONDS_FIELD = "median_native_execution_seconds"
OPENHCS_SECONDS_FIELD = "median_openhcs_execution_seconds"
NATIVE_MEMORY_FIELD = "median_native_peak_memory_mb"
OPENHCS_MEMORY_FIELD = "median_openhcs_peak_memory_mb"
SPEEDUP_FIELD = "median_speedup"
ACCURACY_FIELD = "min_parity_accuracy"
CELLPROFILER_LABEL = "CP"
DEFAULT_OPENHCS_LABEL = "OH1"
DEFAULT_FORMATS = ("png", "svg")
DEFAULT_WRAP_AFTER = 14
ACCURACY_ZOOM_HALF_RANGE_PERCENT = 0.001


@dataclass(frozen=True, slots=True)
class SummarySource:
    """One OpenHCS benchmark variant summary CSV."""

    label: str
    path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkMetricRow:
    """Long-form metric values for one method on one pipeline."""

    pipeline_name: str
    method: str
    accuracy_fraction: float | None
    raw_seconds: float | None
    speedup: float | None
    peak_memory_mb: float | None


@dataclass(frozen=True, slots=True)
class FigureMetricSpec:
    """One grouped-bar chart projection."""

    key: str
    filename_stem: str
    title: str
    ylabel: str
    percentage: bool = False
    baseline_line: float | None = None
    minimum_ylim: float | None = None
    log_variant: bool = False


FIGURE_METRICS = (
    FigureMetricSpec(
        "accuracy_fraction",
        "cppipe_accuracy",
        "Semantic parity accuracy",
        "Parity accuracy (%)",
        percentage=True,
        minimum_ylim=0.0,
    ),
    FigureMetricSpec(
        "raw_seconds",
        "cppipe_raw_seconds",
        "Single-thread execution runtime",
        "Raw execution seconds",
        minimum_ylim=0.0,
        log_variant=True,
    ),
    FigureMetricSpec(
        "speedup",
        "cppipe_speedup",
        "Execution speedup versus native CellProfiler",
        "Speedup (x)",
        baseline_line=1.0,
        minimum_ylim=0.0,
        log_variant=True,
    ),
    FigureMetricSpec(
        "peak_memory_mb",
        "cppipe_peak_memory",
        "Peak process-tree memory usage",
        "Peak RSS (MB)",
        minimum_ylim=0.0,
        log_variant=True,
    ),
)

METHOD_COLORS = (
    "#262626",
    "#0f8b8d",
    "#d95f02",
    "#1b9e77",
    "#7570b3",
)


def parse_summary_source(value: str) -> SummarySource:
    """Parse ``LABEL=/path/to/summary.csv`` CLI syntax."""
    label, separator, path_text = value.partition("=")
    if not separator:
        return SummarySource(DEFAULT_OPENHCS_LABEL, Path(value))
    clean_label = label.strip()
    if not clean_label:
        raise ValueError(f"Summary source label cannot be empty: {value!r}")
    return SummarySource(clean_label, Path(path_text))


def generate_cppipe_benchmark_figures(
    summary_sources: Sequence[SummarySource],
    *,
    output_dir: Path,
    output_formats: Sequence[str] = DEFAULT_FORMATS,
    include_average: bool = True,
    wrap_after: int = DEFAULT_WRAP_AFTER,
) -> tuple[Path, ...]:
    """Generate grouped CP/OH cppipe benchmark figures and a long-form CSV."""
    if not summary_sources:
        raise ValueError("At least one summary source is required.")

    source_tables = tuple(_load_summary_table(source) for source in summary_sources)
    pipeline_names = _pipeline_order(source_tables)
    methods = (CELLPROFILER_LABEL,) + tuple(source.label for source in summary_sources)
    rows = tuple(
        _benchmark_metric_rows(
            source_tables,
            summary_sources=summary_sources,
            pipeline_names=pipeline_names,
            include_average=include_average,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv_path = output_dir / "cppipe_comparison_metrics_long.csv"
    _write_metric_rows(long_csv_path, rows)

    plotted_pipeline_names = tuple(dict.fromkeys(row.pipeline_name for row in rows))
    outputs: list[Path] = [long_csv_path]
    outputs.extend(
        generate_grouped_benchmark_metric_figures(
            rows,
            metrics=FIGURE_METRICS,
            methods=methods,
            pipeline_names=plotted_pipeline_names,
            output_dir=output_dir,
            output_formats=output_formats,
            wrap_after=wrap_after,
        )
    )
    for metric in FIGURE_METRICS:
        if metric.key == "accuracy_fraction":
            outputs.extend(
                _plot_accuracy_zoom(
                    rows,
                    methods=methods,
                    pipeline_names=plotted_pipeline_names,
                    output_dir=output_dir,
                    output_formats=output_formats,
                    wrap_after=wrap_after,
                )
            )
    return tuple(outputs)


def generate_grouped_benchmark_metric_figures(
    rows: Sequence[BenchmarkMetricRow],
    *,
    metrics: Sequence[FigureMetricSpec],
    methods: Sequence[str],
    pipeline_names: Sequence[str],
    output_dir: Path,
    output_formats: Sequence[str] = DEFAULT_FORMATS,
    wrap_after: int = DEFAULT_WRAP_AFTER,
) -> tuple[Path, ...]:
    """Generate v7-style grouped-bar figures for long-form benchmark rows."""
    outputs: list[Path] = []
    for metric in metrics:
        if not any(_metric_value(row, metric) is not None for row in rows):
            continue
        outputs.extend(
            _plot_grouped_metric(
                rows,
                metric=metric,
                methods=methods,
                pipeline_names=pipeline_names,
                output_dir=output_dir,
                output_formats=output_formats,
                wrap_after=wrap_after,
                log_y=False,
            )
        )
        if metric.log_variant:
            outputs.extend(
                _plot_grouped_metric(
                    rows,
                    metric=metric,
                    methods=methods,
                    pipeline_names=pipeline_names,
                    output_dir=output_dir,
                    output_formats=output_formats,
                    wrap_after=wrap_after,
                    log_y=True,
                )
            )
    return tuple(outputs)


def _load_summary_table(source: SummarySource) -> dict[str, dict[str, str]]:
    with source.path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Summary CSV is empty: {source.path}")
    required = {
        CASE_NAME_FIELD,
        NATIVE_SECONDS_FIELD,
        OPENHCS_SECONDS_FIELD,
        ACCURACY_FIELD,
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Summary CSV {source.path} missing columns: {sorted(missing)!r}"
        )
    return {row[CASE_NAME_FIELD]: row for row in rows}


def _pipeline_order(source_tables: Sequence[dict[str, dict[str, str]]]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for table in source_tables:
        for pipeline_name in table:
            if pipeline_name in seen:
                continue
            ordered.append(pipeline_name)
            seen.add(pipeline_name)
    return tuple(ordered)


def _benchmark_metric_rows(
    source_tables: Sequence[dict[str, dict[str, str]]],
    *,
    summary_sources: Sequence[SummarySource],
    pipeline_names: Sequence[str],
    include_average: bool,
) -> Iterable[BenchmarkMetricRow]:
    first_table = source_tables[0]
    for pipeline_name in pipeline_names:
        baseline_row = first_table.get(pipeline_name)
        yield BenchmarkMetricRow(
            pipeline_name=pipeline_name,
            method=CELLPROFILER_LABEL,
            accuracy_fraction=1.0,
            raw_seconds=_optional_float_from_row(baseline_row, NATIVE_SECONDS_FIELD),
            speedup=1.0,
            peak_memory_mb=_optional_float_from_row(baseline_row, NATIVE_MEMORY_FIELD),
        )
        for source, table in zip(summary_sources, source_tables, strict=True):
            row = table.get(pipeline_name)
            native_seconds = _optional_float_from_row(row, NATIVE_SECONDS_FIELD)
            openhcs_seconds = _optional_float_from_row(row, OPENHCS_SECONDS_FIELD)
            yield BenchmarkMetricRow(
                pipeline_name=pipeline_name,
                method=source.label,
                accuracy_fraction=_optional_float_from_row(row, ACCURACY_FIELD),
                raw_seconds=openhcs_seconds,
                speedup=_speedup(row, native_seconds, openhcs_seconds),
                peak_memory_mb=_optional_float_from_row(row, OPENHCS_MEMORY_FIELD),
            )
    if include_average:
        yield from _average_rows(
            _benchmark_metric_rows(
                source_tables,
                summary_sources=summary_sources,
                pipeline_names=pipeline_names,
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


def _write_metric_rows(path: Path, rows: Sequence[BenchmarkMetricRow]) -> None:
    fieldnames = (
        "pipeline_name",
        "method",
        "accuracy_fraction",
        "raw_seconds",
        "speedup",
        "peak_memory_mb",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "pipeline_name": row.pipeline_name,
                    "method": row.method,
                    "accuracy_fraction": row.accuracy_fraction,
                    "raw_seconds": row.raw_seconds,
                    "speedup": row.speedup,
                    "peak_memory_mb": row.peak_memory_mb,
                }
            )


def _plot_grouped_metric(
    rows: Sequence[BenchmarkMetricRow],
    *,
    metric: FigureMetricSpec,
    methods: Sequence[str],
    pipeline_names: Sequence[str],
    output_dir: Path,
    output_formats: Sequence[str],
    wrap_after: int,
    log_y: bool,
) -> tuple[Path, ...]:
    panels = _pipeline_panels(pipeline_names, wrap_after)
    fig_width = max(11.0, max(len(panel) for panel in panels) * 0.82)
    fig_height = 5.4 if len(panels) == 1 else 8.4
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(fig_width, fig_height),
        layout="constrained",
    )
    panel_axes = (axes,) if len(panels) == 1 else tuple(axes)
    width = min(0.16, 0.78 / max(len(methods), 1))
    offsets = _bar_offsets(len(methods), width)
    row_index = {(row.pipeline_name, row.method): row for row in rows}

    for panel_index, (axis, panel_names) in enumerate(zip(panel_axes, panels, strict=True)):
        x_positions = tuple(range(len(panel_names)))
        for method_index, method in enumerate(methods):
            values = [
                _plot_value(
                    _metric_value(row_index.get((pipeline_name, method)), metric)
                )
                for pipeline_name in panel_names
            ]
            axis.bar(
                [x + offsets[method_index] for x in x_positions],
                values,
                width=width,
                label=method if panel_index == 0 else None,
                color=METHOD_COLORS[method_index % len(METHOD_COLORS)],
            )

        if metric.baseline_line is not None:
            axis.axhline(
                metric.baseline_line,
                color="#333333",
                linewidth=1.0,
                alpha=0.8,
            )
        axis.set_ylabel(metric.ylabel)
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(panel_names, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        if metric.minimum_ylim is not None and not log_y:
            axis.set_ylim(bottom=metric.minimum_ylim)
        if metric.percentage:
            axis.set_ylim(0.0, 105.0)
        if log_y:
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
            axis.yaxis.set_minor_locator(NullLocator())
            axis.yaxis.set_major_formatter(FuncFormatter(_plain_log_tick_label))
            axis.yaxis.set_minor_formatter(NullFormatter())

    panel_axes[0].set_title(
        f"{metric.title} (log scale)" if log_y else metric.title
    )
    panel_axes[0].legend(frameon=False, ncol=min(len(methods), 5), loc="upper left")
    outputs: list[Path] = []
    filename_stem = f"{metric.filename_stem}_log" if log_y else metric.filename_stem
    for output_format in output_formats:
        output_path = output_dir / f"{filename_stem}.{output_format}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs.append(output_path)
    plt.close(fig)
    return tuple(outputs)


def _plot_accuracy_zoom(
    rows: Sequence[BenchmarkMetricRow],
    *,
    methods: Sequence[str],
    pipeline_names: Sequence[str],
    output_dir: Path,
    output_formats: Sequence[str],
    wrap_after: int,
) -> tuple[Path, ...]:
    """Plot accuracy with a broken y-axis so tiny parity drift is visible."""
    panels = _pipeline_panels(pipeline_names, wrap_after)
    panel_count = len(panels)
    fig_width = max(11.0, max(len(panel) for panel in panels) * 0.82)
    fig, axes = plt.subplots(
        panel_count * 2,
        1,
        figsize=(fig_width, 5.6 * panel_count),
        gridspec_kw={"height_ratios": tuple((2.7, 1.0) * panel_count)},
        layout="constrained",
    )
    all_axes = (axes,) if panel_count == 1 else tuple(axes)
    width = min(0.16, 0.78 / max(len(methods), 1))
    offsets = _bar_offsets(len(methods), width)
    row_index = {(row.pipeline_name, row.method): row for row in rows}
    metric = FIGURE_METRICS[0]

    for panel_index, panel_names in enumerate(panels):
        zoom_axis = all_axes[panel_index * 2]
        context_axis = all_axes[panel_index * 2 + 1]
        x_positions = tuple(range(len(panel_names)))
        for axis in (zoom_axis, context_axis):
            for method_index, method in enumerate(methods):
                values = [
                    _plot_value(
                        _metric_value(row_index.get((pipeline_name, method)), metric)
                    )
                    for pipeline_name in panel_names
                ]
                axis.bar(
                    [x + offsets[method_index] for x in x_positions],
                    values,
                    width=width,
                    label=method if panel_index == 0 and axis is zoom_axis else None,
                    color=METHOD_COLORS[method_index % len(METHOD_COLORS)],
                )
            axis.grid(axis="y", alpha=0.25)
            axis.set_xticks(list(x_positions))

        zoom_axis.set_ylim(
            100.0 - ACCURACY_ZOOM_HALF_RANGE_PERCENT,
            100.0 + ACCURACY_ZOOM_HALF_RANGE_PERCENT,
        )
        zoom_axis.axhline(100.0, color="#333333", linewidth=0.9, alpha=0.7)
        zoom_axis.yaxis.set_major_formatter(FuncFormatter(_percent_tick_label))
        zoom_axis.yaxis.get_offset_text().set_visible(False)
        zoom_axis.set_ylabel("Accuracy (%)")
        zoom_axis.set_xticklabels(())
        context_axis.set_ylim(0.0, 5.0)
        context_axis.set_ylabel("0-5%")
        context_axis.set_xticklabels(panel_names, rotation=45, ha="right", fontsize=8)
        _mark_axis_break(zoom_axis, context_axis)

    all_axes[0].set_title("Semantic parity accuracy, broken y-axis zoom")
    all_axes[0].legend(frameon=False, ncol=min(len(methods), 5), loc="upper left")
    outputs: list[Path] = []
    for output_format in output_formats:
        output_path = output_dir / f"cppipe_accuracy_zoom.{output_format}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs.append(output_path)
    plt.close(fig)
    return tuple(outputs)


def _mark_axis_break(top_axis, bottom_axis) -> None:
    top_axis.spines.bottom.set_visible(False)
    bottom_axis.spines.top.set_visible(False)
    top_axis.tick_params(labeltop=False, bottom=False)
    bottom_axis.xaxis.tick_bottom()
    marker_size = 0.008
    marker_kwargs = dict(transform=top_axis.transAxes, color="k", clip_on=False)
    top_axis.plot((-marker_size, +marker_size), (-marker_size, +marker_size), **marker_kwargs)
    top_axis.plot(
        (1 - marker_size, 1 + marker_size),
        (-marker_size, +marker_size),
        **marker_kwargs,
    )
    marker_kwargs.update(transform=bottom_axis.transAxes)
    bottom_axis.plot(
        (-marker_size, +marker_size),
        (1 - marker_size, 1 + marker_size),
        **marker_kwargs,
    )
    bottom_axis.plot(
        (1 - marker_size, 1 + marker_size),
        (1 - marker_size, 1 + marker_size),
        **marker_kwargs,
    )


def _pipeline_panels(
    pipeline_names: Sequence[str],
    wrap_after: int,
) -> tuple[tuple[str, ...], ...]:
    if wrap_after <= 0 or len(pipeline_names) <= wrap_after:
        return (tuple(pipeline_names),)
    split = math.ceil(len(pipeline_names) / 2)
    return (tuple(pipeline_names[:split]), tuple(pipeline_names[split:]))


def _plain_log_tick_label(value: float, position: int) -> str:
    del position
    if value <= 0.0 or not math.isfinite(value):
        return ""
    if value >= 100.0:
        return f"{value:.0f}"
    if value >= 10.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if value >= 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _percent_tick_label(value: float, position: int) -> str:
    del position
    if not math.isfinite(value):
        return ""
    return f"{value:.3f}"


def _bar_offsets(method_count: int, width: float) -> tuple[float, ...]:
    center = (method_count - 1) / 2.0
    return tuple((index - center) * width for index in range(method_count))


def _metric_value(
    row: BenchmarkMetricRow | None,
    metric: FigureMetricSpec,
) -> float | None:
    if row is None:
        return None
    value = getattr(row, metric.key)
    if value is None:
        return None
    return float(value) * 100.0 if metric.percentage else float(value)


def _plot_value(value: float | None) -> float:
    return math.nan if value is None else value


def _optional_float_from_row(
    row: dict[str, str] | None,
    field_name: str,
) -> float | None:
    if row is None:
        return None
    value = row.get(field_name)
    if value is None or value == "":
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _speedup(
    row: dict[str, str] | None,
    native_seconds: float | None,
    openhcs_seconds: float | None,
) -> float | None:
    explicit_speedup = _optional_float_from_row(row, SPEEDUP_FIELD)
    if explicit_speedup is not None:
        return explicit_speedup
    if native_seconds is None or openhcs_seconds is None or openhcs_seconds <= 0.0:
        return None
    return native_seconds / openhcs_seconds


def _mean_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)
