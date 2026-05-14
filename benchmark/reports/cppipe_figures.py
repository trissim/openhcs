"""CellProfiler cppipe benchmark figures."""

from __future__ import annotations

import csv
import math
import re
import statistics
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import LogLocator
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import NullLocator

from benchmark.contracts.dataset import BenchmarkCategory
from benchmark.datasets.cppipe_case_catalog import DEFAULT_BENCHMARK_CATEGORY
from benchmark.datasets.cppipe_case_catalog import official_cp3_case_category


CASE_NAME_FIELD = "case_name"
ASSAY_CATEGORY_FIELD = "assay_category"
MODULE_CATEGORY_FIELD = "module_category"
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
DEFAULT_GROUP_WIDTH_INCHES = 0.98
SINGLE_PANEL_HEIGHT_INCHES = 4.4
MULTI_PANEL_HEIGHT_INCHES = 7.2
ACCURACY_ZOOM_PANEL_HEIGHT_INCHES = 4.5
GROUPED_BAR_MAX_WIDTH = 0.22
GROUPED_BAR_FRACTION = 0.9
PIPELINE_LABEL_FONT_SIZE = 7.2
PIPELINE_LABEL_WRAP_THRESHOLD = 29
ACCURACY_ZOOM_HALF_RANGE_PERCENT = 0.001
SPEEDUP_TARGET = 4.0
FIGURE_DPI = 360
PIPELINE_NAME_FIELD = "pipeline_name"
METHOD_FIELD = "method"
AGGREGATE_LABEL = "Aggregate"
ACCURACY_FRACTION_FIELD = "accuracy_fraction"
RAW_SECONDS_FIELD = "raw_seconds"
SPEEDUP_METRIC_KEY = "speedup"
PEAK_MEMORY_MB_FIELD = "peak_memory_mb"
SummaryRow = dict[str, str]
SummaryTable = dict[str, SummaryRow]
SummaryTables = Sequence[SummaryTable]


@dataclass(frozen=True)
class SummarySource:
    """One OpenHCS benchmark variant summary CSV."""

    label: str
    path: Path


@dataclass(frozen=True)
class BenchmarkMetricRow:
    """Long-form metric values for one method on one pipeline."""

    pipeline_name: str
    method: str
    assay_category: str
    module_category: str
    accuracy_fraction: float | None
    raw_seconds: float | None
    speedup: float | None
    peak_memory_mb: float | None


@dataclass(frozen=True)
class FigureMetricSpec:
    """One grouped-bar chart projection."""

    key: str
    filename_stem: str
    title: str
    ylabel: str
    percentage: bool = False
    baseline_line: float | None = None
    target_line: float | None = None
    minimum_ylim: float | None = None
    log_variant: bool = False
    use_axis_break: bool = True


@dataclass(frozen=True)
class SummaryRowNumerics:
    """Typed numeric access to benchmark summary CSV rows."""

    speedup_field: str = SPEEDUP_FIELD

    def optional_float(
        self,
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

    def speedup(
        self,
        row: dict[str, str] | None,
        native_seconds: float | None,
        openhcs_seconds: float | None,
    ) -> float | None:
        explicit_speedup = self.optional_float(row, self.speedup_field)
        if explicit_speedup is not None:
            return explicit_speedup
        if native_seconds is None or openhcs_seconds is None or openhcs_seconds <= 0.0:
            return None
        return native_seconds / openhcs_seconds

    def speedup_from_summary_row(self, row: SummaryRow | None) -> float | None:
        if row is None:
            return None
        speedup = self.optional_float(row, self.speedup_field)
        if speedup is not None:
            return speedup
        return self.speedup(
            row,
            self.optional_float(row, NATIVE_SECONDS_FIELD),
            self.optional_float(row, OPENHCS_SECONDS_FIELD),
        )


@dataclass(frozen=True)
class BenchmarkMetricProjection:
    """Metric-specific projection from long-form rows to plot values."""

    def value(
        self,
        row: BenchmarkMetricRow | None,
        metric: FigureMetricSpec,
    ) -> float | None:
        if row is None:
            return None
        value = getattr(row, metric.key)
        if value is None:
            return None
        return float(value) * 100.0 if metric.percentage else float(value)

    def plot_value(self, value: float | None) -> float:
        return math.nan if value is None else value


@dataclass(frozen=True)
class PipelineLabelLayout:
    """Owns panel partitioning and readable pipeline tick labels."""

    wrap_threshold: int = PIPELINE_LABEL_WRAP_THRESHOLD

    def panels(
        self,
        pipeline_names: Sequence[str],
        wrap_after: int,
    ) -> tuple[tuple[str, ...], ...]:
        if wrap_after <= 0 or len(pipeline_names) <= wrap_after:
            return (tuple(pipeline_names),)
        split = math.ceil(len(pipeline_names) / 2)
        return (tuple(pipeline_names[:split]), tuple(pipeline_names[split:]))

    def split_label(self, label: str) -> str:
        tokens = self._tokens(label)
        if len(tokens) < 2 or len(" ".join(tokens)) < self.wrap_threshold:
            return label
        split_at = min(
            range(1, len(tokens)),
            key=lambda index: self._split_cost(tokens, index),
        )
        return f"{' '.join(tokens[:split_at])}\n{' '.join(tokens[split_at:])}"

    @staticmethod
    def _split_cost(tokens: Sequence[str], index: int) -> tuple[int, int]:
        left_length = len(" ".join(tokens[:index]))
        right_length = len(" ".join(tokens[index:]))
        return max(left_length, right_length), abs(left_length - right_length)

    @staticmethod
    def _tokens(label: str) -> tuple[str, ...]:
        spaced = label.replace("_", " ")
        tokens: list[str] = []
        for part in spaced.split():
            tokens.extend(
                token
                for token in re.findall(
                    r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+",
                    part,
                )
                if token
            )
        return tuple(tokens) or (label,)


@dataclass(frozen=True)
class GroupedFigureRequest:
    """Shared request context for grouped benchmark figure panels."""

    rows: Sequence[BenchmarkMetricRow]
    methods: Sequence[str]
    pipeline_names: Sequence[str]
    output_dir: Path
    output_formats: Sequence[str]
    wrap_after: int
    group_width_inches: float


@dataclass(frozen=True)
class BenchmarkFigureStyle:
    """Publication-oriented styling for CellProfiler benchmark figures."""

    method_colors: tuple[str, ...] = (
        "#252525",
        "#007f7f",
        "#d95f02",
        "#1b9e77",
        "#7570b3",
    )
    background: str = "#fbfaf7"
    grid_color: str = "#dad4c7"
    spine_color: str = "#5c554b"
    text_color: str = "#252525"
    target_color: str = "#b2182b"
    baseline_color: str = "#5c554b"

    @contextmanager
    def context(self):
        with plt.rc_context(self.rc_params):
            yield

    @property
    def rc_params(self) -> dict[str, object]:
        return {
            "figure.facecolor": self.background,
            "axes.facecolor": self.background,
            "axes.edgecolor": self.spine_color,
            "axes.labelcolor": self.text_color,
            "axes.titlecolor": self.text_color,
            "xtick.color": self.text_color,
            "ytick.color": self.text_color,
            "text.color": self.text_color,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": PIPELINE_LABEL_FONT_SIZE,
            "ytick.labelsize": 8.5,
            "savefig.facecolor": self.background,
            "savefig.edgecolor": self.background,
        }

    def color_for_method(self, method_index: int) -> str:
        return self.method_colors[method_index % len(self.method_colors)]

    def decorate_axis(self, axis, *, metric: FigureMetricSpec, panel_index: int) -> None:
        axis.grid(axis="y", color=self.grid_color, linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(self.spine_color)
        axis.spines["bottom"].set_color(self.spine_color)
        if panel_index == 0:
            axis.set_title(metric.title, loc="left", pad=10)

    def save(self, fig, output_path: Path) -> None:
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")


@dataclass(frozen=True)
class LinearAxisBreakPolicy:
    """Automatic linear-axis break policy for extreme benchmark outliers."""

    outlier_ratio: float = 3.0
    max_upper_fraction: float = 0.35
    lower_reference_quantile: float = 0.75
    lower_padding: float = 1.18
    upper_window_bottom: float = 0.92
    break_gap_padding: float = 1.08
    upper_padding: float = 1.06
    marker_size: float = 0.008

    def range_for(self, values: Sequence[float]) -> tuple[float, float, float] | None:
        present = sorted(
            value for value in values if math.isfinite(value) and value > 0.0
        )
        if len(present) < 2:
            return None
        split_index = self._outlier_split_index(present)
        if split_index is None:
            return None
        low_top = present[split_index - 1] * self.lower_padding
        high_bottom = low_top * self.break_gap_padding
        high_top = present[-1] * self.upper_padding
        if high_bottom <= low_top:
            return None
        return low_top, high_bottom, high_top

    def _outlier_split_index(self, present: Sequence[float]) -> int | None:
        max_upper_count = max(1, math.floor(len(present) * self.max_upper_fraction))
        candidates: list[tuple[float, int]] = []
        for index in range(1, len(present)):
            upper_count = len(present) - index
            if upper_count > max_upper_count:
                continue
            lower_values = present[:index]
            reference_index = min(
                len(lower_values) - 1,
                max(0, math.floor((len(lower_values) - 1) * self.lower_reference_quantile)),
            )
            lower_reference = lower_values[reference_index]
            upper_bottom = present[index]
            if upper_bottom < lower_reference * self.outlier_ratio:
                continue
            if upper_bottom * self.upper_window_bottom <= present[index - 1] * self.lower_padding:
                continue
            candidates.append((upper_bottom / present[index - 1], index))
        if candidates:
            return max(candidates)[1]
        return None

    def mark(self, top_axis, bottom_axis) -> None:
        top_axis.spines.bottom.set_visible(False)
        bottom_axis.spines.top.set_visible(False)
        top_axis.tick_params(labeltop=False, bottom=False)
        bottom_axis.xaxis.tick_bottom()
        marker_kwargs = dict(transform=top_axis.transAxes, color="k", clip_on=False)
        top_axis.plot(
            (-self.marker_size, +self.marker_size),
            (-self.marker_size, +self.marker_size),
            **marker_kwargs,
        )
        top_axis.plot(
            (1 - self.marker_size, 1 + self.marker_size),
            (-self.marker_size, +self.marker_size),
            **marker_kwargs,
        )
        marker_kwargs.update(transform=bottom_axis.transAxes)
        bottom_axis.plot(
            (-self.marker_size, +self.marker_size),
            (1 - self.marker_size, 1 + self.marker_size),
            **marker_kwargs,
        )
        bottom_axis.plot(
            (1 - self.marker_size, 1 + self.marker_size),
            (1 - self.marker_size, 1 + self.marker_size),
            **marker_kwargs,
        )


FIGURE_STYLE = BenchmarkFigureStyle()
LINEAR_AXIS_BREAK_POLICY = LinearAxisBreakPolicy()
SUMMARY_ROW_NUMERICS = SummaryRowNumerics()
METRIC_PROJECTION = BenchmarkMetricProjection()
PIPELINE_LABEL_LAYOUT = PipelineLabelLayout()


FIGURE_METRICS = (
    FigureMetricSpec(
        "accuracy_fraction",
        "cppipe_accuracy",
        "Parity accuracy",
        "Parity accuracy (%)",
        percentage=True,
        target_line=100.0,
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
        SPEEDUP_METRIC_KEY,
        "cppipe_speedup",
        "Execution speedup versus native CellProfiler",
        "Speedup (x)",
        baseline_line=1.0,
        target_line=SPEEDUP_TARGET,
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
    group_width_inches: float = DEFAULT_GROUP_WIDTH_INCHES,
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
            group_width_inches=group_width_inches,
        )
    )
    grouped_request = GroupedFigureRequest(
        rows=rows,
        methods=methods,
        pipeline_names=plotted_pipeline_names,
        output_dir=output_dir,
        output_formats=output_formats,
        wrap_after=wrap_after,
        group_width_inches=group_width_inches,
    )
    outputs.extend(
        _plot_average_speedup_points(
            source_tables,
            summary_sources=summary_sources,
            pipeline_names=pipeline_names,
            output_dir=output_dir,
            output_formats=output_formats,
        )
    )
    outputs.extend(
        generate_speedup_distribution_artifacts(
            tuple(
                SpeedupDistributionSeries.from_points(
                    _speedup_point_series(
                        table,
                        source=source,
                        pipeline_names=pipeline_names,
                    )
                )
                for source, table in zip(summary_sources, source_tables, strict=True)
            ),
            output_dir=output_dir,
            filename_prefix="cppipe_speedup",
            title="Speedup cumulative distribution",
            xlabel="Execution speedup versus native CellProfiler (x)",
            output_formats=output_formats,
        )
    )
    category_rows = tuple(_category_metric_rows(rows, category_key=ASSAY_CATEGORY_FIELD))
    module_rows = tuple(_category_metric_rows(rows, category_key=MODULE_CATEGORY_FIELD))
    category_csv_path = output_dir / "cppipe_comparison_category_metrics_long.csv"
    _write_metric_rows(category_csv_path, (*category_rows, *module_rows))
    outputs.append(category_csv_path)
    outputs.extend(
        generate_grouped_benchmark_metric_figures(
            category_rows,
            metrics=_category_metrics(ASSAY_CATEGORY_FIELD),
            methods=methods,
            pipeline_names=_category_order(category_rows),
            output_dir=output_dir,
            output_formats=output_formats,
            wrap_after=wrap_after,
            group_width_inches=group_width_inches,
        )
    )
    outputs.extend(
        generate_grouped_benchmark_metric_figures(
            module_rows,
            metrics=_category_metrics(MODULE_CATEGORY_FIELD),
            methods=methods,
            pipeline_names=_category_order(module_rows),
            output_dir=output_dir,
            output_formats=output_formats,
            wrap_after=wrap_after,
            group_width_inches=group_width_inches,
        )
    )
    for metric in FIGURE_METRICS:
        if metric.key == ACCURACY_FRACTION_FIELD:
            outputs.extend(_plot_accuracy_zoom(grouped_request))
    figure_index_path = output_dir / "benchmark_figure_index.md"
    _write_benchmark_figure_index(figure_index_path, outputs)
    outputs.append(figure_index_path)
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
    group_width_inches: float = DEFAULT_GROUP_WIDTH_INCHES,
) -> tuple[Path, ...]:
    """Generate v7-style grouped-bar figures for long-form benchmark rows."""
    request = GroupedFigureRequest(
        rows=rows,
        methods=methods,
        pipeline_names=pipeline_names,
        output_dir=output_dir,
        output_formats=output_formats,
        wrap_after=wrap_after,
        group_width_inches=group_width_inches,
    )
    outputs: list[Path] = []
    for metric in metrics:
        if not any(METRIC_PROJECTION.value(row, metric) is not None for row in request.rows):
            continue
        outputs.extend(
            _plot_grouped_metric(
                request,
                metric=metric,
                log_y=False,
            )
        )
        if metric.log_variant:
            outputs.extend(
                _plot_grouped_metric(
                    request,
                    metric=metric,
                    log_y=True,
                )
            )
    return tuple(outputs)


def _load_summary_table(source: SummarySource) -> SummaryTable:
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


def _pipeline_order(source_tables: SummaryTables) -> tuple[str, ...]:
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
    source_tables: SummaryTables,
    *,
    summary_sources: Sequence[SummarySource],
    pipeline_names: Sequence[str],
    include_average: bool,
) -> Iterable[BenchmarkMetricRow]:
    first_table = source_tables[0]
    for pipeline_name in pipeline_names:
        baseline_row = first_table.get(pipeline_name)
        baseline_category = _category_from_summary_row(pipeline_name, baseline_row)
        yield BenchmarkMetricRow(
            pipeline_name=pipeline_name,
            method=CELLPROFILER_LABEL,
            assay_category=baseline_category.assay,
            module_category=baseline_category.module,
            accuracy_fraction=1.0,
            raw_seconds=SUMMARY_ROW_NUMERICS.optional_float(
                baseline_row,
                NATIVE_SECONDS_FIELD,
            ),
            speedup=1.0,
            peak_memory_mb=SUMMARY_ROW_NUMERICS.optional_float(
                baseline_row,
                NATIVE_MEMORY_FIELD,
            ),
        )
        for source, table in zip(summary_sources, source_tables, strict=True):
            row = table.get(pipeline_name)
            native_seconds = SUMMARY_ROW_NUMERICS.optional_float(
                row,
                NATIVE_SECONDS_FIELD,
            )
            openhcs_seconds = SUMMARY_ROW_NUMERICS.optional_float(
                row,
                OPENHCS_SECONDS_FIELD,
            )
            category = _category_from_summary_row(pipeline_name, row or baseline_row)
            yield BenchmarkMetricRow(
                pipeline_name=pipeline_name,
                method=source.label,
                assay_category=category.assay,
                module_category=category.module,
                accuracy_fraction=SUMMARY_ROW_NUMERICS.optional_float(
                    row,
                    ACCURACY_FIELD,
                ),
                raw_seconds=openhcs_seconds,
                speedup=SUMMARY_ROW_NUMERICS.speedup(
                    row,
                    native_seconds,
                    openhcs_seconds,
                ),
                peak_memory_mb=SUMMARY_ROW_NUMERICS.optional_float(
                    row,
                    OPENHCS_MEMORY_FIELD,
                ),
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


def _category_from_summary_row(
    pipeline_name: str,
    row: SummaryRow | None,
) -> BenchmarkCategory:
    """Read persisted case metadata, falling back for old pre-category summaries."""
    if row is not None:
        assay_category = row.get(ASSAY_CATEGORY_FIELD)
        module_category = row.get(MODULE_CATEGORY_FIELD)
        if assay_category or module_category:
            return BenchmarkCategory(
                assay=assay_category or DEFAULT_BENCHMARK_CATEGORY.assay,
                module=module_category or DEFAULT_BENCHMARK_CATEGORY.module,
            )
    return official_cp3_case_category(pipeline_name)


def _average_rows(rows: Iterable[BenchmarkMetricRow]) -> Iterable[BenchmarkMetricRow]:
    by_method: dict[str, list[BenchmarkMetricRow]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)
    for method, method_rows in by_method.items():
        yield BenchmarkMetricRow(
            pipeline_name="Average",
            method=method,
            assay_category=AGGREGATE_LABEL,
            module_category=AGGREGATE_LABEL,
            accuracy_fraction=_mean_present(
                row.accuracy_fraction for row in method_rows
            ),
            raw_seconds=_mean_present(row.raw_seconds for row in method_rows),
            speedup=_mean_present(row.speedup for row in method_rows),
            peak_memory_mb=_mean_present(row.peak_memory_mb for row in method_rows),
        )


def _write_metric_rows(path: Path, rows: Sequence[BenchmarkMetricRow]) -> None:
    fieldnames = (
        PIPELINE_NAME_FIELD,
        METHOD_FIELD,
        ASSAY_CATEGORY_FIELD,
        MODULE_CATEGORY_FIELD,
        ACCURACY_FRACTION_FIELD,
        RAW_SECONDS_FIELD,
        SPEEDUP_METRIC_KEY,
        PEAK_MEMORY_MB_FIELD,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    PIPELINE_NAME_FIELD: row.pipeline_name,
                    METHOD_FIELD: row.method,
                    ASSAY_CATEGORY_FIELD: row.assay_category,
                    MODULE_CATEGORY_FIELD: row.module_category,
                    ACCURACY_FRACTION_FIELD: row.accuracy_fraction,
                    RAW_SECONDS_FIELD: row.raw_seconds,
                    SPEEDUP_METRIC_KEY: row.speedup,
                    PEAK_MEMORY_MB_FIELD: row.peak_memory_mb,
                }
            )


def _category_metric_rows(
    rows: Sequence[BenchmarkMetricRow],
    *,
    category_key: str,
) -> Iterable[BenchmarkMetricRow]:
    grouped: dict[tuple[str, str], list[BenchmarkMetricRow]] = {}
    for row in rows:
        if row.pipeline_name == "Average":
            continue
        category_name = str(getattr(row, category_key))
        grouped.setdefault((category_name, row.method), []).append(row)

    for (category_name, method), category_rows in grouped.items():
        yield BenchmarkMetricRow(
            pipeline_name=category_name,
            method=method,
            assay_category=(
                category_name
                if category_key == ASSAY_CATEGORY_FIELD
                else AGGREGATE_LABEL
            ),
            module_category=(
                category_name
                if category_key == MODULE_CATEGORY_FIELD
                else AGGREGATE_LABEL
            ),
            accuracy_fraction=_mean_present(
                row.accuracy_fraction for row in category_rows
            ),
            raw_seconds=_mean_present(row.raw_seconds for row in category_rows),
            speedup=_mean_present(row.speedup for row in category_rows),
            peak_memory_mb=_mean_present(row.peak_memory_mb for row in category_rows),
        )


def _category_metrics(category_key: str) -> tuple[FigureMetricSpec, ...]:
    prefix = (
        "cppipe_assay_category"
        if category_key == ASSAY_CATEGORY_FIELD
        else "cppipe_module_category"
    )
    label = (
        "assay category" if category_key == ASSAY_CATEGORY_FIELD else "module category"
    )
    return tuple(
        FigureMetricSpec(
            key=metric.key,
            filename_stem=f"{prefix}_{metric.key}",
            title=f"{metric.title} by {label}",
            ylabel=metric.ylabel,
            percentage=metric.percentage,
            baseline_line=metric.baseline_line,
            target_line=metric.target_line,
            minimum_ylim=metric.minimum_ylim,
            log_variant=metric.log_variant,
            use_axis_break=metric.use_axis_break,
        )
        for metric in FIGURE_METRICS
    )


def _category_order(rows: Sequence[BenchmarkMetricRow]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(row.pipeline_name for row in rows))


def _plot_grouped_metric(
    request: GroupedFigureRequest,
    *,
    metric: FigureMetricSpec,
    log_y: bool,
) -> tuple[Path, ...]:
    broken_range = (
        LINEAR_AXIS_BREAK_POLICY.range_for(_grouped_metric_values(request, metric))
        if metric.use_axis_break and not log_y and not metric.percentage
        else None
    )
    if broken_range is not None:
        return _plot_grouped_metric_broken(
            request,
            metric=metric,
            broken_range=broken_range,
        )
    panels = PIPELINE_LABEL_LAYOUT.panels(request.pipeline_names, request.wrap_after)
    fig_width = max(
        8.0,
        max(len(panel) for panel in panels) * request.group_width_inches,
    )
    fig_height = (
        SINGLE_PANEL_HEIGHT_INCHES
        if len(panels) == 1
        else MULTI_PANEL_HEIGHT_INCHES
    )
    with FIGURE_STYLE.context():
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(fig_width, fig_height),
            layout="constrained",
        )
        panel_axes = (axes,) if len(panels) == 1 else tuple(axes)
        width = _bar_width(len(request.methods))
        offsets = _bar_offsets(len(request.methods), width)
        row_index = {(row.pipeline_name, row.method): row for row in request.rows}

        for panel_index, (axis, panel_names) in enumerate(
            zip(panel_axes, panels, strict=True)
        ):
            x_positions = tuple(range(len(panel_names)))
            for method_index, method in enumerate(request.methods):
                values = [
                    METRIC_PROJECTION.plot_value(
                        METRIC_PROJECTION.value(
                            row_index.get((pipeline_name, method)),
                            metric,
                        )
                    )
                    for pipeline_name in panel_names
                ]
                axis.bar(
                    [x + offsets[method_index] for x in x_positions],
                    values,
                    width=width,
                    label=method if panel_index == 0 else None,
                    color=FIGURE_STYLE.color_for_method(method_index),
                    edgecolor=FIGURE_STYLE.background,
                    linewidth=0.55,
                )

            _draw_reference_lines(axis, metric=metric, log_y=log_y)
            FIGURE_STYLE.decorate_axis(axis, metric=metric, panel_index=panel_index)
            axis.set_ylabel(metric.ylabel)
            axis.set_xticks(list(x_positions))
            axis.set_xticklabels(
                [PIPELINE_LABEL_LAYOUT.split_label(name) for name in panel_names],
                rotation=42,
                ha="right",
                fontsize=PIPELINE_LABEL_FONT_SIZE,
            )
            axis.margins(x=0.01)
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

        panel_axes[0].legend(
            frameon=False,
            ncol=min(len(request.methods), 5),
            loc="upper left",
        )
        outputs: list[Path] = []
        filename_stem = f"{metric.filename_stem}_log" if log_y else metric.filename_stem
        for output_format in request.output_formats:
            output_path = request.output_dir / f"{filename_stem}.{output_format}"
            FIGURE_STYLE.save(fig, output_path)
            outputs.append(output_path)
        plt.close(fig)
        return tuple(outputs)


def _plot_grouped_metric_broken(
    request: GroupedFigureRequest,
    *,
    metric: FigureMetricSpec,
    broken_range: tuple[float, float, float],
) -> tuple[Path, ...]:
    del broken_range
    panels = PIPELINE_LABEL_LAYOUT.panels(request.pipeline_names, request.wrap_after)
    row_index = {(row.pipeline_name, row.method): row for row in request.rows}
    panel_breaks = tuple(
        LINEAR_AXIS_BREAK_POLICY.range_for(
            tuple(
                value
                for pipeline_name in panel
                for method in request.methods
                if (
                    value := METRIC_PROJECTION.value(
                        row_index.get((pipeline_name, method)),
                        metric,
                    )
                )
                is not None
            )
        )
        for panel in panels
    )
    panel_axis_counts = tuple(2 if panel_break is not None else 1 for panel_break in panel_breaks)
    height_ratios = tuple(
        ratio
        for panel_break in panel_breaks
        for ratio in ((1.0, 3.2) if panel_break is not None else (3.2,))
    )
    fig_width = max(
        8.0,
        max(len(panel) for panel in panels) * request.group_width_inches,
    )
    with FIGURE_STYLE.context():
        fig, axes = plt.subplots(
            sum(panel_axis_counts),
            1,
            figsize=(
                fig_width,
                sum(5.7 if panel_break is not None else 4.8 for panel_break in panel_breaks)
                + 1.5,
            ),
            gridspec_kw={"height_ratios": height_ratios},
            sharex=False,
            layout="constrained",
        )
        all_axes = (axes,) if sum(panel_axis_counts) == 1 else tuple(axes.flat)
        width = _bar_width(len(request.methods))
        offsets = _bar_offsets(len(request.methods), width)

        axis_offset = 0
        for panel_index, panel_names in enumerate(panels):
            panel_break = panel_breaks[panel_index]
            if panel_break is None:
                panel_axes = (all_axes[axis_offset],)
                axis_offset += 1
                label_axis = panel_axes[0]
            else:
                top_axis = all_axes[axis_offset]
                bottom_axis = all_axes[axis_offset + 1]
                axis_offset += 2
                panel_axes = (top_axis, bottom_axis)
                label_axis = bottom_axis
            x_positions = tuple(range(len(panel_names)))
            for axis in panel_axes:
                for method_index, method in enumerate(request.methods):
                    values = [
                        METRIC_PROJECTION.plot_value(
                            METRIC_PROJECTION.value(
                                row_index.get((pipeline_name, method)),
                                metric,
                            )
                        )
                        for pipeline_name in panel_names
                    ]
                    axis.bar(
                        [x + offsets[method_index] for x in x_positions],
                        values,
                        width=width,
                        label=(
                            method
                            if panel_index == 0 and axis is panel_axes[0]
                            else None
                        ),
                        color=FIGURE_STYLE.color_for_method(method_index),
                        edgecolor=FIGURE_STYLE.background,
                        linewidth=0.55,
                    )
                if panel_break is None or axis is panel_axes[-1]:
                    _draw_reference_lines(axis, metric=metric, log_y=False)
                FIGURE_STYLE.decorate_axis(
                    axis,
                    metric=metric,
                    panel_index=panel_index if axis is panel_axes[0] else -1,
                )
                axis.set_ylabel(metric.ylabel)
                axis.set_xticks(list(x_positions))
                axis.margins(x=0.01)

            if panel_break is not None:
                top_axis, bottom_axis = panel_axes
                top_axis.set_ylim(panel_break[1], panel_break[2])
                bottom_axis.set_ylim(
                    metric.minimum_ylim if metric.minimum_ylim is not None else 0.0,
                    panel_break[0],
                )
                top_axis.set_xticklabels(())
                LINEAR_AXIS_BREAK_POLICY.mark(top_axis, bottom_axis)
            elif metric.minimum_ylim is not None:
                label_axis.set_ylim(bottom=metric.minimum_ylim)
            label_axis.set_xticklabels(
                [PIPELINE_LABEL_LAYOUT.split_label(name) for name in panel_names],
                rotation=42,
                ha="right",
                fontsize=PIPELINE_LABEL_FONT_SIZE,
            )

        all_axes[0].legend(
            frameon=False,
            ncol=min(len(request.methods), 5),
            loc="upper left",
        )
        outputs: list[Path] = []
        for output_format in request.output_formats:
            output_path = request.output_dir / f"{metric.filename_stem}.{output_format}"
            FIGURE_STYLE.save(fig, output_path)
            outputs.append(output_path)
        plt.close(fig)
        return tuple(outputs)


def _plot_accuracy_zoom(
    request: GroupedFigureRequest,
) -> tuple[Path, ...]:
    """Plot accuracy with a broken y-axis so tiny parity drift is visible."""
    panels = PIPELINE_LABEL_LAYOUT.panels(request.pipeline_names, request.wrap_after)
    panel_count = len(panels)
    fig_width = max(
        8.0,
        max(len(panel) for panel in panels) * request.group_width_inches,
    )
    with FIGURE_STYLE.context():
        fig, axes = plt.subplots(
            panel_count * 2,
            1,
            figsize=(fig_width, ACCURACY_ZOOM_PANEL_HEIGHT_INCHES * panel_count),
            gridspec_kw={"height_ratios": tuple((2.7, 1.0) * panel_count)},
            layout="constrained",
        )
        all_axes = tuple(axes.flat)
        width = _bar_width(len(request.methods))
        offsets = _bar_offsets(len(request.methods), width)
        row_index = {(row.pipeline_name, row.method): row for row in request.rows}
        metric = FIGURE_METRICS[0]

        for panel_index, panel_names in enumerate(panels):
            zoom_axis = all_axes[panel_index * 2]
            context_axis = all_axes[panel_index * 2 + 1]
            x_positions = tuple(range(len(panel_names)))
            for axis in (zoom_axis, context_axis):
                for method_index, method in enumerate(request.methods):
                    values = [
                        METRIC_PROJECTION.plot_value(
                            METRIC_PROJECTION.value(
                                row_index.get((pipeline_name, method)),
                                metric,
                            )
                        )
                        for pipeline_name in panel_names
                    ]
                    axis.bar(
                        [x + offsets[method_index] for x in x_positions],
                        values,
                        width=width,
                        label=(
                            method if panel_index == 0 and axis is zoom_axis else None
                        ),
                        color=FIGURE_STYLE.color_for_method(method_index),
                        edgecolor=FIGURE_STYLE.background,
                        linewidth=0.55,
                    )
                FIGURE_STYLE.decorate_axis(axis, metric=metric, panel_index=panel_index)
                axis.set_xticks(list(x_positions))

            zoom_axis.set_ylim(
                100.0 - ACCURACY_ZOOM_HALF_RANGE_PERCENT,
                100.0 + ACCURACY_ZOOM_HALF_RANGE_PERCENT,
            )
            zoom_axis.axhline(
                100.0,
                color=FIGURE_STYLE.target_color,
                linewidth=1.15,
                linestyle="--",
                alpha=0.85,
            )
            zoom_axis.yaxis.set_major_formatter(FuncFormatter(_percent_tick_label))
            zoom_axis.yaxis.get_offset_text().set_visible(False)
            zoom_axis.set_ylabel("Accuracy (%)")
            zoom_axis.set_xticklabels(())
            context_axis.set_ylim(0.0, 5.0)
            context_axis.set_ylabel("0-5%")
            context_axis.set_xticklabels(
                [PIPELINE_LABEL_LAYOUT.split_label(name) for name in panel_names],
                rotation=42,
                ha="right",
                fontsize=PIPELINE_LABEL_FONT_SIZE,
            )
            zoom_axis.margins(x=0.01)
            context_axis.margins(x=0.01)
            LINEAR_AXIS_BREAK_POLICY.mark(zoom_axis, context_axis)

        all_axes[0].legend(
            frameon=False,
            ncol=min(len(request.methods), 5),
            loc="upper left",
        )
        outputs: list[Path] = []
        for output_format in request.output_formats:
            output_path = request.output_dir / f"cppipe_accuracy_zoom.{output_format}"
            FIGURE_STYLE.save(fig, output_path)
            outputs.append(output_path)
        plt.close(fig)
        return tuple(outputs)


def _plot_average_speedup_points(
    source_tables: SummaryTables,
    *,
    summary_sources: Sequence[SummarySource],
    pipeline_names: Sequence[str],
    output_dir: Path,
    output_formats: Sequence[str],
) -> tuple[Path, ...]:
    """Plot mean OpenHCS speedup with one point per dataset."""
    series = tuple(
        _speedup_point_series(
            table,
            source=source,
            pipeline_names=pipeline_names,
        )
        for source, table in zip(summary_sources, source_tables, strict=True)
    )
    series = tuple(item for item in series if item.points)
    if not series:
        return ()

    csv_path = output_dir / "cppipe_average_speedup_points.csv"
    _write_average_speedup_points_csv(csv_path, series)

    with FIGURE_STYLE.context():
        fig_width = max(5.2, 1.45 * len(series) + 3.2)
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(fig_width, 4.6),
            layout="constrained",
        )
        x_positions = tuple(range(len(series)))
        for index, speedup_series in enumerate(series):
            color = FIGURE_STYLE.color_for_method(index + 1)
            point_x = [
                index + _deterministic_jitter(point_index, len(speedup_series.points))
                for point_index, _point in enumerate(speedup_series.points)
            ]
            point_y = [point.speedup for point in speedup_series.points]
            axis.scatter(
                point_x,
                point_y,
                s=28,
                color=color,
                alpha=0.76,
                edgecolors=FIGURE_STYLE.background,
                linewidths=0.55,
                zorder=3,
            )
            axis.errorbar(
                [index],
                [speedup_series.mean],
                yerr=[[speedup_series.ci95], [speedup_series.ci95]],
                fmt="o",
                color=FIGURE_STYLE.text_color,
                markerfacecolor=color,
                markeredgecolor=FIGURE_STYLE.text_color,
                markersize=8.5,
                capsize=7,
                elinewidth=1.4,
                zorder=4,
                label=f"{speedup_series.label} mean",
            )
        axis.axhline(
            SPEEDUP_TARGET,
            color=FIGURE_STYLE.target_color,
            linewidth=1.15,
            linestyle="--",
            alpha=0.86,
        )
        axis.annotate(
            "4x target",
            xy=(0.995, SPEEDUP_TARGET),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.8,
            color=FIGURE_STYLE.target_color,
        )
        axis.set_title("Average execution speedup", loc="left", pad=10)
        axis.set_ylabel("Speedup versus native CellProfiler (x)")
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels([item.label for item in series])
        axis.set_xlim(-0.6, len(series) - 0.4)
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", color=FIGURE_STYLE.grid_color, linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(FIGURE_STYLE.spine_color)
        axis.spines["bottom"].set_color(FIGURE_STYLE.spine_color)
        axis.legend(frameon=False, loc="upper left")

        outputs: list[Path] = [csv_path]
        for output_format in output_formats:
            output_path = output_dir / f"cppipe_average_speedup_points.{output_format}"
            FIGURE_STYLE.save(fig, output_path)
            outputs.append(output_path)
        plt.close(fig)
        return tuple(outputs)


@dataclass(frozen=True)
class SpeedupPoint:
    """One dataset speedup point for aggregate plotting."""

    pipeline_name: str
    speedup: float


@dataclass(frozen=True)
class SpeedupPointSeries:
    """Per-method speedup distribution and aggregate interval."""

    label: str
    points: tuple[SpeedupPoint, ...]
    mean: float
    standard_deviation: float
    ci95: float


@dataclass(frozen=True)
class SpeedupDistributionSeries:
    """One labeled speedup distribution for report tables and CDF plots."""

    label: str
    values: tuple[float, ...]

    @classmethod
    def from_points(cls, series: SpeedupPointSeries) -> "SpeedupDistributionSeries":
        """Build a distribution series from per-pipeline speedup points."""
        return cls(series.label, tuple(point.speedup for point in series.points))


@dataclass(frozen=True)
class SpeedupSummaryStatistics:
    """Summary statistics for one speedup distribution."""

    label: str
    sample_count: int
    minimum: float
    maximum: float
    median: float
    mean: float
    standard_deviation: float

    @classmethod
    def from_series(
        cls,
        series: SpeedupDistributionSeries,
    ) -> "SpeedupSummaryStatistics | None":
        """Calculate min/max/median/mean statistics for one distribution."""
        values = tuple(
            value for value in series.values if math.isfinite(value) and value > 0.0
        )
        if not values:
            return None
        return cls(
            label=series.label,
            sample_count=len(values),
            minimum=min(values),
            maximum=max(values),
            median=statistics.median(values),
            mean=sum(values) / len(values),
            standard_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
        )


@dataclass(frozen=True)
class SpeedupDistributionReport:
    """Owns speedup summary tables and cumulative distribution figures."""

    series: tuple[SpeedupDistributionSeries, ...]
    output_dir: Path
    filename_prefix: str
    title: str
    xlabel: str
    output_formats: tuple[str, ...] = DEFAULT_FORMATS

    def outputs(self) -> tuple[Path, ...]:
        """Write all speedup distribution report artifacts."""
        if not self.series:
            return ()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = self.output_dir / f"{self.filename_prefix}_summary_statistics.csv"
        summary_markdown = (
            self.output_dir / f"{self.filename_prefix}_summary_statistics.md"
        )
        cdf_csv = self.output_dir / f"{self.filename_prefix}_cumulative_distribution.csv"
        self.write_summary_statistics(summary_csv)
        self.write_summary_markdown(summary_markdown)
        self.write_cdf_csv(cdf_csv)
        outputs: list[Path] = [summary_csv, summary_markdown, cdf_csv]
        outputs.extend(self.plot_cdf(log_x=False))
        outputs.extend(self.plot_cdf(log_x=True))
        return tuple(outputs)

    def write_summary_statistics(self, path: Path) -> None:
        """Write machine-readable speedup summary statistics."""
        fieldnames = (
            "label",
            "sample_count",
            "min_speedup",
            "max_speedup",
            "median_speedup",
            "mean_speedup",
            "standard_deviation",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.summary_statistics:
                writer.writerow(
                    {
                        "label": row.label,
                        "sample_count": row.sample_count,
                        "min_speedup": row.minimum,
                        "max_speedup": row.maximum,
                        "median_speedup": row.median,
                        "mean_speedup": row.mean,
                        "standard_deviation": row.standard_deviation,
                    }
                )

    def write_summary_markdown(self, path: Path) -> None:
        """Write human-readable speedup summary statistics."""
        lines = [
            "| Series | n | Min speedup | Median speedup | Mean speedup | Max speedup | SD |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        lines.extend(
            "| "
            f"{row.label} | {row.sample_count} | {row.minimum:.3f} | "
            f"{row.median:.3f} | {row.mean:.3f} | {row.maximum:.3f} | "
            f"{row.standard_deviation:.3f} |"
            for row in self.summary_statistics
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_cdf_csv(self, path: Path) -> None:
        """Write empirical percent-at-or-above-threshold speedup data."""
        fieldnames = (
            "label",
            "speedup_threshold",
            "percent_at_or_above",
            "count_at_or_above",
            "sample_count",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in self.series:
                sample_count = len(item.values)
                for threshold in self.thresholds(item.values):
                    count = sum(1 for value in item.values if value >= threshold)
                    writer.writerow(
                        {
                            "label": item.label,
                            "speedup_threshold": threshold,
                            "percent_at_or_above": 100.0 * count / sample_count,
                            "count_at_or_above": count,
                            "sample_count": sample_count,
                        }
                    )

    def plot_cdf(self, *, log_x: bool) -> tuple[Path, ...]:
        """Plot empirical percent-at-or-above-threshold speedup curves."""
        with FIGURE_STYLE.context():
            fig, axis = plt.subplots(
                1,
                1,
                figsize=(7.4, 4.4),
                layout="constrained",
            )
            for index, item in enumerate(self.series):
                thresholds = self.thresholds(item.values)
                y_values = tuple(
                    100.0
                    * sum(1 for value in item.values if value >= threshold)
                    / len(item.values)
                    for threshold in thresholds
                )
                axis.step(
                    thresholds,
                    y_values,
                    where="post",
                    linewidth=2.0,
                    color=FIGURE_STYLE.color_for_method(index + 1),
                    label=item.label,
                )
            axis.axvline(
                SPEEDUP_TARGET,
                color=FIGURE_STYLE.target_color,
                linewidth=1.15,
                linestyle="--",
                alpha=0.86,
            )
            axis.annotate(
                "4x target",
                xy=(SPEEDUP_TARGET, 99.0),
                xycoords=("data", "data"),
                xytext=(3, -2),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=7.8,
                color=FIGURE_STYLE.target_color,
            )
            if log_x:
                axis.set_xscale("log")
                axis.xaxis.set_major_formatter(FuncFormatter(_plain_log_tick_label))
                axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_ylim(0.0, 102.0)
            axis.set_xlabel(self.xlabel)
            axis.set_ylabel("Datasets at or above threshold (%)")
            axis.set_title(
                f"{self.title} (log scale)" if log_x else self.title,
                loc="left",
                pad=10,
            )
            axis.grid(
                axis="both",
                color=FIGURE_STYLE.grid_color,
                linewidth=0.8,
                alpha=0.8,
            )
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(FIGURE_STYLE.spine_color)
            axis.spines["bottom"].set_color(FIGURE_STYLE.spine_color)
            axis.legend(frameon=False, loc="upper right")
            outputs: list[Path] = []
            suffix = (
                "_cumulative_distribution_log"
                if log_x
                else "_cumulative_distribution"
            )
            for output_format in self.output_formats:
                output_path = self.output_dir / f"{self.filename_prefix}{suffix}.{output_format}"
                FIGURE_STYLE.save(fig, output_path)
                outputs.append(output_path)
            plt.close(fig)
            return tuple(outputs)

    @property
    def summary_statistics(self) -> tuple[SpeedupSummaryStatistics, ...]:
        """Return summary rows for every distribution series."""
        return tuple(
            stats
            for item in self.series
            if (stats := SpeedupSummaryStatistics.from_series(item)) is not None
        )

    @staticmethod
    def thresholds(values: Sequence[float]) -> tuple[float, ...]:
        """Return empirical speedup thresholds for CDF/survival reporting."""
        if not values:
            return ()
        return tuple(
            sorted(
                {
                    min(values),
                    max(values),
                    1.0,
                    SPEEDUP_TARGET,
                    *values,
                }
            )
        )


def generate_speedup_distribution_artifacts(
    series: Sequence[SpeedupDistributionSeries],
    *,
    output_dir: Path,
    filename_prefix: str,
    title: str,
    xlabel: str,
    output_formats: Sequence[str] = DEFAULT_FORMATS,
) -> tuple[Path, ...]:
    """Generate speedup summary tables and cumulative distribution figures."""
    clean_series = tuple(
        SpeedupDistributionSeries(
            item.label,
            tuple(value for value in item.values if math.isfinite(value) and value > 0.0),
        )
        for item in series
    )
    clean_series = tuple(item for item in clean_series if item.values)
    if not clean_series:
        return ()
    return SpeedupDistributionReport(
        series=clean_series,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        title=title,
        xlabel=xlabel,
        output_formats=tuple(output_formats),
    ).outputs()


def _speedup_point_series(
    table: SummaryTable,
    *,
    source: SummarySource,
    pipeline_names: Sequence[str],
) -> SpeedupPointSeries:
    points = tuple(
        SpeedupPoint(pipeline_name, speedup)
        for pipeline_name in pipeline_names
        if (
            speedup := SUMMARY_ROW_NUMERICS.speedup_from_summary_row(
                table.get(pipeline_name)
            )
        )
        is not None
    )
    values = tuple(point.speedup for point in points)
    mean = _mean_present(values) or math.nan
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * standard_deviation / math.sqrt(len(values)) if values else math.nan
    return SpeedupPointSeries(
        label=source.label,
        points=points,
        mean=mean,
        standard_deviation=standard_deviation,
        ci95=ci95,
    )


def _write_average_speedup_points_csv(
    path: Path,
    series: Sequence[SpeedupPointSeries],
) -> None:
    fieldnames = (
        "method",
        PIPELINE_NAME_FIELD,
        SPEEDUP_METRIC_KEY,
        "mean_speedup",
        "standard_deviation",
        "ci95",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in series:
            for point in item.points:
                writer.writerow(
                    {
                        "method": item.label,
                        PIPELINE_NAME_FIELD: point.pipeline_name,
                        SPEEDUP_METRIC_KEY: point.speedup,
                        "mean_speedup": item.mean,
                        "standard_deviation": item.standard_deviation,
                        "ci95": item.ci95,
                    }
                )


def _deterministic_jitter(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    spread = 0.18
    return ((index / (count - 1)) - 0.5) * spread


def _grouped_metric_values(
    request: GroupedFigureRequest,
    metric: FigureMetricSpec,
) -> tuple[float, ...]:
    return tuple(
        value
        for row in request.rows
        if (value := METRIC_PROJECTION.value(row, metric)) is not None
        and math.isfinite(value)
        and value > 0.0
    )


def _draw_reference_lines(axis, *, metric: FigureMetricSpec, log_y: bool) -> None:
    if metric.baseline_line is not None:
        axis.axhline(
            metric.baseline_line,
            color=FIGURE_STYLE.baseline_color,
            linewidth=1.0,
            alpha=0.72,
        )
    if metric.target_line is None:
        return
    if log_y and metric.target_line <= 0.0:
        return
    axis.axhline(
        metric.target_line,
        color=FIGURE_STYLE.target_color,
        linewidth=1.15,
        linestyle="--",
        alpha=0.86,
    )
    label = "4x target" if metric.key == SPEEDUP_METRIC_KEY else "target"
    axis.annotate(
        label,
        xy=(0.995, metric.target_line),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=FIGURE_STYLE.target_color,
    )

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


def _bar_width(method_count: int) -> float:
    return min(GROUPED_BAR_MAX_WIDTH, GROUPED_BAR_FRACTION / max(method_count, 1))


def _mean_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _write_benchmark_figure_index(path: Path, outputs: Sequence[Path]) -> None:
    figure_names = {output.name for output in outputs}
    lines = [
        "# CellProfiler Benchmark Figure Index",
        "",
        "Autogenerated benchmark figures for the OpenHCS paper draft.",
        "",
        "## Manuscript Benchmark Panels",
        "",
        "- `cppipe_accuracy.*`: parity across imported `.cppipe` workflows.",
        "- `cppipe_accuracy_zoom.*`: broken-axis parity view for tiny numeric drift near 100%.",
        "- `cppipe_raw_seconds.*`: single-thread execution runtime in seconds.",
        "- `cppipe_raw_seconds_log.*`: runtime on a log scale for mixed short and long pipelines.",
        "- `cppipe_speedup.*`: execution speedup versus native CellProfiler with the 4x target line.",
        "- `cppipe_speedup_log.*`: speedup on a log scale for wide dynamic range.",
        "- `cppipe_speedup_summary_statistics.*`: min, max, median, and mean speedup tables.",
        "- `cppipe_speedup_cumulative_distribution.*`: percent of datasets at or above each speedup threshold.",
        "- `cppipe_speedup_cumulative_distribution_log.*`: cumulative speedup distribution with a log x-axis.",
        "- `cppipe_average_speedup_points.*`: aggregate speedup mean with per-dataset points and a 95% confidence interval.",
        "- `cppipe_average_speedup_points.csv`: per-dataset speedups and aggregate statistics used by the point/error chart.",
        "- `cppipe_peak_memory*`: peak RSS figures when memory metrics are present.",
        "- `cppipe_assay_category_*`: manifest-declared assay category summaries.",
        "- `cppipe_module_category_*`: manifest-declared module category summaries.",
        "- `cppipe_comparison_metrics_long.csv`: long-form per-pipeline plotting table.",
        "- `cppipe_comparison_category_metrics_long.csv`: long-form category plotting table.",
        "",
        "## Files Present",
        "",
    ]
    lines.extend(f"- `{name}`" for name in sorted(figure_names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
