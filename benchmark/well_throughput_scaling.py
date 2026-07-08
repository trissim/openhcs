"""Well-level OpenHCS throughput scaling for converted cppipe pipelines."""

from __future__ import annotations

import csv
import math
import multiprocessing
import queue
import statistics
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import psutil

from benchmark.cellprofiler_comparison import CASE_NAME_FIELD
from benchmark.cellprofiler_comparison import load_comparison_cases
from benchmark.cellprofiler_comparison import MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD
from benchmark.contracts.comparison_manifest import ComparisonManifest
from benchmark.metrics.memory import MemoryMetric
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.core.config import (
    AnalysisConsolidationConfig,
    GlobalPipelineConfig,
    LazyWellFilterConfig,
    LazyPathPlanningConfig,
    MaterializationBackend,
    MultiprocessingStartMethod,
    PipelineConfig,
    VFSConfig,
)
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.orchestrator.execution_result import RuntimeObservationMode
from openhcs.core.progress import set_progress_queue
from openhcs.core.source_schema_workspace import (
    expand_source_schema_workspace_wells,
)
from openhcs.interop.cellprofiler.source_schema_ingestion import (
    CellProfilerSourceSchemaWorkspaceRequest,
    prepare_cellprofiler_source_schema_workspace,
)


WELL_THROUGHPUT_ROWS_CSV = "well_throughput.csv"
WELL_THROUGHPUT_EVENTS_CSV = "well_throughput_progress_events.csv"
WELL_THROUGHPUT_LANES_CSV = "well_throughput_worker_lanes.csv"
WELL_THROUGHPUT_STEPS_CSV = "well_throughput_step_timings.csv"


@dataclass(frozen=True, slots=True)
class WellThroughputMode:
    """One native OpenHCS multiprocessing throughput mode."""

    name: str
    well_count: int
    worker_count: int
    use_threading: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "well_count", int(self.well_count))
        object.__setattr__(self, "worker_count", int(self.worker_count))
        object.__setattr__(self, "use_threading", bool(self.use_threading))
        if self.well_count < 1:
            raise ValueError("well_count must be positive.")
        if self.worker_count < 1:
            raise ValueError("worker_count must be positive.")


@dataclass(frozen=True, slots=True)
class WellThroughputObservationKey:
    """Unique identity for one case/mode throughput observation."""

    case_name: str
    mode_name: str


class WellThroughputPreset(StrEnum):
    """Named preliminary well/core scaling modes."""

    WELL_1_THREAD_1 = "1w_1t"
    WELLS_8_WORKERS_2 = "8w_2c"
    WELLS_12_WORKERS_3 = "12w_3c"
    WELLS_16_WORKERS_4 = "16w_4c"

    def mode(self) -> WellThroughputMode:
        """Return the native OpenHCS mode represented by this preset."""
        return WELL_THROUGHPUT_PRESET_MODES[self]


WELL_THROUGHPUT_PRESET_MODES: Mapping[WellThroughputPreset, WellThroughputMode] = {
    WellThroughputPreset.WELL_1_THREAD_1: WellThroughputMode(
        WellThroughputPreset.WELL_1_THREAD_1.value,
        well_count=1,
        worker_count=1,
        use_threading=True,
    ),
    WellThroughputPreset.WELLS_8_WORKERS_2: WellThroughputMode(
        WellThroughputPreset.WELLS_8_WORKERS_2.value,
        well_count=8,
        worker_count=2,
    ),
    WellThroughputPreset.WELLS_12_WORKERS_3: WellThroughputMode(
        WellThroughputPreset.WELLS_12_WORKERS_3.value,
        well_count=12,
        worker_count=3,
    ),
    WellThroughputPreset.WELLS_16_WORKERS_4: WellThroughputMode(
        WellThroughputPreset.WELLS_16_WORKERS_4.value,
        well_count=16,
        worker_count=4,
    ),
}
DEFAULT_WELL_THROUGHPUT_PRESETS: tuple[WellThroughputPreset, ...] = (
    WellThroughputPreset.WELL_1_THREAD_1,
    WellThroughputPreset.WELLS_8_WORKERS_2,
    WellThroughputPreset.WELLS_12_WORKERS_3,
    WellThroughputPreset.WELLS_16_WORKERS_4,
)


@dataclass(frozen=True, slots=True)
class WellThroughputModeOrder:
    """Publication order for named well-throughput modes."""

    names: tuple[str, ...]

    @classmethod
    def from_presets(
        cls,
        presets: Sequence[WellThroughputPreset],
    ) -> "WellThroughputModeOrder":
        """Build the display order from named throughput presets."""
        return cls(tuple(preset.value for preset in presets))

    def order(self, mode_names: Iterable[str]) -> tuple[str, ...]:
        """Return known modes in preset order and custom modes after them."""
        unique_names = tuple(dict.fromkeys(mode_names))
        known = tuple(name for name in self.names if name in unique_names)
        custom = tuple(name for name in unique_names if name not in self.names)
        return (*known, *custom)


WELL_THROUGHPUT_MODE_ORDER = WellThroughputModeOrder.from_presets(
    DEFAULT_WELL_THROUGHPUT_PRESETS
)


@dataclass(frozen=True, slots=True)
class WellThroughputBenchmarkPlan:
    """Authoritative set of native OpenHCS throughput modes to run."""

    modes: tuple[WellThroughputMode, ...]

    def __post_init__(self) -> None:
        modes = tuple(self.modes)
        if not modes:
            raise ValueError("Well throughput benchmark plan requires at least one mode.")
        object.__setattr__(self, "modes", modes)

    @classmethod
    def from_axes(
        cls,
        *,
        well_counts: Sequence[int],
        worker_counts: Sequence[int],
    ) -> "WellThroughputBenchmarkPlan":
        """Build the legacy cross-product plan from independent axes."""
        modes = tuple(
            WellThroughputMode(
                f"{well_count}w_{worker_count}c",
                well_count=well_count,
                worker_count=worker_count,
            )
            for well_count in tuple(sorted(set(int(value) for value in well_counts)))
            for worker_count in tuple(sorted(set(int(value) for value in worker_counts)))
        )
        return cls(modes)

    @classmethod
    def from_presets(
        cls,
        presets: Sequence[WellThroughputPreset],
    ) -> "WellThroughputBenchmarkPlan":
        """Build a paired-mode plan from named preliminary scaling presets."""
        return cls(tuple(preset.mode() for preset in presets))

    @classmethod
    def from_requested_modes(
        cls,
        *,
        presets: Sequence[WellThroughputPreset] = (),
        well_counts: Sequence[int] = (),
        worker_counts: Sequence[int] = (),
        manifest_path: Path | None = None,
    ) -> "WellThroughputBenchmarkPlan":
        """Resolve the benchmark mode request into one authoritative plan."""
        if presets:
            return cls.from_presets(presets)
        if well_counts or worker_counts:
            if not well_counts or not worker_counts:
                raise ValueError(
                    "Custom well-throughput modes require both well_counts and "
                    "worker_counts."
                )
            return cls.from_axes(
                well_counts=well_counts,
                worker_counts=worker_counts,
            )
        if manifest_path is not None:
            manifest_plan = well_throughput_plan_from_manifest(manifest_path)
            if manifest_plan is not None:
                return manifest_plan
        raise ValueError(
            "Specify presets, both well_counts and worker_counts, or a manifest "
            "with well_throughput_modes."
        )


@dataclass(frozen=True, slots=True)
class WellThroughputPresentationSources:
    """CSV inputs for the publication-oriented throughput figure pack."""

    single_process_summary_csv: Path
    core_scaling_csv: Path
    wells_per_core_csv: Path
    additional_wells_per_core_csvs: tuple[Path, ...] = ()
    module_coverage_semantic_families_csv: Path | None = None


@dataclass(frozen=True, slots=True)
class WellThroughputPresentationMode:
    """One mode shown in the presentation throughput figures."""

    source_mode_name: str
    label: str
    worker_count: int
    wells_per_core: int


class ModuleAbstractionCoverageKind(StrEnum):
    """Presentation coverage state for one absorbed CellProfiler module."""

    EXPLICIT = "explicitly_covered"
    SHARED_ABSTRACTION = "covered_by_shared_abstraction"
    UNCOVERED = "not_covered"

    @property
    def label(self) -> str:
        """Return the publication label for this coverage kind."""
        return {
            ModuleAbstractionCoverageKind.EXPLICIT: "Explicitly covered",
            ModuleAbstractionCoverageKind.SHARED_ABSTRACTION: (
                "Covered by shared abstraction"
            ),
            ModuleAbstractionCoverageKind.UNCOVERED: "Not covered",
        }[self]

    @property
    def sort_key(self) -> int:
        """Return stable presentation order."""
        return {
            ModuleAbstractionCoverageKind.EXPLICIT: 0,
            ModuleAbstractionCoverageKind.SHARED_ABSTRACTION: 1,
            ModuleAbstractionCoverageKind.UNCOVERED: 2,
        }[self]

    @classmethod
    def from_family_coverage(cls, family_coverage: str) -> "ModuleAbstractionCoverageKind":
        """Map compatibility-matrix family coverage to presentation coverage."""
        if family_coverage == "direct_supported":
            return cls.EXPLICIT
        if family_coverage == "semantic_family_supported":
            return cls.SHARED_ABSTRACTION
        if family_coverage == "not_supported":
            return cls.UNCOVERED
        raise ValueError(f"Unsupported module family coverage: {family_coverage!r}")


@dataclass(frozen=True, slots=True)
class ModuleAbstractionCoverageRow:
    """Presentation row for one absorbed CellProfiler module."""

    module_name: str
    coverage: ModuleAbstractionCoverageKind
    abstraction_family: str
    evidence_modules: tuple[str, ...]

    @property
    def evidence_text(self) -> str:
        """Return the semicolon-delimited coverage evidence text."""
        return ";".join(self.evidence_modules)


@dataclass(frozen=True, slots=True)
class ModuleAbstractionCoverageTable:
    """Module coverage projection for presentation figures and tables."""

    rows: tuple[ModuleAbstractionCoverageRow, ...]

    @classmethod
    def from_semantic_family_csv(cls, path: Path) -> "ModuleAbstractionCoverageTable":
        """Build presentation coverage from the compatibility matrix CSV."""
        rows: list[ModuleAbstractionCoverageRow] = []
        with path.open(encoding="utf-8", newline="") as handle:
            for raw_row in csv.DictReader(handle):
                evidence_modules = tuple(
                    module
                    for module in raw_row.get("family_supported_modules", "").split(";")
                    if module
                )
                rows.append(
                    ModuleAbstractionCoverageRow(
                        module_name=raw_row["module_name"],
                        coverage=ModuleAbstractionCoverageKind.from_family_coverage(
                            raw_row["family_coverage"]
                        ),
                        abstraction_family=raw_row["semantic_family"],
                        evidence_modules=evidence_modules,
                    )
                )
        return cls(
            tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        row.coverage.sort_key,
                        row.abstraction_family,
                        row.module_name,
                    ),
                )
            )
        )

    def grouped_rows(
        self,
    ) -> Mapping[ModuleAbstractionCoverageKind, tuple[ModuleAbstractionCoverageRow, ...]]:
        """Return rows grouped by presentation coverage kind."""
        return {
            coverage: tuple(row for row in self.rows if row.coverage is coverage)
            for coverage in ModuleAbstractionCoverageKind
        }


@dataclass(frozen=True, slots=True)
class PresentationAxisBand:
    """One visible y-axis interval in a broken presentation axis."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower < 0.0:
            raise ValueError("Presentation axis bands cannot start below zero.")
        if self.upper <= self.lower:
            raise ValueError("Presentation axis band upper bound must exceed lower bound.")

    def contains(self, value: float) -> bool:
        """Return whether the band displays the given y-value."""
        return self.lower <= value <= self.upper

    def as_ylim(self) -> tuple[float, float]:
        """Return the matplotlib y-limit tuple for this band."""
        return (self.lower, self.upper)


@dataclass(frozen=True, slots=True)
class PresentationAxisBandPolicy:
    """Resolve readable y-axis bands for outlier-heavy presentation figures."""

    max_bands: int = 3
    minimum_absolute_gap: float = 10.0
    minimum_gap_ratio: float = 1.75
    lower_padding: float = 0.92
    upper_padding: float = 1.16

    def bands_for(self, values: Sequence[float | None]) -> tuple[PresentationAxisBand, ...]:
        """Return low-to-high display bands for finite positive values."""
        present = tuple(
            sorted(
                value
                for value in values
                if value is not None and math.isfinite(value) and value > 0.0
            )
        )
        if not present:
            return (PresentationAxisBand(0.0, 1.0),)

        split_indices = self.split_indices_for(present)
        if not split_indices:
            return (PresentationAxisBand(0.0, present[-1] * self.upper_padding),)

        bands: list[PresentationAxisBand] = []
        start = 0
        for split_index in (*split_indices, len(present)):
            segment = present[start:split_index]
            if segment:
                lower = 0.0 if not bands else segment[0] * self.lower_padding
                upper = segment[-1] * self.upper_padding
                bands.append(PresentationAxisBand(lower, upper))
            start = split_index
        return tuple(bands)

    def split_indices_for(self, present: Sequence[float]) -> tuple[int, ...]:
        """Return inter-cluster split points ordered from low to high."""
        candidates = tuple(
            index
            for index in range(1, len(present))
            if present[index] - present[index - 1] >= self.minimum_absolute_gap
            and present[index] / present[index - 1] >= self.minimum_gap_ratio
        )
        return candidates[: max(self.max_bands - 1, 0)]


@dataclass(frozen=True, slots=True)
class WellThroughputPresentationReport:
    """Reusable report generator for the lab-meeting throughput figure pack."""

    sources: WellThroughputPresentationSources
    output_dir: Path
    output_formats: tuple[str, ...] = ("png", "svg")

    core_scaling_modes: tuple[WellThroughputPresentationMode, ...] = (
        WellThroughputPresentationMode("1w_1t", "1 core", 1, 1),
        WellThroughputPresentationMode("8w_2c", "2 cores", 2, 4),
        WellThroughputPresentationMode("12w_3c", "3 cores", 3, 4),
        WellThroughputPresentationMode("16w_4c", "4 cores", 4, 4),
    )
    wells_per_core_counts: tuple[int, ...] = (2, 3, 4)
    multicore_worker_counts: tuple[int, ...] = (2, 3, 4)
    speedup_axis_band_policy: PresentationAxisBandPolicy = PresentationAxisBandPolicy()

    def generate(self) -> tuple[Path, ...]:
        """Generate the complete presentation figure/table pack."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        summary_rows = self.single_process_summary_rows()
        core_rows = self.core_scaling_rows()
        wells_per_core_rows = self.wells_per_core_rows()

        outputs.extend(self.write_source_index())
        outputs.extend(self.generate_parity_figures(summary_rows))
        outputs.extend(self.generate_core_scaling_figures(core_rows, summary_rows))
        outputs.extend(self.generate_wells_per_core_figures(core_rows, wells_per_core_rows))
        outputs.extend(self.generate_module_coverage_figures())
        return tuple(outputs)

    def single_process_summary_rows(self) -> tuple[Mapping[str, str], ...]:
        """Return single-process summary rows in CSV order."""
        return self.read_dict_rows(self.sources.single_process_summary_csv)

    def core_scaling_rows(self) -> tuple[WellThroughputResult, ...]:
        """Return successful rows for the fixed core-scaling comparison."""
        mode_names = {mode.source_mode_name for mode in self.core_scaling_modes}
        return tuple(
            row
            for row in read_well_throughput_csv(self.sources.core_scaling_csv)
            if row.is_successful() and row.mode_name in mode_names
        )

    def wells_per_core_rows(self) -> tuple[WellThroughputResult, ...]:
        """Return successful rows for the variable wells/core sweep."""
        paths = (self.sources.wells_per_core_csv, *self.sources.additional_wells_per_core_csvs)
        return tuple(
            row
            for path in paths
            for row in read_well_throughput_csv(path)
            if row.is_successful()
        )

    def write_source_index(self) -> tuple[Path, ...]:
        """Write a compact manifest for the generated figure pack."""
        path = self.output_dir / "presentation_figure_index.md"
        module_coverage_lines = (
            (
                "- Module coverage: "
                f"`{self.sources.module_coverage_semantic_families_csv}`"
            ),
        ) if self.sources.module_coverage_semantic_families_csv is not None else ()
        module_coverage_figure_lines = (
            ("- `06_module_coverage_by_abstraction.*`",)
            if self.sources.module_coverage_semantic_families_csv is not None
            else ()
        )
        path.write_text(
            "\n".join(
                (
                    "# Official30 Presentation Figure Pack",
                    "",
                    "Source data:",
                    f"- Parity/single-process: `{self.sources.single_process_summary_csv}`",
                    f"- 1/2/3/4 core comparison: `{self.sources.core_scaling_csv}`",
                    f"- Variable wells/core comparison: `{self.sources.wells_per_core_csv}`",
                    *(
                        f"- Additional wells/core comparison: `{path}`"
                        for path in self.sources.additional_wells_per_core_csvs
                    ),
                    *module_coverage_lines,
                    "",
                    "Figure groups:",
                    "- `01_parity_by_pipeline.*`",
                    "- `02_core_scaling_by_pipeline_plus_average_speedup.*`",
                    "- `03_core_scaling_average_with_pipeline_points_speedup.*`",
                    "- `04_core_scaling_by_pipeline_plus_average_ram.*` for per-pipeline RAM.",
                    "- `04_core_scaling_average_with_pipeline_points_ram.*` for aggregate RAM.",
                    "- `05_speedup_summary_by_core_and_wells_per_core.*`",
                    *module_coverage_figure_lines,
                    "",
                    "Interpretation notes:",
                    "- The `1 core` point is same-process, single-well execution latency.",
                    "- The `2/3/4 cores` points are native OpenHCS multiprocessing throughput runs over replicated wells.",
                    "- Very small pipelines can look non-monotonic in `02_core_scaling_by_pipeline_plus_average_speedup.*` because fixed fork/work-queue/file overhead is not amortized at low well counts.",
                    "- Use `05_speedup_summary_by_core_and_wells_per_core.*` to assess throughput scaling as queue depth increases.",
                    "",
                )
            ),
            encoding="utf-8",
        )
        return (path,)

    def generate_parity_figures(
        self,
        summary_rows: Sequence[Mapping[str, str]],
    ) -> tuple[Path, ...]:
        """Generate benchmark-style parity figures from summary.csv."""
        from benchmark.reports.cppipe_figures import (
            ACCURACY_FRACTION_FIELD,
            BenchmarkMetricRow,
            FigureMetricSpec,
            generate_grouped_benchmark_metric_figures,
        )

        rows = tuple(
            BenchmarkMetricRow(
                pipeline_name=row[CASE_NAME_FIELD],
                method="OpenHCS",
                assay_category=row.get("assay_category", ""),
                module_category=row.get("module_category", ""),
                accuracy_fraction=_optional_float(row.get("min_parity_accuracy")),
                raw_seconds=None,
                speedup=None,
                peak_memory_mb=None,
            )
            for row in summary_rows
        )
        csv_path = self.output_dir / "01_parity_by_pipeline.csv"
        self.write_metric_rows(csv_path, rows)
        return (
            csv_path,
            *generate_grouped_benchmark_metric_figures(
                rows,
                metrics=(
                    FigureMetricSpec(
                        ACCURACY_FRACTION_FIELD,
                        "01_parity_by_pipeline",
                        "Parity accuracy by pipeline",
                        "Parity accuracy (%)",
                        percentage=True,
                        baseline_line=100.0,
                        minimum_ylim=0.0,
                    ),
                ),
                methods=("OpenHCS",),
                pipeline_names=tuple(row[CASE_NAME_FIELD] for row in summary_rows),
                output_dir=self.output_dir,
                output_formats=self.output_formats,
            ),
        )

    def generate_core_scaling_figures(
        self,
        core_rows: Sequence[WellThroughputResult],
        summary_rows: Sequence[Mapping[str, str]],
    ) -> tuple[Path, ...]:
        """Generate fixed 1/2/3/4-core throughput and RAM figures."""
        outputs: list[Path] = []
        metric_rows = self.core_scaling_metric_rows(core_rows, summary_rows)
        average_rows = self.core_scaling_average_rows(metric_rows)
        all_rows = (*metric_rows, *average_rows)
        outputs.append(
            self.write_metric_rows(
                self.output_dir / "02_core_scaling_by_pipeline_plus_average_long.csv",
                all_rows,
            )
        )

        from benchmark.reports.cppipe_figures import (
            BenchmarkMetricRow,
            FigureMetricSpec,
            SPEEDUP_TARGET,
            generate_grouped_benchmark_metric_figures,
        )

        methods = tuple(mode.label for mode in self.core_scaling_modes)
        pipeline_names = tuple(row[CASE_NAME_FIELD] for row in summary_rows) + (
            "Average",
        )
        outputs.extend(
            self.generate_core_scaling_pipeline_speedup_figures(
                all_rows,
                methods=methods,
                pipeline_names=pipeline_names,
            )
        )
        outputs.extend(
            self.generate_average_point_figures(
                tuple(
                    BenchmarkMetricRow(
                        pipeline_name=row.pipeline_name,
                        method=row.method,
                        assay_category=row.assay_category,
                        module_category=row.module_category,
                        accuracy_fraction=row.accuracy_fraction,
                        raw_seconds=row.raw_seconds,
                        speedup=row.speedup,
                        peak_memory_mb=row.peak_memory_mb,
                    )
                    for row in metric_rows
                ),
                filename_stem="03_core_scaling_average_with_pipeline_points_speedup",
                title="Average execution speedup by core count",
                ylabel="Execution speedup vs CellProfiler (x)",
                value_key="speedup",
                target_line=SPEEDUP_TARGET,
                log_variant=True,
            )
        )
        outputs.extend(
            generate_grouped_benchmark_metric_figures(
                all_rows,
                metrics=(
                    FigureMetricSpec(
                        "peak_memory_mb",
                        "04_core_scaling_by_pipeline_plus_average_ram",
                        "RAM usage by pipeline and core count",
                        "Peak process-tree RSS (MB)",
                        minimum_ylim=0.0,
                        log_variant=True,
                        use_axis_break=False,
                    ),
                ),
                methods=methods,
                pipeline_names=pipeline_names,
                output_dir=self.output_dir,
                output_formats=self.output_formats,
            )
        )
        outputs.extend(
            self.generate_average_point_figures(
                metric_rows,
                filename_stem="04_core_scaling_average_with_pipeline_points_ram",
                title="Average RAM usage by core count",
                ylabel="Peak process-tree RSS (MB)",
                value_key="peak_memory_mb",
                log_variant=False,
            )
        )
        return tuple(outputs)

    def generate_core_scaling_pipeline_speedup_figures(
        self,
        rows: Sequence["BenchmarkMetricRow"],
        *,
        methods: Sequence[str],
        pipeline_names: Sequence[str],
    ) -> tuple[Path, ...]:
        """Generate the per-pipeline core scaling figure with multi-band breaks."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        from matplotlib.ticker import LogLocator
        from matplotlib.ticker import NullFormatter
        from matplotlib.ticker import NullLocator

        from benchmark.reports.cppipe_figures import FIGURE_STYLE
        from benchmark.reports.cppipe_figures import FIGURE_DPI
        from benchmark.reports.cppipe_figures import DEFAULT_WRAP_AFTER
        from benchmark.reports.cppipe_figures import LINEAR_AXIS_BREAK_POLICY
        from benchmark.reports.cppipe_figures import PIPELINE_LABEL_FONT_SIZE
        from benchmark.reports.cppipe_figures import PIPELINE_LABEL_LAYOUT
        from benchmark.reports.cppipe_figures import SPEEDUP_TARGET

        row_index = {(row.pipeline_name, row.method): row for row in rows}
        panels = PIPELINE_LABEL_LAYOUT.panels(pipeline_names, DEFAULT_WRAP_AFTER)
        width = min(0.18, 0.82 / max(len(methods), 1))
        offsets = _bar_offsets(len(methods), width)
        outputs: list[Path] = []

        global_bands = self.speedup_axis_band_policy.bands_for(
            tuple(row.speedup for row in rows if row.speedup is not None)
        )
        panel_bands = tuple(
            tuple(
                band
                for band in global_bands
                if any(
                    band.contains(row.speedup)
                    for pipeline_name in panel
                    for method in methods
                    if (row := row_index.get((pipeline_name, method))) is not None
                    and row.speedup is not None
                )
            )
            or (global_bands[0],)
            for panel in panels
        )
        with FIGURE_STYLE.context():
            axis_row_specs: list[tuple[int | None, int | None, float]] = []
            for panel_index, bands in enumerate(panel_bands):
                for band_index, ratio in enumerate(
                    (1.0,) * (len(bands) - 1) + (3.0,)
                ):
                    axis_row_specs.append((panel_index, band_index, ratio))
                if panel_index < len(panels) - 1:
                    axis_row_specs.append((None, None, 0.35))

            fig = plt.figure(
                figsize=(
                    max(8.0, max(len(panel) for panel in panels) * 0.98),
                    sum(4.8 + 1.35 * (len(bands) - 1) for bands in panel_bands),
                ),
                layout="constrained",
            )
            grid = fig.add_gridspec(
                len(axis_row_specs),
                1,
                height_ratios=[row_spec[2] for row_spec in axis_row_specs],
            )
            axes_by_panel: list[list[Any]] = [[] for _panel in panels]
            for row_index_, (panel_index, _band_index, _ratio) in enumerate(axis_row_specs):
                axis = fig.add_subplot(grid[row_index_, 0])
                if panel_index is None:
                    axis.set_axis_off()
                    continue
                axes_by_panel[panel_index].append(axis)

            for panel_index, (panel, bands) in enumerate(
                zip(panels, panel_bands, strict=True)
            ):
                panel_axes = tuple(axes_by_panel[panel_index])
                x_positions = tuple(range(len(panel)))
                visible_bands = tuple(reversed(bands))
                for band_index, (axis, band) in enumerate(zip(panel_axes, visible_bands, strict=True)):
                    for method_index, method in enumerate(methods):
                        values = tuple(
                            (
                                row.speedup
                                if (row := row_index.get((pipeline_name, method))) is not None
                                else None
                            )
                            for pipeline_name in panel
                        )
                        axis.bar(
                            [x + offsets[method_index] for x in x_positions],
                            [value if value is not None else float("nan") for value in values],
                            width=width,
                            label=method if panel_index == 0 and band_index == 0 else None,
                            color=FIGURE_STYLE.color_for_method(method_index),
                            edgecolor=FIGURE_STYLE.background,
                            linewidth=0.55,
                        )
                    axis.set_ylim(*band.as_ylim())
                    if band.contains(SPEEDUP_TARGET):
                        axis.axhline(
                            SPEEDUP_TARGET,
                            color=FIGURE_STYLE.target_color,
                            linewidth=1.15,
                            linestyle="--",
                            alpha=0.86,
                        )
                        axis.text(
                            len(panel) - 0.15,
                            SPEEDUP_TARGET * 1.05,
                            "4x target",
                            color=FIGURE_STYLE.target_color,
                            ha="right",
                            va="bottom",
                            fontsize=8,
                        )
                    axis.grid(
                        axis="y",
                        color=FIGURE_STYLE.grid_color,
                        linewidth=0.8,
                        alpha=0.8,
                    )
                    axis.set_axisbelow(True)
                    axis.spines["top"].set_visible(False)
                    axis.spines["right"].set_visible(False)
                    axis.spines["left"].set_color(FIGURE_STYLE.spine_color)
                    axis.spines["bottom"].set_color(FIGURE_STYLE.spine_color)
                    axis.set_xticks(list(x_positions))
                    if band_index < len(visible_bands) - 1:
                        axis.set_xticklabels(())
                    else:
                        axis.set_xticklabels(
                            [PIPELINE_LABEL_LAYOUT.split_label(name) for name in panel],
                            rotation=42,
                            ha="right",
                            fontsize=PIPELINE_LABEL_FONT_SIZE,
                    )
                    if band_index > 0:
                        LINEAR_AXIS_BREAK_POLICY.mark(panel_axes[band_index - 1], axis)
                    if panel_index == 0 and band_index == 0:
                        axis.set_title(
                            "Execution speedup by pipeline and core count",
                            loc="left",
                            pad=10,
                        )
            axes_by_panel[0][0].legend(
                frameon=False,
                ncol=min(len(methods), 5),
                loc="upper left",
            )
            fig.supylabel("Execution speedup vs CellProfiler (x)")
            for output_format in self.output_formats:
                output_path = (
                    self.output_dir
                    / f"02_core_scaling_by_pipeline_plus_average_speedup.{output_format}"
                )
                fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
                outputs.append(output_path)
            plt.close(fig)

        with FIGURE_STYLE.context():
            fig, axes_ = plt.subplots(
                len(panels),
                1,
                figsize=(max(8.0, max(len(panel) for panel in panels) * 0.98), 7.2),
                layout="constrained",
            )
            axes = (axes_,) if len(panels) == 1 else tuple(axes_)
            for panel_index, (axis, panel) in enumerate(zip(axes, panels, strict=True)):
                x_positions = tuple(range(len(panel)))
                for method_index, method in enumerate(methods):
                    values = tuple(
                        (
                            row.speedup
                            if (row := row_index.get((pipeline_name, method))) is not None
                            else None
                        )
                        for pipeline_name in panel
                    )
                    axis.bar(
                        [x + offsets[method_index] for x in x_positions],
                        [value if value is not None else float("nan") for value in values],
                        width=width,
                        label=method if panel_index == 0 else None,
                        color=FIGURE_STYLE.color_for_method(method_index),
                        edgecolor=FIGURE_STYLE.background,
                        linewidth=0.55,
                    )
                axis.set_yscale("log")
                axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
                axis.yaxis.set_minor_locator(NullLocator())
                axis.yaxis.set_major_formatter(FuncFormatter(_plain_numeric_tick_label))
                axis.yaxis.set_minor_formatter(NullFormatter())
                axis.axhline(
                    SPEEDUP_TARGET,
                    color=FIGURE_STYLE.target_color,
                    linewidth=1.15,
                    linestyle="--",
                    alpha=0.86,
                )
                axis.grid(axis="y", color=FIGURE_STYLE.grid_color, linewidth=0.8, alpha=0.8)
                axis.set_axisbelow(True)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.set_ylabel("Execution speedup vs CellProfiler (x)")
                axis.set_xticks(list(x_positions))
                axis.set_xticklabels(
                    [PIPELINE_LABEL_LAYOUT.split_label(name) for name in panel],
                    rotation=42,
                    ha="right",
                    fontsize=PIPELINE_LABEL_FONT_SIZE,
                )
                if panel_index == 0:
                    axis.set_title(
                        "Execution speedup by pipeline and core count (log)",
                        loc="left",
                        pad=10,
                    )
            axes[0].legend(frameon=False, ncol=min(len(methods), 5), loc="upper left")
            for output_format in self.output_formats:
                output_path = (
                    self.output_dir
                    / f"02_core_scaling_by_pipeline_plus_average_speedup_log.{output_format}"
                )
                fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
                outputs.append(output_path)
            plt.close(fig)
        return tuple(outputs)

    def generate_module_coverage_figures(self) -> tuple[Path, ...]:
        """Generate module coverage pie chart and detailed coverage tables."""
        if self.sources.module_coverage_semantic_families_csv is None:
            return ()
        coverage_csv = self.sources.module_coverage_semantic_families_csv
        if not coverage_csv.exists():
            return ()

        table = ModuleAbstractionCoverageTable.from_semantic_family_csv(coverage_csv)
        outputs = [
            self.write_module_coverage_csv(
                self.output_dir / "06_module_coverage_by_abstraction_modules.csv",
                table,
            ),
            self.write_module_coverage_markdown(
                self.output_dir / "06_module_coverage_by_abstraction_modules.md",
                table,
            ),
        ]
        outputs.extend(self.generate_module_coverage_pie_figure(table))
        return tuple(outputs)

    @staticmethod
    def write_module_coverage_csv(
        path: Path,
        table: ModuleAbstractionCoverageTable,
    ) -> Path:
        """Write one row per module for presentation coverage claims."""
        fieldnames = (
            "module_name",
            "coverage",
            "coverage_label",
            "abstraction_family",
            "evidence_modules",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in table.rows:
                writer.writerow(
                    {
                        "module_name": row.module_name,
                        "coverage": row.coverage.value,
                        "coverage_label": row.coverage.label,
                        "abstraction_family": row.abstraction_family,
                        "evidence_modules": row.evidence_text,
                    }
                )
        return path

    @staticmethod
    def write_module_coverage_markdown(
        path: Path,
        table: ModuleAbstractionCoverageTable,
    ) -> Path:
        """Write explicit module lists grouped by presentation coverage state."""
        grouped_rows = table.grouped_rows()
        lines = ["# Module Coverage by Shared Abstractions", ""]
        for coverage in ModuleAbstractionCoverageKind:
            rows = grouped_rows[coverage]
            lines.extend((f"## {coverage.label} ({len(rows)})", ""))
            if not rows:
                lines.extend(("_None_", ""))
                continue
            lines.extend(
                f"- `{row.module_name}`"
                + (
                    f" via `{row.abstraction_family}`"
                    if row.abstraction_family
                    else ""
                )
                + (
                    f" from {row.evidence_text}"
                    if row.coverage is ModuleAbstractionCoverageKind.SHARED_ABSTRACTION
                    and row.evidence_text
                    else ""
                )
                for row in rows
            )
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def generate_module_coverage_pie_figure(
        self,
        table: ModuleAbstractionCoverageTable,
    ) -> tuple[Path, ...]:
        """Plot module coverage shares with a summary count table."""
        import matplotlib.pyplot as plt

        from benchmark.reports.cppipe_figures import FIGURE_DPI
        from benchmark.reports.cppipe_figures import FIGURE_STYLE

        grouped_rows = table.grouped_rows()
        labels = tuple(coverage.label for coverage in ModuleAbstractionCoverageKind)
        counts = tuple(len(grouped_rows[coverage]) for coverage in ModuleAbstractionCoverageKind)
        total = sum(counts)
        if total == 0:
            return ()
        colors = ("#008b8b", "#d95f02", "#7a7f88")
        table_rows = tuple(
            (
                label,
                str(count),
                f"{count / total * 100.0:.1f}%",
            )
            for label, count in zip(labels, counts, strict=True)
        ) + (("Total", str(total), "100.0%"),)
        uncovered_rows = tuple(
            (row.module_name,)
            for row in grouped_rows[ModuleAbstractionCoverageKind.UNCOVERED]
        )

        outputs: list[Path] = []
        with FIGURE_STYLE.context():
            fig, (pie_axis, summary_axis, uncovered_axis) = plt.subplots(
                1,
                3,
                figsize=(15.6, 4.9),
                gridspec_kw={"width_ratios": (1.0, 1.25, 0.95)},
                layout="constrained",
            )
            pie_axis.pie(
                counts,
                labels=labels,
                autopct=lambda percent: f"{percent:.0f}%" if percent >= 3.0 else "",
                startangle=90,
                counterclock=False,
                colors=colors,
                wedgeprops={"linewidth": 1.0, "edgecolor": FIGURE_STYLE.background},
                textprops={"fontsize": 9.0, "color": FIGURE_STYLE.text_color},
            )
            pie_axis.set_title(
                "CellProfiler module coverage",
                loc="left",
                pad=10,
            )
            summary_axis.axis("off")
            summary_table = summary_axis.table(
                cellText=table_rows,
                colLabels=("Coverage", "Modules", "Share"),
                cellLoc="left",
                colLoc="left",
                colWidths=(0.56, 0.22, 0.22),
                loc="center",
            )
            summary_table.auto_set_font_size(False)
            summary_table.set_fontsize(9.0)
            summary_table.scale(1.0, 1.45)
            for (row_index, _column_index), cell in summary_table.get_celld().items():
                cell.set_edgecolor(FIGURE_STYLE.grid_color)
                if row_index == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#f0eee7")
                if row_index == len(table_rows):
                    cell.set_text_props(weight="bold")
            uncovered_axis.axis("off")
            uncovered_axis.set_title(
                "Not covered modules",
                loc="left",
                pad=10,
                fontsize=10,
            )
            uncovered_table = uncovered_axis.table(
                cellText=uncovered_rows,
                colLabels=("Module",),
                cellLoc="left",
                colLoc="left",
                colWidths=(1.0,),
                loc="center",
            )
            uncovered_table.auto_set_font_size(False)
            uncovered_table.set_fontsize(8.5)
            uncovered_table.scale(1.0, 1.25)
            for (row_index, _column_index), cell in uncovered_table.get_celld().items():
                cell.set_edgecolor(FIGURE_STYLE.grid_color)
                if row_index == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#f0eee7")
            for output_format in self.output_formats:
                output_path = (
                    self.output_dir
                    / f"06_module_coverage_by_abstraction.{output_format}"
                )
                fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
                outputs.append(output_path)
            plt.close(fig)
        return tuple(outputs)

    def generate_wells_per_core_figures(
        self,
        core_rows: Sequence[WellThroughputResult],
        wells_per_core_rows: Sequence[WellThroughputResult],
    ) -> tuple[Path, ...]:
        """Generate summary figures for 2/3/4 cores x 2/3/4 wells/core."""
        summary_rows = self.wells_per_core_summary_rows(
            core_rows=core_rows,
            wells_per_core_rows=wells_per_core_rows,
        )
        outputs: list[Path] = []
        outputs.append(
            self.write_wells_per_core_summary_csv(
                self.output_dir / "05_speedup_by_core_and_wells_per_core_summary.csv",
                summary_rows,
            )
        )
        outputs.append(
            self.write_wells_per_core_summary_markdown(
                self.output_dir / "05_speedup_by_core_and_wells_per_core_summary.md",
                summary_rows,
            )
        )

        outputs.extend(self.generate_wells_per_core_summary_figure(summary_rows))
        return tuple(outputs)

    def generate_wells_per_core_summary_figure(
        self,
        summary_rows: Sequence["WellsPerCoreSummaryRow"],
    ) -> tuple[Path, ...]:
        """Plot mean bars with median/min overlays in one wells/core summary."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        from matplotlib.ticker import LogLocator
        from matplotlib.ticker import NullFormatter
        from matplotlib.ticker import NullLocator

        from benchmark.reports.cppipe_figures import FIGURE_DPI
        from benchmark.reports.cppipe_figures import FIGURE_STYLE
        from benchmark.reports.cppipe_figures import SPEEDUP_TARGET

        row_index = {
            (row.worker_count, row.wells_per_core): row for row in summary_rows
        }
        ordered_keys = tuple(
            (worker_count, wells_per_core)
            for worker_count in self.multicore_worker_counts
            for wells_per_core in sorted(
                {
                    row.wells_per_core
                    for row in summary_rows
                    if row.worker_count == worker_count
                }
            )
        )
        outputs: list[Path] = []
        for log_y in (False, True):
            with FIGURE_STYLE.context():
                fig, axis = plt.subplots(
                    1,
                    1,
                    figsize=(12.0, 5.8),
                    layout="constrained",
                )
                x_positions = tuple(range(len(ordered_keys)))
                mean_values = tuple(
                    row_index[key].mean if key in row_index else float("nan")
                    for key in ordered_keys
                )
                median_values = tuple(
                    row_index[key].median if key in row_index else float("nan")
                    for key in ordered_keys
                )
                min_values = tuple(
                    row_index[key].min if key in row_index else float("nan")
                    for key in ordered_keys
                )
                bar_colors = tuple(
                    FIGURE_STYLE.color_for_method(
                        self.multicore_worker_counts.index(worker_count)
                    )
                    for worker_count, _wells_per_core in ordered_keys
                )
                axis.bar(
                    x_positions,
                    mean_values,
                    width=0.72,
                    label="Mean",
                    color=bar_colors,
                    edgecolor=FIGURE_STYLE.background,
                    linewidth=0.55,
                    alpha=0.82,
                )
                axis.scatter(
                    x_positions,
                    median_values,
                    label="Median",
                    marker="D",
                    s=58,
                    color=FIGURE_STYLE.text_color,
                    zorder=4,
                )
                axis.scatter(
                    x_positions,
                    min_values,
                    label="Minimum",
                    marker="v",
                    s=72,
                    color=FIGURE_STYLE.target_color,
                    edgecolor=FIGURE_STYLE.background,
                    linewidth=0.7,
                    zorder=5,
                )
                axis.axhline(
                    SPEEDUP_TARGET,
                    color=FIGURE_STYLE.target_color,
                    linewidth=1.15,
                    linestyle="--",
                    alpha=0.86,
                )
                if log_y:
                    axis.set_yscale("log")
                    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
                    axis.yaxis.set_minor_locator(NullLocator())
                    axis.yaxis.set_major_formatter(
                        FuncFormatter(_plain_numeric_tick_label)
                    )
                    axis.yaxis.set_minor_formatter(NullFormatter())
                else:
                    axis.set_ylim(bottom=0.0)
                axis.set_xticks(list(x_positions))
                axis.set_xticklabels(
                    [
                        f"{worker_count}c\n{wells_per_core} wells/core"
                        for worker_count, wells_per_core in ordered_keys
                    ],
                    rotation=35,
                    ha="right",
                    fontsize=9,
                )
                axis.set_ylabel("Speedup vs CellProfiler (x)", fontsize=10)
                axis.grid(
                    axis="y",
                    color=FIGURE_STYLE.grid_color,
                    linewidth=0.8,
                    alpha=0.8,
                )
                axis.set_axisbelow(True)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
                title_suffix = " (log)" if log_y else ""
                fig.suptitle(
                    f"Multicore speedup summary by wells/core{title_suffix}",
                    x=0.01,
                    ha="left",
                    fontsize=13,
                    fontweight="bold",
                )
                stem = "05_speedup_summary_by_core_and_wells_per_core"
                if log_y:
                    stem = f"{stem}_log"
                for output_format in self.output_formats:
                    output_path = self.output_dir / f"{stem}.{output_format}"
                    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
                    outputs.append(output_path)
                plt.close(fig)
        return tuple(outputs)

    def core_scaling_metric_rows(
        self,
        core_rows: Sequence[WellThroughputResult],
        summary_rows: Sequence[Mapping[str, str]],
    ) -> tuple["BenchmarkMetricRow", ...]:
        """Project fixed core-scaling observations into benchmark metric rows."""
        from benchmark.reports.cppipe_figures import BenchmarkMetricRow

        row_index = {(row.case_name, row.mode_name): row for row in core_rows}
        rows: list[BenchmarkMetricRow] = []
        for summary_row in summary_rows:
            case_name = summary_row[CASE_NAME_FIELD]
            for mode in self.core_scaling_modes:
                row = row_index.get((case_name, mode.source_mode_name))
                if row is None:
                    continue
                rows.append(
                    BenchmarkMetricRow(
                        pipeline_name=case_name,
                        method=mode.label,
                        assay_category=summary_row.get("assay_category", ""),
                        module_category=summary_row.get("module_category", ""),
                        accuracy_fraction=None,
                        raw_seconds=row.execute_seconds,
                        speedup=row.projected_execution_speedup,
                        peak_memory_mb=row.peak_memory_mb,
                    )
                )
        return tuple(rows)

    def core_scaling_average_rows(
        self,
        rows: Sequence["BenchmarkMetricRow"],
    ) -> tuple["BenchmarkMetricRow", ...]:
        """Aggregate one Average row per fixed core-scaling mode."""
        from benchmark.reports.cppipe_figures import BenchmarkMetricRow

        average_rows: list[BenchmarkMetricRow] = []
        for mode in self.core_scaling_modes:
            mode_rows = tuple(row for row in rows if row.method == mode.label)
            average_rows.append(
                BenchmarkMetricRow(
                    pipeline_name="Average",
                    method=mode.label,
                    assay_category="",
                    module_category="",
                    accuracy_fraction=None,
                    raw_seconds=_mean_present(row.raw_seconds for row in mode_rows),
                    speedup=_mean_present(row.speedup for row in mode_rows),
                    peak_memory_mb=_mean_present(row.peak_memory_mb for row in mode_rows),
                )
            )
        return tuple(average_rows)

    def generate_average_point_figures(
        self,
        rows: Sequence["BenchmarkMetricRow"],
        *,
        filename_stem: str,
        title: str,
        ylabel: str,
        value_key: str,
        target_line: float | None = None,
        log_variant: bool,
    ) -> tuple[Path, ...]:
        """Plot mean bars with all per-pipeline points for each method."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        from matplotlib.ticker import LogLocator
        from matplotlib.ticker import NullFormatter
        from matplotlib.ticker import NullLocator

        from benchmark.reports.cppipe_figures import FIGURE_STYLE
        from benchmark.reports.cppipe_figures import LINEAR_AXIS_BREAK_POLICY

        methods = tuple(mode.label for mode in self.core_scaling_modes)
        method_values = tuple(
            (
                method,
                tuple(
                    float(value)
                    for row in rows
                    if row.method == method
                    and (value := getattr(row, value_key)) is not None
                ),
            )
            for method in methods
        )
        values = tuple(value for _method, method_values_ in method_values for value in method_values_)
        if not values:
            return ()
        value_suffix = " MB" if value_key == "peak_memory_mb" else "x"
        outputs: list[Path] = []
        for log_y in (False, True) if log_variant else (False,):
            broken_range = (
                None
                if log_y or value_key == "peak_memory_mb"
                else LINEAR_AXIS_BREAK_POLICY.range_for(values)
            )
            with FIGURE_STYLE.context():
                if broken_range is None:
                    fig, axis = plt.subplots(
                        1,
                        1,
                        figsize=(max(5.6, 1.45 * len(method_values) + 3.2), 4.6),
                        layout="constrained",
                    )
                    axes = (axis,)
                else:
                    fig, axes_ = plt.subplots(
                        2,
                        1,
                        figsize=(max(5.6, 1.45 * len(method_values) + 3.2), 5.6),
                        gridspec_kw={"height_ratios": (1.0, 3.2)},
                        sharex=True,
                        layout="constrained",
                    )
                    top_axis, bottom_axis = tuple(axes_)
                    top_axis.set_ylim(broken_range[1], broken_range[2])
                    bottom_axis.set_ylim(0.0, broken_range[0])
                    LINEAR_AXIS_BREAK_POLICY.mark(top_axis, bottom_axis)
                    axes = (top_axis, bottom_axis)
                x_positions = tuple(range(len(method_values)))
                if log_y:
                    for axis in axes:
                        axis.set_yscale("log")
                        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
                        axis.yaxis.set_minor_locator(NullLocator())
                        axis.yaxis.set_major_formatter(
                            FuncFormatter(_plain_numeric_tick_label)
                        )
                        axis.yaxis.set_minor_formatter(NullFormatter())
                if target_line is not None:
                    for axis in axes:
                        axis.axhline(
                            target_line,
                            color=FIGURE_STYLE.target_color,
                            linewidth=1.15,
                            linestyle="--",
                            alpha=0.86,
                        )
                    target_axis = next(
                        (
                            axis
                            for axis in axes
                            if axis.get_ylim()[0] <= target_line <= axis.get_ylim()[1]
                        ),
                        axes[-1],
                    )
                    target_axis.annotate(
                        f"{target_line:g}x",
                        xy=(-0.012, target_line),
                        xycoords=("axes fraction", "data"),
                        xytext=(-2, 3),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        fontsize=7.8,
                        color=FIGURE_STYLE.target_color,
                        annotation_clip=False,
                    )

                def visible_axis_for(value: float):
                    return next(
                        (
                            axis
                            for axis in axes
                            if axis.get_ylim()[0] <= value <= axis.get_ylim()[1]
                        ),
                        None,
                    )

                def annotate_value(
                    *,
                    x: float,
                    value: float,
                    text: str,
                    x_offset: float = 4.0,
                    y_offset: float = 0.0,
                    ha: str = "left",
                    va: str = "center",
                    color: str | None = None,
                    weight: str = "normal",
                ) -> None:
                    axis = visible_axis_for(value)
                    if axis is None:
                        return
                    axis.annotate(
                        text,
                        xy=(x, value),
                        xytext=(x_offset, y_offset),
                        textcoords="offset points",
                        ha=ha,
                        va=va,
                        fontsize=6.2,
                        color=color or FIGURE_STYLE.text_color,
                        fontweight=weight,
                        bbox={
                            "boxstyle": "round,pad=0.08",
                            "facecolor": FIGURE_STYLE.background,
                            "edgecolor": "none",
                            "alpha": 0.72,
                        },
                        clip_on=True,
                        zorder=6,
                    )

                def adjusted_label_values(
                    axis,
                    stats: Sequence[tuple[str, float, str]],
                    *,
                    min_gap_points: float = 11.0,
                    edge_padding_points: float = 5.0,
                ) -> tuple[tuple[str, float, str], ...]:
                    """Return stat label y-values adjusted to avoid text collisions."""
                    if not stats:
                        return ()
                    renderer = fig.canvas.get_renderer()
                    points_to_pixels = renderer.points_to_pixels
                    min_gap_pixels = points_to_pixels(min_gap_points)
                    edge_padding_pixels = points_to_pixels(edge_padding_points)
                    axis_min = axis.bbox.y0 + edge_padding_pixels
                    axis_max = axis.bbox.y1 - edge_padding_pixels

                    positioned = sorted(
                        (
                            (
                                label,
                                value,
                                weight,
                                axis.transData.transform((0.0, value))[1],
                            )
                            for label, value, weight in stats
                        ),
                        key=lambda item: item[3],
                    )
                    adjusted_pixels: list[float] = []
                    for _label, _value, _weight, original_pixels in positioned:
                        adjusted_pixels.append(
                            max(
                                original_pixels,
                                adjusted_pixels[-1] + min_gap_pixels
                                if adjusted_pixels
                                else axis_min,
                            )
                        )
                    if adjusted_pixels[-1] > axis_max:
                        adjusted_pixels[-1] = axis_max
                    for index in range(len(adjusted_pixels) - 2, -1, -1):
                        adjusted_pixels[index] = min(
                            adjusted_pixels[index],
                            adjusted_pixels[index + 1] - min_gap_pixels,
                        )
                    for index in range(1, len(adjusted_pixels)):
                        adjusted_pixels[index] = max(
                            adjusted_pixels[index],
                            adjusted_pixels[index - 1] + min_gap_pixels,
                        )

                    return tuple(
                        (
                            label,
                            axis.transData.inverted().transform((0.0, adjusted_pixels[index]))[1],
                            weight,
                        )
                        for index, (label, _value, weight, _pixels) in enumerate(positioned)
                    )

                def annotate_stat_labels(
                    *,
                    method_index: int,
                    color: str,
                    stats: Sequence[tuple[str, float, str]],
                ) -> None:
                    label_x = method_index - 0.30
                    for axis in axes:
                        visible_stats = tuple(
                            (label, value, weight)
                            for label, value, weight in stats
                            if axis.get_ylim()[0] <= value <= axis.get_ylim()[1]
                        )
                        for label, label_value, weight in adjusted_label_values(
                            axis, visible_stats
                        ):
                            annotate_value(
                                x=label_x,
                                value=label_value,
                                text=label,
                                x_offset=-3.0,
                                y_offset=0.0,
                                ha="right",
                                va="center",
                                color=color,
                                weight=weight,
                            )

                stat_label_groups: list[tuple[int, str, tuple[tuple[str, float, str], ...]]] = []
                for method_index, (_method, values_) in enumerate(method_values):
                    if not values_:
                        continue
                    mean = sum(values_) / len(values_)
                    median = statistics.median(values_)
                    minimum = min(values_)
                    maximum = max(values_)
                    color = FIGURE_STYLE.color_for_method(method_index + 1)
                    stat_label_groups.append(
                        (
                            method_index,
                            color,
                            (
                                (f"min {minimum:.1f}{value_suffix}", minimum, "normal"),
                                (f"med {median:.1f}{value_suffix}", median, "normal"),
                                (f"mean {mean:.1f}{value_suffix}", mean, "bold"),
                                (f"max {maximum:.1f}{value_suffix}", maximum, "normal"),
                            ),
                        )
                    )
                    point_x = tuple(
                        method_index + _deterministic_jitter(index, len(values_))
                        for index in range(len(values_))
                    )
                    for axis in axes:
                        axis.bar(
                            [method_index],
                            [mean],
                            width=0.62,
                            color=color,
                            alpha=0.72,
                            edgecolor=FIGURE_STYLE.background,
                            linewidth=0.55,
                        )
                        axis.scatter(
                            point_x,
                            values_,
                            s=28,
                            color=FIGURE_STYLE.text_color,
                            alpha=0.58,
                            zorder=3,
                        )
                        axis.hlines(
                            median,
                            method_index - 0.31,
                            method_index + 0.31,
                            color=FIGURE_STYLE.text_color,
                            linewidth=1.35,
                            zorder=4,
                            label=(
                                "Median"
                                if method_index == 0 and axis is axes[0]
                                else None
                            ),
                        )
                        axis.grid(
                            axis="y",
                            color=FIGURE_STYLE.grid_color,
                            linewidth=0.8,
                            alpha=0.8,
                        )
                        axis.set_axisbelow(True)
                        axis.spines["top"].set_visible(False)
                        axis.spines["right"].set_visible(False)
                        axis.spines["left"].set_color(FIGURE_STYLE.spine_color)
                        axis.spines["bottom"].set_color(FIGURE_STYLE.spine_color)
                label_axis = axes[-1]
                for axis in axes:
                    axis.set_xlim(-0.80, len(method_values) - 0.48)
                label_axis.set_xticks(list(x_positions))
                label_axis.set_xticklabels([method for method, _values in method_values])
                label_axis.set_ylabel(ylabel)
                axes[0].set_title(title + (" (log)" if log_y else ""), loc="left", pad=10)
                axes[0].legend(frameon=False, loc="upper left")
                fig.canvas.draw()
                for method_index, color, stats in stat_label_groups:
                    annotate_stat_labels(
                        method_index=method_index,
                        color=color,
                        stats=stats,
                    )
                suffix = "_log" if log_y else ""
                for output_format in self.output_formats:
                    output_path = self.output_dir / f"{filename_stem}{suffix}.{output_format}"
                    FIGURE_STYLE.save(fig, output_path)
                    outputs.append(output_path)
                plt.close(fig)
        return tuple(outputs)

    def wells_per_core_summary_rows(
        self,
        *,
        core_rows: Sequence[WellThroughputResult],
        wells_per_core_rows: Sequence[WellThroughputResult],
    ) -> tuple["WellsPerCoreSummaryRow", ...]:
        """Summarize multicore modes across all available wells/core counts."""
        rows_by_key: dict[tuple[int, int], list[float]] = {}
        seen_observations: set[tuple[str, str]] = set()
        for row in (*wells_per_core_rows, *core_rows):
            if row.worker_count not in self.multicore_worker_counts:
                continue
            if row.well_count % row.worker_count != 0:
                continue
            observation_key = (row.case_name, row.mode_name)
            if observation_key in seen_observations:
                continue
            seen_observations.add(observation_key)
            wells_per_core = row.well_count // row.worker_count
            if row.projected_execution_speedup is not None:
                rows_by_key.setdefault((row.worker_count, wells_per_core), []).append(
                    float(row.projected_execution_speedup)
                )
        return tuple(
            WellsPerCoreSummaryRow.from_values(
                worker_count=worker_count,
                wells_per_core=wells_per_core,
                values=tuple(rows_by_key[(worker_count, wells_per_core)]),
            )
            for worker_count, wells_per_core in sorted(rows_by_key)
        )

    @staticmethod
    def read_dict_rows(path: Path) -> tuple[Mapping[str, str], ...]:
        """Read a CSV file as immutable mapping rows."""
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return tuple(csv.DictReader(handle))

    @staticmethod
    def write_metric_rows(path: Path, rows: Sequence["BenchmarkMetricRow"]) -> Path:
        """Write benchmark metric rows used by generated figures."""
        fieldnames = (
            "pipeline_name",
            "method",
            "assay_category",
            "module_category",
            "accuracy_fraction",
            "raw_seconds",
            "speedup",
            "peak_memory_mb",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: getattr(row, field) for field in fieldnames})
        return path

    @staticmethod
    def write_wells_per_core_summary_csv(
        path: Path,
        rows: Sequence["WellsPerCoreSummaryRow"],
    ) -> Path:
        """Write 2/3/4 cores x 2/3/4 wells/core summary statistics."""
        fieldnames = tuple(asdict(rows[0])) if rows else ()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        return path

    @staticmethod
    def write_wells_per_core_summary_markdown(
        path: Path,
        rows: Sequence["WellsPerCoreSummaryRow"],
    ) -> Path:
        """Write a Markdown table for wells/core summary statistics."""
        lines = [
            "| cores | wells/core | n | minimum x | median x | mean x | max x |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        lines.extend(
            (
                f"| {row.worker_count} | {row.wells_per_core} | {row.n} | "
                f"{row.min:.2f} | {row.median:.2f} | {row.mean:.2f} | "
                f"{row.max:.2f} |"
            )
            for row in rows
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True, slots=True)
class WellsPerCoreSummaryRow:
    """Speedup summary statistics for one core count and wells/core setting."""

    worker_count: int
    wells_per_core: int
    n: int
    min: float
    median: float
    mean: float
    max: float

    @classmethod
    def from_values(
        cls,
        *,
        worker_count: int,
        wells_per_core: int,
        values: Sequence[float],
    ) -> "WellsPerCoreSummaryRow":
        """Build summary statistics from finite speedup values."""
        finite_values = tuple(value for value in values if math.isfinite(value))
        if not finite_values:
            raise ValueError(
                f"No speedup values for {worker_count} cores and "
                f"{wells_per_core} wells/core."
            )
        return cls(
            worker_count=int(worker_count),
            wells_per_core=int(wells_per_core),
            n=len(finite_values),
            min=min(finite_values),
            median=statistics.median(finite_values),
            mean=statistics.mean(finite_values),
            max=max(finite_values),
        )


@dataclass(frozen=True, slots=True)
class NativeCellProfilerExecutionBaseline:
    """Single-sample native CP execution baseline for projected throughput speedups."""

    case_name: str
    execution_seconds: float

    def projected_execution_seconds(self, well_count: int) -> float:
        """Return the preliminary native CP projection for repeated samples."""
        return self.execution_seconds * int(well_count)


class WellThroughputStatus(StrEnum):
    """Terminal status for one well-throughput observation."""

    SUCCESS = "success"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    ERROR = "error"


@dataclass(slots=True)
class ChildProcessTerminator:
    """Terminates OpenHCS worker children when a benchmark guardrail trips."""

    terminate_timeout_seconds: float = 5.0
    kill_timeout_seconds: float = 5.0

    def __call__(
        self,
        _peak_memory_mb: float,
        children: tuple[psutil.Process, ...],
    ) -> None:
        live_children = tuple(child for child in children if child.is_running())
        for child in live_children:
            try:
                child.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        _, still_alive = psutil.wait_procs(
            live_children,
            timeout=self.terminate_timeout_seconds,
        )
        for child in still_alive:
            try:
                child.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        psutil.wait_procs(still_alive, timeout=self.kill_timeout_seconds)


class WorkerLaneEventPhase(str, Enum):
    """Progress-event phases that update benchmark worker-lane summaries."""

    AXIS_STARTED = "axis_started"
    AXIS_COMPLETED = "axis_completed"

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "WorkerLaneEventPhase | None":
        raw_phase = event.get("phase")
        if raw_phase is None:
            return None
        try:
            return cls(str(raw_phase))
        except ValueError:
            return None

    def apply_to_lane(self, lane: dict[str, Any], timestamp: float) -> None:
        """Apply this phase to one worker-lane aggregation row."""
        if self is WorkerLaneEventPhase.AXIS_STARTED:
            if lane["started_at"] == "" or timestamp < float(lane["started_at"]):
                lane["started_at"] = timestamp
            return
        if self is WorkerLaneEventPhase.AXIS_COMPLETED:
            lane["axis_count"] += 1
            if lane["completed_at"] == "" or timestamp > float(lane["completed_at"]):
                lane["completed_at"] = timestamp
            return


@dataclass(frozen=True, slots=True)
class WellThroughputResult:
    """One compile-once, execute-many-wells throughput observation."""

    case_name: str
    mode_name: str
    worker_count: int
    well_count: int
    compile_seconds: float
    prepare_seconds: float
    execute_seconds: float
    total_seconds: float
    wells_per_second: float
    successful_wells: int
    native_single_sample_execution_seconds: float | None = None
    projected_native_execution_seconds: float | None = None
    projected_execution_speedup: float | None = None
    peak_memory_mb: float | None = None
    status: WellThroughputStatus = WellThroughputStatus.SUCCESS
    memory_limit_mb: float | None = None
    error_message: str | None = None

    @classmethod
    def memory_limited(
        cls,
        *,
        case_name: str,
        mode: WellThroughputMode,
        compile_seconds: float,
        prepare_seconds: float,
        execute_seconds: float,
        total_seconds: float,
        peak_memory_mb: float | None,
        memory_limit_mb: float,
        native_execution_baseline: NativeCellProfilerExecutionBaseline | None,
        error_message: str | None = None,
    ) -> "WellThroughputResult":
        projected_native_execution_seconds = (
            native_execution_baseline.projected_execution_seconds(mode.well_count)
            if native_execution_baseline is not None
            else None
        )
        return cls(
            case_name=case_name,
            mode_name=mode.name,
            worker_count=mode.worker_count,
            well_count=mode.well_count,
            compile_seconds=compile_seconds,
            prepare_seconds=prepare_seconds,
            execute_seconds=execute_seconds,
            total_seconds=total_seconds,
            wells_per_second=0.0,
            successful_wells=0,
            native_single_sample_execution_seconds=(
                native_execution_baseline.execution_seconds
                if native_execution_baseline is not None
                else None
            ),
            projected_native_execution_seconds=projected_native_execution_seconds,
            projected_execution_speedup=None,
            peak_memory_mb=peak_memory_mb,
            status=WellThroughputStatus.MEMORY_LIMIT_EXCEEDED,
            memory_limit_mb=memory_limit_mb,
            error_message=error_message,
        )

    @classmethod
    def failed(
        cls,
        *,
        case_name: str,
        mode: WellThroughputMode,
        compile_seconds: float,
        prepare_seconds: float,
        execute_seconds: float,
        total_seconds: float,
        peak_memory_mb: float | None,
        native_execution_baseline: NativeCellProfilerExecutionBaseline | None,
        error_message: str,
    ) -> "WellThroughputResult":
        projected_native_execution_seconds = (
            native_execution_baseline.projected_execution_seconds(mode.well_count)
            if native_execution_baseline is not None
            else None
        )
        return cls(
            case_name=case_name,
            mode_name=mode.name,
            worker_count=mode.worker_count,
            well_count=mode.well_count,
            compile_seconds=compile_seconds,
            prepare_seconds=prepare_seconds,
            execute_seconds=execute_seconds,
            total_seconds=total_seconds,
            wells_per_second=0.0,
            successful_wells=0,
            native_single_sample_execution_seconds=(
                native_execution_baseline.execution_seconds
                if native_execution_baseline is not None
                else None
            ),
            projected_native_execution_seconds=projected_native_execution_seconds,
            projected_execution_speedup=None,
            peak_memory_mb=peak_memory_mb,
            status=WellThroughputStatus.ERROR,
            error_message=error_message,
        )

    def is_successful(self) -> bool:
        """Return whether this observation completed normally."""
        return self.status is WellThroughputStatus.SUCCESS


def run_well_throughput_suite(
    manifest_path: Path,
    *,
    output_root: Path,
    case_names: Sequence[str] = (),
    well_counts: Sequence[int],
    worker_counts: Sequence[int],
    start_method: MultiprocessingStartMethod = MultiprocessingStartMethod.FORK,
    plan: WellThroughputBenchmarkPlan | None = None,
    native_execution_baselines: Mapping[
        str,
        NativeCellProfilerExecutionBaseline,
    ] | None = None,
    existing_results: Sequence[WellThroughputResult] = (),
    skipped_observations: Sequence[WellThroughputObservationKey] = (),
    rerun_missing_memory: bool = False,
    max_memory_mb: float | None = None,
) -> tuple[WellThroughputResult, ...]:
    """Run converted cppipes as one OpenHCS plate with repeated virtual wells."""
    cases = load_comparison_cases(manifest_path)
    selected = set(case_names)
    benchmark_plan = plan or WellThroughputBenchmarkPlan.from_axes(
        well_counts=well_counts,
        worker_counts=worker_counts,
    )
    native_baselines = dict(native_execution_baselines or {})
    results: list[WellThroughputResult] = [
        result
        for result in existing_results
        if result.is_successful()
        and not (
            rerun_missing_memory
            and result.peak_memory_mb is None
        )
    ]
    completed = {
        WellThroughputObservationKey(result.case_name, result.mode_name)
        for result in results
    }
    skipped = set(skipped_observations)
    for case in cases:
        if selected and case.name not in selected:
            continue
        for mode in benchmark_plan.modes:
            observation_key = WellThroughputObservationKey(case.name, mode.name)
            if observation_key in completed or observation_key in skipped:
                continue
            result = run_case_well_throughput(
                case_name=case.name,
                dataset_path=case.dataset_path,
                cppipe_path=case.cppipe_path,
                pipeline_params=case.pipeline_params,
                output_root=(
                    output_root
                    / case.name
                    / f"wells_{mode.well_count}"
                    / f"workers_{mode.worker_count}"
                ),
                mode=mode,
                start_method=start_method,
                native_execution_baseline=native_baselines.get(case.name),
                max_memory_mb=max_memory_mb,
            )
            results.append(result)
            completed.add(observation_key)
            write_well_throughput_csv(output_root / WELL_THROUGHPUT_ROWS_CSV, results)
    return tuple(results)


def run_case_well_throughput(
    *,
    case_name: str,
    dataset_path: Path,
    cppipe_path: Path,
    pipeline_params: Mapping[str, object],
    output_root: Path,
    mode: WellThroughputMode,
    start_method: MultiprocessingStartMethod = MultiprocessingStartMethod.FORK,
    native_execution_baseline: NativeCellProfilerExecutionBaseline | None = None,
    max_memory_mb: float | None = None,
) -> WellThroughputResult:
    """Run one converted cppipe over synthetic wells in a single OpenHCS execution."""
    output_root.mkdir(parents=True, exist_ok=True)
    generated_module_path = output_root / f"{cppipe_path.stem}_openhcs.py"
    ingestion = prepare_cellprofiler_source_schema_workspace(
        CellProfilerSourceSchemaWorkspaceRequest(
            source_root=dataset_path,
            cppipe_path=cppipe_path,
            workspace_root=(
                output_root
                / f"{dataset_path.name}_{cppipe_path.stem}_source_workspace"
            ),
            generated_pipeline_path=generated_module_path,
            prune_dead_unmaterialized_artifact_steps=True,
            materialize_skipped_save_images=False,
            materialize_terminal_images=False,
            force_materialization=True,
        )
    )
    prepared = ingestion.prepared_pipeline
    if prepared.source_schema.is_empty:
        raise ValueError(
            f"Case {case_name} has no source schema; synthetic well expansion requires source-schema input."
        )
    source_workspace_path = ingestion.source_workspace_path
    if source_workspace_path is None:
        raise RuntimeError("Forced source-schema materialization returned no workspace.")
    well_ids = expand_source_schema_workspace_wells(
        source_workspace_path / "openhcs_metadata.json",
        _synthetic_well_ids(mode.well_count),
    )

    global_config = GlobalPipelineConfig(
        num_workers=mode.worker_count,
        use_threading=mode.use_threading,
        multiprocessing_start_method=start_method,
        analysis_consolidation_config=AnalysisConsolidationConfig(enabled=False),
        materialize_runtime_artifacts=False,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = replace(
        prepared.generated_pipeline.pipeline_config,
        well_filter_config=LazyWellFilterConfig(well_filter=list(well_ids)),
        path_planning_config=LazyPathPlanningConfig(
            global_output_folder=output_root,
            output_dir_suffix="_well_throughput",
        ),
        vfs_config=VFSConfig(materialization_backend=MaterializationBackend.DISK),
    )
    orchestrator = PipelineOrchestrator(
        source_workspace_path,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    progress_events: list[dict[str, Any]] = []
    progress_queue = multiprocessing.get_context(
        global_config.multiprocessing_start_method.value
    ).Queue()
    consumer = threading.Thread(
        target=_drain_progress_queue,
        args=(progress_queue, progress_events),
        daemon=True,
    )
    consumer.start()
    progress_context = {
        "execution_id": f"well-throughput::{case_name}::{time.time_ns()}",
        "plate_id": str(source_workspace_path),
        "axis_id": "",
    }

    compile_seconds = 0.0
    prepare_seconds = 0.0
    execute_seconds = 0.0
    execution_results: Mapping[Any, Any] = {}
    started_at = time.perf_counter()
    with MemoryMetric(
        interval_seconds=0.05,
        include_children=True,
        max_memory_mb=max_memory_mb,
        on_limit_exceeded=(
            ChildProcessTerminator()
            if max_memory_mb is not None
            else None
        ),
    ) as memory_metric:
        try:
            try:
                compile_started_at = time.perf_counter()
                set_progress_queue(progress_queue)
                try:
                    compilation = orchestrator.compile_pipelines(
                        pipeline_definition=prepared.runtime_pipeline_steps,
                    )
                finally:
                    set_progress_queue(None)
                compile_seconds = time.perf_counter() - compile_started_at

                execution_bundle = compilation["execution_bundle"]
                compiled_contexts = execution_bundle.runtime_contexts
                pipeline_definition = compilation.get(
                    "pipeline_definition",
                    prepared.runtime_pipeline_steps,
                )

                execute_started_at = time.perf_counter()
                execution_results = orchestrator.execute_compiled_plate(
                    pipeline_definition=pipeline_definition,
                    compiled_contexts=compiled_contexts,
                    execution_bundle=execution_bundle,
                    progress_queue=progress_queue,
                    progress_context=progress_context,
                    runtime_observation_mode=RuntimeObservationMode.OMIT,
                )
                execute_seconds = time.perf_counter() - execute_started_at
            except KeyboardInterrupt as exc:
                peak_memory_mb = memory_metric.get_result()
                total_seconds = time.perf_counter() - started_at
                if memory_metric.limit_exceeded and max_memory_mb is not None:
                    return WellThroughputResult.memory_limited(
                        case_name=case_name,
                        mode=mode,
                        compile_seconds=compile_seconds,
                        prepare_seconds=prepare_seconds,
                        execute_seconds=execute_seconds,
                        total_seconds=total_seconds,
                        peak_memory_mb=peak_memory_mb,
                        memory_limit_mb=max_memory_mb,
                        native_execution_baseline=native_execution_baseline,
                        error_message=(
                            f"Process-tree RSS exceeded {max_memory_mb:.1f} MB."
                        ),
                    )
                raise
            except Exception as exc:
                peak_memory_mb = memory_metric.get_result()
                total_seconds = time.perf_counter() - started_at
                if memory_metric.limit_exceeded and max_memory_mb is not None:
                    return WellThroughputResult.memory_limited(
                        case_name=case_name,
                        mode=mode,
                        compile_seconds=compile_seconds,
                        prepare_seconds=prepare_seconds,
                        execute_seconds=execute_seconds,
                        total_seconds=total_seconds,
                        peak_memory_mb=peak_memory_mb,
                        memory_limit_mb=max_memory_mb,
                        native_execution_baseline=native_execution_baseline,
                        error_message=str(exc),
                    )
                return WellThroughputResult.failed(
                    case_name=case_name,
                    mode=mode,
                    compile_seconds=compile_seconds,
                    prepare_seconds=prepare_seconds,
                    execute_seconds=execute_seconds,
                    total_seconds=total_seconds,
                    peak_memory_mb=peak_memory_mb,
                    native_execution_baseline=native_execution_baseline,
                    error_message=str(exc),
                )
        finally:
            progress_queue.put(None)
            consumer.join(timeout=5.0)
            progress_queue.close()
            progress_queue.join_thread()
    peak_memory_mb = memory_metric.get_result()

    total_seconds = time.perf_counter() - started_at
    if memory_metric.limit_exceeded and max_memory_mb is not None:
        return WellThroughputResult.memory_limited(
            case_name=case_name,
            mode=mode,
            compile_seconds=compile_seconds,
            prepare_seconds=prepare_seconds,
            execute_seconds=execute_seconds,
            total_seconds=total_seconds,
            peak_memory_mb=peak_memory_mb,
            memory_limit_mb=max_memory_mb,
            native_execution_baseline=native_execution_baseline,
            error_message=(
                f"Process-tree RSS exceeded {max_memory_mb:.1f} MB."
            ),
        )
    successful_wells = sum(
        1
        for result in execution_results.values()
        if _execution_result_succeeded(result)
    )
    _write_progress_diagnostics(
        output_root,
        case_name=case_name,
        worker_count=mode.worker_count,
        well_count=mode.well_count,
        events=progress_events,
    )
    projected_native_execution_seconds = (
        native_execution_baseline.projected_execution_seconds(mode.well_count)
        if native_execution_baseline is not None
        else None
    )
    return WellThroughputResult(
        case_name=case_name,
        mode_name=mode.name,
        worker_count=mode.worker_count,
        well_count=mode.well_count,
        compile_seconds=compile_seconds,
        prepare_seconds=prepare_seconds,
        execute_seconds=execute_seconds,
        total_seconds=total_seconds,
        wells_per_second=(
            mode.well_count / execute_seconds if execute_seconds > 0.0 else 0.0
        ),
        successful_wells=successful_wells,
        native_single_sample_execution_seconds=(
            native_execution_baseline.execution_seconds
            if native_execution_baseline is not None
            else None
        ),
        projected_native_execution_seconds=projected_native_execution_seconds,
        projected_execution_speedup=(
            projected_native_execution_seconds / execute_seconds
            if projected_native_execution_seconds is not None and execute_seconds > 0.0
            else None
        ),
        peak_memory_mb=peak_memory_mb,
    )


def native_execution_baselines_from_summary_csv(
    path: Path,
) -> Mapping[str, NativeCellProfilerExecutionBaseline]:
    """Load single-sample native CP execution baselines from official summary CSV."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    baselines: dict[str, NativeCellProfilerExecutionBaseline] = {}
    for row in rows:
        case_name = row.get(CASE_NAME_FIELD)
        execution_seconds = row.get(MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD)
        if not case_name or execution_seconds in (None, ""):
            continue
        baselines[case_name] = NativeCellProfilerExecutionBaseline(
            case_name=case_name,
            execution_seconds=float(execution_seconds),
        )
    return baselines


def well_throughput_plan_from_manifest(
    manifest_path: Path,
) -> WellThroughputBenchmarkPlan | None:
    """Load optional well-throughput modes declared by a comparison manifest."""
    manifest = ComparisonManifest.load(manifest_path)
    raw_modes = manifest.payload.get("well_throughput_modes")
    if raw_modes is None:
        return None
    if not isinstance(raw_modes, Sequence) or isinstance(raw_modes, str):
        raise ValueError("Manifest well_throughput_modes must be a sequence.")
    return WellThroughputBenchmarkPlan.from_presets(
        tuple(WellThroughputPreset(str(raw_mode)) for raw_mode in raw_modes)
    )


def write_well_throughput_csv(
    path: Path,
    rows: Sequence[WellThroughputResult],
) -> None:
    """Write well-level throughput observations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_well_throughput_csv(path: Path) -> tuple[WellThroughputResult, ...]:
    """Read existing throughput observations for resumable benchmark runs."""
    path = Path(path)
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    return tuple(_well_throughput_result_from_row(row) for row in rows)


def generate_well_throughput_figures(
    csv_path: Path,
    output_dir: Path,
    *,
    output_formats: Sequence[str] = ("png", "svg"),
) -> tuple[Path, ...]:
    """Generate well-throughput speedup figures from ``well_throughput.csv``."""
    rows = tuple(row for row in read_well_throughput_csv(csv_path) if row.is_successful())
    if not rows:
        return ()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.ticker import LogLocator
    from matplotlib.ticker import NullFormatter
    from matplotlib.ticker import NullLocator

    from benchmark.reports.cppipe_figures import FIGURE_STYLE
    from benchmark.reports.cppipe_figures import LINEAR_AXIS_BREAK_POLICY
    from benchmark.reports.cppipe_figures import SPEEDUP_TARGET

    output_dir.mkdir(parents=True, exist_ok=True)
    case_names = tuple(dict.fromkeys(row.case_name for row in rows))
    mode_names = WELL_THROUGHPUT_MODE_ORDER.order(row.mode_name for row in rows)
    row_index = {(row.case_name, row.mode_name): row for row in rows}
    outputs: list[Path] = []

    def values_for_mode(mode_name: str) -> tuple[float | None, ...]:
        return tuple(
            (
                row.projected_execution_speedup
                if (row := row_index.get((case_name, mode_name))) is not None
                else None
            )
            for case_name in case_names
        )

    speedup_values = tuple(
        value
        for mode_name in mode_names
        for value in values_for_mode(mode_name)
        if value is not None
    )
    broken_range = LINEAR_AXIS_BREAK_POLICY.range_for(speedup_values)

    with FIGURE_STYLE.context():
        if broken_range is None:
            fig, axis = plt.subplots(
                1,
                1,
                figsize=(max(8.0, len(case_names) * 0.62), 4.8),
                layout="constrained",
            )
            plot_axes = (axis,)
        else:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(max(8.0, len(case_names) * 0.62), 5.8),
                gridspec_kw={"height_ratios": (1.0, 3.2)},
                sharex=True,
                layout="constrained",
            )
            top_axis, bottom_axis = tuple(axes)
            top_axis.set_ylim(broken_range[1], broken_range[2])
            bottom_axis.set_ylim(0.0, broken_range[0])
            LINEAR_AXIS_BREAK_POLICY.mark(top_axis, bottom_axis)
            plot_axes = (top_axis, bottom_axis)

        width = min(0.18, 0.82 / max(len(mode_names), 1))
        offsets = _bar_offsets(len(mode_names), width)
        x_positions = tuple(range(len(case_names)))
        for axis in plot_axes:
            for mode_index, mode_name in enumerate(mode_names):
                values = [
                    value if value is not None else float("nan")
                    for value in values_for_mode(mode_name)
                ]
                axis.bar(
                    [x + offsets[mode_index] for x in x_positions],
                    values,
                    width=width,
                    label=mode_name if axis is plot_axes[0] else None,
                    color=FIGURE_STYLE.color_for_method(mode_index + 1),
                    edgecolor=FIGURE_STYLE.background,
                    linewidth=0.55,
                )
            axis.axhline(
                SPEEDUP_TARGET,
                color=FIGURE_STYLE.target_color,
                linewidth=1.15,
                linestyle="--",
                alpha=0.86,
            )
            axis.grid(axis="y", color=FIGURE_STYLE.grid_color, linewidth=0.8, alpha=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(FIGURE_STYLE.spine_color)
            axis.spines["bottom"].set_color(FIGURE_STYLE.spine_color)

        label_axis = plot_axes[-1]
        label_axis.set_xticks(list(x_positions))
        label_axis.set_xticklabels(case_names, rotation=42, ha="right", fontsize=7.2)
        label_axis.set_ylabel("Projected speedup vs CP (x)")
        plot_axes[0].set_title("OpenHCS well-throughput scaling", loc="left", pad=10)
        plot_axes[0].legend(frameon=False, ncol=min(len(mode_names), 4), loc="upper left")

        for output_format in output_formats:
            output_path = output_dir / f"well_throughput_speedup.{output_format}"
            fig.savefig(output_path, dpi=360, bbox_inches="tight")
            outputs.append(output_path)
        plt.close(fig)

    with FIGURE_STYLE.context():
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(max(8.0, len(case_names) * 0.62), 4.8),
            layout="constrained",
        )
        width = min(0.18, 0.82 / max(len(mode_names), 1))
        offsets = _bar_offsets(len(mode_names), width)
        x_positions = tuple(range(len(case_names)))
        for mode_index, mode_name in enumerate(mode_names):
            axis.bar(
                [x + offsets[mode_index] for x in x_positions],
                [
                    value if value is not None else float("nan")
                    for value in values_for_mode(mode_name)
                ],
                width=width,
                label=mode_name,
                color=FIGURE_STYLE.color_for_method(mode_index + 1),
                edgecolor=FIGURE_STYLE.background,
                linewidth=0.55,
            )
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_major_formatter(FuncFormatter(_plain_numeric_tick_label))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.axhline(
            SPEEDUP_TARGET,
            color=FIGURE_STYLE.target_color,
            linewidth=1.15,
            linestyle="--",
            alpha=0.86,
        )
        axis.grid(axis="y", color=FIGURE_STYLE.grid_color, linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title("OpenHCS well-throughput scaling (log)", loc="left", pad=10)
        axis.set_ylabel("Projected speedup vs CP (x)")
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(case_names, rotation=42, ha="right", fontsize=7.2)
        axis.legend(frameon=False, ncol=min(len(mode_names), 4), loc="upper left")
        for output_format in output_formats:
            output_path = output_dir / f"well_throughput_speedup_log.{output_format}"
            fig.savefig(output_path, dpi=360, bbox_inches="tight")
            outputs.append(output_path)
        plt.close(fig)

    average_csv = output_dir / "well_throughput_average_speedup_points.csv"
    _write_well_throughput_average_speedup_csv(average_csv, rows, mode_names)
    outputs.append(average_csv)
    from benchmark.reports.cppipe_figures import SpeedupDistributionSeries
    from benchmark.reports.cppipe_figures import generate_speedup_distribution_artifacts

    outputs.extend(
        generate_speedup_distribution_artifacts(
            tuple(
                SpeedupDistributionSeries(
                    mode_name,
                    tuple(
                        float(row.projected_execution_speedup)
                        for row in rows
                        if row.mode_name == mode_name
                        and row.projected_execution_speedup is not None
                    ),
                )
                for mode_name in mode_names
            ),
            output_dir=output_dir,
            filename_prefix="well_throughput_speedup",
            title="Well-throughput speedup cumulative distribution",
            xlabel="Projected speedup versus CP (x)",
            output_formats=output_formats,
        )
    )
    outputs.extend(
        _plot_well_throughput_average_speedup_points(
            rows,
            mode_names=mode_names,
            output_dir=output_dir,
            output_formats=output_formats,
        )
    )
    outputs.extend(
        _plot_well_throughput_ram(
            rows,
            case_names=case_names,
            mode_names=mode_names,
            output_dir=output_dir,
            output_formats=output_formats,
        )
    )
    return tuple(outputs)


def _well_throughput_result_from_row(
    row: Mapping[str, str],
) -> WellThroughputResult:
    return WellThroughputResult(
        case_name=row["case_name"],
        mode_name=row["mode_name"],
        worker_count=int(row["worker_count"]),
        well_count=int(row["well_count"]),
        compile_seconds=float(row["compile_seconds"]),
        prepare_seconds=float(row["prepare_seconds"]),
        execute_seconds=float(row["execute_seconds"]),
        total_seconds=float(row["total_seconds"]),
        wells_per_second=float(row["wells_per_second"]),
        successful_wells=int(row["successful_wells"]),
        native_single_sample_execution_seconds=_optional_float(
            row.get("native_single_sample_execution_seconds")
        ),
        projected_native_execution_seconds=_optional_float(
            row.get("projected_native_execution_seconds")
        ),
        projected_execution_speedup=_optional_float(
            row.get("projected_execution_speedup")
        ),
        peak_memory_mb=_optional_float(row.get("peak_memory_mb")),
        status=WellThroughputStatus(row.get("status") or WellThroughputStatus.SUCCESS),
        memory_limit_mb=_optional_float(row.get("memory_limit_mb")),
        error_message=row.get("error_message") or None,
    )


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _mean_present(values: Iterable[float | None]) -> float | None:
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    return sum(present) / len(present)


def _plain_numeric_tick_label(value: float, position: int) -> str:
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


def _bar_offsets(method_count: int, width: float) -> tuple[float, ...]:
    center = (method_count - 1) / 2.0
    return tuple((index - center) * width for index in range(method_count))


def _write_well_throughput_average_speedup_csv(
    path: Path,
    rows: Sequence[WellThroughputResult],
    mode_names: Sequence[str],
) -> None:
    fieldnames = (
        "mode_name",
        "case_name",
        "projected_execution_speedup",
        "mean_speedup",
        "sample_count",
    )
    by_mode = {
        mode_name: tuple(
            row
            for row in rows
            if row.mode_name == mode_name and row.projected_execution_speedup is not None
        )
        for mode_name in mode_names
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode_name, mode_rows in by_mode.items():
            mean_speedup = (
                sum(float(row.projected_execution_speedup) for row in mode_rows)
                / len(mode_rows)
                if mode_rows
                else None
            )
            for row in mode_rows:
                writer.writerow(
                    {
                        "mode_name": mode_name,
                        "case_name": row.case_name,
                        "projected_execution_speedup": row.projected_execution_speedup,
                        "mean_speedup": mean_speedup,
                        "sample_count": len(mode_rows),
                    }
                )


def _plot_well_throughput_average_speedup_points(
    rows: Sequence[WellThroughputResult],
    *,
    mode_names: Sequence[str],
    output_dir: Path,
    output_formats: Sequence[str],
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.ticker import LogLocator
    from matplotlib.ticker import NullFormatter
    from matplotlib.ticker import NullLocator

    from benchmark.reports.cppipe_figures import FIGURE_STYLE
    from benchmark.reports.cppipe_figures import LINEAR_AXIS_BREAK_POLICY
    from benchmark.reports.cppipe_figures import SPEEDUP_TARGET

    mode_rows = tuple(
        (
            mode_name,
            tuple(
                row
                for row in rows
                if row.mode_name == mode_name
                and row.projected_execution_speedup is not None
            ),
        )
        for mode_name in mode_names
    )
    mode_rows = tuple((mode_name, rows_) for mode_name, rows_ in mode_rows if rows_)
    if not mode_rows:
        return ()

    values = tuple(
        float(row.projected_execution_speedup)
        for _mode_name, rows_ in mode_rows
        for row in rows_
        if row.projected_execution_speedup is not None
    )
    broken_range = LINEAR_AXIS_BREAK_POLICY.range_for(values)
    with FIGURE_STYLE.context():
        if broken_range is None:
            fig, axis = plt.subplots(
                1,
                1,
                figsize=(max(5.2, 1.45 * len(mode_rows) + 3.2), 4.6),
                layout="constrained",
            )
            plot_axes = (axis,)
        else:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(max(5.2, 1.45 * len(mode_rows) + 3.2), 5.6),
                gridspec_kw={"height_ratios": (1.0, 3.2)},
                sharex=True,
                layout="constrained",
            )
            top_axis, bottom_axis = tuple(axes)
            top_axis.set_ylim(broken_range[1], broken_range[2])
            bottom_axis.set_ylim(0.0, broken_range[0])
            LINEAR_AXIS_BREAK_POLICY.mark(top_axis, bottom_axis)
            plot_axes = (top_axis, bottom_axis)

        x_positions = tuple(range(len(mode_rows)))
        for mode_index, (mode_name, rows_) in enumerate(mode_rows):
            speedups = tuple(float(row.projected_execution_speedup) for row in rows_)
            mean_speedup = sum(speedups) / len(speedups)
            standard_deviation = statistics.stdev(speedups) if len(speedups) > 1 else 0.0
            ci95 = 1.96 * standard_deviation / math.sqrt(len(speedups))
            color = FIGURE_STYLE.color_for_method(mode_index + 1)
            point_x = [
                mode_index + _deterministic_jitter(point_index, len(speedups))
                for point_index in range(len(speedups))
            ]
            for axis in plot_axes:
                axis.scatter(
                    point_x,
                    speedups,
                    s=28,
                    color=color,
                    alpha=0.76,
                    edgecolors=FIGURE_STYLE.background,
                    linewidths=0.55,
                    zorder=3,
                )
                axis.errorbar(
                    [mode_index],
                    [mean_speedup],
                    yerr=[[ci95], [ci95]],
                    fmt="o",
                    color=FIGURE_STYLE.text_color,
                    markerfacecolor=color,
                    markeredgecolor=FIGURE_STYLE.text_color,
                    markersize=8.5,
                    capsize=7,
                    elinewidth=1.4,
                    zorder=4,
                    label=f"{mode_name} mean" if axis is plot_axes[0] else None,
                )
                axis.axhline(
                    SPEEDUP_TARGET,
                    color=FIGURE_STYLE.target_color,
                    linewidth=1.15,
                    linestyle="--",
                    alpha=0.86,
                )
                axis.grid(axis="y", color=FIGURE_STYLE.grid_color, linewidth=0.8, alpha=0.8)
                axis.set_axisbelow(True)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.spines["left"].set_color(FIGURE_STYLE.spine_color)
                axis.spines["bottom"].set_color(FIGURE_STYLE.spine_color)

        label_axis = plot_axes[-1]
        label_axis.set_xticks(list(x_positions))
        label_axis.set_xticklabels([mode_name for mode_name, _rows in mode_rows])
        label_axis.set_xlim(-0.6, len(mode_rows) - 0.4)
        label_axis.set_ylabel("Projected speedup vs CP (x)")
        plot_axes[0].set_title("Average well-throughput speedup", loc="left", pad=10)
        plot_axes[0].legend(frameon=False, loc="upper left")

        outputs: list[Path] = []
        for output_format in output_formats:
            output_path = output_dir / f"well_throughput_average_speedup_points.{output_format}"
            fig.savefig(output_path, dpi=360, bbox_inches="tight")
            outputs.append(output_path)
        plt.close(fig)

    with FIGURE_STYLE.context():
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(max(5.2, 1.45 * len(mode_rows) + 3.2), 4.6),
            layout="constrained",
        )
        x_positions = tuple(range(len(mode_rows)))
        for mode_index, (mode_name, rows_) in enumerate(mode_rows):
            speedups = tuple(float(row.projected_execution_speedup) for row in rows_)
            mean_speedup = sum(speedups) / len(speedups)
            standard_deviation = statistics.stdev(speedups) if len(speedups) > 1 else 0.0
            ci95 = 1.96 * standard_deviation / math.sqrt(len(speedups))
            color = FIGURE_STYLE.color_for_method(mode_index + 1)
            point_x = [
                mode_index + _deterministic_jitter(point_index, len(speedups))
                for point_index in range(len(speedups))
            ]
            axis.scatter(
                point_x,
                speedups,
                s=28,
                color=color,
                alpha=0.76,
                edgecolors=FIGURE_STYLE.background,
                linewidths=0.55,
                zorder=3,
            )
            axis.errorbar(
                [mode_index],
                [mean_speedup],
                yerr=[[min(ci95, mean_speedup * 0.95)], [ci95]],
                fmt="o",
                color=FIGURE_STYLE.text_color,
                markerfacecolor=color,
                markeredgecolor=FIGURE_STYLE.text_color,
                markersize=8.5,
                capsize=7,
                elinewidth=1.4,
                zorder=4,
                label=f"{mode_name} mean",
            )
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_major_formatter(FuncFormatter(_plain_numeric_tick_label))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.axhline(
            SPEEDUP_TARGET,
            color=FIGURE_STYLE.target_color,
            linewidth=1.15,
            linestyle="--",
            alpha=0.86,
        )
        axis.grid(axis="y", color=FIGURE_STYLE.grid_color, linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(FIGURE_STYLE.spine_color)
        axis.spines["bottom"].set_color(FIGURE_STYLE.spine_color)
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels([mode_name for mode_name, _rows in mode_rows])
        axis.set_xlim(-0.6, len(mode_rows) - 0.4)
        axis.set_ylabel("Projected speedup vs CP (x)")
        axis.set_title("Average well-throughput speedup (log)", loc="left", pad=10)
        axis.legend(frameon=False, loc="upper left")
        for output_format in output_formats:
            output_path = output_dir / f"well_throughput_average_speedup_points_log.{output_format}"
            fig.savefig(output_path, dpi=360, bbox_inches="tight")
            outputs.append(output_path)
        plt.close(fig)
        return tuple(outputs)


def _plot_well_throughput_ram(
    rows: Sequence[WellThroughputResult],
    *,
    case_names: Sequence[str],
    mode_names: Sequence[str],
    output_dir: Path,
    output_formats: Sequence[str],
) -> tuple[Path, ...]:
    if not any(row.peak_memory_mb is not None for row in rows):
        return ()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.ticker import LogLocator
    from matplotlib.ticker import NullFormatter
    from matplotlib.ticker import NullLocator

    from benchmark.reports.cppipe_figures import FIGURE_STYLE

    row_index = {(row.case_name, row.mode_name): row for row in rows}

    def values_for_mode(mode_name: str) -> tuple[float | None, ...]:
        return tuple(
            (
                row.peak_memory_mb
                if (row := row_index.get((case_name, mode_name))) is not None
                else None
            )
            for case_name in case_names
        )

    outputs: list[Path] = []
    with FIGURE_STYLE.context():
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(max(8.0, len(case_names) * 0.62), 4.8),
            layout="constrained",
        )
        plot_axes = (axis,)

        width = min(0.18, 0.82 / max(len(mode_names), 1))
        offsets = _bar_offsets(len(mode_names), width)
        x_positions = tuple(range(len(case_names)))
        for axis in plot_axes:
            for mode_index, mode_name in enumerate(mode_names):
                axis.bar(
                    [x + offsets[mode_index] for x in x_positions],
                    [
                        value if value is not None else float("nan")
                        for value in values_for_mode(mode_name)
                    ],
                    width=width,
                    label=mode_name if axis is plot_axes[0] else None,
                    color=FIGURE_STYLE.color_for_method(mode_index + 1),
                    edgecolor=FIGURE_STYLE.background,
                    linewidth=0.55,
                )
            axis.grid(axis="y", color=FIGURE_STYLE.grid_color, linewidth=0.8, alpha=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(FIGURE_STYLE.spine_color)
            axis.spines["bottom"].set_color(FIGURE_STYLE.spine_color)

        label_axis = plot_axes[-1]
        label_axis.set_xticks(list(x_positions))
        label_axis.set_xticklabels(case_names, rotation=42, ha="right", fontsize=7.2)
        label_axis.set_ylabel("Peak process-tree RSS (MB)")
        plot_axes[0].set_title("OpenHCS well-throughput RAM by core mode", loc="left", pad=10)
        plot_axes[0].legend(frameon=False, ncol=min(len(mode_names), 4), loc="upper left")
        for output_format in output_formats:
            output_path = output_dir / f"well_throughput_peak_memory.{output_format}"
            fig.savefig(output_path, dpi=360, bbox_inches="tight")
            outputs.append(output_path)
        plt.close(fig)

    with FIGURE_STYLE.context():
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(max(8.0, len(case_names) * 0.62), 4.8),
            layout="constrained",
        )
        width = min(0.18, 0.82 / max(len(mode_names), 1))
        offsets = _bar_offsets(len(mode_names), width)
        x_positions = tuple(range(len(case_names)))
        for mode_index, mode_name in enumerate(mode_names):
            axis.bar(
                [x + offsets[mode_index] for x in x_positions],
                [
                    value if value is not None else float("nan")
                    for value in values_for_mode(mode_name)
                ],
                width=width,
                label=mode_name,
                color=FIGURE_STYLE.color_for_method(mode_index + 1),
                edgecolor=FIGURE_STYLE.background,
                linewidth=0.55,
            )
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_major_formatter(FuncFormatter(_plain_numeric_tick_label))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(axis="y", color=FIGURE_STYLE.grid_color, linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title("OpenHCS well-throughput RAM by core mode (log)", loc="left", pad=10)
        axis.set_ylabel("Peak process-tree RSS (MB)")
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(case_names, rotation=42, ha="right", fontsize=7.2)
        axis.legend(frameon=False, ncol=min(len(mode_names), 4), loc="upper left")
        for output_format in output_formats:
            output_path = output_dir / f"well_throughput_peak_memory_log.{output_format}"
            fig.savefig(output_path, dpi=360, bbox_inches="tight")
            outputs.append(output_path)
        plt.close(fig)
    return tuple(outputs)


def _deterministic_jitter(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    spread = 0.18
    return ((index / (count - 1)) - 0.5) * spread


def _synthetic_well_ids(count: int) -> tuple[str, ...]:
    return tuple(f"W{index:03d}" for index in range(1, count + 1))


def _execution_result_succeeded(result: Any) -> bool:
    if isinstance(result, WellThroughputResult):
        return result.is_successful()
    is_success = getattr(result, "is_success", None)
    if callable(is_success):
        return bool(is_success())
    status = getattr(result, "status", None)
    if status is not None:
        status_value = getattr(status, "value", status)
        return str(status_value).lower() == "success"
    if isinstance(result, dict):
        raw_status = result.get("status")
        status_value = getattr(raw_status, "value", raw_status)
        return status_value is None or str(status_value).lower() == "success"
    return True


def _drain_progress_queue(
    progress_queue,
    progress_events: list[dict[str, Any]],
) -> None:
    while True:
        try:
            item = progress_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            return
        if isinstance(item, dict):
            progress_events.append(item)


def _write_progress_diagnostics(
    output_root: Path,
    *,
    case_name: str,
    worker_count: int,
    well_count: int,
    events: Sequence[dict[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_progress_events_csv(
        output_root / WELL_THROUGHPUT_EVENTS_CSV,
        case_name=case_name,
        worker_count=worker_count,
        well_count=well_count,
        events=events,
    )
    _write_worker_lane_csv(
        output_root / WELL_THROUGHPUT_LANES_CSV,
        case_name=case_name,
        worker_count=worker_count,
        well_count=well_count,
        events=events,
    )
    _write_step_timings_csv(
        output_root / WELL_THROUGHPUT_STEPS_CSV,
        case_name=case_name,
        worker_count=worker_count,
        well_count=well_count,
        events=events,
    )


def _write_progress_events_csv(
    path: Path,
    *,
    case_name: str,
    worker_count: int,
    well_count: int,
    events: Sequence[dict[str, Any]],
) -> None:
    fieldnames = (
        "case_name",
        "worker_count",
        "well_count",
        "timestamp",
        "pid",
        "worker_slot",
        "axis_id",
        "step_name",
        "phase",
        "status",
        "percent",
        "completed",
        "total",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "case_name": case_name,
                    "worker_count": worker_count,
                    "well_count": well_count,
                    "timestamp": event.get("timestamp"),
                    "pid": event.get("pid"),
                    "worker_slot": event.get("worker_slot"),
                    "axis_id": event.get("axis_id"),
                    "step_name": event.get("step_name"),
                    "phase": event.get("phase"),
                    "status": event.get("status"),
                    "percent": event.get("percent"),
                    "completed": event.get("completed"),
                    "total": event.get("total"),
                }
            )


def _write_worker_lane_csv(
    path: Path,
    *,
    case_name: str,
    worker_count: int,
    well_count: int,
    events: Sequence[dict[str, Any]],
) -> None:
    lanes: dict[str, dict[str, Any]] = {}
    for event in events:
        worker_slot = event.get("worker_slot")
        if not worker_slot:
            continue
        lane = lanes.setdefault(
            str(worker_slot),
            {
                "case_name": case_name,
                "worker_count": worker_count,
                "well_count": well_count,
                "worker_slot": worker_slot,
                "axis_count": 0,
                "started_at": "",
                "completed_at": "",
                "lane_seconds": "",
            },
        )
        phase = WorkerLaneEventPhase.from_event(event)
        if phase is None:
            continue
        timestamp = float(event["timestamp"])
        phase.apply_to_lane(lane, timestamp)

    for lane in lanes.values():
        if lane["started_at"] != "" and lane["completed_at"] != "":
            lane["lane_seconds"] = float(lane["completed_at"]) - float(
                lane["started_at"]
            )

    fieldnames = (
        "case_name",
        "worker_count",
        "well_count",
        "worker_slot",
        "axis_count",
        "started_at",
        "completed_at",
        "lane_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(lanes.values(), key=lambda value: str(value["worker_slot"])):
            writer.writerow(row)


def _write_step_timings_csv(
    path: Path,
    *,
    case_name: str,
    worker_count: int,
    well_count: int,
    events: Sequence[dict[str, Any]],
) -> None:
    started: dict[tuple[str, str, str], float] = {}
    rows: list[dict[str, Any]] = []
    for event in events:
        phase = event.get("phase")
        if phase not in {"step_started", "step_completed"}:
            continue
        key = (
            str(event.get("worker_slot", "")),
            str(event.get("axis_id", "")),
            str(event.get("step_name", "")),
        )
        timestamp = float(event["timestamp"])
        if phase == "step_started":
            started[key] = timestamp
            continue
        start_timestamp = started.pop(key, None)
        if start_timestamp is None:
            continue
        rows.append(
            {
                "case_name": case_name,
                "worker_count": worker_count,
                "well_count": well_count,
                "worker_slot": key[0],
                "axis_id": key[1],
                "step_name": key[2],
                "started_at": start_timestamp,
                "completed_at": timestamp,
                "step_seconds": timestamp - start_timestamp,
            }
        )

    fieldnames = (
        "case_name",
        "worker_count",
        "well_count",
        "worker_slot",
        "axis_id",
        "step_name",
        "started_at",
        "completed_at",
        "step_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
