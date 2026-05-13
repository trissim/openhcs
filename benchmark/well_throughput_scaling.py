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
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import psutil

from benchmark.cellprofiler_comparison import CASE_NAME_FIELD
from benchmark.cellprofiler_comparison import load_comparison_cases
from benchmark.cellprofiler_comparison import MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD
from benchmark.adapters.openhcs import OpenHCSAxisSelection
from benchmark.contracts.comparison_manifest import ComparisonManifest
from benchmark.metrics.memory import MemoryMetric
from openhcs.interop.cellprofiler.runtime_pipeline import prepare_generated_pipeline
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.constants.constants import Microscope
from openhcs.core.config import (
    AnalysisConsolidationConfig,
    GlobalPipelineConfig,
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
    materialize_source_schema_workspace,
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
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=generated_module_path,
        prune_dead_unmaterialized_artifact_steps=True,
        materialize_skipped_save_images=False,
    )
    if prepared.source_schema.is_empty:
        raise ValueError(
            f"Case {case_name} has no source schema; synthetic well expansion requires source-schema input."
        )

    axis_selection = OpenHCSAxisSelection.from_pipeline_params(pipeline_params)
    source_workspace = materialize_source_schema_workspace(
        dataset_path,
        output_root / f"{dataset_path.name}_{cppipe_path.stem}_source_workspace",
        prepared.source_schema,
        image_set_selection=axis_selection.source_schema_selection(),
    )
    well_ids = expand_source_schema_workspace_wells(
        source_workspace.metadata_path,
        _synthetic_well_ids(mode.well_count),
    )

    global_config = GlobalPipelineConfig(
        num_workers=mode.worker_count,
        use_threading=mode.use_threading,
        multiprocessing_start_method=start_method,
        analysis_consolidation_config=AnalysisConsolidationConfig(enabled=False),
        materialize_runtime_artifacts=False,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            global_output_folder=output_root,
            output_dir_suffix="_well_throughput",
        ),
        vfs_config=VFSConfig(materialization_backend=MaterializationBackend.DISK),
    )
    orchestrator = PipelineOrchestrator(
        source_workspace.workspace_root,
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
        "plate_id": str(source_workspace.workspace_root),
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
                        pipeline_definition=prepared.pipeline.steps,
                        well_filter=list(well_ids),
                    )
                finally:
                    set_progress_queue(None)
                compile_seconds = time.perf_counter() - compile_started_at

                execution_bundle = compilation["execution_bundle"]
                compiled_contexts = execution_bundle.runtime_contexts
                pipeline_definition = compilation.get(
                    "pipeline_definition",
                    prepared.pipeline.steps,
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
    mode_names = tuple(dict.fromkeys(row.mode_name for row in rows))
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
    from benchmark.reports.cppipe_figures import LINEAR_AXIS_BREAK_POLICY

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

    ram_values = tuple(
        value
        for mode_name in mode_names
        for value in values_for_mode(mode_name)
        if value is not None
    )
    broken_range = LINEAR_AXIS_BREAK_POLICY.range_for(ram_values)
    outputs: list[Path] = []
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
