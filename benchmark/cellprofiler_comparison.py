"""CellProfiler versus OpenHCS benchmark result collection."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nominal_refactor_advisor.record_algebra import product_record

from benchmark.contracts.tool_adapter import BenchmarkResult
from benchmark.contracts.tool_adapter import ToolExecutionError
from benchmark.adapters.cellprofiler import native_cellprofiler_reference_is_complete
from benchmark.adapters.openhcs import OPENHCS_AXIS_FILTER_PARAM
from benchmark.adapters.openhcs import OPENHCS_MAX_AXIS_COUNT_PARAM
from benchmark.datasets.visible_source import resolve_visible_source_path
from benchmark.metrics.memory import MemoryMetric
from benchmark.metrics.time import TimeMetric
from benchmark.runner import CellProfilerCompatibilityResult
from benchmark.runner import run_cellprofiler_cppipe_parity


BENCHMARK_CACHE_DOMAINS = frozenset({"harness"})
SUITE_ID_FIELD = "suite_id"
CASE_NAME_FIELD = "case_name"
REPETITION_FIELD = "repetition"
DATASET_ID_FIELD = "dataset_id"
ASSAY_CATEGORY_FIELD = "assay_category"
MODULE_CATEGORY_FIELD = "module_category"
EQUIVALENT_FIELD = "equivalent"
DIFFERENCE_COUNT_FIELD = "difference_count"
NUMERIC_ABS_TOLERANCE_FIELD = "numeric_abs_tolerance"
NUMERIC_REL_TOLERANCE_FIELD = "numeric_rel_tolerance"
NATIVE_EXECUTION_SECONDS_FIELD = "native_execution_seconds"
OPENHCS_EXECUTION_SECONDS_FIELD = "openhcs_execution_seconds"
NATIVE_TOTAL_PHASE_SECONDS_FIELD = "native_total_phase_seconds"
OPENHCS_TOTAL_PHASE_SECONDS_FIELD = "openhcs_total_phase_seconds"
NATIVE_PEAK_MEMORY_MB_FIELD = "native_peak_memory_mb"
OPENHCS_PEAK_MEMORY_MB_FIELD = "openhcs_peak_memory_mb"
SPEEDUP_FIELD = "speedup"
TOTAL_PHASE_SPEEDUP_FIELD = "total_phase_speedup"
SPEEDUP_TARGET_FIELD = "speedup_target"
MEETS_SPEEDUP_TARGET_FIELD = "meets_speedup_target"
MEETS_EXECUTION_SPEEDUP_TARGET_FIELD = "meets_execution_speedup_target"
MEETS_TOTAL_PHASE_SPEEDUP_TARGET_FIELD = "meets_total_phase_speedup_target"
PARITY_ACCURACY_FIELD = "parity_accuracy"
NATIVE_CACHED_FIELD = "native_cached"
OPENHCS_CACHED_FIELD = "openhcs_cached"
NATIVE_ERROR_MESSAGE_FIELD = "native_error_message"
OPENHCS_ERROR_MESSAGE_FIELD = "openhcs_error_message"
NATIVE_OUTPUT_PATH_FIELD = "native_output_path"
OPENHCS_OUTPUT_PATH_FIELD = "openhcs_output_path"
TOOL_FIELD = "tool"
PHASE_FIELD = "phase"
SECONDS_FIELD = "seconds"
N_FIELD = "n"
EQUIVALENT_COUNT_FIELD = "equivalent_count"
MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD = "median_native_execution_seconds"
MEDIAN_OPENHCS_EXECUTION_SECONDS_FIELD = "median_openhcs_execution_seconds"
MEDIAN_NATIVE_TOTAL_PHASE_SECONDS_FIELD = "median_native_total_phase_seconds"
MEDIAN_OPENHCS_TOTAL_PHASE_SECONDS_FIELD = "median_openhcs_total_phase_seconds"
MEDIAN_NATIVE_PEAK_MEMORY_MB_FIELD = "median_native_peak_memory_mb"
MEDIAN_OPENHCS_PEAK_MEMORY_MB_FIELD = "median_openhcs_peak_memory_mb"
MEDIAN_SPEEDUP_FIELD = "median_speedup"
MEDIAN_TOTAL_PHASE_SPEEDUP_FIELD = "median_total_phase_speedup"
MIN_PARITY_ACCURACY_FIELD = "min_parity_accuracy"
DEFAULT_SPEEDUP_TARGET = 5.0
OPENHCS_BENCHMARK_CACHE_MARKER = ".openhcs_benchmark_cache.json"
CsvRow = Mapping[str, object]
CsvRowBuilder = Callable[
    [Sequence["CellProfilerComparisonObservation"]],
    Iterable[CsvRow],
]
CsvTableSpec = product_record(
    "CsvTableSpec",
    "fieldnames: tuple[str, ...]; rows: CsvRowBuilder",
    doc="Authoritative CSV table projection.",
    module_name=__name__,
)


@dataclass(frozen=True, slots=True)
class CellProfilerComparisonCase:
    """One native-CellProfiler versus OpenHCS benchmark case."""

    name: str
    dataset_path: Path
    cppipe_path: Path
    dataset_id: str | None = None
    microscope_type: str | None = None
    assay_category: str | None = None
    module_category: str | None = None
    value_only: bool = False
    equivalence_reference_output_dir: Path | None = None
    cellprofiler_timeout_seconds: float | None = None
    pipeline_params: Mapping[str, object] = field(default_factory=dict)

    @property
    def resolved_dataset_id(self) -> str:
        return self.dataset_id or self.dataset_path.name


ToolExecutionSummary = product_record(
    "ToolExecutionSummary",
    (
        "tool: str; success: bool; output_path: str; "
        "execution_seconds: float | None; total_metric_seconds: float | None; "
        "peak_memory_mb: float | None; cached: bool; error_message: str | None; "
        "phase_seconds: Mapping[str, float]"
    ),
    doc="Execution and phase timing summary for one tool run.",
    module_name=__name__,
)


@dataclass(frozen=True, slots=True)
class NativeReferenceLocation:
    """Resolved native CellProfiler reference location for a benchmark case."""

    output_dir: Path | None
    reference_output_dir: Path | None


@dataclass(frozen=True, slots=True)
class CellProfilerComparisonObservation:
    """Serializable observation for one case/repetition."""

    suite_id: str
    case_name: str
    repetition: int
    dataset_id: str
    assay_category: str | None
    module_category: str | None
    cppipe_path: str
    equivalent: bool
    difference_count: int | None
    numeric_abs_tolerance: float
    numeric_rel_tolerance: float
    native_cellprofiler: ToolExecutionSummary
    openhcs: ToolExecutionSummary
    observed_at_epoch_seconds: float = field(default_factory=time.time)

    @property
    def speedup(self) -> float | None:
        native_seconds = self.native_cellprofiler.execution_seconds
        openhcs_seconds = self.openhcs.execution_seconds
        if native_seconds is None or openhcs_seconds is None or openhcs_seconds <= 0:
            return None
        return native_seconds / openhcs_seconds

    @property
    def total_phase_speedup(self) -> float | None:
        native_seconds = self.native_cellprofiler.total_metric_seconds
        openhcs_seconds = self.openhcs.total_metric_seconds
        if native_seconds is None or openhcs_seconds is None or openhcs_seconds <= 0:
            return None
        return native_seconds / openhcs_seconds

    @property
    def parity_accuracy(self) -> float:
        return 1.0 if self.equivalent else 0.0

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["speedup"] = self.speedup
        payload["total_phase_speedup"] = self.total_phase_speedup
        payload["parity_accuracy"] = self.parity_accuracy
        return payload


@dataclass(frozen=True, slots=True)
class CachedNativeReferenceTimingPolicy:
    """Timing contract for reused native references with timeout-backed evidence."""

    case: CellProfilerComparisonCase
    summary: ToolExecutionSummary

    @property
    def has_timeout_lower_bound(self) -> bool:
        return (
            self.summary.success
            and self.summary.cached
            and self.summary.execution_seconds is None
            and self.case.cellprofiler_timeout_seconds is not None
        )

    def apply(self) -> ToolExecutionSummary:
        if not self.has_timeout_lower_bound:
            return self.summary
        timeout_seconds = float(self.case.cellprofiler_timeout_seconds)
        return ToolExecutionSummary(
            tool=self.summary.tool,
            success=self.summary.success,
            output_path=self.summary.output_path,
            execution_seconds=timeout_seconds,
            total_metric_seconds=(
                self.summary.total_metric_seconds
                if self.summary.total_metric_seconds is not None
                else timeout_seconds
            ),
            peak_memory_mb=self.summary.peak_memory_mb,
            cached=self.summary.cached,
            error_message=self.summary.error_message,
            phase_seconds=self.summary.phase_seconds,
        )


def load_comparison_cases(path: Path) -> tuple[CellProfilerComparisonCase, ...]:
    """Load benchmark cases from a JSON manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Sequence):
        raise ValueError("Benchmark manifest must contain a 'cases' sequence.")
    cases: list[CellProfilerComparisonCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"Benchmark case must be an object: {raw_case!r}")
        raw_pipeline_params = raw_case.get("pipeline_params", {})
        if not isinstance(raw_pipeline_params, Mapping):
            raise ValueError(
                f"Benchmark case pipeline_params must be an object: {raw_pipeline_params!r}"
            )
        cases.append(
            CellProfilerComparisonCase(
                name=str(raw_case["name"]),
                dataset_path=Path(str(raw_case["dataset_path"])),
                cppipe_path=Path(str(raw_case["cppipe_path"])),
                dataset_id=(
                    str(raw_case["dataset_id"])
                    if raw_case.get("dataset_id") is not None
                    else None
                ),
                microscope_type=(
                    str(raw_case["microscope_type"])
                    if raw_case.get("microscope_type") is not None
                    else None
                ),
                assay_category=(
                    str(raw_case["assay_category"])
                    if raw_case.get("assay_category") is not None
                    else None
                ),
                module_category=(
                    str(raw_case["module_category"])
                    if raw_case.get("module_category") is not None
                    else None
                ),
                value_only=bool(raw_case.get("value_only", False)),
                equivalence_reference_output_dir=(
                    Path(str(raw_case["equivalence_reference_output_dir"]))
                    if raw_case.get("equivalence_reference_output_dir") is not None
                    else None
                ),
                cellprofiler_timeout_seconds=(
                    float(raw_case["cellprofiler_timeout_seconds"])
                    if raw_case.get("cellprofiler_timeout_seconds") is not None
                    else None
                ),
                pipeline_params=dict(raw_pipeline_params),
            )
        )
    return tuple(cases)


def run_comparison_suite(
    cases: Iterable[CellProfilerComparisonCase],
    *,
    output_root: Path,
    suite_id: str,
    repeats: int = 1,
    reuse_openhcs_cache: bool = True,
    speedup_target: float = DEFAULT_SPEEDUP_TARGET,
    native_reference_root: Path | None = None,
    discard_openhcs_outputs: bool = False,
    continue_on_error: bool = False,
    openhcs_axis_filter: Sequence[str] = (),
    openhcs_max_axis_count: int | None = None,
) -> tuple[CellProfilerComparisonObservation, ...]:
    """Run all cases and write raw benchmark observations."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if speedup_target <= 0:
        raise ValueError("speedup_target must be positive.")
    if openhcs_max_axis_count is not None and openhcs_max_axis_count <= 0:
        raise ValueError("openhcs_max_axis_count must be positive.")
    output_root.mkdir(parents=True, exist_ok=True)
    observations: list[CellProfilerComparisonObservation] = []
    for repetition in range(1, repeats + 1):
        for case in cases:
            try:
                result = _run_comparison_case(
                    case,
                    output_root=output_root,
                    suite_id=suite_id,
                    repetition=repetition,
                    reuse_openhcs_cache=reuse_openhcs_cache,
                    native_reference_root=native_reference_root,
                    discard_openhcs_outputs=discard_openhcs_outputs,
                    openhcs_axis_filter=tuple(openhcs_axis_filter),
                    openhcs_max_axis_count=openhcs_max_axis_count,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                result = _failed_comparison_observation(
                    case,
                    suite_id=suite_id,
                    repetition=repetition,
                    error=exc,
                )
            observations.append(result)
            append_observations_jsonl(
                output_root / "observations.jsonl",
                (result,),
            )
            write_observations_csv(output_root / "observations.csv", observations)
            write_phase_timing_csv(output_root / "phase_timing.csv", observations)
            write_summary_csv(
                output_root / "summary.csv",
                observations,
                speedup_target=speedup_target,
            )
            write_suite_metadata(
                output_root / "suite_metadata.json",
                suite_id=suite_id,
                speedup_target=speedup_target,
                native_reference_root=native_reference_root,
                discard_openhcs_outputs=discard_openhcs_outputs,
                continue_on_error=continue_on_error,
                openhcs_axis_filter=tuple(openhcs_axis_filter),
                openhcs_max_axis_count=openhcs_max_axis_count,
            )
    return tuple(observations)


def load_observations_jsonl(
    path: Path,
) -> tuple[dict[str, Any], ...]:
    """Load raw observation payloads from a JSONL file."""
    observations: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            observations.append(json.loads(stripped))
    return tuple(observations)


def append_observations_jsonl(
    path: Path,
    observations: Iterable[CellProfilerComparisonObservation],
) -> None:
    """Append raw observations as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation.as_payload(), sort_keys=True) + "\n")


def write_observations_csv(
    path: Path,
    observations: Sequence[CellProfilerComparisonObservation],
) -> None:
    """Write one-row-per-observation benchmark results."""
    _write_csv_table(path, _OBSERVATION_TABLE, observations)


def write_phase_timing_csv(
    path: Path,
    observations: Sequence[CellProfilerComparisonObservation],
) -> None:
    """Write long-form phase timing rows for all observations."""
    _write_csv_table(path, _PHASE_TIMING_TABLE, observations)


def write_summary_csv(
    path: Path,
    observations: Sequence[CellProfilerComparisonObservation],
    *,
    speedup_target: float = DEFAULT_SPEEDUP_TARGET,
) -> None:
    """Write per-case aggregate medians for plotting."""
    _write_csv_table(path, _summary_table(speedup_target), observations)


def _write_csv_table(
    path: Path,
    table: CsvTableSpec,
    observations: Sequence[CellProfilerComparisonObservation],
) -> None:
    """Write a benchmark CSV table through its table-spec authority."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table.fieldnames)
        writer.writeheader()
        writer.writerows(table.rows(observations))


def _observation_csv_rows(
    observations: Sequence[CellProfilerComparisonObservation],
) -> Iterable[CsvRow]:
    for observation in observations:
        yield _observation_csv_row(observation)


def _phase_timing_csv_rows(
    observations: Sequence[CellProfilerComparisonObservation],
) -> Iterable[CsvRow]:
    for observation in observations:
        for tool_summary in (
            observation.native_cellprofiler,
            observation.openhcs,
        ):
            for phase, seconds in tool_summary.phase_seconds.items():
                yield {
                    SUITE_ID_FIELD: observation.suite_id,
                    CASE_NAME_FIELD: observation.case_name,
                    REPETITION_FIELD: observation.repetition,
                    TOOL_FIELD: tool_summary.tool,
                    PHASE_FIELD: phase,
                    SECONDS_FIELD: seconds,
                }


def _summary_csv_rows(
    observations: Sequence[CellProfilerComparisonObservation],
    *,
    speedup_target: float,
) -> Iterable[CsvRow]:
    grouped: dict[str, list[CellProfilerComparisonObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.case_name].append(observation)
    for case_name in sorted(grouped):
        case_observations = grouped[case_name]
        median_speedup = _median_present(
            observation.speedup for observation in case_observations
        )
        median_total_phase_speedup = _median_present(
            observation.total_phase_speedup for observation in case_observations
        )
        meets_execution_speedup_target = (
            median_speedup is not None and median_speedup >= speedup_target
        )
        meets_total_phase_speedup_target = (
            median_total_phase_speedup is not None
            and median_total_phase_speedup >= speedup_target
        )
        yield {
            CASE_NAME_FIELD: case_name,
            ASSAY_CATEGORY_FIELD: _common_value(
                observation.assay_category for observation in case_observations
            ),
            MODULE_CATEGORY_FIELD: _common_value(
                observation.module_category for observation in case_observations
            ),
            N_FIELD: len(case_observations),
            EQUIVALENT_COUNT_FIELD: sum(
                1 for observation in case_observations if observation.equivalent
            ),
            MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD: _median_present(
                observation.native_cellprofiler.execution_seconds
                for observation in case_observations
            ),
            MEDIAN_OPENHCS_EXECUTION_SECONDS_FIELD: _median_present(
                observation.openhcs.execution_seconds
                for observation in case_observations
            ),
            MEDIAN_NATIVE_TOTAL_PHASE_SECONDS_FIELD: _median_present(
                observation.native_cellprofiler.total_metric_seconds
                for observation in case_observations
            ),
            MEDIAN_OPENHCS_TOTAL_PHASE_SECONDS_FIELD: _median_present(
                observation.openhcs.total_metric_seconds
                for observation in case_observations
            ),
            MEDIAN_NATIVE_PEAK_MEMORY_MB_FIELD: _median_present(
                observation.native_cellprofiler.peak_memory_mb
                for observation in case_observations
            ),
            MEDIAN_OPENHCS_PEAK_MEMORY_MB_FIELD: _median_present(
                observation.openhcs.peak_memory_mb
                for observation in case_observations
            ),
            MEDIAN_SPEEDUP_FIELD: median_speedup,
            MEDIAN_TOTAL_PHASE_SPEEDUP_FIELD: median_total_phase_speedup,
            SPEEDUP_TARGET_FIELD: speedup_target,
            MEETS_EXECUTION_SPEEDUP_TARGET_FIELD: meets_execution_speedup_target,
            MEETS_TOTAL_PHASE_SPEEDUP_TARGET_FIELD: meets_total_phase_speedup_target,
            MEETS_SPEEDUP_TARGET_FIELD: (
                meets_execution_speedup_target and meets_total_phase_speedup_target
            ),
            MIN_PARITY_ACCURACY_FIELD: min(
                observation.parity_accuracy for observation in case_observations
            ),
        }


_OBSERVATION_TABLE = CsvTableSpec(
    (
        SUITE_ID_FIELD,
        CASE_NAME_FIELD,
        REPETITION_FIELD,
        DATASET_ID_FIELD,
        ASSAY_CATEGORY_FIELD,
        MODULE_CATEGORY_FIELD,
        EQUIVALENT_FIELD,
        DIFFERENCE_COUNT_FIELD,
        NUMERIC_ABS_TOLERANCE_FIELD,
        NUMERIC_REL_TOLERANCE_FIELD,
        NATIVE_EXECUTION_SECONDS_FIELD,
        OPENHCS_EXECUTION_SECONDS_FIELD,
        NATIVE_TOTAL_PHASE_SECONDS_FIELD,
        OPENHCS_TOTAL_PHASE_SECONDS_FIELD,
        NATIVE_PEAK_MEMORY_MB_FIELD,
        OPENHCS_PEAK_MEMORY_MB_FIELD,
        SPEEDUP_FIELD,
        TOTAL_PHASE_SPEEDUP_FIELD,
        PARITY_ACCURACY_FIELD,
        NATIVE_CACHED_FIELD,
        OPENHCS_CACHED_FIELD,
        NATIVE_ERROR_MESSAGE_FIELD,
        OPENHCS_ERROR_MESSAGE_FIELD,
        NATIVE_OUTPUT_PATH_FIELD,
        OPENHCS_OUTPUT_PATH_FIELD,
    ),
    _observation_csv_rows,
)
_PHASE_TIMING_TABLE = CsvTableSpec(
    (
        SUITE_ID_FIELD,
        CASE_NAME_FIELD,
        REPETITION_FIELD,
        TOOL_FIELD,
        PHASE_FIELD,
        SECONDS_FIELD,
    ),
    _phase_timing_csv_rows,
)
def _summary_table(speedup_target: float) -> CsvTableSpec:
    return CsvTableSpec(
        (
            CASE_NAME_FIELD,
            ASSAY_CATEGORY_FIELD,
            MODULE_CATEGORY_FIELD,
            N_FIELD,
            EQUIVALENT_COUNT_FIELD,
            MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD,
            MEDIAN_OPENHCS_EXECUTION_SECONDS_FIELD,
            MEDIAN_NATIVE_TOTAL_PHASE_SECONDS_FIELD,
            MEDIAN_OPENHCS_TOTAL_PHASE_SECONDS_FIELD,
            MEDIAN_NATIVE_PEAK_MEMORY_MB_FIELD,
            MEDIAN_OPENHCS_PEAK_MEMORY_MB_FIELD,
            MEDIAN_SPEEDUP_FIELD,
            MEDIAN_TOTAL_PHASE_SPEEDUP_FIELD,
            SPEEDUP_TARGET_FIELD,
            MEETS_EXECUTION_SPEEDUP_TARGET_FIELD,
            MEETS_TOTAL_PHASE_SPEEDUP_TARGET_FIELD,
            MEETS_SPEEDUP_TARGET_FIELD,
            MIN_PARITY_ACCURACY_FIELD,
        ),
        lambda observations: _summary_csv_rows(
            observations,
            speedup_target=speedup_target,
        ),
    )


def write_suite_metadata(
    path: Path,
    *,
    suite_id: str,
    speedup_target: float = DEFAULT_SPEEDUP_TARGET,
    native_reference_root: Path | None = None,
    discard_openhcs_outputs: bool = False,
    continue_on_error: bool = False,
    openhcs_axis_filter: Sequence[str] = (),
    openhcs_max_axis_count: int | None = None,
) -> None:
    """Write reproducibility metadata for the benchmark suite."""
    payload = {
        "suite_id": suite_id,
        "speedup_target": speedup_target,
        "created_at_epoch_seconds": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "native_reference_root": (
            str(native_reference_root) if native_reference_root is not None else None
        ),
        "discard_openhcs_outputs": discard_openhcs_outputs,
        "continue_on_error": continue_on_error,
        OPENHCS_AXIS_FILTER_PARAM: tuple(openhcs_axis_filter),
        OPENHCS_MAX_AXIS_COUNT_PARAM: openhcs_max_axis_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_comparison_case(
    case: CellProfilerComparisonCase,
    *,
    output_root: Path,
    suite_id: str,
    repetition: int,
    reuse_openhcs_cache: bool,
    native_reference_root: Path | None,
    discard_openhcs_outputs: bool,
    openhcs_axis_filter: Sequence[str],
    openhcs_max_axis_count: int | None,
) -> CellProfilerComparisonObservation:
    native_reference = _native_reference_location(case, native_reference_root)
    pipeline_params: dict[str, object] = {
        **case.pipeline_params,
        "compare_image_outputs": not case.value_only,
        "raise_on_equivalence_failure": False,
        "cache_candidate_measurement_snapshot": not discard_openhcs_outputs,
    }
    if case.cellprofiler_timeout_seconds is not None:
        pipeline_params["cellprofiler_timeout_seconds"] = (
            case.cellprofiler_timeout_seconds
        )
    if openhcs_axis_filter:
        pipeline_params[OPENHCS_AXIS_FILTER_PARAM] = tuple(openhcs_axis_filter)
    if openhcs_max_axis_count is not None:
        pipeline_params[OPENHCS_MAX_AXIS_COUNT_PARAM] = openhcs_max_axis_count
    result = run_cellprofiler_cppipe_parity(
        case.dataset_path,
        case.cppipe_path,
        metrics=[TimeMetric(), MemoryMetric()],
        dataset_id=case.dataset_id,
        pipeline_name=case.name,
        microscope_type=case.microscope_type,
        pipeline_params=pipeline_params,
        output_root=output_root / "tool_outputs",
        equivalence_reference_output_dir=native_reference.reference_output_dir,
        native_cellprofiler_output_dir=native_reference.output_dir,
        reuse_openhcs_cache=reuse_openhcs_cache,
    )
    observation = comparison_observation_from_result(
        result,
        case=case,
        suite_id=suite_id,
        repetition=repetition,
    )
    if discard_openhcs_outputs:
        _discard_successful_openhcs_benchmark_tree(
            observation,
            suite_output_root=output_root,
        )
    return observation


def _native_reference_location(
    case: CellProfilerComparisonCase,
    native_reference_root: Path | None,
) -> NativeReferenceLocation:
    if case.equivalence_reference_output_dir is not None:
        return NativeReferenceLocation(
            output_dir=None,
            reference_output_dir=case.equivalence_reference_output_dir,
        )
    if native_reference_root is None:
        return NativeReferenceLocation(output_dir=None, reference_output_dir=None)
    native_output_dir = Path(native_reference_root) / _benchmark_path_slug(
        f"{case.resolved_dataset_id}_{case.name}"
    )
    resolved_dataset_path = resolve_visible_source_path(case.dataset_path)
    expected_reference = (
        native_output_dir
        / f"{resolved_dataset_path.name}_{case.name}_native_cellprofiler"
    )
    if native_cellprofiler_reference_is_complete(expected_reference):
        return NativeReferenceLocation(
            output_dir=native_output_dir,
            reference_output_dir=expected_reference,
        )
    return NativeReferenceLocation(
        output_dir=native_output_dir,
        reference_output_dir=None,
    )


def _failed_comparison_observation(
    case: CellProfilerComparisonCase,
    *,
    suite_id: str,
    repetition: int,
    error: Exception,
) -> CellProfilerComparisonObservation:
    return CellProfilerComparisonObservation(
        suite_id=suite_id,
        case_name=case.name,
        repetition=repetition,
        dataset_id=case.resolved_dataset_id,
        assay_category=case.assay_category,
        module_category=case.module_category,
        cppipe_path=str(case.cppipe_path),
        equivalent=False,
        difference_count=None,
        numeric_abs_tolerance=1e-6,
        numeric_rel_tolerance=1e-6,
        native_cellprofiler=ToolExecutionSummary(
            "CellProfiler",
            False,
            "",
            None,
            None,
            None,
            False,
            f"{type(error).__name__}: {error}",
            {},
        ),
        openhcs=ToolExecutionSummary(
            "OpenHCS",
            False,
            "",
            None,
            None,
            None,
            False,
            "skipped after benchmark case failure",
            {},
        ),
    )


def comparison_observation_from_result(
    result: CellProfilerCompatibilityResult,
    *,
    case: CellProfilerComparisonCase,
    suite_id: str,
    repetition: int,
) -> CellProfilerComparisonObservation:
    """Convert adapter results into a stable observation payload."""
    openhcs_provenance = result.openhcs_converted.provenance or {}
    native_summary = CachedNativeReferenceTimingPolicy(
        case=case,
        summary=_tool_execution_summary(
            result.native_cellprofiler,
            execution_phase="EXECUTE_NATIVE_CP",
        ),
    ).apply()
    return CellProfilerComparisonObservation(
        suite_id=suite_id,
        case_name=case.name,
        repetition=repetition,
        dataset_id=case.resolved_dataset_id,
        assay_category=case.assay_category,
        module_category=case.module_category,
        cppipe_path=str(case.cppipe_path),
        equivalent=result.is_equivalent,
        difference_count=_difference_count(result),
        numeric_abs_tolerance=1e-6,
        numeric_rel_tolerance=1e-6,
        native_cellprofiler=native_summary,
        openhcs=_tool_execution_summary(
            result.openhcs_converted,
            execution_phase="EXECUTE_OPENHCS",
            cached=bool(
                openhcs_provenance.get("reused_cached_output")
                or openhcs_provenance.get("reused_runtime_execution_cache")
            ),
        ),
    )


def _tool_execution_summary(
    result: BenchmarkResult,
    *,
    execution_phase: str,
    cached: bool | None = None,
) -> ToolExecutionSummary:
    phase_seconds = _phase_seconds(result)
    metric_seconds = result.metrics.get("execution_time_seconds")
    peak_memory_mb = result.metrics.get("peak_memory_mb")
    total_phase_seconds = sum(phase_seconds.values()) if phase_seconds else None
    return ToolExecutionSummary(
        tool=result.tool_name,
        success=result.success,
        output_path=str(result.output_path),
        execution_seconds=phase_seconds.get(execution_phase),
        total_metric_seconds=total_phase_seconds
        if total_phase_seconds is not None
        else (float(metric_seconds) if metric_seconds is not None else None),
        peak_memory_mb=(
            float(peak_memory_mb) if peak_memory_mb is not None else None
        ),
        cached=bool(cached) if cached is not None else _result_is_cached(result),
        error_message=result.error_message,
        phase_seconds=phase_seconds,
    )


def _phase_seconds(result: BenchmarkResult) -> dict[str, float]:
    phase_totals: dict[str, float] = defaultdict(float)
    provenance = result.provenance or {}
    for raw_record in provenance.get("phase_timing_records", ()):
        if not isinstance(raw_record, Mapping):
            continue
        phase = raw_record.get("phase")
        seconds = raw_record.get("seconds")
        if phase is None or seconds is None:
            continue
        phase_totals[str(phase)] += float(seconds)
    return dict(phase_totals)


def _difference_count(result: CellProfilerCompatibilityResult) -> int | None:
    provenance = result.openhcs_converted.provenance or {}
    value = provenance.get("equivalence_difference_count")
    return int(value) if value is not None else None


def _result_is_cached(result: BenchmarkResult) -> bool:
    provenance = result.provenance or {}
    return bool(
        provenance.get("reused_reference_output")
        or provenance.get("reused_cached_output")
        or provenance.get("reused_runtime_execution_cache")
    )


def _observation_csv_row(
    observation: CellProfilerComparisonObservation,
) -> dict[str, object]:
    return {
        SUITE_ID_FIELD: observation.suite_id,
        CASE_NAME_FIELD: observation.case_name,
        REPETITION_FIELD: observation.repetition,
        DATASET_ID_FIELD: observation.dataset_id,
        ASSAY_CATEGORY_FIELD: observation.assay_category,
        MODULE_CATEGORY_FIELD: observation.module_category,
        EQUIVALENT_FIELD: observation.equivalent,
        DIFFERENCE_COUNT_FIELD: observation.difference_count,
        NUMERIC_ABS_TOLERANCE_FIELD: observation.numeric_abs_tolerance,
        NUMERIC_REL_TOLERANCE_FIELD: observation.numeric_rel_tolerance,
        NATIVE_EXECUTION_SECONDS_FIELD: (
            observation.native_cellprofiler.execution_seconds
        ),
        OPENHCS_EXECUTION_SECONDS_FIELD: observation.openhcs.execution_seconds,
        NATIVE_TOTAL_PHASE_SECONDS_FIELD: (
            observation.native_cellprofiler.total_metric_seconds
        ),
        OPENHCS_TOTAL_PHASE_SECONDS_FIELD: observation.openhcs.total_metric_seconds,
        NATIVE_PEAK_MEMORY_MB_FIELD: observation.native_cellprofiler.peak_memory_mb,
        OPENHCS_PEAK_MEMORY_MB_FIELD: observation.openhcs.peak_memory_mb,
        SPEEDUP_FIELD: observation.speedup,
        TOTAL_PHASE_SPEEDUP_FIELD: observation.total_phase_speedup,
        PARITY_ACCURACY_FIELD: observation.parity_accuracy,
        NATIVE_CACHED_FIELD: observation.native_cellprofiler.cached,
        OPENHCS_CACHED_FIELD: observation.openhcs.cached,
        NATIVE_ERROR_MESSAGE_FIELD: observation.native_cellprofiler.error_message,
        OPENHCS_ERROR_MESSAGE_FIELD: observation.openhcs.error_message,
        NATIVE_OUTPUT_PATH_FIELD: observation.native_cellprofiler.output_path,
        OPENHCS_OUTPUT_PATH_FIELD: observation.openhcs.output_path,
    }


def _median_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return statistics.median(present)


def _common_value(values: Iterable[str | None]) -> str | None:
    present = {value for value in values if value}
    if not present:
        return None
    if len(present) > 1:
        return "Mixed"
    return next(iter(present))


def _benchmark_path_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _discard_openhcs_benchmark_tree(
    output_path: Path,
    *,
    suite_output_root: Path,
) -> None:
    """Delete one OpenHCS output tree only when benchmark ownership is proven."""
    target = _marked_openhcs_output_tree(Path(output_path), suite_output_root)
    suite_root = Path(suite_output_root).resolve()
    if not target.exists():
        return
    if not target.is_dir():
        raise NotADirectoryError(f"OpenHCS discard target is not a directory: {target}")
    if target == Path(".").resolve() or target == suite_root or target.parent == target:
        raise ToolExecutionError(f"Refusing unsafe OpenHCS discard target: {target}")
    try:
        target.relative_to(suite_root)
    except ValueError as exc:
        raise ToolExecutionError(
            "Refusing OpenHCS discard target outside suite output root: "
            f"{target} not under {suite_root}"
        ) from exc
    shutil.rmtree(target)


def _discard_successful_openhcs_benchmark_tree(
    observation: CellProfilerComparisonObservation,
    *,
    suite_output_root: Path,
) -> None:
    """Delete successful OpenHCS outputs while preserving failed debug artifacts."""
    if not observation.openhcs.success:
        return
    _discard_openhcs_benchmark_tree(
        Path(observation.openhcs.output_path),
        suite_output_root=suite_output_root,
    )


def _marked_openhcs_output_tree(output_path: Path, suite_output_root: Path) -> Path:
    """Find the benchmark-owned OpenHCS tree containing an output path."""
    suite_root = Path(suite_output_root).resolve()
    start = Path(output_path).resolve()
    candidates = (start, *start.parents)
    for candidate in candidates:
        if candidate == suite_root:
            break
        try:
            candidate.relative_to(suite_root)
        except ValueError:
            break
        if (candidate / OPENHCS_BENCHMARK_CACHE_MARKER).is_file():
            return candidate
    raise ToolExecutionError(
        "Refusing to discard OpenHCS output because no benchmark cache marker "
        f"was found between {start} and {suite_root}."
    )
