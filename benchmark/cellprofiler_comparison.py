"""CellProfiler versus OpenHCS benchmark result collection."""

from __future__ import annotations

import csv
import json
import platform
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
from benchmark.metrics.time import TimeMetric
from benchmark.runner import CellProfilerCompatibilityResult
from benchmark.runner import run_cellprofiler_cppipe_parity


BENCHMARK_CACHE_DOMAINS = frozenset({"harness"})
SUITE_ID_FIELD = "suite_id"
CASE_NAME_FIELD = "case_name"
REPETITION_FIELD = "repetition"
DATASET_ID_FIELD = "dataset_id"
EQUIVALENT_FIELD = "equivalent"
DIFFERENCE_COUNT_FIELD = "difference_count"
NUMERIC_ABS_TOLERANCE_FIELD = "numeric_abs_tolerance"
NUMERIC_REL_TOLERANCE_FIELD = "numeric_rel_tolerance"
NATIVE_EXECUTION_SECONDS_FIELD = "native_execution_seconds"
OPENHCS_EXECUTION_SECONDS_FIELD = "openhcs_execution_seconds"
SPEEDUP_FIELD = "speedup"
SPEEDUP_TARGET_FIELD = "speedup_target"
MEETS_SPEEDUP_TARGET_FIELD = "meets_speedup_target"
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
MEDIAN_SPEEDUP_FIELD = "median_speedup"
MIN_PARITY_ACCURACY_FIELD = "min_parity_accuracy"
DEFAULT_SPEEDUP_TARGET = 5.0
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
    value_only: bool = False
    equivalence_reference_output_dir: Path | None = None
    cellprofiler_timeout_seconds: float | None = None

    @property
    def resolved_dataset_id(self) -> str:
        return self.dataset_id or self.dataset_path.name


ToolExecutionSummary = product_record(
    "ToolExecutionSummary",
    (
        "tool: str; success: bool; output_path: str; "
        "execution_seconds: float | None; total_metric_seconds: float | None; "
        "cached: bool; error_message: str | None; phase_seconds: Mapping[str, float]"
    ),
    doc="Execution and phase timing summary for one tool run.",
    module_name=__name__,
)


@dataclass(frozen=True, slots=True)
class CellProfilerComparisonObservation:
    """Serializable observation for one case/repetition."""

    suite_id: str
    case_name: str
    repetition: int
    dataset_id: str
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
    def parity_accuracy(self) -> float:
        return 1.0 if self.equivalent else 0.0

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["speedup"] = self.speedup
        payload["parity_accuracy"] = self.parity_accuracy
        return payload


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
) -> tuple[CellProfilerComparisonObservation, ...]:
    """Run all cases and write raw benchmark observations."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if speedup_target <= 0:
        raise ValueError("speedup_target must be positive.")
    output_root.mkdir(parents=True, exist_ok=True)
    observations: list[CellProfilerComparisonObservation] = []
    for repetition in range(1, repeats + 1):
        for case in cases:
            result = _run_comparison_case(
                case,
                output_root=output_root,
                suite_id=suite_id,
                repetition=repetition,
                reuse_openhcs_cache=reuse_openhcs_cache,
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
        yield {
            CASE_NAME_FIELD: case_name,
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
            MEDIAN_SPEEDUP_FIELD: median_speedup,
            SPEEDUP_TARGET_FIELD: speedup_target,
            MEETS_SPEEDUP_TARGET_FIELD: (
                median_speedup is not None and median_speedup >= speedup_target
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
        EQUIVALENT_FIELD,
        DIFFERENCE_COUNT_FIELD,
        NUMERIC_ABS_TOLERANCE_FIELD,
        NUMERIC_REL_TOLERANCE_FIELD,
        NATIVE_EXECUTION_SECONDS_FIELD,
        OPENHCS_EXECUTION_SECONDS_FIELD,
        SPEEDUP_FIELD,
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
            N_FIELD,
            EQUIVALENT_COUNT_FIELD,
            MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD,
            MEDIAN_OPENHCS_EXECUTION_SECONDS_FIELD,
            MEDIAN_SPEEDUP_FIELD,
            SPEEDUP_TARGET_FIELD,
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
) -> None:
    """Write reproducibility metadata for the benchmark suite."""
    payload = {
        "suite_id": suite_id,
        "speedup_target": speedup_target,
        "created_at_epoch_seconds": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
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
) -> CellProfilerComparisonObservation:
    pipeline_params: dict[str, object] = {
        "compare_image_outputs": not case.value_only,
        "raise_on_equivalence_failure": False,
    }
    if case.cellprofiler_timeout_seconds is not None:
        pipeline_params["cellprofiler_timeout_seconds"] = (
            case.cellprofiler_timeout_seconds
        )
    result = run_cellprofiler_cppipe_parity(
        case.dataset_path,
        case.cppipe_path,
        metrics=[TimeMetric()],
        dataset_id=case.dataset_id,
        pipeline_name=case.name,
        microscope_type=case.microscope_type,
        pipeline_params=pipeline_params,
        output_root=output_root / "tool_outputs",
        equivalence_reference_output_dir=case.equivalence_reference_output_dir,
        reuse_openhcs_cache=reuse_openhcs_cache,
    )
    return comparison_observation_from_result(
        result,
        case=case,
        suite_id=suite_id,
        repetition=repetition,
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
    return CellProfilerComparisonObservation(
        suite_id=suite_id,
        case_name=case.name,
        repetition=repetition,
        dataset_id=case.resolved_dataset_id,
        cppipe_path=str(case.cppipe_path),
        equivalent=result.is_equivalent,
        difference_count=_difference_count(result),
        numeric_abs_tolerance=1e-6,
        numeric_rel_tolerance=1e-6,
        native_cellprofiler=_tool_execution_summary(
            result.native_cellprofiler,
            execution_phase="EXECUTE_NATIVE_CP",
        ),
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
    return ToolExecutionSummary(
        tool=result.tool_name,
        success=result.success,
        output_path=str(result.output_path),
        execution_seconds=phase_seconds.get(execution_phase),
        total_metric_seconds=(
            float(metric_seconds) if metric_seconds is not None else None
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
        EQUIVALENT_FIELD: observation.equivalent,
        DIFFERENCE_COUNT_FIELD: observation.difference_count,
        NUMERIC_ABS_TOLERANCE_FIELD: observation.numeric_abs_tolerance,
        NUMERIC_REL_TOLERANCE_FIELD: observation.numeric_rel_tolerance,
        NATIVE_EXECUTION_SECONDS_FIELD: (
            observation.native_cellprofiler.execution_seconds
        ),
        OPENHCS_EXECUTION_SECONDS_FIELD: observation.openhcs.execution_seconds,
        SPEEDUP_FIELD: observation.speedup,
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
