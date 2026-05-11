"""Well-level OpenHCS throughput scaling for converted cppipe pipelines."""

from __future__ import annotations

import csv
import multiprocessing
import queue
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmark.cellprofiler_comparison import load_comparison_cases
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
from openhcs.core.progress import set_progress_queue
from openhcs.core.source_schema_workspace import (
    expand_source_schema_workspace_wells,
    materialize_source_schema_workspace,
)
from openhcs.core.steps.function_runtime import prepare_compiled_context_callables


WELL_THROUGHPUT_ROWS_CSV = "well_throughput.csv"
WELL_THROUGHPUT_EVENTS_CSV = "well_throughput_progress_events.csv"
WELL_THROUGHPUT_LANES_CSV = "well_throughput_worker_lanes.csv"
WELL_THROUGHPUT_STEPS_CSV = "well_throughput_step_timings.csv"


@dataclass(frozen=True, slots=True)
class WellThroughputResult:
    """One compile-once, execute-many-wells throughput observation."""

    case_name: str
    worker_count: int
    well_count: int
    compile_seconds: float
    prepare_seconds: float
    execute_seconds: float
    total_seconds: float
    wells_per_second: float
    successful_wells: int


def run_well_throughput_suite(
    manifest_path: Path,
    *,
    output_root: Path,
    case_names: Sequence[str] = (),
    well_counts: Sequence[int],
    worker_counts: Sequence[int],
) -> tuple[WellThroughputResult, ...]:
    """Run converted cppipes as one OpenHCS plate with repeated virtual wells."""
    cases = load_comparison_cases(manifest_path)
    selected = set(case_names)
    results: list[WellThroughputResult] = []
    for case in cases:
        if selected and case.name not in selected:
            continue
        for well_count in tuple(sorted(set(int(value) for value in well_counts))):
            for worker_count in tuple(sorted(set(int(value) for value in worker_counts))):
                result = run_case_well_throughput(
                    case_name=case.name,
                    dataset_path=case.dataset_path,
                    cppipe_path=case.cppipe_path,
                    output_root=output_root / case.name / f"wells_{well_count}" / f"workers_{worker_count}",
                    well_count=well_count,
                    worker_count=worker_count,
                )
                results.append(result)
                write_well_throughput_csv(output_root / WELL_THROUGHPUT_ROWS_CSV, results)
    return tuple(results)


def run_case_well_throughput(
    *,
    case_name: str,
    dataset_path: Path,
    cppipe_path: Path,
    output_root: Path,
    well_count: int,
    worker_count: int,
) -> WellThroughputResult:
    """Run one converted cppipe over synthetic wells in a single OpenHCS execution."""
    if well_count < 1:
        raise ValueError("well_count must be positive.")
    if worker_count < 1:
        raise ValueError("worker_count must be positive.")

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

    source_workspace = materialize_source_schema_workspace(
        dataset_path,
        output_root / f"{dataset_path.name}_{cppipe_path.stem}_source_workspace",
        prepared.source_schema,
    )
    well_ids = expand_source_schema_workspace_wells(
        source_workspace.metadata_path,
        _synthetic_well_ids(well_count),
    )

    global_config = GlobalPipelineConfig(
        num_workers=worker_count,
        use_threading=False,
        multiprocessing_start_method=MultiprocessingStartMethod.FORK,
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

    started_at = time.perf_counter()
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

        compiled_contexts = compilation["compiled_contexts"]
        pipeline_definition = compilation.get(
            "pipeline_definition",
            prepared.pipeline.steps,
        )
        prepare_started_at = time.perf_counter()
        prepare_compiled_context_callables(compiled_contexts)
        prepare_seconds = time.perf_counter() - prepare_started_at

        execute_started_at = time.perf_counter()
        execution_results = orchestrator.execute_compiled_plate(
            pipeline_definition=pipeline_definition,
            compiled_contexts=compiled_contexts,
            progress_queue=progress_queue,
            progress_context=progress_context,
        )
        execute_seconds = time.perf_counter() - execute_started_at
    finally:
        progress_queue.put(None)
        consumer.join(timeout=5.0)

    total_seconds = time.perf_counter() - started_at
    successful_wells = sum(
        1
        for result in execution_results.values()
        if _execution_result_succeeded(result)
    )
    _write_progress_diagnostics(
        output_root,
        case_name=case_name,
        worker_count=worker_count,
        well_count=well_count,
        events=progress_events,
    )
    return WellThroughputResult(
        case_name=case_name,
        worker_count=worker_count,
        well_count=well_count,
        compile_seconds=compile_seconds,
        prepare_seconds=prepare_seconds,
        execute_seconds=execute_seconds,
        total_seconds=total_seconds,
        wells_per_second=well_count / execute_seconds if execute_seconds > 0.0 else 0.0,
        successful_wells=successful_wells,
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


def _synthetic_well_ids(count: int) -> tuple[str, ...]:
    return tuple(f"W{index:03d}" for index in range(1, count + 1))


def _execution_result_succeeded(result: Any) -> bool:
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
        phase = event.get("phase")
        timestamp = float(event["timestamp"])
        if phase == "axis_started":
            if lane["started_at"] == "" or timestamp < float(lane["started_at"]):
                lane["started_at"] = timestamp
        elif phase == "axis_completed":
            lane["axis_count"] += 1
            if lane["completed_at"] == "" or timestamp > float(lane["completed_at"]):
                lane["completed_at"] = timestamp

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
