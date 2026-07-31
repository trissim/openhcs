from dataclasses import fields, replace as dataclass_replace

from zmqruntime.messages import QueuedExecutionInfo, RunningExecutionInfo

from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.progress.projection import (
    ExecutionRuntimeProjection,
    PlateRuntimeState,
    PlateRuntimeStateDeclarationBase,
    build_execution_runtime_projection,
)


def _event(
    *,
    execution_id: str = "exec-1",
    plate_id: str = "/tmp/plate",
    axis_id: str = "",
    step_name: str = "pipeline",
    phase: ProgressPhase,
    status: ProgressStatus,
    percent: float,
    completed: int = 0,
    total: int = 1,
    total_wells=None,
    worker_assignments=None,
) -> ProgressEvent:
    return ProgressEvent(
        identity=ProgressIdentity(
            execution_id=execution_id,
            plate_id=plate_id,
            axis_id=axis_id,
            step_name=step_name,
        ),
        phase=phase,
        status=status,
        percent=percent,
        completed=completed,
        total=total,
        timestamp=1.0,
        pid=1234,
        total_wells=total_wells,
        worker_assignments=worker_assignments,
    )


def test_projection_marks_plate_compiled_when_all_known_wells_compiled():
    events = [
        _event(
            phase=ProgressPhase.INIT,
            status=ProgressStatus.STARTED,
            percent=0.0,
            total_wells=["A01", "B01"],
        ),
        _event(
            axis_id="A01",
            step_name="compilation",
            phase=ProgressPhase.COMPILE,
            status=ProgressStatus.SUCCESS,
            percent=100.0,
        ),
        _event(
            axis_id="B01",
            step_name="compilation",
            phase=ProgressPhase.COMPILE,
            status=ProgressStatus.SUCCESS,
            percent=100.0,
        ),
    ]

    projection = build_execution_runtime_projection({"exec-1": events})
    plate = projection.get_plate("/tmp/plate", "exec-1")

    assert plate is not None
    assert plate.state == PlateRuntimeState.COMPILED
    assert round(plate.percent, 1) == 100.0
    assert projection.count_for_state(PlateRuntimeState.COMPILED) == 1


def test_projection_marks_plate_executing_from_pipeline_channel():
    events = [
        _event(
            phase=ProgressPhase.INIT,
            status=ProgressStatus.STARTED,
            percent=0.0,
            total_wells=["A01", "B01"],
        ),
        _event(
            axis_id="A01",
            step_name="normalize",
            phase=ProgressPhase.STEP_COMPLETED,
            status=ProgressStatus.SUCCESS,
            percent=50.0,
            completed=1,
            total=2,
        ),
    ]

    projection = build_execution_runtime_projection({"exec-1": events})
    plate = projection.get_plate("/tmp/plate", "exec-1")

    assert plate is not None
    assert plate.state == PlateRuntimeState.EXECUTING
    assert round(plate.percent, 1) == 25.0
    assert projection.count_for_state(PlateRuntimeState.EXECUTING) == 1


def test_projection_dedupes_multiple_execution_ids_for_same_plate():
    exec1 = _event(
        execution_id="exec-1",
        phase=ProgressPhase.COMPILE,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        axis_id="A01",
        step_name="compilation",
    )
    exec2 = _event(
        execution_id="exec-2",
        phase=ProgressPhase.STEP_STARTED,
        status=ProgressStatus.RUNNING,
        percent=50.0,
        axis_id="A01",
        step_name="normalize",
        completed=1,
        total=2,
    )
    # Make second execution newer.
    exec2 = dataclass_replace(exec2, timestamp=2.0)

    projection = build_execution_runtime_projection(
        {
            "exec-1": [exec1],
            "exec-2": [exec2],
        }
    )

    assert len(projection.plates) == 2  # raw snapshots
    assert len(projection.by_plate_latest) == 1  # deduped visible plate set
    assert projection.count_for_state(PlateRuntimeState.EXECUTING) == 1
    assert projection.count_for_state(PlateRuntimeState.COMPILED) == 0


def test_projection_marks_plate_failed_on_axis_error():
    events = [
        _event(
            phase=ProgressPhase.INIT,
            status=ProgressStatus.STARTED,
            percent=0.0,
            total_wells=["A01", "B01"],
        ),
        _event(
            axis_id="A01",
            step_name="normalize",
            phase=ProgressPhase.AXIS_ERROR,
            status=ProgressStatus.ERROR,
            percent=50.0,
            completed=1,
            total=2,
        ),
        _event(
            axis_id="B01",
            step_name="pipeline",
            phase=ProgressPhase.AXIS_COMPLETED,
            status=ProgressStatus.SUCCESS,
            percent=100.0,
            completed=2,
            total=2,
        ),
    ]

    projection = build_execution_runtime_projection({"exec-1": events})
    plate = projection.get_plate("/tmp/plate", "exec-1")

    assert plate is not None
    assert plate.state == PlateRuntimeState.FAILED
    assert round(plate.percent, 1) == 75.0
    assert projection.count_for_state(PlateRuntimeState.FAILED) == 1


def test_plate_runtime_state_declarations_cover_wire_tokens():
    assert set(PlateRuntimeStateDeclarationBase.__registry__) == set(PlateRuntimeState)


def test_execution_projection_does_not_mirror_registered_state_counts_as_fields():
    projection_fields = {field.name for field in fields(ExecutionRuntimeProjection)}

    assert "state_counts" in projection_fields
    assert not {
        f"{state.value}_count" for state in PlateRuntimeState
    } & projection_fields


def test_init_topology_projects_queued_without_a_parallel_ui_override():
    projection = build_execution_runtime_projection(
        {
            "exec-1": [
                _event(
                    phase=ProgressPhase.INIT,
                    status=ProgressStatus.STARTED,
                    percent=0.0,
                    worker_assignments={"worker_0": ["A01"]},
                )
            ]
        }
    )

    plate = projection.get_plate("/tmp/plate", "exec-1")
    assert plate is not None
    assert plate.state is PlateRuntimeState.QUEUED


def test_server_lifecycle_entries_reconcile_by_execution_and_plate_identity():
    old_event = _event(
        execution_id="exec-old",
        phase=ProgressPhase.AXIS_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        axis_id="A01",
    )
    projection = build_execution_runtime_projection(
        {"exec-old": [old_event]},
        queued_executions=(
            QueuedExecutionInfo(
                execution_id="exec-next",
                plate_id="/tmp/plate",
                queue_position=4,
            ),
        ),
        running_executions=(
            RunningExecutionInfo(
                execution_id="exec-running",
                plate_id="/tmp/other",
                start_time=2.0,
                elapsed=0.5,
                compile_only=True,
            ),
        ),
    )

    queued = projection.get_plate("/tmp/plate")
    compiling = projection.get_plate("/tmp/other")
    assert queued is not None
    assert queued.execution_id == "exec-next"
    assert queued.state is PlateRuntimeState.QUEUED
    assert queued.queue_position == 4
    assert compiling is not None
    assert compiling.state is PlateRuntimeState.COMPILING


def test_running_entry_only_supersedes_matching_queued_identity():
    projection = build_execution_runtime_projection(
        {},
        queued_executions=(
            QueuedExecutionInfo(
                execution_id="shared-exec",
                plate_id="/tmp/queued-plate",
                queue_position=2,
            ),
        ),
        running_executions=(
            RunningExecutionInfo(
                execution_id="shared-exec",
                plate_id="/tmp/running-plate",
                start_time=2.0,
                elapsed=0.5,
            ),
        ),
    )

    queued = projection.get_plate("/tmp/queued-plate", "shared-exec")
    running = projection.get_plate("/tmp/running-plate", "shared-exec")
    assert queued is not None
    assert queued.state is PlateRuntimeState.QUEUED
    assert running is not None
    assert running.state is PlateRuntimeState.EXECUTING


def test_server_snapshot_refreshes_queue_position_for_same_identity():
    queued_entry = QueuedExecutionInfo(
        execution_id="exec-queued",
        plate_id="/tmp/plate",
        queue_position=4,
    )
    projection = build_execution_runtime_projection(
        {},
        queued_executions=(queued_entry,),
    )

    projection.reconcile_server_executions(
        running_executions=(),
        queued_executions=(dataclass_replace(queued_entry, queue_position=1),),
    )

    queued = projection.get_plate("/tmp/plate", "exec-queued")
    assert queued is not None
    assert queued.queue_position == 1


def test_running_snapshot_advances_compiled_identity_into_execution():
    compiled_event = _event(
        execution_id="exec-shared",
        phase=ProgressPhase.COMPILE,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        axis_id="A01",
        step_name="compilation",
    )
    projection = build_execution_runtime_projection(
        {"exec-shared": [compiled_event]},
        running_executions=(
            RunningExecutionInfo(
                execution_id="exec-shared",
                plate_id="/tmp/plate",
                start_time=2.0,
                elapsed=0.5,
            ),
        ),
    )

    running = projection.get_plate("/tmp/plate", "exec-shared")
    assert running is not None
    assert running.state is PlateRuntimeState.EXECUTING


def test_lagging_server_snapshot_does_not_downgrade_terminal_event_state():
    completed_event = _event(
        execution_id="exec-terminal",
        phase=ProgressPhase.AXIS_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        axis_id="A01",
    )
    projection = build_execution_runtime_projection(
        {"exec-terminal": [completed_event]},
        running_executions=(
            RunningExecutionInfo(
                execution_id="exec-terminal",
                plate_id="/tmp/plate",
                start_time=0.5,
                elapsed=0.5,
            ),
        ),
    )

    completed = projection.get_plate("/tmp/plate", "exec-terminal")
    assert completed is not None
    assert completed.state is PlateRuntimeState.COMPLETE


def test_plate_runtime_terminal_policy_lives_on_projection():
    events = [
        _event(
            phase=ProgressPhase.INIT,
            status=ProgressStatus.STARTED,
            percent=0.0,
            total_wells=["A01"],
        ),
        _event(
            axis_id="A01",
            phase=ProgressPhase.AXIS_COMPLETED,
            status=ProgressStatus.SUCCESS,
            percent=100.0,
        ),
    ]

    projection = build_execution_runtime_projection({"exec-1": events})
    plate = projection.get_plate("/tmp/plate", "exec-1")

    assert plate is not None
    assert plate.state == PlateRuntimeState.COMPLETE
    assert plate.is_terminal
