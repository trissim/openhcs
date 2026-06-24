from dataclasses import replace as dataclass_replace

from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.progress.projection import (
    PlateRuntimeState,
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
    assert projection.compiled_count == 1


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
    assert projection.executing_count == 1


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
    assert projection.executing_count == 1
    assert projection.compiled_count == 0


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
    assert projection.failed_count == 1
