from pyqt_reactive.services.zmq_server_info import BaseServerInfo
from zmqruntime.messages import PongResponse

from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.pyqt_gui.widgets.shared.server_browser import (
    ExecutionProgressProjection,
    ProgressTreeBuilder,
)
from openhcs.pyqt_gui.widgets.shared.zmq_server_manager import ZMQServerManagerWidget


def _event(
    *,
    phase: ProgressPhase,
    status: ProgressStatus,
    percent: float,
    execution_id: str = "exec-1",
    plate_id: str = "/tmp/plate",
    axis_id: str = "A01",
    step_name: str = "pipeline",
    completed: int = 0,
    total: int = 1,
    timestamp: float = 1.0,
    worker_slot: str | None = None,
    owned_wells: list[str] | None = None,
    worker_assignments: dict[str, list[str]] | None = None,
    total_wells: list[str] | None = None,
    step_names: list[str] | None = None,
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
        timestamp=timestamp,
        pid=1111,
        worker_slot=worker_slot,
        owned_wells=owned_wells,
        worker_assignments=worker_assignments,
        total_wells=total_wells,
        step_names=step_names,
    )


def _init_event(
    *,
    execution_id: str = "exec-1",
    plate_id: str = "/tmp/plate",
    wells: tuple[str, ...] = ("A01",),
    timestamp: float = 0.0,
) -> ProgressEvent:
    return _event(
        execution_id=execution_id,
        plate_id=plate_id,
        phase=ProgressPhase.INIT,
        status=ProgressStatus.RUNNING,
        percent=0.0,
        axis_id="",
        step_name="init",
        timestamp=timestamp,
        worker_assignments={"worker_0": list(wells)},
        total_wells=list(wells),
        step_names=["normalize", "max_project"],
    )


def _execution_server_info(
    *,
    queued: list[dict] | None = None,
    running: list[dict] | None = None,
):
    return BaseServerInfo.from_response(
        PongResponse.from_dict(
            {
                "type": "pong",
                "port": 7777,
                "control_port": 8777,
                "ready": True,
                "server": "OpenHCSExecutionServer",
                "server_type": "execution",
                "server_role": "execution",
                "log_file_path": "/tmp/server.log",
                "workers": [],
                "running_executions": running or [],
                "queued_executions": queued or [],
            }
        )
    )


def _nodes(
    executions: dict[str, list[ProgressEvent]],
    server_info=None,
):
    projection = ExecutionProgressProjection(builder=ProgressTreeBuilder())
    tree = projection.build_runtime_tree(
        executions,
        server_info or _execution_server_info(),
    )
    return list(tree.roots)


def test_worker_tree_uses_pipeline_percent_for_parent_aggregation():
    pipeline_event = _event(
        phase=ProgressPhase.STEP_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=25.0,
        completed=1,
        total=4,
        step_name="normalize",
        timestamp=1.0,
        worker_slot="worker_0",
        owned_wells=["A01"],
    )
    step_event = _event(
        phase=ProgressPhase.PATTERN_GROUP,
        status=ProgressStatus.RUNNING,
        percent=50.0,
        completed=1,
        total=2,
        step_name="max_project",
        timestamp=2.0,
        worker_slot="worker_0",
        owned_wells=["A01"],
    )

    plate = _nodes({"exec-1": [_init_event(), pipeline_event, step_event]})[0]
    worker = plate.children[0]
    well = worker.children[0]
    step = well.children[0]

    assert round(step.percent, 1) == 50.0
    assert round(well.percent, 1) == 25.0
    assert round(worker.percent, 1) == 25.0
    assert round(plate.percent, 1) == 25.0


def test_execution_events_without_topology_render_shallow_node():
    step_event = _event(
        execution_id="exec-run",
        phase=ProgressPhase.STEP_STARTED,
        status=ProgressStatus.RUNNING,
        percent=25.0,
        step_name="ColorToGray",
    )

    plate = _nodes({"exec-run": [step_event]})[0]

    assert plate.status == "⚙️ Executing"
    assert round(plate.percent, 1) == 25.0
    assert plate.children == []


def test_compile_tree_marks_plate_as_compiled():
    compile_events = [
        _event(
            phase=ProgressPhase.COMPILE,
            status=ProgressStatus.SUCCESS,
            percent=100.0,
            axis_id=axis_id,
            step_name="compilation",
            total_wells=["A01", "B01"],
        )
        for axis_id in ("A01", "B01")
    ]

    plate = _nodes({"exec-1": compile_events})[0]

    assert round(plate.percent, 1) == 100.0
    assert plate.status == "✅ Compiled"


def test_compile_tree_marks_pipeline_level_failure():
    compile_failed = _event(
        phase=ProgressPhase.COMPILE,
        status=ProgressStatus.FAILED,
        percent=0.0,
        axis_id="pipeline",
        step_name="compilation",
    )

    plate = _nodes({"exec-1": [compile_failed]})[0]

    assert plate.status == "❌ Compile Failed"
    assert plate.children[0].status == "❌ Failed"


def test_worker_tree_marks_failed_well_and_plate():
    failed_event = _event(
        phase=ProgressPhase.AXIS_ERROR,
        status=ProgressStatus.ERROR,
        percent=42.0,
        completed=1,
        total=2,
        step_name="normalize",
        worker_slot="worker_0",
        owned_wells=["A01"],
    )

    plate = _nodes({"exec-1": [_init_event(), failed_event]})[0]
    worker = plate.children[0]
    well = worker.children[0]

    assert plate.status == "❌ Failed"
    assert worker.status == "❌ 1 failed"
    assert well.status == "❌ Failed"


def test_queued_plates_are_projected_without_progress_events():
    nodes = _nodes(
        {},
        _execution_server_info(
            queued=[
                {
                    "execution_id": "exec-123",
                    "plate_id": "/tmp/plate_a",
                    "queue_position": 1,
                },
                {
                    "execution_id": "exec-456",
                    "plate_id": "/tmp/plate_b",
                    "queue_position": 2,
                },
            ]
        ),
    )

    assert [node.status for node in nodes] == ["⏳ Queued", "⏳ Queued"]
    assert [node.info for node in nodes] == ["0.0% (q#1)", "0.0% (q#2)"]


def test_new_queued_execution_replaces_old_compiled_identity_for_same_plate():
    old_compile = _event(
        execution_id="exec-compile",
        plate_id="/tmp/plate_a",
        phase=ProgressPhase.COMPILE,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        axis_id="A01",
        step_name="compilation",
    )

    nodes = _nodes(
        {"exec-compile": [old_compile]},
        _execution_server_info(
            queued=[
                {
                    "execution_id": "exec-run",
                    "plate_id": "/tmp/plate_a",
                    "queue_position": 2,
                }
            ]
        ),
    )

    assert len(nodes) == 1
    assert nodes[0].execution_id == "exec-run"
    assert nodes[0].status == "⏳ Queued"
    assert nodes[0].children == []


def test_running_plate_is_projected_without_progress_events():
    nodes = _nodes(
        {},
        _execution_server_info(
            running=[
                {
                    "execution_id": "exec-run",
                    "plate_id": "/tmp/plate_a",
                    "compile_only": True,
                }
            ]
        ),
    )

    assert len(nodes) == 1
    assert nodes[0].status == "⏳ Compiling"


def test_running_snapshot_advances_init_only_projection_to_executing():
    nodes = _nodes(
        {"exec-run": [_init_event(execution_id="exec-run")]},
        _execution_server_info(
            running=[{"execution_id": "exec-run", "plate_id": "/tmp/plate"}]
        ),
    )

    assert len(nodes) == 1
    assert nodes[0].status == "⚙️ Executing"


def test_compile_events_with_worker_topology_stay_in_compile_mode():
    compile_event = _event(
        execution_id="exec-compile",
        phase=ProgressPhase.COMPILE,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        step_name="compilation",
        total_wells=["A01"],
    )

    plate = _nodes(
        {
            "exec-compile": [
                _init_event(execution_id="exec-compile"),
                compile_event,
            ]
        }
    )[0]

    assert plate.status == "✅ Compiled"
    assert plate.children[0].node_type == "compilation"


def test_update_from_progress_delegates_execution_server_rows_to_renderer():
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    server_info = _execution_server_info()

    class _FakeItem:
        def data(self, _column, _role):
            return server_info

    class _FakeTree:
        def __init__(self):
            self.item = _FakeItem()

        def topLevelItemCount(self):
            return 1

        def topLevelItem(self, _index):
            return self.item

    class _FakeRenderer:
        def __init__(self):
            self.calls = []

        def update_execution_server_item(self, item, data):
            self.calls.append((item, data))

    renderer = _FakeRenderer()
    manager.server_tree = _FakeTree()
    manager._progress_renderer = renderer
    manager._progress_dirty = True

    manager._update_from_progress()

    assert renderer.calls == [(manager.server_tree.topLevelItem(0), server_info)]
    assert manager._progress_dirty is False


def test_server_browser_does_not_own_a_second_progress_subscriber():
    assert not hasattr(ZMQServerManagerWidget, "_setup_progress_client")
    assert not hasattr(ZMQServerManagerWidget, "sync_progress_client_connection")
