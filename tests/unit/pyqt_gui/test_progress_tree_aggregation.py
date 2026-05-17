from openhcs.core.progress import ProgressEvent, ProgressPhase, ProgressStatus
from openhcs.pyqt_gui.widgets.shared.zmq_server_manager import ZMQServerManagerWidget
from openhcs.pyqt_gui.widgets.shared.server_browser import ProgressTreeBuilder
from pyqt_reactive.services import DefaultServerInfoParser


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
) -> ProgressEvent:
    return ProgressEvent(
        execution_id=execution_id,
        plate_id=plate_id,
        axis_id=axis_id,
        step_name=step_name,
        phase=phase,
        status=status,
        percent=percent,
        completed=completed,
        total=total,
        timestamp=1.0,
        pid=1111,
    )


def _execution_server_info(
    *,
    queued: list[dict] | None = None,
    running: list[dict] | None = None,
    compile_status: str | None = None,
):
    parser = DefaultServerInfoParser()
    payload = {
        "port": 7777,
        "ready": True,
        "server": "OpenHCSExecutionServer",
        "log_file_path": "/tmp/server.log",
        "workers": [],
        "running_executions": running or [],
        "queued_executions": queued or [],
    }
    if compile_status is not None:
        payload["compile_status"] = compile_status
        payload["compile_message"] = ""
    return parser.parse(payload)


def test_worker_tree_uses_pipeline_percent_for_well_and_parent_aggregation(
    monkeypatch,
):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._worker_assignments = {("exec-1", "/tmp/plate"): {"worker_0": ["A01"]}}
    manager._known_wells = {("exec-1", "/tmp/plate"): ["A01"]}

    pipeline_event = _event(
        phase=ProgressPhase.STEP_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=25.0,
        completed=1,
        total=4,
        step_name="normalize",
    )
    step_event = _event(
        phase=ProgressPhase.PATTERN_GROUP,
        status=ProgressStatus.RUNNING,
        percent=50.0,
        completed=1,
        total=2,
        step_name="max_project",
    )

    nodes = manager._build_progress_tree({"exec-1": [pipeline_event, step_event]})

    assert len(nodes) == 1
    plate = nodes[0]
    worker = plate.children[0]
    well = worker.children[0]
    step = well.children[0]

    assert round(step.percent, 1) == 50.0
    assert round(well.percent, 1) == 25.0
    assert round(worker.percent, 1) == 25.0
    assert round(plate.percent, 1) == 25.0


def test_compile_tree_marks_plate_as_compiled_at_100_percent(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._worker_assignments = {}
    manager._known_wells = {("exec-1", "/tmp/plate"): ["A01", "B01"]}

    compile_a = _event(
        phase=ProgressPhase.COMPILE,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        axis_id="A01",
        step_name="compilation",
    )
    compile_b = _event(
        phase=ProgressPhase.COMPILE,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        axis_id="B01",
        step_name="compilation",
    )

    nodes = manager._build_progress_tree({"exec-1": [compile_a, compile_b]})

    assert len(nodes) == 1
    plate = nodes[0]
    assert round(plate.percent, 1) == 100.0
    assert plate.status == "✅ Compiled"


def test_compile_tree_marks_failed_compilation(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._worker_assignments = {}
    manager._known_wells = {("exec-1", "/tmp/plate"): ["A01"]}

    compile_failed = _event(
        phase=ProgressPhase.COMPILE,
        status=ProgressStatus.FAILED,
        percent=35.0,
        axis_id="A01",
        step_name="compilation",
    )

    nodes = manager._build_progress_tree({"exec-1": [compile_failed]})

    assert len(nodes) == 1
    plate = nodes[0]
    assert round(plate.percent, 1) == 35.0
    assert plate.status == "❌ Compile Failed"
    assert plate.children[0].status == "❌ Failed"


def test_worker_tree_marks_failed_well_and_plate_status(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._worker_assignments = {("exec-1", "/tmp/plate"): {"worker_0": ["A01"]}}
    manager._known_wells = {("exec-1", "/tmp/plate"): ["A01"]}

    failed_event = _event(
        phase=ProgressPhase.AXIS_ERROR,
        status=ProgressStatus.ERROR,
        percent=42.0,
        completed=1,
        total=2,
        step_name="normalize",
    )

    nodes = manager._build_progress_tree({"exec-1": [failed_event]})

    assert len(nodes) == 1
    plate = nodes[0]
    worker = plate.children[0]
    well = worker.children[0]

    assert plate.status == "❌ Failed"
    assert worker.status == "❌ 1 failed"
    assert well.status == "❌ Failed"


def test_queued_plates_are_injected_without_progress_events(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()

    nodes = manager._merge_server_snapshot_nodes(
        [],
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

    assert len(nodes) == 2
    assert all(node.status == "⏳ Queued" for node in nodes)


def test_queued_overrides_compiled_plate_node(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._worker_assignments = {}
    manager._known_wells = {("exec-compile", "/tmp/plate_a"): ["A01"]}

    compiled_node = manager._build_progress_tree(
        {
            "exec-compile": [
                _event(
                    execution_id="exec-compile",
                    phase=ProgressPhase.COMPILE,
                    status=ProgressStatus.SUCCESS,
                    percent=100.0,
                    plate_id="/tmp/plate_a",
                    axis_id="A01",
                    step_name="compilation",
                )
            ]
        }
    )[0]

    merged = manager._merge_server_snapshot_nodes(
        [compiled_node],
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

    assert len(merged) == 1
    assert merged[0].status == "⏳ Queued"
    assert merged[0].percent == 0.0
    assert merged[0].info == "0.0% (q#2)"
    assert merged[0].children == []


def test_running_plate_is_injected_without_progress_events(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._get_plate_name = lambda plate_id, exec_id=None: (
        f"{plate_id.split('/')[-1]} ({exec_id[:8]})"
        if exec_id
        else plate_id.split("/")[-1]
    )

    merged = manager._merge_server_snapshot_nodes(
        [],
        _execution_server_info(
            running=[
                {
                    "execution_id": "exec-run",
                    "plate_id": "/tmp/plate_a",
                    "compile_only": True,
                }
            ],
        ),
    )

    assert len(merged) == 1
    assert merged[0].status == "⏳ Compiling"
    assert merged[0].percent == 0.0


def test_running_snapshot_without_progress_defaults_to_executing(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._get_plate_name = lambda plate_id, exec_id=None: plate_id.split("/")[-1]

    merged = manager._merge_server_snapshot_nodes(
        [],
        _execution_server_info(
            running=[{"execution_id": "exec-run", "plate_id": "/tmp/plate_a"}],
            compile_status="compiled success",
        ),
    )

    assert len(merged) == 1
    assert merged[0].status == "⚙️ Executing"
    assert merged[0].percent == 0.0


def test_running_compile_only_snapshot_without_compile_status_is_compiling(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._get_plate_name = lambda plate_id, exec_id=None: plate_id.split("/")[-1]

    merged = manager._merge_server_snapshot_nodes(
        [],
        _execution_server_info(
            running=[
                {
                    "execution_id": "exec-compile",
                    "plate_id": "/tmp/plate_a",
                    "compile_only": True,
                }
            ],
        ),
    )

    assert len(merged) == 1
    assert merged[0].status == "⏳ Compiling"
    assert merged[0].percent == 0.0


def test_progress_client_connection_tracks_execution_server_presence(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)

    class _DummyClient:
        def __init__(self) -> None:
            self.disconnected = False

        def is_connected(self) -> bool:
            return True

        def disconnect(self) -> None:
            self.disconnected = True

    calls = {"setup": 0}
    manager._zmq_client = None

    def _setup() -> None:
        calls["setup"] += 1
        manager._zmq_client = _DummyClient()

    manager._setup_progress_client = _setup

    manager.sync_progress_client_connection(
        [_execution_server_info(running=[], queued=[])]
    )
    assert calls["setup"] == 1
    assert manager._zmq_client is not None

    manager.sync_progress_client_connection(
        [_execution_server_info(running=[], queued=[])]
    )
    assert calls["setup"] == 1

    client = manager._zmq_client
    manager.sync_progress_client_connection([])
    assert manager._zmq_client is None
    assert client.disconnected is True


def test_init_only_execution_with_assignments_renders_as_queued_not_compiling(
    monkeypatch,
):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._worker_assignments = {("exec-run", "/tmp/plate"): {"worker_0": ["A01"]}}
    manager._known_wells = {("exec-run", "/tmp/plate"): ["A01"]}

    init_event = _event(
        execution_id="exec-run",
        phase=ProgressPhase.INIT,
        status=ProgressStatus.RUNNING,
        percent=0.0,
        axis_id="",
        step_name="init",
    )

    nodes = manager._build_progress_tree({"exec-run": [init_event]})

    assert len(nodes) == 1
    plate = nodes[0]
    assert plate.status == "⏳ Queued"
    assert plate.percent == 0.0
    # Worker shows "⚙️ Starting" because execution has started (INIT phase)
    # Wells still show "⏳ Queued" until they receive individual progress events
    assert plate.children[0].status == "⚙️ Starting"
    assert plate.children[0].children[0].status == "⏳ Queued"


def test_running_snapshot_does_not_override_progress_queued_node(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._get_plate_name = lambda plate_id, exec_id=None: plate_id.split("/")[-1]
    manager._worker_assignments = {("exec-run", "/tmp/plate"): {"worker_0": ["A01"]}}
    manager._known_wells = {("exec-run", "/tmp/plate"): ["A01"]}

    queued_plate = manager._build_progress_tree(
        {
            "exec-run": [
                _event(
                    execution_id="exec-run",
                    phase=ProgressPhase.INIT,
                    status=ProgressStatus.RUNNING,
                    percent=0.0,
                    plate_id="/tmp/plate",
                    axis_id="",
                    step_name="init",
                )
            ]
        }
    )[0]
    assert queued_plate.status == "⏳ Queued"

    merged = manager._merge_server_snapshot_nodes(
        [queued_plate],
        _execution_server_info(
            running=[{"execution_id": "exec-run", "plate_id": "/tmp/plate"}],
            compile_status="compiling",
        ),
    )

    assert len(merged) == 1
    assert merged[0].status == "⏳ Queued"


def test_compile_events_with_worker_assignments_stay_in_compile_mode(monkeypatch):
    manager = ZMQServerManagerWidget.__new__(ZMQServerManagerWidget)
    manager._progress_tree_builder = ProgressTreeBuilder()
    manager._worker_assignments = {
        ("exec-compile", "/tmp/plate"): {"worker_0": ["A01"]}
    }
    manager._known_wells = {("exec-compile", "/tmp/plate"): ["A01"]}

    compile_event = _event(
        execution_id="exec-compile",
        phase=ProgressPhase.COMPILE,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        plate_id="/tmp/plate",
        axis_id="A01",
        step_name="compilation",
    )
    init_event = _event(
        execution_id="exec-compile",
        phase=ProgressPhase.INIT,
        status=ProgressStatus.RUNNING,
        percent=0.0,
        plate_id="/tmp/plate",
        axis_id="",
        step_name="init",
    )

    nodes = manager._build_progress_tree({"exec-compile": [init_event, compile_event]})

    assert len(nodes) == 1
    plate = nodes[0]
    assert plate.status == "✅ Compiled"
    assert plate.children[0].node_type == "compilation"
