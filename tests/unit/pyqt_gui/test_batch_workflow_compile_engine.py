import asyncio
from types import SimpleNamespace

import pytest

from openhcs.core.debug import (
    DebugCommandType,
    DebugCursor,
    DebugEvent,
    DebugEventType,
    DebugExecutionConfig,
    DebugProgressEventRequest,
)
from openhcs.pyqt_gui.widgets.shared.services.batch_workflow_service import (
    BatchWorkflowService,
    CompileJob,
    DebugSnapshotAvailableNotification,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugProgressNotificationService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_server_status_presenter import (
    ExecutionServerStatusPresenter,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_control_service import (
    ExecutionControlService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ManagerExecutionState,
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    PlatePipelineRequestBuilder,
    RunSpec,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.progress_workflow_service import (
    ProgressWorkflowService,
)
from pyqt_reactive.services import DefaultServerInfoParser
from zmqruntime.execution import BatchSubmitWaitEngine


def _job(plate_path: str) -> CompileJob:
    return CompileJob(
        plate_path=plate_path,
        plate_name=plate_path,
        definition_pipeline=[],
        pipeline_config={"x": 1},
    )


def test_compile_policy_with_engine_collects_artifacts_and_callbacks():
    service = BatchWorkflowService.__new__(BatchWorkflowService)
    callback_events: list[tuple[str, str, str]] = []

    async def fake_submit_compile_job(*, job: CompileJob, zmq_client, loop) -> str:
        return f"exec-{job.plate_path}"

    async def fake_wait_compile_job(
        *, submission_id: str, job: CompileJob, zmq_client, loop
    ) -> None:
        callback_events.append(("wait", job.plate_path, submission_id))

    service._submit_compile_job = fake_submit_compile_job
    service._wait_compile_job = fake_wait_compile_job

    jobs = [_job("/tmp/a"), _job("/tmp/b")]
    policy = service._make_compile_policy(
        zmq_client=object(),
        loop=object(),
        fail_fast_submit=False,
        fail_fast_wait=False,
        on_wait_success=lambda job, submission_id, _idx, _total: callback_events.append(
            ("success", job.plate_path, submission_id)
        ),
    )
    artifacts = asyncio.run(BatchSubmitWaitEngine[CompileJob]().run(jobs, policy))

    assert artifacts == {
        "/tmp/a": "exec-/tmp/a",
        "/tmp/b": "exec-/tmp/b",
    }
    assert callback_events == [
        ("wait", "/tmp/a", "exec-/tmp/a"),
        ("success", "/tmp/a", "exec-/tmp/a"),
        ("wait", "/tmp/b", "exec-/tmp/b"),
        ("success", "/tmp/b", "exec-/tmp/b"),
    ]


def test_build_compile_job_preserves_debug_config_params():
    debug_config_params = DebugExecutionConfig(
        debug_session_id="debug-1",
        command_type=DebugCommandType.STEP,
    ).to_config_params()
    run_spec = RunSpec(
        plate_path="/tmp/plate",
        definition_pipeline=[],
        global_config=object(),
        pipeline_config={"x": 1},
    )

    job = PlatePipelineRequestBuilder.compile_job_from_run_spec(
        run_spec=run_spec,
        config_params=debug_config_params,
    )

    assert job.config_params == debug_config_params


def test_compile_transport_normalizes_cellprofiler_submodule_function_collision():
    import importlib
    import pickle

    from openhcs.core.steps.function_step import FunctionStep

    crop_module = importlib.import_module("openhcs.processing.backends.cellprofiler.crop")
    pipeline = [
        FunctionStep(
            func=(crop_module, {"crop_shape": "Rectangle"}),
            name="Crop",
        )
    ]

    normalized = CompileWorkflowService.normalize_pipeline_for_transport(pipeline)

    normalized_func = normalized[0].func[0]
    assert callable(normalized_func)
    assert normalized_func.__name__ == "crop"
    assert normalized_func.__module__ == "openhcs.processing.backends.cellprofiler"
    pickle.dumps(normalized)


def test_compile_transport_preserves_stable_cellprofiler_function_wrappers():
    import pickle

    import openhcs.processing.backends.cellprofiler as cellprofiler_backend
    from openhcs.core.steps.function_step import FunctionStep

    crop = cellprofiler_backend.crop
    cellprofiler_backend._cellprofiler_function_maps.cache_clear()

    pipeline = [FunctionStep(func=crop, name="Crop")]

    normalized = CompileWorkflowService.normalize_pipeline_for_transport(pipeline)

    assert normalized[0].func is cellprofiler_backend.crop
    pickle.dumps(normalized)


def test_compile_transport_rejects_unresolved_module_objects():
    import math

    from openhcs.core.steps.function_step import FunctionStep

    pipeline = [FunctionStep(func=math, name="BadModule")]

    with pytest.raises(TypeError, match="module object"):
        CompileWorkflowService.normalize_pipeline_for_transport(pipeline)


def test_compile_policy_fail_fast_submit_raises():
    service = BatchWorkflowService.__new__(BatchWorkflowService)

    async def fake_submit_compile_job(*, job: CompileJob, zmq_client, loop) -> str:
        if job.plate_path == "/tmp/b":
            raise RuntimeError("boom")
        return f"exec-{job.plate_path}"

    async def fake_wait_compile_job(
        *, submission_id: str, job: CompileJob, zmq_client, loop
    ) -> None:
        return None

    service._submit_compile_job = fake_submit_compile_job
    service._wait_compile_job = fake_wait_compile_job

    policy = service._make_compile_policy(
        zmq_client=object(),
        loop=object(),
        fail_fast_submit=True,
        fail_fast_wait=True,
    )

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            BatchSubmitWaitEngine[CompileJob]().run(
                [_job("/tmp/a"), _job("/tmp/b"), _job("/tmp/c")], policy
            )
        )

def test_compile_policy_non_fail_fast_wait_tracks_error_and_finally():
    service = BatchWorkflowService.__new__(BatchWorkflowService)
    callbacks = {"start": [], "success": [], "error": [], "finally": []}

    async def fake_submit_compile_job(*, job: CompileJob, zmq_client, loop) -> str:
        return f"exec-{job.plate_path}"

    async def fake_wait_compile_job(
        *, submission_id: str, job: CompileJob, zmq_client, loop
    ) -> None:
        if job.plate_path == "/tmp/b":
            raise RuntimeError("compile failed")

    service._submit_compile_job = fake_submit_compile_job
    service._wait_compile_job = fake_wait_compile_job

    policy = service._make_compile_policy(
        zmq_client=object(),
        loop=object(),
        fail_fast_submit=False,
        fail_fast_wait=False,
        on_wait_start=lambda job, _idx, _total: callbacks["start"].append(
            job.plate_path
        ),
        on_wait_success=lambda job, execution_id, _idx, _total: callbacks[
            "success"
        ].append((job.plate_path, execution_id)),
        on_wait_error=lambda job, error, _idx, _total: callbacks["error"].append(
            (job.plate_path, str(error))
        ),
        on_wait_finally=lambda job, _idx, _total: callbacks["finally"].append(
            job.plate_path
        ),
    )
    artifacts = asyncio.run(
        BatchSubmitWaitEngine[CompileJob]().run([_job("/tmp/a"), _job("/tmp/b")], policy)
    )

    assert artifacts == {"/tmp/a": "exec-/tmp/a"}
    assert callbacks["start"] == ["/tmp/a", "/tmp/b"]
    assert callbacks["success"] == [("/tmp/a", "exec-/tmp/a")]
    assert callbacks["error"] == [("/tmp/b", "compile failed")]
    assert callbacks["finally"] == ["/tmp/a", "/tmp/b"]


class ExecutionRuntimeHarness:
    def __init__(self) -> None:
        self.active_plates = ["/tmp/a"]
        self.marked_terminal = []
        self._cancellable_plates = ["/tmp/a", "/tmp/b"]

    def all_batch_terminal(self) -> bool:
        return True

    def terminal_counts(self) -> tuple[int, int]:
        return (2, 1)

    def mark_terminal(self, plate_path: str, status: TerminalExecutionStatus) -> None:
        self.marked_terminal.append((plate_path, status))

    def cancellable_plates(self) -> list[str]:
        return list(self._cancellable_plates)


class ExecutionControlHostHarness:
    def __init__(self) -> None:
        self.execution_state = ManagerExecutionState.RUNNING
        self.execution_runtime = ExecutionRuntimeHarness()
        self.completed_notifications = []
        self.execution_completions = []
        self.current_execution_id = "exec-1"
        self.item_updates = 0
        self.button_updates = 0

    def notify_all_plates_completed(self, completed: int, failed: int) -> None:
        self.completed_notifications.append((completed, failed))

    def emit_execution_complete(self, result: dict, plate_path: str) -> None:
        self.execution_completions.append((plate_path, result))

    def update_item_list(self) -> None:
        self.item_updates += 1

    def update_button_states(self) -> None:
        self.button_updates += 1


class ClientServiceHarness:
    def __init__(self) -> None:
        self.zmq_client = object()
        self.disconnect_sync_calls = 0
        self.disconnect_calls = 0

    def disconnect_sync(self) -> None:
        self.disconnect_sync_calls += 1
        self.zmq_client = None

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.zmq_client = None


def test_execution_control_notifies_when_batch_terminal() -> None:
    host = ExecutionControlHostHarness()
    service = ExecutionControlService(
        host=host,
        client_service=ClientServiceHarness(),
        port=7777,
    )

    service.check_all_completed()

    assert host.completed_notifications == [(2, 1)]


def test_execution_control_emits_cancelled_for_cancellable_plates() -> None:
    host = ExecutionControlHostHarness()
    service = ExecutionControlService(
        host=host,
        client_service=ClientServiceHarness(),
        port=7777,
    )

    service.emit_cancelled_for_all_plates()

    assert host.execution_completions == [
        ("/tmp/a", {"status": "cancelled"}),
        ("/tmp/b", {"status": "cancelled"}),
    ]


def test_execution_control_disconnect_uses_client_service() -> None:
    client_service = ClientServiceHarness()
    service = ExecutionControlService(
        host=ExecutionControlHostHarness(),
        client_service=client_service,
        port=7777,
    )

    service.disconnect()

    assert client_service.disconnect_sync_calls == 1
    assert client_service.zmq_client is None


def test_compile_submit_rejects_missing_zmq_client() -> None:
    service = CompileWorkflowService(
        global_config_provider=lambda: object(),
        run_blocking=lambda _loop, func: func(),
    )

    async def run_case() -> None:
        with pytest.raises(RuntimeError, match="ZMQ client is not connected"):
            await service.submit_compile_request(
                zmq_client=None,
                loop=object(),
                plate_path="/tmp/plate",
                definition_pipeline=[],
                pipeline_config=object(),
            )

    asyncio.run(run_case())


class RecordingProgressTracker:
    def __init__(self) -> None:
        self.events = []

    def register_event(self, execution_id, event) -> None:
        self.events.append((execution_id, event))


class BatchWorkflowHostHarness:
    def __init__(self) -> None:
        self._progress_tracker = RecordingProgressTracker()


def _progress_service(
    *,
    host: BatchWorkflowHostHarness,
    client_service,
    on_dirty,
) -> tuple[ProgressWorkflowService, DebugProgressNotificationService]:
    debug_notifications = DebugProgressNotificationService()
    service = ProgressWorkflowService(
        host=host,
        client_service=client_service,
        server_info_parser=DefaultServerInfoParser(),
        debug_notifications=debug_notifications,
        status_presenter=ExecutionServerStatusPresenter(),
        on_dirty=on_dirty,
        start_timer=False,
    )
    return service, debug_notifications


def test_mark_dirty_accepts_progress_tracker_listener_event() -> None:
    host = BatchWorkflowHostHarness()
    client_service = SimpleNamespace(zmq_client=None)
    dirty = {"count": 0}
    service, _debug_notifications = _progress_service(
        host=host,
        client_service=client_service,
        on_dirty=lambda: dirty.__setitem__("count", dirty["count"] + 1),
    )

    service.mark_dirty("exec-1", object())

    assert dirty["count"] == 1


def test_on_progress_notifies_debug_snapshot_listeners() -> None:
    host = BatchWorkflowHostHarness()
    client_service = SimpleNamespace(zmq_client=None)
    dirty = {"count": 0}
    notifications: list[DebugSnapshotAvailableNotification] = []
    service, debug_notifications = _progress_service(
        host=host,
        client_service=client_service,
        on_dirty=lambda: dirty.__setitem__("count", dirty["count"] + 1),
    )
    debug_notifications.add_listener(notifications.append)
    cursor = DebugCursor(
        step_index=1,
        step_scope_id="step-1",
        group_key="default",
        invocation_key="default:0:segment",
    )
    event = DebugEvent(
        event_type=DebugEventType.AFTER_INVOCATION,
        cursor=cursor,
        step_name="IdentifyPrimaryObjects",
        callable_name="IdentifyPrimaryObjects",
        axis_id="A01",
    )
    progress_event = DebugProgressEventRequest(
        debug_session_id="debug-1",
        debug_event=event,
        execution_id="exec-1",
        plate_id="plate-1",
        snapshot_id="snapshot-1",
        snapshot_store_ref="/tmp/debug",
    ).to_progress_event()

    service.on_progress(progress_event.to_dict())

    assert dirty["count"] == 1
    assert host._progress_tracker.events == [("exec-1", progress_event)]
    assert len(notifications) == 1
    assert notifications[0].debug_context.snapshot_id == "snapshot-1"
    assert notifications[0].debug_context.snapshot_store_ref == "/tmp/debug"


def test_on_progress_ignores_debug_events_without_snapshot_id() -> None:
    host = BatchWorkflowHostHarness()
    client_service = SimpleNamespace(zmq_client=None)
    notifications: list[DebugSnapshotAvailableNotification] = []
    service, debug_notifications = _progress_service(
        host=host,
        client_service=client_service,
        on_dirty=lambda: None,
    )
    debug_notifications.add_listener(notifications.append)
    cursor = DebugCursor(
        step_index=1,
        step_scope_id="step-1",
        group_key="default",
        invocation_key="default:0:segment",
    )
    event = DebugEvent(
        event_type=DebugEventType.BEFORE_INVOCATION,
        cursor=cursor,
        step_name="IdentifyPrimaryObjects",
    )
    progress_event = DebugProgressEventRequest(
        debug_session_id="debug-1",
        debug_event=event,
        execution_id="exec-1",
        plate_id="plate-1",
    ).to_progress_event()

    service.on_progress(progress_event.to_dict())

    assert notifications == []


def test_on_progress_attaches_server_read_debug_snapshot() -> None:
    host = BatchWorkflowHostHarness()
    cursor = DebugCursor(
        step_index=1,
        step_scope_id="step-1",
        group_key="default",
        invocation_key="default:0:segment",
    )
    event = DebugEvent(
        event_type=DebugEventType.AFTER_INVOCATION,
        cursor=cursor,
        step_name="IdentifyPrimaryObjects",
        callable_name="IdentifyPrimaryObjects",
        axis_id="A01",
    )
    snapshot = event.to_snapshot(snapshot_id="snapshot-1")
    client_service = SimpleNamespace(
        zmq_client=SimpleNamespace(get_debug_snapshot=lambda **_kwargs: snapshot)
    )
    notifications: list[DebugSnapshotAvailableNotification] = []
    service, debug_notifications = _progress_service(
        host=host,
        client_service=client_service,
        on_dirty=lambda: None,
    )
    debug_notifications.add_listener(notifications.append)
    progress_event = DebugProgressEventRequest(
        debug_session_id="debug-1",
        debug_event=event,
        execution_id="exec-1",
        plate_id="plate-1",
        snapshot_id="snapshot-1",
        snapshot_store_ref="/tmp/debug",
    ).to_progress_event()

    service.on_progress(progress_event.to_dict())

    assert notifications[0].snapshot == snapshot
