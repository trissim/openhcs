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
    DebugReplayMode,
)
from openhcs.core.artifacts import ArtifactKey, ArtifactScope, MeasurementsArtifactType
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.pyqt_gui.services.plate_manager_batch_workflow import (
    DebugSnapshotAvailableNotification,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_batch_workflow_service import (
    CompileBatchWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugProgressNotificationService,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_workflow_service import (
    DebugCompileArtifactCacheKey,
    DebugPlateRunRequest,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_server_status_presenter import (
    ExecutionServerStatusPresenter,
)
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
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
    CompileJob,
    CompileWorkflowService,
    PlatePipelineRequest,
)
from openhcs.pyqt_gui.widgets.shared.services.progress_workflow_service import (
    ProgressWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import (
    ZMQExecutionClientBoundary,
)
from openhcs.runtime.zmq_execution_client import ZMQExecutionRequestBuilder
from openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service import (
    LiveMeasurementAvailableNotification,
    LiveMeasurementProgressNotificationService,
)
from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.progress.live_measurements import (
    LiveMeasurementProgressPayload,
    LiveMeasurementTablePreview,
)
from openhcs.core.runtime_stores import RuntimeArtifactAddress, RuntimeArtifactLocation
from pyqt_reactive.services.zmq_server_info_parser import DefaultServerInfoParser
from zmqruntime.execution import BatchSubmitWaitEngine


def _identity_image(image):
    return image


def _job(plate_path: str) -> CompileJob:
    return CompileJob(
        plate_scope=PlateScopeIdentity.from_scope_id(plate_path),
        execution_plate_path=plate_path,
        selected_pipeline_path=None,
        plate_name=plate_path,
        definition_pipeline=[],
        pipeline_config={"x": 1},
    )


def _debug_run_spec() -> RunSpec:
    return RunSpec(
        plate_scope=PlateScopeIdentity.from_scope_id("/tmp/plate"),
        execution_plate_path="/tmp/plate/.openhcs_debug/execution",
        selected_pipeline_path="/tmp/plate/pipeline.cppipe",
        definition_pipeline=[],
        global_config=GlobalPipelineConfig(),
        pipeline_config=PipelineConfig(),
    )


def _zmq(client_service) -> ZMQExecutionClientBoundary:
    return ZMQExecutionClientBoundary(client_service)


async def _run_blocking_now(_loop, func):
    return func()


async def _connect_progress_client_stub():
    return object()


def _context(
    client_service,
    *,
    global_config_provider=lambda: object(),
    run_blocking=_run_blocking_now,
    connect_progress_client=_connect_progress_client_stub,
) -> BatchWorkflowContext:
    return BatchWorkflowContext(
        zmq=_zmq(client_service),
        global_config_provider=global_config_provider,
        run_blocking=run_blocking,
        connect_progress_client=connect_progress_client,
    )


def test_compile_policy_with_engine_collects_artifacts_and_callbacks():
    service = CompileBatchWorkflowService.__new__(CompileBatchWorkflowService)
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
        plate_scope=PlateScopeIdentity.from_scope_id("/tmp/plate"),
        execution_plate_path="/tmp/plate",
        selected_pipeline_path=None,
        definition_pipeline=[],
        global_config=object(),
        pipeline_config={"x": 1},
    )

    job = PlatePipelineRequestBuilder.compile_job_from_run_spec(
        run_spec=run_spec,
        config_params=debug_config_params,
    )

    assert job.config_params == debug_config_params


def test_debug_compile_cache_key_uses_runtime_replay_signature():
    run_spec = _debug_run_spec()
    debug_request = DebugPlateRunRequest(
        debug_session_id="debug-1",
        snapshot_store_ref="/tmp/snapshots/debug-1",
        snapshot_store_backend="local",
        command_type=DebugCommandType.STEP,
        selected_source_group="A01",
        pause_step_indices=(0,),
        replay_mode=DebugReplayMode.PERSISTENT_PAUSED_WORKER,
    )

    cache_key = DebugCompileArtifactCacheKey.from_run_spec(
        run_spec=run_spec,
        debug_request=debug_request,
    )
    expected_payload = ZMQExecutionRequestBuilder.from_task(
        run_spec.submission(
            global_config=run_spec.global_config,
            config_params=debug_request.compile_config_params,
        )
    ).request_payload

    assert cache_key.debug_replay_signature == expected_payload.debug_replay_signature


def test_debug_compile_cache_key_changes_for_new_debug_session():
    run_spec = _debug_run_spec()
    first_request = DebugPlateRunRequest(
        debug_session_id="debug-1",
        snapshot_store_ref="/tmp/snapshots/debug-1",
        snapshot_store_backend="local",
        command_type=DebugCommandType.STEP,
        selected_source_group="A01",
        pause_step_indices=(0,),
        replay_mode=DebugReplayMode.PERSISTENT_PAUSED_WORKER,
    )
    second_request = DebugPlateRunRequest(
        debug_session_id="debug-2",
        snapshot_store_ref="/tmp/snapshots/debug-2",
        snapshot_store_backend="local",
        command_type=DebugCommandType.STEP,
        selected_source_group="A01",
        pause_step_indices=(0,),
        replay_mode=DebugReplayMode.PERSISTENT_PAUSED_WORKER,
    )

    assert DebugCompileArtifactCacheKey.from_run_spec(
        run_spec=run_spec,
        debug_request=first_request,
    ) != DebugCompileArtifactCacheKey.from_run_spec(
        run_spec=run_spec,
        debug_request=second_request,
    )


def test_debug_compile_cache_key_ignores_cursor_only_debug_commands():
    run_spec = _debug_run_spec()
    first_request = DebugPlateRunRequest(
        debug_session_id="debug-1",
        snapshot_store_ref="/tmp/snapshots/debug-1",
        snapshot_store_backend="local",
        command_type=DebugCommandType.STEP,
        selected_source_group="A01",
        pause_step_indices=(0,),
        start_step_index=0,
        replay_mode=DebugReplayMode.PERSISTENT_PAUSED_WORKER,
    )
    second_request = DebugPlateRunRequest(
        debug_session_id="debug-1",
        snapshot_store_ref="/tmp/snapshots/debug-1",
        snapshot_store_backend="local",
        command_type=DebugCommandType.RUN_TO_PAUSE,
        selected_source_group="A01",
        pause_step_indices=(0,),
        start_step_index=4,
        start_after_invocation_key="default:0:color_to_gray",
        replay_mode=DebugReplayMode.PERSISTENT_PAUSED_WORKER,
    )

    assert DebugCompileArtifactCacheKey.from_run_spec(
        run_spec=run_spec,
        debug_request=first_request,
    ) == DebugCompileArtifactCacheKey.from_run_spec(
        run_spec=run_spec,
        debug_request=second_request,
    )


def test_compile_submission_uses_execution_plate_path_for_transport():
    captured = {}

    class FakeClient:
        def submit_compile(self, submission):
            captured["plate_id"] = submission.plate_id
            captured["execution_plate_id"] = submission.execution_plate_id
            captured["selected_pipeline_path"] = submission.selected_pipeline_path
            return {"status": "accepted", "execution_id": "compile-1"}

    async def run_blocking(_loop, func):
        return func()

    service = CompileWorkflowService(
        context=_context(
            SimpleNamespace(zmq_client=None),
            run_blocking=run_blocking,
        ),
    )
    plate_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/source",
        "/tmp/source/BBBC022_Analysis_Start.cppipe",
    )
    job = CompileJob(
        plate_scope=plate_scope,
        execution_plate_path="/tmp/source/.openhcs_cellprofiler/Analysis_Start",
        selected_pipeline_path="/tmp/source/BBBC022_Analysis_Start.cppipe",
        plate_name="Analysis_Start",
        definition_pipeline=[],
        pipeline_config={"x": 1},
    )

    execution_id = asyncio.run(
        service.submit_compile_job(job=job, zmq_client=FakeClient(), loop=object())
    )

    assert execution_id == "compile-1"
    assert captured["plate_id"] == plate_scope.scope_id
    assert (
        captured["execution_plate_id"]
        == "/tmp/source/.openhcs_cellprofiler/Analysis_Start"
    )
    assert (
        captured["selected_pipeline_path"]
        == "/tmp/source/BBBC022_Analysis_Start.cppipe"
    )


def test_compile_transport_normalizes_cellprofiler_submodule_function_collision():
    import importlib
    import pickle

    from openhcs.core.steps.function_step import FunctionStep

    crop_module = importlib.import_module(
        "openhcs.processing.backends.cellprofiler.crop"
    )
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

    pipeline = [FunctionStep(func=crop, name="Crop")]

    normalized = CompileWorkflowService.normalize_pipeline_for_transport(pipeline)

    assert normalized[0].func is cellprofiler_backend.crop
    pickle.dumps(normalized)


def test_plate_pipeline_submission_normalizes_compile_and_run_signatures():
    import importlib

    from openhcs.core.steps.function_step import FunctionStep

    crop_module = importlib.import_module(
        "openhcs.processing.backends.cellprofiler.crop"
    )
    pipeline = [
        FunctionStep(
            func=(crop_module, {"crop_shape": "Rectangle"}),
            name="Crop",
        )
    ]
    global_config = GlobalPipelineConfig()
    pipeline_config = PipelineConfig()
    run_spec = RunSpec(
        plate_scope=PlateScopeIdentity.from_scope_id("/tmp/plate"),
        execution_plate_path="/tmp/plate",
        selected_pipeline_path="/tmp/plate/pipeline.cppipe",
        definition_pipeline=pipeline,
        global_config=global_config,
        pipeline_config=pipeline_config,
    )
    normalized_compile_request = PlatePipelineRequest(
        plate_scope=run_spec.plate_scope,
        execution_plate_path=run_spec.execution_plate_path,
        selected_pipeline_path=run_spec.selected_pipeline_path,
        definition_pipeline=CompileWorkflowService.normalize_pipeline_for_transport(
            pipeline
        ),
        pipeline_config=pipeline_config,
    )

    run_payload = ZMQExecutionRequestBuilder.from_task(
        run_spec.submission(global_config=global_config)
    ).request_payload
    compile_payload = ZMQExecutionRequestBuilder.from_task(
        normalized_compile_request.submission(global_config=global_config)
    ).request_payload

    assert run_payload.pipeline_sha == compile_payload.pipeline_sha
    assert run_payload.request_signature == compile_payload.request_signature


def test_plate_pipeline_request_builder_rebinds_before_transport_validation(
    monkeypatch,
):
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.pyqt_gui.widgets.shared.services import (
        plate_pipeline_request_builder,
    )

    raw_step = FunctionStep(func=_identity_image, name="raw")
    rebound_step = FunctionStep(func=_identity_image, name="rebound")
    calls = []

    class Host:
        def get_pipeline_definition(self, plate_path: str):
            calls.append(("definition", plate_path))
            return [raw_step]

        def cellprofiler_import_result_for_plate(self, plate_path: str):
            del plate_path
            return None

    def rebind(**kwargs):
        calls.append(
            (
                "rebind",
                kwargs["import_result_provider"],
                kwargs["plate_path"],
                kwargs["pipeline_steps"],
            )
        )
        return [rebound_step]

    monkeypatch.setattr(
        plate_pipeline_request_builder.CellProfilerPipelineRuntimeBindingService,
        "runtime_bound_pipeline_for_plate",
        rebind,
    )

    pipeline = PlatePipelineRequestBuilder(Host())._definition_pipeline_for_plate(
        plate_path="/tmp/plate",
        display_name="plate",
    )

    assert pipeline == [rebound_step]
    assert calls[0] == ("definition", "/tmp/plate")
    assert calls[1][0] == "rebind"
    assert isinstance(calls[1][1], Host)
    assert calls[1][2:] == ("/tmp/plate", [raw_step])


def test_compile_transport_rejects_unresolved_module_objects():
    import math

    from openhcs.core.steps.function_step import FunctionStep

    pipeline = [FunctionStep(func=math, name="BadModule")]

    with pytest.raises(TypeError, match="module object"):
        CompileWorkflowService.normalize_pipeline_for_transport(pipeline)


def test_compile_policy_fail_fast_submit_raises():
    service = CompileBatchWorkflowService.__new__(CompileBatchWorkflowService)

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
    service = CompileBatchWorkflowService.__new__(CompileBatchWorkflowService)
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
        BatchSubmitWaitEngine[CompileJob]().run(
            [_job("/tmp/a"), _job("/tmp/b")], policy
        )
    )

    assert artifacts == {"/tmp/a": "exec-/tmp/a"}
    assert callbacks["start"] == ["/tmp/a", "/tmp/b"]
    assert callbacks["success"] == [("/tmp/a", "exec-/tmp/a")]
    assert callbacks["error"] == [("/tmp/b", "compile failed")]
    assert callbacks["finally"] == ["/tmp/a", "/tmp/b"]


class CompilePlateRowHostHarness:
    """Minimal host surface for compile-only row contract tests."""

    def __init__(self) -> None:
        self.execution_state = ManagerExecutionState.IDLE
        self.plate_compile_pending = set()
        self.plate_compiled_data = {}
        self.cleared_tracking = []
        self.statuses = []
        self.item_updates = 0
        self.button_updates = 0
        self.progress_started = []
        self.progress_updated = []
        self.progress_finished = 0
        self.compilation_errors = []
        self.orchestrator_states = []

    def emit_progress_started(self, total: int) -> None:
        self.progress_started.append(total)

    def clear_plate_execution_tracking(self, plate_path: str) -> None:
        self.cleared_tracking.append(plate_path)

    def update_item_list(self) -> None:
        self.item_updates += 1

    def emit_status(self, status: str) -> None:
        self.statuses.append(status)

    def emit_progress_updated(self, value: int) -> None:
        self.progress_updated.append(value)

    def emit_progress_finished(self) -> None:
        self.progress_finished += 1

    def update_button_states(self) -> None:
        self.button_updates += 1

    def emit_orchestrator_state(self, plate_path: str, state: str) -> None:
        self.orchestrator_states.append((plate_path, state))

    def emit_compilation_error(self, plate_name: str, error: str) -> None:
        self.compilation_errors.append((plate_name, error))


class RecordingPlateRequestBuilder:
    """Compile-job builder that records the row contract it receives."""

    def __init__(self) -> None:
        self.rows = []

    def build_compile_job_from_plate_row(self, row: PlateManagerRow) -> CompileJob:
        self.rows.append(row)
        return CompileJob(
            plate_scope=row.identity,
            execution_plate_path=row.scope_id,
            selected_pipeline_path=row.cppipe_path,
            plate_name=row.name,
            definition_pipeline=[],
            pipeline_config={"x": 1},
        )


class RecordingCompileBatchEngine:
    """Batch engine seam that records jobs without touching transport."""

    def __init__(self) -> None:
        self.jobs = []

    async def run(self, jobs, policy):
        del policy
        self.jobs = list(jobs)
        return {}


def test_compile_plates_accepts_plate_manager_rows() -> None:
    host = CompilePlateRowHostHarness()
    client_service = ClientServiceHarness()
    builder = RecordingPlateRequestBuilder()
    engine = RecordingCompileBatchEngine()

    async def connect_progress_client():
        return object()

    service = CompileBatchWorkflowService(
        host=host,
        context=_context(
            client_service,
            connect_progress_client=connect_progress_client,
        ),
        plate_request_builder=builder,
        compile_batch_engine=engine,
    )
    row = PlateManagerRow.from_scope("/tmp/plate")

    asyncio.run(service.compile_plates([row]))

    assert builder.rows == [row]
    assert [job.plate_path for job in engine.jobs] == [row.scope_id]
    assert host.cleared_tracking == [row.scope_id]
    assert host.progress_started == [1]
    assert host.progress_finished == 1
    assert host.compilation_errors == []
    assert client_service.disconnect_calls == 1


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
        self.plate_terminal_activity_status = ExecutionRuntimeHarness()
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
    service = ExecutionControlService.openhcs_default(
        host=host,
        context=_context(ClientServiceHarness()),
        port=7777,
    )

    service.check_all_completed()

    assert host.completed_notifications == [(2, 1)]


def test_execution_control_emits_cancelled_for_cancellable_plates() -> None:
    host = ExecutionControlHostHarness()
    service = ExecutionControlService.openhcs_default(
        host=host,
        context=_context(ClientServiceHarness()),
        port=7777,
    )

    service.emit_cancelled_for_all_plates()

    assert host.execution_completions == [
        ("/tmp/a", {"status": "cancelled"}),
        ("/tmp/b", {"status": "cancelled"}),
    ]


def test_execution_control_disconnect_uses_client_service() -> None:
    client_service = ClientServiceHarness()
    service = ExecutionControlService.openhcs_default(
        host=ExecutionControlHostHarness(),
        context=_context(client_service),
        port=7777,
    )

    service.disconnect()

    assert client_service.disconnect_sync_calls == 1
    assert client_service.zmq_client is None


def test_compile_submit_rejects_missing_zmq_client() -> None:
    service = CompileWorkflowService(
        context=_context(ClientServiceHarness()),
    )

    async def run_case() -> None:
        with pytest.raises(RuntimeError, match="ZMQ client is not connected"):
            await service.submit_compile_request(
                zmq_client=None,
                loop=object(),
                request=PlatePipelineRequest(
                    plate_scope=PlateScopeIdentity.from_scope_id("/tmp/plate"),
                    execution_plate_path="/tmp/plate",
                    selected_pipeline_path=None,
                    definition_pipeline=[],
                    pipeline_config=object(),
                ),
            )

    asyncio.run(run_case())


class RecordingProgressTracker:
    def __init__(self) -> None:
        self.events = []

    def register_event(self, execution_id, event) -> None:
        self.events.append((execution_id, event))

    def get_execution_ids(self):
        return tuple(
            dict.fromkeys(execution_id for execution_id, _event in self.events)
        )

    def get_events(self, execution_id):
        return [event for event_id, event in self.events if event_id == execution_id]

    def clear_execution(self, execution_id) -> None:
        self.events = [
            (event_id, event)
            for event_id, event in self.events
            if event_id != execution_id
        ]


class BatchWorkflowHostHarness:
    def __init__(self) -> None:
        self._progress_tracker = RecordingProgressTracker()
        self.runtime_progress_projection = None
        self.debug_runtime_projection = None
        self.execution_server_info = None
        self.statuses = []
        self.item_updates = 0

    def emit_status(self, status: str) -> None:
        self.statuses.append(status)

    def update_item_list(self) -> None:
        self.item_updates += 1


def _progress_service(
    *,
    host: BatchWorkflowHostHarness,
    client_service,
    on_dirty,
    live_measurements=None,
) -> tuple[ProgressWorkflowService, DebugProgressNotificationService]:
    debug_notifications = DebugProgressNotificationService()
    service = ProgressWorkflowService(
        host=host,
        context=_context(client_service),
        server_info_parser=DefaultServerInfoParser(),
        debug_notifications=debug_notifications,
        status_presenter=ExecutionServerStatusPresenter(),
        live_measurements=live_measurements,
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


def test_rebuild_runtime_projection_stores_debug_projection_bundle() -> None:
    host = BatchWorkflowHostHarness()
    client_service = SimpleNamespace(zmq_client=None)
    service, _debug_notifications = _progress_service(
        host=host,
        client_service=client_service,
        on_dirty=lambda: None,
    )
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
    service.rebuild_runtime_projection()

    assert host.runtime_progress_projection.get_plate("plate-1", "exec-1") is not None
    assert len(host.debug_runtime_projection.records) == 1
    assert host.debug_runtime_projection.current_frame.snapshot_id == "snapshot-1"
    assert host.item_updates == 1


def test_on_progress_notifies_live_measurement_listeners() -> None:
    host = BatchWorkflowHostHarness()
    client_service = SimpleNamespace(zmq_client=None)
    notifications: list[LiveMeasurementAvailableNotification] = []
    live_measurements = LiveMeasurementProgressNotificationService()
    live_measurements.add_listener(notifications.append)
    service, _debug_notifications = _progress_service(
        host=host,
        client_service=client_service,
        on_dirty=lambda: None,
        live_measurements=live_measurements,
    )
    payload = LiveMeasurementProgressPayload(
        previews=(
            LiveMeasurementTablePreview(
                address=RuntimeArtifactAddress(
                    key=ArtifactKey(
                        name="Measure",
                        artifact_type=MeasurementsArtifactType,
                        scope=ArtifactScope(axis_id="A01"),
                    ),
                    location=RuntimeArtifactLocation(
                        path="/memory/measure.pkl",
                        backend="memory",
                    ),
                ),
                columns=("mean",),
                rows=({"mean": 3.0},),
                row_count=1,
                truncated_rows=False,
                truncated_columns=False,
            ),
        ),
        preview_count=1,
        truncated_previews=False,
    )
    event = ProgressEvent(
        identity=ProgressIdentity(
            execution_id="exec-1",
            plate_id="plate-1",
            axis_id="A01",
            step_name="Measure",
        ),
        phase=ProgressPhase.STEP_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        completed=1,
        total=1,
        timestamp=1.0,
        pid=1234,
        context=payload.to_context(),
    )

    service.on_progress(event.to_dict())

    assert len(notifications) == 1
    assert notifications[0].payload.previews[0].rows == ({"mean": 3.0},)


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
        axis_id="A01",
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
