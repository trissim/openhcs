import threading
import time
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from nominal_refactor_advisor.descriptor_algebra import AliasProperty

from openhcs.constants.constants import MEMORY_TYPE_NUMPY
from openhcs.core.callable_contract import CallableContract
from openhcs.core.debug import (
    DebugArtifactRef,
    DebugArtifactExportControlPayload,
    DebugArtifactExportPlan,
    DebugArtifactExportRequest,
    DebugArtifactExportResponse,
    DebugArtifactIdentity,
    DebugCommandType,
    DebugCursor,
    DebugBoundaryState,
    DebugEvent,
    DebugEventType,
    DebugExecutionConfig,
    DebugExecutionPolicy,
    DebugInvocationParameter,
    DebugProgressContext,
    DebugProgressEventRequest,
    DebugPausedWorkerRegistry,
    DebugSnapshotReadRequest,
    DebugSnapshotReadControlPayload,
    DebugSnapshotReadResponse,
    DebugSnapshot,
    DebugSnapshotFileManagerContext,
    DebugReplayMode,
    DebugSinkInstallRequest,
    DebugSession,
    DebugSessionRequest,
    FileManagerDebugSnapshotStore,
    LocalSnapshotProgressDebugEventSink,
    LocalDebugSnapshotStore,
    NoOpDebugEventSink,
    ProgressDebugEventSink,
    ProgressDebugExecutionPolicy,
    RecordingDebugEventSink,
    DebugWorkerCommandRequest,
    DebugWorkerCommandControlPayload,
    DebugWorkerCommandResponse,
    DebugWarmReplayArtifactReusePlan,
    debug_event_sink_from_context,
)
from openhcs.core.progress import ProgressEvent, ProgressPhase, ProgressStatus
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.function_patterns import (
    CompiledFunctionInvocation,
    FunctionInvocationKey,
)
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    SourceBindingRuntimeContext,
)
from openhcs.core.steps.function_runtime import (
    FunctionChainExecutionRequest,
    execute_function_chain,
)


class DebugRuntimeFixture:
    """Nominal authority for debug-runtime test products."""

    DEBUG_SNAPSHOT_BACKEND = "memory"
    DEBUG_SESSION_ID = "debug-1"
    EXECUTION_ID = "exec-1"
    SNAPSHOT_STORE_REF = "/tmp/debug/debug-1"
    SNAPSHOT_ID = "snap-1"
    STEP_SCOPE_ID = "scope"
    GROUP_KEY = "default"
    AXIS_ID = "A01"
    PLATE_ID = "plate"
    SEGMENT_NAME = "segment"

    @staticmethod
    def double(image):
        return image * 2

    @staticmethod
    def plus_one(image):
        return image + 1

    @staticmethod
    def raising(image):
        raise RuntimeError("debug boom")

    @classmethod
    def compiled_invocation(
        cls,
        func,
        *,
        position: int = 0,
        artifact_input_keys: tuple[str, ...] = (),
        artifact_output_keys: tuple[str, ...] = (),
    ) -> CompiledFunctionInvocation:
        return CompiledFunctionInvocation(
            key=FunctionInvocationKey.from_contract(
                CallableContract.from_callable(func),
                cls.GROUP_KEY,
                position,
            ),
            contract=CallableContract(
                func=func,
                function_name=func.__name__,
                module_name=func.__module__,
                input_memory_type=MEMORY_TYPE_NUMPY,
                output_memory_type=MEMORY_TYPE_NUMPY,
            ),
            artifact_input_keys=artifact_input_keys,
            artifact_output_keys=artifact_output_keys,
        )

    @staticmethod
    def execution_plan():
        return SimpleNamespace(
            step_index=3,
            step_scope_id="plate::functionstep_3",
            step_name="debuggable",
            axis_id=DebugRuntimeFixture.AXIS_ID,
            input_memory_type=MEMORY_TYPE_NUMPY,
            device_id=0,
            source_binding_plan=CompiledSourceBindingPlan.empty(),
        )

    @classmethod
    def cursor(cls) -> DebugCursor:
        return DebugCursor(
            step_index=2,
            step_scope_id=cls.STEP_SCOPE_ID,
            group_key=cls.GROUP_KEY,
            invocation_key=f"{cls.GROUP_KEY}:0:{cls.SEGMENT_NAME}",
        )

    @classmethod
    def debug_event(cls, *, event_type: DebugEventType) -> DebugEvent:
        return DebugEvent(
            event_type=event_type,
            cursor=cls.cursor(),
            step_name=cls.SEGMENT_NAME,
            callable_name=cls.SEGMENT_NAME,
            axis_id=cls.AXIS_ID,
        )

    @classmethod
    def sink_install_request(cls, *, context: object) -> DebugSinkInstallRequest:
        return DebugSinkInstallRequest(
            context=context,
            execution_id=cls.EXECUTION_ID,
            plate_id=cls.PLATE_ID,
            worker_slot="worker_0",
            owned_wells=("A01",),
        )


class DebugSnapshotFileManagerStub:
    """In-memory FileManager surface for VFS-backed debug snapshot tests."""

    def __init__(self) -> None:
        self.saved: dict[str, object] = {}
        self.directories: set[str] = set()

    def ensure_directory(self, directory, backend):
        self.directories.add(str(directory))
        return str(directory)

    def save(self, data, output_path, backend, **kwargs):
        del backend, kwargs
        self.saved[str(output_path)] = data

    def load(self, file_path, backend, **kwargs):
        del backend, kwargs
        return self.saved[str(file_path)]

    def list_files(self, directory, backend, **kwargs):
        del backend
        recursive = bool(kwargs.get("recursive", False))
        directory_prefix = str(directory).rstrip("/") + "/"
        files = []
        for path in self.saved:
            if not path.startswith(directory_prefix):
                continue
            relative_path = path[len(directory_prefix):]
            if recursive or "/" not in relative_path:
                files.append(path)
        return sorted(files)

    def exists(self, path, backend):
        del backend
        path_text = str(path).rstrip("/")
        return path_text in self.directories or any(
            saved_path == str(path) or saved_path.startswith(path_text + "/")
            for saved_path in self.saved
        )

    def is_dir(self, path, backend):
        del backend
        path_text = str(path).rstrip("/")
        return path_text in self.directories


class DebugSnapshotContextStub(DebugSnapshotFileManagerContext):
    """Nominal debug snapshot context used by policy tests."""

    filemanager = AliasProperty[DebugSnapshotFileManagerStub]("_filemanager")

    def __init__(self, filemanager: DebugSnapshotFileManagerStub) -> None:
        self._filemanager = filemanager


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_debug_cursor_uses_invocation_identity():
    invocation = DebugRuntimeFixture.compiled_invocation(
        DebugRuntimeFixture.double,
        position=2,
    )

    cursor = DebugCursor.from_invocation(
        step_index=1,
        step_scope_id=DebugRuntimeFixture.STEP_SCOPE_ID,
        invocation=invocation,
        pattern_group_identity="0",
    )

    assert cursor.step_index == 1
    assert cursor.step_scope_id == DebugRuntimeFixture.STEP_SCOPE_ID
    assert cursor.group_key == DebugRuntimeFixture.GROUP_KEY
    assert cursor.invocation_key == f"{DebugRuntimeFixture.GROUP_KEY}:2:double"
    assert cursor.pattern_group_identity == "0"


def test_debug_cursor_matches_invocation_key_parts():
    cursor = DebugRuntimeFixture.cursor()

    assert cursor.matches_invocation_key_parts(
        group_key=DebugRuntimeFixture.GROUP_KEY,
        position=0,
        function_name=DebugRuntimeFixture.SEGMENT_NAME,
    )
    assert not cursor.matches_invocation_key_parts(
        group_key=DebugRuntimeFixture.GROUP_KEY,
        position=1,
        function_name=DebugRuntimeFixture.SEGMENT_NAME,
    )


def test_debug_execution_config_normalizes_compile_cache_payload():
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        snapshot_store_ref=DebugRuntimeFixture.SNAPSHOT_STORE_REF,
        command_type=DebugCommandType.STEP,
        selected_source_group=DebugRuntimeFixture.AXIS_ID,
        start_step_index=3,
        start_after_invocation_key="default:0:old",
        replay_mode=DebugReplayMode.PERSISTENT_PAUSED_WORKER,
    )

    compile_payload = config.compile_cache_config_params()[
        DebugExecutionConfig.CONFIG_PARAMS_KEY
    ]

    assert compile_payload["command_type"] == DebugCommandType.RUN.value
    assert compile_payload["start_step_index"] == 0
    assert compile_payload["start_after_invocation_key"] is None
    assert compile_payload["selected_source_group"] == DebugRuntimeFixture.AXIS_ID
    assert (
        compile_payload["replay_mode"]
        == DebugReplayMode.PERSISTENT_PAUSED_WORKER.value
    )


def test_execute_chain_emits_debug_invocation_events():
    sink = RecordingDebugEventSink()
    context = SimpleNamespace(debug_event_sink=sink)
    request = FunctionChainExecutionRequest(
        initial_data_stack=np.array([1, 2, 3]),
        invocations=(
            DebugRuntimeFixture.compiled_invocation(DebugRuntimeFixture.double),
        ),
        context=context,
        execution_plan=DebugRuntimeFixture.execution_plan(),
        artifact_inputs={},
        artifact_outputs={},
        runtime_plane_index=0,
        source_binding_context=SourceBindingRuntimeContext.empty(),
    )

    result = execute_function_chain(request)

    np.testing.assert_array_equal(result, np.array([2, 4, 6]))
    assert [event.event_type for event in sink.events] == [
        DebugEventType.BEFORE_INVOCATION,
        DebugEventType.AFTER_INVOCATION,
    ]
    assert (
        sink.events[0].cursor.invocation_key
        == f"{DebugRuntimeFixture.GROUP_KEY}:0:double"
    )


def test_execute_chain_debug_events_include_planned_artifact_refs():
    sink = RecordingDebugEventSink()
    context = SimpleNamespace(debug_event_sink=sink)
    request = FunctionChainExecutionRequest(
        initial_data_stack=np.array([1, 2, 3]),
        invocations=(
            DebugRuntimeFixture.compiled_invocation(
                DebugRuntimeFixture.double,
                artifact_output_keys=("measurements", "relationships"),
            ),
        ),
        context=context,
        execution_plan=DebugRuntimeFixture.execution_plan(),
        artifact_inputs={},
        artifact_outputs={
            "measurements": ArtifactOutputPlan(
                name="measurements",
                path="/debug/measurements.csv",
                kind=ArtifactKind.MEASUREMENTS,
            ),
            "relationships": ArtifactOutputPlan(
                name="relationships",
                path="/debug/relationships.csv",
                kind=ArtifactKind.RELATIONSHIPS,
            ),
        },
        runtime_plane_index=0,
        source_binding_context=SourceBindingRuntimeContext.empty(),
    )

    result = execute_function_chain(request)

    np.testing.assert_array_equal(result, np.array([2, 4, 6]))
    after_event = sink.events[-1]
    assert after_event.input_artifact_refs == ()
    assert [ref.name for ref in after_event.output_artifact_refs] == [
        "measurements",
        "relationships",
    ]
    assert [ref.name for ref in after_event.measurement_refs] == ["measurements"]
    assert [ref.name for ref in after_event.relationship_refs] == ["relationships"]
    assert all(ref.cursor == after_event.cursor for ref in after_event.output_artifact_refs)


def test_debug_step_executes_one_function_pattern_invocation():
    emitted: list[ProgressEvent] = []
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        command_type=DebugCommandType.STEP,
    )
    context = SimpleNamespace()
    ProgressDebugExecutionPolicy(config).install_context_sink(
        DebugRuntimeFixture.sink_install_request(context=context)
    )
    context.debug_event_sink.emit_progress = emitted.append
    request = FunctionChainExecutionRequest(
        initial_data_stack=np.array([1, 2, 3]),
        invocations=(
            DebugRuntimeFixture.compiled_invocation(DebugRuntimeFixture.double, position=0),
            DebugRuntimeFixture.compiled_invocation(DebugRuntimeFixture.plus_one, position=1),
        ),
        context=context,
        execution_plan=DebugRuntimeFixture.execution_plan(),
        artifact_inputs={},
        artifact_outputs={},
        runtime_plane_index=0,
        source_binding_context=SourceBindingRuntimeContext.empty(),
    )

    result = execute_function_chain(request)

    np.testing.assert_array_equal(result, np.array([2, 4, 6]))
    assert [
        DebugProgressContext.from_progress_context(event.context).event_type
        for event in emitted
    ] == [DebugEventType.BEFORE_INVOCATION, DebugEventType.AFTER_INVOCATION]


def test_debug_step_can_advance_past_current_function_pattern_invocation():
    emitted: list[ProgressEvent] = []
    first_invocation = DebugRuntimeFixture.compiled_invocation(
        DebugRuntimeFixture.double,
        position=0,
    )
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        command_type=DebugCommandType.STEP,
        start_after_invocation_key=DebugCursor.invocation_key_text(first_invocation),
    )
    context = SimpleNamespace()
    ProgressDebugExecutionPolicy(config).install_context_sink(
        DebugRuntimeFixture.sink_install_request(context=context)
    )
    context.debug_event_sink.emit_progress = emitted.append
    request = FunctionChainExecutionRequest(
        initial_data_stack=np.array([1, 2, 3]),
        invocations=(
            first_invocation,
            DebugRuntimeFixture.compiled_invocation(DebugRuntimeFixture.plus_one, position=1),
            DebugRuntimeFixture.compiled_invocation(DebugRuntimeFixture.double, position=2),
        ),
        context=context,
        execution_plan=DebugRuntimeFixture.execution_plan(),
        artifact_inputs={},
        artifact_outputs={},
        runtime_plane_index=0,
        source_binding_context=SourceBindingRuntimeContext.empty(),
    )

    result = execute_function_chain(request)

    np.testing.assert_array_equal(result, np.array([2, 3, 4]))
    assert len(emitted) == 2
    assert (
        DebugProgressContext.from_progress_context(emitted[0].context)
        .cursor.invocation_key
        .endswith(":1:plus_one")
    )


def test_execute_chain_emits_debug_exception_event():
    sink = RecordingDebugEventSink()
    context = SimpleNamespace(debug_event_sink=sink)
    request = FunctionChainExecutionRequest(
        initial_data_stack=np.array([1]),
        invocations=(
            DebugRuntimeFixture.compiled_invocation(DebugRuntimeFixture.raising),
        ),
        context=context,
        execution_plan=DebugRuntimeFixture.execution_plan(),
        artifact_inputs={},
        artifact_outputs={},
        runtime_plane_index=0,
        source_binding_context=SourceBindingRuntimeContext.empty(),
    )

    try:
        execute_function_chain(request)
    except RuntimeError:
        pass

    assert [event.event_type for event in sink.events] == [
        DebugEventType.BEFORE_INVOCATION,
        DebugEventType.EXCEPTION,
    ]
    assert sink.events[1].exception == "RuntimeError: debug boom"
    assert "test_debug_runtime.py" in (sink.events[1].traceback_text or "")


def test_debug_sink_from_context_defaults_to_noop():
    sink = debug_event_sink_from_context(SimpleNamespace())

    assert isinstance(sink, NoOpDebugEventSink)


def test_local_debug_snapshot_store_round_trips_metadata(tmp_path):
    cursor = DebugCursor(
        step_index=1,
        step_scope_id=DebugRuntimeFixture.STEP_SCOPE_ID,
        group_key=DebugRuntimeFixture.GROUP_KEY,
        invocation_key=f"{DebugRuntimeFixture.GROUP_KEY}:0:measure",
        pattern_group_identity="0",
    )
    snapshot = DebugSnapshot(
        snapshot_id=DebugRuntimeFixture.SNAPSHOT_ID,
        cursor=cursor,
        step_name="measure",
        callable_name="measure_intensity",
        axis_id="B02",
        source_paths=("B02_s1_w1.tif",),
        output_artifact_refs=(
            DebugArtifactRef(
                kind=ArtifactKind.MEASUREMENTS,
                name="Cells.csv",
                cursor=cursor,
                storage_ref="debug/snap-1/Cells.csv",
                dtype="csv",
            ),
        ),
        timing_seconds=0.25,
    )
    session = DebugSession.create(
        execution_id="exec-1",
        plate_id=DebugRuntimeFixture.PLATE_ID,
        axis_id="B02",
    )
    store = LocalDebugSnapshotStore.for_session(root_path=tmp_path, session=session)

    written_path = store.write_snapshot(snapshot)
    restored = store.read_snapshot("snap-1")

    assert written_path.name == "snap-1.json"
    assert restored == snapshot
    assert store.list_snapshot_ids() == ("snap-1",)
    assert store.manifest().to_json_dict() == {
        "debug_session_id": session.debug_session_id,
        "snapshot_ids": ["snap-1"],
    }


def test_filemanager_debug_snapshot_store_round_trips_metadata():
    cursor = DebugRuntimeFixture.cursor()
    snapshot = DebugSnapshot(
        snapshot_id="snap-vfs",
        cursor=cursor,
        step_name="measure",
        callable_name="measure_intensity",
        axis_id=DebugRuntimeFixture.AXIS_ID,
    )
    filemanager = DebugSnapshotFileManagerStub()
    store = FileManagerDebugSnapshotStore(
        filemanager=filemanager,
        backend=DebugRuntimeFixture.DEBUG_SNAPSHOT_BACKEND,
        root_path="/debug",
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
    )

    written_path = store.write_snapshot(snapshot)
    restored = store.read_snapshot(snapshot.snapshot_id)

    assert written_path == "/debug/debug-1/snap-vfs.json"
    assert restored == snapshot
    assert store.list_snapshot_ids() == ("snap-vfs",)
    assert filemanager.saved["/debug/debug-1/manifest.json"] == {
        "debug_session_id": DebugRuntimeFixture.DEBUG_SESSION_ID,
        "snapshot_ids": ["snap-vfs"],
    }


def test_debug_progress_context_round_trips_through_progress_event():
    event_context = DebugProgressContext(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        snapshot_id="snap-2",
        cursor=DebugRuntimeFixture.cursor(),
        event_type=DebugEventType.AFTER_INVOCATION,
        snapshot_store_ref=DebugRuntimeFixture.SNAPSHOT_STORE_REF,
        snapshot_store_backend=DebugRuntimeFixture.DEBUG_SNAPSHOT_BACKEND,
    )
    progress = ProgressEvent(
        execution_id="exec",
        plate_id=DebugRuntimeFixture.PLATE_ID,
        axis_id=DebugRuntimeFixture.AXIS_ID,
        step_name=DebugRuntimeFixture.SEGMENT_NAME,
        phase=ProgressPhase.PATTERN_GROUP,
        status=ProgressStatus.RUNNING,
        percent=50,
        completed=1,
        total=2,
        timestamp=1.0,
        pid=123,
        context=event_context.to_progress_context(),
    )

    restored = ProgressEvent.from_dict(progress.to_dict())
    restored_context = DebugProgressContext.from_progress_context(restored.context or {})

    assert restored_context == event_context


def test_debug_progress_event_request_builds_lightweight_progress_event():
    debug_event = sink_event = DebugRuntimeFixture.debug_event(
        event_type=DebugEventType.AFTER_INVOCATION,
    )
    request = DebugProgressEventRequest(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        debug_event=sink_event,
        execution_id=DebugRuntimeFixture.EXECUTION_ID,
        plate_id=DebugRuntimeFixture.PLATE_ID,
        snapshot_id="snap-3",
        snapshot_store_ref=DebugRuntimeFixture.SNAPSHOT_STORE_REF,
        snapshot_store_backend=DebugRuntimeFixture.DEBUG_SNAPSHOT_BACKEND,
        completed=1,
        total=1,
        percent=100,
        worker_slot="worker_0",
        owned_wells=("A01",),
        timestamp=2.0,
        pid=456,
    )

    progress = request.to_progress_event()
    restored_context = DebugProgressContext.from_progress_context(progress.context or {})

    assert debug_event.event_type is DebugEventType.AFTER_INVOCATION
    assert progress.phase is ProgressPhase.PATTERN_GROUP
    assert progress.status is ProgressStatus.SUCCESS
    assert progress.message == "after_invocation"
    assert progress.error is None
    assert progress.worker_slot == "worker_0"
    assert progress.owned_wells == ["A01"]
    assert restored_context.snapshot_id == "snap-3"
    assert restored_context.snapshot_store_ref == DebugRuntimeFixture.SNAPSHOT_STORE_REF
    assert (
        restored_context.snapshot_store_backend
        == DebugRuntimeFixture.DEBUG_SNAPSHOT_BACKEND
    )


def test_progress_debug_event_sink_uses_injected_progress_emitter():
    emitted: list[ProgressEvent] = []
    sink = ProgressDebugEventSink(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        execution_id=DebugRuntimeFixture.EXECUTION_ID,
        plate_id=DebugRuntimeFixture.PLATE_ID,
        emit_progress=emitted.append,
        snapshot_store_ref=DebugRuntimeFixture.SNAPSHOT_STORE_REF,
        snapshot_store_backend=DebugRuntimeFixture.DEBUG_SNAPSHOT_BACKEND,
        worker_slot="worker_0",
        owned_wells=("A01",),
    )

    sink.record(DebugRuntimeFixture.debug_event(event_type=DebugEventType.BEFORE_INVOCATION))

    assert len(emitted) == 1
    assert emitted[0].status is ProgressStatus.STARTED
    assert emitted[0].context is not None
    assert (
        emitted[0].context["debug_session_id"]
        == DebugRuntimeFixture.DEBUG_SESSION_ID
    )
    assert (
        emitted[0].context["snapshot_store_ref"]
        == DebugRuntimeFixture.SNAPSHOT_STORE_REF
    )
    assert (
        emitted[0].context["snapshot_store_backend"]
        == DebugRuntimeFixture.DEBUG_SNAPSHOT_BACKEND
    )
    assert emitted[0].worker_slot == "worker_0"
    assert emitted[0].owned_wells == ["A01"]


def test_debug_event_builds_metadata_snapshot():
    event = DebugRuntimeFixture.debug_event(event_type=DebugEventType.AFTER_INVOCATION)

    snapshot = event.to_snapshot(snapshot_id="snap-event")

    assert snapshot.snapshot_id == "snap-event"
    assert snapshot.cursor == event.cursor
    assert snapshot.step_name == event.step_name
    assert snapshot.callable_name == event.callable_name
    assert snapshot.axis_id == event.axis_id


def test_local_snapshot_progress_sink_writes_snapshot_and_emits_id(tmp_path):
    emitted: list[ProgressEvent] = []
    store = LocalDebugSnapshotStore(
        root_path=tmp_path,
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
    )
    sink = LocalSnapshotProgressDebugEventSink(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        execution_id=DebugRuntimeFixture.EXECUTION_ID,
        plate_id=DebugRuntimeFixture.PLATE_ID,
        emit_progress=emitted.append,
        snapshot_store=store,
        snapshot_store_ref=str(tmp_path),
        worker_slot="worker_0",
        owned_wells=("A01",),
    )

    sink.record(DebugRuntimeFixture.debug_event(event_type=DebugEventType.AFTER_INVOCATION))

    restored_context = DebugProgressContext.from_progress_context(
        emitted[0].context or {}
    )
    assert restored_context.snapshot_id is not None
    assert store.list_snapshot_ids() == (restored_context.snapshot_id,)
    assert store.read_snapshot(restored_context.snapshot_id).step_name == (
        DebugRuntimeFixture.SEGMENT_NAME
    )
    assert emitted[0].status is ProgressStatus.SUCCESS
    assert emitted[0].worker_slot == "worker_0"
    assert emitted[0].owned_wells == ["A01"]


def test_debug_execution_config_round_trips_through_config_params():
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        snapshot_store_ref=DebugRuntimeFixture.SNAPSHOT_STORE_REF,
        snapshot_store_backend=DebugRuntimeFixture.DEBUG_SNAPSHOT_BACKEND,
        command_type=DebugCommandType.STEP,
        selected_source_group="A01",
        pause_step_indices=(2, 5),
        start_step_index=2,
    )

    policy = DebugExecutionPolicy.from_config_params(config.to_config_params())
    disabled_policy = DebugExecutionPolicy.from_config_params({})

    assert isinstance(policy, ProgressDebugExecutionPolicy)
    assert policy.config == config
    assert disabled_policy.policy_kind == "noop"


def test_progress_debug_execution_policy_uses_filemanager_snapshot_store():
    filemanager = DebugSnapshotFileManagerStub()
    context = DebugSnapshotContextStub(filemanager=filemanager)
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        snapshot_store_ref="/debug",
        snapshot_store_backend=DebugRuntimeFixture.DEBUG_SNAPSHOT_BACKEND,
    )
    policy = DebugExecutionPolicy.from_config_params(config.to_config_params())

    policy.install_context_sink(
        DebugRuntimeFixture.sink_install_request(context=context)
    )

    assert isinstance(
        context.debug_event_sink.snapshot_store,
        FileManagerDebugSnapshotStore,
    )
    assert context.debug_event_sink.snapshot_store.filemanager is filemanager
    assert (
        context.debug_event_sink.snapshot_store_backend
        == DebugRuntimeFixture.DEBUG_SNAPSHOT_BACKEND
    )


def test_debug_step_policy_stops_after_first_step_only():
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        command_type=DebugCommandType.STEP,
    )
    policy = DebugExecutionPolicy.from_config_params(config.to_config_params())

    assert policy.step_stop_strategy().should_stop_after_step(
        step_index=0,
        step_name="first",
    )
    assert not policy.step_stop_strategy().should_stop_after_step(
        step_index=1,
        step_name="second",
    )
    noop_strategy = DebugExecutionPolicy.from_config_params({}).step_stop_strategy()
    assert not noop_strategy.should_stop_after_step(
        step_index=0,
        step_name="first",
    )


def test_debug_run_to_pause_policy_stops_at_declared_pause_step():
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        command_type=DebugCommandType.RUN_TO_PAUSE,
        pause_step_indices=(2,),
    )
    policy = DebugExecutionPolicy.from_config_params(config.to_config_params())

    strategy = policy.step_stop_strategy()

    assert not strategy.should_stop_after_step(step_index=1, step_name="before")
    assert strategy.should_stop_after_step(step_index=2, step_name="pause")
    assert not strategy.should_stop_after_step(step_index=3, step_name="after")


def test_debug_execution_policy_defaults_to_one_available_axis():
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        command_type=DebugCommandType.RUN,
    )
    policy = DebugExecutionPolicy.from_config_params(config.to_config_params())

    assert policy.axis_filter_for_available(("A01", "B01")) == ["A01"]
    assert policy.axis_filter_for_available(()) == []
    assert DebugExecutionPolicy.from_config_params({}).axis_filter_for_available(
        ("A01", "B01")
    ) == ["A01", "B01"]


def test_debug_execution_policy_uses_selected_source_group_as_axis_filter():
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        command_type=DebugCommandType.CHOOSE_SOURCE_GROUP,
        selected_source_group="B01",
    )
    policy = DebugExecutionPolicy.from_config_params(config.to_config_params())

    assert policy.axis_filter_for_available(("A01", "B01")) == ["B01"]


def test_debug_execution_policy_skips_before_start_step_index():
    config = DebugExecutionConfig(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        start_step_index=2,
        replay_mode=DebugReplayMode.WARM_ARTIFACT,
    )
    policy = DebugExecutionPolicy.from_config_params(config.to_config_params())

    assert not policy.should_execute_step(1)
    assert policy.should_execute_step(2)
    assert policy.should_execute_step(3)
    assert policy.should_reuse_step_outputs(1)
    assert not policy.should_reuse_step_outputs(2)


def test_debug_session_marks_current_cursor_dirty_nominally():
    cursor = DebugRuntimeFixture.cursor()
    session = DebugSession.create(plate_id=DebugRuntimeFixture.PLATE_ID).with_cursor(
        cursor
    )

    dirty_session = session.mark_dirty_from_cursor()

    assert dirty_session.cursor == cursor
    assert dirty_session.dirty_from_cursor == cursor
    assert session.dirty_from_cursor is None


def test_debug_snapshot_read_request_round_trips_snapshot_response():
    snapshot = DebugRuntimeFixture.debug_event(
        event_type=DebugEventType.AFTER_INVOCATION
    ).to_snapshot(snapshot_id=DebugRuntimeFixture.SNAPSHOT_ID)
    request = DebugSnapshotReadRequest(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        snapshot_id=snapshot.snapshot_id,
        snapshot_store_ref=DebugRuntimeFixture.SNAPSHOT_STORE_REF,
    )
    response = DebugSnapshotReadResponse(snapshot=snapshot)

    restored_request = DebugSnapshotReadControlPayload.from_request(
        request
    ).to_request()
    restored_response = DebugSnapshotReadResponse.from_control_response(
        response.to_control_response()
    )

    assert restored_request == request
    assert restored_response.snapshot == snapshot


def test_debug_snapshot_round_trips_invocation_parameters_and_artifact_identity():
    cursor = DebugRuntimeFixture.cursor()
    artifact = ArtifactOutputPlan(
        name="objects",
        path="/tmp/objects.json",
        kind=ArtifactKind.OBJECT_LABELS,
        group_keys=("A01",),
    )
    event = DebugEvent(
        event_type=DebugEventType.AFTER_INVOCATION,
        cursor=cursor,
        step_name="segment",
        callable_name="segment",
        output_artifact_refs=(
            DebugArtifactRef.from_artifact_plan(plan=artifact, cursor=cursor),
        ),
        invocation_parameters=(
            DebugInvocationParameter.from_kwargs({"threshold": 0.5})[0],
        ),
    )

    restored = DebugSnapshot.from_json_dict(
        event.to_snapshot(snapshot_id="snapshot-1").to_json_dict()
    )

    assert restored.invocation_parameters[0].name == "threshold"
    assert restored.output_artifact_refs[0].identity == DebugArtifactIdentity.from_artifact_plan(
        artifact
    )


def test_debug_boundary_state_registry_covers_event_and_snapshot():
    assert DebugBoundaryState.__registry__["event"] is DebugEvent
    assert DebugBoundaryState.__registry__["snapshot"] is DebugSnapshot


def test_debug_session_request_registry_covers_control_request_family():
    assert DebugSessionRequest.__registry__["snapshot_read"] is DebugSnapshotReadRequest
    assert DebugSessionRequest.__registry__["artifact_export"] is DebugArtifactExportRequest
    assert DebugSessionRequest.__registry__["worker_command"] is DebugWorkerCommandRequest
    assert DebugSessionRequest.__registry__["progress_event"] is DebugProgressEventRequest


def test_warm_replay_rejects_missing_artifact_outputs():
    cursor = DebugRuntimeFixture.cursor()
    missing_artifact = ArtifactOutputPlan(
        name="measurements",
        path="/tmp/openhcs-missing-debug-artifact.csv",
        kind=ArtifactKind.MEASUREMENTS,
    )
    plan = DebugWarmReplayArtifactReusePlan.from_artifact_plans(
        artifact_plans={"measurements": missing_artifact},
        cursor=cursor,
    )

    try:
        plan.require_available(DebugSnapshotContextStub(DebugSnapshotFileManagerStub()))
    except RuntimeError as error:
        assert "expected artifact outputs are unavailable" in str(error)
    else:
        raise AssertionError("Missing warm replay artifacts must fail validation.")


def test_warm_replay_hydrates_local_artifact_from_snapshot_identity(tmp_path):
    cursor = DebugRuntimeFixture.cursor()
    source_path = tmp_path / "source.csv"
    destination_path = tmp_path / "hydrated.csv"
    source_path.write_text("x,y\n1,2\n", encoding="utf-8")
    source_plan = ArtifactOutputPlan(
        name="measurements",
        path=str(source_path),
        kind=ArtifactKind.MEASUREMENTS,
    )
    destination_plan = ArtifactOutputPlan(
        name="measurements",
        path=str(destination_path),
        kind=ArtifactKind.MEASUREMENTS,
    )
    store = LocalDebugSnapshotStore(
        root_path=tmp_path / "snapshots",
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
    )
    store.write_snapshot(
        DebugSnapshot(
            snapshot_id="snapshot-1",
            cursor=cursor,
            step_name="measure",
            output_artifact_refs=(
                DebugArtifactRef.from_artifact_plan(plan=source_plan, cursor=cursor),
            ),
        )
    )

    DebugWarmReplayArtifactReusePlan.from_artifact_plans(
        artifact_plans={"measurements": destination_plan},
        cursor=cursor,
        snapshot_store=store,
    ).require_available(DebugSnapshotContextStub(DebugSnapshotFileManagerStub()))

    assert destination_path.read_text(encoding="utf-8") == "x,y\n1,2\n"


def test_warm_replay_hydrates_vfs_artifact_from_snapshot_identity():
    cursor = DebugRuntimeFixture.cursor()
    filemanager = DebugSnapshotFileManagerStub()
    filemanager.save({"value": 1}, "/debug/source/measurements.json", "memory")
    source_plan = ArtifactOutputPlan(
        name="measurements",
        path="/debug/source/measurements.json",
        kind=ArtifactKind.MEASUREMENTS,
    )
    destination_plan = ArtifactOutputPlan(
        name="measurements",
        path="/debug/replay/measurements.json",
        kind=ArtifactKind.MEASUREMENTS,
    )
    store = FileManagerDebugSnapshotStore(
        filemanager=filemanager,
        backend="memory",
        root_path="/snapshots",
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
    )
    store.write_snapshot(
        DebugSnapshot(
            snapshot_id="snapshot-1",
            cursor=cursor,
            step_name="measure",
            output_artifact_refs=(
                DebugArtifactRef.from_artifact_plan(plan=source_plan, cursor=cursor),
            ),
        )
    )

    DebugWarmReplayArtifactReusePlan.from_artifact_plans(
        artifact_plans={"measurements": destination_plan},
        cursor=cursor,
        snapshot_store=store,
    ).require_available(DebugSnapshotContextStub(filemanager))

    assert filemanager.load("/debug/replay/measurements.json", "memory") == {"value": 1}


def test_warm_replay_rejects_snapshot_artifact_with_stale_vfs_content():
    cursor = DebugRuntimeFixture.cursor()
    filemanager = DebugSnapshotFileManagerStub()
    filemanager.save({"value": 1}, "/debug/source/measurements.json", "memory")
    source_plan = ArtifactOutputPlan(
        name="measurements",
        path="/debug/source/measurements.json",
        kind=ArtifactKind.MEASUREMENTS,
    )
    destination_plan = ArtifactOutputPlan(
        name="measurements",
        path="/debug/replay/measurements.json",
        kind=ArtifactKind.MEASUREMENTS,
    )
    store = FileManagerDebugSnapshotStore(
        filemanager=filemanager,
        backend="memory",
        root_path="/snapshots",
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
    )
    store.write_snapshot(
        DebugSnapshot(
            snapshot_id="snapshot-1",
            cursor=cursor,
            step_name="measure",
            output_artifact_refs=(
                DebugArtifactRef.from_artifact_plan(plan=source_plan, cursor=cursor),
            ),
        )
    )
    filemanager.save({"value": 2}, "/debug/source/measurements.json", "memory")

    try:
        DebugWarmReplayArtifactReusePlan.from_artifact_plans(
            artifact_plans={"measurements": destination_plan},
            cursor=cursor,
            snapshot_store=store,
        ).require_available(DebugSnapshotContextStub(filemanager))
    except RuntimeError as error:
        assert "expected artifact outputs are unavailable" in str(error)
    else:
        raise AssertionError("Stale snapshot artifact content must fail validation.")


def test_warm_replay_rejects_snapshot_artifact_with_mismatched_settings_identity():
    cursor = DebugRuntimeFixture.cursor()
    filemanager = DebugSnapshotFileManagerStub()
    filemanager.save({"value": 1}, "/debug/source/measurements.json", "memory")
    source_plan = ArtifactOutputPlan(
        name="measurements",
        path="/debug/source/measurements.json",
        kind=ArtifactKind.MEASUREMENTS,
        materialization="old-settings",
    )
    destination_plan = ArtifactOutputPlan(
        name="measurements",
        path="/debug/replay/measurements.json",
        kind=ArtifactKind.MEASUREMENTS,
        materialization="new-settings",
    )
    store = FileManagerDebugSnapshotStore(
        filemanager=filemanager,
        backend="memory",
        root_path="/snapshots",
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
    )
    store.write_snapshot(
        DebugSnapshot(
            snapshot_id="snapshot-1",
            cursor=cursor,
            step_name="measure",
            output_artifact_refs=(
                DebugArtifactRef.from_artifact_plan(plan=source_plan, cursor=cursor),
            ),
        )
    )

    try:
        DebugWarmReplayArtifactReusePlan.from_artifact_plans(
            artifact_plans={"measurements": destination_plan},
            cursor=cursor,
            snapshot_store=store,
        ).require_available(DebugSnapshotContextStub(filemanager))
    except RuntimeError as error:
        assert "expected artifact outputs are unavailable" in str(error)
    else:
        raise AssertionError("Mismatched artifact settings identity must fail validation.")


def test_debug_artifact_export_plan_materializes_vfs_payload(tmp_path):
    cursor = DebugRuntimeFixture.cursor()
    filemanager = DebugSnapshotFileManagerStub()
    filemanager.save("payload", "/debug/artifact.txt", "memory")
    artifact_ref = DebugArtifactRef(
        kind=ArtifactKind.MEASUREMENTS,
        name="artifact",
        cursor=cursor,
        storage_ref="/debug/artifact.txt",
        storage_backend="memory",
    )

    exported = DebugArtifactExportPlan(
        artifact_ref=artifact_ref,
        export_root=tmp_path,
        filemanager=filemanager,
    ).export()

    assert exported.read_text(encoding="utf-8") == "payload"


def test_zmq_server_reads_local_debug_snapshot_by_control_request(tmp_path):
    from openhcs.runtime.zmq_debug_control import DebugControlMessageRouter

    snapshot = DebugRuntimeFixture.debug_event(
        event_type=DebugEventType.AFTER_INVOCATION
    ).to_snapshot(snapshot_id=DebugRuntimeFixture.SNAPSHOT_ID)
    store = LocalDebugSnapshotStore(
        root_path=tmp_path,
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
    )
    store.write_snapshot(snapshot)
    request = DebugSnapshotReadRequest(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        snapshot_id=snapshot.snapshot_id,
        snapshot_store_ref=str(tmp_path),
    )

    message = DebugSnapshotReadControlPayload.from_request(request).to_dict()
    response = DebugControlMessageRouter.handle(message)

    assert DebugSnapshotReadResponse.from_control_response(response).snapshot == snapshot


def test_zmq_server_exports_debug_artifact_by_control_request(tmp_path):
    from openhcs.runtime.zmq_debug_control import DebugControlMessageRouter

    source_path = tmp_path / "source.csv"
    source_path.write_text("value\n1\n", encoding="utf-8")
    cursor = DebugRuntimeFixture.cursor()
    artifact_ref = DebugArtifactRef(
        kind=ArtifactKind.MEASUREMENTS,
        name="measurements",
        cursor=cursor,
        storage_ref=str(source_path),
    )
    request = DebugArtifactExportRequest(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        artifact_ref=artifact_ref,
        export_root=str(tmp_path / "exports"),
    )

    message = DebugArtifactExportControlPayload.from_request(request).to_dict()
    response = DebugControlMessageRouter.handle(message)

    exported = Path(DebugArtifactExportResponse.from_control_response(response).exported_ref)
    assert exported.read_text(encoding="utf-8") == "value\n1\n"


def test_live_zmq_debug_worker_command_loop_round_trips_paused_worker_status():
    from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
    from openhcs.runtime.zmq_execution_server import ZMQExecutionServer

    port = _free_tcp_port()
    server = ZMQExecutionServer(port=port, host="127.0.0.1")

    def pump_server() -> None:
        while server.is_running():
            server.process_messages()
            time.sleep(0.01)

    server.start()
    thread = threading.Thread(target=pump_server, daemon=True)
    thread.start()
    client = ZMQExecutionClient(port=port, host="127.0.0.1", persistent=True)
    try:
        response = client.send_debug_worker_command(
            debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
            command_type=DebugCommandType.STEP,
        )
    finally:
        client.disconnect()
        server.stop()
        thread.join(timeout=2.0)
        DebugPausedWorkerRegistry.remove(DebugRuntimeFixture.DEBUG_SESSION_ID)

    assert response.status.debug_session_id == DebugRuntimeFixture.DEBUG_SESSION_ID


def test_paused_worker_controller_blocks_and_steps_after_snapshot_boundary():
    controller = DebugPausedWorkerRegistry.controller_for(
        DebugRuntimeFixture.DEBUG_SESSION_ID
    )
    event = DebugRuntimeFixture.debug_event(
        event_type=DebugEventType.AFTER_INVOCATION
    )
    completed: list[str] = []

    def wait_at_boundary() -> None:
        controller.wait_at_boundary(event)
        completed.append("resumed")

    thread = threading.Thread(target=wait_at_boundary)
    thread.start()
    deadline = time.time() + 2.0
    while (
        controller.status.state.value != "paused"
        and time.time() < deadline
    ):
        time.sleep(0.01)

    assert controller.status.state.value == "paused"
    controller.apply_command(DebugCommandType.STEP)
    thread.join(timeout=2.0)

    assert completed == ["resumed"]
    DebugPausedWorkerRegistry.remove(DebugRuntimeFixture.DEBUG_SESSION_ID)


def test_zmq_server_routes_paused_worker_command_by_control_request():
    from openhcs.runtime.zmq_debug_control import DebugControlMessageRouter

    request = DebugWorkerCommandRequest(
        debug_session_id=DebugRuntimeFixture.DEBUG_SESSION_ID,
        command_type=DebugCommandType.STEP,
    )

    message = DebugWorkerCommandControlPayload.from_request(request).to_dict()
    response = DebugControlMessageRouter.handle(message)

    assert DebugWorkerCommandResponse.from_control_response(response).status.debug_session_id == (
        DebugRuntimeFixture.DEBUG_SESSION_ID
    )
    DebugPausedWorkerRegistry.remove(DebugRuntimeFixture.DEBUG_SESSION_ID)
