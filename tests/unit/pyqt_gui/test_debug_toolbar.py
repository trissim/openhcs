from __future__ import annotations

import asyncio

from metaclass_registry import AutoRegisterMeta
from PyQt6.QtWidgets import QApplication

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.debug import (
    DebugArtifactRef,
    DebugCommand,
    DebugCommandType,
    DebugCursor,
    DebugEventType,
    DebugProgressContext,
    DebugReplayMode,
    DebugSession,
    FileManagerDebugSnapshotStore,
)
from openhcs.pyqt_gui.windows.debug_inspector_window import (
    DebugArtifactMaterializeRequest,
)
from openhcs.core.progress import ProgressEvent, ProgressPhase, ProgressStatus
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.widgets.shared.services.batch_workflow_service import (
    DebugSnapshotAvailableNotification,
)
from openhcs.pyqt_gui.widgets.debug_toolbar import DebugToolbarWidget
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget


class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


def test_debug_toolbar_emits_typed_command() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()
    commands: list[DebugCommand] = []
    toolbar.command_requested.connect(commands.append)

    toolbar.buttons[DebugCommandType.STEP].click()

    assert commands == [DebugCommand(DebugCommandType.STEP)]


def test_debug_toolbar_enables_controls_together() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()

    toolbar.set_controls_enabled(False)

    assert all(not button.isEnabled() for button in toolbar.buttons.values())


class StatusSignalRecorder:
    """Signal-like recorder for command routing tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, message: str) -> None:
        self.messages.append(message)


class PlateManagerStopRecorder:
    """Plate-manager seam used by the pipeline editor stop command."""

    def __init__(self) -> None:
        self.stop_calls = 0

    def action_stop_execution(self) -> None:
        self.stop_calls += 1


class PlateManagerRunRecorder:
    """Plate-manager seam used by debug run dispatch tests."""

    def __init__(self) -> None:
        self.run_calls = []

    def action_run_debug_plate(self, plate_path, **kwargs):
        self.run_calls.append((plate_path, kwargs))
        return ("debug-run", plate_path, kwargs)


class DebugBatchWorkflowRecorder:
    """Batch workflow seam for paused-worker GUI command routing tests."""

    def __init__(self) -> None:
        self.run_calls = []
        self.worker_commands = []

    async def run_debug_plate(self, **kwargs) -> None:
        self.run_calls.append(kwargs)

    async def send_debug_worker_command(
        self,
        *,
        debug_session_id: str,
        command_type: DebugCommandType,
    ) -> None:
        self.worker_commands.append((debug_session_id, command_type))


class PlateManagerDebugHarness:
    """Minimal plate-manager state used by paused-worker UX tests."""

    def __init__(self) -> None:
        self._active_debug_sessions = {}
        self._batch_workflow_service = DebugBatchWorkflowRecorder()
        self.execution_error = StatusSignalRecorder()

    def get_selected_items(self) -> list[dict[str, str]]:
        return []


class PipelineEditorHarnessBase(metaclass=AutoRegisterMeta):
    """Shared pipeline-editor harness fields for debug command tests."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    def __init__(self, plate_manager) -> None:
        self.plate_manager = plate_manager
        self.status_message = StatusSignalRecorder()


class PipelineEditorCommandHarness(PipelineEditorHarnessBase):
    """Minimal object carrying the attributes used by debug command dispatch."""

    registry_key = "command"
    DEBUG_COMMAND_ROUTES = PipelineEditorWidget.DEBUG_COMMAND_ROUTES

    def __init__(self, plate_manager: PlateManagerStopRecorder | None) -> None:
        super().__init__(plate_manager)
        self.debug_run_commands: list[DebugCommandType] = []

    def _dispatch_debug_run_command(
        self,
        command_type: DebugCommandType = DebugCommandType.RUN,
    ) -> None:
        self.debug_run_commands.append(command_type)

    def _dispatch_debug_stop_command(self) -> None:
        PipelineEditorWidget._dispatch_debug_stop_command(self)


class PipelineEditorRunHarness(PipelineEditorHarnessBase):
    """Minimal object carrying attributes used by debug run dispatch."""

    registry_key = "run"

    def __init__(self) -> None:
        super().__init__(PlateManagerRunRecorder())
        self.current_plate = "plate"
        self.debug_session_state = None
        self.pipeline_steps = [
            FunctionStep(func=lambda image: image, name="first"),
            FunctionStep(func=lambda image: image, name="pause", debug_pause=True),
        ]

    def _debug_pause_step_indices(self) -> tuple[int, ...]:
        return PipelineEditorWidget._debug_pause_step_indices(self)

    def _debug_start_step_index(self, command_type: DebugCommandType) -> int:
        return PipelineEditorWidget._debug_start_step_index(self, command_type)

    def _debug_start_after_invocation_key(
        self,
        command_type: DebugCommandType,
    ) -> str | None:
        return PipelineEditorWidget._debug_start_after_invocation_key(
            self,
            command_type,
        )


class PipelineEditorDirtyHarness(PipelineEditorHarnessBase):
    """Minimal object carrying state used by debug dirty invalidation."""

    registry_key = "dirty"

    def __init__(self) -> None:
        super().__init__(PlateManagerRunRecorder())
        cursor = DebugCursor(
            step_index=1,
            step_scope_id="step-1",
            group_key="default",
            invocation_key="default:0:segment",
        )
        self.current_plate = "plate"
        self.saved = []
        self._suppress_pipeline_state_sync = False
        self.debug_session_state = DebugSession.create(plate_id="plate").with_cursor(
            cursor
        )

    def save_pipeline_for_plate(self, plate, steps) -> None:
        self.saved.append((plate, steps))


class DebugInspectorRecorder:
    """Recorder replacing the heavy Qt inspector in route tests."""

    def __init__(self, parent=None) -> None:
        del parent
        self.snapshots = []
        self.local_loads = []
        self.store_loads = []
        self.show_calls = 0
        self.raise_calls = 0
        self.artifact_export_requested = SignalConnectRecorder()
        self.artifact_open_requested = SignalConnectRecorder()

    def set_snapshot(self, snapshot) -> None:
        self.snapshots.append(snapshot)

    def load_snapshot(self, **kwargs) -> None:
        self.local_loads.append(kwargs)

    def load_snapshot_from_store(self, *, store, snapshot_id) -> None:
        self.store_loads.append((store, snapshot_id))

    def show(self) -> None:
        self.show_calls += 1

    def raise_(self) -> None:
        self.raise_calls += 1


class FileManagerRecorder:
    """FileManager identity used by VFS debug snapshot route tests."""


class SignalConnectRecorder:
    """Signal-like object that records connected callables."""

    def __init__(self) -> None:
        self.connected = []

    def connect(self, callback) -> None:
        self.connected.append(callback)


class PipelineEditorSnapshotHarness:
    """Minimal object carrying the attributes used by snapshot display."""

    def __init__(self) -> None:
        self.status_message = StatusSignalRecorder()
        self.debug_inspector_window = None
        self.filemanager = FileManagerRecorder()
        self.service_adapter = self
        self.plate_manager = None

    def get_file_manager(self) -> FileManagerRecorder:
        return self.filemanager

    def _handle_debug_artifact_export_request(self, request) -> None:
        PipelineEditorWidget._handle_debug_artifact_export_request(self, request)

    def _handle_debug_artifact_open_request(self, request) -> None:
        PipelineEditorWidget._handle_debug_artifact_open_request(self, request)


def debug_snapshot_notification(
    *,
    snapshot_store_backend: str | None,
) -> DebugSnapshotAvailableNotification:
    cursor = DebugCursor(
        step_index=1,
        step_scope_id="plate::step-1",
        group_key="default",
        invocation_key="default:0:segment",
    )
    debug_context = DebugProgressContext(
        debug_session_id="debug-1",
        snapshot_id="snap-1",
        cursor=cursor,
        event_type=DebugEventType.AFTER_INVOCATION,
        snapshot_store_ref="/debug",
        snapshot_store_backend=snapshot_store_backend,
    )
    return DebugSnapshotAvailableNotification(
        progress_event=ProgressEvent(
            execution_id="exec-1",
            plate_id="plate",
            axis_id="A01",
            step_name="step",
            phase=ProgressPhase.PATTERN_GROUP,
            status=ProgressStatus.SUCCESS,
            percent=100,
            completed=1,
            total=1,
            timestamp=1.0,
            pid=123,
            context=debug_context.to_progress_context(),
        ),
        debug_context=debug_context,
    )


def test_pipeline_editor_routes_stop_debug_command_to_plate_manager() -> None:
    plate_manager = PlateManagerStopRecorder()
    harness = PipelineEditorCommandHarness(plate_manager)

    PipelineEditorWidget._handle_debug_command(
        harness,
        DebugCommand(DebugCommandType.STOP),
    )

    assert plate_manager.stop_calls == 1
    assert harness.status_message.messages == ["Requested debug execution stop."]


def test_plate_manager_reuses_persistent_paused_worker_across_commands(tmp_path) -> None:
    plate_path = str(tmp_path / "plate")
    harness = PlateManagerDebugHarness()

    asyncio.run(
        PlateManagerWidget.action_run_debug_plate(
            harness,
            plate_path,
            command_type=DebugCommandType.RUN_TO_PAUSE,
            pause_step_indices=(1,),
        )
    )
    session = harness._active_debug_sessions[plate_path]
    asyncio.run(
        PlateManagerWidget.action_run_debug_plate(
            harness,
            plate_path,
            command_type=DebugCommandType.STEP,
        )
    )
    asyncio.run(
        PlateManagerWidget.action_run_debug_plate(
            harness,
            plate_path,
            command_type=DebugCommandType.RUN,
        )
    )
    asyncio.run(
        PlateManagerWidget.action_run_debug_plate(
            harness,
            plate_path,
            command_type=DebugCommandType.STOP,
        )
    )

    assert harness._batch_workflow_service.run_calls == [
        {
            "plate_path": plate_path,
            "debug_session_id": session.debug_session_id,
            "snapshot_store_ref": str(tmp_path / ".openhcs_debug"),
            "snapshot_store_backend": None,
            "command_type": DebugCommandType.RUN_TO_PAUSE,
            "selected_source_group": None,
            "pause_step_indices": (1,),
            "start_step_index": 0,
            "start_after_invocation_key": None,
            "replay_mode": DebugReplayMode.PERSISTENT_PAUSED_WORKER,
        }
    ]
    assert harness._batch_workflow_service.worker_commands == [
        (session.debug_session_id, DebugCommandType.STEP),
        (session.debug_session_id, DebugCommandType.RUN),
        (session.debug_session_id, DebugCommandType.STOP),
    ]
    assert plate_path not in harness._active_debug_sessions


def test_pipeline_editor_routes_step_debug_command_to_bounded_run() -> None:
    harness = PipelineEditorCommandHarness(PlateManagerStopRecorder())

    PipelineEditorWidget._handle_debug_command(
        harness,
        DebugCommand(DebugCommandType.STEP),
    )

    assert harness.debug_run_commands == [DebugCommandType.STEP]
    assert harness.plate_manager.stop_calls == 0


def test_pipeline_editor_has_route_for_every_debug_command() -> None:
    assert set(PipelineEditorWidget.DEBUG_COMMAND_ROUTES) == set(DebugCommandType)


def test_pipeline_editor_dispatches_pause_step_indices(monkeypatch) -> None:
    created_tasks = []

    def record_task(task):
        created_tasks.append(task)
        return None

    import openhcs.pyqt_gui.widgets.pipeline_editor as pipeline_editor_module

    monkeypatch.setattr(pipeline_editor_module.asyncio, "create_task", record_task)
    harness = PipelineEditorRunHarness()

    PipelineEditorWidget._dispatch_debug_run_command(
        harness,
        DebugCommandType.RUN_TO_PAUSE,
    )

    assert created_tasks == [
        (
            "debug-run",
            "plate",
            {
                "command_type": DebugCommandType.RUN_TO_PAUSE,
                "pause_step_indices": (1,),
                "start_step_index": 0,
                "start_after_invocation_key": None,
            },
        )
    ]
    assert harness.plate_manager.run_calls == [
        (
            "plate",
            {
                "command_type": DebugCommandType.RUN_TO_PAUSE,
                "pause_step_indices": (1,),
                "start_step_index": 0,
                "start_after_invocation_key": None,
            },
        )
    ]
    assert "Submitting debug run to pause" in harness.status_message.messages[0]


def test_pipeline_editor_derives_pause_step_indices() -> None:
    harness = PipelineEditorRunHarness()

    assert PipelineEditorWidget._debug_pause_step_indices(harness) == (1,)


def test_pipeline_editor_restarts_from_dirty_debug_cursor() -> None:
    harness = PipelineEditorDirtyHarness()

    harness.debug_session_state = harness.debug_session_state.mark_dirty_from_cursor()

    assert (
        PipelineEditorWidget._debug_start_step_index(
            harness,
            DebugCommandType.RESTART,
    )
    == 1
    )


def test_pipeline_editor_step_advances_from_current_debug_invocation() -> None:
    harness = PipelineEditorDirtyHarness()

    assert (
        PipelineEditorWidget._debug_start_step_index(
            harness,
            DebugCommandType.STEP,
        )
        == 1
    )
    assert (
        PipelineEditorWidget._debug_start_after_invocation_key(
            harness,
            DebugCommandType.STEP,
        )
        == "default:0:segment"
    )


def test_pipeline_editor_marks_debug_session_dirty_on_pipeline_change() -> None:
    harness = PipelineEditorDirtyHarness()
    steps = [FunctionStep(func=lambda image: image, name="changed")]

    PipelineEditorWidget.on_pipeline_changed(harness, steps)

    assert harness.saved == [("plate", steps)]
    assert harness.debug_session_state.dirty_from_cursor is not None
    assert harness.status_message.messages == [
        "Debug snapshots downstream of the current cursor are dirty."
    ]


def test_pipeline_editor_formats_invocation_badges_with_debug_cursor() -> None:
    def segment(image):
        return image

    def finish(image):
        return image

    harness = PipelineEditorDirtyHarness()
    harness.debug_session_state = harness.debug_session_state.mark_dirty_from_cursor()

    badges = PipelineEditorWidget.function_pattern_invocation_badges(
        harness,
        [segment, finish],
    )

    assert tuple(badge.text for badge in badges) == (
        "▶ default[0] segment *",
        "default[1] finish",
    )


def test_pipeline_editor_reports_stop_without_plate_manager() -> None:
    harness = PipelineEditorCommandHarness(None)

    PipelineEditorWidget._handle_debug_command(
        harness,
        DebugCommand(DebugCommandType.STOP),
    )

    assert harness.status_message.messages == [
        "Debug stop requires a connected Plate Manager.",
    ]


def test_pipeline_editor_loads_vfs_debug_snapshot_store(monkeypatch) -> None:
    import openhcs.pyqt_gui.widgets.pipeline_editor as pipeline_editor_module

    monkeypatch.setattr(
        pipeline_editor_module,
        "DebugInspectorWindow",
        DebugInspectorRecorder,
    )
    harness = PipelineEditorSnapshotHarness()

    PipelineEditorWidget.show_debug_snapshot(
        harness,
        debug_snapshot_notification(snapshot_store_backend="memory"),
    )

    inspector = harness.debug_inspector_window
    assert inspector.local_loads == []
    assert len(inspector.store_loads) == 1
    store, snapshot_id = inspector.store_loads[0]
    assert isinstance(store, FileManagerDebugSnapshotStore)
    assert store.filemanager is harness.filemanager
    assert store.backend == "memory"
    assert snapshot_id == "snap-1"


def test_pipeline_editor_connects_debug_inspector_artifact_actions(monkeypatch) -> None:
    import openhcs.pyqt_gui.widgets.pipeline_editor as pipeline_editor_module

    monkeypatch.setattr(
        pipeline_editor_module,
        "DebugInspectorWindow",
        DebugInspectorRecorder,
    )
    harness = PipelineEditorSnapshotHarness()

    PipelineEditorWidget.show_debug_snapshot(
        harness,
        debug_snapshot_notification(snapshot_store_backend="memory"),
    )

    inspector = harness.debug_inspector_window
    assert inspector.artifact_export_requested.connected == [
        harness._handle_debug_artifact_export_request
    ]
    assert inspector.artifact_open_requested.connected == [
        harness._handle_debug_artifact_open_request
    ]


def test_pipeline_editor_exports_debug_artifact_through_plate_manager(monkeypatch) -> None:
    import openhcs.pyqt_gui.widgets.pipeline_editor as pipeline_editor_module

    created_tasks = []
    monkeypatch.setattr(
        pipeline_editor_module.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: "/tmp/debug-export",
    )
    monkeypatch.setattr(
        pipeline_editor_module.asyncio,
        "create_task",
        lambda task: created_tasks.append(task),
    )
    harness = PipelineEditorSnapshotHarness()
    harness.debug_session_state = DebugSession(
        debug_session_id="debug-1",
        snapshot_store_ref="/debug",
        snapshot_store_backend="memory",
    )
    harness.plate_manager = PlateManagerRunRecorder()
    harness.plate_manager.action_export_debug_artifact = lambda **kwargs: (
        "export-task",
        kwargs,
    )
    artifact_ref = DebugArtifactRef(
        kind=ArtifactKind.MEASUREMENTS,
        name="Measurements",
        cursor=DebugCursor(0, "scope", "default", "default:0:measure"),
        storage_ref="/debug/measurements.csv",
        storage_backend="memory",
    )

    PipelineEditorWidget._handle_debug_artifact_export_request(
        harness,
        DebugArtifactMaterializeRequest(artifact_ref=artifact_ref),
    )

    assert created_tasks == [
        (
            "export-task",
            {
                "debug_session_id": "debug-1",
                "artifact_ref": artifact_ref,
                "export_root": "/tmp/debug-export",
                "snapshot_store_ref": "/debug",
                "snapshot_store_backend": "memory",
            },
        )
    ]
