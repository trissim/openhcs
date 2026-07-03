from __future__ import annotations

import asyncio
from types import SimpleNamespace

from objectstate.lazy_factory import PREVIEW_LABEL_REGISTRY
from metaclass_registry import AutoRegisterMeta
from PyQt6.QtWidgets import QApplication

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.config import NapariStreamingConfig
from openhcs.core.execution_state import ManagerExecutionState
from openhcs.core.debug import (
    DebugArtifactRef,
    DebugCommand,
    DebugCommandType,
    DebugCursor,
    DebugEventType,
    DebugProgressContext,
    DebugReplayMode,
    DebugSession,
    DebugTerminalSummary,
    FileManagerDebugSnapshotStore,
)
from openhcs.core.debug_views import DebugViewModel
from openhcs.pyqt_gui.windows.debug_inspector_window import (
    DebugArtifactMaterializeRequest,
)
from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.services.plate_manager_batch_workflow import (
    DebugSnapshotAvailableNotification,
)
from openhcs.pyqt_gui.widgets.debug_toolbar import DebugToolbarWidget
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
    PipelineDebugPauseBoundaryState,
    PipelineDebugSessionContext,
    PipelineDebugTargetState,
)
from openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions import (
    PipelineDebugActionDeclarationBase,
    PipelineDebugCommandActionDeclaration,
)
from openhcs.pyqt_gui.widgets.shared.services.pipeline_editor_workflows import (
    PipelineEditorDebugWorkflow,
    PipelineEditorFunctionPresentation,
)
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget


class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


def debug_toolbar_context(
    *,
    initialized: bool = True,
    compiled: bool = True,
    session: DebugSession | None = None,
    terminal_summary: DebugTerminalSummary | None = None,
    manager_execution_state: ManagerExecutionState = ManagerExecutionState.IDLE,
    pause_step_indices: tuple[int, ...] = (1,),
) -> PipelineDebugSessionContext:
    return PipelineDebugSessionContext(
        target=PipelineDebugTargetState(
            current_plate_scope_id="plate",
            pipeline_scope_id="plate::pipeline",
            initialized=initialized,
            compiled=compiled,
        ),
        session=session,
        terminal_summary=terminal_summary,
        pause_boundaries=PipelineDebugPauseBoundaryState(pause_step_indices),
        manager_execution_state=manager_execution_state,
    )


def active_debug_session() -> DebugSession:
    return DebugSession.create(plate_id="plate")


def test_debug_toolbar_emits_typed_command() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()
    toolbar.set_debug_session_context(debug_toolbar_context())
    commands: list[DebugCommand] = []
    toolbar.command_requested.connect(commands.append)

    toolbar.buttons[DebugCommandType.STEP].click()

    assert commands == [DebugCommand(DebugCommandType.STEP)]


def test_debug_toolbar_debug_button_runs_debug_execution() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()
    toolbar.set_debug_session_context(debug_toolbar_context())
    commands: list[DebugCommand] = []
    toolbar.command_requested.connect(commands.append)

    toolbar.buttons[DebugCommandType.RUN].click()

    assert toolbar.buttons[DebugCommandType.RUN].text() == "Start Debug"
    assert toolbar.phase_label.text() == "Ready"
    assert commands == [DebugCommand(DebugCommandType.RUN)]


def test_debug_toolbar_renders_secondary_commands_as_session_buttons() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()
    toolbar.set_debug_session_context(debug_toolbar_context())
    commands: list[DebugCommand] = []
    toolbar.command_requested.connect(commands.append)

    assert DebugCommandType.TOGGLE not in toolbar.buttons
    assert DebugCommandType.CHOOSE_SOURCE_GROUP in toolbar.buttons
    assert DebugCommandType.RANDOM_SOURCE_GROUP not in toolbar.buttons
    assert DebugCommandType.STOP in toolbar.buttons

    toolbar.buttons[DebugCommandType.CHOOSE_SOURCE_GROUP].click()
    toolbar.set_debug_session_context(
        debug_toolbar_context(session=active_debug_session())
    )
    toolbar.buttons[DebugCommandType.STOP].click()

    assert commands == [
        DebugCommand(DebugCommandType.CHOOSE_SOURCE_GROUP),
        DebugCommand(DebugCommandType.STOP),
    ]


def test_debug_toolbar_runtime_inspection_action_emits_separate_signal() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()
    requests = []
    toolbar.runtime_inspection_requested.connect(lambda: requests.append("runtime"))

    assert toolbar.runtime_inspection_button is not None
    toolbar.set_debug_session_context(
        debug_toolbar_context(session=active_debug_session())
    )
    toolbar.runtime_inspection_button.click()

    assert requests == ["runtime"]


def test_debug_toolbar_enables_controls_together() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()

    assert all(not button.isEnabled() for button in toolbar.buttons.values())
    assert toolbar.runtime_inspection_button is not None
    assert not toolbar.runtime_inspection_button.isEnabled()


def test_debug_toolbar_session_only_controls_follow_active_session() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()
    toolbar.set_debug_session_context(debug_toolbar_context())

    assert toolbar.buttons[DebugCommandType.RUN].isEnabled()
    assert toolbar.buttons[DebugCommandType.STEP].isEnabled()
    assert toolbar.buttons[DebugCommandType.CHOOSE_SOURCE_GROUP].isEnabled()
    assert not toolbar.buttons[DebugCommandType.RESTART].isEnabled()
    assert not toolbar.buttons[DebugCommandType.STOP].isEnabled()
    assert toolbar.runtime_inspection_button is not None
    assert not toolbar.runtime_inspection_button.isEnabled()

    toolbar.set_debug_session_context(
        debug_toolbar_context(session=active_debug_session())
    )

    assert toolbar.phase_label.text() == "Debug Active"
    assert toolbar.buttons[DebugCommandType.RUN].text() == "Continue"
    assert toolbar.buttons[DebugCommandType.RESTART].isEnabled()
    assert toolbar.buttons[DebugCommandType.STOP].isEnabled()
    assert toolbar.runtime_inspection_button.isEnabled()


def test_debug_toolbar_terminal_summary_retire_matching_local_session() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()
    session = active_debug_session()
    cursor = DebugCursor(
        step_index=0,
        step_scope_id="step-0",
        group_key="default",
        invocation_key="default:0:correct_illumination_apply",
    )
    session = session.with_cursor(cursor).with_command(DebugCommandType.STEP)
    terminal_summary = DebugTerminalSummary(
        debug_session_id=session.debug_session_id,
        plate_id="plate",
        terminal_status="complete",
        cursor=cursor,
        command_type=DebugCommandType.STEP,
        axis_id=session.axis_id,
        snapshot_id="snapshot-1",
        snapshot_store_ref="/debug",
    )

    toolbar.set_debug_session_context(
        debug_toolbar_context(
            session=session,
            terminal_summary=terminal_summary,
        )
    )

    assert toolbar.phase_label.text() == "Debug Complete"
    assert toolbar.buttons[DebugCommandType.RUN].text() == "Start Debug"
    assert not toolbar.buttons[DebugCommandType.RESTART].isEnabled()
    assert not toolbar.buttons[DebugCommandType.STOP].isEnabled()
    assert toolbar.runtime_inspection_button is not None
    assert not toolbar.runtime_inspection_button.isEnabled()


def test_debug_toolbar_projects_pending_execution_state() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()
    toolbar.set_debug_session_context(
        debug_toolbar_context(
            manager_execution_state=ManagerExecutionState.RUNNING,
        )
    )

    assert not toolbar.buttons[DebugCommandType.RUN].isEnabled()
    assert not toolbar.buttons[DebugCommandType.STEP].isEnabled()
    assert not toolbar.buttons[DebugCommandType.RESTART].isEnabled()
    assert not toolbar.buttons[DebugCommandType.CHOOSE_SOURCE_GROUP].isEnabled()
    assert toolbar.buttons[DebugCommandType.STOP].isEnabled()
    assert toolbar.runtime_inspection_button is not None
    assert not toolbar.runtime_inspection_button.isEnabled()


def test_debug_toolbar_omits_random_source_group_action() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()

    assert DebugCommandType.RANDOM_SOURCE_GROUP not in toolbar.buttons


def test_debug_toolbar_uses_shared_button_panel_styling() -> None:
    QtApplicationHarness.app()
    style_generator = SimpleNamespace(generate_button_style=lambda: "QPushButton { color: red; }")

    toolbar = DebugToolbarWidget(style_generator=style_generator)

    assert toolbar.button_panel is not None
    assert "color: red" in toolbar.buttons[DebugCommandType.STEP].styleSheet()


def test_debug_toolbar_disables_run_to_pause_without_pause_boundary() -> None:
    QtApplicationHarness.app()
    toolbar = DebugToolbarWidget()

    toolbar.set_debug_session_context(debug_toolbar_context(pause_step_indices=()))

    assert not toolbar.buttons[DebugCommandType.RUN_TO_PAUSE].isEnabled()
    assert (
        "debug-pause step"
        in toolbar.buttons[DebugCommandType.RUN_TO_PAUSE].toolTip()
    )


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
        self.force_values: list[bool | None] = []

    def action_stop_execution(self, force: bool | None = None) -> None:
        self.stop_calls += 1
        self.force_values.append(force)


class PlateManagerRunRecorder:
    """Plate-manager seam used by debug run dispatch tests."""

    def __init__(self) -> None:
        self.run_calls = []
        self.terminal_summary = None
        self.debug_session_state = None
        self.execution_state = ManagerExecutionState.IDLE

    async def action_run_debug_plate(self, plate_path, **kwargs):
        self.run_calls.append((plate_path, kwargs))

    async def action_export_debug_artifact(self, **kwargs) -> None:
        self.run_calls.append(("export", kwargs))

    async def action_inspect_debug_runtime(self, *, debug_session_id: str):
        return DebugViewModel(
            title=f"Runtime {debug_session_id}",
            sections=(),
        )

    def debug_terminal_summary_for_plate(self, plate_path: str):
        del plate_path
        return self.terminal_summary

    def debug_session_for_plate(self, plate_path: str):
        del plate_path
        return self.debug_session_state

    def debug_session_context_for_plate(
        self,
        plate_path: str,
    ) -> PipelineDebugSessionContext:
        return PipelineDebugSessionContext(
            target=PipelineDebugTargetState(
                current_plate_scope_id=plate_path,
                pipeline_scope_id="plate::pipeline",
                initialized=True,
                compiled=True,
            ),
            session=self.debug_session_for_plate(plate_path),
            terminal_summary=self.debug_terminal_summary_for_plate(plate_path),
            manager_execution_state=self.execution_state,
        )


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
        self._debug_terminal_summaries_by_plate = {}
        self.manager_execution_state_changed = StatusSignalRecorder()
        self.execution_state = ManagerExecutionState.IDLE
        self._batch_workflow_service = DebugBatchWorkflowRecorder()
        self.execution_error = StatusSignalRecorder()
        self.export_calls = []
        self.item_updates = 0
        self.button_updates = 0
        self.plate_terminal_activity_status = SimpleNamespace(
            terminal_status=lambda plate_path: None
        )

    def get_selected_items(self) -> list[dict[str, str]]:
        return []

    async def action_export_debug_artifact(self, **kwargs) -> None:
        self.export_calls.append(kwargs)

    def debug_session_for_plate(self, plate_path: str) -> DebugSession | None:
        return self._active_debug_sessions.get(plate_path)

    def debug_terminal_summary_for_plate(self, plate_path: str):
        del plate_path
        return None

    def update_item_list(self) -> None:
        self.item_updates += 1

    def update_button_states(self) -> None:
        self.button_updates += 1


class PipelineEditorDebugStateRecorder:
    """Pipeline-editor state used by terminal debug-session cleanup tests."""

    def __init__(self, session: DebugSession) -> None:
        self.current_plate = None if session.plate_id is None else session.plate_id
        self.debug_session_state: DebugSession | None = session
        self.debug_terminal_summary = None
        self.item_updates = 0
        self.button_updates = 0

    def update_item_list(self) -> None:
        self.item_updates += 1

    def update_button_states(self) -> None:
        self.button_updates += 1


class ImmediatePipelineEditorCoroutineRunner:
    """Test runner that executes workflow coroutines synchronously."""

    def __init__(self, editor) -> None:
        self.editor = editor

    def submit(self, coroutine) -> None:
        asyncio.run(coroutine)


class PipelineEditorHarnessBase(metaclass=AutoRegisterMeta):
    """Shared pipeline-editor harness fields for debug command tests."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    def __init__(self, plate_manager) -> None:
        self.plate_manager = plate_manager
        self.status_message = StatusSignalRecorder()
        self.debug_workflow = PipelineEditorDebugWorkflow(self)
        self.function_presentation = PipelineEditorFunctionPresentation(self)
        self.pipeline_steps = []


class PipelineEditorCommandHarness(PipelineEditorHarnessBase):
    """Minimal object carrying the attributes used by debug command dispatch."""

    registry_key = "command"
    def __init__(self, plate_manager: PlateManagerStopRecorder | None) -> None:
        super().__init__(plate_manager)
        self.debug_run_commands: list[DebugCommandType] = []


class PipelineEditorRunHarness(PipelineEditorHarnessBase):
    """Minimal object carrying attributes used by debug run dispatch."""

    registry_key = "run"

    def __init__(self) -> None:
        super().__init__(PlateManagerRunRecorder())
        self.current_plate = "plate"
        self.debug_session_state = None
        self.debug_terminal_summary = None
        self.debug_inspector_window = None
        self.pipeline_steps = [
            FunctionStep(func=lambda image: image, name="first"),
            FunctionStep(func=lambda image: image, name="pause", debug_pause=True),
        ]

    def debug_session_context(self) -> PipelineDebugSessionContext:
        return PipelineEditorWidget.debug_session_context(self)


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
        self.debug_terminal_summary = None

    def save_pipeline_for_plate(self, plate, steps) -> None:
        self.saved.append((plate, steps))

    def notify_pipeline_definition_changed(self, plate) -> None:
        return None


class DebugInspectorRecorder:
    """Recorder replacing the heavy Qt inspector in route tests."""

    def __init__(self, parent=None) -> None:
        del parent
        self.snapshots = []
        self.local_loads = []
        self.store_loads = []
        self.show_calls = 0
        self.raise_calls = 0
        self.inspection_view_models = []
        self.artifact_export_requested = SignalConnectRecorder()
        self.artifact_open_requested = SignalConnectRecorder()

    def set_inspection_view_model(self, view_model) -> None:
        self.inspection_view_models.append(view_model)

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
        self.debug_session_state = None
        self.debug_terminal_summary = None
        self.debug_workflow = PipelineEditorDebugWorkflow(self)
        self.button_state_updates = 0
        self.item_updates = 0

    def get_file_manager(self) -> FileManagerRecorder:
        return self.filemanager

    def update_item_list(self) -> None:
        self.item_updates += 1

    def update_button_states(self) -> None:
        self.button_state_updates += 1

    def _handle_debug_artifact_export_request(self, request) -> None:
        self.debug_workflow.handle_artifact_export_request(request)

    def _handle_debug_artifact_open_request(self, request) -> None:
        self.debug_workflow.handle_artifact_open_request(request)


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
            identity=ProgressIdentity(
                execution_id="exec-1",
                plate_id="plate",
                axis_id="A01",
                step_name="step",
            ),
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

    harness.debug_workflow.handle_command(DebugCommand(DebugCommandType.STOP))

    assert plate_manager.stop_calls == 1
    assert plate_manager.force_values == [True]
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


def test_plate_manager_completion_clears_matching_pipeline_editor_debug_session() -> None:
    plate_path = "plate"
    manager = PlateManagerDebugHarness()
    session = DebugSession.create(plate_id=plate_path)
    manager._active_debug_sessions[plate_path] = session

    PlateManagerWidget._clear_debug_session_for_plate(manager, plate_path)

    assert plate_path not in manager._active_debug_sessions
    assert manager._debug_terminal_summaries_by_plate == {}


def test_pipeline_editor_routes_step_debug_command_to_bounded_run(monkeypatch) -> None:
    from openhcs.pyqt_gui.widgets.shared.services import pipeline_editor_workflows

    monkeypatch.setattr(
        pipeline_editor_workflows,
        "PipelineEditorCoroutineRunner",
        ImmediatePipelineEditorCoroutineRunner,
    )
    harness = PipelineEditorRunHarness()

    harness.debug_workflow.handle_command(DebugCommand(DebugCommandType.STEP))

    assert harness.plate_manager.run_calls[0][1]["command_type"] is DebugCommandType.STEP


def test_pipeline_editor_has_route_for_every_debug_command() -> None:
    command_types = {
        declaration.command_type()
        for declaration in PipelineDebugActionDeclarationBase.__registry__.values()
        if issubclass(declaration, PipelineDebugCommandActionDeclaration)
    }
    assert command_types == set(DebugCommandType)


def test_streaming_preview_label_is_declared_on_config_class() -> None:
    assert PREVIEW_LABEL_REGISTRY[NapariStreamingConfig] == "NAP"


def test_pipeline_editor_dispatches_pause_step_indices(monkeypatch) -> None:
    from openhcs.pyqt_gui.widgets.shared.services import pipeline_editor_workflows

    monkeypatch.setattr(
        pipeline_editor_workflows,
        "PipelineEditorCoroutineRunner",
        ImmediatePipelineEditorCoroutineRunner,
    )
    harness = PipelineEditorRunHarness()

    harness.debug_workflow.run_command(DebugCommandType.RUN_TO_PAUSE)

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

    assert harness.debug_workflow.pause_step_indices() == (1,)


def test_pipeline_editor_restarts_from_dirty_debug_cursor() -> None:
    harness = PipelineEditorDirtyHarness()

    harness.debug_session_state = harness.debug_session_state.mark_dirty_from_cursor()

    assert (
        harness.debug_workflow.start_step_index(DebugCommandType.RESTART)
    == 1
    )


def test_pipeline_editor_step_advances_from_current_debug_invocation() -> None:
    harness = PipelineEditorDirtyHarness()

    assert (
        harness.debug_workflow.start_step_index(DebugCommandType.STEP)
        == 1
    )
    assert (
        harness.debug_workflow.start_after_invocation_key(DebugCommandType.STEP)
        == "default:0:segment"
    )


def test_pipeline_editor_step_replays_from_terminal_debug_cursor(monkeypatch) -> None:
    from openhcs.pyqt_gui.widgets.shared.services import pipeline_editor_workflows

    monkeypatch.setattr(
        pipeline_editor_workflows,
        "PipelineEditorCoroutineRunner",
        ImmediatePipelineEditorCoroutineRunner,
    )
    harness = PipelineEditorRunHarness()
    cursor = DebugCursor(
        step_index=2,
        step_scope_id="step-2",
        group_key="default",
        invocation_key="default:0:measure",
    )
    harness.plate_manager.terminal_summary = DebugTerminalSummary(
        debug_session_id="debug-terminal",
        plate_id="plate",
        terminal_status="complete",
        cursor=cursor,
        command_type=DebugCommandType.STEP,
        axis_id="A01",
    )

    harness.debug_workflow.run_command(DebugCommandType.STEP)

    assert harness.plate_manager.run_calls == [
        (
            "plate",
            {
                "command_type": DebugCommandType.STEP,
                "pause_step_indices": (1,),
                "start_step_index": 2,
                "start_after_invocation_key": "default:0:measure",
            },
        )
    ]


def test_pipeline_editor_context_uses_manager_terminal_state_over_stale_local_session() -> None:
    harness = PipelineEditorRunHarness()
    local_cursor = DebugCursor(
        step_index=0,
        step_scope_id="step-0",
        group_key="default",
        invocation_key="default:0:first",
    )
    harness.debug_session_state = DebugSession.create(
        plate_id="plate",
        execution_id="old-exec",
        axis_id="A01",
    ).with_cursor(local_cursor)
    terminal_cursor = DebugCursor(
        step_index=2,
        step_scope_id="step-2",
        group_key="default",
        invocation_key="default:0:measure",
    )
    harness.plate_manager.terminal_summary = DebugTerminalSummary(
        debug_session_id="debug-terminal",
        plate_id="plate",
        terminal_status="complete",
        cursor=terminal_cursor,
        command_type=DebugCommandType.STEP,
        axis_id="A01",
    )

    context = PipelineEditorWidget.debug_session_context(harness)

    assert context.active_session is None
    assert context.phase.value == "terminal_complete"
    assert context.terminal_summary is harness.plate_manager.terminal_summary


def test_pipeline_editor_runtime_inspection_posts_render_from_active_context(
    monkeypatch,
) -> None:
    from openhcs.pyqt_gui.widgets.shared.services import pipeline_editor_workflows

    class ImmediateUiThreadDispatcher:
        def post(self, callback) -> None:
            callback()

    monkeypatch.setattr(
        pipeline_editor_workflows,
        "PipelineEditorCoroutineRunner",
        ImmediatePipelineEditorCoroutineRunner,
    )
    monkeypatch.setattr(
        pipeline_editor_workflows,
        "UiThreadDispatcher",
        ImmediateUiThreadDispatcher,
    )
    monkeypatch.setattr(
        pipeline_editor_workflows,
        "DebugInspectorWindow",
        DebugInspectorRecorder,
    )
    harness = PipelineEditorRunHarness()
    session = DebugSession.create(
        plate_id="plate",
        execution_id="exec-1",
        axis_id="A01",
    )
    harness.debug_session_state = session

    harness.debug_workflow.show_runtime_inspection()

    inspector = harness.debug_inspector_window
    assert inspector.inspection_view_models == [
        DebugViewModel(
            title=f"Runtime {session.debug_session_id}",
            sections=(),
        )
    ]
    assert inspector.show_calls == 1
    assert inspector.raise_calls == 1


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

    badges = harness.function_presentation.invocation_badges([segment, finish])

    assert tuple(badge.text for badge in badges) == (
        "default[0] segment",
        "default[1] finish",
    )

    non_cursor_row_badges = harness.function_presentation.invocation_badges(
        [segment, finish],
        step_index=0,
    )
    cursor_row_badges = harness.function_presentation.invocation_badges(
        [segment, finish],
        step_index=1,
    )

    assert tuple(badge.text for badge in non_cursor_row_badges) == (
        "default[0] segment",
        "default[1] finish",
    )
    assert tuple(badge.text for badge in cursor_row_badges) == (
        "▶ default[0] segment *",
        "default[1] finish",
    )


def test_pipeline_editor_badge_provider_uses_rendered_step_index() -> None:
    def segment(image):
        return image

    harness = PipelineEditorDirtyHarness()
    first = FunctionStep(func=segment, name="first")
    second = FunctionStep(func=segment, name="second")
    harness.pipeline_steps = [first, second]

    first_badge_provider = harness.function_presentation.badge_provider(
        first,
        step_index=0,
    )
    second_badge_provider = harness.function_presentation.badge_provider(
        second,
        step_index=1,
    )

    assert first_badge_provider("default", 0, segment) is None
    assert second_badge_provider("default", 0, segment) == "▶ default[0] segment"


def test_pipeline_editor_hides_inactive_invocation_badges_from_titles() -> None:
    def crop(image):
        return image

    harness = PipelineEditorDirtyHarness()
    step = FunctionStep(func=crop)

    badge_provider = harness.function_presentation.badge_provider(step, step_index=0)

    assert badge_provider("default", 0, crop) is None
    assert harness.function_presentation.format_func_preview(crop) == "func=crop"


def test_pipeline_editor_reports_stop_without_plate_manager() -> None:
    harness = PipelineEditorCommandHarness(None)

    harness.debug_workflow.handle_command(DebugCommand(DebugCommandType.STOP))

    assert harness.status_message.messages == [
        "Debug stop requires a connected Plate Manager.",
    ]


def test_pipeline_editor_loads_vfs_debug_snapshot_store(monkeypatch) -> None:
    from openhcs.pyqt_gui.widgets.shared.services import pipeline_editor_workflows

    monkeypatch.setattr(
        pipeline_editor_workflows,
        "DebugInspectorWindow",
        DebugInspectorRecorder,
    )
    harness = PipelineEditorSnapshotHarness()

    harness.debug_workflow.show_snapshot(
        debug_snapshot_notification(snapshot_store_backend="memory")
    )

    inspector = harness.debug_inspector_window
    assert inspector.local_loads == []
    assert len(inspector.store_loads) == 1
    store, snapshot_id = inspector.store_loads[0]
    assert isinstance(store, FileManagerDebugSnapshotStore)
    assert store.filemanager is harness.filemanager
    assert store.backend == "memory"
    assert snapshot_id == "snap-1"
    assert harness.item_updates == 1
    assert harness.button_state_updates == 1


def test_pipeline_editor_connects_debug_inspector_artifact_actions(monkeypatch) -> None:
    from openhcs.pyqt_gui.widgets.shared.services import pipeline_editor_workflows

    monkeypatch.setattr(
        pipeline_editor_workflows,
        "DebugInspectorWindow",
        DebugInspectorRecorder,
    )
    harness = PipelineEditorSnapshotHarness()

    harness.debug_workflow.show_snapshot(
        debug_snapshot_notification(snapshot_store_backend="memory")
    )

    inspector = harness.debug_inspector_window
    assert inspector.artifact_export_requested.connected == [
        harness.debug_workflow.handle_artifact_export_request
    ]
    assert inspector.artifact_open_requested.connected == [
        harness.debug_workflow.handle_artifact_open_request
    ]


def test_pipeline_editor_exports_debug_artifact_through_plate_manager(monkeypatch) -> None:
    from openhcs.pyqt_gui.widgets.shared.services import pipeline_editor_workflows

    monkeypatch.setattr(
        pipeline_editor_workflows.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: "/tmp/debug-export",
    )
    monkeypatch.setattr(
        pipeline_editor_workflows,
        "PipelineEditorCoroutineRunner",
        ImmediatePipelineEditorCoroutineRunner,
    )
    harness = PipelineEditorSnapshotHarness()
    harness.debug_session_state = DebugSession(
        debug_session_id="debug-1",
        snapshot_store_ref="/debug",
        snapshot_store_backend="memory",
    )
    harness.plate_manager = PlateManagerRunRecorder()
    artifact_ref = DebugArtifactRef(
        kind=ArtifactKind.MEASUREMENTS,
        name="Measurements",
        cursor=DebugCursor(0, "scope", "default", "default:0:measure"),
        storage_ref="/debug/measurements.csv",
        storage_backend="memory",
    )

    harness.debug_workflow.handle_artifact_export_request(
        DebugArtifactMaterializeRequest(artifact_ref=artifact_ref)
    )

    assert harness.plate_manager.run_calls == [
        (
            "export",
            {
                "debug_session_id": "debug-1",
                "artifact_ref": artifact_ref,
                "export_root": "/tmp/debug-export",
                "snapshot_store_ref": "/debug",
                "snapshot_store_backend": "memory",
            },
        )
    ]


def test_debug_gui_workflow_runs_commands_inspects_snapshot_and_exports(
    monkeypatch,
    tmp_path,
) -> None:
    from openhcs.pyqt_gui.widgets.shared.services import pipeline_editor_workflows

    monkeypatch.setattr(
        pipeline_editor_workflows,
        "DebugInspectorWindow",
        DebugInspectorRecorder,
    )
    monkeypatch.setattr(
        pipeline_editor_workflows.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path / "export"),
    )
    monkeypatch.setattr(
        pipeline_editor_workflows,
        "PipelineEditorCoroutineRunner",
        ImmediatePipelineEditorCoroutineRunner,
    )

    plate_path = str(tmp_path / "plate")
    plate_manager = PlateManagerDebugHarness()
    editor = PipelineEditorSnapshotHarness()
    editor.plate_manager = plate_manager

    asyncio.run(
        PlateManagerWidget.action_run_debug_plate(
            plate_manager,
            plate_path,
            command_type=DebugCommandType.RUN_TO_PAUSE,
            pause_step_indices=(1,),
        )
    )
    session = plate_manager._active_debug_sessions[plate_path]
    asyncio.run(
        PlateManagerWidget.action_run_debug_plate(
            plate_manager,
            plate_path,
            command_type=DebugCommandType.STEP,
        )
    )
    asyncio.run(
        PlateManagerWidget.action_run_debug_plate(
            plate_manager,
            plate_path,
            command_type=DebugCommandType.RUN,
        )
    )

    editor.debug_workflow.show_snapshot(
        debug_snapshot_notification(snapshot_store_backend="memory")
    )
    artifact_ref = DebugArtifactRef(
        kind=ArtifactKind.MEASUREMENTS,
        name="Measurements",
        cursor=DebugCursor(0, "scope", "default", "default:0:measure"),
        storage_ref="/debug/measurements.csv",
        storage_backend="memory",
    )
    editor.debug_workflow.handle_artifact_export_request(
        DebugArtifactMaterializeRequest(artifact_ref=artifact_ref)
    )
    asyncio.run(
        PlateManagerWidget.action_run_debug_plate(
            plate_manager,
            plate_path,
            command_type=DebugCommandType.STOP,
        )
    )

    assert plate_manager._batch_workflow_service.run_calls[0]["replay_mode"] is (
        DebugReplayMode.PERSISTENT_PAUSED_WORKER
    )
    assert plate_manager._batch_workflow_service.worker_commands == [
        (session.debug_session_id, DebugCommandType.STEP),
        (session.debug_session_id, DebugCommandType.RUN),
        (session.debug_session_id, DebugCommandType.STOP),
    ]
    assert plate_path not in plate_manager._active_debug_sessions
    assert editor.debug_session_state.debug_session_id == "debug-1"
    assert len(editor.debug_inspector_window.store_loads) == 1
    assert plate_manager.export_calls == [
        {
            "debug_session_id": "debug-1",
            "artifact_ref": artifact_ref,
            "export_root": str(tmp_path / "export"),
            "snapshot_store_ref": "/debug",
            "snapshot_store_backend": "memory",
        }
    ]
