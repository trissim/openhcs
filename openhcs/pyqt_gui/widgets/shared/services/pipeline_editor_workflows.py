"""Workflow services owned by the pipeline editor widget."""

from __future__ import annotations

import logging
from collections.abc import Coroutine, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypeAlias

from objectstate import patch_lazy_constructors
from objectstate.object_state import ObjectState, ObjectStateRegistry
from PyQt6.QtWidgets import QFileDialog
from pyqt_reactive.services.scope_token_service import ScopeTokenService
from pyqt_reactive.widgets.shared.manager_workflows import (
    ManagerCodeExecutionWorkflow,
    ManagerDeletionWorkflow,
)

from openhcs.core.callable_contract import CallableContract
from openhcs.core.debug import (
    DebugCommandType,
    DebugCursor,
    DebugSession,
    FileManagerDebugSnapshotStore,
)
from openhcs.core.debug_views import DebugViewModel
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.core.steps.abstract import AbstractStep
from openhcs.pyqt_gui.services.pipeline_object_state_binding import (
    PipelineObjectStateBinding,
)
from openhcs.pyqt_gui.widgets.shared.services.gui_event_bus_broadcast import (
    GuiEventBusBroadcaster,
)
from openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions import (
    PipelineDebugActionDeclarationBase,
)
from openhcs.pyqt_gui.windows.debug_inspector_window import DebugInspectorWindow
from openhcs.utils.pipeline_migration import patch_step_constructors_for_migration

logger = logging.getLogger(__name__)

TimeTravelDirtyStates: TypeAlias = Iterable[tuple[str, ObjectState]]


class WorkflowSignal(Protocol):
    """Signal-like object used by workflow services."""

    def emit(self, *values) -> None:
        """Emit a workflow event."""


class PipelineStepSaveEditor(Protocol):
    """Editor surface required by PipelineStepSaveWorkflow."""

    pipeline_steps: list[AbstractStep]
    pipeline_changed: WorkflowSignal
    status_message: WorkflowSignal

    def update_item_list(self) -> None:
        """Refresh the visible pipeline list."""

    def require_pipeline_definition_mutation_allowed(
        self,
        plate_path: str | None = None,
    ) -> None:
        """Reject mutation while the owning manager is executing."""


@dataclass(frozen=True, slots=True)
class PipelineEditorCoroutineRunner:
    """Run async GUI workflow commands from Qt slots without requiring qasync."""

    editor: Any

    def submit(self, coroutine: Coroutine[Any, Any, None]) -> None:
        self.editor.service_adapter.execute_async_operation(
            self._guard,
            coroutine,
        )

    async def _guard(self, coroutine: Coroutine[Any, Any, None]) -> None:
        try:
            await coroutine
        except Exception as exc:
            logger.exception("Pipeline editor async command failed")
            self.editor.status_message.emit(str(exc))


@dataclass(frozen=True, slots=True)
class FunctionPatternInvocationBadge:
    """GUI badge for one invocation in a FunctionStep function pattern."""

    group_key: str
    position: int
    function_name: str
    is_current_cursor: bool = False
    is_dirty_replay_start: bool = False

    @property
    def text(self) -> str:
        prefix = "▶ " if self.is_current_cursor else ""
        suffix = " *" if self.is_dirty_replay_start else ""
        return f"{prefix}{self.group_key}[{self.position}] {self.function_name}{suffix}"

    @property
    def is_visible(self) -> bool:
        """Only debug-significant invocations need a title badge."""
        return self.is_current_cursor or self.is_dirty_replay_start


@dataclass(frozen=True, slots=True)
class PipelineEditorFunctionPresentation:
    """Owns function-pattern names, preview text, and invocation badges."""

    editor: Any

    def format_func_preview(
        self,
        func,
        state=None,
        *,
        step_index: int | None = None,
    ) -> str | None:
        badges = self.visible_invocation_badges(func, step_index=step_index)
        if badges:
            return "func=" + " | ".join(badge.text for badge in badges)
        if isinstance(func, tuple) and len(func) >= 1:
            return f"func={self.func_name(func)}"
        if isinstance(func, list) and func:
            func_names = [self.func_name(f) for f in func if f is not None]
            return f"func=[{', '.join(func_names)}]"
        if callable(func):
            func_name = CallableContract.from_callable(func).function_name
            return f"func={func_name}"
        if isinstance(func, dict):
            orchestrator = self.editor._get_current_orchestrator()
            metadata_cache = orchestrator.metadata_cache if orchestrator else None
            step = state.to_object(update_delegate=False) if state else None
            group_by = (
                step.processing_config.group_by
                if isinstance(step, AbstractStep)
                else None
            )
            entries = []
            for key in sorted(func.keys()):
                display_name = None
                if group_by and metadata_cache:
                    display_name = metadata_cache.get_component_metadata(
                        group_by, str(key)
                    )
                if display_name is None:
                    display_name = str(key)
                entries.append(f"{display_name}: {self.func_name(func[key])}")
            return f"func={{{', '.join(entries)}}}"
        return None

    def invocation_badges(
        self,
        func,
        *,
        step_index: int | None = None,
    ) -> tuple[FunctionPatternInvocationBadge, ...]:
        if not func:
            return ()
        normalized = normalize_function_pattern(func)
        cursor = (
            None
            if self.editor.debug_session_state is None
            else self.editor.debug_session_state.cursor
        )
        dirty_cursor = (
            None
            if self.editor.debug_session_state is None
            else self.editor.debug_session_state.dirty_from_cursor
        )
        return tuple(
            FunctionPatternInvocationBadge(
                group_key=item.key.group_key,
                position=item.key.position,
                function_name=item.key.function_name,
                is_current_cursor=(
                    self._cursor_matches_step_index(cursor, step_index)
                    and cursor.matches_invocation_key_parts(
                        group_key=item.key.group_key,
                        position=item.key.position,
                        function_name=item.key.function_name,
                    )
                    if cursor is not None
                    else False
                ),
                is_dirty_replay_start=(
                    self._cursor_matches_step_index(dirty_cursor, step_index)
                    and dirty_cursor.matches_invocation_key_parts(
                        group_key=item.key.group_key,
                        position=item.key.position,
                        function_name=item.key.function_name,
                    )
                    if dirty_cursor is not None
                    else False
                ),
            )
            for item in normalized.iter_items()
        )

    def visible_invocation_badges(
        self,
        func,
        *,
        step_index: int | None = None,
    ) -> tuple[FunctionPatternInvocationBadge, ...]:
        return tuple(
            badge
            for badge in self.invocation_badges(func, step_index=step_index)
            if badge.is_visible
        )

    def badge_provider(
        self,
        step: AbstractStep,
        *,
        step_index: int,
    ) -> Callable[[str, int, Callable], str | None]:
        badges = {
            (badge.group_key, badge.position, badge.function_name): badge
            for badge in self.invocation_badges(step.func, step_index=step_index)
        }

        def badge_text(group_key: str, position: int, func: Callable) -> str | None:
            badge = badges.get((group_key, position, self.func_name(func)))
            return badge.text if badge is not None and badge.is_visible else None

        return badge_text

    @staticmethod
    def _cursor_matches_step_index(cursor: DebugCursor, step_index: int | None) -> bool:
        return step_index is not None and cursor.step_index == step_index

    def func_name(self, func_entry) -> str:
        if isinstance(func_entry, tuple) and len(func_entry) >= 1:
            return CallableContract.from_callable(func_entry[0]).function_name
        if isinstance(func_entry, list) and func_entry:
            first = self.func_name(func_entry[0])
            if len(func_entry) > 1:
                last = self.func_name(func_entry[-1])
                return f"{first}→{last}"
            return first
        if callable(func_entry):
            return CallableContract.from_callable(func_entry).function_name
        return str(func_entry)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PipelineEditorDebugWorkflow:
    """Owns debug-toolbar command dispatch and snapshot inspector routing."""

    editor: Any

    def handle_command(self, command) -> None:
        declaration = PipelineDebugActionDeclarationBase.for_command_type(
            command.command_type
        )
        declaration.dispatch_editor(self.editor)

    def run_command(
        self,
        command_type: DebugCommandType = DebugCommandType.RUN,
    ) -> None:
        if not self.editor.current_plate:
            self.editor.status_message.emit("Select a plate before running debug mode.")
            return
        if self.editor.plate_manager is None:
            self.editor.status_message.emit(
                "Debug run requires a connected Plate Manager."
            )
            return
        command_label = command_type.value.replace("_", " ")
        self.editor.status_message.emit(
            f"Submitting debug {command_label} for {self.editor.current_plate}"
        )
        start_step_index = self.start_step_index(command_type)
        start_after_invocation_key = self.start_after_invocation_key(command_type)
        self.editor.debug_terminal_summary = None
        PipelineEditorCoroutineRunner(self.editor).submit(
            self.editor.plate_manager.action_run_debug_plate(
                self.editor.current_plate,
                command_type=command_type,
                pause_step_indices=self.pause_step_indices(),
                start_step_index=start_step_index,
                start_after_invocation_key=start_after_invocation_key,
            )
        )

    def pause_step_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, step in enumerate(self.editor.pipeline_steps)
            if step.debug_pause
        )

    def start_step_index(self, command_type: DebugCommandType) -> int:
        cursor = self._replay_cursor(command_type)
        if cursor is not None:
            return cursor.step_index
        return 0

    def start_after_invocation_key(
        self,
        command_type: DebugCommandType,
    ) -> str | None:
        cursor = self._replay_cursor(command_type)
        if cursor is None:
            return None
        return cursor.invocation_key

    def _replay_cursor(self, command_type: DebugCommandType) -> DebugCursor | None:
        session = self.editor.debug_session_state
        if (
            command_type is DebugCommandType.RESTART
            and session is not None
            and session.dirty_from_cursor is not None
        ):
            return session.dirty_from_cursor
        if (
            command_type is DebugCommandType.STEP
            and session is not None
            and session.cursor is not None
        ):
            return session.cursor
        if command_type is DebugCommandType.STEP:
            terminal_summary = None
            if self.editor.current_plate and self.editor.plate_manager is not None:
                terminal_summary = (
                    self.editor.plate_manager.debug_terminal_summary_for_plate(
                        self.editor.current_plate
                    )
                )
            if terminal_summary is None:
                terminal_summary = self.editor.debug_terminal_summary
            if terminal_summary is not None:
                return terminal_summary.cursor
        return None

    def stop_command(self) -> None:
        if self.editor.plate_manager is None:
            self.editor.status_message.emit(
                "Debug stop requires a connected Plate Manager."
            )
            return
        self.editor.plate_manager.action_stop_execution(force=True)
        self.editor.status_message.emit("Requested debug execution stop.")

    def show_runtime_inspection(self) -> None:
        if self.editor.plate_manager is None:
            self.editor.status_message.emit(
                "Runtime inspection requires an active debug session."
            )
            return
        session = self.editor.debug_session_context().active_session
        if session is None:
            self.editor.status_message.emit(
                "Runtime inspection requires an active debug session."
            )
            return
        PipelineEditorCoroutineRunner(self.editor).submit(
            self._show_runtime_inspection(session)
        )

    async def _show_runtime_inspection(self, session: DebugSession) -> None:
        view_model = await self.editor.plate_manager.action_inspect_debug_runtime(
            debug_session_id=session.debug_session_id,
        )
        self.editor.service_adapter.ui_dispatcher.post(
            lambda: self._render_runtime_inspection(view_model)
        )

    def _render_runtime_inspection(self, view_model: DebugViewModel) -> None:
        if self.editor.debug_inspector_window is None:
            self.editor.debug_inspector_window = DebugInspectorWindow(self.editor)
            self.editor.debug_inspector_window.artifact_export_requested.connect(
                self.handle_artifact_export_request
            )
            self.editor.debug_inspector_window.artifact_open_requested.connect(
                self.handle_artifact_open_request
            )
        self.editor.debug_inspector_window.set_inspection_view_model(view_model)
        self.editor.debug_inspector_window.show()
        self.editor.debug_inspector_window.raise_()

    def show_snapshot(self, notification) -> None:
        debug_context = notification.debug_context
        snapshot_store_ref = debug_context.snapshot_store_ref
        snapshot_store_backend = debug_context.snapshot_store_backend
        snapshot_id = debug_context.snapshot_id
        if snapshot_store_ref is None or snapshot_id is None:
            self.editor.status_message.emit(
                "Debug snapshot event did not include a snapshot store."
            )
            return

        if self.editor.debug_inspector_window is None:
            self.editor.debug_inspector_window = DebugInspectorWindow(self.editor)
            self.editor.debug_inspector_window.artifact_export_requested.connect(
                self.handle_artifact_export_request
            )
            self.editor.debug_inspector_window.artifact_open_requested.connect(
                self.handle_artifact_open_request
            )
        active_session = None
        if self.editor.plate_manager is not None:
            active_session = self.editor.plate_manager.debug_session_for_plate(
                notification.progress_event.plate_id
            )
        terminal_summary = self.editor.debug_terminal_summary
        loaded_session = DebugSession(
            debug_session_id=debug_context.debug_session_id,
            plate_id=notification.progress_event.plate_id,
            axis_id=notification.progress_event.axis_id,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
        )
        if active_session is not None:
            self.editor.debug_session_state = active_session.with_snapshot_store(
                snapshot_store_ref=snapshot_store_ref,
                snapshot_store_backend=snapshot_store_backend,
                axis_id=notification.progress_event.axis_id,
            ).with_cursor(debug_context.cursor)
            self.editor.debug_terminal_summary = None
        elif (
            terminal_summary is not None
            and terminal_summary.debug_session_id == debug_context.debug_session_id
        ):
            self.editor.debug_session_state = None
        else:
            self.editor.debug_session_state = loaded_session.with_cursor(
                debug_context.cursor
            )
        self.editor.update_item_list()
        self.editor.update_button_states()
        if notification.snapshot is not None:
            self.editor.debug_inspector_window.set_snapshot(notification.snapshot)
            if (
                terminal_summary is not None
                and terminal_summary.debug_session_id == debug_context.debug_session_id
            ):
                self.editor.debug_terminal_summary = terminal_summary.with_snapshot(
                    snapshot=notification.snapshot,
                    snapshot_id=snapshot_id,
                    snapshot_store_ref=snapshot_store_ref,
                    snapshot_store_backend=snapshot_store_backend,
                )
        elif snapshot_store_backend is None:
            snapshot = self.editor.debug_inspector_window.load_snapshot(
                root_path=snapshot_store_ref,
                debug_session_id=debug_context.debug_session_id,
                snapshot_id=snapshot_id,
            )
            if (
                terminal_summary is not None
                and terminal_summary.debug_session_id == debug_context.debug_session_id
            ):
                self.editor.debug_terminal_summary = terminal_summary.with_snapshot(
                    snapshot=snapshot,
                    snapshot_id=snapshot_id,
                    snapshot_store_ref=snapshot_store_ref,
                    snapshot_store_backend=snapshot_store_backend,
                )
        else:
            snapshot = self.editor.debug_inspector_window.load_snapshot_from_store(
                store=FileManagerDebugSnapshotStore(
                    filemanager=self.editor.service_adapter.get_file_manager(),
                    backend=snapshot_store_backend,
                    root_path=snapshot_store_ref,
                    debug_session_id=debug_context.debug_session_id,
                ),
                snapshot_id=snapshot_id,
            )
            if (
                terminal_summary is not None
                and terminal_summary.debug_session_id == debug_context.debug_session_id
            ):
                self.editor.debug_terminal_summary = terminal_summary.with_snapshot(
                    snapshot=snapshot,
                    snapshot_id=snapshot_id,
                    snapshot_store_ref=snapshot_store_ref,
                    snapshot_store_backend=snapshot_store_backend,
                )
        self.editor.debug_inspector_window.show()
        self.editor.debug_inspector_window.raise_()
        self.editor.status_message.emit(
            f"Loaded debug snapshot {snapshot_id} for {notification.progress_event.step_name}"
        )

    def handle_artifact_open_request(self, request) -> None:
        self.editor.status_message.emit(
            "Debug artifact viewer request queued for "
            f"{request.viewer_type}: {request.artifact_ref.name}"
        )

    def handle_artifact_export_request(self, request) -> None:
        if self.editor.plate_manager is None or self.editor.debug_session_state is None:
            self.editor.status_message.emit(
                "Debug artifact export requires an active debug session."
            )
            return
        export_root = QFileDialog.getExistingDirectory(
            self.editor,
            "Export Debug Artifact",
            str(Path.home()),
        )
        if not export_root:
            return
        task = self.editor.plate_manager.action_export_debug_artifact(
            debug_session_id=self.editor.debug_session_state.debug_session_id,
            artifact_ref=request.artifact_ref,
            export_root=export_root,
            snapshot_store_ref=self.editor.debug_session_state.snapshot_store_ref,
            snapshot_store_backend=self.editor.debug_session_state.snapshot_store_backend,
        )
        PipelineEditorCoroutineRunner(self.editor).submit(task)


@dataclass(frozen=True, slots=True)
class PipelineEditorCodeWorkflow(ManagerCodeExecutionWorkflow):
    """Applies edited pipeline-step code to pipeline editor state."""

    workflow_key = "pipeline_editor"
    editor: Any

    def migration_namespace(self, code: str, error: Exception) -> dict | None:
        error_msg = str(error)
        if "unexpected keyword argument" not in error_msg:
            return None
        if "group_by" not in error_msg and "variable_components" not in error_msg:
            return None

        logger.info(
            "Detected old-format step constructor, retrying with migration patch: %s",
            error,
        )
        namespace: dict[str, Any] = {}
        with (
            patch_lazy_constructors(),
            patch_step_constructors_for_migration(),
        ):
            exec(code, namespace)
        return namespace

    def apply_namespace(self, namespace: dict) -> bool:
        if not self.validate_namespace(namespace):
            return False

        document = PipelineDocumentAuthority.from_namespace(namespace)
        self.editor.require_pipeline_definition_mutation_allowed(
            self.editor.current_plate
        )
        pipeline_steps = document.pipeline_steps
        self.editor.pipeline_steps = pipeline_steps
        self.editor._normalize_step_scope_tokens(register=False)

        if self.editor.current_plate:
            if self.editor.plate_manager is not None:
                from openhcs.pyqt_gui.widgets.shared.services.plate_manager_workflows import (
                    PlateManagerCodeWorkflow,
                )

                PlateManagerCodeWorkflow(
                    self.editor.plate_manager
                ).apply_per_plate_configs(
                    {self.editor.current_plate: document.pipeline_config}
                )
            self.editor.update_pipeline_for_plate(
                self.editor.current_plate,
                self.editor.pipeline_steps,
            )
            PipelineObjectStateBinding.commit_plate_state(
                self.editor.current_plate,
            )
            self.editor.notify_pipeline_definition_changed(self.editor.current_plate)
            logger.debug(
                "Updated Pipeline ObjectState (%d steps) for plate: %s",
                len(self.editor.pipeline_steps),
                self.editor.current_plate,
            )

        self.editor.update_item_list()
        self.editor._suppress_pipeline_state_sync = True
        try:
            self.editor.pipeline_changed.emit(self.editor.pipeline_steps)
        finally:
            self.editor._suppress_pipeline_state_sync = False
        self.editor.status_message.emit(
            f"Pipeline updated with {len(pipeline_steps)} steps"
        )
        GuiEventBusBroadcaster(self.editor.event_bus).pipeline_changed(pipeline_steps)
        return True

    def validate_namespace(self, namespace: dict) -> bool:
        try:
            PipelineDocumentAuthority.from_namespace(namespace)
        except (TypeError, ValueError):
            return False
        return True


@dataclass(frozen=True, slots=True)
class PipelineEditorDeletionWorkflow(ManagerDeletionWorkflow):
    """Deletes pipeline steps and updates backing ObjectState atomically."""

    workflow_key = "pipeline_editor"
    editor: Any

    def validate(self, items: list[Any]) -> bool:
        del items
        return True

    def delete(self, items: list[Any]) -> None:
        self.editor.require_pipeline_definition_mutation_allowed(
            self.editor.current_plate
        )
        step_names = [step.name for step in items]
        label = f"delete step{'s' if len(items) > 1 else ''} {', '.join(step_names)}"

        with ObjectStateRegistry.atomic(label):
            for step in items:
                self.unregister_step_state(step)

            deleted_step_ids = {id(step) for step in items}
            self.editor.pipeline_steps = [
                step
                for step in self.editor.pipeline_steps
                if id(step) not in deleted_step_ids
            ]
            self.editor._normalize_step_scope_tokens(register=False)

            if self.editor.current_plate:
                self.editor.update_pipeline_for_plate(
                    self.editor.current_plate,
                    self.editor.pipeline_steps,
                )

        if self.editor.selected_step in [step.name for step in items]:
            self.editor.selected_step = ""

    def unregister_step_state(self, step: Any) -> None:
        scope_id = self.editor._build_step_scope_id(step)
        count = ObjectStateRegistry.unregister_scope_and_descendants(scope_id)
        logger.debug(
            "Cascade unregistered %d ObjectState(s) for deleted step: %s",
            count,
            scope_id,
        )


@dataclass(frozen=True, slots=True)
class PipelineEditorListWorkflow:
    """Owns pipeline editor list refresh side effects."""

    editor: Any

    def prepare_update(self) -> None:
        self.editor._normalize_step_scope_tokens(register=False)

    def post_reorder(self) -> None:
        self.editor.require_pipeline_definition_mutation_allowed(
            self.editor.current_plate
        )
        self.editor._normalize_step_scope_tokens(register=False)
        if self.editor.current_plate:
            self.editor.update_pipeline_for_plate(
                self.editor.current_plate,
                self.editor.pipeline_steps,
            )
        self.editor.pipeline_changed.emit(self.editor.pipeline_steps)
        GuiEventBusBroadcaster(self.editor.event_bus).pipeline_changed(
            self.editor.pipeline_steps
        )
        ObjectStateRegistry.record_snapshot(
            "reorder steps",
            scope_id=str(self.editor.current_plate),
        )

    def restore_after_time_travel(
        self,
        dirty_states: TimeTravelDirtyStates | None = None,
        triggering_scope: str | None = None,
    ) -> None:
        from objectstate.time_travel_profile import TimeTravelProfiler

        del triggering_scope

        with TimeTravelProfiler.phase(
            "openhcs.pipeline_editor.restore_after_time_travel"
        ):
            with TimeTravelProfiler.phase("openhcs.pipeline_editor.load_steps"):
                if self.editor.current_plate:
                    self.editor.pipeline_steps = (
                        self.editor._get_steps_from_pipeline_state(
                            self.editor.current_plate
                        )
                    )
                else:
                    self.editor.pipeline_steps = []

            with TimeTravelProfiler.phase("openhcs.pipeline_editor.normalize_tokens"):
                self.editor._normalize_step_scope_tokens(register=False)
            with TimeTravelProfiler.phase("openhcs.pipeline_editor.update_item_list"):
                self.editor.update_item_list()
            with TimeTravelProfiler.phase(
                "openhcs.pipeline_editor.update_button_states"
            ):
                self.editor.update_button_states()
            if not self._changed_pipeline_structure(dirty_states):
                return

            with TimeTravelProfiler.phase(
                "openhcs.pipeline_editor.broadcast_pipeline_changed"
            ):
                self.editor._suppress_pipeline_state_sync = True
                try:
                    self.editor.pipeline_changed.emit(self.editor.pipeline_steps)
                finally:
                    self.editor._suppress_pipeline_state_sync = False
                GuiEventBusBroadcaster(self.editor.event_bus).pipeline_changed(
                    self.editor.pipeline_steps
                )

    def _changed_pipeline_structure(
        self,
        dirty_states: TimeTravelDirtyStates | None,
    ) -> bool:
        from openhcs.ui.shared.plate_scope_identity import PipelineScopeIdentity

        if not self.editor.current_plate:
            return False
        if dirty_states is None:
            return False

        pipeline_scope = PipelineScopeIdentity.from_plate_scope(
            self.editor.current_plate
        ).scope_id
        for scope_id, _state in dirty_states:
            if scope_id == pipeline_scope:
                return True
        return False


@dataclass(frozen=True, slots=True)
class PipelineStepSaveWorkflow:
    """Updates one edited step while preserving scope-token continuity."""

    editor: PipelineStepSaveEditor
    step_to_edit: AbstractStep
    plate_scope: str

    def save(self, edited_step: AbstractStep) -> None:
        self.editor.require_pipeline_definition_mutation_allowed(self.plate_scope)
        for index, step in enumerate(self.editor.pipeline_steps):
            if not self._matches_step_to_edit(step):
                continue
            ScopeTokenService.transfer_token(
                self.plate_scope,
                self.step_to_edit,
                edited_step,
            )
            self.editor.pipeline_steps[index] = edited_step
            break

        self.editor.update_item_list()
        self.editor.pipeline_changed.emit(self.editor.pipeline_steps)
        self.editor.status_message.emit(f"Updated step: {edited_step.name}")

    def _matches_step_to_edit(self, step: AbstractStep) -> bool:
        if step is self.step_to_edit:
            return True
        return ScopeTokenService.same_object_token(step, self.step_to_edit)
