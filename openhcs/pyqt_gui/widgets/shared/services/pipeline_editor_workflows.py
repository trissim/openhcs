"""Workflow services owned by the pipeline editor widget."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable

from objectstate import patch_lazy_constructors, spawn_thread_with_context
from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.core.callable_contract import CallableContract
from openhcs.core.debug import DebugCommandType, DebugSession, FileManagerDebugSnapshotStore
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
    CellProfilerPipelineRuntimeRebinder,
)
from openhcs.pyqt_gui.windows.debug_inspector_window import DebugInspectorWindow
from openhcs.utils.pipeline_migration import patch_step_constructors_for_migration
from PyQt6.QtWidgets import QFileDialog
from pyqt_reactive.services.scope_token_service import ScopeTokenService
from pyqt_reactive.widgets.shared.manager_workflows import (
    ManagerCodeExecutionWorkflow,
    ManagerDeletionWorkflow,
)
from openhcs.pyqt_gui.widgets.shared.services.gui_event_bus_broadcast import (
    GuiEventBusBroadcaster,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineEditorCoroutineRunner:
    """Run async GUI workflow commands from Qt slots without requiring qasync."""

    editor: Any

    def submit(self, coroutine: Coroutine[Any, Any, None]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._submit_background(coroutine)
            return
        loop.create_task(self._guard(coroutine))

    def _submit_background(self, coroutine: Coroutine[Any, Any, None]) -> None:
        spawn_thread_with_context(
            lambda: asyncio.run(self._guard(coroutine)),
            name="pipeline-editor-async-command",
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

    def format_func_preview(self, func, state=None) -> str | None:
        badges = self.visible_invocation_badges(func)
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
            group_by = (
                state.get_resolved_value("processing_config.group_by")
                if state
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
                    cursor.matches_invocation_key_parts(
                        group_key=item.key.group_key,
                        position=item.key.position,
                        function_name=item.key.function_name,
                    )
                    if cursor is not None
                    else False
                ),
                is_dirty_replay_start=(
                    dirty_cursor.matches_invocation_key_parts(
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
    ) -> tuple[FunctionPatternInvocationBadge, ...]:
        return tuple(badge for badge in self.invocation_badges(func) if badge.is_visible)

    def badge_provider(self, step: Any) -> Callable[[str, int, Callable], str | None]:
        badges = {
            (badge.group_key, badge.position, badge.function_name): badge
            for badge in self.invocation_badges(step.func)
        }

        def badge_text(group_key: str, position: int, func: Callable) -> str | None:
            badge = badges.get((group_key, position, self.func_name(func)))
            return badge.text if badge is not None and badge.is_visible else None

        return badge_text

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

    def format_input_source_preview(self, input_source) -> str | None:
        source_name = input_source.name
        if source_name != "PREVIOUS_STEP":
            return f"input={source_name}"
        return None


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PipelineEditorDebugWorkflow:
    """Owns debug-toolbar command dispatch and snapshot inspector routing."""

    editor: Any

    def handle_command(self, command) -> None:
        route = self.editor.DEBUG_COMMAND_ROUTES.get(command.command_type)
        if route is None:
            raise RuntimeError(
                f"Unhandled debug command route: {command.command_type.value}"
            )
        route.dispatch(self.editor)

    def run_command(
        self,
        command_type: DebugCommandType = DebugCommandType.RUN,
    ) -> None:
        if not self.editor.current_plate:
            self.editor.status_message.emit("Select a plate before running debug mode.")
            return
        if self.editor.plate_manager is None:
            self.editor.status_message.emit("Debug run requires a connected Plate Manager.")
            return
        command_label = command_type.value.replace("_", " ")
        self.editor.status_message.emit(
            f"Submitting debug {command_label} for {self.editor.current_plate}"
        )
        PipelineEditorCoroutineRunner(self.editor).submit(
            self.editor.plate_manager.action_run_debug_plate(
                self.editor.current_plate,
                command_type=command_type,
                pause_step_indices=self.pause_step_indices(),
                start_step_index=self.start_step_index(command_type),
                start_after_invocation_key=self.start_after_invocation_key(
                    command_type
                ),
            )
        )

    def pause_step_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, step in enumerate(self.editor.pipeline_steps)
            if step.debug_pause
        )

    def start_step_index(self, command_type: DebugCommandType) -> int:
        session = self.editor.debug_session_state
        if (
            command_type is DebugCommandType.RESTART
            and session is not None
            and session.dirty_from_cursor is not None
        ):
            return session.dirty_from_cursor.step_index
        if (
            command_type is DebugCommandType.STEP
            and session is not None
            and session.cursor is not None
        ):
            return session.cursor.step_index
        return 0

    def start_after_invocation_key(
        self,
        command_type: DebugCommandType,
    ) -> str | None:
        session = self.editor.debug_session_state
        if (
            command_type is DebugCommandType.STEP
            and session is not None
            and session.cursor is not None
        ):
            return session.cursor.invocation_key
        return None

    def stop_command(self) -> None:
        if self.editor.plate_manager is None:
            self.editor.status_message.emit("Debug stop requires a connected Plate Manager.")
            return
        self.editor.plate_manager.action_stop_execution()
        self.editor.status_message.emit("Requested debug execution stop.")

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
        self.editor.debug_session_state = DebugSession(
            debug_session_id=debug_context.debug_session_id,
            plate_id=notification.progress_event.plate_id,
            axis_id=notification.progress_event.axis_id,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
        ).with_cursor(debug_context.cursor)
        if notification.snapshot is not None:
            self.editor.debug_inspector_window.set_snapshot(notification.snapshot)
        elif snapshot_store_backend is None:
            self.editor.debug_inspector_window.load_snapshot(
                root_path=snapshot_store_ref,
                debug_session_id=debug_context.debug_session_id,
                snapshot_id=snapshot_id,
            )
        else:
            self.editor.debug_inspector_window.load_snapshot_from_store(
                store=FileManagerDebugSnapshotStore(
                    filemanager=self.editor.service_adapter.get_file_manager(),
                    backend=snapshot_store_backend,
                    root_path=snapshot_store_ref,
                    debug_session_id=debug_context.debug_session_id,
                ),
                snapshot_id=snapshot_id,
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
        if "pipeline_steps" not in namespace:
            return False

        pipeline_steps = namespace["pipeline_steps"]
        import_result = self.editor.cellprofiler_import_result_for_current_plate()
        if import_result is not None:
            pipeline_steps = CellProfilerPipelineRuntimeRebinder.from_import_result(
                import_result,
            ).rebind(pipeline_steps)
        self.editor.pipeline_steps = pipeline_steps
        self.editor._normalize_step_scope_tokens(register=False)

        if self.editor.current_plate:
            self.editor.update_pipeline_for_plate(
                self.editor.current_plate,
                self.editor.pipeline_steps,
            )
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


@dataclass(frozen=True, slots=True)
class PipelineEditorDeletionWorkflow(ManagerDeletionWorkflow):
    """Deletes pipeline steps and updates backing ObjectState atomically."""

    workflow_key = "pipeline_editor"
    editor: Any

    def validate(self, items: list[Any]) -> bool:
        del items
        return True

    def delete(self, items: list[Any]) -> None:
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

    def restore_after_time_travel(self) -> None:
        if self.editor.current_plate:
            self.editor.pipeline_steps = self.editor._get_steps_from_pipeline_state(
                self.editor.current_plate
            )
        else:
            self.editor.pipeline_steps = []

        self.editor._normalize_step_scope_tokens(register=False)
        self.editor.update_item_list()
        self.editor.update_button_states()


@dataclass(frozen=True, slots=True)
class PipelineStepSaveWorkflow:
    """Updates one edited step while preserving scope-token continuity."""

    editor: Any
    step_to_edit: Any
    plate_scope: str

    def save(self, edited_step: Any) -> None:
        for index, step in enumerate(self.editor.pipeline_steps):
            if step is not self.step_to_edit:
                continue
            prefix = ScopeTokenService._get_prefix(self.step_to_edit)
            ScopeTokenService.get_generator(self.plate_scope, prefix).transfer(
                self.step_to_edit,
                edited_step,
            )
            self.editor.pipeline_steps[index] = edited_step
            break

        self.editor.update_item_list()
        self.editor.pipeline_changed.emit(self.editor.pipeline_steps)
        self.editor.status_message.emit(f"Updated step: {edited_step.name}")
