"""Workflow services owned by the pipeline editor widget."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.utils.pipeline_migration import patch_step_constructors_for_migration
from pyqt_reactive.services.scope_token_service import ScopeTokenService


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineEditorCodeWorkflow:
    """Applies edited pipeline-step code to pipeline editor state."""

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
            self.editor._patch_lazy_constructors(),
            patch_step_constructors_for_migration(),
        ):
            exec(code, namespace)
        return namespace

    def apply_namespace(self, namespace: dict) -> bool:
        if "pipeline_steps" not in namespace:
            return False

        pipeline_steps = namespace["pipeline_steps"]
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
        self.editor._broadcast_to_event_bus("pipeline", pipeline_steps)
        return True


@dataclass(frozen=True, slots=True)
class PipelineEditorDeletionWorkflow:
    """Deletes pipeline steps and updates backing ObjectState atomically."""

    editor: Any

    def delete(self, items: list[Any]) -> None:
        step_names = [step.name for step in items]
        label = f"delete step{'s' if len(items) > 1 else ''} {', '.join(step_names)}"

        with ObjectStateRegistry.atomic(label):
            for step in items:
                self.editor._unregister_step_state(step)

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
        self.editor._broadcast_to_event_bus("pipeline", self.editor.pipeline_steps)
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
