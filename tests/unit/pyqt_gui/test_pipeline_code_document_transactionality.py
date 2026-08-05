"""Transactional validation for Pipeline Editor code documents."""

from __future__ import annotations

from objectstate import patch_lazy_constructors
import pytest
from pyqt_reactive.widgets.shared.manager_action_controller import (
    CodeEditorPayload,
    ManagerActionController,
    ManagerActionOperations,
)

from openhcs.constants import GroupBy
from openhcs.core.pipeline_document import PipelineDocument
from openhcs.pyqt_gui.widgets.shared.services.pipeline_editor_workflows import (
    PipelineEditorCodeWorkflow,
)


class _Signal:
    def __init__(self) -> None:
        self.values: list[tuple[object, ...]] = []

    def emit(self, *values: object) -> None:
        self.values.append(values)


class _PipelineEditorHarness:
    def __init__(self) -> None:
        self.current_plate = ""
        self.plate_manager = None
        self.pipeline_steps = []
        self.pipeline_changed = _Signal()
        self.status_message = _Signal()
        self.event_bus = None
        self.item_list_update_count = 0
        self._suppress_pipeline_state_sync = False

    @staticmethod
    def require_pipeline_definition_mutation_allowed(
        plate_path: str | None = None,
    ) -> None:
        del plate_path

    @staticmethod
    def _normalize_step_scope_tokens(*, register: bool) -> None:
        assert register is False

    def update_item_list(self) -> None:
        self.item_list_update_count += 1


def _operations(editor: _PipelineEditorHarness) -> ManagerActionOperations:
    workflow = PipelineEditorCodeWorkflow(editor)
    return ManagerActionOperations(
        widget=editor,
        action_handlers={},
        dynamic_action_handlers={},
        run_async=lambda _operation: None,
        selected_items=list,
        item_name_singular="step",
        item_name_plural="steps",
        show_error=lambda _message: None,
        validate_delete=lambda _items: True,
        perform_delete=lambda _items: None,
        update_item_list=editor.update_item_list,
        emit_items_changed=lambda: None,
        emit_status=lambda _message: None,
        show_item_editor=lambda _item: None,
        validate_code_action=lambda: True,
        code_payload=CodeEditorPayload(
            declaration_type=PipelineDocument,
            missing_error_message=(
                "Pipeline code must define 'pipeline_config' and 'pipeline_steps'."
            ),
        ),
        pre_code_execution=lambda: None,
        patch_lazy_constructors=patch_lazy_constructors,
        migrate_code_namespace=(
            lambda code, error, _namespace: workflow.migration_namespace(code, error)
        ),
        validate_code_namespace=workflow.validate_namespace,
        apply_code_namespace=workflow.apply_namespace,
        post_code_execution=lambda: None,
    )


INVALID_LEGACY_SOURCE = """
from openhcs.core.config import PipelineConfig
from openhcs.core.steps.function_step import FunctionStep

pipeline_config = PipelineConfig()
pipeline_steps = [FunctionStep(func=[], group_by='banana')]
"""

VALID_LEGACY_SOURCE = """
from openhcs.constants import GroupBy
from openhcs.core.config import PipelineConfig
from openhcs.core.steps.function_step import FunctionStep

pipeline_config = PipelineConfig()
pipeline_steps = [FunctionStep(func=[], group_by=GroupBy.CHANNEL)]
"""


def test_invalid_legacy_config_is_rejected_before_pipeline_editor_mutation() -> None:
    editor = _PipelineEditorHarness()
    operations = _operations(editor)
    controller = ManagerActionController()

    with pytest.raises(TypeError, match="LazyProcessingConfig.group_by"):
        controller.validate_edited_code(operations, INVALID_LEGACY_SOURCE)
    with pytest.raises(TypeError, match="LazyProcessingConfig.group_by"):
        controller.apply_edited_code(operations, INVALID_LEGACY_SOURCE)

    assert editor.pipeline_steps == []
    assert editor.item_list_update_count == 0
    assert editor.pipeline_changed.values == []

    controller.validate_edited_code(operations, VALID_LEGACY_SOURCE)
    controller.apply_edited_code(operations, VALID_LEGACY_SOURCE)

    assert len(editor.pipeline_steps) == 1
    assert editor.pipeline_steps[0].processing_config.group_by is GroupBy.CHANNEL
    assert editor.item_list_update_count == 1
    assert len(editor.pipeline_changed.values) == 1
