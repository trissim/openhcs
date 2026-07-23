from __future__ import annotations

from dataclasses import dataclass, field

from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow
from pyqt_reactive.services.scope_token_service import ScopeTokenService
from pyqt_reactive.widgets.shared.base_form_dialog import BaseFormDialog
from pyqt_reactive.widgets.shared.dirty_window_presenter import DirtyWindowStateTracker


@dataclass
class NestedConfig:
    group_by: str = "site"


@dataclass
class EditableStep:
    name: str = "old"
    func: object = None
    processing_config: NestedConfig = field(default_factory=NestedConfig)


class NonDataclassEditableStep:
    def __init__(self) -> None:
        self.name = "old"
        self.func = None
        self.processing_config = NestedConfig()


class StepState:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.is_raw_dirty = False
        self.signature_diff_fields = set()

    def get_current_values(self):
        return self.parameters


class StepEditor:
    def __init__(self, state: StepState):
        self.state = state
        self.step_index = None
        self.tree_helper = TreeHelper()


class TreeHelper:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup_subscriptions(self) -> None:
        self.cleanup_calls += 1


class FunctionEditor:
    def __init__(self) -> None:
        self.scope_indices = []

    def set_scope_index(self, index: int) -> None:
        self.scope_indices.append(index)


class ArtifactPlanView:
    def __init__(self) -> None:
        self.inspections = []

    def set_inspection(self, inspection):
        self.inspections.append(inspection)


def test_artifact_plan_is_invalidated_when_authored_step_changes() -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.artifact_plan_view = ArtifactPlanView()

    window._invalidate_artifact_plan()

    assert window.artifact_plan_view.inspections == [None]


def test_dual_editor_title_prefixes_current_step_number() -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.step_editor = StepEditor(StepState({"name": "Measure"}))
    window.editing_step = FunctionStep(name="Fallback")
    window.is_new = False
    window._step_index = 2
    window._dirty_window_state = DirtyWindowStateTracker(
        state_provider=lambda: None,
        change_emitter=lambda _: None,
    )

    presentation = window.dirty_window_presentation()

    assert presentation.window_title == "Edit Step: 3. Measure"
    assert presentation.header_text == "Edit Step: 3. Measure"


def test_dual_editor_pipeline_reorder_refreshes_title_number_with_scope_colors() -> None:
    plate_scope = "plate::dual-title-test"
    ScopeTokenService.clear_scope(plate_scope)
    tracked_step = FunctionStep(name="Tracked")
    other_step = FunctionStep(name="Other")
    window_scope_id = ScopeTokenService.build_scope_id(plate_scope, tracked_step)
    ScopeTokenService.build_scope_id(plate_scope, other_step)

    window = DualEditorWindow.__new__(DualEditorWindow)
    window.scope_id = window_scope_id
    window._step_index = 0
    window.step_editor = StepEditor(StepState({"name": "Tracked"}))
    window.func_editor = FunctionEditor()
    window.original_step_reference = tracked_step
    window.artifact_plan_view = None
    title_refreshes = []
    border_refreshes = []
    window._update_window_title = lambda: title_refreshes.append(window._step_index)
    window._refresh_scope_border = lambda: border_refreshes.append(window._step_index)

    window._on_pipeline_changed([other_step, tracked_step])

    assert window._step_index == 1
    assert window.step_editor.step_index == 1
    assert window.func_editor.scope_indices == [1]
    assert border_refreshes == [1]
    assert title_refreshes == [1]


def test_form_parameter_change_accepts_prefixed_nested_paths() -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.editing_step = EditableStep()
    sync_calls = []
    window._schedule_function_editor_sync = lambda: sync_calls.append(True)

    window.on_form_parameter_changed("FunctionStep.processing_config.group_by", "well")

    assert window.editing_step.name == "old"
    assert not hasattr(window.editing_step, "group_by")
    assert sync_calls == [True]


def test_form_parameter_change_accepts_unprefixed_nested_paths() -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.editing_step = EditableStep()
    sync_calls = []
    window._schedule_function_editor_sync = lambda: sync_calls.append(True)

    window.on_form_parameter_changed("processing_config.group_by", "well")

    assert window.editing_step.name == "old"
    assert not hasattr(window.editing_step, "group_by")
    assert sync_calls == [True]


def test_form_parameter_change_accepts_non_dataclass_step_objects() -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.editing_step = NonDataclassEditableStep()
    sync_calls = []
    window._schedule_function_editor_sync = lambda: sync_calls.append(True)

    window.on_form_parameter_changed("FunctionStep.processing_config.group_by", "well")

    assert window.editing_step.name == "old"
    assert not hasattr(window.editing_step, "group_by")
    assert sync_calls == [True]


def test_form_parameter_change_syncs_prefixed_top_level_paths() -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.editing_step = EditableStep()
    sync_calls = []
    window._schedule_function_editor_sync = lambda: sync_calls.append(True)

    window.on_form_parameter_changed("FunctionStep.name", "new")

    assert window.editing_step.name == "old"
    assert sync_calls == [True]


def test_close_event_does_not_require_legacy_dirty_callbacks(monkeypatch) -> None:
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.step_editor = StepEditor(StepState({"name": "Closing"}))
    close_calls = []

    def record_base_close(self, event):
        close_calls.append((self, event))

    monkeypatch.setattr(BaseFormDialog, "closeEvent", record_base_close)

    event = object()
    window.closeEvent(event)

    assert window.step_editor.tree_helper.cleanup_calls == 1
    assert close_calls == [(window, event)]
