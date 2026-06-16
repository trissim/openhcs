from __future__ import annotations

import re

from PyQt6.QtWidgets import QApplication
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.services.pattern_data_manager import (
    FUNC_EDITOR_PATTERN_TOKENS_META_KEY,
)
from pyqt_reactive.services.scope_token_service import ScopeTokenService

from openhcs.constants import GroupBy
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyProcessingConfig,
    LazyStepWellFilterConfig,
    PipelineConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.step_scope_identity import StepEditorScope
from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow
from openhcs.pyqt_gui.services.service_adapter import GlobalEventBus
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.pyqt_gui.widgets.shared.services.pipeline_editor_workflows import (
    PipelineEditorListWorkflow,
)


TEST_PLATE_SCOPE = "plate"


class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


class PipelineEditorServiceStub:
    """Minimal service adapter surface needed by PipelineEditorWidget construction."""

    def __init__(self) -> None:
        self.global_config = GlobalPipelineConfig()
        self.color_scheme = ColorScheme()
        self.event_bus = GlobalEventBus()

    def get_global_config(self) -> GlobalPipelineConfig:
        return self.global_config

    def get_current_color_scheme(self) -> ColorScheme:
        return self.color_scheme

    def get_event_bus(self) -> GlobalEventBus:
        return self.event_bus

    def get_file_manager(self):
        return None


class SignalRecorder:
    """Signal-like recorder for workflow unit tests."""

    def __init__(self) -> None:
        self.emissions = []

    def emit(self, value):
        self.emissions.append(value)


class EventBusRecorder:
    """Minimal event bus surface used by GuiEventBusBroadcaster."""

    def __init__(self) -> None:
        self.pipeline_emissions = []

    def emit_pipeline_changed(self, pipeline_steps):
        self.pipeline_emissions.append(pipeline_steps)


def test_pipeline_editor_constructor_connects_debug_toolbar_signal() -> None:
    QtApplicationHarness.app()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())

    assert widget.debug_toolbar is not None
    widget.close()


def test_pipeline_editor_clear_selection_does_not_require_plate_scope() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())

    try:
        widget.set_current_plate("")
        widget.update_item_list()

        assert widget.current_plate == ""
        assert widget.pipeline_steps == []
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_pipeline_editor_time_travel_restore_broadcasts_restored_pipeline() -> None:
    restored_steps = [FunctionStep(name="First"), FunctionStep(name="Second")]

    class Editor:
        current_plate = TEST_PLATE_SCOPE

        def __init__(self) -> None:
            self.pipeline_steps = []
            self.pipeline_changed = SignalRecorder()
            self.event_bus = EventBusRecorder()
            self.normalized = False
            self.list_updated = False
            self.buttons_updated = False

        def _get_steps_from_pipeline_state(self, plate_path):
            assert plate_path == self.current_plate
            return restored_steps

        def _normalize_step_scope_tokens(self, *, register):
            assert register is False
            self.normalized = True

        def update_item_list(self):
            self.list_updated = True

        def update_button_states(self):
            self.buttons_updated = True

    editor = Editor()

    PipelineEditorListWorkflow(editor).restore_after_time_travel()

    assert editor.pipeline_steps == restored_steps
    assert editor.normalized is True
    assert editor.list_updated is True
    assert editor.buttons_updated is True
    assert editor.pipeline_changed.emissions == [restored_steps]
    assert editor.event_bus.pipeline_emissions == [restored_steps]


def test_pipeline_editor_step_display_is_numbered_without_renaming_step() -> None:
    QtApplicationHarness.app()

    step_name = "Measure"
    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    steps = [
        FunctionStep(name=step_name),
        FunctionStep(name=step_name),
    ]
    widget.pipeline_steps = steps

    first_display = widget._format_item_content(steps[0], 0, None)
    second_display = widget._format_item_content(steps[1], 1, None)
    _, semantic_name = widget.format_item_for_display(steps[1], step_index=1)

    assert first_display.layout.name.text == f"1. {step_name}"
    assert second_display.layout.name.text == f"2. {step_name}"
    assert semantic_name == step_name
    assert [step.name for step in steps] == [step_name, step_name]
    widget.close()


class RuntimeCallable:
    """Callable object with a non-function scope prefix."""

    def __call__(self, image, threshold: int = 1):
        return image


def test_step_registration_persists_function_editor_scope_tokens() -> None:
    ObjectStateRegistry._states.clear()
    ScopeTokenService.clear_scope("plate::functionstep_0")

    runtime_callable = RuntimeCallable()
    step = FunctionStep(
        func=(runtime_callable, {"threshold": 3}),
        name="Crop",
    )
    editor = PipelineEditorWidget.__new__(PipelineEditorWidget)

    step_state, states = editor._collect_step_registration_states(
        step=step,
        scope_id="plate::functionstep_0",
        parent_state=None,
    )

    assert step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY] == [
        "runtimecallable_0",
    ]
    assert [state.scope_id for state in states] == [
        "plate::functionstep_0",
        "plate::functionstep_0::runtimecallable_0",
    ]


def test_pipeline_update_refreshes_existing_step_scope_state() -> None:
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(TEST_PLATE_SCOPE)

    editor = PipelineEditorWidget.__new__(PipelineEditorWidget)
    original = FunctionStep(
        name="IdentifyPrimaryObjects",
        processing_config=LazyProcessingConfig(group_by=GroupBy.CHANNEL),
    )
    editor.update_pipeline_for_plate(TEST_PLATE_SCOPE, [original])

    replacement = FunctionStep(
        name="IdentifyPrimaryObjects",
        processing_config=LazyProcessingConfig(group_by=GroupBy.NONE),
    )
    replacement._scope_token = original._scope_token

    editor.update_pipeline_for_plate(TEST_PLATE_SCOPE, [replacement])

    resolved = editor.get_pipeline_for_plate(TEST_PLATE_SCOPE)
    assert resolved[0].processing_config.group_by is GroupBy.NONE


def test_dual_editor_step_scope_uses_logical_plate_scope() -> None:
    logical_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/plate",
        "/tmp/plate/Analysis_Final.cppipe",
    ).scope_id
    ScopeTokenService.clear_scope(logical_scope)

    step = FunctionStep(name="Threshold")
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.plate_scope = logical_scope
    window.editing_step = step

    assert window._build_step_scope_id() == f"{logical_scope}::functionstep_0"


def test_step_editor_scope_parse_preserves_cppipe_plate_scope() -> None:
    plate_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/plate",
        "/tmp/plate/Analysis_Final.cppipe",
    ).scope_id
    scope_id = f"{plate_scope}::functionstep_17::cellprofilerruntimecallable_0"

    parsed = StepEditorScope.parse(scope_id)

    assert parsed.plate_scope == plate_scope
    assert parsed.step_scope_id == f"{plate_scope}::functionstep_17"
    assert parsed.step_token.raw == "functionstep_17"
    assert parsed.is_function_scope is True


def test_step_editor_scope_handler_pattern_accepts_runtime_callable_tokens() -> None:
    plate_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/plate",
        "/tmp/plate/Analysis_Final.cppipe",
    ).scope_id
    scope_id = f"{plate_scope}::functionstep_17::runtimecallable_0"

    assert re.match(StepEditorScope.handler_pattern(), scope_id)


def test_step_well_filter_live_resolution_is_visible_in_pipeline_row() -> None:
    """Inherited step well filters should fan out into the visible row preview."""
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(TEST_PLATE_SCOPE)

    root_state = ObjectState(GlobalPipelineConfig(), scope_id="")
    ObjectStateRegistry.register(root_state)
    plate_state = ObjectState(
        PipelineConfig(),
        scope_id=TEST_PLATE_SCOPE,
        parent_state=root_state,
    )
    ObjectStateRegistry.register(plate_state)

    step = FunctionStep(
        name="Threshold",
        step_well_filter_config=LazyStepWellFilterConfig(),
    )
    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    widget.current_plate = TEST_PLATE_SCOPE
    widget.pipeline_steps = [step]
    widget.update_pipeline_for_plate(TEST_PLATE_SCOPE, [step])

    plate_state.update_parameter("well_filter_config.well_filter", "A01")
    ObjectStateRegistry._notify_change()

    styled_text, _ = widget.format_item_for_display(step, step_index=0)
    layout = styled_text.layout
    preview_by_path = {
        segment.field_path: segment.text
        for segment in layout.preview_segments
        if segment.field_path
    }

    assert preview_by_path["step_well_filter_config.well_filter"] == ":A01"
    widget.close()


def test_pipeline_config_scope_is_not_treated_as_current_orchestrator() -> None:
    """Live row refreshes can see pipeline config state before an orchestrator exists."""
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    pipeline_state = ObjectState(PipelineConfig(), scope_id=TEST_PLATE_SCOPE)
    ObjectStateRegistry.register(pipeline_state)

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    widget.current_plate = TEST_PLATE_SCOPE

    assert widget._get_current_orchestrator() is None
    assert widget._is_current_plate_initialized() is False
    widget.update_button_states()
    widget.close()
