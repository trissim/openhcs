from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import re

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.services.pattern_data_manager import (
    FUNC_EDITOR_PATTERN_TOKENS_META_KEY,
)
from pyqt_reactive.services.function_navigation import (
    build_function_token_field_path,
)
from pyqt_reactive.forms.parameter_form_manager import (
    FormManagerConfig,
    ParameterFormManager,
)
from pyqt_reactive.services.function_pattern_code_document import (
    FunctionPatternCodeDocumentService,
)
from pyqt_reactive.services.scope_token_service import ScopeTokenService

from openhcs.constants import GroupBy
from openhcs.constants.constants import OrchestratorState
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyProcessingConfig,
    LazyStepWellFilterConfig,
    PipelineConfig,
)
from openhcs.core.debug import DebugCommandType, DebugTerminalSummary
from openhcs.core.execution_state import ManagerExecutionState
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.core.pipeline.function_contracts import artifact_inputs
from objectstate.object_state import ObjectState, ObjectStateRegistry
from openhcs.ui.shared.plate_scope_identity import (
    PipelineScopeIdentity,
    PlateScopeIdentity,
)
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater
from openhcs.pyqt_gui.services.pipeline_object_state_binding import (
    PipelineEditorStateRoot,
    PipelineObjectStateBinding,
)
from openhcs.pyqt_gui.services.step_scope_identity import StepEditorScope
from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow
from openhcs.pyqt_gui.services.service_adapter import GlobalEventBus
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
    PipelineDebugPauseBoundaryState,
    PipelineDebugSessionContext,
    PipelineDebugTargetState,
)
from openhcs.pyqt_gui.widgets.shared.services.pipeline_editor_workflows import (
    PipelineEditorListWorkflow,
)
from openhcs.processing.backends.processors.numpy_processor import (
    stack_percentile_normalize,
)
from openhcs.processing.backends.cellprofiler import correct_illumination_apply
from openhcs.processing.backends.cellprofiler.illumination import (
    IlluminationCorrectionMethod,
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

    def execute_async_operation(self, async_func, *args, **kwargs):
        return async_func(*args, **kwargs)

    def show_error_dialog(self, error_message: str, title: str = "Error") -> None:
        del error_message, title


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


class PlateTerminalStatusRecorder:
    """Minimal terminal-status surface read by PipelineEditor debug projection."""

    def __init__(self) -> None:
        self.terminal_status_by_plate: dict[str, object] = {}


class PlateManagerDefinitionChangeRecorder:
    """Minimal plate-manager surface for pipeline invalidation notifications."""

    def __init__(self) -> None:
        self.changed_plates: list[str] = []
        self.plate_configs: dict[str, PipelineConfig] = {}
        self.event_bus = None
        self.plate_compiled_data: dict[str, object] = {}
        self.plate_terminal_activity_status = PlateTerminalStatusRecorder()
        self.execution_state = ManagerExecutionState.IDLE

    def notify_pipeline_definition_changed(self, plate_path: str) -> None:
        self.changed_plates.append(plate_path)

    def require_pipeline_definition_mutation_allowed(
        self,
        plate_path: str | None = None,
    ) -> None:
        del plate_path
        if self.execution_state is not ManagerExecutionState.IDLE:
            raise RuntimeError(
                "Pipeline definitions cannot change while plate execution is active."
            )

    def authored_pipeline_config_for_code_document(
        self,
        plate_path: str,
    ) -> PipelineConfig:
        return self.plate_configs.get(plate_path, PipelineConfig())

    def debug_session_context_for_plate(
        self,
        plate_path: str,
    ) -> PipelineDebugSessionContext:
        target = PipelineDebugTargetState(
            current_plate_scope_id=plate_path,
            pipeline_scope_id=f"{plate_path}::pipeline",
            initialized=True,
            compiled=plate_path in self.plate_compiled_data,
            terminal_status=None,
        )
        return PipelineDebugSessionContext(
            target=target,
            session=None,
            terminal_summary=None,
            pause_boundaries=PipelineDebugPauseBoundaryState(),
            manager_execution_state=self.execution_state,
        )

    def debug_terminal_summary_for_plate(self, plate_path: str):
        del plate_path
        return None


class PlateManagerCompiledStateRecorder:
    """Minimal plate-manager compiled-state authority for editor tests."""

    def __init__(self) -> None:
        self.plate_compiled_data: dict[str, object] = {}
        self.plate_terminal_activity_status = PlateTerminalStatusRecorder()
        self.execution_state = ManagerExecutionState.IDLE

    def debug_session_context_for_plate(
        self,
        plate_path: str,
    ) -> PipelineDebugSessionContext:
        target = PipelineDebugTargetState(
            current_plate_scope_id=plate_path,
            pipeline_scope_id=PlateScopeIdentity.from_plate_root(
                plate_path,
            ).scope_id
            + "::pipeline",
            initialized=True,
            compiled=plate_path in self.plate_compiled_data,
            terminal_status=None,
        )
        return PipelineDebugSessionContext(
            target=target,
            session=None,
            terminal_summary=None,
            pause_boundaries=PipelineDebugPauseBoundaryState(),
            manager_execution_state=self.execution_state,
        )

    def debug_terminal_summary_for_plate(self, plate_path: str):
        del plate_path
        return None

    def require_pipeline_definition_mutation_allowed(
        self,
        plate_path: str | None = None,
    ) -> None:
        del plate_path
        if self.execution_state is not ManagerExecutionState.IDLE:
            raise RuntimeError(
                "Pipeline definitions cannot change while plate execution is active."
            )


def test_pipeline_editor_constructor_connects_debug_toolbar_signal() -> None:
    QtApplicationHarness.app()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())

    assert widget.debug_toolbar is not None
    widget.close()


def test_standard_execution_state_retires_local_debug_summary() -> None:
    QtApplicationHarness.app()
    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    widget.current_plate = TEST_PLATE_SCOPE
    widget.debug_terminal_summary = DebugTerminalSummary(
        debug_session_id="debug-1",
        plate_id=TEST_PLATE_SCOPE,
        terminal_status="failed",
    )

    try:
        widget.on_orchestrator_state_changed(
            TEST_PLATE_SCOPE,
            OrchestratorState.EXECUTING,
        )

        assert widget.debug_terminal_summary is None
    finally:
        widget.close()


def test_drag_reorder_uses_transport_safe_row_identity() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    plate_manager = PlateManagerDefinitionChangeRecorder()
    widget.current_plate = TEST_PLATE_SCOPE
    widget.plate_manager = plate_manager

    def locally_declared_function(image):
        return image

    steps = [
        FunctionStep(func=locally_declared_function, name="Local"),
        FunctionStep(func=stack_percentile_normalize, name="Normalize"),
    ]
    widget.pipeline_steps = steps
    widget.update_pipeline_for_plate(TEST_PLATE_SCOPE, steps)
    widget._get_list_placeholder = lambda: None
    widget.update_item_list()
    source_item = widget.item_list.item(0)
    source_token = ScopeTokenService.object_token(steps[0])
    target_token = ScopeTokenService.object_token(steps[1])

    try:
        assert source_token is not None
        assert target_token is not None
        assert source_item.data(Qt.ItemDataRole.UserRole) == source_token
        assert widget.item_list.mimeData([source_item]) is not None

        moved_item = widget.item_list.takeItem(0)
        widget.item_list.insertItem(1, moved_item)
        widget._on_items_reordered(0, 1)

        assert [step.name for step in widget.pipeline_steps] == ["Normalize", "Local"]
        assert [
            widget.item_list.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(widget.item_list.count())
        ] == [target_token, source_token]
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_pipeline_editor_code_document_driver_reads_validates_and_applies() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    widget.pipeline_steps = [FunctionStep(name="Original")]
    driver = widget.code_document_driver()

    try:
        assert driver is not None
        document = driver.read_document(clean=True)

        assert document.title == "Edit Pipeline"
        assert "pipeline_config" in document.source
        assert "pipeline_steps" in document.source
        assert "Original" in document.source
        driver.validate_source(
            PipelineDocumentAuthority.render(
                PipelineDocumentAuthority.from_values(
                    pipeline_config=PipelineConfig(),
                    pipeline_steps=[FunctionStep(name="Applied")],
                )
            )
        )
        with pytest.raises(SyntaxError):
            driver.validate_source("pipeline_steps = [\n")
        with pytest.raises(ValueError, match="pipeline_steps"):
            driver.validate_source("not_pipeline_steps = []\n")
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_function_pattern_form_exposes_explicit_kwargs_outside_callable_signature() -> (
    None
):
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    step = FunctionStep(
        func=(
            correct_illumination_apply,
            {
                "name_the_output_image": "CorrectedStain1",
                "truncate_low": True,
                "truncate_high": True,
                "method": IlluminationCorrectionMethod.DIVIDE,
                "select_the_illumination_function": "IllumStain1",
                "select_the_input_image": "OrigStain1",
                "enabled": True,
            },
        ),
        name="CorrectIlluminationApply",
    )

    try:
        PipelineObjectStateBinding.update_plate_steps("plate", [step])
        editor_state = PipelineObjectStateBinding.editor_state_for_plate("plate")
        step_state = ObjectStateRegistry.get_by_scope(editor_state.step_scope_ids[0])
        assert step_state is not None
        [function_token] = step_state.metadata[
            FUNC_EDITOR_PATTERN_TOKENS_META_KEY
        ]
        child_state = ObjectStateRegistry.get_by_scope(
            f"{step_state.scope_id}::{function_token}"
        )
        assert child_state is not None
        manager = ParameterFormManager(
            child_state,
            FormManagerConfig(color_scheme=ColorScheme()),
        )

        expected = {
            "method",
            "truncate_low",
            "truncate_high",
            "enabled",
            "select_the_input_image",
            "select_the_illumination_function",
            "name_the_output_image",
        }
        assert expected <= set(manager.parameters)
        assert expected <= set(manager.parameter_types)
        entry = FunctionPatternCodeDocumentService().child_scope_entry(
            child_state.scope_id
        )
        assert entry.func is correct_illumination_apply
        assert entry.kwargs["name_the_output_image"] == "CorrectedStain1"
    finally:
        ObjectStateRegistry.clear()


def test_pipeline_editor_code_document_driver_apply_mutates_pipeline() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    driver = widget.code_document_driver()

    try:
        assert driver is not None
        driver.apply_source(
            PipelineDocumentAuthority.render(
                PipelineDocumentAuthority.from_values(
                    pipeline_config=PipelineConfig(),
                    pipeline_steps=[FunctionStep(name="Applied")],
                )
            )
        )

        assert [step.name for step in widget.pipeline_steps] == ["Applied"]
        assert widget.item_list.count() == 1
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_pipeline_editor_code_document_apply_notifies_plate_manager() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    plate_manager = PlateManagerDefinitionChangeRecorder()
    widget.current_plate = TEST_PLATE_SCOPE
    widget.plate_manager = plate_manager
    driver = widget.code_document_driver()

    try:
        assert driver is not None
        driver.apply_source(
            PipelineDocumentAuthority.render(
                PipelineDocumentAuthority.from_values(
                    pipeline_config=PipelineConfig(),
                    pipeline_steps=[FunctionStep(name="Replacement")],
                )
            )
        )

        assert [step.name for step in widget.pipeline_steps] == ["Replacement"]
        assert plate_manager.changed_plates == [TEST_PLATE_SCOPE]
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_pipeline_editor_code_document_commits_reconciled_step_tree() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    initial_step = FunctionStep(
        name="Normalize before",
        func=(stack_percentile_normalize, {"low_percentile": 0.5}),
    )
    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    widget.current_plate = TEST_PLATE_SCOPE
    widget.plate_manager = PlateManagerDefinitionChangeRecorder()
    widget.pipeline_steps = [initial_step]
    widget.update_pipeline_for_plate(TEST_PLATE_SCOPE, [initial_step])
    editor_state = PipelineObjectStateBinding.editor_state_for_plate(TEST_PLATE_SCOPE)
    [step_scope_id] = editor_state.step_scope_ids
    step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
    assert step_state is not None
    [function_token] = step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY]
    function_state = ObjectStateRegistry.get_by_scope(
        f"{step_scope_id}::{function_token}"
    )
    assert function_state is not None
    driver = widget.code_document_driver()

    try:
        assert driver is not None
        driver.apply_source(
            PipelineDocumentAuthority.render(
                PipelineDocumentAuthority.from_values(
                    pipeline_config=PipelineConfig(),
                    pipeline_steps=[
                        FunctionStep(
                            name="Normalize after",
                            func=(
                                stack_percentile_normalize,
                                {"low_percentile": 0.75},
                            ),
                        )
                    ],
                )
            )
        )

        [reconciled_scope_id] = PipelineObjectStateBinding.editor_state_for_plate(
            TEST_PLATE_SCOPE
        ).step_scope_ids
        assert reconciled_scope_id == step_scope_id
        assert ObjectStateRegistry.get_by_scope(step_scope_id) is step_state
        [reconciled_function_token] = step_state.metadata[
            FUNC_EDITOR_PATTERN_TOKENS_META_KEY
        ]
        reconciled_function_state = ObjectStateRegistry.get_by_scope(
            f"{step_scope_id}::{reconciled_function_token}"
        )
        assert reconciled_function_state is not None
        editor_object_state = ObjectStateRegistry.get_by_scope(
            PipelineScopeIdentity.from_plate_scope(TEST_PLATE_SCOPE).scope_id
        )
        assert editor_object_state is not None
        assert step_state.saved_object.name == "Normalize after"
        assert reconciled_function_state.parameters["low_percentile"] == 0.75
        assert not step_state.is_raw_dirty
        assert not step_state.dirty_fields
        assert not reconciled_function_state.is_raw_dirty
        assert not reconciled_function_state.dirty_fields
        assert not editor_object_state.is_raw_dirty
        assert not editor_object_state.dirty_fields
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_step_code_mode_applies_callable_pattern_through_parameter_form() -> None:
    """A parsed FunctionStep can update the live form's Callable field."""

    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    original = FunctionStep(func=stack_percentile_normalize, name="Normalize")
    replacement = FunctionStep(
        func=(stack_percentile_normalize, {"low_percentile": 0.75}),
        name="Normalize edited",
    )
    manager = None

    try:
        PipelineObjectStateBinding.update_plate_steps(TEST_PLATE_SCOPE, [original])
        [step_scope_id] = PipelineObjectStateBinding.editor_state_for_plate(
            TEST_PLATE_SCOPE
        ).step_scope_ids
        step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
        assert step_state is not None
        manager = ParameterFormManager(
            step_state,
            FormManagerConfig(color_scheme=ColorScheme()),
        )

        CodeEditorFormUpdater.update_form_from_instance(manager, replacement)

        assert step_state.parameters["name"] == "Normalize edited"
        func, kwargs = step_state.parameters["func"]
        assert func is stack_percentile_normalize
        assert kwargs["low_percentile"] == 0.75
    finally:
        if manager is not None:
            manager.deleteLater()
        ObjectStateRegistry.clear()


def test_pipeline_editor_code_document_rejects_active_execution_before_mutation() -> (
    None
):
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    plate_manager = PlateManagerDefinitionChangeRecorder()
    widget.current_plate = TEST_PLATE_SCOPE
    widget.plate_manager = plate_manager
    original_step = FunctionStep(name="Original")
    widget.pipeline_steps = [original_step]
    widget.update_pipeline_for_plate(TEST_PLATE_SCOPE, [original_step])
    original_config = PipelineConfig()
    plate_manager.plate_configs[TEST_PLATE_SCOPE] = original_config
    plate_manager.execution_state = ManagerExecutionState.RUNNING
    driver = widget.code_document_driver()

    try:
        assert driver is not None
        with pytest.raises(
            RuntimeError,
            match="cannot change while plate execution is active",
        ):
            driver.apply_source(
                PipelineDocumentAuthority.render(
                    PipelineDocumentAuthority.from_values(
                        pipeline_config=PipelineConfig(),
                        pipeline_steps=[FunctionStep(name="Replacement")],
                    )
                )
            )

        assert [step.name for step in widget.pipeline_steps] == ["Original"]
        assert [
            step.name
            for step in PipelineObjectStateBinding.steps_for_plate(TEST_PLATE_SCOPE)
        ] == ["Original"]
        assert plate_manager.plate_configs[TEST_PLATE_SCOPE] is original_config
        assert plate_manager.changed_plates == []
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_pipeline_editor_code_document_reads_function_child_object_state() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    step = FunctionStep(
        name="Normalize",
        func=(
            stack_percentile_normalize,
            {
                "low_percentile": 0.5,
                "high_percentile": 99.5,
            },
        ),
    )
    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    widget.current_plate = TEST_PLATE_SCOPE
    widget.pipeline_steps = [step]
    widget.update_pipeline_for_plate(TEST_PLATE_SCOPE, [step])

    try:
        step_scope = widget._build_step_scope_id(step)
        step_state = ObjectStateRegistry.get_by_scope(step_scope)
        assert step_state is not None
        [function_token] = step_state.metadata[
            FUNC_EDITOR_PATTERN_TOKENS_META_KEY
        ]
        function_scope = f"{step_scope}::{function_token}"
        function_state = ObjectStateRegistry.get_by_scope(function_scope)
        assert function_state is not None

        function_state.update_parameter("low_percentile", 0.75)

        source = widget.code_document_source(clean=True)

        assert "'low_percentile': 0.75" in source
        assert "'low_percentile': 0.5" not in source
    finally:
        widget.close()
        ObjectStateRegistry.clear()


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

    class SuppressionAwareSignalRecorder(SignalRecorder):
        def __init__(self, editor) -> None:
            super().__init__()
            self.editor = editor
            self.suppression_values = []

        def emit(self, value):
            self.suppression_values.append(self.editor._suppress_pipeline_state_sync)
            super().emit(value)

    class Editor:
        current_plate = TEST_PLATE_SCOPE

        def __init__(self) -> None:
            self.pipeline_steps = []
            self._suppress_pipeline_state_sync = False
            self.pipeline_changed = SuppressionAwareSignalRecorder(self)
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

    PipelineEditorListWorkflow(editor).restore_after_time_travel(
        dirty_states=[
            (
                PipelineScopeIdentity.from_plate_scope(TEST_PLATE_SCOPE).scope_id,
                object(),
            )
        ],
    )

    assert editor.pipeline_steps == restored_steps
    assert editor.normalized is True
    assert editor.list_updated is True
    assert editor.buttons_updated is True
    assert editor.pipeline_changed.emissions == [restored_steps]
    assert editor.pipeline_changed.suppression_values == [True]
    assert editor._suppress_pipeline_state_sync is False
    assert editor.event_bus.pipeline_emissions == [restored_steps]


def test_pipeline_editor_time_travel_step_field_restore_stays_local() -> None:
    restored_steps = [FunctionStep(name="First"), FunctionStep(name="Second")]

    class Editor:
        current_plate = TEST_PLATE_SCOPE

        def __init__(self) -> None:
            self.pipeline_steps = []
            self._suppress_pipeline_state_sync = False
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

    PipelineEditorListWorkflow(editor).restore_after_time_travel(
        dirty_states=[(f"{TEST_PLATE_SCOPE}::functionstep_0", object())],
    )

    assert editor.pipeline_steps == restored_steps
    assert editor.normalized is True
    assert editor.list_updated is True
    assert editor.buttons_updated is True
    assert editor.pipeline_changed.emissions == []
    assert editor.event_bus.pipeline_emissions == []


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


def test_add_step_action_registers_state_before_opening_and_supports_edit(
    monkeypatch,
) -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(TEST_PLATE_SCOPE)

    class CallbackSignal:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

        def emit(self) -> None:
            for callback in self.callbacks:
                callback()

    opened_editors = []

    class EditorRecorder:
        def __init__(
            self,
            *,
            step_data,
            is_new,
            on_save_callback,
            plate_scope,
            **kwargs,
        ) -> None:
            del kwargs
            self.step_data = step_data
            self.is_new = is_new
            self.on_save_callback = on_save_callback
            self.rejected = CallbackSignal()
            self.scope_id = ScopeTokenService.build_scope_id(plate_scope, step_data)
            assert ObjectStateRegistry.get_by_scope(self.scope_id) is not None
            opened_editors.append(self)

        def set_original_step_for_change_detection(self) -> None:
            pass

        def show(self) -> None:
            pass

        def raise_(self) -> None:
            pass

        def activateWindow(self) -> None:
            pass

    monkeypatch.setattr(
        "openhcs.pyqt_gui.widgets.pipeline_editor.DualEditorWindow",
        EditorRecorder,
    )

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    widget.current_plate = TEST_PLATE_SCOPE
    widget.buttons["add_step"].setEnabled(True)

    try:
        widget.buttons["add_step"].click()

        assert len(opened_editors) == 1
        add_editor = opened_editors[0]
        assert add_editor.is_new is True
        assert widget.pipeline_steps == []
        assert PipelineObjectStateBinding.editor_state_for_plate(
            TEST_PLATE_SCOPE
        ).step_scope_ids == (add_editor.scope_id,)

        step_state = ObjectStateRegistry.get_by_scope(add_editor.scope_id)
        assert step_state is not None
        step_state.update_parameter("name", "Added Step")
        edited_step = step_state.to_object()
        assert edited_step is not add_editor.step_data
        add_editor.on_save_callback(edited_step)

        assert [step.name for step in widget.pipeline_steps] == ["Added Step"]
        assert [
            step.name
            for step in PipelineObjectStateBinding.steps_for_plate(TEST_PLATE_SCOPE)
        ] == ["Added Step"]
        accepted_history = ObjectStateRegistry.get_branch_history()
        assert accepted_history[-2].label.startswith("edit name")
        assert accepted_history[-1].label.startswith("add step Added Step")
        assert accepted_history[-1].parent_id == accepted_history[-2].id
        assert add_editor.scope_id in accepted_history[-2].all_states
        assert add_editor.scope_id in accepted_history[-1].all_states

        widget.show_item_editor(widget.pipeline_steps[0])

        assert len(opened_editors) == 2
        edit_editor = opened_editors[1]
        assert edit_editor.is_new is False
        assert edit_editor.scope_id == add_editor.scope_id
        assert ObjectStateRegistry.get_by_scope(edit_editor.scope_id) is not None

        widget.buttons["add_step"].setEnabled(True)
        widget.buttons["add_step"].click()
        rejected_editor = opened_editors[2]
        rejected_state = ObjectStateRegistry.get_by_scope(rejected_editor.scope_id)
        assert rejected_state is not None
        rejected_state.update_parameter("name", "Rejected Staged Edit")
        rejected_editor.rejected.emit()

        assert PipelineObjectStateBinding.editor_state_for_plate(
            TEST_PLATE_SCOPE
        ).step_scope_ids == (add_editor.scope_id,)
        assert ObjectStateRegistry.get_by_scope(rejected_editor.scope_id) is None
        assert [
            step.name
            for step in PipelineObjectStateBinding.steps_for_plate(TEST_PLATE_SCOPE)
        ] == ["Added Step"]

        discard_snapshot = ObjectStateRegistry.get_branch_history()[-1]
        assert discard_snapshot.label.startswith("discard staged step Step_2")
        assert rejected_editor.scope_id not in discard_snapshot.all_states

        assert ObjectStateRegistry.time_travel_back()
        assert (
            ObjectStateRegistry.get_by_scope(rejected_editor.scope_id) is rejected_state
        )
        assert [step.name for step in widget.pipeline_steps] == ["Added Step"]

        assert ObjectStateRegistry.time_travel_forward()
        assert ObjectStateRegistry.get_by_scope(rejected_editor.scope_id) is None
        assert ObjectStateRegistry.get_by_scope(add_editor.scope_id) is step_state
        assert [step.name for step in widget.pipeline_steps] == ["Added Step"]

        history_head_id = ObjectStateRegistry.get_branch_history()[-1].id
        widget.buttons["add_step"].setEnabled(True)
        widget.buttons["add_step"].click()
        unedited_rejected_editor = opened_editors[3]
        unedited_rejected_editor.rejected.emit()

        assert (
            ObjectStateRegistry.get_by_scope(unedited_rejected_editor.scope_id) is None
        )
        assert ObjectStateRegistry.get_branch_history()[-1].id == history_head_id
        assert [step.name for step in widget.pipeline_steps] == ["Added Step"]
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_add_step_history_preserves_open_step_across_edit_rewind_and_forward(
    monkeypatch,
) -> None:
    """Accepted Add owns a snapshot before later field edits can be rewound."""

    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(TEST_PLATE_SCOPE)

    class CallbackSignal:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

    opened_editors = []

    class EditorRecorder:
        def __init__(
            self,
            *,
            step_data,
            is_new,
            on_save_callback,
            plate_scope,
            **kwargs,
        ) -> None:
            del kwargs
            self.step_data = step_data
            self.is_new = is_new
            self.on_save_callback = on_save_callback
            self.rejected = CallbackSignal()
            self.scope_id = ScopeTokenService.build_scope_id(plate_scope, step_data)
            self.state = ObjectStateRegistry.get_by_scope(self.scope_id)
            assert self.state is not None
            opened_editors.append(self)

        def set_original_step_for_change_detection(self) -> None:
            pass

        def show(self) -> None:
            pass

        def raise_(self) -> None:
            pass

        def activateWindow(self) -> None:
            pass

    monkeypatch.setattr(
        "openhcs.pyqt_gui.widgets.pipeline_editor.DualEditorWindow",
        EditorRecorder,
    )

    unrelated_state = ObjectState(
        FunctionStep(name="Existing History"),
        scope_id="other-plate::functionstep_0",
    )
    ObjectStateRegistry.register(unrelated_state, _skip_snapshot=True)
    unrelated_state.update_parameter("name", "Existing History Edited")

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    widget.current_plate = TEST_PLATE_SCOPE
    widget.buttons["add_step"].setEnabled(True)

    try:
        widget.buttons["add_step"].click()
        add_editor = opened_editors[0]
        add_editor.on_save_callback(add_editor.step_data)

        add_history = ObjectStateRegistry.get_branch_history()
        add_snapshot = add_history[-1]
        assert add_snapshot.label.startswith("add step Step_1")
        assert add_editor.scope_id in add_snapshot.all_states
        add_parent = add_history[-2]
        assert add_snapshot.parent_id == add_parent.id
        assert add_editor.scope_id not in add_parent.all_states

        add_editor.state.update_parameter("name", "Edited Step")
        edit_snapshot = ObjectStateRegistry.get_branch_history()[-1]
        assert edit_snapshot.label.startswith("edit name")
        assert edit_snapshot.parent_id == add_snapshot.id
        assert ObjectStateRegistry.time_travel_back()

        assert ObjectStateRegistry.get_by_scope(add_editor.scope_id) is add_editor.state
        assert [step.name for step in widget.pipeline_steps] == ["Step_1"]
        assert [
            step.name
            for step in PipelineObjectStateBinding.steps_for_plate(TEST_PLATE_SCOPE)
        ] == ["Step_1"]

        assert ObjectStateRegistry.time_travel_back()
        assert ObjectStateRegistry.get_by_scope(add_editor.scope_id) is None
        assert widget.pipeline_steps == []

        assert ObjectStateRegistry.time_travel_forward()
        assert ObjectStateRegistry.get_by_scope(add_editor.scope_id) is add_editor.state
        assert [step.name for step in widget.pipeline_steps] == ["Step_1"]

        assert ObjectStateRegistry.time_travel_forward()
        assert ObjectStateRegistry.get_by_scope(add_editor.scope_id) is add_editor.state
        assert [step.name for step in widget.pipeline_steps] == ["Edited Step"]

        add_editor.state.update_parameter("name", "Editable After Rewind")
        assert [
            step.name
            for step in PipelineObjectStateBinding.steps_for_plate(TEST_PLATE_SCOPE)
        ] == ["Editable After Rewind"]
    finally:
        widget.close()
        ObjectStateRegistry.clear()


class RuntimeCallable:
    """Callable object with a non-function scope prefix."""

    def __call__(self, image, threshold: int = 1):
        return image


@dataclass(frozen=True)
class RuntimeSettings:
    threshold: int = 1


def runtime_with_settings(
    image,
    settings: RuntimeSettings = RuntimeSettings(),
):
    return image


def test_step_registration_preserves_and_updates_nested_function_kwargs() -> None:
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope("plate")
    initial_settings = RuntimeSettings(threshold=3)
    replacement_settings = RuntimeSettings(threshold=7)

    PipelineObjectStateBinding.update_plate_steps(
        "plate",
        [
            FunctionStep(
                func=(runtime_with_settings, {"settings": initial_settings}),
                name="Nested settings",
            )
        ],
    )

    initial_editor_state = PipelineObjectStateBinding.editor_state_for_plate("plate")
    initial_step_scope_id = initial_editor_state.step_scope_ids[0]
    initial_step_state = ObjectStateRegistry.get_by_scope(initial_step_scope_id)
    assert initial_step_state is not None
    function_token = initial_step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY][0]
    function_scope_id = f"{initial_step_scope_id}::{function_token}"
    initial_function_state = ObjectStateRegistry.get_by_scope(function_scope_id)
    assert initial_function_state is not None
    assert initial_function_state.parameters["settings"] == initial_settings
    assert initial_function_state.parameters["settings.threshold"] == 3
    assert initial_function_state.reconstruct_top_level_parameters() == {
        "settings": initial_settings,
    }

    initial_step = PipelineObjectStateBinding.steps_for_plate("plate")[0]
    assert initial_step.func[1]["settings"] == initial_settings

    PipelineObjectStateBinding.update_plate_steps(
        "plate",
        [
            FunctionStep(
                func=(runtime_with_settings, {"settings": replacement_settings}),
                name="Nested settings",
            )
        ],
    )

    replacement_function_state = ObjectStateRegistry.get_by_scope(function_scope_id)
    assert replacement_function_state is initial_function_state
    assert replacement_function_state.parameters["settings"] == replacement_settings
    assert replacement_function_state.parameters["settings.threshold"] == 7
    assert replacement_function_state.reconstruct_top_level_parameters() == {
        "settings": replacement_settings,
    }

    replacement_step = PipelineObjectStateBinding.steps_for_plate("plate")[0]
    assert replacement_step.func[1]["settings"] == replacement_settings

    PipelineObjectStateBinding.update_plate_steps(
        "plate",
        [
            FunctionStep(
                func=runtime_with_settings,
                name="Nested settings",
            )
        ],
    )

    reset_function_state = ObjectStateRegistry.get_by_scope(function_scope_id)
    assert reset_function_state is initial_function_state
    assert reset_function_state.parameters["settings"] == RuntimeSettings()
    assert reset_function_state.parameters["settings.threshold"] == 1
    assert reset_function_state.reconstruct_top_level_parameters() == {
        "settings": RuntimeSettings(),
    }

    reset_step = PipelineObjectStateBinding.steps_for_plate("plate")[0]
    assert reset_step.func[1]["settings"] == RuntimeSettings()


def test_reconstructed_pipeline_save_notifies_only_edited_step() -> None:
    """Selecting a new test plate must not expand every callable baseline."""

    from openhcs.tests.test_pipeline import pipeline_steps

    plate_scope = "synthetic-plate"
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(plate_scope)
    events: list[tuple[str, set[str]]] = []

    def record_change(scope_id: str, changed_paths: set[str]) -> None:
        events.append((scope_id, set(changed_paths)))

    try:
        PipelineObjectStateBinding.update_plate_steps(
            plate_scope,
            list(pipeline_steps),
        )
        selected_steps = PipelineObjectStateBinding.steps_for_plate(plate_scope)
        editor_state = PipelineObjectStateBinding.editor_state_for_plate(plate_scope)

        edited_steps = list(selected_steps)
        edited_steps[2] = copy(edited_steps[2])
        edited_steps[2].name = "Edited only"

        ObjectStateRegistry.add_resolved_changed_callback(record_change)
        PipelineObjectStateBinding.update_plate_steps(plate_scope, edited_steps)

        assert events == [(editor_state.step_scope_ids[2], {"name"})]
    finally:
        ObjectStateRegistry.remove_resolved_changed_callback(record_change)
        ObjectStateRegistry.clear()


def test_reconstructed_pipeline_preserves_explicit_default_function_kwarg() -> None:
    """Canonical parent baselines retain explicitly authored default kwargs."""

    def threshold_image(image, threshold: int = 1):
        del threshold
        return image

    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope("plate")
    try:
        PipelineObjectStateBinding.update_plate_steps(
            "plate",
            [
                FunctionStep(
                    func=(threshold_image, {"threshold": 1}),
                    name="Threshold",
                )
            ],
        )

        reconstructed = PipelineObjectStateBinding.steps_for_plate("plate")

        assert reconstructed[0].func == (threshold_image, {"threshold": 1})
    finally:
        ObjectStateRegistry.clear()


def test_step_registration_persists_function_editor_scope_tokens() -> None:
    ObjectStateRegistry._states.clear()
    ScopeTokenService.clear_scope("plate")

    runtime_callable = RuntimeCallable()
    step = FunctionStep(
        func=(runtime_callable, {"threshold": 3}),
        name="Crop",
    )
    PipelineObjectStateBinding.update_plate_steps("plate", [step])
    editor_state = PipelineObjectStateBinding.editor_state_for_plate("plate")
    step_state = ObjectStateRegistry.get_by_scope(editor_state.step_scope_ids[0])
    assert step_state is not None

    assert step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY] == [
        "func_0",
    ]
    assert len(editor_state.step_scope_ids) == 1
    step_scope_id = editor_state.step_scope_ids[0]
    child_scope_id = f"{step_scope_id}::func_0"
    assert ObjectStateRegistry.get_by_scope(child_scope_id) is not None
    assert sorted(
        scope_id
        for scope_id in ObjectStateRegistry._states
        if scope_id.startswith(step_scope_id)
    ) == [
        step_scope_id,
        child_scope_id,
    ]


def test_complete_pipeline_diff_preserves_reordered_and_edited_step_scopes() -> None:
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(TEST_PLATE_SCOPE)
    try:
        PipelineObjectStateBinding.update_plate_steps(
            TEST_PLATE_SCOPE,
            [
                FunctionStep(func=stack_percentile_normalize, name="Normalize"),
                FunctionStep(func=correct_illumination_apply, name="Correct"),
            ],
        )
        previous_scope_ids = PipelineObjectStateBinding.editor_state_for_plate(
            TEST_PLATE_SCOPE
        ).step_scope_ids
        normalize_scope, correct_scope = previous_scope_ids
        normalize_state = ObjectStateRegistry.get_by_scope(normalize_scope)
        assert normalize_state is not None

        PipelineObjectStateBinding.update_plate_steps(
            TEST_PLATE_SCOPE,
            [
                FunctionStep(func=correct_illumination_apply, name="Correct"),
                FunctionStep(func=stack_percentile_normalize, name="New normalize"),
            ],
        )

        replacement = PipelineObjectStateBinding.steps_for_plate(TEST_PLATE_SCOPE)
        replacement_scope_ids = PipelineObjectStateBinding.editor_state_for_plate(
            TEST_PLATE_SCOPE
        ).step_scope_ids
        assert replacement_scope_ids == (correct_scope, normalize_scope)
        assert [
            FunctionPatternCodeDocumentService.function_and_kwargs(step.func)[0]
            for step in replacement
        ] == [
            correct_illumination_apply,
            stack_percentile_normalize,
        ]
        assert replacement[1].name == "New normalize"
        assert ObjectStateRegistry.get_by_scope(normalize_scope) is normalize_state
        assert normalize_state.dirty_fields == {"name"}
    finally:
        ObjectStateRegistry.clear()


def test_complete_pipeline_diff_adds_and_removes_only_changed_occurrences() -> None:
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(TEST_PLATE_SCOPE)
    try:
        PipelineObjectStateBinding.update_plate_steps(
            TEST_PLATE_SCOPE,
            [
                FunctionStep(func=stack_percentile_normalize, name="Normalize"),
                FunctionStep(func=correct_illumination_apply, name="Correct"),
            ],
        )
        normalize_scope, correct_scope = (
            PipelineObjectStateBinding.editor_state_for_plate(
                TEST_PLATE_SCOPE
            ).step_scope_ids
        )

        PipelineObjectStateBinding.update_plate_steps(
            TEST_PLATE_SCOPE,
            [FunctionStep(func=correct_illumination_apply, name="Correct")],
        )

        assert PipelineObjectStateBinding.editor_state_for_plate(
            TEST_PLATE_SCOPE
        ).step_scope_ids == (correct_scope,)
        assert ObjectStateRegistry.get_by_scope(normalize_scope) is None

        PipelineObjectStateBinding.update_plate_steps(
            TEST_PLATE_SCOPE,
            [
                FunctionStep(func=correct_illumination_apply, name="Correct"),
                FunctionStep(func=stack_percentile_normalize, name="Added"),
            ],
        )
        correct_after_add, added_scope = (
            PipelineObjectStateBinding.editor_state_for_plate(
                TEST_PLATE_SCOPE
            ).step_scope_ids
        )
        assert correct_after_add == correct_scope
        assert added_scope not in {normalize_scope, correct_scope}
    finally:
        ObjectStateRegistry.clear()


def test_step_registration_does_not_publish_runtime_artifact_parameters() -> None:
    ObjectStateRegistry._states.clear()
    ScopeTokenService.clear_scope("plate")

    @artifact_inputs("positions")
    def assemble(image, positions=None, threshold: int = 1):
        del positions, threshold
        return image

    PipelineObjectStateBinding.update_plate_steps(
        "plate",
        [FunctionStep(func=assemble, name="Assemble")],
    )
    editor_state = PipelineObjectStateBinding.editor_state_for_plate("plate")
    step_scope_id = editor_state.step_scope_ids[0]
    step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
    assert step_state is not None
    token = step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY][0]
    child_state = ObjectStateRegistry.get_by_scope(f"{step_scope_id}::{token}")
    assert child_state is not None

    assert "positions" not in child_state.parameters
    reconstructed_step = PipelineObjectStateBinding.steps_for_plate("plate")[0]
    assert reconstructed_step.func == (assemble, {"threshold": 1})


def test_step_registration_exposes_public_cellprofiler_settings_in_function_child_state() -> (
    None
):
    ObjectStateRegistry._states.clear()
    ScopeTokenService.clear_scope("plate")

    runtime_callable = RuntimeCallable()
    step = FunctionStep(
        func=(
            runtime_callable,
            {
                "threshold": 3,
                "select_the_input_image": "OrigBlue",
            },
        ),
        name="Crop",
    )
    PipelineObjectStateBinding.update_plate_steps("plate", [step])
    editor_state = PipelineObjectStateBinding.editor_state_for_plate("plate")
    step_scope_id = editor_state.step_scope_ids[0]
    step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
    assert step_state is not None
    token = step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY][0]
    child_state = ObjectStateRegistry.get_by_scope(f"{step_scope_id}::{token}")
    assert child_state is not None
    reconstructed_step = PipelineObjectStateBinding.steps_for_plate("plate")[0]
    reconstructed_kwargs = reconstructed_step.func[1]

    assert child_state.reconstruct_top_level_parameters() == {
        "threshold": 3,
        "select_the_input_image": "OrigBlue",
    }
    assert reconstructed_kwargs["threshold"] == 3
    assert reconstructed_kwargs["select_the_input_image"] == "OrigBlue"


def test_pipeline_editor_root_preserves_only_text_and_step_scope_ids() -> None:
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope("plate")

    steps = [
        FunctionStep(
            func=(RuntimeCallable(), {"threshold": 3}),
            name="ImportedStep",
        )
    ]
    PipelineObjectStateBinding.update_plate_steps("plate", steps)
    PipelineObjectStateBinding.update_editor_text(
        "plate",
        name="ImportedPipeline",
        description="Imported description",
    )

    editor_state = PipelineObjectStateBinding.editor_state_for_plate("plate")
    step_scope_id = editor_state.step_scope_ids[0]
    assert editor_state == PipelineEditorStateRoot(
        name="ImportedPipeline",
        description="Imported description",
        step_scope_ids=(step_scope_id,),
    )
    assert [
        step.name for step in PipelineObjectStateBinding.steps_for_plate("plate")
    ] == ["ImportedStep"]
    assert not hasattr(editor_state, "metadata")
    assert not hasattr(editor_state, "pipeline_config")


def test_pipeline_object_state_binding_public_surface_is_editor_list_only() -> None:
    public_methods = {
        name
        for name, value in PipelineObjectStateBinding.__dict__.items()
        if not name.startswith("_")
        and (isinstance(value, (classmethod, staticmethod)) or callable(value))
    }

    assert public_methods == {
        "commit_plate_state",
        "editor_state_for_plate",
        "registered_plate_steps",
        "steps_for_plate",
        "update_editor_text",
        "update_plate_steps",
    }
    assert tuple(PipelineEditorStateRoot.__dataclass_fields__) == (
        "name",
        "description",
        "step_scope_ids",
    )
    assert PipelineEditorStateRoot.__slots__ == (
        "name",
        "description",
        "step_scope_ids",
    )


def test_pipeline_update_refreshes_existing_step_scope_state() -> None:
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(TEST_PLATE_SCOPE)

    editor = PipelineEditorWidget.__new__(PipelineEditorWidget)
    editor.plate_manager = None
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


def test_groupby_none_is_concrete_object_state_override() -> None:
    ObjectStateRegistry.clear()

    assert not (GroupBy.NONE == None)  # noqa: E711
    assert not (None == GroupBy.NONE)  # noqa: E711

    state = ObjectState(
        FunctionStep(
            name="IdentifyPrimaryObjects",
            processing_config=LazyProcessingConfig(),
        ),
        scope_id="plate::functionstep_0",
    )

    state.update_parameter("processing_config.group_by", GroupBy.NONE)

    assert state.parameters["processing_config.group_by"] is GroupBy.NONE
    assert "processing_config.group_by" in state.signature_diff_fields

    state.reset_parameter("processing_config.group_by")

    assert state.parameters["processing_config.group_by"] is None
    assert "processing_config.group_by" not in state.signature_diff_fields


def test_pipeline_update_transfers_existing_step_scope_token_for_reapply() -> None:
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(TEST_PLATE_SCOPE)

    editor = PipelineEditorWidget.__new__(PipelineEditorWidget)
    editor.plate_manager = None
    original = FunctionStep(name="CountCells")
    editor.update_pipeline_for_plate(TEST_PLATE_SCOPE, [original])

    replacement = FunctionStep(name="CountCells")
    editor.update_pipeline_for_plate(TEST_PLATE_SCOPE, [replacement])

    pipeline_scope = f"{TEST_PLATE_SCOPE}::pipeline"
    pipeline_state = ObjectStateRegistry.get_by_scope(pipeline_scope)
    assert pipeline_state is not None
    assert pipeline_state.parameters["step_scope_ids"] == (
        f"{TEST_PLATE_SCOPE}::functionstep_0",
    )
    assert (
        ObjectStateRegistry.get_by_scope(f"{TEST_PLATE_SCOPE}::functionstep_1") is None
    )
    assert ScopeTokenService.object_token(replacement) == "functionstep_0"


def test_pipeline_update_unregisters_removed_step_scopes() -> None:
    ObjectStateRegistry.clear()
    ScopeTokenService.clear_scope(TEST_PLATE_SCOPE)

    editor = PipelineEditorWidget.__new__(PipelineEditorWidget)
    editor.plate_manager = None
    first = FunctionStep(name="First")
    second = FunctionStep(name="Second")
    editor.update_pipeline_for_plate(TEST_PLATE_SCOPE, [first, second])

    replacement = FunctionStep(name="First")
    editor.update_pipeline_for_plate(TEST_PLATE_SCOPE, [replacement])

    assert (
        ObjectStateRegistry.get_by_scope(f"{TEST_PLATE_SCOPE}::functionstep_0")
        is not None
    )
    assert (
        ObjectStateRegistry.get_by_scope(f"{TEST_PLATE_SCOPE}::functionstep_1") is None
    )


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


def test_step_editor_child_scope_resolves_to_parent_window_navigation() -> None:
    plate_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        "/tmp/plate",
        "/tmp/plate/Analysis_Final.cppipe",
    ).scope_id
    child_token = "cellprofilerruntimecallable_0"
    scope_id = f"{plate_scope}::functionstep_17::{child_token}"

    assert (
        StepEditorScope.window_scope_id_for_scope(scope_id)
        == f"{plate_scope}::functionstep_17"
    )
    assert StepEditorScope.window_field_path_for_scope(
        scope_id,
        "adaptive_window_size",
    ) == build_function_token_field_path(
        child_token,
        fallback_base_field_path="func.adaptive_window_size",
    )


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


def test_delete_and_edit_buttons_require_step_selection() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    widget.current_plate = TEST_PLATE_SCOPE
    widget._is_current_plate_initialized = lambda: True
    step = FunctionStep(name="One")
    widget.pipeline_steps = [step]

    try:
        widget.get_selected_items = lambda: []
        widget.update_button_states()

        assert widget.buttons["del_step"].isEnabled() is False
        assert widget.buttons["edit_step"].isEnabled() is False

        widget.get_selected_items = lambda: [step]
        widget.update_button_states()

        assert widget.buttons["del_step"].isEnabled() is True
        assert widget.buttons["edit_step"].isEnabled() is True
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_debug_toolbar_requires_compiled_current_plate() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    plate_manager = PlateManagerCompiledStateRecorder()
    widget.current_plate = TEST_PLATE_SCOPE
    widget.plate_manager = plate_manager
    widget._is_current_plate_initialized = lambda: True

    try:
        widget.update_button_states()

        assert widget.debug_toolbar is not None
        command_type = DebugCommandType.STEP
        assert widget.debug_toolbar.command_enabled(command_type) is False

        plate_manager.plate_compiled_data[TEST_PLATE_SCOPE] = object()
        widget.update_button_states()

        assert widget.debug_toolbar.command_enabled(command_type) is True
    finally:
        widget.close()
        ObjectStateRegistry.clear()


def test_orchestrator_state_change_refreshes_debug_toolbar() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    widget = PipelineEditorWidget(PipelineEditorServiceStub())
    plate_manager = PlateManagerCompiledStateRecorder()
    widget.current_plate = TEST_PLATE_SCOPE
    widget.plate_manager = plate_manager
    widget._is_current_plate_initialized = lambda: True

    try:
        assert widget.debug_toolbar is not None
        widget.update_button_states()
        assert widget.debug_toolbar.command_enabled(DebugCommandType.STEP) is False

        plate_manager.plate_compiled_data[TEST_PLATE_SCOPE] = object()
        widget.on_orchestrator_state_changed(
            TEST_PLATE_SCOPE,
            OrchestratorState.COMPILED,
        )

        assert widget.debug_toolbar.command_enabled(DebugCommandType.STEP) is True
    finally:
        widget.close()
        ObjectStateRegistry.clear()
