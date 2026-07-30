from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QListWidget, QPushButton
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager
from pyqt_reactive.theming import ColorScheme

import openhcs.processing.backends.cellprofiler as cellprofiler_backend
import openhcs.pyqt_gui.widgets.shared.services.execution_submission_service as execution_submission_service
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyFijiStreamingConfig,
    LazyNapariStreamingConfig,
    LazySourceBindingsConfig,
    PipelineConfig,
    WellFilterConfig,
)
from openhcs.constants.constants import OrchestratorState
from openhcs.core.input_workspace import InputWorkspacePreparationResult
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_bindings import (
    LazyStepSourceBindingsConfig,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceSelector,
)
from openhcs.core.steps.function_step import FunctionStep
from objectstate.lazy_factory import ensure_global_config_context
from objectstate.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.ui.shared.plate_scope_identity import PlateScopeIdentity
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
)
from openhcs.pyqt_gui.services.pipeline_object_state_binding import (
    PipelineObjectStateBinding,
)
from openhcs.pyqt_gui.services.ui_agent_bridge import UiAgentBridgeService
from openhcs.pyqt_gui.services.ui_bridge_object_state import (
    ObjectStateBridgeProviderSet,
)
from openhcs.pyqt_gui.services.ui_bridge_plate_manager import (
    PlateManagerBridgeProviderSet,
)
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.services.service_adapter import GlobalEventBus
from openhcs.pyqt_gui.config import get_default_ui_config
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.pyqt_gui.widgets.plate_manager import (
    PlateManagerAction,
    PlateManagerWidget,
    PlateOperation,
    PlateOperationValidator,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_manager_workflows import (
    PlateManagerCodeWorkflow,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ExecutionBatchRuntime,
    ManagerExecutionState,
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_submission_service import (
    ExecutionSubmissionService,
)
from openhcs.pyqt_gui.widgets.shared.services.terminal_result_builder import (
    TerminalExecutionResultBuilder,
)
from openhcs.agent.dto.ui_bridge import (
    UiBridgeConfirmationRequirement,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentId,
    UiCodeDocumentRequest,
    UiCodeDocumentSelectionMode,
    UiStateSurfaceId,
    UiStateSurfaceRequest,
)
from openhcs.agent.ui_bridge_identities import PipelineEditorWidgetIdentity
from pyqt_reactive.services.window_manager import WindowManager
from openhcs.processing.backends.processors.numpy_processor import (
    percentile_normalize,
)


class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


def test_scope_mutation_authorization_uses_plate_identity_ownership() -> None:
    calls: list[str | None] = []
    manager = SimpleNamespace(
        plates=(PlateManagerRow.from_scope("/plate"),),
        require_pipeline_definition_mutation_allowed=(
            lambda plate_path=None: calls.append(plate_path)
        ),
    )

    PlateManagerWidget.require_pipeline_definition_mutation_allowed_for_scope(
        manager,
        "/plate::pipeline::functionstep_0",
    )
    PlateManagerWidget.require_pipeline_definition_mutation_allowed_for_scope(
        manager,
        "",
    )
    PlateManagerWidget.require_pipeline_definition_mutation_allowed_for_scope(
        manager,
        "ui_config",
    )

    assert calls == ["/plate", None]


def close_widget(widget) -> None:
    """Drain queued GUI updates from smoke widgets with monkeypatched setup."""

    if widget.item_list is None:
        widget.item_list = QListWidget()
    if not widget.buttons:
        widget.buttons = {
            name: QPushButton()
            for name in (
                "del_plate",
                "edit_config",
                "init_plate",
                "compile_plate",
                "code_plate",
                "view_metadata",
                "run_plate",
            )
        }
    widget.cleanup()
    widget.close()
    QApplication.processEvents()


class PlateManagerServiceStub:
    """Minimal service adapter surface needed by PlateManagerWidget construction."""

    def __init__(self) -> None:
        ensure_storage_registry()
        self.global_config = GlobalPipelineConfig()
        self.color_scheme = ColorScheme()
        self.event_bus = GlobalEventBus()
        self.filemanager = FileManager(storage_registry)

    def get_global_config(self) -> GlobalPipelineConfig:
        return self.global_config

    def set_global_config(self, global_config: GlobalPipelineConfig) -> None:
        self.global_config = global_config

    def get_current_color_scheme(self) -> ColorScheme:
        return self.color_scheme

    def get_event_bus(self) -> GlobalEventBus:
        return self.event_bus

    def get_file_manager(self) -> FileManager:
        return self.filemanager

    def execute_async_operation(self, operation):
        return operation()

    def show_error_dialog(self, message: str) -> None:
        self.last_error_message = message


class PlateManagerWidgetTestHarness:
    """Nominal owner for headless PlateManager test setup."""

    @staticmethod
    def widget(monkeypatch) -> PlateManagerWidget:
        QtApplicationHarness.app()
        monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
        monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
        monkeypatch.setattr(
            PlateManagerWidget, "update_button_states", lambda self: None
        )
        return PlateManagerWidget(
            PlateManagerServiceStub(),
            gui_config=get_default_ui_config(),
        )


class InlineUiThreadDispatcher:
    """Execute UI bridge work inline for a widget already owned by this thread."""

    def call(self, callback, *, timeout_ms: int = 5000):
        del timeout_ms
        return callback()

    def post(self, callback) -> None:
        callback()


class TestPlateManagerWidget:
    def test_constructor_initializes_qobject_before_signal_use(
        self, monkeypatch
    ) -> None:
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)

        assert widget.debug_snapshot_available is not None
        close_widget(widget)

    def test_loads_cellprofiler_pipeline_into_empty_plate(self, monkeypatch) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        widget.selected_plate_path = "/plate"

        widget._load_cellprofiler_pipeline_from_workspace(
            "/plate",
            CellProfilerWorkspaceResultFixture.with_steps(
                (FunctionStep(func=lambda image: image, name="Imported"),)
            ),
        )

        pipeline_steps = PipelineObjectStateBinding.steps_for_plate("/plate")
        assert [step.name for step in pipeline_steps] == ["Imported"]
        assert widget.source_binding_context_for_plate("/plate") is None
        close_widget(widget)
        ObjectStateRegistry.clear()

    def test_source_binding_context_projects_current_orchestrator_state(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        source_root = tmp_path / "source"
        execution_root = tmp_path / "execution"
        source_root.mkdir()
        execution_root.mkdir()
        logical_plate_id = "/logical/plate"
        orchestrator = PipelineOrchestrator(
            source_root,
            pipeline_config=PipelineConfig(
                source_bindings_config=LazySourceBindingsConfig(
                    bindings=(NamedSourceBinding(alias="Before"),),
                ),
            ),
        )
        orchestrator.bind_input_workspace(
            InputWorkspacePreparationResult(
                original_source_root=source_root,
                execution_plate_path=execution_root,
                pipeline_steps=[],
                pipeline_config=orchestrator.pipeline_config,
            )
        )
        ObjectStateRegistry.register(
            ObjectState(orchestrator, scope_id=logical_plate_id),
            _skip_snapshot=True,
        )

        try:
            before = widget.source_binding_context_for_plate(logical_plate_id)
            assert before is not None
            assert before.logical_plate_id == logical_plate_id
            assert before.display_plate_root == source_root
            assert before.execution_plate_path == execution_root
            assert tuple(
                binding.alias for binding in before.source_bindings.bindings
            ) == ("Before",)

            orchestrator.apply_pipeline_config(
                PipelineConfig(
                    source_bindings_config=LazySourceBindingsConfig(
                        bindings=(NamedSourceBinding(alias="After"),),
                    ),
                )
            )
            after = widget.source_binding_context_for_plate(logical_plate_id)
            assert after is not None
            assert tuple(
                binding.alias for binding in after.source_bindings.bindings
            ) == ("After",)
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_cellprofiler_workspace_import_keeps_public_steps_unbound(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        plate_root = tmp_path / "CropExample"
        plate_root.mkdir()
        plate_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            plate_root / "crop.cppipe",
        ).scope_id
        raw_step = FunctionStep(
            func=(
                cellprofiler_backend.crop,
                {
                    "crop_shape": "Rectangle",
                    "select_the_input_image": "OrigBlue",
                    "name_the_output_image": "CropBlue",
                },
            ),
            name="Crop",
        )

        try:
            widget._load_cellprofiler_pipeline_from_workspace(
                plate_scope,
                CellProfilerWorkspaceResultFixture.with_steps((raw_step,)),
            )

            stored_step = PipelineObjectStateBinding.steps_for_plate(plate_scope)[0]
            stored_func, stored_kwargs = stored_step.func
            assert stored_func.__name__ == "crop"
            assert stored_func is cellprofiler_backend.crop
            assert stored_kwargs["crop_shape"] == "Rectangle"
            assert stored_kwargs["select_the_input_image"] == "OrigBlue"
            assert stored_kwargs["name_the_output_image"] == "CropBlue"
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_compile_validator_requires_initialized_orchestrator(self) -> None:
        plate_scope = "/plate"
        ObjectStateRegistry.register(
            ObjectState(
                SimpleNamespace(state=OrchestratorState.CREATED),
                scope_id=plate_scope,
            ),
            _skip_snapshot=True,
        )
        manager = SimpleNamespace(
            _get_current_pipeline_definition=lambda scope_id: [
                FunctionStep(func=lambda image: image, name="Defined")
            ],
        )
        row = PlateManagerRow.from_scope(plate_scope)

        result = PlateOperationValidator.for_operation(PlateOperation.COMPILE).validate(
            manager,
            row,
        )

        assert not result.valid
        assert result.reason == "orchestrator_not_initialized"
        assert result.recovery_action is PlateManagerAction.INIT_PLATE

    def test_live_results_use_source_orchestrator_for_source_and_output_rows(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        source_root = tmp_path / "source_plate"
        output_root = tmp_path / "source_plate_openhcs"
        source_root.mkdir()
        output_root.mkdir()
        source_row = PlateManagerRow.from_scope(str(source_root))
        output_row = PlateManagerRow.from_scope(str(output_root))
        widget._ensure_root_state().update_parameter(
            "orchestrator_scope_ids",
            [source_row.scope_id, output_row.scope_id],
        )
        source_orchestrator = SimpleNamespace(state=OrchestratorState.COMPLETED)
        output_orchestrator = SimpleNamespace(state=OrchestratorState.CREATED)
        ObjectStateRegistry.register(
            ObjectState(source_orchestrator, scope_id=source_row.scope_id),
            _skip_snapshot=True,
        )
        ObjectStateRegistry.register(
            ObjectState(output_orchestrator, scope_id=output_row.scope_id),
            _skip_snapshot=True,
        )

        try:
            assert (
                widget._live_results_viewer_orchestrator_for_row(source_row)
                is source_orchestrator
            )
            assert (
                widget._live_results_viewer_orchestrator_for_row(output_row)
                is source_orchestrator
            )
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_refreshes_existing_pipeline_for_cellprofiler_plate(
        self,
        monkeypatch,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        PipelineObjectStateBinding.update_plate_steps(
            "/plate",
            [FunctionStep(func=lambda image: image, name="Existing")],
        )

        widget._load_cellprofiler_pipeline_from_workspace(
            "/plate",
            CellProfilerWorkspaceResultFixture.with_steps(
                (FunctionStep(func=lambda image: image, name="Imported"),)
            ),
        )

        assert (
            PipelineObjectStateBinding.steps_for_plate("/plate")[0].name == "Imported"
        )
        close_widget(widget)
        ObjectStateRegistry.clear()

    def test_cellprofiler_import_applies_pipeline_config_to_orchestrator_state(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        ensure_global_config_context(GlobalPipelineConfig, widget.global_config)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)
        match_plan = SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER)
        pipeline_config = PipelineConfig(
            source_bindings_config=LazySourceBindingsConfig(match_plan=match_plan),
        )
        step = FunctionStep(
            func=lambda image: image,
            name="Imported",
            source_bindings=LazyStepSourceBindingsConfig(
                enabled=False,
                bindings=(
                    NamedSourceBinding(
                        alias="OrigDNA",
                        selector=SourceSelector(
                            metadata=(
                                MetadataSelector(
                                    field="ChannelNumber",
                                    value="1",
                                ),
                            )
                        ),
                    ),
                ),
            ),
        )

        try:
            widget._create_orchestrator_for_plate(plate_scope, plate_root=plate_root)
            widget._load_cellprofiler_pipeline_from_workspace(
                plate_scope,
                CellProfilerWorkspaceResultFixture.with_steps(
                    (step,),
                    pipeline_config=pipeline_config,
                ),
            )

            plate_state = ObjectStateRegistry.get_by_scope(plate_scope)
            editor_state = PipelineObjectStateBinding.editor_state_for_plate(
                plate_scope
            )
            assert plate_state is not None
            assert widget.plate_configs[plate_scope] == pipeline_config
            assert (
                plate_state.get_saved_resolved_value(
                    "source_bindings_config.match_plan"
                )
                == match_plan
            )

            step_scope_id = editor_state.step_scope_ids[0]
            step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
            assert step_state is not None
            snapshot = StepSnapshot(
                index=0,
                scope_id=step_state.scope_id,
                step=step_state.to_saved_resolved_object(),
            )
            assert snapshot.step.source_bindings.match_plan == match_plan
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_explicit_cellprofiler_import_seeds_pipeline_before_editor_signal(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        ensure_global_config_context(GlobalPipelineConfig, widget.global_config)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            plate_root / "second.cppipe",
        ).scope_id
        orchestrator = PipelineOrchestrator(plate_path=plate_root)
        orchestrator.bind_input_workspace(
            CellProfilerWorkspaceResultFixture.with_steps(
                (FunctionStep(func=lambda image: image, name="Second"),)
            )
        )
        ObjectStateRegistry.register(
            ObjectState(object_instance=orchestrator, scope_id=plate_scope),
            _skip_snapshot=True,
        )
        signal_observations = []
        widget.plate_selected.connect(
            lambda _plate_path: signal_observations.append(
                PipelineObjectStateBinding.steps_for_plate(plate_scope)
            )
        )

        try:
            widget._load_cellprofiler_pipeline_from_orchestrator(plate_scope)
            widget.selected_plate_path = plate_scope
            widget.plate_selected.emit(plate_scope)

            assert signal_observations
            assert [step.name for step in signal_observations[0]] == ["Second"]
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_code_mode_per_plate_config_refreshes_delegate_saved_baseline(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        ensure_global_config_context(GlobalPipelineConfig, widget.global_config)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)
        config = PipelineConfig(
            napari_streaming_config=LazyNapariStreamingConfig(
                enabled=True,
                port=5557,
            ),
        )

        try:
            applied = PlateManagerCodeWorkflow(widget).apply_namespace(
                {
                    "plate_paths": [plate_scope],
                    "global_config": widget.global_config,
                    "per_plate_configs": {plate_scope: config},
                    "pipeline_data": {plate_scope: []},
                }
            )

            state = ObjectStateRegistry.get_by_scope(plate_scope)
            assert applied is True
            assert state is not None
            assert state.get_resolved_value("napari_streaming_config.port") == 5557
            assert (
                state.get_saved_resolved_value("napari_streaming_config.port") == 5557
            )
            assert state.dirty_fields == set()
            assert {
                "napari_streaming_config.enabled",
                "napari_streaming_config.port",
            } <= state.signature_diff_fields

            state.update_parameter("napari_streaming_config.port", 5558)
            assert state.dirty_fields == {"napari_streaming_config.port"}

            state.update_parameter("napari_streaming_config.port", 5557)
            assert state.dirty_fields == set()
            assert {
                "napari_streaming_config.enabled",
                "napari_streaming_config.port",
            } <= state.signature_diff_fields
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_global_config_save_preserves_dirty_pipeline_config_delegate(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        initial_global_config = GlobalPipelineConfig(num_workers=1)
        widget.global_config = initial_global_config
        ensure_global_config_context(GlobalPipelineConfig, initial_global_config)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)

        try:
            global_state = ObjectState(initial_global_config, scope_id="")
            ObjectStateRegistry.register(global_state, _skip_snapshot=True)
            orchestrator = PipelineOrchestrator(
                plate_path=plate_root,
                pipeline_config=PipelineConfig(),
            )
            plate_state = ObjectState(orchestrator, scope_id=plate_scope)
            ObjectStateRegistry.register(plate_state, _skip_snapshot=True)
            emitted_effective_configs = []
            widget.orchestrator_config_changed.connect(
                lambda scope_id, config: emitted_effective_configs.append(
                    (scope_id, config.num_workers)
                )
            )

            plate_state.update_parameter("num_workers", 3)
            assert plate_state.get_resolved_value("num_workers") == 3
            assert plate_state.get_saved_resolved_value("num_workers") == 1
            assert plate_state.dirty_fields == {"num_workers"}

            global_state.update_parameter("num_workers", 7)
            new_global_config = global_state.to_object(update_delegate=False)
            widget._update_orchestrator_global_config(
                plate_scope,
                orchestrator,
                new_global_config,
            )

            assert (
                object.__getattribute__(
                    orchestrator.pipeline_config,
                    "num_workers",
                )
                is None
            )
            assert plate_state.parameters["num_workers"] == 3
            assert plate_state._saved_parameters["num_workers"] is None
            assert plate_state.get_resolved_value("num_workers") == 3
            assert plate_state.get_saved_resolved_value("num_workers") == 1
            assert plate_state.dirty_fields == {"num_workers"}
            assert emitted_effective_configs == [(plate_scope, 7)]

            global_state.mark_saved()

            assert plate_state.parameters["num_workers"] == 3
            assert plate_state._saved_parameters["num_workers"] is None
            assert plate_state.get_resolved_value("num_workers") == 3
            assert plate_state.get_saved_resolved_value("num_workers") == 7
            assert plate_state.dirty_fields == {"num_workers"}
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_code_mode_renders_default_orchestrator_per_plate_config(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)
        orchestrator = PipelineOrchestrator(plate_path=plate_root)
        ObjectStateRegistry.register(
            ObjectState(orchestrator, scope_id=plate_scope),
            _skip_snapshot=True,
        )

        try:
            context = widget.orchestrator_code_document_context_for_rows(
                [PlateManagerRow.from_scope(plate_scope)]
            )

            assert "per_plate_configs = {" in context.source
            config_source = context.source.split("per_plate_configs = {", 1)[1].split(
                "pipeline_data = {", 1
            )[0]
            assert "path_1" in config_source
            assert context.source.count(repr(plate_scope)) == 1
            assert "PipelineConfig(" in config_source
            assert tuple(context.payload.per_plate_configs) == (plate_scope,)
            assert isinstance(
                context.payload.per_plate_configs[plate_scope],
                PipelineConfig,
            )
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_clean_code_mode_emits_only_authored_fiji_streaming_override(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)
        orchestrator = PipelineOrchestrator(
            plate_path=plate_root,
            pipeline_config=PipelineConfig(
                fiji_streaming_config=LazyFijiStreamingConfig(
                    well_filter="333",
                    enabled=False,
                ),
            ),
        )
        ObjectStateRegistry.register(
            ObjectState(orchestrator, scope_id=plate_scope),
            _skip_snapshot=True,
        )

        try:
            context = widget.orchestrator_code_document_context_for_rows(
                [PlateManagerRow.from_scope(plate_scope)]
            )
            restored = PlateManagerCodeDocumentAuthority.from_source(context.source)
            imported_config_names = {
                alias.name
                for node in ast.parse(context.source).body
                if isinstance(node, ast.ImportFrom)
                and node.module == "openhcs.core.config"
                for alias in node.names
            }

            assert imported_config_names == {
                "GlobalPipelineConfig",
                "LazyFijiStreamingConfig",
                "PipelineConfig",
            }
            assert "from openhcs.core.source_bindings import" not in context.source
            assert "fiji_streaming_config=LazyFijiStreamingConfig(" in context.source
            assert "well_filter='333'" in context.source
            assert "enabled=False" in context.source
            assert "napari_streaming_config=" not in context.source
            assert "path_planning_config=" not in context.source
            assert "step_materialization_config=" not in context.source

            restored_config = restored.per_plate_configs[plate_scope]
            restored_fiji_config = object.__getattribute__(
                restored_config,
                "fiji_streaming_config",
            )
            assert object.__getattribute__(restored_fiji_config, "well_filter") == "333"
            assert object.__getattribute__(restored_fiji_config, "enabled") is False
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_code_mode_factors_common_plate_root_once(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        plate_roots = (tmp_path / "screen" / "plate_A", tmp_path / "screen" / "plate_B")
        for plate_root in plate_roots:
            plate_root.mkdir(parents=True)
            scope_id = str(plate_root)
            ObjectStateRegistry.register(
                ObjectState(
                    PipelineOrchestrator(plate_path=plate_root),
                    scope_id=scope_id,
                ),
                _skip_snapshot=True,
            )

        try:
            context = widget.orchestrator_code_document_context_for_rows(
                [PlateManagerRow.from_scope(str(path)) for path in plate_roots]
            )

            common_root = str(tmp_path / "screen")
            assert f"path_root = Path({common_root!r})" in context.source
            assert context.source.count(common_root) == 1
            assert "path_1 = path_root / 'plate_A'" in context.source
            assert "path_2 = path_root / 'plate_B'" in context.source
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_code_mode_renders_pipeline_data_as_function_step_lists(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)
        orchestrator = PipelineOrchestrator(plate_path=plate_root)
        ObjectStateRegistry.register(
            ObjectState(orchestrator, scope_id=plate_scope),
            _skip_snapshot=True,
        )
        PipelineObjectStateBinding.update_plate_steps(
            plate_scope,
            [FunctionStep(func=cellprofiler_backend.crop, name="Crop")],
        )

        try:
            context = widget.orchestrator_code_document_context_for_rows(
                [PlateManagerRow.from_scope(plate_scope)]
            )

            assert "from openhcs.core.pipeline import Pipeline" not in context.source
            assert "Pipeline(" not in context.source
            assert "FunctionStep(" in context.source
            assert isinstance(context.payload.pipeline_data[plate_scope], list)
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_code_mode_keeps_authored_orchestrator_per_plate_config(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)
        orchestrator = PipelineOrchestrator(plate_path=plate_root)
        state = ObjectState(orchestrator, scope_id=plate_scope)
        ObjectStateRegistry.register(state, _skip_snapshot=True)
        state.update_parameter("napari_streaming_config.port", 5557)

        try:
            context = widget.orchestrator_code_document_context_for_rows(
                [PlateManagerRow.from_scope(plate_scope)]
            )

            assert "per_plate_configs = {" in context.source
            assert "PipelineConfig(" in context.source
            assert "port=5557" in context.source
            assert context.payload.per_plate_configs == {
                plate_scope: state.to_object(update_delegate=False)
            }
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_code_mode_reads_global_config_from_object_state(
        self,
        monkeypatch,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        widget.global_config = GlobalPipelineConfig(
            well_filter_config=WellFilterConfig(well_filter="A01")
        )
        ensure_global_config_context(GlobalPipelineConfig, widget.global_config)
        global_state = ObjectState(widget.global_config, scope_id="")
        ObjectStateRegistry.register(global_state, _skip_snapshot=True)

        try:
            global_state.update_parameter("well_filter_config.well_filter", "A02")

            context = widget.orchestrator_code_document_context_for_rows([])

            assert "well_filter='A02'" in context.source
            assert "well_filter='A01'" not in context.source
            assert (
                object.__getattribute__(
                    context.payload.global_pipeline_config.well_filter_config,
                    "well_filter",
                )
                == "A02"
            )
            assert (
                object.__getattribute__(
                    widget.global_config.well_filter_config,
                    "well_filter",
                )
                == "A01"
            )
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_plate_selection_does_not_reseed_dirty_cellprofiler_config(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        ensure_global_config_context(GlobalPipelineConfig, widget.global_config)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)
        imported_config = PipelineConfig(
            source_bindings_config=LazySourceBindingsConfig(
                match_plan=SourceBindingMatchPlan(
                    method=SourceBindingMatchMethod.ORDER,
                ),
            ),
        )
        imported_steps = [FunctionStep(func=lambda image: image, name="Imported")]
        orchestrator = PipelineOrchestrator(plate_path=plate_root)
        orchestrator.bind_input_workspace(
            CellProfilerWorkspaceResultFixture.with_steps(
                tuple(imported_steps),
                pipeline_config=imported_config,
            )
        )
        state = ObjectState(orchestrator, scope_id=plate_scope)
        ObjectStateRegistry.register(state, _skip_snapshot=True)

        try:
            widget._load_cellprofiler_pipeline_from_workspace(
                plate_scope,
                orchestrator.input_workspace_preparation_result,
            )
            state.update_parameter(
                "source_bindings_config.match_plan.method",
                SourceBindingMatchMethod.METADATA,
            )

            widget.plate_selected.emit(plate_scope)

            assert (
                state.parameters["source_bindings_config.match_plan.method"]
                is SourceBindingMatchMethod.METADATA
            )
            assert "source_bindings_config.match_plan.method" in state.dirty_fields
            assert [
                step.name
                for step in PipelineObjectStateBinding.steps_for_plate(plate_scope)
            ] == ["Imported"]
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_cellprofiler_import_does_not_write_into_stale_current_plate(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        QtApplicationHarness.app()
        ObjectStateRegistry.clear()
        service_adapter = PlateManagerServiceStub()
        ensure_global_config_context(
            GlobalPipelineConfig, service_adapter.global_config
        )
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        editor = PipelineEditorWidget(service_adapter)
        widget.pipeline_editor = editor
        plate_root = tmp_path / "AdvancedSegmentation"
        plate_root.mkdir()
        start_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            plate_root / "BBBC022_Analysis_Start.cppipe",
        ).scope_id
        final_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            plate_root / "BBBC022_Analysis_Final.cppipe",
        ).scope_id
        start_steps = [FunctionStep(func=lambda image: image, name="StartOnly")]
        final_steps = [FunctionStep(func=lambda image: image, name="FinalOnly")]

        try:
            editor.current_plate = start_scope
            editor.pipeline_steps = start_steps
            editor.update_pipeline_for_plate(start_scope, start_steps)
            widget.selected_plate_path = final_scope

            widget._load_cellprofiler_pipeline_from_workspace(
                final_scope,
                CellProfilerWorkspaceResultFixture.with_steps(tuple(final_steps)),
            )

            assert [
                step.name for step in editor.get_pipeline_for_plate(start_scope)
            ] == ["StartOnly"]
            assert [
                step.name for step in editor.get_pipeline_for_plate(final_scope)
            ] == ["FinalOnly"]
            assert editor.current_plate == start_scope
            assert [step.name for step in editor.pipeline_steps] == ["StartOnly"]
        finally:
            editor.close()
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_plate_manager_code_mode_keeps_public_cellprofiler_steps_unbound(
        self,
        tmp_path: Path,
    ) -> None:
        plate_root = tmp_path / "BeginnerSegmentation"
        plate_root.mkdir()
        cppipe_path = plate_root / "segmentation_final.cppipe"
        cppipe_path.write_text("Version:5", encoding="utf-8")
        plate_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            cppipe_path,
        ).scope_id
        manager = PlateManagerCodeWorkflowHarness(selected_plate_path=plate_scope)
        raw_step = FunctionStep(
            func=(cellprofiler_backend.crop, {"crop_shape": "Rectangle"}),
            name="Crop",
        )

        PlateManagerCodeWorkflow(manager).apply_pipeline_data({plate_scope: [raw_step]})

        updated_steps = PipelineObjectStateBinding.steps_for_plate(plate_scope)
        rebound_func = updated_steps[0].func[0]
        assert rebound_func.__name__ == "crop"
        assert rebound_func is cellprofiler_backend.crop

    def test_code_mode_pipeline_change_clears_stale_execution_state(self) -> None:
        ObjectStateRegistry.clear()
        plate_scope = "/plate"
        manager = PlateManagerCodeWorkflowHarness(selected_plate_path=plate_scope)
        manager.plate_compiled_data[plate_scope] = ("compiled",)
        manager.plate_execution_ids[plate_scope] = "execution-1"
        manager.plate_terminal_activity_status.mark_terminal(
            plate_scope,
            TerminalExecutionStatus.COMPLETE,
        )
        orchestrator = OrchestratorStateHolder(OrchestratorState.EXECUTING)
        ObjectStateRegistry.register(
            ObjectState(object_instance=orchestrator, scope_id=plate_scope),
            _skip_snapshot=True,
        )

        try:
            PlateManagerCodeWorkflow(manager).apply_pipeline_data(
                {
                    plate_scope: [
                        FunctionStep(
                            func=lambda image: image,
                            name="Replacement",
                        )
                    ]
                }
            )

            assert plate_scope not in manager.plate_compiled_data
            assert plate_scope not in manager.plate_execution_ids
            assert (
                manager.plate_terminal_activity_status.terminal_status(plate_scope)
                is None
            )
            assert orchestrator.state is OrchestratorState.READY
            assert manager.orchestrator_state_changed.emissions == [
                (plate_scope, "READY")
            ]
            assert manager.status_message.messages == [
                "Loaded 1 steps from plate-manager code document"
            ]
        finally:
            ObjectStateRegistry.clear()

    def test_ui_code_document_recovers_failed_stale_executing_state(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        QtApplicationHarness.app()
        ObjectStateRegistry.clear()
        ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
        manager = PlateManagerWidgetTestHarness.widget(monkeypatch)
        manager.item_list = QListWidget()
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)
        manager._create_orchestrator_for_plate(plate_scope)
        root_state = manager._ensure_root_state()
        root_state.update_parameter("orchestrator_scope_ids", [plate_scope])
        manager.selected_plate_path = plate_scope
        manager.update_item_list()
        PipelineObjectStateBinding.update_plate_steps(
            plate_scope,
            [FunctionStep(func=percentile_normalize, name="Failing")],
        )

        orchestrator = ObjectStateRegistry.get_object(plate_scope)
        orchestrator._initialized = True
        orchestrator._state = OrchestratorState.EXECUTING
        manager.execution_state = ManagerExecutionState.RUNNING
        manager.current_execution_id = "execution-1"
        manager.plate_execution_ids[plate_scope] = "execution-1"
        manager.plate_terminal_activity_status.begin_batch((plate_scope,))
        manager.plate_compiled_data[plate_scope] = SimpleNamespace()

        class ImmediateThread:
            def __init__(self, *, target, daemon):
                assert daemon is True
                self._target = target

            def start(self) -> None:
                self._target()

        class DeferredFailurePoller:
            policy = None

            def run(self, execution_id, policy) -> None:
                assert execution_id == "execution-1"
                self.policy = policy

            def fail(self) -> None:
                assert self.policy is not None
                self.policy.on_terminal(
                    "execution-1",
                    TerminalExecutionStatus.FAILED.value,
                    {
                        "status": TerminalExecutionStatus.FAILED.value,
                        "error": "expected test failure",
                    },
                )

        monkeypatch.setattr(
            execution_submission_service.threading,
            "Thread",
            ImmediateThread,
        )
        completion_poller = DeferredFailurePoller()
        submission_service = ExecutionSubmissionService(
            host=manager,
            context=SimpleNamespace(),
            completion_poller=completion_poller,
            terminal_result_builder=TerminalExecutionResultBuilder(),
        )
        submission_service.start_completion_poller("execution-1", plate_scope)

        replacement_source = PlateManagerCodeDocumentAuthority.render(
            PlateManagerCodeDocumentAuthority.from_values(
                plate_paths=[plate_scope],
                global_pipeline_config=manager.global_config,
                per_plate_configs={
                    plate_scope: manager.authored_pipeline_config_for_code_document(
                        plate_scope
                    )
                },
                pipeline_data={
                    plate_scope: [
                        FunctionStep(
                            func=percentile_normalize,
                            name="Replacement",
                        )
                    ]
                },
            )
        )
        bridge = UiAgentBridgeService(
            provider_set=PlateManagerBridgeProviderSet(manager),
            dispatcher=InlineUiThreadDispatcher(),
        )
        document = bridge.get_document(
            UiCodeDocumentRequest(
                document_id=UiCodeDocumentId.PLATE_MANAGER_ORCHESTRATOR.value,
                selection_mode=UiCodeDocumentSelectionMode.ALL.value,
            )
        )

        try:
            premature_result = bridge.apply_document(
                UiCodeDocumentApplyRequest(
                    document_id=UiCodeDocumentId.PLATE_MANAGER_ORCHESTRATOR.value,
                    source=replacement_source,
                    base_revision_token=document.current_revision_token,
                    confirmation_requirement=(
                        UiBridgeConfirmationRequirement.from_flag(False)
                    ),
                )
            )
            assert not premature_result.applied
            assert manager.plate_execution_ids[plate_scope] == "execution-1"
            assert [
                step.name
                for step in PipelineObjectStateBinding.steps_for_plate(plate_scope)
            ] == ["Failing"]

            completion_poller.fail()
            assert manager.execution_state is ManagerExecutionState.IDLE
            assert orchestrator.state is OrchestratorState.EXEC_FAILED
            assert plate_scope not in manager.plate_execution_ids

            terminal_document = bridge.get_document(
                UiCodeDocumentRequest(
                    document_id=UiCodeDocumentId.PLATE_MANAGER_ORCHESTRATOR.value,
                    selection_mode=UiCodeDocumentSelectionMode.ALL.value,
                )
            )
            result = bridge.apply_document(
                UiCodeDocumentApplyRequest(
                    document_id=UiCodeDocumentId.PLATE_MANAGER_ORCHESTRATOR.value,
                    source=replacement_source,
                    base_revision_token=terminal_document.current_revision_token,
                    confirmation_requirement=(
                        UiBridgeConfirmationRequirement.from_flag(False)
                    ),
                )
            )
            state = bridge.get_state_surface(
                UiStateSurfaceRequest(
                    surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
                    selection_mode=UiCodeDocumentSelectionMode.ALL.value,
                )
            )

            row = state.payload["rows"][0]
            assert result.applied
            assert row["orchestrator_state"] == OrchestratorState.READY.value
            assert row["compiled"] is False
            assert row["execution_active"] is False
            assert row["execution_id"] is None
            assert row["terminal_status"] is None
            assert [
                step.name
                for step in PipelineObjectStateBinding.steps_for_plate(plate_scope)
            ] == ["Replacement"]
        finally:
            close_widget(manager)
            ObjectStateRegistry.clear()

    def test_code_document_replacement_is_rejected_before_active_batch_mutation(
        self,
    ) -> None:
        ObjectStateRegistry.clear()
        plate_scope = "/plate"
        manager = PlateManagerCodeWorkflowHarness(selected_plate_path=plate_scope)
        manager.execution_state = ManagerExecutionState.RUNNING
        initial_steps = [FunctionStep(func=percentile_normalize, name="Initial")]
        PipelineObjectStateBinding.update_plate_steps(plate_scope, initial_steps)

        try:
            with pytest.raises(
                RuntimeError,
                match="cannot change while plate execution is active",
            ):
                PlateManagerCodeWorkflow(manager).apply_pipeline_data(
                    {
                        plate_scope: [
                            FunctionStep(
                                func=percentile_normalize,
                                name="Replacement",
                            )
                        ]
                    }
                )

            assert [
                step.name
                for step in PipelineObjectStateBinding.steps_for_plate(plate_scope)
            ] == ["Initial"]
        finally:
            ObjectStateRegistry.clear()

    def test_async_failure_finalizes_before_pipeline_editor_bridge_presentation(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        QtApplicationHarness.app()
        ObjectStateRegistry.clear()
        ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
        manager = PlateManagerWidgetTestHarness.widget(monkeypatch)
        manager.item_list = QListWidget()
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = str(plate_root)
        manager._create_orchestrator_for_plate(plate_scope)
        manager._ensure_root_state().update_parameter(
            "orchestrator_scope_ids",
            [plate_scope],
        )
        manager.selected_plate_path = plate_scope
        manager.update_item_list()
        original_step = FunctionStep(func=percentile_normalize, name="Original")
        PipelineObjectStateBinding.update_plate_steps(
            plate_scope,
            [original_step],
        )

        orchestrator = ObjectStateRegistry.get_object(plate_scope)
        orchestrator._initialized = True
        orchestrator._state = OrchestratorState.EXECUTING
        manager.execution_state = ManagerExecutionState.RUNNING
        manager.current_execution_id = "execution-1"
        manager.plate_execution_ids[plate_scope] = "execution-1"
        manager.plate_terminal_activity_status.begin_batch((plate_scope,))
        manager.plate_compiled_data[plate_scope] = SimpleNamespace()

        editor = PipelineEditorWidget(manager.service_adapter)
        editor.plate_manager = manager
        editor.set_current_plate(plate_scope)
        pipeline_scope = PipelineEditorWidgetIdentity.require_value()
        WindowManager.register(
            pipeline_scope,
            editor,
            code_document_driver=editor.code_document_driver(),
        )
        bridge = UiAgentBridgeService(
            provider_set=ObjectStateBridgeProviderSet(),
            dispatcher=InlineUiThreadDispatcher(),
        )

        class BlockingFailurePoller:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()

            def run(self, execution_id, policy) -> None:
                assert execution_id == "execution-1"
                self.started.set()
                assert self.release.wait(timeout=5)
                policy.on_terminal(
                    execution_id,
                    TerminalExecutionStatus.FAILED.value,
                    {
                        "status": TerminalExecutionStatus.FAILED.value,
                        "traceback": "expected asynchronous test failure",
                    },
                )

        completion_poller = BlockingFailurePoller()
        submission_service = ExecutionSubmissionService(
            host=manager,
            context=SimpleNamespace(),
            completion_poller=completion_poller,
            terminal_result_builder=TerminalExecutionResultBuilder(),
        )
        presented_states: list[tuple[ManagerExecutionState, str | None, bool]] = []

        def show_error_dialog(_message: str) -> None:
            presented_states.append(
                (
                    manager.execution_state,
                    manager.current_execution_id,
                    plate_scope in manager.plate_execution_ids,
                )
            )

        manager.service_adapter.show_error_dialog = show_error_dialog
        manager.execution_error.connect(manager._handle_execution_error)
        replacement_source = PipelineDocumentAuthority.render(
            PipelineDocumentAuthority.from_values(
                pipeline_config=PipelineConfig(),
                pipeline_steps=[
                    FunctionStep(
                        func=percentile_normalize,
                        name="Replacement",
                    )
                ],
            )
        )

        try:
            submission_service.start_completion_poller(
                "execution-1",
                plate_scope,
            )
            assert completion_poller.started.wait(timeout=2)

            document = bridge.get_document(
                UiCodeDocumentRequest(
                    document_id="window_code_document:pipeline_editor",
                    selection_mode=UiCodeDocumentSelectionMode.SELECTED.value,
                )
            )
            rejected = bridge.apply_document(
                UiCodeDocumentApplyRequest(
                    document_id=document.summary.identity.document_id,
                    source=replacement_source,
                    base_revision_token=document.current_revision_token,
                    confirmation_requirement=(
                        UiBridgeConfirmationRequirement.from_flag(False)
                    ),
                )
            )
            assert not rejected.applied
            assert [
                step.name
                for step in PipelineObjectStateBinding.steps_for_plate(plate_scope)
            ] == ["Original"]

            completion_poller.release.set()
            deadline = time.monotonic() + 5
            while (
                manager.execution_state is not ManagerExecutionState.IDLE
                or not presented_states
            ):
                QApplication.processEvents()
                if time.monotonic() >= deadline:
                    raise AssertionError(
                        "Asynchronous terminal completion did not finalize: "
                        f"state={manager.execution_state!r}, "
                        f"current_execution_id={manager.current_execution_id!r}, "
                        f"plate_execution_ids={manager.plate_execution_ids!r}, "
                        "terminal_statuses="
                        f"{manager.plate_terminal_activity_status.terminal_status_by_plate!r}, "
                        f"presented_states={presented_states!r}."
                    )

            assert presented_states == [(ManagerExecutionState.IDLE, None, False)]
            assert orchestrator.state is OrchestratorState.EXEC_FAILED

            terminal_document = bridge.get_document(
                UiCodeDocumentRequest(
                    document_id="window_code_document:pipeline_editor",
                    selection_mode=UiCodeDocumentSelectionMode.SELECTED.value,
                )
            )
            applied = bridge.apply_document(
                UiCodeDocumentApplyRequest(
                    document_id=terminal_document.summary.identity.document_id,
                    source=replacement_source,
                    base_revision_token=terminal_document.current_revision_token,
                    confirmation_requirement=(
                        UiBridgeConfirmationRequirement.from_flag(False)
                    ),
                )
            )
            assert applied.applied
            assert [
                step.name
                for step in PipelineObjectStateBinding.steps_for_plate(plate_scope)
            ] == ["Replacement"]
            assert orchestrator.state is OrchestratorState.READY
            assert plate_scope not in manager.plate_compiled_data
        finally:
            completion_poller.release.set()
            WindowManager.unregister(pipeline_scope)
            editor.close()
            close_widget(manager)
            ObjectStateRegistry.clear()

    def test_stop_completion_resets_force_kill_state_despite_stale_server_info(
        self,
    ) -> None:
        manager = PlateManagerWidget.__new__(PlateManagerWidget)
        manager._execution_state = ManagerExecutionState.IDLE
        manager.manager_execution_state_changed = (
            PlateManagerExecutionStateSignalRecorder()
        )
        manager.execution_state = ManagerExecutionState.FORCE_KILL_READY
        manager.current_execution_id = "execution-1"
        manager.execution_server_info = SimpleNamespace(
            running_execution_entries=("stale-running",),
            queued_execution_entries=(),
        )
        manager.plate_terminal_activity_status = ExecutionBatchRuntime()
        manager.plate_terminal_activity_status.begin_batch(("/plate",))
        manager.plate_terminal_activity_status.mark_terminal(
            "/plate",
            TerminalExecutionStatus.CANCELLED,
        )

        manager._maybe_reset_execution_state_after_stop()

        assert manager.execution_state is ManagerExecutionState.IDLE
        assert manager.current_execution_id is None


class PlatePipelineChangedSignalRecorder:
    """Signal-like recorder for pipeline_changed emissions."""

    def __init__(self) -> None:
        self.steps = None

    def emit(self, steps) -> None:
        self.steps = steps


class PlatePipelineStatusSignalRecorder:
    """Signal-like recorder for pipeline editor status messages."""

    def __init__(self) -> None:
        self.messages = []

    def emit(self, message: str) -> None:
        self.messages.append(message)


class OrchestratorStateHolder:
    """Small state alias matching PipelineOrchestrator's _state storage."""

    def __init__(self, state: OrchestratorState) -> None:
        self._state = state

    @property
    def state(self) -> OrchestratorState:
        return self._state


class PlatePipelineEditorRecorder:
    """Pipeline editor seam used by PlateManager auto-import tests."""

    def __init__(self, existing_steps: tuple[FunctionStep, ...] = ()) -> None:
        self.existing_steps = existing_steps
        self.pipeline_steps = []
        self.pipeline_changed = PlatePipelineChangedSignalRecorder()
        self.updated_pipeline = None
        self.current_plate = None
        self.status_message = PlatePipelineStatusSignalRecorder()

    @property
    def changed_steps(self):
        return self.pipeline_changed.steps

    def get_pipeline_for_plate(self, plate_path: str):
        return list(self.existing_steps)

    def update_pipeline_for_plate(self, plate_path: str, pipeline_steps) -> None:
        self.updated_pipeline = (plate_path, pipeline_steps)
        self.existing_steps = tuple(pipeline_steps)

    def update_item_list(self) -> None:
        return None


class PlateManagerCodeWorkflowHarness:
    """Minimal plate-manager surface consumed by PlateManagerCodeWorkflow."""

    def __init__(self, selected_plate_path: str | None = None) -> None:
        self.selected_plate_path = selected_plate_path
        self.pipeline_data_changed = PlatePipelineDataChangedSignalRecorder()
        self.orchestrator_state_changed = PlateOrchestratorStateChangedSignalRecorder()
        self.event_bus = PlateManagerEventBusRecorder()
        self.status_message = PlatePipelineStatusSignalRecorder()
        self.plate_compiled_data = {}
        self.compiled_state_emissions = []
        self.plate_execution_ids = {}
        self.plate_terminal_activity_status = ExecutionBatchRuntime()
        self.execution_state = ManagerExecutionState.IDLE

    def emit_compiled_state(self, plate_path: str, state) -> None:
        if state is None:
            self.plate_compiled_data.pop(plate_path, None)
        else:
            self.plate_compiled_data[plate_path] = state
        self.compiled_state_emissions.append((plate_path, state))

    def clear_plate_execution_tracking(
        self,
        plate_path: str,
        *,
        clear_terminal: bool = True,
    ) -> None:
        self.plate_execution_ids.pop(plate_path, None)
        self.plate_terminal_activity_status.clear_plate(
            plate_path,
            clear_terminal=clear_terminal,
        )

    def is_any_plate_running(self) -> bool:
        return self.execution_state is not ManagerExecutionState.IDLE

    def require_pipeline_definition_mutation_allowed(
        self,
        plate_path: str | None = None,
    ) -> None:
        del plate_path
        if self.is_any_plate_running():
            raise RuntimeError(
                "Pipeline definitions cannot change while plate execution is active."
            )


class PlatePipelineDataChangedSignalRecorder:
    """Signal-like recorder for plate-manager pipeline data changes."""

    def __init__(self) -> None:
        self.count = 0

    def emit(self) -> None:
        self.count += 1


class PlateManagerExecutionStateSignalRecorder:
    """Signal-like recorder for manager execution state changes."""

    def __init__(self) -> None:
        self.states = []

    def emit(self, state: ManagerExecutionState) -> None:
        self.states.append(state)


class PlateOrchestratorStateChangedSignalRecorder:
    """Signal-like recorder for orchestrator state changes."""

    def __init__(self) -> None:
        self.emissions = []

    def emit(self, plate_path: str, state: str) -> None:
        self.emissions.append((plate_path, state))


class PlateManagerEventBusRecorder:
    """Event-bus recorder for workflow pipeline-changed broadcasts."""

    def __init__(self) -> None:
        self.pipeline_steps = None

    def emit_pipeline_changed(self, pipeline_steps: list) -> None:
        self.pipeline_steps = pipeline_steps


@dataclass(frozen=True, slots=True)
class CellProfilerWorkspaceResultFixture:
    """Minimal CellProfiler workspace result carrying prepared pipeline steps."""

    @classmethod
    def with_steps(
        cls,
        steps,
        pipeline_config: PipelineConfig | None = None,
    ):
        if pipeline_config is None:
            pipeline_config = PipelineConfig()
        return InputWorkspacePreparationResult(
            original_source_root=Path("/source"),
            execution_plate_path=Path("/execution"),
            pipeline_path=Path("/source/pipeline.cppipe"),
            pipeline_steps=list(steps),
            pipeline_config=pipeline_config,
        )
