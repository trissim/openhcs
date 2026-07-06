from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtWidgets import QApplication, QListWidget, QPushButton
from pyqt_reactive.theming import ColorScheme

import openhcs.processing.backends.cellprofiler as cellprofiler_backend
from openhcs.core.artifacts import ArtifactSpec, ImageArtifactType
from openhcs.core.callable_contract import CallableContract
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyNapariStreamingConfig,
    PipelineConfig,
    WellFilterConfig,
)
from openhcs.constants.constants import OrchestratorState
from openhcs.core.input_workspace import InputWorkspacePreparationResult
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    RecordedArtifactOutputPartition,
    RuntimeArtifactInputPartition,
    SourceArtifactInputPartition,
)
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.source_bindings import (
    LazyStepSourceBindingsConfig,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingsConfig,
    SourceSelector,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.pipeline_object_state_binding import (
    PipelineObjectStateBinding,
)
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.services.service_adapter import GlobalEventBus
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.pyqt_gui.widgets.plate_manager import (
    PlateManagerAction,
    PlateManagerWidget,
    PlateOperation,
    PlateOperationValidator,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    cellprofiler_module_callable,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_manager_workflows import (
    PlateManagerCodeWorkflow,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ExecutionBatchRuntime,
    ManagerExecutionState,
    TerminalExecutionStatus,
)


class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


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
        self.global_config = GlobalPipelineConfig()
        self.color_scheme = ColorScheme()
        self.event_bus = GlobalEventBus()

    def get_global_config(self) -> GlobalPipelineConfig:
        return self.global_config

    def set_global_config(self, global_config: GlobalPipelineConfig) -> None:
        self.global_config = global_config

    def get_current_color_scheme(self) -> ColorScheme:
        return self.color_scheme

    def get_event_bus(self) -> GlobalEventBus:
        return self.event_bus

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
        return PlateManagerWidget(PlateManagerServiceStub())


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
        context = widget.source_binding_context_for_plate("/plate")
        assert context is not None
        assert context.logical_plate_id == "/plate"
        assert context.display_plate_root == Path("/source")
        assert context.execution_plate_path == Path("/execution")
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

    def test_live_results_prefers_output_plate_orchestrator(
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
            result = widget._live_results_viewer_orchestrator_for_row(source_row)

            assert result is output_orchestrator
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_live_results_allows_selected_output_plate_created_state(
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
            result = widget._live_results_viewer_orchestrator_for_row(output_row)

            assert result is output_orchestrator
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
            source_bindings_config=SourceBindingsConfig(match_plan=match_plan),
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
            binding = PipelineObjectStateBinding.for_plate(plate_scope)
            assert plate_state is not None
            assert binding is not None
            assert widget.plate_configs[plate_scope] == pipeline_config
            assert (
                plate_state.get_saved_resolved_value(
                    "source_bindings_config.match_plan"
                )
                == match_plan
            )

            step_scope_id = binding.step_scope_ids[0]
            step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
            assert step_state is not None
            snapshot = StepSnapshot.from_resolved_step(
                index=0,
                step=step_state.to_object(),
                step_state=step_state,
            )
            assert snapshot.source_bindings.match_plan == match_plan
        finally:
            close_widget(widget)
            ObjectStateRegistry.clear()

    def test_plate_selection_seeds_cellprofiler_pipeline_before_editor_signal(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        service_adapter = PlateManagerServiceStub()
        ensure_global_config_context(
            GlobalPipelineConfig, service_adapter.global_config
        )
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            plate_root / "second.cppipe",
        ).scope_id
        orchestrator = PipelineOrchestrator(plate_path=plate_root)
        orchestrator.input_workspace_preparation_result = (
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

        widget.selected_plate_path = plate_scope
        widget.plate_selected.emit(plate_scope)

        assert signal_observations
        assert [step.name for step in signal_observations[0]] == ["Second"]
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

    def test_code_mode_omits_default_orchestrator_per_plate_config(
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

            assert "per_plate_configs = {}" in context.source
            assert "per_plate_configs = {}\n\npipeline_data" in context.source
            assert context.payload.per_plate_configs == {}
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

    def test_plate_manager_code_mode_rebinds_cellprofiler_runtime_callables(
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
        crop_metadata = cellprofiler_backend.CellProfilerFunctionCatalog.runtime_metadata(
            cellprofiler_backend.crop,
        )
        contract = ModuleArtifactContract(
            module_name=crop_metadata.module_name,
            items=(
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("CropBlue", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("CropBlue", ImageArtifactType),),
                ),
            ),
        )
        import_result = SimpleNamespace(
            generated_module_name="generated_cellprofiler_pipeline",
            provenance=SimpleNamespace(
                processing_modules=(SimpleNamespace(module_num=1),),
            ),
            artifact_contracts=(contract,),
        )
        manager = PlateManagerCodeWorkflowHarness(selected_plate_path=plate_scope)
        manager._cellprofiler_import_results_by_plate[plate_scope] = import_result
        raw_step = FunctionStep(
            func=(cellprofiler_backend.crop, {"crop_shape": "Rectangle"}),
            name="Crop",
        )

        PlateManagerCodeWorkflow(manager).apply_pipeline_data({plate_scope: [raw_step]})

        updated_steps = PipelineObjectStateBinding.steps_for_plate(plate_scope)
        rebound_func = updated_steps[0].func[0]
        rebound_contract = CallableContract.from_callable(
            rebound_func,
        ).module_artifact_contract
        assert rebound_func.__name__ == "crop"
        assert rebound_contract == contract

    def test_plate_manager_code_mode_keeps_bound_cellprofiler_steps_without_import_context(
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
        crop_metadata = cellprofiler_backend.CellProfilerFunctionCatalog.runtime_metadata(
            cellprofiler_backend.crop,
        )
        contract = ModuleArtifactContract(
            module_name=crop_metadata.module_name,
            items=(
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition,
                    (ArtifactSpec.output("CropBlue", ImageArtifactType),),
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition,
                    (ArtifactSpec.output("CropBlue", ImageArtifactType),),
                ),
            ),
        )
        bound_step = FunctionStep(
            func=(
                cellprofiler_module_callable(
                    cellprofiler_backend.crop,
                    contract,
                    processing_contract=crop_metadata.processing_contract,
                    declared_processing_contract=(
                        crop_metadata.declared_processing_contract
                    ),
                ),
                {"crop_shape": "Rectangle"},
            ),
            name="Crop",
        )
        manager = PlateManagerCodeWorkflowHarness(selected_plate_path=plate_scope)

        PlateManagerCodeWorkflow(manager).apply_pipeline_data(
            {plate_scope: [bound_step]}
        )

        updated_steps = PipelineObjectStateBinding.steps_for_plate(plate_scope)
        updated_func = updated_steps[0].func[0]
        updated_contract = CallableContract.from_callable(
            updated_func,
        ).module_artifact_contract
        assert updated_contract == contract

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
        orchestrator = OrchestratorStateHolder(OrchestratorState.COMPLETED)
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
                    ],
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
        self.cellprofiler_import_result = None
        self.cellprofiler_import_results_by_plate = {}
        self.source_binding_context = None
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

    def refresh_loaded_pipeline_for_plate(
        self,
        plate_path: str,
        import_result,
        pipeline_steps,
    ) -> None:
        if self.current_plate != plate_path:
            return
        self.cellprofiler_import_result = import_result
        self.pipeline_steps = pipeline_steps
        self.pipeline_changed.emit(pipeline_steps)

    def set_source_binding_context_for_plate(
        self,
        plate_path: str,
        context: SourceBindingContext,
    ) -> None:
        self.source_binding_context = context

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
        self.plate_execution_ids = {}
        self.plate_terminal_activity_status = ExecutionBatchRuntime()
        self._cellprofiler_import_results_by_plate = {}

    def cellprofiler_import_result_for_plate(self, plate_path: str):
        return self._cellprofiler_import_results_by_plate.get(str(plate_path))

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
    def with_steps(cls, steps, pipeline_config: PipelineConfig | None = None):
        return InputWorkspacePreparationResult(
            original_source_root=Path("/source"),
            execution_plate_path=Path("/execution"),
            pipeline_path=Path("/source/pipeline.cppipe"),
            source_schema=PipelineImageSchema.empty(),
            prepared_pipeline=CellProfilerPreparedPipelineFixture(
                pipeline=CellProfilerPipelineFixture(steps=steps),
                import_result=SimpleNamespace(
                    pipeline=CellProfilerPipelineFixture(steps=steps),
                    source_schema=PipelineImageSchema.empty(),
                    pipeline_config=pipeline_config,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class CellProfilerPreparedPipelineFixture:
    pipeline: object
    import_result: object


@dataclass(frozen=True, slots=True)
class CellProfilerPipelineFixture:
    steps: tuple[FunctionStep, ...]
