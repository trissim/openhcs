from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtWidgets import QApplication, QListWidget, QPushButton
from pyqt_reactive.theming import ColorScheme

from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.input_workspace import InputWorkspacePreparationResult
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.steps.function_step import FunctionStep
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.pyqt_gui.services.service_adapter import GlobalEventBus
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget


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

    def get_current_color_scheme(self) -> ColorScheme:
        return self.color_scheme

    def get_event_bus(self) -> GlobalEventBus:
        return self.event_bus


class PlateManagerWidgetTestHarness:
    """Nominal owner for headless PlateManager test setup."""

    @staticmethod
    def widget(monkeypatch) -> PlateManagerWidget:
        QtApplicationHarness.app()
        monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
        monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
        monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
        return PlateManagerWidget(PlateManagerServiceStub())


class TestPlateManagerWidget:
    def test_constructor_initializes_qobject_before_signal_use(self, monkeypatch) -> None:
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)

        assert widget.debug_snapshot_available is not None
        close_widget(widget)

    def test_loads_cellprofiler_pipeline_into_empty_plate(self, monkeypatch) -> None:
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        pipeline_editor = PlatePipelineEditorRecorder()
        widget.pipeline_editor = pipeline_editor
        widget.selected_plate_path = "/plate"

        widget._load_cellprofiler_pipeline_for_empty_plate(
            "/plate",
            CellProfilerWorkspaceResultFixture.with_steps(
                (FunctionStep(func=lambda image: image, name="Imported"),)
            ),
        )

        assert pipeline_editor.updated_pipeline == (
            "/plate",
            pipeline_editor.pipeline_steps,
        )
        assert len(pipeline_editor.pipeline_steps) == 1
        assert pipeline_editor.changed_steps == pipeline_editor.pipeline_steps
        assert pipeline_editor.source_binding_context is not None
        assert pipeline_editor.source_binding_context.logical_plate_id == "/plate"
        assert pipeline_editor.source_binding_context.display_plate_root == Path(
            "/source"
        )
        assert pipeline_editor.source_binding_context.execution_plate_path == Path(
            "/execution"
        )
        close_widget(widget)

    def test_keeps_existing_pipeline_for_cellprofiler_plate(
        self,
        monkeypatch,
    ) -> None:
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        pipeline_editor = PlatePipelineEditorRecorder(
            existing_steps=(FunctionStep(func=lambda image: image, name="Existing"),)
        )
        widget.pipeline_editor = pipeline_editor

        widget._load_cellprofiler_pipeline_for_empty_plate(
            "/plate",
            CellProfilerWorkspaceResultFixture.with_steps(
                (FunctionStep(func=lambda image: image, name="Imported"),)
            ),
        )

        assert pipeline_editor.updated_pipeline is None
        assert pipeline_editor.changed_steps is None
        close_widget(widget)

    def test_plate_selection_seeds_cellprofiler_pipeline_before_editor_signal(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        ObjectStateRegistry.clear()
        widget = PlateManagerWidgetTestHarness.widget(monkeypatch)
        pipeline_editor = PlatePipelineEditorRecorder()
        widget.pipeline_editor = pipeline_editor
        service_adapter = PlateManagerServiceStub()
        ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
        plate_root = tmp_path / "plate"
        plate_root.mkdir()
        plate_scope = f"{plate_root}::cppipe::second"
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
                pipeline_editor.updated_pipeline
            )
        )

        widget.selected_plate_path = plate_scope
        widget.plate_selected.emit(plate_scope)

        assert signal_observations == [
            (plate_scope, pipeline_editor.pipeline_steps)
        ]
        assert len(pipeline_editor.pipeline_steps) == 1
        close_widget(widget)
        ObjectStateRegistry.clear()


class PlatePipelineChangedSignalRecorder:
    """Signal-like recorder for pipeline_changed emissions."""

    def __init__(self) -> None:
        self.steps = None

    def emit(self, steps) -> None:
        self.steps = steps


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

    @property
    def changed_steps(self):
        return self.pipeline_changed.steps

    def get_pipeline_for_plate(self, plate_path: str):
        return list(self.existing_steps)

    def update_pipeline_for_plate(self, plate_path: str, pipeline_steps) -> None:
        self.updated_pipeline = (plate_path, pipeline_steps)

    def set_source_binding_context_for_plate(
        self,
        plate_path: str,
        context: SourceBindingContext,
    ) -> None:
        self.source_binding_context = context

    def update_item_list(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class CellProfilerWorkspaceResultFixture:
    """Minimal CellProfiler workspace result carrying prepared pipeline steps."""

    @classmethod
    def with_steps(cls, steps):
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
