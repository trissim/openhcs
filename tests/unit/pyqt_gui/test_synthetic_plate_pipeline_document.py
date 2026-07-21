from __future__ import annotations

from pathlib import Path

from openhcs.core.config import PipelineConfig
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.widgets.shared.services.pipeline_editor_workflows import (
    PipelineEditorCodeWorkflow,
)
from openhcs.pyqt_gui.windows import synthetic_plate_generator_window
from openhcs.pyqt_gui.windows.synthetic_plate_generator_window import (
    SyntheticPlateGeneratorWindow,
)
from pyqt_reactive.services.service_registry import ServiceRegistry


class _SignalRecorder:
    def __init__(self) -> None:
        self.emissions: list[tuple[object, ...]] = []

    def emit(self, *values: object) -> None:
        self.emissions.append(values)


class _SyntheticPlateGenerationHarness:
    def __init__(self, output_dir: Path) -> None:
        self.state = self
        self.output_dir = str(output_dir)
        self.plate_generated = _SignalRecorder()
        self.accepted = False

    @staticmethod
    def get_current_values() -> dict[str, object]:
        return {}

    def accept(self) -> None:
        self.accepted = True


class _PipelineEditorHarness:
    def __init__(self) -> None:
        self.current_plate = ""
        self.plate_manager = _PlateManagerHarness()
        self.pipeline_steps = []
        self.pipeline_changed = _SignalRecorder()
        self.status_message = _SignalRecorder()
        self.event_bus = None
        self.updated_plate_steps: list[tuple[str, list[object]]] = []
        self.changed_plates: list[str] = []
        self.item_list_updates = 0
        self.applied = False

    def _handle_edited_code(self, source: str) -> None:
        namespace: dict[str, object] = {}
        exec(compile(source, "<synthetic-plate-pipeline>", "exec"), namespace)
        self.applied = PipelineEditorCodeWorkflow(self).apply_namespace(namespace)

    def _normalize_step_scope_tokens(self, *, register: bool) -> None:
        assert register is False

    def update_pipeline_for_plate(self, plate_path: str, steps: list[object]) -> None:
        self.updated_plate_steps.append((plate_path, list(steps)))

    def notify_pipeline_definition_changed(self, plate_path: str) -> None:
        self.changed_plates.append(plate_path)

    def update_item_list(self) -> None:
        self.item_list_updates += 1


class _PlateManagerHarness:
    def __init__(self) -> None:
        self.plate_configs: dict[str, PipelineConfig] = {}
        self.event_bus = None


class _MainWindowHarness:
    def __init__(self) -> None:
        self.pipeline_editor_shown = 0

    def show_pipeline_editor(self) -> None:
        self.pipeline_editor_shown += 1


def test_synthetic_plate_generation_emits_complete_pipeline_document(
    monkeypatch,
    tmp_path: Path,
) -> None:
    generated_parameters: list[dict[str, object]] = []

    class _Generator:
        def __init__(self, **parameters: object) -> None:
            generated_parameters.append(parameters)

        def generate_dataset(self) -> None:
            return None

    monkeypatch.setattr(
        synthetic_plate_generator_window,
        "SyntheticMicroscopyGenerator",
        _Generator,
    )
    harness = _SyntheticPlateGenerationHarness(tmp_path)

    SyntheticPlateGeneratorWindow.generate_plate(harness)

    assert harness.accepted is True
    assert generated_parameters == [{"output_dir": str(tmp_path)}]
    assert len(harness.plate_generated.emissions) == 1
    output_dir, pipeline_path = harness.plate_generated.emissions[0]
    assert output_dir == str(tmp_path)
    document = PipelineDocumentAuthority.from_source(Path(pipeline_path).read_text())
    assert isinstance(document.pipeline_config, PipelineConfig)
    assert len(document.pipeline_steps) == 8


def test_main_window_loads_emitted_synthetic_pipeline_document(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from openhcs.tests import test_pipeline

    editor = _PipelineEditorHarness()
    monkeypatch.setattr(
        ServiceRegistry,
        "get",
        classmethod(lambda cls, service_type: editor),
    )
    main_window = _MainWindowHarness()
    plate_path = str(tmp_path / "plate")

    OpenHCSMainWindow._load_pipeline_file(
        main_window,
        str(Path(test_pipeline.__file__)),
        plate_path=plate_path,
    )

    assert main_window.pipeline_editor_shown == 1
    assert editor.current_plate == plate_path
    assert editor.applied is True
    assert isinstance(editor.plate_manager.plate_configs[plate_path], PipelineConfig)
    assert isinstance(
        PipelineDocumentAuthority.from_source(
            Path(test_pipeline.__file__).read_text()
        ).pipeline_config,
        PipelineConfig,
    )
    assert len(editor.pipeline_steps) == 8
    assert editor.updated_plate_steps == [(plate_path, editor.pipeline_steps)]
    assert editor.changed_plates == [plate_path]
    assert editor.item_list_updates == 1
