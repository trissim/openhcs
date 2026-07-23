from types import SimpleNamespace

from openhcs.constants.constants import OrchestratorState
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.textual_tui.services.plate_manager_code_document import (
    TextualPlateManagerCodeDocumentController,
)
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
)


class FakeOrchestrator:
    def __init__(self, pipeline_config: PipelineConfig) -> None:
        self.pipeline_config = pipeline_config
        self.state = OrchestratorState.READY
        self.applied_configs = []

    def apply_pipeline_config(self, pipeline_config: PipelineConfig) -> None:
        self.pipeline_config = pipeline_config
        self.applied_configs.append(pipeline_config)


class FakePipelineEditor:
    def __init__(self) -> None:
        self.plate_pipelines = {"/plate-a": []}
        self.current_plate = "/plate-a"
        self.visible_steps = None

    def get_pipeline_for_plate(self, plate_path: str):
        return self.plate_pipelines[plate_path]

    def _set_pipeline_steps_without_save(self, steps) -> None:
        self.visible_steps = list(steps)


class TextualPlateManagerHarness:
    def __init__(self) -> None:
        self.app = SimpleNamespace(
            global_config=GlobalPipelineConfig(num_workers=1),
            current_status="",
        )
        self.global_config = self.app.global_config
        self.pipeline_editor = FakePipelineEditor()
        self.orchestrators = {
            "/plate-a": FakeOrchestrator(PipelineConfig(num_workers=2))
        }
        self.plate_configs = {}
        self.items = [{"name": "plate-a", "path": "/plate-a"}]
        self.selected_plate = "/plate-a"
        self.plate_compiled_data = {"/plate-a": object()}
        self.refresh_count = 0

    def _trigger_ui_refresh(self) -> None:
        self.refresh_count += 1

    def _update_button_states(self) -> None:
        pass


def test_textual_plate_manager_collects_and_applies_complete_document_atomically():
    manager = TextualPlateManagerHarness()
    original_orchestrator = manager.orchestrators["/plate-a"]
    controller = TextualPlateManagerCodeDocumentController(manager)
    payload = controller.payload(manager.items)
    source = PlateManagerCodeDocumentAuthority.render(payload)
    assert "per_plate_configs =" in source

    new_global_config = GlobalPipelineConfig(num_workers=8)
    new_plate_config = PipelineConfig(num_workers=5)
    edited_payload = PlateManagerCodeDocumentAuthority.from_values(
        plate_paths=("/plate-a", "/plate-b"),
        global_pipeline_config=new_global_config,
        per_plate_configs={
            "/plate-a": new_plate_config,
            "/plate-b": PipelineConfig(num_workers=6),
        },
        pipeline_data={"/plate-a": [], "/plate-b": []},
    )

    controller.apply(edited_payload)

    assert manager.app.global_config == new_global_config
    assert manager.global_config == new_global_config
    assert manager.plate_configs == edited_payload.per_plate_configs
    assert [item["path"] for item in manager.items] == ["/plate-a", "/plate-b"]
    assert manager.pipeline_editor.plate_pipelines == edited_payload.pipeline_data
    assert original_orchestrator.applied_configs == [new_plate_config]
    assert manager.plate_compiled_data == {}
    assert manager.refresh_count == 1


def test_textual_plate_manager_empty_document_clears_selection_and_visible_steps():
    manager = TextualPlateManagerHarness()
    controller = TextualPlateManagerCodeDocumentController(manager)
    payload = PlateManagerCodeDocumentAuthority.from_values(
        plate_paths=(),
        global_pipeline_config=GlobalPipelineConfig(num_workers=3),
        per_plate_configs={},
        pipeline_data={},
    )

    controller.apply(payload)

    assert manager.items == []
    assert manager.orchestrators == {}
    assert manager.selected_plate == ""
    assert manager.pipeline_editor.current_plate == ""
    assert manager.pipeline_editor.visible_steps == []
