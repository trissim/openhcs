"""Textual adapter for the framework-neutral PlateManager document."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.constants.constants import OrchestratorState
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
    PlateManagerOrchestratorCodePayload,
)
from openhcs.ui.shared.plate_scope_identity import PlateScopeIdentity


@dataclass(frozen=True, slots=True)
class TextualPlateManagerCodeDocumentController:
    """Collect and atomically apply PlateManager documents for Textual."""

    manager: object

    def payload(
        self, selected_items: list[dict]
    ) -> PlateManagerOrchestratorCodePayload:
        pipeline_editor = self.manager.pipeline_editor
        if pipeline_editor is None:
            raise RuntimeError(
                "PlateManager code documents require a connected PipelineEditor."
            )
        if not isinstance(self.manager.app.global_config, GlobalPipelineConfig):
            raise TypeError(
                "The Textual app global config must be GlobalPipelineConfig."
            )

        plate_paths = [str(item["path"]) for item in selected_items]
        per_plate_configs: dict[str, PipelineConfig] = {}
        pipeline_data = {}
        for plate_path in plate_paths:
            orchestrator = self.manager.orchestrators[plate_path]
            pipeline_config = orchestrator.pipeline_config
            if not isinstance(pipeline_config, PipelineConfig):
                raise TypeError(
                    f"Plate {plate_path!r} must own a PipelineConfig before code mode."
                )
            per_plate_configs[plate_path] = pipeline_config
            pipeline_data[plate_path] = pipeline_editor.get_pipeline_for_plate(
                plate_path
            )

        return PlateManagerCodeDocumentAuthority.from_values(
            plate_paths=plate_paths,
            global_pipeline_config=self.manager.app.global_config,
            per_plate_configs=per_plate_configs,
            pipeline_data=pipeline_data,
        )

    def apply(self, payload: PlateManagerOrchestratorCodePayload) -> None:
        pipeline_editor = self.manager.pipeline_editor
        if pipeline_editor is None:
            raise RuntimeError(
                "PlateManager code documents require a connected PipelineEditor."
            )
        if any(
            orchestrator.state is OrchestratorState.EXECUTING
            for orchestrator in self.manager.orchestrators.values()
        ):
            raise RuntimeError(
                "PlateManager code documents cannot be applied during execution."
            )

        payload = PlateManagerCodeDocumentAuthority.from_values(
            plate_paths=payload.plate_paths,
            global_pipeline_config=payload.global_pipeline_config,
            per_plate_configs=payload.per_plate_configs,
            pipeline_data=payload.pipeline_data,
        )

        retained_orchestrators = {
            plate_path: self.manager.orchestrators[plate_path]
            for plate_path in payload.plate_paths
            if plate_path in self.manager.orchestrators
        }
        for plate_path, orchestrator in retained_orchestrators.items():
            orchestrator.apply_pipeline_config(payload.per_plate_configs[plate_path])

        plate_items = [
            {
                "name": PlateScopeIdentity.from_scope_id(plate_path).display_name,
                "path": plate_path,
            }
            for plate_path in payload.plate_paths
        ]
        selected_plate = (
            self.manager.selected_plate
            if self.manager.selected_plate in payload.plate_paths
            else (payload.plate_paths[0] if payload.plate_paths else "")
        )

        self.manager.global_config = payload.global_pipeline_config
        self.manager.app.global_config = payload.global_pipeline_config
        self.manager.orchestrators = retained_orchestrators
        self.manager.plate_configs = dict(payload.per_plate_configs)
        self.manager.items = plate_items
        self.manager.plate_compiled_data = {}
        pipeline_editor.plate_pipelines = {
            plate_path: list(steps)
            for plate_path, steps in payload.pipeline_data.items()
        }
        self.manager.selected_plate = selected_plate
        pipeline_editor.current_plate = selected_plate
        visible_steps = payload.pipeline_data[selected_plate] if selected_plate else []
        pipeline_editor._set_pipeline_steps_without_save(visible_steps)
        self.manager._trigger_ui_refresh()
        self.manager._update_button_states()
        self.manager.app.current_status = (
            f"Applied PlateManager document for {len(payload.plate_paths)} plates"
        )
