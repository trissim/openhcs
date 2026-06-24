"""Build plate-scoped compile/run requests for PyQt batch workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openhcs.core.config import GlobalPipelineConfig
from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileJob,
    CompileWorkflowService,
    PlatePipelineRequest,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_config_resolver import (
    resolve_pipeline_config_for_plate,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSpec(PlatePipelineRequest):
    """Execution request assembled from one plate and its resolved config."""

    global_config: GlobalPipelineConfig


class PlatePipelineRequestBuilder:
    """Builds validated plate pipeline requests from the host GUI state."""

    def __init__(self, host) -> None:
        self._host = host

    def build_compile_job_from_plate_row(
        self,
        row: PlateManagerRow,
    ) -> CompileJob:
        plate_path = row.scope_id
        execution_plate_path = self._execution_plate_path_for_scope(plate_path)
        selected_pipeline_path = self._selected_pipeline_path_for_scope(plate_path)
        definition_pipeline = self._definition_pipeline_for_plate(
            plate_path=plate_path,
            display_name=row.name,
        )
        pipeline_config = resolve_pipeline_config_for_plate(self._host, plate_path)
        logger.info(
            "Compile snapshot: plate=%s steps=%d fingerprint=%s step_names=%s",
            plate_path,
            len(definition_pipeline),
            CompileWorkflowService.pipeline_fingerprint(definition_pipeline),
            CompileWorkflowService.pipeline_step_names(definition_pipeline),
        )
        return CompileJob(
            plate_scope=row.identity,
            execution_plate_path=execution_plate_path,
            selected_pipeline_path=selected_pipeline_path,
            plate_name=row.name,
            definition_pipeline=definition_pipeline,
            pipeline_config=pipeline_config,
        )

    def build_run_spec(self, plate_path: str) -> RunSpec:
        resolved_plate_path = str(plate_path)
        execution_plate_path = self._execution_plate_path_for_scope(
            resolved_plate_path
        )
        selected_pipeline_path = self._selected_pipeline_path_for_scope(
            resolved_plate_path
        )
        definition_pipeline = self._definition_pipeline_for_plate(
            plate_path=resolved_plate_path,
            display_name=resolved_plate_path,
        )
        pipeline_config = resolve_pipeline_config_for_plate(
            self._host,
            resolved_plate_path,
        )
        return RunSpec(
            plate_scope=PlateScopeIdentity.from_scope_id(resolved_plate_path),
            execution_plate_path=execution_plate_path,
            selected_pipeline_path=selected_pipeline_path,
            definition_pipeline=definition_pipeline,
            global_config=self._host.global_config,
            pipeline_config=pipeline_config,
        )

    @staticmethod
    def compile_job_from_run_spec(
        run_spec: RunSpec,
        *,
        config_params: dict | None = None,
    ) -> CompileJob:
        plate_path = run_spec.plate_path
        definition_pipeline = run_spec.definition_pipeline
        logger.info(
            "Compile-before-run snapshot: plate=%s steps=%d fingerprint=%s step_names=%s",
            plate_path,
            len(definition_pipeline),
            CompileWorkflowService.pipeline_fingerprint(definition_pipeline),
            CompileWorkflowService.pipeline_step_names(definition_pipeline),
        )
        return CompileJob(
            plate_scope=run_spec.plate_scope,
            execution_plate_path=run_spec.execution_plate_path,
            selected_pipeline_path=run_spec.selected_pipeline_path,
            plate_name=plate_path,
            definition_pipeline=definition_pipeline,
            pipeline_config=run_spec.pipeline_config,
            config_params=config_params,
        )

    @staticmethod
    def _execution_plate_path_for_scope(plate_path: str) -> str:
        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if not isinstance(orchestrator, PipelineOrchestrator):
            raise RuntimeError(
                f"No PipelineOrchestrator registered for plate scope {plate_path!r}."
            )
        input_workspace = orchestrator.input_workspace_preparation_result
        if input_workspace is not None:
            return str(input_workspace.execution_plate_path)
        if orchestrator.plate_path is None:
            raise RuntimeError(
                f"PipelineOrchestrator for plate scope {plate_path!r} has no execution plate path."
            )
        return str(orchestrator.plate_path)

    @staticmethod
    def _selected_pipeline_path_for_scope(plate_path: str) -> str | None:
        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if not isinstance(orchestrator, PipelineOrchestrator):
            raise RuntimeError(
                f"No PipelineOrchestrator registered for plate scope {plate_path!r}."
            )
        input_workspace = orchestrator.input_workspace_preparation_result
        if input_workspace is not None and input_workspace.pipeline_path is not None:
            return str(input_workspace.pipeline_path)
        request = orchestrator.input_workspace_preparation
        if request is not None and request.selected_pipeline_path is not None:
            return str(request.selected_pipeline_path)
        return None

    def _definition_pipeline_for_plate(
        self,
        *,
        plate_path: str,
        display_name: str,
    ) -> list:
        definition_pipeline = self._host.get_pipeline_definition(plate_path)
        if not definition_pipeline:
            logger.warning(
                "No pipeline defined for %s, using empty pipeline",
                display_name,
            )
            definition_pipeline = []
        definition_pipeline = CompileWorkflowService.normalize_pipeline_for_transport(
            definition_pipeline
        )
        self.validate_pipeline_steps(definition_pipeline)
        return definition_pipeline

    @staticmethod
    def validate_pipeline_steps(pipeline: list) -> None:
        for step in pipeline:
            if step.func is None:
                raise AttributeError(
                    f"Step '{step.name}' has func=None. "
                    "This usually means the pipeline was loaded from a compiled state."
                )


def is_plate_pipeline_request_builder_export(name: str, value) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_plate_pipeline_request_builder_export(name, value)
)
