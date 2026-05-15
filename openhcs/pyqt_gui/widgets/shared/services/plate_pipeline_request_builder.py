"""Build plate-scoped compile/run requests for PyQt batch workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

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

    global_config: Any


class PlatePipelineRequestBuilder:
    """Builds validated plate pipeline requests from the host GUI state."""

    def __init__(self, host) -> None:
        self._host = host

    def build_compile_job_from_plate_data(
        self,
        plate_data: Dict[str, Any],
    ) -> CompileJob:
        plate_path = str(plate_data["path"])
        definition_pipeline = self._definition_pipeline_for_plate(
            plate_path=plate_path,
            display_name=str(plate_data["name"]),
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
            plate_path=plate_path,
            plate_name=str(plate_data["name"]),
            definition_pipeline=definition_pipeline,
            pipeline_config=pipeline_config,
        )

    def build_run_spec(self, plate_path: str) -> RunSpec:
        resolved_plate_path = str(plate_path)
        definition_pipeline = self._definition_pipeline_for_plate(
            plate_path=resolved_plate_path,
            display_name=resolved_plate_path,
        )
        pipeline_config = resolve_pipeline_config_for_plate(
            self._host,
            resolved_plate_path,
        )
        return RunSpec(
            plate_path=resolved_plate_path,
            definition_pipeline=definition_pipeline,
            global_config=self._host.global_config,
            pipeline_config=pipeline_config,
        )

    @staticmethod
    def compile_job_from_run_spec(
        run_spec: RunSpec,
        *,
        config_params: dict[str, Any] | None = None,
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
            plate_path=plate_path,
            plate_name=plate_path,
            definition_pipeline=definition_pipeline,
            pipeline_config=run_spec.pipeline_config,
            config_params=config_params,
        )

    def _definition_pipeline_for_plate(
        self,
        *,
        plate_path: str,
        display_name: str,
    ) -> List:
        definition_pipeline = self._host.get_pipeline_definition(plate_path)
        if not definition_pipeline:
            logger.warning(
                "No pipeline defined for %s, using empty pipeline",
                display_name,
            )
            definition_pipeline = []
        self.validate_pipeline_steps(definition_pipeline)
        return definition_pipeline

    @staticmethod
    def validate_pipeline_steps(pipeline: List) -> None:
        for step in pipeline:
            if step.func is None:
                raise AttributeError(
                    f"Step '{step.name}' has func=None. "
                    "This usually means the pipeline was loaded from a compiled state."
                )


def is_plate_pipeline_request_builder_export(name: str, value: object) -> bool:
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
