"""Input workspace preparation contracts owned by orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openhcs.core.config import PipelineConfig
from openhcs.core.source_binding_workspace import (
    SourceBindingWorkspaceMaterialization,
)
from openhcs.core.steps.function_step import FunctionStep


@dataclass(frozen=True, slots=True)
class PipelineImportDiagnostic:
    """Non-fatal diagnostic from importing an external pipeline dialect."""

    pipeline_path: Path
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pipeline_path", Path(self.pipeline_path))


@dataclass(frozen=True, slots=True)
class InputWorkspacePreparationRequest:
    """Request to prepare a selected input tree before microscope initialization."""

    selected_path: Path
    selected_pipeline_path: Path | None = None
    workspace_root: Path | None = None
    generated_source_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_path", Path(self.selected_path))
        if self.selected_pipeline_path is not None:
            object.__setattr__(
                self,
                "selected_pipeline_path",
                Path(self.selected_pipeline_path),
            )
        if self.workspace_root is not None:
            object.__setattr__(self, "workspace_root", Path(self.workspace_root))
        if self.generated_source_path is not None:
            object.__setattr__(
                self,
                "generated_source_path",
                Path(self.generated_source_path),
            )


@dataclass(frozen=True, slots=True)
class InputWorkspacePreparationResult:
    """Prepared input workspace plus optional external pipeline import product."""

    original_source_root: Path
    execution_plate_path: Path
    pipeline_path: Path | None = None
    pipeline_steps: list[FunctionStep] | None = None
    pipeline_config: PipelineConfig | None = None
    materialization: SourceBindingWorkspaceMaterialization | None = None
    pipeline_import_error: PipelineImportDiagnostic | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_source_root", Path(self.original_source_root))
        object.__setattr__(self, "execution_plate_path", Path(self.execution_plate_path))
        if self.pipeline_path is not None:
            object.__setattr__(self, "pipeline_path", Path(self.pipeline_path))
        if (self.pipeline_steps is None) is not (self.pipeline_config is None):
            raise ValueError(
                "InputWorkspacePreparationResult requires pipeline_steps and "
                "pipeline_config together."
            )
        if self.pipeline_steps is not None:
            pipeline_steps = list(self.pipeline_steps)
            for step in pipeline_steps:
                if not isinstance(step, FunctionStep):
                    raise TypeError(
                        "InputWorkspacePreparationResult.pipeline_steps must "
                        f"contain FunctionStep values, got {type(step).__name__}."
                    )
            object.__setattr__(self, "pipeline_steps", pipeline_steps)
        if self.pipeline_config is not None and not isinstance(
            self.pipeline_config,
            PipelineConfig,
        ):
            raise TypeError(
                "InputWorkspacePreparationResult.pipeline_config must be "
                f"PipelineConfig, got {type(self.pipeline_config).__name__}."
            )
        if self.materialization is not None and not isinstance(
            self.materialization,
            SourceBindingWorkspaceMaterialization,
        ):
            raise TypeError(
                "InputWorkspacePreparationResult.materialization must be "
                "SourceBindingWorkspaceMaterialization, got "
                f"{type(self.materialization).__name__}."
            )
