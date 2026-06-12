"""Input workspace preparation contracts owned by orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openhcs.core.pipeline_image_schema import PipelineImageSchema


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
    generated_pipeline_path: Path | None = None

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
        if self.generated_pipeline_path is not None:
            object.__setattr__(
                self,
                "generated_pipeline_path",
                Path(self.generated_pipeline_path),
            )


@dataclass(frozen=True, slots=True)
class InputWorkspacePreparationResult:
    """Prepared input workspace plus optional external pipeline import product."""

    original_source_root: Path
    execution_plate_path: Path
    pipeline_path: Path | None = None
    source_schema: PipelineImageSchema | None = None
    materialization: Any | None = None
    prepared_pipeline: Any | None = None
    pipeline_import_error: PipelineImportDiagnostic | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_source_root", Path(self.original_source_root))
        object.__setattr__(self, "execution_plate_path", Path(self.execution_plate_path))
        if self.pipeline_path is not None:
            object.__setattr__(self, "pipeline_path", Path(self.pipeline_path))
