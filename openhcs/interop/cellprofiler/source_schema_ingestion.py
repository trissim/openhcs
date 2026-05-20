"""CellProfiler source-schema workspace ingestion for OpenHCS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openhcs.constants import Backend
from openhcs.core.source_schema_workspace import (
    SourceSchemaImageSetSelection,
    SourceSchemaWorkspaceMaterialization,
    materialize_source_schema_workspace,
)
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.runtime_pipeline import (
    PreparedGeneratedPipeline,
    prepare_generated_pipeline,
)


@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaWorkspaceRequest:
    """Request to prepare a `.cppipe` plus image source as an OpenHCS workspace."""

    source_root: Path
    cppipe_path: Path
    workspace_root: Path
    generated_pipeline_path: Path
    filemanager: FileManagerLike | None = None
    cppipe_filemanager: FileManagerLike | None = None
    generated_pipeline_filemanager: FileManagerLike | None = None
    source_backend: Backend = Backend.DISK
    workspace_backend: Backend = Backend.DISK
    cppipe_backend: Backend = Backend.DISK
    generated_pipeline_backend: Backend = Backend.DISK
    image_set_selection: SourceSchemaImageSetSelection | None = None
    prune_dead_unmaterialized_artifact_steps: bool = False
    materialize_skipped_save_images: bool = True
    materialize_terminal_images: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "cppipe_path", Path(self.cppipe_path))
        object.__setattr__(self, "workspace_root", Path(self.workspace_root))
        object.__setattr__(
            self,
            "generated_pipeline_path",
            Path(self.generated_pipeline_path),
        )
        if self.cppipe_path.suffix != ".cppipe":
            raise ValueError(
                "CellProfiler source-schema ingestion requires a .cppipe file, "
                f"got {self.cppipe_path}."
            )
        if self.generated_pipeline_path.suffix != ".py":
            raise ValueError(
                "generated_pipeline_path must point to a Python module, "
                f"got {self.generated_pipeline_path}."
            )


@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaWorkspace:
    """Prepared CellProfiler pipeline plus optional OpenHCS source workspace."""

    prepared_pipeline: PreparedGeneratedPipeline
    materialization: SourceSchemaWorkspaceMaterialization | None
    source_root: Path

    @property
    def execution_plate_path(self) -> Path:
        """Return the plate path that should be passed to `PipelineOrchestrator`."""
        if self.materialization is None:
            return self.source_root
        return self.materialization.workspace_root

    @property
    def source_workspace_path(self) -> Path | None:
        """Return materialized workspace root when source-schema projection was needed."""
        if self.materialization is None:
            return None
        return self.materialization.workspace_root


class CellProfilerSourceSchemaIngestionError(RuntimeError):
    """Base error for CellProfiler source-schema ingestion failures."""


class CellProfilerPipelinePreparationError(CellProfilerSourceSchemaIngestionError):
    """Raised when `.cppipe` conversion fails before source workspace creation."""


class CellProfilerSourceWorkspaceMaterializationError(
    CellProfilerSourceSchemaIngestionError
):
    """Raised when source-schema workspace materialization fails."""


def prepare_cellprofiler_source_schema_workspace(
    request: CellProfilerSourceSchemaWorkspaceRequest,
) -> CellProfilerSourceSchemaWorkspace:
    """Prepare a CellProfiler pipeline and source-schema OpenHCS workspace."""
    try:
        prepared = prepare_generated_pipeline(
            request.cppipe_path,
            output_path=request.generated_pipeline_path,
            prune_dead_unmaterialized_artifact_steps=(
                request.prune_dead_unmaterialized_artifact_steps
            ),
            materialize_skipped_save_images=request.materialize_skipped_save_images,
            materialize_terminal_images=request.materialize_terminal_images,
            cppipe_filemanager=request.cppipe_filemanager,
            generated_pipeline_filemanager=request.generated_pipeline_filemanager,
            cppipe_backend=request.cppipe_backend,
            generated_pipeline_backend=request.generated_pipeline_backend,
        )
    except ValueError as exc:
        raise CellProfilerPipelinePreparationError(
            f"Failed to prepare converted .cppipe pipeline "
            f"{request.cppipe_path.name}: {exc}"
        ) from exc

    if prepared.source_schema.is_empty:
        return CellProfilerSourceSchemaWorkspace(
            prepared_pipeline=prepared,
            materialization=None,
            source_root=request.source_root,
        )

    try:
        materialization = materialize_source_schema_workspace(
            request.source_root,
            request.workspace_root,
            prepared.source_schema,
            filemanager=request.filemanager,
            source_backend=request.source_backend,
            workspace_backend=request.workspace_backend,
            image_set_selection=request.image_set_selection,
        )
    except Exception as exc:
        raise CellProfilerSourceWorkspaceMaterializationError(
            f"Failed to materialize CellProfiler source schema for "
            f"{request.cppipe_path.name}: {exc}"
        ) from exc

    return CellProfilerSourceSchemaWorkspace(
        prepared_pipeline=prepared,
        materialization=materialization,
        source_root=request.source_root,
    )
