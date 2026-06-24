"""CellProfiler source folders as OpenHCS plate workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

from openhcs.core.input_workspace import (
    InputWorkspacePreparationRequest,
    InputWorkspacePreparationResult,
    PipelineImportDiagnostic,
)
from openhcs.interop.cellprofiler.runtime_pipeline import prepare_generated_pipeline
from openhcs.interop.cellprofiler.source_schema_ingestion import (
    CellProfilerSourceSchemaWorkspace,
    CellProfilerSourceSchemaWorkspaceRequest,
    prepare_cellprofiler_source_schema_only_workspace,
    prepare_cellprofiler_source_schema_workspace,
)


@dataclass(frozen=True, slots=True)
class CellProfilerPlateWorkspaceRequest:
    """Request to prepare a selected folder for OpenHCS plate initialization."""

    plate_root: Path
    cppipe_path: Path | None = None

    @classmethod
    def from_paths(
        cls,
        plate_root: Path | str,
        cppipe_path: Path | str | None = None,
    ) -> "CellProfilerPlateWorkspaceRequest":
        return cls(
            plate_root=Path(plate_root),
            cppipe_path=cls._optional_cppipe_path(cppipe_path),
        )

    @staticmethod
    def _optional_cppipe_path(cppipe_path: Path | str | None) -> Path | None:
        if cppipe_path is None:
            return None
        return Path(cppipe_path)


@dataclass(frozen=True, slots=True)
class CellProfilerPlateWorkspaceResult(CellProfilerPlateWorkspaceRequest):
    """Result of preparing a CellProfiler folder as an OpenHCS workspace."""

    ingestion: CellProfilerSourceSchemaWorkspace | None = None

    @property
    def materialized(self) -> bool:
        return self.ingestion is not None and self.ingestion.materialization is not None


class CellProfilerPipelineStage(Enum):
    """Tutorial-stage semantics encoded in CellProfiler pipeline filenames."""

    START = 0
    FINAL = 1
    UNLABELED = 2

    @classmethod
    def from_path(cls, cppipe_path: Path) -> "CellProfilerPipelineStage":
        tokens = tuple(
            token.casefold()
            for token in (
                cppipe_path.stem.replace("-", "_").replace(" ", "_").split("_")
            )
            if token
        )
        if "start" in tokens:
            return cls.START
        if "final" in tokens:
            return cls.FINAL
        return cls.UNLABELED


class CellProfilerPipelineFileKind(Enum):
    """Nominal file categories considered during `.cppipe` discovery."""

    PIPELINE = "pipeline"
    APPLEDOUBLE_SIDECAR = "appledouble_sidecar"
    OTHER = "other"

    @classmethod
    def from_path(cls, path: Path) -> "CellProfilerPipelineFileKind":
        if path.suffix != ".cppipe":
            return cls.OTHER
        if AppleDoubleResourceForkFilename.matches(path.name):
            return cls.APPLEDOUBLE_SIDECAR
        return cls.PIPELINE


class AppleDoubleResourceForkFilename:
    """Filename convention for macOS AppleDouble resource-fork sidecars."""

    PREFIX: ClassVar[str] = "._"

    @classmethod
    def matches(cls, filename: str) -> bool:
        return filename[: len(cls.PREFIX)] == cls.PREFIX


@dataclass(frozen=True, slots=True)
class CellProfilerPipelineFile:
    """Visible `.cppipe` file with ordering semantics for one source folder."""

    path: Path
    stage: CellProfilerPipelineStage

    @classmethod
    def from_path(cls, cppipe_path: Path) -> "CellProfilerPipelineFile":
        return cls(
            path=cppipe_path,
            stage=CellProfilerPipelineStage.from_path(cppipe_path),
        )

    @property
    def sort_key(self) -> tuple[int, str]:
        return (self.stage.value, self.path.name.casefold())

    @staticmethod
    def is_visible_cppipe(cppipe_path: Path) -> bool:
        return (
            CellProfilerPipelineFileKind.from_path(cppipe_path)
            is CellProfilerPipelineFileKind.PIPELINE
        )


@dataclass(frozen=True, slots=True)
class CellProfilerPlateWorkspacePreparer(CellProfilerPlateWorkspaceRequest):
    """Prepare direct `.cppipe` folders for OpenHCS plate initialization."""

    def prepare(self) -> CellProfilerPlateWorkspaceResult:
        plate_root = self.plate_root
        cppipe_path = self.resolved_cppipe_path()
        if cppipe_path is None:
            return CellProfilerPlateWorkspaceResult(
                plate_root=plate_root,
                cppipe_path=cppipe_path,
                ingestion=None,
            )
        ingestion = prepare_cellprofiler_source_schema_workspace(
            CellProfilerSourceSchemaWorkspaceRequest.from_paths(
                source_root=plate_root,
                cppipe_path=cppipe_path,
                workspace_root=plate_root,
                generated_pipeline_path=self.generated_pipeline_path(cppipe_path),
            )
        )
        return CellProfilerPlateWorkspaceResult(
            plate_root=plate_root,
            cppipe_path=cppipe_path,
            ingestion=ingestion,
        )

    def prepare_input_workspace(self) -> InputWorkspacePreparationResult:
        """Prepare source workspace first, reporting pipeline import as diagnostic."""

        return prepare_cellprofiler_input_workspace(
            InputWorkspacePreparationRequest(
                selected_path=self.plate_root,
                selected_pipeline_path=self.cppipe_path,
                workspace_root=self.plate_root,
            )
        )

    def resolved_cppipe_path(self) -> Path | None:
        candidates = self.cppipe_paths()
        if self.cppipe_path is not None:
            cppipe_path = self.cppipe_path
            if not cppipe_path.is_absolute():
                cppipe_path = self.plate_root / cppipe_path
            if cppipe_path.suffix != ".cppipe":
                raise ValueError(f"Expected a .cppipe file, got {cppipe_path}.")
            if cppipe_path not in candidates:
                raise FileNotFoundError(
                    f"Requested CellProfiler pipeline not found in plate workspace: {cppipe_path}"
                )
            return cppipe_path
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        same_named_cppipe = self.plate_root / f"{self.plate_root.name}.cppipe"
        if same_named_cppipe in candidates:
            return same_named_cppipe
        candidate_names = ", ".join(path.name for path in candidates)
        raise ValueError(
            "CellProfiler plate workspace contains multiple .cppipe files; "
            f"expected one or {same_named_cppipe.name}. Found: {candidate_names}"
        )

    def cppipe_paths(self) -> tuple[Path, ...]:
        pipeline_files = tuple(
            CellProfilerPipelineFile.from_path(path)
            for path in self.plate_root.glob("*.cppipe")
            if CellProfilerPipelineFile.is_visible_cppipe(path)
        )
        return tuple(
            file.path
            for file in sorted(pipeline_files, key=lambda file: file.sort_key)
        )

    def default_cppipe_path(self) -> Path | None:
        """Return the default CellProfiler pipeline for this physical plate."""
        candidates = self.cppipe_paths()
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        same_named_cppipe = self.plate_root / f"{self.plate_root.name}.cppipe"
        if same_named_cppipe in candidates:
            return same_named_cppipe

        final_pipelines = tuple(
            path
            for path in candidates
            if CellProfilerPipelineStage.from_path(path)
            is CellProfilerPipelineStage.FINAL
        )
        if len(final_pipelines) == 1:
            return final_pipelines[0]

        unlabeled_pipelines = tuple(
            path
            for path in candidates
            if CellProfilerPipelineStage.from_path(path)
            is CellProfilerPipelineStage.UNLABELED
        )
        if len(unlabeled_pipelines) == 1:
            return unlabeled_pipelines[0]

        return candidates[0]

    def generated_pipeline_path(self, cppipe_path: Path) -> Path:
        generated_dir = self.plate_root / ".openhcs_cellprofiler"
        generated_dir.mkdir(parents=True, exist_ok=True)
        return generated_dir / f"{cppipe_path.stem}_openhcs.py"


def prepare_cellprofiler_input_workspace(
    request: InputWorkspacePreparationRequest,
) -> InputWorkspacePreparationResult:
    """Prepare a CellProfiler folder through the generic input workspace contract."""

    plate_root = request.selected_path
    preparer = CellProfilerPlateWorkspacePreparer.from_paths(
        plate_root,
        cppipe_path=request.selected_pipeline_path,
    )
    cppipe_path = preparer.resolved_cppipe_path()
    if cppipe_path is None:
        return InputWorkspacePreparationResult(
            original_source_root=plate_root,
            execution_plate_path=plate_root,
        )

    generated_pipeline_path = (
        request.generated_pipeline_path
        if request.generated_pipeline_path is not None
        else preparer.generated_pipeline_path(cppipe_path)
    )
    workspace_root = request.workspace_root or plate_root
    source_schema_request = CellProfilerSourceSchemaWorkspaceRequest.from_paths(
        source_root=plate_root,
        cppipe_path=cppipe_path,
        workspace_root=workspace_root,
        generated_pipeline_path=generated_pipeline_path,
    )

    source_preparation = prepare_cellprofiler_source_schema_only_workspace(
        source_schema_request
    )
    source_schema = source_preparation.source_schema
    execution_plate_path = source_preparation.execution_plate_path

    prepared_pipeline = None
    import_error = None
    try:
        prepared_pipeline = prepare_generated_pipeline(
            cppipe_path,
            output_path=generated_pipeline_path,
        )
        source_schema = prepared_pipeline.source_schema
    except ValueError as exc:
        import_error = PipelineImportDiagnostic(
            pipeline_path=cppipe_path,
            exception_type=type(exc).__name__,
            message=str(exc),
        )

    return InputWorkspacePreparationResult(
        original_source_root=source_preparation.source_root,
        execution_plate_path=execution_plate_path,
        pipeline_path=cppipe_path,
        source_schema=source_schema,
        materialization=source_preparation.materialization,
        prepared_pipeline=prepared_pipeline,
        pipeline_import_error=import_error,
    )
