"""CellProfiler source folders as OpenHCS plate workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from openhcs.core.input_workspace import (
    InputWorkspacePreparationRequest,
    InputWorkspacePreparationResult,
    PipelineImportDiagnostic,
)
from openhcs.interop.cellprofiler.runtime_pipeline import prepare_generated_pipeline
from openhcs.interop.cellprofiler.source_schema_ingestion import (
    CellProfilerSourceSchemaWorkspace,
    CellProfilerSourceSchemaWorkspaceRequest,
    compile_cellprofiler_source_schema,
    prepare_cellprofiler_source_schema_only_workspace,
    prepare_cellprofiler_source_schema_workspace,
)
from openhcs.microscopes.openhcs import OpenHCSMetadataHandler


@dataclass(frozen=True, slots=True)
class CellProfilerPlateWorkspaceRequest:
    """Request to prepare a selected folder for OpenHCS plate initialization."""

    plate_root: Path
    cppipe_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plate_root", Path(self.plate_root))
        if self.cppipe_path is not None:
            object.__setattr__(self, "cppipe_path", Path(self.cppipe_path))


@dataclass(frozen=True, slots=True)
class CellProfilerPlateWorkspaceResult:
    """Result of preparing a CellProfiler folder as an OpenHCS workspace."""

    plate_root: Path
    cppipe_path: Path | None
    ingestion: CellProfilerSourceSchemaWorkspace | None

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
        return cppipe_path.suffix == ".cppipe" and not cppipe_path.name.startswith("._")


@dataclass(frozen=True, slots=True)
class CellProfilerPlateWorkspacePreparer:
    """Prepare direct `.cppipe` folders for OpenHCS plate initialization."""

    request: CellProfilerPlateWorkspaceRequest

    def prepare(self) -> CellProfilerPlateWorkspaceResult:
        plate_root = self.request.plate_root
        cppipe_path = self.cppipe_path()
        if cppipe_path is None:
            return CellProfilerPlateWorkspaceResult(
                plate_root=plate_root,
                cppipe_path=cppipe_path,
                ingestion=None,
            )
        if self.openhcs_metadata_path().exists():
            prepared = prepare_generated_pipeline(
                cppipe_path,
                output_path=self.generated_pipeline_path(cppipe_path),
            )
            return CellProfilerPlateWorkspaceResult(
                plate_root=plate_root,
                cppipe_path=cppipe_path,
                ingestion=CellProfilerSourceSchemaWorkspace(
                    prepared_pipeline=prepared,
                    materialization=None,
                    source_root=plate_root,
                ),
            )
        ingestion = prepare_cellprofiler_source_schema_workspace(
            CellProfilerSourceSchemaWorkspaceRequest(
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
                selected_path=self.request.plate_root,
                selected_pipeline_path=self.request.cppipe_path,
                workspace_root=self.request.plate_root,
            )
        )

    def openhcs_metadata_path(self) -> Path:
        return self.request.plate_root / OpenHCSMetadataHandler.METADATA_FILENAME

    def cppipe_path(self) -> Path | None:
        candidates = self.cppipe_paths()
        if self.request.cppipe_path is not None:
            cppipe_path = self.request.cppipe_path
            if not cppipe_path.is_absolute():
                cppipe_path = self.request.plate_root / cppipe_path
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
        preferred = self.request.plate_root / f"{self.request.plate_root.name}.cppipe"
        if preferred in candidates:
            return preferred
        candidate_names = ", ".join(path.name for path in candidates)
        raise ValueError(
            "CellProfiler plate workspace contains multiple .cppipe files; "
            f"expected one or {preferred.name}. Found: {candidate_names}"
        )

    def cppipe_paths(self) -> tuple[Path, ...]:
        pipeline_files = tuple(
            CellProfilerPipelineFile.from_path(path)
            for path in self.request.plate_root.glob("*.cppipe")
            if CellProfilerPipelineFile.is_visible_cppipe(path)
        )
        return tuple(
            file.path
            for file in sorted(pipeline_files, key=lambda file: file.sort_key)
        )

    def generated_pipeline_path(self, cppipe_path: Path) -> Path:
        generated_dir = self.request.plate_root / ".openhcs_cellprofiler"
        generated_dir.mkdir(parents=True, exist_ok=True)
        return generated_dir / f"{cppipe_path.stem}_openhcs.py"


def prepare_cellprofiler_input_workspace(
    request: InputWorkspacePreparationRequest,
) -> InputWorkspacePreparationResult:
    """Prepare a CellProfiler folder through the generic input workspace contract."""

    plate_root = request.selected_path
    preparer = CellProfilerPlateWorkspacePreparer(
        CellProfilerPlateWorkspaceRequest(
            plate_root,
            cppipe_path=request.selected_pipeline_path,
        )
    )
    cppipe_path = preparer.cppipe_path()
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
    source_schema_request = CellProfilerSourceSchemaWorkspaceRequest(
        source_root=plate_root,
        cppipe_path=cppipe_path,
        workspace_root=workspace_root,
        generated_pipeline_path=generated_pipeline_path,
    )

    source_preparation = None
    if preparer.openhcs_metadata_path().exists():
        source_schema = compile_cellprofiler_source_schema(source_schema_request)
        execution_plate_path = plate_root
    else:
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
        original_source_root=(
            source_preparation.source_root
            if source_preparation is not None
            else plate_root
        ),
        execution_plate_path=execution_plate_path,
        pipeline_path=cppipe_path,
        source_schema=source_schema,
        materialization=(
            source_preparation.materialization
            if source_preparation is not None
            else None
        ),
        prepared_pipeline=prepared_pipeline,
        pipeline_import_error=import_error,
    )
