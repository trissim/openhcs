"""CellProfiler source folders as OpenHCS plate workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

from polystore.base import _create_storage_registry
from polystore.filemanager import FileManager

from openhcs.constants import Backend
from openhcs.core.input_workspace import (
    InputWorkspacePreparationRequest,
    InputWorkspacePreparationResult,
    PipelineImportDiagnostic,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.source_binding_workspace import (
    SourceBindingWorkspaceMaterialization,
    materialize_source_binding_workspace,
)
from openhcs.core.source_bindings import source_bindings_defaults_to_base
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


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
class CellProfilerPlateWorkspacePreparer:
    """Prepare direct `.cppipe` folders for OpenHCS plate initialization."""

    plate_root: Path
    cppipe_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plate_root", Path(self.plate_root))
        if self.cppipe_path is not None:
            object.__setattr__(self, "cppipe_path", Path(self.cppipe_path))

    @classmethod
    def from_paths(
        cls,
        plate_root: Path | str,
        cppipe_path: Path | str | None = None,
    ) -> "CellProfilerPlateWorkspacePreparer":
        return cls(
            plate_root=Path(plate_root),
            cppipe_path=None if cppipe_path is None else Path(cppipe_path),
        )

    def prepare(self) -> InputWorkspacePreparationResult:
        plate_root = self.plate_root
        cppipe_path = self.resolved_cppipe_path()
        if cppipe_path is None:
            return InputWorkspacePreparationResult(
                original_source_root=plate_root,
                execution_plate_path=plate_root,
            )
        return prepare_cellprofiler_input_workspace(
            InputWorkspacePreparationRequest(
                selected_path=plate_root,
                selected_pipeline_path=cppipe_path,
                workspace_root=self.source_workspace_root(cppipe_path),
            )
        )

    def resolved_cppipe_path(self) -> Path | None:
        if self.cppipe_path is not None:
            cppipe_path = self.cppipe_path
            if not cppipe_path.is_absolute():
                cppipe_path = self.plate_root / cppipe_path
            if cppipe_path.suffix != ".cppipe":
                raise ValueError(f"Expected a .cppipe file, got {cppipe_path}.")
            if not cppipe_path.is_file():
                raise FileNotFoundError(
                    f"Requested CellProfiler pipeline does not exist: {cppipe_path}"
                )
            return cppipe_path
        candidates = self.cppipe_paths()
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

    def source_workspace_root(self, cppipe_path: Path) -> Path:
        generated_dir = self.plate_root / ".openhcs_cellprofiler"
        generated_dir.mkdir(parents=True, exist_ok=True)
        return generated_dir / f"{cppipe_path.stem}_source_workspace"


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

    workspace_root = (
        request.workspace_root
        if request.workspace_root is not None
        else preparer.source_workspace_root(cppipe_path)
    )
    pipeline_steps = None
    pipeline_config = None
    import_error = None
    filemanager = FileManager(_create_storage_registry())
    try:
        pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
            cppipe_path,
            source_root=plate_root,
        )
        if request.generated_source_path is not None:
            filemanager.save(
                FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps),
                request.generated_source_path,
                Backend.DISK.value,
            )
    except ValueError as exc:
        import_error = PipelineImportDiagnostic(
            pipeline_path=cppipe_path,
            exception_type=type(exc).__name__,
            message=str(exc),
        )

    materialization: SourceBindingWorkspaceMaterialization | None = None
    execution_plate_path = plate_root
    if pipeline_config is not None:
        source_bindings = source_bindings_defaults_to_base(
            pipeline_config.source_bindings_config
        ).resolved_imported_metadata_locations(
            plate_root,
            portable_roots=(cppipe_path.parent,),
        )
        if not source_bindings.is_empty:
            source_files = tuple(
                filemanager.list_files(
                    plate_root,
                    Backend.DISK.value,
                    recursive=True,
                )
            )
            materialization = materialize_source_binding_workspace(
                plate_root,
                workspace_root,
                source_bindings,
                filemanager=filemanager,
                source_backend=Backend.DISK,
                workspace_backend=Backend.DISK,
                source_files=source_files,
                parser=SourceSchemaFilenameParser(),
            )
            execution_plate_path = materialization.workspace_root

    return InputWorkspacePreparationResult(
        original_source_root=plate_root,
        execution_plate_path=execution_plate_path,
        pipeline_path=cppipe_path,
        pipeline_steps=pipeline_steps,
        pipeline_config=pipeline_config,
        materialization=materialization,
        pipeline_import_error=import_error,
    )
