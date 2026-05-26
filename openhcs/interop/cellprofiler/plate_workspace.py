"""CellProfiler source folders as OpenHCS plate workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openhcs.interop.cellprofiler.runtime_pipeline import prepare_generated_pipeline
from openhcs.interop.cellprofiler.source_schema_ingestion import (
    CellProfilerSourceSchemaWorkspace,
    CellProfilerSourceSchemaWorkspaceRequest,
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

    @property
    def execution_plate_path(self) -> Path:
        """Return the plate workspace that should back orchestrator execution."""

        if self.ingestion is None:
            return self.plate_root
        return self.ingestion.execution_plate_path


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
        return tuple(sorted(self.request.plate_root.glob("*.cppipe")))

    def generated_pipeline_path(self, cppipe_path: Path) -> Path:
        generated_dir = self.request.plate_root / ".openhcs_cellprofiler"
        generated_dir.mkdir(parents=True, exist_ok=True)
        return generated_dir / f"{cppipe_path.stem}_openhcs.py"
