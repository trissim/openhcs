"""Service contract for importing CellProfiler pipelines into OpenHCS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from openhcs.constants import Backend
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.import_records import (
    CellProfilerPipelineImportResult,
)
from openhcs.interop.cellprofiler.pipeline_compiler import CellProfilerDialectCompiler


@dataclass(frozen=True, slots=True)
class CellProfilerPipelineImportRequest:
    """Typed request to convert a `.cppipe` file into an OpenHCS pipeline."""

    cppipe_path: Path
    generated_pipeline_path: Path
    prune_dead_unmaterialized_artifact_steps: bool = False
    filemanager: FileManagerLike | None = None
    cppipe_backend: Backend = Backend.DISK
    generated_pipeline_backend: Backend = Backend.DISK

    def __post_init__(self) -> None:
        object.__setattr__(self, "cppipe_path", Path(self.cppipe_path))
        object.__setattr__(
            self,
            "generated_pipeline_path",
            Path(self.generated_pipeline_path),
        )
        if self.cppipe_path.suffix != ".cppipe":
            raise ValueError(
                "CellProfilerPipelineImportRequest.cppipe_path must point to "
                f"a .cppipe file, got {self.cppipe_path}."
            )
        if self.generated_pipeline_path.suffix != ".py":
            raise ValueError(
                "CellProfilerPipelineImportRequest.generated_pipeline_path must point "
                f"to a .py file, got {self.generated_pipeline_path}."
            )
        if not isinstance(self.cppipe_backend, Backend):
            raise TypeError(
                "CellProfilerPipelineImportRequest.cppipe_backend must be "
                f"Backend, got {type(self.cppipe_backend).__name__}."
            )
        if not isinstance(self.generated_pipeline_backend, Backend):
            raise TypeError(
                "CellProfilerPipelineImportRequest.generated_pipeline_backend must be "
                f"Backend, got {type(self.generated_pipeline_backend).__name__}."
            )


class CellProfilerPipelineImporter(CellProfilerDialectCompiler, ABC):
    """Backend-neutral product contract for CellProfiler pipeline import."""

    def compile_pipeline(
        self,
        request: CellProfilerPipelineImportRequest,
    ) -> CellProfilerPipelineImportResult:
        """Compile one CellProfiler pipeline into ordinary OpenHCS state."""
        return self.import_pipeline(request)

    @abstractmethod
    def import_pipeline(
        self,
        request: CellProfilerPipelineImportRequest,
    ) -> CellProfilerPipelineImportResult:
        """Import a CellProfiler pipeline into ordinary OpenHCS state."""
