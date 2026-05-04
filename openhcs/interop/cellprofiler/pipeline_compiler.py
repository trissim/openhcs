"""Compiler contract for CellProfiler-to-OpenHCS pipeline import."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from openhcs.interop.cellprofiler.import_records import (
    CellProfilerPipelineImportResult,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.import_service import (
        CellProfilerPipelineImportRequest,
    )


class CellProfilerDialectCompiler(ABC):
    """Product contract for compiling `.cppipe` pipelines into OpenHCS."""

    @abstractmethod
    def compile_pipeline(
        self,
        request: CellProfilerPipelineImportRequest,
    ) -> CellProfilerPipelineImportResult:
        """Compile one CellProfiler pipeline into ordinary OpenHCS state."""
