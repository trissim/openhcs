"""Nominal service contracts consumed by the CellProfiler runtime adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from openhcs.core.config import ZarrConfig
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFilePayload,
    ImagePayloadValue,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerParsedMetadataValue,
)

CellProfilerFileManagerOption = (
    str
    | int
    | float
    | bool
    | ZarrConfig
    | None
)


class CellProfilerFileManager(ABC):
    """Nominal filemanager surface used by the CellProfiler runtime adapter."""

    @abstractmethod
    def load(
        self,
        file_path: str | Path,
        backend: str,
        **backend_options: CellProfilerFileManagerOption,
    ) -> ImagePayloadValue:
        """Load one payload from a backend."""

    @abstractmethod
    def load_batch(
        self,
        file_paths: Sequence[str | Path],
        backend: str,
        **backend_options: CellProfilerFileManagerOption,
    ) -> Sequence[ImagePayloadValue]:
        """Load several payloads from a backend."""

    @abstractmethod
    def save(
        self,
        data: CellProfilerFilePayload,
        output_path: str | Path,
        backend: str,
        **backend_options: CellProfilerFileManagerOption,
    ) -> None:
        """Save one payload to a backend."""

    @abstractmethod
    def delete(
        self,
        path: str | Path,
        backend: str,
        recursive: bool = False,
    ) -> bool:
        """Delete one path from a backend."""

    @abstractmethod
    def ensure_directory(self, directory: str | Path, backend: str) -> str:
        """Ensure a backend directory exists."""


class CellProfilerFilenameParser(ABC):
    """Microscope filename parser surface needed for source ordering."""

    @abstractmethod
    def semantic_identity(self) -> tuple[Hashable, ...]:
        """Return the parser semantics that affect filename parsing."""

    @abstractmethod
    def parse_filename(
        self,
        filename: str,
    ) -> Mapping[str, CellProfilerParsedMetadataValue] | None:
        """Parse source filename metadata."""


class CellProfilerMicroscopeHandler(ABC):
    """Microscope handler surface needed by the adapter."""

    @property
    @abstractmethod
    def parser(self) -> CellProfilerFilenameParser:
        """Return the filename parser."""


class CellProfilerProcessingContext(ABC):
    """Processing context surface consumed by CellProfiler runtime adapter."""

    @property
    @abstractmethod
    def filemanager(self) -> CellProfilerFileManager:
        """Return the OpenHCS VFS filemanager."""

    @property
    @abstractmethod
    def input_dir(self) -> str:
        """Return the processing input directory."""

    @property
    @abstractmethod
    def microscope_handler(self) -> CellProfilerMicroscopeHandler:
        """Return the initialized microscope handler."""


@dataclass(frozen=True, slots=True)
class RequireProcessingContextBoundaryPolicy:
    """Boundary policy for adapter paths that require source-loading services."""

    adapter: "CellProfilerRuntimeAdapter"

    @property
    def context(self) -> CellProfilerProcessingContext:
        if self.adapter.processing_context is None:
            raise RuntimeError(
                "CellProfilerRuntimeAdapter.processing_context is required for "
                "selector-bearing source resolution."
            )
        return self.adapter.processing_context
