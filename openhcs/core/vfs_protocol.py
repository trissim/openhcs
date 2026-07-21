"""Nominal contract for OpenHCS virtual file-system operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, get_args, get_origin


class FileManagerLike(ABC):
    """Minimal FileManager surface required by backend-explicit callers."""

    @property
    def registry(self) -> Mapping[str, object]:
        """Return backend registry metadata when the filemanager exposes it."""
        return MappingProxyType({})

    @abstractmethod
    def list_files(
        self,
        directory: str | Path,
        backend: str,
        **kwargs: object,
    ) -> list[str]:
        """List files from the selected backend."""

    @abstractmethod
    def exists(self, path: str | Path, backend: str) -> bool:
        """Return whether a path exists in the selected backend."""

    @abstractmethod
    def is_dir(self, path: str | Path, backend: str) -> bool:
        """Return whether a path is a directory in the selected backend."""

    @abstractmethod
    def load(self, file_path: str | Path, backend: str, **kwargs: object) -> object:
        """Load an object from the selected backend."""

    @abstractmethod
    def load_batch(
        self,
        file_paths: Sequence[str | Path],
        backend: str,
        **kwargs: object,
    ) -> Sequence[object]:
        """Load several objects from the selected backend."""

    @abstractmethod
    def resolve_address(
        self,
        backend_address: str | Path,
        backend: str,
        *,
        base_path: str | Path,
    ) -> str | Path:
        """Resolve one backend-owned source address."""

    def source_path(
        self,
        backend_address: str | Path,
        backend: str,
        *,
        base_path: str | Path,
    ) -> str | Path:
        """Return the physical source path represented by a backend address."""

        return self.resolve_address(
            backend_address,
            backend,
            base_path=base_path,
        )

    @abstractmethod
    def save(
        self,
        data: object,
        output_path: str | Path,
        backend: str,
        **kwargs: object,
    ) -> None:
        """Save an object through the selected backend."""

    @abstractmethod
    def ensure_directory(self, directory: str | Path, backend: str) -> str:
        """Ensure a directory exists in the selected backend."""


class PlatePathDeclaration(ABC):
    """Nominal declaration for a path resolved against one compilation plate."""

    @abstractmethod
    def validate_target(
        self,
        target: Path,
        *,
        filemanager: FileManagerLike,
        backend: str,
    ) -> None:
        """Validate the resolved target without creating it."""

    @classmethod
    def from_annotation(cls, annotation: Any) -> "PlatePathDeclaration | None":
        """Return the sole declaration carried by one structured annotation."""

        declarations = cls._from_annotation(annotation)
        unique = tuple(dict.fromkeys(declarations))
        if len(unique) > 1:
            raise TypeError(
                "Path annotation declares conflicting plate path policies: "
                f"{unique!r}."
            )
        return unique[0] if unique else None

    @classmethod
    def _from_annotation(cls, annotation: Any) -> tuple["PlatePathDeclaration", ...]:
        args = get_args(annotation)
        if get_origin(annotation) is Annotated:
            base, *metadata = args
            return (
                *(item for item in metadata if isinstance(item, cls)),
                *cls._from_annotation(base),
            )
        return tuple(
            declaration
            for member in args
            for declaration in cls._from_annotation(member)
        )


@dataclass(frozen=True, slots=True)
class PlateInputFileDeclaration(PlatePathDeclaration):
    """Existing plate-relative input file."""

    def validate_target(
        self,
        target: Path,
        *,
        filemanager: FileManagerLike,
        backend: str,
    ) -> None:
        if not filemanager.exists(target, backend):
            raise FileNotFoundError(f"Declared input file does not exist: {target}")
        if filemanager.is_dir(target, backend):
            raise IsADirectoryError(f"Declared input file is a directory: {target}")


@dataclass(frozen=True, slots=True)
class PlateInputDirectoryDeclaration(PlatePathDeclaration):
    """Existing plate-relative input directory."""

    def validate_target(
        self,
        target: Path,
        *,
        filemanager: FileManagerLike,
        backend: str,
    ) -> None:
        if not filemanager.exists(target, backend):
            raise FileNotFoundError(
                f"Declared input directory does not exist: {target}"
            )
        if not filemanager.is_dir(target, backend):
            raise NotADirectoryError(
                f"Declared input directory is not a directory: {target}"
            )


@dataclass(frozen=True, slots=True)
class PlateOutputFileDeclaration(PlatePathDeclaration):
    """Plate-relative output file whose writer owns creation."""

    def validate_target(
        self,
        target: Path,
        *,
        filemanager: FileManagerLike,
        backend: str,
    ) -> None:
        del target, filemanager, backend


@dataclass(frozen=True, slots=True)
class PlateOutputDirectoryDeclaration(PlatePathDeclaration):
    """Plate-relative output directory whose writer owns creation."""

    def validate_target(
        self,
        target: Path,
        *,
        filemanager: FileManagerLike,
        backend: str,
    ) -> None:
        del target, filemanager, backend


PlateInputFile = Annotated[Path, PlateInputFileDeclaration()]
PlateInputDirectory = Annotated[Path, PlateInputDirectoryDeclaration()]
PlateOutputFile = Annotated[Path, PlateOutputFileDeclaration()]
PlateOutputDirectory = Annotated[Path, PlateOutputDirectoryDeclaration()]
