"""Nominal contract for OpenHCS virtual file-system operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class FileManagerLike(ABC):
    """Minimal FileManager surface required by backend-explicit callers."""

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
