"""Structural protocol for OpenHCS virtual file-system operations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FileManagerLike(Protocol):
    """Minimal FileManager surface required by backend-explicit callers."""

    def load(self, file_path: str | Path, backend: str, **kwargs: object) -> object:
        """Load an object from the selected backend."""

    def save(
        self,
        data: object,
        output_path: str | Path,
        backend: str,
        **kwargs: object,
    ) -> None:
        """Save an object through the selected backend."""

    def ensure_directory(self, directory: str | Path, backend: str) -> str:
        """Ensure a directory exists in the selected backend."""
