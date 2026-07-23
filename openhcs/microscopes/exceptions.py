"""Typed microscope metadata exceptions."""

from __future__ import annotations

from pathlib import Path


class MicroscopeMetadataError(ValueError):
    """Base class for microscope metadata contract failures."""


class MicroscopePixelSizeUnavailableError(MicroscopeMetadataError):
    """Raised when a microscope handler cannot determine physical pixel size."""

    def __init__(self, image_path: str | Path) -> None:
        self.image_path = Path(image_path)
        super().__init__(
            f"Pixel size not found in TIFF metadata for {self.image_path}"
        )
