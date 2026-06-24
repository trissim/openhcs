"""Shared current-image runtime context for CellProfiler projection authorities."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.runtime_values import ImagePayloadMetadataInput


@dataclass(frozen=True, kw_only=True)
class CellProfilerOptionalCurrentImageContext:
    """Nominal current-image carrier for contexts where absence is valid."""

    current_image: ImagePayloadMetadataInput | None = None

    def required_current_image(self, purpose: str) -> ImagePayloadMetadataInput:
        """Return the current image or fail with the consuming semantic purpose."""
        if self.current_image is None:
            raise RuntimeError(f"{purpose} requires a current image payload.")
        return self.current_image


@dataclass(frozen=True, kw_only=True)
class CellProfilerRequiredCurrentImageContext:
    """Nominal current-image carrier for contexts where absence is invalid."""

    current_image: ImagePayloadMetadataInput
