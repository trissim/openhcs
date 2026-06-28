"""Compiled source-loading options consumed by runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from openhcs.constants.constants import Backend
from openhcs.core.config import ZarrConfig

SourceLoadBackendOption: TypeAlias = str | int | float | bool | ZarrConfig | None


@dataclass(frozen=True, slots=True)
class SourceLoadPlan:
    """Typed compile artifact for loading pipeline-start sources."""

    zarr_config: ZarrConfig | None = None

    def filemanager_load_kwargs(
        self,
        backend: Backend | str,
    ) -> dict[str, SourceLoadBackendOption]:
        """Return backend-specific filemanager load options."""
        backend_value = backend.value if isinstance(backend, Backend) else str(backend)
        if backend_value == Backend.ZARR.value:
            return {"zarr_config": self.zarr_config}
        return {}
