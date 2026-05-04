"""Object outline backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit, prange

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)


class ObjectOutlineBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Object outline operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def outline(self, labels: np.ndarray) -> np.ndarray:
        """Return a labeled inner outline image."""


class NumbaNumpyObjectOutlineBackendStrategy(ObjectOutlineBackendStrategy):
    """Numba-accelerated NumPy object outline primitives."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def outline(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            raise NotImplementedError("Object outlines currently support 2-D labels.")
        return _outline_numba(np.ascontiguousarray(label_array))


class CentrosomeNumpyObjectOutlineBackendStrategy(ObjectOutlineBackendStrategy):
    """Explicit centrosome provider for NumPy object outlines."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def outline(self, labels: np.ndarray) -> np.ndarray:
        from centrosome.outline import outline

        return outline(labels)


def object_outline_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> ObjectOutlineBackendStrategy:
    """Return the selected CellProfiler object outline backend."""
    return ObjectOutlineBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    )


@njit(cache=True, parallel=True)
def _outline_numba(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    output = np.zeros((height, width), dtype=labels.dtype)
    for y in prange(height):
        for x in range(width):
            center = labels[y, x]
            if center <= 0:
                continue
            min_label = center
            max_label = center
            for dy in range(-1, 2):
                ny = y + dy
                for dx in range(-1, 2):
                    nx = x + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        value = 0
                    else:
                        value = labels[ny, nx]
                    if value < min_label:
                        min_label = value
                    if value > max_label:
                        max_label = value
            if max_label != min_label:
                output[y, x] = center
    return output


__all__ = [
    "CentrosomeNumpyObjectOutlineBackendStrategy",
    "NumbaNumpyObjectOutlineBackendStrategy",
    "ObjectOutlineBackendStrategy",
    "object_outline_backend",
]
