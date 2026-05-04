"""Secondary-object backend strategies for CellProfiler-compatible processing."""

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


class SecondaryDistanceTransformBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Distance transform operations used by secondary segmentation."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def distance_to_foreground(self, labels: np.ndarray) -> np.ndarray:
        """Return Euclidean distance to the nearest positive label."""

    @abstractmethod
    def nearest_label_expansion(
        self,
        labels: np.ndarray,
        max_distance: float,
    ) -> np.ndarray:
        """Expand labels to pixels within ``max_distance`` of a seed."""


class NumpySecondaryDistanceTransformBackendStrategy(
    SecondaryDistanceTransformBackendStrategy,
):
    """Reference NumPy/SciPy secondary distance-transform backend."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False

    def distance_to_foreground(self, labels: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(np.asarray(labels) == 0)

    def nearest_label_expansion(
        self,
        labels: np.ndarray,
        max_distance: float,
    ) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt

        label_array = np.asarray(labels, dtype=np.int32)
        distances, indices = distance_transform_edt(
            label_array == 0,
            return_indices=True,
        )
        output = np.zeros_like(label_array)
        mask = distances <= float(max_distance)
        output[mask] = label_array[indices[0][mask], indices[1][mask]]
        return output


class NumbaSecondaryDistanceTransformBackendStrategy(
    SecondaryDistanceTransformBackendStrategy,
):
    """Numba-accelerated exact 2-D Euclidean distance-transform backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def distance_to_foreground(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            raise NotImplementedError(
                "Numba secondary distance backend currently supports 2-D labels."
            )
        return _distance_to_positive_labels_numba(np.ascontiguousarray(label_array))

    def nearest_label_expansion(
        self,
        labels: np.ndarray,
        max_distance: float,
    ) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            raise NotImplementedError(
                "Numba secondary distance backend currently supports 2-D labels."
            )
        return _nearest_label_expansion_numba(
            np.ascontiguousarray(label_array),
            float(max_distance),
        )


class SecondaryPropagationBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Regularized label propagation backend for secondary segmentation."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
    ) -> np.ndarray:
        """Propagate seed labels through a mask."""


class CentrosomeSecondaryPropagationBackendStrategy(
    SecondaryPropagationBackendStrategy,
):
    """Centrosome provider for exact CellProfiler propagation semantics."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = True

    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
    ) -> np.ndarray:
        import centrosome.propagate

        if np.max(labels) == 0:
            return np.asarray(labels, dtype=np.int32).copy()
        result, _distance = centrosome.propagate.propagate(
            image,
            labels,
            mask,
            regularization,
        )
        return np.asarray(result, dtype=np.int32)


@njit(cache=True, parallel=True)
def _distance_to_positive_labels_numba(labels: np.ndarray) -> np.ndarray:
    distances, _nearest_y, _nearest_x = _edt_feature_transform_numba(labels)
    return distances


@njit(cache=True, parallel=True)
def _nearest_label_expansion_numba(
    labels: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    distances, nearest_y, nearest_x = _edt_feature_transform_numba(labels)
    height, width = labels.shape
    output = np.zeros((height, width), dtype=np.int32)
    for y in prange(height):
        for x in range(width):
            if distances[y, x] <= max_distance:
                output[y, x] = labels[nearest_y[y, x], nearest_x[y, x]]
    return output


@njit(cache=True, parallel=True)
def _edt_feature_transform_numba(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = labels.shape
    inf = 1.0e20
    row_distances = np.empty((height, width), dtype=np.float64)
    row_nearest_x = np.empty((height, width), dtype=np.int64)
    distances_sq = np.empty((height, width), dtype=np.float64)
    nearest_y = np.empty((height, width), dtype=np.int64)
    nearest_x = np.empty((height, width), dtype=np.int64)

    for y in prange(height):
        source = np.empty(width, dtype=np.float64)
        for x in range(width):
            source[x] = 0.0 if labels[y, x] > 0 else inf
        row_output = np.empty(width, dtype=np.float64)
        row_arg = np.empty(width, dtype=np.int64)
        _edt_1d_numba(source, row_output, row_arg)
        for x in range(width):
            row_distances[y, x] = row_output[x]
            row_nearest_x[y, x] = row_arg[x]

    for x in prange(width):
        source = np.empty(height, dtype=np.float64)
        for y in range(height):
            source[y] = row_distances[y, x]
        column_output = np.empty(height, dtype=np.float64)
        column_arg = np.empty(height, dtype=np.int64)
        _edt_1d_numba(source, column_output, column_arg)
        for y in range(height):
            seed_y = column_arg[y]
            distances_sq[y, x] = column_output[y]
            nearest_y[y, x] = seed_y
            nearest_x[y, x] = row_nearest_x[seed_y, x]

    distances = np.empty((height, width), dtype=np.float64)
    for y in prange(height):
        for x in range(width):
            distances[y, x] = np.sqrt(distances_sq[y, x])
    return distances, nearest_y, nearest_x


@njit(cache=True)
def _edt_1d_numba(
    source: np.ndarray,
    output: np.ndarray,
    argmin: np.ndarray,
) -> None:
    length = source.size
    locations = np.empty(length, dtype=np.int64)
    boundaries = np.empty(length + 1, dtype=np.float64)
    k = 0
    locations[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf

    for q in range(1, length):
        s = _edt_intersection_numba(source, q, locations[k])
        while s <= boundaries[k]:
            k -= 1
            s = _edt_intersection_numba(source, q, locations[k])
        k += 1
        locations[k] = q
        boundaries[k] = s
        boundaries[k + 1] = np.inf

    k = 0
    for q in range(length):
        while boundaries[k + 1] < q:
            k += 1
        location = locations[k]
        delta = q - location
        output[q] = delta * delta + source[location]
        argmin[q] = location


@njit(cache=True)
def _edt_intersection_numba(source: np.ndarray, q: int, v: int) -> float:
    return ((source[q] + q * q) - (source[v] + v * v)) / (2.0 * q - 2.0 * v)


__all__ = [
    "CentrosomeSecondaryPropagationBackendStrategy",
    "NumbaSecondaryDistanceTransformBackendStrategy",
    "NumpySecondaryDistanceTransformBackendStrategy",
    "SecondaryDistanceTransformBackendStrategy",
    "SecondaryPropagationBackendStrategy",
]
