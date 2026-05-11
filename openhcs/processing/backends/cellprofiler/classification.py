"""Classification backends for CellProfiler-compatible object measurements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)


class ObjectClassificationBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Object classification primitives keyed by memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def positive_label_ids(self, labels: np.ndarray) -> np.ndarray:
        """Return positive label ids present in ``labels``."""

    @abstractmethod
    def mean_intensity_values(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        """Return mean intensity for ``label_ids``."""

    @abstractmethod
    def apply_object_bins(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
        object_bins: np.ndarray,
    ) -> np.ndarray:
        """Map source labels to classification bin ids in one image pass."""


class NumbaNumpyObjectClassificationBackendStrategy(
    ObjectClassificationBackendStrategy
):
    """Numba-backed NumPy object classification primitives."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def positive_label_ids(self, labels: np.ndarray) -> np.ndarray:
        labels_array = np.asarray(labels, dtype=np.int32)
        if labels_array.size == 0:
            return np.zeros(0, dtype=np.int32)
        max_label = int(labels_array.max())
        if max_label <= 0:
            return np.zeros(0, dtype=np.int32)
        present = np.bincount(labels_array.ravel(), minlength=max_label + 1) > 0
        return np.flatnonzero(present[1:]).astype(np.int32) + 1

    def mean_intensity_values(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        return _mean_intensity_values_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(image, dtype=np.float64),
            np.asarray(label_ids, dtype=np.int32),
        )

    def apply_object_bins(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
        object_bins: np.ndarray,
    ) -> np.ndarray:
        return _apply_object_bins_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
            np.asarray(object_bins, dtype=np.int32),
        )


def object_classification_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> ObjectClassificationBackendStrategy:
    """Return the selected object-classification backend."""
    return ObjectClassificationBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


@njit(cache=True)
def _mean_intensity_values_numba(
    labels: np.ndarray,
    image: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    max_label = 0
    for i in range(label_ids.size):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id
    sums = np.zeros(max_label + 1, dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    rows, cols = labels.shape
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                sums[label] += image[row, col]
                counts[label] += 1
    values = np.empty(label_ids.size, dtype=np.float64)
    for i in range(label_ids.size):
        label = int(label_ids[i])
        if label <= 0 or label > max_label or counts[label] == 0:
            values[i] = np.nan
        else:
            values[i] = sums[label] / counts[label]
    return values


@njit(cache=True)
def _apply_object_bins_numba(
    labels: np.ndarray,
    label_ids: np.ndarray,
    object_bins: np.ndarray,
) -> np.ndarray:
    max_label = 0
    for i in range(label_ids.size):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id
    bin_by_label = np.zeros(max_label + 1, dtype=np.int32)
    count = label_ids.size
    if object_bins.size < count:
        count = object_bins.size
    for i in range(count):
        label = int(label_ids[i])
        if label > 0 and label <= max_label:
            bin_by_label[label] = int(object_bins[i])

    output = np.zeros(labels.shape, dtype=np.int32)
    rows, cols = labels.shape
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                output[row, col] = bin_by_label[label]
    return output


__all__ = [
    "NumbaNumpyObjectClassificationBackendStrategy",
    "ObjectClassificationBackendStrategy",
    "object_classification_backend",
]
