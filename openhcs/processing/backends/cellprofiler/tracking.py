"""Tracking backends for CellProfiler-compatible TrackObjects."""

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


class ObjectTrackingBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """TrackObjects primitives keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def label_centers(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return y/x centers for dense positive labels."""

    @abstractmethod
    def track_by_overlap(
        self,
        current_labels: np.ndarray,
        old_labels: np.ndarray | None,
        old_object_numbers: np.ndarray,
        max_object_number: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Assign track labels using maximum object overlap."""

    @abstractmethod
    def track_by_distance(
        self,
        current_labels: np.ndarray,
        old_labels: np.ndarray | None,
        old_object_numbers: np.ndarray,
        max_object_number: int,
        pixel_radius: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Assign track labels using nearest centroid within a radius."""


class NumbaNumpyObjectTrackingBackendStrategy(ObjectTrackingBackendStrategy):
    """Numba implementation of TrackObjects dense-label primitives."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def label_centers(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels_array = np.asarray(labels)
        label_count = int(labels_array.max()) if labels_array.size else 0
        if label_count == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        centers = _label_centers_numba(
            np.ascontiguousarray(labels_array),
            label_count,
        )
        return centers[1:, 0], centers[1:, 1]

    def track_by_overlap(
        self,
        current_labels: np.ndarray,
        old_labels: np.ndarray | None,
        old_object_numbers: np.ndarray,
        max_object_number: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        current = np.asarray(current_labels)
        current_count = int(current.max()) if current.size else 0
        if old_labels is None or current_count == 0:
            return _new_track_labels(current_count, max_object_number)

        old = np.asarray(old_labels)
        old_count = int(old.max()) if old.size else 0
        if old_count == 0:
            return _new_track_labels(current_count, max_object_number)

        return _track_by_overlap_numba(
            np.ascontiguousarray(current),
            np.ascontiguousarray(old),
            np.asarray(old_object_numbers, dtype=np.int64),
            int(max_object_number),
            current_count,
            old_count,
        )

    def track_by_distance(
        self,
        current_labels: np.ndarray,
        old_labels: np.ndarray | None,
        old_object_numbers: np.ndarray,
        max_object_number: int,
        pixel_radius: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        current = np.asarray(current_labels)
        current_count = int(current.max()) if current.size else 0
        if old_labels is None or current_count == 0:
            return _new_track_labels(current_count, max_object_number)

        old = np.asarray(old_labels)
        old_count = int(old.max()) if old.size else 0
        if old_count == 0:
            return _new_track_labels(current_count, max_object_number)

        return _track_by_distance_numba(
            np.ascontiguousarray(current),
            np.ascontiguousarray(old),
            np.asarray(old_object_numbers, dtype=np.int64),
            int(max_object_number),
            current_count,
            old_count,
            int(pixel_radius),
        )


def object_tracking_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> ObjectTrackingBackendStrategy:
    """Return the selected CellProfiler TrackObjects backend."""
    return ObjectTrackingBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    )


def _new_track_labels(
    object_count: int,
    max_object_number: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if object_count == 0:
        return (
            np.array([], dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            max_object_number,
        )
    new_labels = np.arange(1, object_count + 1, dtype=int) + max_object_number
    return (
        new_labels,
        np.zeros(object_count, dtype=int),
        np.zeros(object_count, dtype=int),
        max_object_number + object_count,
    )


@njit(cache=True)
def _label_centers_numba(labels: np.ndarray, label_count: int) -> np.ndarray:
    sums = np.zeros((label_count + 1, 2), dtype=np.float64)
    counts = np.zeros(label_count + 1, dtype=np.int64)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label_id = int(labels[y, x])
            if label_id > 0 and label_id <= label_count:
                sums[label_id, 0] += y
                sums[label_id, 1] += x
                counts[label_id] += 1

    centers = np.empty((label_count + 1, 2), dtype=np.float64)
    for label_id in range(label_count + 1):
        if counts[label_id] == 0:
            centers[label_id, 0] = np.nan
            centers[label_id, 1] = np.nan
        else:
            centers[label_id, 0] = sums[label_id, 0] / counts[label_id]
            centers[label_id, 1] = sums[label_id, 1] / counts[label_id]
    return centers


@njit(cache=True)
def _track_by_overlap_numba(
    current_labels: np.ndarray,
    old_labels: np.ndarray,
    old_object_numbers: np.ndarray,
    max_object_number: int,
    current_count: int,
    old_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    overlap = np.zeros((current_count + 1, old_count + 1), dtype=np.int64)
    height, width = current_labels.shape
    for y in range(height):
        for x in range(width):
            current_label = int(current_labels[y, x])
            old_label = int(old_labels[y, x])
            if (
                current_label > 0
                and current_label <= current_count
                and old_label > 0
                and old_label <= old_count
            ):
                overlap[current_label, old_label] += 1

    new_labels = np.zeros(current_count, dtype=np.int64)
    parent_object_numbers = np.zeros(current_count, dtype=np.int64)
    parent_image_numbers = np.zeros(current_count, dtype=np.int64)
    for current_index in range(current_count):
        current_label = current_index + 1
        best_old = 0
        best_overlap = 0
        for old_label in range(1, old_count + 1):
            current_overlap = overlap[current_label, old_label]
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_old = old_label
        if best_old > 0 and best_overlap > 0:
            new_labels[current_index] = old_object_numbers[best_old - 1]
            parent_object_numbers[current_index] = best_old
            parent_image_numbers[current_index] = 1
        else:
            max_object_number += 1
            new_labels[current_index] = max_object_number
    return new_labels, parent_object_numbers, parent_image_numbers, max_object_number


@njit(cache=True)
def _track_by_distance_numba(
    current_labels: np.ndarray,
    old_labels: np.ndarray,
    old_object_numbers: np.ndarray,
    max_object_number: int,
    current_count: int,
    old_count: int,
    pixel_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    current_centers = _label_centers_numba(current_labels, current_count)
    old_centers = _label_centers_numba(old_labels, old_count)
    new_labels = np.zeros(current_count, dtype=np.int64)
    parent_object_numbers = np.zeros(current_count, dtype=np.int64)
    parent_image_numbers = np.zeros(current_count, dtype=np.int64)
    radius_squared = float(pixel_radius * pixel_radius)

    for current_index in range(current_count):
        current_label = current_index + 1
        current_y = current_centers[current_label, 0]
        current_x = current_centers[current_label, 1]
        best_old = -1
        best_distance_squared = float((pixel_radius + 1) * (pixel_radius + 1))
        for old_index in range(old_count):
            old_label = old_index + 1
            old_y = old_centers[old_label, 0]
            old_x = old_centers[old_label, 1]
            dy = current_y - old_y
            dx = current_x - old_x
            distance_squared = dy * dy + dx * dx
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_old = old_index
        if best_old >= 0 and best_distance_squared <= radius_squared:
            new_labels[current_index] = old_object_numbers[best_old]
            parent_object_numbers[current_index] = best_old + 1
            parent_image_numbers[current_index] = 1
        else:
            max_object_number += 1
            new_labels[current_index] = max_object_number
    return new_labels, parent_object_numbers, parent_image_numbers, max_object_number


__all__ = [
    "NumbaNumpyObjectTrackingBackendStrategy",
    "ObjectTrackingBackendStrategy",
    "object_tracking_backend",
]
