"""Neighbor-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)


@dataclass(frozen=True, slots=True)
class NeighborTopologyArrays:
    """Dense per-variant-object neighbor topology measurements."""

    neighbor_count: np.ndarray
    touching_pixel_count: np.ndarray


@dataclass(frozen=True, slots=True)
class NeighborClosestArrays:
    """Dense nearest-neighbor vectors and final object IDs."""

    first_x_vector: np.ndarray
    first_y_vector: np.ndarray
    second_x_vector: np.ndarray
    second_y_vector: np.ndarray
    angle_between_neighbors: np.ndarray
    final_first_object_number: np.ndarray
    final_second_object_number: np.ndarray


class NeighborTopologyBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Neighbor topology operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def measure_topology(
        self,
        working_labels: np.ndarray,
        neighbor_working_labels: np.ndarray,
        perimeter_outlines: np.ndarray,
        object_numbers: np.ndarray,
        *,
        distance: int,
        neighbors_are_same_objects: bool,
        footprint: np.ndarray,
        touching_footprint: np.ndarray,
        variant_object_count: int,
        variant_neighbor_count: int,
    ) -> NeighborTopologyArrays:
        """Return neighbor counts and touching-pixel counts."""

    @abstractmethod
    def variant_numbers_for_final_labels(
        self,
        final_labels: np.ndarray,
        variant_labels: np.ndarray,
    ) -> np.ndarray:
        """Map final object IDs to their dominant variant object ID."""

    @abstractmethod
    def perimeter_counts(
        self,
        perimeter_outlines: np.ndarray,
        *,
        variant_object_count: int,
    ) -> np.ndarray:
        """Return per-object perimeter pixel counts."""

    @abstractmethod
    def closest_neighbors(
        self,
        object_centers: np.ndarray,
        neighbor_centers: np.ndarray,
        object_numbers: np.ndarray,
        neighbor_numbers: np.ndarray,
        final_has_pixels: np.ndarray,
        neighbor_has_pixels: np.ndarray,
        *,
        neighbors_are_same_objects: bool,
        variant_object_count: int,
        variant_neighbor_count: int,
        final_object_count: int,
    ) -> NeighborClosestArrays:
        """Return nearest-neighbor vectors and final object numbering."""


class NumbaNumpyNeighborTopologyBackendStrategy(NeighborTopologyBackendStrategy):
    """Numba-accelerated NumPy backend for neighbor topology."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def measure_topology(
        self,
        working_labels: np.ndarray,
        neighbor_working_labels: np.ndarray,
        perimeter_outlines: np.ndarray,
        object_numbers: np.ndarray,
        *,
        distance: int,
        neighbors_are_same_objects: bool,
        footprint: np.ndarray,
        touching_footprint: np.ndarray,
        variant_object_count: int,
        variant_neighbor_count: int,
    ) -> NeighborTopologyArrays:
        working_array = np.ascontiguousarray(working_labels, dtype=np.int32)
        neighbor_array = np.ascontiguousarray(neighbor_working_labels, dtype=np.int32)
        outline_array = np.ascontiguousarray(perimeter_outlines, dtype=np.int32)
        object_numbers_array = np.ascontiguousarray(object_numbers, dtype=np.int32)
        if working_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler neighbor topology currently supports 2-D labels."
            )
        if neighbor_array.shape != working_array.shape:
            raise ValueError(
                "Neighbor topology labels must share a shape; got "
                f"{working_array.shape!r} and {neighbor_array.shape!r}."
            )
        measured_object_mask = np.zeros(int(variant_object_count) + 1, dtype=np.bool_)
        for object_number in object_numbers_array:
            if 0 < object_number <= int(variant_object_count):
                measured_object_mask[int(object_number)] = True

        offset_y, offset_x = _footprint_offsets(footprint)
        touching_offset_y, touching_offset_x = _footprint_offsets(touching_footprint)
        neighbor_count, touching_pixel_count = _measure_neighbor_topology_numba(
            working_array,
            neighbor_array,
            outline_array,
            measured_object_mask,
            offset_y,
            offset_x,
            touching_offset_y,
            touching_offset_x,
            bool(neighbors_are_same_objects),
            int(variant_object_count),
            int(variant_neighbor_count),
        )
        return NeighborTopologyArrays(
            neighbor_count=neighbor_count,
            touching_pixel_count=touching_pixel_count,
        )

    def variant_numbers_for_final_labels(
        self,
        final_labels: np.ndarray,
        variant_labels: np.ndarray,
    ) -> np.ndarray:
        final_array = np.ascontiguousarray(final_labels, dtype=np.int32)
        variant_array = np.ascontiguousarray(variant_labels, dtype=np.int32)
        if final_array.shape != variant_array.shape:
            raise ValueError(
                "Final and variant labels must share a shape; got "
                f"{final_array.shape!r} and {variant_array.shape!r}."
            )
        final_count = int(final_array.max()) if final_array.size else 0
        variant_count = int(variant_array.max()) if variant_array.size else 0
        return _variant_numbers_for_final_labels_numba(
            final_array.ravel(),
            variant_array.ravel(),
            final_count,
            variant_count,
        )

    def perimeter_counts(
        self,
        perimeter_outlines: np.ndarray,
        *,
        variant_object_count: int,
    ) -> np.ndarray:
        outline_array = np.ascontiguousarray(perimeter_outlines, dtype=np.int32)
        return np.maximum(
            _perimeter_counts_numba(outline_array.ravel(), int(variant_object_count)),
            1,
        )

    def closest_neighbors(
        self,
        object_centers: np.ndarray,
        neighbor_centers: np.ndarray,
        object_numbers: np.ndarray,
        neighbor_numbers: np.ndarray,
        final_has_pixels: np.ndarray,
        neighbor_has_pixels: np.ndarray,
        *,
        neighbors_are_same_objects: bool,
        variant_object_count: int,
        variant_neighbor_count: int,
        final_object_count: int,
    ) -> NeighborClosestArrays:
        result = _closest_neighbors_numba(
            np.ascontiguousarray(object_centers, dtype=np.float64),
            np.ascontiguousarray(neighbor_centers, dtype=np.float64),
            np.ascontiguousarray(object_numbers, dtype=np.int32),
            np.ascontiguousarray(neighbor_numbers, dtype=np.int32),
            np.ascontiguousarray(final_has_pixels, dtype=np.bool_),
            np.ascontiguousarray(neighbor_has_pixels, dtype=np.bool_),
            bool(neighbors_are_same_objects),
            int(variant_object_count),
            int(variant_neighbor_count),
            int(final_object_count),
        )
        return NeighborClosestArrays(*result)


def neighbor_topology_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> NeighborTopologyBackendStrategy:
    """Return the selected neighbor topology backend."""
    return NeighborTopologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


def _footprint_offsets(footprint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    footprint_array = np.asarray(footprint, dtype=bool)
    if footprint_array.ndim != 2:
        raise NotImplementedError(
            "CellProfiler neighbor topology currently supports 2-D footprints."
        )
    center_y = footprint_array.shape[0] // 2
    center_x = footprint_array.shape[1] // 2
    coords = np.argwhere(footprint_array)
    offsets = np.ascontiguousarray(
        np.column_stack((coords[:, 0] - center_y, coords[:, 1] - center_x)),
        dtype=np.int64,
    )
    return offsets[:, 0], offsets[:, 1]


@njit(cache=True)
def _measure_neighbor_topology_numba(
    working_labels: np.ndarray,
    neighbor_working_labels: np.ndarray,
    perimeter_outlines: np.ndarray,
    measured_object_mask: np.ndarray,
    offset_y: np.ndarray,
    offset_x: np.ndarray,
    touching_offset_y: np.ndarray,
    touching_offset_x: np.ndarray,
    neighbors_are_same_objects: bool,
    variant_object_count: int,
    variant_neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = working_labels.shape
    adjacency = np.zeros(
        (variant_object_count, variant_neighbor_count + 1),
        dtype=np.bool_,
    )
    touching_pixel_count = np.zeros(variant_object_count, dtype=np.float64)

    for y in range(height):
        for x in range(width):
            object_number = working_labels[y, x]
            if (
                object_number <= 0
                or object_number > variant_object_count
                or not measured_object_mask[object_number]
            ):
                continue
            object_index = object_number - 1
            for offset_index in range(offset_y.size):
                neighbor_y = y + offset_y[offset_index]
                neighbor_x = x + offset_x[offset_index]
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                ):
                    continue
                neighbor_number = neighbor_working_labels[neighbor_y, neighbor_x]
                if neighbor_number <= 0 or neighbor_number > variant_neighbor_count:
                    continue
                if neighbors_are_same_objects and neighbor_number == object_number:
                    continue
                adjacency[object_index, neighbor_number] = True

            if perimeter_outlines[y, x] != object_number:
                continue
            for offset_index in range(touching_offset_y.size):
                neighbor_y = y + touching_offset_y[offset_index]
                neighbor_x = x + touching_offset_x[offset_index]
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                ):
                    continue
                if neighbors_are_same_objects:
                    touches = (
                        working_labels[neighbor_y, neighbor_x] != 0
                        and working_labels[neighbor_y, neighbor_x] != object_number
                    )
                else:
                    touches = neighbor_working_labels[neighbor_y, neighbor_x] != 0
                if touches:
                    touching_pixel_count[object_index] += 1.0
                    break

    neighbor_count = np.zeros(variant_object_count, dtype=np.float64)
    for object_index in range(variant_object_count):
        count = 0.0
        for neighbor_number in range(1, variant_neighbor_count + 1):
            if adjacency[object_index, neighbor_number]:
                count += 1.0
        neighbor_count[object_index] = count
    return neighbor_count, touching_pixel_count


@njit(cache=True)
def _variant_numbers_for_final_labels_numba(
    final_labels_flat: np.ndarray,
    variant_labels_flat: np.ndarray,
    final_count: int,
    variant_count: int,
) -> np.ndarray:
    numbers = np.zeros(final_count, dtype=np.int32)
    if final_count == 0 or variant_count == 0:
        return numbers
    overlaps = np.zeros((final_count + 1, variant_count + 1), dtype=np.int32)
    for index in range(final_labels_flat.size):
        final_number = final_labels_flat[index]
        variant_number = variant_labels_flat[index]
        if (
            final_number > 0
            and final_number <= final_count
            and variant_number > 0
            and variant_number <= variant_count
        ):
            overlaps[final_number, variant_number] += 1
    for final_number in range(1, final_count + 1):
        best_variant = 0
        best_count = 0
        for variant_number in range(1, variant_count + 1):
            count = overlaps[final_number, variant_number]
            if count > best_count:
                best_count = count
                best_variant = variant_number
        numbers[final_number - 1] = best_variant
    return numbers


@njit(cache=True)
def _perimeter_counts_numba(
    perimeter_outlines_flat: np.ndarray,
    variant_object_count: int,
) -> np.ndarray:
    counts = np.zeros(variant_object_count, dtype=np.float64)
    for index in range(perimeter_outlines_flat.size):
        object_number = perimeter_outlines_flat[index]
        if object_number > 0 and object_number <= variant_object_count:
            counts[object_number - 1] += 1.0
    return counts


@njit(cache=True)
def _closest_neighbors_numba(
    object_centers: np.ndarray,
    neighbor_centers: np.ndarray,
    object_numbers: np.ndarray,
    neighbor_numbers: np.ndarray,
    final_has_pixels: np.ndarray,
    neighbor_has_pixels: np.ndarray,
    neighbors_are_same_objects: bool,
    variant_object_count: int,
    variant_neighbor_count: int,
    final_object_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_x_vector = np.zeros(variant_object_count, dtype=np.float64)
    first_y_vector = np.zeros(variant_object_count, dtype=np.float64)
    second_x_vector = np.zeros(variant_object_count, dtype=np.float64)
    second_y_vector = np.zeros(variant_object_count, dtype=np.float64)
    angle = np.zeros(variant_object_count, dtype=np.float64)
    final_first_object_number = np.zeros(final_object_count, dtype=np.int64)
    final_second_object_number = np.zeros(final_object_count, dtype=np.int64)

    for object_index in range(variant_object_count):
        if object_index >= object_centers.shape[0]:
            continue
        object_y = object_centers[object_index, 0]
        object_x = object_centers[object_index, 1]
        if not (np.isfinite(object_y) and np.isfinite(object_x)):
            continue
        first_distance = np.inf
        second_distance = np.inf
        first_neighbor = -1
        second_neighbor = -1
        for neighbor_index in range(variant_neighbor_count):
            if neighbor_index >= neighbor_centers.shape[0]:
                continue
            if neighbors_are_same_objects and neighbor_index == object_index:
                continue
            neighbor_y = neighbor_centers[neighbor_index, 0]
            neighbor_x = neighbor_centers[neighbor_index, 1]
            if not (np.isfinite(neighbor_y) and np.isfinite(neighbor_x)):
                continue
            dy = object_y - neighbor_y
            dx = object_x - neighbor_x
            distance = dy * dy + dx * dx
            if distance < first_distance:
                second_distance = first_distance
                second_neighbor = first_neighbor
                first_distance = distance
                first_neighbor = neighbor_index
            elif distance < second_distance:
                second_distance = distance
                second_neighbor = neighbor_index
        if first_neighbor >= 0:
            first_x_vector[object_index] = (
                neighbor_centers[first_neighbor, 1] - object_x
            )
            first_y_vector[object_index] = (
                neighbor_centers[first_neighbor, 0] - object_y
            )
        if second_neighbor >= 0:
            second_x_vector[object_index] = (
                neighbor_centers[second_neighbor, 1] - object_x
            )
            second_y_vector[object_index] = (
                neighbor_centers[second_neighbor, 0] - object_y
            )

        norm1 = np.sqrt(
            first_x_vector[object_index] * first_x_vector[object_index]
            + first_y_vector[object_index] * first_y_vector[object_index]
        )
        norm2 = np.sqrt(
            second_x_vector[object_index] * second_x_vector[object_index]
            + second_y_vector[object_index] * second_y_vector[object_index]
        )
        if norm1 > 0.0 and norm2 > 0.0:
            dot = (
                first_x_vector[object_index] * second_x_vector[object_index]
                + first_y_vector[object_index] * second_y_vector[object_index]
            ) / (norm1 * norm2)
            if dot < -1.0:
                dot = -1.0
            elif dot > 1.0:
                dot = 1.0
            angle[object_index] = np.arccos(dot) * 180.0 / np.pi

    for final_object_index in range(final_object_count):
        if (
            final_object_index >= final_has_pixels.size
            or not final_has_pixels[final_object_index]
        ):
            continue
        object_number = object_numbers[final_object_index]
        object_index = object_number - 1
        if (
            object_index < 0
            or object_index >= variant_object_count
            or object_index >= object_centers.shape[0]
        ):
            continue
        object_y = object_centers[object_index, 0]
        object_x = object_centers[object_index, 1]
        if not (np.isfinite(object_y) and np.isfinite(object_x)):
            continue

        first_distance = np.inf
        second_distance = np.inf
        first_final_neighbor = 0
        second_final_neighbor = 0
        for final_neighbor_index in range(neighbor_numbers.size):
            if (
                final_neighbor_index >= neighbor_has_pixels.size
                or not neighbor_has_pixels[final_neighbor_index]
            ):
                continue
            if neighbors_are_same_objects and final_neighbor_index == final_object_index:
                continue
            neighbor_number = neighbor_numbers[final_neighbor_index]
            neighbor_index = neighbor_number - 1
            if (
                neighbor_index < 0
                or neighbor_index >= variant_neighbor_count
                or neighbor_index >= neighbor_centers.shape[0]
            ):
                continue
            neighbor_y = neighbor_centers[neighbor_index, 0]
            neighbor_x = neighbor_centers[neighbor_index, 1]
            if not (np.isfinite(neighbor_y) and np.isfinite(neighbor_x)):
                continue
            dy = object_y - neighbor_y
            dx = object_x - neighbor_x
            distance = dy * dy + dx * dx
            if distance < first_distance:
                second_distance = first_distance
                second_final_neighbor = first_final_neighbor
                first_distance = distance
                first_final_neighbor = final_neighbor_index + 1
            elif distance < second_distance:
                second_distance = distance
                second_final_neighbor = final_neighbor_index + 1
        final_first_object_number[final_object_index] = first_final_neighbor
        final_second_object_number[final_object_index] = second_final_neighbor

    return (
        first_x_vector,
        first_y_vector,
        second_x_vector,
        second_y_vector,
        angle,
        final_first_object_number,
        final_second_object_number,
    )


__all__ = [
    "NeighborTopologyArrays",
    "NeighborClosestArrays",
    "NeighborTopologyBackendStrategy",
    "NumbaNumpyNeighborTopologyBackendStrategy",
    "neighbor_topology_backend",
]
