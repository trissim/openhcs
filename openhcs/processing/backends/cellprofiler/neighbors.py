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


__all__ = [
    "NeighborTopologyArrays",
    "NeighborTopologyBackendStrategy",
    "NumbaNumpyNeighborTopologyBackendStrategy",
    "neighbor_topology_backend",
]
