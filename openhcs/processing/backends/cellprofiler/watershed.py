"""Watershed backend strategies for CellProfiler-compatible processing."""

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


class LegacyWatershedBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Legacy watershed operations keyed by OpenHCS memory type."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def legacy_watershed(
        self,
        image: np.ndarray,
        *,
        markers: np.ndarray,
        mask: np.ndarray,
        connectivity: int | np.ndarray = 1,
    ) -> np.ndarray:
        """Run CellProfiler 4.2/skimage 0.18 watershed semantics."""


class NumpyLegacyWatershedBackendStrategy(LegacyWatershedBackendStrategy):
    """NumPy-memory reference legacy watershed backend."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False

    def legacy_watershed(
        self,
        image: np.ndarray,
        *,
        markers: np.ndarray,
        mask: np.ndarray,
        connectivity: int | np.ndarray = 1,
    ) -> np.ndarray:
        return _cellprofiler_legacy_watershed_numpy(
            image,
            markers=markers,
            mask=mask,
            connectivity=connectivity,
            prefer_fast=False,
        )


class NumbaNumpyLegacyWatershedBackendStrategy(LegacyWatershedBackendStrategy):
    """NumPy-memory legacy watershed backend with required Numba acceleration."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def legacy_watershed(
        self,
        image: np.ndarray,
        *,
        markers: np.ndarray,
        mask: np.ndarray,
        connectivity: int | np.ndarray = 1,
    ) -> np.ndarray:
        return _cellprofiler_legacy_watershed_numpy(
            image,
            markers=markers,
            mask=mask,
            connectivity=connectivity,
            prefer_fast=True,
        )


class CellProfiler4DistanceMarkerBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Build CellProfiler 4 distance-watershed markers through a typed backend."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def distance_markers(
        self,
        distance: np.ndarray,
        peak_footprint: np.ndarray,
        seed_connectivity: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Return seed markers, marker count, and the regional-maxima mask."""


class MahotasCellProfiler4DistanceMarkerBackendStrategy(
    CellProfiler4DistanceMarkerBackendStrategy
):
    """Reference CellProfiler 4 marker backend."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE)
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def distance_markers(
        self,
        distance: np.ndarray,
        peak_footprint: np.ndarray,
        seed_connectivity: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        import mahotas

        peaks = mahotas.regmax(distance, peak_footprint)
        seed_markers, marker_count = mahotas.label(peaks, seed_connectivity)
        return seed_markers, int(marker_count), peaks


class NumbaCellProfiler4DistanceMarkerBackendStrategy(
    CellProfiler4DistanceMarkerBackendStrategy
):
    """Exact numba backend for CellProfiler 4 regional-maxima markers."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA)
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def distance_markers(
        self,
        distance: np.ndarray,
        peak_footprint: np.ndarray,
        seed_connectivity: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        import mahotas
        import scipy.ndimage

        distance_array = np.ascontiguousarray(distance)
        peak_footprint_array = np.asarray(peak_footprint, dtype=bool)
        if distance_array.ndim != 3 or peak_footprint_array.ndim != 3:
            return MahotasCellProfiler4DistanceMarkerBackendStrategy().distance_markers(
                distance_array,
                peak_footprint_array,
                seed_connectivity,
            )
        local_maxima = (
            distance_array
            == scipy.ndimage.maximum_filter(
                distance_array,
                footprint=peak_footprint_array,
                mode="constant",
                cval=0,
            )
        ) & (distance_array > 0)
        peaks = _cellprofiler4_regional_maxima_from_candidates_3d_numba(
            distance_array,
            np.ascontiguousarray(local_maxima),
            _footprint_offsets_3d(peak_footprint_array),
        )
        seed_markers, marker_count = mahotas.label(peaks, seed_connectivity)
        return seed_markers, int(marker_count), peaks

    def prepare_backend(self) -> None:
        distance = np.zeros((8, 16, 16), dtype=np.uint8)
        distance[2:6, 4:12, 4:12] = 1
        distance[3:5, 6:10, 6:10] = 2
        footprint = np.ones((4, 4, 4), dtype=bool)
        connectivity = np.ones((4, 4, 4), dtype=bool)
        self.distance_markers(distance, footprint, connectivity)


def cellprofiler_legacy_watershed(
    image: np.ndarray,
    *,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int | np.ndarray = 1,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Run CellProfiler 4.2/skimage 0.18 watershed semantics."""
    return LegacyWatershedBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    ).legacy_watershed(
        image,
        markers=markers,
        mask=mask,
        connectivity=connectivity,
    )


def _cellprofiler_legacy_watershed_numpy(
    image: np.ndarray,
    *,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int | np.ndarray = 1,
    prefer_fast: bool = True,
) -> np.ndarray:
    from skimage.morphology._util import (
        _offsets_to_raveled_neighbors,
        _validate_connectivity,
    )
    from skimage.util import crop

    image_array = np.asarray(image, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    marker_array = np.asarray(markers) * mask_array
    if marker_array.shape != image_array.shape:
        raise ValueError("markers must have the same shape as image")
    if mask_array.shape != image_array.shape:
        raise ValueError("mask must have the same shape as image")
    if _is_planewise_watershed(
        image_array,
        connectivity,
    ):
        return _cellprofiler_legacy_watershed_planewise(
            image_array,
            markers=marker_array,
            mask=mask_array,
            connectivity=connectivity,
            prefer_fast=prefer_fast,
        )

    connectivity_array, offset = _validate_connectivity(
        image_array.ndim,
        connectivity,
        None,
    )
    pad_width = [(int(width), int(width)) for width in offset]
    padded_image = np.pad(image_array, pad_width, mode="constant")
    padded_mask = np.pad(
        mask_array.astype(np.bool_, copy=False),
        pad_width,
        mode="constant",
    ).ravel()
    output = np.pad(
        marker_array.astype(np.int32, copy=False),
        pad_width,
        mode="constant",
    )
    output_flat = output.ravel()
    image_flat = padded_image.ravel()
    neighbor_offsets = _offsets_to_raveled_neighbors(
        padded_image.shape,
        connectivity_array,
        center=offset,
    ).astype(np.int64, copy=False)
    marker_locations = np.flatnonzero(output_flat).astype(np.int64, copy=False)

    if prefer_fast:
        _legacy_watershed_raveled_numba(
            image_flat,
            padded_mask,
            output_flat,
            neighbor_offsets,
            marker_locations,
        )
        return crop(output, pad_width, copy=True)

    _legacy_watershed_raveled_python(
        image_flat,
        padded_mask,
        output_flat,
        neighbor_offsets,
        marker_locations,
    )
    return crop(output, pad_width, copy=True)


def _footprint_offsets_3d(footprint: np.ndarray) -> np.ndarray:
    center = np.asarray(footprint.shape, dtype=np.int64) // 2
    offsets = np.argwhere(footprint).astype(np.int64) - center
    return np.ascontiguousarray(offsets[np.any(offsets != 0, axis=1)])


@njit(cache=True)
def _cellprofiler4_regional_maxima_from_candidates_3d_numba(
    image: np.ndarray,
    candidates: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    z_size, y_size, x_size = image.shape
    voxel_count = image.size
    visited = np.zeros(voxel_count, np.uint8)
    output = np.zeros(image.shape, np.bool_)
    stack = np.empty(voxel_count, np.int64)
    component = np.empty(voxel_count, np.int64)
    plane_size = y_size * x_size

    for start_index in range(voxel_count):
        z_start = start_index // plane_size
        start_remainder = start_index - z_start * plane_size
        y_start = start_remainder // x_size
        x_start = start_remainder - y_start * x_size
        if not candidates[z_start, y_start, x_start] or visited[start_index]:
            continue

        value = image[z_start, y_start, x_start]
        visited[start_index] = 1
        stack_size = 1
        stack[0] = start_index
        component_size = 0
        has_higher_neighbor = False

        while stack_size > 0:
            stack_size -= 1
            index = stack[stack_size]
            component[component_size] = index
            component_size += 1
            z_index = index // plane_size
            remainder = index - z_index * plane_size
            y_index = remainder // x_size
            x_index = remainder - y_index * x_size

            for offset_index in range(offsets.shape[0]):
                z_neighbor = z_index + offsets[offset_index, 0]
                y_neighbor = y_index + offsets[offset_index, 1]
                x_neighbor = x_index + offsets[offset_index, 2]
                if (
                    z_neighbor < 0
                    or y_neighbor < 0
                    or x_neighbor < 0
                    or z_neighbor >= z_size
                    or y_neighbor >= y_size
                    or x_neighbor >= x_size
                ):
                    continue

                neighbor_value = image[z_neighbor, y_neighbor, x_neighbor]
                if neighbor_value > value:
                    has_higher_neighbor = True
                elif neighbor_value == value:
                    neighbor_index = (
                        z_neighbor * plane_size
                        + y_neighbor * x_size
                        + x_neighbor
                    )
                    if not visited[neighbor_index]:
                        visited[neighbor_index] = 1
                        stack[stack_size] = neighbor_index
                        stack_size += 1

        if not has_higher_neighbor:
            for component_index in range(component_size):
                index = component[component_index]
                z_index = index // plane_size
                remainder = index - z_index * plane_size
                y_index = remainder // x_size
                x_index = remainder - y_index * x_size
                output[z_index, y_index, x_index] = True

    return output


def _is_planewise_watershed(
    image: np.ndarray,
    connectivity: int | np.ndarray,
) -> bool:
    if image.ndim <= 2:
        return False
    if np.isscalar(connectivity):
        return False
    return np.asarray(connectivity).ndim == 2


def _cellprofiler_legacy_watershed_planewise(
    image: np.ndarray,
    *,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int | np.ndarray,
    prefer_fast: bool,
) -> np.ndarray:
    output = np.empty(markers.shape, dtype=np.int32)
    image_planes = image.reshape((-1, *image.shape[-2:]))
    marker_planes = markers.reshape((-1, *markers.shape[-2:]))
    mask_planes = mask.reshape((-1, *mask.shape[-2:]))
    output_planes = output.reshape((-1, *output.shape[-2:]))
    for plane_index in range(image_planes.shape[0]):
        output_planes[plane_index] = _cellprofiler_legacy_watershed_numpy(
            image_planes[plane_index],
            markers=marker_planes[plane_index],
            mask=mask_planes[plane_index],
            connectivity=connectivity,
            prefer_fast=prefer_fast,
        )
    return output


def _legacy_watershed_raveled_python(
    image_flat: np.ndarray,
    mask_flat: np.ndarray,
    output_flat: np.ndarray,
    neighbor_offsets: np.ndarray,
    marker_locations: np.ndarray,
) -> None:
    heap_values: list[float] = []
    heap_ages: list[int] = []
    heap_indexes: list[int] = []
    heap_sources: list[int] = []

    for marker_location in marker_locations:
        location = int(marker_location)
        _heap_push_python(
            heap_values,
            heap_ages,
            heap_indexes,
            heap_sources,
            float(image_flat[location]),
            0,
            location,
            location,
        )

    age = 1
    while heap_values:
        _value, _entry_age, index, source = _heap_pop_python(
            heap_values,
            heap_ages,
            heap_indexes,
            heap_sources,
        )
        label = int(output_flat[index])
        if label == 0:
            label = int(output_flat[source])
        for offset_value in neighbor_offsets:
            neighbor_index = int(index + offset_value)
            if not mask_flat[neighbor_index] or output_flat[neighbor_index] != 0:
                continue
            output_flat[neighbor_index] = label
            age += 1
            _heap_push_python(
                heap_values,
                heap_ages,
                heap_indexes,
                heap_sources,
                float(image_flat[neighbor_index]),
                age,
                neighbor_index,
                source,
            )


def _heap_item_less_python(
    left_value: float,
    left_age: int,
    right_value: float,
    right_age: int,
) -> bool:
    if left_value != right_value:
        return left_value < right_value
    return left_age < right_age


def _heap_swap_python(
    values: list[float],
    ages: list[int],
    indexes: list[int],
    sources: list[int],
    left: int,
    right: int,
) -> None:
    values[left], values[right] = values[right], values[left]
    ages[left], ages[right] = ages[right], ages[left]
    indexes[left], indexes[right] = indexes[right], indexes[left]
    sources[left], sources[right] = sources[right], sources[left]


def _heap_push_python(
    values: list[float],
    ages: list[int],
    indexes: list[int],
    sources: list[int],
    value: float,
    age: int,
    index: int,
    source: int,
) -> None:
    values.append(value)
    ages.append(age)
    indexes.append(index)
    sources.append(source)
    position = len(values) - 1
    while position > 0:
        parent = (position - 1) // 2
        if not _heap_item_less_python(
            values[position],
            ages[position],
            values[parent],
            ages[parent],
        ):
            break
        _heap_swap_python(values, ages, indexes, sources, position, parent)
        position = parent


def _heap_pop_python(
    values: list[float],
    ages: list[int],
    indexes: list[int],
    sources: list[int],
) -> tuple[float, int, int, int]:
    value = values[0]
    age = ages[0]
    index = indexes[0]
    source = sources[0]
    last = len(values) - 1
    if last == 0:
        values.pop()
        ages.pop()
        indexes.pop()
        sources.pop()
        return value, age, index, source

    values[0] = values.pop()
    ages[0] = ages.pop()
    indexes[0] = indexes.pop()
    sources[0] = sources.pop()
    size = len(values)
    position = 0
    while True:
        left = position * 2 + 1
        right = left + 1
        if left >= size:
            break
        smallest = left
        if right < size and _heap_item_less_python(
            values[right],
            ages[right],
            values[left],
            ages[left],
        ):
            smallest = right
        if not _heap_item_less_python(
            values[smallest],
            ages[smallest],
            values[position],
            ages[position],
        ):
            break
        _heap_swap_python(values, ages, indexes, sources, position, smallest)
        position = smallest
    return value, age, index, source


@njit(cache=True)
def _heap_item_less(
    left_value: float,
    left_age: int,
    left_index: int,
    left_source: int,
    right_value: float,
    right_age: int,
    right_index: int,
    right_source: int,
) -> bool:
    if left_value != right_value:
        return left_value < right_value
    return left_age < right_age


@njit(cache=True)
def _heap_swap(
    values: np.ndarray,
    ages: np.ndarray,
    indexes: np.ndarray,
    sources: np.ndarray,
    left: int,
    right: int,
) -> None:
    value = values[left]
    age = ages[left]
    index = indexes[left]
    source = sources[left]
    values[left] = values[right]
    ages[left] = ages[right]
    indexes[left] = indexes[right]
    sources[left] = sources[right]
    values[right] = value
    ages[right] = age
    indexes[right] = index
    sources[right] = source


@njit(cache=True)
def _heap_push(
    values: np.ndarray,
    ages: np.ndarray,
    indexes: np.ndarray,
    sources: np.ndarray,
    size: int,
    value: float,
    age: int,
    index: int,
    source: int,
) -> int:
    values[size] = value
    ages[size] = age
    indexes[size] = index
    sources[size] = source
    size += 1
    position = size - 1
    while position > 0:
        parent = (position - 1) // 2
        if not _heap_item_less(
            values[position],
            ages[position],
            indexes[position],
            sources[position],
            values[parent],
            ages[parent],
            indexes[parent],
            sources[parent],
        ):
            break
        _heap_swap(values, ages, indexes, sources, position, parent)
        position = parent
    return size


@njit(cache=True)
def _heap_pop(
    values: np.ndarray,
    ages: np.ndarray,
    indexes: np.ndarray,
    sources: np.ndarray,
    size: int,
) -> tuple[int, float, int, int, int]:
    value = values[0]
    age = ages[0]
    index = indexes[0]
    source = sources[0]
    size -= 1
    if size > 0:
        values[0] = values[size]
        ages[0] = ages[size]
        indexes[0] = indexes[size]
        sources[0] = sources[size]
        position = 0
        while True:
            left = position * 2 + 1
            right = left + 1
            if left >= size:
                break
            smallest = left
            if right < size and _heap_item_less(
                values[right],
                ages[right],
                indexes[right],
                sources[right],
                values[left],
                ages[left],
                indexes[left],
                sources[left],
            ):
                smallest = right
            if not _heap_item_less(
                values[smallest],
                ages[smallest],
                indexes[smallest],
                sources[smallest],
                values[position],
                ages[position],
                indexes[position],
                sources[position],
            ):
                break
            _heap_swap(values, ages, indexes, sources, position, smallest)
            position = smallest
    return size, value, age, index, source


@njit(cache=True)
def _legacy_watershed_raveled_numba(
    image_flat: np.ndarray,
    mask_flat: np.ndarray,
    output_flat: np.ndarray,
    neighbor_offsets: np.ndarray,
    marker_locations: np.ndarray,
) -> None:
    capacity = output_flat.size
    heap_values = np.empty(capacity, dtype=np.float64)
    heap_ages = np.empty(capacity, dtype=np.int64)
    heap_indexes = np.empty(capacity, dtype=np.int64)
    heap_sources = np.empty(capacity, dtype=np.int64)
    heap_size = 0

    for marker_location in marker_locations:
        location = int(marker_location)
        heap_size = _heap_push(
            heap_values,
            heap_ages,
            heap_indexes,
            heap_sources,
            heap_size,
            float(image_flat[location]),
            0,
            location,
            location,
        )

    age = 1
    while heap_size > 0:
        heap_size, _value, _entry_age, index, source = _heap_pop(
            heap_values,
            heap_ages,
            heap_indexes,
            heap_sources,
            heap_size,
        )
        label = int(output_flat[index])
        if label == 0:
            label = int(output_flat[source])
        for offset_value in neighbor_offsets:
            neighbor_index = int(index + offset_value)
            if (not mask_flat[neighbor_index]) or output_flat[neighbor_index] != 0:
                continue
            output_flat[neighbor_index] = label
            age += 1
            heap_size = _heap_push(
                heap_values,
                heap_ages,
                heap_indexes,
                heap_sources,
                heap_size,
                float(image_flat[neighbor_index]),
                age,
                neighbor_index,
                source,
            )


__all__ = [
    "LegacyWatershedBackendStrategy",
    "NumbaNumpyLegacyWatershedBackendStrategy",
    "NumpyLegacyWatershedBackendStrategy",
    "cellprofiler_legacy_watershed",
]
