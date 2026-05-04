"""Intensity-measurement backends for CellProfiler-compatible processing."""

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
class ObjectIntensityArrays:
    """Dense per-object intensity measurement arrays."""

    object_labels: np.ndarray
    integrated_intensity: np.ndarray
    mean_intensity: np.ndarray
    std_intensity: np.ndarray
    min_intensity: np.ndarray
    max_intensity: np.ndarray
    integrated_intensity_edge: np.ndarray
    mean_intensity_edge: np.ndarray
    std_intensity_edge: np.ndarray
    min_intensity_edge: np.ndarray
    max_intensity_edge: np.ndarray
    mass_displacement: np.ndarray
    lower_quartile_intensity: np.ndarray
    median_intensity: np.ndarray
    mad_intensity: np.ndarray
    upper_quartile_intensity: np.ndarray
    center_mass_intensity_x: np.ndarray
    center_mass_intensity_y: np.ndarray
    max_intensity_x: np.ndarray
    max_intensity_y: np.ndarray


class ObjectIntensityBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Object-intensity operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def measure(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> ObjectIntensityArrays:
        """Measure object intensity arrays for one image plane."""


class NumbaNumpyObjectIntensityBackendStrategy(ObjectIntensityBackendStrategy):
    """Numba-accelerated NumPy object-intensity backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def measure(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> ObjectIntensityArrays:
        image_array = np.ascontiguousarray(image, dtype=np.float64)
        label_array = np.ascontiguousarray(labels, dtype=np.int64)
        if image_array.ndim != 2 or label_array.ndim != 2:
            raise NotImplementedError(
                "Numba object-intensity backend currently supports 2-D arrays."
            )
        if image_array.shape != label_array.shape:
            raise ValueError("image and labels must have matching shapes.")

        object_labels = np.unique(label_array)
        object_labels = np.ascontiguousarray(
            object_labels[object_labels > 0],
            dtype=np.int64,
        )
        object_count = int(object_labels.size)
        if object_count == 0:
            return _empty_intensity_arrays(object_labels)

        max_label = int(label_array.max())
        label_to_index = np.full(max_label + 1, -1, dtype=np.int64)
        for index, object_label in enumerate(object_labels):
            label_to_index[int(object_label)] = index

        arrays = _object_intensity_scan_numba(
            image_array,
            label_array,
            object_labels,
            label_to_index,
        )
        lower, median, upper, mad = _object_intensity_quantiles(
            image_array,
            label_array,
            object_labels,
            label_to_index,
            arrays[0].astype(np.int64, copy=False),
        )
        return ObjectIntensityArrays(
            object_labels=object_labels.astype(np.int32, copy=False),
            integrated_intensity=arrays[1],
            mean_intensity=arrays[2],
            std_intensity=arrays[3],
            min_intensity=arrays[4],
            max_intensity=arrays[5],
            integrated_intensity_edge=arrays[6],
            mean_intensity_edge=arrays[7],
            std_intensity_edge=arrays[8],
            min_intensity_edge=arrays[9],
            max_intensity_edge=arrays[10],
            mass_displacement=arrays[11],
            lower_quartile_intensity=lower,
            median_intensity=median,
            mad_intensity=mad,
            upper_quartile_intensity=upper,
            center_mass_intensity_x=arrays[12],
            center_mass_intensity_y=arrays[13],
            max_intensity_x=arrays[14],
            max_intensity_y=arrays[15],
        )


def object_intensity_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> ObjectIntensityBackendStrategy:
    """Return the selected object-intensity backend."""
    return ObjectIntensityBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


def _empty_intensity_arrays(object_labels: np.ndarray) -> ObjectIntensityArrays:
    empty = np.zeros(0, dtype=float)
    return ObjectIntensityArrays(
        object_labels=object_labels.astype(np.int32, copy=False),
        integrated_intensity=empty,
        mean_intensity=empty,
        std_intensity=empty,
        min_intensity=empty,
        max_intensity=empty,
        integrated_intensity_edge=empty,
        mean_intensity_edge=empty,
        std_intensity_edge=empty,
        min_intensity_edge=empty,
        max_intensity_edge=empty,
        mass_displacement=empty,
        lower_quartile_intensity=empty,
        median_intensity=empty,
        mad_intensity=empty,
        upper_quartile_intensity=empty,
        center_mass_intensity_x=empty,
        center_mass_intensity_y=empty,
        max_intensity_x=empty,
        max_intensity_y=empty,
    )


def _object_intensity_quantiles(
    image: np.ndarray,
    labels: np.ndarray,
    object_labels: np.ndarray,
    label_to_index: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _object_intensity_quantiles_grouped_numba(
        image,
        labels,
        label_to_index,
        counts.astype(np.int64, copy=False),
    )


@njit(cache=True)
def _object_intensity_quantiles_grouped_numba(
    image: np.ndarray,
    labels: np.ndarray,
    label_to_index: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_count = counts.size
    lower = np.zeros(object_count, dtype=np.float64)
    median = np.zeros(object_count, dtype=np.float64)
    upper = np.zeros(object_count, dtype=np.float64)
    mad = np.zeros(object_count, dtype=np.float64)

    total_count = 0
    for index in range(object_count):
        total_count += int(counts[index])
    if total_count <= 0:
        return lower, median, upper, mad

    offsets = np.empty(object_count + 1, dtype=np.int64)
    offsets[0] = 0
    for index in range(object_count):
        offsets[index + 1] = offsets[index] + int(counts[index])

    write_offsets = offsets[:-1].copy()
    values = np.empty(total_count, dtype=np.float64)
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0 or label >= label_to_index.size:
                continue
            index = int(label_to_index[label])
            if index < 0:
                continue
            value = float(image[y, x])
            if not np.isfinite(value):
                continue
            offset = write_offsets[index]
            values[offset] = value
            write_offsets[index] = offset + 1

    for index in range(object_count):
        start = int(offsets[index])
        count = int(counts[index])
        if count <= 0:
            continue
        sorted_group = np.sort(values[start:start + count].copy())
        lower[index] = _quantile_from_dense_sorted_group(sorted_group, 0.25)
        median[index] = _quantile_from_dense_sorted_group(sorted_group, 0.5)
        upper[index] = _quantile_from_dense_sorted_group(sorted_group, 0.75)

    write_offsets = offsets[:-1].copy()
    deviations = np.empty(total_count, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0 or label >= label_to_index.size:
                continue
            index = int(label_to_index[label])
            if index < 0:
                continue
            value = float(image[y, x])
            if not np.isfinite(value):
                continue
            offset = write_offsets[index]
            deviations[offset] = abs(value - median[index])
            write_offsets[index] = offset + 1

    for index in range(object_count):
        start = int(offsets[index])
        count = int(counts[index])
        if count <= 0:
            continue
        sorted_group = np.sort(deviations[start:start + count].copy())
        mad[index] = _quantile_from_dense_sorted_group(sorted_group, 0.5)

    return lower, median, upper, mad


@njit(cache=True)
def _quantile_from_dense_sorted_group(
    sorted_values: np.ndarray,
    fraction: float,
) -> float:
    count = sorted_values.size
    if count <= 0:
        return 0.0
    qindex = count * fraction
    low = int(qindex)
    qfraction = qindex - low
    last = count - 1
    if low < last:
        return sorted_values[low] * (1.0 - qfraction) + sorted_values[low + 1] * qfraction
    return sorted_values[last]


@njit(cache=True)
def _group_starts_numba(
    sorted_labels: np.ndarray,
    label_to_index: np.ndarray,
    starts: np.ndarray,
) -> None:
    for offset in range(sorted_labels.size):
        label = sorted_labels[offset]
        if label < 0 or label >= label_to_index.size:
            continue
        index = label_to_index[label]
        if index >= 0 and starts[index] < 0:
            starts[index] = offset


@njit(cache=True)
def _quartiles_from_sorted_numba(
    sorted_values: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_count = counts.size
    lower = np.zeros(object_count, dtype=np.float64)
    median = np.zeros(object_count, dtype=np.float64)
    upper = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        lower[index] = _quantile_from_sorted_group(
            sorted_values,
            starts[index],
            counts[index],
            0.25,
        )
        median[index] = _quantile_from_sorted_group(
            sorted_values,
            starts[index],
            counts[index],
            0.5,
        )
        upper[index] = _quantile_from_sorted_group(
            sorted_values,
            starts[index],
            counts[index],
            0.75,
        )
    return lower, median, upper


@njit(cache=True)
def _median_from_sorted_numba(
    sorted_values: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    object_count = counts.size
    output = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        output[index] = _quantile_from_sorted_group(
            sorted_values,
            starts[index],
            counts[index],
            0.5,
        )
    return output


@njit(cache=True)
def _quantile_from_sorted_group(
    sorted_values: np.ndarray,
    start: int,
    count: int,
    fraction: float,
) -> float:
    if count <= 0:
        return 0.0
    qindex = start + count * fraction
    low = int(qindex)
    qfraction = qindex - low
    last = start + count - 1
    if low < last:
        return sorted_values[low] * (1.0 - qfraction) + sorted_values[low + 1] * qfraction
    return sorted_values[last]


@njit(cache=True)
def _object_intensity_scan_numba(
    image: np.ndarray,
    labels: np.ndarray,
    object_labels: np.ndarray,
    label_to_index: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    height, width = image.shape
    object_count = object_labels.size
    counts = np.zeros(object_count, dtype=np.float64)
    sums = np.zeros(object_count, dtype=np.float64)
    sumsq = np.zeros(object_count, dtype=np.float64)
    min_values = np.full(object_count, np.inf, dtype=np.float64)
    max_values = np.full(object_count, -np.inf, dtype=np.float64)
    sum_x = np.zeros(object_count, dtype=np.float64)
    sum_y = np.zeros(object_count, dtype=np.float64)
    weighted_x = np.zeros(object_count, dtype=np.float64)
    weighted_y = np.zeros(object_count, dtype=np.float64)
    max_x = np.zeros(object_count, dtype=np.float64)
    max_y = np.zeros(object_count, dtype=np.float64)

    edge_counts = np.zeros(object_count, dtype=np.float64)
    edge_sums = np.zeros(object_count, dtype=np.float64)
    edge_sumsq = np.zeros(object_count, dtype=np.float64)
    edge_min_values = np.full(object_count, np.inf, dtype=np.float64)
    edge_max_values = np.full(object_count, -np.inf, dtype=np.float64)

    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label <= 0 or label >= label_to_index.size:
                continue
            index = label_to_index[label]
            if index < 0:
                continue
            value = image[y, x]
            if not np.isfinite(value):
                continue

            counts[index] += 1.0
            sums[index] += value
            sumsq[index] += value * value
            sum_x[index] += x
            sum_y[index] += y
            weighted_x[index] += x * value
            weighted_y[index] += y * value
            if value < min_values[index]:
                min_values[index] = value
            if value > max_values[index]:
                max_values[index] = value
                max_x[index] = x
                max_y[index] = y

            if _is_inner_boundary_pixel(labels, y, x, label):
                edge_counts[index] += 1.0
                edge_sums[index] += value
                edge_sumsq[index] += value * value
                if value < edge_min_values[index]:
                    edge_min_values[index] = value
                if value > edge_max_values[index]:
                    edge_max_values[index] = value

    means = np.zeros(object_count, dtype=np.float64)
    stds = np.zeros(object_count, dtype=np.float64)
    edge_means = np.zeros(object_count, dtype=np.float64)
    edge_stds = np.zeros(object_count, dtype=np.float64)
    mass_displacement = np.zeros(object_count, dtype=np.float64)
    center_mass_x = np.zeros(object_count, dtype=np.float64)
    center_mass_y = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        if counts[index] > 0.0:
            means[index] = sums[index] / counts[index]
            variance = sumsq[index] / counts[index] - means[index] * means[index]
            if variance < 0.0 and variance > -1e-15:
                variance = 0.0
            stds[index] = np.sqrt(variance)
            center_x = sum_x[index] / counts[index]
            center_y = sum_y[index] / counts[index]
            if sums[index] != 0.0:
                center_mass_x[index] = weighted_x[index] / sums[index]
                center_mass_y[index] = weighted_y[index] / sums[index]
            diff_x = center_x - center_mass_x[index]
            diff_y = center_y - center_mass_y[index]
            mass_displacement[index] = np.sqrt(diff_x * diff_x + diff_y * diff_y)
        else:
            min_values[index] = 0.0
            max_values[index] = 0.0

        if edge_counts[index] > 0.0:
            edge_means[index] = edge_sums[index] / edge_counts[index]
            edge_variance = (
                edge_sumsq[index] / edge_counts[index]
                - edge_means[index] * edge_means[index]
            )
            if edge_variance < 0.0 and edge_variance > -1e-15:
                edge_variance = 0.0
            edge_stds[index] = np.sqrt(edge_variance)
        else:
            edge_min_values[index] = 0.0
            edge_max_values[index] = 0.0

    return (
        counts,
        sums,
        means,
        stds,
        min_values,
        max_values,
        edge_sums,
        edge_means,
        edge_stds,
        edge_min_values,
        edge_max_values,
        mass_displacement,
        center_mass_x,
        center_mass_y,
        max_x,
        max_y,
    )


@njit(cache=True)
def _is_inner_boundary_pixel(
    labels: np.ndarray,
    y: int,
    x: int,
    label: int,
) -> bool:
    height, width = labels.shape
    if y == 0 or x == 0 or y == height - 1 or x == width - 1:
        return True
    return (
        labels[y - 1, x] != label
        or labels[y + 1, x] != label
        or labels[y, x - 1] != label
        or labels[y, x + 1] != label
    )


__all__ = [
    "NumbaNumpyObjectIntensityBackendStrategy",
    "ObjectIntensityArrays",
    "ObjectIntensityBackendStrategy",
    "object_intensity_backend",
]
