"""Object-intensity quantile and scan kernels for CellProfiler-compatible backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields as dataclass_fields, replace
from operator import attrgetter
from typing import Generic, TypeAlias, TypeVar

import numpy as np
from numba import njit


ObjectIntensity3DScanResult: TypeAlias = tuple[np.ndarray, ...]
ObjectIntensity3DQuantileResult: TypeAlias = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]
ObjectIntensityFeatureValueT = TypeVar("ObjectIntensityFeatureValueT")


@dataclass(frozen=True, slots=True)
class ObjectIntensityFeatureValues(Generic[ObjectIntensityFeatureValueT]):
    """Nominal CellProfiler object-intensity feature value family."""

    integrated_intensity: ObjectIntensityFeatureValueT
    mean_intensity: ObjectIntensityFeatureValueT
    std_intensity: ObjectIntensityFeatureValueT
    min_intensity: ObjectIntensityFeatureValueT
    max_intensity: ObjectIntensityFeatureValueT
    integrated_intensity_edge: ObjectIntensityFeatureValueT
    mean_intensity_edge: ObjectIntensityFeatureValueT
    std_intensity_edge: ObjectIntensityFeatureValueT
    min_intensity_edge: ObjectIntensityFeatureValueT
    max_intensity_edge: ObjectIntensityFeatureValueT
    mass_displacement: ObjectIntensityFeatureValueT
    lower_quartile_intensity: ObjectIntensityFeatureValueT
    median_intensity: ObjectIntensityFeatureValueT
    mad_intensity: ObjectIntensityFeatureValueT
    upper_quartile_intensity: ObjectIntensityFeatureValueT
    center_mass_intensity_x: ObjectIntensityFeatureValueT
    center_mass_intensity_y: ObjectIntensityFeatureValueT
    center_mass_intensity_z: ObjectIntensityFeatureValueT
    max_intensity_x: ObjectIntensityFeatureValueT
    max_intensity_y: ObjectIntensityFeatureValueT
    max_intensity_z: ObjectIntensityFeatureValueT

    @staticmethod
    def scalar_kwargs_from_columns(
        columns: Mapping[str, np.ndarray],
        index: int,
    ) -> dict[str, float]:
        """Return one row's scalar feature kwargs from columnar storage."""
        return {
            feature_field.name: float(columns[feature_field.name][index])
            for feature_field in dataclass_fields(ObjectIntensityFeatureValues)
        }


@dataclass(frozen=True, slots=True)
class ObjectIntensityArrays(ObjectIntensityFeatureValues[np.ndarray]):
    """Dense per-object intensity measurement arrays."""

    object_labels: np.ndarray

    @classmethod
    def empty(cls, object_labels: np.ndarray) -> "ObjectIntensityArrays":
        """Build an empty measurement array set for a declared object domain."""
        empty = np.zeros(0, dtype=float)
        return cls(
            object_labels=object_labels.astype(np.int32, copy=False),
            **{
                feature_field.name: empty
                for feature_field in dataclass_fields(ObjectIntensityFeatureValues)
            },
        )

    def scalar_kwargs(self, index: int) -> dict[str, float]:
        """Return scalar feature kwargs for one measured object index."""
        feature_fields = dataclass_fields(ObjectIntensityFeatureValues)
        feature_values = attrgetter(
            *(feature_field.name for feature_field in feature_fields)
        )(self)
        return {
            feature_field.name: float(values[index])
            for feature_field, values in zip(
                feature_fields, feature_values, strict=True
            )
        }

    def aligned_feature_columns(
        self,
        align_column: Callable[[np.ndarray], np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Return feature columns aligned through the supplied label-domain mapper."""
        feature_fields = dataclass_fields(ObjectIntensityFeatureValues)
        feature_values = attrgetter(
            *(feature_field.name for feature_field in feature_fields)
        )(self)
        return {
            feature_field.name: align_column(values)
            for feature_field, values in zip(
                feature_fields, feature_values, strict=True
            )
        }

    def with_max_intensity_positions(
        self,
        positions: tuple[np.ndarray, ...],
    ) -> "ObjectIntensityArrays":
        """Return these measurements with CP-native max-location coordinates."""
        if len(positions) == 2:
            max_intensity_y, max_intensity_x = positions
            max_intensity_z = np.zeros_like(max_intensity_x)
        elif len(positions) == 3:
            max_intensity_z, max_intensity_y, max_intensity_x = positions
        else:
            raise ValueError(
                "Object-intensity maximum positions require a 2-D or 3-D domain, "
                f"got {len(positions)} axes."
            )
        return replace(
            self,
            max_intensity_x=max_intensity_x,
            max_intensity_y=max_intensity_y,
            max_intensity_z=max_intensity_z,
        )

    @classmethod
    def from_3d_scan_result(
        cls,
        *,
        object_labels: np.ndarray,
        scan_result: ObjectIntensity3DScanResult,
        quantile_result: ObjectIntensity3DQuantileResult,
    ) -> "ObjectIntensityArrays":
        """Build dense intensity arrays from the 3-D scan kernel ABI."""
        lower, median, upper, mad = quantile_result
        return cls(
            object_labels=object_labels.astype(np.int32, copy=False),
            integrated_intensity=scan_result[1],
            mean_intensity=scan_result[2],
            std_intensity=scan_result[3],
            min_intensity=scan_result[4],
            max_intensity=scan_result[5],
            integrated_intensity_edge=scan_result[6],
            mean_intensity_edge=scan_result[7],
            std_intensity_edge=scan_result[8],
            min_intensity_edge=scan_result[9],
            max_intensity_edge=scan_result[10],
            mass_displacement=scan_result[11],
            lower_quartile_intensity=lower,
            median_intensity=median,
            mad_intensity=mad,
            upper_quartile_intensity=upper,
            center_mass_intensity_x=scan_result[12],
            center_mass_intensity_y=scan_result[13],
            center_mass_intensity_z=scan_result[14],
            max_intensity_x=np.zeros(object_labels.size, dtype=np.float64),
            max_intensity_y=np.zeros(object_labels.size, dtype=np.float64),
            max_intensity_z=np.zeros(object_labels.size, dtype=np.float64),
        )

    @classmethod
    def from_3d_scan_batch_result(
        cls,
        *,
        object_labels: np.ndarray,
        scan_result: ObjectIntensity3DScanResult,
        quantile_result: ObjectIntensity3DQuantileResult,
        image_index: int,
    ) -> "ObjectIntensityArrays":
        """Build one image's arrays from image-major 3-D batch kernel output."""
        return cls.from_3d_scan_result(
            object_labels=object_labels,
            scan_result=tuple(column[image_index] for column in scan_result),
            quantile_result=tuple(column[image_index] for column in quantile_result),
        )


@dataclass(frozen=True, slots=True)
class ObjectIntensityForegroundIndex:
    """Prepared foreground coordinates for sparse 3-D object-intensity kernels."""

    shape: tuple[int, int, int]
    z_indices: np.ndarray
    y_indices: np.ndarray
    x_indices: np.ndarray
    object_indexes: np.ndarray
    edge_flags: np.ndarray

    @classmethod
    def from_3d_labels(
        cls,
        labels: np.ndarray,
        label_to_index: np.ndarray,
    ) -> "ObjectIntensityForegroundIndex":
        label_array = np.ascontiguousarray(labels, dtype=np.int32)
        if label_array.ndim != 3:
            raise ValueError(
                "ObjectIntensityForegroundIndex requires 3-D labels, got "
                f"shape {label_array.shape!r}."
            )
        z_indices, y_indices, x_indices = (
            np.ascontiguousarray(axis, dtype=np.int64)
            for axis in np.nonzero(label_array > 0)
        )
        object_indexes = np.ascontiguousarray(
            label_to_index[label_array[z_indices, y_indices, x_indices]],
            dtype=np.int64,
        )
        valid = object_indexes >= 0
        if not bool(np.all(valid)):
            z_indices = np.ascontiguousarray(z_indices[valid], dtype=np.int64)
            y_indices = np.ascontiguousarray(y_indices[valid], dtype=np.int64)
            x_indices = np.ascontiguousarray(x_indices[valid], dtype=np.int64)
            object_indexes = np.ascontiguousarray(object_indexes[valid], dtype=np.int64)
        edge_flags = _object_intensity_foreground_edges_3d_numba(
            label_array,
            z_indices,
            y_indices,
            x_indices,
        )
        return cls(
            shape=(
                int(label_array.shape[0]),
                int(label_array.shape[1]),
                int(label_array.shape[2]),
            ),
            z_indices=z_indices,
            y_indices=y_indices,
            x_indices=x_indices,
            object_indexes=object_indexes,
            edge_flags=np.ascontiguousarray(edge_flags, dtype=np.bool_),
        )

    @property
    def voxel_count(self) -> int:
        """Return the number of foreground voxels represented by this index."""
        return int(self.object_indexes.size)

    def matches_shape(self, shape: tuple[int, int, int]) -> bool:
        """Return whether this index was prepared for the requested image shape."""
        return self.shape == shape

    def foreground_fraction(self) -> float:
        """Return foreground occupancy relative to the full 3-D volume."""
        volume = int(self.shape[0]) * int(self.shape[1]) * int(self.shape[2])
        if volume <= 0:
            return 0.0
        return float(self.voxel_count) / float(volume)

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
        group = values[start:start + count]
        (
            lower[index],
            median[index],
            upper[index],
        ) = _quartiles_from_dense_group_partition(group)
        mad[index] = _median_absolute_deviation_from_dense_group_partition(
            group,
            median[index],
            0.5,
        )

    return lower, median, upper, mad


@njit(cache=True)
def _quantile_from_dense_group_partition(
    values: np.ndarray,
    fraction: float,
) -> float:
    count = values.size
    if count <= 0:
        return 0.0
    ranks = np.empty(2, dtype=np.int64)
    low, qfraction = _write_quantile_rank_pair(count, fraction, ranks, 0)
    partitioned = np.partition(values, ranks)
    return _quantile_from_partitioned_rank_pair(partitioned, low, qfraction)


@njit(cache=True)
def _quartiles_from_dense_group_partition(
    values: np.ndarray,
) -> tuple[float, float, float]:
    count = values.size
    if count <= 0:
        return 0.0, 0.0, 0.0
    ranks = np.empty(6, dtype=np.int64)
    lower_low, lower_fraction = _write_quantile_rank_pair(count, 0.25, ranks, 0)
    median_low, median_fraction = _write_quantile_rank_pair(count, 0.5, ranks, 2)
    upper_low, upper_fraction = _write_quantile_rank_pair(count, 0.75, ranks, 4)
    partitioned = np.partition(values, ranks)
    return (
        _quantile_from_partitioned_rank_pair(
            partitioned,
            lower_low,
            lower_fraction,
        ),
        _quantile_from_partitioned_rank_pair(
            partitioned,
            median_low,
            median_fraction,
        ),
        _quantile_from_partitioned_rank_pair(
            partitioned,
            upper_low,
            upper_fraction,
        ),
    )


@njit(cache=True)
def _median_absolute_deviation_from_dense_group_partition(
    values: np.ndarray,
    median: float,
    fraction: float,
) -> float:
    deviations = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        deviations[index] = abs(float(values[index]) - median)
    return _quantile_from_dense_group_partition(deviations, fraction)


@njit(cache=True)
def _write_quantile_rank_pair(
    count: int,
    fraction: float,
    ranks: np.ndarray,
    offset: int,
) -> tuple[int, float]:
    qindex = count * fraction
    low = int(qindex)
    qfraction = qindex - low
    last = count - 1
    if low >= last:
        ranks[offset] = last
        ranks[offset + 1] = last
        return low, 0.0
    ranks[offset] = low
    ranks[offset + 1] = low + 1
    return low, qfraction


@njit(cache=True)
def _quantile_from_partitioned_rank_pair(
    partitioned: np.ndarray,
    low: int,
    qfraction: float,
) -> float:
    last = partitioned.size - 1
    if low >= last:
        return float(partitioned[last])
    low_value = float(partitioned[low])
    high_value = float(partitioned[low + 1])
    return low_value * (1.0 - qfraction) + high_value * qfraction


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
    min_values = np.full(object_count, np.inf, dtype=np.float64)
    max_values = np.full(object_count, -np.inf, dtype=np.float64)
    sum_x = np.zeros(object_count, dtype=np.float64)
    sum_y = np.zeros(object_count, dtype=np.float64)
    weighted_x = np.zeros(object_count, dtype=np.float64)
    weighted_y = np.zeros(object_count, dtype=np.float64)

    edge_counts = np.zeros(object_count, dtype=np.float64)
    edge_sums = np.zeros(object_count, dtype=np.float64)
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
            sum_x[index] += x
            sum_y[index] += y
            weighted_x[index] += x * value
            weighted_y[index] += y * value
            if value < min_values[index]:
                min_values[index] = value
            if value > max_values[index]:
                max_values[index] = value

            if _is_inner_boundary_pixel(labels, y, x, label):
                edge_counts[index] += 1.0
                edge_sums[index] += value
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
        else:
            edge_min_values[index] = 0.0
            edge_max_values[index] = 0.0

    stds, edge_stds = _object_intensity_std_2d_numba(
        image,
        labels,
        label_to_index,
        means,
        edge_means,
    )

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
    )


@njit(cache=True)
def _is_inner_boundary_pixel(
    labels: np.ndarray,
    y: int,
    x: int,
    label: int,
) -> bool:
    height, width = labels.shape
    if y > 0 and labels[y - 1, x] != label:
        return True
    if y + 1 < height and labels[y + 1, x] != label:
        return True
    if x > 0 and labels[y, x - 1] != label:
        return True
    if x + 1 < width and labels[y, x + 1] != label:
        return True
    return False


@njit(cache=True)
def _object_intensity_std_2d_numba(
    image: np.ndarray,
    labels: np.ndarray,
    label_to_index: np.ndarray,
    means: np.ndarray,
    edge_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_count = means.size
    variance_sums = np.zeros(object_count, dtype=np.float64)
    counts = np.zeros(object_count, dtype=np.float64)
    edge_variance_sums = np.zeros(object_count, dtype=np.float64)
    edge_counts = np.zeros(object_count, dtype=np.float64)
    height, width = image.shape
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
            diff = value - means[index]
            variance_sums[index] += diff * diff
            counts[index] += 1.0
            if _is_inner_boundary_pixel(labels, y, x, label):
                edge_diff = value - edge_means[index]
                edge_variance_sums[index] += edge_diff * edge_diff
                edge_counts[index] += 1.0

    stds = np.zeros(object_count, dtype=np.float64)
    edge_stds = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        if counts[index] > 0.0:
            stds[index] = np.sqrt(variance_sums[index] / counts[index])
        if edge_counts[index] > 0.0:
            edge_stds[index] = np.sqrt(
                edge_variance_sums[index] / edge_counts[index]
            )
    return stds, edge_stds


@njit(cache=True)
def _object_intensity_quantiles_3d_batch_numba(
    images: np.ndarray,
    labels: np.ndarray,
    label_to_index: np.ndarray,
    counts: np.ndarray,
    mad_fraction: float,
) -> ObjectIntensity3DQuantileResult:
    image_count, z_size, y_size, x_size = images.shape
    object_count = counts.shape[1]
    lower = np.zeros((image_count, object_count), dtype=np.float64)
    median = np.zeros((image_count, object_count), dtype=np.float64)
    upper = np.zeros((image_count, object_count), dtype=np.float64)
    mad = np.zeros((image_count, object_count), dtype=np.float64)

    total_count = 0
    for image_index in range(image_count):
        for object_index in range(object_count):
            total_count += int(counts[image_index, object_index])
    if total_count <= 0:
        return lower, median, upper, mad

    offsets = np.empty((image_count, object_count + 1), dtype=np.int64)
    cursor = 0
    for image_index in range(image_count):
        offsets[image_index, 0] = cursor
        for object_index in range(object_count):
            cursor += int(counts[image_index, object_index])
            offsets[image_index, object_index + 1] = cursor

    write_offsets = offsets[:, :-1].copy()
    values = np.empty(total_count, dtype=np.float64)
    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                label = int(labels[z_index, y_index, x_index])
                if label <= 0 or label >= label_to_index.size:
                    continue
                object_index = int(label_to_index[label])
                if object_index < 0:
                    continue
                for image_index in range(image_count):
                    value = float(images[image_index, z_index, y_index, x_index])
                    if not np.isfinite(value):
                        continue
                    offset = write_offsets[image_index, object_index]
                    values[offset] = value
                    write_offsets[image_index, object_index] = offset + 1

    for image_index in range(image_count):
        for object_index in range(object_count):
            start = int(offsets[image_index, object_index])
            count = int(counts[image_index, object_index])
            if count <= 0:
                continue
            group = values[start:start + count]
            (
                lower[image_index, object_index],
                median[image_index, object_index],
                upper[image_index, object_index],
            ) = _quartiles_from_dense_group_partition(group)
            mad[image_index, object_index] = (
                _median_absolute_deviation_from_dense_group_partition(
                    group,
                    median[image_index, object_index],
                    mad_fraction,
                )
            )

    return lower, median, upper, mad


@njit(cache=True)
def _object_intensity_scan_3d_batch_numba(
    images: np.ndarray,
    labels: np.ndarray,
    object_labels: np.ndarray,
    label_to_index: np.ndarray,
) -> ObjectIntensity3DScanResult:
    image_count, z_size, y_size, x_size = images.shape
    object_count = object_labels.size
    counts = np.zeros((image_count, object_count), dtype=np.float64)
    sums = np.zeros((image_count, object_count), dtype=np.float64)
    min_values = np.full((image_count, object_count), np.inf, dtype=np.float64)
    max_values = np.full((image_count, object_count), -np.inf, dtype=np.float64)
    sum_x = np.zeros((image_count, object_count), dtype=np.float64)
    sum_y = np.zeros((image_count, object_count), dtype=np.float64)
    sum_z = np.zeros((image_count, object_count), dtype=np.float64)
    weighted_x = np.zeros((image_count, object_count), dtype=np.float64)
    weighted_y = np.zeros((image_count, object_count), dtype=np.float64)
    weighted_z = np.zeros((image_count, object_count), dtype=np.float64)

    edge_counts = np.zeros((image_count, object_count), dtype=np.float64)
    edge_sums = np.zeros((image_count, object_count), dtype=np.float64)
    edge_min_values = np.full(
        (image_count, object_count),
        np.inf,
        dtype=np.float64,
    )
    edge_max_values = np.full(
        (image_count, object_count),
        -np.inf,
        dtype=np.float64,
    )

    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                label = labels[z_index, y_index, x_index]
                if label <= 0 or label >= label_to_index.size:
                    continue
                object_index = label_to_index[label]
                if object_index < 0:
                    continue
                is_edge = _is_inner_boundary_voxel(
                    labels,
                    z_index,
                    y_index,
                    x_index,
                    label,
                )
                for image_index in range(image_count):
                    value = images[image_index, z_index, y_index, x_index]
                    if not np.isfinite(value):
                        continue

                    counts[image_index, object_index] += 1.0
                    sums[image_index, object_index] += value
                    sum_x[image_index, object_index] += x_index
                    sum_y[image_index, object_index] += y_index
                    sum_z[image_index, object_index] += z_index
                    weighted_x[image_index, object_index] += x_index * value
                    weighted_y[image_index, object_index] += y_index * value
                    weighted_z[image_index, object_index] += z_index * value
                    if value < min_values[image_index, object_index]:
                        min_values[image_index, object_index] = value
                    if value > max_values[image_index, object_index]:
                        max_values[image_index, object_index] = value

                    if is_edge:
                        edge_counts[image_index, object_index] += 1.0
                        edge_sums[image_index, object_index] += value
                        if value < edge_min_values[image_index, object_index]:
                            edge_min_values[image_index, object_index] = value
                        if value > edge_max_values[image_index, object_index]:
                            edge_max_values[image_index, object_index] = value

    means = np.zeros((image_count, object_count), dtype=np.float64)
    stds = np.zeros((image_count, object_count), dtype=np.float64)
    edge_means = np.zeros((image_count, object_count), dtype=np.float64)
    edge_stds = np.zeros((image_count, object_count), dtype=np.float64)
    mass_displacement = np.zeros((image_count, object_count), dtype=np.float64)
    center_mass_x = np.zeros((image_count, object_count), dtype=np.float64)
    center_mass_y = np.zeros((image_count, object_count), dtype=np.float64)
    center_mass_z = np.zeros((image_count, object_count), dtype=np.float64)
    for image_index in range(image_count):
        for object_index in range(object_count):
            if counts[image_index, object_index] > 0.0:
                means[image_index, object_index] = (
                    sums[image_index, object_index]
                    / counts[image_index, object_index]
                )
                center_x = sum_x[image_index, object_index] / counts[
                    image_index,
                    object_index,
                ]
                center_y = sum_y[image_index, object_index] / counts[
                    image_index,
                    object_index,
                ]
                center_z = sum_z[image_index, object_index] / counts[
                    image_index,
                    object_index,
                ]
                if sums[image_index, object_index] != 0.0:
                    center_mass_x[image_index, object_index] = (
                        weighted_x[image_index, object_index]
                        / sums[image_index, object_index]
                    )
                    center_mass_y[image_index, object_index] = (
                        weighted_y[image_index, object_index]
                        / sums[image_index, object_index]
                    )
                    center_mass_z[image_index, object_index] = (
                        weighted_z[image_index, object_index]
                        / sums[image_index, object_index]
                    )
                diff_x = center_x - center_mass_x[image_index, object_index]
                diff_y = center_y - center_mass_y[image_index, object_index]
                diff_z = center_z - center_mass_z[image_index, object_index]
                mass_displacement[image_index, object_index] = np.sqrt(
                    diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                )
            else:
                min_values[image_index, object_index] = 0.0
                max_values[image_index, object_index] = 0.0

            if edge_counts[image_index, object_index] > 0.0:
                edge_means[image_index, object_index] = (
                    edge_sums[image_index, object_index]
                    / edge_counts[image_index, object_index]
                )
            else:
                edge_min_values[image_index, object_index] = 0.0
                edge_max_values[image_index, object_index] = 0.0

    stds, edge_stds = _object_intensity_std_3d_batch_numba(
        images,
        labels,
        label_to_index,
        means,
        edge_means,
    )

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
        center_mass_z,
    )


@njit(cache=True)
def _is_inner_boundary_voxel(
    labels: np.ndarray,
    z_index: int,
    y_index: int,
    x_index: int,
    label: int,
) -> bool:
    z_size, y_size, x_size = labels.shape
    if z_index > 0 and labels[z_index - 1, y_index, x_index] != label:
        return True
    if z_index + 1 < z_size and labels[z_index + 1, y_index, x_index] != label:
        return True
    if y_index > 0 and labels[z_index, y_index - 1, x_index] != label:
        return True
    if y_index + 1 < y_size and labels[z_index, y_index + 1, x_index] != label:
        return True
    if x_index > 0 and labels[z_index, y_index, x_index - 1] != label:
        return True
    if x_index + 1 < x_size and labels[z_index, y_index, x_index + 1] != label:
        return True
    return False


@njit(cache=True)
def _object_intensity_std_3d_numba(
    image: np.ndarray,
    labels: np.ndarray,
    label_to_index: np.ndarray,
    means: np.ndarray,
    edge_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_count = means.size
    variance_sums = np.zeros(object_count, dtype=np.float64)
    counts = np.zeros(object_count, dtype=np.float64)
    edge_variance_sums = np.zeros(object_count, dtype=np.float64)
    edge_counts = np.zeros(object_count, dtype=np.float64)
    z_size, y_size, x_size = image.shape
    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                label = labels[z_index, y_index, x_index]
                if label <= 0 or label >= label_to_index.size:
                    continue
                object_index = label_to_index[label]
                if object_index < 0:
                    continue
                value = image[z_index, y_index, x_index]
                if not np.isfinite(value):
                    continue
                diff = value - means[object_index]
                variance_sums[object_index] += diff * diff
                counts[object_index] += 1.0
                if _is_inner_boundary_voxel(
                    labels,
                    z_index,
                    y_index,
                    x_index,
                    label,
                ):
                    edge_diff = value - edge_means[object_index]
                    edge_variance_sums[object_index] += edge_diff * edge_diff
                    edge_counts[object_index] += 1.0

    stds = np.zeros(object_count, dtype=np.float64)
    edge_stds = np.zeros(object_count, dtype=np.float64)
    for object_index in range(object_count):
        if counts[object_index] > 0.0:
            stds[object_index] = np.sqrt(
                variance_sums[object_index] / counts[object_index]
            )
        if edge_counts[object_index] > 0.0:
            edge_stds[object_index] = np.sqrt(
                edge_variance_sums[object_index] / edge_counts[object_index]
            )
    return stds, edge_stds


@njit(cache=True)
def _object_intensity_std_3d_batch_numba(
    images: np.ndarray,
    labels: np.ndarray,
    label_to_index: np.ndarray,
    means: np.ndarray,
    edge_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image_count, z_size, y_size, x_size = images.shape
    object_count = means.shape[1]
    variance_sums = np.zeros((image_count, object_count), dtype=np.float64)
    counts = np.zeros((image_count, object_count), dtype=np.float64)
    edge_variance_sums = np.zeros((image_count, object_count), dtype=np.float64)
    edge_counts = np.zeros((image_count, object_count), dtype=np.float64)
    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                label = labels[z_index, y_index, x_index]
                if label <= 0 or label >= label_to_index.size:
                    continue
                object_index = label_to_index[label]
                if object_index < 0:
                    continue
                is_edge = _is_inner_boundary_voxel(
                    labels,
                    z_index,
                    y_index,
                    x_index,
                    label,
                )
                for image_index in range(image_count):
                    value = images[image_index, z_index, y_index, x_index]
                    if not np.isfinite(value):
                        continue
                    diff = value - means[image_index, object_index]
                    variance_sums[image_index, object_index] += diff * diff
                    counts[image_index, object_index] += 1.0
                    if is_edge:
                        edge_diff = value - edge_means[image_index, object_index]
                        edge_variance_sums[
                            image_index,
                            object_index,
                        ] += edge_diff * edge_diff
                        edge_counts[image_index, object_index] += 1.0

    stds = np.zeros((image_count, object_count), dtype=np.float64)
    edge_stds = np.zeros((image_count, object_count), dtype=np.float64)
    for image_index in range(image_count):
        for object_index in range(object_count):
            if counts[image_index, object_index] > 0.0:
                stds[image_index, object_index] = np.sqrt(
                    variance_sums[image_index, object_index]
                    / counts[image_index, object_index]
                )
            if edge_counts[image_index, object_index] > 0.0:
                edge_stds[image_index, object_index] = np.sqrt(
                    edge_variance_sums[image_index, object_index]
                    / edge_counts[image_index, object_index]
                )
    return stds, edge_stds


@njit(cache=True)
def _object_intensity_std_3d_sparse_batch_numba(
    images: np.ndarray,
    z_indices: np.ndarray,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    object_indexes: np.ndarray,
    edge_flags: np.ndarray,
    means: np.ndarray,
    edge_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image_count = images.shape[0]
    object_count = means.shape[1]
    variance_sums = np.zeros((image_count, object_count), dtype=np.float64)
    counts = np.zeros((image_count, object_count), dtype=np.float64)
    edge_variance_sums = np.zeros((image_count, object_count), dtype=np.float64)
    edge_counts = np.zeros((image_count, object_count), dtype=np.float64)
    for foreground_index in range(object_indexes.size):
        object_index = int(object_indexes[foreground_index])
        z_index = int(z_indices[foreground_index])
        y_index = int(y_indices[foreground_index])
        x_index = int(x_indices[foreground_index])
        is_edge = edge_flags[foreground_index]
        for image_index in range(image_count):
            value = images[image_index, z_index, y_index, x_index]
            if not np.isfinite(value):
                continue
            diff = value - means[image_index, object_index]
            variance_sums[image_index, object_index] += diff * diff
            counts[image_index, object_index] += 1.0
            if is_edge:
                edge_diff = value - edge_means[image_index, object_index]
                edge_variance_sums[
                    image_index,
                    object_index,
                ] += edge_diff * edge_diff
                edge_counts[image_index, object_index] += 1.0

    stds = np.zeros((image_count, object_count), dtype=np.float64)
    edge_stds = np.zeros((image_count, object_count), dtype=np.float64)
    for image_index in range(image_count):
        for object_index in range(object_count):
            if counts[image_index, object_index] > 0.0:
                stds[image_index, object_index] = np.sqrt(
                    variance_sums[image_index, object_index]
                    / counts[image_index, object_index]
                )
            if edge_counts[image_index, object_index] > 0.0:
                edge_stds[image_index, object_index] = np.sqrt(
                    edge_variance_sums[image_index, object_index]
                    / edge_counts[image_index, object_index]
                )
    return stds, edge_stds


@njit(cache=True)
def _object_intensity_foreground_edges_3d_numba(
    labels: np.ndarray,
    z_indices: np.ndarray,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
) -> np.ndarray:
    edge_flags = np.zeros(z_indices.size, dtype=np.bool_)
    for index in range(z_indices.size):
        z_index = int(z_indices[index])
        y_index = int(y_indices[index])
        x_index = int(x_indices[index])
        label = int(labels[z_index, y_index, x_index])
        edge_flags[index] = _is_inner_boundary_voxel(
            labels,
            z_index,
            y_index,
            x_index,
            label,
        )
    return edge_flags


@njit(cache=True)
def _object_intensity_scan_3d_sparse_batch_numba(
    images: np.ndarray,
    z_indices: np.ndarray,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    object_indexes: np.ndarray,
    edge_flags: np.ndarray,
    object_count: int,
) -> ObjectIntensity3DScanResult:
    image_count = images.shape[0]
    counts = np.zeros((image_count, object_count), dtype=np.float64)
    sums = np.zeros((image_count, object_count), dtype=np.float64)
    min_values = np.full((image_count, object_count), np.inf, dtype=np.float64)
    max_values = np.full((image_count, object_count), -np.inf, dtype=np.float64)
    sum_x = np.zeros((image_count, object_count), dtype=np.float64)
    sum_y = np.zeros((image_count, object_count), dtype=np.float64)
    sum_z = np.zeros((image_count, object_count), dtype=np.float64)
    weighted_x = np.zeros((image_count, object_count), dtype=np.float64)
    weighted_y = np.zeros((image_count, object_count), dtype=np.float64)
    weighted_z = np.zeros((image_count, object_count), dtype=np.float64)

    edge_counts = np.zeros((image_count, object_count), dtype=np.float64)
    edge_sums = np.zeros((image_count, object_count), dtype=np.float64)
    edge_min_values = np.full(
        (image_count, object_count),
        np.inf,
        dtype=np.float64,
    )
    edge_max_values = np.full(
        (image_count, object_count),
        -np.inf,
        dtype=np.float64,
    )

    for foreground_index in range(object_indexes.size):
        object_index = int(object_indexes[foreground_index])
        z_index = int(z_indices[foreground_index])
        y_index = int(y_indices[foreground_index])
        x_index = int(x_indices[foreground_index])
        is_edge = edge_flags[foreground_index]
        for image_index in range(image_count):
            value = images[image_index, z_index, y_index, x_index]
            if not np.isfinite(value):
                continue

            counts[image_index, object_index] += 1.0
            sums[image_index, object_index] += value
            sum_x[image_index, object_index] += x_index
            sum_y[image_index, object_index] += y_index
            sum_z[image_index, object_index] += z_index
            weighted_x[image_index, object_index] += x_index * value
            weighted_y[image_index, object_index] += y_index * value
            weighted_z[image_index, object_index] += z_index * value
            if value < min_values[image_index, object_index]:
                min_values[image_index, object_index] = value
            if value > max_values[image_index, object_index]:
                max_values[image_index, object_index] = value

            if is_edge:
                edge_counts[image_index, object_index] += 1.0
                edge_sums[image_index, object_index] += value
                if value < edge_min_values[image_index, object_index]:
                    edge_min_values[image_index, object_index] = value
                if value > edge_max_values[image_index, object_index]:
                    edge_max_values[image_index, object_index] = value

    means = np.zeros((image_count, object_count), dtype=np.float64)
    stds = np.zeros((image_count, object_count), dtype=np.float64)
    edge_means = np.zeros((image_count, object_count), dtype=np.float64)
    edge_stds = np.zeros((image_count, object_count), dtype=np.float64)
    mass_displacement = np.zeros((image_count, object_count), dtype=np.float64)
    center_mass_x = np.zeros((image_count, object_count), dtype=np.float64)
    center_mass_y = np.zeros((image_count, object_count), dtype=np.float64)
    center_mass_z = np.zeros((image_count, object_count), dtype=np.float64)
    for image_index in range(image_count):
        for object_index in range(object_count):
            if counts[image_index, object_index] > 0.0:
                means[image_index, object_index] = (
                    sums[image_index, object_index]
                    / counts[image_index, object_index]
                )
                center_x = sum_x[image_index, object_index] / counts[
                    image_index,
                    object_index,
                ]
                center_y = sum_y[image_index, object_index] / counts[
                    image_index,
                    object_index,
                ]
                center_z = sum_z[image_index, object_index] / counts[
                    image_index,
                    object_index,
                ]
                if sums[image_index, object_index] != 0.0:
                    center_mass_x[image_index, object_index] = (
                        weighted_x[image_index, object_index]
                        / sums[image_index, object_index]
                    )
                    center_mass_y[image_index, object_index] = (
                        weighted_y[image_index, object_index]
                        / sums[image_index, object_index]
                    )
                    center_mass_z[image_index, object_index] = (
                        weighted_z[image_index, object_index]
                        / sums[image_index, object_index]
                    )
                diff_x = center_x - center_mass_x[image_index, object_index]
                diff_y = center_y - center_mass_y[image_index, object_index]
                diff_z = center_z - center_mass_z[image_index, object_index]
                mass_displacement[image_index, object_index] = np.sqrt(
                    diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                )
            else:
                min_values[image_index, object_index] = 0.0
                max_values[image_index, object_index] = 0.0

            if edge_counts[image_index, object_index] > 0.0:
                edge_means[image_index, object_index] = (
                    edge_sums[image_index, object_index]
                    / edge_counts[image_index, object_index]
                )
            else:
                edge_min_values[image_index, object_index] = 0.0
                edge_max_values[image_index, object_index] = 0.0

    stds, edge_stds = _object_intensity_std_3d_sparse_batch_numba(
        images,
        z_indices,
        y_indices,
        x_indices,
        object_indexes,
        edge_flags,
        means,
        edge_means,
    )

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
        center_mass_z,
    )


@njit(cache=True)
def _object_intensity_quantiles_3d_sparse_batch_numba(
    images: np.ndarray,
    z_indices: np.ndarray,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    object_indexes: np.ndarray,
    counts: np.ndarray,
    mad_fraction: float,
) -> ObjectIntensity3DQuantileResult:
    image_count = images.shape[0]
    object_count = counts.shape[1]
    lower = np.zeros((image_count, object_count), dtype=np.float64)
    median = np.zeros((image_count, object_count), dtype=np.float64)
    upper = np.zeros((image_count, object_count), dtype=np.float64)
    mad = np.zeros((image_count, object_count), dtype=np.float64)

    total_count = 0
    for image_index in range(image_count):
        for object_index in range(object_count):
            total_count += int(counts[image_index, object_index])
    if total_count <= 0:
        return lower, median, upper, mad

    offsets = np.empty((image_count, object_count + 1), dtype=np.int64)
    cursor = 0
    for image_index in range(image_count):
        offsets[image_index, 0] = cursor
        for object_index in range(object_count):
            cursor += int(counts[image_index, object_index])
            offsets[image_index, object_index + 1] = cursor

    write_offsets = offsets[:, :-1].copy()
    values = np.empty(total_count, dtype=np.float64)
    for foreground_index in range(object_indexes.size):
        object_index = int(object_indexes[foreground_index])
        z_index = int(z_indices[foreground_index])
        y_index = int(y_indices[foreground_index])
        x_index = int(x_indices[foreground_index])
        for image_index in range(image_count):
            value = float(images[image_index, z_index, y_index, x_index])
            if not np.isfinite(value):
                continue
            offset = write_offsets[image_index, object_index]
            values[offset] = value
            write_offsets[image_index, object_index] = offset + 1

    for image_index in range(image_count):
        for object_index in range(object_count):
            start = int(offsets[image_index, object_index])
            count = int(counts[image_index, object_index])
            if count <= 0:
                continue
            group = values[start:start + count]
            (
                lower[image_index, object_index],
                median[image_index, object_index],
                upper[image_index, object_index],
            ) = _quartiles_from_dense_group_partition(group)
            mad[image_index, object_index] = (
                _median_absolute_deviation_from_dense_group_partition(
                    group,
                    median[image_index, object_index],
                    mad_fraction,
                )
            )

    return lower, median, upper, mad
