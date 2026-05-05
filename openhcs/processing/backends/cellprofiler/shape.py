"""Shape-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.secondary import _edt_1d_numba


class ShapeMeasurementBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Shape-measurement operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def form_factor_values(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        """Return CP-compatible AreaShape_FormFactor values."""

    @abstractmethod
    def radius_features(
        self,
        object_images: np.ndarray,
        object_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return maximum, mean, and median object radii."""

    @abstractmethod
    def radius_features_from_labels(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return maximum, mean, and median object radii from dense labels."""

    @abstractmethod
    def feret_diameters(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return minimum and maximum Feret diameters."""

    @abstractmethod
    def minimum_enclosing_circle(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return object center coordinates and radii."""

    @abstractmethod
    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        """Return per-pixel distance-to-edge for labeled objects."""

    @abstractmethod
    def maximum_position_of_labels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return maximum-value positions for each label."""

    @abstractmethod
    def color_labels(self, labels: np.ndarray) -> np.ndarray:
        """Return non-touching label color classes."""

    @abstractmethod
    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate labels through a mask and return labels plus distances."""

    @abstractmethod
    def zernike_indexes(self, max_order: int) -> np.ndarray:
        """Return Zernike index pairs up to ``max_order``."""

    @abstractmethod
    def construct_zernike_polynomials(
        self,
        x: np.ndarray,
        y: np.ndarray,
        zernike_indexes: np.ndarray,
    ) -> np.ndarray:
        """Return Zernike polynomial values at normalized coordinates."""


class CentrosomeNumpyShapeMeasurementBackendStrategy(ShapeMeasurementBackendStrategy):
    """Centrosome-backed NumPy shape measurements."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def form_factor_values(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        import centrosome.cpmorphology

        labels_array = np.asarray(labels, dtype=np.int32)
        label_ids_array = np.asarray(label_ids, dtype=np.int32)
        if label_ids_array.size == 0:
            return np.array([], dtype=float)
        areas = np.bincount(
            labels_array.ravel(),
            minlength=int(label_ids_array[-1]) + 1,
        )[label_ids_array]
        perimeters = np.asarray(
            centrosome.cpmorphology.calculate_perimeters(
                labels_array,
                label_ids_array,
            ),
            dtype=float,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return 4.0 * np.pi * areas.astype(float) / perimeters**2

    def radius_features(
        self,
        object_images: np.ndarray,
        object_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import centrosome.cpmorphology

        max_radius = np.zeros(object_count)
        mean_radius = np.zeros(object_count)
        median_radius = np.zeros(object_count)
        for index, object_image in enumerate(object_images):
            mini_image = np.pad(object_image, 1)
            distances = scipy.ndimage.distance_transform_edt(mini_image)
            max_radius[index] = _first_scalar(
                centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.maximum(distances, mini_image)
                )
            )
            mean_radius[index] = _first_scalar(
                centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.mean(distances, mini_image)
                )
            )
            median_radius[index] = _first_scalar(
                centrosome.cpmorphology.median_of_labels(
                    distances,
                    mini_image.astype("int"),
                    [1],
                )
            )
        return max_radius, mean_radius, median_radius

    def radius_features_from_labels(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return radius features via one full-image distance-to-edge pass."""
        distances = self.distance_to_edge(labels)
        return _radius_features_from_distance_image_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(distances, dtype=np.float64),
            np.asarray(label_ids, dtype=np.int32),
        )

    def feret_diameters(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import centrosome.cpmorphology

        chulls, chull_counts = centrosome.cpmorphology.convex_hull(
            labels,
            label_ids,
        )
        return centrosome.cpmorphology.feret_diameter(
            chulls,
            chull_counts,
            label_ids,
        )

    def minimum_enclosing_circle(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import centrosome.cpmorphology

        return centrosome.cpmorphology.minimum_enclosing_circle(
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
        )

    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        import centrosome.cpmorphology

        return centrosome.cpmorphology.distance_to_edge(
            np.asarray(labels, dtype=np.int32)
        )

    def maximum_position_of_labels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import centrosome.cpmorphology

        centers_i, centers_j = centrosome.cpmorphology.maximum_position_of_labels(
            np.asarray(image),
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
        )
        return centers_i, centers_j

    def color_labels(self, labels: np.ndarray) -> np.ndarray:
        import centrosome.cpmorphology

        return centrosome.cpmorphology.color_labels(
            np.asarray(labels, dtype=np.int32)
        )

    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        import centrosome.propagate

        return centrosome.propagate.propagate(
            np.asarray(image, dtype=np.float64),
            np.asarray(labels, dtype=np.int32),
            np.asarray(mask, dtype=bool),
            regularization_factor,
        )

    def zernike_indexes(self, max_order: int) -> np.ndarray:
        import centrosome.zernike

        return centrosome.zernike.get_zernike_indexes(int(max_order))

    def construct_zernike_polynomials(
        self,
        x: np.ndarray,
        y: np.ndarray,
        zernike_indexes: np.ndarray,
    ) -> np.ndarray:
        import centrosome.zernike

        return centrosome.zernike.construct_zernike_polynomials(
            x,
            y,
            zernike_indexes,
        )


class LegacyFastNumpyShapeMeasurementBackendStrategy(
    CentrosomeNumpyShapeMeasurementBackendStrategy
):
    """Mixed legacy-fast shape backend with explicit centrosome exact leaves."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.LEGACY_FAST,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.LEGACY_FAST
    is_default_backend = True

    def radius_features(
        self,
        object_images: np.ndarray,
        object_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_radius = np.zeros(object_count, dtype=np.float64)
        mean_radius = np.zeros(object_count, dtype=np.float64)
        median_radius = np.zeros(object_count, dtype=np.float64)
        for index, object_image in enumerate(object_images):
            max_value, mean_value, median_value = _object_radius_features_numba(
                np.asarray(object_image, dtype=np.bool_),
            )
            max_radius[index] = max_value
            mean_radius[index] = mean_value
            median_radius[index] = median_value
        return max_radius, mean_radius, median_radius

    def radius_features_from_labels(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _radius_features_from_labels_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
        )

    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            return _distance_to_edge_planewise(self, label_array)
        return _distance_to_label_edge_numba(np.ascontiguousarray(label_array))

    def maximum_position_of_labels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _maximum_position_of_labels_numba(
            np.ascontiguousarray(np.asarray(image, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(labels, dtype=np.int32)),
            np.ascontiguousarray(np.asarray(label_ids, dtype=np.int32)),
        )


class NumbaNumpyShapeMeasurementBackendStrategy(ShapeMeasurementBackendStrategy):
    """Pure Numba shape backend. Unsupported leaves fail explicitly."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def form_factor_values(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Pure Numba form-factor values are not implemented yet. "
            "Select LEGACY_FAST or CENTROSOME explicitly for this leaf."
        )

    def radius_features(
        self,
        object_images: np.ndarray,
        object_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_radius = np.zeros(object_count, dtype=np.float64)
        mean_radius = np.zeros(object_count, dtype=np.float64)
        median_radius = np.zeros(object_count, dtype=np.float64)
        for index, object_image in enumerate(object_images):
            max_value, mean_value, median_value = _object_radius_features_numba(
                np.asarray(object_image, dtype=np.bool_),
            )
            max_radius[index] = max_value
            mean_radius[index] = mean_value
            median_radius[index] = median_value
        return max_radius, mean_radius, median_radius

    def radius_features_from_labels(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _radius_features_from_labels_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
        )

    def feret_diameters(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Pure Numba Feret diameters are not implemented yet. "
            "Select LEGACY_FAST or CENTROSOME explicitly for this leaf."
        )

    def minimum_enclosing_circle(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Pure Numba minimum enclosing circle is not implemented yet. "
            "Select LEGACY_FAST or CENTROSOME explicitly for this leaf."
        )

    def distance_to_edge(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            return _distance_to_edge_planewise(self, label_array)
        return _distance_to_label_edge_numba(np.ascontiguousarray(label_array))

    def maximum_position_of_labels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _maximum_position_of_labels_numba(
            np.ascontiguousarray(np.asarray(image, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(labels, dtype=np.int32)),
            np.ascontiguousarray(np.asarray(label_ids, dtype=np.int32)),
        )

    def color_labels(self, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Pure Numba label coloring is not implemented yet. "
            "Select LEGACY_FAST or CENTROSOME explicitly for this leaf."
        )

    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Pure Numba propagation is not implemented yet. "
            "Select CENTROSOME explicitly for this leaf."
        )

    def zernike_indexes(self, max_order: int) -> np.ndarray:
        return _zernike_indexes_numpy(int(max_order))

    def construct_zernike_polynomials(
        self,
        x: np.ndarray,
        y: np.ndarray,
        zernike_indexes: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Pure Numba Zernike polynomial construction is not implemented in "
            "the shape backend. Use the zernike backend family instead."
        )


def _distance_to_edge_planewise(
    backend: ShapeMeasurementBackendStrategy,
    labels: np.ndarray,
) -> np.ndarray:
    if labels.ndim < 2:
        raise ValueError("Distance-to-edge requires at least two dimensions.")
    distances = np.empty(labels.shape, dtype=np.float64)
    plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
    source_planes = labels.reshape((plane_count, *labels.shape[-2:]))
    target_planes = distances.reshape((plane_count, *labels.shape[-2:]))
    for plane_index in range(plane_count):
        target_planes[plane_index] = backend.distance_to_edge(source_planes[plane_index])
    return distances


def shape_measurement_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> ShapeMeasurementBackendStrategy:
    """Return the selected shape-measurement backend."""
    return ShapeMeasurementBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


def form_factor_values(
    labels: np.ndarray,
    label_ids: np.ndarray,
    *,
    backend_provider: BackendProviderInput | None = None,
) -> np.ndarray:
    """Return CP-compatible AreaShape_FormFactor values through a backend."""
    return ShapeMeasurementBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    ).form_factor_values(labels, label_ids)


def _zernike_indexes_numpy(max_order: int) -> np.ndarray:
    indexes: list[tuple[int, int]] = []
    for n_value in range(max_order + 1):
        for m_value in range(n_value + 1):
            if (n_value - m_value) % 2 == 0:
                indexes.append((n_value, m_value))
    return np.asarray(indexes, dtype=np.int64)


def _first_scalar(value: object) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    return float(array.reshape(-1)[0])


@njit(cache=True)
def _object_radius_features_numba(mask: np.ndarray) -> tuple[float, float, float]:
    height = mask.shape[0] + 2
    width = mask.shape[1] + 2
    inf = 1.0e20

    row_distances = np.empty((height, width), dtype=np.float64)
    distances_sq = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        source = np.empty(width, dtype=np.float64)
        for x in range(width):
            source[x] = 0.0
            if 0 < y < height - 1 and 0 < x < width - 1:
                if mask[y - 1, x - 1]:
                    source[x] = inf
        row_output = np.empty(width, dtype=np.float64)
        row_arg = np.empty(width, dtype=np.int64)
        _edt_1d_numba(source, row_output, row_arg)
        for x in range(width):
            row_distances[y, x] = row_output[x]

    for x in range(width):
        source = np.empty(height, dtype=np.float64)
        for y in range(height):
            source[y] = row_distances[y, x]
        column_output = np.empty(height, dtype=np.float64)
        column_arg = np.empty(height, dtype=np.int64)
        _edt_1d_numba(source, column_output, column_arg)
        for y in range(height):
            distances_sq[y, x] = column_output[y]

    count = 0
    total = 0.0
    maximum = 0.0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if mask[y - 1, x - 1]:
                value = np.sqrt(distances_sq[y, x])
                total += value
                if value > maximum:
                    maximum = value
                count += 1

    if count == 0:
        return 0.0, 0.0, 0.0

    values = np.empty(count, dtype=np.float64)
    index = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if mask[y - 1, x - 1]:
                values[index] = np.sqrt(distances_sq[y, x])
                index += 1
    values.sort()
    middle = count // 2
    if count % 2 == 1:
        median = values[middle]
    else:
        median = 0.5 * (values[middle - 1] + values[middle])
    return maximum, total / count, median


@njit(cache=True)
def _radius_features_from_distance_image_numba(
    labels: np.ndarray,
    distances: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_count = label_ids.size
    max_label = 0
    for i in range(object_count):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id

    counts_by_label = np.zeros(max_label + 1, dtype=np.int64)
    sums_by_label = np.zeros(max_label + 1, dtype=np.float64)
    max_by_label = np.zeros(max_label + 1, dtype=np.float64)
    rows, cols = labels.shape
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                value = distances[row, col]
                counts_by_label[label] += 1
                sums_by_label[label] += value
                if value > max_by_label[label]:
                    max_by_label[label] = value

    offsets = np.zeros(max_label + 2, dtype=np.int64)
    for label in range(max_label + 1):
        offsets[label + 1] = offsets[label] + counts_by_label[label]
    cursor = offsets.copy()
    ordered = np.empty(offsets[max_label + 1], dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                index = cursor[label]
                ordered[index] = distances[row, col]
                cursor[label] = index + 1

    max_radius = np.zeros(object_count, dtype=np.float64)
    mean_radius = np.zeros(object_count, dtype=np.float64)
    median_radius = np.zeros(object_count, dtype=np.float64)
    for i in range(object_count):
        label = int(label_ids[i])
        if label <= 0 or label > max_label:
            continue
        count = counts_by_label[label]
        if count <= 0:
            continue
        start = offsets[label]
        values = ordered[start : start + count].copy()
        values.sort()
        max_radius[i] = max_by_label[label]
        mean_radius[i] = sums_by_label[label] / count
        middle = count // 2
        if count % 2 == 1:
            median_radius[i] = values[middle]
        else:
            median_radius[i] = 0.5 * (values[middle - 1] + values[middle])
    return max_radius, mean_radius, median_radius


@njit(cache=True)
def _radius_features_from_labels_numba(
    labels: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_count = label_ids.size
    max_label = 0
    for i in range(object_count):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id

    height, width = labels.shape
    min_y = np.full(max_label + 1, height, dtype=np.int64)
    min_x = np.full(max_label + 1, width, dtype=np.int64)
    max_y = np.zeros(max_label + 1, dtype=np.int64)
    max_x = np.zeros(max_label + 1, dtype=np.int64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0 or label > max_label:
                continue
            counts[label] += 1
            if y < min_y[label]:
                min_y[label] = y
            if x < min_x[label]:
                min_x[label] = x
            if y + 1 > max_y[label]:
                max_y[label] = y + 1
            if x + 1 > max_x[label]:
                max_x[label] = x + 1

    max_radius = np.zeros(object_count, dtype=np.float64)
    mean_radius = np.zeros(object_count, dtype=np.float64)
    median_radius = np.zeros(object_count, dtype=np.float64)
    inf = 1.0e20
    for object_index in range(object_count):
        label = int(label_ids[object_index])
        if label <= 0 or label > max_label or counts[label] <= 0:
            continue

        crop_height = max_y[label] - min_y[label] + 2
        crop_width = max_x[label] - min_x[label] + 2
        row_distances = np.empty((crop_height, crop_width), dtype=np.float64)
        distances_sq = np.empty((crop_height, crop_width), dtype=np.float64)

        for yy in range(crop_height):
            source = np.empty(crop_width, dtype=np.float64)
            source_y = min_y[label] + yy - 1
            for xx in range(crop_width):
                source_x = min_x[label] + xx - 1
                if (
                    source_y >= 0
                    and source_y < height
                    and source_x >= 0
                    and source_x < width
                    and labels[source_y, source_x] == label
                ):
                    source[xx] = inf
                else:
                    source[xx] = 0.0
            row_output = np.empty(crop_width, dtype=np.float64)
            row_arg = np.empty(crop_width, dtype=np.int64)
            _edt_1d_numba(source, row_output, row_arg)
            for xx in range(crop_width):
                row_distances[yy, xx] = row_output[xx]

        for xx in range(crop_width):
            source = np.empty(crop_height, dtype=np.float64)
            for yy in range(crop_height):
                source[yy] = row_distances[yy, xx]
            column_output = np.empty(crop_height, dtype=np.float64)
            column_arg = np.empty(crop_height, dtype=np.int64)
            _edt_1d_numba(source, column_output, column_arg)
            for yy in range(crop_height):
                distances_sq[yy, xx] = column_output[yy]

        object_pixel_count = counts[label]
        values = np.empty(object_pixel_count, dtype=np.float64)
        value_index = 0
        total = 0.0
        maximum = 0.0
        for yy in range(1, crop_height - 1):
            source_y = min_y[label] + yy - 1
            for xx in range(1, crop_width - 1):
                source_x = min_x[label] + xx - 1
                if labels[source_y, source_x] != label:
                    continue
                value = np.sqrt(distances_sq[yy, xx])
                values[value_index] = value
                value_index += 1
                total += value
                if value > maximum:
                    maximum = value

        values.sort()
        middle = object_pixel_count // 2
        max_radius[object_index] = maximum
        mean_radius[object_index] = total / object_pixel_count
        if object_pixel_count % 2 == 1:
            median_radius[object_index] = values[middle]
        else:
            median_radius[object_index] = 0.5 * (
                values[middle - 1] + values[middle]
            )
    return max_radius, mean_radius, median_radius


@njit(cache=True)
def _distance_to_label_edge_numba(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label

    output = np.zeros((height, width), dtype=np.float64)
    if max_label <= 0:
        return output

    min_y = np.full(max_label + 1, height, dtype=np.int64)
    min_x = np.full(max_label + 1, width, dtype=np.int64)
    max_y = np.zeros(max_label + 1, dtype=np.int64)
    max_x = np.zeros(max_label + 1, dtype=np.int64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1
            if y < min_y[label]:
                min_y[label] = y
            if x < min_x[label]:
                min_x[label] = x
            if y + 1 > max_y[label]:
                max_y[label] = y + 1
            if x + 1 > max_x[label]:
                max_x[label] = x + 1

    inf = 1.0e20
    for label in range(1, max_label + 1):
        if counts[label] <= 0:
            continue
        crop_y0 = min_y[label] - 1
        if crop_y0 < 0:
            crop_y0 = 0
        crop_x0 = min_x[label] - 1
        if crop_x0 < 0:
            crop_x0 = 0
        crop_y1 = max_y[label] + 1
        if crop_y1 > height:
            crop_y1 = height
        crop_x1 = max_x[label] + 1
        if crop_x1 > width:
            crop_x1 = width
        crop_height = crop_y1 - crop_y0
        crop_width = crop_x1 - crop_x0
        has_background = False
        for yy in range(crop_height):
            source_y = crop_y0 + yy
            for xx in range(crop_width):
                source_x = crop_x0 + xx
                if labels[source_y, source_x] != label:
                    has_background = True
                    break
            if has_background:
                break

        if not has_background:
            for yy in range(crop_height):
                source_y = crop_y0 + yy
                y_distance = yy + 1
                for xx in range(crop_width):
                    source_x = crop_x0 + xx
                    output[source_y, source_x] = np.sqrt(
                        (y_distance * y_distance) + (xx * xx)
                    )
            continue

        row_distances = np.empty((crop_height, crop_width), dtype=np.float64)
        distances_sq = np.empty((crop_height, crop_width), dtype=np.float64)

        for yy in range(crop_height):
            source = np.empty(crop_width, dtype=np.float64)
            source_y = crop_y0 + yy
            for xx in range(crop_width):
                source_x = crop_x0 + xx
                if labels[source_y, source_x] == label:
                    source[xx] = inf
                else:
                    source[xx] = 0.0
            row_output = np.empty(crop_width, dtype=np.float64)
            row_arg = np.empty(crop_width, dtype=np.int64)
            _edt_1d_numba(source, row_output, row_arg)
            for xx in range(crop_width):
                row_distances[yy, xx] = row_output[xx]

        for xx in range(crop_width):
            source = np.empty(crop_height, dtype=np.float64)
            for yy in range(crop_height):
                source[yy] = row_distances[yy, xx]
            column_output = np.empty(crop_height, dtype=np.float64)
            column_arg = np.empty(crop_height, dtype=np.int64)
            _edt_1d_numba(source, column_output, column_arg)
            for yy in range(crop_height):
                distances_sq[yy, xx] = column_output[yy]

        for yy in range(crop_height):
            source_y = crop_y0 + yy
            for xx in range(crop_width):
                source_x = crop_x0 + xx
                if labels[source_y, source_x] == label:
                    output[source_y, source_x] = np.sqrt(distances_sq[yy, xx])
    return output


@njit(cache=True)
def _maximum_position_of_labels_numba(
    image: np.ndarray,
    labels: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_count = label_ids.size
    max_label = 0
    for index in range(object_count):
        label = int(label_ids[index])
        if label > max_label:
            max_label = label

    best_values = np.full(max_label + 1, -np.inf, dtype=np.float64)
    best_y = np.full(max_label + 1, -1, dtype=np.int64)
    best_x = np.full(max_label + 1, -1, dtype=np.int64)
    seen = np.zeros(max_label + 1, dtype=np.bool_)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0 or label > max_label:
                continue
            value = image[y, x]
            if (not seen[label]) or value > best_values[label]:
                seen[label] = True
                best_values[label] = value
                best_y[label] = y
                best_x[label] = x

    centers_i = np.zeros(object_count, dtype=np.float64)
    centers_j = np.zeros(object_count, dtype=np.float64)
    for index in range(object_count):
        label = int(label_ids[index])
        if label > 0 and label <= max_label and seen[label]:
            centers_i[index] = float(best_y[label])
            centers_j[index] = float(best_x[label])
    return centers_i, centers_j


__all__ = [
    "CentrosomeNumpyShapeMeasurementBackendStrategy",
    "LegacyFastNumpyShapeMeasurementBackendStrategy",
    "NumbaNumpyShapeMeasurementBackendStrategy",
    "ShapeMeasurementBackendStrategy",
    "form_factor_values",
    "shape_measurement_backend",
]
