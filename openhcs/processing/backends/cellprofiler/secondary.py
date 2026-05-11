"""Secondary-object backend strategies for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit, prange

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    CellProfilerPlaneGeometry,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    cellprofiler_legacy_watershed,
)


class SecondaryMethod(Enum):
    """CellProfiler IdentifySecondaryObjects segmentation method."""

    PROPAGATION = ("propagation", True)
    WATERSHED_GRADIENT = ("watershed_gradient", True)
    WATERSHED_IMAGE = ("watershed_image", True)
    DISTANCE_N = ("distance_n", False)
    DISTANCE_B = ("distance_b", True)

    def __new__(cls, value: str, requires_threshold: bool):
        method = object.__new__(cls)
        method._value_ = value
        method.requires_threshold = requires_threshold
        return method


@dataclass(frozen=True, slots=True)
class LabelPropagationResult:
    """Regularized propagation labels and cumulative distances."""

    labels: np.ndarray
    distances: np.ndarray


@dataclass(frozen=True)
class SecondarySegmentationRequest:
    """Inputs needed by nominal secondary-object segmentation strategies."""

    image: np.ndarray
    labels: np.ndarray
    unedited_labels: np.ndarray
    thresholded: np.ndarray
    distance_to_dilate: int
    regularization_factor: float
    watershed_backend_provider: CellProfilerBackendProvider | None
    distance_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
    propagation_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION

    @property
    def has_primary_objects(self) -> bool:
        return self.unedited_labels.max() > 0

    @property
    def object_mask(self) -> np.ndarray:
        return self.thresholded | (self.unedited_labels > 0)


class SecondarySegmentationStrategy(ABC, metaclass=AutoRegisterMeta):
    """Segmentation strategy for one closed secondary-object method."""

    __registry_key__ = "method_label"
    method_label: ClassVar[str | None] = None
    method: ClassVar[SecondaryMethod | None] = None

    @classmethod
    def for_method(cls, method: SecondaryMethod) -> "SecondarySegmentationStrategy":
        return cls.__registry__[method.value]()

    def segment(self, request: SecondarySegmentationRequest) -> np.ndarray:
        if not request.has_primary_objects:
            return np.zeros_like(request.labels)
        return self._segment_non_empty(request)

    def propagate_labels(
        self,
        request: SecondarySegmentationRequest,
        *,
        regularization: float,
        max_distance: float | None = None,
    ) -> np.ndarray:
        """Propagate secondary labels through the configured backend."""
        geometry = CellProfilerPlaneGeometry.from_image_plane(request.image)
        labels = geometry.label_plane(request.unedited_labels)
        mask = geometry.binary_mask(request.thresholded)
        return SecondaryPropagationBackendStrategy.for_memory_type(
            backend_provider=request.propagation_backend_provider,
        ).propagate(
            request.image,
            labels,
            mask,
            regularization,
            max_distance=max_distance,
        )

    def watershed_secondary_labels(
        self,
        request: SecondarySegmentationRequest,
        watershed_image: np.ndarray,
    ) -> np.ndarray:
        """Build secondary labels from watershed markers and object mask."""
        return cellprofiler_legacy_watershed(
            watershed_image,
            markers=request.unedited_labels,
            mask=request.object_mask,
            connectivity=np.ones((3, 3), bool),
            backend_provider=request.watershed_backend_provider,
        )

    @abstractmethod
    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        """Segment secondary objects when primary labels are present."""


class DistanceOnlySegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.DISTANCE_N
    method_label = method.value

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        return SecondaryDistanceTransformBackendStrategy.for_memory_type(
            backend_provider=request.distance_backend_provider,
        ).nearest_label_expansion(
            request.unedited_labels,
            float(request.distance_to_dilate),
        )


class DistanceMaskedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.DISTANCE_B
    method_label = method.value

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        labels_out = self.propagate_labels(
            request,
            regularization=1.0,
            max_distance=float(request.distance_to_dilate),
        )
        labels_out[request.labels > 0] = request.labels[request.labels > 0]
        accepted_labels = np.unique(request.labels[request.labels > 0])
        if accepted_labels.size:
            labels_out[~np.isin(labels_out, accepted_labels)] = 0
        return labels_out


class PropagationSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.PROPAGATION
    method_label = method.value

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        return self.propagate_labels(
            request,
            regularization=request.regularization_factor,
        )


class GradientWatershedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.WATERSHED_GRADIENT
    method_label = method.value

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        from scipy.ndimage import sobel

        sobel_image = np.abs(sobel(request.image, axis=0)) + np.abs(
            sobel(request.image, axis=1)
        )
        return self.watershed_secondary_labels(request, sobel_image)


class ImageWatershedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.WATERSHED_IMAGE
    method_label = method.value

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        return self.watershed_secondary_labels(request, 1.0 - request.image)


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
    def propagate_result(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
    ) -> LabelPropagationResult:
        """Propagate seed labels through a mask and retain cumulative distances."""

    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
        *,
        max_distance: float | None = None,
    ) -> np.ndarray:
        """Propagate seed labels through a mask."""
        result = self.propagate_result(image, labels, mask, regularization)
        propagated = np.asarray(result.labels, dtype=np.int32)
        if max_distance is None:
            return propagated
        filtered = propagated.copy()
        source_labels = np.asarray(labels, dtype=np.int32)
        filtered[np.asarray(result.distances, dtype=np.float64) > float(max_distance)] = 0
        filtered[source_labels > 0] = source_labels[source_labels > 0]
        return filtered


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
    is_default_backend = False

    def propagate_result(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
    ) -> LabelPropagationResult:
        import centrosome.propagate

        if np.max(labels) == 0:
            label_array = np.asarray(labels, dtype=np.int32).copy()
            return LabelPropagationResult(
                labels=label_array,
                distances=np.zeros(label_array.shape, dtype=np.float64),
            )
        result, distance = centrosome.propagate.propagate(
            image,
            labels,
            mask,
            regularization,
        )
        return LabelPropagationResult(
            labels=np.asarray(result, dtype=np.int32),
            distances=np.asarray(distance, dtype=np.float64),
        )


class NumbaSecondaryPropagationBackendStrategy(
    SecondaryPropagationBackendStrategy,
):
    """Numba implementation of centrosome's regularized propagation semantics."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def propagate_result(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
    ) -> LabelPropagationResult:
        image_array = np.asarray(image, dtype=np.float64)
        label_array = np.asarray(labels, dtype=np.int32)
        mask_array = np.asarray(mask, dtype=np.bool_)
        if image_array.ndim != 2 or label_array.ndim != 2 or mask_array.ndim != 2:
            raise NotImplementedError(
                "Numba secondary propagation backend currently supports 2-D arrays."
            )
        if image_array.shape != label_array.shape or image_array.shape != mask_array.shape:
            raise ValueError("image, labels, and mask must have the same shape.")
        if np.max(label_array) == 0:
            return LabelPropagationResult(
                labels=label_array.copy(),
                distances=np.zeros(label_array.shape, dtype=np.float64),
            )
        propagated, distances = _propagate_labels_and_distances_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(label_array),
            np.ascontiguousarray(mask_array),
            float(regularization),
        )
        return LabelPropagationResult(labels=propagated, distances=distances)


def secondary_propagation_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> SecondaryPropagationBackendStrategy:
    """Return the selected secondary propagation backend."""
    return SecondaryPropagationBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


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


@njit(cache=True)
def _propagation_cost_numba(
    image: np.ndarray,
    y1: int,
    x1: int,
    y2: int,
    x2: int,
    weight: float,
) -> float:
    height, width = image.shape
    pixel_diff = 0.0
    for dy in range(-1, 2):
        yy1 = y1 + dy
        yy2 = y2 + dy
        if yy1 < 0:
            yy1 = 0
        elif yy1 >= height:
            yy1 = height - 1
        if yy2 < 0:
            yy2 = 0
        elif yy2 >= height:
            yy2 = height - 1
        for dx in range(-1, 2):
            xx1 = x1 + dx
            xx2 = x2 + dx
            if xx1 < 0:
                xx1 = 0
            elif xx1 >= width:
                xx1 = width - 1
            if xx2 < 0:
                xx2 = 0
            elif xx2 >= width:
                xx2 = width - 1
            v1 = image[yy1, xx1]
            v2 = image[yy2, xx2]
            if v1 > v2:
                pixel_diff += v1 - v2
            else:
                pixel_diff += v2 - v1
    manhattan_distance = abs(y1 - y2) + abs(x1 - x2)
    return np.sqrt(pixel_diff * pixel_diff + manhattan_distance * weight * weight)


@njit(cache=True)
def _propagation_heap_less(
    left_value: float,
    left_label: int,
    left_y: int,
    left_x: int,
    right_value: float,
    right_label: int,
    right_y: int,
    right_x: int,
) -> bool:
    if left_value != right_value:
        return left_value < right_value
    if left_label != right_label:
        return left_label < right_label
    if left_y != right_y:
        return left_y < right_y
    return left_x < right_x


@njit(cache=True)
def _propagation_heap_swap(
    values: np.ndarray,
    labels: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    left: int,
    right: int,
) -> None:
    value = values[left]
    label = labels[left]
    y = ys[left]
    x = xs[left]
    values[left] = values[right]
    labels[left] = labels[right]
    ys[left] = ys[right]
    xs[left] = xs[right]
    values[right] = value
    labels[right] = label
    ys[right] = y
    xs[right] = x


@njit(cache=True)
def _propagation_heap_push(
    values: np.ndarray,
    labels: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    size: int,
    value: float,
    label: int,
    y: int,
    x: int,
) -> int:
    values[size] = value
    labels[size] = label
    ys[size] = y
    xs[size] = x
    size += 1
    position = size - 1
    while position > 0:
        parent = (position - 1) // 2
        if not _propagation_heap_less(
            values[position],
            labels[position],
            ys[position],
            xs[position],
            values[parent],
            labels[parent],
            ys[parent],
            xs[parent],
        ):
            break
        _propagation_heap_swap(values, labels, ys, xs, position, parent)
        position = parent
    return size


@njit(cache=True)
def _propagation_heap_pop(
    values: np.ndarray,
    labels: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    size: int,
) -> tuple[int, float, int, int, int]:
    value = values[0]
    label = labels[0]
    y = ys[0]
    x = xs[0]
    size -= 1
    if size > 0:
        values[0] = values[size]
        labels[0] = labels[size]
        ys[0] = ys[size]
        xs[0] = xs[size]
        position = 0
        while True:
            left = position * 2 + 1
            right = left + 1
            if left >= size:
                break
            smallest = left
            if right < size and _propagation_heap_less(
                values[right],
                labels[right],
                ys[right],
                xs[right],
                values[left],
                labels[left],
                ys[left],
                xs[left],
            ):
                smallest = right
            if not _propagation_heap_less(
                values[smallest],
                labels[smallest],
                ys[smallest],
                xs[smallest],
                values[position],
                labels[position],
                ys[position],
                xs[position],
            ):
                break
            _propagation_heap_swap(values, labels, ys, xs, position, smallest)
            position = smallest
    return size, value, label, y, x


@njit(cache=True)
def _propagate_labels_numba(
    image: np.ndarray,
    seed_labels: np.ndarray,
    mask: np.ndarray,
    weight: float,
    max_distance: float,
) -> np.ndarray:
    output, distances = _propagate_labels_and_distances_numba(
        image,
        seed_labels,
        mask,
        weight,
    )
    if max_distance >= 0.0:
        height, width = output.shape
        for y in range(height):
            for x in range(width):
                if distances[y, x] > max_distance and seed_labels[y, x] <= 0:
                    output[y, x] = 0
    return output


@njit(cache=True)
def _propagate_labels_and_distances_numba(
    image: np.ndarray,
    seed_labels: np.ndarray,
    mask: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.int32)
    distances = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            if seed_labels[y, x] > 0:
                distances[y, x] = 0.0
            else:
                distances[y, x] = -1.0

    capacity = height * width * 8
    heap_values = np.empty(capacity, dtype=np.float64)
    heap_labels = np.empty(capacity, dtype=np.int32)
    heap_ys = np.empty(capacity, dtype=np.int32)
    heap_xs = np.empty(capacity, dtype=np.int32)
    heap_size = 0
    for y in range(height):
        for x in range(width):
            label = int(seed_labels[y, x])
            if (
                label > 0
                and mask[y, x]
                and _propagation_seed_frontier_pixel(seed_labels, mask, y, x)
            ):
                heap_size = _propagation_heap_push(
                    heap_values,
                    heap_labels,
                    heap_ys,
                    heap_xs,
                    heap_size,
                    0.0,
                    label,
                    y,
                    x,
                )

    delta_y = np.array((-1, -1, -1, 0, 0, 1, 1, 1), dtype=np.int32)
    delta_x = np.array((-1, 0, 1, -1, 1, -1, 0, 1), dtype=np.int32)
    while heap_size > 0:
        heap_size, _value, label, y1, x1 = _propagation_heap_pop(
            heap_values,
            heap_labels,
            heap_ys,
            heap_xs,
            heap_size,
        )
        if output[y1, x1] != 0:
            continue
        output[y1, x1] = label
        d0 = distances[y1, x1]
        for index in range(8):
            y2 = y1 + int(delta_y[index])
            x2 = x1 + int(delta_x[index])
            if y2 < 0 or y2 >= height or x2 < 0 or x2 >= width:
                continue
            if output[y2, x2] > 0 or not mask[y2, x2]:
                continue
            distance = _propagation_cost_numba(
                image,
                y1,
                x1,
                y2,
                x2,
                weight,
            ) + d0
            if distances[y2, x2] == -1.0 or distances[y2, x2] > distance:
                distances[y2, x2] = distance
                heap_size = _propagation_heap_push(
                    heap_values,
                    heap_labels,
                    heap_ys,
                    heap_xs,
                    heap_size,
                    distance,
                    label,
                    y2,
                    x2,
                )
    for y in range(height):
        for x in range(width):
            if seed_labels[y, x] > 0:
                output[y, x] = seed_labels[y, x]
    return output, distances


@njit(cache=True)
def _propagation_seed_frontier_pixel(
    seed_labels: np.ndarray,
    mask: np.ndarray,
    y: int,
    x: int,
) -> bool:
    height, width = seed_labels.shape
    for dy in range(-1, 2):
        yy = y + dy
        if yy < 0 or yy >= height:
            continue
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            xx = x + dx
            if xx < 0 or xx >= width:
                continue
            if mask[yy, xx] and seed_labels[yy, xx] == 0:
                return True
    return False


__all__ = [
    "CentrosomeSecondaryPropagationBackendStrategy",
    "DistanceMaskedSegmentationStrategy",
    "DistanceOnlySegmentationStrategy",
    "GradientWatershedSegmentationStrategy",
    "ImageWatershedSegmentationStrategy",
    "LabelPropagationResult",
    "NumbaSecondaryPropagationBackendStrategy",
    "NumbaSecondaryDistanceTransformBackendStrategy",
    "NumpySecondaryDistanceTransformBackendStrategy",
    "PropagationSegmentationStrategy",
    "SecondaryDistanceTransformBackendStrategy",
    "SecondaryMethod",
    "SecondaryPropagationBackendStrategy",
    "SecondarySegmentationRequest",
    "SecondarySegmentationStrategy",
    "secondary_propagation_backend",
]
