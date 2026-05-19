"""Secondary-object backend strategies for CellProfiler-compatible processing."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, ClassVar, Tuple

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import (
    RuntimePure2DSliceBatchRequest,
    pure_2d_batch_executor,
)
from openhcs.core.registry_strategies import RegisteredLeafClassSpec
from openhcs.core.runtime_semantics import (
    DenseObjectLabelPairAligner,
    ExplicitObjectLabelDomainDeclaration,
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ParentChildRelationshipPayload,
    dense_object_label_plane_id_domains,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    ObjectLabelSet,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    object_label_dense_array,
    object_label_payload_from_source_image,
    object_label_payload_with_dense_labels,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler.image_geometry import (
    CellProfilerPlaneGeometry,
    collapse_singleton_plane_stack,
)
from openhcs.processing.backends.cellprofiler.morphology import MorphologyBackendStrategy
from openhcs.processing.backends.cellprofiler.outlines import ObjectOutlineBackendStrategy
from openhcs.processing.backends.cellprofiler.thresholding import (
    CellProfilerAveragingMethod,
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
    CellProfilerThresholdMethod,
    CellProfilerThresholdScope,
    CellProfilerVarianceMethod,
    ThresholdPrimitiveBackendStrategy,
    cellprofiler_threshold,
    cellprofiler_threshold_diagnostics,
    normalize_cellprofiler_image,
    threshold_primitives,
    unit_interval_scale_for_threshold_diagnostics,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    cellprofiler_legacy_watershed,
)

logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)


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

    def prepare_backend(self) -> None:
        labels = np.array([[0, 1, 0], [0, 0, 2], [0, 0, 0]], dtype=np.int32)
        self.distance_to_foreground(labels)
        self.nearest_label_expansion(labels, 2.0)

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
        *,
        max_distance: float | None = None,
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
        result = self.propagate_result(
            image,
            labels,
            mask,
            regularization,
            max_distance=max_distance,
        )
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
        *,
        max_distance: float | None = None,
    ) -> LabelPropagationResult:
        del max_distance
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

    def prepare_backend(self) -> None:
        image = np.arange(9, dtype=np.float64).reshape((3, 3))
        labels = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]], dtype=np.int32)
        mask = np.ones(labels.shape, dtype=np.bool_)
        self.propagate_result(image, labels, mask, 0.1)
        self.propagate_result(image, labels, mask, 1.0, max_distance=2.0)

    def propagate_result(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
        *,
        max_distance: float | None = None,
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
            -1.0 if max_distance is None else float(max_distance),
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


@njit(cache=True)
def _distance_to_positive_labels_numba(labels: np.ndarray) -> np.ndarray:
    distances, _nearest_y, _nearest_x = _edt_feature_transform_numba(labels)
    return distances


@njit(cache=True)
def _nearest_label_expansion_numba(
    labels: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    distances, nearest_y, nearest_x = _edt_feature_transform_numba(labels)
    height, width = labels.shape
    output = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            if distances[y, x] <= max_distance:
                output[y, x] = labels[nearest_y[y, x], nearest_x[y, x]]
    return output


@njit(cache=True)
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

    for y in range(height):
        source = np.empty(width, dtype=np.float64)
        for x in range(width):
            source[x] = 0.0 if labels[y, x] > 0 else inf
        row_output = np.empty(width, dtype=np.float64)
        row_arg = np.empty(width, dtype=np.int64)
        _edt_1d_numba(source, row_output, row_arg)
        for x in range(width):
            row_distances[y, x] = row_output[x]
            row_nearest_x[y, x] = row_arg[x]

    for x in range(width):
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
    for y in range(height):
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
        max_distance,
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
    max_distance: float,
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
        if max_distance >= 0.0 and _value > max_distance:
            break
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
            if max_distance >= 0.0 and distance > max_distance:
                continue
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


class ThresholdMethod(Enum):
    OTSU = "otsu"
    LI = "li"
    MINIMUM = "minimum"
    TRIANGLE = "triangle"


@dataclass
class SecondaryObjectStats:
    slice_index: int
    object_count: int
    mean_area: float
    median_area: float
    total_area: int
    area_coverage_percent: float
    threshold_value: float
    original_threshold: float = 0.0
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0


@dataclass(frozen=True)
class SecondaryImageInputs:
    image: np.ndarray
    labels: np.ndarray
    unedited_labels: np.ndarray


@dataclass(frozen=True)
class SecondaryThresholdResult:
    value: float
    original_value: float
    mask: np.ndarray
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0


@dataclass(frozen=True)
class SecondaryThresholdRequest:
    image: np.ndarray
    image_mask: np.ndarray | None
    method: SecondaryMethod
    threshold_scope: CellProfilerThresholdScope
    threshold_method: CellProfilerThresholdMethod | ThresholdMethod | str
    threshold_correction_factor: float
    threshold_min: float
    threshold_max: float
    threshold_smoothing_scale: float
    otsu_class_count: CellProfilerOtsuMethod
    assign_middle_to_foreground: CellProfilerThresholdAssignment
    log_transform: bool
    adaptive_window_size: int
    lower_outlier_fraction: float
    upper_outlier_fraction: float
    averaging_method: CellProfilerAveragingMethod
    variance_method: CellProfilerVarianceMethod
    number_of_deviations: float
    manual_threshold: float
    diagnostics_unit_interval_scale: int | None = None


@dataclass(frozen=True)
class SecondaryObjectLabels:
    """CellProfiler object-container label variants for secondary objects."""

    segmented: np.ndarray
    unedited_segmented: np.ndarray
    small_removed_segmented: np.ndarray

    @classmethod
    def from_raw_labels(
        cls,
        labels: np.ndarray,
        *,
        fill_holes: bool,
        discard_edge_objects: bool,
        primary_labels: np.ndarray,
        morphology: MorphologyBackendStrategy,
    ) -> "SecondaryObjectLabels":
        small_removed = labels
        if fill_holes and small_removed.max() > 0:
            small_removed = morphology.fill_labeled_holes(small_removed)
        segmented = _filter_labels(small_removed, primary_labels)
        if discard_edge_objects and segmented.max() > 0:
            segmented = _discard_edge_objects(segmented, morphology)
        segmented = segmented.astype(np.int32, copy=False)
        small_removed = small_removed.astype(np.int32, copy=False)
        return cls(
            segmented=segmented,
            unedited_segmented=small_removed,
            small_removed_segmented=small_removed,
        )

    @property
    def object_count(self) -> int:
        return int(np.max(self.segmented)) if self.segmented.size else 0

    def payload_for_image(self, image: object) -> ObjectLabelPayload:
        return object_label_payload_from_source_image(
            image,
            self.segmented,
            unedited_labels=self.unedited_segmented,
            small_removed_labels=self.small_removed_segmented,
            declared_object_count=self.object_count,
        )


class ThresholdCalculator(ABC, metaclass=AutoRegisterMeta):
    """Threshold strategy for one closed CellProfiler threshold method."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    method_label: ClassVar[str | None] = None
    method: ClassVar[ThresholdMethod | None] = None
    primitive: ClassVar[
        Callable[[ThresholdPrimitiveBackendStrategy, np.ndarray], float]
    ]

    @classmethod
    def for_method(cls, method: ThresholdMethod) -> "ThresholdCalculator":
        return cls.__registry__[method.value]()

    def calculate(self, image: np.ndarray) -> float:
        """Calculate a threshold value for a normalized intensity image."""
        return type(self).primitive(threshold_primitives(), image)


@dataclass(frozen=True)
class ThresholdCalculatorDeclaration(RegisteredLeafClassSpec):
    """Typed declaration for metadata-only secondary threshold calculators."""

    method: ThresholdMethod
    primitive: Callable[[ThresholdPrimitiveBackendStrategy, np.ndarray], float]
    base_type: type[ThresholdCalculator] = field(
        default=ThresholdCalculator,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def class_name(self) -> str:
        method_name = "".join(part.title() for part in self.method.name.split("_"))
        return f"{method_name}ThresholdCalculator"

    def class_attributes(self) -> dict[str, object]:
        primitive = self.primitive

        def concrete_primitive(
            backend: ThresholdPrimitiveBackendStrategy,
            image: np.ndarray,
        ) -> float:
            return primitive(backend, image)

        return {
            "method": self.method,
            "method_label": self.method.value,
            "primitive": staticmethod(concrete_primitive),
        }


THRESHOLD_CALCULATOR_DECLARATIONS = (
    ThresholdCalculatorDeclaration(
        ThresholdMethod.OTSU,
        ThresholdPrimitiveBackendStrategy.otsu_threshold,
    ),
    ThresholdCalculatorDeclaration(
        ThresholdMethod.LI,
        ThresholdPrimitiveBackendStrategy.li_threshold,
    ),
    ThresholdCalculatorDeclaration(
        ThresholdMethod.MINIMUM,
        ThresholdPrimitiveBackendStrategy.minimum_threshold,
    ),
    ThresholdCalculatorDeclaration(
        ThresholdMethod.TRIANGLE,
        ThresholdPrimitiveBackendStrategy.triangle_threshold,
    ),
)

for threshold_calculator_declaration in THRESHOLD_CALCULATOR_DECLARATIONS:
    threshold_calculator_declaration.declare_in(globals())


def _normalize_secondary_inputs(
    image: np.ndarray,
    primary_labels: np.ndarray | ObjectLabelPayload,
) -> SecondaryImageInputs:
    image = collapse_singleton_plane_stack(np.asarray(image))
    if isinstance(primary_labels, ObjectLabelPayload):
        final_labels = collapse_singleton_plane_stack(
            np.asarray(primary_labels.labels, dtype=np.int32)
        )
        unedited_labels = np.asarray(
            primary_labels.labels_for_variant("unedited"),
            dtype=np.int32,
        )
        unedited_labels = collapse_singleton_plane_stack(unedited_labels)
        return SecondaryImageInputs(
            image=image,
            labels=final_labels,
            unedited_labels=_secondary_seed_labels(final_labels, unedited_labels),
        )
    if image.ndim == 3 and image.shape[0] == 2:
        labels = image[1].astype(np.int32)
        return SecondaryImageInputs(
            image=image[0],
            labels=labels,
            unedited_labels=labels,
        )
    labels = collapse_singleton_plane_stack(np.asarray(primary_labels, dtype=np.int32))
    return SecondaryImageInputs(
        image=image,
        labels=labels,
        unedited_labels=labels,
    )


def _secondary_seed_labels(
    final_labels: np.ndarray,
    unedited_labels: np.ndarray,
) -> np.ndarray:
    """Match CellProfiler's secondary-object seed contract.

    CellProfiler seeds secondary segmentation from unedited primary labels, but
    removes non-edge labels that were rejected from the final primary objects.
    Edge-touching rejected labels remain so they can constrain propagated
    secondary boundaries without becoming accepted parent objects.
    """
    labels_in = np.asarray(unedited_labels, dtype=np.int32).copy()
    if labels_in.size == 0 or labels_in.max() <= 0:
        return labels_in

    final = np.asarray(final_labels, dtype=np.int32)
    if final.shape != labels_in.shape:
        aligned_final = np.zeros(labels_in.shape, dtype=final.dtype)
        i_max = min(labels_in.shape[0], final.shape[0])
        j_max = min(labels_in.shape[1], final.shape[1])
        aligned_final[:i_max, :j_max] = final[:i_max, :j_max]
        final = aligned_final

    edge_labels = np.unique(
        np.concatenate(
            (
                labels_in[0, :],
                labels_in[-1, :],
                labels_in[:, 0],
                labels_in[:, -1],
            )
        )
    )
    is_touching_lookup = np.zeros(int(labels_in.max()) + 1, dtype=bool)
    is_touching_lookup[edge_labels.astype(int)] = True
    return _secondary_seed_label_remap_numba(
        np.ascontiguousarray(labels_in, dtype=np.int32),
        np.ascontiguousarray(final, dtype=np.int32),
        is_touching_lookup,
    )


@njit(cache=True)
def _secondary_seed_label_remap_numba(
    unedited_labels: np.ndarray,
    final_labels: np.ndarray,
    is_touching_edge: np.ndarray,
) -> np.ndarray:
    max_unedited = int(unedited_labels.max())
    max_final = int(final_labels.max())
    accepted_mapping = np.zeros(max_unedited + 1, dtype=np.int32)

    flat_unedited = unedited_labels.ravel()
    flat_final = final_labels.ravel()
    for index in range(flat_unedited.size):
        unedited_label = int(flat_unedited[index])
        final_label = int(flat_final[index])
        if unedited_label > 0 and final_label > accepted_mapping[unedited_label]:
            accepted_mapping[unedited_label] = final_label

    edge_mapping = np.zeros(max_unedited + 1, dtype=np.int32)
    next_edge_label = max_final + 1
    for label in range(1, max_unedited + 1):
        if accepted_mapping[label] == 0 and is_touching_edge[label]:
            edge_mapping[label] = next_edge_label
            next_edge_label += 1

    output = np.zeros(unedited_labels.shape, dtype=np.int32)
    output_flat = output.ravel()
    for index in range(flat_unedited.size):
        unedited_label = int(flat_unedited[index])
        if unedited_label == 0:
            continue
        accepted_label = accepted_mapping[unedited_label]
        if accepted_label > 0:
            output_flat[index] = accepted_label
        else:
            output_flat[index] = edge_mapping[unedited_label]
    return output


def _normalize_intensity_image(image: np.ndarray) -> np.ndarray:
    return normalize_cellprofiler_image(image)


def _threshold_secondary_objects(
    request: SecondaryThresholdRequest,
) -> SecondaryThresholdResult:
    if not request.method.requires_threshold:
        return SecondaryThresholdResult(
            value=0.0,
            original_value=0.0,
            mask=(
                np.ones_like(request.image, dtype=bool)
                if request.image_mask is None
                else np.asarray(request.image_mask, dtype=bool)
            ),
        )

    thresholded, threshold_value, original_threshold = cellprofiler_threshold(
        request.image,
        use_advanced_settings=True,
        threshold_scope=request.threshold_scope,
        threshold_method=_coerce_threshold_method(request.threshold_method),
        threshold_smoothing_scale=request.threshold_smoothing_scale,
        threshold_correction_factor=request.threshold_correction_factor,
        threshold_min=request.threshold_min,
        threshold_max=request.threshold_max,
        manual_threshold=request.manual_threshold,
        otsu_class_count=request.otsu_class_count,
        assign_middle_to_foreground=request.assign_middle_to_foreground,
        log_transform=request.log_transform,
        adaptive_window_size=request.adaptive_window_size,
        lower_outlier_fraction=request.lower_outlier_fraction,
        upper_outlier_fraction=request.upper_outlier_fraction,
        averaging_method=request.averaging_method,
        variance_method=request.variance_method,
        number_of_deviations=request.number_of_deviations,
        mask=request.image_mask,
    )
    diagnostics = cellprofiler_threshold_diagnostics(
        request.image,
        thresholded,
        final_threshold=threshold_value,
        original_threshold=original_threshold,
        mask=request.image_mask,
        proven_unit_interval_scale=request.diagnostics_unit_interval_scale,
    )
    return SecondaryThresholdResult(
        value=threshold_value,
        original_value=diagnostics.original_threshold,
        mask=thresholded,
        weighted_variance=diagnostics.weighted_variance,
        sum_of_entropies=diagnostics.sum_of_entropies,
    )


def _coerce_threshold_method(
    threshold_method: CellProfilerThresholdMethod | ThresholdMethod | str,
) -> CellProfilerThresholdMethod:
    if isinstance(threshold_method, CellProfilerThresholdMethod):
        return threshold_method
    if isinstance(threshold_method, str):
        return coerce_cellprofiler_enum(CellProfilerThresholdMethod, threshold_method)
    return {
        ThresholdMethod.OTSU: CellProfilerThresholdMethod.OTSU,
        ThresholdMethod.LI: CellProfilerThresholdMethod.LI,
        ThresholdMethod.MINIMUM: CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
        ThresholdMethod.TRIANGLE: CellProfilerThresholdMethod.TRIANGLE,
    }[threshold_method]


def _filter_labels(labels_out: np.ndarray, primary_labels: np.ndarray) -> np.ndarray:
    """Keep secondary labels associated with accepted primary labels."""
    max_out = int(np.max(labels_out))
    if max_out <= 0:
        return labels_out.copy()
    if primary_labels.shape != labels_out.shape:
        aligned_primary = np.zeros(labels_out.shape, primary_labels.dtype)
        i_max = min(labels_out.shape[0], primary_labels.shape[0])
        j_max = min(labels_out.shape[1], primary_labels.shape[1])
        aligned_primary[:i_max, :j_max] = primary_labels[:i_max, :j_max]
    else:
        aligned_primary = primary_labels
    return _filter_labels_numba(
        np.ascontiguousarray(labels_out, dtype=np.int32),
        np.ascontiguousarray(aligned_primary, dtype=np.int32),
        max_out,
    )


@njit(cache=True)
def _filter_labels_numba(
    labels_out: np.ndarray,
    aligned_primary: np.ndarray,
    max_out: int,
) -> np.ndarray:
    lookup = np.zeros(max_out + 1, dtype=np.int32)
    labels_flat = labels_out.ravel()
    primary_flat = aligned_primary.ravel()
    for index in range(labels_flat.size):
        label = int(labels_flat[index])
        if label <= 0:
            continue
        primary_label = int(primary_flat[index])
        if primary_label > lookup[label]:
            lookup[label] = primary_label
    lookup[0] = 0

    filtered = np.empty(labels_out.shape, dtype=np.int32)
    filtered_flat = filtered.ravel()
    for index in range(labels_flat.size):
        filtered_flat[index] = lookup[int(labels_flat[index])]
    return filtered


def _discard_edge_objects(
    labels: np.ndarray,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    edge_labels = np.unique(np.concatenate([
        labels[0, :],
        labels[-1, :],
        labels[:, 0],
        labels[:, -1],
    ]))
    labels_out = labels.copy()
    for edge_label in edge_labels:
        if edge_label > 0:
            labels_out[labels_out == edge_label] = 0

    if labels_out.max() == 0:
        return labels_out
    relabeled, _count = morphology.connected_components(labels_out > 0, connectivity=2)
    return relabeled.astype(np.int32, copy=False)


def _secondary_label_area_statistics(labels: np.ndarray) -> tuple[int, float, float, int]:
    areas = np.bincount(np.asarray(labels).ravel())[1:]
    positive_areas = areas[areas > 0]
    object_count = int(positive_areas.size)
    if object_count == 0:
        return 0, 0.0, 0.0, 0
    return (
        object_count,
        float(np.mean(positive_areas)),
        float(np.median(positive_areas)),
        int(np.sum(positive_areas)),
    )


def _secondary_object_stats(
    labels: np.ndarray,
    *,
    image_shape: tuple[int, int],
    threshold_value: float,
    original_threshold: float,
    weighted_variance: float,
    sum_of_entropies: float,
) -> SecondaryObjectStats:
    object_count, mean_area, median_area, total_area = (
        _secondary_label_area_statistics(labels)
    )

    height, width = image_shape
    area_coverage = 100.0 * total_area / (height * width) if height * width else 0.0
    return SecondaryObjectStats(
        slice_index=0,
        object_count=object_count,
        mean_area=mean_area,
        median_area=median_area,
        total_area=total_area,
        area_coverage_percent=area_coverage,
        threshold_value=float(threshold_value),
        original_threshold=float(original_threshold),
        weighted_variance=float(weighted_variance),
        sum_of_entropies=float(sum_of_entropies),
    )


@numpy
def identify_secondary_objects(
    image: np.ndarray,
    primary_labels: np.ndarray,
    method: SecondaryMethod = SecondaryMethod.PROPAGATION,
    threshold_scope: CellProfilerThresholdScope = CellProfilerThresholdScope.GLOBAL,
    threshold_method: CellProfilerThresholdMethod = CellProfilerThresholdMethod.OTSU,
    threshold_smoothing_scale: float = 0.0,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    manual_threshold: float = 0.0,
    otsu_class_count: CellProfilerOtsuMethod = CellProfilerOtsuMethod.TWO_CLASS,
    assign_middle_to_foreground: CellProfilerThresholdAssignment = (
        CellProfilerThresholdAssignment.FOREGROUND
    ),
    log_transform: bool = False,
    adaptive_window_size: int = 10,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: CellProfilerAveragingMethod = CellProfilerAveragingMethod.MEAN,
    variance_method: CellProfilerVarianceMethod = (
        CellProfilerVarianceMethod.STANDARD_DEVIATION
    ),
    number_of_deviations: float = 2.0,
    distance_to_dilate: int = 10,
    regularization_factor: float = 0.05,
    fill_holes: bool = True,
    discard_edge_objects: bool = False,
    watershed_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    distance_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    propagation_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[
    np.ndarray,
    SecondaryObjectStats,
    ParentChildRelationshipPayload,
    np.ndarray,
]:
    """
    Identify secondary objects using primary objects as seeds.

    Args:
        image: Input intensity image, shape (2, H, W) where [0] is intensity, [1] is primary labels
               OR shape (H, W) if primary_labels provided separately
        primary_labels: Label image of primary objects (seeds)
        method: Method for identifying secondary objects
        threshold_method: Method for thresholding the image
        threshold_correction_factor: Factor to multiply threshold by
        threshold_min: Minimum threshold value
        threshold_max: Maximum threshold value
        distance_to_dilate: Pixels to expand for distance methods
        regularization_factor: Lambda for propagation method (0=gradient only, higher=more distance)
        fill_holes: Whether to fill holes in identified objects
        discard_edge_objects: Whether to discard objects touching image border

    Returns:
        Tuple of (image, stats, parent-child relationships, secondary_labels)
    """
    profile_total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    method = coerce_cellprofiler_enum(SecondaryMethod, method)
    morphology = MorphologyBackendStrategy.for_callable(
        identify_secondary_objects,
        backend_provider=morphology_backend_provider,
    )
    input_mask = image_payload_mask(image)
    if input_mask is not None:
        input_mask = collapse_singleton_plane_stack(np.asarray(input_mask, dtype=bool))
    raw_image_data = image_payload_data(image)
    diagnostics_unit_interval_scale = unit_interval_scale_for_threshold_diagnostics(
        np.asarray(raw_image_data),
        image_payload_metadata(image),
    )
    inputs = _normalize_secondary_inputs(raw_image_data, primary_labels)
    img = _normalize_intensity_image(inputs.image)
    runtime_profiler.log(
        "iso_prepare_inputs",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    threshold = _threshold_secondary_objects(
        SecondaryThresholdRequest(
            image=img,
            image_mask=input_mask,
            method=method,
            threshold_scope=threshold_scope,
            threshold_method=threshold_method,
            threshold_smoothing_scale=threshold_smoothing_scale,
            threshold_correction_factor=threshold_correction_factor,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            manual_threshold=manual_threshold,
            otsu_class_count=otsu_class_count,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            adaptive_window_size=adaptive_window_size,
            lower_outlier_fraction=lower_outlier_fraction,
            upper_outlier_fraction=upper_outlier_fraction,
            averaging_method=averaging_method,
            variance_method=variance_method,
            number_of_deviations=number_of_deviations,
            diagnostics_unit_interval_scale=diagnostics_unit_interval_scale,
        )
    )
    runtime_profiler.log(
        "iso_threshold",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    raw_labels = SecondarySegmentationStrategy.for_method(method).segment(
        SecondarySegmentationRequest(
            image=img,
            labels=inputs.labels,
            unedited_labels=inputs.unedited_labels,
            thresholded=threshold.mask,
            distance_to_dilate=distance_to_dilate,
            regularization_factor=regularization_factor,
            watershed_backend_provider=watershed_backend_provider,
            distance_backend_provider=distance_backend_provider,
            propagation_backend_provider=propagation_backend_provider,
        )
    )
    runtime_profiler.log(
        "iso_segment",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    object_labels = SecondaryObjectLabels.from_raw_labels(
        raw_labels,
        fill_holes=fill_holes,
        discard_edge_objects=discard_edge_objects,
        primary_labels=inputs.labels,
        morphology=morphology,
    )
    runtime_profiler.log(
        "iso_label_variants",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    stats = _secondary_object_stats(
        object_labels.segmented,
        image_shape=img.shape,
        threshold_value=threshold.value,
        original_threshold=threshold.original_value,
        weighted_variance=threshold.weighted_variance,
        sum_of_entropies=threshold.sum_of_entropies,
    )
    runtime_profiler.log(
        "iso_stats",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    relationships = object_label_parent_child_payload(
        primary_labels if isinstance(primary_labels, ObjectLabelPayload) else inputs.labels,
        object_labels.segmented,
    )
    runtime_profiler.log(
        "iso_relationships",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    runtime_profiler.log(
        "iso_total",
        time.perf_counter() - profile_total_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )

    return img.astype(np.float32), stats, relationships, object_labels.payload_for_image(image)


@processing_prepare(identify_secondary_objects)
def _prepare_identify_secondary_objects() -> None:
    """Compile secondary-object threshold, distance, and propagation kernels."""
    image = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.ogrid[:64, :64]
    image[((yy - 24) ** 2 + (xx - 24) ** 2) <= 18 * 18] = 0.7
    image[((yy - 40) ** 2 + (xx - 40) ** 2) <= 14 * 14] = 0.5
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[20:28, 20:28] = 1
    labels[36:44, 36:44] = 2
    identify_secondary_objects.__wrapped__(
        image,
        labels,
        method=SecondaryMethod.PROPAGATION,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        threshold_smoothing_scale=1.3488,
        regularization_factor=0.05,
    )
    identify_secondary_objects.__wrapped__(
        image,
        labels,
        method=SecondaryMethod.DISTANCE_B,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        threshold_smoothing_scale=1.3488,
        distance_to_dilate=8,
    )


@dataclass
class TertiaryObjectStats:
    """Runtime measurements emitted by IdentifyTertiaryObjects."""

    slice_index: int
    object_count: int
    mean_area: float
    primary_parent_count: int
    secondary_parent_count: int


@dataclass(frozen=True, slots=True)
class TertiaryObjectLabelOutput:
    """Typed tertiary labels preserving the secondary object-label domain."""

    source: object
    labels: np.ndarray

    def value(self) -> object:
        if not isinstance(self.source, (ObjectLabelPayload, ObjectLabelSet)):
            return self.labels
        return object_label_payload_with_dense_labels(
            self.source,
            self.labels,
            domain_declaration=ExplicitObjectLabelDomainDeclaration(
                ObjectLabelDomain(
                    declared_object_id_domains=dense_object_label_plane_id_domains(
                        self.labels,
                        domain_scope=ObjectLabelDomainScope.PLANE,
                    ),
                    scope=ObjectLabelDomainScope.PLANE,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class TertiaryObjectMeasurement:
    """Nominal measurement summary for one tertiary label plane."""

    labels: np.ndarray

    @property
    def positive_label_count(self) -> int:
        return int(np.count_nonzero(np.bincount(np.asarray(self.labels).ravel())[1:]))

    @property
    def positive_label_mean_area(self) -> tuple[int, float]:
        areas = np.bincount(np.asarray(self.labels).ravel())[1:]
        positive_areas = areas[areas > 0]
        if positive_areas.size == 0:
            return 0, 0.0
        return int(positive_areas.size), float(np.mean(positive_areas))


@dataclass(frozen=True, slots=True)
class TertiaryObjectInputs:
    """Dense aligned label inputs and source payloads for tertiary segmentation."""

    primary_source: object
    secondary_source: object
    primary_array: np.ndarray
    secondary_array: np.ndarray

    @classmethod
    def from_labels(
        cls,
        primary_labels: np.ndarray | ObjectLabelPayload,
        secondary_labels: np.ndarray | ObjectLabelPayload,
    ) -> "TertiaryObjectInputs":
        primary_array, secondary_array = DenseObjectLabelPairAligner(
            primary_labels,
            secondary_labels,
        ).aligned()
        primary_array = np.asarray(primary_array, dtype=np.int32)
        secondary_array = np.asarray(secondary_array, dtype=np.int32)
        if primary_array.ndim == 3:
            primary_array = primary_array[0]
        if secondary_array.ndim == 3:
            secondary_array = secondary_array[0]
        if primary_array.shape != secondary_array.shape:
            raise ValueError(
                f"Primary and secondary label shapes must match. "
                f"Got {primary_array.shape} vs {secondary_array.shape}"
            )
        return cls(
            primary_source=primary_labels,
            secondary_source=secondary_labels,
            primary_array=primary_array,
            secondary_array=secondary_array,
        )


class TertiaryObjectSegmentation:
    """Load-bearing IdentifyTertiaryObjects segmentation policy."""

    def segment(
        self,
        inputs: TertiaryObjectInputs,
        *,
        shrink_primary: bool,
        outline_backend_provider: BackendProviderInput,
    ) -> np.ndarray:
        primary_outline = ObjectOutlineBackendStrategy.for_memory_type(
            backend_provider=outline_backend_provider,
        ).outline(inputs.primary_array)

        tertiary_labels = inputs.secondary_array.copy()
        if shrink_primary:
            primary_mask = np.logical_or(inputs.primary_array == 0, primary_outline > 0)
        else:
            primary_mask = inputs.primary_array == 0
        tertiary_labels[~primary_mask] = 0
        return tertiary_labels

    def stats(
        self,
        inputs: TertiaryObjectInputs,
        tertiary_labels: np.ndarray,
        *,
        slice_index: int = 0,
    ) -> TertiaryObjectStats:
        object_count, mean_area = TertiaryObjectMeasurement(
            tertiary_labels,
        ).positive_label_mean_area
        return TertiaryObjectStats(
            slice_index=slice_index,
            object_count=object_count,
            mean_area=float(mean_area),
            primary_parent_count=TertiaryObjectMeasurement(
                inputs.primary_array,
            ).positive_label_count,
            secondary_parent_count=TertiaryObjectMeasurement(
                inputs.secondary_array,
            ).positive_label_count,
        )


def _identify_tertiary_objects_batch(
    request: RuntimePure2DSliceBatchRequest,
) -> list[object]:
    kwargs = request.kwargs
    slice_count = request.slice_count
    alignment = DenseObjectLabelPairAligner(
        kwargs["primary_labels"],
        kwargs["secondary_labels"],
    ).aligned_stack_context(slice_count)
    if alignment is None:
        return [request.execute_one(slice_index) for slice_index in range(slice_count)]

    primary_stack = alignment.first_stack
    secondary_stack = alignment.second_stack
    tertiary_stack, object_counts, mean_areas, primary_counts, secondary_counts = (
        _tertiary_stack_numba(
            primary_stack,
            secondary_stack,
            bool(kwargs.get("shrink_primary", True)),
        )
    )
    output_tertiary_stack = alignment.restore_second_stack(tertiary_stack)

    return [
        (
            request.slices_2d[slice_index],
            object_label_parent_child_payload(
                secondary_stack[slice_index],
                tertiary_stack[slice_index],
            ),
            object_label_parent_child_payload(
                primary_stack[slice_index],
                tertiary_stack[slice_index],
                child_region_labels=secondary_stack[slice_index],
            ),
            TertiaryObjectStats(
                slice_index=slice_index,
                object_count=int(object_counts[slice_index]),
                mean_area=float(mean_areas[slice_index]),
                primary_parent_count=int(primary_counts[slice_index]),
                secondary_parent_count=int(secondary_counts[slice_index]),
            ),
            TertiaryObjectLabelOutput(
                kwargs["secondary_labels"],
                output_tertiary_stack[slice_index],
            ).value(),
        )
        for slice_index in range(slice_count)
    ]


@njit(cache=True)
def _tertiary_stack_numba(
    primary_stack: np.ndarray,
    secondary_stack: np.ndarray,
    shrink_primary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slice_count, height, width = secondary_stack.shape
    max_primary = 0
    max_secondary = 0
    for z in range(slice_count):
        for y in range(height):
            for x in range(width):
                primary_label = primary_stack[z, y, x]
                secondary_label = secondary_stack[z, y, x]
                if primary_label > max_primary:
                    max_primary = primary_label
                if secondary_label > max_secondary:
                    max_secondary = secondary_label

    tertiary_stack = np.zeros_like(secondary_stack)
    primary_present = np.zeros((slice_count, max_primary + 1), dtype=np.uint8)
    secondary_present = np.zeros((slice_count, max_secondary + 1), dtype=np.uint8)
    tertiary_present = np.zeros((slice_count, max_secondary + 1), dtype=np.uint8)
    tertiary_areas = np.zeros((slice_count, max_secondary + 1), dtype=np.int64)
    first_y = np.full((slice_count, max_secondary + 1), -1, dtype=np.int64)
    first_x = np.full((slice_count, max_secondary + 1), -1, dtype=np.int64)

    for z in range(slice_count):
        for y in range(height):
            for x in range(width):
                primary_label = primary_stack[z, y, x]
                secondary_label = secondary_stack[z, y, x]
                if primary_label > 0:
                    primary_present[z, primary_label] = 1
                if secondary_label > 0:
                    secondary_present[z, secondary_label] = 1
                    if first_y[z, secondary_label] < 0:
                        first_y[z, secondary_label] = y
                        first_x[z, secondary_label] = x

                keep_pixel = primary_label <= 0
                if shrink_primary and primary_label > 0:
                    for dy in range(-1, 2):
                        ny = y + dy
                        for dx in range(-1, 2):
                            nx = x + dx
                            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                                keep_pixel = True
                            elif primary_stack[z, ny, nx] != primary_label:
                                keep_pixel = True

                if keep_pixel and secondary_label > 0:
                    tertiary_stack[z, y, x] = secondary_label
                    tertiary_present[z, secondary_label] = 1
                    tertiary_areas[z, secondary_label] += 1

    for z in range(slice_count):
        for label in range(1, max_secondary + 1):
            if secondary_present[z, label] == 0 or tertiary_present[z, label] != 0:
                continue
            y = first_y[z, label]
            x = first_x[z, label]
            if y >= 0:
                tertiary_stack[z, y, x] = label
                tertiary_present[z, label] = 1
                tertiary_areas[z, label] += 1

    object_counts = np.zeros(slice_count, dtype=np.int64)
    mean_areas = np.zeros(slice_count, dtype=np.float64)
    primary_counts = np.zeros(slice_count, dtype=np.int64)
    secondary_counts = np.zeros(slice_count, dtype=np.int64)
    for z in range(slice_count):
        total_area = 0
        for label in range(1, max_primary + 1):
            if primary_present[z, label] != 0:
                primary_counts[z] += 1
        for label in range(1, max_secondary + 1):
            if secondary_present[z, label] != 0:
                secondary_counts[z] += 1
            if tertiary_present[z, label] != 0:
                object_counts[z] += 1
                total_area += tertiary_areas[z, label]
        if object_counts[z] > 0:
            mean_areas[z] = total_area / object_counts[z]
    return tertiary_stack, object_counts, mean_areas, primary_counts, secondary_counts


@numpy
def identify_tertiary_objects(
    image: np.ndarray,
    primary_labels: np.ndarray | ObjectLabelPayload,
    secondary_labels: np.ndarray | ObjectLabelPayload,
    shrink_primary: bool = True,
    outline_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[
    np.ndarray,
    ParentChildRelationshipPayload,
    ParentChildRelationshipPayload,
    TertiaryObjectStats,
    np.ndarray,
]:
    """Identify tertiary objects by subtracting primary labels from secondary labels."""
    inputs = TertiaryObjectInputs.from_labels(primary_labels, secondary_labels)
    segmentation = TertiaryObjectSegmentation()
    tertiary_labels = segmentation.segment(
        inputs,
        shrink_primary=shrink_primary,
        outline_backend_provider=outline_backend_provider,
    )

    tertiary_labels_out = (
        np.expand_dims(tertiary_labels, axis=0) if image.ndim == 3 else tertiary_labels
    )

    return (
        image,
        object_label_parent_child_payload(inputs.secondary_source, tertiary_labels),
        object_label_parent_child_payload(
            inputs.primary_source,
            tertiary_labels,
            child_region_labels=inputs.secondary_array,
        ),
        segmentation.stats(inputs, tertiary_labels),
        TertiaryObjectLabelOutput(inputs.secondary_source, tertiary_labels_out).value(),
    )


@processing_prepare(identify_tertiary_objects)
def _prepare_identify_tertiary_objects() -> None:
    """Compile tertiary-object kernels before benchmarked execution."""
    image = np.zeros((32, 32), dtype=np.float32)
    primary = np.zeros((32, 32), dtype=np.int32)
    secondary = np.zeros((32, 32), dtype=np.int32)
    primary[10:20, 10:20] = 1
    secondary[6:24, 6:24] = 1
    identify_tertiary_objects.__wrapped__(
        image,
        primary,
        secondary,
        shrink_primary=True,
    )
    _tertiary_stack_numba(
        np.expand_dims(primary, axis=0),
        np.expand_dims(secondary, axis=0),
        True,
    )


pure_2d_batch_executor(_identify_tertiary_objects_batch)(identify_tertiary_objects)


__all__ = [
    "CentrosomeSecondaryPropagationBackendStrategy",
    "DistanceMaskedSegmentationStrategy",
    "DistanceOnlySegmentationStrategy",
    "GradientWatershedSegmentationStrategy",
    "ImageWatershedSegmentationStrategy",
    "LabelPropagationResult",
    "NumbaSecondaryDistanceTransformBackendStrategy",
    "NumbaSecondaryPropagationBackendStrategy",
    "NumpySecondaryDistanceTransformBackendStrategy",
    "PropagationSegmentationStrategy",
    "SecondaryDistanceTransformBackendStrategy",
    "SecondaryImageInputs",
    "SecondaryMethod",
    "SecondaryObjectLabels",
    "SecondaryObjectStats",
    "SecondaryPropagationBackendStrategy",
    "SecondarySegmentationRequest",
    "SecondarySegmentationStrategy",
    "SecondaryThresholdRequest",
    "SecondaryThresholdResult",
    "ThresholdCalculator",
    "ThresholdCalculatorDeclaration",
    "ThresholdMethod",
    "TertiaryObjectInputs",
    "TertiaryObjectLabelOutput",
    "TertiaryObjectMeasurement",
    "TertiaryObjectSegmentation",
    "TertiaryObjectStats",
    "THRESHOLD_CALCULATOR_DECLARATIONS",
    "identify_secondary_objects",
    "identify_tertiary_objects",
    "secondary_propagation_backend",
]
