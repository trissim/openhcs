"""
Converted from CellProfiler: ExpandOrShrinkObjects
Original: expand_or_shrink_objects
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np
from numba import njit
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_semantics import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    dense_object_label_max_present_id,
)
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    object_label_dense_array,
    object_label_payload_with_dense_labels,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import segmentation_mask_rois
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)


class ExpandShrinkMode(Enum):
    EXPAND_DEFINED_PIXELS = "expand_defined_pixels"
    EXPAND_INFINITE = "expand_infinite"
    SHRINK_DEFINED_PIXELS = "shrink_defined_pixels"
    SHRINK_TO_POINT = "shrink_to_point"
    ADD_DIVIDING_LINES = "add_dividing_lines"
    DESPUR = "despur"
    SKELETONIZE = "skeletonize"


class CellProfilerExpandShrinkOperation(str, Enum):
    """Closed CellProfiler UI operation dialect for ExpandOrShrinkObjects."""

    SHRINK_TO_POINT = "Shrink objects to a point"
    EXPAND_UNTIL_TOUCHING = "Expand objects until touching"
    ADD_DIVIDING_LINES = "Add partial dividing lines between objects"
    SHRINK_DEFINED_PIXELS = "Shrink objects by a specified number of pixels"
    SHRINK_BY_MEASUREMENT = "Shrink objects by a previous measurement"
    EXPAND_DEFINED_PIXELS = "Expand objects by a specified number of pixels"
    EXPAND_BY_MEASUREMENT = "Expand objects by a previous measurement"
    SKELETONIZE = "Skeletonize each object"
    DESPUR = "Remove spurs"


class ExpandShrinkOperationStrategy(
    EnumKeyedStrategyMixin[ExpandShrinkMode],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal operation strategy for one ExpandOrShrinkObjects mode."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "mode"
    mode: ClassVar[ExpandShrinkMode | None] = None
    strategy_label: ClassVar[str | None] = None
    cellprofiler_operations: ClassVar[tuple[CellProfilerExpandShrinkOperation, ...]] = ()

    @classmethod
    def for_mode(
        cls,
        mode: ExpandShrinkMode | str,
    ) -> "ExpandShrinkOperationStrategy":
        resolved = _coerce_function_enum(ExpandShrinkMode, mode)
        return cls.for_enum_member(resolved)

    @abstractmethod
    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        """Return transformed labels for this operation mode."""

    def output_domain(self, labels: np.ndarray) -> ObjectLabelDomain:
        """Return CP's semantic object domain for transformed labels."""
        return ObjectLabelDomain(
            declared_object_count=dense_object_label_max_present_id(labels),
            scope=ObjectLabelDomainScope.PLANE,
        )


class ExpandDefinedPixelsStrategy(ExpandShrinkOperationStrategy):
    """Expand labeled objects by a fixed pixel radius."""

    mode = ExpandShrinkMode.EXPAND_DEFINED_PIXELS
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.EXPAND_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.EXPAND_BY_MEASUREMENT,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _expand_defined_pixels(labels, iterations)


class ExpandInfiniteStrategy(ExpandShrinkOperationStrategy):
    """Expand labeled objects until all background is assigned."""

    mode = ExpandShrinkMode.EXPAND_INFINITE
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.EXPAND_UNTIL_TOUCHING,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _expand_until_touching(labels)


class ShrinkDefinedPixelsStrategy(ExpandShrinkOperationStrategy):
    """Shrink labeled objects by a fixed pixel radius."""

    mode = ExpandShrinkMode.SHRINK_DEFINED_PIXELS
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.SHRINK_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.SHRINK_BY_MEASUREMENT,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _shrink_defined_pixels(labels, iterations, fill_holes)


class ShrinkToPointStrategy(ExpandShrinkOperationStrategy):
    """Shrink each object to its center point."""

    mode = ExpandShrinkMode.SHRINK_TO_POINT
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.SHRINK_TO_POINT,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _shrink_to_point(labels, fill_holes)


class AddDividingLinesStrategy(ExpandShrinkOperationStrategy):
    """Remove touching object boundary pixels."""

    mode = ExpandShrinkMode.ADD_DIVIDING_LINES
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.ADD_DIVIDING_LINES,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _add_dividing_lines(labels)


class DespurStrategy(ExpandShrinkOperationStrategy):
    """Remove object spurs by repeated opening."""

    mode = ExpandShrinkMode.DESPUR
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.DESPUR,)

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _despur(labels, iterations)


class SkeletonizeStrategy(ExpandShrinkOperationStrategy):
    """Reduce each object to a skeleton."""

    mode = ExpandShrinkMode.SKELETONIZE
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.SKELETONIZE,)

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _skeletonize_labels(labels)


def _expand_defined_pixels(labels: np.ndarray, iterations: int) -> np.ndarray:
    """Expand labeled objects by a defined number of pixels."""
    from scipy.ndimage import distance_transform_edt
    
    if iterations <= 0:
        return labels.copy()
    labels_int = labels.astype(np.int32, copy=False)
    if labels_int.ndim > 2:
        return _apply_label_planes(
            labels_int,
            lambda plane: _expand_defined_pixels(plane, iterations),
        )
    if _labels_are_points_numba(np.ascontiguousarray(labels_int)):
        return _expand_point_labels_defined_pixels_numba(
            np.ascontiguousarray(labels_int),
            int(iterations),
        )

    result = labels_int.copy()
    background = labels_int == 0
    distances, indices = distance_transform_edt(background, return_indices=True)
    expand_mask = background & (distances <= iterations)
    result[expand_mask] = labels_int[indices[0][expand_mask], indices[1][expand_mask]]
    return result


@njit(cache=True)
def _labels_are_points_numba(labels: np.ndarray) -> bool:
    max_label = 0
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label
    if max_label <= 0:
        return True

    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1
            if counts[label] > 1:
                return False
    return True


@njit(cache=True)
def _expand_point_labels_defined_pixels_numba(
    labels: np.ndarray,
    radius: int,
) -> np.ndarray:
    height, width = labels.shape
    output = labels.copy()
    radius_squared = radius * radius
    initial_distance = radius_squared + 1
    best_distance = np.full(labels.shape, initial_distance, dtype=np.int32)
    best_y = np.full(labels.shape, 2147483647, dtype=np.int32)
    best_x = np.full(labels.shape, 2147483647, dtype=np.int32)

    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            for dy in range(-radius, radius + 1):
                yy = y + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-radius, radius + 1):
                    xx = x + dx
                    if xx < 0 or xx >= width:
                        continue
                    distance = dy * dy + dx * dx
                    if distance > radius_squared:
                        continue
                    if (
                        distance < best_distance[yy, xx]
                        or (
                            distance == best_distance[yy, xx]
                            and (
                                x < best_x[yy, xx]
                                or (x == best_x[yy, xx] and y < best_y[yy, xx])
                            )
                        )
                    ):
                        best_distance[yy, xx] = distance
                        best_y[yy, xx] = y
                        best_x[yy, xx] = x
                        output[yy, xx] = label
    return output


def _expand_until_touching(labels: np.ndarray) -> np.ndarray:
    """Expand labeled objects until they touch (Voronoi-like expansion)."""
    from scipy.ndimage import distance_transform_edt

    if labels.ndim > 2:
        return _apply_label_planes(labels, _expand_until_touching)
    
    if labels.max() == 0:
        return labels.copy()
    
    # Use distance transform to find nearest labeled pixel for each background pixel
    mask = labels > 0
    distances, indices = distance_transform_edt(~mask, return_indices=True)
    
    # Assign each pixel to its nearest labeled object
    result = labels[indices[0], indices[1]]
    
    return result


def _shrink_defined_pixels(labels: np.ndarray, iterations: int, fill: bool) -> np.ndarray:
    """Shrink labeled objects by a defined number of pixels."""
    if iterations <= 0:
        return labels.copy()

    original = labels.astype(np.int32, copy=False)
    if original.ndim > 2:
        return _apply_label_planes(
            original,
            lambda plane: _shrink_defined_pixels(plane, iterations, fill),
        )
    result = original.copy()
    for _ in range(iterations):
        same_neighbors = np.zeros(result.shape, dtype=bool)
        center = result[1:-1, 1:-1]
        same_neighbors[1:-1, 1:-1] = (
            (center > 0)
            & (center == result[:-2, 1:-1])
            & (center == result[2:, 1:-1])
            & (center == result[1:-1, :-2])
            & (center == result[1:-1, 2:])
        )
        result = np.where(same_neighbors, result, 0).astype(np.int32, copy=False)

    if fill:
        _restore_eroded_objects_to_centroids(original, result)

    return result


def _restore_eroded_objects_to_centroids(
    original: np.ndarray,
    eroded: np.ndarray,
) -> None:
    """Preserve one centroid pixel for labels fully removed by shrinking."""
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        original.astype(np.int32, copy=False)
    )
    if region_props.label.size == 0:
        return
    remaining_ids = set(int(label_id) for label_id in np.unique(eroded) if label_id > 0)
    for index, label_id in enumerate(region_props.label):
        label_int = int(label_id)
        if label_int in remaining_ids:
            continue
        cy = int(region_props.centroid_y[index])
        cx = int(region_props.centroid_x[index])
        eroded[cy, cx] = label_int


def _shrink_to_point(labels: np.ndarray, fill: bool) -> np.ndarray:
    """Shrink each labeled object to a single point at its centroid."""
    labels_int = labels.astype(np.int32, copy=False)
    if labels_int.ndim > 2:
        return _apply_label_planes(
            labels_int,
            lambda plane: _shrink_to_point(plane, fill),
        )
    if labels_int.size == 0 or int(labels_int.max()) <= 0:
        return np.zeros_like(labels_int)
    return _shrink_to_point_numba(np.ascontiguousarray(labels_int))


@njit(cache=True)
def _shrink_to_point_numba(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label

    y_sums = np.zeros(max_label + 1, dtype=np.float64)
    x_sums = np.zeros(max_label + 1, dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            y_sums[label] += float(y)
            x_sums[label] += float(x)
            counts[label] += 1

    result = np.zeros(labels.shape, dtype=np.int32)
    for label in range(1, max_label + 1):
        count = counts[label]
        if count <= 0:
            continue
        cy = int(y_sums[label] / float(count))
        cx = int(x_sums[label] / float(count))
        if cy < 0:
            cy = 0
        elif cy >= height:
            cy = height - 1
        if cx < 0:
            cx = 0
        elif cx >= width:
            cx = width - 1
        result[cy, cx] = label
    return result


def _add_dividing_lines(labels: np.ndarray) -> np.ndarray:
    """Add 1-pixel dividing lines between touching objects."""
    from scipy.ndimage import maximum_filter, minimum_filter

    if labels.ndim > 2:
        return _apply_label_planes(labels, _add_dividing_lines)
    
    if labels.max() == 0:
        return labels.copy()
    
    result = labels.copy()
    
    # Find pixels where neighboring labels differ (boundaries)
    max_filt = maximum_filter(labels, size=3)
    min_filt = minimum_filter(labels, size=3)
    
    # Boundary pixels are where max != min and both are > 0
    boundary = (max_filt != min_filt) & (min_filt > 0)
    
    result[boundary] = 0
    
    return result


def _despur(labels: np.ndarray, iterations: int) -> np.ndarray:
    """Remove spurs (small protrusions) from labeled objects."""
    from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
    
    if iterations <= 0:
        return labels.copy()
    if labels.ndim > 2:
        return _apply_label_planes(
            labels,
            lambda plane: _despur(plane, iterations),
        )
    
    result = np.zeros_like(labels)
    struct = generate_binary_structure(2, 1)
    
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        # Opening operation removes small protrusions
        opened = binary_erosion(obj_mask, structure=struct, iterations=iterations)
        opened = binary_dilation(opened, structure=struct, iterations=iterations)
        result[opened] = label_id
    
    return result


def _skeletonize_labels(labels: np.ndarray) -> np.ndarray:
    """Reduce labeled objects to their skeletons."""
    from skimage.morphology import skeletonize

    if labels.ndim > 2:
        return _apply_label_planes(labels, _skeletonize_labels)
    
    result = np.zeros_like(labels)
    
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        skeleton = skeletonize(obj_mask)
        result[skeleton] = label_id
    
    return result


def _apply_label_planes(
    labels: np.ndarray,
    operation,
) -> np.ndarray:
    output = np.empty_like(labels, dtype=np.int32)
    label_planes = labels.reshape((-1, *labels.shape[-2:]))
    output_planes = output.reshape((-1, *output.shape[-2:]))
    for plane_index in range(label_planes.shape[0]):
        output_planes[plane_index] = operation(label_planes[plane_index])
    return output


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("labels", segmentation_mask_rois()))
def expand_or_shrink_objects(
    image: np.ndarray,
    labels: np.ndarray | ObjectLabelPayload,
    mode: ExpandShrinkMode | str = ExpandShrinkMode.EXPAND_DEFINED_PIXELS,
    iterations: int = 1,
    fill_holes: bool = True,
) -> tuple:
    """
    Expand or shrink labeled objects using various methods.
    
    Args:
        image: Input image (H, W) - passed through unchanged
        labels: Label image (H, W) - integer labels for each object
        mode: Operation mode - expand, shrink, skeletonize, etc.
        iterations: Number of pixels to expand/shrink (for applicable modes)
        fill_holes: Whether to preserve objects that would disappear (for shrink modes)
    
    Returns:
        Tuple of (image, modified_labels)
    """
    labels_int = object_label_dense_array(labels, dtype=np.int32)

    operation = ExpandShrinkOperationStrategy.for_mode(mode)
    result_labels = operation.apply(
        labels_int,
        iterations=iterations,
        fill_holes=fill_holes,
    )
    output_domain = operation.output_domain(result_labels)

    return image, object_label_payload_with_dense_labels(
        labels,
        result_labels.astype(np.int32, copy=False),
        declared_object_count=output_domain.declared_object_count,
        declared_object_ids=output_domain.declared_object_ids,
        domain_scope=output_domain.scope,
    )


def _prepare_expand_or_shrink_objects() -> None:
    """Compile Numba kernels used by common object expansion/shrink modes."""
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:5, 3:7] = 1
    labels[8:12, 9:14] = 2
    points = _shrink_to_point(labels, False)
    _expand_defined_pixels(points, 2)


expand_or_shrink_objects.__openhcs_prepare__ = _prepare_expand_or_shrink_objects
