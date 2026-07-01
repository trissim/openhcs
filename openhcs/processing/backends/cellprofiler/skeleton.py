"""CellProfiler-compatible skeleton measurement backends."""

from __future__ import annotations
from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule

from dataclasses import dataclass

import numpy as np
import scipy.ndimage
from skimage.morphology import remove_small_holes, skeletonize

from openhcs.core.memory.decorators import numpy as numpy_backend
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer

SKELETON_MEASUREMENT_FIELDS = ["slice_index", "branches", "endpoints"]
OBJECT_SKELETON_MEASUREMENT_FIELDS = [
    "slice_index",
    "object_label",
    "number_trunks",
    "number_non_trunk_branches",
    "number_branch_ends",
    "total_skeleton_length",
]
EIGHT_NEIGHBOR_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    dtype=np.uint8,
)


@dataclass(frozen=True, slots=True)
class SkeletonMeasurement:
    """Measurements from skeleton analysis."""

    slice_index: int
    branches: int
    endpoints: int


@dataclass(frozen=True, slots=True)
class ObjectSkeletonMeasurement:
    """Measurements for skeleton branching structures per seed object."""

    slice_index: int
    object_label: int
    number_trunks: int
    number_non_trunk_branches: int
    number_branch_ends: int
    total_skeleton_length: float


@dataclass(frozen=True, slots=True)
class SkeletonNeighborhood:
    """Neighbor-count semantics for 2-D and 3-D skeleton measurements."""

    image: np.ndarray

    @property
    def binary(self) -> np.ndarray:
        return (self.image > 0).astype(np.uint8)

    def neighbor_counts(self) -> np.ndarray:
        binary = self.binary
        padding = np.pad(binary, 1, mode="constant", constant_values=0)
        mask = padding > 0
        response = (3**binary.ndim) * scipy.ndimage.uniform_filter(
            padding.astype(np.float64),
            size=3,
        ) - 1
        interior = tuple(slice(1, -1) for _ in range(binary.ndim))
        return (response * mask)[interior].astype(np.uint16)

    def measurement(self, *, slice_index: int = 0) -> SkeletonMeasurement:
        neighbors = self.neighbor_counts()
        return SkeletonMeasurement(
            slice_index=slice_index,
            branches=int(np.count_nonzero(neighbors > 2)),
            endpoints=int(np.count_nonzero(neighbors == 1)),
        )


@dataclass(frozen=True, slots=True)
class DiskStructuringElement:
    """Disk footprint used by CellProfiler object-skeleton measurements."""

    radius: float

    def footprint(self) -> np.ndarray:
        radius = int(self.radius + 0.5)
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        return (x * x + y * y <= self.radius * self.radius).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class SkeletonLabelPropagation:
    """Propagate seed labels onto the skeleton support."""

    labels: np.ndarray
    mask: np.ndarray

    def propagate(self) -> tuple[np.ndarray, np.ndarray]:
        distance = scipy.ndimage.distance_transform_edt(self.labels == 0)
        propagated = self.labels.copy()
        max_distance = int(np.max(distance[self.mask])) + 1 if np.any(self.mask) else 0
        for _ in range(max_distance):
            dilated = scipy.ndimage.grey_dilation(propagated, size=3)
            propagated = np.where((propagated == 0) & self.mask, dilated, propagated)
        return propagated, distance


@dataclass(frozen=True, slots=True)
class ObjectSkeletonSliceMeasurement:
    """Seed-relative skeleton measurements for one 2-D plane."""

    skeleton: np.ndarray
    seed_labels: np.ndarray
    slice_index: int
    fill_small_holes: bool
    maximum_hole_size: int

    def measurements(self) -> list[ObjectSkeletonMeasurement]:
        labels = self.seed_labels.astype(np.int32)
        label_count = int(np.max(labels))
        if label_count == 0:
            return []

        label_range = np.arange(1, label_count + 1, dtype=np.int32)
        disk = DiskStructuringElement(1.5).footprint()
        dilated_labels = scipy.ndimage.grey_dilation(labels, footprint=disk)
        seed_mask = dilated_labels > 0
        combined_skeleton = (self.skeleton > 0) | seed_mask
        closed_labels = scipy.ndimage.grey_erosion(dilated_labels, footprint=disk)
        combined_skeleton = combined_skeleton & ~(closed_labels > 0)
        if self.fill_small_holes:
            combined_skeleton = remove_small_holes(
                combined_skeleton,
                area_threshold=self.maximum_hole_size,
            )
        combined_skeleton = skeletonize(combined_skeleton)
        outside_skeleton = combined_skeleton & (dilated_labels == 0)

        propagated_labels, distance_map = SkeletonLabelPropagation(
            labels=dilated_labels,
            mask=combined_skeleton,
        ).propagate()
        combined_skeleton = combined_skeleton & (propagated_labels > 0)
        branch_points = SkeletonConvolutionFeatures(combined_skeleton).branchpoints()
        end_points = SkeletonConvolutionFeatures(combined_skeleton).endpoints()
        branching_counts = SkeletonConvolutionFeatures(combined_skeleton).branching_counts()
        dilated_skeleton = scipy.ndimage.binary_dilation(
            outside_skeleton,
            structure=np.ones((3, 3)),
        )
        branching_counts[~dilated_skeleton] = 0

        nearby_labels = propagated_labels.copy()
        nearby_labels[distance_map > 1.5] = 0
        outside_labels = propagated_labels.copy()
        outside_labels[nearby_labels > 0] = 0

        trunk_counts = np.array(
            [int(np.sum(branching_counts[nearby_labels == label])) for label in label_range],
            dtype=np.int32,
        )
        branch_counts = np.array(
            [int(np.sum(branch_points[outside_labels == label])) for label in label_range],
            dtype=np.int32,
        )
        end_counts = np.array(
            [int(np.sum(end_points[outside_labels == label])) for label in label_range],
            dtype=np.int32,
        )
        total_distance = SkeletonLengthByLabel(
            labels=propagated_labels * outside_skeleton.astype(np.int32),
            label_range=label_range,
        ).lengths()

        return [
            ObjectSkeletonMeasurement(
                slice_index=self.slice_index,
                object_label=int(label),
                number_trunks=int(trunk_counts[index]),
                number_non_trunk_branches=int(branch_counts[index]),
                number_branch_ends=int(end_counts[index]),
                total_skeleton_length=(
                    float(total_distance[index])
                    if index < len(total_distance)
                    else 0.0
                ),
            )
            for index, label in enumerate(label_range)
        ]


@dataclass(frozen=True, slots=True)
class SkeletonConvolutionFeatures:
    """2-D skeleton branch and endpoint features from CP neighbor semantics."""

    skeleton: np.ndarray

    def neighbor_counts(self) -> np.ndarray:
        return scipy.ndimage.convolve(
            self.skeleton.astype(np.uint8),
            EIGHT_NEIGHBOR_KERNEL,
            mode="constant",
            cval=0,
        )

    def branchpoints(self) -> np.ndarray:
        return (self.skeleton > 0) & (self.neighbor_counts() > 2)

    def endpoints(self) -> np.ndarray:
        return (self.skeleton > 0) & (self.neighbor_counts() == 1)

    def branching_counts(self) -> np.ndarray:
        counts = np.clip(self.neighbor_counts() - 2, 0, 2)
        counts[~self.skeleton] = 0
        return counts


@dataclass(frozen=True, slots=True)
class SkeletonLengthByLabel:
    """Skeleton length aggregation over propagated seed labels."""

    labels: np.ndarray
    label_range: np.ndarray

    def lengths(self) -> np.ndarray:
        if len(self.label_range) == 0:
            return np.zeros(0)
        lengths = scipy.ndimage.sum(
            self.labels > 0,
            self.labels,
            self.label_range,
        )
        return np.atleast_1d(lengths).astype(float)


@numpy_backend(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "skeleton_measurements",
        csv_materializer(
            fields=SKELETON_MEASUREMENT_FIELDS,
            analysis_type="skeleton_measurement",
        ),
    )
)
def measure_image_skeleton(image: np.ndarray) -> tuple[np.ndarray, SkeletonMeasurement]:
    """Measure branches and endpoints in a 2-D skeletonized image."""
    return image, SkeletonNeighborhood(image).measurement()


@numpy_backend(contract=ProcessingContract.PURE_3D)
@special_outputs(
    (
        "skeleton_measurements_3d",
        csv_materializer(
            fields=SKELETON_MEASUREMENT_FIELDS,
            analysis_type="skeleton_measurement_3d",
        ),
    )
)
def measure_image_skeleton_3d(image: np.ndarray) -> tuple[np.ndarray, SkeletonMeasurement]:
    """Measure branches and endpoints in a 3-D skeletonized image."""
    return image, SkeletonNeighborhood(image).measurement()


@numpy_backend
@special_inputs("seed_labels")
@special_outputs(
    (
        "skeleton_measurements",
        csv_materializer(
            fields=OBJECT_SKELETON_MEASUREMENT_FIELDS,
            analysis_type="object_skeleton",
        ),
    )
)
def measure_object_skeleton(
    image: np.ndarray,
    seed_labels: np.ndarray,
    fill_small_holes: bool = True,
    maximum_hole_size: int = 10,
) -> tuple[np.ndarray, list[ObjectSkeletonMeasurement]]:
    """Measure branching structures in skeletonized images relative to seed objects."""
    image_stack = image[np.newaxis, :, :] if image.ndim == 2 else image
    label_stack = object_label_dense_array(seed_labels, dtype=np.int32)
    if label_stack.ndim == 2:
        label_stack = label_stack[np.newaxis, :, :]

    measurements: list[ObjectSkeletonMeasurement] = []
    for slice_index in range(image_stack.shape[0]):
        labels_slice = (
            label_stack[slice_index]
            if slice_index < label_stack.shape[0]
            else label_stack[0]
        )
        measurements.extend(
            ObjectSkeletonSliceMeasurement(
                skeleton=image_stack[slice_index],
                seed_labels=labels_slice,
                slice_index=slice_index,
                fill_small_holes=fill_small_holes,
                maximum_hole_size=maximum_hole_size,
            ).measurements()
        )
    return image, measurements


class MeasureImageSkeletonModule(CellProfilerModule):
    module_name = 'MeasureImageSkeleton'
    function_name = 'measure_image_skeleton'
    validated = True
    contract = 'unknown'
    confidence = 1.0

class MeasureObjectSkeletonModule(CellProfilerModule):
    module_name = 'MeasureObjectSkeleton'
    function_name = 'measure_object_skeleton'
    validated = True
    confidence = 1.0

__all__ = public_names_from_objects(
    DiskStructuringElement,
    "EIGHT_NEIGHBOR_KERNEL",
    "OBJECT_SKELETON_MEASUREMENT_FIELDS",
    ObjectSkeletonMeasurement,
    ObjectSkeletonSliceMeasurement,
    "SKELETON_MEASUREMENT_FIELDS",
    SkeletonConvolutionFeatures,
    SkeletonLabelPropagation,
    SkeletonLengthByLabel,
    SkeletonMeasurement,
    SkeletonNeighborhood,
    measure_image_skeleton,
    measure_image_skeleton_3d,
    measure_object_skeleton,
)
