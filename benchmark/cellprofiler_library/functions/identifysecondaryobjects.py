"""
Converted from CellProfiler: IdentifySecondaryObjects
Original: IdentifySecondaryObjects.run

Identifies secondary objects (e.g., cells) using primary objects (e.g., nuclei)
as seeds, expanding them based on intensity gradients or distance.
"""

import logging
import os
import time

import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple
from dataclasses import dataclass
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory import numpy
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    object_label_payload_from_source_image,
)
from openhcs.core.runtime_semantics import (
    ParentChildRelationshipPayload,
    object_label_parent_child_payload,
)
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from benchmark.cellprofiler_library.functions.thresholding import (
    CellProfilerAveragingMethod,
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
    CellProfilerThresholdMethod,
    CellProfilerThresholdScope,
    CellProfilerVarianceMethod,
    cellprofiler_threshold,
    cellprofiler_threshold_diagnostics,
    normalize_cellprofiler_image,
    unit_interval_scale_for_threshold_diagnostics,
)
from benchmark.cellprofiler_library.functions.watershed import (
    cellprofiler_legacy_watershed,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    CellProfilerPlaneGeometry,
    collapse_singleton_plane_stack,
)
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.morphology import MorphologyBackendStrategy
from openhcs.processing.backends.cellprofiler.secondary import (
    SecondaryDistanceTransformBackendStrategy,
    SecondaryPropagationBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.thresholding import threshold_primitives

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


class SecondaryMethod(Enum):
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


def _propagate_labels(
    image: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    regularization: float,
    backend_provider: CellProfilerBackendProvider | None = None,
    max_distance: float | None = None,
) -> np.ndarray:
    """Propagate labels using the configured explicit backend provider."""
    geometry = CellProfilerPlaneGeometry.from_image_plane(image)
    labels = geometry.label_plane(labels)
    mask = geometry.binary_mask(mask)
    return SecondaryPropagationBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    ).propagate(
        image,
        labels,
        mask,
        regularization,
        max_distance=max_distance,
    )
    return result


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


def _parent_child_relationship(
    parent_labels: np.ndarray | ObjectLabelPayload,
    child_labels: np.ndarray,
) -> ParentChildRelationshipPayload:
    return object_label_parent_child_payload(parent_labels, child_labels)


@dataclass(frozen=True)
class SecondarySegmentationRequest:
    image: np.ndarray
    labels: np.ndarray
    unedited_labels: np.ndarray
    thresholded: np.ndarray
    distance_to_dilate: int
    regularization_factor: float
    watershed_backend_provider: CellProfilerBackendProvider | None
    distance_backend_provider: CellProfilerBackendProvider | None = None
    propagation_backend_provider: CellProfilerBackendProvider | None = None

    @property
    def has_primary_objects(self) -> bool:
        return self.unedited_labels.max() > 0

    @property
    def object_mask(self) -> np.ndarray:
        return self.thresholded | (self.unedited_labels > 0)


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
    method_label: ClassVar[str | None] = None
    method: ClassVar[ThresholdMethod | None] = None

    @classmethod
    def for_method(cls, method: ThresholdMethod) -> "ThresholdCalculator":
        return cls.__registry__[method.value]()

    @abstractmethod
    def calculate(self, image: np.ndarray) -> float:
        """Calculate a threshold value for a normalized intensity image."""


class OtsuThresholdCalculator(ThresholdCalculator):
    method = ThresholdMethod.OTSU
    method_label = method.value

    def calculate(self, image: np.ndarray) -> float:
        return threshold_primitives().otsu_threshold(image)


class LiThresholdCalculator(ThresholdCalculator):
    method = ThresholdMethod.LI
    method_label = method.value

    def calculate(self, image: np.ndarray) -> float:
        return threshold_primitives().li_threshold(image)


class MinimumThresholdCalculator(ThresholdCalculator):
    method = ThresholdMethod.MINIMUM
    method_label = method.value

    def calculate(self, image: np.ndarray) -> float:
        return threshold_primitives().minimum_threshold(image)


class TriangleThresholdCalculator(ThresholdCalculator):
    method = ThresholdMethod.TRIANGLE
    method_label = method.value

    def calculate(self, image: np.ndarray) -> float:
        return threshold_primitives().triangle_threshold(image)


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
        labels_out = (
            _propagate_labels(
                request.image,
                request.unedited_labels,
                request.thresholded,
                1.0,
                max_distance=float(request.distance_to_dilate),
            )
            if request.propagation_backend_provider is None
            else _propagate_labels(
                request.image,
                request.unedited_labels,
                request.thresholded,
                1.0,
                backend_provider=request.propagation_backend_provider,
                max_distance=float(request.distance_to_dilate),
            )
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
        return (
            _propagate_labels(
                request.image,
                request.unedited_labels,
                request.thresholded,
                request.regularization_factor,
            )
            if request.propagation_backend_provider is None
            else _propagate_labels(
                request.image,
                request.unedited_labels,
                request.thresholded,
                request.regularization_factor,
                backend_provider=request.propagation_backend_provider,
            )
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
        return _watershed_secondary_labels(request, sobel_image)


class ImageWatershedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.WATERSHED_IMAGE
    method_label = method.value

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        return _watershed_secondary_labels(request, 1.0 - request.image)


def _watershed_secondary_labels(
    request: SecondarySegmentationRequest,
    watershed_image: np.ndarray,
) -> np.ndarray:
    return cellprofiler_legacy_watershed(
        watershed_image,
        markers=request.unedited_labels,
        mask=request.object_mask,
        connectivity=np.ones((3, 3), bool),
        backend_provider=request.watershed_backend_provider,
    )


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
        return _coerce_function_enum(CellProfilerThresholdMethod, threshold_method)
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
    watershed_backend_provider: CellProfilerBackendProvider | None = None,
    morphology_backend_provider: CellProfilerBackendProvider | None = None,
    distance_backend_provider: CellProfilerBackendProvider | None = None,
    propagation_backend_provider: CellProfilerBackendProvider | None = None,
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
    method = _coerce_function_enum(SecondaryMethod, method)
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
    _log_profile(
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
    _log_profile(
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
    _log_profile(
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
    _log_profile(
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
    _log_profile(
        "iso_stats",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    relationships = _parent_child_relationship(
        primary_labels if isinstance(primary_labels, ObjectLabelPayload) else inputs.labels,
        object_labels.segmented,
    )
    _log_profile(
        "iso_relationships",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    _log_profile(
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
