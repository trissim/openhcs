"""
Converted from CellProfiler: IdentifyPrimaryObjects
Original: IdentifyPrimaryObjects.run

Identifies primary objects (e.g., nuclei) in grayscale images using
thresholding, declumping, and watershed segmentation.
"""

import logging
import numpy as np
import os
import time
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple
from dataclasses import dataclass
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from numba import njit, prange
from openhcs.core.memory import numpy
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ObjectLabelPayload,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
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
)
from benchmark.cellprofiler_library.functions.watershed import (
    cellprofiler_legacy_watershed,
)
from openhcs.processing.backends.cellprofiler.morphology import CellProfilerDeclumpMethod
from openhcs.processing.backends.cellprofiler.shape import shape_measurement_backend
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider

CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE = 7.0
_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


class UnclumpMethod(Enum):
    INTENSITY = "intensity"
    SHAPE = "shape"
    NONE = "none"


class WatershedMethod(Enum):
    INTENSITY = "intensity"
    SHAPE = "shape"
    PROPAGATE = "propagate"
    NONE = "none"


class FillHolesOption(Enum):
    NEVER = ("never", False, False)
    AFTER_BOTH = ("after_both", True, True)
    AFTER_DECLUMP = ("after_declump", False, True)

    def __new__(
        cls,
        value: str,
        fill_before_declump: bool,
        fill_after_declump: bool,
    ):
        option = object.__new__(cls)
        option._value_ = value
        option.fill_before_declump = fill_before_declump
        option.fill_after_declump = fill_after_declump
        return option


class ExcessObjectHandling(Enum):
    CONTINUE = ("Continue", False)
    ERASE = ("Erase", True)

    def __new__(cls, value: str, erase_excess: bool):
        option = object.__new__(cls)
        option._value_ = value
        option.erase_excess = erase_excess
        return option


def _fill_before_declump_requested(
    *,
    use_advanced_settings: bool,
    fill_holes: FillHolesOption,
) -> bool:
    """Return whether CP fills binary foreground holes before declumping."""
    return (not use_advanced_settings) or fill_holes.fill_before_declump


def _fill_after_declump_requested(
    *,
    use_advanced_settings: bool,
    fill_holes: FillHolesOption,
) -> bool:
    """Return whether CP fills labeled-object holes after declumping/filtering."""
    return (not use_advanced_settings) or fill_holes.fill_after_declump


def _declumping_maxima_geometry(
    *,
    min_diameter: int,
    low_res_maxima: bool,
    automatic_suppression: bool,
    maxima_suppression_size: float,
    declump_method: "CellProfilerDeclumpMethod",
    median_initial_object_radius: float | None = None,
) -> tuple[float, float]:
    """Return ``(image_resize_factor, suppress_size)`` for declumping maxima."""
    if min_diameter > 10 and low_res_maxima:
        image_resize_factor = 10.0 / float(min_diameter)
        if automatic_suppression:
            return image_resize_factor, 7.0
        return image_resize_factor, maxima_suppression_size * image_resize_factor + 0.5

    if automatic_suppression:
        return 1.0, float(min_diameter) / 1.5
    return 1.0, _manual_declumping_size(maxima_suppression_size)


def _manual_declumping_size(size: float) -> float:
    """Return the configured manual CP declumping size."""
    size = float(size)
    if size <= 0:
        return 0.0
    return size


class WatershedImageBuilder(ABC, metaclass=AutoRegisterMeta):
    """Build the watershed surface for one closed watershed method."""

    __registry_key__ = "method_label"
    method_label: ClassVar[str | None] = None
    method: ClassVar[WatershedMethod | None] = None

    @classmethod
    def for_method(cls, method: WatershedMethod) -> "WatershedImageBuilder":
        return cls.__registry__[method.value]()

    @abstractmethod
    def build(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        """Return the image used as the watershed surface."""


class IntensityWatershedImageBuilder(WatershedImageBuilder):
    method = WatershedMethod.INTENSITY
    method_label = method.value

    def build(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        return 1 - image


class ShapeWatershedImageBuilder(WatershedImageBuilder):
    method = WatershedMethod.SHAPE
    method_label = method.value

    def build(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        from scipy import ndimage as ndi

        return -ndi.distance_transform_edt(binary)


class PropagateWatershedImageBuilder(WatershedImageBuilder):
    method = WatershedMethod.PROPAGATE
    method_label = method.value

    def build(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        return 1 - image


@dataclass
class PrimaryObjectStats:
    slice_index: int
    object_count: int
    mean_area: float
    median_area: float
    total_area: float
    threshold_used: float
    original_threshold: float = 0.0
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0


@numpy
def identify_primary_objects(
    image: np.ndarray,
    min_diameter: int = 10,
    max_diameter: int = 40,
    exclude_size: bool = True,
    exclude_border_objects: bool = True,
    unclump_method: UnclumpMethod = UnclumpMethod.INTENSITY,
    watershed_method: WatershedMethod = WatershedMethod.INTENSITY,
    automatic_smoothing: bool = True,
    smoothing_filter_size: int = 10,
    automatic_suppression: bool = True,
    maxima_suppression_size: float = 7.0,
    low_res_maxima: bool = True,
    fill_holes: FillHolesOption = FillHolesOption.AFTER_BOTH,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    use_advanced_settings: bool = True,
    threshold_scope: CellProfilerThresholdScope = CellProfilerThresholdScope.GLOBAL,
    threshold_method: CellProfilerThresholdMethod = CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
    threshold_smoothing_scale: float = 1.3488,
    otsu_class_count: CellProfilerOtsuMethod = CellProfilerOtsuMethod.TWO_CLASS,
    assign_middle_to_foreground: CellProfilerThresholdAssignment = CellProfilerThresholdAssignment.FOREGROUND,
    log_transform: bool = False,
    adaptive_window_size: int = 10,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: CellProfilerAveragingMethod = CellProfilerAveragingMethod.MEAN,
    variance_method: CellProfilerVarianceMethod = CellProfilerVarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2.0,
    manual_threshold: float = 0.0,
    maximum_object_count: int = 500,
    limit_erase: ExcessObjectHandling = ExcessObjectHandling.CONTINUE,
    morphology_backend_provider: CellProfilerBackendProvider | None = None,
    watershed_backend_provider: CellProfilerBackendProvider | None = None,
    respect_source_border_metadata: bool = False,
) -> Tuple[np.ndarray, PrimaryObjectStats, np.ndarray]:
    """
    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select the input image' -> (pipeline-handled)
        'Name the primary objects to be identified' -> (pipeline-handled)
        'Typical diameter of objects, in pixel units (Min,Max)' -> [min_diameter, max_diameter]
        'Discard objects outside the diameter range?' -> exclude_size
        'Discard objects touching the border of the image?' -> exclude_border_objects
        'Method to distinguish clumped objects' -> unclump_method
        'Method to draw dividing lines between clumped objects' -> watershed_method
        'Size of smoothing filter' -> smoothing_filter_size
        'Suppress local maxima that are closer than this minimum allowed distance' -> maxima_suppression_size
        'Speed up by using lower-resolution image to find local maxima?' -> low_res_maxima
        'Fill holes in identified objects?' -> fill_holes
        'Automatically calculate size of smoothing filter for declumping?' -> automatic_smoothing
        'Automatically calculate minimum allowed distance between local maxima?' -> automatic_suppression
        'Handling of objects if excessive number of objects identified' -> limit_erase
        'Maximum number of objects' -> maximum_object_count
        'Threshold correction factor' -> threshold_correction_factor
        'Lower and upper bounds on threshold' -> [threshold_min, threshold_max]
        'Use advanced settings?' -> use_advanced_settings
        'Threshold strategy' -> threshold_scope
        'Thresholding method' -> threshold_method
        'Threshold smoothing scale' -> threshold_smoothing_scale
        'Two-class or three-class thresholding?' -> otsu_class_count
        'Assign pixels in the middle intensity class to the foreground or the background?' -> assign_middle_to_foreground
        'Log transform before thresholding?' -> log_transform
        'Size of adaptive window' -> adaptive_window_size
        'Lower outlier fraction' -> lower_outlier_fraction
        'Upper outlier fraction' -> upper_outlier_fraction
        'Averaging method' -> averaging_method
        'Variance method' -> variance_method
        '# of deviations' -> number_of_deviations
        'Manual threshold' -> manual_threshold

    Identify primary objects in a grayscale image.
    
    Args:
        image: Input grayscale image (H, W)
        min_diameter: Minimum object diameter in pixels
        max_diameter: Maximum object diameter in pixels
        exclude_size: Discard objects outside diameter range
        exclude_border_objects: Discard objects touching image border
        unclump_method: Method to distinguish clumped objects
        watershed_method: Method to draw dividing lines between clumped objects
        automatic_smoothing: Auto-calculate smoothing filter size
        smoothing_filter_size: Size of smoothing filter for declumping
        automatic_suppression: Auto-calculate maxima suppression distance
        maxima_suppression_size: Minimum distance between local maxima
        low_res_maxima: Use lower resolution for finding maxima (faster)
        fill_holes: When to fill holes in objects
        threshold_correction_factor: Multiply threshold by this factor
        threshold_min: Minimum threshold value
        threshold_max: Maximum threshold value
        maximum_object_count: Max objects before erasing (if limit_erase=True)
        limit_erase: Erase all objects if count exceeds maximum
        respect_source_border_metadata: Treat only source-image edges as
            physical borders. The CellProfiler-compatible default filters the
            current image plane border.
    
    Returns:
        Tuple of (original image, object statistics, labeled image)
    """
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    profile_total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    morphology = MorphologyBackendStrategy.for_callable(
        identify_primary_objects,
        backend_provider=morphology_backend_provider,
    )
    unclump_method = _coerce_function_enum(UnclumpMethod, unclump_method)
    watershed_method = _coerce_function_enum(WatershedMethod, watershed_method)
    fill_holes = _coerce_function_enum(FillHolesOption, fill_holes)
    limit_erase = _coerce_function_enum(ExcessObjectHandling, limit_erase)
    threshold_method = _coerce_function_enum(
        CellProfilerThresholdMethod,
        threshold_method,
    )
    _log_profile(
        "ipo_prepare_inputs",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )
    phase_started_at = time.perf_counter()
    input_mask_payload = image_payload_mask(image)
    input_mask = (
        None
        if input_mask_payload is None
        else np.asarray(input_mask_payload, dtype=bool)
    )
    input_metadata = (
        image_payload_metadata(image)
        if respect_source_border_metadata
        else ImagePayloadMetadata()
    )
    img = normalize_cellprofiler_image(image)
    effective_threshold_smoothing = threshold_smoothing_scale
    _log_profile(
        "ipo_normalize_image",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )
    
    phase_started_at = time.perf_counter()
    binary, thresh, original_threshold = cellprofiler_threshold(
        img,
        use_advanced_settings=use_advanced_settings,
        threshold_scope=threshold_scope,
        threshold_method=threshold_method,
        threshold_smoothing_scale=effective_threshold_smoothing,
        otsu_class_count=otsu_class_count,
        assign_middle_to_foreground=assign_middle_to_foreground,
        log_transform=log_transform,
        threshold_correction_factor=threshold_correction_factor,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        adaptive_window_size=adaptive_window_size,
        lower_outlier_fraction=lower_outlier_fraction,
        upper_outlier_fraction=upper_outlier_fraction,
        averaging_method=averaging_method,
        variance_method=variance_method,
        number_of_deviations=number_of_deviations,
        manual_threshold=manual_threshold,
        mask=input_mask,
        smooth_threshold_application=True,
    )
    threshold_binary = np.asarray(binary, dtype=bool).copy()
    _log_profile(
        "ipo_threshold",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )
    # Fill holes if requested (before declumping)
    if _fill_before_declump_requested(
        use_advanced_settings=use_advanced_settings,
        fill_holes=fill_holes,
    ):
        phase_started_at = time.perf_counter()
        max_hole_size = max_diameter * max_diameter
        binary = morphology.fill_labeled_holes_below_size(
            binary,
            max_hole_size,
        )
        _log_profile(
            "ipo_fill_before_declump",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )
    
    # Initial labeling
    phase_started_at = time.perf_counter()
    labeled_image, object_count = morphology.connected_components(
        binary,
        connectivity=2,
    )
    pre_declump_labels = labeled_image.copy()
    declump_backend_method = CellProfilerDeclumpMethod.SHAPE
    _log_profile(
        "ipo_initial_label",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
        object_count=object_count,
    )
    
    # Declumping and watershed
    if unclump_method != UnclumpMethod.NONE and watershed_method != WatershedMethod.NONE and object_count > 0:
        declump_started_at = time.perf_counter()
        if automatic_smoothing:
            smooth_size = 2.35 * min_diameter / 3.5
        else:
            smooth_size = _manual_declumping_size(smoothing_filter_size)

        initial_areas = np.bincount(labeled_image.ravel())[1:]
        median_initial_object_radius = (
            None
            if initial_areas.size == 0
            else float(np.sqrt(float(np.median(initial_areas)) / np.pi))
        )
        declump_backend_method = (
            CellProfilerDeclumpMethod.INTENSITY
            if unclump_method is UnclumpMethod.INTENSITY
            else CellProfilerDeclumpMethod.SHAPE
        )
        image_resize_factor, suppress_size = _declumping_maxima_geometry(
            min_diameter=min_diameter,
            low_res_maxima=low_res_maxima,
            automatic_suppression=automatic_suppression,
            maxima_suppression_size=maxima_suppression_size,
            declump_method=declump_backend_method,
            median_initial_object_radius=median_initial_object_radius,
        )

        maxima_mask = _declumping_suppression_footprint(
            morphology,
            suppress_size,
            min_diameter=min_diameter,
            declump_method=declump_backend_method,
        )
        image_mask = (
            np.ones(img.shape, dtype=bool)
            if input_mask is None
            else np.asarray(input_mask, dtype=bool)
        )
        phase_started_at = time.perf_counter()
        smoothed = morphology.smooth_image_for_declumping(
            img,
            image_mask,
            smooth_size,
            declump_method=declump_backend_method,
            suppress_size=suppress_size,
            min_diameter=min_diameter,
        )
        _log_profile(
            "ipo_declump_smooth",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )
        distance = None
        if unclump_method == UnclumpMethod.INTENSITY:
            maxima_image = smoothed
        else:
            phase_started_at = time.perf_counter()
            distance = shape_measurement_backend().distance_to_edge(
                np.asarray(labeled_image, dtype=np.int32)
            )
            distance = distance + np.random.RandomState(0).uniform(
                0,
                0.001,
                distance.shape,
            )
            maxima_image = distance
            _log_profile(
                "ipo_declump_distance",
                time.perf_counter() - phase_started_at,
                function="identify_primary_objects",
            )

        phase_started_at = time.perf_counter()
        maxima = morphology.declumping_seed_points(
            maxima_image,
            labeled_image,
            maxima_mask,
            image_resize_factor,
        )
        _log_profile(
            "ipo_declump_seed_points",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )
        phase_started_at = time.perf_counter()
        markers, object_count = morphology.connected_components(
            maxima,
            connectivity=2,
        )
        _log_profile(
            "ipo_declump_marker_label",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
            object_count=object_count,
        )
        if object_count > 0:
            phase_started_at = time.perf_counter()
            if watershed_method == WatershedMethod.SHAPE:
                if distance is None:
                    distance = shape_measurement_backend().distance_to_edge(
                        np.asarray(labeled_image, dtype=np.int32)
                    )
                watershed_image = -distance
                watershed_image = watershed_image - np.min(watershed_image)
            else:
                watershed_image = WatershedImageBuilder.for_method(
                    watershed_method
                ).build(img, binary)
            watershed_markers = np.zeros(watershed_image.shape, np.int32)
            watershed_markers[markers > 0] = -markers[markers > 0]
            _log_profile(
                "ipo_watershed_prepare",
                time.perf_counter() - phase_started_at,
                function="identify_primary_objects",
            )
            phase_started_at = time.perf_counter()
            labeled_image = -cellprofiler_legacy_watershed(
                watershed_image,
                markers=watershed_markers,
                mask=labeled_image != 0,
                connectivity=np.ones((3, 3), bool),
                backend_provider=watershed_backend_provider,
            )
            object_count = int(labeled_image.max())
            _log_profile(
                "ipo_watershed_execute",
                time.perf_counter() - phase_started_at,
                function="identify_primary_objects",
                object_count=object_count,
            )
        _log_profile(
            "ipo_declump_total",
            time.perf_counter() - declump_started_at,
            function="identify_primary_objects",
        )
    
    phase_started_at = time.perf_counter()
    unedited_labels = labeled_image.copy()
    small_removed_labels = labeled_image.copy()
    _log_profile(
        "ipo_copy_label_variants",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )

    # Filter objects touching the image border, or the mask border when the
    # image has a crop/mask domain. This mirrors CellProfiler's legacy rule:
    # mask-border filtering is applied only when no labels touch the physical
    # image border.
    if exclude_border_objects and object_count > 0:
        phase_started_at = time.perf_counter()
        labeled_image = _filter_border_objects(
            labeled_image,
            image_mask=input_mask,
            image_metadata=input_metadata,
        )
        _log_profile(
            "ipo_filter_border",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )
    
    phase_started_at = time.perf_counter()
    labels_before_size_filter = labeled_image.copy()
    _log_profile(
        "ipo_copy_size_reference",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )

    # Filter objects by size. Keep CellProfiler's small-removed variant: small
    # objects are removed, large objects are still present.
    if exclude_size and object_count > 0:
        phase_started_at = time.perf_counter()
        labeled_image = _filter_labels_below_minimum_diameter(
            labeled_image,
            min_diameter,
        )
        small_removed_labels = labeled_image.copy()
        labeled_image = _filter_labels_above_maximum_diameter(
            labeled_image,
            max_diameter,
        )
        _log_profile(
            "ipo_filter_size",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )

    # CellProfiler fills segmented-object holes after border and size filtering.
    if _fill_after_declump_requested(
        use_advanced_settings=use_advanced_settings,
        fill_holes=fill_holes,
    ):
        phase_started_at = time.perf_counter()
        labeled_image = morphology.fill_labeled_holes(labeled_image)
        if exclude_size and object_count > 0:
            labeled_image = _filter_labels_below_minimum_diameter(
                labeled_image,
                min_diameter,
            )
            labeled_image = _filter_labels_above_maximum_diameter(
                labeled_image,
                max_diameter,
            )
        _log_profile(
            "ipo_fill_after_declump",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )
    
    # Relabel while preserving watershed boundaries between touching objects.
    phase_started_at = time.perf_counter()
    labeled_image, object_count = morphology.relabel_sequential(labeled_image)
    _log_profile(
        "ipo_relabel",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
        object_count=object_count,
    )
    
    # Check object count limit
    if limit_erase.erase_excess and object_count > maximum_object_count:
        labeled_image = np.zeros_like(labeled_image)
        object_count = 0
    
    # Calculate statistics
    phase_started_at = time.perf_counter()
    mean_area, median_area, total_area = _label_area_statistics(labeled_image)
    threshold_diagnostics = cellprofiler_threshold_diagnostics(
        img,
        threshold_binary,
        final_threshold=thresh,
        original_threshold=original_threshold,
        mask=input_mask,
    )
    _log_profile(
        "ipo_statistics_diagnostics",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )
    
    stats = PrimaryObjectStats(
        slice_index=0,
        object_count=object_count,
        mean_area=mean_area,
        median_area=median_area,
        total_area=total_area,
        threshold_used=float(thresh),
        original_threshold=threshold_diagnostics.original_threshold,
        weighted_variance=threshold_diagnostics.weighted_variance,
        sum_of_entropies=threshold_diagnostics.sum_of_entropies,
    )
    _log_profile(
        "ipo_total",
        time.perf_counter() - profile_total_started_at,
        function="identify_primary_objects",
    )
    
    return (
        image,
        stats,
        ObjectLabelPayload(
            labels=labeled_image.astype(np.int32, copy=False),
            unedited_labels=unedited_labels.astype(np.int32, copy=False),
            small_removed_labels=small_removed_labels.astype(np.int32, copy=False),
            declared_object_count=object_count,
        ),
    )


def _label_area_statistics(labels: np.ndarray) -> tuple[float, float, float]:
    """Return mean, median, and total positive-label area."""
    areas = np.bincount(np.asarray(labels).ravel())[1:]
    positive_areas = areas[areas > 0]
    if positive_areas.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(positive_areas)),
        float(np.median(positive_areas)),
        float(np.sum(positive_areas)),
    )


def _filter_labels_below_minimum_diameter(
    labels: np.ndarray,
    min_diameter: float,
) -> np.ndarray:
    min_area = np.pi * (float(min_diameter) ** 2) / 4.0
    areas = np.bincount(np.asarray(labels).ravel())
    return _filter_labels_by_area_numba(
        np.ascontiguousarray(labels),
        np.ascontiguousarray(areas),
        float(min_area),
        np.inf,
    )


def _filter_labels_above_maximum_diameter(
    labels: np.ndarray,
    max_diameter: float,
) -> np.ndarray:
    max_area = np.pi * (float(max_diameter) ** 2) / 4.0
    areas = np.bincount(np.asarray(labels).ravel())
    return _filter_labels_by_area_numba(
        np.ascontiguousarray(labels),
        np.ascontiguousarray(areas),
        0.0,
        float(max_area),
    )


@njit(cache=True, parallel=True)
def _filter_labels_by_area_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> np.ndarray:
    output = labels.copy()
    height, width = labels.shape
    for row in prange(height):
        for col in range(width):
            label = int(labels[row, col])
            if label <= 0:
                continue
            area = float(areas[label])
            if area < min_area or area > max_area:
                output[row, col] = 0
    return output


def _filter_border_objects(
    labeled_image: np.ndarray,
    *,
    image_mask: np.ndarray | None,
    image_metadata: ImagePayloadMetadata = ImagePayloadMetadata(),
) -> np.ndarray:
    """Remove labels touching the physical border or masked image border."""
    output = labeled_image.copy()
    max_label = int(output.max())
    if max_label <= 0:
        return output

    height, width = output.shape[:2]
    physical_edges = image_metadata.physical_border_edges_for_shape((height, width))
    border_slices = []
    if height > 0 and physical_edges[0]:
        border_slices.append(output[0, :])
    if height > 0 and physical_edges[1]:
        border_slices.append(output[-1, :])
    if width > 0 and physical_edges[2]:
        border_slices.append(output[:, 0])
    if width > 0 and physical_edges[3]:
        border_slices.append(output[:, -1])
    if border_slices:
        border_labels = np.concatenate(border_slices).astype(np.int64, copy=False)
        border_histogram = np.bincount(border_labels, minlength=max_label + 1)
        labels_to_remove = np.flatnonzero(border_histogram[1:] > 0) + 1
        if labels_to_remove.size:
            output[np.isin(output, labels_to_remove)] = 0
            return output

    if image_mask is None or image_metadata.mask_defines_border is False:
        return output

    from scipy import ndimage as ndi

    mask = np.asarray(image_mask, dtype=bool)
    mask_border = np.logical_not(ndi.binary_erosion(mask, border_value=1)) & mask
    masked_border_labels = output[mask_border].astype(np.int64, copy=False)
    masked_border_histogram = np.bincount(
        masked_border_labels,
        minlength=max_label + 1,
    )
    labels_to_remove = np.flatnonzero(masked_border_histogram[1:] > 0) + 1
    if labels_to_remove.size:
        output[np.isin(output, labels_to_remove)] = 0
    return output


def _declumping_suppression_footprint(
    morphology: object,
    suppress_size: float,
    *,
    min_diameter: float,
    declump_method: "CellProfilerDeclumpMethod",
) -> np.ndarray:
    """Return the backend-provided local-maxima suppression footprint."""
    return np.asarray(
        morphology.declumping_suppression_footprint(
            suppress_size,
            min_diameter=min_diameter,
            declump_method=declump_method,
        ),
        dtype=bool,
    )


def _prepare_identify_primary_objects() -> None:
    """Compile Numba/backend kernels used by IdentifyPrimaryObjects."""
    labels = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 2],
            [3, 3, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.int32,
    )
    areas = np.bincount(labels.ravel())
    _filter_labels_by_area_numba(
        np.ascontiguousarray(labels),
        np.ascontiguousarray(areas),
        2.0,
        4.0,
    )
    image = np.zeros((96, 96), dtype=np.float32)
    yy, xx = np.ogrid[:96, :96]
    image[((yy - 32) ** 2 + (xx - 32) ** 2) <= 12 * 12] = 0.8
    image[((yy - 56) ** 2 + (xx - 56) ** 2) <= 12 * 12] = 0.75
    identify_primary_objects.__wrapped__(
        image,
        min_diameter=10,
        max_diameter=45,
        unclump_method=UnclumpMethod.SHAPE,
        watershed_method=WatershedMethod.SHAPE,
        low_res_maxima=True,
        use_advanced_settings=True,
        threshold_method=CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
    )


identify_primary_objects.__openhcs_prepare__ = _prepare_identify_primary_objects
