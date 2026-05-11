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
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from numba import njit, prange
from openhcs.core.memory import numpy
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    object_label_payload_from_source_image,
)

from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
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
from openhcs.processing.backends.cellprofiler.perf_fixtures import capture_array_fixture
from benchmark.cellprofiler_library.functions.watershed import (
    cellprofiler_legacy_watershed,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE,
    CellProfilerDeclumpMethod,
    DeclumpingMaximaGeometry,
    FillHolesOption,
    manual_declumping_size,
)
from openhcs.processing.backends.cellprofiler.shape import shape_measurement_backend
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)

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


class ExcessObjectHandling(Enum):
    CONTINUE = ("Continue", False)
    ERASE = ("Erase", True)

    def __new__(cls, value: str, erase_excess: bool):
        option = object.__new__(cls)
        option._value_ = value
        option.erase_excess = erase_excess
        return option


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
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    watershed_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
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
    unclump_method = coerce_cellprofiler_enum(UnclumpMethod, unclump_method)
    watershed_method = coerce_cellprofiler_enum(WatershedMethod, watershed_method)
    fill_holes = coerce_cellprofiler_enum(FillHolesOption, fill_holes)
    limit_erase = coerce_cellprofiler_enum(ExcessObjectHandling, limit_erase)
    threshold_method = coerce_cellprofiler_enum(
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
    raw_image_data = np.asarray(image_payload_data(image))
    diagnostics_unit_interval_scale = unit_interval_scale_for_threshold_diagnostics(
        raw_image_data,
        image_payload_metadata(image),
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
    if fill_holes.before_declump_requested(use_advanced_settings=use_advanced_settings):
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
            smooth_size = manual_declumping_size(smoothing_filter_size)

        declump_backend_method = (
            CellProfilerDeclumpMethod.INTENSITY
            if unclump_method is UnclumpMethod.INTENSITY
            else CellProfilerDeclumpMethod.SHAPE
        )
        maxima_geometry = DeclumpingMaximaGeometry.from_cellprofiler_settings(
            min_diameter=min_diameter,
            low_res_maxima=low_res_maxima,
            automatic_suppression=automatic_suppression,
            maxima_suppression_size=maxima_suppression_size,
        )
        image_resize_factor = maxima_geometry.image_resize_factor
        suppress_size = maxima_geometry.suppress_size

        maxima_mask = np.asarray(
            morphology.declumping_suppression_footprint(
                suppress_size,
                min_diameter=min_diameter,
                declump_method=declump_backend_method,
            ),
            dtype=bool,
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
                watershed_image = 1 - img
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
    
    # Filter objects by size. Keep CellProfiler's small-removed variant: small
    # objects are removed, large objects are still present.
    if exclude_size and object_count > 0:
        phase_started_at = time.perf_counter()
        small_removed_labels, labeled_image = _filter_labels_by_diameter_range(
            labeled_image,
            min_diameter,
            max_diameter,
        )
        _log_profile(
            "ipo_filter_size",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )

    # CellProfiler fills segmented-object holes after border and size filtering.
    if fill_holes.after_declump_requested(use_advanced_settings=use_advanced_settings):
        phase_started_at = time.perf_counter()
        capture_array_fixture("ipo_fill_after", labels=labeled_image)
        labeled_image = morphology.fill_labeled_holes(labeled_image)
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
        proven_unit_interval_scale=diagnostics_unit_interval_scale,
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
        object_label_payload_from_source_image(
            image,
            labeled_image.astype(np.int32, copy=False),
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
    labels_array = np.ascontiguousarray(labels)
    areas = np.bincount(np.asarray(labels_array).ravel())
    return _filter_labels_by_area_numba(
        labels_array,
        np.ascontiguousarray(areas),
        float(min_area),
        np.inf,
    )


def _filter_labels_above_maximum_diameter(
    labels: np.ndarray,
    max_diameter: float,
) -> np.ndarray:
    max_area = np.pi * (float(max_diameter) ** 2) / 4.0
    labels_array = np.ascontiguousarray(labels)
    areas = np.bincount(np.asarray(labels_array).ravel())
    return _filter_labels_by_area_numba(
        labels_array,
        np.ascontiguousarray(areas),
        0.0,
        float(max_area),
    )


def _filter_labels_by_diameter_range(
    labels: np.ndarray,
    min_diameter: float,
    max_diameter: float,
) -> tuple[np.ndarray, np.ndarray]:
    min_area = np.pi * (float(min_diameter) ** 2) / 4.0
    max_area = np.pi * (float(max_diameter) ** 2) / 4.0
    labels_array = np.ascontiguousarray(labels)
    areas = np.ascontiguousarray(np.bincount(np.asarray(labels_array).ravel()))
    return _filter_labels_by_diameter_range_numba(
        labels_array,
        areas,
        float(min_area),
        float(max_area),
    )


def _filter_labels_by_area_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> np.ndarray:
    if labels.ndim == 2:
        return _filter_labels_by_area_2d_numba(
            labels,
            areas,
            min_area,
            max_area,
        )
    if labels.ndim == 3:
        return _filter_labels_by_area_3d_numba(
            labels,
            areas,
            min_area,
            max_area,
        )
    raise ValueError(
        "IdentifyPrimaryObjects area filtering expects 2-D planes or stacked "
        f"planes, got shape {labels.shape!r}."
    )


@njit(cache=True, parallel=True)
def _filter_labels_by_area_2d_numba(
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


@njit(cache=True, parallel=True)
def _filter_labels_by_area_3d_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> np.ndarray:
    output = labels.copy()
    plane_count, height, width = labels.shape
    for plane_index in prange(plane_count):
        for row in range(height):
            for col in range(width):
                label = int(labels[plane_index, row, col])
                if label <= 0:
                    continue
                area = float(areas[label])
                if area < min_area or area > max_area:
                    output[plane_index, row, col] = 0
    return output


def _filter_labels_by_diameter_range_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim == 2:
        return _filter_labels_by_diameter_range_2d_numba(
            labels,
            areas,
            min_area,
            max_area,
        )
    if labels.ndim == 3:
        return _filter_labels_by_diameter_range_3d_numba(
            labels,
            areas,
            min_area,
            max_area,
        )
    raise ValueError(
        "IdentifyPrimaryObjects size filtering expects 2-D planes or stacked "
        f"planes, got shape {labels.shape!r}."
    )


@njit(cache=True, parallel=True)
def _filter_labels_by_diameter_range_2d_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    small_removed = labels.copy()
    final = labels.copy()
    height, width = labels.shape
    for row in prange(height):
        for col in range(width):
            label = int(labels[row, col])
            if label <= 0:
                continue
            area = float(areas[label])
            if area < min_area:
                small_removed[row, col] = 0
                final[row, col] = 0
            elif area > max_area:
                final[row, col] = 0
    return small_removed, final


@njit(cache=True, parallel=True)
def _filter_labels_by_diameter_range_3d_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    small_removed = labels.copy()
    final = labels.copy()
    plane_count, height, width = labels.shape
    for plane_index in prange(plane_count):
        for row in range(height):
            for col in range(width):
                label = int(labels[plane_index, row, col])
                if label <= 0:
                    continue
                area = float(areas[label])
                if area < min_area:
                    small_removed[plane_index, row, col] = 0
                    final[plane_index, row, col] = 0
                elif area > max_area:
                    final[plane_index, row, col] = 0
    return small_removed, final


def _filter_border_objects(
    labeled_image: np.ndarray,
    *,
    image_mask: np.ndarray | None,
    image_metadata: ImagePayloadMetadata = ImagePayloadMetadata(),
) -> np.ndarray:
    """Remove labels touching the physical border or masked image border."""
    labeled_array = np.asarray(labeled_image)
    if labeled_array.ndim > 2:
        return _filter_border_objects_planewise(
            labeled_array,
            image_mask=image_mask,
            image_metadata=image_metadata,
        )

    height, width = labeled_array.shape[:2]
    physical_edges = image_metadata.physical_border_edges_for_shape((height, width))
    output, removed_physical = _filter_physical_border_objects_numba(
        np.ascontiguousarray(labeled_array),
        bool(physical_edges[0]),
        bool(physical_edges[1]),
        bool(physical_edges[2]),
        bool(physical_edges[3]),
    )
    if removed_physical:
        return output

    if image_mask is None or image_metadata.mask_defines_border is False:
        return output

    from scipy import ndimage as ndi

    max_label = int(output.max())
    if max_label <= 0:
        return output
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


def _filter_border_objects_planewise(
    labeled_image: np.ndarray,
    *,
    image_mask: np.ndarray | None,
    image_metadata: ImagePayloadMetadata,
) -> np.ndarray:
    output = np.empty_like(labeled_image)
    label_planes = labeled_image.reshape((-1, *labeled_image.shape[-2:]))
    output_planes = output.reshape((-1, *output.shape[-2:]))
    mask_planes = _mask_planes_for_labels(image_mask, label_planes.shape[0])
    for plane_index in range(label_planes.shape[0]):
        output_planes[plane_index] = _filter_border_objects(
            label_planes[plane_index],
            image_mask=None if mask_planes is None else mask_planes[plane_index],
            image_metadata=image_metadata.for_channel(plane_index),
        )
    return output


def _mask_planes_for_labels(
    image_mask: np.ndarray | None,
    plane_count: int,
) -> np.ndarray | None:
    if image_mask is None:
        return None
    mask = np.asarray(image_mask, dtype=bool)
    if mask.ndim == 2:
        return np.broadcast_to(mask, (plane_count, *mask.shape))
    mask_planes = mask.reshape((-1, *mask.shape[-2:]))
    if mask_planes.shape[0] == plane_count:
        return mask_planes
    if mask_planes.shape[0] == 1:
        return np.broadcast_to(mask_planes[0], (plane_count, *mask_planes.shape[-2:]))
    raise ValueError(
        "IdentifyPrimaryObjects mask stack must align with label stack; got "
        f"{mask.shape!r} for {plane_count} label planes."
    )


@njit(cache=True)
def _filter_physical_border_objects_numba(
    labels: np.ndarray,
    top: bool,
    bottom: bool,
    left: bool,
    right: bool,
) -> tuple[np.ndarray, bool]:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label
    if max_label <= 0:
        return labels, False

    remove = np.zeros(max_label + 1, dtype=np.bool_)
    if top and height > 0:
        for x in range(width):
            label = int(labels[0, x])
            if label > 0:
                remove[label] = True
    if bottom and height > 0:
        for x in range(width):
            label = int(labels[height - 1, x])
            if label > 0:
                remove[label] = True
    if left and width > 0:
        for y in range(height):
            label = int(labels[y, 0])
            if label > 0:
                remove[label] = True
    if right and width > 0:
        for y in range(height):
            label = int(labels[y, width - 1])
            if label > 0:
                remove[label] = True

    any_removed = False
    for label in range(1, max_label + 1):
        if remove[label]:
            any_removed = True
            break
    if not any_removed:
        return labels, False

    output = labels.copy()
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > 0 and remove[label]:
                output[y, x] = 0
    return output, True


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
    _filter_labels_by_diameter_range(labels, 2.0, 4.0)
    _filter_physical_border_objects_numba(labels, True, True, True, True)
    image = np.zeros((96, 96), dtype=np.float32)
    yy, xx = np.ogrid[:96, :96]
    image[((yy - 32) ** 2 + (xx - 32) ** 2) <= 12 * 12] = 0.8
    image[((yy - 56) ** 2 + (xx - 56) ** 2) <= 12 * 12] = 0.75
    binary = image > np.float32(0.5)
    cellprofiler_threshold_diagnostics(
        image,
        binary,
        final_threshold=0.5,
        original_threshold=0.5,
        proven_unit_interval_scale=65535,
    )
    rectangular_mask = np.zeros_like(binary, dtype=bool)
    rectangular_mask[16:80, 16:80] = True
    cellprofiler_threshold_diagnostics(
        image,
        binary,
        final_threshold=0.5,
        original_threshold=0.5,
        mask=rectangular_mask,
        proven_unit_interval_scale=65535,
    )
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
    identify_primary_objects.__wrapped__(
        image,
        min_diameter=3,
        max_diameter=15,
        unclump_method=UnclumpMethod.INTENSITY,
        watershed_method=WatershedMethod.INTENSITY,
        low_res_maxima=True,
        use_advanced_settings=True,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.THREE_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.BACKGROUND,
        threshold_smoothing_scale=1.3488,
    )
    identify_primary_objects.__wrapped__(
        image,
        min_diameter=3,
        max_diameter=15,
        unclump_method=UnclumpMethod.INTENSITY,
        watershed_method=WatershedMethod.INTENSITY,
        low_res_maxima=True,
        use_advanced_settings=True,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.BACKGROUND,
        threshold_smoothing_scale=1.3488,
    )


identify_primary_objects.__openhcs_prepare__ = _prepare_identify_primary_objects
