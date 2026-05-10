"""
Converted from CellProfiler: Threshold
Original: threshold
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
)
from benchmark.cellprofiler_library.functions.thresholding import (
    cellprofiler_threshold,
    cellprofiler_threshold_diagnostics,
)
from openhcs.processing.backends.cellprofiler.thresholding import threshold_primitives
from openhcs.processing.materialization import csv_materializer


class ThresholdScope(Enum):
    GLOBAL = "global"
    ADAPTIVE = "adaptive"


class ThresholdMethod(Enum):
    OTSU = "otsu"
    MANUAL = "manual"
    MINIMUM_CROSS_ENTROPY = "minimum_cross_entropy"
    LI = "li"
    TRIANGLE = "triangle"
    ISODATA = "isodata"
    ROBUST_BACKGROUND = "robust_background"


class Assignment(Enum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"


class AveragingMethod(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"


class VarianceMethod(Enum):
    STANDARD_DEVIATION = "standard_deviation"
    MEDIAN_ABSOLUTE_DEVIATION = "median_absolute_deviation"


@dataclass
class ThresholdResult:
    slice_index: int
    final_threshold: float
    original_threshold: float
    guide_threshold: float
    sigma: float
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0


def _get_global_threshold(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    threshold_method: ThresholdMethod,
    log_transform: bool,
    lower_outlier_fraction: float,
    upper_outlier_fraction: float,
    averaging_method: AveragingMethod,
    variance_method: VarianceMethod,
    number_of_deviations: float,
) -> float:
    """Calculate global threshold using specified method."""
    primitives = threshold_primitives()
    
    # Apply mask if provided
    if mask is not None:
        data = image[mask > 0]
    else:
        data = image.ravel()
    
    # Remove zeros and invalid values
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return 0.0
    
    # Log transform if requested
    if log_transform:
        data = data[data > 0]
        if len(data) == 0:
            return 0.0
        data, conversion = primitives.log_transform(data)
    else:
        conversion = None

    if threshold_method == ThresholdMethod.OTSU:
        thresh = primitives.otsu_threshold(data)
    elif threshold_method == ThresholdMethod.MINIMUM_CROSS_ENTROPY:
        thresh = primitives.minimum_cross_entropy_threshold(data)
    elif threshold_method == ThresholdMethod.LI:
        thresh = primitives.li_threshold(data)
    elif threshold_method == ThresholdMethod.TRIANGLE:
        thresh = primitives.triangle_threshold(data)
    elif threshold_method == ThresholdMethod.ISODATA:
        thresh = primitives.isodata_threshold(data)
    elif threshold_method == ThresholdMethod.ROBUST_BACKGROUND:
        # Robust background method
        sorted_data = np.sort(data)
        n = len(sorted_data)
        lower_idx = int(n * lower_outlier_fraction)
        upper_idx = int(n * (1 - upper_outlier_fraction))
        trimmed = sorted_data[lower_idx:upper_idx]
        
        if len(trimmed) == 0:
            trimmed = sorted_data
        
        if averaging_method == AveragingMethod.MEAN:
            center = np.mean(trimmed)
        elif averaging_method == AveragingMethod.MEDIAN:
            center = np.median(trimmed)
        else:  # MODE
            center = primitives.binned_mode(trimmed)
        
        if variance_method == VarianceMethod.STANDARD_DEVIATION:
            spread = np.std(trimmed)
        else:  # MEDIAN_ABSOLUTE_DEVIATION
            spread = primitives.mad(trimmed) * 1.4826

        thresh = center + number_of_deviations * spread
    else:
        thresh = primitives.otsu_threshold(data)

    # Reverse log transform if applied
    if conversion is not None:
        thresh = primitives.inverse_log_transform(thresh, conversion)

    return float(thresh)


def _get_adaptive_threshold(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    threshold_method: ThresholdMethod,
    window_size: int,
    log_transform: bool,
    lower_outlier_fraction: float,
    upper_outlier_fraction: float,
    averaging_method: AveragingMethod,
    variance_method: VarianceMethod,
    number_of_deviations: float,
) -> np.ndarray:
    """Calculate adaptive (local) threshold."""
    from scipy.ndimage import uniform_filter
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    work_image = image.copy().astype(np.float64)
    
    if log_transform:
        work_image = np.where(work_image > 0, np.log(work_image), 0)
    
    # Local mean
    local_mean = uniform_filter(work_image, size=window_size, mode='reflect')
    
    # Local variance for adaptive offset
    local_sq_mean = uniform_filter(work_image ** 2, size=window_size, mode='reflect')
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)
    local_std = np.sqrt(local_var)
    
    # Get global threshold as guide
    global_thresh = _get_global_threshold(
        image, mask, threshold_method, log_transform,
        lower_outlier_fraction, upper_outlier_fraction,
        averaging_method, variance_method, number_of_deviations
    )
    
    # Adaptive threshold based on local statistics
    # Use local mean adjusted by relationship to global threshold
    adaptive_thresh = local_mean + 0.5 * local_std
    
    if log_transform:
        adaptive_thresh = np.exp(adaptive_thresh)
    
    return adaptive_thresh


def _apply_threshold(
    image: np.ndarray,
    threshold: np.ndarray,
    mask: Optional[np.ndarray],
    smoothing: float,
) -> Tuple[np.ndarray, float]:
    """Apply threshold to image and return binary mask."""
    from scipy.ndimage import gaussian_filter
    
    sigma = smoothing
    
    if smoothing > 0:
        smoothed = gaussian_filter(image.astype(np.float64), sigma=smoothing)
    else:
        smoothed = image
    
    if isinstance(threshold, np.ndarray):
        binary = smoothed > threshold
    else:
        binary = smoothed > threshold
    
    if mask is not None:
        binary = binary & (mask > 0)
    
    return binary.astype(np.float32), sigma


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("threshold_results", csv_materializer(
    fields=["slice_index", "final_threshold", "original_threshold", "guide_threshold", "sigma"],
    analysis_type="threshold"
)))
def threshold(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold_scope: ThresholdScope | str = ThresholdScope.GLOBAL,
    threshold_method: ThresholdMethod | str = ThresholdMethod.OTSU,
    assign_middle_to_foreground: Assignment | str = Assignment.FOREGROUND,
    log_transform: bool = False,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    window_size: int = 50,
    smoothing: float = 0.0,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: AveragingMethod | str = AveragingMethod.MEAN,
    variance_method: VarianceMethod | str = VarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2.0,
    predefined_threshold: Optional[float] = None,
    automatic: bool = False,
    otsu_class_count: str = "Two classes",
    use_advanced_settings: bool = True,
) -> Tuple[np.ndarray, ThresholdResult]:
    """
    Apply threshold to image and return binary mask with threshold metrics.
    
    Returns three threshold values and a binary image.
    Thresholds returned are:
    
    Final threshold: Threshold following application of the
    threshold_correction_factor and clipping to min/max threshold
    
    orig_threshold: The threshold following either adaptive or global
    thresholding strategies, prior to correction
    
    guide_threshold: Only produced by adaptive threshold, otherwise 0.
    This is the global threshold that constrains the adaptive threshold.
    
    Args:
        image: Input grayscale image (H, W)
        mask: Optional mask to apply to the image
        threshold_scope: GLOBAL or ADAPTIVE thresholding
        threshold_method: Method to calculate threshold
        assign_middle_to_foreground: How to assign middle values
        log_transform: Apply log transform before thresholding
        threshold_correction_factor: Factor to multiply threshold by
        threshold_min: Minimum allowed threshold
        threshold_max: Maximum allowed threshold
        window_size: Window size for adaptive thresholding
        smoothing: Gaussian smoothing sigma
        lower_outlier_fraction: Lower outlier fraction for robust background
        upper_outlier_fraction: Upper outlier fraction for robust background
        averaging_method: Averaging method for robust background
        variance_method: Variance method for robust background
        number_of_deviations: Number of deviations for robust background
        predefined_threshold: Use this threshold value directly
        automatic: Use automatic settings
    
    Returns:
        Tuple of (binary_mask, ThresholdResult)
    """
    source_payload = image
    image = np.asarray(image_payload_data(source_payload), dtype=np.float32)
    if mask is not None:
        mask = np.asarray(image_payload_data(mask))
    else:
        mask = image_payload_mask(source_payload)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    threshold_scope = coerce_cellprofiler_enum(ThresholdScope, threshold_scope)
    threshold_method = coerce_cellprofiler_enum(ThresholdMethod, threshold_method)
    assign_middle_to_foreground = coerce_cellprofiler_enum(
        Assignment,
        assign_middle_to_foreground,
    )
    averaging_method = coerce_cellprofiler_enum(AveragingMethod, averaging_method)
    variance_method = coerce_cellprofiler_enum(VarianceMethod, variance_method)

    guide_threshold = 0.0

    if automatic:
        smoothing = 1.0
        log_transform = False
        threshold_scope = ThresholdScope.GLOBAL
        threshold_method = ThresholdMethod.MINIMUM_CROSS_ENTROPY
    
    if threshold_method is ThresholdMethod.MANUAL and predefined_threshold is None:
        predefined_threshold = 0.0
    if predefined_threshold is not None:
        threshold_method = ThresholdMethod.MANUAL

    binary_image, final_threshold, original_threshold = cellprofiler_threshold(
        image,
        use_advanced_settings=use_advanced_settings,
        threshold_scope=threshold_scope.value,
        threshold_method=threshold_method.value,
        otsu_class_count=otsu_class_count,
        assign_middle_to_foreground=assign_middle_to_foreground.value,
        log_transform=log_transform,
        threshold_correction_factor=threshold_correction_factor,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_smoothing_scale=smoothing,
        adaptive_window_size=window_size,
        lower_outlier_fraction=lower_outlier_fraction,
        upper_outlier_fraction=upper_outlier_fraction,
        averaging_method=averaging_method.value,
        variance_method=variance_method.value,
        number_of_deviations=number_of_deviations,
        manual_threshold=(
            float(predefined_threshold)
            if predefined_threshold is not None
            else 0.0
        ),
        mask=mask,
    )
    diagnostics = cellprofiler_threshold_diagnostics(
        image,
        binary_image,
        final_threshold=final_threshold,
        original_threshold=original_threshold,
        mask=mask,
    )
    output_image = image_payload_with_context(
        binary_image.astype(np.float32),
        mask=mask,
        metadata=image_payload_metadata(source_payload).without_unit_interval_intensity_scale(),
    )
    return output_image, ThresholdResult(
        slice_index=0,
        final_threshold=float(final_threshold),
        original_threshold=float(original_threshold),
        guide_threshold=guide_threshold,
        sigma=float(smoothing),
        weighted_variance=diagnostics.weighted_variance,
        sum_of_entropies=diagnostics.sum_of_entropies,
    )
