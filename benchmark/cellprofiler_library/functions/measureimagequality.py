"""
Converted from CellProfiler: MeasureImageQuality
Original: MeasureImageQuality module

Measures features that indicate image quality including blur metrics,
saturation metrics, intensity metrics, and threshold metrics.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler._backend import BackendProviderInput
from openhcs.processing.backends.cellprofiler.image_quality import image_quality_backend
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.backends.cellprofiler.thresholding import threshold_primitives
from openhcs.processing.materialization import csv_materializer
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum


class ThresholdMethod(Enum):
    OTSU = "otsu"
    LI = "li"
    TRIANGLE = "triangle"
    ISODATA = "isodata"
    MINIMUM = "minimum"
    MEAN = "mean"
    YEN = "yen"


@dataclass
class ImageQualityMetrics:
    """Dataclass containing all image quality measurements."""
    slice_index: int = 0
    # Blur metrics
    focus_score: float = 0.0
    local_focus_score: float = 0.0
    correlation: float = 0.0
    power_log_log_slope: float = 0.0
    # Saturation metrics
    percent_maximal: float = 0.0
    percent_minimal: float = 0.0
    # Intensity metrics
    total_area: int = 0
    total_intensity: float = 0.0
    mean_intensity: float = 0.0
    median_intensity: float = 0.0
    std_intensity: float = 0.0
    mad_intensity: float = 0.0
    min_intensity: float = 0.0
    max_intensity: float = 0.0
    # Threshold metrics
    threshold_otsu: float = 0.0


def _calculate_focus_score(pixel_data: np.ndarray) -> float:
    """Calculate normalized variance focus score."""
    if pixel_data.size == 0:
        return 0.0
    mean_val = np.mean(pixel_data)
    if mean_val <= 0:
        return 0.0
    squared_normalized = (pixel_data - mean_val) ** 2
    focus_score = np.sum(squared_normalized) / (pixel_data.size * mean_val)
    return float(focus_score)


def _calculate_local_focus_score(pixel_data: np.ndarray, scale: int) -> float:
    """Calculate local focus score using grid-based normalized variance."""
    from scipy.ndimage import mean as ndimage_mean, sum as ndimage_sum
    
    shape = pixel_data.shape
    if pixel_data.size == 0:
        return 0.0
    
    # Create grid labels
    i, j = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    m, n = (np.array(shape) + scale - 1) // scale
    i = (i * float(m) / float(shape[0])).astype(int)
    j = (j * float(n) / float(shape[1])).astype(int)
    grid = i * n + j + 1
    grid_range = np.arange(0, m * n + 1, dtype=np.int32)
    
    # Calculate local means
    local_means = ndimage_mean(pixel_data, grid, grid_range)
    if not isinstance(local_means, np.ndarray):
        local_means = np.array([local_means])
    
    # Handle NaN values
    local_means = np.nan_to_num(local_means, nan=0.0)
    
    # Calculate local squared normalized image
    local_squared_normalized = (pixel_data - local_means[grid]) ** 2
    
    # Compute for non-zero means
    grid_mask = (local_means != 0) & np.isfinite(local_means)
    nz_grid_range = grid_range[grid_mask]
    
    if len(nz_grid_range) == 0:
        return 0.0
    
    if nz_grid_range[0] == 0:
        nz_grid_range = nz_grid_range[1:]
        local_means = local_means[1:]
        grid_mask = grid_mask[1:]
    
    if len(nz_grid_range) == 0:
        return 0.0
    
    sums = ndimage_sum(local_squared_normalized, grid, nz_grid_range)
    if not isinstance(sums, np.ndarray):
        sums = np.array([sums])
    
    pixel_counts = ndimage_sum(np.ones(shape), grid, nz_grid_range)
    if not isinstance(pixel_counts, np.ndarray):
        pixel_counts = np.array([pixel_counts])
    
    valid_means = local_means[grid_mask] if len(local_means) > len(nz_grid_range) else local_means[:len(nz_grid_range)]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        local_norm_var = sums / (pixel_counts * valid_means[:len(sums)])
    
    local_norm_var = local_norm_var[np.isfinite(local_norm_var)]
    
    if len(local_norm_var) == 0:
        return 0.0
    
    local_norm_median = np.median(local_norm_var)
    if np.isfinite(local_norm_median) and local_norm_median > 0:
        return float(np.var(local_norm_var) / local_norm_median)
    
    return 0.0


def _calculate_correlation(
    pixel_data: np.ndarray,
    scale: int,
    *,
    backend_provider: BackendProviderInput | None = None,
) -> float:
    """Calculate CellProfiler's Haralick H3 image-quality correlation."""
    if pixel_data.size == 0:
        return 0.0

    return image_quality_backend(
        backend_provider=backend_provider,
    ).haralick_h3(pixel_data, scale=scale)


def _calculate_power_spectrum_slope(
    pixel_data: np.ndarray,
    *,
    backend_provider: BackendProviderInput | None = None,
) -> float:
    """Calculate CellProfiler's log-log radial power spectrum slope."""
    if pixel_data.size == 0 or len(np.unique(pixel_data)) <= 1:
        return 0.0

    import scipy.linalg

    radii, magnitude, power = _cellprofiler_radial_power_spectrum(
        pixel_data,
        backend_provider=backend_provider,
    )
    if np.sum(magnitude) <= 0:
        return 0.0

    valid = magnitude > 0
    radii = radii[valid].reshape((-1, 1))
    power = power[valid].reshape((-1, 1))
    if radii.shape[0] <= 1:
        return 0.0

    idx = np.isfinite(np.log(power))
    design = np.hstack(
        (
            np.log(radii)[idx][:, np.newaxis],
            np.ones(radii.shape)[idx][:, np.newaxis],
        )
    )
    slope = scipy.linalg.lstsq(design, np.log(power)[idx][:, np.newaxis])[0][0]
    slope_value = float(np.asarray(slope).ravel()[0])
    return slope_value if np.isfinite(slope_value) else 0.0


def _cellprofiler_radial_power_spectrum(
    pixel_data: np.ndarray,
    *,
    backend_provider: BackendProviderInput | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return image_quality_backend(
        backend_provider=backend_provider,
    ).radial_power_spectrum(pixel_data)


def _calculate_saturation(pixel_data: np.ndarray) -> Tuple[float, float]:
    """Calculate percent of pixels at max and min values."""
    if pixel_data.size == 0:
        return 0.0, 0.0
    
    pixel_count = pixel_data.size
    max_val = np.max(pixel_data)
    min_val = np.min(pixel_data)
    
    num_maximal = np.sum(pixel_data == max_val)
    num_minimal = np.sum(pixel_data == min_val)
    
    percent_maximal = 100.0 * float(num_maximal) / float(pixel_count)
    percent_minimal = 100.0 * float(num_minimal) / float(pixel_count)
    
    return percent_maximal, percent_minimal


def _calculate_intensity_metrics(pixel_data: np.ndarray) -> dict:
    """Calculate intensity-based metrics."""
    if pixel_data.size == 0:
        return {
            'total_area': 0,
            'total_intensity': 0.0,
            'mean_intensity': 0.0,
            'median_intensity': 0.0,
            'std_intensity': 0.0,
            'mad_intensity': 0.0,
            'min_intensity': 0.0,
            'max_intensity': 0.0
        }
    
    pixel_median = np.median(pixel_data)
    
    return {
        'total_area': int(pixel_data.size),
        'total_intensity': float(np.sum(pixel_data)),
        'mean_intensity': float(np.mean(pixel_data)),
        'median_intensity': float(pixel_median),
        'std_intensity': float(np.std(pixel_data)),
        'mad_intensity': float(np.median(np.abs(pixel_data - pixel_median))),
        'min_intensity': float(np.min(pixel_data)),
        'max_intensity': float(np.max(pixel_data))
    }


def _calculate_threshold(pixel_data: np.ndarray, method: ThresholdMethod) -> float:
    """Calculate automatic threshold using specified method."""
    if pixel_data.size == 0 or len(np.unique(pixel_data)) <= 1:
        return 0.0

    method = _coerce_function_enum(ThresholdMethod, method)
    primitives = threshold_primitives()
    values = pixel_data.astype(np.float32, copy=False)

    if method == ThresholdMethod.OTSU:
        return primitives.weighted_otsu_threshold(values)
    if method == ThresholdMethod.LI:
        return primitives.li_threshold(values)
    if method == ThresholdMethod.TRIANGLE:
        return primitives.triangle_threshold(values)
    if method == ThresholdMethod.ISODATA:
        return primitives.isodata_threshold(values)
    if method == ThresholdMethod.MINIMUM:
        return primitives.minimum_threshold(values)
    if method == ThresholdMethod.MEAN:
        return primitives.mean_threshold(values)
    if method == ThresholdMethod.YEN:
        return primitives.yen_threshold(values)
    raise NotImplementedError(f"Threshold method {method} not supported.")


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("quality_metrics", csv_materializer(
    fields=["slice_index", "focus_score", "local_focus_score", "correlation",
            "power_log_log_slope", "percent_maximal", "percent_minimal",
            "total_area", "total_intensity", "mean_intensity", "median_intensity",
            "std_intensity", "mad_intensity", "min_intensity", "max_intensity",
            "threshold_otsu"],
    analysis_type="image_quality"
)))
def measure_image_quality(
    image: np.ndarray,
    calculate_blur: bool = True,
    calculate_saturation: bool = True,
    calculate_intensity: bool = True,
    calculate_threshold: bool = True,
    blur_scale: int = 20,
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
    backend_provider: BackendProviderInput | None = None,
) -> Tuple[np.ndarray, ImageQualityMetrics]:
    """
    Measure image quality metrics including blur, saturation, intensity, and threshold.
    
    Args:
        image: Input grayscale image with shape (H, W)
        calculate_blur: Whether to calculate blur metrics (FocusScore, LocalFocusScore,
                       Correlation, PowerLogLogSlope)
        calculate_saturation: Whether to calculate saturation metrics (PercentMaximal,
                             PercentMinimal)
        calculate_intensity: Whether to calculate intensity metrics (TotalIntensity,
                            MeanIntensity, etc.)
        calculate_threshold: Whether to calculate automatic threshold
        blur_scale: Spatial scale for blur measurements (window size in pixels)
        threshold_method: Thresholding method to use
    
    Returns:
        Tuple of (original image, ImageQualityMetrics dataclass)
    """
    metrics = ImageQualityMetrics(slice_index=0)
    
    pixel_data = np.asarray(image, dtype=np.float32)
    
    # Calculate blur metrics
    if calculate_blur:
        metrics.focus_score = _calculate_focus_score(pixel_data)
        metrics.local_focus_score = _calculate_local_focus_score(pixel_data, blur_scale)
        metrics.correlation = _calculate_correlation(
            pixel_data,
            blur_scale,
            backend_provider=backend_provider,
        )
        metrics.power_log_log_slope = _calculate_power_spectrum_slope(
            pixel_data,
            backend_provider=backend_provider,
        )
    
    # Calculate saturation metrics
    if calculate_saturation:
        metrics.percent_maximal, metrics.percent_minimal = _calculate_saturation(pixel_data)
    
    # Calculate intensity metrics
    if calculate_intensity:
        intensity_metrics = _calculate_intensity_metrics(pixel_data)
        metrics.total_area = intensity_metrics['total_area']
        metrics.total_intensity = intensity_metrics['total_intensity']
        metrics.mean_intensity = intensity_metrics['mean_intensity']
        metrics.median_intensity = intensity_metrics['median_intensity']
        metrics.std_intensity = intensity_metrics['std_intensity']
        metrics.mad_intensity = intensity_metrics['mad_intensity']
        metrics.min_intensity = intensity_metrics['min_intensity']
        metrics.max_intensity = intensity_metrics['max_intensity']
    
    # Calculate threshold
    if calculate_threshold:
        metrics.threshold_otsu = _calculate_threshold(pixel_data, threshold_method)
    
    return image, metrics
