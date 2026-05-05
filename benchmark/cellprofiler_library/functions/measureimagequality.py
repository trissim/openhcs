"""
Converted from CellProfiler: MeasureImageQuality
Original: MeasureImageQuality module

Measures features that indicate image quality including blur metrics,
saturation metrics, intensity metrics, and threshold metrics.
"""

import numpy as np
import os
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from numba import njit
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler._backend import BackendProviderInput
from openhcs.processing.backends.cellprofiler.image_quality import image_quality_backend
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.backends.cellprofiler.thresholding import threshold_primitives
from openhcs.processing.materialization import csv_materializer
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    import logging

    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logging.getLogger(__name__).info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


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
    return float(
        _focus_score_numba(
            np.ascontiguousarray(pixel_data, dtype=np.float64),
        )
    )


@njit(cache=True)
def _focus_score_numba(pixel_data: np.ndarray) -> float:
    flat = pixel_data.ravel()
    count = flat.size
    if count == 0:
        return 0.0

    total = 0.0
    for index in range(count):
        total += flat[index]
    mean_value = total / float(count)
    if mean_value <= 0.0:
        return 0.0

    squared_sum = 0.0
    for index in range(count):
        diff = flat[index] - mean_value
        squared_sum += diff * diff
    return squared_sum / (float(count) * mean_value)


def _calculate_local_focus_score(pixel_data: np.ndarray, scale: int) -> float:
    """Calculate local focus score using grid-based normalized variance."""
    if pixel_data.size == 0 or scale <= 0:
        return 0.0
    return float(
        _local_focus_score_numba(
            np.ascontiguousarray(pixel_data, dtype=np.float64),
            int(scale),
        )
    )


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


@njit(cache=True)
def _local_focus_score_numba(pixel_data: np.ndarray, scale: int) -> float:
    height, width = pixel_data.shape
    if height == 0 or width == 0 or scale <= 0:
        return 0.0

    grid_rows = (height + scale - 1) // scale
    grid_cols = (width + scale - 1) // scale
    grid_count = grid_rows * grid_cols

    sums = np.zeros(grid_count, dtype=np.float64)
    counts = np.zeros(grid_count, dtype=np.int64)
    for row in range(height):
        grid_row = int(row * float(grid_rows) / float(height))
        if grid_row >= grid_rows:
            grid_row = grid_rows - 1
        for col in range(width):
            grid_col = int(col * float(grid_cols) / float(width))
            if grid_col >= grid_cols:
                grid_col = grid_cols - 1
            grid_index = grid_row * grid_cols + grid_col
            sums[grid_index] += pixel_data[row, col]
            counts[grid_index] += 1

    means = np.zeros(grid_count, dtype=np.float64)
    valid_count = 0
    for grid_index in range(grid_count):
        count = counts[grid_index]
        if count <= 0:
            continue
        mean_value = sums[grid_index] / count
        if mean_value != 0.0 and np.isfinite(mean_value):
            means[grid_index] = mean_value
            valid_count += 1

    if valid_count == 0:
        return 0.0

    squared_sums = np.zeros(grid_count, dtype=np.float64)
    for row in range(height):
        grid_row = int(row * float(grid_rows) / float(height))
        if grid_row >= grid_rows:
            grid_row = grid_rows - 1
        for col in range(width):
            grid_col = int(col * float(grid_cols) / float(width))
            if grid_col >= grid_cols:
                grid_col = grid_cols - 1
            grid_index = grid_row * grid_cols + grid_col
            mean_value = means[grid_index]
            diff = pixel_data[row, col] - mean_value
            squared_sums[grid_index] += diff * diff

    local_norm_var = np.empty(valid_count, dtype=np.float64)
    output_index = 0
    for grid_index in range(grid_count):
        mean_value = means[grid_index]
        if mean_value == 0.0 or not np.isfinite(mean_value):
            continue
        value = squared_sums[grid_index] / (counts[grid_index] * mean_value)
        if np.isfinite(value):
            local_norm_var[output_index] = value
            output_index += 1

    if output_index == 0:
        return 0.0

    values = local_norm_var[:output_index]
    median_value = np.median(values)
    if (not np.isfinite(median_value)) or median_value <= 0.0:
        return 0.0

    mean_value = 0.0
    for index in range(output_index):
        mean_value += values[index]
    mean_value /= output_index

    variance = 0.0
    for index in range(output_index):
        diff = values[index] - mean_value
        variance += diff * diff
    variance /= output_index
    return variance / median_value


def _calculate_power_spectrum_slope(
    pixel_data: np.ndarray,
    *,
    backend_provider: BackendProviderInput | None = None,
) -> float:
    """Calculate CellProfiler's log-log radial power spectrum slope."""
    if pixel_data.size == 0 or not _has_multiple_unique_values(pixel_data):
        return 0.0

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

    slope_value = _least_squares_log_log_slope_numba(
        np.ascontiguousarray(radii.ravel(), dtype=np.float64),
        np.ascontiguousarray(power.ravel(), dtype=np.float64),
    )
    return float(slope_value) if np.isfinite(slope_value) else 0.0


@njit(cache=True)
def _least_squares_log_log_slope_numba(
    radii: np.ndarray,
    power: np.ndarray,
) -> float:
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    for index in range(radii.size):
        radius = radii[index]
        power_value = power[index]
        if radius <= 0.0 or power_value <= 0.0:
            continue
        x_value = np.log(radius)
        y_value = np.log(power_value)
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            continue
        count += 1
        sum_x += x_value
        sum_y += y_value
        sum_xx += x_value * x_value
        sum_xy += x_value * y_value
    if count <= 1:
        return 0.0
    denominator = float(count) * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        return 0.0
    return (float(count) * sum_xy - sum_x * sum_y) / denominator


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
    if pixel_data.size == 0 or not _has_multiple_unique_values(pixel_data):
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


def _has_multiple_unique_values(pixel_data: np.ndarray) -> bool:
    """Return whether ``np.unique(pixel_data)`` would contain more than one value."""
    return bool(
        _has_multiple_unique_values_numba(
            np.ascontiguousarray(pixel_data, dtype=np.float32),
        )
    )


@njit(cache=True)
def _has_multiple_unique_values_numba(pixel_data: np.ndarray) -> bool:
    flat_size = pixel_data.size
    if flat_size <= 1:
        return False
    flat = pixel_data.ravel()
    first = flat[0]
    first_is_nan = np.isnan(first)
    for index in range(1, flat_size):
        value = flat[index]
        value_is_nan = np.isnan(value)
        if first_is_nan:
            if not value_is_nan:
                return True
        elif value_is_nan or value != first:
            return True
    return False


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
    total_started_at = time.perf_counter()
    metrics = ImageQualityMetrics(slice_index=0)

    phase_started_at = time.perf_counter()
    pixel_data = np.asarray(image, dtype=np.float32)
    _log_profile(
        "miq_prepare_image",
        time.perf_counter() - phase_started_at,
        function="measure_image_quality",
    )

    # Calculate blur metrics
    if calculate_blur:
        phase_started_at = time.perf_counter()
        metrics.focus_score = _calculate_focus_score(pixel_data)
        _log_profile(
            "miq_focus_score",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
        phase_started_at = time.perf_counter()
        metrics.local_focus_score = _calculate_local_focus_score(pixel_data, blur_scale)
        _log_profile(
            "miq_local_focus_score",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
        phase_started_at = time.perf_counter()
        metrics.correlation = _calculate_correlation(
            pixel_data,
            blur_scale,
            backend_provider=backend_provider,
        )
        _log_profile(
            "miq_correlation",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
        phase_started_at = time.perf_counter()
        metrics.power_log_log_slope = _calculate_power_spectrum_slope(
            pixel_data,
            backend_provider=backend_provider,
        )
        _log_profile(
            "miq_power_log_log_slope",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
    
    if calculate_saturation:
        phase_started_at = time.perf_counter()
        metrics.percent_maximal, metrics.percent_minimal = _calculate_saturation(pixel_data)
        _log_profile(
            "miq_saturation",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )

    if calculate_intensity:
        phase_started_at = time.perf_counter()
        intensity_metrics = _calculate_intensity_metrics(pixel_data)
        metrics.total_area = intensity_metrics['total_area']
        metrics.total_intensity = intensity_metrics['total_intensity']
        metrics.mean_intensity = intensity_metrics['mean_intensity']
        metrics.median_intensity = intensity_metrics['median_intensity']
        metrics.std_intensity = intensity_metrics['std_intensity']
        metrics.mad_intensity = intensity_metrics['mad_intensity']
        metrics.min_intensity = intensity_metrics['min_intensity']
        metrics.max_intensity = intensity_metrics['max_intensity']
        _log_profile(
            "miq_intensity",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
    
    # Calculate threshold
    if calculate_threshold:
        phase_started_at = time.perf_counter()
        metrics.threshold_otsu = _calculate_threshold(pixel_data, threshold_method)
        _log_profile(
            "miq_threshold",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
            method=threshold_method.value,
        )

    _log_profile(
        "miq_total",
        time.perf_counter() - total_started_at,
        function="measure_image_quality",
    )
    
    return image, metrics


def _prepare_measure_image_quality() -> None:
    sample = (
        (np.arange(64 * 64, dtype=np.uint16) % 256)
        .astype(np.float32)
        .reshape((64, 64))
    )
    measure_image_quality.__wrapped__(sample)


measure_image_quality.__openhcs_prepare__ = _prepare_measure_image_quality
