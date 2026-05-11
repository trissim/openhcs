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
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler._backend import BackendProviderInput
from openhcs.processing.backends.cellprofiler.image_quality import (
    ThresholdMethod,
    image_quality_focus_score,
    image_quality_haralick_correlation,
    image_quality_intensity_metrics,
    image_quality_local_focus_score,
    image_quality_power_spectrum_slope,
    image_quality_saturation,
    image_quality_threshold,
)
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    import logging

    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logging.getLogger(__name__).info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


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
        metrics.focus_score = image_quality_focus_score(pixel_data)
        _log_profile(
            "miq_focus_score",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
        phase_started_at = time.perf_counter()
        metrics.local_focus_score = image_quality_local_focus_score(pixel_data, blur_scale)
        _log_profile(
            "miq_local_focus_score",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
        phase_started_at = time.perf_counter()
        metrics.correlation = image_quality_haralick_correlation(
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
        metrics.power_log_log_slope = image_quality_power_spectrum_slope(
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
        metrics.percent_maximal, metrics.percent_minimal = image_quality_saturation(pixel_data)
        _log_profile(
            "miq_saturation",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )

    if calculate_intensity:
        phase_started_at = time.perf_counter()
        intensity_metrics = image_quality_intensity_metrics(pixel_data)
        metrics.total_area = intensity_metrics.total_area
        metrics.total_intensity = intensity_metrics.total_intensity
        metrics.mean_intensity = intensity_metrics.mean_intensity
        metrics.median_intensity = intensity_metrics.median_intensity
        metrics.std_intensity = intensity_metrics.std_intensity
        metrics.mad_intensity = intensity_metrics.mad_intensity
        metrics.min_intensity = intensity_metrics.min_intensity
        metrics.max_intensity = intensity_metrics.max_intensity
        _log_profile(
            "miq_intensity",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
    
    # Calculate threshold
    if calculate_threshold:
        phase_started_at = time.perf_counter()
        threshold_method = coerce_cellprofiler_enum(ThresholdMethod, threshold_method)
        metrics.threshold_otsu = image_quality_threshold(pixel_data, threshold_method)
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
