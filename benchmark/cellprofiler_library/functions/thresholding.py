"""Shared CellProfiler image normalization and threshold semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import logging
import os
import time

import numpy as np
import scipy.interpolate

from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler.perf_fixtures import (
    capture_array_fixture,
    capture_enabled,
)
from openhcs.core.runtime_values import ImagePayloadMetadata
from openhcs.core.runtime_values import image_intensity_scale_for_dtype
from openhcs.core.runtime_values import image_payload_data
from openhcs.core.runtime_values import normalize_image_payload_intensity
from openhcs.processing.backends.cellprofiler.thresholding import threshold_primitives

CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE = 1.3488
CELLPROFILER_MULTI_OTSU_BINS = 128
CELLPROFILER_LOG_MULTI_OTSU_BINS = 128
CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET = 0.0
_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


class CellProfilerThresholdAssignment(Enum):
    FOREGROUND = "Foreground"
    BACKGROUND = "Background"


class CellProfilerAveragingMethod(Enum):
    MEAN = "Mean"
    MEDIAN = "Median"
    MODE = "Mode"


class CellProfilerThresholdMethod(Enum):
    OTSU = "Otsu"
    MINIMUM_CROSS_ENTROPY = "Minimum Cross-Entropy"
    ROBUST_BACKGROUND = "Robust Background"
    MULTI_OTSU = "Multi-Otsu"
    SAUVOLA = "Sauvola"
    MAX_INTENSITY_PERCENTAGE = "Max Intensity Percentage"
    MANUAL = "Manual"
    MEASUREMENT = "Measurement"
    LI = "Li"
    TRIANGLE = "Triangle"
    ISODATA = "Isodata"


class CellProfilerOtsuMethod(Enum):
    TWO_CLASS = "Two classes"
    THREE_CLASS = "Three classes"


class CellProfilerThresholdScope(Enum):
    GLOBAL = "Global"
    ADAPTIVE = "Adaptive"


class CellProfilerVarianceMethod(Enum):
    STANDARD_DEVIATION = "Standard deviation"
    MEDIAN_ABSOLUTE_DEVIATION = "Median absolute deviation"


@dataclass(frozen=True, slots=True)
class CellProfilerThresholdDiagnostics:
    """CellProfiler threshold measurements emitted as runtime facts."""

    final_threshold: float
    original_threshold: float
    weighted_variance: float
    sum_of_entropies: float


def normalize_cellprofiler_image(image: np.ndarray) -> np.ndarray:
    """Return an image in CellProfiler's normalized pixel-data convention."""
    return image_payload_data(normalize_image_payload_intensity(image, dtype=np.float32))


def unit_interval_scale_for_threshold_diagnostics(
    image_data: np.ndarray,
    metadata: ImagePayloadMetadata,
) -> int | None:
    """Return a proof scale for exact unit-interval threshold diagnostics."""
    metadata_scale = metadata.unit_interval_intensity_scale_for_channel(0)
    if metadata_scale is not None and metadata_scale > 1:
        return int(metadata_scale)
    image_array = np.asarray(image_data)
    if not np.issubdtype(image_array.dtype, np.integer):
        return None
    scale = image_intensity_scale_for_dtype(image_array.dtype)
    if scale is None or scale <= 1:
        return None
    return int(scale)


def cellprofiler_get_global_threshold(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold_method: CellProfilerThresholdMethod | str = CellProfilerThresholdMethod.OTSU,
    threshold_min: float = 0,
    threshold_max: float = 1,
    threshold_correction_factor: float = 1,
    assign_middle_to_foreground: (
        CellProfilerThresholdAssignment | str
    ) = CellProfilerThresholdAssignment.FOREGROUND,
    log_transform: bool = False,
    **kwargs: object,
) -> float:
    """Compute one global threshold using independent CP-compatible semantics."""
    primitives = threshold_primitives()
    method = coerce_cellprofiler_enum(CellProfilerThresholdMethod, threshold_method)
    assignment = coerce_cellprofiler_enum(
        CellProfilerThresholdAssignment,
        assign_middle_to_foreground,
    )
    threshold_image = np.asarray(image, dtype=np.float32)
    if log_transform:
        threshold_image, conversion = primitives.log_transform(threshold_image)
    else:
        conversion = None
    threshold_mask = None if mask is None else np.asarray(mask, dtype=bool)
    values = threshold_image
    if mask is not None:
        values = values[threshold_mask]
    else:
        values = values.ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        threshold = 0.0
    elif np.all(values == values.ravel()[0]):
        threshold = float(values.ravel()[0])
    elif method is CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY:
        threshold = primitives.minimum_cross_entropy_threshold(
            threshold_image,
            threshold_mask,
        )
    elif method is CellProfilerThresholdMethod.LI:
        threshold = primitives.li_threshold(values)
    elif method is CellProfilerThresholdMethod.ROBUST_BACKGROUND:
        threshold = _get_threshold_robust_background(values, **kwargs)
    elif method is CellProfilerThresholdMethod.OTSU:
        threshold = primitives.otsu_threshold(values)
    elif method is CellProfilerThresholdMethod.MULTI_OTSU:
        bin_wanted = (
            0
            if assignment is CellProfilerThresholdAssignment.FOREGROUND
            else 1
        )
        nbins = int(kwargs.get("nbins", CELLPROFILER_MULTI_OTSU_BINS))
        thresholds = _threshold_multiotsu(
            values,
            nbins=nbins,
        )
        threshold = float(thresholds[bin_wanted])
        if log_transform:
            threshold += (
                _threshold_histogram_bin_width(values, nbins)
                * CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET
            )
    elif method is CellProfilerThresholdMethod.SAUVOLA:
        threshold = float(
            np.mean(
                primitives.sauvola_threshold_image(
                    values.reshape(1, -1),
                    window_size=int(kwargs.get("window_size", 15)),
                )
            )
        )
    elif method is CellProfilerThresholdMethod.TRIANGLE:
        threshold = primitives.triangle_threshold(values)
    elif method is CellProfilerThresholdMethod.ISODATA:
        threshold = primitives.isodata_threshold(values)
    elif method is CellProfilerThresholdMethod.MAX_INTENSITY_PERCENTAGE:
        threshold = float(np.max(values) * float(kwargs.get("fraction", 0.75)))
    else:
        raise NotImplementedError(f"Threshold method {method} not supported.")

    if conversion is not None:
        threshold = float(primitives.inverse_log_transform(threshold, conversion))
    threshold *= threshold_correction_factor
    return float(min(max(threshold, threshold_min), threshold_max))


def cellprofiler_get_adaptive_threshold(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold_method: CellProfilerThresholdMethod | str = CellProfilerThresholdMethod.OTSU,
    window_size: int = 50,
    threshold_min: float = 0,
    threshold_max: float = 1,
    threshold_correction_factor: float = 1,
    assign_middle_to_foreground: (
        CellProfilerThresholdAssignment | str
    ) = CellProfilerThresholdAssignment.FOREGROUND,
    global_limits: tuple[float, float] = (0.7, 1.5),
    log_transform: bool = False,
    **kwargs: object,
) -> np.ndarray:
    """Compute CP-style adaptive thresholds without depending on CP packages."""
    primitives = threshold_primitives()
    method = coerce_cellprofiler_enum(CellProfilerThresholdMethod, threshold_method)
    assignment = coerce_cellprofiler_enum(
        CellProfilerThresholdAssignment,
        assign_middle_to_foreground,
    )
    data = np.asarray(image, dtype=np.float32)
    if mask is not None:
        data = np.where(np.asarray(mask, dtype=bool), data, False)

    if log_transform:
        transformed, conversion = primitives.log_transform(data)
    else:
        transformed = data
        conversion = None

    if transformed.size == 0 or np.all(np.isnan(transformed)):
        thresholds = np.zeros_like(transformed)
    elif np.all(transformed == transformed.ravel()[0]):
        thresholds = np.full_like(transformed, transformed.ravel()[0])
    elif method is CellProfilerThresholdMethod.SAUVOLA:
        if window_size % 2 == 0:
            window_size += 1
        thresholds = primitives.sauvola_threshold_image(
            transformed,
            window_size=window_size,
        )
    else:
        thresholds = _adaptive_threshold_blocks(
            transformed,
            window_size=window_size,
            threshold_method=method,
            assign_middle_to_foreground=assignment,
            **kwargs,
        )

    global_threshold = cellprofiler_get_global_threshold(
        transformed,
        mask=None,
        threshold_method=method,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_correction_factor=threshold_correction_factor,
        assign_middle_to_foreground=assignment,
        log_transform=False,
        **kwargs,
    )
    if conversion is not None:
        thresholds = primitives.inverse_log_transform(thresholds, conversion)
        global_threshold = float(
            primitives.inverse_log_transform(global_threshold, conversion)
        )

    thresholds = thresholds * threshold_correction_factor
    t_min = max(threshold_min, global_threshold * global_limits[0])
    t_max = min(threshold_max, global_threshold * global_limits[1])
    thresholds[thresholds < t_min] = t_min
    thresholds[thresholds > t_max] = t_max
    return thresholds


def cellprofiler_apply_threshold(
    image: np.ndarray,
    *,
    threshold: float | np.ndarray,
    mask: np.ndarray | None = None,
    smoothing: float = 0,
) -> tuple[np.ndarray, float]:
    """Apply threshold with CP's mask-aware smoothing convention."""
    if smoothing == 0:
        thresholded = np.asarray(image) >= threshold
        if mask is None:
            return thresholded, 0.0
        return thresholded & np.asarray(mask, dtype=bool), 0.0

    resolved_mask = _resolved_threshold_mask(image, mask)
    blurred_image, sigma = _threshold_application_smoothed_image(
        image,
        resolved_mask,
        smoothing,
    )
    return (blurred_image >= threshold) & resolved_mask, sigma


def _resolved_threshold_mask(
    image: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    return (
        np.full(np.asarray(image).shape, True)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )


def _threshold_smoothed_image(
    image: np.ndarray,
    mask: np.ndarray | None,
    smoothing: float,
    threshold_method: CellProfilerThresholdMethod | None = None,
    log_transform: bool = False,
) -> tuple[np.ndarray, float]:
    resolved_mask = _resolved_threshold_mask(image, mask)
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdSmoothingBackendStrategy,
    )

    return ThresholdSmoothingBackendStrategy.for_memory_type().smooth_threshold_image(
        np.asarray(image),
        resolved_mask,
        smoothing,
        threshold_method=threshold_method,
        log_transform=log_transform,
    )


def _threshold_application_smoothed_image(
    image: np.ndarray,
    mask: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, float]:
    """Return the image CP thresholds against after threshold estimation."""
    from scipy import ndimage as ndi

    sigma = float(smoothing) / CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE
    if sigma <= 0.0:
        return np.asarray(image), 0.0
    image_array = np.asarray(image, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != image_array.shape:
        raise ValueError(
            "Threshold application mask must match the image shape; got "
            f"mask {mask_array.shape!r} for image {image_array.shape!r}."
        )
    full_mask = bool(np.all(mask_array))
    capture_array_fixture(
        "threshold_application",
        image=image_array,
        mask=mask_array,
        smoothing=np.asarray(smoothing, dtype=np.float64),
    )
    masked_image = image_array if full_mask else np.where(mask_array, image_array, 0.0)
    smoothed_image = ndi.gaussian_filter(
        masked_image,
        sigma=sigma,
        mode="constant",
        cval=0,
        truncate=4.0,
    )
    mask_weight = (
        _full_threshold_mask_weight(image_array.shape, sigma)
        if full_mask
        else ndi.gaussian_filter(
            mask_array.astype(np.float64),
            sigma=sigma,
            mode="constant",
            cval=0,
            truncate=4.0,
        )
    )
    if full_mask:
        smoothed_image /= mask_weight
        return smoothed_image, sigma
    output = np.zeros_like(image_array)
    valid = mask_weight != 0
    output[valid] = smoothed_image[valid] / mask_weight[valid]
    return (
        output,
        sigma,
    )


@lru_cache(maxsize=32)
def _full_threshold_mask_weight(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Return CP's threshold-application boundary weights for a full mask."""
    from scipy import ndimage as ndi

    return ndi.gaussian_filter(
        np.ones(shape, dtype=np.float64),
        sigma=sigma,
        mode="constant",
        cval=0,
        truncate=4.0,
    )


def _threshold_multiotsu(values: np.ndarray, *, nbins: int) -> np.ndarray:
    """Compute CP-compatible multi-Otsu thresholds for the observed value range."""
    if values.size == 0:
        return np.zeros((2,), dtype=float)
    return threshold_primitives().multiotsu_thresholds(values, nbins=nbins)


def _threshold_histogram_bin_width(values: np.ndarray, nbins: int) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    finite_values = values_array[np.isfinite(values_array)]
    if finite_values.size == 0 or nbins <= 0:
        return 0.0
    value_min = float(np.min(finite_values))
    value_max = float(np.max(finite_values))
    if value_max == value_min:
        return 0.0
    return (value_max - value_min) / float(nbins)


def _get_threshold_robust_background(
    image: np.ndarray,
    *,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: CellProfilerAveragingMethod | str = CellProfilerAveragingMethod.MEAN,
    variance_method: (
        CellProfilerVarianceMethod | str
    ) = CellProfilerVarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2,
    **_ignored: object,
) -> float:
    primitives = threshold_primitives()
    averaging_method = coerce_cellprofiler_enum(
        CellProfilerAveragingMethod,
        averaging_method,
    )
    variance_method = coerce_cellprofiler_enum(
        CellProfilerVarianceMethod,
        variance_method,
    )
    flat = np.asarray(image).flatten()
    if flat.size < 3:
        return 0.0
    flat.sort()
    if flat[0] == flat[-1]:
        return float(flat[0])
    low_chop = int(round(flat.size * lower_outlier_fraction))
    high_chop = flat.size - int(round(flat.size * upper_outlier_fraction))
    trimmed = flat if low_chop == 0 else flat[low_chop:high_chop]

    if averaging_method is CellProfilerAveragingMethod.MEAN:
        center = np.mean(trimmed)
    elif averaging_method is CellProfilerAveragingMethod.MEDIAN:
        center = np.median(trimmed)
    else:
        center = primitives.binned_mode(trimmed)

    if variance_method is CellProfilerVarianceMethod.STANDARD_DEVIATION:
        spread = np.std(trimmed)
    else:
        spread = primitives.mad(trimmed)
    return float(center + spread * number_of_deviations)


def _adaptive_threshold_blocks(
    image: np.ndarray,
    *,
    window_size: int,
    threshold_method: CellProfilerThresholdMethod,
    assign_middle_to_foreground: CellProfilerThresholdAssignment,
    **kwargs: object,
) -> np.ndarray:
    image_size = np.array(image.shape[:2], dtype=int)
    nblocks = image_size // window_size
    if any(count < 2 for count in nblocks):
        raise ValueError(
            "Adaptive window cannot exceed 50% of an image dimension.\n"
            f"Window of {window_size}px is too large for a "
            f"{image_size[1]}x{image_size[0]} image"
        )

    increment = np.array(image_size, dtype=float) / np.array(nblocks, dtype=float)
    block_threshold = np.zeros([nblocks[0], nblocks[1]])
    for row in range(nblocks[0]):
        row_start = int(row * increment[0])
        row_stop = int((row + 1) * increment[0])
        for column in range(nblocks[1]):
            column_start = int(column * increment[1])
            column_stop = int((column + 1) * increment[1])
            block = image[row_start:row_stop, column_start:column_stop]
            block = block[~np.logical_not(block)]
            block_threshold[row, column] = _block_threshold(
                block,
                threshold_method=threshold_method,
                assign_middle_to_foreground=assign_middle_to_foreground,
                **kwargs,
            )

    spline_order = min(3, int(np.min(nblocks)) - 1)
    row_start = int(increment[0] / 2)
    row_end = int((nblocks[0] - 0.5) * increment[0])
    column_start = int(increment[1] / 2)
    column_end = int((nblocks[1] - 0.5) * increment[1])
    interpolation = scipy.interpolate.RectBivariateSpline(
        np.linspace(row_start, row_end, nblocks[0]),
        np.linspace(column_start, column_end, nblocks[1]),
        block_threshold,
        bbox=(0.5, image.shape[0] - 0.5, 0.5, image.shape[1] - 0.5),
        kx=spline_order,
        ky=spline_order,
    )
    return interpolation(
        np.linspace(0.5, int(nblocks[0] * increment[0]) - 0.5, image.shape[0]),
        np.linspace(0.5, int(nblocks[1] * increment[1]) - 0.5, image.shape[1]),
    )


def _block_threshold(
    block: np.ndarray,
    *,
    threshold_method: CellProfilerThresholdMethod,
    assign_middle_to_foreground: CellProfilerThresholdAssignment,
    **kwargs: object,
) -> float:
    if block.size == 0:
        return 0.0
    if np.all(block == block[0]):
        return float(block[0])
    if (
        threshold_method is CellProfilerThresholdMethod.MULTI_OTSU
        and np.unique(block).size < 3
    ):
        return threshold_primitives().otsu_threshold(block)
    return cellprofiler_get_global_threshold(
        block,
        threshold_method=threshold_method,
        assign_middle_to_foreground=assign_middle_to_foreground,
        threshold_min=0,
        threshold_max=1,
        threshold_correction_factor=1,
        log_transform=False,
        **kwargs,
    )


def _masked_linear_filter(
    image: np.ndarray,
    mask: np.ndarray,
    operation,
) -> np.ndarray:
    masked_image = np.zeros(image.shape, dtype=image.dtype)
    masked_image[mask] = image[mask]
    weights = operation(mask.astype(float))
    filtered = operation(masked_image)
    return filtered / (weights + np.finfo(float).eps)


def _threshold_method_for_class_count(
    threshold_method: CellProfilerThresholdMethod,
    otsu_class_count: CellProfilerOtsuMethod,
) -> CellProfilerThresholdMethod:
    if (
        threshold_method is CellProfilerThresholdMethod.OTSU
        and otsu_class_count is CellProfilerOtsuMethod.THREE_CLASS
    ):
        return CellProfilerThresholdMethod.MULTI_OTSU
    return threshold_method


def _threshold_method_kwargs(
    threshold_method: CellProfilerThresholdMethod,
    *,
    lower_outlier_fraction: float,
    upper_outlier_fraction: float,
    averaging_method: CellProfilerAveragingMethod,
    variance_method: CellProfilerVarianceMethod,
    number_of_deviations: float,
) -> dict[str, object]:
    """Return kwargs that are meaningful for the selected threshold algorithm."""
    if threshold_method is not CellProfilerThresholdMethod.ROBUST_BACKGROUND:
        return {}
    return {
        "lower_outlier_fraction": lower_outlier_fraction,
        "upper_outlier_fraction": upper_outlier_fraction,
        "averaging_method": averaging_method,
        "variance_method": variance_method,
        "number_of_deviations": number_of_deviations,
    }


def _clip_threshold(threshold: float, threshold_min: float, threshold_max: float) -> float:
    return float(min(max(float(threshold), threshold_min), threshold_max))


def _global_threshold_uses_raw_image(
    *,
    effective_method: CellProfilerThresholdMethod,
    log_transform: bool,
) -> bool:
    if effective_method in (
        CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
        CellProfilerThresholdMethod.LI,
        CellProfilerThresholdMethod.OTSU,
    ):
        return True
    return effective_method is CellProfilerThresholdMethod.MULTI_OTSU and log_transform


def _global_threshold_selection_image(
    *,
    effective_method: CellProfilerThresholdMethod,
    log_transform: bool,
    image: np.ndarray,
    threshold_image: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the image/kwargs used to estimate a global threshold."""
    if _global_threshold_uses_raw_image(
        effective_method=effective_method,
        log_transform=log_transform,
    ):
        if effective_method is CellProfilerThresholdMethod.MULTI_OTSU:
            return np.asarray(image), {"nbins": CELLPROFILER_LOG_MULTI_OTSU_BINS}
        return np.asarray(image), {}
    return threshold_image, {}


def cellprofiler_threshold(
    image: np.ndarray,
    *,
    use_advanced_settings: bool,
    threshold_scope: CellProfilerThresholdScope,
    threshold_method: CellProfilerThresholdMethod,
    otsu_class_count: CellProfilerOtsuMethod,
    assign_middle_to_foreground: CellProfilerThresholdAssignment,
    log_transform: bool,
    threshold_correction_factor: float,
    threshold_min: float,
    threshold_max: float,
    threshold_smoothing_scale: float,
    adaptive_window_size: int,
    lower_outlier_fraction: float,
    upper_outlier_fraction: float,
    averaging_method: CellProfilerAveragingMethod,
    variance_method: CellProfilerVarianceMethod,
    number_of_deviations: float,
    manual_threshold: float,
    mask: np.ndarray | None = None,
    smooth_threshold_application: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Apply CellProfiler threshold semantics without a CP workspace."""
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    threshold_mask = None if mask is None else np.asarray(mask, dtype=bool)
    threshold_scope = coerce_cellprofiler_enum(
        CellProfilerThresholdScope,
        threshold_scope,
    )
    threshold_method = coerce_cellprofiler_enum(
        CellProfilerThresholdMethod,
        threshold_method,
    )
    otsu_class_count = coerce_cellprofiler_enum(
        CellProfilerOtsuMethod,
        otsu_class_count,
    )
    assign_middle_to_foreground = coerce_cellprofiler_enum(
        CellProfilerThresholdAssignment,
        assign_middle_to_foreground,
    )
    averaging_method = coerce_cellprofiler_enum(
        CellProfilerAveragingMethod,
        averaging_method,
    )
    variance_method = coerce_cellprofiler_enum(
        CellProfilerVarianceMethod,
        variance_method,
    )
    _log_profile(
        "threshold_coerce_settings",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold",
    )

    if not use_advanced_settings:
        threshold_scope = CellProfilerThresholdScope.GLOBAL
        threshold_method = CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY
        log_transform = False
        threshold_smoothing_scale = CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE

    if threshold_method is CellProfilerThresholdMethod.MEASUREMENT:
        raise NotImplementedError(
            "Measurement-based thresholding requires a prior measurement source."
        )

    effective_method = _threshold_method_for_class_count(
        threshold_method,
        otsu_class_count,
    )
    threshold_image = np.asarray(image)
    if threshold_method is CellProfilerThresholdMethod.MANUAL:
        final_threshold: float | np.ndarray = float(manual_threshold)
        original_threshold = float(manual_threshold)
    else:
        phase_started_at = time.perf_counter()
        threshold_kwargs = _threshold_method_kwargs(
            effective_method,
            lower_outlier_fraction=lower_outlier_fraction,
            upper_outlier_fraction=upper_outlier_fraction,
            averaging_method=averaging_method,
            variance_method=variance_method,
            number_of_deviations=number_of_deviations,
        )
        _log_profile(
            "threshold_method_kwargs",
            time.perf_counter() - phase_started_at,
            function="cellprofiler_threshold",
            method=effective_method.value,
        )
        if threshold_scope is CellProfilerThresholdScope.ADAPTIVE:
            phase_started_at = time.perf_counter()
            final_threshold = cellprofiler_get_adaptive_threshold(
                threshold_image,
                mask=threshold_mask,
                threshold_method=effective_method,
                window_size=adaptive_window_size,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_correction_factor=threshold_correction_factor,
                assign_middle_to_foreground=assign_middle_to_foreground,
                log_transform=log_transform,
                **threshold_kwargs,
            )
            _log_profile(
                "threshold_adaptive_final",
                time.perf_counter() - phase_started_at,
                function="cellprofiler_threshold",
                method=effective_method.value,
            )
            phase_started_at = time.perf_counter()
            original_threshold = float(
                np.mean(
                    np.atleast_1d(
                        cellprofiler_get_adaptive_threshold(
                            threshold_image,
                            mask=threshold_mask,
                            threshold_method=effective_method,
                            window_size=adaptive_window_size,
                            threshold_min=(
                                threshold_min if not use_advanced_settings else 0
                            ),
                            threshold_max=(
                                threshold_max if not use_advanced_settings else 1
                            ),
                            threshold_correction_factor=(
                                threshold_correction_factor
                                if not use_advanced_settings
                                else 1
                            ),
                            assign_middle_to_foreground=assign_middle_to_foreground,
                            log_transform=log_transform,
                            **threshold_kwargs,
                        )
                    )
                )
            )
            _log_profile(
                "threshold_adaptive_original",
                time.perf_counter() - phase_started_at,
                function="cellprofiler_threshold",
                method=effective_method.value,
            )
        else:
            selection_image, selection_kwargs = _global_threshold_selection_image(
                effective_method=effective_method,
                log_transform=log_transform,
                image=image,
                threshold_image=threshold_image,
            )
            phase_started_at = time.perf_counter()
            raw_threshold = cellprofiler_get_global_threshold(
                selection_image,
                mask=threshold_mask,
                threshold_method=effective_method,
                threshold_min=0,
                threshold_max=1,
                threshold_correction_factor=1,
                assign_middle_to_foreground=assign_middle_to_foreground,
                log_transform=log_transform,
                **threshold_kwargs,
                **selection_kwargs,
            )
            _log_profile(
                "threshold_global_raw",
                time.perf_counter() - phase_started_at,
                function="cellprofiler_threshold",
                method=effective_method.value,
                pixels=np.asarray(selection_image).size,
            )
            phase_started_at = time.perf_counter()
            final_threshold = _clip_threshold(
                raw_threshold * threshold_correction_factor,
                threshold_min,
                threshold_max,
            )
            original_threshold = (
                final_threshold
                if not use_advanced_settings
                else _clip_threshold(raw_threshold, 0, 1)
            )
            _log_profile(
                "threshold_clip",
                time.perf_counter() - phase_started_at,
                function="cellprofiler_threshold",
                method=effective_method.value,
            )

    application_image = image
    application_smoothing = (
        threshold_smoothing_scale if smooth_threshold_application else 0.0
    )
    phase_started_at = time.perf_counter()
    binary, _sigma = cellprofiler_apply_threshold(
        application_image,
        threshold=final_threshold,
        mask=threshold_mask,
        smoothing=application_smoothing,
    )
    _log_profile(
        "threshold_apply",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold",
        smoothing=float(application_smoothing),
    )
    phase_started_at = time.perf_counter()
    if threshold_mask is not None:
        binary = np.asarray(binary, dtype=bool) & threshold_mask
    result = (
        binary.astype(bool),
        float(np.mean(np.atleast_1d(final_threshold))),
        float(original_threshold),
    )
    _log_profile(
        "threshold_finalize",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold",
    )
    _log_profile(
        "threshold_total",
        time.perf_counter() - total_started_at,
        function="cellprofiler_threshold",
    )
    return result


def cellprofiler_threshold_diagnostics(
    image: np.ndarray,
    binary: np.ndarray,
    *,
    final_threshold: float,
    original_threshold: float,
    mask: np.ndarray | None = None,
    proven_unit_interval_scale: int | None = None,
) -> CellProfilerThresholdDiagnostics:
    """Return CellProfiler's image-level threshold quality measurements."""
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    measurement_mask = None if mask is None else np.asarray(mask, dtype=bool)
    binary_image = np.asarray(binary, dtype=bool)
    if capture_enabled():
        capture_array_fixture(
            "threshold_diagnostics",
            image=np.asarray(image),
            binary=binary_image,
            mask=(
                np.ones_like(binary_image, dtype=bool)
                if measurement_mask is None
                else measurement_mask
            ),
            final_threshold=np.asarray(final_threshold, dtype=np.float64),
            original_threshold=np.asarray(original_threshold, dtype=np.float64),
        )
    _log_profile(
        "threshold_diagnostics_prepare",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold_diagnostics",
    )
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdDiagnosticsBackendStrategy,
    )

    phase_started_at = time.perf_counter()
    weighted_variance, sum_of_entropies = (
        ThresholdDiagnosticsBackendStrategy.for_memory_type().diagnostics(
            image,
            measurement_mask,
            binary_image,
            proven_unit_interval_scale=proven_unit_interval_scale,
        )
    )
    _log_profile(
        "threshold_diagnostics_backend",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold_diagnostics",
    )
    phase_started_at = time.perf_counter()
    result = CellProfilerThresholdDiagnostics(
        final_threshold=float(final_threshold),
        original_threshold=float(original_threshold),
        weighted_variance=float(np.mean(np.atleast_1d(weighted_variance))),
        sum_of_entropies=float(np.mean(np.atleast_1d(sum_of_entropies))),
    )
    _log_profile(
        "threshold_diagnostics_finalize",
        time.perf_counter() - phase_started_at,
        function="cellprofiler_threshold_diagnostics",
    )
    _log_profile(
        "threshold_diagnostics_total",
        time.perf_counter() - total_started_at,
        function="cellprofiler_threshold_diagnostics",
    )
    return result
