"""Quantized threshold diagnostic kernels for CellProfiler thresholding."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_diagnostics import (
    CELLPROFILER_THRESHOLD_ENTROPY_BINS,
    CELLPROFILER_THRESHOLD_ENTROPY_DELTA,
    _histogram_entropy_numba,
)


class QuantizedThresholdDiagnosticContext(NamedTuple):
    """Shared quantized diagnostic arrays for CellProfiler threshold scoring."""

    codes: np.ndarray
    binary_image: np.ndarray
    noise: np.ndarray
    values: np.ndarray
    weighted_log_values: np.ndarray
    entropy_log_values: np.ndarray
    entropy_log_delta_values: np.ndarray


class QuantizedThresholdCodebook(NamedTuple):
    """Exact runtime values associated with one proven quantized image."""

    codes: np.ndarray
    values: np.ndarray


@njit(cache=True)
def _populate_quantized_threshold_codebook_numba(
    image: np.ndarray,
    scale: int,
    codes: np.ndarray,
    values: np.ndarray,
    populated: np.ndarray,
) -> bool:
    flat_image = image.ravel()
    flat_codes = codes.ravel()
    scale_float = float(scale)
    for index in range(flat_image.size):
        value = flat_image[index]
        if not np.isfinite(value):
            return False
        code = int(np.rint(value * scale_float))
        if code < 0 or code > scale:
            return False
        if populated[code]:
            if values[code] != value:
                return False
        else:
            values[code] = value
            populated[code] = True
        flat_codes[index] = code
    return True


def quantized_threshold_codebook(
    image: np.ndarray, scale: int
) -> QuantizedThresholdCodebook | None:
    """Return exact per-code runtime values for a proven quantized image."""
    image_array = np.asarray(image)
    scale_value = int(scale)
    if scale_value <= 0 or not np.issubdtype(image_array.dtype, np.floating):
        return None
    code_dtype = (
        np.uint8 if scale_value <= int(np.iinfo(np.uint8).max) else np.uint16
    )
    codes = np.empty(image_array.shape, dtype=code_dtype)
    values = np.zeros(scale_value + 1, dtype=image_array.dtype)
    populated = np.zeros(scale_value + 1, dtype=np.bool_)
    if not _populate_quantized_threshold_codebook_numba(
        image_array,
        scale_value,
        codes,
        values,
        populated,
    ):
        return None
    return QuantizedThresholdCodebook(
        codes=np.ascontiguousarray(codes),
        values=values,
    )


def quantized_threshold_diagnostic_context(
    image: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
    scale: int,
) -> QuantizedThresholdDiagnosticContext | None:
    """Build exact producer-dtype log tables for the quantized fast path."""
    codebook = quantized_threshold_codebook(image, scale)
    if codebook is None:
        return None
    values = codebook.values
    weighted_log_values = np.zeros(values.shape, dtype=np.float64)
    positive_values = values > 0.0
    weighted_log_values[positive_values] = np.log2(
        values[positive_values].astype(np.float64)
    )
    delta = np.asarray(CELLPROFILER_THRESHOLD_ENTROPY_DELTA, dtype=values.dtype)
    entropy_values = np.clip(
        values,
        delta,
        np.asarray(1.0, dtype=values.dtype),
    )
    return QuantizedThresholdDiagnosticContext(
        codes=codebook.codes,
        binary_image=np.ascontiguousarray(binary_image),
        noise=noise,
        values=values,
        weighted_log_values=weighted_log_values,
        entropy_log_values=np.log2(entropy_values),
        entropy_log_delta_values=np.log2(entropy_values + delta),
    )


def exact_quantized_threshold_codes(
    image: np.ndarray, scale: int
) -> np.ndarray | None:
    """Return quantized codes only when image values exactly match ``scale``."""
    scale_value = int(scale)
    if scale_value <= 0:
        return None
    scaled = np.asarray(image) * scale_value
    rounded = np.rint(scaled)
    if not np.array_equal(scaled, rounded):
        return None
    code_dtype = (
        np.uint8 if scale_value <= int(np.iinfo(np.uint8).max) else np.uint16
    )
    return np.ascontiguousarray(rounded.astype(code_dtype, copy=False))


@njit(cache=True, inline="always")
def _numpy_uniform_histogram_bin_index_numba(
    value: float,
    lower: float,
    upper: float,
    bin_edges: np.ndarray,
) -> int:
    """Return NumPy's corrected uniform-histogram bin index."""
    bin_count = CELLPROFILER_THRESHOLD_ENTROPY_BINS
    bin_index = int(((value - lower) / (upper - lower)) * float(bin_count))
    if bin_index == bin_count:
        bin_index -= 1
    if bin_index < 0 or bin_index >= bin_count:
        return -1
    if value < bin_edges[bin_index]:
        bin_index -= 1
    if bin_index < 0:
        return -1
    if value >= bin_edges[bin_index + 1] and bin_index != bin_count - 1:
        bin_index += 1
    return bin_index


@njit(cache=True, inline="always")
def _producer_dtype_clamped_entropy_logs_numba(
    minval: float,
    values: np.ndarray,
) -> tuple[float, float]:
    """Return clamp logs after NumPy's producer-dtype scalar cast."""
    typed_scalars = np.empty(2, dtype=values.dtype)
    typed_scalars[0] = minval
    typed_scalars[1] = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    clipped = typed_scalars[0]
    delta = typed_scalars[1]
    if clipped < delta:
        clipped = delta
    elif clipped > 1.0:
        clipped = 1.0
    clipped_plus_delta = clipped + delta
    typed_scalars[0] = math.log2(float(clipped))
    typed_scalars[1] = math.log2(float(clipped_plus_delta))
    return float(typed_scalars[0]), float(typed_scalars[1])


@njit(cache=True)
def _threshold_diagnostics_unmasked_finite_quantized_numba(
    context: QuantizedThresholdDiagnosticContext,
) -> tuple[float, float]:
    codes = context.codes
    binary_image = context.binary_image
    noise = context.noise
    values = context.values
    weighted_log_values = context.weighted_log_values
    entropy_log_values = context.entropy_log_values
    entropy_log_delta_values = context.entropy_log_delta_values
    height, width = codes.shape
    if height == 0 or width == 0:
        return 0.0, 0.0

    max_value = values[codes[0, 0]]
    for y in range(height):
        for x in range(width):
            value = values[codes[y, x]]
            if value > max_value:
                max_value = value

    weighted_variance = 0.0
    minval = max_value / 256.0
    minval_log = 0.0
    (
        clamped_entropy_log,
        clamped_entropy_log_delta,
    ) = _producer_dtype_clamped_entropy_logs_numba(minval, values)
    lower = np.inf
    upper = -np.inf
    foreground_count = 0
    background_count = 0
    if minval > 0.0:
        minval_log = math.log2(minval)
        fg_count = 0
        bg_count = 0
        fg_sum = 0.0
        bg_sum = 0.0
        fg_sumsq = 0.0
        bg_sumsq = 0.0
        for y in range(height):
            for x in range(width):
                code = codes[y, x]
                value = values[code]
                if value < minval:
                    weighted_log_value = minval_log
                    entropy_log_value = clamped_entropy_log
                    log_delta_value = clamped_entropy_log_delta
                else:
                    weighted_log_value = weighted_log_values[code]
                    entropy_log_value = entropy_log_values[code]
                    log_delta_value = entropy_log_delta_values[code]
                noise_value = noise[y, x]
                log_smoothed_value = (
                    log_delta_value * noise_value
                    + (1.0 - noise_value) * entropy_log_value
                )
                if log_smoothed_value > 0.0:
                    log_smoothed_value = 0.0
                if log_smoothed_value < lower:
                    lower = log_smoothed_value
                if log_smoothed_value > upper:
                    upper = log_smoothed_value
                if binary_image[y, x]:
                    fg_count += 1
                    foreground_count += 1
                    fg_sum += weighted_log_value
                    fg_sumsq += weighted_log_value * weighted_log_value
                else:
                    bg_count += 1
                    background_count += 1
                    bg_sum += weighted_log_value
                    bg_sumsq += weighted_log_value * weighted_log_value

        if fg_count == 0 and bg_count == 0:
            weighted_variance = 0.0
        elif fg_count == 0:
            bg_mean = bg_sum / bg_count
            weighted_variance = bg_sumsq / bg_count - bg_mean * bg_mean
        elif bg_count == 0:
            fg_mean = fg_sum / fg_count
            weighted_variance = fg_sumsq / fg_count - fg_mean * fg_mean
        else:
            fg_mean = fg_sum / fg_count
            bg_mean = bg_sum / bg_count
            fg_variance = fg_sumsq / fg_count - fg_mean * fg_mean
            bg_variance = bg_sumsq / bg_count - bg_mean * bg_mean
            weighted_variance = (
                fg_variance * fg_count + bg_variance * bg_count
            ) / (fg_count + bg_count)

    if minval == 0.0:
        return weighted_variance, 0.0

    if upper == lower:
        return weighted_variance, math.log2(float(foreground_count + background_count))
    if foreground_count == 0 or background_count == 0:
        return weighted_variance, 0.0

    foreground_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    background_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    bin_edges = np.linspace(
        lower,
        upper,
        CELLPROFILER_THRESHOLD_ENTROPY_BINS + 1,
    )
    for y in range(height):
        for x in range(width):
            code = codes[y, x]
            value = values[code]
            if value < minval:
                entropy_log_value = clamped_entropy_log
                log_delta_value = clamped_entropy_log_delta
            else:
                entropy_log_value = entropy_log_values[code]
                log_delta_value = entropy_log_delta_values[code]
            noise_value = noise[y, x]
            log_smoothed_value = (
                log_delta_value * noise_value
                + (1.0 - noise_value) * entropy_log_value
            )
            if log_smoothed_value > 0.0:
                log_smoothed_value = 0.0
            bin_index = _numpy_uniform_histogram_bin_index_numba(
                log_smoothed_value,
                lower,
                upper,
                bin_edges,
            )
            if bin_index < 0:
                continue
            if binary_image[y, x]:
                foreground_hist[bin_index] += 1
            else:
                background_hist[bin_index] += 1

    return weighted_variance, _histogram_entropy_numba(
        foreground_hist,
        foreground_count,
    ) + _histogram_entropy_numba(
        background_hist,
        background_count,
    )


@njit(cache=True)
def _threshold_diagnostics_rectangular_mask_quantized_numba(
    context: QuantizedThresholdDiagnosticContext,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[float, float]:
    codes = context.codes
    binary_image = context.binary_image
    noise = context.noise
    values = context.values
    weighted_log_values = context.weighted_log_values
    entropy_log_values = context.entropy_log_values
    entropy_log_delta_values = context.entropy_log_delta_values
    height, width = codes.shape
    if height == 0 or width == 0 or y0 >= y1 or x0 >= x1:
        return 0.0, 0.0

    masked_max = values[codes[y0, x0]]
    for y in range(y0, y1):
        for x in range(x0, x1):
            value = values[codes[y, x]]
            if value > masked_max:
                masked_max = value

    weighted_variance = 0.0
    minval = masked_max / 256.0
    (
        clamped_entropy_log,
        clamped_entropy_log_delta,
    ) = _producer_dtype_clamped_entropy_logs_numba(minval, values)
    if minval > 0.0:
        minval_log = math.log2(minval)
        fg_count = 0
        bg_count = 0
        fg_sum = 0.0
        bg_sum = 0.0
        fg_sumsq = 0.0
        bg_sumsq = 0.0
        for y in range(y0, y1):
            for x in range(x0, x1):
                code = codes[y, x]
                value = values[code]
                log_value = minval_log if value < minval else weighted_log_values[code]
                if binary_image[y, x]:
                    fg_count += 1
                    fg_sum += log_value
                    fg_sumsq += log_value * log_value
                else:
                    bg_count += 1
                    bg_sum += log_value
                    bg_sumsq += log_value * log_value

        if fg_count == 0 and bg_count == 0:
            weighted_variance = 0.0
        elif fg_count == 0:
            bg_mean = bg_sum / bg_count
            weighted_variance = bg_sumsq / bg_count - bg_mean * bg_mean
        elif bg_count == 0:
            fg_mean = fg_sum / fg_count
            weighted_variance = fg_sumsq / fg_count - fg_mean * fg_mean
        else:
            fg_mean = fg_sum / fg_count
            bg_mean = bg_sum / bg_count
            fg_variance = fg_sumsq / fg_count - fg_mean * fg_mean
            bg_variance = bg_sumsq / bg_count - bg_mean * bg_mean
            weighted_variance = (
                fg_variance * fg_count + bg_variance * bg_count
            ) / (fg_count + bg_count)

    if minval == 0.0:
        return weighted_variance, 0.0

    lower = np.inf
    upper = -np.inf
    for y in range(height):
        for x in range(width):
            code = codes[y, x]
            value = values[code]
            if value < minval:
                log_value = clamped_entropy_log
                log_delta_value = clamped_entropy_log_delta
            else:
                log_value = entropy_log_values[code]
                log_delta_value = entropy_log_delta_values[code]
            noise_value = noise[y, x]
            log_smoothed_value = (
                log_delta_value * noise_value
                + (1.0 - noise_value) * log_value
            )
            if log_smoothed_value > 0.0:
                log_smoothed_value = 0.0
            if log_smoothed_value < lower:
                lower = log_smoothed_value
            if log_smoothed_value > upper:
                upper = log_smoothed_value

    foreground_count = 0
    background_count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if binary_image[y, x]:
                foreground_count += 1
            else:
                background_count += 1

    if upper == lower:
        return weighted_variance, math.log2(float(foreground_count + background_count))
    if foreground_count == 0 or background_count == 0:
        return weighted_variance, 0.0

    foreground_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    background_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    bin_edges = np.linspace(
        lower,
        upper,
        CELLPROFILER_THRESHOLD_ENTROPY_BINS + 1,
    )
    for y in range(y0, y1):
        for x in range(x0, x1):
            code = codes[y, x]
            value = values[code]
            if value < minval:
                log_value = clamped_entropy_log
                log_delta_value = clamped_entropy_log_delta
            else:
                log_value = entropy_log_values[code]
                log_delta_value = entropy_log_delta_values[code]
            noise_value = noise[y, x]
            log_smoothed_value = (
                log_delta_value * noise_value
                + (1.0 - noise_value) * log_value
            )
            if log_smoothed_value > 0.0:
                log_smoothed_value = 0.0
            bin_index = _numpy_uniform_histogram_bin_index_numba(
                log_smoothed_value,
                lower,
                upper,
                bin_edges,
            )
            if bin_index < 0:
                continue
            if binary_image[y, x]:
                foreground_hist[bin_index] += 1
            else:
                background_hist[bin_index] += 1

    return weighted_variance, _histogram_entropy_numba(
        foreground_hist,
        foreground_count,
    ) + _histogram_entropy_numba(
        background_hist,
        background_count,
    )
