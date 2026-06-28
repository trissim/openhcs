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


def quantized_threshold_codes(image: np.ndarray, scale: int) -> np.ndarray:
    """Return dense unit-interval integer codes for quantized diagnostics."""
    code_dtype = np.uint8 if scale <= int(np.iinfo(np.uint8).max) else np.uint16
    return np.ascontiguousarray(
        np.rint(image * int(scale)).astype(code_dtype, copy=False)
    )


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
    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
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
                    clipped = minval
                    if clipped < delta:
                        clipped = delta
                    elif clipped > 1.0:
                        clipped = 1.0
                    weighted_log_value = minval_log
                    entropy_log_value = math.log2(clipped)
                    log_delta_value = math.log2(clipped + delta)
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
    scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    for y in range(height):
        for x in range(width):
            code = codes[y, x]
            value = values[code]
            if value < minval:
                clipped = minval
                if clipped < delta:
                    clipped = delta
                elif clipped > 1.0:
                    clipped = 1.0
                entropy_log_value = math.log2(clipped)
                log_delta_value = math.log2(clipped + delta)
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
            bin_index = int((log_smoothed_value - lower) * scale)
            if bin_index < 0:
                continue
            if bin_index >= CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                if bin_index == CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                    bin_index = CELLPROFILER_THRESHOLD_ENTROPY_BINS - 1
                else:
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

    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    lower = np.inf
    upper = -np.inf
    for y in range(height):
        for x in range(width):
            code = codes[y, x]
            value = values[code]
            if value < minval:
                clipped = minval
                if clipped < delta:
                    clipped = delta
                elif clipped > 1.0:
                    clipped = 1.0
                log_value = math.log2(clipped)
                log_delta_value = math.log2(clipped + delta)
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
    histogram_scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    for y in range(y0, y1):
        for x in range(x0, x1):
            code = codes[y, x]
            value = values[code]
            if value < minval:
                clipped = minval
                if clipped < delta:
                    clipped = delta
                elif clipped > 1.0:
                    clipped = 1.0
                log_value = math.log2(clipped)
                log_delta_value = math.log2(clipped + delta)
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
            bin_index = int((log_smoothed_value - lower) * histogram_scale)
            if bin_index < 0:
                continue
            if bin_index >= CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                if bin_index == CELLPROFILER_THRESHOLD_ENTROPY_BINS:
                    bin_index = CELLPROFILER_THRESHOLD_ENTROPY_BINS - 1
                else:
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
