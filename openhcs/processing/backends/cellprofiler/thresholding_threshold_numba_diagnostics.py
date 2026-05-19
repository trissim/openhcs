"""Threshold diagnostic kernels for CellProfiler-compatible thresholding."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from numba import njit


CELLPROFILER_THRESHOLD_ENTROPY_DELTA = 2.0 ** -8
CELLPROFILER_THRESHOLD_ENTROPY_BINS = 256


@dataclass(frozen=True, slots=True)
class QuantizedThresholdLogTables:
    """Log lookup tables for CellProfiler threshold diagnostics."""

    values: np.ndarray
    weighted_log_values: np.ndarray
    entropy_log_values: np.ndarray
    entropy_log_delta_values: np.ndarray


@dataclass(frozen=True, slots=True)
class RectangularMaskDomain:
    """The true region of a mask that is exactly one filled 2D rectangle."""

    y: slice
    x: slice

    @property
    def slices(self) -> tuple[slice, slice]:
        return self.y, self.x


def _rectangular_mask_domain(mask: np.ndarray) -> RectangularMaskDomain | None:
    """Return the true rectangle for masks that are exactly one filled rectangle."""
    if mask.ndim != 2:
        return None
    row_indices = np.flatnonzero(np.any(mask, axis=1))
    if row_indices.size == 0:
        return None
    column_indices = np.flatnonzero(np.any(mask, axis=0))
    y0 = int(row_indices[0])
    y1 = int(row_indices[-1]) + 1
    x0 = int(column_indices[0])
    x1 = int(column_indices[-1]) + 1
    if int(np.sum(mask)) != (y1 - y0) * (x1 - x0):
        return None
    return RectangularMaskDomain(slice(y0, y1), slice(x0, x1))


@lru_cache(maxsize=8)
def _quantized_log_tables(
    scale: int,
) -> QuantizedThresholdLogTables:
    codes = np.arange(int(scale) + 1, dtype=np.float32)
    values = (codes / np.float32(scale)).astype(np.float64, copy=False)
    weighted_log_values = np.zeros_like(values)
    positive_values = values > 0.0
    weighted_log_values[positive_values] = np.log2(values[positive_values])
    entropy_values = np.clip(values, CELLPROFILER_THRESHOLD_ENTROPY_DELTA, 1.0)
    return QuantizedThresholdLogTables(
        values=values,
        weighted_log_values=weighted_log_values,
        entropy_log_values=np.log2(entropy_values),
        entropy_log_delta_values=np.log2(
            entropy_values + CELLPROFILER_THRESHOLD_ENTROPY_DELTA
        ),
    )


@njit(cache=True)
def _threshold_diagnostics_unmasked_finite_quantized_numba(
    codes: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
    values: np.ndarray,
    weighted_log_values: np.ndarray,
    entropy_log_values: np.ndarray,
    entropy_log_delta_values: np.ndarray,
) -> tuple[float, float]:
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
    codes: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
    values: np.ndarray,
    weighted_log_values: np.ndarray,
    entropy_log_values: np.ndarray,
    entropy_log_delta_values: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[float, float]:
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


@njit(cache=True)
def _threshold_diagnostics_unmasked_finite_numba(
    image: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
) -> tuple[float, float]:
    height, width = image.shape
    if height == 0 or width == 0:
        return 0.0, 0.0

    max_value = image[0, 0]
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value > max_value:
                max_value = value

    weighted_variance = 0.0
    minval = max_value / 256.0
    if minval != 0.0:
        fg_count = 0
        bg_count = 0
        fg_sum = 0.0
        bg_sum = 0.0
        fg_sumsq = 0.0
        bg_sumsq = 0.0
        for y in range(height):
            for x in range(width):
                value = image[y, x]
                if value < minval:
                    value = minval
                log_value = math.log2(value)
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
    foreground_count = 0
    background_count = 0
    log_smoothed = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < minval:
                value = minval
            if value < delta:
                clipped = delta
            elif value > 1.0:
                clipped = 1.0
            else:
                clipped = value

            noise_value = noise[y, x]
            log_smoothed_value = (
                math.log2(clipped + delta) * noise_value
                + (1.0 - noise_value) * math.log2(clipped)
            )
            if log_smoothed_value > 0.0:
                log_smoothed_value = 0.0
            log_smoothed[y, x] = log_smoothed_value
            if log_smoothed_value < lower:
                lower = log_smoothed_value
            if log_smoothed_value > upper:
                upper = log_smoothed_value
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
    scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    for y in range(height):
        for x in range(width):
            bin_index = int((log_smoothed[y, x] - lower) * scale)
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
def _threshold_diagnostics_numba(
    image: np.ndarray,
    mask: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
) -> tuple[float, float]:
    return (
        _threshold_weighted_variance_numba(image, mask, binary_image),
        _threshold_sum_of_entropies_numba(image, mask, binary_image, noise),
    )


@njit(cache=True)
def _threshold_weighted_variance_numba(
    image: np.ndarray,
    mask: np.ndarray,
    binary_image: np.ndarray,
) -> float:
    height, width = image.shape
    any_masked = False
    max_value = -np.inf
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            any_masked = True
            value = image[y, x]
            if value > max_value:
                max_value = value

    if not any_masked:
        return 0.0
    minval = max_value / 256.0
    if minval == 0.0:
        return 0.0

    fg_count = 0
    bg_count = 0
    fg_sum = 0.0
    bg_sum = 0.0
    fg_sumsq = 0.0
    bg_sumsq = 0.0
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            value = image[y, x]
            if value < minval:
                value = minval
            log_value = math.log2(value)
            if binary_image[y, x]:
                fg_count += 1
                fg_sum += log_value
                fg_sumsq += log_value * log_value
            else:
                bg_count += 1
                bg_sum += log_value
                bg_sumsq += log_value * log_value

    if fg_count == 0 and bg_count == 0:
        return 0.0
    if fg_count == 0:
        bg_mean = bg_sum / bg_count
        return bg_sumsq / bg_count - bg_mean * bg_mean
    if bg_count == 0:
        fg_mean = fg_sum / fg_count
        return fg_sumsq / fg_count - fg_mean * fg_mean

    fg_mean = fg_sum / fg_count
    bg_mean = bg_sum / bg_count
    fg_variance = fg_sumsq / fg_count - fg_mean * fg_mean
    bg_variance = bg_sumsq / bg_count - bg_mean * bg_mean
    return (
        fg_variance * fg_count + bg_variance * bg_count
    ) / (fg_count + bg_count)


@njit(cache=True)
def _threshold_sum_of_entropies_numba(
    image: np.ndarray,
    mask: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
) -> float:
    height, width = image.shape
    any_masked = False
    max_value = -np.inf
    for y in range(height):
        for x in range(width):
            if (not mask[y, x]) or np.isnan(image[y, x]):
                continue
            any_masked = True
            value = image[y, x]
            if value > max_value:
                max_value = value

    if not any_masked:
        return 0.0
    minval = max_value / 256.0
    if minval == 0.0:
        return 0.0

    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    im_min = np.inf
    im_max = -np.inf
    foreground_count = 0
    background_count = 0
    smoothed = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < minval:
                value = minval
            if value < delta:
                clipped = delta
            elif value > 1.0:
                clipped = 1.0
            else:
                clipped = value

            noise_value = noise[y, x]
            smoothed_value = 2.0 ** (
                math.log2(clipped + delta) * noise_value
                + (1.0 - noise_value) * math.log2(clipped)
            )
            if smoothed_value > 1.0:
                smoothed_value = 1.0
            elif smoothed_value < 0.0:
                smoothed_value = 0.0
            smoothed[y, x] = smoothed_value
            if smoothed_value < im_min:
                im_min = smoothed_value
            if smoothed_value > im_max:
                im_max = smoothed_value

            if mask[y, x] and not np.isnan(image[y, x]):
                if binary_image[y, x]:
                    foreground_count += 1
                else:
                    background_count += 1

    upper = math.log2(im_max)
    lower = math.log2(im_min)
    if upper == lower:
        return math.log2(float(foreground_count + background_count))
    if foreground_count == 0 or background_count == 0:
        return 0.0

    foreground_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    background_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    for y in range(height):
        for x in range(width):
            if (not mask[y, x]) or np.isnan(image[y, x]):
                continue
            log_value = math.log2(smoothed[y, x])
            bin_index = int((log_value - lower) * scale)
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

    return _histogram_entropy_numba(
        foreground_hist,
        foreground_count,
    ) + _histogram_entropy_numba(
        background_hist,
        background_count,
    )


@njit(cache=True)
def _histogram_entropy_numba(histogram: np.ndarray, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    entropy = 0.0
    total = float(total_count)
    for index in range(histogram.size):
        count = histogram[index]
        if count <= 0:
            continue
        probability = float(count) / total
        entropy += probability * math.log2(probability)
    return entropy


def _smooth_with_deterministic_noise(image: np.ndarray, *, bits: int) -> np.ndarray:
    delta = pow(2.0, -bits)
    image_copy = np.clip(image, delta, 1)
    noise = _deterministic_normal_noise(image_copy.shape)
    result = np.exp2(
        np.log2(image_copy + delta) * noise
        + (1 - noise) * np.log2(image_copy)
    )
    result[result > 1] = 1
    result[result < 0] = 0
    return result


@lru_cache(maxsize=16)
def _deterministic_normal_noise(shape: tuple[int, ...]) -> np.ndarray:
    random_state = np.random.RandomState()
    random_state.seed(0)
    noise = random_state.normal(size=shape)
    noise.setflags(write=False)
    return noise
