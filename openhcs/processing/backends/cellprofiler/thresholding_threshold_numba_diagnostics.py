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


def rectangular_mask_domain(mask: np.ndarray) -> RectangularMaskDomain | None:
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


@njit(cache=True, inline="always")
def _threshold_weighted_variance_from_sums(
    fg_count: int,
    bg_count: int,
    fg_sum: float,
    bg_sum: float,
    fg_sumsq: float,
    bg_sumsq: float,
) -> float:
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
def _threshold_weighted_variance_unmasked_finite_numba(
    image: np.ndarray,
    binary_image: np.ndarray,
) -> float:
    height, width = image.shape
    if height == 0 or width == 0:
        return 0.0

    max_value = image[0, 0]
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value > max_value:
                max_value = value

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

    return _threshold_weighted_variance_from_sums(
        fg_count,
        bg_count,
        fg_sum,
        bg_sum,
        fg_sumsq,
        bg_sumsq,
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

    minval = max_value / 256.0
    if minval == 0.0:
        return 0.0, 0.0

    fg_count = 0
    bg_count = 0
    fg_sum = 0.0
    bg_sum = 0.0
    fg_sumsq = 0.0
    bg_sumsq = 0.0
    lower = np.inf
    upper = -np.inf
    foreground_count = 0
    background_count = 0
    smoothed_logs = np.empty(height * width, dtype=np.float64)
    smoothed_index = 0
    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < minval:
                value = minval
            log_value = math.log2(value)
            if binary_image[y, x]:
                fg_count += 1
                foreground_count += 1
                fg_sum += log_value
                fg_sumsq += log_value * log_value
            else:
                bg_count += 1
                background_count += 1
                bg_sum += log_value
                bg_sumsq += log_value * log_value

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
            smoothed_logs[smoothed_index] = log_smoothed_value
            smoothed_index += 1
            if log_smoothed_value < lower:
                lower = log_smoothed_value
            if log_smoothed_value > upper:
                upper = log_smoothed_value

    weighted_variance = _threshold_weighted_variance_from_sums(
        fg_count,
        bg_count,
        fg_sum,
        bg_sum,
        fg_sumsq,
        bg_sumsq,
    )

    if upper == lower:
        return weighted_variance, math.log2(float(foreground_count + background_count))
    if foreground_count == 0 or background_count == 0:
        return weighted_variance, 0.0

    foreground_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    background_hist = np.zeros(CELLPROFILER_THRESHOLD_ENTROPY_BINS, dtype=np.int64)
    scale = float(CELLPROFILER_THRESHOLD_ENTROPY_BINS) / (upper - lower)
    smoothed_index = 0
    for y in range(height):
        for x in range(width):
            log_smoothed_value = smoothed_logs[smoothed_index]
            smoothed_index += 1
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
def _threshold_diagnostics_numba(
    image: np.ndarray,
    mask: np.ndarray,
    binary_image: np.ndarray,
    noise: np.ndarray,
) -> tuple[float, float]:
    height, width = image.shape
    weighted_any_masked = False
    entropy_any_masked = False
    weighted_max_value = -np.inf
    entropy_max_value = -np.inf
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            weighted_any_masked = True
            value = image[y, x]
            if value > weighted_max_value:
                weighted_max_value = value
            if not np.isnan(value):
                entropy_any_masked = True
                if value > entropy_max_value:
                    entropy_max_value = value

    weighted_variance = 0.0
    weighted_minval = weighted_max_value / 256.0
    if weighted_any_masked and weighted_minval != 0.0:
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
                if value < weighted_minval:
                    value = weighted_minval
                log_value = math.log2(value)
                if binary_image[y, x]:
                    fg_count += 1
                    fg_sum += log_value
                    fg_sumsq += log_value * log_value
                else:
                    bg_count += 1
                    bg_sum += log_value
                    bg_sumsq += log_value * log_value

        weighted_variance = _threshold_weighted_variance_from_sums(
            fg_count,
            bg_count,
            fg_sum,
            bg_sum,
            fg_sumsq,
            bg_sumsq,
        )

    if not entropy_any_masked:
        return weighted_variance, 0.0
    entropy_minval = entropy_max_value / 256.0
    if entropy_minval == 0.0:
        return weighted_variance, 0.0

    lower = np.inf
    upper = -np.inf
    foreground_count = 0
    background_count = 0
    smoothed_logs = np.empty(height * width, dtype=np.float64)
    smoothed_index = 0
    delta = CELLPROFILER_THRESHOLD_ENTROPY_DELTA
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < entropy_minval:
                value = entropy_minval
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
            smoothed_logs[smoothed_index] = log_smoothed_value
            smoothed_index += 1
            if log_smoothed_value < lower:
                lower = log_smoothed_value
            if log_smoothed_value > upper:
                upper = log_smoothed_value

            if mask[y, x] and not np.isnan(image[y, x]):
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
    smoothed_index = 0
    for y in range(height):
        for x in range(width):
            log_value = smoothed_logs[smoothed_index]
            smoothed_index += 1
            if (not mask[y, x]) or np.isnan(image[y, x]):
                continue
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

    return weighted_variance, _histogram_entropy_numba(
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


def smooth_with_deterministic_noise(image: np.ndarray, *, bits: int) -> np.ndarray:
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
