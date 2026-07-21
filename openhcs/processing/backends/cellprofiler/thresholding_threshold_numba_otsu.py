"""Threshold primitive/Otsu-family Numba kernels for CellProfiler thresholding."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_otsu_weighted import (
    weighted_otsu_threshold_numba_compatible as _weighted_otsu_threshold_numba_compatible,
)
from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_otsu_histogram_multiotsu import (
    _binned_mode_numba,
    _histogram_range_numba,
    _isodata_threshold_numba,
    _mean_threshold_numba,
    _minimum_cross_entropy_threshold_numba,
    _minimum_threshold_numba,
    _multiotsu_three_class_thresholds_numba,
    _otsu_threshold_numba,
    _sauvola_threshold_image_numba,
    _triangle_threshold_numba,
    _yen_threshold_numba,
)


CELLPROFILER_LI_TOLERANCE = 0.5 / 65536.0


def _finite_flat_float64(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(flat)
    if bool(np.all(finite)):
        return np.ascontiguousarray(flat)
    return np.ascontiguousarray(flat[finite], dtype=np.float64)


def _finite_flat_float32(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).ravel()
    finite = np.isfinite(flat)
    if bool(np.all(finite)):
        return np.ascontiguousarray(flat)
    return np.ascontiguousarray(flat[finite], dtype=np.float32)


@njit(cache=True)
def _log_transform_numba(
    values: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    transformed = np.zeros(values.shape, dtype=np.float64)
    if values.size == 0:
        return transformed, 0.0, 0.0, 0.0

    min_value = np.inf
    max_value = -np.inf
    for index in range(values.size):
        value = float(values[index])
        if not np.isfinite(value):
            continue
        if value < min_value:
            min_value = value
        if value > max_value:
            max_value = value

    if not np.isfinite(max_value) or max_value <= 0.0:
        return transformed, 0.0, 0.0, 0.0
    if not np.isfinite(min_value):
        min_value = 0.0
    noise_min = float(
        np.float32(
            np.float32(min_value)
            + np.float32(max_value - min_value) / np.float32(256.0)
            + np.finfo(np.float32).eps
        )
    )

    log_min = float(np.float32(math.log(noise_min)))
    log_max = float(np.float32(math.log(np.float32(max_value))))
    denominator = log_max - log_min
    if denominator == 0.0:
        return transformed, noise_min, log_min, log_max

    for index in range(values.size):
        value = float(values[index])
        if not np.isfinite(value) or value < noise_min:
            value = noise_min
        transformed[index] = (math.log(value) - log_min) / denominator
    return transformed, noise_min, log_min, log_max


@njit(cache=True)
def _inverse_log_transform_numba(
    values: np.ndarray,
    log_min: float,
    log_max: float,
) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.float64)
    scale = log_max - log_min
    for index in range(values.size):
        output[index] = np.float32(math.exp(log_min + float(values[index]) * scale))
    return output


@njit(cache=True)
def _mad_numba(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values.copy())
    median = _median_sorted_numba(sorted_values)
    deviations = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        deviations[index] = abs(values[index] - median)
    deviations = np.sort(deviations)
    return _median_sorted_numba(deviations)


@njit(cache=True)
def _median_sorted_numba(sorted_values: np.ndarray) -> float:
    size = sorted_values.size
    if size == 0:
        return 0.0
    middle = size // 2
    if size % 2 == 1:
        return float(sorted_values[middle])
    return (float(sorted_values[middle - 1]) + float(sorted_values[middle])) / 2.0


@njit(cache=True)
def _li_tolerance_numba(values: np.ndarray) -> float:
    tolerance = CELLPROFILER_LI_TOLERANCE
    if values.size < 2:
        return tolerance
    sorted_values = np.sort(values.copy())
    min_diff = np.inf
    previous = sorted_values[0]
    for index in range(1, sorted_values.size):
        current = sorted_values[index]
        difference = current - previous
        if difference > 0.0 and difference < min_diff:
            min_diff = difference
        previous = current
    if min_diff == np.inf:
        return tolerance
    half_diff = min_diff / 2.0
    if half_diff > tolerance:
        return half_diff
    return tolerance


@njit(cache=True)
def _li_threshold_numba(values: np.ndarray, tolerance: float) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    threshold = 0.0
    for index in range(values.size):
        threshold += values[index] - minimum
    threshold /= float(values.size)
    previous_threshold = -2.0 * tolerance
    tiny = np.finfo(np.float64).tiny
    iterations = 0
    while abs(threshold - previous_threshold) > tolerance and iterations < 1000:
        previous_threshold = threshold
        background_count = 0
        foreground_count = 0
        background_sum = 0.0
        foreground_sum = 0.0
        for index in range(values.size):
            value = values[index] - minimum
            if value <= previous_threshold:
                background_count += 1
                background_sum += value
            else:
                foreground_count += 1
                foreground_sum += value

        if background_count == 0 or foreground_count == 0:
            return previous_threshold + minimum

        background_mean = background_sum / float(background_count)
        foreground_mean = foreground_sum / float(foreground_count)
        if background_mean <= tiny:
            background_mean = tiny
        if foreground_mean <= tiny:
            foreground_mean = tiny
        if background_mean == foreground_mean:
            return background_mean + minimum

        threshold = (
            background_mean - foreground_mean
        ) / (
            math.log(background_mean) - math.log(foreground_mean)
        )
        iterations += 1
    return threshold + minimum


