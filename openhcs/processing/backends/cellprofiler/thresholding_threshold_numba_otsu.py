"""Threshold primitive/Otsu-family Numba kernels for CellProfiler thresholding."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_otsu_weighted import (
    weighted_otsu_threshold_numba_compatible as _weighted_otsu_threshold_numba_compatible,
)


CELLPROFILER_LI_TOLERANCE = 0.5 / 65536.0

def _finite_flat_float64(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return np.ascontiguousarray(flat[np.isfinite(flat)], dtype=np.float64)


def _finite_flat_float32(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).ravel()
    return np.ascontiguousarray(flat[np.isfinite(flat)], dtype=np.float32)



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
def _binned_mode_numba(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    bin_count = int(math.ceil(math.sqrt(float(values.size))))
    if bin_count < 2:
        bin_count = 2
    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)

    best_index = 0
    best_count = counts[0]
    for index in range(1, bin_count):
        if counts[index] > best_count:
            best_index = index
            best_count = counts[index]
    return minimum + (float(best_index) + 0.5) * (maximum - minimum) / bin_count


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
def _histogram_range_numba(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    minimum = values[0]
    maximum = values[0]
    for index in range(1, values.size):
        value = values[index]
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value
    return minimum, maximum


@njit(cache=True)
def _histogram_counts_numba(
    values: np.ndarray,
    bin_count: int,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    counts = np.zeros(bin_count, dtype=np.int64)
    if values.size == 0 or maximum == minimum:
        return counts
    scale = float(bin_count) / (maximum - minimum)
    for index in range(values.size):
        bin_index = int((values[index] - minimum) * scale)
        if bin_index < 0:
            bin_index = 0
        elif bin_index >= bin_count:
            bin_index = bin_count - 1
        counts[bin_index] += 1
    return counts


@njit(cache=True)
def _otsu_threshold_numba(values: np.ndarray, bin_count: int) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)
    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    width = (maximum - minimum) / float(bin_count)
    total = 0
    total_weighted = 0.0
    for index in range(bin_count):
        count = counts[index]
        center = minimum + (float(index) + 0.5) * width
        total += count
        total_weighted += float(count) * center
    if total == 0:
        return 0.0

    background_count = 0
    background_weighted = 0.0
    best_index = 0
    best_variance = -1.0
    for index in range(bin_count - 1):
        count = counts[index]
        center = minimum + (float(index) + 0.5) * width
        background_count += count
        background_weighted += float(count) * center
        foreground_count = total - background_count
        if background_count <= 0 or foreground_count <= 0:
            continue
        background_mean = background_weighted / float(background_count)
        foreground_mean = (
            total_weighted - background_weighted
        ) / float(foreground_count)
        mean_delta = background_mean - foreground_mean
        variance = (
            float(background_count)
            * float(foreground_count)
            * mean_delta
            * mean_delta
        )
        if variance > best_variance:
            best_variance = variance
            best_index = index
    return minimum + (float(best_index) + 0.5) * width


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


@njit(cache=True)
def _triangle_threshold_numba(values: np.ndarray, bin_count: int) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    first = 0
    while first < bin_count and counts[first] == 0:
        first += 1
    last = bin_count - 1
    while last >= 0 and counts[last] == 0:
        last -= 1
    if first >= last:
        return minimum + (float(first) + 0.5) * (maximum - minimum) / bin_count

    peak = first
    peak_count = counts[first]
    for index in range(first + 1, last + 1):
        count = counts[index]
        if count > peak_count:
            peak = index
            peak_count = count

    if peak - first < last - peak:
        original_first = first
        first = bin_count - 1 - last
        last = bin_count - 1 - original_first
        peak = bin_count - 1 - peak
        reversed_counts = np.empty(bin_count, dtype=np.int64)
        for index in range(bin_count):
            reversed_counts[index] = counts[bin_count - 1 - index]
        counts = reversed_counts
        is_reversed = True
    else:
        is_reversed = False

    x1 = float(first)
    y1 = float(counts[first])
    x2 = float(last)
    y2 = float(counts[last])
    dx = x2 - x1
    dy = y2 - y1
    normalizer = math.sqrt(dx * dx + dy * dy)
    if normalizer == 0.0:
        threshold_index = peak
    else:
        threshold_index = first
        max_distance = -1.0
        for index in range(first, last + 1):
            distance = abs(
                dy * float(index)
                - dx * float(counts[index])
                + x2 * y1
                - y2 * x1
            ) / normalizer
            if distance > max_distance:
                max_distance = distance
                threshold_index = index

    if is_reversed:
        threshold_index = bin_count - 1 - threshold_index
    return minimum + (float(threshold_index) + 0.5) * (maximum - minimum) / bin_count


@njit(cache=True)
def _isodata_threshold_numba(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    threshold = _mean_threshold_numba(values)
    for _ in range(1000):
        lower_count = 0
        upper_count = 0
        lower_sum = 0.0
        upper_sum = 0.0
        for index in range(values.size):
            value = values[index]
            if value <= threshold:
                lower_count += 1
                lower_sum += value
            else:
                upper_count += 1
                upper_sum += value
        if lower_count == 0 or upper_count == 0:
            return threshold
        next_threshold = (
            lower_sum / float(lower_count) + upper_sum / float(upper_count)
        ) / 2.0
        if next_threshold == threshold:
            return threshold
        if abs(next_threshold - threshold) <= 0.5 / 65536.0:
            return next_threshold
        threshold = next_threshold
    return threshold


@njit(cache=True)
def _mean_threshold_numba(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    total = 0.0
    for index in range(values.size):
        total += values[index]
    return total / float(values.size)


@njit(cache=True)
def _yen_threshold_numba(values: np.ndarray, bin_count: int) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    total_count = 0.0
    for index in range(bin_count):
        total_count += float(counts[index])
    if total_count == 0.0:
        return 0.0

    p1 = np.zeros(bin_count, dtype=np.float64)
    p1_sq = np.zeros(bin_count, dtype=np.float64)
    p2_sq = np.zeros(bin_count, dtype=np.float64)
    running_probability = 0.0
    running_square = 0.0
    for index in range(bin_count):
        probability = float(counts[index]) / total_count
        running_probability += probability
        running_square += probability * probability
        p1[index] = running_probability
        p1_sq[index] = running_square

    running_square = 0.0
    for index in range(bin_count - 1, -1, -1):
        probability = float(counts[index]) / total_count
        running_square += probability * probability
        p2_sq[index] = running_square

    best_index = 0
    best_criterion = -np.inf
    for index in range(bin_count - 1):
        foreground_probability = p1[index]
        background_probability = 1.0 - foreground_probability
        square_product = p1_sq[index] * p2_sq[index + 1]
        probability_product = foreground_probability * background_probability
        if square_product <= 0.0 or probability_product <= 0.0:
            continue
        criterion = -math.log(square_product) + 2.0 * math.log(
            probability_product
        )
        if criterion > best_criterion:
            best_criterion = criterion
            best_index = index

    return minimum + (float(best_index) + 0.5) * (maximum - minimum) / bin_count


@njit(cache=True)
def _minimum_threshold_numba(values: np.ndarray, bin_count: int) -> float:
    if values.size == 0:
        return 0.0
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    histogram = _histogram_counts_numba(values, bin_count, minimum, maximum).astype(
        np.float64
    )
    maxima = np.empty(bin_count, dtype=np.int64)
    maxima_count = 0
    for _ in range(10000):
        maxima_count = _histogram_local_maxima_numba(histogram, maxima)
        if maxima_count <= 2:
            break
        previous = histogram.copy()
        for index in range(bin_count):
            left = previous[index - 1] if index > 0 else previous[index]
            center = previous[index]
            right = previous[index + 1] if index < bin_count - 1 else previous[index]
            histogram[index] = (left + center + right) / 3.0

    if maxima_count != 2:
        return np.nan

    first = maxima[0]
    second = maxima[1]
    if first > second:
        tmp = first
        first = second
        second = tmp
    valley_index = first
    valley_value = histogram[first]
    for index in range(first + 1, second + 1):
        if histogram[index] < valley_value:
            valley_value = histogram[index]
            valley_index = index
    return minimum + (float(valley_index) + 0.5) * (maximum - minimum) / bin_count


@njit(cache=True)
def _histogram_local_maxima_numba(
    histogram: np.ndarray,
    maxima: np.ndarray,
) -> int:
    count = 0
    if histogram.size == 0:
        return count
    if histogram.size == 1:
        maxima[count] = 0
        return 1
    if histogram[0] > histogram[1]:
        maxima[count] = 0
        count += 1
    for index in range(1, histogram.size - 1):
        if (
            histogram[index - 1] < histogram[index]
            and histogram[index + 1] < histogram[index]
        ):
            maxima[count] = index
            count += 1
    if histogram[histogram.size - 1] > histogram[histogram.size - 2]:
        maxima[count] = histogram.size - 1
        count += 1
    return count


@njit(cache=True)
def _multiotsu_three_class_thresholds_numba(
    values: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    thresholds = np.zeros(2, dtype=np.float64)
    if values.size == 0:
        return thresholds
    if bin_count < 3:
        bin_count = 3
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        thresholds[0] = minimum
        thresholds[1] = minimum
        return thresholds

    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    nonzero_count = 0
    last_nonzero = 0
    for index in range(bin_count):
        if counts[index] > 0:
            if nonzero_count < 2:
                thresholds[nonzero_count] = (
                    minimum
                    + (float(index) + 0.5) * (maximum - minimum) / float(bin_count)
                )
            last_nonzero = index
            nonzero_count += 1
    if nonzero_count < 3:
        if nonzero_count == 2:
            thresholds[1] = (
                minimum
                + (float(last_nonzero) + 0.5)
                * (maximum - minimum)
                / float(bin_count)
            )
        return thresholds
    if nonzero_count == 3:
        return thresholds

    width = (maximum - minimum) / float(bin_count)
    total_count = 0
    for index in range(bin_count):
        total_count += counts[index]
    if total_count == 0:
        return thresholds

    cumulative_probability = np.zeros(bin_count, dtype=np.float32)
    cumulative_weighted_index = np.zeros(bin_count, dtype=np.float32)
    running_probability = np.float32(float(counts[0]) / float(total_count))
    running_weighted_index = running_probability
    cumulative_probability[0] = running_probability
    cumulative_weighted_index[0] = running_weighted_index
    for index in range(1, bin_count):
        probability = np.float32(float(counts[index]) / float(total_count))
        running_probability = np.float32(running_probability + probability)
        running_weighted_index = np.float32(
            running_weighted_index + np.float32(float(index)) * probability
        )
        cumulative_probability[index] = running_probability
        cumulative_weighted_index[index] = running_weighted_index

    best_first = 0
    best_second = 1
    best_variance = np.float32(0.0)
    for first in range(bin_count - 2):
        for second in range(first + 1, bin_count - 1):
            variance = (
                _multiotsu_interval_score_numba(
                    cumulative_probability,
                    cumulative_weighted_index,
                    0,
                    first,
                )
                + _multiotsu_interval_score_numba(
                    cumulative_probability,
                    cumulative_weighted_index,
                    first + 1,
                    second,
                )
                + _multiotsu_interval_score_numba(
                    cumulative_probability,
                    cumulative_weighted_index,
                    second + 1,
                    bin_count - 1,
                )
            )
            if variance > best_variance:
                best_variance = variance
                best_first = first
                best_second = second
    thresholds[0] = minimum + (float(best_first) + 0.5) * width
    thresholds[1] = minimum + (float(best_second) + 0.5) * width
    return thresholds


@njit(cache=True)
def _multiotsu_interval_score_numba(
    cumulative_probability: np.ndarray,
    cumulative_weighted_index: np.ndarray,
    first: int,
    last: int,
) -> np.float32:
    if first == 0:
        probability = cumulative_probability[last]
        weighted_index = cumulative_weighted_index[last]
    else:
        probability = np.float32(
            cumulative_probability[last] - cumulative_probability[first - 1]
        )
        weighted_index = np.float32(
            cumulative_weighted_index[last]
            - cumulative_weighted_index[first - 1]
        )
    if probability <= np.float32(0.0):
        return np.float32(0.0)
    return np.float32((weighted_index * weighted_index) / probability)


@njit(cache=True)
def _minimum_cross_entropy_threshold_numba(
    values: np.ndarray,
    bin_count: int,
) -> float:
    if values.size == 0:
        return 0.0
    if bin_count < 2:
        bin_count = 2
    minimum, maximum = _histogram_range_numba(values)
    if maximum == minimum:
        return float(minimum)

    counts = _histogram_counts_numba(values, bin_count, minimum, maximum)
    width = (maximum - minimum) / float(bin_count)
    cumulative_count = np.zeros(bin_count, dtype=np.float64)
    cumulative_weighted = np.zeros(bin_count, dtype=np.float64)
    total_count = 0.0
    total_weighted = 0.0
    for index in range(bin_count):
        center = minimum + (float(index) + 0.5) * width
        count = float(counts[index])
        total_count += count
        total_weighted += count * center
        cumulative_count[index] = total_count
        cumulative_weighted[index] = total_weighted

    if total_count == 0.0:
        return 0.0

    best_index = 0
    best_cross_entropy = np.inf
    for index in range(bin_count - 1):
        foreground_count = cumulative_count[index]
        background_count = total_count - foreground_count
        foreground_weighted = cumulative_weighted[index]
        background_weighted = total_weighted - foreground_weighted
        if (
            foreground_count <= 0.0
            or background_count <= 0.0
            or foreground_weighted <= 0.0
            or background_weighted <= 0.0
        ):
            continue
        foreground_mean = foreground_weighted / foreground_count
        background_mean = background_weighted / background_count
        if foreground_mean <= 0.0 or background_mean <= 0.0:
            continue
        cross_entropy = -(
            foreground_weighted * math.log(foreground_mean)
            + background_weighted * math.log(background_mean)
        )
        if cross_entropy < best_cross_entropy:
            best_cross_entropy = cross_entropy
            best_index = index
    return minimum + (float(best_index) + 0.5) * width


@njit(cache=True)
def _sauvola_threshold_image_numba(
    image: np.ndarray,
    window_size: int,
    k: float,
    dynamic_range: float,
) -> np.ndarray:
    height, width = image.shape
    if window_size < 1:
        window_size = 1
    if window_size % 2 == 0:
        window_size += 1
    radius = window_size // 2
    padded_height = height + 2 * radius
    padded_width = width + 2 * radius
    integral = np.zeros((padded_height + 1, padded_width + 1), dtype=np.float64)
    integral_squared = np.zeros(
        (padded_height + 1, padded_width + 1),
        dtype=np.float64,
    )

    for padded_y in range(padded_height):
        image_y = _reflect_index_numba(padded_y - radius, height)
        row_sum = 0.0
        row_squared_sum = 0.0
        for padded_x in range(padded_width):
            image_x = _reflect_index_numba(padded_x - radius, width)
            value = image[image_y, image_x]
            row_sum += value
            row_squared_sum += value * value
            integral[padded_y + 1, padded_x + 1] = (
                integral[padded_y, padded_x + 1] + row_sum
            )
            integral_squared[padded_y + 1, padded_x + 1] = (
                integral_squared[padded_y, padded_x + 1] + row_squared_sum
            )

    output = np.empty((height, width), dtype=np.float64)
    area = float(window_size * window_size)

    for y in range(height):
        for x in range(width):
            y0 = y
            x0 = x
            y1 = y0 + window_size
            x1 = x0 + window_size
            total = (
                integral[y1, x1]
                - integral[y0, x1]
                - integral[y1, x0]
                + integral[y0, x0]
            )
            total_squared = (
                integral_squared[y1, x1]
                - integral_squared[y0, x1]
                - integral_squared[y1, x0]
                + integral_squared[y0, x0]
            )
            mean = total / area
            variance = total_squared / area - mean * mean
            if variance < 0.0:
                variance = 0.0
            stddev = math.sqrt(variance)
            output[y, x] = mean * (1.0 + k * ((stddev / dynamic_range) - 1.0))
    return output


@njit(cache=True)
def _reflect_index_numba(index: int, size: int) -> int:
    if size <= 1:
        return 0
    while index < 0 or index >= size:
        if index < 0:
            index = -index - 1
        elif index >= size:
            index = 2 * size - index - 1
    return index
