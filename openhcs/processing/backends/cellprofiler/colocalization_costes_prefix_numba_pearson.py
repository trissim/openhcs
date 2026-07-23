"""Private Numba Pearson/Costes kernels for CellProfiler colocalization."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _correlation_slopes_numba(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float, float]:
    count = first.size
    if count <= 1:
        return np.nan, 0.0, 0.0

    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    for index in range(count):
        x = first[index]
        y = second[index]
        sum_x += x
        sum_y += y
        sum_x2 += x * x
        sum_y2 += y * y
        sum_xy += x * y

    x_centered_sum = sum_x2 - (sum_x * sum_x / count)
    y_centered_sum = sum_y2 - (sum_y * sum_y / count)
    xy_centered_sum = sum_xy - (sum_x * sum_y / count)

    correlation = np.nan
    denominator = np.sqrt(x_centered_sum * y_centered_sum)
    if denominator != 0.0:
        correlation = xy_centered_sum / denominator

    forward_slope = 0.0
    if x_centered_sum != 0.0:
        forward_slope = xy_centered_sum / x_centered_sum

    reverse_slope = 0.0
    if y_centered_sum != 0.0:
        reverse_slope = xy_centered_sum / y_centered_sum

    return correlation, forward_slope, reverse_slope


@njit(cache=True)
def _regression_line_numba(first: np.ndarray, second: np.ndarray) -> tuple[bool, float, float]:
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_z = 0.0
    sum_z2 = 0.0
    for index in range(first.size):
        x = first[index]
        y = second[index]
        if x > 0.0 or y > 0.0:
            z = x + y
            count += 1
            sum_x += x
            sum_y += y
            sum_x2 += x * x
            sum_y2 += y * y
            sum_z += z
            sum_z2 += z * z

    if count <= 1:
        return False, 0.0, 0.0

    denom_count = count - 1
    xmean = sum_x / count
    ymean = sum_y / count
    xvar = (sum_x2 - (sum_x * sum_x / count)) / denom_count
    yvar = (sum_y2 - (sum_y * sum_y / count)) / denom_count
    zvar = (sum_z2 - (sum_z * sum_z / count)) / denom_count
    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2.0 * covar
    if denom == 0.0:
        return False, 0.0, 0.0

    delta = yvar - xvar
    radicand = delta * delta + 4.0 * covar * covar
    if radicand < 0.0:
        return False, 0.0, 0.0
    slope = (delta + np.sqrt(radicand)) / denom
    intercept = ymean - slope * xmean
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return False, 0.0, 0.0
    return True, slope, intercept


@njit(cache=True)
def _pearson_below_threshold_numba(
    first: np.ndarray,
    second: np.ndarray,
    first_threshold: float,
    second_threshold: float,
) -> tuple[int, float]:
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    for index in range(first.size):
        x = first[index]
        y = second[index]
        if x < first_threshold or y < second_threshold:
            count += 1
            sum_x += x
            sum_y += y
            sum_x2 += x * x
            sum_y2 += y * y
            sum_xy += x * y

    if count <= 1:
        return count, np.nan

    x_centered_sum = sum_x2 - (sum_x * sum_x / count)
    y_centered_sum = sum_y2 - (sum_y * sum_y / count)
    denominator = np.sqrt(x_centered_sum * y_centered_sum)
    if denominator == 0.0:
        return count, np.nan
    numerator = sum_xy - (sum_x * sum_y / count)
    return count, numerator / denominator


@njit(cache=True)
def _max_pair_numba(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    first_max = 0.0
    second_max = 0.0
    for index in range(first.size):
        if first[index] > first_max:
            first_max = first[index]
        if second[index] > second_max:
            second_max = second[index]
    return first_max, second_max, max(first_max, second_max)


@njit(cache=True)
def _linear_costes_numba(
    first: np.ndarray,
    second: np.ndarray,
    scale_max: int,
    fast_mode: bool,
) -> tuple[float, float]:
    valid, slope, intercept = _regression_line_numba(first, second)
    if not valid:
        return 0.0, 0.0

    intensity_step = 1.0 / scale_max
    first_max, second_max, image_max = _max_pair_numba(first, second)
    threshold = intensity_step * (np.floor(image_max / intensity_step) + 1.0)
    threshold_1 = threshold
    threshold_2 = slope * threshold_1 + intercept

    while threshold > first_max and (slope * threshold + intercept) > second_max:
        threshold -= intensity_step

    previous_count = -1
    cost_regression = np.nan
    while threshold > intensity_step:
        threshold_1 = threshold
        threshold_2 = slope * threshold_1 + intercept
        count, pearson = _pearson_below_threshold_numba(
            first,
            second,
            threshold_1,
            threshold_2,
        )
        if count != previous_count:
            cost_regression = pearson
            previous_count = count

        if not np.isfinite(cost_regression):
            break
        if cost_regression <= 0.0:
            break
        if (not fast_mode) or threshold < intensity_step * 10.0:
            threshold -= intensity_step
        elif cost_regression > 0.45:
            threshold -= intensity_step * 10.0
        elif cost_regression > 0.35:
            threshold -= intensity_step * 5.0
        elif cost_regression > 0.25:
            threshold -= intensity_step * 2.0
        else:
            threshold -= intensity_step

    return threshold_1, threshold_2


@njit(cache=True)
def _prefix_pearson_numba(
    count: int,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float:
    if count <= 1:
        return np.nan
    x_centered_sum = sum_x2 - (sum_x * sum_x / count)
    y_centered_sum = sum_y2 - (sum_y * sum_y / count)
    denominator = np.sqrt(x_centered_sum * y_centered_sum)
    if denominator == 0.0:
        return np.nan
    numerator = sum_xy - (sum_x * sum_y / count)
    return numerator / denominator


@njit(cache=True)
def _event_count_for_threshold_numba(
    sorted_event_threshold: np.ndarray,
    threshold: float,
) -> int:
    low = 0
    high = sorted_event_threshold.size
    while low < high:
        middle = (low + high) // 2
        if sorted_event_threshold[middle] < threshold:
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=True)
def _prefix_value_numba(prefix: np.ndarray, count: int) -> float:
    if count <= 0:
        return 0.0
    return prefix[count - 1]


@njit(cache=True)
def _prefix_pearson_from_count_numba(
    count: int,
    prefix_x: np.ndarray,
    prefix_y: np.ndarray,
    prefix_x2: np.ndarray,
    prefix_y2: np.ndarray,
    prefix_xy: np.ndarray,
) -> float:
    return _prefix_pearson_numba(
        count,
        _prefix_value_numba(prefix_x, count),
        _prefix_value_numba(prefix_y, count),
        _prefix_value_numba(prefix_x2, count),
        _prefix_value_numba(prefix_y2, count),
        _prefix_value_numba(prefix_xy, count),
    )


@njit(cache=True)
def _max_from_prefix_numba(prefix: np.ndarray) -> float:
    if prefix.size == 0:
        return 0.0
    previous = 0.0
    max_value = 0.0
    for index in range(prefix.size):
        value = prefix[index] - previous
        previous = prefix[index]
        if value > max_value:
            max_value = value
    return max_value


@njit(cache=True)
def _linear_costes_sorted_events_numba(
    costes_context: tuple,
    fast_mode: bool,
) -> tuple[float, float]:
    (
        sorted_event_threshold,
        prefix_x,
        prefix_y,
        prefix_x2,
        prefix_y2,
        prefix_xy,
        scale_max,
        slope,
        intercept,
    ) = costes_context
    intensity_step = 1.0 / scale_max
    first_max = _max_from_prefix_numba(prefix_x)
    second_max = _max_from_prefix_numba(prefix_y)
    image_max = max(first_max, second_max)
    threshold = intensity_step * (np.floor(image_max / intensity_step) + 1.0)
    threshold_1 = threshold
    threshold_2 = slope * threshold_1 + intercept

    while threshold > first_max and (slope * threshold + intercept) > second_max:
        threshold -= intensity_step

    previous_count = -1
    cost_regression = np.nan
    while threshold > intensity_step:
        threshold_1 = threshold
        threshold_2 = slope * threshold_1 + intercept
        count = _event_count_for_threshold_numba(
            sorted_event_threshold,
            threshold_1,
        )
        if count != previous_count:
            cost_regression = _prefix_pearson_from_count_numba(
                count,
                prefix_x,
                prefix_y,
                prefix_x2,
                prefix_y2,
                prefix_xy,
            )
            previous_count = count

        if not np.isfinite(cost_regression):
            break
        if cost_regression <= 0.0:
            break
        if (not fast_mode) or threshold < intensity_step * 10.0:
            threshold -= intensity_step
        elif cost_regression > 0.45:
            threshold -= intensity_step * 10.0
        elif cost_regression > 0.35:
            threshold -= intensity_step * 5.0
        elif cost_regression > 0.25:
            threshold -= intensity_step * 2.0
        else:
            threshold -= intensity_step

    return threshold_1, threshold_2

@njit(cache=True)
def _integer_unit_interval_codes_for_scale_numba(
    values: np.ndarray,
    scale: int,
) -> tuple[bool, np.ndarray]:
    codes = np.empty(values.size, dtype=np.int64)
    scale_float32 = np.float32(scale)
    for index in range(values.size):
        value = values[index]
        code = int(np.rint(value * scale))
        if code < 0 or code > scale:
            return False, codes
        reconstructed = np.float64(np.float32(code) / scale_float32)
        if value != reconstructed:
            return False, codes
        codes[index] = code
    return True, codes


@njit(cache=True)
def _scaled_second_channel_costes_numba(
    first: np.ndarray,
    second: np.ndarray,
    scale_max: int,
) -> tuple[float, float]:
    valid, slope, intercept = _regression_line_numba(first, second)
    if not valid:
        return 0.0, 0.0

    minimum_scale_index = min(scale_max, 5)
    selected_first_threshold = 0.0
    selected_second_threshold = minimum_scale_index / scale_max
    selected_correlation = np.nan

    for second_scale_index in range(scale_max, minimum_scale_index - 1, -1):
        second_threshold = second_scale_index / scale_max
        if slope == 0.0:
            first_threshold = 0.0
        else:
            first_threshold = (second_threshold - intercept) / slope
            if first_threshold < 0.0:
                first_threshold = 0.0

        _count, cost_regression = _pearson_below_threshold_numba(
            first,
            second,
            first_threshold,
            second_threshold,
        )
        selected_first_threshold = first_threshold
        selected_second_threshold = second_threshold
        selected_correlation = cost_regression
        if np.isfinite(cost_regression) and cost_regression <= 0.0:
            break

    if (
        (not np.isfinite(selected_correlation))
        or selected_correlation > 0.0
        or selected_first_threshold <= 0.0
    ):
        selected_first_threshold = 0.0

    return selected_first_threshold, selected_second_threshold


@njit(cache=True)
def _scaled_second_channel_costes_sorted_events_numba(
    costes_context: tuple,
) -> tuple[float, float]:
    (
        sorted_event_threshold,
        prefix_x,
        prefix_y,
        prefix_x2,
        prefix_y2,
        prefix_xy,
        scale_max,
        slope,
        intercept,
    ) = costes_context
    minimum_scale_index = min(scale_max, 5)
    selected_first_threshold = 0.0
    selected_second_threshold = minimum_scale_index / scale_max
    selected_correlation = np.nan

    for second_scale_index in range(scale_max, minimum_scale_index - 1, -1):
        second_threshold = second_scale_index / scale_max
        first_threshold = (second_threshold - intercept) / slope
        if first_threshold < 0.0:
            first_threshold = 0.0
        count = _event_count_for_threshold_numba(
            sorted_event_threshold,
            second_threshold,
        )
        cost_regression = _prefix_pearson_from_count_numba(
            count,
            prefix_x,
            prefix_y,
            prefix_x2,
            prefix_y2,
            prefix_xy,
        )
        selected_first_threshold = first_threshold
        selected_second_threshold = second_threshold
        selected_correlation = cost_regression
        if np.isfinite(cost_regression) and cost_regression <= 0.0:
            break

    if (
        (not np.isfinite(selected_correlation))
        or selected_correlation > 0.0
        or selected_first_threshold <= 0.0
    ):
        selected_first_threshold = 0.0

    return selected_first_threshold, selected_second_threshold


@njit(cache=True)
def _scaled_second_channel_costes_grouped_events_numba(
    costes_context: tuple,
    prefix_count: np.ndarray,
) -> tuple[float, float]:
    (
        sorted_event_threshold,
        prefix_x,
        prefix_y,
        prefix_x2,
        prefix_y2,
        prefix_xy,
        scale_max,
        slope,
        intercept,
    ) = costes_context
    minimum_scale_index = min(scale_max, 5)
    selected_first_threshold = 0.0
    selected_second_threshold = minimum_scale_index / scale_max
    selected_correlation = np.nan

    for second_scale_index in range(scale_max, minimum_scale_index - 1, -1):
        second_threshold = second_scale_index / scale_max
        first_threshold = (second_threshold - intercept) / slope
        if first_threshold < 0.0:
            first_threshold = 0.0
        group_count = _event_count_for_threshold_numba(
            sorted_event_threshold,
            second_threshold,
        )
        cost_regression = _prefix_pearson_from_group_count_numba(
            group_count,
            prefix_count,
            prefix_x,
            prefix_y,
            prefix_x2,
            prefix_y2,
            prefix_xy,
        )
        selected_first_threshold = first_threshold
        selected_second_threshold = second_threshold
        selected_correlation = cost_regression
        if np.isfinite(cost_regression) and cost_regression <= 0.0:
            break

    if (
        (not np.isfinite(selected_correlation))
        or selected_correlation > 0.0
        or selected_first_threshold <= 0.0
    ):
        selected_first_threshold = 0.0

    return selected_first_threshold, selected_second_threshold


@njit(cache=True)
def _prefix_pearson_from_group_count_numba(
    group_count: int,
    prefix_count: np.ndarray,
    prefix_x: np.ndarray,
    prefix_y: np.ndarray,
    prefix_x2: np.ndarray,
    prefix_y2: np.ndarray,
    prefix_xy: np.ndarray,
) -> float:
    if group_count <= 0:
        return np.nan
    count = int(prefix_count[group_count - 1])
    return _prefix_pearson_numba(
        count,
        _prefix_value_numba(prefix_x, group_count),
        _prefix_value_numba(prefix_y, group_count),
        _prefix_value_numba(prefix_x2, group_count),
        _prefix_value_numba(prefix_y2, group_count),
        _prefix_value_numba(prefix_xy, group_count),
    )




@njit(cache=True)
def _costes_manders_numba(
    first: np.ndarray,
    second: np.ndarray,
    first_threshold: float,
    second_threshold: float,
) -> tuple[float, float]:
    first_threshold_total = 0.0
    second_threshold_total = 0.0
    combined_first_total = 0.0
    combined_second_total = 0.0
    combined_count = 0

    for index in range(first.size):
        first_value = first[index]
        second_value = second[index]
        if first_threshold <= 0.0:
            first_above = first_value >= first_threshold
        else:
            first_above = first_value > first_threshold
        if second_threshold <= 0.0:
            second_above = second_value >= second_threshold
        else:
            second_above = second_value > second_threshold

        if first_above:
            first_threshold_total += first_value
        if second_above:
            second_threshold_total += second_value
        if first_above and second_above:
            combined_count += 1
            combined_first_total += first_value
            combined_second_total += second_value

    costes_m1 = np.nan
    costes_m2 = np.nan
    if combined_count > 0:
        if first_threshold_total > 0.0:
            costes_m1 = combined_first_total / first_threshold_total
        if second_threshold_total > 0.0:
            costes_m2 = combined_second_total / second_threshold_total
    return costes_m1, costes_m2
