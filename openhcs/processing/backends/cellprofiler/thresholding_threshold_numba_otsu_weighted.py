"""Weighted Otsu Numba kernels for CellProfiler thresholding."""

from __future__ import annotations

import numpy as np
from numba import njit


def weighted_otsu_threshold_numba_compatible(
    values: np.ndarray,
    bin_count: int,
) -> float:
    """Return weighted Otsu using the fastest exact sorted-rank representation."""
    values_array = np.ascontiguousarray(values, dtype=np.float64)
    unique_values, counts = np.unique(values_array, return_counts=True)
    if unique_values.size < values_array.size:
        return float(
            _counted_sorted_weighted_otsu_threshold_numba(
                np.ascontiguousarray(unique_values, dtype=np.float64),
                np.ascontiguousarray(counts, dtype=np.int64),
                int(values_array.size),
                int(bin_count),
            )
        )
    return float(_sorted_weighted_otsu_threshold_numba(values_array, int(bin_count)))



@njit(cache=True)
def _sorted_weighted_otsu_threshold_numba(
    values: np.ndarray,
    bin_count: int,
) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values.copy())
    size = sorted_values.size
    if size == 1:
        return float(sorted_values[0])
    if bin_count > size:
        bin_count = size
    step = size // bin_count
    if step < 1:
        step = 1

    variance = _running_variance_numba(sorted_values)
    reversed_values = np.empty(size, dtype=np.float64)
    for index in range(size):
        reversed_values[index] = sorted_values[size - 1 - index]
    reversed_variance = _running_variance_numba(reversed_values)

    best_score = np.inf
    best_candidate = 0
    candidate_count = 0
    for candidate_index in range(0, size - 1, step):
        high_index = candidate_index + 1
        score = (
            variance[candidate_index] * float(candidate_index)
            + reversed_variance[size - 1 - high_index]
            * float(size - high_index)
        )
        if score < best_score:
            best_score = score
            best_candidate = candidate_count
        candidate_count += 1

    if candidate_count == 0:
        return float(sorted_values[1])

    best_index = 1 + best_candidate * step
    low_candidate = best_candidate - 1
    high_candidate = best_candidate + 1
    if low_candidate < 0:
        low_candidate = 0
    if high_candidate >= candidate_count:
        high_candidate = candidate_count - 1
    low_index = 1 + low_candidate * step
    high_index = 1 + high_candidate * step
    if low_index >= size:
        low_index = size - 1
    if high_index >= size:
        high_index = size - 1
    return (float(sorted_values[low_index]) + float(sorted_values[high_index])) / 2.0


@njit(cache=True)
def _counted_sorted_weighted_otsu_threshold_numba(
    unique_values: np.ndarray,
    counts: np.ndarray,
    size: int,
    bin_count: int,
) -> float:
    if size == 0:
        return 0.0
    if size == 1:
        return float(unique_values[0])
    if bin_count > size:
        bin_count = size
    step = size // bin_count
    if step < 1:
        step = 1

    unique_count = unique_values.size
    cumulative_counts = np.empty(unique_count, dtype=np.int64)
    cumulative_sums = np.empty(unique_count, dtype=np.float64)
    cumulative_squares = np.empty(unique_count, dtype=np.float64)

    running_count = 0
    running_sum = 0.0
    running_square = 0.0
    for index in range(unique_count):
        count = counts[index]
        value = unique_values[index]
        running_count += count
        running_sum += value * float(count)
        running_square += value * value * float(count)
        cumulative_counts[index] = running_count
        cumulative_sums[index] = running_sum
        cumulative_squares[index] = running_square

    total_sum = cumulative_sums[unique_count - 1]
    total_square = cumulative_squares[unique_count - 1]

    best_score = np.inf
    best_candidate = 0
    candidate_count = 0
    for candidate_index in range(0, size - 1, step):
        high_index = candidate_index + 1
        foreground_count = candidate_index + 1
        background_count = size - high_index
        foreground_variance = _counted_prefix_variance_at_rank_numba(
            unique_values,
            cumulative_counts,
            cumulative_sums,
            cumulative_squares,
            candidate_index,
        )
        background_sum = total_sum - _counted_prefix_sum_at_rank_numba(
            unique_values,
            cumulative_counts,
            cumulative_sums,
            high_index - 1,
        )
        background_square = total_square - _counted_prefix_square_at_rank_numba(
            unique_values,
            cumulative_counts,
            cumulative_squares,
            high_index - 1,
        )
        background_variance = _sample_variance_numba(
            background_count,
            background_sum,
            background_square,
        )
        score = (
            foreground_variance * float(candidate_index)
            + background_variance * float(background_count)
        )
        if score < best_score:
            best_score = score
            best_candidate = candidate_count
        candidate_count += 1

    if candidate_count == 0:
        return _counted_value_at_rank_numba(unique_values, cumulative_counts, 1)

    low_candidate = best_candidate - 1
    high_candidate = best_candidate + 1
    if low_candidate < 0:
        low_candidate = 0
    if high_candidate >= candidate_count:
        high_candidate = candidate_count - 1
    low_index = 1 + low_candidate * step
    high_index = 1 + high_candidate * step
    if low_index >= size:
        low_index = size - 1
    if high_index >= size:
        high_index = size - 1
    return (
        _counted_value_at_rank_numba(unique_values, cumulative_counts, low_index)
        + _counted_value_at_rank_numba(unique_values, cumulative_counts, high_index)
    ) / 2.0


@njit(cache=True)
def _counted_prefix_variance_at_rank_numba(
    unique_values: np.ndarray,
    cumulative_counts: np.ndarray,
    cumulative_sums: np.ndarray,
    cumulative_squares: np.ndarray,
    rank: int,
) -> float:
    prefix_count = rank + 1
    prefix_sum = _counted_prefix_sum_at_rank_numba(
        unique_values,
        cumulative_counts,
        cumulative_sums,
        rank,
    )
    prefix_square = _counted_prefix_square_at_rank_numba(
        unique_values,
        cumulative_counts,
        cumulative_squares,
        rank,
    )
    return _sample_variance_numba(prefix_count, prefix_sum, prefix_square)


@njit(cache=True)
def _counted_prefix_sum_at_rank_numba(
    unique_values: np.ndarray,
    cumulative_counts: np.ndarray,
    cumulative_sums: np.ndarray,
    rank: int,
) -> float:
    bucket = _counted_bucket_for_rank_numba(cumulative_counts, rank)
    previous_count = 0 if bucket == 0 else cumulative_counts[bucket - 1]
    previous_sum = 0.0 if bucket == 0 else cumulative_sums[bucket - 1]
    partial_count = rank - previous_count + 1
    return previous_sum + unique_values[bucket] * float(partial_count)


@njit(cache=True)
def _counted_prefix_square_at_rank_numba(
    unique_values: np.ndarray,
    cumulative_counts: np.ndarray,
    cumulative_squares: np.ndarray,
    rank: int,
) -> float:
    bucket = _counted_bucket_for_rank_numba(cumulative_counts, rank)
    previous_count = 0 if bucket == 0 else cumulative_counts[bucket - 1]
    previous_square = 0.0 if bucket == 0 else cumulative_squares[bucket - 1]
    partial_count = rank - previous_count + 1
    value = unique_values[bucket]
    return previous_square + value * value * float(partial_count)


@njit(cache=True)
def _counted_value_at_rank_numba(
    unique_values: np.ndarray,
    cumulative_counts: np.ndarray,
    rank: int,
) -> float:
    return float(unique_values[_counted_bucket_for_rank_numba(cumulative_counts, rank)])


@njit(cache=True)
def _counted_bucket_for_rank_numba(cumulative_counts: np.ndarray, rank: int) -> int:
    low = 0
    high = cumulative_counts.size - 1
    while low < high:
        middle = (low + high) // 2
        if rank < cumulative_counts[middle]:
            high = middle
        else:
            low = middle + 1
    return low


@njit(cache=True)
def _sample_variance_numba(count: int, total: float, square_total: float) -> float:
    if count <= 1:
        return 0.0
    variance = (square_total - total * total / float(count)) / float(count - 1)
    if variance > 0.0:
        return variance
    return 0.0


@njit(cache=True)
def _running_variance_numba(values: np.ndarray) -> np.ndarray:
    size = values.size
    output = np.zeros(size, dtype=np.float64)
    if size < 2:
        return output

    running_sum = float(values[0])
    previous_mean = running_sum
    accumulator = 0.0
    for index in range(1, size):
        value = float(values[index])
        running_sum += value
        mean = running_sum / float(index + 1)
        accumulator += (value - previous_mean) * (value - mean)
        output[index] = accumulator / float(index)
        previous_mean = mean
    return output
