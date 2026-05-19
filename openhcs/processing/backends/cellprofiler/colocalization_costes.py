"""Costes and thresholded colocalization kernels for CellProfiler-compatible measurements."""

from __future__ import annotations

import numpy as np
from numba import njit

from openhcs.processing.backends.cellprofiler.colocalization_costes_prefix import (
    _correlation_slopes_numba,
    _costes_manders_numba,
    _event_count_for_threshold_numba,
    _integer_unit_interval_codes_for_scale_numba,
    _linear_costes_numba,
    _linear_costes_sorted_events_numba,
    _max_from_prefix_numba,
    _max_pair_numba,
    _pearson_below_threshold_numba,
    _prefix_pearson_from_count_numba,
    _prefix_pearson_from_group_count_numba,
    _prefix_pearson_numba,
    _prefix_value_numba,
    _regression_line_numba,
    _scaled_second_channel_costes_grouped_events_numba,
    _scaled_second_channel_costes_numba,
    _scaled_second_channel_costes_sorted_events_numba,
    UnitIntervalDenseRankSemantics,
    quantized_unit_interval_event_summaries,
)


def thresholded_colocalization_metrics(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    threshold_percent: float,
    do_manders: bool,
    do_rwc: bool,
    do_overlap: bool,
    preferred_scale: int | None = None,
    proven_unit_interval_scale: int | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    """Return CellProfiler thresholded Manders, RWC, and overlap metrics."""
    first = np.ascontiguousarray(first_pixels)
    second = np.ascontiguousarray(second_pixels)
    if not do_rwc:
        empty_ranks = np.empty(0, dtype=np.int64)
        return _thresholded_colocalization_metrics_with_ranks_numba(
            first,
            second,
            empty_ranks,
            empty_ranks,
            threshold_percent,
            do_manders,
            False,
            do_overlap,
        )

    first_ranks = UnitIntervalDenseRankSemantics.ranks(
        first,
        preferred_scale=preferred_scale,
        proven_unit_interval_scale=proven_unit_interval_scale,
    )
    second_ranks = UnitIntervalDenseRankSemantics.ranks(
        second,
        preferred_scale=preferred_scale,
        proven_unit_interval_scale=proven_unit_interval_scale,
    )
    return _thresholded_colocalization_metrics_with_ranks_numba(
        first,
        second,
        first_ranks,
        second_ranks,
        threshold_percent,
        do_manders,
        True,
        do_overlap,
    )


def costes_above_threshold_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    """Return CellProfiler Costes inclusion mask for a scalar threshold."""
    if threshold <= 0:
        return values >= threshold
    return values > threshold


@njit(cache=True)
def object_colocalization_base_reductions(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    object_labels: np.ndarray,
    object_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(object_count, dtype=np.float64)
    sum1 = np.zeros(object_count, dtype=np.float64)
    sum2 = np.zeros(object_count, dtype=np.float64)
    sum1_sq = np.zeros(object_count, dtype=np.float64)
    sum2_sq = np.zeros(object_count, dtype=np.float64)
    product_sum = np.zeros(object_count, dtype=np.float64)
    max1 = np.zeros(object_count, dtype=np.float64)
    max2 = np.zeros(object_count, dtype=np.float64)
    for index in range(object_labels.size):
        label_index = int(object_labels[index]) - 1
        first_value = float(first_pixels[index])
        second_value = float(second_pixels[index])
        counts[label_index] += 1.0
        sum1[label_index] += first_value
        sum2[label_index] += second_value
        sum1_sq[label_index] += first_value * first_value
        sum2_sq[label_index] += second_value * second_value
        product_sum[label_index] += first_value * second_value
        if first_value > max1[label_index]:
            max1[label_index] = first_value
        if second_value > max2[label_index]:
            max2[label_index] = second_value
    return counts, sum1, sum2, sum1_sq, sum2_sq, product_sum, max1, max2


@njit(cache=True)
def object_colocalization_threshold_reductions(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    object_labels: np.ndarray,
    threshold_1: np.ndarray,
    threshold_2: np.ndarray,
    costes_threshold_1: float,
    costes_threshold_2: float,
    first_costes_denominator_threshold: float,
    second_costes_denominator_threshold: float,
    object_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    total_first_threshold = np.zeros(object_count, dtype=np.float64)
    total_second_threshold = np.zeros(object_count, dtype=np.float64)
    threshold_sum1 = np.zeros(object_count, dtype=np.float64)
    threshold_sum2 = np.zeros(object_count, dtype=np.float64)
    threshold_sum1_sq = np.zeros(object_count, dtype=np.float64)
    threshold_sum2_sq = np.zeros(object_count, dtype=np.float64)
    threshold_product_sum = np.zeros(object_count, dtype=np.float64)
    total_first_costes = np.zeros(object_count, dtype=np.float64)
    total_second_costes = np.zeros(object_count, dtype=np.float64)
    costes_sum1 = np.zeros(object_count, dtype=np.float64)
    costes_sum2 = np.zeros(object_count, dtype=np.float64)
    for index in range(object_labels.size):
        label_index = int(object_labels[index]) - 1
        first_value = float(first_pixels[index])
        second_value = float(second_pixels[index])
        first_above = first_value >= threshold_1[label_index]
        second_above = second_value >= threshold_2[label_index]
        if first_above:
            total_first_threshold[label_index] += first_value
        if second_above:
            total_second_threshold[label_index] += second_value
        if first_above and second_above:
            threshold_sum1[label_index] += first_value
            threshold_sum2[label_index] += second_value
            threshold_sum1_sq[label_index] += first_value * first_value
            threshold_sum2_sq[label_index] += second_value * second_value
            threshold_product_sum[label_index] += first_value * second_value
        first_above_costes = (
            first_value >= costes_threshold_1
            if costes_threshold_1 <= 0.0
            else first_value > costes_threshold_1
        )
        second_above_costes = (
            second_value >= costes_threshold_2
            if costes_threshold_2 <= 0.0
            else second_value > costes_threshold_2
        )
        if first_value >= first_costes_denominator_threshold:
            total_first_costes[label_index] += first_value
        if second_value >= second_costes_denominator_threshold:
            total_second_costes[label_index] += second_value
        if first_above_costes and second_above_costes:
            costes_sum1[label_index] += first_value
            costes_sum2[label_index] += second_value
    return (
        total_first_threshold,
        total_second_threshold,
        threshold_sum1,
        threshold_sum2,
        threshold_sum1_sq,
        threshold_sum2_sq,
        threshold_product_sum,
        total_first_costes,
        total_second_costes,
        costes_sum1,
        costes_sum2,
    )


@njit(cache=True)
def _thresholded_colocalization_metrics_with_ranks_numba(
    first: np.ndarray,
    second: np.ndarray,
    first_ranks: np.ndarray,
    second_ranks: np.ndarray,
    threshold_percent: float,
    do_manders: bool,
    do_rwc: bool,
    do_overlap: bool,
) -> tuple[float, float, float, float, float, float, float]:
    count = first.size
    nan_value = np.nan
    if count == 0:
        return (
            nan_value,
            nan_value,
            nan_value,
            nan_value,
            nan_value,
            nan_value,
            nan_value,
        )

    first_max = first[0]
    second_max = second[0]
    for index in range(1, count):
        if first[index] > first_max:
            first_max = first[index]
        if second[index] > second_max:
            second_max = second[index]

    first_threshold = threshold_percent * first_max / 100.0
    second_threshold = threshold_percent * second_max / 100.0

    first_threshold_total = 0.0
    second_threshold_total = 0.0
    combined_first_total = 0.0
    combined_second_total = 0.0
    combined_first_square_total = 0.0
    combined_second_square_total = 0.0
    combined_product_total = 0.0
    combined_count = 0
    for index in range(count):
        first_value = first[index]
        second_value = second[index]
        first_above = first_value > first_threshold
        second_above = second_value > second_threshold
        if first_above:
            first_threshold_total += first_value
        if second_above:
            second_threshold_total += second_value
        if first_above and second_above:
            combined_count += 1
            combined_first_total += first_value
            combined_second_total += second_value
            combined_first_square_total += first_value * first_value
            combined_second_square_total += second_value * second_value
            combined_product_total += first_value * second_value

    if combined_count == 0:
        return (
            nan_value,
            nan_value,
            nan_value,
            nan_value,
            nan_value,
            nan_value,
            nan_value,
        )

    manders_m1 = nan_value
    manders_m2 = nan_value
    if do_manders and first_threshold_total > 0.0 and second_threshold_total > 0.0:
        manders_m1 = combined_first_total / first_threshold_total
        manders_m2 = combined_second_total / second_threshold_total

    overlap = nan_value
    k1 = nan_value
    k2 = nan_value
    if do_overlap:
        denominator = np.sqrt(
            combined_first_square_total * combined_second_square_total
        )
        if denominator > 0.0:
            overlap = combined_product_total / denominator
        if combined_first_square_total > 0.0:
            k1 = combined_product_total / combined_first_square_total
        if combined_second_square_total > 0.0:
            k2 = combined_product_total / combined_second_square_total

    rwc1 = nan_value
    rwc2 = nan_value
    if do_rwc and first_threshold_total > 0.0 and second_threshold_total > 0.0:
        max_rank = 0
        for index in range(count):
            if first_ranks[index] > max_rank:
                max_rank = first_ranks[index]
            if second_ranks[index] > max_rank:
                max_rank = second_ranks[index]
        rank_count = float(max_rank + 1)
        weighted_first_total = 0.0
        weighted_second_total = 0.0
        for index in range(count):
            first_value = first[index]
            second_value = second[index]
            if first_value <= first_threshold or second_value <= second_threshold:
                continue
            rank_delta = first_ranks[index] - second_ranks[index]
            if rank_delta < 0:
                rank_delta = -rank_delta
            weight = (rank_count - float(rank_delta)) / rank_count
            weighted_first_total += first_value * weight
            weighted_second_total += second_value * weight
        rwc1 = weighted_first_total / first_threshold_total
        rwc2 = weighted_second_total / second_threshold_total

    return manders_m1, manders_m2, rwc1, rwc2, overlap, k1, k2

