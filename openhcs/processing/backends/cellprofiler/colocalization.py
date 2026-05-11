"""Colocalization backends for CellProfiler-compatible measurements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)


class ColocalizationCostesBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Costes thresholding primitives keyed by OpenHCS memory/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def linear_costes(
        self,
        first_pixels: np.ndarray,
        second_pixels: np.ndarray,
        scale_max: int,
        fast_mode: bool,
    ) -> tuple[float, float]:
        """Return CellProfiler linear Costes thresholds."""

    @abstractmethod
    def scaled_second_channel_costes(
        self,
        first_pixels: np.ndarray,
        second_pixels: np.ndarray,
        scale_max: int,
    ) -> tuple[float, float]:
        """Return CellProfiler scaled-bin second-channel Costes thresholds."""

    @abstractmethod
    def correlation_slopes(
        self,
        first_pixels: np.ndarray,
        second_pixels: np.ndarray,
    ) -> tuple[float, float, float]:
        """Return Pearson correlation plus forward/reverse regression slopes."""


class NumbaNumpyColocalizationCostesBackendStrategy(
    ColocalizationCostesBackendStrategy
):
    """Numba implementation of Costes threshold searches."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def linear_costes(
        self,
        first_pixels: np.ndarray,
        second_pixels: np.ndarray,
        scale_max: int,
        fast_mode: bool,
    ) -> tuple[float, float]:
        first = np.ascontiguousarray(first_pixels, dtype=np.float64)
        second = np.ascontiguousarray(second_pixels, dtype=np.float64)
        valid, slope, intercept = _regression_line_numba(first, second)
        if not valid:
            return 0.0, 0.0
        if slope > 0.0:
            event_threshold = np.minimum(first, (second - intercept) / slope)
            order = np.argsort(event_threshold)
            sorted_first = np.ascontiguousarray(first[order])
            sorted_second = np.ascontiguousarray(second[order])
            return _linear_costes_sorted_events_numba(
                np.ascontiguousarray(event_threshold[order]),
                np.ascontiguousarray(np.cumsum(sorted_first)),
                np.ascontiguousarray(np.cumsum(sorted_second)),
                np.ascontiguousarray(np.cumsum(sorted_first * sorted_first)),
                np.ascontiguousarray(np.cumsum(sorted_second * sorted_second)),
                np.ascontiguousarray(np.cumsum(sorted_first * sorted_second)),
                int(scale_max),
                bool(fast_mode),
                slope,
                intercept,
            )
        return _linear_costes_numba(
            first,
            second,
            int(scale_max),
            bool(fast_mode),
        )

    def scaled_second_channel_costes(
        self,
        first_pixels: np.ndarray,
        second_pixels: np.ndarray,
        scale_max: int,
    ) -> tuple[float, float]:
        first = np.ascontiguousarray(first_pixels, dtype=np.float64)
        second = np.ascontiguousarray(second_pixels, dtype=np.float64)
        valid, slope, intercept = _regression_line_numba(first, second)
        if not valid:
            return 0.0, 0.0
        if slope > 0.0:
            event_summaries = _quantized_unit_interval_event_summaries(
                first,
                second,
                slope,
                intercept,
                int(scale_max),
            )
            if event_summaries is None:
                event_threshold = np.minimum(second, (slope * first) + intercept)
                unique_events, inverse = np.unique(event_threshold, return_inverse=True)
                counts = np.bincount(inverse)
                first_sum = np.bincount(inverse, weights=first)
                second_sum = np.bincount(inverse, weights=second)
                first_square_sum = np.bincount(inverse, weights=first * first)
                second_square_sum = np.bincount(inverse, weights=second * second)
                product_sum = np.bincount(inverse, weights=first * second)
            else:
                (
                    unique_events,
                    counts,
                    first_sum,
                    second_sum,
                    first_square_sum,
                    second_square_sum,
                    product_sum,
                ) = event_summaries
            return _scaled_second_channel_costes_grouped_events_numba(
                np.ascontiguousarray(unique_events, dtype=np.float64),
                np.ascontiguousarray(np.cumsum(counts), dtype=np.int64),
                np.ascontiguousarray(np.cumsum(first_sum), dtype=np.float64),
                np.ascontiguousarray(np.cumsum(second_sum), dtype=np.float64),
                np.ascontiguousarray(np.cumsum(first_square_sum), dtype=np.float64),
                np.ascontiguousarray(np.cumsum(second_square_sum), dtype=np.float64),
                np.ascontiguousarray(np.cumsum(product_sum), dtype=np.float64),
                int(scale_max),
                slope,
                intercept,
            )
        return _scaled_second_channel_costes_numba(
            first,
            second,
            int(scale_max),
        )

    def correlation_slopes(
        self,
        first_pixels: np.ndarray,
        second_pixels: np.ndarray,
    ) -> tuple[float, float, float]:
        return _correlation_slopes_numba(
            np.ascontiguousarray(first_pixels, dtype=np.float64),
            np.ascontiguousarray(second_pixels, dtype=np.float64),
        )

    def prepare_backend(self) -> None:
        """Compile numba Costes kernels outside measured execution."""
        first = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32)
        second = np.flipud(first.reshape((64, 64))).ravel().copy()
        self.correlation_slopes(first, second)
        thresholded_colocalization_metrics(first, second, 15.0, True, True, True)
        self.linear_costes(first, second, 255, False)
        quantized_codes = (np.arange(64 * 64, dtype=np.uint16) % 512) + 1024
        quantized = quantized_codes.astype(np.float32) / np.float32(65535)
        self.scaled_second_channel_costes(quantized, quantized.copy(), 255)


def costes_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> ColocalizationCostesBackendStrategy:
    """Resolve the explicit/default Costes backend for NumPy data."""
    return ColocalizationCostesBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


class UnitIntervalDenseRankSemantics:
    """Shared dense-code/rank semantics for quantized CellProfiler intensities."""

    @classmethod
    def ranks(
        cls,
        values: np.ndarray,
        *,
        preferred_scale: int | None = None,
        proven_unit_interval_scale: int | None = None,
    ) -> np.ndarray:
        if proven_unit_interval_scale is not None:
            return cls.ranks_for_proven_unit_interval(
                values,
                int(proven_unit_interval_scale),
            )
        quantized_ranks = cls.ranks_for_integer_unit_interval(
            values,
            preferred_scale=preferred_scale,
        )
        if quantized_ranks is not None:
            return quantized_ranks
        return np.ascontiguousarray(
            np.unique(values, return_inverse=True)[1],
            dtype=np.int64,
        )

    @staticmethod
    def ranks_for_proven_unit_interval(
        values: np.ndarray,
        scale: int,
    ) -> np.ndarray:
        """Return dense ranks when metadata proves values are exact code / scale."""
        codes = np.rint(np.asarray(values) * int(scale)).astype(np.int64, copy=False)
        present = np.bincount(codes.ravel(), minlength=int(scale) + 1) > 0
        lookup = np.cumsum(present, dtype=np.int64) - 1
        return np.ascontiguousarray(lookup[codes], dtype=np.int64)

    @classmethod
    def ranks_for_integer_unit_interval(
        cls,
        values: np.ndarray,
        *,
        preferred_scale: int | None = None,
    ) -> np.ndarray | None:
        code_result = cls.integer_codes(values, preferred_scale=preferred_scale)
        if code_result is None:
            return None
        codes, scale = code_result
        return cls.dense_codes_and_values(codes, scale)[0]

    @staticmethod
    def integer_codes(
        values: np.ndarray,
        *,
        preferred_scale: int | None = None,
    ) -> tuple[np.ndarray, int] | None:
        values_array = np.asarray(values)
        if values_array.size == 0 or values_array.dtype.kind not in {"f", "u", "i"}:
            return None
        minimum = np.min(values_array)
        maximum = np.max(values_array)
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            return None
        if minimum < 0.0 or maximum > 1.0:
            return None
        scales = (
            (preferred_scale, 65535, 255)
            if preferred_scale is not None
            else (65535, 255)
        )
        for scale in dict.fromkeys(int(candidate) for candidate in scales if candidate):
            valid, codes = _integer_unit_interval_codes_for_scale_numba(
                np.ascontiguousarray(values_array.ravel(), dtype=np.float64),
                scale,
            )
            if valid:
                return codes, scale
        return None

    @staticmethod
    def dense_codes_and_values(
        codes: np.ndarray,
        scale: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        present = np.bincount(codes.ravel()) > 0
        lookup = np.cumsum(present, dtype=np.int64) - 1
        dense = np.ascontiguousarray(lookup[codes], dtype=np.int64)
        values = np.flatnonzero(present).astype(np.float32, copy=False)
        return dense, (values / np.float32(scale)).astype(np.float64, copy=False)


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
    sorted_event_threshold: np.ndarray,
    prefix_x: np.ndarray,
    prefix_y: np.ndarray,
    prefix_x2: np.ndarray,
    prefix_y2: np.ndarray,
    prefix_xy: np.ndarray,
    scale_max: int,
    fast_mode: bool,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
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


def _quantized_unit_interval_event_summaries(
    first: np.ndarray,
    second: np.ndarray,
    slope: float,
    intercept: float,
    preferred_scale: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    first_code_result = UnitIntervalDenseRankSemantics.integer_codes(
        first,
        preferred_scale=preferred_scale,
    )
    second_code_result = UnitIntervalDenseRankSemantics.integer_codes(
        second,
        preferred_scale=preferred_scale,
    )
    if first_code_result is None or second_code_result is None:
        return None
    first_codes, first_scale = first_code_result
    second_codes, second_scale = second_code_result
    first_dense, first_values = UnitIntervalDenseRankSemantics.dense_codes_and_values(
        first_codes,
        first_scale,
    )
    second_dense, second_values = UnitIntervalDenseRankSemantics.dense_codes_and_values(
        second_codes,
        second_scale,
    )
    second_value_count = second_values.size
    combined = first_dense * second_value_count + second_dense
    pair_counts = np.bincount(
        combined,
        minlength=first_values.size * second_value_count,
    )
    active_pairs = np.flatnonzero(pair_counts)
    if active_pairs.size == 0:
        return None
    first_index = active_pairs // second_value_count
    second_index = active_pairs - first_index * second_value_count
    counts = pair_counts[active_pairs].astype(np.float64, copy=False)
    first_pair_values = first_values[first_index]
    second_pair_values = second_values[second_index]
    event_thresholds = np.minimum(
        second_pair_values,
        slope * first_pair_values + intercept,
    )
    unique_events, inverse = np.unique(event_thresholds, return_inverse=True)
    return (
        unique_events,
        np.bincount(inverse, weights=counts).astype(np.int64, copy=False),
        np.bincount(inverse, weights=counts * first_pair_values),
        np.bincount(inverse, weights=counts * second_pair_values),
        np.bincount(inverse, weights=counts * first_pair_values * first_pair_values),
        np.bincount(inverse, weights=counts * second_pair_values * second_pair_values),
        np.bincount(inverse, weights=counts * first_pair_values * second_pair_values),
    )


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
    sorted_event_threshold: np.ndarray,
    prefix_x: np.ndarray,
    prefix_y: np.ndarray,
    prefix_x2: np.ndarray,
    prefix_y2: np.ndarray,
    prefix_xy: np.ndarray,
    scale_max: int,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
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
    sorted_event_threshold: np.ndarray,
    prefix_count: np.ndarray,
    prefix_x: np.ndarray,
    prefix_y: np.ndarray,
    prefix_x2: np.ndarray,
    prefix_y2: np.ndarray,
    prefix_xy: np.ndarray,
    scale_max: int,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
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


__all__ = [
    "ColocalizationCostesBackendStrategy",
    "NumbaNumpyColocalizationCostesBackendStrategy",
    "UnitIntervalDenseRankSemantics",
    "costes_backend",
    "thresholded_colocalization_metrics",
]
