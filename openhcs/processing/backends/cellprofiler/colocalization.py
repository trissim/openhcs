"""Colocalization backends for CellProfiler-compatible measurements."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, make_dataclass, replace
from enum import Enum
from typing import Tuple

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import AlignedImageStack
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import (
    measurement_image_batch_executor,
    special_inputs,
    special_outputs,
)
from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
from openhcs.core.runtime_values import (
    DenseObjectLabelAggregation,
    ImagePayloadChannelProjection,
    image_intensity_scale_for_dtype,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.materialization import csv_materializer


_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


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
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
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


class CostesMethod(Enum):
    FASTER = "faster"
    FAST = "fast"
    ACCURATE = "accurate"


@dataclass(slots=True)
class ColocalizationMeasurements:
    """Colocalization measurements between two channels."""
    slice_index: int
    correlation: float
    slope: float
    slope_reverse: float
    overlap: float
    k1: float
    k2: float
    manders_m1: float
    manders_m2: float
    rwc1: float
    rwc2: float
    costes_m1: float
    costes_m2: float
    costes_threshold_1: float
    costes_threshold_2: float


class ColocalizationMeasurementSchema:
    """Authoritative row schema for image- and object-scoped colocalization."""

    object_label_field = ("object_label", int)

    @classmethod
    def object_measurement_type(
        cls,
        measurement_type: type[ColocalizationMeasurements],
    ) -> type:
        measurement_fields = tuple(fields(measurement_type))
        row_fields = (
            (measurement_fields[0].name, measurement_fields[0].type),
            cls.object_label_field,
            *(
                (field.name, field.type)
                for field in measurement_fields[1:]
            ),
        )
        return make_dataclass(
            "ObjectColocalizationMeasurements",
            row_fields,
            slots=True,
            namespace={
                "__module__": __name__,
                "__doc__": "Colocalization measurements scoped to one labeled object.",
                "from_measurement": classmethod(cls.from_measurement),
                "from_values": classmethod(cls.from_values),
            },
        )

    @staticmethod
    def finite_or_zero(value: float) -> float:
        return float(value) if np.isfinite(value) else 0.0

    @staticmethod
    def from_measurement(
        row_type: type,
        *,
        object_label: int,
        measurement: ColocalizationMeasurements,
    ) -> object:
        measurement_values = asdict(measurement)
        return row_type(
            object_label=object_label,
            **measurement_values,
        )

    @classmethod
    def from_values(
        cls,
        row_type: type,
        object_label: int,
        *,
        correlation: float = 0.0,
        slope: float = 0.0,
        slope_reverse: float = 0.0,
        overlap: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        manders_m1: float = 0.0,
        manders_m2: float = 0.0,
        rwc1: float = 0.0,
        rwc2: float = 0.0,
        costes_m1: float = 0.0,
        costes_m2: float = 0.0,
        costes_threshold_1: float = 0.0,
        costes_threshold_2: float = 0.0,
    ) -> object:
        """Build one object-row record using CellProfiler finite-value semantics."""
        return row_type(
            slice_index=0,
            object_label=object_label,
            correlation=cls.finite_or_zero(correlation),
            slope=cls.finite_or_zero(slope),
            slope_reverse=cls.finite_or_zero(slope_reverse),
            overlap=cls.finite_or_zero(overlap),
            k1=cls.finite_or_zero(k1),
            k2=cls.finite_or_zero(k2),
            manders_m1=cls.finite_or_zero(manders_m1),
            manders_m2=cls.finite_or_zero(manders_m2),
            rwc1=cls.finite_or_zero(rwc1),
            rwc2=cls.finite_or_zero(rwc2),
            costes_m1=float(costes_m1),
            costes_m2=float(costes_m2),
            costes_threshold_1=cls.finite_or_zero(costes_threshold_1),
            costes_threshold_2=cls.finite_or_zero(costes_threshold_2),
        )


ObjectColocalizationMeasurements = (
    ColocalizationMeasurementSchema.object_measurement_type(ColocalizationMeasurements)
)


@dataclass(frozen=True)
class ColocalizationMeasurementOptions:
    """Metric switches shared by image- and object-scoped colocalization."""

    threshold_percent: float
    do_correlation: bool
    do_manders: bool
    do_rwc: bool
    do_overlap: bool
    do_costes: bool
    costes_method: CostesMethod
    scale_max: int
    unit_interval_intensity_scale: int | None = None
    costes_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "costes_method", CostesMethod(self.costes_method))


@dataclass(frozen=True, slots=True)
class ColocalizationCostesThresholds:
    """Precomputed Costes thresholds for one resolved image source pair."""

    first: float
    second: float
    first_denominator: float
    second_denominator: float

    @classmethod
    def from_thresholds(
        cls,
        first: float,
        second: float,
        *,
        scale_max: int,
    ) -> "ColocalizationCostesThresholds":
        first_denominator = float(first)
        second_denominator = float(second)
        first_threshold = float(
            np.nextafter(np.float32(first_denominator), np.float32(np.inf))
        )
        second_threshold = float(
            np.nextafter(np.float32(second_denominator), np.float32(np.inf))
        )
        scaled_first = first_denominator * scale_max
        nearest_first_bin = round(scaled_first)
        first_denominator = (
            nearest_first_bin / scale_max
            if scale_max > 0
            and np.isfinite(first)
            and np.isclose(scaled_first, nearest_first_bin, rtol=0.0, atol=1e-3)
            else first_denominator
        )
        return cls(
            first=first_threshold,
            second=second_threshold,
            first_denominator=float(first_denominator),
            second_denominator=second_denominator,
        )


@dataclass(frozen=True, slots=True)
class ColocalizationImagePairCacheKey:
    """Batch-local identity for one resolved colocalization image pair."""

    image_payload_id: int
    image_data_id: int
    channel_1: int
    channel_2: int


@dataclass(frozen=True, slots=True)
class ColocalizationObjectLabelCacheKey:
    """Batch-local identity for labels projected into one image-pair mask."""

    label_data_id: int
    pair_valid_mask_id: int


@dataclass(frozen=True, slots=True)
class ColocalizationCostesThresholdCacheKey:
    """Batch-local identity for Costes thresholds over one image pair."""

    image_payload_id: int
    image_data_id: int
    channel_1: int
    channel_2: int
    method: CostesMethod
    scale_max: int
    backend_provider: object


@dataclass(frozen=True, slots=True)
class ColocalizationImagePairContext:
    """Resolved image-pair pixels shared by batched object colocalization calls."""

    image_data: np.ndarray
    image_float: np.ndarray
    first_image: np.ndarray
    second_image: np.ndarray
    pair_valid_mask: np.ndarray | None
    full_first_pixels: np.ndarray
    full_second_pixels: np.ndarray

    @staticmethod
    def valid_mask(
        image: object,
        image_data: np.ndarray,
        channel_1: int,
        channel_2: int,
    ) -> np.ndarray | None:
        """Return CellProfiler-style valid pixels for a two-image measurement."""
        first_pixels = image_data[channel_1]
        second_pixels = image_data[channel_2]
        mask = image_payload_mask(image)
        if mask is None:
            if bool(np.all(np.isfinite(first_pixels))) and bool(
                np.all(np.isfinite(second_pixels))
            ):
                return None
            return np.isfinite(first_pixels) & np.isfinite(second_pixels)

        valid = np.isfinite(first_pixels) & np.isfinite(second_pixels)
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape == valid.shape:
            return valid & mask_array
        if mask_array.shape == image_data.shape:
            return valid & mask_array[channel_1] & mask_array[channel_2]
        if (
            mask_array.ndim >= 3
            and mask_array.shape[0] == image_data.shape[0]
            and mask_array.shape[1:] == valid.shape
        ):
            return valid & mask_array[channel_1] & mask_array[channel_2]
        raise ValueError(
            "MeasureColocalization image mask must match the shared spatial "
            f"domain or channel stack; got mask {mask_array.shape!r} for image "
            f"{image_data.shape!r}."
        )

    @classmethod
    def from_request(
        cls,
        image: object,
        *,
        channel_1: int,
        channel_2: int,
    ) -> "ColocalizationImagePairContext":
        image_data = cls.measurement_pixels(image)
        image_float = cls.cellprofiler_float_pixels(image)
        first_image = image_float[channel_1]
        second_image = image_float[channel_2]
        pair_valid_mask = cls.valid_mask(
            image,
            image_float,
            channel_1,
            channel_2,
        )
        if pair_valid_mask is None:
            full_first_pixels = first_image.ravel()
            full_second_pixels = second_image.ravel()
        else:
            full_first_pixels = first_image[pair_valid_mask]
            full_second_pixels = second_image[pair_valid_mask]
        return cls(
            image_data=image_data,
            image_float=image_float,
            first_image=first_image,
            second_image=second_image,
            pair_valid_mask=pair_valid_mask,
            full_first_pixels=full_first_pixels,
            full_second_pixels=full_second_pixels,
        )

    @staticmethod
    def requires_slice_local_context(image: object) -> bool:
        """Return whether context resolution must happen after slice projection."""
        return isinstance(image_payload_data(image), AlignedImageStack)

    @staticmethod
    def measurement_pixels(image: object) -> np.ndarray:
        """Return stacked image pixels for colocalization measurement."""
        image_data = image_payload_data(image)
        if isinstance(image_data, AlignedImageStack):
            return np.stack(
                tuple(
                    np.asarray(image_payload_data(slice_payload))
                    for slice_payload in image_data.slices
                ),
                axis=0,
            )
        return np.asarray(image_data)

    @classmethod
    def cellprofiler_float_pixels(cls, image: object) -> np.ndarray:
        """Return image pixels in CellProfiler's native float image domain."""
        return np.asarray(cls.measurement_pixels(image), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ColocalizationObjectLabelContext:
    """Resolved object-label reductions shared by batched image-pair calls."""

    labels: np.ndarray
    max_label: int
    label_range: np.ndarray
    object_mask: np.ndarray
    object_labels: np.ndarray
    aggregation: DenseObjectLabelAggregation
    object_counts: np.ndarray

    @classmethod
    def from_labels(
        cls,
        labels: object,
        *,
        pair_valid_mask: np.ndarray | None,
    ) -> "ColocalizationObjectLabelContext":
        return cls.from_dense_labels(
            object_label_dense_array(labels, dtype=np.int32),
            pair_valid_mask=pair_valid_mask,
        )

    @classmethod
    def from_dense_labels(
        cls,
        label_array: np.ndarray,
        *,
        pair_valid_mask: np.ndarray | None,
    ) -> "ColocalizationObjectLabelContext":
        """Build reductions from an already-resolved dense label array."""
        max_label = int(np.max(label_array)) if label_array.size else 0
        label_range = np.arange(1, max_label + 1, dtype=np.int32)
        object_mask = label_array > 0
        if pair_valid_mask is not None:
            object_mask = object_mask & pair_valid_mask
        object_labels = label_array[object_mask].astype(np.int32, copy=False)
        aggregation = DenseObjectLabelAggregation(
            labels=object_labels,
            object_count=max_label,
        )
        return cls(
            labels=label_array,
            max_label=max_label,
            label_range=label_range,
            object_mask=object_mask,
            object_labels=object_labels,
            aggregation=aggregation,
            object_counts=aggregation.counts(),
        )


def _colocalization_measurement(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    *,
    options: ColocalizationMeasurementOptions,
    valid_mask: np.ndarray | None = None,
) -> ColocalizationMeasurements:
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    corr = np.nan
    slope = np.nan
    slope_reverse = np.nan
    overlap = np.nan
    k1 = np.nan
    k2 = np.nan
    m1 = np.nan
    m2 = np.nan
    rwc1 = np.nan
    rwc2 = np.nan
    c1 = np.nan
    c2 = np.nan
    thr_fi_c = np.nan
    thr_si_c = np.nan

    if valid_mask is None:
        first_array = np.asarray(first_pixels)
        second_array = np.asarray(second_pixels)
        finite_mask = np.isfinite(first_array) & np.isfinite(second_array)
        if np.any(finite_mask):
            if bool(np.all(finite_mask)):
                fi = np.ravel(first_array)
                si = np.ravel(second_array)
            else:
                fi = first_array[finite_mask]
                si = second_array[finite_mask]
        else:
            fi = np.empty(0, dtype=np.asarray(first_pixels).dtype)
            si = np.empty(0, dtype=np.asarray(second_pixels).dtype)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if np.any(mask):
            fi = first_pixels[mask]
            si = second_pixels[mask]
        else:
            fi = np.empty(0, dtype=np.asarray(first_pixels).dtype)
            si = np.empty(0, dtype=np.asarray(second_pixels).dtype)

    _log_profile(
        "coloc_prepare_pixels",
        time.perf_counter() - phase_started_at,
        function="_colocalization_measurement",
        pixels=fi.size,
    )
    if fi.size:
        if options.do_correlation:
            phase_started_at = time.perf_counter()
            corr, slope, slope_reverse = (
                ColocalizationCostesBackendStrategy.for_memory_type(
                    backend_provider=options.costes_backend_provider,
                ).correlation_slopes(fi, si)
            )
            _log_profile(
                "coloc_correlation",
                time.perf_counter() - phase_started_at,
                function="_colocalization_measurement",
            )

        if any((options.do_manders, options.do_rwc, options.do_overlap)):
            phase_started_at = time.perf_counter()
            (
                m1,
                m2,
                rwc1,
                rwc2,
                overlap,
                k1,
                k2,
            ) = thresholded_colocalization_metrics(
                np.ascontiguousarray(fi),
                np.ascontiguousarray(si),
                float(options.threshold_percent),
                bool(options.do_manders),
                bool(options.do_rwc),
                bool(options.do_overlap),
                int(options.scale_max),
                options.unit_interval_intensity_scale,
            )
            _log_profile(
                "coloc_thresholded_metrics",
                time.perf_counter() - phase_started_at,
                function="_colocalization_measurement",
            )

        if options.do_costes:
            phase_started_at = time.perf_counter()
            if options.costes_method == CostesMethod.FASTER:
                thr_fi_c, thr_si_c = costes_backend(
                    backend_provider=options.costes_backend_provider,
                ).scaled_second_channel_costes(
                    fi,
                    si,
                    options.scale_max,
                )
            else:
                fast_mode = options.costes_method == CostesMethod.FAST
                thr_fi_c, thr_si_c = costes_backend(
                    backend_provider=options.costes_backend_provider,
                ).linear_costes(
                    fi,
                    si,
                    options.scale_max,
                    fast_mode,
                )
            _log_profile(
                "coloc_costes_thresholds",
                time.perf_counter() - phase_started_at,
                function="_colocalization_measurement",
                method=options.costes_method.value,
            )

            phase_started_at = time.perf_counter()
            c1, c2 = _costes_manders_numba(
                np.ascontiguousarray(fi),
                np.ascontiguousarray(si),
                _pixel_dtype_threshold(fi, thr_fi_c),
                _pixel_dtype_threshold(si, thr_si_c),
            )
            _log_profile(
                "coloc_costes_manders",
                time.perf_counter() - phase_started_at,
                function="_colocalization_measurement",
            )

    result = ColocalizationMeasurements(
        slice_index=0,
        correlation=float(corr) if not np.isnan(corr) else 0.0,
        slope=float(slope) if not np.isnan(slope) else 0.0,
        slope_reverse=float(slope_reverse) if not np.isnan(slope_reverse) else 0.0,
        overlap=float(overlap) if not np.isnan(overlap) else 0.0,
        k1=float(k1) if not np.isnan(k1) else 0.0,
        k2=float(k2) if not np.isnan(k2) else 0.0,
        manders_m1=float(m1) if not np.isnan(m1) else 0.0,
        manders_m2=float(m2) if not np.isnan(m2) else 0.0,
        rwc1=float(rwc1) if not np.isnan(rwc1) else 0.0,
        rwc2=float(rwc2) if not np.isnan(rwc2) else 0.0,
        costes_m1=float(c1) if not np.isnan(c1) else 0.0,
        costes_m2=float(c2) if not np.isnan(c2) else 0.0,
        costes_threshold_1=float(thr_fi_c) if not np.isnan(thr_fi_c) else 0.0,
        costes_threshold_2=float(thr_si_c) if not np.isnan(thr_si_c) else 0.0,
    )
    _log_profile(
        "coloc_total",
        time.perf_counter() - total_started_at,
        function="_colocalization_measurement",
    )
    return result


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


def _pixel_dtype_threshold(pixels: np.ndarray, threshold: float) -> float:
    """Round scalar thresholds into the pixel dtype before bin comparisons."""
    return float(np.asarray(threshold, dtype=np.asarray(pixels).dtype).item())


def _cellprofiler_float_pixels(image: np.ndarray) -> np.ndarray:
    """Return image pixels in CellProfiler's native float image domain."""
    return ColocalizationImagePairContext.cellprofiler_float_pixels(image)


def _colocalization_unit_interval_scale(
    image: object,
    channel_1: int,
    channel_2: int,
) -> int | None:
    """Return a shared proof scale when both channels are exact unit interval."""
    metadata = image_payload_metadata(image)
    first_scale = metadata.unit_interval_intensity_scale_for_channel(channel_1)
    second_scale = metadata.unit_interval_intensity_scale_for_channel(channel_2)
    if first_scale is None or second_scale is None:
        return None
    if int(first_scale) != int(second_scale):
        return None
    return int(first_scale)


@numpy
@special_outputs(("colocalization_measurements", csv_materializer(
    fields=["slice_index", "correlation", "slope", "slope_reverse", "overlap", "k1", "k2",
            "manders_m1", "manders_m2", "rwc1", "rwc2",
            "costes_m1", "costes_m2", "costes_threshold_1", "costes_threshold_2"],
    analysis_type="colocalization"
)))
def measure_colocalization(
    image: np.ndarray,
    channel_1: int = 0,
    channel_2: int = 1,
    threshold_percent: float = 15.0,
    do_correlation: bool = True,
    do_manders: bool = True,
    do_rwc: bool = True,
    do_overlap: bool = True,
    do_costes: bool = True,
    costes_method: CostesMethod = CostesMethod.FASTER,
    scale_max: int | None = None,
    costes_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[np.ndarray, ColocalizationMeasurements]:
    """
    Measure colocalization between two channels from an N-channel image.

    Args:
        image: Shape (N, H, W) - N channel images stacked along dim 0
        channel_1: Index of first channel to compare (default 0)
        channel_2: Index of second channel to compare (default 1)
        threshold_percent: Threshold as percentage of max intensity (0-99)
        do_correlation: Calculate Pearson correlation and slope
        do_manders: Calculate Manders coefficients
        do_rwc: Calculate Rank Weighted Colocalization coefficients
        do_overlap: Calculate Overlap coefficients
        do_costes: Calculate Manders coefficients using Costes auto threshold
        costes_method: Method for Costes thresholding (faster, fast, accurate)
        scale_max: Optional explicit maximum scale for Costes calculation. When
            omitted, OpenHCS resolves it from generic source image metadata.
        costes_backend_provider: Optional explicit Costes backend provider.

    Returns:
        Tuple of (first channel image, ColocalizationMeasurements)

    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select images to measure' -> (pipeline-handled)
        'Set threshold as percentage of maximum intensity for the images' -> threshold_percent
        'Run all metrics?' -> (pipeline-handled)
        'Calculate correlation and slope metrics?' -> do_correlation
        'Calculate the Manders coefficients?' -> do_manders
        'Calculate the Rank Weighted Colocalization coefficients?' -> do_rwc
        'Calculate the Overlap coefficients?' -> do_overlap
        'Calculate the Manders coefficients using Costes auto threshold?' -> do_costes
        'Method for Costes thresholding' -> costes_method
    """
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    # Select the two channels to compare
    image_data = image_payload_data(image)
    if channel_1 >= image_data.shape[0] or channel_2 >= image_data.shape[0]:
        raise ValueError(f"Channel indices ({channel_1}, {channel_2}) out of range for image with {image_data.shape[0]} channels")
    _log_profile(
        "measure_coloc_input",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
    )

    phase_started_at = time.perf_counter()
    options = ColocalizationMeasurementOptions(
        threshold_percent=threshold_percent,
        do_correlation=do_correlation,
        do_manders=do_manders,
        do_rwc=do_rwc,
        do_overlap=do_overlap,
        do_costes=do_costes,
        costes_method=costes_method,
        scale_max=ColocalizationCostesThresholdRequest.scale_max_for_image_pair(
            image,
            image_data,
            channel_1,
            channel_2,
            scale_max,
        ),
        unit_interval_intensity_scale=_colocalization_unit_interval_scale(
            image,
            channel_1,
            channel_2,
        ),
        costes_backend_provider=costes_backend_provider,
    )
    _log_profile(
        "measure_coloc_options",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
    )
    phase_started_at = time.perf_counter()
    image_float = _cellprofiler_float_pixels(image_data)
    valid_mask = ColocalizationImagePairContext.valid_mask(
        image,
        image_float,
        channel_1,
        channel_2,
    )
    _log_profile(
        "measure_coloc_prepare_arrays",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
        full_valid=valid_mask is None,
    )
    phase_started_at = time.perf_counter()
    measurements = _colocalization_measurement(
        image_float[channel_1],
        image_float[channel_2],
        options=options,
        valid_mask=valid_mask,
    )
    _log_profile(
        "measure_coloc_metrics",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
    )

    # Return first selected channel as the output image
    phase_started_at = time.perf_counter()
    output = ImagePayloadChannelProjection.from_channel(
        image,
        image_data,
        channel_1,
    ).payload()
    _log_profile(
        "measure_coloc_output_payload",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
    )
    _log_profile(
        "measure_coloc_total",
        time.perf_counter() - total_started_at,
        function="measure_colocalization",
    )
    return output, measurements


@numpy
@special_inputs("labels")
@special_outputs(("object_colocalization_measurements", csv_materializer(
    fields=[
        "slice_index",
        "object_label",
        "correlation",
        "slope",
        "slope_reverse",
        "overlap",
        "k1",
        "k2",
        "manders_m1",
        "manders_m2",
        "rwc1",
        "rwc2",
        "costes_m1",
        "costes_m2",
        "costes_threshold_1",
        "costes_threshold_2",
    ],
    analysis_type="object_colocalization",
)))
def measure_colocalization_objects(
    image: np.ndarray,
    labels: np.ndarray,
    channel_1: int = 0,
    channel_2: int = 1,
    threshold_percent: float = 15.0,
    do_correlation: bool = True,
    do_manders: bool = True,
    do_rwc: bool = True,
    do_overlap: bool = True,
    do_costes: bool = True,
    costes_method: CostesMethod = CostesMethod.FASTER,
    scale_max: int | None = None,
    costes_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    costes_thresholds: ColocalizationCostesThresholds | None = None,
    image_pair_context: ColocalizationImagePairContext | None = None,
    object_label_context: ColocalizationObjectLabelContext | None = None,
) -> Tuple[np.ndarray, list[ObjectColocalizationMeasurements]]:
    """Measure colocalization between two channels within labeled objects."""
    if image_pair_context is None:
        image_pair_context = ColocalizationImagePairContext.from_request(
            image,
            channel_1=channel_1,
            channel_2=channel_2,
        )
    image_data = image_pair_context.image_data
    if object_label_context is None:
        object_label_context = ColocalizationObjectLabelContext.from_labels(
            labels,
            pair_valid_mask=image_pair_context.pair_valid_mask,
        )
    max_label = object_label_context.max_label
    if max_label <= 0:
        return (
            ImagePayloadChannelProjection.from_channel(
                image,
                image_data,
                channel_1,
            ).payload(),
            [],
        )

    label_range = object_label_context.label_range
    options = ColocalizationMeasurementOptions(
        threshold_percent=threshold_percent,
        do_correlation=do_correlation,
        do_manders=do_manders,
        do_rwc=do_rwc,
        do_overlap=do_overlap,
        do_costes=do_costes,
        costes_method=costes_method,
        scale_max=ColocalizationCostesThresholdRequest.scale_max_for_image_pair(
            image,
            image_data,
            channel_1,
            channel_2,
            scale_max,
        ),
        costes_backend_provider=costes_backend_provider,
    )

    first_image = image_pair_context.first_image
    second_image = image_pair_context.second_image
    object_mask = object_label_context.object_mask
    if not object_label_context.object_labels.size:
        return (
            ImagePayloadChannelProjection.from_channel(
                image,
                image_data,
                channel_1,
            ).payload(),
            [
                ObjectColocalizationMeasurements.from_values(int(object_label))
                for object_label in label_range
            ],
        )

    first_pixels = first_image[object_mask]
    second_pixels = second_image[object_mask]
    object_labels = object_label_context.object_labels
    label_aggregation = object_label_context.aggregation
    full_fi = image_pair_context.full_first_pixels
    full_si = image_pair_context.full_second_pixels
    (
        object_counts,
        sum1,
        sum2,
        sum1_sq,
        sum2_sq,
        product_sum,
        max1,
        max2,
    ) = object_colocalization_base_reductions(
        first_pixels,
        second_pixels,
        object_labels,
        max_label,
    )

    corr = np.zeros(max_label, dtype=float)
    slope = np.zeros(max_label, dtype=float)
    slope_reverse = np.zeros(max_label, dtype=float)
    overlap = np.zeros(max_label, dtype=float)
    k1 = np.zeros(max_label, dtype=float)
    k2 = np.zeros(max_label, dtype=float)
    manders_m1 = np.zeros(max_label, dtype=float)
    manders_m2 = np.zeros(max_label, dtype=float)
    rwc1 = np.zeros(max_label, dtype=float)
    rwc2 = np.zeros(max_label, dtype=float)
    costes_m1 = np.zeros(max_label, dtype=float)
    costes_m2 = np.zeros(max_label, dtype=float)
    costes_threshold_1 = np.zeros(max_label, dtype=float)
    costes_threshold_2 = np.zeros(max_label, dtype=float)

    if options.do_correlation:
        with np.errstate(divide="ignore", invalid="ignore"):
            centered_product = product_sum - ((sum1 * sum2) / object_counts)
            centered_first = sum1_sq - ((sum1 * sum1) / object_counts)
            centered_second = sum2_sq - ((sum2 * sum2) / object_counts)
            corr = centered_product / np.sqrt(centered_first * centered_second)
        corr[~np.isfinite(corr)] = np.nan

    threshold_metrics_requested = any(
        (options.do_manders, options.do_rwc, options.do_overlap)
    )
    if threshold_metrics_requested:
        threshold_1 = (
            options.threshold_percent / 100 * max1
        )
        threshold_2 = (
            options.threshold_percent / 100 * max2
        )
        first_above_threshold = first_pixels >= threshold_1[object_labels - 1]
        second_above_threshold = second_pixels >= threshold_2[object_labels - 1]
        combined_threshold = first_above_threshold & second_above_threshold
        combined_threshold_has_values = bool(np.any(combined_threshold))
        fi_thresh = first_pixels[combined_threshold]
        si_thresh = second_pixels[combined_threshold]
        threshold_aggregation = label_aggregation.subset(combined_threshold)

    threshold_c1 = 0.0
    threshold_c2 = 0.0
    first_costes_denominator_threshold = 0.0
    second_costes_denominator_threshold = 0.0
    if options.do_costes and full_fi.size:
        if costes_thresholds is not None:
            resolved_costes_thresholds = costes_thresholds
        elif options.costes_method == CostesMethod.FASTER:
            threshold_c1, threshold_c2 = costes_backend(
                backend_provider=options.costes_backend_provider,
            ).scaled_second_channel_costes(
                full_fi,
                full_si,
                options.scale_max,
            )
            resolved_costes_thresholds = ColocalizationCostesThresholds.from_thresholds(
                threshold_c1,
                threshold_c2,
                scale_max=options.scale_max,
            )
        else:
            threshold_c1, threshold_c2 = costes_backend(
                backend_provider=options.costes_backend_provider,
            ).linear_costes(
                full_fi,
                full_si,
                options.scale_max,
                options.costes_method == CostesMethod.FAST,
            )
            resolved_costes_thresholds = ColocalizationCostesThresholds.from_thresholds(
                threshold_c1,
                threshold_c2,
                scale_max=options.scale_max,
            )
        threshold_c1 = resolved_costes_thresholds.first
        threshold_c2 = resolved_costes_thresholds.second
        first_costes_denominator_threshold = (
            resolved_costes_thresholds.first_denominator
        )
        second_costes_denominator_threshold = (
            resolved_costes_thresholds.second_denominator
        )
        costes_threshold_1.fill(threshold_c1)
        costes_threshold_2.fill(threshold_c2)

    threshold_reductions_requested = threshold_metrics_requested or (
        options.do_costes and full_fi.size
    )
    if threshold_reductions_requested:
        if not threshold_metrics_requested:
            threshold_1 = np.zeros(max_label, dtype=float)
            threshold_2 = np.zeros(max_label, dtype=float)
        (
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
        ) = object_colocalization_threshold_reductions(
            first_pixels,
            second_pixels,
            object_labels,
            threshold_1,
            threshold_2,
            threshold_c1,
            threshold_c2,
            first_costes_denominator_threshold,
            second_costes_denominator_threshold,
            max_label,
        )

    if options.do_manders and combined_threshold_has_values:
        manders_m1 = _divide_measurements(threshold_sum1, total_first_threshold)
        manders_m2 = _divide_measurements(threshold_sum2, total_second_threshold)

    if options.do_rwc:
        rank_image_1 = UnitIntervalDenseRankSemantics.ranks(
            first_pixels,
            preferred_scale=options.scale_max,
            proven_unit_interval_scale=options.unit_interval_intensity_scale,
        )
        rank_image_2 = UnitIntervalDenseRankSemantics.ranks(
            second_pixels,
            preferred_scale=options.scale_max,
            proven_unit_interval_scale=options.unit_interval_intensity_scale,
        )

        max_rank = max(rank_image_1.max(), rank_image_2.max()) + 1
        rank_delta = abs(rank_image_1 - rank_image_2)
        weight = (max_rank - rank_delta) * 1.0 / max_rank
        weight_threshold = weight[combined_threshold]
        if combined_threshold_has_values:
            rwc1 = _divide_measurements(
                threshold_aggregation.sum(fi_thresh * weight_threshold),
                total_first_threshold,
            )
            rwc2 = _divide_measurements(
                threshold_aggregation.sum(si_thresh * weight_threshold),
                total_second_threshold,
            )

    if options.do_overlap and combined_threshold_has_values:
        overlap = _divide_measurements(
            threshold_product_sum,
            np.sqrt(threshold_sum1_sq * threshold_sum2_sq),
        )
        k1 = _divide_measurements(threshold_product_sum, threshold_sum1_sq)
        k2 = _divide_measurements(threshold_product_sum, threshold_sum2_sq)

    if options.do_costes and full_fi.size:
        costes_m1 = _divide_costes_measurements(costes_sum1, total_first_costes)
        costes_m2 = _divide_costes_measurements(costes_sum2, total_second_costes)

    return (
        ImagePayloadChannelProjection.from_channel(
            image,
            image_data,
            channel_1,
        ).payload(),
        [
            ObjectColocalizationMeasurements.from_values(
                int(object_label),
                correlation=corr[index],
                slope=slope[index],
                slope_reverse=slope_reverse[index],
                overlap=overlap[index],
                k1=k1[index],
                k2=k2[index],
                manders_m1=manders_m1[index],
                manders_m2=manders_m2[index],
                rwc1=rwc1[index],
                rwc2=rwc2[index],
                costes_m1=costes_m1[index],
                costes_m2=costes_m2[index],
                costes_threshold_1=costes_threshold_1[index],
                costes_threshold_2=costes_threshold_2[index],
            )
            for index, object_label in enumerate(label_range)
        ],
    )


@dataclass(frozen=True)
class ColocalizationCostesThresholdRequest:
    """Resolved inputs needed to compute one image-pair Costes threshold."""

    image: object
    image_data: np.ndarray
    channel_1: int
    channel_2: int
    method: CostesMethod
    scale_max: int
    backend_provider: BackendProviderInput | None
    image_pair_context: ColocalizationImagePairContext | None = None

    @property
    def cache_key(self) -> ColocalizationCostesThresholdCacheKey:
        """Return the batch-local identity for this resolved source pair."""
        return ColocalizationCostesThresholdCacheKey(
            id(self.image),
            id(self.image_data),
            self.channel_1,
            self.channel_2,
            self.method,
            self.scale_max,
            self.backend_provider,
        )

    @staticmethod
    def scale_max_for_image_pair(
        image: object,
        image_data: np.ndarray,
        channel_1: int,
        channel_2: int,
        explicit_scale_max: int | None,
    ) -> int:
        """Resolve Costes scale from image metadata, with dtype fallback."""
        if explicit_scale_max is not None:
            return int(explicit_scale_max)

        metadata = image_payload_metadata(image)
        metadata_scales = tuple(
            scale
            for scale in (
                metadata.intensity_scale_for_channel(channel_1),
                metadata.intensity_scale_for_channel(channel_2),
            )
            if scale is not None and scale > 0
        )
        if metadata_scales:
            return int(round(max(metadata_scales)))

        dtype_scale = image_intensity_scale_for_dtype(np.asarray(image_data).dtype)
        if dtype_scale is not None and dtype_scale > 0:
            return int(round(dtype_scale))
        return 255

    @classmethod
    def from_batch_request(
        cls,
        request: RuntimeBatchInvocationRequest,
        image_pair_context: ColocalizationImagePairContext | None = None,
    ) -> "ColocalizationCostesThresholdRequest | None":
        """Build a Costes request from runtime invocation metadata."""
        kwargs = request.kwargs
        if not bool(kwargs.get("do_costes", True)):
            return None
        image_data = (
            image_pair_context.image_data
            if image_pair_context is not None
            else image_payload_data(request.image)
        )
        channel_1 = int(kwargs.get("channel_1", 0))
        channel_2 = int(kwargs.get("channel_2", 1))
        return cls(
            image=request.image,
            image_data=image_data,
            channel_1=channel_1,
            channel_2=channel_2,
            method=CostesMethod(kwargs.get("costes_method", CostesMethod.FASTER)),
            scale_max=cls.scale_max_for_image_pair(
                request.image,
                image_data,
                channel_1,
                channel_2,
                kwargs.get("scale_max"),
            ),
            backend_provider=kwargs.get("costes_backend_provider"),
            image_pair_context=image_pair_context,
        )

    def thresholds(self) -> ColocalizationCostesThresholds:
        """Compute Costes thresholds for this resolved image source pair."""
        if self.image_pair_context is None:
            image_pair_context = ColocalizationImagePairContext.from_request(
                self.image,
                channel_1=self.channel_1,
                channel_2=self.channel_2,
            )
        else:
            image_pair_context = self.image_pair_context
        first_pixels = image_pair_context.full_first_pixels
        second_pixels = image_pair_context.full_second_pixels
        if not first_pixels.size:
            return ColocalizationCostesThresholds.from_thresholds(
                0.0,
                0.0,
                scale_max=self.scale_max,
            )
        if self.method is CostesMethod.FASTER:
            first, second = costes_backend(
                backend_provider=self.backend_provider,
            ).scaled_second_channel_costes(
                first_pixels,
                second_pixels,
                self.scale_max,
            )
        else:
            first, second = costes_backend(
                backend_provider=self.backend_provider,
            ).linear_costes(
                first_pixels,
                second_pixels,
                self.scale_max,
                self.method is CostesMethod.FAST,
            )
        return ColocalizationCostesThresholds.from_thresholds(
            first,
            second,
            scale_max=self.scale_max,
        )


class ColocalizationCostesThresholdBatch:
    """Batch-local Costes threshold cache keyed by resolved image-pair identity."""

    def __init__(self) -> None:
        self._thresholds: dict[
            ColocalizationCostesThresholdCacheKey,
            ColocalizationCostesThresholds,
        ] = {}
        self._image_pairs: dict[
            ColocalizationImagePairCacheKey,
            ColocalizationImagePairContext,
        ] = {}
        self._label_contexts: dict[
            ColocalizationObjectLabelCacheKey,
            ColocalizationObjectLabelContext,
        ] = {}

    def image_pair_context(
        self,
        request: RuntimeBatchInvocationRequest,
    ) -> ColocalizationImagePairContext:
        """Return the batch-local resolved image-pair context."""
        kwargs = request.kwargs
        image_data = image_payload_data(request.image)
        channel_1 = int(kwargs.get("channel_1", 0))
        channel_2 = int(kwargs.get("channel_2", 1))
        key = ColocalizationImagePairCacheKey(
            id(request.image),
            id(image_data),
            channel_1,
            channel_2,
        )
        context = self._image_pairs.get(key)
        if context is None:
            context = ColocalizationImagePairContext.from_request(
                request.image,
                channel_1=channel_1,
                channel_2=channel_2,
            )
            self._image_pairs[key] = context
        return context

    def object_label_context(
        self,
        request: RuntimeBatchInvocationRequest,
        image_pair_context: ColocalizationImagePairContext,
    ) -> ColocalizationObjectLabelContext:
        """Return the batch-local resolved object-label context."""
        labels = request.kwargs["labels"]
        label_array = object_label_dense_array(labels, dtype=np.int32)
        key = ColocalizationObjectLabelCacheKey(
            id(label_array),
            id(image_pair_context.pair_valid_mask),
        )
        context = self._label_contexts.get(key)
        if context is None:
            context = ColocalizationObjectLabelContext.from_dense_labels(
                label_array,
                pair_valid_mask=image_pair_context.pair_valid_mask,
            )
            self._label_contexts[key] = context
        return context

    def request_kwargs(
        self,
        request: RuntimeBatchInvocationRequest,
    ) -> dict[str, object]:
        """Return request kwargs with source-pair thresholds materialized once."""
        if ColocalizationImagePairContext.requires_slice_local_context(request.image):
            return dict(request.kwargs)
        image_pair_context = self.image_pair_context(request)
        object_label_context = self.object_label_context(request, image_pair_context)
        threshold_request = ColocalizationCostesThresholdRequest.from_batch_request(
            request,
            image_pair_context,
        )
        thresholds = None
        if threshold_request is not None:
            key = threshold_request.cache_key
            thresholds = self._thresholds.get(key)
            if thresholds is None:
                thresholds = threshold_request.thresholds()
                self._thresholds[key] = thresholds
        kwargs = {
            **request.kwargs,
            "image_pair_context": image_pair_context,
            "object_label_context": object_label_context,
        }
        if thresholds is not None:
            kwargs["costes_thresholds"] = thresholds
        return kwargs


def measure_colocalization_objects_batch(
    func: Callable[..., object],
    requests: tuple[RuntimeBatchInvocationRequest, ...],
    execute_request: Callable[
        [Callable[..., object], RuntimeBatchInvocationRequest],
        object,
    ],
) -> list[object]:
    """Batch object colocalization invocations over shared image-pair thresholds."""
    threshold_batch = ColocalizationCostesThresholdBatch()
    return [
        execute_request(
            func,
            replace(request, kwargs=threshold_batch.request_kwargs(request)),
        )
        for request in requests
    ]


measurement_image_batch_executor(measure_colocalization_objects_batch)(
    measure_colocalization_objects
)


def _prepare_measure_colocalization_objects() -> None:
    """Compile object-colocalization reduction kernels before measured execution."""
    first_pixels = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    second_pixels = np.linspace(1.0, 0.0, 16, dtype=np.float64)
    object_labels = np.repeat(np.arange(1, 5, dtype=np.int32), 4)
    object_count = 4
    reductions = object_colocalization_base_reductions(
        first_pixels,
        second_pixels,
        object_labels,
        object_count,
    )
    threshold_1 = 0.15 * reductions[6]
    threshold_2 = 0.15 * reductions[7]
    object_colocalization_threshold_reductions(
        first_pixels,
        second_pixels,
        object_labels,
        threshold_1,
        threshold_2,
        0.1,
        0.1,
        0.1,
        0.1,
        object_count,
    )


measure_colocalization_objects.__openhcs_prepare__ = (
    _prepare_measure_colocalization_objects
)


def _divide_measurements(numerator: object, denominator: object) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator_array / denominator_array
    result[~np.isfinite(result)] = 0
    return result


def _divide_costes_measurements(numerator: object, denominator: object) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator_array / denominator_array


__all__ = [
    "ColocalizationCostesBackendStrategy",
    "ColocalizationCostesThresholdBatch",
    "ColocalizationCostesThresholdRequest",
    "ColocalizationCostesThresholds",
    "ColocalizationImagePairContext",
    "ColocalizationMeasurementOptions",
    "ColocalizationMeasurementSchema",
    "ColocalizationMeasurements",
    "ColocalizationObjectLabelContext",
    "CostesMethod",
    "NumbaNumpyColocalizationCostesBackendStrategy",
    "ObjectColocalizationMeasurements",
    "UnitIntervalDenseRankSemantics",
    "costes_above_threshold_mask",
    "costes_backend",
    "measure_colocalization",
    "measure_colocalization_objects",
    "measure_colocalization_objects_batch",
    "object_colocalization_base_reductions",
    "object_colocalization_threshold_reductions",
    "thresholded_colocalization_metrics",
]
