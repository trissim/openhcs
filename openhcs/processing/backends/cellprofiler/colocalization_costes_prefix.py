"""Low-level Numba Costes prefix/Pearson kernels."""

from __future__ import annotations

import numpy as np
from openhcs.processing.backends.cellprofiler.colocalization_costes_prefix_numba_pearson import (
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


def quantized_unit_interval_event_summaries(
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
