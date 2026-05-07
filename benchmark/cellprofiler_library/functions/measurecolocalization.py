"""
Converted from CellProfiler: MeasureColocalization
Original: MeasureColocalization

Measures colocalization and correlation between intensities in different images
(e.g., different color channels) on a pixel-by-pixel basis.
"""

import numpy as np
import logging
import os
import time
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.core.runtime_values import (
    image_intensity_scale_for_dtype,
    image_payload_data,
    image_payload_metadata,
    image_payload_mask,
    image_payload_with_context,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
)
from openhcs.processing.backends.cellprofiler.colocalization import (
    ColocalizationCostesBackendStrategy,
)
from openhcs.processing.materialization import csv_materializer
import scipy.ndimage
from numba import njit

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


class CostesMethod(Enum):
    FASTER = "faster"
    FAST = "fast"
    ACCURATE = "accurate"


@dataclass
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


@dataclass
class ObjectColocalizationMeasurements:
    """Colocalization measurements scoped to one labeled object."""

    slice_index: int
    object_label: int
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

    @classmethod
    def from_measurement(
        cls,
        *,
        object_label: int,
        measurement: ColocalizationMeasurements,
    ) -> "ObjectColocalizationMeasurements":
        return cls(
            slice_index=measurement.slice_index,
            object_label=object_label,
            correlation=measurement.correlation,
            slope=measurement.slope,
            slope_reverse=measurement.slope_reverse,
            overlap=measurement.overlap,
            k1=measurement.k1,
            k2=measurement.k2,
            manders_m1=measurement.manders_m1,
            manders_m2=measurement.manders_m2,
            rwc1=measurement.rwc1,
            rwc2=measurement.rwc2,
            costes_m1=measurement.costes_m1,
            costes_m2=measurement.costes_m2,
            costes_threshold_1=measurement.costes_threshold_1,
            costes_threshold_2=measurement.costes_threshold_2,
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
    costes_backend_provider: BackendProviderInput | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "costes_method", CostesMethod(self.costes_method))


@dataclass(frozen=True)
class CostesRegressionLine:
    """Regression line used by Costes automatic threshold search."""

    slope: float
    intercept: float

    def second_threshold(self, first_threshold: float) -> float:
        return (self.slope * first_threshold) + self.intercept


def _costes_regression_line(
    fi: np.ndarray,
    si: np.ndarray,
) -> CostesRegressionLine | None:
    non_zero = (fi > 0) | (si > 0)
    if np.count_nonzero(non_zero) <= 1:
        return None

    first_values = fi[non_zero]
    second_values = si[non_zero]
    xvar = np.var(first_values, axis=0, ddof=1)
    yvar = np.var(second_values, axis=0, ddof=1)
    xmean = np.mean(first_values, axis=0)
    ymean = np.mean(second_values, axis=0)

    z = first_values + second_values
    zvar = np.var(z, axis=0, ddof=1)
    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    if denom == 0:
        return None

    num = (yvar - xvar) + np.sqrt((yvar - xvar) ** 2 + 4 * covar**2)
    slope = num / denom
    intercept = ymean - slope * xmean
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return CostesRegressionLine(float(slope), float(intercept))


def _costes_intensity_step(
    fi: np.ndarray,
    si: np.ndarray,
    scale_max: int,
) -> float:
    if scale_max <= 0:
        raise ValueError("scale_max must be positive.")
    image_max = float(max(np.max(fi), np.max(si)))
    if image_max <= 1.0:
        return 1.0 / scale_max
    return max(1.0, image_max / scale_max)


def _costes_minimum_scale_index(scale_max: int) -> int:
    """Lowest Costes bin tested by CellProfiler's scaled threshold search."""
    if scale_max <= 0:
        raise ValueError("scale_max must be positive.")
    return min(scale_max, 5)


def _costes_scale_threshold(scale_index: int, scale_max: int) -> float:
    if scale_max <= 0:
        raise ValueError("scale_max must be positive.")
    return scale_index / scale_max


def _costes_above_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return values >= threshold
    return values > threshold


def _pearson_correlation_or_nan(first: np.ndarray, second: np.ndarray) -> float:
    if first.size <= 1 or second.size <= 1:
        return np.nan
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    first_centered = first_values - np.mean(first_values)
    second_centered = second_values - np.mean(second_values)
    denominator = np.sqrt(
        np.sum(first_centered * first_centered)
        * np.sum(second_centered * second_centered)
    )
    if denominator == 0:
        return np.nan
    return float(np.sum(first_centered * second_centered) / denominator)


def _initial_costes_scale_index(
    fi: np.ndarray,
    si: np.ndarray,
    regression_line: CostesRegressionLine,
    intensity_step: float,
    scale_max: int,
) -> int:
    fi_max = float(np.max(fi))
    si_max = float(np.max(si))
    image_max = max(fi_max, si_max)
    scale_index = min(scale_max, max(1, int(np.ceil(image_max / intensity_step))))

    while scale_index > 1:
        first_threshold = scale_index * intensity_step
        second_threshold = regression_line.second_threshold(first_threshold)
        if first_threshold <= fi_max or second_threshold <= si_max:
            break
        scale_index -= 1

    return scale_index


def _linear_costes(
    fi: np.ndarray,
    si: np.ndarray,
    scale_max: int = 255,
    fast_mode: bool = True,
    *,
    backend_provider: BackendProviderInput | None = None,
) -> Tuple[float, float]:
    """Find Costes Automatic Threshold using CellProfiler's linear algorithm."""
    return ColocalizationCostesBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    ).linear_costes(
        fi,
        si,
        scale_max,
        fast_mode,
    )


def _linear_costes_numpy_reference(fi: np.ndarray, si: np.ndarray, scale_max: int = 255, fast_mode: bool = True) -> Tuple[float, float]:
    """Reference Python implementation used to validate backend semantics."""
    regression_line = _costes_regression_line(fi, si)
    if regression_line is None:
        return 0.0, 0.0

    intensity_step = 1 / scale_max
    threshold = intensity_step * ((max(fi.max(), si.max()) // intensity_step) + 1)
    num_true = None
    thr_fi_c = threshold
    thr_si_c = regression_line.second_threshold(thr_fi_c)

    while threshold > fi.max() and regression_line.second_threshold(threshold) > si.max():
        threshold -= intensity_step
    while threshold > intensity_step:
        thr_fi_c = threshold
        thr_si_c = regression_line.second_threshold(thr_fi_c)
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        positives = np.count_nonzero(combt)
        if positives != num_true:
            costReg = _pearson_correlation_or_nan(fi[combt], si[combt])
            num_true = positives

        if not np.isfinite(costReg):
            break
        if costReg <= 0:
            break
        elif not fast_mode or threshold < intensity_step * 10:
            threshold -= intensity_step
        elif costReg > 0.45:
            threshold -= intensity_step * 10
        elif costReg > 0.35:
            threshold -= intensity_step * 5
        elif costReg > 0.25:
            threshold -= intensity_step * 2
        else:
            threshold -= intensity_step

    return thr_fi_c, thr_si_c


def _threshold_for_second_costes_bin(
    regression_line: CostesRegressionLine,
    second_scale_index: int,
    scale_max: int,
) -> float:
    second_threshold = _costes_scale_threshold(second_scale_index, scale_max)
    if regression_line.slope == 0:
        return 0.0
    return max(0.0, (second_threshold - regression_line.intercept) / regression_line.slope)


def _scaled_second_channel_costes(
    fi: np.ndarray,
    si: np.ndarray,
    scale_max: int,
    *,
    backend_provider: BackendProviderInput | None = None,
) -> Tuple[float, float]:
    """Search Costes thresholds over scaled intensity bins.

    CellProfiler's fast image-level Costes behavior is quantized by the image
    intensity scale, and the second channel uses a low-bin floor when the fitted
    first-channel threshold clips to zero.
    """
    return ColocalizationCostesBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    ).scaled_second_channel_costes(
        fi,
        si,
        scale_max,
    )


def _scaled_second_channel_costes_numpy_reference(
    fi: np.ndarray,
    si: np.ndarray,
    scale_max: int,
) -> Tuple[float, float]:
    """Reference Python implementation used to validate backend semantics."""
    regression_line = _costes_regression_line(fi, si)
    if regression_line is None:
        return 0.0, 0.0

    minimum_scale_index = _costes_minimum_scale_index(scale_max)
    selected_first_threshold = 0.0
    selected_second_threshold = _costes_scale_threshold(
        minimum_scale_index,
        scale_max,
    )
    selected_correlation = np.nan

    for second_scale_index in range(scale_max, minimum_scale_index - 1, -1):
        second_threshold = _costes_scale_threshold(second_scale_index, scale_max)
        first_threshold = _threshold_for_second_costes_bin(
            regression_line,
            second_scale_index,
            scale_max,
        )
        below_threshold = (fi < first_threshold) | (si < second_threshold)
        cost_regression = _pearson_correlation_or_nan(
            fi[below_threshold],
            si[below_threshold],
        )
        selected_first_threshold = first_threshold
        selected_second_threshold = second_threshold
        selected_correlation = cost_regression
        if np.isfinite(cost_regression) and cost_regression <= 0:
            break

    if (
        not np.isfinite(selected_correlation)
        or selected_correlation > 0
        or selected_first_threshold <= 0
    ):
        selected_first_threshold = 0.0

    return selected_first_threshold, selected_second_threshold


def _bisection_costes(
    fi: np.ndarray,
    si: np.ndarray,
    scale_max: int = 255,
    *,
    backend_provider: BackendProviderInput | None = None,
) -> Tuple[float, float]:
    """Find Costes Automatic Threshold using CellProfiler's bisection algorithm."""
    return _scaled_second_channel_costes(
        fi,
        si,
        scale_max,
        backend_provider=backend_provider,
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
            ) = _thresholded_colocalization_metrics(
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
                thr_fi_c, thr_si_c = _bisection_costes(
                    fi,
                    si,
                    options.scale_max,
                    backend_provider=options.costes_backend_provider,
                )
            else:
                fast_mode = options.costes_method == CostesMethod.FAST
                thr_fi_c, thr_si_c = _linear_costes(
                    fi,
                    si,
                    options.scale_max,
                    fast_mode,
                    backend_provider=options.costes_backend_provider,
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


def _thresholded_colocalization_metrics(
    first: np.ndarray,
    second: np.ndarray,
    threshold_percent: float,
    do_manders: bool,
    do_rwc: bool,
    do_overlap: bool,
    preferred_scale: int | None = None,
    proven_unit_interval_scale: int | None = None,
) -> tuple[float, float, float, float, float, float, float]:
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

    first_ranks = _dense_ranks(
        first,
        preferred_scale=preferred_scale,
        proven_unit_interval_scale=proven_unit_interval_scale,
    )
    second_ranks = _dense_ranks(
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


def _dense_ranks(
    values: np.ndarray,
    *,
    preferred_scale: int | None = None,
    proven_unit_interval_scale: int | None = None,
) -> np.ndarray:
    if proven_unit_interval_scale is not None:
        return _dense_ranks_for_proven_unit_interval(
            values,
            int(proven_unit_interval_scale),
        )
    quantized_ranks = _dense_ranks_for_integer_unit_interval(
        values,
        preferred_scale=preferred_scale,
    )
    if quantized_ranks is not None:
        return quantized_ranks
    return np.ascontiguousarray(
        np.unique(values, return_inverse=True)[1],
        dtype=np.int64,
    )


def _dense_ranks_for_proven_unit_interval(
    values: np.ndarray,
    scale: int,
) -> np.ndarray:
    """Return dense ranks when metadata proves values are exact code / scale."""
    codes = np.rint(np.asarray(values) * int(scale)).astype(np.int64, copy=False)
    present = np.bincount(codes.ravel(), minlength=int(scale) + 1) > 0
    lookup = np.cumsum(present, dtype=np.int64) - 1
    return np.ascontiguousarray(lookup[codes], dtype=np.int64)


def _dense_ranks_for_integer_unit_interval(
    values: np.ndarray,
    *,
    preferred_scale: int | None = None,
) -> np.ndarray | None:
    values_array = np.asarray(values)
    if values_array.size == 0 or values_array.dtype.kind not in {"f", "u", "i"}:
        return None
    minimum = np.min(values_array)
    maximum = np.max(values_array)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return None
    if minimum < 0 or maximum > 1:
        return None
    scales = (
        (preferred_scale, 65535, 255)
        if preferred_scale is not None
        else (65535, 255)
    )
    for scale in dict.fromkeys(int(candidate) for candidate in scales if candidate):
        codes = np.rint(values_array * scale).astype(np.int64, copy=False)
        if np.any(codes < 0) or np.any(codes > scale):
            continue
        reconstructed = (
            codes.astype(values_array.dtype, copy=False)
            / values_array.dtype.type(scale)
        )
        if not np.array_equal(values_array, reconstructed):
            continue
        present = np.bincount(codes.ravel(), minlength=scale + 1) > 0
        lookup = np.cumsum(present, dtype=np.int64) - 1
        return np.ascontiguousarray(lookup[codes], dtype=np.int64)
    return None


def _thresholded_colocalization_metrics_numba(
    first: np.ndarray,
    second: np.ndarray,
    threshold_percent: float,
    do_manders: bool,
    do_rwc: bool,
    do_overlap: bool,
    preferred_scale: int | None = None,
    proven_unit_interval_scale: int | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    """Compatibility entrypoint for the Numba-backed metric reducer."""
    return _thresholded_colocalization_metrics(
        first,
        second,
        threshold_percent,
        do_manders,
        do_rwc,
        do_overlap,
        preferred_scale,
        proven_unit_interval_scale,
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


@njit(cache=True)
def _dense_rank_values_numba(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.int64)
    current_rank = 0
    if values.size == 0:
        return ranks
    ranks[order[0]] = current_rank
    previous = values[order[0]]
    for sorted_index in range(1, values.size):
        value = values[order[sorted_index]]
        if value != previous:
            current_rank += 1
            previous = value
        ranks[order[sorted_index]] = current_rank
    return ranks


def _cellprofiler_float_pixels(image: np.ndarray) -> np.ndarray:
    """Return image pixels in CellProfiler's native float image domain."""
    return np.asarray(image_payload_data(image), dtype=np.float32)


def _channel_pair_valid_mask(
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


def _channel_output_payload(
    image: object,
    image_data: np.ndarray,
    channel_index: int,
) -> object:
    """Return one channel while preserving compatible image-mask semantics."""
    output = image_data[channel_index : channel_index + 1]
    mask = image_payload_mask(image)
    metadata = image_payload_metadata(image).for_channel(channel_index)
    if mask is None:
        return image_payload_with_context(output, metadata=metadata)

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape == image_data.shape:
        mask_array = mask_array[channel_index : channel_index + 1]
    return image_payload_with_context(output, mask=mask_array, metadata=metadata)


def _costes_scale_max(
    image: object,
    image_data: np.ndarray,
    channel_1: int,
    channel_2: int,
    explicit_scale_max: int | None,
) -> int:
    """Resolve Costes scale from generic image metadata, with dtype fallback."""
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
    costes_backend_provider: BackendProviderInput | None = None,
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
        scale_max=_costes_scale_max(
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
    valid_mask = _channel_pair_valid_mask(
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
    output = _channel_output_payload(image, image_data, channel_1)
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
    costes_backend_provider: BackendProviderInput | None = None,
) -> Tuple[np.ndarray, list[ObjectColocalizationMeasurements]]:
    """Measure colocalization between two channels within labeled objects."""
    image_data = image_payload_data(image)
    labels = object_label_dense_array(labels, dtype=np.int32)
    max_label = int(np.max(labels)) if labels.size else 0
    if max_label <= 0:
        return _channel_output_payload(image, image_data, channel_1), []

    label_range = np.arange(1, max_label + 1, dtype=np.int32)
    image_float = _cellprofiler_float_pixels(image_data)
    options = ColocalizationMeasurementOptions(
        threshold_percent=threshold_percent,
        do_correlation=do_correlation,
        do_manders=do_manders,
        do_rwc=do_rwc,
        do_overlap=do_overlap,
        do_costes=do_costes,
        costes_method=costes_method,
        scale_max=_costes_scale_max(
            image,
            image_data,
            channel_1,
            channel_2,
            scale_max,
        ),
        costes_backend_provider=costes_backend_provider,
    )

    first_image = image_float[channel_1]
    second_image = image_float[channel_2]
    pair_valid_mask = _channel_pair_valid_mask(
        image,
        image_float,
        channel_1,
        channel_2,
    )
    object_mask = labels > 0
    if pair_valid_mask is not None:
        object_mask = object_mask & pair_valid_mask
    if not np.any(object_mask):
        return (
            _channel_output_payload(image, image_data, channel_1),
            [
                _object_colocalization_row(int(object_label))
                for object_label in label_range
            ],
        )

    first_pixels = first_image[object_mask]
    second_pixels = second_image[object_mask]
    object_labels = labels[object_mask].astype(np.int32, copy=False)
    if pair_valid_mask is None:
        full_fi = first_image.ravel()
        full_si = second_image.ravel()
    else:
        full_fi = first_image[pair_valid_mask]
        full_si = second_image[pair_valid_mask]
    object_counts = scipy.ndimage.sum(
        np.ones(len(first_pixels)),
        object_labels,
        label_range,
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
        mean1 = scipy.ndimage.mean(first_pixels, object_labels, label_range)
        mean2 = scipy.ndimage.mean(second_pixels, object_labels, label_range)
        std1 = np.sqrt(
            scipy.ndimage.sum(
                (first_pixels - mean1[object_labels - 1]) ** 2,
                object_labels,
                label_range,
            )
        )
        std2 = np.sqrt(
            scipy.ndimage.sum(
                (second_pixels - mean2[object_labels - 1]) ** 2,
                object_labels,
                label_range,
            )
        )
        denominator = std1[object_labels - 1] * std2[object_labels - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_pixel_corr = (
                (first_pixels - mean1[object_labels - 1])
                * (second_pixels - mean2[object_labels - 1])
                / denominator
            )
        corr = np.asarray(
            scipy.ndimage.sum(per_pixel_corr, object_labels, label_range),
            dtype=float,
        )
        corr[np.asarray(object_counts) == 0] = np.nan

    if any((options.do_manders, options.do_rwc, options.do_overlap)):
        threshold_1 = options.threshold_percent / 100 * scipy.ndimage.maximum(
            first_pixels,
            object_labels,
            label_range,
        )
        threshold_2 = options.threshold_percent / 100 * scipy.ndimage.maximum(
            second_pixels,
            object_labels,
            label_range,
        )
        first_above_threshold = first_pixels >= threshold_1[object_labels - 1]
        second_above_threshold = second_pixels >= threshold_2[object_labels - 1]
        combined_threshold = first_above_threshold & second_above_threshold
        fi_thresh = first_pixels[combined_threshold]
        si_thresh = second_pixels[combined_threshold]
        labels_thresh = object_labels[combined_threshold]
        total_first_threshold = np.asarray(
            scipy.ndimage.sum(
                first_pixels[first_above_threshold],
                object_labels[first_above_threshold],
                label_range,
            ),
            dtype=float,
        )
        total_second_threshold = np.asarray(
            scipy.ndimage.sum(
                second_pixels[second_above_threshold],
                object_labels[second_above_threshold],
                label_range,
            ),
            dtype=float,
        )

    if options.do_manders and np.any(combined_threshold):
        manders_m1 = _divide_measurements(
            scipy.ndimage.sum(fi_thresh, labels_thresh, label_range),
            total_first_threshold,
        )
        manders_m2 = _divide_measurements(
            scipy.ndimage.sum(si_thresh, labels_thresh, label_range),
            total_second_threshold,
        )

    if options.do_rwc:
        rank1 = np.lexsort((object_labels, first_pixels))
        rank2 = np.lexsort((object_labels, second_pixels))
        rank1_unique = np.hstack(
            [[False], first_pixels[rank1[:-1]] != first_pixels[rank1[1:]]]
        )
        rank2_unique = np.hstack(
            [[False], second_pixels[rank2[:-1]] != second_pixels[rank2[1:]]]
        )
        rank1_serial = np.cumsum(rank1_unique)
        rank2_serial = np.cumsum(rank2_unique)
        rank_image_1 = np.zeros(first_pixels.shape, dtype=int)
        rank_image_2 = np.zeros(second_pixels.shape, dtype=int)
        rank_image_1[rank1] = rank1_serial
        rank_image_2[rank2] = rank2_serial

        max_rank = max(rank_image_1.max(), rank_image_2.max()) + 1
        rank_delta = abs(rank_image_1 - rank_image_2)
        weight = (max_rank - rank_delta) * 1.0 / max_rank
        weight_threshold = weight[combined_threshold]
        if np.any(combined_threshold):
            rwc1 = _divide_measurements(
                scipy.ndimage.sum(
                    fi_thresh * weight_threshold,
                    labels_thresh,
                    label_range,
                ),
                total_first_threshold,
            )
            rwc2 = _divide_measurements(
                scipy.ndimage.sum(
                    si_thresh * weight_threshold,
                    labels_thresh,
                    label_range,
                ),
                total_second_threshold,
            )

    if options.do_overlap:
        if np.any(combined_threshold):
            first_sq = np.asarray(
                scipy.ndimage.sum(
                    first_pixels[combined_threshold] ** 2,
                    labels_thresh,
                    label_range,
                ),
                dtype=float,
            )
            second_sq = np.asarray(
                scipy.ndimage.sum(
                    second_pixels[combined_threshold] ** 2,
                    labels_thresh,
                    label_range,
                ),
                dtype=float,
            )
            product_sum = np.asarray(
                scipy.ndimage.sum(
                    fi_thresh * si_thresh,
                    labels_thresh,
                    label_range,
                ),
                dtype=float,
            )
            overlap = _divide_measurements(
                product_sum,
                np.sqrt(first_sq * second_sq),
            )
            k1 = _divide_measurements(product_sum, first_sq)
            k2 = _divide_measurements(product_sum, second_sq)

    if options.do_costes and full_fi.size:
        if options.costes_method == CostesMethod.FASTER:
            threshold_c1, threshold_c2 = _bisection_costes(
                full_fi,
                full_si,
                options.scale_max,
                backend_provider=options.costes_backend_provider,
            )
        else:
            threshold_c1, threshold_c2 = _linear_costes(
                full_fi,
                full_si,
                options.scale_max,
                options.costes_method == CostesMethod.FAST,
                backend_provider=options.costes_backend_provider,
            )
        costes_threshold_1.fill(threshold_c1)
        costes_threshold_2.fill(threshold_c2)
        first_above_costes = _costes_above_threshold(first_pixels, threshold_c1)
        second_above_costes = _costes_above_threshold(second_pixels, threshold_c2)
        combined_costes = first_above_costes & second_above_costes
        first_costes_denominator_threshold = _costes_first_channel_bin_threshold(
            threshold_c1,
            options.scale_max,
        )
        total_first_costes = (
            np.asarray(
                scipy.ndimage.sum(
                    first_pixels[first_pixels >= first_costes_denominator_threshold],
                    object_labels[first_pixels >= first_costes_denominator_threshold],
                    label_range,
                ),
                dtype=float,
            )
            if np.any(first_above_costes)
            else np.zeros(max_label, dtype=float)
        )
        total_second_costes = (
            np.asarray(
                scipy.ndimage.sum(
                    second_pixels[second_pixels >= threshold_c2],
                    object_labels[second_pixels >= threshold_c2],
                    label_range,
                ),
                dtype=float,
            )
            if np.any(second_above_costes)
            else np.zeros(max_label, dtype=float)
        )
        if np.any(combined_costes):
            costes_m1 = _divide_costes_measurements(
                scipy.ndimage.sum(
                    first_pixels[combined_costes],
                    object_labels[combined_costes],
                    label_range,
                ),
                total_first_costes,
            )
            costes_m2 = _divide_costes_measurements(
                scipy.ndimage.sum(
                    second_pixels[combined_costes],
                    object_labels[combined_costes],
                    label_range,
                ),
                total_second_costes,
            )

    return (
        _channel_output_payload(image, image_data, channel_1),
        [
            _object_colocalization_row(
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


def _costes_first_channel_bin_threshold(threshold: float, scale_max: int) -> float:
    if scale_max <= 0 or not np.isfinite(threshold):
        return float(threshold)
    scaled_threshold = float(threshold) * scale_max
    nearest_bin = round(scaled_threshold)
    if np.isclose(scaled_threshold, nearest_bin, rtol=0.0, atol=1e-3):
        return nearest_bin / scale_max
    return float(threshold)


def _object_colocalization_row(
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
) -> ObjectColocalizationMeasurements:
    return ObjectColocalizationMeasurements(
        slice_index=0,
        object_label=object_label,
        correlation=float(correlation) if np.isfinite(correlation) else 0.0,
        slope=float(slope) if np.isfinite(slope) else 0.0,
        slope_reverse=(
            float(slope_reverse) if np.isfinite(slope_reverse) else 0.0
        ),
        overlap=float(overlap) if np.isfinite(overlap) else 0.0,
        k1=float(k1) if np.isfinite(k1) else 0.0,
        k2=float(k2) if np.isfinite(k2) else 0.0,
        manders_m1=float(manders_m1) if np.isfinite(manders_m1) else 0.0,
        manders_m2=float(manders_m2) if np.isfinite(manders_m2) else 0.0,
        rwc1=float(rwc1) if np.isfinite(rwc1) else 0.0,
        rwc2=float(rwc2) if np.isfinite(rwc2) else 0.0,
        costes_m1=float(costes_m1),
        costes_m2=float(costes_m2),
        costes_threshold_1=(
            float(costes_threshold_1) if np.isfinite(costes_threshold_1) else 0.0
        ),
        costes_threshold_2=(
            float(costes_threshold_2) if np.isfinite(costes_threshold_2) else 0.0
        ),
    )


def _prepare_measure_colocalization() -> None:
    """Compile colocalization kernels outside measured execution."""
    first = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    second = np.flipud(first).copy()
    _colocalization_measurement(
        first,
        second,
        options=ColocalizationMeasurementOptions(
            threshold_percent=15.0,
            do_correlation=True,
            do_manders=True,
            do_rwc=True,
            do_overlap=True,
            do_costes=True,
            costes_method=CostesMethod.ACCURATE,
            scale_max=255,
        ),
    )
    quantized_codes = (np.arange(64 * 64, dtype=np.uint16) % 512) + 1024
    quantized_first = (quantized_codes.astype(np.float32) / np.float32(65535)).reshape(
        (64, 64)
    )
    quantized_second = np.flipud(quantized_first).copy()
    _colocalization_measurement(
        quantized_first,
        quantized_second,
        options=ColocalizationMeasurementOptions(
            threshold_percent=15.0,
            do_correlation=True,
            do_manders=True,
            do_rwc=True,
            do_overlap=True,
            do_costes=True,
            costes_method=CostesMethod.ACCURATE,
            scale_max=255,
        ),
    )


measure_colocalization.__openhcs_prepare__ = _prepare_measure_colocalization
