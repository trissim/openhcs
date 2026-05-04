"""
Converted from CellProfiler: MeasureColocalization
Original: MeasureColocalization

Measures colocalization and correlation between intensities in different images
(e.g., different color channels) on a pixel-by-pixel basis.
"""

import numpy as np
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
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
)
from openhcs.processing.backends.cellprofiler.colocalization import (
    ColocalizationCostesBackendStrategy,
)
from openhcs.processing.materialization import csv_materializer
import scipy.ndimage


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
    mask = (~np.isnan(first_pixels)) & (~np.isnan(second_pixels))
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)

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

    if np.any(mask):
        fi = first_pixels[mask]
        si = second_pixels[mask]

        if options.do_correlation:
            corr, slope, slope_reverse = (
                ColocalizationCostesBackendStrategy.for_memory_type(
                    backend_provider=options.costes_backend_provider,
                ).correlation_slopes(fi, si)
            )

        if any((options.do_manders, options.do_rwc, options.do_overlap)):
            thr_fi = options.threshold_percent * np.max(fi) / 100
            thr_si = options.threshold_percent * np.max(si) / 100
            thr_fi_out = fi > thr_fi
            thr_si_out = si > thr_si
            combined_thresh = thr_fi_out & thr_si_out

            if np.any(combined_thresh):
                fi_thresh = fi[combined_thresh]
                si_thresh = si[combined_thresh]
                tot_fi_thr = fi[thr_fi_out].sum()
                tot_si_thr = si[thr_si_out].sum()

                if options.do_manders and tot_fi_thr > 0 and tot_si_thr > 0:
                    m1 = fi_thresh.sum() / tot_fi_thr
                    m2 = si_thresh.sum() / tot_si_thr

                if options.do_rwc and tot_fi_thr > 0 and tot_si_thr > 0:
                    rank1 = np.lexsort([fi])
                    rank2 = np.lexsort([si])
                    rank1_u = np.hstack(
                        [[False], fi[rank1[:-1]] != fi[rank1[1:]]]
                    )
                    rank2_u = np.hstack(
                        [[False], si[rank2[:-1]] != si[rank2[1:]]]
                    )
                    rank1_s = np.cumsum(rank1_u)
                    rank2_s = np.cumsum(rank2_u)
                    rank_im1 = np.zeros(fi.shape, dtype=int)
                    rank_im2 = np.zeros(si.shape, dtype=int)
                    rank_im1[rank1] = rank1_s
                    rank_im2[rank2] = rank2_s

                    r = max(rank_im1.max(), rank_im2.max()) + 1
                    di = np.abs(rank_im1 - rank_im2)
                    weight = (r - di) / r
                    weight_thresh = weight[combined_thresh]
                    rwc1 = (fi_thresh * weight_thresh).sum() / tot_fi_thr
                    rwc2 = (si_thresh * weight_thresh).sum() / tot_si_thr

                if options.do_overlap:
                    denom = np.sqrt(
                        (fi_thresh ** 2).sum() * (si_thresh ** 2).sum()
                    )
                    if denom > 0:
                        overlap = (fi_thresh * si_thresh).sum() / denom
                    fi_sq_sum = (fi_thresh ** 2).sum()
                    si_sq_sum = (si_thresh ** 2).sum()
                    if fi_sq_sum > 0:
                        k1 = (fi_thresh * si_thresh).sum() / fi_sq_sum
                    if si_sq_sum > 0:
                        k2 = (fi_thresh * si_thresh).sum() / si_sq_sum

        if options.do_costes:
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

            first_above_costes = _costes_above_threshold(fi, thr_fi_c)
            second_above_costes = _costes_above_threshold(si, thr_si_c)
            combined_thresh_c = first_above_costes & second_above_costes
            if np.any(combined_thresh_c):
                fi_thresh_c = fi[combined_thresh_c]
                si_thresh_c = si[combined_thresh_c]
                tot_fi_thr_c = fi[first_above_costes].sum()
                tot_si_thr_c = si[second_above_costes].sum()

                if tot_fi_thr_c > 0:
                    c1 = fi_thresh_c.sum() / tot_fi_thr_c
                if tot_si_thr_c > 0:
                    c2 = si_thresh_c.sum() / tot_si_thr_c

    return ColocalizationMeasurements(
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


def _cellprofiler_float_pixels(image: np.ndarray) -> np.ndarray:
    """Return image pixels in CellProfiler's native float image domain."""
    return np.asarray(image_payload_data(image), dtype=np.float32)


def _channel_pair_valid_mask(
    image: object,
    image_data: np.ndarray,
    channel_1: int,
    channel_2: int,
) -> np.ndarray:
    """Return CellProfiler-style valid pixels for a two-image measurement."""
    first_pixels = image_data[channel_1]
    second_pixels = image_data[channel_2]
    valid = np.isfinite(first_pixels) & np.isfinite(second_pixels)
    mask = image_payload_mask(image)
    if mask is None:
        return valid

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
    # Select the two channels to compare
    image_data = image_payload_data(image)
    if channel_1 >= image_data.shape[0] or channel_2 >= image_data.shape[0]:
        raise ValueError(f"Channel indices ({channel_1}, {channel_2}) out of range for image with {image_data.shape[0]} channels")

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
    image_float = _cellprofiler_float_pixels(image_data)
    measurements = _colocalization_measurement(
        image_float[channel_1],
        image_float[channel_2],
        options=options,
        valid_mask=_channel_pair_valid_mask(
            image,
            image_float,
            channel_1,
            channel_2,
        ),
    )
    
    # Return first selected channel as the output image
    return _channel_output_payload(image, image_data, channel_1), measurements


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
    object_mask = (
        (labels > 0)
        & pair_valid_mask
    )
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
    full_mask = pair_valid_mask
    full_fi = first_image[full_mask]
    full_si = second_image[full_mask]
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
