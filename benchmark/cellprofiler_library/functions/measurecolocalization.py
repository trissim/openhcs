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
from openhcs.processing.materialization import csv_materializer
import scipy.ndimage
import scipy.stats
from scipy.linalg import lstsq


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


def _linear_costes(fi: np.ndarray, si: np.ndarray, scale_max: int = 255, fast_mode: bool = True) -> Tuple[float, float]:
    """Find Costes Automatic Threshold using linear algorithm."""
    i_step = 1 / scale_max
    non_zero = (fi > 0) | (si > 0)
    
    if not np.any(non_zero):
        return 0.0, 0.0
    
    xvar = np.var(fi[non_zero], axis=0, ddof=1)
    yvar = np.var(si[non_zero], axis=0, ddof=1)
    xmean = np.mean(fi[non_zero], axis=0)
    ymean = np.mean(si[non_zero], axis=0)
    
    z = fi[non_zero] + si[non_zero]
    zvar = np.var(z, axis=0, ddof=1)
    covar = 0.5 * (zvar - (xvar + yvar))
    
    denom = 2 * covar
    if denom == 0:
        return 0.0, 0.0
    
    num = (yvar - xvar) + np.sqrt((yvar - xvar) ** 2 + 4 * covar ** 2)
    a = num / denom
    b = ymean - a * xmean
    
    img_max = max(fi.max(), si.max())
    i = i_step * ((img_max // i_step) + 1)
    
    num_true = None
    fi_max = fi.max()
    si_max = si.max()
    
    thr_fi_c = i
    thr_si_c = (a * i) + b
    
    while i > fi_max and (a * i) + b > si_max:
        i -= i_step
    
    while i > i_step:
        thr_fi_c = i
        thr_si_c = (a * i) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        try:
            positives = np.count_nonzero(combt)
            if positives != num_true and positives > 2:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                num_true = positives
            else:
                costReg = 1.0
            
            if costReg <= 0:
                break
            elif not fast_mode or i < i_step * 10:
                i -= i_step
            elif costReg > 0.45:
                i -= i_step * 10
            elif costReg > 0.35:
                i -= i_step * 5
            elif costReg > 0.25:
                i -= i_step * 2
            else:
                i -= i_step
        except (ValueError, RuntimeWarning):
            break
    
    return thr_fi_c, thr_si_c


def _bisection_costes(fi: np.ndarray, si: np.ndarray, scale_max: int = 255) -> Tuple[float, float]:
    """Find Costes Automatic Threshold using bisection algorithm."""
    non_zero = (fi > 0) | (si > 0)
    
    if not np.any(non_zero):
        return 0.0, 0.0
    
    xvar = np.var(fi[non_zero], axis=0, ddof=1)
    yvar = np.var(si[non_zero], axis=0, ddof=1)
    xmean = np.mean(fi[non_zero], axis=0)
    ymean = np.mean(si[non_zero], axis=0)
    
    z = fi[non_zero] + si[non_zero]
    zvar = np.var(z, axis=0, ddof=1)
    covar = 0.5 * (zvar - (xvar + yvar))
    
    denom = 2 * covar
    if denom == 0:
        return 0.0, 0.0
    
    num = (yvar - xvar) + np.sqrt((yvar - xvar) ** 2 + 4 * covar ** 2)
    a = num / denom
    b = ymean - a * xmean
    
    left = 1
    right = scale_max
    mid = int(((right - left) // (6/5)) + left)
    lastmid = 0
    valid = 1
    
    while lastmid != mid:
        thr_fi_c = mid / scale_max
        thr_si_c = (a * thr_fi_c) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        
        if np.count_nonzero(combt) <= 2:
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                if costReg < 0:
                    left = mid - 1
                else:
                    right = mid + 1
                    valid = mid
            except (ValueError, RuntimeWarning):
                left = mid - 1
        
        lastmid = mid
        if right - left > 6:
            mid = int(((right - left) // (6 / 5)) + left)
        else:
            mid = int(((right - left) // 2) + left)
    
    thr_fi_c = (valid - 1) / scale_max
    thr_si_c = (a * thr_fi_c) + b
    
    return thr_fi_c, thr_si_c


def _colocalization_measurement(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    *,
    threshold_percent: float,
    do_correlation: bool,
    do_manders: bool,
    do_rwc: bool,
    do_overlap: bool,
    do_costes: bool,
    costes_method: CostesMethod,
    scale_max: int,
) -> ColocalizationMeasurements:
    mask = (~np.isnan(first_pixels)) & (~np.isnan(second_pixels))

    corr = np.nan
    slope = np.nan
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

        if do_correlation:
            corr = np.corrcoef(fi, si)[1, 0]
            coeffs = lstsq(
                np.array((fi, np.ones_like(fi))).T,
                si,
                lapack_driver="gelsy",
            )[0]
            slope = coeffs[0]

        if any((do_manders, do_rwc, do_overlap)):
            thr_fi = threshold_percent * np.max(fi) / 100
            thr_si = threshold_percent * np.max(si) / 100
            thr_fi_out = fi > thr_fi
            thr_si_out = si > thr_si
            combined_thresh = thr_fi_out & thr_si_out

            if np.any(combined_thresh):
                fi_thresh = fi[combined_thresh]
                si_thresh = si[combined_thresh]
                tot_fi_thr = fi[thr_fi_out].sum()
                tot_si_thr = si[thr_si_out].sum()

                if do_manders and tot_fi_thr > 0 and tot_si_thr > 0:
                    m1 = fi_thresh.sum() / tot_fi_thr
                    m2 = si_thresh.sum() / tot_si_thr

                if do_rwc and tot_fi_thr > 0 and tot_si_thr > 0:
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

                if do_overlap:
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

        if do_costes:
            if costes_method == CostesMethod.FASTER:
                thr_fi_c, thr_si_c = _bisection_costes(fi, si, scale_max)
            else:
                fast_mode = costes_method == CostesMethod.FAST
                thr_fi_c, thr_si_c = _linear_costes(
                    fi,
                    si,
                    scale_max,
                    fast_mode,
                )

            combined_thresh_c = (fi > thr_fi_c) & (si > thr_si_c)
            if np.any(combined_thresh_c):
                fi_thresh_c = fi[combined_thresh_c]
                si_thresh_c = si[combined_thresh_c]
                tot_fi_thr_c = fi[fi > thr_fi_c].sum()
                tot_si_thr_c = si[si > thr_si_c].sum()

                if tot_fi_thr_c > 0:
                    c1 = fi_thresh_c.sum() / tot_fi_thr_c
                if tot_si_thr_c > 0:
                    c2 = si_thresh_c.sum() / tot_si_thr_c

    return ColocalizationMeasurements(
        slice_index=0,
        correlation=float(corr) if not np.isnan(corr) else 0.0,
        slope=float(slope) if not np.isnan(slope) else 0.0,
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


@numpy
@special_outputs(("colocalization_measurements", csv_materializer(
    fields=["slice_index", "correlation", "slope", "overlap", "k1", "k2",
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
    scale_max: int = 255,
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
        scale_max: Maximum scale for Costes calculation (255 for 8-bit, 65535 for 16-bit)

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
    if channel_1 >= image.shape[0] or channel_2 >= image.shape[0]:
        raise ValueError(f"Channel indices ({channel_1}, {channel_2}) out of range for image with {image.shape[0]} channels")

    measurements = _colocalization_measurement(
        image[channel_1].astype(np.float64),
        image[channel_2].astype(np.float64),
        threshold_percent=threshold_percent,
        do_correlation=do_correlation,
        do_manders=do_manders,
        do_rwc=do_rwc,
        do_overlap=do_overlap,
        do_costes=do_costes,
        costes_method=costes_method,
        scale_max=scale_max,
    )
    
    # Return first selected channel as the output image
    return image[channel_1:channel_1+1], measurements


@numpy
@special_inputs("labels")
@special_outputs(("object_colocalization_measurements", csv_materializer(
    fields=[
        "slice_index",
        "object_label",
        "correlation",
        "slope",
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
    scale_max: int = 255,
) -> Tuple[np.ndarray, list[ObjectColocalizationMeasurements]]:
    """Measure colocalization between two channels within labeled objects."""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    if len(unique_labels) == 0:
        return image[channel_1:channel_1+1], []

    measurements: list[ObjectColocalizationMeasurements] = []
    image_float = image.astype(np.float64, copy=False)
    for object_label in unique_labels:
        mask = labels == object_label
        masked_image = image_float.copy()
        masked_image[:, ~mask] = np.nan
        object_measurement = _colocalization_measurement(
            masked_image[channel_1],
            masked_image[channel_2],
            threshold_percent=threshold_percent,
            do_correlation=do_correlation,
            do_manders=do_manders,
            do_rwc=do_rwc,
            do_overlap=do_overlap,
            do_costes=do_costes,
            costes_method=costes_method,
            scale_max=scale_max,
        )
        measurements.append(
            ObjectColocalizationMeasurements(
                slice_index=object_measurement.slice_index,
                object_label=int(object_label),
                correlation=object_measurement.correlation,
                slope=object_measurement.slope,
                overlap=object_measurement.overlap,
                k1=object_measurement.k1,
                k2=object_measurement.k2,
                manders_m1=object_measurement.manders_m1,
                manders_m2=object_measurement.manders_m2,
                rwc1=object_measurement.rwc1,
                rwc2=object_measurement.rwc2,
                costes_m1=object_measurement.costes_m1,
                costes_m2=object_measurement.costes_m2,
                costes_threshold_1=object_measurement.costes_threshold_1,
                costes_threshold_2=object_measurement.costes_threshold_2,
            )
        )

    return image[channel_1:channel_1+1], measurements
