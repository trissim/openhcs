"""
Converted from CellProfiler: CalculateStatistics
Original: CalculateStatistics module

Calculates measures of assay quality (V and Z' factors) and dose-response
data (EC50) for all measured features. This is an experiment-level analysis
that operates on aggregated measurements across all images.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import scipy.optimize
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


@dataclass
class StatisticsResult:
    """Results from calculate_statistics analysis."""
    feature_name: str
    object_name: str
    z_factor: float
    z_factor_one_tailed: float
    v_factor: float
    ec50: float


def _loc_vector_labels(x: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """Identify unique labels from the vector of image labels.
    
    Args:
        x: A vector of one label or dose per image
        
    Returns:
        labels: Ordinal per image indexing into unique labels
        labnum: Number of unique labels
        uniqsortvals: Vector of unique labels
    """
    order = np.lexsort((x,))
    reverse_order = np.lexsort((order,))
    sorted_x = x[order]
    
    first_occurrence = np.ones(len(x), bool)
    first_occurrence[1:] = sorted_x[:-1] != sorted_x[1:]
    sorted_labels = np.cumsum(first_occurrence) - 1
    labels = sorted_labels[reverse_order]
    uniqsortvals = sorted_x[first_occurrence]
    return labels, len(uniqsortvals), uniqsortvals


def _loc_shrink_mean_std(xcol: np.ndarray, ymatr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and standard deviation per label.
    
    Args:
        xcol: Column of image labels or doses
        ymatr: Matrix with rows of values per image, columns for measurements
        
    Returns:
        xs: Vector of unique doses
        avers: Average value per label
        stds: Standard deviation per label
    """
    ncols = ymatr.shape[1]
    labels, labnum, xs = _loc_vector_labels(xcol)
    avers = np.zeros((labnum, ncols))
    stds = avers.copy()
    
    for ilab in range(labnum):
        labinds = labels == ilab
        labmatr = ymatr[labinds, :]
        if labmatr.shape[0] == 1:
            avers[ilab, :] = labmatr[0, :]
        else:
            avers[ilab, :] = np.mean(labmatr, 0)
            stds[ilab, :] = np.std(labmatr, 0)
    return xs, avers, stds


def _z_factors(xcol: np.ndarray, ymatr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Z' factors for assay quality.
    
    Args:
        xcol: Grouping values (positive/negative control designations)
        ymatr: Matrix of measurements (observations x measures)
        
    Returns:
        z: Z' factors
        z_one_tailed: One-tailed Z' factors
        xs: Ordered unique doses
        avers: Ordered average values
    """
    xs, avers, stds = _loc_shrink_mean_std(xcol, ymatr)
    
    # Z' factor from positive and negative controls (extremes by dose)
    zrange = np.abs(avers[0, :] - avers[-1, :])
    zstd = stds[0, :] + stds[-1, :]
    zstd[zrange == 0] = 1
    zrange[zrange == 0] = 0.000001
    z = 1 - 3 * (zstd / zrange)
    
    # One-tailed Z' factor using only samples between means
    zrange = np.abs(avers[0, :] - avers[-1, :])
    exp1_vals = ymatr[xcol == xs[0], :]
    exp2_vals = ymatr[xcol == xs[-1], :]
    sort_avers = np.sort(np.array((avers[0, :], avers[-1, :])), 0)
    
    for i in range(sort_avers.shape[1]):
        exp1_cvals = exp1_vals[:, i]
        exp2_cvals = exp2_vals[:, i]
        vals1 = exp1_cvals[(exp1_cvals >= sort_avers[0, i]) & (exp1_cvals <= sort_avers[1, i])]
        vals2 = exp2_cvals[(exp2_cvals >= sort_avers[0, i]) & (exp2_cvals <= sort_avers[1, i])]
        if len(vals1) > 0:
            stds[0, i] = np.sqrt(np.sum((vals1 - sort_avers[0, i]) ** 2) / len(vals1))
        if len(vals2) > 0:
            stds[1, i] = np.sqrt(np.sum((vals2 - sort_avers[1, i]) ** 2) / len(vals2))
    
    zstd = stds[0, :] + stds[1, :]
    z_one_tailed = 1 - 3 * (zstd / zrange)
    z_one_tailed[(~np.isfinite(zstd)) | (zrange == 0)] = -1e5
    
    return z, z_one_tailed, xs, avers


def _v_factors(xcol: np.ndarray, ymatr: np.ndarray) -> np.ndarray:
    """Calculate V factors for assay quality.
    
    V factor = 1 - 6 * mean(std) / range
    
    Args:
        xcol: Grouping values (doses)
        ymatr: Matrix of measurements
        
    Returns:
        v: V factors for each measurement
    """
    xs, avers, stds = _loc_shrink_mean_std(xcol, ymatr)
    vrange = np.max(avers, 0) - np.min(avers, 0)
    
    vstd = np.zeros(len(vrange))
    vstd[vrange == 0] = 1
    vstd[vrange != 0] = np.mean(stds[:, vrange != 0], 0)
    vrange[vrange == 0] = 0.000001
    v = 1 - 6 * (vstd / vrange)
    return v


def _sigmoid(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """EC50 sigmoid function.
    
    Args:
        v: Parameters [min, max, ec50, hill_coefficient]
        x: Input values
        
    Returns:
        Sigmoid response values
    """
    p_min, p_max, ec50, hill = v
    return p_min + ((p_max - p_min) / (1 + (x / ec50) ** hill))


def _calc_init_params(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate initial parameters for sigmoid fitting.
    
    Args:
        x: Dose values
        y: Response values
        
    Returns:
        Initial parameters (min, max, ec50, hill)
    """
    min_0 = float(np.min(y))
    max_0 = float(np.max(y))
    
    y_mid = (min_0 + max_0) / 2
    dist = np.abs(y - y_mid)
    loc = np.argmin(dist)
    x_mid = x[loc]
    
    if x_mid == np.min(x) or x_mid == np.max(x):
        ec50 = float((np.min(x) + np.max(x)) / 2)
    else:
        ec50 = float(x_mid)
    
    min_idx = np.argmin(x)
    max_idx = np.argmax(x)
    y0 = y[min_idx]
    y1 = y[max_idx]
    
    if y1 > y0:
        hillc = -1.0
    else:
        hillc = 1.0
    
    return min_0, max_0, ec50, hillc


def _calculate_ec50(conc: np.ndarray, responses: np.ndarray, log_transform: bool = False) -> np.ndarray:
    """Calculate EC50 values by fitting dose-response curves.
    
    Args:
        conc: Concentration/dose values
        responses: Response matrix (observations x measurements)
        log_transform: Whether to log-transform concentrations
        
    Returns:
        EC50 coefficients matrix (measurements x 4 parameters)
    """
    if log_transform:
        conc = np.log(conc + 1e-10)  # Avoid log(0)
    
    n = responses.shape[1]
    results = np.zeros((n, 4))
    
    def error_fn(v, x, y):
        return np.sum((_sigmoid(v, x) - y) ** 2)
    
    for i in range(n):
        response = responses[:, i]
        try:
            v0 = _calc_init_params(conc, response)
            v = scipy.optimize.fmin(
                error_fn, v0, args=(conc, response),
                maxiter=1000, maxfun=1000, disp=False
            )
            results[i, :] = v
        except (ValueError, RuntimeError):
            results[i, :] = [np.nan, np.nan, np.nan, np.nan]
    
    return results


@numpy
@special_outputs(("statistics_results", csv_materializer(
    fields=["feature_name", "object_name", "z_factor", "z_factor_one_tailed", "v_factor", "ec50"],
    analysis_type="statistics"
)))
def calculate_statistics(
    image: np.ndarray,
    grouping_data: Optional[np.ndarray] = None,
    dose_data: Optional[np.ndarray] = None,
    measurement_data: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    object_names: Optional[List[str]] = None,
    log_transform_dose: bool = False,
) -> Tuple[np.ndarray, List[StatisticsResult]]:
    """
    Calculate assay quality statistics (Z' factor, V factor, EC50).
    
    This function calculates experiment-level statistics for assay quality
    assessment. It requires pre-aggregated measurement data from all images.
    
    Args:
        image: Input image array (D, H, W) - passed through unchanged
        grouping_data: Array of positive/negative control designations per image.
                      Positive controls should have max value, negative controls min value.
        dose_data: Array of dose/concentration values per image
        measurement_data: Matrix of measurements (n_images x n_features)
        feature_names: Names of features being measured
        object_names: Names of objects for each feature
        log_transform_dose: Whether to log-transform dose values for EC50 fitting
        
    Returns:
        image: Input image passed through
        results: List of StatisticsResult dataclasses with computed statistics
    """
    results = []
    
    # If no measurement data provided, return empty results
    if measurement_data is None or grouping_data is None:
        return image, results
    
    # Ensure proper shapes
    if measurement_data.ndim == 1:
        measurement_data = measurement_data.reshape(-1, 1)
    
    grouping_data = np.asarray(grouping_data).flatten()
    
    n_features = measurement_data.shape[1]
    
    # Default names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    if object_names is None:
        object_names = ["Image"] * n_features
    
    # Calculate Z' factors
    z_factors, z_one_tailed, _, _ = _z_factors(grouping_data, measurement_data)
    
    # Calculate V factors
    if dose_data is not None:
        dose_data = np.asarray(dose_data).flatten()
        v_factors = _v_factors(dose_data, measurement_data)
        ec50_coeffs = _calculate_ec50(dose_data, measurement_data, log_transform_dose)
        ec50_values = ec50_coeffs[:, 2]  # EC50 is the 3rd parameter
    else:
        v_factors = z_factors  # V factor equals Z' when only two doses
        ec50_values = np.full(n_features, np.nan)
    
    # Build results
    for i in range(n_features):
        results.append(StatisticsResult(
            feature_name=feature_names[i] if i < len(feature_names) else f"Feature_{i}",
            object_name=object_names[i] if i < len(object_names) else "Image",
            z_factor=float(z_factors[i]) if np.isfinite(z_factors[i]) else 0.0,
            z_factor_one_tailed=float(z_one_tailed[i]) if np.isfinite(z_one_tailed[i]) else 0.0,
            v_factor=float(v_factors[i]) if np.isfinite(v_factors[i]) else 0.0,
            ec50=float(ec50_values[i]) if np.isfinite(ec50_values[i]) else 0.0,
        ))
    
    return image, results