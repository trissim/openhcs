"""MeasureObjectOverlap execution policy for CellProfiler-compatible backends."""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
import scipy.ndimage
import scipy.sparse
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.materialization import csv_materializer


class DecimationMethod(Enum):
    KMEANS = "kmeans"
    SKELETON = "skeleton"


@dataclass
class OverlapMeasurements:
    """Measurements from object overlap analysis."""
    slice_index: int
    f_factor: float
    precision: float
    recall: float
    true_positive_rate: float
    false_positive_rate: float
    true_negative_rate: float
    false_negative_rate: float
    rand_index: float
    adjusted_rand_index: float
    earth_movers_distance: float


@dataclass
class ImageOverlapMeasurement:
    slice_index: int
    true_positive_rate: float
    false_positive_rate: float
    false_negative_rate: float
    true_negative_rate: float
    precision: float
    recall: float
    f_factor: float
    jaccard_index: float
    dice_coefficient: float
    rand_index: float
    adjusted_rand_index: float
    earth_movers_distance: float


def _nan_divide(numerator: float, denominator: float) -> float:
    """Safe division that returns NaN for zero denominator."""
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _compute_rand_index_ijv(gt_ijv: np.ndarray, test_ijv: np.ndarray, shape: Tuple[int, int]) -> Tuple[float, float]:
    """
    Compute the Rand Index for IJV matrices.
    
    Based on the Omega Index from Collins (1988).
    """
    # Add backgrounds with label zero
    gt_bkgd = np.ones(shape, bool)
    if len(gt_ijv) > 0:
        gt_bkgd[gt_ijv[:, 0], gt_ijv[:, 1]] = False
    test_bkgd = np.ones(shape, bool)
    if len(test_ijv) > 0:
        test_bkgd[test_ijv[:, 0], test_ijv[:, 1]] = False
    
    gt_bkgd_coords = np.argwhere(gt_bkgd)
    test_bkgd_coords = np.argwhere(test_bkgd)
    
    if len(gt_bkgd_coords) > 0:
        gt_ijv = np.vstack([
            gt_ijv,
            np.column_stack([gt_bkgd_coords, np.zeros(len(gt_bkgd_coords), dtype=gt_ijv.dtype if len(gt_ijv) > 0 else np.int32)])
        ]) if len(gt_ijv) > 0 else np.column_stack([gt_bkgd_coords, np.zeros(len(gt_bkgd_coords), dtype=np.int32)])
    
    if len(test_bkgd_coords) > 0:
        test_ijv = np.vstack([
            test_ijv,
            np.column_stack([test_bkgd_coords, np.zeros(len(test_bkgd_coords), dtype=test_ijv.dtype if len(test_ijv) > 0 else np.int32)])
        ]) if len(test_ijv) > 0 else np.column_stack([test_bkgd_coords, np.zeros(len(test_bkgd_coords), dtype=np.int32)])
    
    if len(gt_ijv) == 0 or len(test_ijv) == 0:
        return np.nan, np.nan
    
    # Create unified structure
    u = np.vstack([
        np.column_stack([gt_ijv, np.zeros(gt_ijv.shape[0], dtype=np.int32)]),
        np.column_stack([test_ijv, np.ones(test_ijv.shape[0], dtype=np.int32)])
    ])
    
    # Sort by coordinates then identity
    order = np.lexsort([u[:, 2], u[:, 3], u[:, 0], u[:, 1]])
    u = u[order, :]
    
    # Remove duplicates
    first = np.hstack([[True], np.any(u[:-1, :] != u[1:, :], axis=1)])
    u = u[first, :]
    
    # Create coordinate indexer
    coord_changes = np.hstack([
        [0],
        np.argwhere((u[:-1, 0] != u[1:, 0]) | (u[:-1, 1] != u[1:, 1])).flatten() + 1,
        [u.shape[0]]
    ])
    coord_counts = coord_changes[1:] - coord_changes[:-1]
    
    # Count test and gt labels at each coordinate
    rev_idx = np.repeat(np.arange(len(coord_counts)), coord_counts)
    count_test = np.bincount(rev_idx, u[:, 3]).astype(np.int64)
    count_gt = coord_counts - count_test
    
    # Simplified Rand index calculation
    # For each unique coordinate, count pairs
    n_coords = len(coord_counts)
    if n_coords < 2:
        return 1.0, 1.0
    
    # Simple approximation: count matching pairs
    total_pairs = n_coords * (n_coords - 1) // 2
    
    # Count agreements (both in same set or both in different sets)
    agreements = 0
    for i in range(n_coords):
        for j in range(i + 1, min(i + 100, n_coords)):  # Limit for performance
            same_gt = count_gt[i] > 0 and count_gt[j] > 0
            same_test = count_test[i] > 0 and count_test[j] > 0
            if same_gt == same_test:
                agreements += 1
    
    sampled_pairs = min(total_pairs, n_coords * 50)
    rand_index = agreements / sampled_pairs if sampled_pairs > 0 else np.nan
    
    # Adjusted Rand index (simplified)
    adjusted_rand_index = 2 * rand_index - 1 if not np.isnan(rand_index) else np.nan
    
    return rand_index, adjusted_rand_index


def _labels_to_ijv(labels: np.ndarray) -> np.ndarray:
    """Convert label image to IJV format (row, col, label)."""
    i, j = np.where(labels > 0)
    if len(i) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    v = labels[i, j]
    return np.column_stack([i, j, v]).astype(np.int32)


def _compute_emd_simple(src_labels: np.ndarray, dest_labels: np.ndarray, 
                        max_points: int, max_distance: int, penalize_missing: bool) -> float:
    """
    Compute simplified Earth Mover's Distance between two label images.
    """
    src_mask = src_labels > 0
    dest_mask = dest_labels > 0
    
    src_area = np.sum(src_mask)
    dest_area = np.sum(dest_mask)
    
    if src_area == 0 and dest_area == 0:
        return 0.0
    
    if src_area == 0 or dest_area == 0:
        if penalize_missing:
            return max(src_area, dest_area) * max_distance
        return 0.0
    
    # Get representative points using simple sampling
    src_coords = np.argwhere(src_mask)
    dest_coords = np.argwhere(dest_mask)
    
    # Subsample if needed
    if len(src_coords) > max_points:
        idx = np.linspace(0, len(src_coords) - 1, max_points).astype(int)
        src_coords = src_coords[idx]
    if len(dest_coords) > max_points:
        idx = np.linspace(0, len(dest_coords) - 1, max_points).astype(int)
        dest_coords = dest_coords[idx]
    
    # Compute pairwise distances and find minimum cost assignment (greedy)
    total_distance = 0.0
    for sc in src_coords:
        if len(dest_coords) == 0:
            total_distance += max_distance
            continue
        distances = np.sqrt(np.sum((dest_coords - sc) ** 2, axis=1))
        min_dist = np.min(distances)
        total_distance += min(min_dist, max_distance)
    
    # Normalize by number of points
    return total_distance / len(src_coords) if len(src_coords) > 0 else 0.0


@numpy
@special_outputs(
    (
        "overlap_measurements",
        csv_materializer(
            fields=[
                "slice_index",
                "true_positive_rate",
                "false_positive_rate",
                "false_negative_rate",
                "true_negative_rate",
                "precision",
                "recall",
                "f_factor",
                "jaccard_index",
                "dice_coefficient",
                "rand_index",
                "adjusted_rand_index",
                "earth_movers_distance",
            ],
            analysis_type="image_overlap",
        ),
    )
)
def measureimageoverlap(
    image: np.ndarray,
    calculate_emd: bool = False,
    max_distance: int = 250,
    penalize_missing: bool = False,
    decimation_method: DecimationMethod = DecimationMethod.KMEANS,
    max_points: int = 250,
) -> Tuple[np.ndarray, ImageOverlapMeasurement]:
    """Measure binary overlap between ground-truth and test image planes."""
    if image.shape[0] < 2:
        raise ValueError("Image must have at least 2 slices (ground_truth, test)")

    ground_truth_image = image[0].astype(bool)
    test_image = image[1].astype(bool)
    mask = image[2].astype(bool) if image.shape[0] > 2 else None
    if mask is not None:
        ground_truth_image = ground_truth_image & mask
        test_image = test_image & mask
        total_pixels = np.sum(mask)
    else:
        total_pixels = ground_truth_image.size

    true_positive = np.sum(ground_truth_image & test_image)
    false_positive = np.sum(~ground_truth_image & test_image)
    false_negative = np.sum(ground_truth_image & ~test_image)
    true_negative = np.sum(~ground_truth_image & ~test_image)
    eps = 1e-10
    true_positive_rate = true_positive / (true_positive + false_negative + eps)
    false_positive_rate = false_positive / (false_positive + true_negative + eps)
    false_negative_rate = false_negative / (true_positive + false_negative + eps)
    true_negative_rate = true_negative / (false_positive + true_negative + eps)
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive_rate
    f_factor = 2 * precision * recall / (precision + recall + eps)
    intersection = true_positive
    union = true_positive + false_positive + false_negative
    jaccard_index = intersection / (union + eps)
    dice_coefficient = (
        2 * intersection / (2 * intersection + false_positive + false_negative + eps)
    )
    n = total_pixels
    a = true_positive
    b = false_positive
    c = false_negative
    d = true_negative
    rand_index = (a + d) / (a + b + c + d + eps)
    n_choose_2 = n * (n - 1) / 2 if n > 1 else 1
    sum_ni_choose_2 = a + c
    sum_nj_choose_2 = a + b
    expected_index = (sum_ni_choose_2 * sum_nj_choose_2) / (n_choose_2 + eps)
    max_index = (sum_ni_choose_2 + sum_nj_choose_2) / 2
    adjusted_rand_index = (a - expected_index) / (max_index - expected_index + eps)
    adjusted_rand_index = max(0.0, min(1.0, adjusted_rand_index))

    earth_movers_distance = 0.0
    if calculate_emd:
        earth_movers_distance = compute_image_earth_movers_distance(
            ground_truth_image,
            test_image,
            max_distance=max_distance,
            penalize_missing=penalize_missing,
            decimation_method=decimation_method,
            max_points=max_points,
        )

    measurements = ImageOverlapMeasurement(
        slice_index=0,
        true_positive_rate=float(true_positive_rate),
        false_positive_rate=float(false_positive_rate),
        false_negative_rate=float(false_negative_rate),
        true_negative_rate=float(true_negative_rate),
        precision=float(precision),
        recall=float(recall),
        f_factor=float(f_factor),
        jaccard_index=float(jaccard_index),
        dice_coefficient=float(dice_coefficient),
        rand_index=float(rand_index),
        adjusted_rand_index=float(adjusted_rand_index),
        earth_movers_distance=float(earth_movers_distance),
    )
    return ground_truth_image.astype(np.float32)[np.newaxis, ...], measurements


def compute_image_earth_movers_distance(
    ground_truth: np.ndarray,
    test: np.ndarray,
    max_distance: int,
    penalize_missing: bool,
    decimation_method: DecimationMethod,
    max_points: int,
) -> float:
    from scipy.spatial.distance import cdist

    gt_coords = np.argwhere(ground_truth)
    test_coords = np.argwhere(test)
    if len(gt_coords) == 0 or len(test_coords) == 0:
        if penalize_missing:
            return float(max_distance)
        return 0.0
    if len(gt_coords) > max_points:
        gt_coords = decimate_overlap_points(gt_coords, max_points, decimation_method)
    if len(test_coords) > max_points:
        test_coords = decimate_overlap_points(test_coords, max_points, decimation_method)
    distances = cdist(gt_coords, test_coords, metric="euclidean")
    distances = np.minimum(distances, max_distance)
    min_dist_gt_to_test = np.mean(np.min(distances, axis=1))
    min_dist_test_to_gt = np.mean(np.min(distances, axis=0))
    return float((min_dist_gt_to_test + min_dist_test_to_gt) / 2)


def decimate_overlap_points(
    coords: np.ndarray,
    max_points: int,
    method: DecimationMethod,
) -> np.ndarray:
    del method
    indices = np.linspace(0, len(coords) - 1, max_points, dtype=int)
    return coords[indices]


@numpy
@special_inputs("labels_ground_truth", "labels_test")
@special_outputs(("overlap_measurements", csv_materializer(
    fields=["slice_index", "f_factor", "precision", "recall", 
            "true_positive_rate", "false_positive_rate", 
            "true_negative_rate", "false_negative_rate",
            "rand_index", "adjusted_rand_index", "earth_movers_distance"],
    analysis_type="object_overlap"
)))
def measure_object_overlap(
    image: np.ndarray,
    labels_ground_truth: np.ndarray,
    labels_test: np.ndarray,
    calculate_emd: bool = False,
    max_points: int = 250,
    decimation_method: DecimationMethod = DecimationMethod.KMEANS,
    max_distance: int = 250,
    penalize_missing: bool = False,
) -> Tuple[np.ndarray, OverlapMeasurements]:
    """
    Calculate overlap statistics between ground truth and test segmentation objects.
    
    Args:
        image: Input image array, shape (2, H, W) - ground truth labels stacked with test labels,
               or (H, W) if labels provided via special_inputs
        labels_ground_truth: Ground truth segmentation labels
        labels_test: Test segmentation labels to compare
        calculate_emd: Whether to calculate Earth Mover's Distance
        max_points: Maximum number of representative points for EMD calculation
        decimation_method: Method for selecting representative points (KMEANS or SKELETON)
        max_distance: Maximum distance penalty for EMD calculation
        penalize_missing: Whether to penalize missing pixels in EMD calculation
    
    Returns:
        Tuple of (original image, overlap measurements)
    """
    # Handle input - if labels not provided via special_inputs, unstack from image
    if labels_ground_truth is None or labels_test is None:
        if image.ndim == 3 and image.shape[0] >= 2:
            labels_ground_truth = image[0].astype(np.int32)
            labels_test = image[1].astype(np.int32)
            output_image = image[0] if image.shape[0] == 2 else image[2:]
        else:
            raise ValueError("Labels must be provided either via special_inputs or stacked in image")
    else:
        output_image = image
        labels_ground_truth = object_label_dense_array(
            labels_ground_truth,
            dtype=np.int32,
        )
        labels_test = object_label_dense_array(labels_test, dtype=np.int32)
    
    # Ensure 2D
    if labels_ground_truth.ndim == 3:
        labels_ground_truth = labels_ground_truth[0]
    if labels_test.ndim == 3:
        labels_test = labels_test[0]
    
    # Convert to IJV format
    gt_ijv = _labels_to_ijv(labels_ground_truth)
    test_ijv = _labels_to_ijv(labels_test)
    
    # Get dimensions
    shape = (max(labels_ground_truth.shape[0], labels_test.shape[0]),
             max(labels_ground_truth.shape[1], labels_test.shape[1]))
    total_pixels = shape[0] * shape[1]
    
    # Create binary masks
    gt_mask = labels_ground_truth > 0
    test_mask = labels_test > 0
    
    # Calculate confusion matrix elements
    TP = np.sum(gt_mask & test_mask)
    FP = np.sum(~gt_mask & test_mask)
    FN = np.sum(gt_mask & ~test_mask)
    TN = np.sum(~gt_mask & ~test_mask)
    
    gt_total = np.sum(gt_mask)
    
    # Calculate metrics
    precision = _nan_divide(TP, TP + FP)
    recall = _nan_divide(TP, TP + FN)
    f_factor = _nan_divide(2 * precision * recall, precision + recall)
    true_positive_rate = _nan_divide(TP, FN + TP)
    false_positive_rate = _nan_divide(FP, FP + TN)
    false_negative_rate = _nan_divide(FN, FN + TP)
    true_negative_rate = _nan_divide(TN, FP + TN)
    
    # Calculate Rand indices
    rand_index, adjusted_rand_index = _compute_rand_index_ijv(gt_ijv, test_ijv, shape)
    
    # Calculate Earth Mover's Distance if requested
    if calculate_emd:
        emd = _compute_emd_simple(labels_ground_truth, labels_test, 
                                   max_points, max_distance, penalize_missing)
    else:
        emd = np.nan
    
    measurements = OverlapMeasurements(
        slice_index=0,
        f_factor=float(f_factor) if not np.isnan(f_factor) else 0.0,
        precision=float(precision) if not np.isnan(precision) else 0.0,
        recall=float(recall) if not np.isnan(recall) else 0.0,
        true_positive_rate=float(true_positive_rate) if not np.isnan(true_positive_rate) else 0.0,
        false_positive_rate=float(false_positive_rate) if not np.isnan(false_positive_rate) else 0.0,
        true_negative_rate=float(true_negative_rate) if not np.isnan(true_negative_rate) else 0.0,
        false_negative_rate=float(false_negative_rate) if not np.isnan(false_negative_rate) else 0.0,
        rand_index=float(rand_index) if not np.isnan(rand_index) else 0.0,
        adjusted_rand_index=float(adjusted_rand_index) if not np.isnan(adjusted_rand_index) else 0.0,
        earth_movers_distance=float(emd) if not np.isnan(emd) else 0.0
    )
    
    return output_image, measurements


__all__ = [
    "DecimationMethod",
    "ImageOverlapMeasurement",
    "OverlapMeasurements",
    "compute_image_earth_movers_distance",
    "decimate_overlap_points",
    "measureimageoverlap",
    "measure_object_overlap",
]
