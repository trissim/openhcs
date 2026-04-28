"""
Converted from CellProfiler: ClassifyObjects
Original: ClassifyObjects module

Classifies objects into different classes based on measurements or thresholds.
This is a measurement-based classification module that operates on pre-computed
measurements from segmented objects.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer


class ClassificationMethod(Enum):
    SINGLE_MEASUREMENT = "single_measurement"
    TWO_MEASUREMENTS = "two_measurements"


class ThresholdMethod(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    CUSTOM = "custom"


class BinChoice(Enum):
    EVEN = "even"
    CUSTOM = "custom"


@dataclass
class ClassificationResult:
    """Results from object classification."""
    slice_index: int
    total_objects: int
    bin_counts: str  # JSON-encoded dict of bin_name -> count
    bin_percentages: str  # JSON-encoded dict of bin_name -> percentage


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("classification_results", csv_materializer(
        fields=["slice_index", "total_objects", "bin_counts", "bin_percentages"],
        analysis_type="classification"
    ))
)
def classify_objects_single_measurement(
    image: np.ndarray,
    labels: np.ndarray,
    measurement_values: Optional[np.ndarray] = None,
    bin_choice: BinChoice = BinChoice.EVEN,
    bin_count: int = 3,
    low_threshold: float = 0.0,
    high_threshold: float = 1.0,
    wants_low_bin: bool = False,
    wants_high_bin: bool = False,
    custom_thresholds: str = "0,1",
    bin_names: Optional[str] = None,
) -> Tuple[np.ndarray, ClassificationResult]:
    """
    Classify objects based on a single measurement into bins.
    
    Args:
        image: Input image (H, W)
        labels: Label image with segmented objects (H, W)
        measurement_values: Pre-computed measurement values per object.
                          If None, uses mean intensity per object.
        bin_choice: How to define bins (EVEN or CUSTOM)
        bin_count: Number of bins between low and high threshold (for EVEN)
        low_threshold: Lower threshold value (for EVEN)
        high_threshold: Upper threshold value (for EVEN)
        wants_low_bin: Include bin for objects below low threshold
        wants_high_bin: Include bin for objects above high threshold
        custom_thresholds: Comma-separated threshold values (for CUSTOM)
        bin_names: Optional comma-separated custom bin names
    
    Returns:
        Tuple of (classified_labels, classification_results)
    """
    import json
    from skimage.measure import regionprops
    
    # Get unique object labels (excluding background)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_objects = len(unique_labels)
    
    if num_objects == 0:
        return labels, ClassificationResult(
            slice_index=0,
            total_objects=0,
            bin_counts=json.dumps({}),
            bin_percentages=json.dumps({})
        )
    
    # Get measurement values if not provided
    if measurement_values is None:
        # Default to mean intensity per object
        props = regionprops(labels.astype(np.int32), intensity_image=image)
        values = np.array([p.mean_intensity for p in props])
    else:
        values = measurement_values.copy()
    
    # Pad values if needed
    if len(values) < num_objects:
        values = np.concatenate([values, np.full(num_objects - len(values), np.nan)])
    
    # Determine thresholds
    if bin_choice == BinChoice.EVEN:
        if low_threshold >= high_threshold:
            low_threshold, high_threshold = high_threshold, low_threshold
        thresholds = np.linspace(low_threshold, high_threshold, bin_count + 1)
    else:
        thresholds = np.array([float(x.strip()) for x in custom_thresholds.split(",")])
    
    # Add infinite bounds if needed
    threshold_list = []
    if wants_low_bin:
        threshold_list.append(-np.inf)
    threshold_list.extend(thresholds.tolist())
    if wants_high_bin:
        threshold_list.append(np.inf)
    thresholds = np.array(threshold_list)
    
    num_bins = len(thresholds) - 1
    
    # Generate bin names
    if bin_names is not None:
        names = [n.strip() for n in bin_names.split(",")]
    else:
        names = [f"Bin_{i+1}" for i in range(num_bins)]
    
    # Ensure we have enough names
    while len(names) < num_bins:
        names.append(f"Bin_{len(names)+1}")
    
    # Classify each object
    object_bins = np.zeros(num_objects, dtype=np.int32)
    for i, val in enumerate(values):
        if np.isnan(val):
            object_bins[i] = 0  # Unclassified
        else:
            for bin_idx in range(num_bins):
                if thresholds[bin_idx] < val <= thresholds[bin_idx + 1]:
                    object_bins[i] = bin_idx + 1
                    break
    
    # Count objects per bin
    bin_counts = {}
    bin_percentages = {}
    for bin_idx in range(num_bins):
        count = np.sum(object_bins == (bin_idx + 1))
        bin_counts[names[bin_idx]] = int(count)
        bin_percentages[names[bin_idx]] = float(count / num_objects * 100) if num_objects > 0 else 0.0
    
    # Create classified label image
    classified_labels = np.zeros_like(labels, dtype=np.int32)
    for i, label_val in enumerate(unique_labels):
        if object_bins[i] > 0:
            classified_labels[labels == label_val] = object_bins[i]
    
    return classified_labels, ClassificationResult(
        slice_index=0,
        total_objects=num_objects,
        bin_counts=json.dumps(bin_counts),
        bin_percentages=json.dumps(bin_percentages)
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("classification_results", csv_materializer(
        fields=["slice_index", "total_objects", "bin_counts", "bin_percentages"],
        analysis_type="classification"
    ))
)
def classify_objects_two_measurements(
    image: np.ndarray,
    labels: np.ndarray,
    measurement1_values: Optional[np.ndarray] = None,
    measurement2_values: Optional[np.ndarray] = None,
    threshold1_method: ThresholdMethod = ThresholdMethod.MEAN,
    threshold1_value: float = 0.5,
    threshold2_method: ThresholdMethod = ThresholdMethod.MEAN,
    threshold2_value: float = 0.5,
    low_low_name: str = "low_low",
    low_high_name: str = "low_high",
    high_low_name: str = "high_low",
    high_high_name: str = "high_high",
) -> Tuple[np.ndarray, ClassificationResult]:
    """
    Classify objects based on two measurements into four quadrants.
    
    Args:
        image: Input image (H, W)
        labels: Label image with segmented objects (H, W)
        measurement1_values: First measurement values per object
        measurement2_values: Second measurement values per object
        threshold1_method: How to determine threshold for measurement 1
        threshold1_value: Custom threshold for measurement 1
        threshold2_method: How to determine threshold for measurement 2
        threshold2_value: Custom threshold for measurement 2
        low_low_name: Name for low-low bin
        low_high_name: Name for low-high bin
        high_low_name: Name for high-low bin
        high_high_name: Name for high-high bin
    
    Returns:
        Tuple of (classified_labels, classification_results)
    """
    import json
    from skimage.measure import regionprops
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_objects = len(unique_labels)
    
    if num_objects == 0:
        return labels, ClassificationResult(
            slice_index=0,
            total_objects=0,
            bin_counts=json.dumps({}),
            bin_percentages=json.dumps({})
        )
    
    # Get measurement values if not provided
    props = regionprops(labels.astype(np.int32), intensity_image=image)
    
    if measurement1_values is None:
        values1 = np.array([p.mean_intensity for p in props])
    else:
        values1 = measurement1_values.copy()
    
    if measurement2_values is None:
        values2 = np.array([p.area for p in props])
    else:
        values2 = measurement2_values.copy()
    
    # Determine thresholds
    def get_threshold(values, method, custom_value):
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return custom_value
        if method == ThresholdMethod.MEAN:
            return np.mean(valid_values)
        elif method == ThresholdMethod.MEDIAN:
            return np.median(valid_values)
        else:
            return custom_value
    
    t1 = get_threshold(values1, threshold1_method, threshold1_value)
    t2 = get_threshold(values2, threshold2_method, threshold2_value)
    
    # Classify into quadrants
    high1 = values1 >= t1
    high2 = values2 >= t2
    has_nan = np.isnan(values1) | np.isnan(values2)
    
    # Quadrant assignments: 1=low_low, 2=high_low, 3=low_high, 4=high_high
    object_class = np.zeros(num_objects, dtype=np.int32)
    object_class[(~high1) & (~high2) & (~has_nan)] = 1  # low_low
    object_class[(high1) & (~high2) & (~has_nan)] = 2   # high_low
    object_class[(~high1) & (high2) & (~has_nan)] = 3   # low_high
    object_class[(high1) & (high2) & (~has_nan)] = 4    # high_high
    
    names = [low_low_name, high_low_name, low_high_name, high_high_name]
    
    bin_counts = {}
    bin_percentages = {}
    for i, name in enumerate(names):
        count = np.sum(object_class == (i + 1))
        bin_counts[name] = int(count)
        bin_percentages[name] = float(count / num_objects * 100) if num_objects > 0 else 0.0
    
    # Create classified label image
    classified_labels = np.zeros_like(labels, dtype=np.int32)
    for i, label_val in enumerate(unique_labels):
        if object_class[i] > 0:
            classified_labels[labels == label_val] = object_class[i]
    
    return classified_labels, ClassificationResult(
        slice_index=0,
        total_objects=num_objects,
        bin_counts=json.dumps(bin_counts),
        bin_percentages=json.dumps(bin_percentages)
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("classification_results", csv_materializer(
        fields=["slice_index", "total_objects", "bin_counts", "bin_percentages"],
        analysis_type="classification"
    ))
)
def classify_objects_by_intensity_bins(
    image: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 3,
    use_percentiles: bool = True,
) -> Tuple[np.ndarray, ClassificationResult]:
    """
    Classify objects by mean intensity into evenly distributed bins.
    
    Args:
        image: Input intensity image (H, W)
        labels: Label image with segmented objects (H, W)
        num_bins: Number of classification bins
        use_percentiles: If True, use percentile-based thresholds for even distribution
    
    Returns:
        Tuple of (classified_labels, classification_results)
    """
    import json
    from skimage.measure import regionprops
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_objects = len(unique_labels)
    
    if num_objects == 0:
        return labels, ClassificationResult(
            slice_index=0,
            total_objects=0,
            bin_counts=json.dumps({}),
            bin_percentages=json.dumps({})
        )
    
    # Measure mean intensity per object
    props = regionprops(labels.astype(np.int32), intensity_image=image)
    values = np.array([p.mean_intensity for p in props])
    
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    
    if len(valid_values) == 0:
        return labels, ClassificationResult(
            slice_index=0,
            total_objects=num_objects,
            bin_counts=json.dumps({}),
            bin_percentages=json.dumps({})
        )
    
    # Determine thresholds
    if use_percentiles:
        percentiles = np.linspace(0, 100, num_bins + 1)
        thresholds = np.percentile(valid_values, percentiles)
    else:
        thresholds = np.linspace(np.min(valid_values), np.max(valid_values), num_bins + 1)
    
    # Classify objects
    object_bins = np.zeros(num_objects, dtype=np.int32)
    for i, val in enumerate(values):
        if np.isnan(val):
            continue
        for bin_idx in range(num_bins):
            if bin_idx == num_bins - 1:
                if thresholds[bin_idx] <= val <= thresholds[bin_idx + 1]:
                    object_bins[i] = bin_idx + 1
            else:
                if thresholds[bin_idx] <= val < thresholds[bin_idx + 1]:
                    object_bins[i] = bin_idx + 1
                    break
    
    # Generate results
    bin_names = [f"Intensity_Bin_{i+1}" for i in range(num_bins)]
    bin_counts = {}
    bin_percentages = {}
    for i, name in enumerate(bin_names):
        count = np.sum(object_bins == (i + 1))
        bin_counts[name] = int(count)
        bin_percentages[name] = float(count / num_objects * 100) if num_objects > 0 else 0.0
    
    # Create classified label image
    classified_labels = np.zeros_like(labels, dtype=np.int32)
    for i, label_val in enumerate(unique_labels):
        if object_bins[i] > 0:
            classified_labels[labels == label_val] = object_bins[i]
    
    return classified_labels, ClassificationResult(
        slice_index=0,
        total_objects=num_objects,
        bin_counts=json.dumps(bin_counts),
        bin_percentages=json.dumps(bin_percentages)
    )