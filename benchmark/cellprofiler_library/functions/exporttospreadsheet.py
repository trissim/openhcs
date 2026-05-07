"""
Converted from CellProfiler: ExportToSpreadsheet
Original: ExportToSpreadsheet

Note: ExportToSpreadsheet is a data export module, not an image processing module.
In OpenHCS, data export is handled by the materialization system, not by processing functions.
This conversion provides a measurement aggregation function that can be used with csv_materializer.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


class Delimiter(Enum):
    TAB = "tab"
    COMMA = "comma"


class NanRepresentation(Enum):
    NULLS = "null"
    NANS = "nan"


@dataclass
class ImageMeasurements:
    """Container for image-level measurements to be exported."""
    image_number: int
    measurements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectMeasurements:
    """Container for object-level measurements to be exported."""
    image_number: int
    object_number: int
    object_name: str
    measurements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateStats:
    """Aggregate statistics for objects in an image."""
    image_number: int
    object_name: str
    measurement_name: str
    mean_value: float
    median_value: float
    std_value: float


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("aggregate_stats", csv_materializer(
    fields=["image_number", "object_name", "measurement_name", "mean_value", "median_value", "std_value"],
    analysis_type="aggregate_measurements"
)))
def compute_aggregate_measurements(
    image: np.ndarray,
    labels: np.ndarray,
    object_name: str = "Objects",
    compute_mean: bool = False,
    compute_median: bool = False,
    compute_std: bool = False,
    nan_representation: NanRepresentation = NanRepresentation.NANS,
) -> Tuple[np.ndarray, AggregateStats]:
    """
    Compute aggregate measurements (mean, median, std) for objects in an image.
    
    This function computes per-image aggregate statistics over all objects,
    which can then be exported via the materialization system.
    
    In OpenHCS, actual file export is handled by materializers configured in the
    pipeline, not by processing functions. This function prepares measurement
    data for export.
    
    Args:
        image: Input intensity image, shape (H, W)
        labels: Label image where each object has a unique integer ID, shape (H, W)
        object_name: Name of the object type being measured
        compute_mean: Whether to compute mean values
        compute_median: Whether to compute median values  
        compute_std: Whether to compute standard deviation values
        nan_representation: How to represent NaN values in output
    
    Returns:
        Tuple of (original image, aggregate statistics dataclass)
    """
    from skimage.measure import regionprops
    labels = object_label_dense_array(labels, dtype=np.int32)
    
    # Get object properties
    props = regionprops(labels, intensity_image=image)
    
    if len(props) == 0:
        # No objects found
        mean_val = np.nan if nan_representation == NanRepresentation.NANS else 0.0
        median_val = np.nan if nan_representation == NanRepresentation.NANS else 0.0
        std_val = np.nan if nan_representation == NanRepresentation.NANS else 0.0
    else:
        # Compute intensity measurements for each object
        intensities = [prop.mean_intensity for prop in props]
        areas = [prop.area for prop in props]
        
        # Compute aggregates
        if compute_mean:
            mean_val = float(np.mean(intensities))
        else:
            mean_val = np.nan
            
        if compute_median:
            median_val = float(np.median(intensities))
        else:
            median_val = np.nan
            
        if compute_std:
            std_val = float(np.std(intensities))
        else:
            std_val = np.nan
    
    # Handle NaN representation
    if nan_representation == NanRepresentation.NULLS:
        if np.isnan(mean_val):
            mean_val = 0.0
        if np.isnan(median_val):
            median_val = 0.0
        if np.isnan(std_val):
            std_val = 0.0
    
    stats = AggregateStats(
        image_number=0,  # Will be set by pipeline context
        object_name=object_name,
        measurement_name="Intensity",
        mean_value=mean_val,
        median_value=median_val,
        std_value=std_val
    )
    
    return image, stats


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("object_measurements", csv_materializer(
    fields=["image_number", "object_number", "area", "mean_intensity", "centroid_x", "centroid_y"],
    analysis_type="object_measurements"
)))
def extract_object_measurements(
    image: np.ndarray,
    labels: np.ndarray,
    add_metadata: bool = False,
    add_filepath: bool = False,
    nan_representation: NanRepresentation = NanRepresentation.NANS,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extract per-object measurements for export.
    
    This function extracts measurements for each segmented object,
    preparing them for CSV export via the materialization system.
    
    Args:
        image: Input intensity image, shape (H, W)
        labels: Label image where each object has a unique integer ID, shape (H, W)
        add_metadata: Whether to include image metadata columns
        add_filepath: Whether to include file path columns
        nan_representation: How to represent NaN values
    
    Returns:
        Tuple of (original image, list of measurement dictionaries)
    """
    from skimage.measure import regionprops
    labels = object_label_dense_array(labels, dtype=np.int32)
    
    props = regionprops(labels, intensity_image=image)
    
    measurements = []
    for i, prop in enumerate(props):
        centroid = prop.centroid
        
        meas = {
            "image_number": 0,  # Set by pipeline
            "object_number": i + 1,
            "area": float(prop.area),
            "mean_intensity": float(prop.mean_intensity),
            "centroid_x": float(centroid[1]),
            "centroid_y": float(centroid[0]),
        }
        
        # Handle NaN values
        if nan_representation == NanRepresentation.NULLS:
            for key, val in meas.items():
                if isinstance(val, float) and np.isnan(val):
                    meas[key] = None
        
        measurements.append(meas)
    
    return image, measurements
