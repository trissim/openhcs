"""
Converted from CellProfiler: DisplayHistogram
Original: DisplayHistogram

DisplayHistogram plots a histogram of measurement data.
This is a data visualization/analysis module that computes histogram statistics
from measurement values rather than processing images directly.

Note: In OpenHCS, this module computes histogram statistics and returns them
as measurements. The actual visualization is handled by the pipeline's
visualization layer, not by this function.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer


class AxisScale(Enum):
    LINEAR = "linear"
    LOG = "log"


@dataclass
class HistogramResult:
    """Histogram computation results."""
    slice_index: int
    bin_count: int
    data_min: float
    data_max: float
    data_mean: float
    data_std: float
    data_median: float
    total_count: int
    # Histogram bin edges and counts stored as comma-separated strings for CSV
    bin_edges: str
    bin_counts: str


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("histogram_results", csv_materializer(
    fields=["slice_index", "bin_count", "data_min", "data_max", "data_mean", 
            "data_std", "data_median", "total_count", "bin_edges", "bin_counts"],
    analysis_type="histogram"
)))
def display_histogram(
    image: np.ndarray,
    labels: np.ndarray,
    measurement_type: str = "intensity_mean",
    num_bins: int = 100,
    x_scale: AxisScale = AxisScale.LINEAR,
    y_scale: AxisScale = AxisScale.LINEAR,
    use_x_bounds: bool = False,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> Tuple[np.ndarray, HistogramResult]:
    """
    Compute histogram statistics from object measurements.
    
    This function extracts measurements from labeled objects and computes
    histogram statistics. The actual histogram visualization is handled
    by the pipeline's visualization layer.
    
    Args:
        image: Input intensity image, shape (H, W)
        labels: Label image from segmentation, shape (H, W)
        measurement_type: Type of measurement to histogram
            - "intensity_mean": Mean intensity per object
            - "intensity_sum": Sum of intensity per object  
            - "area": Area of each object in pixels
            - "perimeter": Perimeter of each object
        num_bins: Number of histogram bins (1-1000)
        x_scale: Scale for X-axis (linear or log)
        y_scale: Scale for Y-axis (linear or log)
        use_x_bounds: Whether to apply min/max bounds to X-axis
        x_min: Minimum X-axis value (if use_x_bounds is True)
        x_max: Maximum X-axis value (if use_x_bounds is True)
    
    Returns:
        Tuple of (original image, histogram results)
    """
    from skimage.measure import regionprops
    
    # Handle empty labels
    if labels.max() == 0:
        return image, HistogramResult(
            slice_index=0,
            bin_count=num_bins,
            data_min=0.0,
            data_max=0.0,
            data_mean=0.0,
            data_std=0.0,
            data_median=0.0,
            total_count=0,
            bin_edges="",
            bin_counts=""
        )
    
    # Extract measurements from labeled objects
    props = regionprops(labels.astype(np.int32), intensity_image=image)
    
    if len(props) == 0:
        return image, HistogramResult(
            slice_index=0,
            bin_count=num_bins,
            data_min=0.0,
            data_max=0.0,
            data_mean=0.0,
            data_std=0.0,
            data_median=0.0,
            total_count=0,
            bin_edges="",
            bin_counts=""
        )
    
    # Get measurement values based on type
    if measurement_type == "intensity_mean":
        values = np.array([p.mean_intensity for p in props])
    elif measurement_type == "intensity_sum":
        values = np.array([p.mean_intensity * p.area for p in props])
    elif measurement_type == "area":
        values = np.array([p.area for p in props])
    elif measurement_type == "perimeter":
        values = np.array([p.perimeter for p in props])
    else:
        # Default to mean intensity
        values = np.array([p.mean_intensity for p in props])
    
    # Apply log transform if needed for x-axis
    if x_scale == AxisScale.LOG:
        # Avoid log(0) by filtering out zeros and negatives
        values = values[values > 0]
        if len(values) > 0:
            values = np.log(values)
    
    # Apply X bounds if specified
    if use_x_bounds and len(values) > 0:
        values = values[values >= x_min]
        values = values[values <= x_max]
    
    # Handle empty values after filtering
    if len(values) == 0:
        return image, HistogramResult(
            slice_index=0,
            bin_count=num_bins,
            data_min=0.0,
            data_max=0.0,
            data_mean=0.0,
            data_std=0.0,
            data_median=0.0,
            total_count=0,
            bin_edges="",
            bin_counts=""
        )
    
    # Compute histogram
    counts, bin_edges = np.histogram(values, bins=num_bins)
    
    # Apply log transform to counts if y-scale is log
    if y_scale == AxisScale.LOG:
        counts = np.log1p(counts)  # log(1 + x) to handle zeros
    
    # Compute statistics
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    data_mean = float(np.mean(values))
    data_std = float(np.std(values))
    data_median = float(np.median(values))
    
    # Convert arrays to comma-separated strings for CSV storage
    bin_edges_str = ",".join([f"{x:.6f}" for x in bin_edges])
    bin_counts_str = ",".join([f"{x:.6f}" for x in counts])
    
    result = HistogramResult(
        slice_index=0,
        bin_count=num_bins,
        data_min=data_min,
        data_max=data_max,
        data_mean=data_mean,
        data_std=data_std,
        data_median=data_median,
        total_count=len(values),
        bin_edges=bin_edges_str,
        bin_counts=bin_counts_str
    )
    
    return image, result