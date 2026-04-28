"""
Converted from CellProfiler: DisplayScatterPlot
Original: DisplayScatterPlot

Note: This module is a visualization/data tool that plots measurement values.
In OpenHCS, visualization is handled differently - this function extracts
and returns scatter plot data that can be visualized by the frontend.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer


class MeasurementSource(Enum):
    IMAGE = "Image"
    OBJECT = "Object"


class ScaleType(Enum):
    LINEAR = "linear"
    LOG = "log"


@dataclass
class ScatterPlotData:
    """Data structure for scatter plot output."""
    slice_index: int
    x_values: str  # JSON-encoded array of x values
    y_values: str  # JSON-encoded array of y values
    x_label: str
    y_label: str
    x_scale: str
    y_scale: str
    title: str
    point_count: int


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("measurements_x", "measurements_y")
@special_outputs(("scatter_plot_data", csv_materializer(
    fields=["slice_index", "x_values", "y_values", "x_label", "y_label", 
            "x_scale", "y_scale", "title", "point_count"],
    analysis_type="scatter_plot"
)))
def display_scatter_plot(
    image: np.ndarray,
    measurements_x: np.ndarray,
    measurements_y: np.ndarray,
    x_source: MeasurementSource = MeasurementSource.OBJECT,
    y_source: MeasurementSource = MeasurementSource.OBJECT,
    x_axis_label: str = "X Measurement",
    y_axis_label: str = "Y Measurement",
    x_scale: ScaleType = ScaleType.LINEAR,
    y_scale: ScaleType = ScaleType.LINEAR,
    title: str = "",
) -> Tuple[np.ndarray, ScatterPlotData]:
    """
    Extract scatter plot data from two measurement arrays.
    
    This function prepares data for scatter plot visualization by pairing
    corresponding measurements from two arrays. The actual visualization
    is handled by the OpenHCS frontend.
    
    Args:
        image: Input image array (H, W), passed through unchanged
        measurements_x: Array of x-axis measurement values
        measurements_y: Array of y-axis measurement values
        x_source: Source type for x measurements (Image or Object)
        y_source: Source type for y measurements (Image or Object)
        x_axis_label: Label for x-axis
        y_axis_label: Label for y-axis
        x_scale: Scale type for x-axis (linear or log)
        y_scale: Scale type for y-axis (linear or log)
        title: Plot title (empty string for auto-generated title)
    
    Returns:
        Tuple of (original image, scatter plot data)
    """
    import json
    
    # Flatten measurements if needed
    x_vals = np.asarray(measurements_x).flatten()
    y_vals = np.asarray(measurements_y).flatten()
    
    # Handle mismatched lengths - take minimum length
    min_len = min(len(x_vals), len(y_vals))
    x_vals = x_vals[:min_len]
    y_vals = y_vals[:min_len]
    
    # Filter out NaN and None values
    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[valid_mask]
    y_vals = y_vals[valid_mask]
    
    # Apply log transform if needed (filter out non-positive values)
    if x_scale == ScaleType.LOG:
        positive_x = x_vals > 0
        x_vals = x_vals[positive_x]
        y_vals = y_vals[positive_x]
    
    if y_scale == ScaleType.LOG:
        positive_y = y_vals > 0
        x_vals = x_vals[positive_y]
        y_vals = y_vals[positive_y]
    
    # Generate title if not provided
    plot_title = title if title else f"{x_axis_label} vs {y_axis_label}"
    
    # Create scatter plot data
    scatter_data = ScatterPlotData(
        slice_index=0,
        x_values=json.dumps(x_vals.tolist()),
        y_values=json.dumps(y_vals.tolist()),
        x_label=x_axis_label,
        y_label=y_axis_label,
        x_scale=x_scale.value,
        y_scale=y_scale.value,
        title=plot_title,
        point_count=len(x_vals)
    )
    
    return image, scatter_data