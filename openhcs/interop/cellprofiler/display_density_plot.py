"""
Converted from CellProfiler: DisplayDensityPlot
Original: DisplayDensityPlot

Note: This module is a visualization/data tool that creates density plots from
measurements. In OpenHCS, this is converted to a measurement aggregation function
that computes 2D histogram data from measurement arrays.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Tuple

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


class ScaleType(Enum):
    LINEAR = "linear"
    LOG = "log"


class ColorMap(Enum):
    JET = "jet"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    HOT = "hot"
    COOL = "cool"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"
    GRAY = "gray"
    BONE = "bone"
    COPPER = "copper"
    PINK = "pink"


@dataclass
class DensityPlotAxis:
    """One density plot axis range and scale."""

    min_value: float
    max_value: float
    scale: str


@dataclass
class DensityPlotData:
    """Density plot histogram data for visualization."""

    slice_index: int
    x_axis: DensityPlotAxis
    y_axis: DensityPlotAxis
    gridsize: int
    num_points: int
    colorbar_scale: str

    @classmethod
    def from_axes(
        cls,
        *,
        slice_index: int,
        x_axis: DensityPlotAxis,
        y_axis: DensityPlotAxis,
        gridsize: int,
        num_points: int,
        colorbar_scale: str,
    ) -> "DensityPlotData":
        return cls(
            slice_index=slice_index,
            x_axis=x_axis,
            y_axis=y_axis,
            gridsize=gridsize,
            num_points=num_points,
            colorbar_scale=colorbar_scale,
        )

    @property
    def x_min(self) -> float:
        return self.x_axis.min_value

    @property
    def x_max(self) -> float:
        return self.x_axis.max_value

    @property
    def y_min(self) -> float:
        return self.y_axis.min_value

    @property
    def y_max(self) -> float:
        return self.y_axis.max_value

    @property
    def x_scale(self) -> str:
        return self.x_axis.scale

    @property
    def y_scale(self) -> str:
        return self.y_axis.scale


class DensityPlotMeasurementPairExtractor(ABC, metaclass=AutoRegisterMeta):
    """Nominal policy for extracting X/Y measurement pairs by array dimensionality."""

    __registry_key__ = "ndim"
    __skip_if_no_key__ = True
    ndim: ClassVar[int | None] = None

    @classmethod
    def for_image(cls, image: np.ndarray) -> "DensityPlotMeasurementPairExtractor":
        extractor_type = cls.__registry__.get(
            int(image.ndim),
            ScalarDensityPlotMeasurementPairExtractor,
        )
        return extractor_type()

    @abstractmethod
    def extract(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned X/Y measurement arrays for histogramming."""


class SpatialStackDensityPlotMeasurementPairExtractor(DensityPlotMeasurementPairExtractor):
    """Extract measurement pairs from a two-channel spatial stack."""

    ndim = 3

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return image[0].ravel(), image[1].ravel()


class VectorStackDensityPlotMeasurementPairExtractor(DensityPlotMeasurementPairExtractor):
    """Extract measurement pairs from a two-row measurement matrix."""

    ndim = 2

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return image[0], image[1]


class ScalarDensityPlotMeasurementPairExtractor(DensityPlotMeasurementPairExtractor):
    """Extract a single measurement pair from scalar/vector input."""

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.array([image[0]]), np.array([image[1]])


@numpy
@special_outputs(("density_plot_data", csv_materializer(
    fields=["slice_index", "x_min", "x_max", "y_min", "y_max", "gridsize", 
            "num_points", "x_scale", "y_scale", "colorbar_scale"],
    analysis_type="density_plot"
)))
def display_density_plot(
    image: np.ndarray,
    gridsize: int = 100,
    x_scale: ScaleType = ScaleType.LINEAR,
    y_scale: ScaleType = ScaleType.LINEAR,
    colorbar_scale: ScaleType = ScaleType.LINEAR,
    colormap: ColorMap = ColorMap.JET,
    title: str = "",
) -> Tuple[np.ndarray, DensityPlotData]:
    """
    Compute 2D density histogram from two measurement arrays.
    
    This function takes two measurement arrays stacked along dimension 0
    and computes a 2D histogram (density plot) representation.
    
    Args:
        image: Shape (2, N) where image[0] contains X measurements and
               image[1] contains Y measurements. N is the number of objects.
        gridsize: Number of grid regions on each axis (1-1000). Higher values
                  increase resolution.
        x_scale: Scale for X-axis - linear or log (base 10).
        y_scale: Scale for Y-axis - linear or log (base 10).
        colorbar_scale: Scale for colorbar - linear or log (base 10).
        colormap: Colormap for the density plot visualization.
        title: Optional title for the plot.
    
    Returns:
        Tuple of:
        - 2D histogram array of shape (gridsize, gridsize) representing density
        - DensityPlotData with metadata about the plot
    """
    x_data, y_data = DensityPlotMeasurementPairExtractor.for_image(image).extract(image)
    
    # Remove NaN and infinite values
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    
    if len(x_data) == 0:
        # No valid data, return empty histogram
        histogram = np.zeros((gridsize, gridsize), dtype=np.float32)
        return histogram[np.newaxis, :, :], DensityPlotData.from_axes(
            slice_index=0,
            x_axis=DensityPlotAxis(0.0, 1.0, x_scale.value),
            y_axis=DensityPlotAxis(0.0, 1.0, y_scale.value),
            gridsize=gridsize,
            num_points=0,
            colorbar_scale=colorbar_scale.value
        )
    
    # Apply log transform if requested
    if x_scale == ScaleType.LOG:
        # Filter out non-positive values for log scale
        pos_mask = x_data > 0
        x_data = x_data[pos_mask]
        y_data = y_data[pos_mask]
        if len(x_data) > 0:
            x_data = np.log10(x_data)
    
    if y_scale == ScaleType.LOG:
        # Filter out non-positive values for log scale
        pos_mask = y_data > 0
        x_data = x_data[pos_mask]
        y_data = y_data[pos_mask]
        if len(y_data) > 0:
            y_data = np.log10(y_data)
    
    if len(x_data) == 0:
        # No valid data after log transform
        histogram = np.zeros((gridsize, gridsize), dtype=np.float32)
        return histogram[np.newaxis, :, :], DensityPlotData.from_axes(
            slice_index=0,
            x_axis=DensityPlotAxis(0.0, 1.0, x_scale.value),
            y_axis=DensityPlotAxis(0.0, 1.0, y_scale.value),
            gridsize=gridsize,
            num_points=0,
            colorbar_scale=colorbar_scale.value
        )
    
    # Compute data ranges
    x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
    y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
    
    # Handle edge case where min == max
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    
    # Compute 2D histogram
    histogram, x_edges, y_edges = np.histogram2d(
        x_data, y_data,
        bins=gridsize,
        range=[[x_min, x_max], [y_min, y_max]]
    )
    
    # Apply log transform to histogram counts if requested
    if colorbar_scale == ScaleType.LOG:
        # Add 1 to avoid log(0), then take log
        histogram = np.log10(histogram + 1)
    
    # Normalize to 0-1 range for visualization
    if histogram.max() > 0:
        histogram = histogram / histogram.max()
    
    histogram = histogram.astype(np.float32)
    
    # Return with batch dimension
    return histogram[np.newaxis, :, :], DensityPlotData.from_axes(
        slice_index=0,
        x_axis=DensityPlotAxis(x_min, x_max, x_scale.value),
        y_axis=DensityPlotAxis(y_min, y_max, y_scale.value),
        gridsize=gridsize,
        num_points=len(x_data),
        colorbar_scale=colorbar_scale.value
    )
