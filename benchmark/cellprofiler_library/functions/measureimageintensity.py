"""
Converted from CellProfiler: MeasureImageIntensity
Original: MeasureImageIntensity.measure

Measures several intensity features across an entire image (excluding masked pixels).
Measurements include: TotalIntensity, MeanIntensity, MedianIntensity, StdIntensity,
MADIntensity, MinIntensity, MaxIntensity, TotalArea, PercentMaximal,
LowerQuartileIntensity, UpperQuartileIntensity, and custom percentiles.
"""

import numpy as np
from typing import Tuple

from openhcs.core.memory import numpy
from openhcs.processing.backends.cellprofiler.intensity import (
    ImageIntensityMeasurement,
    ImageIntensityPercentileSpec,
)


@numpy
def measure_image_intensity(
    image: np.ndarray,
    calculate_percentiles: bool = False,
    percentiles: str = "10,90",
) -> Tuple[np.ndarray, ImageIntensityMeasurement]:
    """
    Measure intensity features across an entire image.

    Args:
        image: Input grayscale image (H, W)
        calculate_percentiles: Whether to calculate custom percentiles
        percentiles: Comma-separated list of percentiles to calculate (0-100)

    Returns:
        Tuple of (original image, intensity measurements)
    """
    measurements = ImageIntensityMeasurement.from_pixels(
        image.flatten(),
        percentile_spec=ImageIntensityPercentileSpec(
            enabled=calculate_percentiles,
            raw_percentiles=percentiles,
        ),
    )
    
    return image, measurements


@numpy
def measure_image_intensity_masked(
    image: np.ndarray,
    labels: np.ndarray,
    calculate_percentiles: bool = False,
    percentiles: str = "10,90",
) -> Tuple[np.ndarray, ImageIntensityMeasurement]:
    """
    Measure intensity features within labeled object regions.
    
    This measures aggregate intensity across ALL objects in the label image,
    not per-object measurements. For per-object measurements, use
    measure_object_intensity instead.
    
    Args:
        image: Input grayscale image (H, W)
        labels: Label image where non-zero pixels indicate object regions (H, W)
        calculate_percentiles: Whether to calculate custom percentiles
        percentiles: Comma-separated list of percentiles to calculate (0-100)
    
    Returns:
        Tuple of (original image, intensity measurements)
    """
    # Extract pixels within labeled regions
    mask = labels > 0
    measurements = ImageIntensityMeasurement.from_pixels(
        image[mask].flatten(),
        percentile_spec=ImageIntensityPercentileSpec(
            enabled=calculate_percentiles,
            raw_percentiles=percentiles,
        ),
    )
    
    return image, measurements
