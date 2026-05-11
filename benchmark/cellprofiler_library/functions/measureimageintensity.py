"""
Converted from CellProfiler: MeasureImageIntensity
Original: MeasureImageIntensity.measure

Measures several intensity features across an entire image (excluding masked pixels).
Measurements include: TotalIntensity, MeanIntensity, MedianIntensity, StdIntensity,
MADIntensity, MinIntensity, MaxIntensity, TotalArea, PercentMaximal,
LowerQuartileIntensity, UpperQuartileIntensity, and custom percentiles.
"""

import json
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from openhcs.core.memory import numpy


@dataclass(frozen=True, slots=True)
class ImageIntensityPercentileSpec:
    """Percentile calculation policy for image-intensity rows."""

    enabled: bool = False
    raw_percentiles: str = "10,90"

    @property
    def values(self) -> List[int]:
        percentiles = []
        for percentile in self.raw_percentiles.replace(" ", "").split(","):
            if percentile == "":
                continue
            if percentile.isdigit() and 0 <= int(percentile) <= 100:
                percentiles.append(int(percentile))
        return sorted(set(percentiles))

    def measurements_for(self, pixels: np.ndarray) -> dict[int, float]:
        if not self.enabled:
            return {}
        parsed_percentiles = self.values
        if pixels.size == 0:
            return {percentile: 0.0 for percentile in parsed_percentiles}
        if not parsed_percentiles:
            return {}
        percentile_results = np.percentile(pixels, parsed_percentiles)
        return {
            percentile: float(value)
            for percentile, value in zip(parsed_percentiles, percentile_results)
        }


@dataclass
class ImageIntensityMeasurement:
    """Intensity measurements for an image or masked region."""
    slice_index: int
    total_intensity: float
    mean_intensity: float
    median_intensity: float
    std_intensity: float
    mad_intensity: float
    min_intensity: float
    max_intensity: float
    total_area: int
    percent_maximal: float
    lower_quartile_intensity: float
    upper_quartile_intensity: float
    percentile_values: str  # JSON-encoded dict of percentile -> value

    @classmethod
    def from_pixels(
        cls,
        pixels: np.ndarray,
        *,
        percentile_spec: ImageIntensityPercentileSpec,
    ) -> "ImageIntensityMeasurement":
        """Build the authoritative image-intensity measurement row."""
        pixels = pixels[np.isfinite(pixels)]
        pixel_count = pixels.size
        percentile_dict = percentile_spec.measurements_for(pixels)

        if pixel_count == 0:
            pixel_sum = 0.0
            pixel_mean = 0.0
            pixel_std = 0.0
            pixel_mad = 0.0
            pixel_median = 0.0
            pixel_min = 0.0
            pixel_max = 0.0
            pixel_pct_max = 0.0
            pixel_lower_qrt = 0.0
            pixel_upper_qrt = 0.0
        else:
            pixel_sum = float(np.sum(pixels))
            pixel_mean = pixel_sum / float(pixel_count)
            pixel_std = float(np.std(pixels))
            pixel_median = float(np.median(pixels))
            pixel_mad = float(np.median(np.abs(pixels - pixel_median)))
            pixel_min = float(np.min(pixels))
            pixel_max = float(np.max(pixels))
            pixel_pct_max = (
                100.0 * float(np.sum(pixels == pixel_max)) / float(pixel_count)
            )
            quartiles = np.percentile(pixels, [25, 75])
            pixel_lower_qrt = float(quartiles[0])
            pixel_upper_qrt = float(quartiles[1])

        return cls(
            slice_index=0,
            total_intensity=pixel_sum,
            mean_intensity=pixel_mean,
            median_intensity=pixel_median,
            std_intensity=pixel_std,
            mad_intensity=pixel_mad,
            min_intensity=pixel_min,
            max_intensity=pixel_max,
            total_area=int(pixel_count),
            percent_maximal=pixel_pct_max,
            lower_quartile_intensity=pixel_lower_qrt,
            upper_quartile_intensity=pixel_upper_qrt,
            percentile_values=json.dumps(percentile_dict),
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
