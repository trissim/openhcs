"""
Converted from CellProfiler: CorrectIlluminationCalculate.

Compatibility facade for the OpenHCS CellProfiler backend implementation.
"""

from openhcs.processing.backends.cellprofiler.illumination import (
    CalculationScope,
    FilterSizeMethod,
    IlluminationStats,
    IntensityChoice,
    RescaleOption,
    SmoothingFilterSizeStrategy,
    SmoothingMethod,
    SmoothingPlaneStrategy,
    SplineBgMode,
    correct_illumination_calculate,
)

__all__ = [
    "CalculationScope",
    "FilterSizeMethod",
    "IlluminationStats",
    "IntensityChoice",
    "RescaleOption",
    "SmoothingFilterSizeStrategy",
    "SmoothingMethod",
    "SmoothingPlaneStrategy",
    "SplineBgMode",
    "correct_illumination_calculate",
]
