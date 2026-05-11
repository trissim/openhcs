"""Benchmark-library facade for CellProfiler RescaleIntensity."""

from openhcs.processing.backends.cellprofiler.intensity import (
    AutomaticHigh,
    AutomaticLow,
    RescaleMethod,
    rescale_intensity,
    rescale_intensity_match_maximum,
    rescale_source_range,
)

__all__ = [
    "AutomaticHigh",
    "AutomaticLow",
    "RescaleMethod",
    "rescale_intensity",
    "rescale_intensity_match_maximum",
    "rescale_source_range",
]
