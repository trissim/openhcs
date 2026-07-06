"""Benchmark-library facade for CellProfiler MeasureGranularity."""

from openhcs.processing.backends.cellprofiler.granularity import (
    GranularityMeasurement,
    ObjectGranularityMeasurement,
    measure_granularity,
    measure_granularity_objects,
)

__all__ = [
    "GranularityMeasurement",
    "ObjectGranularityMeasurement",
    "measure_granularity",
    "measure_granularity_objects",
]
