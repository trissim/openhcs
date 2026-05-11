"""Benchmark-library facade for CellProfiler MeasureGranularity."""

from openhcs.processing.backends.cellprofiler.granularity import (
    GRANULARITY_FIELDS,
    GranularityMeasurement,
    ObjectGranularityMeasurement,
    measure_granularity,
    measure_granularity_objects,
)

__all__ = [
    "GRANULARITY_FIELDS",
    "GranularityMeasurement",
    "ObjectGranularityMeasurement",
    "measure_granularity",
    "measure_granularity_objects",
]
