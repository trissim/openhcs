"""Benchmark-library facade for CellProfiler MeasureObjectOverlap semantics."""

from openhcs.processing.backends.cellprofiler.object_overlap import (
    DecimationMethod,
    OverlapMeasurements,
    measure_object_overlap,
)

__all__ = [
    "DecimationMethod",
    "OverlapMeasurements",
    "measure_object_overlap",
]
