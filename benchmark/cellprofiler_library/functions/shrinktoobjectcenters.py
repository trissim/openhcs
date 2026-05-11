"""Benchmark-library facade for CellProfiler ShrinkToObjectCenters."""

from openhcs.processing.backends.cellprofiler.morphology import (
    CentroidStats,
    shrink_to_object_centers,
    shrink_to_object_centers_3d,
)

__all__ = [
    "CentroidStats",
    "shrink_to_object_centers",
    "shrink_to_object_centers_3d",
]
