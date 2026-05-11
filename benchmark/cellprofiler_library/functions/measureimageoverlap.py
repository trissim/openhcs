"""Benchmark-library facade for CellProfiler MeasureImageOverlap."""

from openhcs.processing.backends.cellprofiler.object_overlap import (
    DecimationMethod,
    ImageOverlapMeasurement,
    compute_image_earth_movers_distance,
    decimate_overlap_points,
    measureimageoverlap,
)

__all__ = [
    "DecimationMethod",
    "ImageOverlapMeasurement",
    "compute_image_earth_movers_distance",
    "decimate_overlap_points",
    "measureimageoverlap",
]
