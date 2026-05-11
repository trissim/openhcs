"""Benchmark-library facade for CellProfiler DilateObjects."""

from openhcs.processing.backends.cellprofiler.morphology import (
    DilationStats,
    DilationStats3D,
    StructuringElement as StructuringElementShape,
    dilate_objects,
    dilate_objects_3d,
)

__all__ = [
    "DilationStats",
    "DilationStats3D",
    "StructuringElementShape",
    "dilate_objects",
    "dilate_objects_3d",
]
