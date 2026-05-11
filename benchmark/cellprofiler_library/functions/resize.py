"""Benchmark-library facade for CellProfiler Resize."""

from openhcs.processing.backends.cellprofiler.image_geometry import (
    InterpolationMethod,
    ResizeGeometry,
    ResizeMethod,
    resize,
    resize_volumetric,
)

__all__ = [
    "InterpolationMethod",
    "ResizeGeometry",
    "ResizeMethod",
    "resize",
    "resize_volumetric",
]
