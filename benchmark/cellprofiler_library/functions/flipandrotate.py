"""Benchmark-library facade for CellProfiler FlipAndRotate."""

from openhcs.processing.backends.cellprofiler.image_geometry import (
    AlignmentDirection,
    FlipMethod,
    RotateMethod,
    RotationResult,
    flip_and_rotate,
)

__all__ = [
    "AlignmentDirection",
    "FlipMethod",
    "RotateMethod",
    "RotationResult",
    "flip_and_rotate",
]
