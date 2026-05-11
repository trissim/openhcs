"""Benchmark-library facade for CellProfiler FillObjects."""

from openhcs.processing.backends.cellprofiler.morphology import (
    FillMode,
    fill_objects,
)

__all__ = [
    "FillMode",
    "fill_objects",
]
