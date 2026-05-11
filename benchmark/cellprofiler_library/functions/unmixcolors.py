"""Benchmark-library facade for CellProfiler UnmixColors."""

from openhcs.processing.backends.cellprofiler.color import (
    StainDefinition,
    StainType,
    unmix_colors,
)

__all__ = [
    "StainDefinition",
    "StainType",
    "unmix_colors",
]
