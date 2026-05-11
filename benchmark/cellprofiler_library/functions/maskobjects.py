"""Converted from CellProfiler: MaskObjects.

Removes objects outside of a specified region or regions.
"""

from openhcs.interop.cellprofiler.mask_objects_settings import (
    MaskObjectsNumberingChoice as NumberingChoice,
    MaskObjectsOverlapHandling as OverlapHandling,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    MaskChoice,
    MaskObjectsOutputLabels,
    MaskObjectsPlaneOperation,
    MaskObjectsPlaneResult,
    MaskObjectsStats,
    mask_objects,
)

__all__ = [
    "MaskChoice",
    "MaskObjectsOutputLabels",
    "MaskObjectsPlaneOperation",
    "MaskObjectsPlaneResult",
    "MaskObjectsStats",
    "NumberingChoice",
    "OverlapHandling",
    "mask_objects",
]
