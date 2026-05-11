"""
Converted from CellProfiler: IdentifyTertiaryObjects.

Compatibility facade for the OpenHCS CellProfiler backend implementation.
"""

from openhcs.processing.backends.cellprofiler.secondary import (
    TertiaryObjectInputs,
    TertiaryObjectLabelOutput,
    TertiaryObjectMeasurement,
    TertiaryObjectSegmentation,
    TertiaryObjectStats,
    _identify_tertiary_objects_batch,
    identify_tertiary_objects,
)

__all__ = [
    "TertiaryObjectInputs",
    "TertiaryObjectLabelOutput",
    "TertiaryObjectMeasurement",
    "TertiaryObjectSegmentation",
    "TertiaryObjectStats",
    "_identify_tertiary_objects_batch",
    "identify_tertiary_objects",
]
