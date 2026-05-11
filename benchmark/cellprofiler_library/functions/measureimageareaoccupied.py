"""
Converted from CellProfiler: MeasureImageAreaOccupied.

Compatibility facade for the OpenHCS CellProfiler backend implementation.
"""

from openhcs.processing.backends.cellprofiler.area_occupied import (
    AreaOccupiedMeasurement,
    AreaOccupiedRuntimeRow,
    BinaryAreaOccupiedRequest,
    ObjectLabelsAreaOccupiedRequest,
    OperandChoice,
    SurfaceAreaRequest,
    VolumeOccupiedMeasurement,
    VolumeOccupiedRequest,
    label_area_and_perimeter,
    measure_image_area_occupied,
    measure_image_area_occupied_binary,
    measure_image_area_occupied_objects,
    measure_image_volume_occupied_binary,
    measure_image_volume_occupied_objects,
)

__all__ = [
    "AreaOccupiedMeasurement",
    "AreaOccupiedRuntimeRow",
    "BinaryAreaOccupiedRequest",
    "ObjectLabelsAreaOccupiedRequest",
    "OperandChoice",
    "SurfaceAreaRequest",
    "VolumeOccupiedMeasurement",
    "VolumeOccupiedRequest",
    "label_area_and_perimeter",
    "measure_image_area_occupied",
    "measure_image_area_occupied_binary",
    "measure_image_area_occupied_objects",
    "measure_image_volume_occupied_binary",
    "measure_image_volume_occupied_objects",
]
