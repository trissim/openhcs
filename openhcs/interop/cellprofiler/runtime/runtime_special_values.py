"""Shared CellProfiler runtime value type aliases."""

from __future__ import annotations

from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_values import (
    ImagePayloadMetadataInput,
    MeasurementTable,
    ObjectLabelData,
    ObjectLabelValue,
    ObjectRelationship,
    SpatialGrid,
)

CellProfilerSpecialInputValue = (
    ImagePayloadMetadataInput
    | ObjectLabelData
    | ObjectLabelValue
    | MeasurementTable
    | ObjectRelationship
    | SpatialGrid
    | str
    | int
    | float
    | bool
    | None
)
CellProfilerSpecialInputKwargs = dict[str, CellProfilerSpecialInputValue]
CellProfilerRuntimePlaneKwargValue = (
    RuntimeSliceAlignedValueSet | CellProfilerSpecialInputValue
)
