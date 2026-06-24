"""Runtime payload type aliases shared by the CellProfiler adapter."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Protocol, TypeAlias

import numpy as np

from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from openhcs.core.runtime_values import (
    ColumnarRows,
    ImagePayloadMetadataInput,
    MeasurementTable,
    NamedImage,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectRelationship,
    RuntimeArrayData,
    RuntimeArrayPayload,
    RuntimeSliceAlignedValueSet,
    RuntimeValue,
    SparseIJVLabelRows,
    SpatialGrid,
)
from openhcs.core.runtime_semantics import ParentChildRelationshipPayload


CellProfilerRuntimePrimitive: TypeAlias = str | int | float | bool | None
CellProfilerRuntimeMapping: TypeAlias = Mapping[str, "CellProfilerRuntimeValue"]
CellProfilerRuntimeSequence: TypeAlias = Sequence["CellProfilerRuntimeValue"]
CellProfilerRuntimeValue: TypeAlias = (
    RuntimeArrayPayload
    | np.ndarray
    | ObjectLabelSet
    | ObjectLabelPayload
    | SparseIJVLabelRows
    | MeasurementTable
    | ObjectRelationship
    | ParentChildRelationshipPayload
    | RuntimeSliceAlignedValueSet
    | CellProfilerRuntimePrimitive
    | CellProfilerRuntimeMapping
    | CellProfilerRuntimeSequence
)


class CellProfilerFunction(Protocol):
    """Callable surface for absorbed CellProfiler-compatible functions."""

    def __call__(self, *args: object, **kwargs: object) -> CellProfilerRuntimeValue:
        """Execute the absorbed function and return its runtime value."""
        ...


CellProfilerOptionalFunction: TypeAlias = CellProfilerFunction | None
CellProfilerKwargs: TypeAlias = Mapping[str, CellProfilerRuntimeValue]
CellProfilerKwargDict: TypeAlias = dict[str, CellProfilerRuntimeValue]
CellProfilerProfileFields: TypeAlias = tuple[tuple[str, CellProfilerRuntimeValue], ...]
CellProfilerRuntimeValues: TypeAlias = tuple[CellProfilerRuntimeValue, ...]
CellProfilerRuntimeValueSequence: TypeAlias = Sequence[CellProfilerRuntimeValue]
CellProfilerRuntimeType: TypeAlias = type[CellProfilerRuntimeValue]
CellProfilerRuntimeTypeOrNone: TypeAlias = CellProfilerRuntimeType | None
CellProfilerClassAttributes: TypeAlias = Mapping[str, CellProfilerRuntimeValue]
CellProfilerClassAttributeDict: TypeAlias = dict[str, CellProfilerRuntimeValue]
CellProfilerMutableClassNamespace: TypeAlias = MutableMapping[
    str,
    CellProfilerRuntimeValue,
]


CellProfilerMeasurementCellValue: TypeAlias = str | int | float | bool | None
CellProfilerMeasurementVector: TypeAlias = (
    np.ndarray | Sequence[CellProfilerMeasurementCellValue]
)
MeasurementRowMapping: TypeAlias = Mapping[str, CellProfilerRuntimeValue]
MeasurementObjectName: TypeAlias = str | None | CellProfilerRuntimeValue
MissingObjectMeasurementCellValue: TypeAlias = float | CellProfilerRuntimeValue
MeasurementRowsInput: TypeAlias = (
    ColumnarRows
    | Sequence[Mapping[str, CellProfilerMeasurementCellValue]]
)
MeasurementJsonRow: TypeAlias = CellProfilerKwargDict
MeasurementJsonRows: TypeAlias = list[MeasurementJsonRow]


ImagePayloadValue = RuntimeArrayData
ImagePayloadMaskValue = ImagePayloadValue | None
CellProfilerFilePayload = (
    ImagePayloadValue
    | ObjectLabelPayload
    | ObjectLabelSet
    | MeasurementTable
    | ObjectRelationship
    | SpatialGrid
    | SparseIJVLabelRows
    | ColumnarRows
)
RuntimeArtifactPayloadValue = (
    CellProfilerFilePayload
    | NamedImage
    | ImagePayloadMetadataInput
    | ObjectLabelSet
    | ObjectRelationship
    | SpatialGrid
    | RuntimeSliceAlignedValues[ImagePayloadValue]
    | RuntimeSliceAlignedValues[SpatialGrid]
)
RuntimeArtifactNormalizationInput = (
    RuntimeArtifactPayloadValue | MeasurementRowsInput | RuntimeValue
)
DenseLabelPayload = (
    ObjectLabelSet
    | ObjectLabelPayload
    | SparseIJVLabelRows
    | np.ndarray
    | Sequence[int]
    | Sequence[np.ndarray]
)
