"""Typed CellProfiler measurement target-scope controls."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType

from openhcs.core.runtime_semantics import MeasurementScope, MeasurementScopeSelection


CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG = (
    "_cellprofiler_measurement_target_scope"
)


class CellProfilerMeasurementTargetScope(str, Enum):
    """Closed measurement target scopes used by CellProfiler modules."""

    IMAGE = "image"
    OBJECT = "object"
    BOTH = "both"


def coerce_cellprofiler_measurement_target_scope(
    value: CellProfilerMeasurementTargetScope | str | None,
    default: CellProfilerMeasurementTargetScope,
) -> CellProfilerMeasurementTargetScope:
    """Coerce an invocation value into a closed measurement target scope."""
    if value is None:
        return default
    if isinstance(value, CellProfilerMeasurementTargetScope):
        return value
    return CellProfilerMeasurementTargetScope(str(value))


CELLPROFILER_MEASUREMENT_SCOPE_SELECTIONS = MappingProxyType(
    {
        CellProfilerMeasurementTargetScope.IMAGE: MeasurementScopeSelection.of(
            MeasurementScope.IMAGE,
        ),
        CellProfilerMeasurementTargetScope.OBJECT: MeasurementScopeSelection.of(
            MeasurementScope.OBJECT,
        ),
        CellProfilerMeasurementTargetScope.BOTH: MeasurementScopeSelection.of(
            MeasurementScope.IMAGE,
            MeasurementScope.OBJECT,
        ),
    }
)


def cellprofiler_measurement_scope_selection(
    value: CellProfilerMeasurementTargetScope | str | None,
    default: MeasurementScopeSelection,
) -> MeasurementScopeSelection:
    """Coerce a CellProfiler target-scope value into OpenHCS measurement scopes."""
    if value is None:
        return default
    target_scope = coerce_cellprofiler_measurement_target_scope(
        value,
        CellProfilerMeasurementTargetScope.BOTH,
    )
    return CELLPROFILER_MEASUREMENT_SCOPE_SELECTIONS[target_scope]
