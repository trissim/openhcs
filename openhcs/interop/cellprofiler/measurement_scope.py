"""Typed CellProfiler measurement target-scope controls."""

from __future__ import annotations

from enum import Enum


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
    *,
    default: CellProfilerMeasurementTargetScope,
) -> CellProfilerMeasurementTargetScope:
    """Coerce an invocation value into a closed measurement target scope."""
    if value is None:
        return default
    if isinstance(value, CellProfilerMeasurementTargetScope):
        return value
    return CellProfilerMeasurementTargetScope(str(value))
