"""Typed CellProfiler measurement target-scope controls."""

from __future__ import annotations

from enum import Enum

from openhcs.core.runtime_semantics import MeasurementScope, MeasurementScopeSelection


class CellProfilerMeasurementTargetScope(str, Enum):
    """Closed measurement target scopes used by CellProfiler modules."""

    IMAGE = "image"
    OBJECT = "object"
    BOTH = "both"

    @property
    def measurement_scope_selection(self) -> MeasurementScopeSelection:
        """Return the OpenHCS measurement scopes represented by this target."""
        match self:
            case CellProfilerMeasurementTargetScope.IMAGE:
                return MeasurementScopeSelection.of(MeasurementScope.IMAGE)
            case CellProfilerMeasurementTargetScope.OBJECT:
                return MeasurementScopeSelection.of(MeasurementScope.OBJECT)
            case CellProfilerMeasurementTargetScope.BOTH:
                return MeasurementScopeSelection.of(
                    MeasurementScope.IMAGE,
                    MeasurementScope.OBJECT,
                )
        raise TypeError(f"Unsupported CellProfiler measurement target {self!r}.")


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
    return target_scope.measurement_scope_selection
