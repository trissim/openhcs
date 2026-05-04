"""Compatibility import path for CellProfiler measurement target scopes."""

from openhcs.interop.cellprofiler.measurement_scope import (
    CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
    CellProfilerMeasurementTargetScope,
    coerce_cellprofiler_measurement_target_scope,
)

__all__ = (
    "CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG",
    "CellProfilerMeasurementTargetScope",
    "coerce_cellprofiler_measurement_target_scope",
)

