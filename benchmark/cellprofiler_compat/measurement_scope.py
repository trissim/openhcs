"""Compatibility import path for CellProfiler measurement target scopes."""

from openhcs.interop.cellprofiler.measurement_scope import (
    CellProfilerMeasurementTargetScope,
    coerce_cellprofiler_measurement_target_scope,
)

__all__ = (
    "CellProfilerMeasurementTargetScope",
    "coerce_cellprofiler_measurement_target_scope",
)
