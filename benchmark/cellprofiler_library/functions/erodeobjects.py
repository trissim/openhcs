"""Benchmark-library facade for CellProfiler ErodeObjects."""

from openhcs.processing.backends.cellprofiler.morphology import (
    ErosionStats,
    MidpointPreservationPolicy,
    SimpleDiskMidpointPreservationPolicy,
    erode_objects,
    log_function_runtime_profile,
    profile_function_runtime_enabled,
)

__all__ = [
    "ErosionStats",
    "MidpointPreservationPolicy",
    "SimpleDiskMidpointPreservationPolicy",
    "erode_objects",
    "log_function_runtime_profile",
    "profile_function_runtime_enabled",
]
