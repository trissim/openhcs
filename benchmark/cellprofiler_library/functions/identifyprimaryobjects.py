"""Benchmark-library facade for CellProfiler IdentifyPrimaryObjects semantics."""

from openhcs.processing.backends.cellprofiler.primary_objects import (
    ExcessObjectHandling,
    FillHolesOption,
    PrimaryObjectStats,
    UnclumpMethod,
    WatershedMethod,
    identify_primary_objects,
)

__all__ = [
    "ExcessObjectHandling",
    "FillHolesOption",
    "PrimaryObjectStats",
    "UnclumpMethod",
    "WatershedMethod",
    "identify_primary_objects",
]
