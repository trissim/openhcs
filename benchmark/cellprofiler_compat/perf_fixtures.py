"""Compatibility exports for CellProfiler runtime fixture capture."""

from openhcs.processing.backends.cellprofiler.perf_fixtures import (
    capture_array_fixture,
    capture_enabled,
)

__all__ = (
    "capture_array_fixture",
    "capture_enabled",
)
