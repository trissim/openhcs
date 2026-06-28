"""CellProfiler CalculateStatistics module declaration."""

from __future__ import annotations

from openhcs.processing.backends.cellprofiler.module_classes import (
    CellProfilerModule,
    MeasurementDebugViewModule,
)


class CalculateStatisticsModule(MeasurementDebugViewModule, CellProfilerModule):
    module_name = 'CalculateStatistics'
    function_name = 'calculate_statistics'
    validated = True
    confidence = 1.0
