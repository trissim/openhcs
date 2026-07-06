"""CellProfiler CalculateStatistics module declaration."""

from __future__ import annotations
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.statistics import calculate_statistics


class CalculateStatisticsModule(CellProfilerModule):
    module_name = "CalculateStatistics"
    function_name = "calculate_statistics"
    validated = True
    confidence = 1.0
