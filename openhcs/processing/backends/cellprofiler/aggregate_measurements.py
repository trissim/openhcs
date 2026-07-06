"""CellProfiler aggregate measurement module declarations."""

from __future__ import annotations
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.spreadsheet_export import (
    compute_aggregate_measurements,
)


class ComputeAggregateMeasurementsModule(CellProfilerModule):
    module_name = "ComputeAggregateMeasurements"
    function_name = "compute_aggregate_measurements"
    validated = True
    confidence = 0.0
