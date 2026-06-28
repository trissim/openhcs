"""CellProfiler aggregate measurement module declarations."""

from __future__ import annotations

from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule

class ComputeAggregateMeasurementsModule(CellProfilerModule):
    module_name = 'ComputeAggregateMeasurements'
    function_name = 'compute_aggregate_measurements'
    validated = True
    confidence = 0.0
