"""CellProfiler batch-file infrastructure module declarations."""

from __future__ import annotations

from openhcs.processing.backends.cellprofiler.module_classes import (
    InfrastructureCellProfilerModule,
)

class CreateBatchFilesModule(InfrastructureCellProfilerModule):
    module_name = 'CreateBatchFiles'
    function_name = 'create_batch_files'
    validated = True
    contract = 'unknown'
    confidence = 1.0
