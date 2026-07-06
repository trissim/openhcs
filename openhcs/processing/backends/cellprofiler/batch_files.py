"""CellProfiler batch-file infrastructure module declarations."""

from __future__ import annotations
from openhcs.interop.cellprofiler.module_declarations import (
    InfrastructureCellProfilerModule,
)


class CreateBatchFilesModule(InfrastructureCellProfilerModule):
    module_name = "CreateBatchFiles"
    function_name = "create_batch_files"
    validated = True
    contract = None
    confidence = 1.0
