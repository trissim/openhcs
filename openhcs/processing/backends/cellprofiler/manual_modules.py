"""CellProfiler manual-interaction module declarations."""

from __future__ import annotations
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.edit_objects import edit_objects_manually
from openhcs.interop.cellprofiler.manual_objects import identify_objects_manually


class EditObjectsManuallyModule(CellProfilerModule):
    module_name = "EditObjectsManually"
    function_name = "edit_objects_manually"
    validated = True
    contract = None
    confidence = 1.0


class IdentifyObjectsManuallyModule(CellProfilerModule):
    module_name = "IdentifyObjectsManually"
    function_name = "identify_objects_manually"
    validated = True
    contract = None
    confidence = 1.0
