"""CellProfiler manual-interaction module declarations."""

from __future__ import annotations

from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule

class EditObjectsManuallyModule(CellProfilerModule):
    module_name = 'EditObjectsManually'
    function_name = 'edit_objects_manually'
    validated = True
    contract = 'unknown'
    confidence = 1.0


class IdentifyObjectsManuallyModule(CellProfilerModule):
    module_name = 'IdentifyObjectsManually'
    function_name = 'identify_objects_manually'
    validated = True
    contract = 'unknown'
    confidence = 1.0
