"""CellProfiler LabelImages module declaration."""

from __future__ import annotations
from openhcs.interop.cellprofiler.module_declarations import (
    InfrastructureCellProfilerModule,
)


class LabelImagesModule(InfrastructureCellProfilerModule):
    module_name = "LabelImages"
    function_name = "label_images"
    validated = True
    confidence = 1.0
