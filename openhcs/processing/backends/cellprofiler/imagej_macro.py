"""CellProfiler ImageJ macro module declaration."""

from __future__ import annotations
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    CellProfilerModule,
)


class RunImagejMacroModule(CellProfilerModule):
    module_name = "RunImagejMacro"
    function_name = "run_imagej_macro"
    validated = True
    contract = ProcessingContract.FLEXIBLE
    confidence = 0.95
