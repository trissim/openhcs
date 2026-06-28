"""CellProfiler ImageJ macro module declaration."""

from __future__ import annotations

from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule

class RunImagejMacroModule(CellProfilerModule):
    module_name = 'RunImagejMacro'
    function_name = 'run_imagej_macro'
    validated = True
    contract = 'flexible'
    confidence = 0.95
