"""Compatibility aliases for CellProfiler `.cppipe` parsing."""

from openhcs.interop.cellprofiler.parser import (
    CPPipeParser,
    ModuleBlock,
    ModuleSetting,
)

__all__ = ("CPPipeParser", "ModuleBlock", "ModuleSetting")
