"""Compatibility aliases for CellProfiler `.cppipe` module roles."""

from openhcs.interop.cellprofiler.module_roles import (
    CellProfilerModuleRole as CPPipeModuleRole,
    CellProfilerModuleRoleSpec as CPPipeModuleRoleSpec,
    cellprofiler_module_role as cppipe_module_role,
)

__all__ = (
    "CPPipeModuleRole",
    "CPPipeModuleRoleSpec",
    "cppipe_module_role",
)
