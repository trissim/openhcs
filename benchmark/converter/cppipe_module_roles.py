"""Compatibility aliases for CellProfiler `.cppipe` module roles."""

from openhcs.interop.cellprofiler.module_roles import (
    INFRASTRUCTURE_MODULE_NAMES,
    INFRASTRUCTURE_MODULE_NAMES_BY_KEY,
    CellProfilerModuleRole as CPPipeModuleRole,
    CellProfilerModuleRoleSpec as CPPipeModuleRoleSpec,
    cellprofiler_module_role as cppipe_module_role,
)

__all__ = (
    "CPPipeModuleRole",
    "CPPipeModuleRoleSpec",
    "INFRASTRUCTURE_MODULE_NAMES",
    "INFRASTRUCTURE_MODULE_NAMES_BY_KEY",
    "cppipe_module_role",
)
