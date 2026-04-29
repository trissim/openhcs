"""Typed roles for CellProfiler modules inside parsed .cppipe pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CPPipeModuleRole(str, Enum):
    """Runtime role for one parsed CellProfiler module."""

    INFRASTRUCTURE = "infrastructure"
    PROCESSING = "processing"


INFRASTRUCTURE_MODULE_NAMES = frozenset(
    {
        "LoadData",
        "LoadImages",
        "Images",
        "Metadata",
        "NamesAndTypes",
        "Groups",
        "SaveImages",
        "ExportToSpreadsheet",
    }
)


@dataclass(frozen=True, slots=True)
class CPPipeModuleRoleSpec:
    """Typed role classification for one CellProfiler module name."""

    module_name: str
    role: CPPipeModuleRole

    @property
    def is_infrastructure(self) -> bool:
        return self.role is CPPipeModuleRole.INFRASTRUCTURE


def cppipe_module_role(module_name: str) -> CPPipeModuleRoleSpec:
    """Classify one CellProfiler module name from parser output."""

    normalized_name = module_name.strip()
    if not normalized_name:
        raise ValueError("CellProfiler module name cannot be empty.")
    role = (
        CPPipeModuleRole.INFRASTRUCTURE
        if normalized_name in INFRASTRUCTURE_MODULE_NAMES
        else CPPipeModuleRole.PROCESSING
    )
    return CPPipeModuleRoleSpec(
        module_name=normalized_name,
        role=role,
    )
