"""Typed roles for CellProfiler modules inside parsed .cppipe pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


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
INFRASTRUCTURE_MODULE_NAMES_BY_KEY = MappingProxyType(
    {module_name.casefold(): module_name for module_name in INFRASTRUCTURE_MODULE_NAMES}
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
    canonical_infrastructure_name = INFRASTRUCTURE_MODULE_NAMES_BY_KEY.get(
        normalized_name.casefold()
    )
    role = (
        CPPipeModuleRole.INFRASTRUCTURE
        if canonical_infrastructure_name is not None
        else CPPipeModuleRole.PROCESSING
    )
    return CPPipeModuleRoleSpec(
        module_name=canonical_infrastructure_name or normalized_name,
        role=role,
    )
