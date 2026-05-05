"""Typed CellProfiler module roles used during `.cppipe` import."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType


class CellProfilerModuleRole(Enum):
    """Semantic role for one parsed CellProfiler module."""

    INFRASTRUCTURE = auto()
    PROCESSING = auto()
    DISABLED = auto()


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
class CellProfilerModuleRoleSpec:
    """Typed role classification for one CellProfiler module name."""

    module_name: str
    role: CellProfilerModuleRole

    @property
    def is_infrastructure(self) -> bool:
        return self.role is CellProfilerModuleRole.INFRASTRUCTURE


def cellprofiler_module_role(module_name: str) -> CellProfilerModuleRoleSpec:
    """Classify one CellProfiler module name from parser output."""
    normalized_name = module_name.strip()
    if not normalized_name:
        raise ValueError("CellProfiler module name cannot be empty.")
    canonical_infrastructure_name = INFRASTRUCTURE_MODULE_NAMES_BY_KEY.get(
        normalized_name.casefold()
    )
    role = (
        CellProfilerModuleRole.INFRASTRUCTURE
        if canonical_infrastructure_name is not None
        else CellProfilerModuleRole.PROCESSING
    )
    return CellProfilerModuleRoleSpec(
        module_name=canonical_infrastructure_name or normalized_name,
        role=role,
    )
