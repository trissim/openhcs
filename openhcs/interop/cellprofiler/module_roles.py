"""Typed CellProfiler module roles used during `.cppipe` import."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from .module_semantics import (
    CELLPROFILER_MODULE_SEMANTICS,
    CellProfilerModuleCategory,
    cellprofiler_module_semantics,
)

class CellProfilerModuleRole(Enum):
    """Semantic role for one parsed CellProfiler module."""

    INFRASTRUCTURE = auto()
    PROCESSING = auto()
    DISABLED = auto()


INFRASTRUCTURE_MODULE_NAMES = frozenset(
    semantics.module_name
    for semantics in CELLPROFILER_MODULE_SEMANTICS.values()
    if semantics.category
    in {
        CellProfilerModuleCategory.INPUT,
        CellProfilerModuleCategory.FILE_PROCESSING,
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
    semantics = cellprofiler_module_semantics(normalized_name)
    canonical_infrastructure_name = None
    if semantics is not None and semantics.is_infrastructure:
        canonical_infrastructure_name = semantics.module_name
    role = (
        CellProfilerModuleRole.INFRASTRUCTURE
        if canonical_infrastructure_name is not None
        else CellProfilerModuleRole.PROCESSING
    )
    return CellProfilerModuleRoleSpec(
        module_name=canonical_infrastructure_name or normalized_name,
        role=role,
    )
