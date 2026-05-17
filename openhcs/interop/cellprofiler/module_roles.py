"""Typed CellProfiler module roles used during `.cppipe` import."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

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


class CellProfilerInfrastructureImportNote(metaclass=AutoRegisterMeta):
    """Auto-registered note for OpenHCS-owned infrastructure module handling."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True

    module_name: ClassVar[str | None] = None
    note_text: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "CellProfilerInfrastructureImportNote":
        role = cellprofiler_module_role(module_name)
        note_type = cls.__registry__.get(role.module_name)
        if note_type is None:
            return DefaultInfrastructureImportNote(role.module_name)
        return note_type()

    @property
    def text(self) -> str:
        """Return the generated-source note for this infrastructure module."""
        return self.note_text or ""


@dataclass(frozen=True, slots=True)
class DefaultInfrastructureImportNote(CellProfilerInfrastructureImportNote):
    """Default generated-source note for infrastructure modules."""

    note_text: str


class LoadDataInfrastructureImportNote(CellProfilerInfrastructureImportNote):
    """Declare OpenHCS source metadata handling for LoadData."""

    module_name = "LoadData"
    note_text = "LoadData -> handled by plate_path + openhcs_metadata.json"


class ExportToSpreadsheetInfrastructureImportNote(CellProfilerInfrastructureImportNote):
    """Declare OpenHCS table materialization handling for ExportToSpreadsheet."""

    module_name = "ExportToSpreadsheet"
    note_text = "ExportToSpreadsheet -> handled by @special_outputs(csv_materializer(...))"


def cellprofiler_infrastructure_import_note(module_name: str) -> str:
    """Return the generated-source note for an OpenHCS-owned infrastructure module."""
    return CellProfilerInfrastructureImportNote.for_module(module_name).text
