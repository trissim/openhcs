"""Typed CellProfiler module roles used during `.cppipe` import."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec

from .parser import ModuleBlock
from .source_schema import SetupModuleCompiler

if TYPE_CHECKING:
    from .symbol_table import ModuleArtifactContracts


class CellProfilerModuleRole(Enum):
    """Semantic role for one parsed CellProfiler module."""

    INFRASTRUCTURE = auto()
    PROCESSING = auto()
    DISABLED = auto()


@dataclass(frozen=True, slots=True)
class CellProfilerModuleRoleSpec:
    """Typed role classification for one CellProfiler module name."""

    module_name: str
    role: CellProfilerModuleRole

    @property
    def is_infrastructure(self) -> bool:
        return self.role is CellProfilerModuleRole.INFRASTRUCTURE


@dataclass(frozen=True)
class ArtifactSpecKey:
    """Scope-free artifact identity used while pruning generated CP steps."""

    kind: ArtifactKind
    name: str

    @classmethod
    def from_spec(cls, spec: ArtifactSpec) -> "ArtifactSpecKey":
        return cls(kind=spec.kind, name=spec.name)


def cellprofiler_module_role(module_name: str) -> CellProfilerModuleRoleSpec:
    """Classify one CellProfiler module name from parser output."""
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
        InfrastructureCellProfilerModule,
    )

    normalized_name = module_name.strip()
    if not normalized_name:
        raise ValueError("CellProfiler module name cannot be empty.")
    module_type = CellProfilerModule.for_module(normalized_name)
    canonical_module_name = (
        str(module_type.module_name)
        if module_type is not None
        else normalized_name
    )
    is_infrastructure = (
        module_type is not None
        and issubclass(module_type, InfrastructureCellProfilerModule)
    ) or SetupModuleCompiler.for_module(normalized_name) is not None
    role = (
        CellProfilerModuleRole.INFRASTRUCTURE
        if is_infrastructure
        else CellProfilerModuleRole.PROCESSING
    )
    return CellProfilerModuleRoleSpec(
        module_name=canonical_module_name,
        role=role,
    )

def cellprofiler_infrastructure_import_note(module_name: str) -> str:
    """Return the generated-source note for an OpenHCS-owned infrastructure module."""
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
    )

    module_type = CellProfilerModule.for_module(module_name)
    if (
        module_type is not None
        and module_type.infrastructure_import_note is not None
    ):
        return module_type.infrastructure_import_note
    role = cellprofiler_module_role(module_name)
    return f"{role.module_name} -> handled by OpenHCS infrastructure"


def cellprofiler_infrastructure_retained_artifacts(
    module: ModuleBlock,
    *,
    contracts_by_module_num: Mapping[int, "ModuleArtifactContracts"],
) -> frozenset[ArtifactSpecKey]:
    """Return artifacts retained by one OpenHCS-owned infrastructure module."""
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
    )

    module_type = CellProfilerModule.for_module(module.name)
    if module_type is None:
        return frozenset()
    return module_type.infrastructure_retained_artifacts(
        module,
        contracts_by_module_num=contracts_by_module_num,
    )
