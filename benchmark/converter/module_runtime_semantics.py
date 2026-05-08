"""Typed runtime-semantics selection for CellProfiler module revisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from benchmark.cellprofiler_library import canonical_module_name
from benchmark.cellprofiler_library.functions.watershed import WatershedRuntimeFamily

from .parser import ModuleBlock


@dataclass(frozen=True, slots=True)
class ModuleRevisionRange:
    """Closed revision interval for CellProfiler module schema semantics."""

    maximum: int | None = None

    def contains(self, module: ModuleBlock) -> bool:
        revision = module.variable_revision_number
        if revision is None:
            return self.maximum is None
        return self.maximum is None or revision <= self.maximum


class ModuleRuntimeSemanticsBinding(ABC, metaclass=AutoRegisterMeta):
    """Nominal hook for module-revision-specific runtime kwargs."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "ModuleRuntimeSemanticsBinding | None":
        binding_type = cls.__registry__.get(canonical_module_name(module_name))
        return None if binding_type is None else binding_type()

    @abstractmethod
    def kwargs(self, module: ModuleBlock) -> Mapping[str, Any]:
        """Return runtime-selection kwargs for one parsed module."""


class WatershedRuntimeSemanticsBinding(ModuleRuntimeSemanticsBinding):
    """Select CP4 module semantics for legacy Watershed revisions."""

    module_name = "Watershed"
    cellprofiler4_revisions = ModuleRevisionRange(maximum=3)

    def kwargs(self, module: ModuleBlock) -> Mapping[str, Any]:
        runtime_family = (
            WatershedRuntimeFamily.CELLPROFILER4
            if type(self).cellprofiler4_revisions.contains(module)
            else WatershedRuntimeFamily.LIBRARY
        )
        return {"runtime_family": runtime_family.value}
