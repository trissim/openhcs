"""Module-declared CellProfiler artifact input semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.processing.backends.cellprofiler.library import (
    canonical_module_name,
    get_function,
)


@dataclass(frozen=True, slots=True)
class ModuleArtifactInput:
    """One module-declared CellProfiler artifact input name."""

    name: str
    kind: ArtifactKind

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("ModuleArtifactInput.name cannot be empty.")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "kind", ArtifactKind(self.kind))


class ModuleArtifactInputProvider(ABC, metaclass=AutoRegisterMeta):
    """Nominal hook for module-specific artifact input declarations."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True

    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "ModuleArtifactInputProvider | None":
        canonical_name = canonical_module_name(module_name)
        provider_type = cls.__registry__.get(canonical_name)
        if provider_type is None:
            get_function(canonical_name)
            provider_type = cls.__registry__.get(canonical_name)
        return None if provider_type is None else provider_type()

    @abstractmethod
    def inputs(
        self,
        module: ModuleBlock,
        source_schema: PipelineImageSchema,
    ) -> tuple[ModuleArtifactInput, ...]:
        """Return artifact inputs declared by module-specific semantics."""


def module_declared_artifact_inputs(
    module: ModuleBlock,
    source_schema: PipelineImageSchema,
) -> tuple[ModuleArtifactInput, ...]:
    """Return module-declared artifact inputs for one parsed module."""

    provider = ModuleArtifactInputProvider.for_module(module.name)
    if provider is None:
        return ()
    return provider.inputs(module, source_schema)
