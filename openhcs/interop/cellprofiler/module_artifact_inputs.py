"""Module-declared CellProfiler artifact input semantics."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.interop.cellprofiler.parser import ModuleBlock


@dataclass(slots=True)
class ModuleArtifactInput:
    """One module-declared CellProfiler artifact input name."""

    name: str
    kind: ArtifactKind

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("ModuleArtifactInput.name cannot be empty.")
        self.name = normalized_name
        self.kind = ArtifactKind(self.kind)


def module_declared_artifact_inputs(
    module: ModuleBlock,
    source_schema: PipelineImageSchema,
) -> tuple[ModuleArtifactInput, ...]:
    """Return module-declared artifact inputs for one parsed module."""
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
    )

    module_type = CellProfilerModule.for_module(module.name)
    if module_type is not None:
        return module_type.artifact_inputs(module, source_schema)
    return ()
