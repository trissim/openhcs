"""Typed records for importing CellProfiler pipelines into OpenHCS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.pipeline import Pipeline
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.interop.cellprofiler.module_roles import CellProfilerModuleRole


@dataclass(frozen=True, slots=True)
class CellProfilerModuleReference:
    """CellProfiler module identity preserved from the source pipeline."""

    name: str
    module_num: int
    role: CellProfilerModuleRole

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CellProfilerModuleReference.name cannot be empty.")
        if self.module_num < 1:
            raise ValueError(
                "CellProfilerModuleReference.module_num must be one-based."
            )
        object.__setattr__(self, "role", CellProfilerModuleRole(self.role))


@dataclass(frozen=True, slots=True)
class CellProfilerPipelineProvenance:
    """Source-level provenance for a `.cppipe` imported as an OpenHCS pipeline."""

    cppipe_path: Path
    modules: tuple[CellProfilerModuleReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cppipe_path", Path(self.cppipe_path))
        object.__setattr__(self, "modules", tuple(self.modules))
        if not self.modules:
            raise ValueError("CellProfilerPipelineProvenance.modules cannot be empty.")
        for module in self.modules:
            if not isinstance(module, CellProfilerModuleReference):
                raise TypeError(
                    "CellProfilerPipelineProvenance.modules must contain "
                    f"CellProfilerModuleReference values, got {type(module).__name__}."
                )

    @property
    def processing_modules(self) -> tuple[CellProfilerModuleReference, ...]:
        """Modules converted into executable OpenHCS steps."""
        return self.modules_with_role(CellProfilerModuleRole.PROCESSING)

    @property
    def infrastructure_modules(self) -> tuple[CellProfilerModuleReference, ...]:
        """Modules whose semantics are handled by import/runtime infrastructure."""
        return self.modules_with_role(CellProfilerModuleRole.INFRASTRUCTURE)

    def modules_with_role(
        self,
        role: CellProfilerModuleRole,
    ) -> tuple[CellProfilerModuleReference, ...]:
        """Return modules with the requested typed import role."""
        normalized_role = CellProfilerModuleRole(role)
        return tuple(module for module in self.modules if module.role is normalized_role)


@dataclass(frozen=True, slots=True)
class CellProfilerPipelineImportResult:
    """Product-facing result of importing a `.cppipe` into OpenHCS."""

    provenance: CellProfilerPipelineProvenance
    pipeline: Pipeline
    source_schema: PipelineImageSchema
    generated_source: str
    generated_module_name: str
    generated_module_path: Path
    artifact_contracts: tuple[ModuleArtifactContract, ...] = ()
    registered_functions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, CellProfilerPipelineProvenance):
            raise TypeError(
                "CellProfilerPipelineImportResult.provenance must be "
                "CellProfilerPipelineProvenance."
            )
        if not isinstance(self.pipeline, Pipeline):
            raise TypeError(
                "CellProfilerPipelineImportResult.pipeline must be Pipeline, "
                f"got {type(self.pipeline).__name__}."
            )
        if not isinstance(self.source_schema, PipelineImageSchema):
            raise TypeError(
                "CellProfilerPipelineImportResult.source_schema must be "
                f"PipelineImageSchema, got {type(self.source_schema).__name__}."
            )
        object.__setattr__(
            self,
            "generated_module_path",
            Path(self.generated_module_path),
        )
        object.__setattr__(self, "artifact_contracts", tuple(self.artifact_contracts))
        object.__setattr__(
            self,
            "registered_functions",
            tuple(self.registered_functions),
        )
        for contract in self.artifact_contracts:
            if not isinstance(contract, ModuleArtifactContract):
                raise TypeError(
                    "CellProfilerPipelineImportResult.artifact_contracts must contain "
                    f"ModuleArtifactContract values, got {type(contract).__name__}."
                )
        if not self.generated_source:
            raise ValueError(
                "CellProfilerPipelineImportResult.generated_source must be "
                "a non-empty string."
            )
        if not self.generated_module_name:
            raise ValueError(
                "CellProfilerPipelineImportResult.generated_module_name must be "
                "a non-empty string."
            )
