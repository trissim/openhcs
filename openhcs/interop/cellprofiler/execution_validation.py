"""Validation for converted CellProfiler pipeline executions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionExpectation,
    RuntimeArtifactExecutionObservation,
    runtime_artifact_execution_failures,
)
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    RuntimeImageExportSpec,
    artifact_kind_exports_as_table,
)
from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule

from openhcs.interop.cellprofiler.runtime_pipeline import (
    DirectPipelineExecution,
    PreparedGeneratedPipeline,
)
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock


class CPPipeExecutionValidationError(RuntimeError):
    """Converted CellProfiler execution violated compiled expectations."""


@dataclass(frozen=True, slots=True)
class CPPipeInfrastructureProfile:
    """External output semantics declared by CellProfiler infrastructure modules."""

    exports_tables: bool
    exports_images: bool
    image_export_specs: tuple[RuntimeImageExportSpec, ...]

    @classmethod
    def from_cppipe_path(
        cls,
        cppipe_path: Path,
        *,
        parser: CPPipeParser | None = None,
    ) -> "CPPipeInfrastructureProfile":
        return cls.from_modules((parser or CPPipeParser()).parse(cppipe_path))

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedGeneratedPipeline,
    ) -> "CPPipeInfrastructureProfile":
        return cls.from_modules(prepared.infrastructure_modules)

    @classmethod
    def from_modules(
        cls,
        modules: Iterable[ModuleBlock],
    ) -> "CPPipeInfrastructureProfile":
        module_tuple = tuple(modules)
        module_types = tuple(
            module_type
            for module in module_tuple
            for module_type in (CellProfilerModule.for_module(module.name),)
            if module_type is not None
        )
        return cls(
            exports_tables=any(
                module_type.infrastructure_exports_tables
                for module_type in module_types
            ),
            exports_images=any(
                module_type.infrastructure_exports_images
                for module_type in module_types
            ),
            image_export_specs=_image_export_specs(module_tuple),
        )

    @property
    def image_exports_without_table_exports(self) -> bool:
        return bool(self.image_export_specs) and not self.exports_tables


@dataclass(frozen=True, slots=True)
class CPPipeExecutionValidation:
    """Successful validation result for converted CellProfiler execution."""

    expectation: RuntimeArtifactExecutionExpectation
    observation: RuntimeArtifactExecutionObservation


def validate_cppipe_execution(
    prepared: PreparedGeneratedPipeline,
    execution: DirectPipelineExecution,
    output_root: Path,
    *,
    validate_table_exports: bool = True,
    validate_image_exports: bool = True,
) -> CPPipeExecutionValidation:
    """Validate runtime artifacts and exports implied by a prepared .cppipe."""
    expectation = _runtime_expectation(
        prepared,
        validate_table_exports=validate_table_exports,
        validate_image_exports=validate_image_exports,
    )
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        execution.compiled_contexts,
        output_root,
    )
    failures = (
        *_execution_failures(execution),
        *runtime_artifact_execution_failures(expectation, observation),
    )
    if failures:
        raise CPPipeExecutionValidationError(
            "Converted CellProfiler pipeline violated compiled expectations:\n"
            + "\n".join(f"- {failure}" for failure in failures)
        )
    return CPPipeExecutionValidation(
        expectation=expectation,
        observation=observation,
    )


def _runtime_expectation(
    prepared: PreparedGeneratedPipeline,
    *,
    validate_table_exports: bool = True,
    validate_image_exports: bool = True,
) -> RuntimeArtifactExecutionExpectation:
    output_specs = _output_specs(prepared)
    artifact_kinds = frozenset(spec.kind for spec in output_specs)
    infrastructure = CPPipeInfrastructureProfile.from_prepared(prepared)
    return RuntimeArtifactExecutionExpectation.from_output_specs(
        output_specs,
        exports=_runtime_exports(
            infrastructure,
            artifact_kinds,
            validate_table_exports=validate_table_exports,
            validate_image_exports=validate_image_exports,
        ),
    )


def _output_specs(
    prepared: PreparedGeneratedPipeline,
) -> tuple[ArtifactSpec, ...]:
    return tuple(
        spec
        for contract in prepared.generated_pipeline.artifact_contracts
        for spec in contract.outputs
    )


def _image_export_specs(
    modules: Iterable[ModuleBlock],
) -> tuple[RuntimeImageExportSpec, ...]:
    return tuple(
        spec
        for module in modules
        for module_type in (CellProfilerModule.for_module(module.name),)
        if module_type is not None
        for spec in module_type.image_export_specs(module)
    )


def _runtime_exports(
    infrastructure: CPPipeInfrastructureProfile,
    artifact_kinds: frozenset[ArtifactKind],
    *,
    validate_table_exports: bool = True,
    validate_image_exports: bool = True,
) -> RuntimeExportExpectation:
    return RuntimeExportExpectation.from_flags(
        table_exports=(
            validate_table_exports
            and infrastructure.exports_tables
        ),
        image_exports=(
            validate_image_exports
            and infrastructure.exports_images
        ),
        table_artifact_kinds=frozenset(
            kind
            for kind in artifact_kinds
            if artifact_kind_exports_as_table(kind)
        ),
        image_export_specs=(
            infrastructure.image_export_specs if validate_image_exports else ()
        ),
    )

def _execution_failures(
    execution: DirectPipelineExecution,
) -> tuple[str, ...]:
    unsuccessful_results = {
        axis: result
        for axis, result in execution.execution_results.items()
        if not result.is_success()
    }
    if not unsuccessful_results:
        return ()
    return (f"unsuccessful execution results: {unsuccessful_results!r}",)
