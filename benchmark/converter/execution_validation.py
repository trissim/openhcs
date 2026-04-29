"""Validation for converted CellProfiler pipeline executions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionExpectation,
    RuntimeArtifactExecutionObservation,
    runtime_artifact_execution_failures,
)
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    artifact_kind_exports_as_table,
)

from benchmark.converter.runtime_pipeline import (
    DirectPipelineExecution,
    PreparedGeneratedPipeline,
)


class CPPipeInfrastructureFeature(Enum):
    """CellProfiler infrastructure behavior expected after conversion."""

    EXPORT_TO_SPREADSHEET = "ExportToSpreadsheet"
    SAVE_IMAGES = "SaveImages"


class CPPipeExecutionValidationError(RuntimeError):
    """Converted CellProfiler execution violated compiled expectations."""


@dataclass(frozen=True, slots=True)
class CPPipeExecutionValidation:
    """Successful validation result for converted CellProfiler execution."""

    expectation: RuntimeArtifactExecutionExpectation
    observation: RuntimeArtifactExecutionObservation


def validate_cppipe_execution(
    prepared: PreparedGeneratedPipeline,
    execution: DirectPipelineExecution,
    output_root: Path,
) -> CPPipeExecutionValidation:
    """Validate runtime artifacts and exports implied by a prepared .cppipe."""
    expectation = _runtime_expectation(prepared)
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
) -> RuntimeArtifactExecutionExpectation:
    output_specs = _output_specs(prepared)
    artifact_kinds = frozenset(spec.kind for spec in output_specs)
    return RuntimeArtifactExecutionExpectation.from_output_specs(
        output_specs,
        exports=_runtime_exports(
            _infrastructure_features(prepared),
            artifact_kinds,
        ),
    )


def _infrastructure_features(
    prepared: PreparedGeneratedPipeline,
) -> frozenset[CPPipeInfrastructureFeature]:
    module_names = {module.name for module in prepared.infrastructure_modules}
    return frozenset(
        feature
        for feature in CPPipeInfrastructureFeature
        if feature.value in module_names
    )


def _output_specs(
    prepared: PreparedGeneratedPipeline,
) -> tuple[ArtifactSpec, ...]:
    return tuple(
        spec
        for contract in prepared.generated_pipeline.artifact_contracts
        for spec in contract.outputs
    )


def _runtime_exports(
    infrastructure_features: frozenset[CPPipeInfrastructureFeature],
    artifact_kinds: frozenset[ArtifactKind],
) -> RuntimeExportExpectation:
    return RuntimeExportExpectation.from_flags(
        table_exports=(
            CPPipeInfrastructureFeature.EXPORT_TO_SPREADSHEET
            in infrastructure_features
        ),
        image_exports=CPPipeInfrastructureFeature.SAVE_IMAGES in infrastructure_features,
        table_artifact_kinds=frozenset(
            kind
            for kind in artifact_kinds
            if artifact_kind_exports_as_table(kind)
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
