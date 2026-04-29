"""Validation for converted CellProfiler pipeline executions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    RuntimeExportObservation,
    artifact_kind_exports_as_table,
    runtime_export_failures,
)
from openhcs.core.runtime_stores import (
    StoredRuntimeValue,
    require_runtime_value_store,
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
class CPPipeExecutionExpectation:
    """Compiled runtime/export expectations for one prepared .cppipe."""

    infrastructure_features: frozenset[CPPipeInfrastructureFeature]
    runtime_artifact_kinds: frozenset[ArtifactKind]
    runtime_exports: RuntimeExportExpectation

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedGeneratedPipeline,
    ) -> "CPPipeExecutionExpectation":
        infrastructure_features = _infrastructure_features(prepared)
        runtime_artifact_kinds = _runtime_artifact_kinds(prepared)
        return cls(
            infrastructure_features=infrastructure_features,
            runtime_artifact_kinds=runtime_artifact_kinds,
            runtime_exports=_runtime_exports(
                infrastructure_features,
                runtime_artifact_kinds,
            ),
        )

    @property
    def expects_csv_exports(self) -> bool:
        return self.runtime_exports.expects_table_files

    @property
    def expects_image_exports(self) -> bool:
        return self.runtime_exports.expects_image_files


@dataclass(frozen=True, slots=True)
class CPPipeExecutionObservation:
    """Observed runtime/export outputs from one converted .cppipe execution."""

    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]]
    runtime_exports: RuntimeExportObservation

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_records_by_axis",
            MappingProxyType(
                {
                    axis: tuple(records)
                    for axis, records in self.runtime_records_by_axis.items()
                }
            ),
        )
        if not isinstance(self.runtime_exports, RuntimeExportObservation):
            raise TypeError(
                "CPPipeExecutionObservation.runtime_exports must be "
                f"RuntimeExportObservation, got "
                f"{type(self.runtime_exports).__name__}."
            )

    @property
    def runtime_record_counts_by_axis(self) -> Mapping[str, Mapping[ArtifactKind, int]]:
        return MappingProxyType(
            {
                axis: MappingProxyType(Counter(record.key.kind for record in records))
                for axis, records in self.runtime_records_by_axis.items()
            }
        )

    @property
    def csv_outputs(self) -> tuple[Path, ...]:
        return self.runtime_exports.table_outputs

    @property
    def image_outputs(self) -> tuple[Path, ...]:
        return self.runtime_exports.image_outputs

    @property
    def csv_headers_by_path(self) -> Mapping[Path, tuple[str, ...]]:
        return self.runtime_exports.table_headers_by_path

    @property
    def csv_row_counts_by_path(self) -> Mapping[Path, int]:
        return self.runtime_exports.table_row_counts_by_path


@dataclass(frozen=True, slots=True)
class CPPipeExecutionValidation:
    """Successful validation result for converted CellProfiler execution."""

    expectation: CPPipeExecutionExpectation
    observation: CPPipeExecutionObservation


def validate_cppipe_execution(
    prepared: PreparedGeneratedPipeline,
    execution: DirectPipelineExecution,
    output_root: Path,
) -> CPPipeExecutionValidation:
    """Validate runtime artifacts and exports implied by a prepared .cppipe."""
    expectation = CPPipeExecutionExpectation.from_prepared(prepared)
    observation = CPPipeExecutionObservation(
        runtime_records_by_axis=_runtime_records(execution),
        runtime_exports=RuntimeExportObservation.from_output_root(output_root),
    )
    failures = [
        *_execution_failures(execution),
        *_runtime_artifact_failures(expectation, observation),
        *_export_failures(expectation, observation),
    ]
    if failures:
        raise CPPipeExecutionValidationError(
            "Converted CellProfiler pipeline violated compiled expectations:\n"
            + "\n".join(f"- {failure}" for failure in failures)
        )
    return CPPipeExecutionValidation(
        expectation=expectation,
        observation=observation,
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


def _runtime_artifact_kinds(
    prepared: PreparedGeneratedPipeline,
) -> frozenset[ArtifactKind]:
    return frozenset(
        spec.kind
        for contract in prepared.generated_pipeline.artifact_contracts
        for spec in contract.outputs
    )


def _runtime_exports(
    infrastructure_features: frozenset[CPPipeInfrastructureFeature],
    runtime_artifact_kinds: frozenset[ArtifactKind],
) -> RuntimeExportExpectation:
    return RuntimeExportExpectation.from_flags(
        table_exports=(
            CPPipeInfrastructureFeature.EXPORT_TO_SPREADSHEET
            in infrastructure_features
        ),
        image_exports=CPPipeInfrastructureFeature.SAVE_IMAGES in infrastructure_features,
        table_artifact_kinds=frozenset(
            kind
            for kind in runtime_artifact_kinds
            if artifact_kind_exports_as_table(kind)
        ),
    )


def _runtime_records(
    execution: DirectPipelineExecution,
) -> Mapping[str, tuple[StoredRuntimeValue, ...]]:
    records_by_axis: dict[str, tuple[StoredRuntimeValue, ...]] = {}
    for axis_id, context in execution.compiled_contexts.items():
        store = require_runtime_value_store(
            context,
            owner_name=f"compiled context {axis_id!r}",
        )
        records_by_axis[str(axis_id)] = tuple(store.values())
    return MappingProxyType(records_by_axis)


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


def _runtime_artifact_failures(
    expectation: CPPipeExecutionExpectation,
    observation: CPPipeExecutionObservation,
) -> tuple[str, ...]:
    failures: list[str] = []
    for axis_id, counts in observation.runtime_record_counts_by_axis.items():
        for kind in sorted(
            expectation.runtime_artifact_kinds,
            key=lambda artifact_kind: artifact_kind.value,
        ):
            if counts.get(kind, 0) == 0:
                failures.append(
                    f"axis {axis_id!r} produced no runtime records for "
                    f"declared artifact kind {kind.value!r}"
                )
    return tuple(failures)


def _export_failures(
    expectation: CPPipeExecutionExpectation,
    observation: CPPipeExecutionObservation,
) -> tuple[str, ...]:
    return runtime_export_failures(
        expectation.runtime_exports,
        observation.runtime_exports,
        observation.runtime_records_by_axis,
    )
