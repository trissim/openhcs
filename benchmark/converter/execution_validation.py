"""Validation for converted CellProfiler pipeline executions."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_stores import RuntimeValueStore

from benchmark.converter.runtime_pipeline import (
    DirectPipelineExecution,
    PreparedGeneratedPipeline,
)

CSV_EXPORT_ARTIFACT_KINDS = frozenset(
    {
        ArtifactKind.MEASUREMENTS,
        ArtifactKind.RELATIONSHIPS,
        ArtifactKind.TABLE,
    }
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

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedGeneratedPipeline,
    ) -> "CPPipeExecutionExpectation":
        return cls(
            infrastructure_features=_infrastructure_features(prepared),
            runtime_artifact_kinds=_runtime_artifact_kinds(prepared),
        )

    @property
    def expects_csv_exports(self) -> bool:
        return (
            CPPipeInfrastructureFeature.EXPORT_TO_SPREADSHEET
            in self.infrastructure_features
            and bool(self.runtime_artifact_kinds & CSV_EXPORT_ARTIFACT_KINDS)
        )

    @property
    def expects_image_exports(self) -> bool:
        return CPPipeInfrastructureFeature.SAVE_IMAGES in self.infrastructure_features


@dataclass(frozen=True, slots=True)
class CPPipeExecutionObservation:
    """Observed runtime/export outputs from one converted .cppipe execution."""

    runtime_record_counts_by_axis: Mapping[str, Mapping[ArtifactKind, int]]
    csv_outputs: tuple[Path, ...]
    image_outputs: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_record_counts_by_axis",
            MappingProxyType(
                {
                    axis: MappingProxyType(dict(counts))
                    for axis, counts in self.runtime_record_counts_by_axis.items()
                }
            ),
        )
        object.__setattr__(self, "csv_outputs", tuple(self.csv_outputs))
        object.__setattr__(self, "image_outputs", tuple(self.image_outputs))


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
        runtime_record_counts_by_axis=_runtime_record_counts(execution),
        csv_outputs=_csv_outputs(output_root),
        image_outputs=_image_outputs(output_root),
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


def _runtime_record_counts(
    execution: DirectPipelineExecution,
) -> Mapping[str, Mapping[ArtifactKind, int]]:
    counts_by_axis: dict[str, Mapping[ArtifactKind, int]] = {}
    for axis_id, context in execution.compiled_contexts.items():
        store = getattr(context, "runtime_value_store", None)
        if not isinstance(store, RuntimeValueStore):
            counts_by_axis[str(axis_id)] = MappingProxyType({})
            continue
        counts_by_axis[str(axis_id)] = MappingProxyType(
            Counter(record.key.kind for record in store.values())
        )
    return MappingProxyType(counts_by_axis)


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
    failures: list[str] = []
    if expectation.expects_csv_exports and not observation.csv_outputs:
        failures.append("ExportToSpreadsheet declared but no CSV outputs exist")
    for path in observation.csv_outputs:
        if not _csv_header(path):
            failures.append(f"CSV output {path} has an empty header")
    if expectation.expects_image_exports and not observation.image_outputs:
        failures.append("SaveImages declared but no image outputs exist")
    return tuple(failures)


def _csv_outputs(output_root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(Path(output_root).rglob("*.csv"))
        if path.is_file() and path.stat().st_size > 0
    )


def _image_outputs(output_root: Path) -> tuple[Path, ...]:
    image_dir = Path(output_root) / "images"
    if not image_dir.exists():
        return ()
    return tuple(path for path in sorted(image_dir.iterdir()) if path.is_file())


def _csv_header(path: Path) -> tuple[str, ...]:
    with path.open(newline="") as handle:
        try:
            return tuple(next(csv.reader(handle)))
        except StopIteration:
            return ()
