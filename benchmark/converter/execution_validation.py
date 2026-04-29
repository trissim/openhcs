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
from openhcs.core.runtime_stores import (
    StoredRuntimeValue,
    require_runtime_value_store,
)

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

    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]]
    csv_outputs: tuple[Path, ...]
    image_outputs: tuple[Path, ...]
    csv_headers_by_path: Mapping[Path, tuple[str, ...]]
    csv_row_counts_by_path: Mapping[Path, int]

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
        object.__setattr__(self, "csv_outputs", tuple(self.csv_outputs))
        object.__setattr__(self, "image_outputs", tuple(self.image_outputs))
        object.__setattr__(
            self,
            "csv_headers_by_path",
            MappingProxyType(dict(self.csv_headers_by_path)),
        )
        object.__setattr__(
            self,
            "csv_row_counts_by_path",
            MappingProxyType(dict(self.csv_row_counts_by_path)),
        )

    @property
    def runtime_record_counts_by_axis(self) -> Mapping[str, Mapping[ArtifactKind, int]]:
        return MappingProxyType(
            {
                axis: MappingProxyType(Counter(record.key.kind for record in records))
                for axis, records in self.runtime_records_by_axis.items()
            }
        )


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
    csv_outputs = _csv_outputs(output_root)
    observation = CPPipeExecutionObservation(
        runtime_records_by_axis=_runtime_records(execution),
        csv_outputs=csv_outputs,
        image_outputs=_image_outputs(output_root),
        csv_headers_by_path=_csv_headers_by_path(csv_outputs),
        csv_row_counts_by_path=_csv_row_counts_by_path(csv_outputs),
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
    failures: list[str] = []
    if expectation.expects_csv_exports and not observation.csv_outputs:
        failures.append("ExportToSpreadsheet declared but no CSV outputs exist")
    for path in observation.csv_outputs:
        if not observation.csv_headers_by_path[path]:
            failures.append(f"CSV output {path} has an empty header")
        if observation.csv_row_counts_by_path[path] == 0:
            failures.append(f"CSV output {path} has no data rows")
    if expectation.expects_csv_exports:
        failures.extend(_csv_artifact_failures(observation))
    if expectation.expects_image_exports and not observation.image_outputs:
        failures.append("SaveImages declared but no image outputs exist")
    return tuple(failures)


def _csv_artifact_failures(
    observation: CPPipeExecutionObservation,
) -> tuple[str, ...]:
    failures: list[str] = []
    for axis_id, records in observation.runtime_records_by_axis.items():
        for record in _csv_runtime_records(records):
            matching_outputs = _matching_csv_outputs(
                record,
                observation.csv_outputs,
            )
            if not matching_outputs:
                failures.append(
                    f"axis {axis_id!r} produced CSV artifact "
                    f"{record.key.name!r} ({record.key.kind.value}) but no "
                    "matching CSV output exists"
                )
                continue
            failures.extend(
                _csv_schema_field_failures(
                    record,
                    matching_outputs,
                    observation.csv_headers_by_path,
                )
            )
    return tuple(failures)


def _csv_runtime_records(
    records: tuple[StoredRuntimeValue, ...],
) -> tuple[StoredRuntimeValue, ...]:
    return tuple(
        record
        for record in records
        if record.key.kind in CSV_EXPORT_ARTIFACT_KINDS
    )


def _matching_csv_outputs(
    record: StoredRuntimeValue,
    csv_outputs: tuple[Path, ...],
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in csv_outputs
        if _csv_output_matches_artifact(path, record.key.name)
    )


def _csv_output_matches_artifact(path: Path, artifact_name: str) -> bool:
    return f"_{artifact_name}_step" in path.stem


def _csv_schema_field_failures(
    record: StoredRuntimeValue,
    csv_outputs: tuple[Path, ...],
    headers_by_path: Mapping[Path, tuple[str, ...]],
) -> tuple[str, ...]:
    expected_fields = tuple(field.name for field in record.value.schema.fields)
    if not expected_fields:
        return ()

    failures: list[str] = []
    for path in csv_outputs:
        header = headers_by_path[path]
        missing_fields = tuple(
            field for field in expected_fields if field not in header
        )
        if missing_fields:
            failures.append(
                f"CSV output {path} for artifact {record.key.name!r} is "
                f"missing schema fields {missing_fields!r}"
            )
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


def _csv_headers_by_path(paths: tuple[Path, ...]) -> Mapping[Path, tuple[str, ...]]:
    return MappingProxyType({path: _csv_header(path) for path in paths})


def _csv_row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _row in reader)


def _csv_row_counts_by_path(paths: tuple[Path, ...]) -> Mapping[Path, int]:
    return MappingProxyType({path: _csv_row_count(path) for path in paths})
