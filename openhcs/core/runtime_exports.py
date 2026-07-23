"""Runtime artifact export expectations and observations."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import cast

from openhcs.core.artifacts import (
    ArtifactPayloadShape,
    ArtifactSpec,
    MeasurementsArtifactType,
)
from openhcs.core.image_file_serialization import ImageFileFormat
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.processing.materialization import (
    CsvOptions,
    FileBundleOptions,
    FileOutputOptions,
    ImageFileOptions,
    MaterializationSpec,
)
from openhcs.processing.materialization.core import materialization_is_empty


@dataclass(frozen=True, slots=True)
class RuntimeExportExpectation:
    """Explicitly materialized artifact outputs expected from one execution."""

    output_specs: tuple[ArtifactSpec, ...] = ()

    @classmethod
    def from_output_specs(
        cls,
        output_specs: Sequence[ArtifactSpec],
    ) -> "RuntimeExportExpectation":
        specs = tuple(output_specs)
        for spec in specs:
            if not isinstance(spec, ArtifactSpec):
                raise TypeError(
                    "RuntimeExportExpectation output specs must be ArtifactSpec "
                    f"values, got {type(spec).__name__}."
                )
        return cls(
            tuple(
                dict.fromkeys(
                    spec
                    for spec in specs
                    if isinstance(spec.materialization, MaterializationSpec)
                    and spec.materialization.participates_in_runtime_export_observation()
                )
            )
        )

    def __post_init__(self) -> None:
        specs = tuple(self.output_specs)
        for spec in specs:
            if not isinstance(spec, ArtifactSpec):
                raise TypeError(
                    "RuntimeExportExpectation.output_specs must contain "
                    f"ArtifactSpec values, got {type(spec).__name__}."
                )
            if not isinstance(spec.materialization, MaterializationSpec):
                raise TypeError(
                    "RuntimeExportExpectation.output_specs must be explicitly "
                    f"materialized; {spec.name!r} has "
                    f"{type(spec.materialization).__name__}."
                )
            if not spec.materialization.participates_in_runtime_export_observation():
                raise TypeError(
                    "RuntimeExportExpectation.output_specs must contain pipeline-"
                    f"declared exports; {spec.name!r} is terminal persistence."
                )
        object.__setattr__(self, "output_specs", specs)

    @property
    def table_output_specs(self) -> tuple[ArtifactSpec, ...]:
        return _output_specs_with_options(self.output_specs, CsvOptions)

    @property
    def image_output_specs(self) -> tuple[ArtifactSpec, ...]:
        return _output_specs_with_options(self.output_specs, ImageFileOptions)

    @property
    def file_bundle_output_specs(self) -> tuple[ArtifactSpec, ...]:
        return _output_specs_with_options(self.output_specs, FileBundleOptions)

    @property
    def expects_table_files(self) -> bool:
        return bool(self.table_output_specs)

    @property
    def expects_image_files(self) -> bool:
        return bool(self.image_output_specs)


@dataclass(frozen=True, slots=True)
class RuntimeExportObservation:
    """Observed file exports from one runtime execution."""

    table_outputs: tuple[Path, ...]
    image_outputs: tuple[Path, ...]
    table_headers_by_path: Mapping[Path, tuple[str, ...]]
    table_row_counts_by_path: Mapping[Path, int]
    output_files: tuple[Path, ...] = ()

    @classmethod
    def from_output_root(
        cls,
        output_root: Path,
    ) -> "RuntimeExportObservation":
        """Build an export observation from one runtime output root."""
        return cls.from_output_roots((Path(output_root),))

    @classmethod
    def from_output_roots(
        cls,
        output_roots: tuple[Path, ...],
    ) -> "RuntimeExportObservation":
        """Build an export observation from compiled runtime output roots."""
        roots = tuple(dict.fromkeys(Path(root) for root in output_roots))
        return cls.from_output_paths(_output_files_from_roots(roots))

    @classmethod
    def from_output_paths(
        cls,
        output_paths: Sequence[str | Path],
    ) -> "RuntimeExportObservation":
        """Build an export observation from exact contract-owned output paths."""

        output_files = tuple(
            dict.fromkeys(
                path
                for value in output_paths
                for path in (Path(value),)
                if path.is_file()
            )
        )
        table_outputs = tuple(
            path
            for path in output_files
            if path.suffix.lower() == ".csv" and path.stat().st_size > 0
        )
        image_outputs = tuple(
            path for path in output_files if _is_image_output_path(path)
        )
        return cls(
            table_outputs=table_outputs,
            image_outputs=image_outputs,
            table_headers_by_path=_table_headers_by_path(table_outputs),
            table_row_counts_by_path=_table_row_counts_by_path(table_outputs),
            output_files=output_files,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "table_outputs", tuple(self.table_outputs))
        object.__setattr__(self, "image_outputs", tuple(self.image_outputs))
        object.__setattr__(self, "output_files", tuple(self.output_files))
        object.__setattr__(
            self,
            "table_headers_by_path",
            MappingProxyType(dict(self.table_headers_by_path)),
        )
        object.__setattr__(
            self,
            "table_row_counts_by_path",
            MappingProxyType(dict(self.table_row_counts_by_path)),
        )

    def with_runtime_artifact_tables(
        self,
        runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
    ) -> "RuntimeExportObservation":
        """Return this observation scoped to table files owned by runtime artifacts."""
        table_outputs = tuple(
            dict.fromkeys(
                path
                for records in runtime_records_by_axis.values()
                for record in _table_runtime_records(records)
                for path in matching_table_outputs(record, self.table_outputs)
            )
        )
        return RuntimeExportObservation(
            table_outputs=table_outputs,
            image_outputs=self.image_outputs,
            table_headers_by_path={
                path: self.table_headers_by_path[path] for path in table_outputs
            },
            table_row_counts_by_path={
                path: self.table_row_counts_by_path[path] for path in table_outputs
            },
            output_files=self.output_files,
        )


def runtime_export_failures(
    expectation: RuntimeExportExpectation,
    observation: RuntimeExportObservation,
    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
) -> tuple[str, ...]:
    """Return validation failures for expected runtime artifact exports."""
    failures = list(
        _materialized_artifact_record_failures(
            expectation.output_specs,
            runtime_records_by_axis,
        )
    )
    if expectation.expects_table_files and not observation.table_outputs:
        failures.append("table artifact exports were expected but no table files exist")
    for path in observation.table_outputs:
        if not observation.table_headers_by_path[path]:
            failures.append(f"table output {path} has an empty header")
    if expectation.expects_table_files:
        failures.extend(
            _table_artifact_failures(
                expectation.table_output_specs,
                observation,
                runtime_records_by_axis,
            )
        )
    if expectation.expects_image_files and not observation.image_outputs:
        failures.append("image exports were expected but no image outputs exist")
    failures.extend(
        _file_bundle_failures(
            expectation.file_bundle_output_specs,
            observation,
            runtime_records_by_axis,
        )
    )
    return tuple(failures)


def _output_specs_with_options(
    output_specs: tuple[ArtifactSpec, ...],
    options_type: type[FileOutputOptions],
) -> tuple[ArtifactSpec, ...]:
    return tuple(
        spec
        for spec in output_specs
        if isinstance(spec.materialization, MaterializationSpec)
        and any(
            isinstance(options, options_type)
            for options in spec.materialization.outputs
        )
    )


def _runtime_records_for_spec(
    spec: ArtifactSpec,
    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
) -> tuple[StoredRuntimeValue, ...]:
    return tuple(
        record
        for records in runtime_records_by_axis.values()
        for record in records
        if record.key.name == spec.name
        and record.key.artifact_type is spec.artifact_type
    )


def _runtime_records_for_specs(
    output_specs: tuple[ArtifactSpec, ...],
    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
) -> tuple[StoredRuntimeValue, ...]:
    return tuple(
        record
        for spec in output_specs
        for record in _runtime_records_for_spec(spec, runtime_records_by_axis)
    )


def _materialized_artifact_record_failures(
    output_specs: tuple[ArtifactSpec, ...],
    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
) -> tuple[str, ...]:
    return tuple(
        f"produced no runtime record for materialized artifact {spec.name!r} "
        f"({spec.artifact_type.require_value()})"
        for spec in output_specs
        if not _runtime_records_for_spec(spec, runtime_records_by_axis)
    )


def matching_table_outputs(
    record: StoredRuntimeValue,
    table_outputs: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Return table output files matching one runtime artifact record."""
    return tuple(
        path
        for path in table_outputs
        if table_output_matches_artifact(
            path,
            record.key.name,
            axis_id=record.key.scope.axis_id,
        )
    )


def table_output_matches_artifact(
    path: Path,
    artifact_name: str,
    *,
    axis_id: str | None = None,
) -> bool:
    """Return whether a materialized table filename belongs to an artifact."""
    stem = path.stem
    if f"_{artifact_name}_step" not in stem:
        return False
    if axis_id is None:
        return True
    return stem.startswith(f"{axis_id}_")


def _table_artifact_failures(
    output_specs: tuple[ArtifactSpec, ...],
    observation: RuntimeExportObservation,
    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
) -> tuple[str, ...]:
    failures: list[str] = []
    for record in _runtime_records_for_specs(output_specs, runtime_records_by_axis):
        matching_outputs = matching_table_outputs(
            record,
            observation.table_outputs,
        )
        if not matching_outputs:
            failures.append(
                f"axis {record.key.scope.axis_id!r} produced table artifact "
                f"{record.key.name!r} ({record.key.artifact_type.value}) but no "
                "matching table output exists"
            )
            continue
        failures.extend(
            _table_row_count_failures(
                record,
                matching_outputs,
                observation.table_row_counts_by_path,
            )
        )
        failures.extend(
            _table_schema_field_failures(
                record,
                matching_outputs,
                observation.table_headers_by_path,
                observation.table_row_counts_by_path,
            )
        )
    return tuple(failures)


def _table_runtime_records(
    records: tuple[StoredRuntimeValue, ...],
) -> tuple[StoredRuntimeValue, ...]:
    return tuple(
        record
        for record in records
        if record.key.artifact_type.payload_shape is ArtifactPayloadShape.TABLE
    )


def _file_bundle_failures(
    output_specs: tuple[ArtifactSpec, ...],
    observation: RuntimeExportObservation,
    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
) -> tuple[str, ...]:
    failures: list[str] = []
    for record in _runtime_records_for_specs(output_specs, runtime_records_by_axis):
        payload = record.value.materialization_payload()
        if type(payload) is not dict:
            failures.append(
                f"materialized file-bundle artifact {record.key.name!r} has "
                f"non-dict payload {type(payload).__name__}"
            )
            continue
        for relative_path in payload:
            normalized = _normalized_bundle_relative_path(relative_path)
            if normalized is None:
                failures.append(
                    f"materialized file-bundle artifact {record.key.name!r} has "
                    f"invalid relative output path {relative_path!r}"
                )
                continue
            if not any(
                _output_path_has_relative_suffix(path, normalized)
                for path in observation.output_files
            ):
                failures.append(
                    f"materialized file-bundle artifact {record.key.name!r} "
                    f"declared missing output path {normalized.as_posix()!r}"
                )
    return tuple(failures)


def _normalized_bundle_relative_path(value: object) -> PurePosixPath | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or str(path) in {"", "."} or ".." in path.parts:
        return None
    return path


def _output_path_has_relative_suffix(path: Path, relative_path: PurePosixPath) -> bool:
    expected_parts = relative_path.parts
    return (
        len(path.parts) >= len(expected_parts)
        and path.parts[-len(expected_parts) :] == expected_parts
    )


def _table_schema_field_failures(
    record: StoredRuntimeValue,
    table_outputs: tuple[Path, ...],
    headers_by_path: Mapping[Path, tuple[str, ...]],
    row_counts_by_path: Mapping[Path, int],
) -> tuple[str, ...]:
    if record.key.artifact_type is not MeasurementsArtifactType:
        return ()
    expected_fields = tuple(
        field.name for field in cast(MeasurementTable, record.value.data).rows.fields
    )
    if not expected_fields:
        return ()

    failures: list[str] = []
    for path in table_outputs:
        if row_counts_by_path[path] == 0:
            continue
        missing_fields = _missing_schema_fields(
            headers_by_path[path],
            expected_fields,
        )
        if missing_fields:
            failures.append(
                f"table output {path} for artifact {record.key.name!r} is "
                f"missing schema fields {missing_fields!r}"
            )
    return tuple(failures)


def _table_row_count_failures(
    record: StoredRuntimeValue,
    table_outputs: tuple[Path, ...],
    row_counts_by_path: Mapping[Path, int],
) -> tuple[str, ...]:
    if materialization_is_empty(record.value.materialization_payload()):
        return ()
    return tuple(
        f"table output {path} has no data rows"
        for path in table_outputs
        if row_counts_by_path[path] == 0
    )


def _missing_schema_fields(
    header: tuple[str, ...],
    expected_fields: tuple[str, ...],
) -> tuple[str, ...]:
    normalized_header = {normalize_runtime_identifier(field) for field in header}
    return tuple(
        field
        for field in expected_fields
        if normalize_runtime_identifier(field) not in normalized_header
    )


def _output_files(output_root: Path) -> tuple[Path, ...]:
    return tuple(
        path for path in sorted(Path(output_root).rglob("*")) if path.is_file()
    )


def _output_files_from_roots(output_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(path for root in output_roots for path in _output_files(root))
    )


def _is_image_output_path(path: Path) -> bool:
    return ImageFileFormat.is_image_path(path)


def _table_header(path: Path) -> tuple[str, ...]:
    with path.open(newline="") as handle:
        try:
            return tuple(next(csv.reader(handle)))
        except StopIteration:
            return ()


def _table_headers_by_path(paths: tuple[Path, ...]) -> Mapping[Path, tuple[str, ...]]:
    return MappingProxyType({path: _table_header(path) for path in paths})


def _table_row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _row in reader)


def _table_row_counts_by_path(paths: tuple[Path, ...]) -> Mapping[Path, int]:
    return MappingProxyType({path: _table_row_count(path) for path in paths})
