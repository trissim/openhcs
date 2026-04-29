"""Runtime artifact export expectations and observations."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from openhcs.core.artifacts import ArtifactKind, ArtifactPayloadShape
from openhcs.core.runtime_stores import StoredRuntimeValue


class RuntimeExportFormat(str, Enum):
    """File export format families for runtime artifacts."""

    TABLE = "table"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class RuntimeExportExpectation:
    """Expected export formats for one runtime execution."""

    formats: frozenset[RuntimeExportFormat]
    table_artifact_kinds: frozenset[ArtifactKind] = frozenset()

    @classmethod
    def from_flags(
        cls,
        *,
        table_exports: bool,
        image_exports: bool,
        table_artifact_kinds: frozenset[ArtifactKind] = frozenset(),
    ) -> "RuntimeExportExpectation":
        formats = {
            format_
            for format_, enabled in (
                (RuntimeExportFormat.TABLE, table_exports),
                (RuntimeExportFormat.IMAGE, image_exports),
            )
            if enabled
        }
        return cls(
            formats=frozenset(formats),
            table_artifact_kinds=frozenset(table_artifact_kinds),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "formats",
            frozenset(
                format_
                if isinstance(format_, RuntimeExportFormat)
                else RuntimeExportFormat(format_)
                for format_ in self.formats
            ),
        )
        object.__setattr__(
            self,
            "table_artifact_kinds",
            frozenset(
                kind if isinstance(kind, ArtifactKind) else ArtifactKind(kind)
                for kind in self.table_artifact_kinds
            ),
        )

    @property
    def expects_table_files(self) -> bool:
        return RuntimeExportFormat.TABLE in self.formats and any(
            artifact_kind_exports_as_table(kind)
            for kind in self.table_artifact_kinds
        )

    @property
    def expects_image_files(self) -> bool:
        return RuntimeExportFormat.IMAGE in self.formats


@dataclass(frozen=True, slots=True)
class RuntimeExportObservation:
    """Observed file exports from one runtime execution."""

    table_outputs: tuple[Path, ...]
    image_outputs: tuple[Path, ...]
    table_headers_by_path: Mapping[Path, tuple[str, ...]]
    table_row_counts_by_path: Mapping[Path, int]

    @classmethod
    def from_output_root(
        cls,
        output_root: Path,
        *,
        image_dir_name: str = "images",
    ) -> "RuntimeExportObservation":
        table_outputs = _table_outputs(output_root)
        return cls(
            table_outputs=table_outputs,
            image_outputs=_image_outputs(output_root, image_dir_name),
            table_headers_by_path=_table_headers_by_path(table_outputs),
            table_row_counts_by_path=_table_row_counts_by_path(table_outputs),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "table_outputs", tuple(self.table_outputs))
        object.__setattr__(self, "image_outputs", tuple(self.image_outputs))
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


def runtime_export_failures(
    expectation: RuntimeExportExpectation,
    observation: RuntimeExportObservation,
    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
) -> tuple[str, ...]:
    """Return validation failures for expected runtime artifact exports."""
    failures: list[str] = []
    if expectation.expects_table_files and not observation.table_outputs:
        failures.append("table artifact exports were expected but no table files exist")
    for path in observation.table_outputs:
        if not observation.table_headers_by_path[path]:
            failures.append(f"table output {path} has an empty header")
        if observation.table_row_counts_by_path[path] == 0:
            failures.append(f"table output {path} has no data rows")
    if expectation.expects_table_files:
        failures.extend(_table_artifact_failures(observation, runtime_records_by_axis))
    if expectation.expects_image_files and not observation.image_outputs:
        failures.append("image exports were expected but no image outputs exist")
    return tuple(failures)


def artifact_kind_exports_as_table(kind: ArtifactKind) -> bool:
    """Return whether an artifact kind materializes as a table export."""
    return kind.payload_shape is ArtifactPayloadShape.TABLE


def matching_table_outputs(
    record: StoredRuntimeValue,
    table_outputs: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Return table output files matching one runtime artifact record."""
    return tuple(
        path
        for path in table_outputs
        if table_output_matches_artifact(path, record.key.name)
    )


def table_output_matches_artifact(path: Path, artifact_name: str) -> bool:
    """Return whether a materialized table filename belongs to an artifact."""
    return f"_{artifact_name}_step" in path.stem


def _table_artifact_failures(
    observation: RuntimeExportObservation,
    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
) -> tuple[str, ...]:
    failures: list[str] = []
    for axis_id, records in runtime_records_by_axis.items():
        for record in _table_runtime_records(records):
            matching_outputs = matching_table_outputs(
                record,
                observation.table_outputs,
            )
            if not matching_outputs:
                failures.append(
                    f"axis {axis_id!r} produced table artifact "
                    f"{record.key.name!r} ({record.key.kind.value}) but no "
                    "matching table output exists"
                )
                continue
            failures.extend(
                _table_schema_field_failures(
                    record,
                    matching_outputs,
                    observation.table_headers_by_path,
                )
            )
    return tuple(failures)


def _table_runtime_records(
    records: tuple[StoredRuntimeValue, ...],
) -> tuple[StoredRuntimeValue, ...]:
    return tuple(
        record
        for record in records
        if artifact_kind_exports_as_table(record.key.kind)
    )


def _table_schema_field_failures(
    record: StoredRuntimeValue,
    table_outputs: tuple[Path, ...],
    headers_by_path: Mapping[Path, tuple[str, ...]],
) -> tuple[str, ...]:
    expected_fields = tuple(field.name for field in record.value.schema.fields)
    if not expected_fields:
        return ()

    failures: list[str] = []
    for path in table_outputs:
        header = headers_by_path[path]
        missing_fields = tuple(
            field for field in expected_fields if field not in header
        )
        if missing_fields:
            failures.append(
                f"table output {path} for artifact {record.key.name!r} is "
                f"missing schema fields {missing_fields!r}"
            )
    return tuple(failures)


def _table_outputs(output_root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(Path(output_root).rglob("*.csv"))
        if path.is_file() and path.stat().st_size > 0
    )


def _image_outputs(output_root: Path, image_dir_name: str) -> tuple[Path, ...]:
    image_dir = Path(output_root) / image_dir_name
    if not image_dir.exists():
        return ()
    return tuple(path for path in sorted(image_dir.iterdir()) if path.is_file())


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
