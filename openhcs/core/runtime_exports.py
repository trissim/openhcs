"""Runtime artifact export expectations and observations."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from openhcs.core.artifacts import (
    ArtifactType,
    ImageArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.image_file_serialization import (
    collapse_singleton_image_stack,
    image_payload_as_uint8,
)
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.runtime_values import (
    ColumnarRows,
    MeasurementTable,
)


class RuntimeExportFormat(str, Enum):
    """File export format families for runtime artifacts."""

    TABLE = "table"
    IMAGE = "image"


class RuntimeImageExportBitDepth(str, Enum):
    """Runtime image export encoding families."""

    NATIVE = "native"
    UINT8 = "uint8"
    UINT16 = "uint16"
    FLOAT32 = "float32"


@dataclass(frozen=True, slots=True)
class RuntimeImageExportSpec:
    """Declared image artifact export and file encoding semantics."""

    artifact_name: str
    bit_depth: RuntimeImageExportBitDepth = RuntimeImageExportBitDepth.NATIVE
    file_format: str | None = None

    def __post_init__(self) -> None:
        artifact_name = self.artifact_name.strip()
        if not artifact_name:
            raise ValueError("RuntimeImageExportSpec.artifact_name cannot be empty.")
        bit_depth = (
            self.bit_depth
            if isinstance(self.bit_depth, RuntimeImageExportBitDepth)
            else RuntimeImageExportBitDepth(self.bit_depth)
        )
        file_format = (
            self.file_format.strip().lower()
            if self.file_format is not None and self.file_format.strip()
            else None
        )
        object.__setattr__(self, "artifact_name", artifact_name)
        object.__setattr__(self, "bit_depth", bit_depth)
        object.__setattr__(self, "file_format", file_format)

    def prepare_payload(self, payload: Any) -> np.ndarray:
        """Return payload pixels as they should appear in the exported image."""
        array = collapse_singleton_image_stack(payload)
        if self.bit_depth is RuntimeImageExportBitDepth.UINT8:
            return image_payload_as_uint8(array)
        if self.bit_depth is RuntimeImageExportBitDepth.UINT16:
            return _image_payload_as_uint16(array)
        if self.bit_depth is RuntimeImageExportBitDepth.FLOAT32:
            return array.astype(np.float32, copy=False)
        return array


@dataclass(frozen=True, slots=True)
class RuntimeExportExpectation:
    """Expected export formats for one runtime execution."""

    formats: frozenset[RuntimeExportFormat]
    table_artifact_kinds: frozenset[ArtifactType] = frozenset()
    image_artifact_names: frozenset[str] = frozenset()
    image_export_specs: tuple[RuntimeImageExportSpec, ...] = ()

    @classmethod
    def from_flags(
        cls,
        *,
        table_exports: bool,
        image_exports: bool,
        table_artifact_kinds: frozenset[ArtifactType] = frozenset(),
        image_artifact_names: frozenset[str] = frozenset(),
        image_export_specs: tuple[RuntimeImageExportSpec, ...] = (),
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
            image_artifact_names=frozenset(image_artifact_names),
            image_export_specs=tuple(image_export_specs),
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
                ArtifactType.coerce(kind)
                for kind in self.table_artifact_kinds
            ),
        )
        object.__setattr__(
            self,
            "image_export_specs",
            _normalized_image_export_specs(
                self.image_export_specs,
                self.image_artifact_names,
            ),
        )
        object.__setattr__(
            self,
            "image_artifact_names",
            frozenset(spec.artifact_name for spec in self.image_export_specs),
        )

    @property
    def expects_table_files(self) -> bool:
        return RuntimeExportFormat.TABLE in self.formats and any(
            kind.exports_as_table()
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
        """Build an export observation from one runtime output root."""
        return cls.from_output_roots(
            (Path(output_root),),
            image_dir_name=image_dir_name,
        )

    @classmethod
    def from_output_roots(
        cls,
        output_roots: tuple[Path, ...],
        *,
        image_dir_name: str = "images",
    ) -> "RuntimeExportObservation":
        """Build an export observation from compiled runtime output roots."""
        roots = tuple(dict.fromkeys(Path(root) for root in output_roots))
        table_outputs = _table_outputs_from_roots(roots)
        image_outputs = _image_outputs_from_roots(roots, image_dir_name)
        return cls(
            table_outputs=table_outputs,
            image_outputs=image_outputs,
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
    if expectation.expects_table_files:
        failures.extend(_table_artifact_failures(observation, runtime_records_by_axis))
    if expectation.expects_image_files and not observation.image_outputs:
        failures.append("image exports were expected but no image outputs exist")
    if expectation.expects_image_files and expectation.image_artifact_names:
        failures.extend(
            _image_artifact_failures(
                expectation.image_artifact_names,
                runtime_records_by_axis,
            )
        )
    return tuple(failures)


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
        if record.key.artifact_type.exports_as_table()
    )


def image_runtime_records(
    records: tuple[StoredRuntimeValue, ...],
    *,
    artifact_names: frozenset[str] = frozenset(),
) -> tuple[StoredRuntimeValue, ...]:
    """Return runtime image artifact records, optionally scoped by artifact name."""
    return tuple(
        record
        for record in records
        if record.key.artifact_type is ImageArtifactType
        and (not artifact_names or record.key.name in artifact_names)
    )


def _image_artifact_failures(
    image_artifact_names: frozenset[str],
    runtime_records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
) -> tuple[str, ...]:
    failures: list[str] = []
    for axis_id, records in runtime_records_by_axis.items():
        record_names = {
            record.key.name
            for record in image_runtime_records(records)
        }
        for image_name in sorted(image_artifact_names - record_names):
            failures.append(
                f"axis {axis_id!r} produced no runtime image artifact "
                f"{image_name!r} for declared image export"
            )
    return tuple(failures)


def _normalized_image_export_specs(
    specs: tuple[RuntimeImageExportSpec, ...],
    artifact_names: frozenset[str],
) -> tuple[RuntimeImageExportSpec, ...]:
    normalized_specs = tuple(
        spec if isinstance(spec, RuntimeImageExportSpec) else RuntimeImageExportSpec(
            **spec
        )
        for spec in specs
    )
    declared_names = tuple(
        name.strip()
        for name in artifact_names
        if name.strip()
    )
    if not normalized_specs:
        return tuple(RuntimeImageExportSpec(name) for name in declared_names)
    spec_names = {spec.artifact_name for spec in normalized_specs}
    return (
        *normalized_specs,
        *(
            RuntimeImageExportSpec(name)
            for name in declared_names
            if name not in spec_names
        ),
    )


def _image_payload_as_uint16(payload: Any) -> np.ndarray:
    array = np.asarray(payload)
    if array.dtype == np.uint16:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint16) * np.uint16(65535)
    values = array.astype(np.float64, copy=False)
    if _is_unit_interval(values):
        values = values * 65535.0
    sanitized = np.nan_to_num(values, nan=0.0, posinf=65535.0, neginf=0.0)
    return np.rint(np.clip(sanitized, 0.0, 65535.0)).astype(np.uint16)


def _is_unit_interval(values: np.ndarray) -> bool:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return True
    return float(finite_values.min()) >= 0.0 and float(finite_values.max()) <= 1.0


def _table_schema_field_failures(
    record: StoredRuntimeValue,
    table_outputs: tuple[Path, ...],
    headers_by_path: Mapping[Path, tuple[str, ...]],
    row_counts_by_path: Mapping[Path, int],
) -> tuple[str, ...]:
    expected_fields = tuple(field.name for field in record.value.schema.fields)
    if not expected_fields:
        return ()

    failures: list[str] = []
    for path in table_outputs:
        if row_counts_by_path[path] == 0:
            continue
        header = headers_by_path[path]
        if not _header_has_schema_compatible_field(header, expected_fields):
            failures.append(
                f"table output {path} for artifact {record.key.name!r} is "
                f"missing schema fields {expected_fields!r}"
            )
    return tuple(failures)


def _table_row_count_failures(
    record: StoredRuntimeValue,
    table_outputs: tuple[Path, ...],
    row_counts_by_path: Mapping[Path, int],
) -> tuple[str, ...]:
    runtime_row_count = _runtime_table_row_count(record)
    if runtime_row_count == 0:
        return ()
    return tuple(
        f"table output {path} has no data rows"
        for path in table_outputs
        if row_counts_by_path[path] == 0
    )


def _runtime_table_row_count(record: StoredRuntimeValue) -> int:
    data = record.value.data
    if record.key.artifact_type is MeasurementsArtifactType:
        data = MeasurementTable.from_runtime_value(record.value).rows
    if isinstance(data, ColumnarRows):
        try:
            return len(data)  # type: ignore[arg-type]
        except TypeError:
            return 1
    if isinstance(data, Mapping):
        return _mapping_table_row_count(data)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return len(data)
    return 1 if data is not None else 0


def _mapping_table_row_count(data: Mapping[Any, Any]) -> int:
    if not data:
        return 0
    column_lengths = tuple(
        length
        for value in data.values()
        if (length := _table_column_length(value)) is not None
    )
    if column_lengths:
        return max(column_lengths)
    return 1


def _table_column_length(value: Any) -> int | None:
    if isinstance(value, Mapping):
        return None
    if isinstance(value, (str, bytes, bytearray)):
        return None
    if isinstance(value, ColumnarRows):
        try:
            return len(value)  # type: ignore[arg-type]
        except TypeError:
            return None
    shape = getattr(value, "shape", None)
    if shape:
        return int(shape[0])
    if isinstance(value, Sequence):
        return len(value)
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _header_has_schema_compatible_field(
    header: tuple[str, ...],
    expected_fields: tuple[str, ...],
) -> bool:
    normalized_header = {_normalize_export_field(field) for field in header}
    return any(
        field in header or _normalize_export_field(field) in normalized_header
        for field in expected_fields
    )


def _normalize_export_field(field: str) -> str:
    text = str(field).strip()
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def _table_outputs(output_root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(Path(output_root).rglob("*.csv"))
        if path.is_file() and path.stat().st_size > 0
    )


def _table_outputs_from_roots(output_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            path for root in output_roots for path in _table_outputs(root)
        )
    )


def _image_outputs(output_root: Path, image_dir_name: str) -> tuple[Path, ...]:
    image_dir = Path(output_root) / image_dir_name
    if image_dir.exists():
        return tuple(path for path in sorted(image_dir.iterdir()) if path.is_file())
    return tuple(
        path
        for path in sorted(Path(output_root).rglob("*"))
        if path.is_file() and _is_image_output_path(path)
    )


def _is_image_output_path(path: Path) -> bool:
    return path.suffix.lower() in {
        ".bmp",
        ".jpeg",
        ".jpg",
        ".npy",
        ".png",
        ".tif",
        ".tiff",
    }


def _image_outputs_from_roots(
    output_roots: tuple[Path, ...],
    image_dir_name: str,
) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            path
            for root in output_roots
            for path in _image_outputs(root, image_dir_name)
        )
    )


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
