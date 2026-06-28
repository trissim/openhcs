"""Measurement row materialization and columnar view semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from dataclasses import replace as dataclass_replace
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias, cast

from metaclass_registry import AutoRegisterMeta
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
import numpy as np

from openhcs.core.registry_strategies import NominalTypeStrategyFamilyMixin
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
    MeasurementScalarLiteral,
    MeasurementScope,
    measurement_axis_integer_domain,
    measurement_axis_integer_value,
    measurement_row_mapping,
)
from openhcs.core.runtime_values import ColumnarRows, MeasurementTable

MEASUREMENT_OBJECT_NAME_FIELD = "object_name"
MEASUREMENT_SOURCE_IMAGE_NAME_FIELD = "source_image_name"
MEASUREMENT_OBJECT_LABEL_FIELD = "object_label"
MEASUREMENT_OBJECT_NUMBER_FIELD = "object_number"
MEASUREMENT_OBJECT_ID_FIELD = "object_id"
MEASUREMENT_LABEL_FIELD = "label"
MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD = "openhcs_object_row_identity"
MEASUREMENT_OBJECT_ID_FIELDS = (
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_OBJECT_ID_FIELD,
    MEASUREMENT_LABEL_FIELD,
)
MEASUREMENT_FEATURE_NAME_FIELD = "feature_name"
MEASUREMENT_MEASUREMENT_NAME_FIELD = "measurement_name"
MEASUREMENT_OUTPUT_NAME_FIELD = "output_name"
MEASUREMENT_FEATURE_NAME_FIELDS = (
    MEASUREMENT_FEATURE_NAME_FIELD,
    MEASUREMENT_MEASUREMENT_NAME_FIELD,
    MEASUREMENT_OUTPUT_NAME_FIELD,
)
MEASUREMENT_RESULT_VALUE_FIELD = "result_value"
MEASUREMENT_MEASUREMENT_VALUE_FIELD = "measurement_value"
MEASUREMENT_VALUE_FIELD = "value"
MEASUREMENT_MEAN_VALUE_FIELD = "mean_value"
MEASUREMENT_VALUE_FIELDS = (
    MEASUREMENT_RESULT_VALUE_FIELD,
    MEASUREMENT_MEASUREMENT_VALUE_FIELD,
    MEASUREMENT_VALUE_FIELD,
    MEASUREMENT_MEAN_VALUE_FIELD,
)


def is_structural_missing_measurement_cell(value: object) -> bool:
    """Return whether a columnar value marks structural absence, not a value."""
    return isinstance(value, MeasurementSparseCell)


def columnar_row_values(rows: ColumnarRows, column: str) -> Sequence[object]:
    """Return one column from a nominal columnar payload."""
    return rows.column_values(column)


def column_mapping_row_count(columns: Mapping[str, Sequence[object]]) -> int:
    """Return row count for a concrete column mapping."""
    if not columns:
        return 0
    return len(next(iter(columns.values())))


def iter_measurement_rows(
    measurement_tables: Iterable[MeasurementTable],
) -> Iterator[object]:
    """Yield row payloads from measurement tables without materializing them."""
    for table in measurement_tables:
        yield from table.iter_rows()


def measurement_rows(
    measurement_tables: tuple[MeasurementTable, ...],
) -> tuple[object, ...]:
    """Flatten row payloads from measurement tables."""
    return tuple(iter_measurement_rows(measurement_tables))


def measurement_table_axis_values(
    table: MeasurementTable,
    axis: MeasurementRowAxisField,
) -> set[int]:
    """Return declared row-axis values for one measurement table."""
    axis_field = axis.value
    if isinstance(table.rows, ColumnarRows):
        column_names = tuple(str(column) for column in table.rows.columns)
        if axis_field not in column_names:
            return set()
        return set(
            measurement_axis_integer_domain(
                columnar_row_values(table.rows, axis_field),
                axis,
            )
        )
    return {
        axis_integer
        for row in measurement_rows((table,))
        for row_mapping in (measurement_row_mapping(row),)
        for axis_integer in (
            measurement_axis_integer_value(row_mapping.get(axis_field), axis),
        )
        if axis_integer is not None
    }


ProjectedMeasurementRows: TypeAlias = Sequence[Mapping[str, Any]] | ColumnarRows


@dataclass(frozen=True, slots=True)
class MeasurementProjectedColumnarRows(ColumnarRows):
    """Columnar measurement rows with projected row-axis values."""

    columns: Mapping[str, Sequence[Any]]
    declared_object_measurement_domain_covered: bool = False

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        """Return whether projection preserved complete object-domain rows."""
        return bool(self.declared_object_measurement_domain_covered)

    def __len__(self) -> int:
        return column_mapping_row_count(self.columns)

    def __iter__(self):
        yield from self.row_mappings()

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                field_name: value
                for field_name, value in row.items()
                if not is_structural_missing_measurement_cell(value)
            }
            for row in ColumnarRows.row_mappings(self)
        )


@dataclass(frozen=True, slots=True)
class MeasurementSparseCell:
    """Structural missing-cell marker for sparse columnar row materialization."""


MEASUREMENT_SPARSE_CELL = MeasurementSparseCell()


@dataclass(frozen=True, slots=True)
class ColumnarRowColumnOverlay(Mapping[str, Sequence[Any]]):
    """Lazy column mapping that overlays projected columns on an existing table."""

    base_columns: Mapping[str, Sequence[Any]]
    overlay_columns: Mapping[str, Sequence[Any]]
    column_names: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "column_names",
            tuple(
                dict.fromkeys(
                    (
                        *(str(column) for column in self.base_columns),
                        *(str(column) for column in self.overlay_columns),
                    )
                )
            ),
        )

    def __getitem__(self, column_name: str) -> Sequence[Any]:
        if column_name in self.overlay_columns:
            return self.overlay_columns[column_name]
        return self.base_columns[column_name]

    def __iter__(self):
        return iter(self.column_names)

    def __len__(self) -> int:
        return len(self.column_names)


@dataclass(frozen=True, slots=True)
class MeasurementSparseColumnarRows(ColumnarRows):
    """Columnar measurement rows whose missing cells are structural, not values."""

    columns: Mapping[str, Sequence[Any]]
    missing_cell: object = MEASUREMENT_SPARSE_CELL

    def __len__(self) -> int:
        return column_mapping_row_count(self.columns)

    def __iter__(self):
        columns = self.columns
        for row_index in range(len(self)):
            yield {
                field_name: value
                for field_name, values in columns.items()
                for value in (values[row_index],)
                if not is_structural_missing_measurement_cell(value)
            }

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self)


@dataclass(frozen=True, slots=True)
class MeasurementSliceIndexImageNumberProjection:
    """Map runtime slice indices onto external image-number row values."""

    start: int
    image_numbers_by_slice: Mapping[int, int]

    def image_number_for_slice(self, slice_index: int) -> int:
        mapped = self.image_numbers_by_slice.get(slice_index)
        if mapped is not None:
            return mapped
        return slice_index + self.start


class MeasurementRowsAxisProjection(
    NominalTypeStrategyFamilyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project measurement rows from runtime-axis space into image-number space."""

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[object] | ColumnarRows,
    ) -> "MeasurementRowsAxisProjection":
        projection_type = cls.strategy_types_for_nominal_value(rows)[0]
        return cast(MeasurementRowsAxisProjection, projection_type(rows=rows))

    @staticmethod
    def row_has_axis(row: Mapping[str, object]) -> bool:
        return (
            MeasurementRowAxisField.IMAGE_NUMBER.value in row
            or MeasurementRowAxisField.SLICE_INDEX.value in row
        )

    @property
    @abstractmethod
    def has_rows(self) -> bool:
        """Return whether this projection has row payloads to project."""

    @property
    @abstractmethod
    def row_count(self) -> int:
        """Return the number of measurement rows represented by this projection."""

    @property
    @abstractmethod
    def columns(self) -> Mapping[str, Sequence[Any]]:
        """Return column vectors when the underlying rows are columnar."""

    @property
    @abstractmethod
    def has_axis(self) -> bool:
        """Return whether rows declare a runtime or CellProfiler row axis."""

    @property
    @abstractmethod
    def has_image_number(self) -> bool:
        """Return whether rows declare CellProfiler ImageNumber values."""

    @property
    @abstractmethod
    def has_slice_index(self) -> bool:
        """Return whether rows declare runtime slice-index values."""

    @property
    @abstractmethod
    def has_source_qualified_image_rows(self) -> bool:
        """Return whether rows describe source images without object ownership."""

    @abstractmethod
    def present_axis_values(self, field_name: str) -> tuple[int, ...]:
        """Return present integer value domain for one measurement row-axis field."""

    @abstractmethod
    def project_runtime_slice_index(
        self,
        slice_index: int,
    ) -> Sequence[object] | ColumnarRows:
        """Return rows stamped into one runtime slice."""

    @abstractmethod
    def aggregate_runtime_slice_index(
        self,
        slice_index: int,
    ) -> Sequence[object] | ColumnarRows:
        """Return rows for PURE_2D aggregation, preserving explicit row axes."""

    @abstractmethod
    def project_current_image_number(
        self,
        start: int,
    ) -> Sequence[Mapping[str, object]] | ColumnarRows:
        """Return rows with a current ImageNumber added where absent."""

    def apply(
        self,
        image_numbers: MeasurementSliceIndexImageNumberProjection,
        *,
        image_number_start_for_source_qualified_image_rows: int | None = None,
    ) -> Sequence[Any] | ColumnarRows | None:
        if (
            image_number_start_for_source_qualified_image_rows is not None
            and self.has_source_qualified_image_rows
        ):
            return self.project_current_image_number(
                image_number_start_for_source_qualified_image_rows
            )
        if not self.has_rows or not self.has_axis:
            return None
        if self.has_slice_index:
            return self.project_slice_index(image_numbers)
        if self.has_image_number:
            return self.project_image_number(image_numbers.start)
        return None

    @abstractmethod
    def project_slice_index(
        self,
        image_numbers: MeasurementSliceIndexImageNumberProjection,
    ) -> Sequence[Mapping[str, Any]] | ColumnarRows:
        """Project runtime slice-index values into CellProfiler ImageNumber values."""

    @abstractmethod
    def project_image_number(
        self,
        start: int,
    ) -> Sequence[Mapping[str, Any]] | ColumnarRows | None:
        """Project local ImageNumber values into global CellProfiler ImageNumber space."""


@dataclass(frozen=True, slots=True)
class SequenceMeasurementRowsAxisProjection(MeasurementRowsAxisProjection):
    """Row-axis projection for row-sequence measurement payloads."""

    value_type = Sequence
    rows: Sequence[object]

    @property
    def has_rows(self) -> bool:
        return bool(self.rows)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def columns(self) -> Mapping[str, Sequence[Any]]:
        return MappingProxyType({})

    @property
    def has_axis(self) -> bool:
        return any(self.row_has_axis(measurement_row_mapping(row)) for row in self.rows)

    @property
    def has_image_number(self) -> bool:
        return any(
            MeasurementRowAxisField.IMAGE_NUMBER.value in measurement_row_mapping(row)
            for row in self.rows
        )

    @property
    def has_slice_index(self) -> bool:
        return any(
            MeasurementRowAxisField.SLICE_INDEX.value in measurement_row_mapping(row)
            for row in self.rows
        )

    @property
    def has_source_qualified_image_rows(self) -> bool:
        return any(
            MEASUREMENT_SOURCE_IMAGE_NAME_FIELD in row_mapping
            and not measurement_row_has_object_identity(row_mapping)
            for row_mapping in (measurement_row_mapping(row) for row in self.rows)
        )

    def project_current_image_number(
        self,
        start: int,
    ) -> Sequence[Mapping[str, object]]:
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        projected_rows = [dict(measurement_row_mapping(row)) for row in self.rows]
        for row in projected_rows:
            if image_number_field not in row:
                row[image_number_field] = start
        return projected_rows

    def project_runtime_slice_index(
        self,
        slice_index: int,
    ) -> Sequence[object]:
        """Stamp runtime-slice index while preserving row dataclass types."""
        slice_index_field = MeasurementRowAxisField.SLICE_INDEX.value
        projected_rows: list[object] = []
        for row in self.rows:
            row_mapping = measurement_row_mapping(row)
            if (
                is_dataclass(row)
                and slice_index_field in row_mapping
                and slice_index_field in dataclass_init_field_names(type(row))
            ):
                projected_rows.append(
                    dataclass_replace(row, **{slice_index_field: int(slice_index)})
                )
                continue
            projected_row = dict(row_mapping)
            projected_row[slice_index_field] = int(slice_index)
            projected_rows.append(projected_row)
        return projected_rows

    def aggregate_runtime_slice_index(
        self,
        slice_index: int,
    ) -> Sequence[object]:
        """Preserve already-projected row axes; stamp axisless per-slice rows."""
        if self.has_axis:
            return self.rows
        return self.project_runtime_slice_index(slice_index)

    def present_axis_values(self, field_name: str) -> tuple[int, ...]:
        """Return present integer axis values for one measurement row field."""
        axis = MeasurementRowAxisField(field_name)
        return tuple(
            dict.fromkeys(
                integer_value
                for row in (measurement_row_mapping(row) for row in self.rows)
                for integer_value in (
                    measurement_axis_integer_value(row.get(field_name), axis),
                )
                if integer_value is not None
            )
        )

    def project_axis_values(
        self,
        *,
        source_field_name: str,
        target_field_name: str,
        transform: Callable[[int], int],
    ) -> Sequence[Mapping[str, Any]]:
        """Return rows with present source-axis values projected into a target."""
        source_axis = MeasurementRowAxisField(source_field_name)
        projected_rows = [dict(measurement_row_mapping(row)) for row in self.rows]
        for row in projected_rows:
            axis_value = measurement_axis_integer_value(
                row.get(source_field_name),
                source_axis,
            )
            if axis_value is not None:
                row[target_field_name] = transform(axis_value)
        return projected_rows

    def project_slice_index(
        self,
        image_numbers: MeasurementSliceIndexImageNumberProjection,
    ) -> Sequence[Mapping[str, Any]]:
        slice_index_field = MeasurementRowAxisField.SLICE_INDEX.value
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        return self.project_axis_values(
            source_field_name=slice_index_field,
            target_field_name=image_number_field,
            transform=image_numbers.image_number_for_slice,
        )

    def project_image_number(
        self,
        start: int,
    ) -> Sequence[Mapping[str, Any]] | None:
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        image_numbers = self.present_axis_values(image_number_field)
        if not image_numbers or min(image_numbers) >= start:
            return None

        offset = start - 1
        return self.project_axis_values(
            source_field_name=image_number_field,
            target_field_name=image_number_field,
            transform=lambda value: value + offset,
        )


def dataclass_init_field_names(row_type: type[object]) -> frozenset[str]:
    """Return constructor-backed dataclass field names for row replacement."""
    return frozenset(
        field.name
        for field in dataclass_fields(row_type)
        if field.init
    )


@dataclass(frozen=True, slots=True)
class ColumnarMeasurementRowsAxisProjection(MeasurementRowsAxisProjection):
    """Row-axis projection for nominal columnar measurement payloads."""

    value_type = ColumnarRows
    rows: ColumnarRows
    _columns: Mapping[str, Sequence[Any]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        columns = self.rows.columns
        if isinstance(columns, Mapping) and all(
            isinstance(column, str)
            for column in columns
        ):
            object.__setattr__(self, "_columns", columns)
            return
        object.__setattr__(
            self,
            "_columns",
            MappingProxyType(
                {
                    str(column): columnar_row_values(self.rows, str(column))
                    for column in self.rows.columns
                }
            ),
        )

    @property
    def has_rows(self) -> bool:
        return self.rows.row_count() > 0

    @property
    def row_count(self) -> int:
        return self.rows.row_count()

    @property
    def columns(self) -> Mapping[str, Sequence[Any]]:
        return self._columns

    @property
    def has_axis(self) -> bool:
        return self.has_image_number or self.has_slice_index

    @property
    def has_image_number(self) -> bool:
        return MeasurementRowAxisField.IMAGE_NUMBER.value in self.columns

    @property
    def has_slice_index(self) -> bool:
        return MeasurementRowAxisField.SLICE_INDEX.value in self.columns

    @property
    def has_source_qualified_image_rows(self) -> bool:
        columns = self.columns
        return (
            MEASUREMENT_SOURCE_IMAGE_NAME_FIELD in columns
            and not any(field in columns for field in MEASUREMENT_OBJECT_ID_FIELDS)
        )

    def present_axis_values(self, field_name: str) -> tuple[int, ...]:
        """Return present integer axis values for one measurement column."""
        return measurement_axis_integer_domain(
            self.columns.get(field_name, ()),
            MeasurementRowAxisField(field_name),
        )

    def project_runtime_slice_index(
        self,
        slice_index: int,
    ) -> ColumnarRows:
        slice_index_field = MeasurementRowAxisField.SLICE_INDEX.value
        return MeasurementProjectedColumnarRows(
            ColumnarRowColumnOverlay(
                self.columns,
                MappingProxyType(
                    {slice_index_field: (int(slice_index),) * self.rows.row_count()}
                ),
            ),
            declared_object_measurement_domain_covered=(
                self.rows.covers_declared_object_measurement_domain
            ),
        )

    def aggregate_runtime_slice_index(
        self,
        slice_index: int,
    ) -> ColumnarRows:
        """Preserve already-projected row axes; stamp axisless per-slice rows."""
        if self.has_axis:
            return self.rows
        return self.project_runtime_slice_index(slice_index)

    def project_current_image_number(self, start: int) -> ColumnarRows:
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        overlay_columns: Mapping[str, Sequence[Any]]
        if image_number_field in self.columns:
            overlay_columns = MappingProxyType({})
        else:
            overlay_columns = MappingProxyType(
                {image_number_field: (start,) * self.rows.row_count()}
            )
        return MeasurementProjectedColumnarRows(
            ColumnarRowColumnOverlay(self.columns, overlay_columns),
            declared_object_measurement_domain_covered=(
                self.rows.covers_declared_object_measurement_domain
            ),
        )

    def project_slice_index(
        self,
        image_numbers: MeasurementSliceIndexImageNumberProjection,
    ) -> ColumnarRows:
        slice_index_field = MeasurementRowAxisField.SLICE_INDEX.value
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        return MeasurementProjectedColumnarRows(
            ColumnarRowColumnOverlay(
                self.columns,
                MappingProxyType(
                    {
                        image_number_field: self.projected_image_numbers(
                            image_numbers,
                            self.columns[slice_index_field],
                        )
                    }
                ),
            ),
            declared_object_measurement_domain_covered=(
                self.rows.covers_declared_object_measurement_domain
            ),
        )

    def project_image_number(self, start: int) -> ColumnarRows | None:
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        columns = self.columns
        image_numbers = measurement_axis_integer_domain(
            columns[image_number_field],
            MeasurementRowAxisField.IMAGE_NUMBER,
        )
        if not image_numbers or min(image_numbers) >= start:
            return None
        offset = start - 1
        return MeasurementProjectedColumnarRows(
            ColumnarRowColumnOverlay(
                columns,
                MappingProxyType(
                    {
                        image_number_field: tuple(
                            int(value) + offset
                            if MeasurementScalarLiteral(value).is_present_axis_value
                            else value
                            for value in columns[image_number_field]
                        )
                    }
                ),
            ),
            declared_object_measurement_domain_covered=(
                self.rows.covers_declared_object_measurement_domain
            ),
        )

    @staticmethod
    def projected_image_numbers(
        image_numbers: MeasurementSliceIndexImageNumberProjection,
        slice_indices: Sequence[Any],
    ) -> Sequence[Any]:
        """Return projected image numbers for one columnar slice-index vector."""
        values = np.asarray(slice_indices)
        if values.size == 0:
            return ()
        if np.issubdtype(values.dtype, np.integer):
            unique_values = np.unique(values)
            if unique_values.size == 1:
                return np.full(
                    values.shape,
                    image_numbers.image_number_for_slice(int(unique_values[0])),
                    dtype=np.int64,
                )
            mapping = {
                int(slice_index): image_numbers.image_number_for_slice(int(slice_index))
                for slice_index in unique_values
            }
            return np.asarray(
                [mapping[int(slice_index)] for slice_index in values],
                dtype=np.int64,
            )
        return tuple(
            image_numbers.image_number_for_slice(int(value))
            if measurement_axis_integer_value(
                value,
                MeasurementRowAxisField.SLICE_INDEX,
            ) is not None
            else value
            for value in slice_indices
        )


def measurement_row_object_name(row: Mapping[str, object]) -> str | None:
    """Return the object owner encoded on one measurement row."""
    value = row.get(MEASUREMENT_OBJECT_NAME_FIELD)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def measurement_row_source_image_name(row: Mapping[str, object]) -> str | None:
    """Return the source-image owner encoded on one measurement row."""
    value = row.get(MEASUREMENT_SOURCE_IMAGE_NAME_FIELD)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


@dataclass(frozen=True, slots=True)
class MeasurementObjectLabelResolution:
    """Integer object label resolved from runtime/CSV scalar encodings."""

    value: object

    @property
    def object_label(self) -> int | None:
        return MeasurementScalarLiteral(self.value).integer_value


def measurement_object_label(
    row: Mapping[str, object],
    *,
    object_id_field: str | None = None,
) -> int | None:
    """Return the resolved object label encoded on a measurement row."""
    if object_id_field is not None:
        value = measurement_row_field_value(row, object_id_field)
        if value is not None:
            return MeasurementObjectLabelResolution(value).object_label
    for key in MEASUREMENT_OBJECT_ID_FIELDS:
        value = measurement_row_field_value(row, key)
        if value is not None:
            return MeasurementObjectLabelResolution(value).object_label
    return None


def measurement_row_has_object_identity(
    row: Mapping[str, object],
    *,
    object_id_field: str | None = None,
) -> bool:
    """Return whether a measurement row carries resolved object identity."""
    return measurement_object_label(row, object_id_field=object_id_field) is not None


def measurement_row_has_long_form_measurement_fields(
    row: Mapping[str, object],
) -> bool:
    """Return whether a row carries long-form measurement feature/value fields."""
    return any(
        measurement_row_field_value(row, field) is not None
        for field in MEASUREMENT_FEATURE_NAME_FIELDS
    ) and any(
        measurement_row_field_value(row, field) is not None
        for field in MEASUREMENT_VALUE_FIELDS
    )


def measurement_row_identity_role(
    row: Mapping[str, object],
) -> MeasurementObjectRowIdentity | None:
    """Return the explicit OpenHCS row-identity role encoded on a measurement row."""
    value = measurement_row_field_value(row, MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD)
    if value not in MeasurementObjectRowIdentity._value2member_map_:
        return None
    return MeasurementObjectRowIdentity(value)


def measurement_row_field_value(
    row: Mapping[str, object],
    field_name: str,
) -> object | None:
    """Return a row value by normalized measurement field name."""
    if field_name in row:
        return row[field_name]
    normalized_target = normalize_runtime_identifier(field_name)
    for candidate_field, value in row.items():
        if normalize_runtime_identifier(str(candidate_field)) == normalized_target:
            return value
    return None


def measurement_table_object_id_field(table: MeasurementTable) -> str | None:
    """Return the authoritative object-id field declared by a measurement table."""
    if table.object_id_field is not None:
        return table.object_id_field
    if table.subject and table.subject.scope is MeasurementScope.OBJECT:
        return table.subject.id_field
    return None


def measurement_table_object_name(table: MeasurementTable) -> str | None:
    """Return the authoritative object name for object-scoped measurement tables."""
    if table.object_name is not None:
        return table.object_name
    if table.subject and table.subject.scope is MeasurementScope.OBJECT:
        return table.subject.name
    return None


@dataclass(frozen=True, slots=True)
class MeasurementRowQualifier:
    """One typed ownership qualifier attached to a measurement row."""

    field_name: str
    value: str

    @classmethod
    def optional(
        cls,
        *,
        field_name: str,
        value: str | None,
    ) -> "MeasurementRowQualifier | None":
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} cannot be empty.")
        return cls(field_name=field_name, value=normalized)

    def apply(self, row: MutableMapping[str, object]) -> None:
        row[self.field_name] = self.value


@dataclass(frozen=True, slots=True)
class MeasurementRowOwnership:
    """Shared object/source ownership qualifiers for measurement rows."""

    object_name: str | None = None
    source_image_name: str | None = None

    @property
    def qualifiers(self) -> tuple[MeasurementRowQualifier, ...]:
        return tuple(
            qualifier
            for qualifier in (
                MeasurementRowQualifier.optional(
                    field_name=MEASUREMENT_OBJECT_NAME_FIELD,
                    value=self.object_name,
                ),
                MeasurementRowQualifier.optional(
                    field_name=MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
                    value=self.source_image_name,
                ),
            )
            if qualifier is not None
        )

    def annotate_rows(self, rows: Sequence[object] | ColumnarRows) -> Sequence[object] | ColumnarRows:
        """Attach ownership qualifiers, copying only non-mutable row values."""
        qualifiers = self.qualifiers
        if not qualifiers:
            return rows
        if isinstance(rows, ColumnarRows):
            return QualifiedMeasurementColumnarRows(rows, qualifiers)
        if (
            rows
            and is_dataclass(type(rows[0]))
            and all(type(row) is type(rows[0]) for row in rows)
        ):
            return QualifiedMeasurementColumnarRows(
                DataclassMeasurementColumnarRows(rows),
                qualifiers,
            )
        return [self.annotate_row(row, qualifiers=qualifiers) for row in rows]

    def annotate_row(
        self,
        row: object,
        *,
        qualifiers: Sequence[MeasurementRowQualifier] | None = None,
    ) -> Mapping[str, object]:
        if qualifiers is None:
            qualifiers = self.qualifiers
        annotated_row: MutableMapping[str, object] = (
            row
            if isinstance(row, MutableMapping)
            else dict(measurement_row_mapping(row))
        )
        for qualifier in qualifiers:
            qualifier.apply(annotated_row)
        return annotated_row


@dataclass(slots=True)
class MeasurementColumnarRowsView(ColumnarRows, ABC):
    """Base for columnar measurement views that derive columns from another table."""

    _columns: Mapping[str, Sequence[object]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    columns: ClassVar[AliasProperty[Mapping[str, Sequence[object]]]] = (
        AliasProperty("_columns")
    )

    def __len__(self) -> int:
        return column_mapping_row_count(self._columns)


def columnar_row_count(rows: ColumnarRows) -> int:
    """Return row count for a nominal columnar payload."""
    return rows.row_count()


@dataclass(frozen=True, slots=True)
class ConcatenatedColumnarRowColumns(Mapping[str, Sequence[object]]):
    """Lazy mapping over columns concatenated from multiple columnar batches."""

    row_batches: tuple[ColumnarRows, ...]
    column_names: tuple[str, ...]

    @classmethod
    def from_row_batches(
        cls,
        row_batches: tuple[ColumnarRows, ...],
    ) -> "ConcatenatedColumnarRowColumns":
        return cls(
            row_batches=row_batches,
            column_names=tuple(
                dict.fromkeys(
                    str(column)
                    for row_batch in row_batches
                    for column in row_batch.columns
                )
            ),
        )

    def __getitem__(self, column_name: str) -> Sequence[object]:
        if column_name not in self.column_names:
            raise KeyError(column_name)
        return np.concatenate(
            tuple(
                self._batch_column_values(row_batch, column_name)
                for row_batch in self.row_batches
            )
        )

    def _batch_column_values(
        self,
        row_batch: ColumnarRows,
        column_name: str,
    ) -> Sequence[object]:
        batch_columns = {str(column): column for column in row_batch.columns}
        if column_name in batch_columns:
            return columnar_row_values(row_batch, batch_columns[column_name])
        return (None,) * columnar_row_count(row_batch)

    def __iter__(self):
        return iter(self.column_names)

    def __len__(self) -> int:
        return len(self.column_names)


@dataclass(slots=True)
class ConcatenatedColumnarRows(MeasurementColumnarRowsView):
    """Columnar table view over multiple columnar row batches."""

    row_batches: tuple[ColumnarRows, ...]

    def __post_init__(self) -> None:
        self._columns = ConcatenatedColumnarRowColumns.from_row_batches(
            self.row_batches,
        )

    def __len__(self) -> int:
        return sum(columnar_row_count(row_batch) for row_batch in self.row_batches)

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        """Return whether every concatenated batch covers its declared domain."""
        return bool(self.row_batches) and all(
            row_batch.covers_declared_object_measurement_domain
            for row_batch in self.row_batches
        )

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            row
            for row_batch in self.row_batches
            for row in row_batch.row_mappings()
        )


@dataclass(slots=True)
class DataclassMeasurementColumnarRows(ColumnarRows):
    """Columnar view over homogeneous dataclass measurement rows."""

    rows: Sequence[object]
    _columns: Mapping[str, Sequence[object]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    columns: ClassVar[AliasProperty[Mapping[str, Sequence[object]]]] = (
        AliasProperty("_columns")
    )

    def __post_init__(self) -> None:
        if not self.rows:
            self._columns = {}
            return
        row_type = type(self.rows[0])
        if not is_dataclass(row_type):
            raise TypeError(
                "DataclassMeasurementColumnarRows requires dataclass rows, "
                f"got {row_type.__name__}."
            )
        if not all(type(row) is row_type for row in self.rows):
            raise TypeError(
                "DataclassMeasurementColumnarRows requires homogeneous row types."
            )
        row_mappings = tuple(measurement_row_mapping(row) for row in self.rows)
        column_names = tuple(row_mappings[0])
        self._columns = {
            column_name: tuple(row[column_name] for row in row_mappings)
            for column_name in column_names
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        columns = self._columns
        for row_index in range(len(self)):
            yield {
                field_name: values[row_index]
                for field_name, values in columns.items()
            }


@dataclass(slots=True)
class QualifiedMeasurementColumnarRows(MeasurementColumnarRowsView):
    """Columnar measurement rows with table-ownership qualifiers attached."""

    rows: ColumnarRows
    qualifiers: tuple[MeasurementRowQualifier, ...]

    def __post_init__(self) -> None:
        columns = dict(self.rows.columns)
        row_count = column_mapping_row_count(columns)
        for qualifier in self.qualifiers:
            columns[qualifier.field_name] = (qualifier.value,) * row_count
        self._columns = columns

    def __iter__(self):
        columns = self._columns
        for row_index in range(len(self)):
            yield {
                field_name: values[row_index]
                for field_name, values in columns.items()
            }
