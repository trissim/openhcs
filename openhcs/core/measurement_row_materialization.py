"""Measurement row materialization and columnar view semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from dataclasses import replace as dataclass_replace
from functools import lru_cache
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias, cast

from metaclass_registry import AutoRegisterMeta
from openhcs.core.alias_property import AliasProperty
import numpy as np

from openhcs.core.registry_strategies import NominalTypeStrategyFamilyMixin
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    MeasurementScalarLiteral,
    MeasurementScope,
    RuntimeMeasurementRowIdentityContract,
    measurement_axis_integer_domain,
    measurement_axis_integer_value,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
    measurement_row_mapping,
)
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)

from enum import Enum
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_measurements import ObjectMeasurementValueRow


@lru_cache(maxsize=32768)
def normalized_measurement_row_fields(
    fields: tuple[str, ...],
) -> Mapping[str, str]:
    """Return normalized field-name lookup for one measurement row shape."""
    return MappingProxyType(
        {normalize_runtime_identifier(field): field for field in fields}
    )


def normalized_measurement_row_fields_for_row(
    row: Mapping[str, object],
) -> Mapping[str, str]:
    """Return cached normalized field names for one row mapping."""
    return normalized_measurement_row_fields(tuple(str(field) for field in row))


def is_structural_missing_measurement_cell(value: object) -> bool:
    """Return whether a columnar value marks structural absence, not a value."""
    return isinstance(value, MeasurementSparseCell)


def columnar_row_values(rows: ColumnarRows, column: str) -> Sequence[object]:
    """Return one column from a nominal columnar payload."""
    return rows.column_values(column)


def projected_columnar_fields(
    rows: ColumnarRows,
    field_spec: FieldSpec,
) -> tuple[FieldSpec, ...]:
    """Return fields for a projection that adds or replaces one exact column."""
    return FieldSpec.merge_exact(
        (rows.fields, (field_spec,)),
        context="projected column fields",
    )


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
        yield from table.rows.iter_row_mappings()


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
MeasurementFeatureNameProjection: TypeAlias = Callable[
    [str, tuple[tuple[str, object], ...]],
    str,
]


class MeasurementRowDeclaredValue(ABC, metaclass=AutoRegisterMeta):
    """Nominal declaration for values projected from a measurement row."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True
    declares_row_value: ClassVar[bool] = False

    @classmethod
    def declared_value_types(cls) -> tuple[type["MeasurementRowDeclaredValue"], ...]:
        """Return registered concrete row-value declarations."""
        return tuple(
            dict.fromkeys(
                declaration_type
                for declaration_type in cls.__registry__.values()
                if declaration_type.declares_row_value
                and not declaration_type.__abstractmethods__
            )
        )

    @classmethod
    def values_for_row(
        cls,
        row: Mapping[str, object],
        *,
        normalized_fields: Mapping[str, str] | None = None,
    ) -> Mapping[type["MeasurementRowDeclaredValue"], object | None]:
        """Return every declared row value keyed by its declaration type."""
        return MappingProxyType(
            {
                declaration_type: declaration_type.value_from_row(
                    row,
                    normalized_fields=normalized_fields,
                )
                for declaration_type in cls.declared_value_types()
            }
        )

    @classmethod
    @abstractmethod
    def value_from_row(
        cls,
        row: Mapping[str, object],
        *,
        normalized_fields: Mapping[str, str] | None = None,
        object_id_field: str | None = None,
    ) -> object | None:
        """Return this declared value from one measurement row."""


class MeasurementRowTextValue(MeasurementRowDeclaredValue):
    """Shared declaration for normalized text values stored in row fields."""

    field_names: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def value_from_row(
        cls,
        row: Mapping[str, object],
        *,
        normalized_fields: Mapping[str, str] | None = None,
        object_id_field: str | None = None,
    ) -> str | None:
        del object_id_field
        for field_name in cls.field_names:
            value = measurement_row_declared_field_value(
                row,
                field_name,
                normalized_fields,
            )
            if value is None:
                continue
            normalized = str(value).strip()
            if normalized:
                return normalized
        return None


class MeasurementRowObjectName(MeasurementRowTextValue):
    """Object owner encoded on a measurement row."""

    declares_row_value = True
    field_names = (MeasurementRowAxisField.OBJECT_NAME.value,)


class MeasurementRowSourceImageName(MeasurementRowTextValue):
    """Source-image owner encoded on a measurement row."""

    declares_row_value = True
    field_names = (MeasurementRowAxisField.SOURCE_IMAGE_NAME.value,)


class MeasurementRowObjectLabel(MeasurementRowDeclaredValue):
    """Resolved object label encoded on a measurement row."""

    declares_row_value = True

    @classmethod
    def value_from_row(
        cls,
        row: Mapping[str, object],
        *,
        normalized_fields: Mapping[str, str] | None = None,
        object_id_field: str | None = None,
    ) -> int | None:
        if object_id_field is not None:
            value = measurement_row_declared_field_value(
                row,
                object_id_field,
                normalized_fields,
            )
            if value is not None:
                return measurement_object_label_value(value)
        for key in MeasurementRowAxisField.object_id_field_names():
            value = measurement_row_declared_field_value(row, key, normalized_fields)
            if value is not None:
                return measurement_object_label_value(value)
        return None


class MeasurementRowObjectIdentityRole(MeasurementRowDeclaredValue):
    """Explicit OpenHCS object-row identity role encoded on a row."""

    declares_row_value = True

    @classmethod
    def explicit_value_from_row(
        cls,
        row: Mapping[str, object],
        *,
        normalized_fields: Mapping[str, str] | None = None,
    ) -> MeasurementObjectRowIdentity | None:
        """Return only the identity role explicitly encoded on the row."""
        value = measurement_row_declared_field_value(
            row,
            MeasurementRowAxisField.OBJECT_ROW_IDENTITY.value,
            normalized_fields,
        )
        if value in MeasurementObjectRowIdentity._value2member_map_:
            return MeasurementObjectRowIdentity(value)
        return None

    @classmethod
    def value_from_row(
        cls,
        row: Mapping[str, object],
        *,
        normalized_fields: Mapping[str, str] | None = None,
        object_id_field: str | None = None,
    ) -> MeasurementObjectRowIdentity | None:
        del object_id_field
        explicit_identity = cls.explicit_value_from_row(
            row,
            normalized_fields=normalized_fields,
        )
        if explicit_identity is not None:
            return explicit_identity
        if (
            measurement_row_declared_field_value(
                row,
                MeasurementRowAxisField.OBJECT_LABEL.value,
                normalized_fields,
            )
            is not None
        ):
            return MeasurementObjectRowIdentity.LABEL_ID
        return None

    @classmethod
    def resolve_value_from_row(
        cls,
        row: Mapping[str, object],
        *,
        carrier_identity: MeasurementObjectRowIdentity | None,
        normalized_fields: Mapping[str, str] | None = None,
    ) -> MeasurementObjectRowIdentity | None:
        """Resolve row identity with the nominal columnar carrier as authority."""
        explicit_identity = cls.explicit_value_from_row(
            row,
            normalized_fields=normalized_fields,
        )
        if (
            explicit_identity is not None
            and carrier_identity is not None
            and explicit_identity is not carrier_identity
        ):
            raise ValueError(
                "Measurement row identity conflicts with its nominal columnar "
                f"carrier: {explicit_identity.value!r} != "
                f"{carrier_identity.value!r}."
            )
        if carrier_identity is not None:
            return carrier_identity
        return cls.value_from_row(
            row,
            normalized_fields=normalized_fields,
        )


@dataclass(frozen=True, slots=True)
class MeasurementProjectedColumnarRows(ColumnarRows):
    """Columnar measurement rows with projected row-axis values."""

    columns: Mapping[str, Sequence[Any]]
    fields: tuple[FieldSpec, ...] = ()
    declared_object_measurement_domain_covered: bool = False
    object_row_identity: MeasurementObjectRowIdentity | None = None

    def __post_init__(self) -> None:
        self.validate_fields()

    @classmethod
    def from_columnar_rows(
        cls,
        rows: ColumnarRows,
        *,
        row_indices: Sequence[int] | None = None,
        declared_object_measurement_domain_covered: bool,
        object_row_identity: MeasurementObjectRowIdentity | None,
    ) -> "MeasurementProjectedColumnarRows":
        """Project declared columns without reconstructing row mappings."""
        fields = rows.fields
        if row_indices is None:
            columns = {
                field_spec.name: rows.column_values(field_spec.name)
                for field_spec in fields
            }
        else:
            selected_indices = tuple(int(row_index) for row_index in row_indices)
            row_count = rows.row_count()
            invalid_indices = tuple(
                row_index
                for row_index in selected_indices
                if row_index < 0 or row_index >= row_count
            )
            if invalid_indices:
                raise IndexError(
                    "Columnar row projection indices are outside the row domain: "
                    f"{invalid_indices!r} for {row_count} rows."
                )
            numpy_indices = np.asarray(selected_indices, dtype=np.intp)

            def selected_values(field_spec: FieldSpec) -> Sequence[Any]:
                values = rows.column_values(field_spec.name)
                if isinstance(values, np.ndarray):
                    return values[numpy_indices]
                return tuple(values[row_index] for row_index in selected_indices)

            columns = {
                field_spec.name: selected_values(field_spec) for field_spec in fields
            }
        return cls(
            MappingProxyType(columns),
            fields=fields,
            declared_object_measurement_domain_covered=(
                declared_object_measurement_domain_covered
            ),
            object_row_identity=object_row_identity,
        )

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        """Return whether projection preserved complete object-domain rows."""
        return bool(self.declared_object_measurement_domain_covered)

    def __len__(self) -> int:
        return column_mapping_row_count(self.columns)

    def __iter__(self):
        yield from self.iter_row_mappings()

    def __getitem__(
        self, row_index: int | slice
    ) -> Mapping[str, object] | tuple[Mapping[str, object], ...]:
        if not isinstance(row_index, (int, slice)):
            raise TypeError(
                f"{type(self).__name__} indices must be integers or slices, got "
                f"{type(row_index).__name__}."
            )
        if isinstance(row_index, slice):
            return tuple(
                self._row_mapping_at(selected_index)
                for selected_index in range(*row_index.indices(len(self)))
            )
        selected_index = row_index if row_index >= 0 else len(self) + row_index
        if selected_index < 0 or selected_index >= len(self):
            raise IndexError(row_index)
        return self._row_mapping_at(selected_index)

    def _row_mapping_at(self, row_index: int) -> Mapping[str, object]:
        return {
            field_name: value
            for field_name, values in self.columns.items()
            for value in (values[row_index],)
            if not is_structural_missing_measurement_cell(value)
        }

    def iter_row_mappings(self):
        columns = tuple(str(column) for column in self.columns)
        column_values = tuple(self.column_values(column) for column in columns)
        for row_index in range(len(self)):
            yield {
                field_name: values[row_index]
                for field_name, values in zip(columns, column_values, strict=True)
                if not is_structural_missing_measurement_cell(values[row_index])
            }

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self.iter_row_mappings())


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
    fields: tuple[FieldSpec, ...] = ()
    missing_cell: object = MEASUREMENT_SPARSE_CELL
    declared_object_measurement_domain_covered: bool = False
    object_row_identity: MeasurementObjectRowIdentity | None = None

    def __post_init__(self) -> None:
        self.validate_fields()

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[object],
        *,
        fields: tuple[FieldSpec, ...],
        declared_object_measurement_domain_covered: bool = False,
        missing_cell: object = MEASUREMENT_SPARSE_CELL,
        object_row_identity: MeasurementObjectRowIdentity | None = None,
    ) -> "MeasurementSparseColumnarRows":
        """Return a sparse columnar view over heterogeneous measurement rows."""
        row_mappings = coalesced_sparse_measurement_row_mappings(
            tuple(measurement_row_mapping(row) for row in rows),
            missing_cell=missing_cell,
        )
        field_names = tuple(field_spec.name for field_spec in fields)
        declared_names = frozenset(field_names)
        undeclared_names = tuple(
            dict.fromkeys(
                field_name
                for row_mapping in row_mappings
                for field_name in row_mapping
                if field_name not in declared_names
            )
        )
        if undeclared_names:
            raise ValueError(
                "Sparse measurement rows contain columns absent from their "
                f"declared fields: {undeclared_names!r}."
            )
        return cls(
            MappingProxyType(
                {
                    field_name: tuple(
                        row_mapping.get(field_name, missing_cell)
                        for row_mapping in row_mappings
                    )
                    for field_name in field_names
                }
            ),
            fields=fields,
            missing_cell=missing_cell,
            declared_object_measurement_domain_covered=(
                declared_object_measurement_domain_covered
            ),
            object_row_identity=object_row_identity,
        )

    @classmethod
    def from_columnar_batches(
        cls,
        batches: Sequence[ColumnarRows],
        *,
        declared_object_measurement_domain_covered: bool = False,
        missing_cell: object = MEASUREMENT_SPARSE_CELL,
    ) -> "MeasurementSparseColumnarRows":
        """Return sparse rows coalesced across columnar batches."""
        fields = FieldSpec.merge_exact(
            (batch.fields for batch in batches),
            context="columnar batch fields",
        )
        return cls.from_rows(
            tuple(row for batch in batches for row in batch.iter_row_mappings()),
            fields=fields,
            declared_object_measurement_domain_covered=(
                declared_object_measurement_domain_covered
            ),
            missing_cell=missing_cell,
            object_row_identity=ColumnarRows.common_object_row_identity(batches),
        )

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        """Return whether these sparse rows were completed over object domain."""
        return bool(self.declared_object_measurement_domain_covered)

    def __len__(self) -> int:
        return column_mapping_row_count(self.columns)

    def __iter__(self):
        yield from self.iter_row_mappings()

    def __getitem__(
        self, row_index: int | slice
    ) -> Mapping[str, object] | tuple[Mapping[str, object], ...]:
        if not isinstance(row_index, (int, slice)):
            raise TypeError(
                f"{type(self).__name__} indices must be integers or slices, got "
                f"{type(row_index).__name__}."
            )
        return self.row_mappings()[row_index]

    def iter_row_mappings(self):
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


def coalesced_sparse_measurement_row_mappings(
    row_mappings: Sequence[Mapping[str, object]],
    *,
    missing_cell: object = MEASUREMENT_SPARSE_CELL,
) -> tuple[Mapping[str, object], ...]:
    """Merge sparse feature fragments that share the same row-axis identity."""
    if not row_mappings:
        return ()
    identity_fields = tuple(
        field.value
        for field in MeasurementRowAxisField
        if any(field.value in row for row in row_mappings)
    )
    if not identity_fields:
        return tuple(row_mappings)

    coalesced: dict[tuple[tuple[str, object], ...], dict[str, object]] = {}
    order: list[tuple[tuple[str, object], ...]] = []
    passthrough_index = 0
    for row in row_mappings:
        identity = tuple(
            (field_name, row[field_name])
            for field_name in identity_fields
            if field_name in row
            and not is_structural_missing_measurement_cell(row[field_name])
        )
        if not identity:
            identity = (("__row_index__", passthrough_index),)
            passthrough_index += 1
        merged = coalesced.get(identity)
        if merged is None:
            merged = {}
            coalesced[identity] = merged
            order.append(identity)
        for field_name, value in row.items():
            if value is missing_cell or is_structural_missing_measurement_cell(value):
                continue
            existing = merged.get(field_name, missing_cell)
            if (
                existing is not missing_cell
                and not is_structural_missing_measurement_cell(existing)
                and not _measurement_sparse_cell_values_equal(existing, value)
            ):
                raise ValueError(
                    "Conflicting sparse measurement values for row identity "
                    f"{identity!r}, field {field_name!r}: {existing!r} vs {value!r}."
                )
            merged[field_name] = value
    return tuple(MappingProxyType(coalesced[identity]) for identity in order)


def measurement_column_carries_scalar_values(values: Sequence[object]) -> bool:
    """Return whether a column carries scalar measurement values."""

    saw_none = False
    for value in values:
        if is_structural_missing_measurement_cell(value):
            continue
        if value is None:
            saw_none = True
            continue
        return MeasurementScalarLiteral(value).token is not None
    return saw_none


def wide_measurement_feature_columns(
    columns: Mapping[str, Sequence[object]],
    *,
    object_id_field: str | None = None,
    qualifier_field_names: Iterable[str] = (),
) -> tuple[tuple[str, Sequence[object]], ...]:
    """Return scalar feature columns after excluding declared row structure."""

    object_id_fields = tuple(
        dict.fromkeys(
            (
                *((object_id_field,) if object_id_field is not None else ()),
                *MeasurementRowAxisField.object_id_field_names(),
            )
        )
    )
    folded_axis_fields = frozenset(
        (
            MeasurementRowAxisField.OBJECT_NAME.value,
            MeasurementRowAxisField.SOURCE_IMAGE_NAME.value,
            MeasurementRowAxisField.OBJECT_ROW_IDENTITY.value,
            *object_id_fields,
            *qualifier_field_names,
        )
    )
    return tuple(
        (field_name, values)
        for field_name, values in columns.items()
        if field_name not in MeasurementRowAxisField.field_names()
        and field_name not in MeasurementRowValueField.field_names()
        and field_name not in folded_axis_fields
        and measurement_column_carries_scalar_values(values)
    )


@dataclass(slots=True)
class WideMeasurementRowAccumulator:
    """Consolidate measurement columns directly into final subject-owned rows."""

    row_identity_contract: RuntimeMeasurementRowIdentityContract
    _rows: dict[str, dict[tuple[object, ...], dict[str, object]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _order: dict[str, list[tuple[object, ...]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _object_subjects: list[str] = field(default_factory=list, init=False, repr=False)
    _passthrough_index: int = field(default=0, init=False, repr=False)
    _absent_identity: object = field(default_factory=object, init=False, repr=False)
    _passthrough_identity: object = field(
        default_factory=object, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.row_identity_contract,
            RuntimeMeasurementRowIdentityContract,
        ):
            raise TypeError(
                "WideMeasurementRowAccumulator requires a "
                "RuntimeMeasurementRowIdentityContract."
            )

    def add(
        self,
        rows: Sequence[object] | ColumnarRows,
        project_feature_name: MeasurementFeatureNameProjection,
        *,
        default_subject: str,
        default_scope: MeasurementScope = MeasurementScope.ARTIFACT,
        source_image_name: str | None = None,
        object_id_field: str | None = None,
        qualifier_field_names: Iterable[str] = (),
        missing_cell: object = MEASUREMENT_SPARSE_CELL,
    ) -> None:
        """Add one measurement payload without materializing input row mappings."""
        if not isinstance(rows, ColumnarRows):
            rows = self._columnar_rows(rows, missing_cell)
        row_count = rows.row_count()
        if row_count == 0:
            return
        columns = {
            str(column): rows.column_values(str(column)) for column in rows.columns
        }
        feature_fields = MeasurementRowAxisField.feature_name_field_names_ordered()
        value_fields = MeasurementRowValueField.field_names_ordered()
        qualifier_fields = tuple(
            field_name
            for field_name in dict.fromkeys(qualifier_field_names)
            if field_name in columns
        )
        object_id_fields = tuple(
            dict.fromkeys(
                (
                    *((object_id_field,) if object_id_field is not None else ()),
                    *MeasurementRowAxisField.object_id_field_names(),
                )
            )
        )
        identity_field_names = (
            self.row_identity_contract.selected_image_identity_fields(
                frozenset(
                    normalize_runtime_identifier(field_name) for field_name in columns
                )
            )
        )
        identity_columns = tuple(
            (field_name, values)
            for field_name, values in columns.items()
            if normalize_runtime_identifier(field_name) in identity_field_names
        )
        feature_columns = wide_measurement_feature_columns(
            columns,
            object_id_field=object_id_field,
            qualifier_field_names=qualifier_fields,
        )
        source_values = columns.get(MeasurementRowAxisField.SOURCE_IMAGE_NAME.value)
        object_name_values = columns.get(MeasurementRowAxisField.OBJECT_NAME.value)
        object_id_columns = tuple(
            columns[field_name]
            for field_name in object_id_fields
            if field_name in columns
        )
        feature_field_columns = tuple(
            columns[field_name]
            for field_name in feature_fields
            if field_name in columns
        )
        value_field_columns = tuple(
            columns[field_name] for field_name in value_fields if field_name in columns
        )
        qualifier_columns = tuple(
            (field_name, columns[field_name]) for field_name in qualifier_fields
        )

        if feature_field_columns:
            if not value_field_columns:
                raise ValueError("Long-form measurement columns have no value column.")
            feature_values = feature_field_columns[0]
            measurement_values = value_field_columns[0]
            subject_cache: dict[object, str] = {}
            source_cache: dict[object, str | None] = {}
            projected_feature_cache: dict[
                tuple[str, tuple[tuple[str, object], ...]],
                str,
            ] = {}
            for row_index in range(row_count):
                row_subject = default_subject
                row_owned = False
                if object_name_values is not None:
                    object_name = object_name_values[row_index]
                    if not is_structural_missing_measurement_cell(object_name):
                        if object_name in subject_cache:
                            row_subject = subject_cache[object_name]
                        elif object_name is not None:
                            normalized_object_name = str(object_name).strip()
                            if normalized_object_name:
                                row_subject = normalized_object_name
                                subject_cache[object_name] = row_subject
                        row_owned = row_subject != default_subject
                identity_values = tuple(
                    values[row_index] for _field_name, values in identity_columns
                )
                object_label = None
                for values in object_id_columns:
                    object_label = measurement_object_label_value(values[row_index])
                    if object_label is not None:
                        break
                identity = (
                    *identity_values,
                    object_label if object_label is not None else self._absent_identity,
                )
                if all(value is self._absent_identity for value in identity):
                    identity = (self._passthrough_identity, self._passthrough_index)
                    self._passthrough_index += 1
                subject_rows = self._rows.setdefault(row_subject, {})
                target = subject_rows.get(identity)
                if target is None:
                    target = {
                        field_name: value
                        for (field_name, _values), value in zip(
                            identity_columns,
                            identity_values,
                            strict=True,
                        )
                    }
                    if object_label is not None:
                        target[MeasurementRowAxisField.OBJECT_LABEL.value] = (
                            object_label
                        )
                    subject_rows[identity] = target
                    self._order.setdefault(row_subject, []).append(identity)

                row_source_name = source_image_name
                if source_values is not None:
                    source_value = source_values[row_index]
                    if source_value in source_cache:
                        row_source_name = source_cache[source_value]
                    elif not is_structural_missing_measurement_cell(source_value):
                        if source_value is not None:
                            normalized_source = str(source_value).strip()
                            if normalized_source:
                                row_source_name = normalized_source
                        source_cache[source_value] = row_source_name
                qualifier_values = (
                    tuple(
                        (field_name, value)
                        for field_name, values in qualifier_columns
                        for value in (values[row_index],)
                        if not is_structural_missing_measurement_cell(value)
                    )
                    if qualifier_columns
                    else ()
                )
                row_scope = MeasurementRowOwnership(
                    object_name=row_subject if row_owned else None,
                    source_image_name=row_source_name,
                ).scope(default_scope)
                if (
                    row_scope is MeasurementScope.OBJECT
                    and row_subject not in self._object_subjects
                ):
                    self._object_subjects.append(row_subject)
                feature_value = feature_values[row_index]
                if is_structural_missing_measurement_cell(feature_value):
                    for field_name, values in feature_columns:
                        value = values[row_index]
                        if is_structural_missing_measurement_cell(value):
                            continue
                        self._assign(
                            target,
                            identity,
                            project_feature_name(
                                field_name,
                                qualifier_values,
                            ),
                            value,
                            missing_cell,
                        )
                    continue
                feature_name = (
                    feature_value
                    if isinstance(feature_value, str)
                    else str(feature_value)
                )
                if not feature_name:
                    raise ValueError(
                        "Long-form measurement row has an empty feature name."
                    )
                projection_key = (
                    feature_name,
                    qualifier_values,
                )
                projected_feature = projected_feature_cache.get(projection_key)
                if projected_feature is None:
                    projected_feature = project_feature_name(*projection_key)
                    projected_feature_cache[projection_key] = projected_feature
                for field_name, values in feature_columns:
                    value = values[row_index]
                    if not is_structural_missing_measurement_cell(value):
                        self._assign(target, identity, field_name, value, missing_cell)
                measurement_value = measurement_values[row_index]
                if is_structural_missing_measurement_cell(measurement_value):
                    raise ValueError(
                        f"Long-form measurement feature {feature_name!r} has no value."
                    )
                self._assign(
                    target,
                    identity,
                    projected_feature,
                    measurement_value,
                    missing_cell,
                )
            return

        for row_index in range(row_count):
            row_subject = default_subject
            row_owned = False
            if object_name_values is not None:
                object_name = object_name_values[row_index]
                if (
                    not is_structural_missing_measurement_cell(object_name)
                    and object_name is not None
                ):
                    normalized_object_name = str(object_name).strip()
                    if normalized_object_name:
                        row_subject = normalized_object_name
                        row_owned = True
            identity_values = tuple(
                (
                    self._absent_identity
                    if is_structural_missing_measurement_cell(values[row_index])
                    else values[row_index]
                )
                for _field_name, values in identity_columns
            )
            object_label = None
            for values in object_id_columns:
                value = values[row_index]
                if is_structural_missing_measurement_cell(value):
                    continue
                object_label = measurement_object_label_value(value)
                if object_label is not None:
                    break
            identity = (
                *identity_values,
                object_label if object_label is not None else self._absent_identity,
            )
            if all(value is self._absent_identity for value in identity):
                identity = (self._passthrough_identity, self._passthrough_index)
                self._passthrough_index += 1
            subject_rows = self._rows.setdefault(row_subject, {})
            target = subject_rows.get(identity)
            if target is None:
                target = {
                    field_name: value
                    for (field_name, _values), value in zip(
                        identity_columns,
                        identity_values,
                        strict=True,
                    )
                    if value is not self._absent_identity
                }
                if object_label is not None:
                    target[MeasurementRowAxisField.OBJECT_LABEL.value] = object_label
                subject_rows[identity] = target
                self._order.setdefault(row_subject, []).append(identity)

            row_source_name = source_image_name
            if source_values is not None:
                source_value = source_values[row_index]
                if (
                    not is_structural_missing_measurement_cell(source_value)
                    and source_value is not None
                ):
                    normalized_source = str(source_value).strip()
                    if normalized_source:
                        row_source_name = normalized_source
            qualifier_values = (
                tuple(
                    (field_name, value)
                    for field_name, values in qualifier_columns
                    for value in (values[row_index],)
                    if not is_structural_missing_measurement_cell(value)
                )
                if qualifier_columns
                else ()
            )
            row_scope = MeasurementRowOwnership(
                object_name=row_subject if row_owned else None,
                source_image_name=row_source_name,
            ).scope(default_scope)
            if (
                row_scope is MeasurementScope.OBJECT
                and row_subject not in self._object_subjects
            ):
                self._object_subjects.append(row_subject)
            for field_name, values in feature_columns:
                value = values[row_index]
                if is_structural_missing_measurement_cell(value):
                    continue
                self._assign(
                    target,
                    identity,
                    project_feature_name(
                        field_name,
                        qualifier_values,
                    ),
                    value,
                    missing_cell,
                )

    def row_mappings_by_subject(
        self,
    ) -> dict[str, tuple[Mapping[str, object], ...]]:
        """Return final rows in first-seen subject and identity order."""
        return {
            subject: tuple(
                MappingProxyType(self._rows[subject][identity])
                for identity in identities
            )
            for subject, identities in self._order.items()
        }

    def object_subjects(self) -> tuple[str, ...]:
        """Return subjects carrying object-scoped measurement rows."""
        return tuple(self._object_subjects)

    @staticmethod
    def _columnar_rows(
        rows: Sequence[object],
        missing_cell: object,
    ) -> ColumnarRows:
        del missing_cell
        if not rows:
            return MeasurementSparseColumnarRows(
                MappingProxyType({}),
                fields=(),
            )
        row_type = type(rows[0])
        if is_dataclass(row_type):
            return DataclassMeasurementColumnarRows(rows, row_type=row_type)
        raise TypeError(
            "Measurement row mappings require an explicit schema-bearing "
            "ColumnarRows carrier."
        )

    @staticmethod
    def _assign(
        target: MutableMapping[str, object],
        identity: tuple[object, ...],
        field_name: str,
        value: object,
        missing_cell: object,
    ) -> None:
        existing = target.get(field_name, missing_cell)
        if existing is not missing_cell and not _measurement_sparse_cell_values_equal(
            existing,
            value,
        ):
            raise ValueError(
                "Conflicting sparse measurement values for row identity "
                f"{identity!r}, field {field_name!r}: {existing!r} vs {value!r}."
            )
        target[field_name] = value


def _measurement_sparse_cell_values_equal(left: object, right: object) -> bool:
    """Return scalar truth for equality without treating arrays as ambiguous."""
    if left is right:
        return True
    try:
        if np.array_equal(
            np.asarray(left),
            np.asarray(right),
            equal_nan=True,
        ):
            return True
    except (TypeError, ValueError):
        pass
    try:
        equality = left == right
    except Exception:
        return False
    if isinstance(equality, bool):
        return equality
    if hasattr(equality, "all"):
        try:
            return bool(equality.all())
        except Exception:
            return False
    try:
        return bool(equality)
    except Exception:
        return False


class MeasurementRowsAxisProjection(
    NominalTypeStrategyFamilyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project measurement rows along the OpenHCS runtime slice axis."""

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[object] | ColumnarRows,
    ) -> "MeasurementRowsAxisProjection":
        projection_type = cls.strategy_types_for_nominal_value(rows)[0]
        return cast(MeasurementRowsAxisProjection, projection_type(rows=rows))

    @staticmethod
    def row_has_axis(row: Mapping[str, object]) -> bool:
        return MeasurementRowAxisField.SLICE_INDEX.value in row

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

    @abstractmethod
    def declares_axis_field(self, axis: MeasurementRowAxisField) -> bool:
        """Return whether the row representation declares ``axis``."""

    @abstractmethod
    def has_axisless_rows(self, axis: MeasurementRowAxisField) -> bool:
        """Return whether any represented row omits an exact ``axis`` value."""

    @property
    def has_axis(self) -> bool:
        """Return whether rows declare the runtime slice coordinate."""
        return self.declares_axis_field(MeasurementRowAxisField.SLICE_INDEX)

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
    def remap_runtime_slice_indices(
        self,
        values: Mapping[int, int],
        *,
        axisless_value: int | None = None,
    ) -> Sequence[object] | ColumnarRows:
        """Return rows with declared runtime-slice values remapped exactly."""


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

    def declares_axis_field(self, axis: MeasurementRowAxisField) -> bool:
        return any(axis.value in measurement_row_mapping(row) for row in self.rows)

    def has_axisless_rows(self, axis: MeasurementRowAxisField) -> bool:
        return any(
            measurement_axis_integer_value(
                measurement_row_mapping(row).get(axis.value),
                axis,
            )
            is None
            for row in self.rows
        )

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

    def remap_runtime_slice_indices(
        self,
        values: Mapping[int, int],
        *,
        axisless_value: int | None = None,
    ) -> Sequence[object]:
        slice_index_field = MeasurementRowAxisField.SLICE_INDEX.value
        projected_rows: list[object] = []
        for row in self.rows:
            row_mapping = measurement_row_mapping(row)
            slice_index = measurement_axis_integer_value(
                row_mapping.get(slice_index_field),
                MeasurementRowAxisField.SLICE_INDEX,
            )
            if slice_index is None:
                if axisless_value is None:
                    projected_rows.append(row)
                    continue
                projected_row = dict(row_mapping)
                projected_row[slice_index_field] = int(axisless_value)
                projected_rows.append(projected_row)
                continue
            if slice_index not in values:
                raise ValueError(
                    "Measurement row runtime-slice remapping has no value for "
                    f"slice_index={slice_index}."
                )
            projected_index = int(values[slice_index])
            if (
                is_dataclass(row)
                and slice_index_field in row_mapping
                and slice_index_field in dataclass_init_field_names(type(row))
            ):
                projected_rows.append(
                    dataclass_replace(row, **{slice_index_field: projected_index})
                )
                continue
            projected_row = dict(row_mapping)
            projected_row[slice_index_field] = projected_index
            projected_rows.append(projected_row)
        return projected_rows

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


def dataclass_init_field_names(row_type: type[object]) -> frozenset[str]:
    """Return constructor-backed dataclass field names for row replacement."""
    return frozenset(field.name for field in dataclass_fields(row_type) if field.init)


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
            isinstance(column, str) for column in columns
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

    def declares_axis_field(self, axis: MeasurementRowAxisField) -> bool:
        return axis.value in self.columns

    def has_axisless_rows(self, axis: MeasurementRowAxisField) -> bool:
        values = self.columns.get(axis.value)
        if values is None:
            return self.has_rows
        return any(
            measurement_axis_integer_value(value, axis) is None for value in values
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
            fields=projected_columnar_fields(
                self.rows,
                FieldSpec(slice_index_field, int),
            ),
            declared_object_measurement_domain_covered=(
                self.rows.covers_declared_object_measurement_domain
            ),
            object_row_identity=self.rows.object_row_identity,
        )

    def remap_runtime_slice_indices(
        self,
        values: Mapping[int, int],
        *,
        axisless_value: int | None = None,
    ) -> ColumnarRows:
        slice_index_field = MeasurementRowAxisField.SLICE_INDEX.value
        if slice_index_field not in self.columns:
            if axisless_value is None:
                return self.rows
            return self.project_runtime_slice_index(axisless_value)
        projected_values = []
        for value in self.columns[slice_index_field]:
            if is_structural_missing_measurement_cell(value):
                projected_values.append(
                    value if axisless_value is None else int(axisless_value)
                )
                continue
            slice_index = measurement_axis_integer_value(
                value,
                MeasurementRowAxisField.SLICE_INDEX,
            )
            if slice_index is None:
                if axisless_value is None:
                    projected_values.append(value)
                    continue
                projected_values.append(int(axisless_value))
                continue
            if slice_index not in values:
                raise ValueError(
                    "Measurement row runtime-slice remapping has no value for "
                    f"slice_index={slice_index!r}."
                )
            projected_values.append(int(values[slice_index]))
        return MeasurementProjectedColumnarRows(
            ColumnarRowColumnOverlay(
                self.columns,
                MappingProxyType({slice_index_field: tuple(projected_values)}),
            ),
            fields=projected_columnar_fields(
                self.rows,
                FieldSpec(slice_index_field, int),
            ),
            declared_object_measurement_domain_covered=(
                self.rows.covers_declared_object_measurement_domain
            ),
            object_row_identity=self.rows.object_row_identity,
        )


def measurement_row_object_name(row: Mapping[str, object]) -> str | None:
    """Return the object owner encoded on one measurement row."""
    return cast(str | None, MeasurementRowObjectName.value_from_row(row))


def measurement_row_source_image_name(row: Mapping[str, object]) -> str | None:
    """Return the source-image owner encoded on one measurement row."""
    return cast(str | None, MeasurementRowSourceImageName.value_from_row(row))


@dataclass(frozen=True, slots=True)
class MeasurementObjectLabelResolution:
    """Integer object label resolved from runtime/CSV scalar encodings."""

    value: object

    @property
    def object_label(self) -> int | None:
        return measurement_object_label_value(self.value)


def measurement_object_label_value(value: object) -> int | None:
    """Return the integer object label represented by one scalar value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        integer = int(value)
        return integer if float(integer) == float(value) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        signless = stripped[1:] if stripped[:1] in ("+", "-") else stripped
        if signless.isdecimal():
            return int(stripped)
    return MeasurementScalarLiteral(value).integer_value


def measurement_object_label(
    row: Mapping[str, object],
    *,
    object_id_field: str | None = None,
) -> int | None:
    """Return the resolved object label encoded on a measurement row."""
    return cast(
        int | None,
        MeasurementRowObjectLabel.value_from_row(
            row,
            object_id_field=object_id_field,
        ),
    )


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
    if any(
        field in row for field in MeasurementRowAxisField.feature_name_field_names()
    ) and any(field in row for field in MeasurementRowValueField.field_names()):
        return True
    normalized_fields = frozenset(
        normalize_runtime_identifier(str(field)) for field in row
    )
    return bool(
        normalized_fields
        & MeasurementRowAxisField.normalized_feature_name_field_names()
    ) and bool(normalized_fields & MeasurementRowValueField.normalized_field_names())


def measurement_row_identity_role(
    row: Mapping[str, object],
) -> MeasurementObjectRowIdentity | None:
    """Return the explicit OpenHCS row-identity role encoded on a measurement row."""
    return cast(
        MeasurementObjectRowIdentity | None,
        MeasurementRowObjectIdentityRole.value_from_row(row),
    )


def measurement_row_field_value(
    row: Mapping[str, object],
    field_name: str,
) -> object | None:
    """Return a row value by normalized measurement field name."""
    if field_name in row:
        return row[field_name]
    normalized_target = normalize_runtime_identifier(field_name)
    field = normalized_measurement_row_fields_for_row(row).get(normalized_target)
    return None if field is None else row[field]


def measurement_row_declared_field_value(
    row: Mapping[str, object],
    field_name: str,
    normalized_fields: Mapping[str, str] | None,
) -> object | None:
    """Return a declared row value using cached normalized fields when available."""
    resolved_field = field_name if field_name in row else None
    if resolved_field is None:
        if normalized_fields is None:
            normalized_fields = normalized_measurement_row_fields_for_row(row)
        resolved_field = normalized_fields.get(normalize_runtime_identifier(field_name))
    if resolved_field is None:
        return None
    value = row[resolved_field]
    return None if is_structural_missing_measurement_cell(value) else value


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

    def field_spec(self) -> FieldSpec:
        """Return the output field declared by this string qualifier."""
        return FieldSpec(self.field_name, str)


@dataclass(frozen=True, slots=True)
class MeasurementRowOwnership:
    """Shared object/source ownership qualifiers for measurement rows."""

    object_name: str | None = None
    source_image_name: str | None = None

    def scope(self, default: MeasurementScope) -> MeasurementScope:
        """Return the semantic row scope declared by this ownership."""

        if self.object_name is not None:
            return MeasurementScope.OBJECT
        if default is not MeasurementScope.ARTIFACT:
            return default
        if self.source_image_name is not None:
            return MeasurementScope.IMAGE
        return default

    @staticmethod
    def rows_declare_source_image(field_names: Iterable[str]) -> bool:
        """Return whether row fields own source-image qualification."""

        return MeasurementRowAxisField.SOURCE_IMAGE_NAME.value in field_names

    @property
    def qualifiers(self) -> tuple[MeasurementRowQualifier, ...]:
        return tuple(
            qualifier
            for qualifier in (
                MeasurementRowQualifier.optional(
                    field_name=MeasurementRowAxisField.OBJECT_NAME.value,
                    value=self.object_name,
                ),
                MeasurementRowQualifier.optional(
                    field_name=MeasurementRowAxisField.SOURCE_IMAGE_NAME.value,
                    value=self.source_image_name,
                ),
            )
            if qualifier is not None
        )

    def annotate_rows(
        self, rows: Sequence[object] | ColumnarRows
    ) -> Sequence[object] | ColumnarRows:
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
    _fields: tuple[FieldSpec, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    object_row_identity: MeasurementObjectRowIdentity | None = field(
        default=None,
        init=False,
    )

    columns: ClassVar[AliasProperty[Mapping[str, Sequence[object]]]] = AliasProperty(
        "_columns"
    )
    fields: ClassVar[AliasProperty[tuple[FieldSpec, ...]]] = AliasProperty("_fields")

    def __len__(self) -> int:
        return column_mapping_row_count(self._columns)

    def iter_row_mappings(self) -> Iterator[Mapping[str, object]]:
        columns = tuple(str(column) for column in self.columns)
        column_values = tuple(self.column_values(column) for column in columns)
        for values in zip(*column_values, strict=True):
            yield {
                column: value
                for column, value in zip(columns, values, strict=True)
                if not is_structural_missing_measurement_cell(value)
            }


def columnar_row_count(rows: ColumnarRows) -> int:
    """Return row count for a nominal columnar payload."""
    return rows.row_count()


@dataclass(frozen=True, slots=True)
class ConcatenatedColumnarRowColumns(Mapping[str, Sequence[object]]):
    """Lazy mapping over columns concatenated from multiple columnar batches."""

    row_batches: tuple[ColumnarRows, ...]
    column_names: tuple[str, ...]
    _batch_columns: tuple[Mapping[str, object], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _batch_row_counts: tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _column_cache: dict[str, Sequence[object]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_batch_columns",
            tuple(
                MappingProxyType(
                    {str(column): column for column in row_batch.columns}
                )
                for row_batch in self.row_batches
            ),
        )
        object.__setattr__(
            self,
            "_batch_row_counts",
            tuple(columnar_row_count(row_batch) for row_batch in self.row_batches),
        )

    @classmethod
    def from_row_batches(
        cls,
        row_batches: tuple[ColumnarRows, ...],
        fields: tuple[FieldSpec, ...],
    ) -> "ConcatenatedColumnarRowColumns":
        return cls(
            row_batches=row_batches,
            column_names=tuple(field_spec.name for field_spec in fields),
        )

    def __getitem__(self, column_name: str) -> Sequence[object]:
        if column_name not in self.column_names:
            raise KeyError(column_name)
        cached = self._column_cache.get(column_name)
        if cached is not None:
            return cached
        batch_column_keys = tuple(
            batch_columns.get(column_name) for batch_columns in self._batch_columns
        )
        if all(column_key is not None for column_key in batch_column_keys):
            values = np.concatenate(
                tuple(
                    columnar_row_values(row_batch, column_key)
                    for row_batch, column_key in zip(
                        self.row_batches,
                        batch_column_keys,
                        strict=True,
                    )
                )
            )
        else:
            values = np.empty(sum(self._batch_row_counts), dtype=object)
            values.fill(MEASUREMENT_SPARSE_CELL)
            row_offset = 0
            for row_batch, row_count, column_key in zip(
                self.row_batches,
                self._batch_row_counts,
                batch_column_keys,
                strict=True,
            ):
                if column_key is not None:
                    values[row_offset : row_offset + row_count] = columnar_row_values(
                        row_batch,
                        column_key,
                    )
                row_offset += row_count
        self._column_cache[column_name] = values
        return values

    def __iter__(self):
        return iter(self.column_names)

    def __len__(self) -> int:
        return len(self.column_names)


@dataclass(slots=True)
class ConcatenatedColumnarRows(MeasurementColumnarRowsView):
    """Columnar table view over multiple columnar row batches."""

    row_batches: tuple[ColumnarRows, ...]

    def __post_init__(self) -> None:
        self.object_row_identity = ColumnarRows.common_object_row_identity(
            self.row_batches
        )
        self._fields = FieldSpec.merge_exact(
            (row_batch.fields for row_batch in self.row_batches),
            context="concatenated columnar row fields",
        )
        self._columns = ConcatenatedColumnarRowColumns.from_row_batches(
            self.row_batches,
            self._fields,
        )
        self.validate_fields()

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
        return tuple(self.iter_row_mappings())

    def iter_row_mappings(self):
        for row_batch in self.row_batches:
            yield from row_batch.iter_row_mappings()


@dataclass(frozen=True, slots=True)
class ConcatenatedMeasurementRowsAxisProjection(ColumnarMeasurementRowsAxisProjection):
    """Project a concatenated table without erasing its batch schemas."""

    value_type = ConcatenatedColumnarRows
    rows: ConcatenatedColumnarRows

    def project_runtime_slice_index(
        self,
        slice_index: int,
    ) -> ConcatenatedColumnarRows:
        return ConcatenatedColumnarRows(
            tuple(
                MeasurementRowsAxisProjection.from_rows(
                    row_batch
                ).project_runtime_slice_index(slice_index)
                for row_batch in self.rows.row_batches
            )
        )

    def remap_runtime_slice_indices(
        self,
        values: Mapping[int, int],
        *,
        axisless_value: int | None = None,
    ) -> ConcatenatedColumnarRows:
        return ConcatenatedColumnarRows(
            tuple(
                MeasurementRowsAxisProjection.from_rows(
                    row_batch
                ).remap_runtime_slice_indices(
                    values,
                    axisless_value=axisless_value,
                )
                for row_batch in self.rows.row_batches
            )
        )


@dataclass(slots=True)
class DataclassMeasurementColumnarRows(ColumnarRows):
    """Columnar view over homogeneous dataclass measurement rows."""

    rows: Sequence[object]
    row_type: type[object] | None = None
    _columns: Mapping[str, Sequence[object]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    columns: ClassVar[AliasProperty[Mapping[str, Sequence[object]]]] = AliasProperty(
        "_columns"
    )
    _fields: tuple[FieldSpec, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    fields: ClassVar[AliasProperty[tuple[FieldSpec, ...]]] = AliasProperty("_fields")

    def __post_init__(self) -> None:
        row_type = self.row_type
        if row_type is None:
            if not self.rows:
                raise TypeError(
                    "DataclassMeasurementColumnarRows requires row_type for zero rows."
                )
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
        self.row_type = row_type
        self._fields = FieldSpec.from_dataclass_type(row_type)
        column_names = tuple(field_spec.name for field_spec in self._fields)
        row_mappings = tuple(measurement_row_mapping(row) for row in self.rows)
        self._columns = {
            column_name: tuple(row[column_name] for row in row_mappings)
            for column_name in column_names
        }
        self.validate_fields()

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        yield from self.iter_row_mappings()

    def iter_row_mappings(self):
        columns = self._columns
        for row_index in range(len(self)):
            yield {
                field_name: values[row_index] for field_name, values in columns.items()
            }


@dataclass(slots=True)
class QualifiedMeasurementColumnarRows(MeasurementColumnarRowsView):
    """Columnar measurement rows with table-ownership qualifiers attached."""

    rows: ColumnarRows
    qualifiers: tuple[MeasurementRowQualifier, ...]

    def __post_init__(self) -> None:
        self.object_row_identity = self.rows.object_row_identity
        columns = dict(self.rows.columns)
        row_count = column_mapping_row_count(columns)
        qualifier_fields = tuple(
            qualifier.field_spec() for qualifier in self.qualifiers
        )
        self._fields = FieldSpec.merge_exact(
            (self.rows.fields, qualifier_fields),
            context="qualified columnar row fields",
        )
        for qualifier in self.qualifiers:
            columns[qualifier.field_name] = (qualifier.value,) * row_count
        self._columns = columns
        self.validate_fields()

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        """Return whether the owned row carrier still covers its object domain."""
        return self.rows.covers_declared_object_measurement_domain

    def __iter__(self):
        yield from self.iter_row_mappings()

    def iter_row_mappings(self):
        columns = self._columns
        for row_index in range(len(self)):
            yield {
                field_name: values[row_index] for field_name, values in columns.items()
            }


class MeasurementTableRowLayout(str, Enum):
    """Nominal row layout for measurement tables."""

    EMPTY = "empty"
    LONG = "long"
    WIDE = "wide"

    @classmethod
    def for_row(cls, row: object) -> "MeasurementTableRowLayout":
        """Classify one row from its declared measurement fields."""
        field_names = frozenset(
            str(field_name) for field_name in measurement_row_mapping(row)
        )
        has_feature_field = bool(
            field_names & MeasurementRowAxisField.feature_name_field_names()
        )
        has_value_field = bool(
            field_names & MeasurementRowValueField.field_names()
        )
        if has_feature_field and not has_value_field:
            raise ValueError(
                "Long-form measurement rows must declare both a feature field "
                f"and a value field, got fields {sorted(field_names)!r}."
            )
        return cls.LONG if has_feature_field else cls.WIDE


class MeasurementRowLayoutProjectionStrategy(
    EnumKeyedStrategyMixin[MeasurementTableRowLayout],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project one nominal measurement row layout into canonical long form."""

    __enum_member_attr__ = "layout"
    layout: ClassVar[MeasurementTableRowLayout | None] = None

    @abstractmethod
    def long_rows(self, row: object) -> tuple[Mapping[str, object], ...]:
        """Return canonical long-form rows for one source row."""


class LongMeasurementRowProjectionStrategy(MeasurementRowLayoutProjectionStrategy):
    """Preserve already-long rows."""

    layout = MeasurementTableRowLayout.LONG

    def long_rows(self, row: object) -> tuple[Mapping[str, object], ...]:
        return (measurement_row_mapping(row),)


class WideMeasurementRowProjectionStrategy(MeasurementRowLayoutProjectionStrategy):
    """Explode wide feature columns into canonical long-form rows."""

    layout = MeasurementTableRowLayout.WIDE

    def long_rows(self, row: object) -> tuple[Mapping[str, object], ...]:
        row_mapping = measurement_row_mapping(row)
        axis_fields = MeasurementRowAxisField.field_names()
        axis_values = {
            str(field_name): value
            for field_name, value in row_mapping.items()
            if str(field_name) in axis_fields
        }
        long_rows: list[Mapping[str, object]] = []
        for field_name, value in row_mapping.items():
            field_text = str(field_name)
            if field_text in axis_fields:
                continue
            long_row = dict(axis_values)
            long_row[MeasurementRowAxisField.FEATURE_NAME.value] = field_text
            long_row[MeasurementRowValueField.RESULT_VALUE.value] = value
            long_rows.append(long_row)
        return tuple(long_rows)


def measurement_row_semantic_field_names() -> frozenset[str]:
    """Return fields that identify a payload as a measurement row."""
    return MeasurementRowAxisField.field_names() | MeasurementRowValueField.field_names()


def carries_measurement_row_semantics(row: object) -> bool:
    """Return whether a row-like object declares measurement-row fields."""
    semantic_fields = measurement_row_semantic_field_names()
    if isinstance(row, Mapping):
        field_names = frozenset((str(field_name) for field_name in row.keys()))
    elif is_dataclass(row):
        field_names = frozenset((field.name for field in dataclass_fields(row)))
    elif type(row).__dictoffset__ != 0:
        field_names = frozenset((str(field_name) for field_name in vars(row).keys()))
    else:
        return False
    return bool(field_names & semantic_fields)


def measurement_table_row_layout(rows: object) -> MeasurementTableRowLayout:
    """Return the declared layout implied by a table row payload."""
    observed_layouts = measurement_table_row_layouts(rows)
    if not observed_layouts:
        return MeasurementTableRowLayout.EMPTY
    if len(observed_layouts) != 1:
        raise ValueError(
            f"MeasurementTable rows must not mix long-form and wide-form layouts; got {sorted((layout.value for layout in observed_layouts))!r}."
        )
    return next(iter(observed_layouts))


def measurement_table_row_layout_from_fields(
    fields: Iterable[FieldSpec],
) -> MeasurementTableRowLayout | None:
    """Return row layout declared by table fields when fields are authoritative."""
    return _measurement_table_row_layout_from_field_names(
        tuple((field.name for field in fields))
    )


@lru_cache(maxsize=256)
def _measurement_table_row_layout_from_field_names(
    field_names_tuple: tuple[str, ...],
) -> MeasurementTableRowLayout | None:
    """Return row layout declared by field names."""
    field_names = frozenset(field_names_tuple)
    if not field_names:
        return None
    has_feature_field = bool(
        field_names & MeasurementRowAxisField.feature_name_field_names()
    )
    has_value_field = bool(field_names & MeasurementRowValueField.field_names())
    if has_feature_field and (not has_value_field):
        raise ValueError(
            f"Long-form measurement table fields must declare both a feature field and a value field, got fields {sorted(field_names)!r}."
        )
    return (
        MeasurementTableRowLayout.LONG
        if has_feature_field
        else MeasurementTableRowLayout.WIDE
    )


def measurement_table_row_layouts(rows: object) -> frozenset[MeasurementTableRowLayout]:
    """Return every nominal row layout observed in a measurement payload."""
    if rows is None:
        return frozenset()
    row_sequence = rows if isinstance(rows, list | tuple) else (rows,)
    if not row_sequence:
        return frozenset()
    if isinstance(row_sequence[0], ObjectMeasurementValueRow) and all(
        (isinstance(row, ObjectMeasurementValueRow) for row in row_sequence)
    ):
        return frozenset((MeasurementTableRowLayout.LONG,))
    return frozenset(
        (MeasurementTableRowLayout.for_row(row) for row in row_sequence)
    )


def normalize_measurement_table_rows(
    rows: object, *, fields: Iterable[FieldSpec] = ()
) -> object:
    """Return homogeneous measurement rows, canonicalizing mixed tables to long form."""
    declared_layout = measurement_table_row_layout_from_fields(fields)
    if declared_layout is not None:
        return rows
    observed_layouts = measurement_table_row_layouts(rows)
    if len(observed_layouts) <= 1:
        return rows
    return measurement_rows_as_layout(rows, MeasurementTableRowLayout.LONG)


def measurement_rows_as_layout(
    rows: object, layout: MeasurementTableRowLayout
) -> object:
    """Project measurement rows into a declared table layout."""
    if layout is not MeasurementTableRowLayout.LONG:
        raise ValueError(
            f"Unsupported measurement row layout projection: {layout.value}."
        )
    row_sequence = rows if isinstance(rows, list | tuple) else (rows,)
    return [
        projected_row
        for row in row_sequence
        for projected_row in MeasurementRowLayoutProjectionStrategy.for_enum_member(
            MeasurementTableRowLayout.for_row(row)
        ).long_rows(row)
    ]
