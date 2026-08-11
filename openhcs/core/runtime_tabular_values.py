"""Generic columnar and tabular runtime payload values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import asdict, dataclass, fields as dataclass_fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from openhcs.core.process_local_cache import RegisteredProcessLocalBoundedCache


class ColumnarRows(ABC):
    """Nominal ABC for schema-bearing table payloads exposing named columns."""

    object_row_identity: MeasurementObjectRowIdentity | None = None

    @staticmethod
    def common_object_row_identity(
        row_batches: Iterable["ColumnarRows"],
    ) -> MeasurementObjectRowIdentity | None:
        """Return the single object-row identity declared by columnar batches."""
        identities = tuple(
            dict.fromkeys(
                row_batch.object_row_identity
                for row_batch in row_batches
                if row_batch.object_row_identity is not None
            )
        )
        if len(identities) > 1:
            raise ValueError(
                "Columnar measurement batches declare conflicting object-row "
                f"identities {identities!r}."
            )
        return identities[0] if identities else None

    @property
    @abstractmethod
    def columns(self) -> Any: ...

    @property
    @abstractmethod
    def fields(self) -> tuple[FieldSpec, ...]:
        """Return exact field declarations in physical column order."""

    def validate_fields(self) -> None:
        """Require exact ordered agreement between field and column names."""
        fields = self.fields
        if not isinstance(fields, tuple):
            raise TypeError(
                f"{type(self).__name__}.fields must be a tuple of FieldSpec values."
            )
        if not all(isinstance(field, FieldSpec) for field in fields):
            raise TypeError(
                f"{type(self).__name__}.fields must contain only FieldSpec values."
            )
        field_names = tuple(field.name for field in fields)
        if len(field_names) != len(set(field_names)):
            raise ValueError(
                f"{type(self).__name__}.fields contains duplicate names: "
                f"{field_names!r}."
            )
        column_names = tuple(self.columns)
        if field_names != column_names:
            raise ValueError(
                f"{type(self).__name__} field/column names and order must match "
                f"exactly: fields={field_names!r}, columns={column_names!r}."
            )

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        """Return whether this payload already spans its declared object domain."""
        return False

    def column_values(self, column: str) -> Sequence[object]:
        """Return one named column from this nominal columnar payload."""
        columns = self.columns
        if isinstance(columns, Mapping):
            return columns[column]
        return self[column]

    def row_count(self) -> int:
        """Return the number of rows represented by this columnar payload."""
        columns = self.columns
        if not columns:
            return 0
        if isinstance(columns, Mapping):
            return len(next(iter(columns.values())))
        first_column = next(iter(columns))
        return len(self.column_values(first_column))

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        """Return row-wise mappings for this columnar payload."""
        return tuple(self.iter_row_mappings())

    def iter_row_mappings(self) -> Iterable[Mapping[str, object]]:
        """Yield row-wise mappings for this columnar payload."""
        columns = tuple(str(column) for column in self.columns)
        column_values = tuple(self.column_values(column) for column in columns)
        for values in zip(*column_values, strict=True):
            yield dict(zip(columns, values, strict=True))


def is_table_payload(data: Any) -> bool:
    """Return whether data is a nominal schema-bearing table payload."""
    return isinstance(data, ColumnarRows)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One named field expected in a tabular runtime value."""

    name: str
    """Exact field or column name carried by the tabular value."""

    dtype: type[object] | str | None = None
    """Optional scalar Python type or external dtype label declared for the field."""

    required: bool = True
    """Whether every validated row must provide a non-null value for this field."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Runtime value field name cannot be empty.")

    def coerce_scalar(self, value: object) -> object:
        """Coerce one scalar through this field's concrete Python dtype."""
        if value is None:
            if self.required:
                raise ValueError(f"Required field {self.name!r} cannot be None.")
            return None
        if self.dtype is None:
            return value
        if self.dtype not in (str, int, float, bool):
            raise TypeError(
                f"Field {self.name!r} does not declare a coercible scalar dtype: "
                f"{self.dtype!r}."
            )
        if self.dtype is bool and not isinstance(value, bool):
            raise TypeError(
                f"Boolean field {self.name!r} requires a bool value, got {value!r}."
            )
        try:
            return self.dtype(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Field {self.name!r} cannot convert {value!r} to "
                f"{self.dtype.__name__}."
            ) from exc

    @classmethod
    def from_annotation(cls, name: str, annotation: object) -> "FieldSpec":
        """Build one field declaration from a resolved type annotation."""
        return cls(
            name=name,
            dtype=cls.annotation_dtype(annotation),
            required=cls.annotation_required(annotation),
        )

    @classmethod
    def from_dataclass_type(
        cls,
        row_type: type[object],
    ) -> tuple["FieldSpec", ...]:
        """Return fields declared by one nominal dataclass row type."""
        if not isinstance(row_type, type) or not is_dataclass(row_type):
            type_name = (
                row_type.__name__
                if isinstance(row_type, type)
                else type(row_type).__name__
            )
            raise TypeError(f"Columnar row type must be a dataclass, got {type_name}.")
        annotations = get_type_hints(row_type, include_extras=True)
        return tuple(
            cls.from_annotation(row_field.name, annotations[row_field.name])
            for row_field in dataclass_fields(row_type)
        )

    @classmethod
    def annotation_dtype(cls, annotation: object) -> type[object] | str | None:
        """Return the scalar dtype declared by one resolved annotation."""
        origin = get_origin(annotation)
        if origin is Annotated:
            declared_type, *_metadata = get_args(annotation)
            return cls.annotation_dtype(declared_type)
        if origin in (Union, UnionType):
            member_dtypes = frozenset(
                dtype
                for member in get_args(annotation)
                if member is not type(None)
                for dtype in (cls.annotation_dtype(member),)
                if dtype is not None
            )
            return next(iter(member_dtypes)) if len(member_dtypes) == 1 else None
        if not isinstance(annotation, type):
            return None
        for scalar_type in (bool, int, float, str, bytes):
            if issubclass(annotation, scalar_type):
                return scalar_type
        if issubclass(annotation, Enum):
            member_dtypes = frozenset(
                cls.annotation_dtype(type(member.value)) for member in annotation
            )
            if len(member_dtypes) == 1:
                return next(iter(member_dtypes))
        return None

    @classmethod
    def annotation_required(cls, annotation: object) -> bool:
        """Return whether one resolved annotation excludes ``None``."""
        origin = get_origin(annotation)
        if origin is Annotated:
            declared_type, *_metadata = get_args(annotation)
            return cls.annotation_required(declared_type)
        if origin in (Union, UnionType):
            return type(None) not in get_args(annotation)
        return True

    @classmethod
    def merge_exact(
        cls,
        field_groups: Iterable[Iterable["FieldSpec"]],
        *,
        context: str = "field declarations",
    ) -> tuple["FieldSpec", ...]:
        """Merge ordered declarations and reject same-name schema conflicts."""
        merged: list[FieldSpec] = []
        by_name: dict[str, FieldSpec] = {}
        for fields in field_groups:
            for field_spec in fields:
                if not isinstance(field_spec, cls):
                    raise TypeError(
                        f"{context} must contain FieldSpec values, got "
                        f"{type(field_spec).__name__}."
                    )
                previous = by_name.get(field_spec.name)
                if previous is None:
                    by_name[field_spec.name] = field_spec
                    merged.append(field_spec)
                    continue
                if previous != field_spec:
                    raise ValueError(
                        f"Conflicting {context} for field {field_spec.name!r}: "
                        f"{previous!r} and {field_spec!r}."
                    )
        return tuple(merged)


def supports_measurement_row_mapping(row: object) -> bool:
    """Return whether ``measurement_row_mapping`` can project this row."""
    return isinstance(row, Mapping) or is_dataclass(row)


def measurement_row_mapping(row: object) -> Mapping[str, object]:
    """Return a mapping view for a supported measurement row payload."""
    if isinstance(row, Mapping):
        return row
    if is_dataclass(row):
        return MeasurementRowMappingCache.process_cache().mapping(row)
    raise TypeError(f"Unsupported measurement row type {type(row).__name__}.")


@dataclass(slots=True)
class MeasurementRowMappingCache(
    RegisteredProcessLocalBoundedCache[int, tuple[object, Mapping[str, object]]]
):
    """Bounded process-local cache for immutable dataclass measurement rows."""

    max_entries: int = 262144

    def mapping(self, row: object) -> Mapping[str, object]:
        row_id = id(row)
        cached = self.cached_value(row_id)
        if cached is not None:
            cached_row, row_mapping = cached
            if cached_row is row:
                return row_mapping
            del self.entries[row_id]
        row_mapping = asdict(row)
        self.store_value(row_id, (row, row_mapping))
        return row_mapping


class MeasurementObjectRowIdentity(str, Enum):
    """How object-scoped measurement rows identify their measured object."""

    LABEL_ID = "label_id"
    ROW_ORDINAL = "row_ordinal"
    ROW_SEQUENCE = "row_sequence"
