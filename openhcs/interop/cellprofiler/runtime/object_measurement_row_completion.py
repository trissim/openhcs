"""Object-measurement row identity and completion semantics."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, TYPE_CHECKING

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.measurement_row_materialization import (
    ColumnarRowColumnOverlay,
    MEASUREMENT_SPARSE_CELL,
    MeasurementProjectedColumnarRows,
    MeasurementSparseColumnarRows,
    is_structural_missing_measurement_cell,
    measurement_object_label,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    StrategyLabelRegistryMixin,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomainScope,
    ObjectLabelPlaneDomainStrategy,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.steps.function_runtime import RuntimeCallableArgument

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
        CellProfilerObjectMeasurementRowPolicy,
    )

ObjectMeasurementAxisKey = tuple[RuntimeCallableArgument, ...]
ObjectMeasurementIdSet = tuple[int, ...]
ObjectMeasurementIdsByAxis = dict[ObjectMeasurementAxisKey, ObjectMeasurementIdSet]
ObjectMeasurementIdsByAxisView = Mapping[
    ObjectMeasurementAxisKey,
    ObjectMeasurementIdSet,
]
MeasurementAxisKeyTuple = tuple[ObjectMeasurementAxisKey, ...]
ObjectMeasurementRowKey = tuple[int | None, ObjectMeasurementAxisKey]
ObjectMeasurementConcreteRowKey = tuple[int, ObjectMeasurementAxisKey]
ObjectMeasurementConcreteRowKeys = list[ObjectMeasurementConcreteRowKey]
ObjectMeasurementSliceRowKeys = list[tuple[int, ObjectMeasurementAxisKey]]
ObjectMeasurementProjectedRowKey = tuple[int | None, tuple[RuntimeCallableArgument, ...]]
ObjectMeasurementProjectedRowKeysTuple = tuple[ObjectMeasurementProjectedRowKey, ...]
ObjectMeasurementPresentRowKey = tuple[int, tuple[RuntimeCallableArgument, ...]]
ObjectMeasurementPresentRowKeySet = set[ObjectMeasurementPresentRowKey]
ObjectMeasurementAxisOrder = dict[ObjectMeasurementAxisKey, int]
ObjectMeasurementRowsByName = dict[str, list[RuntimeCallableArgument]]

MISSING_MEASUREMENT_ROW_VALUE = object()


class MissingObjectMeasurementValuePolicy(str, Enum):
    """How missing per-object measurement result fields are materialized."""

    NAN = "nan"
    ZERO_WITHIN_POSITIVE_EXTENT = "zero_within_positive_extent"


@dataclass(frozen=True, slots=True)
class MissingObjectMeasurementValueRequest:
    """Inputs needed to materialize one missing object-measurement cell."""

    object_id: int
    label_payload: ObjectLabelValue
    field_name: str
    positive_label_extent: int | None = None


class MissingObjectMeasurementValueStrategy(
    EnumKeyedStrategyMixin,
    StrategyLabelRegistryMixin,
    metaclass=AutoRegisterMeta,
):
    """Registered materialization policy for missing object-measurement values."""

    __enum_member_attr__ = "value_policy"
    value_policy: ClassVar[MissingObjectMeasurementValuePolicy]

    @abstractmethod
    def missing_value(self, request: MissingObjectMeasurementValueRequest) -> float:
        """Return the materialized value for one missing measurement cell."""

    def missing_values(
        self,
        *,
        object_ids: Sequence[int],
        label_payload: ObjectLabelValue,
        field_name: str,
        positive_label_extents: Sequence[int | None],
    ) -> tuple[float, ...]:
        """Return one field's missing values with a single strategy resolution."""
        if len(object_ids) != len(positive_label_extents):
            raise ValueError(
                "Missing measurement object IDs and extents must align exactly."
            )
        return tuple(
            self.missing_value(
                MissingObjectMeasurementValueRequest(
                    object_id=int(object_id),
                    label_payload=label_payload,
                    field_name=field_name,
                    positive_label_extent=positive_label_extent,
                )
            )
            for object_id, positive_label_extent in zip(
                object_ids,
                positive_label_extents,
                strict=True,
            )
        )


class NanMissingObjectMeasurementValueStrategy(MissingObjectMeasurementValueStrategy):
    """Materialize every missing object-measurement value as NaN."""

    value_policy = MissingObjectMeasurementValuePolicy.NAN

    def missing_value(self, request: MissingObjectMeasurementValueRequest) -> float:
        del request
        return np.nan

    def missing_values(
        self,
        *,
        object_ids: Sequence[int],
        label_payload: ObjectLabelValue,
        field_name: str,
        positive_label_extents: Sequence[int | None],
    ) -> tuple[float, ...]:
        del label_payload, field_name
        if len(object_ids) != len(positive_label_extents):
            raise ValueError(
                "Missing measurement object IDs and extents must align exactly."
            )
        return (np.nan,) * len(object_ids)


class ZeroWithinPositiveExtentMissingObjectMeasurementValueStrategy(
    MissingObjectMeasurementValueStrategy
):
    """Use zero for rows inside the positive label extent and NaN beyond it."""

    value_policy = MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT

    def missing_value(self, request: MissingObjectMeasurementValueRequest) -> float:
        extent = request.positive_label_extent
        if extent is None:
            extent = self.positive_label_extent(request.label_payload)
        if request.object_id <= extent:
            return 0.0
        return np.nan

    def missing_values(
        self,
        *,
        object_ids: Sequence[int],
        label_payload: ObjectLabelValue,
        field_name: str,
        positive_label_extents: Sequence[int | None],
    ) -> tuple[float, ...]:
        del field_name
        if len(object_ids) != len(positive_label_extents):
            raise ValueError(
                "Missing measurement object IDs and extents must align exactly."
            )
        fallback_extent: int | None = None
        values: list[float] = []
        for object_id, positive_label_extent in zip(
            object_ids,
            positive_label_extents,
            strict=True,
        ):
            extent = positive_label_extent
            if extent is None:
                if fallback_extent is None:
                    fallback_extent = self.positive_label_extent(label_payload)
                extent = fallback_extent
            values.append(0.0 if int(object_id) <= extent else np.nan)
        return tuple(values)

    @staticmethod
    def positive_label_extent(label_payload: RuntimeCallableArgument) -> int:
        """Return the largest object ID in an explicit payload declaration."""
        if not isinstance(label_payload, ObjectLabelValue):
            raise TypeError(
                "Missing object-measurement values require an ObjectLabelValue, "
                f"got {type(label_payload).__name__}."
            )
        domain = label_payload.object_label_domain()
        if domain.scope is not ObjectLabelDomainScope.PAYLOAD:
            raise ValueError(
                "Missing object-measurement values require a projected payload "
                "object-ID domain."
            )
        object_ids = domain.explicit_id_domain()
        if object_ids is None:
            raise ValueError(
                "Missing object-measurement values require an explicit object-ID domain."
            )
        return max(object_ids, default=0)


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowCompletionSchema:
    """Nominal table schema for completing object-scoped measurement rows."""

    fields: tuple[FieldSpec, ...]
    object_id_field: str
    axis_fields: tuple[str, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return exact carrier field names in physical column order."""
        return tuple(field.name for field in self.fields)

    @property
    def metadata_fields(self) -> frozenset[str]:
        """Return declared identity and ownership fields, excluding results."""
        return frozenset(
            (
                self.object_id_field,
                *MeasurementRowAxisField.object_id_field_names(),
                *self.axis_fields,
                *MeasurementRowAxisField.object_ownership_field_names(),
            )
        )

    @classmethod
    def for_completion_fields(
        cls,
        *,
        object_id_field: str,
        axis_fields: Sequence[str],
        field_names: Sequence[str] = (),
    ) -> "ObjectMeasurementRowCompletionSchema":
        """Build a completion schema from explicit object and axis fields."""
        return cls(
            fields=tuple(FieldSpec(str(field_name)) for field_name in field_names),
            object_id_field=str(object_id_field),
            axis_fields=tuple(str(field_name) for field_name in axis_fields),
        )

    @classmethod
    def from_fields(
        cls,
        fields: tuple[FieldSpec, ...],
    ) -> "ObjectMeasurementRowCompletionSchema":
        """Build completion semantics from an exact producer-owned schema."""
        field_names = tuple(field.name for field in fields)
        if not field_names:
            raise ValueError(
                "Object measurement row completion requires a declared object-row "
                "schema."
            )
        return cls(
            fields=fields,
            object_id_field=cls.object_id_field_from_fields(field_names),
            axis_fields=cls.axis_fields_from_fields(field_names),
        )

    @staticmethod
    def object_id_field_from_fields(field_names: Sequence[str]) -> str:
        object_id_field_names = MeasurementRowAxisField.object_id_field_names()
        object_id_fields = tuple(
            field_name
            for field_name in field_names
            if field_name in object_id_field_names
        )
        if len(object_id_fields) != 1:
            raise ValueError(
                "Object measurement row schemas must declare exactly one object-ID "
                f"field, got {object_id_fields!r}."
            )
        return object_id_fields[0]

    @staticmethod
    def axis_fields_from_fields(field_names: Sequence[str]) -> tuple[str, ...]:
        axis_field_names = MeasurementRowAxisField.field_names()
        object_id_field_names = MeasurementRowAxisField.object_id_field_names()
        return tuple(
            field_name
            for field_name in field_names
            if (
                field_name in axis_field_names
                and field_name not in object_id_field_names
            )
        )

    def object_label(self, row: Mapping[str, RuntimeCallableArgument]) -> int | None:
        """Return the object identity encoded by one declared row."""
        return measurement_object_label(row, object_id_field=self.object_id_field)

    def axis_key_from_mapping(
        self,
        row: Mapping[str, RuntimeCallableArgument],
    ) -> tuple[RuntimeCallableArgument, ...]:
        """Return the exact measurement-axis key encoded by one row."""
        missing_fields = tuple(
            field_name for field_name in self.axis_fields if field_name not in row
        )
        if missing_fields:
            raise ValueError(
                "Object measurement row is missing declared identity axes "
                f"{missing_fields!r}."
            )
        return tuple(row[field_name] for field_name in self.axis_fields)

    def axis_key_from_columns(
        self,
        rows: ColumnarRows,
        row_index: int,
    ) -> tuple[RuntimeCallableArgument, ...]:
        """Return one exact axis key directly from declared row columns."""
        axis_values = tuple(
            rows.column_values(field_name)[row_index]
            for field_name in self.axis_fields
        )
        missing_fields = tuple(
            field_name
            for field_name, value in zip(
                self.axis_fields,
                axis_values,
                strict=True,
            )
            if is_structural_missing_measurement_cell(value)
        )
        if missing_fields:
            raise ValueError(
                "Object measurement row is missing declared identity axes "
                f"{missing_fields!r}."
            )
        return axis_values

    def row_with_object_id(
        self,
        row: Mapping[str, RuntimeCallableArgument],
        object_id: int,
    ) -> dict[str, RuntimeCallableArgument]:
        """Return one row projected to this schema's object identity field."""
        projected_row = dict(row)
        projected_row[self.object_id_field] = object_id
        return projected_row

    def axis_keys_for_label_payload(
        self,
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        *,
        label_payload: ObjectLabelValue,
    ) -> tuple[tuple[RuntimeCallableArgument, ...], ...]:
        """Return measurement axes valid for completing rows against labels."""
        if not self.axis_fields:
            return ((),)
        if projection.rows.row_count():
            return projection.axis_keys
        slice_axis = MeasurementRowAxisField.SLICE_INDEX.value
        if self.axis_fields != (slice_axis,):
            raise ValueError(
                "Empty object-measurement rows can only be completed when the sole "
                f"declared axis is {slice_axis!r}; got {self.axis_fields!r}."
            )
        axis_values = ObjectLabelPlaneDomainStrategy.for_enum_member(
            label_payload.object_label_domain().scope
        ).declared_measurement_axis_values(label_payload)
        return tuple((axis_value,) for axis_value in axis_values)

    def object_ids_for_axis(
        self,
        *,
        label_payload: ObjectLabelValue,
        object_identity: MeasurementObjectRowIdentity,
        axis_key: tuple[RuntimeCallableArgument, ...],
    ) -> tuple[int, ...]:
        label_ids = self.label_ids_for_axis(
            label_payload=label_payload,
            axis_key=axis_key,
        )
        return MeasurementObjectRowIdentityProjectionStrategy.for_enum_member(
            object_identity
        ).object_ids_for_label_ids(label_ids)

    def label_ids_for_axis(
        self,
        *,
        label_payload: ObjectLabelValue,
        axis_key: tuple[RuntimeCallableArgument, ...],
    ) -> tuple[int, ...]:
        """Return source label IDs in the declared domain for one measurement axis."""
        label_ids = self.explicit_label_ids_for_axis(
            label_payload=label_payload,
            axis_key=axis_key,
        )
        if label_ids is None:
            raise ValueError(
                "Object measurement completion requires an explicit object-ID domain."
            )
        return tuple(label_ids)

    def explicit_label_ids_for_axis(
        self,
        *,
        label_payload: ObjectLabelValue,
        axis_key: tuple[RuntimeCallableArgument, ...],
    ) -> tuple[int, ...] | None:
        """Return explicitly declared source label IDs for one measurement axis."""
        if not isinstance(label_payload, ObjectLabelValue):
            raise TypeError(
                "Object measurement completion requires an ObjectLabelValue, "
                f"got {type(label_payload).__name__}."
            )
        domain_axis_key = self.label_domain_axis_key(
            label_payload,
            axis_key=axis_key,
        )
        if not domain_axis_key:
            domain = label_payload.object_label_domain()
        else:
            plane_count = label_payload.runtime_slice_plane_count()
            if plane_count is None:
                raise ValueError(
                    "Plane-scoped object labels must declare an exact plane count."
                )
            domain = label_payload.runtime_slice_domain(
                slice_index=int(domain_axis_key[0]),
                slice_count=plane_count,
            )
        object_ids = domain.explicit_id_domain()
        return None if object_ids is None else tuple(object_ids)

    def label_domain_axis_key(
        self,
        label_payload: ObjectLabelValue,
        *,
        axis_key: tuple[RuntimeCallableArgument, ...],
    ) -> tuple[RuntimeCallableArgument, ...]:
        """Return the axis subset that changes the label-id domain."""
        label_domain = label_payload.object_label_domain()
        if label_domain.scope is ObjectLabelDomainScope.PAYLOAD:
            return ()
        slice_axis_name = MeasurementRowAxisField.SLICE_INDEX.value
        if label_payload.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            raise ValueError(
                "Plane-scoped measurement domains require the runtime-slice axis."
            )
        if slice_axis_name not in self.axis_fields:
            raise ValueError("Plane-scoped measurement rows must declare slice_index.")
        slice_axis_position = self.axis_fields.index(slice_axis_name)
        if slice_axis_position >= len(axis_key):
            raise ValueError(
                "Object measurement axis key does not contain its declared "
                "slice_index value."
            )
        slice_index = int(axis_key[slice_axis_position])
        plane_count = label_payload.runtime_slice_plane_count()
        if plane_count is None:
            raise ValueError(
                "Plane-scoped object labels must declare an exact plane count."
            )
        if slice_index < 0 or slice_index >= plane_count:
            raise ValueError(
                f"Measurement slice_index {slice_index} is outside label stack "
                f"with {plane_count} declared planes."
            )
        return (slice_index,)

    def positive_extent_for_missing_measurements(
        self,
        *,
        label_payload: ObjectLabelValue,
        axis_key: tuple[RuntimeCallableArgument, ...],
        row_policy: CellProfilerObjectMeasurementRowPolicy,
    ) -> int | None:
        policy = MissingObjectMeasurementValuePolicy(
            type(row_policy).missing_value_policy
        )
        if (
            policy
            is not MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
        ):
            return None
        return max(
            self.label_ids_for_axis(
                label_payload=label_payload,
                axis_key=axis_key,
            ),
            default=0,
        )

    def positive_extent_by_axis(
        self,
        *,
        label_payload: ObjectLabelValue,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
        row_keys: Sequence[tuple[int, tuple[RuntimeCallableArgument, ...]]],
    ) -> dict[tuple[RuntimeCallableArgument, ...], int | None]:
        unique_axis_keys = tuple(
            dict.fromkeys(axis_key for _object_id, axis_key in row_keys)
        )
        return {
            axis_key: self.positive_extent_for_missing_measurements(
                label_payload=label_payload,
                axis_key=axis_key,
                row_policy=row_policy,
            )
            for axis_key in unique_axis_keys
        }

    def missing_rows(
        self,
        *,
        missing_row_keys: Sequence[tuple[int, tuple[RuntimeCallableArgument, ...]]],
        label_payload: ObjectLabelValue,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
    ) -> tuple[dict[str, RuntimeCallableArgument], ...]:
        positive_extent_by_axis = self.positive_extent_by_axis(
            label_payload=label_payload,
            row_policy=row_policy,
            row_keys=missing_row_keys,
        )
        return tuple(
            self.missing_row(
                object_id=object_id,
                axis_key=axis_key,
                label_payload=label_payload,
                row_policy=row_policy,
                positive_label_extent=positive_extent_by_axis[axis_key],
            )
            for object_id, axis_key in missing_row_keys
        )

    def missing_columnar_rows(
        self,
        *,
        missing_row_keys: Sequence[tuple[int, tuple[RuntimeCallableArgument, ...]]],
        label_payload: ObjectLabelValue,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
        object_row_identity: MeasurementObjectRowIdentity,
    ) -> MeasurementSparseColumnarRows:
        """Return missing object rows directly in the declared column schema."""
        row_keys = tuple(missing_row_keys)
        positive_extent_by_axis = self.positive_extent_by_axis(
            label_payload=label_payload,
            row_policy=row_policy,
            row_keys=row_keys,
        )
        object_ids = tuple(object_id for object_id, _axis_key in row_keys)
        axis_positions = {
            field_name: axis_index
            for axis_index, field_name in enumerate(self.axis_fields)
        }
        positive_label_extents = tuple(
            positive_extent_by_axis[axis_key] for _object_id, axis_key in row_keys
        )
        object_id_field_names = MeasurementRowAxisField.object_id_field_names()
        columns: dict[str, Sequence[RuntimeCallableArgument]] = {}
        for field_name in self.field_names:
            if field_name == self.object_id_field:
                columns[field_name] = object_ids
                continue
            if field_name in object_id_field_names:
                columns[field_name] = (MEASUREMENT_SPARSE_CELL,) * len(row_keys)
                continue
            axis_position = axis_positions.get(field_name)
            if axis_position is not None:
                columns[field_name] = tuple(
                    axis_key[axis_position] for _object_id, axis_key in row_keys
                )
                continue
            values = row_policy.missing_measurement_values(
                object_ids=object_ids,
                label_payload=label_payload,
                field_name=field_name,
                positive_label_extents=positive_label_extents,
            )
            columns[field_name] = tuple(
                MEASUREMENT_SPARSE_CELL
                if value is MISSING_MEASUREMENT_ROW_VALUE
                else value
                for value in values
            )
        return MeasurementSparseColumnarRows(
            MappingProxyType(columns),
            fields=self.fields,
            object_row_identity=object_row_identity,
        )

    def missing_row(
        self,
        *,
        object_id: int,
        axis_key: Sequence[RuntimeCallableArgument],
        label_payload: ObjectLabelValue,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
        positive_label_extent: int | None = None,
    ) -> dict[str, RuntimeCallableArgument]:
        axis_values = self.axis_values_for_key(axis_key)
        object_id_field_names = MeasurementRowAxisField.object_id_field_names()
        row: dict[str, RuntimeCallableArgument] = {}
        for field_name in self.field_names:
            if (
                field_name in object_id_field_names
                or field_name in axis_values
            ):
                continue
            missing_value = row_policy.missing_measurement_value(
                object_id=object_id,
                label_payload=label_payload,
                field_name=field_name,
                positive_label_extent=positive_label_extent,
            )
            if missing_value is MISSING_MEASUREMENT_ROW_VALUE:
                continue
            row[field_name] = missing_value
        row.update(axis_values)
        row[self.object_id_field] = object_id
        return row

    def axis_values_for_key(
        self, axis_key: Sequence[RuntimeCallableArgument]
    ) -> dict[str, RuntimeCallableArgument]:
        if len(axis_key) != len(self.axis_fields):
            raise ValueError(
                "Measurement axis key cardinality must match declared axis fields; got "
                f"{tuple(axis_key)!r} for fields {tuple(self.axis_fields)!r}."
            )
        return dict(zip(self.axis_fields, axis_key, strict=True))


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowIdentityProjectionResult:
    """Rows plus their nominal object/axis identity after projection."""

    rows: ColumnarRows
    row_keys: "ObjectMeasurementProjectedRowKeys"
    measured_row_keys: "ObjectMeasurementProjectedRowKeys"
    axis_keys: tuple[tuple[RuntimeCallableArgument, ...], ...]

    def within_axis_domain(
        self,
        *,
        axis_keys: Sequence[tuple[RuntimeCallableArgument, ...]],
        object_ids_by_axis: ObjectMeasurementIdsByAxis,
    ) -> "ObjectMeasurementRowIdentityProjectionResult":
        """Return projected rows constrained to the required object/axis domain."""
        required_row_keys = set(
            ObjectMeasurementProjectedRowKeys.required_axis_domain(
                axis_keys,
                object_ids_by_axis,
            )
        )
        if self.rows.row_count() != len(self.row_keys):
            raise ValueError(
                "Projected measurement row count must match its row-key count."
            )
        selected_indices = tuple(
            row_index
            for row_index, row_key in enumerate(self.row_keys)
            if self.row_key_is_within_axis_domain(row_key, required_row_keys)
        )
        measured_row_keys = tuple(
            row_key
            for row_key in self.measured_row_keys
            if self.row_key_is_within_axis_domain(row_key, required_row_keys)
        )
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=MeasurementProjectedColumnarRows.from_columnar_rows(
                self.rows,
                row_indices=selected_indices,
                declared_object_measurement_domain_covered=(
                    self.rows.covers_declared_object_measurement_domain
                ),
                object_row_identity=self.rows.object_row_identity,
            ),
            row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(self.row_keys.entries[row_index] for row_index in selected_indices)
            ),
            measured_row_keys=ObjectMeasurementProjectedRowKeys(measured_row_keys),
            axis_keys=self.axis_keys,
        )

    @staticmethod
    def row_key_is_within_axis_domain(
        row_key: ObjectMeasurementProjectedRowKey,
        required_row_keys: set[ObjectMeasurementConcreteRowKey],
    ) -> bool:
        """Return whether a projected row key belongs to the required domain."""
        object_id, axis_key = row_key
        return object_id is None or (object_id, axis_key) in required_row_keys

    def ordered_rows(
        self,
        *,
        object_ids: Sequence[int],
        axis_keys: Sequence[tuple[RuntimeCallableArgument, ...]],
        rows: ColumnarRows | None = None,
        row_keys: "ObjectMeasurementProjectedRowKeys | None" = None,
    ) -> ColumnarRows:
        """Return exact columnar rows in dense object/axis order."""
        ordered_rows = self.rows if rows is None else rows
        ordered_row_keys = self.row_keys if row_keys is None else row_keys
        object_order = {object_id: index for index, object_id in enumerate(object_ids)}
        axis_order = {axis_key: index for index, axis_key in enumerate(axis_keys)}
        if ordered_rows.row_count() != len(ordered_row_keys):
            raise ValueError(
                "Projected measurement row count must match its row-key count."
            )
        if len(set(ordered_row_keys)) != len(ordered_row_keys):
            raise ValueError(
                "Object measurement rows contain duplicate declared row identities."
            )
        ordered_indices = tuple(
            row_index
            for row_index, _row_key in sorted(
                enumerate(ordered_row_keys),
                key=lambda item: self.row_order_key(
                    item[1],
                    object_order=object_order,
                    axis_order=axis_order,
                ),
            )
        )
        return MeasurementProjectedColumnarRows.from_columnar_rows(
            ordered_rows,
            row_indices=ordered_indices,
            declared_object_measurement_domain_covered=(
                ordered_rows.covers_declared_object_measurement_domain
            ),
            object_row_identity=ordered_rows.object_row_identity,
        )

    @staticmethod
    def row_order_key(
        row_key: tuple[int | None, tuple[RuntimeCallableArgument, ...]],
        *,
        object_order: Mapping[int, int],
        axis_order: Mapping[tuple[RuntimeCallableArgument, ...], int],
    ) -> tuple[int, int]:
        """Return the exact declared ordering key for one measurement row."""
        object_id, axis_key = row_key
        if axis_key not in axis_order:
            raise ValueError(
                f"Object measurement row axis {axis_key!r} is not declared."
            )
        if object_id is None or object_id not in object_order:
            raise ValueError(
                f"Object measurement row identity {object_id!r} is not declared."
            )
        return (axis_order[axis_key], object_order[object_id])


@dataclass(frozen=True, slots=True)
class ObjectMeasurementProjectedRowKeys:
    """Projected object-row keys with aggregate diagnostics."""

    entries: ObjectMeasurementProjectedRowKeysTuple

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def max_object_id_or_none(self) -> int | None:
        object_ids = tuple(
            object_id for object_id, _axis_key in self.entries if object_id is not None
        )
        if not object_ids:
            return None
        return max(object_ids)

    def has_object_ids(self) -> bool:
        """Return whether any projected row has object identity."""
        return any(object_id is not None for object_id, _axis_key in self.entries)

    def axis_keys(self) -> tuple[tuple[RuntimeCallableArgument, ...], ...]:
        """Return ordered unique measurement-axis keys represented by entries."""
        return tuple(dict.fromkeys(axis_key for _object_id, axis_key in self.entries))

    def present_in(
        self,
        required_row_keys: ObjectMeasurementPresentRowKeySet,
    ) -> ObjectMeasurementPresentRowKeySet:
        """Return present concrete row keys constrained to a required domain."""
        present_row_keys: ObjectMeasurementPresentRowKeySet = set()
        for object_id, axis_key in self.entries:
            if object_id is None:
                continue
            row_key = (object_id, axis_key)
            if row_key not in required_row_keys:
                continue
            present_row_keys.add(row_key)
            if len(present_row_keys) == len(required_row_keys):
                break
        return present_row_keys

    @staticmethod
    def required_axis_domain(
        axis_keys: Sequence[tuple[RuntimeCallableArgument, ...]],
        object_ids_by_axis: ObjectMeasurementIdsByAxis,
    ) -> ObjectMeasurementSliceRowKeys:
        """Return required row keys for a dense object/axis domain."""
        return [
            (object_id, axis_key)
            for axis_key in axis_keys
            for object_id in object_ids_by_axis[axis_key]
        ]

    @staticmethod
    def object_ids_from_axis_domain(
        object_ids_by_axis: ObjectMeasurementIdsByAxis,
    ) -> tuple[int, ...]:
        """Return all object IDs represented by an axis-keyed object domain."""
        return tuple(
            sorted(
                {
                    object_id
                    for axis_object_ids in object_ids_by_axis.values()
                    for object_id in axis_object_ids
                }
            )
        )

    def missing_from_axis_domain(
        self,
        axis_keys: Sequence[tuple[RuntimeCallableArgument, ...]],
        object_ids_by_axis: ObjectMeasurementIdsByAxis,
    ) -> ObjectMeasurementSliceRowKeys:
        """Return required row keys not already represented by projected keys."""
        required_row_keys = self.required_axis_domain(axis_keys, object_ids_by_axis)
        present_row_keys = self.present_in(set(required_row_keys))
        return [
            row_key for row_key in required_row_keys if row_key not in present_row_keys
        ]

    def appended(
        self,
        entries: Sequence[ObjectMeasurementProjectedRowKey],
    ) -> "ObjectMeasurementProjectedRowKeys":
        """Return keys with appended completion entries."""
        return ObjectMeasurementProjectedRowKeys((*self.entries, *tuple(entries)))


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowOrdinalProjectionState:
    """Declared source-label to row-ordinal mapping for compact projection."""

    ordinal_by_axis_label: Mapping[
        tuple[RuntimeCallableArgument, ...],
        Mapping[int, int],
    ]

    def ordinal_for_declared_object(
        self,
        row_mapping: Mapping[str, RuntimeCallableArgument],
        *,
        axis_key: tuple[RuntimeCallableArgument, ...],
        object_id_field: str,
    ) -> int:
        """Return the ordinal declared for one source object and measurement axis."""
        source_object_id = measurement_object_label(
            row_mapping,
            object_id_field=object_id_field,
        )
        if source_object_id is None:
            raise ValueError(
                "Declared row-ordinal projection requires a source object ID."
            )
        if axis_key not in self.ordinal_by_axis_label:
            raise ValueError(
                f"Row-ordinal projection axis {axis_key!r} is not declared."
            )
        ordinal_by_label = self.ordinal_by_axis_label[axis_key]
        if source_object_id not in ordinal_by_label:
            raise ValueError(
                f"Source object {source_object_id} is absent from the declared "
                f"row-ordinal domain for axis {axis_key!r}."
            )
        return ordinal_by_label[source_object_id]


class MeasurementObjectRowIdentityProjectionStrategy(
    EnumKeyedStrategyMixin,
    StrategyLabelRegistryMixin,
    metaclass=AutoRegisterMeta,
):
    """Registered projection from source object IDs to exported row identity."""

    __enum_member_attr__ = "object_identity"
    object_identity: ClassVar[MeasurementObjectRowIdentity]

    @abstractmethod
    def project_rows(
        self,
        rows: ColumnarRows,
        schema: ObjectMeasurementRowCompletionSchema,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        """Return rows with the requested object-row identity."""

    @abstractmethod
    def object_ids_for_label_ids(self, label_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Return required exported object IDs for a source label-ID domain."""

    def row_with_object_id(
        self,
        schema: ObjectMeasurementRowCompletionSchema,
        row: Mapping[str, RuntimeCallableArgument],
        object_id: int,
    ) -> dict[str, RuntimeCallableArgument]:
        """Return a row projected into this strategy's object identity domain."""
        return schema.row_with_object_id(row, object_id)


class LabelIdMeasurementObjectRowIdentityProjectionStrategy(
    MeasurementObjectRowIdentityProjectionStrategy
):
    """Preserve label IDs as exported object-row identities."""

    object_identity = MeasurementObjectRowIdentity.LABEL_ID

    def object_ids_for_label_ids(self, label_ids: tuple[int, ...]) -> tuple[int, ...]:
        return label_ids

    def project_rows(
        self,
        rows: ColumnarRows,
        schema: ObjectMeasurementRowCompletionSchema,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        row_mappings = rows.row_mappings()
        row_keys: list[ObjectMeasurementProjectedRowKey] = []
        measured_row_keys: list[ObjectMeasurementProjectedRowKey] = []
        metadata_fields = schema.metadata_fields
        for row_mapping in row_mappings:
            row_key = (
                schema.object_label(row_mapping),
                schema.axis_key_from_mapping(row_mapping),
            )
            row_keys.append(row_key)
            if row_key[0] is None:
                continue
            if row_policy.row_has_measured_object(
                row_mapping,
                metadata_fields=metadata_fields,
            ):
                measured_row_keys.append(row_key)
        projected_row_keys = ObjectMeasurementProjectedRowKeys(tuple(row_keys))
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=rows,
            row_keys=projected_row_keys,
            measured_row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(measured_row_keys)
            ),
            axis_keys=projected_row_keys.axis_keys(),
        )


class CompactMeasurementObjectIdProjectionMixin:
    """Map source label IDs onto compact 1-based exported object IDs."""

    def object_ids_for_label_ids(self, label_ids: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(range(1, len(label_ids) + 1))


class RowOrdinalMeasurementObjectRowIdentityProjectionStrategy(
    CompactMeasurementObjectIdProjectionMixin,
    MeasurementObjectRowIdentityProjectionStrategy,
):
    """Project measured objects into CP's compact row-ordinal identity."""

    object_identity = MeasurementObjectRowIdentity.ROW_ORDINAL

    def project_rows(
        self,
        rows: ColumnarRows,
        schema: ObjectMeasurementRowCompletionSchema,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        projected_rows: list[Mapping[str, RuntimeCallableArgument]] = []
        projected_row_keys: ObjectMeasurementSliceRowKeys = []
        measured_projected_row_keys: ObjectMeasurementSliceRowKeys = []
        axis_keys: list[tuple[RuntimeCallableArgument, ...]] = []
        metadata_fields = schema.metadata_fields
        for row_mapping in rows.iter_row_mappings():
            object_id = schema.object_label(row_mapping)
            if object_id is None or object_id <= 0:
                raise ValueError(
                    "Row-ordinal projection requires every source row to declare a "
                    "positive object ID."
                )
            axis_key = schema.axis_key_from_mapping(row_mapping)
            axis_keys.append(axis_key)
            measured = row_policy.row_has_measured_object(
                row_mapping,
                metadata_fields=metadata_fields,
            )
            if not measured and not row_policy.retains_unmeasured_compact_row(
                row_mapping,
                schema=schema,
            ):
                continue
            projected_rows.append(row_mapping)
            projected_row_keys.append((object_id, axis_key))
            if measured:
                measured_projected_row_keys.append((object_id, axis_key))
        projected_axis_keys = tuple(dict.fromkeys(axis_keys))
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=MeasurementSparseColumnarRows.from_rows(
                tuple(projected_rows),
                fields=rows.fields,
                declared_object_measurement_domain_covered=(
                    rows.covers_declared_object_measurement_domain
                    and len(projected_rows) == rows.row_count()
                ),
                object_row_identity=MeasurementObjectRowIdentity.ROW_ORDINAL,
            ),
            row_keys=ObjectMeasurementProjectedRowKeys(tuple(projected_row_keys)),
            measured_row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(measured_projected_row_keys)
            ),
            axis_keys=projected_axis_keys,
        )


class RowSequenceMeasurementObjectRowIdentityProjectionStrategy(
    CompactMeasurementObjectIdProjectionMixin,
    MeasurementObjectRowIdentityProjectionStrategy,
):
    """Project each measured source row to its own compact row ordinal."""

    object_identity = MeasurementObjectRowIdentity.ROW_SEQUENCE

    def project_rows(
        self,
        rows: ColumnarRows,
        schema: ObjectMeasurementRowCompletionSchema,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        ordinal_by_axis: dict[tuple[RuntimeCallableArgument, ...], int] = {}
        selected_indices: list[int] = []
        projected_object_ids: list[int] = []
        projected_row_keys: ObjectMeasurementSliceRowKeys = []
        measured_projected_row_keys: ObjectMeasurementSliceRowKeys = []
        axis_keys: list[tuple[RuntimeCallableArgument, ...]] = []
        row_count = rows.row_count()
        field_columns = {
            field_spec.name: rows.column_values(field_spec.name)
            for field_spec in rows.fields
        }
        metadata_fields = schema.metadata_fields
        measurement_columns = tuple(
            values
            for field_name, values in field_columns.items()
            if field_name not in metadata_fields
        )
        for row_index in range(row_count):
            axis_key = schema.axis_key_from_columns(rows, row_index)
            axis_keys.append(axis_key)
            measured = row_policy.measurement_values_have_result_payload(
                values[row_index] for values in measurement_columns
            )
            if not measured:
                row_mapping = {
                    field_name: value
                    for field_name, values in field_columns.items()
                    for value in (values[row_index],)
                    if not is_structural_missing_measurement_cell(value)
                }
                if not row_policy.retains_unmeasured_compact_row(
                    row_mapping,
                    schema=schema,
                ):
                    continue
            ordinal = (
                ordinal_by_axis[axis_key] if axis_key in ordinal_by_axis else 0
            ) + 1
            ordinal_by_axis[axis_key] = ordinal
            selected_indices.append(row_index)
            projected_object_ids.append(ordinal)
            projected_row_keys.append((ordinal, axis_key))
            if measured:
                measured_projected_row_keys.append((ordinal, axis_key))
        projected_rows = MeasurementProjectedColumnarRows.from_columnar_rows(
            rows,
            row_indices=selected_indices,
            declared_object_measurement_domain_covered=(
                rows.covers_declared_object_measurement_domain
                and len(selected_indices) == row_count
            ),
            object_row_identity=MeasurementObjectRowIdentity.ROW_SEQUENCE,
        )
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=MeasurementProjectedColumnarRows(
                ColumnarRowColumnOverlay(
                    projected_rows.columns,
                    MappingProxyType(
                        {schema.object_id_field: tuple(projected_object_ids)}
                    ),
                ),
                fields=rows.fields,
                object_row_identity=MeasurementObjectRowIdentity.ROW_SEQUENCE,
            ),
            row_keys=ObjectMeasurementProjectedRowKeys(tuple(projected_row_keys)),
            measured_row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(measured_projected_row_keys)
            ),
            axis_keys=tuple(dict.fromkeys(axis_keys)),
        )
