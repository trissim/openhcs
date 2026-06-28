"""Object-measurement row identity and completion semantics."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_ID_FIELD,
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
    measurement_object_label,
)
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
    ObjectLabelDomainMetadataStrategy,
    dense_object_label_id_domain,
    measurement_row_mapping,
    measurement_row_axis_field_names,
)
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
    RuntimeProjectionAxis,
)
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup
from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
    CellProfilerMeasurementFieldSchema,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import LABEL_PAYLOAD_FINAL
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargDict,
    CellProfilerKwargs,
    CellProfilerProfileFields,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    CellProfilerRuntimeValueSequence,
    MeasurementRowMapping,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    EnumStrategyLabelRegistryMixin,
)

ObjectMeasurementAxisKey = CellProfilerRuntimeValues
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
ObjectMeasurementProjectedRowKey = tuple[int | None, CellProfilerRuntimeValues]
ObjectMeasurementProjectedRowKeysTuple = tuple[ObjectMeasurementProjectedRowKey, ...]
ObjectMeasurementPresentRowKey = tuple[int, CellProfilerRuntimeValues]
ObjectMeasurementPresentRowKeySet = set[ObjectMeasurementPresentRowKey]
ObjectMeasurementAxisOrder = dict[ObjectMeasurementAxisKey, int]
ObjectMeasurementRowsByName = dict[str, list[CellProfilerRuntimeValue]]

MeasurementSourceFieldPairs = CellProfilerProfileFields

MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS = (
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_OBJECT_ID_FIELD,
)
MISSING_MEASUREMENT_ROW_VALUE = object()


class MissingObjectMeasurementValuePolicy(str, Enum):
    """How missing per-object measurement result fields are materialized."""

    NAN = "nan"
    ZERO_WITHIN_POSITIVE_EXTENT = "zero_within_positive_extent"


@dataclass(frozen=True, slots=True)
class MissingObjectMeasurementValueRequest:
    """Inputs needed to materialize one missing object-measurement cell."""

    object_id: int
    label_payload: CellProfilerRuntimeValue
    field_name: str
    positive_label_extent: int | None = None


class MissingObjectMeasurementValueStrategy(
    EnumStrategyLabelRegistryMixin,
    metaclass=AutoRegisterMeta,
):
    """Registered materialization policy for missing object-measurement values."""

    __enum_member_attr__ = "value_policy"
    value_policy: ClassVar[MissingObjectMeasurementValuePolicy]

    @abstractmethod
    def missing_value(self, request: MissingObjectMeasurementValueRequest) -> float:
        """Return the materialized value for one missing measurement cell."""


class NanMissingObjectMeasurementValueStrategy(MissingObjectMeasurementValueStrategy):
    """Materialize every missing object-measurement value as NaN."""

    value_policy = MissingObjectMeasurementValuePolicy.NAN

    def missing_value(self, request: MissingObjectMeasurementValueRequest) -> float:
        del request
        return np.nan


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

    @staticmethod
    def positive_label_extent(label_payload: CellProfilerRuntimeValue) -> int:
        """Return the largest positive label ID present in a label payload."""
        labels = np.asarray(LABEL_PAYLOAD_FINAL.value(label_payload))
        if labels.size == 0:
            return 0
        positive_labels = labels[labels > 0]
        if positive_labels.size == 0:
            return 0
        return int(np.max(positive_labels))

@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowIdentityProjectionRequest:
    """Typed context for projecting source rows into CellProfiler row identity."""

    rows: CellProfilerRuntimeValueSequence
    object_id_field: str
    axis_fields: Sequence[str]
    row_policy: CellProfilerObjectMeasurementRowPolicy

    def object_label(self, row: CellProfilerRuntimeValue) -> int | None:
        """Return the object identity encoded by one source row."""
        return measurement_object_label(
            measurement_row_mapping(row),
            object_id_field=self.object_id_field,
        )

    def axis_key(self, row: CellProfilerRuntimeValue) -> CellProfilerRuntimeValues:
        """Return the measurement-axis key encoded by one source row."""
        return self.axis_key_from_mapping(measurement_row_mapping(row))

    def axis_key_from_mapping(self, row: CellProfilerKwargs) -> CellProfilerRuntimeValues:
        """Return the measurement-axis key encoded by one row mapping."""
        axis_fields = self.row_policy.row_identity_axis_fields(self.axis_fields)
        if not axis_fields:
            return ()
        if len(axis_fields) == 1:
            return (row.get(axis_fields[0]),)
        return tuple(row.get(field_name) for field_name in axis_fields)

    def row_with_object_id(self, row: CellProfilerRuntimeValue, object_id: int) -> CellProfilerKwargDict:
        """Return a row projected to the requested object identity field."""
        projected_row = dict(measurement_row_mapping(row))
        projected_row[self.object_id_field] = object_id
        return projected_row

    def axis_keys_for_label_payload(
        self,
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        *,
        label_payload: CellProfilerRuntimeValue,
    ) -> tuple[CellProfilerRuntimeValues, ...]:
        """Return measurement axes valid for completing rows against labels."""
        if not self.axis_fields:
            return ((),)
        if not projection.rows:
            return ()
        return projection.axis_keys


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowCompletionSchema:
    """Nominal table schema for completing object-scoped measurement rows."""

    field_names: tuple[str, ...]
    object_id_field: str
    axis_fields: tuple[str, ...]

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
            field_names=tuple(str(field_name) for field_name in field_names),
            object_id_field=str(object_id_field),
            axis_fields=tuple(str(field_name) for field_name in axis_fields),
        )

    @classmethod
    def from_rows(
        cls,
        rows: CellProfilerRuntimeValueSequence,
        func: CellProfilerFunction,
    ) -> "ObjectMeasurementRowCompletionSchema":
        field_names = cls.field_names_from_rows(rows, func)
        return cls(
            field_names=field_names,
            object_id_field=cls.object_id_field_from_fields(field_names),
            axis_fields=cls.axis_fields_from_fields(field_names),
        )

    @staticmethod
    def field_names_from_rows(
        rows: CellProfilerRuntimeValueSequence,
        func: CellProfilerFunction,
    ) -> tuple[str, ...]:
        if rows:
            return tuple(str(key) for key in measurement_row_mapping(rows[0]).keys())
        return tuple(
            field.name for field in CellProfilerMeasurementFieldSchema.from_callable(func)
        )

    @staticmethod
    def object_id_field_from_fields(field_names: Sequence[str]) -> str:
        for field_name in field_names:
            if field_name in MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS:
                return field_name
        return MEASUREMENT_OBJECT_LABEL_FIELD

    @staticmethod
    def axis_fields_from_fields(field_names: Sequence[str]) -> tuple[str, ...]:
        axis_field_names = measurement_row_axis_field_names()
        return tuple(
            field_name
            for field_name in field_names
            if (
                field_name in axis_field_names
                and field_name not in MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS
            )
        )

    def object_ids_for_axis(
        self,
        *,
        label_payload: CellProfilerRuntimeValue,
        object_identity: MeasurementObjectRowIdentity,
        axis_key: CellProfilerRuntimeValues,
    ) -> tuple[int, ...]:
        axis_payload = self.label_payload_for_axis(label_payload, axis_key=axis_key)
        label_domain = ObjectLabelDomainMetadataStrategy.for_value(
            axis_payload
        ).object_label_domain(axis_payload)
        label_ids = label_domain.explicit_id_domain()
        if label_ids is None:
            label_ids = dense_object_label_id_domain(axis_payload)
        return (
            MeasurementObjectRowIdentityProjectionStrategy
            .for_enum_member(object_identity)
            .object_ids_for_label_ids(label_ids)
        )

    def label_payload_for_axis(
        self,
        label_payload: CellProfilerRuntimeValue,
        *,
        axis_key: CellProfilerRuntimeValues,
    ) -> CellProfilerRuntimeValue:
        normalized_axis_fields = tuple(
            str(field_name).strip().lower() for field_name in self.axis_fields
        )
        slice_axis_name = MeasurementRowAxisField.SLICE_INDEX.value
        if slice_axis_name not in normalized_axis_fields:
            return label_payload
        slice_axis_position = normalized_axis_fields.index(slice_axis_name)
        if slice_axis_position >= len(axis_key):
            return label_payload
        slice_index = int(axis_key[slice_axis_position])
        labels = np.asarray(LABEL_PAYLOAD_FINAL.value(label_payload))
        if labels.ndim < 3:
            return label_payload
        if slice_index < 0 or slice_index >= labels.shape[0]:
            raise ValueError(
                f"Measurement slice_index {slice_index} is outside label stack "
                f"with {labels.shape[0]} slices."
            )
        return RuntimeSliceProjection.value_for_slice(
            label_payload,
            RuntimeProjectionAxis(
                slice_index=slice_index,
                extent=labels.shape[0],
            ),
        )

    def label_domain_axis_key(
        self,
        label_payload: CellProfilerRuntimeValue,
        *,
        axis_key: CellProfilerRuntimeValues,
    ) -> CellProfilerRuntimeValues:
        """Return the axis subset that changes the label-id domain."""
        normalized_axis_fields = tuple(
            str(field_name).strip().lower() for field_name in self.axis_fields
        )
        slice_axis_name = MeasurementRowAxisField.SLICE_INDEX.value
        if slice_axis_name not in normalized_axis_fields:
            return ()
        slice_axis_position = normalized_axis_fields.index(slice_axis_name)
        if slice_axis_position >= len(axis_key):
            return ()
        labels = np.asarray(LABEL_PAYLOAD_FINAL.value(label_payload))
        if labels.ndim < 3:
            return ()
        return (int(axis_key[slice_axis_position]),)

    def positive_extent_for_missing_measurements(
        self,
        *,
        label_payload: CellProfilerRuntimeValue,
        axis_key: CellProfilerRuntimeValues,
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
        axis_payload = self.label_payload_for_axis(label_payload, axis_key=axis_key)
        return self.positive_object_label_extent(axis_payload)

    @staticmethod
    def positive_object_label_extent(label_payload: CellProfilerRuntimeValue) -> int:
        return (
            ZeroWithinPositiveExtentMissingObjectMeasurementValueStrategy
            .positive_label_extent(label_payload)
        )

    def positive_extent_by_axis(
        self,
        *,
        label_payload: CellProfilerRuntimeValue,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
        row_keys: Sequence[tuple[int, CellProfilerRuntimeValues]],
        measured_positive_extent_by_axis: Mapping[CellProfilerRuntimeValues, int] | None = None,
    ) -> dict[CellProfilerRuntimeValues, int | None]:
        unique_axis_keys = tuple(
            dict.fromkeys(axis_key for _object_id, axis_key in row_keys)
        )
        return {
            axis_key: self.positive_extent_for_axis(
                axis_key=axis_key,
                label_payload=label_payload,
                row_policy=row_policy,
                measured_positive_extent_by_axis=measured_positive_extent_by_axis,
            )
            for axis_key in unique_axis_keys
        }

    def positive_extent_for_axis(
        self,
        *,
        axis_key: CellProfilerRuntimeValues,
        label_payload: CellProfilerRuntimeValue,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
        measured_positive_extent_by_axis: Mapping[CellProfilerRuntimeValues, int] | None,
    ) -> int | None:
        """Return missing-value extent without shrinking the declared label domain."""
        label_extent = self.positive_extent_for_missing_measurements(
            label_payload=label_payload,
            axis_key=axis_key,
            row_policy=row_policy,
        )
        measured_extent = (
            measured_positive_extent_by_axis.get(axis_key)
            if measured_positive_extent_by_axis is not None
            else None
        )
        if self.axis_fields and measured_extent is not None:
            return measured_extent
        if label_extent is None:
            return measured_extent
        if measured_extent is None:
            return label_extent
        return max(label_extent, measured_extent)

    def missing_rows(
        self,
        *,
        missing_row_keys: Sequence[tuple[int, CellProfilerRuntimeValues]],
        label_payload: CellProfilerRuntimeValue,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
        measured_positive_extent_by_axis: Mapping[CellProfilerRuntimeValues, int] | None = None,
    ) -> tuple[CellProfilerKwargDict, ...]:
        positive_extent_by_axis = self.positive_extent_by_axis(
            label_payload=label_payload,
            row_policy=row_policy,
            row_keys=missing_row_keys,
            measured_positive_extent_by_axis=measured_positive_extent_by_axis,
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

    def missing_row(
        self,
        *,
        object_id: int,
        axis_key: CellProfilerRuntimeValueSequence,
        label_payload: CellProfilerRuntimeValue,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
        positive_label_extent: int | None = None,
    ) -> CellProfilerKwargDict:
        axis_values = self.axis_values_for_key(axis_key)
        row: CellProfilerKwargDict = {}
        for field_name in self.field_names:
            if (
                field_name in MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS
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
        object_identity = row_policy.object_identity_for_label_payload(label_payload)
        if object_identity is not MeasurementObjectRowIdentity.LABEL_ID:
            row[MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD] = object_identity.value
        return row

    def axis_values_for_key(self, axis_key: CellProfilerRuntimeValueSequence) -> CellProfilerKwargDict:
        if len(axis_key) > len(self.axis_fields):
            raise ValueError(
                "Measurement axis key has more values than axis fields; got "
                f"{tuple(axis_key)!r} for fields {tuple(self.axis_fields)!r}."
            )
        return dict(zip(self.axis_fields, axis_key, strict=False))


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowIdentityProjectionResult:
    """Rows plus their nominal object/axis identity after projection."""

    rows: CellProfilerRuntimeValues
    row_keys: "ObjectMeasurementProjectedRowKeys"
    measured_row_keys: "ObjectMeasurementProjectedRowKeys"
    axis_keys: tuple[CellProfilerRuntimeValues, ...]

    def measured_positive_extent_by_axis(self) -> dict[CellProfilerRuntimeValues, int]:
        """Return the largest measured object ID represented on each axis."""
        return self.measured_row_keys.measured_positive_extent_by_axis()

    def within_axis_domain(
        self,
        *,
        axis_keys: Sequence[CellProfilerRuntimeValues],
        object_ids_by_axis: ObjectMeasurementIdsByAxis,
    ) -> "ObjectMeasurementRowIdentityProjectionResult":
        """Return projected rows constrained to the required object/axis domain."""
        required_row_keys = set(
            ObjectMeasurementProjectedRowKeys.required_axis_domain(
                axis_keys,
                object_ids_by_axis,
            )
        )
        row_pairs = tuple(
            (row, row_key)
            for row, row_key in zip(self.rows, self.row_keys, strict=True)
            if self.row_key_is_within_axis_domain(row_key, required_row_keys)
        )
        measured_row_keys = tuple(
            row_key
            for row_key in self.measured_row_keys
            if self.row_key_is_within_axis_domain(row_key, required_row_keys)
        )
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=tuple(row for row, _row_key in row_pairs),
            row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(row_key for _row, row_key in row_pairs)
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
        axis_keys: Sequence[CellProfilerRuntimeValues],
        rows: CellProfilerRuntimeValueSequence | None = None,
        row_keys: "ObjectMeasurementProjectedRowKeys | None" = None,
    ) -> list[CellProfilerRuntimeValue]:
        """Return rows in dense object/axis order using projected identities."""
        ordered_rows = self.rows if rows is None else rows
        ordered_row_keys = self.row_keys if row_keys is None else row_keys
        object_order = {object_id: index for index, object_id in enumerate(object_ids)}
        axis_order = {axis_key: index for index, axis_key in enumerate(axis_keys)}
        indexed_rows = tuple(
            enumerate(zip(ordered_rows, ordered_row_keys, strict=True))
        )
        return [
            row
            for _index, (row, _row_key) in sorted(
                indexed_rows,
                key=lambda item: self.row_order_key(
                    item[1][1],
                    item[0],
                    object_order=object_order,
                    axis_order=axis_order,
                ),
            )
        ]

    @staticmethod
    def row_order_key(
        row_key: tuple[int | None, CellProfilerRuntimeValues],
        fallback_index: int,
        *,
        object_order: Mapping[int, int],
        axis_order: Mapping[CellProfilerRuntimeValues, int],
    ) -> tuple[int, int, int]:
        """Return a stable ordering key for one projected measurement row."""
        object_id, axis_key = row_key
        axis_order_index = MappingValueLookup(axis_order, axis_key).value_or(
            len(axis_order)
        )
        object_order_index = len(object_order)
        if object_id is not None:
            object_order_index = MappingValueLookup(
                object_order,
                object_id,
            ).value_or(len(object_order))
        return (
            axis_order_index,
            object_order_index,
            fallback_index,
        )


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
            object_id
            for object_id, _axis_key in self.entries
            if object_id is not None
        )
        if not object_ids:
            return None
        return max(object_ids)

    def has_object_ids(self) -> bool:
        """Return whether any projected row has object identity."""
        return any(object_id is not None for object_id, _axis_key in self.entries)

    def axis_keys(self) -> tuple[CellProfilerRuntimeValues, ...]:
        """Return ordered unique measurement-axis keys represented by entries."""
        return tuple(dict.fromkeys(axis_key for _object_id, axis_key in self.entries))

    def measured_positive_extent_by_axis(self) -> dict[CellProfilerRuntimeValues, int]:
        """Return the largest measured object ID represented on each axis."""
        extents: dict[CellProfilerRuntimeValues, int] = {}
        for object_id, axis_key in self.entries:
            if object_id is None:
                continue
            current_extent = extents.get(axis_key, 0)
            if object_id > current_extent:
                extents[axis_key] = object_id
        return extents

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
        axis_keys: Sequence[CellProfilerRuntimeValues],
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
        axis_keys: Sequence[CellProfilerRuntimeValues],
        object_ids_by_axis: ObjectMeasurementIdsByAxis,
    ) -> ObjectMeasurementSliceRowKeys:
        """Return required row keys not already represented by projected keys."""
        required_row_keys = self.required_axis_domain(axis_keys, object_ids_by_axis)
        present_row_keys = self.present_in(set(required_row_keys))
        return [
            row_key
            for row_key in required_row_keys
            if row_key not in present_row_keys
        ]

    def appended(
        self,
        entries: Sequence[ObjectMeasurementProjectedRowKey],
    ) -> "ObjectMeasurementProjectedRowKeys":
        """Return keys with appended completion entries."""
        return ObjectMeasurementProjectedRowKeys((*self.entries, *tuple(entries)))


@dataclass(slots=True)
class ObjectMeasurementRowOrdinalProjectionState:
    """Mutable ordinal ownership state for one compact row projection pass."""

    ordinal_by_axis: dict[CellProfilerRuntimeValues, int] = field(default_factory=dict)
    ordinal_by_original_id: dict[tuple[CellProfilerRuntimeValues, int], int] = field(
        default_factory=dict
    )

    def register_measured_object(
        self,
        row_mapping: MeasurementRowMapping,
        *,
        axis_key: CellProfilerRuntimeValues,
        object_id_field: str,
    ) -> None:
        """Register a measured source object before compact row projection."""
        original_id = measurement_object_label(
            row_mapping,
            object_id_field=object_id_field,
        )
        ordinal_key = (axis_key, original_id) if original_id is not None else None
        ordinal = (
            self.ordinal_by_original_id.get(ordinal_key)
            if ordinal_key is not None
            else None
        )
        if ordinal is not None:
            return
        ordinal = MappingValueLookup(self.ordinal_by_axis, axis_key).value_or(0) + 1
        self.ordinal_by_axis[axis_key] = ordinal
        if ordinal_key is not None:
            self.ordinal_by_original_id[ordinal_key] = ordinal

    def ordinal_for_measured_object(
        self,
        row_mapping: MeasurementRowMapping,
        *,
        axis_key: CellProfilerRuntimeValues,
        object_id_field: str,
    ) -> int:
        """Return the compact row ordinal for a registered measured object."""
        original_id = measurement_object_label(
            row_mapping,
            object_id_field=object_id_field,
        )
        ordinal_key = (axis_key, original_id) if original_id is not None else None
        if ordinal_key is not None:
            return self.ordinal_by_original_id[ordinal_key]
        return self.next_unbound_ordinal(axis_key)

    def next_unbound_ordinal(self, axis_key: CellProfilerRuntimeValues) -> int:
        """Allocate an ordinal for retained rows that have no measured object."""
        ordinal = MappingValueLookup(self.ordinal_by_axis, axis_key).value_or(0) + 1
        self.ordinal_by_axis[axis_key] = ordinal
        return ordinal


class MeasurementObjectRowIdentityProjectionStrategy(
    EnumStrategyLabelRegistryMixin,
    metaclass=AutoRegisterMeta,
):
    """Registered projection from source object IDs to exported row identity."""

    __enum_member_attr__ = "object_identity"
    object_identity: ClassVar[MeasurementObjectRowIdentity]

    @abstractmethod
    def project_rows(
        self,
        request: ObjectMeasurementRowIdentityProjectionRequest,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        """Return rows with the requested object-row identity."""

    @abstractmethod
    def object_ids_for_label_ids(self, label_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Return required exported object IDs for a source label-ID domain."""

    def row_with_object_id(
        self,
        request: ObjectMeasurementRowIdentityProjectionRequest,
        row: CellProfilerRuntimeValue,
        object_id: int,
    ) -> CellProfilerKwargDict:
        """Return a row projected into this strategy's object identity domain."""
        projected_row = request.row_with_object_id(row, object_id)
        if request.row_policy.annotates_projected_object_identity(
            measurement_row_mapping(projected_row),
            object_identity=type(self).object_identity,
        ):
            projected_row[MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD] = (
                type(self).object_identity.value
            )
        return projected_row


class LabelIdMeasurementObjectRowIdentityProjectionStrategy(
    MeasurementObjectRowIdentityProjectionStrategy
):
    """Preserve label IDs as exported object-row identities."""

    object_identity = MeasurementObjectRowIdentity.LABEL_ID

    def object_ids_for_label_ids(self, label_ids: tuple[int, ...]) -> tuple[int, ...]:
        return label_ids

    def project_rows(
        self,
        request: ObjectMeasurementRowIdentityProjectionRequest,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        rows = tuple(request.rows)
        row_keys: list[ObjectMeasurementProjectedRowKey] = []
        measured_row_keys: list[ObjectMeasurementProjectedRowKey] = []
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            row_key = (
                request.object_label(row),
                request.axis_key_from_mapping(row_mapping),
            )
            row_keys.append(row_key)
            if row_key[0] is None:
                continue
            if request.row_policy.row_has_measured_object(
                row_mapping,
                object_id_field=request.object_id_field,
                axis_fields=request.axis_fields,
            ):
                measured_row_keys.append(row_key)
        projected_row_keys = ObjectMeasurementProjectedRowKeys(row_keys)
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
    MeasurementObjectRowIdentityProjectionStrategy
):
    """Project measured objects into CP's compact row-ordinal identity."""

    object_identity = MeasurementObjectRowIdentity.ROW_ORDINAL

    def project_rows(
        self,
        request: ObjectMeasurementRowIdentityProjectionRequest,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        ordinal_state = ObjectMeasurementRowOrdinalProjectionState()
        row_entries: list[tuple[CellProfilerRuntimeValue, MeasurementRowMapping, CellProfilerRuntimeValues, bool]] = []
        for row in request.rows:
            row_mapping = measurement_row_mapping(row)
            axis_key = request.axis_key_from_mapping(row_mapping)
            measured = request.row_policy.row_has_measured_object(
                row_mapping,
                object_id_field=request.object_id_field,
                axis_fields=request.axis_fields,
            )
            row_entries.append((row, row_mapping, axis_key, measured))
            if not measured:
                continue
            ordinal_state.register_measured_object(
                row_mapping,
                axis_key=axis_key,
                object_id_field=request.object_id_field,
            )

        projected_rows: list[CellProfilerRuntimeValue] = []
        projected_row_keys: ObjectMeasurementSliceRowKeys = []
        measured_projected_row_keys: ObjectMeasurementSliceRowKeys = []
        for row, row_mapping, axis_key, measured in row_entries:
            if measured:
                ordinal = ordinal_state.ordinal_for_measured_object(
                    row_mapping,
                    axis_key=axis_key,
                    object_id_field=request.object_id_field,
                )
            else:
                if not request.row_policy.retains_unmeasured_compact_row(
                    row_mapping,
                    object_id_field=request.object_id_field,
                    axis_fields=request.axis_fields,
                ):
                    continue
                ordinal = ordinal_state.next_unbound_ordinal(axis_key)
            projected_rows.append(self.row_with_object_id(request, row, ordinal))
            projected_row_keys.append((ordinal, axis_key))
            if measured:
                measured_projected_row_keys.append((ordinal, axis_key))
        object_ids = tuple(
            sorted(dict.fromkeys(ordinal for ordinal, _axis_key in projected_row_keys))
        )
        axis_keys = tuple(
            dict.fromkeys(
                axis_key for _row, _mapping, axis_key, _measured in row_entries
            )
        )
        projection = ObjectMeasurementRowIdentityProjectionResult(
            rows=tuple(projected_rows),
            row_keys=ObjectMeasurementProjectedRowKeys(tuple(projected_row_keys)),
            measured_row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(measured_projected_row_keys)
            ),
            axis_keys=axis_keys,
        )
        object_order = {object_id: index for index, object_id in enumerate(object_ids)}
        axis_order = {axis_key: index for index, axis_key in enumerate(axis_keys)}
        ordered_entries = tuple(
            sorted(
                enumerate(zip(projection.rows, projection.row_keys, strict=True)),
                key=lambda item: projection.row_order_key(
                    item[1][1],
                    item[0],
                    object_order=object_order,
                    axis_order=axis_order,
                ),
            )
        )
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=tuple(row for _index, (row, _row_key) in ordered_entries),
            row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(row_key for _index, (_row, row_key) in ordered_entries)
            ),
            measured_row_keys=projection.measured_row_keys,
            axis_keys=axis_keys,
        )


class RowSequenceMeasurementObjectRowIdentityProjectionStrategy(
    CompactMeasurementObjectIdProjectionMixin,
    MeasurementObjectRowIdentityProjectionStrategy
):
    """Project each measured source row to its own compact row ordinal."""

    object_identity = MeasurementObjectRowIdentity.ROW_SEQUENCE

    def project_rows(
        self,
        request: ObjectMeasurementRowIdentityProjectionRequest,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        ordinal_by_axis: dict[CellProfilerRuntimeValues, int] = {}
        projected_rows: list[CellProfilerRuntimeValue] = []
        projected_row_keys: ObjectMeasurementSliceRowKeys = []
        measured_projected_row_keys: ObjectMeasurementSliceRowKeys = []
        axis_keys: list[CellProfilerRuntimeValues] = []
        for row in request.rows:
            row_mapping = measurement_row_mapping(row)
            axis_key = request.axis_key_from_mapping(row_mapping)
            axis_keys.append(axis_key)
            measured = request.row_policy.row_has_measured_object(
                row_mapping,
                object_id_field=request.object_id_field,
                axis_fields=request.axis_fields,
            )
            if not measured and not request.row_policy.retains_unmeasured_compact_row(
                row_mapping,
                object_id_field=request.object_id_field,
                axis_fields=request.axis_fields,
            ):
                continue
            ordinal = MappingValueLookup(ordinal_by_axis, axis_key).value_or(0) + 1
            ordinal_by_axis[axis_key] = ordinal
            projected_rows.append(self.row_with_object_id(request, row, ordinal))
            projected_row_keys.append((ordinal, axis_key))
            if measured:
                measured_projected_row_keys.append((ordinal, axis_key))
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=tuple(projected_rows),
            row_keys=ObjectMeasurementProjectedRowKeys(tuple(projected_row_keys)),
            measured_row_keys=ObjectMeasurementProjectedRowKeys(
                tuple(measured_projected_row_keys)
            ),
            axis_keys=tuple(dict.fromkeys(axis_keys)),
        )
