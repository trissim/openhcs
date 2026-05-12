"""Nominal runtime-slice projection contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.image_shapes import is_color_image_slice, is_color_image_stack
from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.runtime_artifact_queries import (
    MEASUREMENT_OBJECT_NAME_FIELD,
    columnar_row_values,
    measurement_row_mapping,
    measurement_rows,
    measurement_table_for_slice,
    measurement_table_object_name,
)
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    ParentChildRelationshipPayload,
    RuntimePlaneAxis,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_values import (
    ColumnarRows,
    MeasurementTable,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectRelationship,
    SparseIJVLabelRows,
    collapse_singleton_object_label_stack,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
    object_label_dense_array,
    project_image_mask_to_data_domain,
)


@dataclass(frozen=True, slots=True)
class RuntimeSliceProjectionContext:
    """Execution context for projecting one runtime slice from a value."""

    slice_index: int
    slice_count: int

    def __post_init__(self) -> None:
        if self.slice_count <= 0:
            raise ValueError("RuntimeSliceProjectionContext.slice_count must be positive.")
        if self.slice_index < 0 or self.slice_index >= self.slice_count:
            raise ValueError(
                "RuntimeSliceProjectionContext.slice_index must be within "
                f"[0, {self.slice_count}), got {self.slice_index}."
)


@dataclass(frozen=True, slots=True)
class MeasurementTableRepeatedScalarGroupKey:
    """Nominal identity for append-ordered scalar measurement table groups."""

    table_name: str
    object_name: str | None
    source_image_name: str | None

    @classmethod
    def from_table(cls, table: MeasurementTable) -> "MeasurementTableRepeatedScalarGroupKey":
        """Return the repeated-scalar group identity declared by one table."""
        return cls(
            table_name=table.name,
            object_name=measurement_table_object_name(table),
            source_image_name=table.source_image_name,
        )


class RuntimeSliceProjectionStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered strategy for projecting values into runtime-slice scope."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True
    value_type: ClassVar[type[Any] | tuple[type[Any], ...] | None] = None
    value_type_label: ClassVar[str | None] = None

    @classmethod
    def strategy_for_value(cls, value: Any) -> "RuntimeSliceProjectionStrategy":
        strategy = cls.for_nominal_value(value)
        return strategy if strategy is not None else DefaultRuntimeSliceProjectionStrategy()

    @abstractmethod
    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        """Return the value projected into one runtime slice."""

    def stack_views(self, value: Any) -> tuple[np.ndarray, ...]:
        """Return stack views that declare runtime-slice cardinality."""
        stack = self.stack_view(value)
        return () if stack is None else (stack,)

    def stack_view(
        self,
        value: Any,
        *,
        slice_count: int | None = None,
    ) -> np.ndarray | None:
        """Return a plane-stack view when this value carries slice planes."""
        return RuntimeSliceProjection.stack_view(value, slice_count=slice_count)


class RuntimeSliceProjection:
    """SSOT for runtime-slice count and value projection."""

    @classmethod
    def value_for_slice(
        cls,
        value: Any,
        slice_index: int,
        slice_count: int,
    ) -> Any:
        return RuntimeSliceProjectionStrategy.strategy_for_value(value).value_for_slice(
            value,
            RuntimeSliceProjectionContext(
                slice_index=slice_index,
                slice_count=slice_count,
            ),
        )

    @classmethod
    def object_label_endpoint(
        cls,
        value: Any,
        *,
        slice_index: int | None = None,
        slice_count: int | None = None,
    ) -> Any:
        """Resolve one object-label endpoint through runtime-slice semantics."""
        if slice_index is None:
            return value
        effective_slice_count = slice_count or cls.slice_count_from_values((value,))
        if effective_slice_count is None:
            stack_counts = {
                int(stack.shape[0])
                for stack in cls.stack_views(value)
                if stack.shape[0] > slice_index
            }
            effective_slice_count = cls.single_slice_count(
                stack_counts,
                source_description="object-label endpoint stack views",
            )
        if effective_slice_count is None:
            raise ValueError(
                "Cannot project object-label endpoint without a declared "
                "runtime slice count."
            )
        return cls.value_for_slice(value, slice_index, effective_slice_count)

    @classmethod
    def object_label_endpoint_dense_array(
        cls,
        value: Any,
        *,
        slice_index: int | None = None,
        slice_count: int | None = None,
        dtype: object | None = None,
    ) -> np.ndarray:
        """Resolve and materialize one object-label endpoint as dense labels."""
        return object_label_dense_array(
            cls.object_label_endpoint(
                value,
                slice_index=slice_index,
                slice_count=slice_count,
            ),
            dtype=dtype,
        )

    @classmethod
    def kwargs_for_slice(
        cls,
        kwargs: Mapping[str, Any],
        slice_index: int,
        slice_count: int,
        *,
        sequence_kwargs: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        return {
            name: (
                tuple(
                    cls.value_for_slice(item, slice_index, slice_count)
                    for item in value
                )
                if name in sequence_kwargs and isinstance(value, tuple)
                else cls.value_for_slice(value, slice_index, slice_count)
            )
            for name, value in kwargs.items()
        }

    @classmethod
    def slice_count_from_values(cls, values: Any) -> int | None:
        tensor_slice_counts = {
            stack.shape[0]
            for value in values
            for stack in cls.stack_views(value)
            if stack.shape[0] > 1
        }
        tensor_slice_counts.update(
            value.slice_count
            for value in values
            if isinstance(value, RuntimeSliceAlignedValueSet)
            and value.slice_count >= 1
        )
        tensor_slice_counts.update(
            count
            for value in values
            if isinstance(value, (ParentChildRelationshipPayload, ObjectRelationship))
            for count in (cls.relationship_slice_count(value),)
            if count is not None and count > 1
        )
        tensor_slice_count = cls.single_slice_count(
            tensor_slice_counts,
            source_description="tensor/vector values",
        )
        if tensor_slice_count is not None:
            return tensor_slice_count

        measurement_table_slice_count = cls.measurement_table_collection_slice_count(
            values
        )
        if measurement_table_slice_count is not None:
            return measurement_table_slice_count

        if any(
            stack.shape[0] == 1
            for value in values
            for stack in cls.stack_views(value)
        ):
            return 1
        return None

    @classmethod
    def stack_views(cls, value: Any) -> tuple[np.ndarray, ...]:
        return RuntimeSliceProjectionStrategy.strategy_for_value(value).stack_views(value)

    @classmethod
    def stack_view(
        cls,
        value: Any,
        *,
        slice_count: int | None = None,
    ) -> np.ndarray | None:
        if isinstance(value, (str, bytes, bytearray, Mapping)):
            return None
        if isinstance(value, (tuple, list)):
            try:
                value = np.asarray(value)
            except ValueError:
                return None
        elif not isinstance(value, np.ndarray):
            try:
                value = np.asarray(value)
            except ValueError:
                return None
        return cls.grayscale_plane_stack_view(value, slice_count=slice_count)

    @classmethod
    def grayscale_plane_stack_view(
        cls,
        value: Any,
        *,
        slice_count: int | None = None,
        flatten_high_rank: bool = False,
    ) -> np.ndarray | None:
        array = image_payload_data(value)
        if is_color_image_slice(array) or is_color_image_stack(array):
            return None
        if not isinstance(array, np.ndarray):
            return None
        if array.ndim < 3:
            return None
        if array.ndim > 3:
            if flatten_high_rank:
                return array.reshape((-1, *array.shape[-2:]))
            if slice_count is not None:
                if array.shape[0] == slice_count:
                    return array
                flattened = array.reshape((-1, *array.shape[-2:]))
                if flattened.shape[0] == slice_count:
                    return flattened
            return array
        return array.reshape((-1, *array.shape[-2:]))

    @classmethod
    def first_axis_slice_if_aligned(
        cls,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any | None:
        array = image_payload_data(value)
        if is_color_image_slice(array) or is_color_image_stack(array):
            return None
        if not isinstance(array, np.ndarray):
            return None
        if array.ndim < 3 or array.shape[0] != context.slice_count:
            return None
        return array[context.slice_index]

    @classmethod
    def grouped_label_stack_slice(
        cls,
        stack: np.ndarray,
        context: RuntimeSliceProjectionContext,
    ) -> Any | None:
        if stack.shape[0] <= context.slice_count:
            return None
        if stack.shape[0] % context.slice_count != 0:
            return None
        dtype = getattr(stack, "dtype", None)
        if dtype is None:
            return None
        if not (
            np.issubdtype(dtype, np.integer)
            or np.issubdtype(dtype, np.bool_)
        ):
            return None
        grouped = stack[context.slice_index :: context.slice_count]
        if grouped.shape[0] == 1:
            return grouped[0]
        if np.issubdtype(dtype, np.bool_):
            return np.any(grouped, axis=0)
        return np.max(grouped, axis=0).astype(dtype, copy=False)

    @staticmethod
    def relationship_slice_count(
        value: ParentChildRelationshipPayload | ObjectRelationship,
    ) -> int | None:
        if value.slice_count is not None:
            return value.slice_count
        if not value.slice_indices:
            return None
        return max(value.slice_indices) + 1

    @staticmethod
    def measurement_table_slice_count(value: Any) -> int | None:
        if isinstance(value, (tuple, list)):
            return RuntimeSliceProjection.measurement_table_collection_slice_count(value)
        if not isinstance(value, MeasurementTable):
            return None
        if not value.subject.scope.projects_runtime_slices:
            return None
        slice_indices = RuntimeSliceProjection.measurement_table_slice_indices(value)
        if not slice_indices:
            return None
        expected_indices = set(range(max(slice_indices) + 1))
        if slice_indices != expected_indices:
            raise ValueError(
                f"MeasurementTable '{value.name}' has non-contiguous slice_index "
                f"values {sorted(slice_indices)}; expected "
                f"{sorted(expected_indices)}."
            )
        return len(expected_indices)

    @staticmethod
    def measurement_table_effective_slice_count(value: MeasurementTable) -> int:
        """Return declared measurement-table slice count, treating scalar tables as one slice."""
        if RuntimeSliceProjection.measurement_table_declares_no_slice_index(value):
            return 1
        return RuntimeSliceProjection.measurement_table_slice_count(value) or 1

    @staticmethod
    def measurement_table_declares_no_slice_index(value: MeasurementTable) -> bool:
        """Return whether declared fields make row-level slice scanning unnecessary."""
        if not value.fields:
            return False
        slice_field = MeasurementRowAxisField.SLICE_INDEX.value
        return all(field.name != slice_field for field in value.fields)

    @staticmethod
    def measurement_table_collection_slice_count(values: Any) -> int | None:
        slice_indices: set[int] = set()
        slice_counts: set[int] = set()
        for value in values:
            if isinstance(value, MeasurementTable):
                table_indices = RuntimeSliceProjection.measurement_table_slice_indices(
                    value
                )
                if table_indices:
                    slice_indices.update(table_indices)
                continue
            count = RuntimeSliceProjection.measurement_table_slice_count(value)
            if count is not None:
                slice_counts.add(count)
        if slice_indices:
            expected_indices = set(range(max(slice_indices) + 1))
            if slice_indices != expected_indices:
                raise ValueError(
                    "Measurement table collection has non-contiguous slice_index "
                    f"values {sorted(slice_indices)}; expected "
                    f"{sorted(expected_indices)}."
                )
            slice_counts.add(len(expected_indices))
        return RuntimeSliceProjection.single_slice_count(
            slice_counts,
            source_description="measurement table collection values",
        )

    @staticmethod
    def measurement_table_slice_indices(value: MeasurementTable) -> set[int]:
        if not value.subject.scope.projects_runtime_slices:
            return set()
        slice_field = MeasurementRowAxisField.SLICE_INDEX.value
        if isinstance(value.rows, ColumnarRows):
            column_names = tuple(str(column) for column in value.rows.columns)
            if slice_field not in column_names:
                return set()
            return {
                int(slice_index)
                for slice_index in columnar_row_values(value.rows, slice_field)
                if slice_index is not None
            }
        return {
            int(row[slice_field])
            for row in value.rows
            if isinstance(row, Mapping) and row.get(slice_field) is not None
        }

    @staticmethod
    def measurement_table_matches_object(
        table: MeasurementTable,
        object_name: str,
    ) -> bool:
        """Return whether a measurement table declares or rows-match an object name."""
        table_object_name = measurement_table_object_name(table)
        if table_object_name is not None:
            return table_object_name == object_name
        return any(
            measurement_row_mapping(row).get(MEASUREMENT_OBJECT_NAME_FIELD) == object_name
            for row in measurement_rows((table,))
        )

    @staticmethod
    def measurement_table_with_slice_offset(
        table: MeasurementTable,
        slice_offset: int,
    ) -> MeasurementTable:
        """Return a table with row slice indexes shifted by ``slice_offset``."""
        if slice_offset == 0:
            return table
        return MeasurementTable(
            name=table.name,
            rows=[
                {
                    **dict(measurement_row_mapping(row)),
                    "slice_index": int(
                        measurement_row_mapping(row).get("slice_index", 0)
                    )
                    + slice_offset,
                }
                for row in measurement_rows((table,))
            ],
            object_name=table.object_name,
            fields=table.fields,
            object_id_field=table.object_id_field,
            source_image_name=table.source_image_name,
            subject=table.subject,
        )

    @staticmethod
    def measurement_table_broadcast_to_slice_count(
        table: MeasurementTable,
        slice_count: int,
    ) -> MeasurementTable:
        """Return a scalar table repeated across every runtime slice."""
        if slice_count <= 1:
            return table
        return MeasurementTable(
            name=table.name,
            rows=[
                {
                    **dict(measurement_row_mapping(row)),
                    "slice_index": slice_index,
                }
                for slice_index in range(slice_count)
                for row in measurement_rows((table,))
            ],
            object_name=table.object_name,
            fields=table.fields,
            object_id_field=table.object_id_field,
            source_image_name=table.source_image_name,
            subject=table.subject,
        )

    @staticmethod
    def measurement_tables_with_repeated_scalar_slice_offsets(
        tables: tuple[MeasurementTable, ...],
    ) -> tuple[MeasurementTable, ...]:
        """Offset repeated scalar measurement tables onto consecutive slice indexes."""
        grouped: dict[tuple[str, str | None, str | None], list[int]] = {}
        for index, table in enumerate(tables):
            grouped.setdefault(
                (table.name, measurement_table_object_name(table), table.source_image_name),
                [],
            ).append(index)

        aligned = list(tables)
        for indexes in grouped.values():
            if len(indexes) <= 1:
                continue
            if any(
                RuntimeSliceProjection.measurement_table_effective_slice_count(
                    tables[index]
                )
                != 1
                for index in indexes
            ):
                continue
            for slice_offset, table_index in enumerate(indexes):
                aligned[table_index] = (
                    RuntimeSliceProjection.measurement_table_with_slice_offset(
                        tables[table_index],
                        slice_offset,
                    )
                )
        return tuple(aligned)

    @staticmethod
    def measurement_table_appended_with_repeated_scalar_slice_offset(
        existing_tables: tuple[MeasurementTable, ...],
        table: MeasurementTable,
    ) -> MeasurementTable:
        """Return a new table offset for append-only repeated scalar table indexing."""
        scalar_group_size = sum(
            1
            for existing_table in existing_tables
            if RuntimeSliceProjection.measurement_tables_share_repeated_scalar_group(
                existing_table,
                table,
            )
        )
        if scalar_group_size == 0:
            return table
        if RuntimeSliceProjection.measurement_table_effective_slice_count(table) != 1:
            return table
        return RuntimeSliceProjection.measurement_table_with_slice_offset(
            table,
            scalar_group_size,
        )

    @staticmethod
    def measurement_tables_share_repeated_scalar_group(
        left: MeasurementTable,
        right: MeasurementTable,
    ) -> bool:
        """Return whether two tables participate in one repeated scalar group."""
        return (
            RuntimeSliceProjection.measurement_table_repeated_scalar_group_key(left)
            == RuntimeSliceProjection.measurement_table_repeated_scalar_group_key(right)
        )

    @staticmethod
    def measurement_table_repeated_scalar_group_key(
        table: MeasurementTable,
    ) -> MeasurementTableRepeatedScalarGroupKey:
        """Return the nominal repeated-scalar group identity for one table."""
        return MeasurementTableRepeatedScalarGroupKey.from_table(table)

    @staticmethod
    def single_slice_count(
        slice_counts: set[int],
        *,
        source_description: str,
    ) -> int | None:
        if not slice_counts:
            return None
        if len(slice_counts) > 1:
            raise ValueError(
                f"Conflicting runtime slice counts from {source_description}: "
                f"{sorted(slice_counts)!r}."
            )
        return next(iter(slice_counts))


class RuntimeSliceAlignedValueProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project nominal slice-aligned value sets through their own contract."""

    value_type = RuntimeSliceAlignedValueSet

    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        if not isinstance(value, RuntimeSliceAlignedValueSet):
            raise TypeError("RuntimeSliceAlignedValueProjectionStrategy requires RuntimeSliceAlignedValueSet.")
        return value.value_for_slice(context.slice_index)

    def stack_views(self, value: Any) -> tuple[np.ndarray, ...]:
        return ()


class MeasurementTableRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project measurement tables by their declared slice-index rows."""

    value_type = MeasurementTable

    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        if not isinstance(value, MeasurementTable):
            raise TypeError("MeasurementTableRuntimeSliceProjectionStrategy requires MeasurementTable.")
        return measurement_table_for_slice(value, context.slice_index)


class ParentChildRelationshipRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project relationship payloads by their runtime slice ids."""

    value_type = ParentChildRelationshipPayload

    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        if not isinstance(value, ParentChildRelationshipPayload):
            raise TypeError(
                "ParentChildRelationshipRuntimeSliceProjectionStrategy requires "
                "ParentChildRelationshipPayload."
            )
        if not value.slice_indices:
            if value.slice_count is not None and value.slice_count > 1 and value.parent_ids:
                raise ValueError(
                    "Cannot slice multi-plane ParentChildRelationshipPayload "
                    "without slice_indices."
                )
            return value
        parent_ids: list[int] = []
        child_ids: list[int] = []
        for parent_id, child_id, relationship_slice_index in zip(
            value.parent_ids,
            value.child_ids,
            value.slice_indices,
            strict=True,
        ):
            if relationship_slice_index != context.slice_index:
                continue
            parent_ids.append(parent_id)
            child_ids.append(child_id)
        return ParentChildRelationshipPayload(
            parent_ids=tuple(parent_ids),
            child_ids=tuple(child_ids),
            slice_count=1,
        )


class ObjectRelationshipRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project object relationship wrappers by runtime slice ids."""

    value_type = ObjectRelationship

    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        if not isinstance(value, ObjectRelationship):
            raise TypeError("ObjectRelationshipRuntimeSliceProjectionStrategy requires ObjectRelationship.")
        source_ids_all = tuple(int(source_id) for source_id in value.source_ids)
        target_ids_all = tuple(int(target_id) for target_id in value.target_ids)
        if not value.slice_indices:
            if value.slice_count is not None and value.slice_count > 1 and source_ids_all:
                raise ValueError(
                    "Cannot slice multi-plane ObjectRelationship without "
                    "slice_indices."
                )
            return value
        source_ids: list[int] = []
        target_ids: list[int] = []
        for source_id, target_id, relationship_slice_index in zip(
            source_ids_all,
            target_ids_all,
            value.slice_indices,
            strict=True,
        ):
            if relationship_slice_index != context.slice_index:
                continue
            source_ids.append(source_id)
            target_ids.append(target_id)
        return ObjectRelationship(
            name=value.name,
            source=value.source,
            target=value.target,
            source_ids=tuple(source_ids),
            target_ids=tuple(target_ids),
            relationship_type=value.relationship_type,
            slice_count=1,
        )


class SparseIJVLabelRowsRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project sparse IJV label rows by their declared slice-index column."""

    value_type = SparseIJVLabelRows

    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        if not isinstance(value, SparseIJVLabelRows):
            raise TypeError(
                "SparseIJVLabelRowsRuntimeSliceProjectionStrategy requires "
                "SparseIJVLabelRows."
            )
        return value.slice(context.slice_index)

    def stack_views(self, value: Any) -> tuple[np.ndarray, ...]:
        if not isinstance(value, SparseIJVLabelRows):
            raise TypeError(
                "SparseIJVLabelRowsRuntimeSliceProjectionStrategy requires "
                "SparseIJVLabelRows."
            )
        return ()


class ObjectLabelSetRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project runtime-slice object label sets while preserving label metadata."""

    value_type = ObjectLabelSet

    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        if not isinstance(value, ObjectLabelSet):
            raise TypeError("ObjectLabelSetRuntimeSliceProjectionStrategy requires ObjectLabelSet.")
        if value.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return value
        slice_domain = value.object_label_domain().project_slice(
            context.slice_index,
            context.slice_count,
        )
        return ObjectLabelSet(
            name=value.name,
            labels=RuntimeSliceProjection.value_for_slice(
                value.labels,
                context.slice_index,
                context.slice_count,
            ),
            unedited_labels=(
                None
                if value.unedited_labels is None
                else RuntimeSliceProjection.value_for_slice(
                    value.unedited_labels,
                    context.slice_index,
                    context.slice_count,
                )
            ),
            small_removed_labels=(
                None
                if value.small_removed_labels is None
                else RuntimeSliceProjection.value_for_slice(
                    value.small_removed_labels,
                    context.slice_index,
                    context.slice_count,
                )
            ),
            representation=value.representation,
            declared_object_count=slice_domain.declared_object_count,
            declared_object_ids=slice_domain.declared_object_ids,
            declared_object_id_domains=slice_domain.declared_object_id_domains,
            domain_scope=slice_domain.scope,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            spatial_origin_yx=value.spatial_origin_yx,
            source_spatial_shape_yx=value.source_spatial_shape_yx,
            dimensions=value.dimensions,
            source_image_name=value.source_image_name,
        )

    def stack_views(self, value: Any) -> tuple[np.ndarray, ...]:
        if not isinstance(value, ObjectLabelSet):
            raise TypeError("ObjectLabelSetRuntimeSliceProjectionStrategy requires ObjectLabelSet.")
        if value.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return ()
        return RuntimeSliceProjection.stack_views(value.runtime_payload())


class ObjectLabelPayloadRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project runtime-slice object label payloads while preserving metadata."""

    value_type = ObjectLabelPayload

    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        if not isinstance(value, ObjectLabelPayload):
            raise TypeError("ObjectLabelPayloadRuntimeSliceProjectionStrategy requires ObjectLabelPayload.")
        if value.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return value
        slice_domain = value.object_label_domain().project_slice(
            context.slice_index,
            context.slice_count,
        )
        return ObjectLabelPayload(
            labels=RuntimeSliceProjection.value_for_slice(
                value.labels,
                context.slice_index,
                context.slice_count,
            ),
            unedited_labels=(
                None
                if value.unedited_labels is None
                else RuntimeSliceProjection.value_for_slice(
                    value.unedited_labels,
                    context.slice_index,
                    context.slice_count,
                )
            ),
            small_removed_labels=(
                None
                if value.small_removed_labels is None
                else RuntimeSliceProjection.value_for_slice(
                    value.small_removed_labels,
                    context.slice_index,
                    context.slice_count,
                )
            ),
            declared_object_count=slice_domain.declared_object_count,
            declared_object_ids=slice_domain.declared_object_ids,
            declared_object_id_domains=slice_domain.declared_object_id_domains,
            domain_scope=slice_domain.scope,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            spatial_origin_yx=value.spatial_origin_yx,
            source_spatial_shape_yx=value.source_spatial_shape_yx,
        )

    def stack_views(self, value: Any) -> tuple[np.ndarray, ...]:
        if not isinstance(value, ObjectLabelPayload):
            raise TypeError("ObjectLabelPayloadRuntimeSliceProjectionStrategy requires ObjectLabelPayload.")
        if value.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return ()
        values = (value.labels, value.unedited_labels, value.small_removed_labels)
        return tuple(
            stack
            for item in values
            if item is not None
            if (stack := RuntimeSliceProjection.stack_view(item)) is not None
        )


class SequenceRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project tuple/list containers through their item semantics."""

    value_type = (tuple, list)

    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        stack = (
            RuntimeSliceProjection.stack_view(value)
            if len(value) > 1
            else None
        )
        if stack is not None:
            if stack.shape[0] == context.slice_count:
                return stack[context.slice_index]
            if stack.shape[0] == 1:
                return stack[0]
            return value
        projected = [
            RuntimeSliceProjection.value_for_slice(
                item,
                context.slice_index,
                context.slice_count,
            )
            for item in value
        ]
        return tuple(projected) if isinstance(value, tuple) else projected

    def stack_views(self, value: Any) -> tuple[np.ndarray, ...]:
        stack = RuntimeSliceProjection.stack_view(value) if len(value) > 1 else None
        if stack is not None:
            return (stack,)
        return tuple(
            stack
            for item in value
            for stack in RuntimeSliceProjection.stack_views(item)
        )


class DefaultRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Default projection for image-like array payloads."""

    def value_for_slice(
        self,
        value: Any,
        context: RuntimeSliceProjectionContext,
    ) -> Any:
        metadata = image_payload_metadata(value)
        mask = image_payload_mask(value)
        if mask is not None or metadata.has_values:
            data = RuntimeSliceProjection.value_for_slice(
                image_payload_data(value),
                context.slice_index,
                context.slice_count,
            )
            return image_payload_with_context(
                data=data,
                mask=project_image_mask_to_data_domain(mask, data),
                metadata=metadata.for_channel(context.slice_index),
            )
        axis_sliced = RuntimeSliceProjection.first_axis_slice_if_aligned(
            value,
            context,
        )
        if axis_sliced is not None:
            return axis_sliced
        stack = RuntimeSliceProjection.stack_view(
            value,
            slice_count=context.slice_count,
        )
        if stack is None:
            return value
        if stack.shape[0] == context.slice_count:
            return stack[context.slice_index]
        if stack.shape[0] == 1:
            return stack[0]
        grouped_slice = RuntimeSliceProjection.grouped_label_stack_slice(
            stack,
            context,
        )
        if grouped_slice is not None:
            return grouped_slice
        return value


def collapse_singleton_runtime_slice_value(value: Any) -> Any:
    """Collapse singleton object-label stack wrappers at runtime boundaries."""
    return collapse_singleton_object_label_stack(value)
