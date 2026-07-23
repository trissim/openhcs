"""Object-measurement vector binding authorities for CellProfiler runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time
from types import MappingProxyType
from typing import cast, TYPE_CHECKING

import numpy as np

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
)
from openhcs.core.measurement_row_materialization import (
    measurement_row_has_object_identity,
)
from openhcs.core.runtime_artifact_queries import (
    MeasurementLabelSliceFeatureQuery,
    MeasurementLabelSliceFeatureBatchQuery,
)
from openhcs.core.measurement_feature_queries import MeasurementFeatureQuery
from openhcs.core.measurement_row_materialization import measurement_rows
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
)
from openhcs.core.runtime_tabular_values import (
    measurement_row_mapping,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxisValueProjection,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerSliceAlignedValues,
)
from openhcs.interop.cellprofiler.runtime.object_label_measurements import (
    ObjectLabelMeasurementSliceRequest,
    object_label_measurement_values_cache,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import ObjectMeasurementTableIndex
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter


@dataclass(slots=True)
class CellProfilerMeasurementVector:
    """CellProfiler-facing projection of one object/image measurement vector."""

    slices: tuple[RuntimeCallableArgument, ...]
    plane_projection: RuntimePlaneAxisValueProjection | None = None

    def __post_init__(self) -> None:
        self.slices = tuple(self.slices)

    @property
    def slice_aligned_value(self) -> CellProfilerSliceAlignedValues:
        return CellProfilerSliceAlignedValues(
            tuple(np.asarray(value) for value in self.slices)
        )

    @property
    def runtime_value(self) -> RuntimeCallableArgument:
        if self.plane_projection is None or self.plane_projection.plane_index is not None:
            if len(self.slices) != 1:
                raise ValueError(
                    "A non-stacked object-measurement vector requires exactly one "
                    f"resolved vector, got {len(self.slices)}."
                )
            return np.asarray(self.slices[0])
        return self.slice_aligned_value


@dataclass(frozen=True, slots=True)
class MeasurementImageOperandRow:
    """One image-level measurement row accepted by a CalculateMath operand."""

    row_mapping: Mapping[str, RuntimeCallableArgument]
    value: float
    source: str

    @classmethod
    def from_mapping(
        cls,
        *,
        table_name: str,
        row_mapping: Mapping[str, RuntimeCallableArgument],
        value: float,
    ) -> "MeasurementImageOperandRow":
        return cls(
            row_mapping=row_mapping,
            value=value,
            source=cls.source_text(table_name, row_mapping),
        )

    @staticmethod
    def source_text(
        table_name: str,
        row_mapping: Mapping[str, RuntimeCallableArgument],
    ) -> str:
        row_axis = MeasurementRowAxisField.SLICE_INDEX
        return (
            f"{table_name}:"
            f"{MeasurementImageOperandRow.field_text(row_mapping, MeasurementRowAxisField.SOURCE_IMAGE_NAME.value, '<none>')}:"
            f"{MeasurementImageOperandRow.field_text(row_mapping, MeasurementRowAxisField.FEATURE_NAME.value, '<wide>')}:"
            f"{row_axis.value}="
            f"{MeasurementImageOperandRow.field_text(row_mapping, row_axis.value, '<none>')}"
        )

    @staticmethod
    def field_text(
        row_mapping: Mapping[str, RuntimeCallableArgument],
        field_name: str,
        missing_label: str,
    ) -> str:
        if field_name not in row_mapping:
            return missing_label
        value = row_mapping[field_name]
        if not value:
            return missing_label
        return str(value)

    def has_axis(self, axis: MeasurementRowAxisField) -> bool:
        return axis.value in self.row_mapping and self.row_mapping[axis.value] not in (
            None,
            "",
        )

    def axis_value(self, axis: MeasurementRowAxisField) -> int:
        if not self.has_axis(axis):
            raise ValueError(f"Measurement row does not declare axis {axis.value!r}.")
        return int(float(self.row_mapping[axis.value]))


@dataclass(frozen=True, slots=True)
class MeasurementImageOperandAxisResolution:
    """Explicit row-axis authority for image-level CalculateMath operands."""

    measurement_tables: tuple[MeasurementTable, ...]

    def axis_field(self) -> MeasurementRowAxisField | None:
        if not self.measurement_tables:
            return None
        axis = MeasurementRowAxisField.SLICE_INDEX
        declarations = tuple(
            axis.value in {field.name for field in table.rows.fields}
            for table in self.measurement_tables
        )
        if all(declarations):
            return axis
        if not any(declarations):
            return None
        raise ValueError(
            f"Image-level measurement table schemas disagree about the "
            f"{axis.value!r} axis."
        )


MeasurementImageOperandRowsBySlice = Mapping[
    int,
    tuple[MeasurementImageOperandRow, ...],
]


@dataclass(frozen=True, slots=True)
class MeasurementImageOperandVectorResolution:
    """Resolve image-level CalculateMath operands into slice-aligned values."""

    measurement_tables: tuple[MeasurementTable, ...]
    feature_name: str

    @classmethod
    def from_runtime_feature(
        cls,
        adapter: "CellProfilerRuntimeAdapter",
        feature_name: str,
        *,
        group_key: str | None = None,
    ) -> "MeasurementImageOperandVectorResolution":
        """Build an image-operand resolver from runtime measurement tables."""
        return cls(
            measurement_tables=cls.runtime_axis_scope_tables(
                adapter,
                feature_name,
                group_key=group_key,
            ),
            feature_name=feature_name,
        )

    @classmethod
    def runtime_axis_scope_tables(
        cls,
        adapter: "CellProfilerRuntimeAdapter",
        feature_name: str,
        *,
        group_key: str | None = None,
    ) -> tuple[MeasurementTable, ...]:
        """Return declared image-measurement inputs that carry the feature."""
        query = MeasurementFeatureQuery(
            feature_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )
        return cls.runtime_feature_tables(
            adapter,
            query,
            group_key=group_key,
            match_group=False,
        )

    @staticmethod
    def runtime_feature_tables(
        adapter: "CellProfilerRuntimeAdapter",
        query: MeasurementFeatureQuery,
        *,
        group_key: str | None,
        match_group: bool,
    ) -> tuple[MeasurementTable, ...]:
        """Return visible image-measurement tables that may carry one feature."""
        source_records = adapter.declared_measurement_input_records(
            group_key=group_key,
            match_group=match_group,
        )
        if not source_records:
            raise ValueError(
                f"Image feature {query.feature_name!r} requires a declared "
                "MeasurementsArtifactType runtime input."
            )
        table_records = []
        for record in source_records:
            table = cast(MeasurementTable, record.value.data)
            if (
                query.table_may_carry_feature(table)
                or query.optional_value_index((table,)) is not None
            ):
                table_records.append((record, table))

        return tuple(table for _record, table in table_records)

    @property
    def query(self) -> MeasurementFeatureQuery:
        return MeasurementFeatureQuery(
            self.feature_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )

    def resolve(self) -> tuple[np.ndarray, ...] | None:
        rows = self.matching_rows()
        if not rows:
            raise ValueError(
                f"Measurement feature {self.feature_name!r} has no declared rows."
            )
        row_axis = MeasurementImageOperandAxisResolution(
            self.measurement_tables
        ).axis_field()
        if row_axis is None:
            return None
        return self.slice_values(
            self.index_rows_by_slice(rows),
        )

    def matching_rows(self) -> tuple[MeasurementImageOperandRow, ...]:
        matched: list[MeasurementImageOperandRow] = []
        query = self.query
        for table in self.measurement_tables:
            declared_fields = {field.name for field in table.rows.fields}
            declares_slice_axis = (
                MeasurementRowAxisField.SLICE_INDEX.value in declared_fields
            )
            for row in measurement_rows((table,)):
                row_mapping = measurement_row_mapping(row)
                if measurement_row_has_object_identity(row_mapping):
                    continue
                value = query.row_value(row)
                if value is None:
                    continue
                row_has_slice_axis = (
                    MeasurementRowAxisField.SLICE_INDEX.value in row_mapping
                )
                if declared_fields and row_has_slice_axis is not declares_slice_axis:
                    raise ValueError(
                        f"Measurement table {table.name!r} rows disagree with its "
                        "declared slice_index schema."
                    )
                matched.append(
                    MeasurementImageOperandRow.from_mapping(
                        table_name=table.name,
                        row_mapping=row_mapping,
                        value=float(value),
                    )
                )
        return tuple(matched)

    def index_rows_by_slice(
        self,
        rows: tuple[MeasurementImageOperandRow, ...],
    ) -> MeasurementImageOperandRowsBySlice:
        row_axis = MeasurementRowAxisField.SLICE_INDEX
        rows_by_slice: dict[int, list[MeasurementImageOperandRow]] = {}
        for row in rows:
            slice_index = row.axis_value(row_axis)
            slice_rows = rows_by_slice.get(slice_index)
            if slice_rows is None:
                slice_rows = []
                rows_by_slice[slice_index] = slice_rows
            slice_rows.append(row)
        return MappingProxyType(
            {
                slice_index: tuple(slice_rows)
                for slice_index, slice_rows in rows_by_slice.items()
            }
        )

    def slice_values(
        self,
        rows_by_slice: MeasurementImageOperandRowsBySlice,
    ) -> tuple[np.ndarray, ...]:
        expected_indices = set(range(len(rows_by_slice)))
        if set(rows_by_slice) != expected_indices:
            raise ValueError(
                f"Measurement feature {self.feature_name!r} has non-contiguous "
                f"slice_index projection values {sorted(rows_by_slice)}; "
                f"expected {sorted(expected_indices)}."
            )

        slice_values: list[np.ndarray] = []
        for slice_index in range(len(expected_indices)):
            rows = rows_by_slice[slice_index]
            if len(rows) != 1:
                raise ValueError(
                    f"Measurement feature {self.feature_name!r} resolved to "
                    f"{len(rows)} values on slice {slice_index}; expected exactly "
                    f"one scalar value. Sources: {[row.source for row in rows]!r}."
                )
            slice_values.append(np.asarray(rows[0].value, dtype=float))
        return tuple(slice_values)


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerObjectMeasurementVectorBinding(ObjectLabelMeasurementSliceRequest):
    """Nominal binding from object-label runtime inputs to measurement vectors."""

    request: RuntimeInputBindingRequest

    @classmethod
    def for_object(
        cls,
        request: RuntimeInputBindingRequest,
        *,
        object_ref: ArtifactSpec | str,
        feature_name: str,
        labels: ObjectLabelValue | None = None,
        slice_index: int | None = None,
    ) -> "CellProfilerObjectMeasurementVectorBinding":
        object_ref_is_spec = isinstance(object_ref, ArtifactSpec)
        object_spec = (
            object_ref
            if object_ref_is_spec
            else ArtifactSpecCollection(request.object_inputs).by_name(object_ref)
        )
        if object_spec is None:
            raise ValueError(
                f"Object measurement vector input {object_ref!r} is not declared "
                f"for {request.adapter.request.require_callable_contract().module_name}; available object inputs are "
                f"{ArtifactSpecCollection(request.object_inputs).names()}."
            )
        resolved_slice_index = slice_index
        if resolved_slice_index is None:
            resolved_slice_index = request.adapter.runtime_slice_plane_index()
        label_payload = labels
        if label_payload is None:
            label_payload = request.label_payload_for(object_spec)
        return cls(
            object_name=object_spec.name,
            feature_name=feature_name,
            group_key=request.adapter.request.group_key,
            slice_index=resolved_slice_index,
            labels=label_payload,
            request=request,
        )

    def values(
        self,
        adapter: "CellProfilerRuntimeAdapter",
    ) -> tuple[np.ndarray, ...]:
        del adapter
        return MeasurementLabelSliceFeatureQuery(
            measurement_tables=self.measurement_tables(self.request.adapter),
            feature_name=self.feature_name,
            object_name=self.object_name,
            row_axis=MeasurementRowAxisField.SLICE_INDEX,
            plane_projector=self.request.adapter,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        ).values_for_labels(self.labels)

    def measurement_tables(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        *,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        """Return feature-bearing tables from exact declared measurement inputs."""

        del adapter, match_group
        declared = self.request.declared_measurement_tables()
        if not declared:
            raise ValueError(
                f"{self.request.adapter.request.require_callable_contract().module_name} feature {self.feature_name!r} "
                "requires a declared MeasurementsArtifactType runtime input."
            )
        matches = ObjectMeasurementTableIndex.from_tables(declared).for_object_feature(
            self.object_name,
            self.feature_name,
        )
        if not matches:
            raise ValueError(
                f"{self.request.adapter.request.require_callable_contract().module_name} declared measurement inputs do "
                f"not provide feature {self.feature_name!r} for object "
                f"{self.object_name!r}."
            )
        return matches

    def vector(self) -> CellProfilerMeasurementVector:
        return self.vector_from_slices(self.values(self.request.adapter))

    def vector_from_slices(
        self,
        slices: tuple[RuntimeCallableArgument, ...],
    ) -> CellProfilerMeasurementVector:
        """Attach this binding's exact runtime-plane selection to resolved values."""

        return CellProfilerMeasurementVector(
            slices,
            plane_projection=self.labels.declared_plane_projection(),
        )


@dataclass(frozen=True, slots=True)
class CellProfilerObjectMeasurementVectorBatchBinding:
    """Batch object-measurement vector bindings that share runtime semantics."""

    bindings: tuple[CellProfilerObjectMeasurementVectorBinding, ...]

    def vectors(self) -> tuple[CellProfilerMeasurementVector, ...]:
        if not self.bindings:
            return ()
        if len(self.bindings) == 1:
            return self.independent_vectors()
        shared_feature_name = self.shared_feature_name
        if shared_feature_name is None:
            return self.independent_vectors()
        return self.runtime_batch_vectors(shared_feature_name)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return unique feature names in binding order."""
        return tuple(dict.fromkeys(binding.feature_name for binding in self.bindings))

    @property
    def shared_feature_name(self) -> str | None:
        """Return the shared feature name required for runtime batch resolution."""
        feature_names = self.feature_names
        if len(feature_names) != 1:
            return None
        return feature_names[0]

    def independent_vectors(self) -> tuple[CellProfilerMeasurementVector, ...]:
        """Resolve bindings independently when no single feature owns the batch."""
        return tuple(binding.vector() for binding in self.bindings)

    def runtime_batch_vectors(
        self,
        feature_name: str,
    ) -> tuple[CellProfilerMeasurementVector, ...]:
        """Resolve same-feature object vectors through one runtime batch query."""
        started_at = time.perf_counter()
        adapter = self.bindings[0].request.adapter
        group_key = self.bindings[0].group_key
        mismatched = tuple(
            binding.object_name
            for binding in self.bindings
            if binding.feature_name != feature_name or binding.group_key != group_key
        )
        if mismatched:
            raise ValueError(
                "Object measurement vector batch bindings must share feature_name "
                f"and group_key; mismatched objects: {mismatched!r}."
            )
        cached_vectors = self.cached_runtime_batch_vectors(adapter)
        missing_bindings = tuple(
            binding
            for binding in self.bindings
            if binding.object_name not in cached_vectors
        )
        queried_vectors = {}
        if missing_bindings:
            queried_vectors = self.query_runtime_batch_vectors(
                adapter,
                feature_name,
                missing_bindings,
            )
        vectors = {**cached_vectors, **queried_vectors}
        resolved_vectors = tuple(
            binding.vector_from_slices(vectors[binding.object_name])
            for binding in self.bindings
        )
        CellProfilerRuntimeProfileLogger.label_batch(
            "adapter_object_label_batch_values",
            time.perf_counter() - started_at,
            feature_name=feature_name,
            count=len(vectors),
        )
        return resolved_vectors

    def query_runtime_batch_vectors(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        feature_name: str,
        bindings: tuple[CellProfilerObjectMeasurementVectorBinding, ...],
    ) -> Mapping[str, tuple[np.ndarray, ...]]:
        table_started_at = time.perf_counter()
        measurement_tables_by_object = MappingProxyType(
            {
                binding.object_name: binding.measurement_tables(adapter)
                for binding in bindings
            }
        )
        measurement_tables_by_identity: dict[int, MeasurementTable] = {}
        for tables in measurement_tables_by_object.values():
            for table in tables:
                table_identity = id(table)
                if table_identity not in measurement_tables_by_identity:
                    measurement_tables_by_identity[table_identity] = table
        measurement_tables = tuple(measurement_tables_by_identity.values())
        CellProfilerRuntimeProfileLogger.label_batch(
            "adapter_object_label_batch_tables",
            time.perf_counter() - table_started_at,
            feature_name=feature_name,
            count=len(measurement_tables),
        )
        if not measurement_tables:
            raise ValueError(
                f"Object feature {feature_name!r} requires declared measurement tables."
            )
        query_started_at = time.perf_counter()
        vectors = MeasurementLabelSliceFeatureBatchQuery(
            measurement_tables=measurement_tables,
            feature_name=feature_name,
            labels_by_object={
                binding.object_name: binding.labels for binding in bindings
            },
            row_axis=MeasurementRowAxisField.SLICE_INDEX,
            plane_projector=adapter,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        ).values_by_object()
        CellProfilerRuntimeProfileLogger.label_batch(
            "adapter_object_label_batch_query",
            time.perf_counter() - query_started_at,
            feature_name=feature_name,
            count=len(vectors),
        )
        cache_started_at = time.perf_counter()
        self.cache_runtime_batch_vectors(adapter, vectors, bindings)
        CellProfilerRuntimeProfileLogger.label_batch(
            "adapter_object_label_batch_cache",
            time.perf_counter() - cache_started_at,
            feature_name=feature_name,
            count=len(vectors),
        )
        return vectors

    def cache_runtime_batch_vectors(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        vectors: Mapping[str, tuple[np.ndarray, ...]],
        bindings: tuple[CellProfilerObjectMeasurementVectorBinding, ...],
    ) -> None:
        process_cache = object_label_measurement_values_cache(
            adapter.request.context.runtime_value_store
        )
        for binding in bindings:
            values = vectors[binding.object_name]
            query = binding.measurement_query(adapter)
            process_cache[query] = values

    def cached_runtime_batch_vectors(
        self,
        adapter: "CellProfilerRuntimeAdapter",
    ) -> dict[str, tuple[np.ndarray, ...]]:
        process_cache = object_label_measurement_values_cache(
            adapter.request.context.runtime_value_store
        )
        cached_vectors: dict[str, tuple[np.ndarray, ...]] = {}
        for binding in self.bindings:
            query = binding.measurement_query(adapter)
            cached = process_cache.get(query)
            if cached is not None:
                cached_vectors[binding.object_name] = cached
        return cached_vectors
