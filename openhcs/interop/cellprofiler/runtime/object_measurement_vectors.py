"""Object-measurement vector binding authorities for CellProfiler runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import time
from types import MappingProxyType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    MeasurementsArtifactType,
)
from openhcs.core.measurement_row_materialization import (
    measurement_row_has_object_identity,
)
from openhcs.core.runtime_artifact_queries import (
    MeasurementLabelSliceFeatureQuery,
    MeasurementLabelSliceFeatureBatchQuery,
    MeasurementTableAxisProjection,
)
from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureAxisScopeSelection,
    MeasurementFeatureQuery,
)
from openhcs.core.measurement_row_materialization import measurement_rows
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    ObjectLabelMeasurementValues,
    dense_object_label_id_domain,
    measurement_row_mapping,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    MeasurementTable,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeArtifactTypeStrategy,
    RuntimeInputBindingRequestBase,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerSliceAlignedValues,
)
from openhcs.interop.cellprofiler.runtime.object_label_measurements import (
    ObjectFeatureMeasurementContext,
    ObjectLabelMeasurementSliceRequest,
    object_label_measurement_values_cache,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import (
    object_measurement_tables_for_object,
)
from openhcs.interop.cellprofiler.runtime.adapter_scope import (
    RuntimeGroupMatchScope,
)
from openhcs.interop.cellprofiler.runtime.adapter_profile import AdapterProfileLog
from openhcs.interop.cellprofiler.runtime.payload_types import (
    DenseLabelPayload,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    MeasurementRowMapping,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    EnumStrategyLabelRegistryMixin,
)
from openhcs.interop.cellprofiler.runtime.runtime_artifact_records import (
    RuntimeArtifactRecordResolver,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerCurrentImage,
    RuntimeRecordSourceImageSetSelector,
)


class CellProfilerObjectInputCountAuthority:
    """Validate CellProfiler policies that require a fixed object-input arity."""

    @staticmethod
    def require_exact(
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        expected_count: int,
    ) -> None:
        if len(object_inputs) != expected_count:
            raise NotImplementedError(
                f"{module_name} requires {expected_count} object runtime input(s), "
                f"got {[spec.name for spec in object_inputs]}."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectInputBindingRequest(RuntimeInputBindingRequestBase):
    """Authoritative runtime context for binding object-label inputs."""

    registry_key = "object_input"

    object_inputs: tuple[ArtifactSpec, ...]
    runtime_inputs: tuple[ArtifactSpec, ...] = ()

    def with_object_inputs(
        self,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> "ObjectInputBindingRequest":
        return type(self)(
            module_name=self.module_name,
            func=self.func,
            object_inputs=object_inputs,
            adapter=self.adapter,
            kwargs=self.kwargs,
            current_image=self.current_image,
            binding_scope=self.binding_scope,
            runtime_inputs=self.runtime_inputs,
            project_object_labels_to_current_plane=(
                self.project_object_labels_to_current_plane
            ),
        )

    def require_exact_object_count(self, expected_count: int) -> None:
        CellProfilerObjectInputCountAuthority.require_exact(
            self.module_name,
            self.object_inputs,
            expected_count,
        )

    def labels_for_inputs(self) -> CellProfilerRuntimeValues:
        return tuple(self.labels_for(spec) for spec in self.object_inputs)

    def measurement_tables_for_primary_object(self) -> CellProfilerRuntimeValues:
        primary_object = None
        if self.object_inputs:
            primary_object = self.object_inputs[0]
        if primary_object is None:
            return ()
        return object_measurement_tables_for_object(self.adapter, primary_object.name)


@dataclass(slots=True)
class CellProfilerMeasurementVector:
    """CellProfiler-facing projection of one object/image measurement vector."""

    slices: CellProfilerRuntimeValues

    def __post_init__(self) -> None:
        self.slices = tuple(self.slices)

    @property
    def slice_aligned_value(self) -> np.ndarray | CellProfilerSliceAlignedValues:
        if len(self.slices) == 1:
            return np.asarray(self.slices[0])
        return CellProfilerSliceAlignedValues(
            tuple(np.asarray(value) for value in self.slices)
        )

    @property
    def calculate_math_operand_value(self) -> CellProfilerRuntimeValue:
        if len(self.slices) != 1:
            return self.slice_aligned_value
        values = np.asarray(self.slices[0])
        return float(values[0]) if values.size == 1 else values


@dataclass(frozen=True, slots=True)
class MeasurementImageOperandRow:
    """One image-level measurement row accepted by a CalculateMath operand."""

    row_mapping: MeasurementRowMapping
    value: float
    source: str

    @classmethod
    def from_mapping(
        cls,
        *,
        table_name: str,
        row_mapping: MeasurementRowMapping,
        value: float,
    ) -> "MeasurementImageOperandRow":
        return cls(
            row_mapping=row_mapping,
            value=value,
            source=cls.source_text(table_name, row_mapping),
        )

    @staticmethod
    def source_text(table_name: str, row_mapping: MeasurementRowMapping) -> str:
        return (
            f"{table_name}:"
            f"{MeasurementImageOperandRow.field_text(row_mapping, MeasurementRowAxisField.SOURCE_IMAGE_NAME.value, '<none>')}:"
            f"{MeasurementImageOperandRow.field_text(row_mapping, MeasurementRowAxisField.FEATURE_NAME.value, '<wide>')}:"
            "image_number="
            f"{MeasurementImageOperandRow.field_text(row_mapping, MeasurementRowAxisField.IMAGE_NUMBER.value, '<none>')}"
        )

    @staticmethod
    def field_text(
        row_mapping: MeasurementRowMapping,
        field_name: str,
        missing_label: str,
    ) -> str:
        value = row_mapping.get(field_name)
        if not value:
            return missing_label
        return str(value)

    def has_axis(self, axis: MeasurementRowAxisField) -> bool:
        return self.row_mapping.get(axis.value) not in (None, "")

    def axis_value(self, axis: MeasurementRowAxisField) -> int:
        if not self.has_axis(axis):
            raise ValueError(f"Measurement row does not declare axis {axis.value!r}.")
        return int(float(self.row_mapping[axis.value]))


@dataclass(frozen=True, slots=True)
class MeasurementImageOperandAxisResolution:
    """Explicit row-axis authority for image-level CalculateMath operands."""

    rows: tuple[MeasurementImageOperandRow, ...]

    def axis_field(self) -> MeasurementRowAxisField | None:
        if self.all_rows_declare(MeasurementRowAxisField.SLICE_INDEX):
            slice_values = self.axis_values(MeasurementRowAxisField.SLICE_INDEX)
            image_values = self.axis_values(MeasurementRowAxisField.IMAGE_NUMBER)
            if len(slice_values) > 1 and len(image_values) <= 1:
                return MeasurementRowAxisField.SLICE_INDEX
        if self.all_rows_declare(MeasurementRowAxisField.IMAGE_NUMBER):
            return MeasurementRowAxisField.IMAGE_NUMBER
        if self.no_rows_declare(
            MeasurementRowAxisField.IMAGE_NUMBER
        ) and self.all_rows_declare(MeasurementRowAxisField.SLICE_INDEX):
            return MeasurementRowAxisField.SLICE_INDEX
        if self.no_rows_declare(
            MeasurementRowAxisField.IMAGE_NUMBER
        ) and self.no_rows_declare(MeasurementRowAxisField.SLICE_INDEX):
            return None
        raise ValueError(
            "Image-level measurement rows must declare one consistent axis: "
            f"{MeasurementRowAxisField.IMAGE_NUMBER.value!r} or "
            f"{MeasurementRowAxisField.SLICE_INDEX.value!r}."
        )

    def all_rows_declare(self, axis: MeasurementRowAxisField) -> bool:
        return bool(self.rows) and all(row.has_axis(axis) for row in self.rows)

    def no_rows_declare(self, axis: MeasurementRowAxisField) -> bool:
        return all(not row.has_axis(axis) for row in self.rows)

    def axis_values(self, axis: MeasurementRowAxisField) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                row.axis_value(axis) for row in self.rows if row.has_axis(axis)
            )
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
        current_image: CellProfilerCurrentImage | None = None,
        group_key: str | None = None,
    ) -> "MeasurementImageOperandVectorResolution":
        """Build an image-operand resolver from runtime measurement tables."""
        return cls(
            measurement_tables=cls.runtime_axis_scope_tables(
                adapter,
                feature_name,
                current_image=current_image,
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
        current_image: CellProfilerCurrentImage | None = None,
        group_key: str | None = None,
    ) -> tuple[MeasurementTable, ...]:
        """Return image measurement tables in the narrowest scope carrying a feature."""
        query = MeasurementFeatureQuery(
            feature_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )
        candidates = (
            cls.runtime_feature_tables(
                adapter,
                query,
                group_key=group_key,
                match_group=True,
                current_image=current_image,
            ),
            RuntimeSliceProjection.measurement_tables_with_repeated_scalar_slice_offsets(
                cls.runtime_feature_tables(
                    adapter,
                    query,
                    group_key=group_key,
                    match_group=False,
                    current_image=current_image,
                )
            ),
            RuntimeSliceProjection.measurement_tables_with_repeated_scalar_slice_offsets(
                cls.runtime_feature_tables(
                    adapter,
                    query,
                    group_key=group_key,
                    match_group=False,
                    current_image=None,
                )
            ),
        )
        fallback = adapter.measurement_tables(
            group_key=group_key,
            current_image=current_image,
        )
        return MeasurementFeatureAxisScopeSelection(
            candidates=candidates,
            query=query,
            fallback=fallback,
        ).select()

    @staticmethod
    def runtime_feature_tables(
        adapter: "CellProfilerRuntimeAdapter",
        query: MeasurementFeatureQuery,
        *,
        group_key: str | None,
        match_group: bool,
        current_image: CellProfilerCurrentImage | None,
    ) -> tuple[MeasurementTable, ...]:
        """Return visible image-measurement tables that may carry one feature."""
        runtime_scope = RuntimeGroupMatchScope(
            group_key=group_key,
            match_group=match_group,
        ).runtime_scope(adapter, current_image=current_image)
        cache_key = (
            "image_feature",
            adapter.runtime_value_store.revision,
            runtime_scope.group_cache_component,
            match_group,
            runtime_scope.current_image_cache_component,
            query.feature_name,
        )
        cached = adapter._measurement_cache.get(cache_key)
        if cached is not None:
            return cached

        source_records = adapter.declared_measurement_input_records(
            group_key=group_key,
            match_group=match_group,
            current_image=current_image,
        )
        if not source_records:
            source_records = runtime_scope.artifact_query_context().find(
                artifact_type=MeasurementsArtifactType
            )
        table_records = []
        for record in source_records:
            table = MeasurementTable.from_runtime_value(record.value)
            if (
                query.table_may_carry_feature(table)
                or query.optional_value_index((table,)) is not None
            ):
                table_records.append((record, table))

        records = tuple(record for record, _table in table_records)
        tables_by_record_id = {id(record): table for record, table in table_records}
        records = RuntimeRecordSourceImageSetSelector(
            adapter,
            current_image,
        ).select_runtime_scope(records)
        tables = tuple(tables_by_record_id[id(record)] for record in records)
        adapter._measurement_cache[cache_key] = tables
        return tables

    @property
    def query(self) -> MeasurementFeatureQuery:
        return MeasurementFeatureQuery(
            self.feature_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )

    def resolve(self) -> tuple[np.ndarray, ...] | None:
        rows = self.matching_rows()
        row_axis = MeasurementImageOperandAxisResolution(rows).axis_field()
        if row_axis is None:
            return None
        return self.slice_values(
            self.index_rows_by_slice(rows, row_axis),
            row_axis=row_axis,
        )

    def matching_rows(self) -> tuple[MeasurementImageOperandRow, ...]:
        matched: list[MeasurementImageOperandRow] = []
        query = self.query
        for table in self.measurement_tables:
            for row in measurement_rows((table,)):
                row_mapping = measurement_row_mapping(row)
                if measurement_row_has_object_identity(row_mapping):
                    continue
                value = query.row_value(row)
                if value is None:
                    continue
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
        row_axis: MeasurementRowAxisField,
    ) -> MeasurementImageOperandRowsBySlice:
        axis_values = tuple(dict.fromkeys(row.axis_value(row_axis) for row in rows))
        axis_to_slice = {
            axis_value: index for index, axis_value in enumerate(sorted(axis_values))
        }
        rows_by_slice: dict[int, list[MeasurementImageOperandRow]] = {}
        for row in rows:
            slice_index = axis_to_slice[row.axis_value(row_axis)]
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
        *,
        row_axis: MeasurementRowAxisField,
    ) -> tuple[np.ndarray, ...]:
        expected_indices = set(range(len(rows_by_slice)))
        if set(rows_by_slice) != expected_indices:
            raise ValueError(
                f"Measurement feature {self.feature_name!r} has non-contiguous "
                f"{row_axis.value} projection values {sorted(rows_by_slice)}; "
                f"expected {sorted(expected_indices)}."
            )

        slice_values: list[np.ndarray] = []
        for slice_index in range(len(expected_indices)):
            rows = rows_by_slice[slice_index]
            unique_by_source: dict[str, MeasurementImageOperandRow] = {}
            conflicting_rows: list[MeasurementImageOperandRow] = []
            for row in rows:
                existing = unique_by_source.get(row.source)
                if existing is None:
                    unique_by_source[row.source] = row
                    continue
                if np.allclose(
                    np.asarray(existing.value, dtype=float),
                    np.asarray(row.value, dtype=float),
                    equal_nan=True,
                ):
                    continue
                conflicting_rows.extend((existing, row))
            if conflicting_rows:
                rows_by_identity: dict[int, MeasurementImageOperandRow] = {}
                for row in (*unique_by_source.values(), *conflicting_rows):
                    rows_by_identity[id(row)] = row
                unique_rows = tuple(rows_by_identity.values())
            else:
                unique_rows = tuple(unique_by_source.values())
            if unique_rows:
                rows = unique_rows
            if len(rows) != 1:
                raise ValueError(
                    f"Measurement feature {self.feature_name!r} resolved to "
                    f"{len(rows)} values on slice {slice_index}; expected exactly "
                    f"one scalar value. Sources: {[row.source for row in rows]!r}."
                )
            slice_values.append(np.asarray(rows[0].value, dtype=float))
        return tuple(slice_values)


class CellProfilerObjectMeasurementVectorSource(Enum):
    """Source authority for object-measurement vector binding."""

    RUNTIME_MEASUREMENTS = "runtime_measurements"
    CURRENT_OBJECT_SHAPE_FEATURE = "current_object_shape_feature"


class CellProfilerObjectMeasurementVectorSourceStrategy(
    EnumStrategyLabelRegistryMixin[CellProfilerObjectMeasurementVectorSource],
    metaclass=AutoRegisterMeta,
):
    """Nominal source strategy for object-measurement vector binding."""

    __enum_member_attr__ = "source"
    stable_key_axis: ClassVar[str] = "source"
    source: ClassVar[CellProfilerObjectMeasurementVectorSource]

    @abstractmethod
    def vector(
        self,
        binding: "CellProfilerObjectMeasurementVectorBinding",
    ) -> CellProfilerMeasurementVector | None:
        """Return a source-owned vector or None to use runtime measurement rows."""


class RuntimeMeasurementsVectorSourceStrategy(
    CellProfilerObjectMeasurementVectorSourceStrategy
):
    """Use persisted runtime measurement rows for object-vector binding."""

    source = CellProfilerObjectMeasurementVectorSource.RUNTIME_MEASUREMENTS

    def vector(
        self,
        binding: "CellProfilerObjectMeasurementVectorBinding",
    ) -> CellProfilerMeasurementVector | None:
        del binding
        return None


class CurrentObjectShapeFeatureVectorStatus(Enum):
    """Resolution status for current-label AreaShape vector derivation."""

    AVAILABLE = "available"
    UNSUPPORTED_LABEL_DIMENSION = "unsupported_label_dimension"
    UNKNOWN_SHAPE_FEATURE = "unknown_shape_feature"
    UNMEASURED_SHAPE_FEATURE = "unmeasured_shape_feature"


@dataclass(frozen=True, slots=True)
class CurrentObjectShapeFeatureVectorResult:
    """Typed result for deriving an object-measurement vector from live labels."""

    status: CurrentObjectShapeFeatureVectorStatus
    vector: CellProfilerMeasurementVector | None = None

    @classmethod
    def available(
        cls,
        vector: CellProfilerMeasurementVector,
    ) -> "CurrentObjectShapeFeatureVectorResult":
        return cls(CurrentObjectShapeFeatureVectorStatus.AVAILABLE, vector)

    @classmethod
    def unavailable(
        cls,
        status: CurrentObjectShapeFeatureVectorStatus,
    ) -> "CurrentObjectShapeFeatureVectorResult":
        if status is CurrentObjectShapeFeatureVectorStatus.AVAILABLE:
            raise ValueError("Available shape-vector results require a vector.")
        return cls(status)


class CurrentObjectShapeFeatureVectorSourceStrategy(
    CellProfilerObjectMeasurementVectorSourceStrategy
):
    """Derive AreaShape vectors from the current object-label image."""

    source = CellProfilerObjectMeasurementVectorSource.CURRENT_OBJECT_SHAPE_FEATURE

    def vector(
        self,
        binding: "CellProfilerObjectMeasurementVectorBinding",
    ) -> CellProfilerMeasurementVector | None:
        labels = binding.labels
        label_array = np.asarray(labels)
        runtime_vector = self.runtime_current_image_vector(
            binding,
            label_array,
        )
        if runtime_vector is not None:
            return runtime_vector
        if label_array.ndim > 2 and self.runtime_tables_declare_feature(binding):
            return None
        return self.current_label_shape_vector(binding.feature_name, label_array).vector

    def runtime_tables_declare_feature(
        self,
        binding: "CellProfilerObjectMeasurementVectorBinding",
    ) -> bool:
        """Return whether persisted runtime rows can own this shape feature."""
        tables = binding.measurement_tables(binding.request.adapter)
        return any(
            binding.feature_query.table_may_carry_feature(table) for table in tables
        )

    def runtime_current_image_vector(
        self,
        binding: "CellProfilerObjectMeasurementVectorBinding",
        label_array: np.ndarray,
    ) -> CellProfilerMeasurementVector | None:
        if binding.image_number is None:
            return None
        if not self.runtime_tables_declare_image_number(binding):
            return None
        value_slices = self.current_image_positional_value_slices(
            binding,
            label_array,
        )
        if value_slices is None:
            return None
        return CellProfilerMeasurementVector(value_slices)

    def current_image_positional_value_slices(
        self,
        binding: "CellProfilerObjectMeasurementVectorBinding",
        label_array: np.ndarray,
    ) -> tuple[np.ndarray, ...] | None:
        if binding.image_number is None:
            return None
        tables = binding.measurement_tables(binding.request.adapter)
        label_planes = (
            (label_array,)
            if label_array.ndim <= 2
            else tuple(label_array[index] for index in range(label_array.shape[0]))
        )
        values_by_slice = tuple(
            self.current_image_positional_values(
                MeasurementTableAxisProjection(
                    MeasurementRowAxisField.IMAGE_NUMBER,
                    binding.image_number + slice_index,
                ).tables(
                    tables,
                ),
                binding,
            )
            for slice_index, _label_plane in enumerate(label_planes)
        )
        if any(values is None for values in values_by_slice):
            return None
        return tuple(
            np.asarray(values, dtype=np.float64)
            for values in values_by_slice
            if values is not None
        )

    def current_image_positional_values(
        self,
        tables: tuple[MeasurementTable, ...],
        binding: "CellProfilerObjectMeasurementVectorBinding",
    ) -> tuple[float, ...] | None:
        value_index = binding.feature_query.optional_value_index(tables)
        if value_index is None:
            return None
        values_by_label, positional_values = value_index
        if values_by_label:
            values = tuple(values_by_label[label] for label in sorted(values_by_label))
        else:
            values = tuple(positional_values)
        if values:
            return values
        return None

    def runtime_tables_declare_image_number(
        self,
        binding: "CellProfilerObjectMeasurementVectorBinding",
    ) -> bool:
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        tables = binding.measurement_tables(binding.request.adapter)
        for table in tables:
            if isinstance(table.rows, ColumnarRows):
                if image_number_field in tuple(
                    str(column) for column in table.rows.columns
                ):
                    return True
                continue
            if any(
                image_number_field in measurement_row_mapping(row)
                for row in measurement_rows((table,))
            ):
                return True
        return False

    def current_label_shape_vector(
        self,
        feature_name: str,
        label_array: np.ndarray,
    ) -> CurrentObjectShapeFeatureVectorResult:
        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerModule,
            CurrentObjectFeatureVectorAuthority,
        )

        for module_type in CellProfilerModule.__registry__.values():
            if not issubclass(module_type, CurrentObjectFeatureVectorAuthority):
                continue
            result = module_type.current_object_feature_vector(
                feature_name,
                label_array,
            )
            if not isinstance(result, CurrentObjectShapeFeatureVectorResult):
                raise TypeError(
                    f"{module_type.__name__}.current_object_feature_vector must "
                    "return CurrentObjectShapeFeatureVectorResult."
                )
            if (
                result.status
                is not CurrentObjectShapeFeatureVectorStatus.UNKNOWN_SHAPE_FEATURE
            ):
                return result
        return CurrentObjectShapeFeatureVectorResult.unavailable(
            CurrentObjectShapeFeatureVectorStatus.UNKNOWN_SHAPE_FEATURE
        )


def current_object_label_measurement_vector(
    value_slices: tuple[np.ndarray, ...],
) -> CellProfilerMeasurementVector:
    """Return a measurement vector from current object-label value slices."""
    return CellProfilerMeasurementVector(value_slices)


def dense_object_label_values(
    label_plane: np.ndarray,
    *,
    measured_labels: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Align measured object values to the dense label id domain of one plane."""
    values_by_label = {
        int(label): float(value)
        for label, value in zip(measured_labels, values, strict=True)
    }
    object_ids = dense_object_label_id_domain(label_plane)
    return ObjectLabelMeasurementValues.from_value_mapping(
        object_ids,
        values_by_label,
    ).values


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerObjectMeasurementVectorBinding(ObjectLabelMeasurementSliceRequest):
    """Nominal binding from object-label runtime inputs to measurement vectors."""

    request: RuntimeInputBindingRequestBase
    source: CellProfilerObjectMeasurementVectorSource = (
        CellProfilerObjectMeasurementVectorSource.RUNTIME_MEASUREMENTS
    )

    @classmethod
    def for_object(
        cls,
        request: RuntimeInputBindingRequestBase,
        *,
        object_ref: ArtifactSpec | str,
        feature_name: str,
        labels: DenseLabelPayload | None = None,
        image_number: int | None = None,
        source: CellProfilerObjectMeasurementVectorSource = (
            CellProfilerObjectMeasurementVectorSource.RUNTIME_MEASUREMENTS
        ),
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
                f"for {request.module_name}; available object inputs are "
                f"{ArtifactSpecCollection(request.object_inputs).names()}."
            )
        resolved_image_number = image_number
        if resolved_image_number is None and object_ref_is_spec:
            resolved_image_number = RuntimeArtifactTypeStrategy.for_artifact_type(
                object_spec.artifact_type
            ).cellprofiler_image_number(request.artifact_input_request(object_spec))
        label_payload = labels
        if label_payload is None:
            label_payload = request.label_domain_payload_for(object_spec)
        return cls(
            object_name=object_spec.name,
            feature_name=feature_name,
            group_key=None,
            image_number=resolved_image_number,
            current_image=request.current_image,
            labels=label_payload,
            request=request,
            source=source,
        )

    @property
    def feature_query(self) -> MeasurementFeatureQuery:
        return MeasurementFeatureQuery(
            self.feature_name,
            object_name=self.object_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )

    def values(
        self,
        adapter: "CellProfilerRuntimeAdapter",
    ) -> CellProfilerMeasurementSliceValues:
        declared_measurement_tables = self.request.declared_measurement_tables()
        if declared_measurement_tables:
            return MeasurementLabelSliceFeatureQuery(
                measurement_tables=declared_measurement_tables,
                feature_name=self.feature_name,
                object_name=self.object_name,
                row_axis=MeasurementRowAxisField.SLICE_INDEX,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ).values_for_labels(self.labels)
        return ObjectLabelMeasurementSliceRequest.values(self, adapter)

    def vector(self) -> CellProfilerMeasurementVector:
        source_vector = (
            CellProfilerObjectMeasurementVectorSourceStrategy.for_enum_member(
                self.source
            ).vector(self)
        )
        if source_vector is not None:
            return source_vector
        return CellProfilerMeasurementVector(self.values(self.request.adapter))


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
        source_vectors = self.source_vectors()
        if source_vectors is not None:
            return source_vectors
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
            CellProfilerMeasurementVector(vectors[binding.object_name])
            for binding in self.bindings
        )
        AdapterProfileLog.label_batch(
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
    ) -> Mapping[str, CellProfilerMeasurementSliceValues]:
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
        AdapterProfileLog.label_batch(
            "adapter_object_label_batch_tables",
            time.perf_counter() - table_started_at,
            feature_name=feature_name,
            count=len(measurement_tables),
        )
        if not measurement_tables:
            return {
                binding.object_name: binding.values(adapter) for binding in bindings
            }
        query_started_at = time.perf_counter()
        vectors = MeasurementLabelSliceFeatureBatchQuery(
            measurement_tables=measurement_tables,
            feature_name=feature_name,
            labels_by_object={
                binding.object_name: binding.labels for binding in bindings
            },
            row_axis_starts_by_object={
                binding.object_name: binding.table_scope_image_number
                for binding in bindings
                if binding.table_scope_image_number is not None
            },
            row_axis=MeasurementRowAxisField.SLICE_INDEX,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        ).values_by_object()
        AdapterProfileLog.label_batch(
            "adapter_object_label_batch_query",
            time.perf_counter() - query_started_at,
            feature_name=feature_name,
            count=len(vectors),
        )
        cache_started_at = time.perf_counter()
        self.cache_runtime_batch_vectors(adapter, vectors, bindings)
        AdapterProfileLog.label_batch(
            "adapter_object_label_batch_cache",
            time.perf_counter() - cache_started_at,
            feature_name=feature_name,
            count=len(vectors),
        )
        return vectors

    def cache_runtime_batch_vectors(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        vectors: Mapping[str, CellProfilerMeasurementSliceValues],
        bindings: tuple[CellProfilerObjectMeasurementVectorBinding, ...],
    ) -> None:
        process_cache = object_label_measurement_values_cache(
            adapter.runtime_value_store
        )
        for binding in bindings:
            values = vectors[binding.object_name]
            query = binding.measurement_query(adapter)
            adapter._measurement_cache[query] = values
            process_cache[query] = values

    def cached_runtime_batch_vectors(
        self,
        adapter: "CellProfilerRuntimeAdapter",
    ) -> dict[str, CellProfilerMeasurementSliceValues]:
        process_cache = object_label_measurement_values_cache(
            adapter.runtime_value_store
        )
        cached_vectors: dict[str, CellProfilerMeasurementSliceValues] = {}
        for binding in self.bindings:
            query = binding.measurement_query(adapter)
            cached = adapter._measurement_cache.get(query)
            if cached is None:
                cached = process_cache.get(query)
                if cached is not None:
                    adapter._measurement_cache[query] = cached
            if cached is not None:
                cached_vectors[binding.object_name] = cached
        return cached_vectors

    def source_vectors(self) -> tuple[CellProfilerMeasurementVector, ...] | None:
        """Return vectors from the declared source strategy when it owns all bindings."""
        vectors: list[CellProfilerMeasurementVector | None] = []
        for binding in self.bindings:
            vector = CellProfilerObjectMeasurementVectorSourceStrategy.for_enum_member(
                binding.source
            ).vector(binding)
            vectors.append(vector)
        if not any(vector is not None for vector in vectors):
            return None
        if any(vector is None for vector in vectors):
            raise ValueError(
                "Object measurement vector batch cannot mix source-owned and "
                "runtime-table-owned bindings."
            )
        return tuple(vector for vector in vectors if vector is not None)
