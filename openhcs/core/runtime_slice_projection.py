"""Nominal runtime-slice projection contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, TypeAlias, cast

from metaclass_registry import AutoRegisterMeta
import numpy as np
import numpy.typing as npt

from openhcs.constants.constants import VariableComponents
from openhcs.core.image_shapes import is_color_image_slice, is_color_image_stack
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_NAME_FIELD,
    MeasurementRowsAxisProjection,
    measurement_table_object_name,
)
from openhcs.core.runtime_artifact_queries import (
    MeasurementTableAxisProjection,
    measurement_table_slice_indices as runtime_measurement_table_slice_indices,
)
from openhcs.core.measurement_row_materialization import measurement_rows
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementRowAxisField,
    MeasurementTableRowLayout,
    ParentChildRelationshipPayload,
    RuntimeSliceIdentityProjectableValue,
    RuntimeSliceProjectableValue,
    RuntimePlaneAxis,
    RuntimePlaneAxisSliceProjectionPolicy,
    carries_measurement_row_semantics,
    measurement_table_row_layout,
    measurement_table_row_layouts,
    measurement_table_row_layout_from_fields,
    measurement_row_mapping,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    MeasurementTable,
    ObjectLabelPayload,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelSet,
    ObjectLabelValue,
    ObjectRelationship,
    RuntimeArrayData,
    RuntimeImagePayloadContext,
    SingletonObjectLabelStackCollapseStrategy,
    SparseIJVLabelRows,
    dense_label_stack_reduce_planes,
    dense_label_stack_supports_plane_reduction,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    object_label_dense_array,
    project_image_mask_to_data_domain,
)
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
    VariableComponentAxisProjection,
)

SINGLE_OBSERVED_ROW_LAYOUT_COUNT = 1
RuntimeProjectionPrimitive: TypeAlias = str | bytes | int | float | bool | None
RuntimeProjectionMapping: TypeAlias = Mapping[str, "RuntimeProjectionData"]
RuntimeProjectionSequence: TypeAlias = Sequence["RuntimeProjectionData"]
RuntimeProjectionDType: TypeAlias = npt.DTypeLike
MeasurementTableSchemaRow: TypeAlias = Mapping[str, "RuntimeProjectionData"]
RuntimeProjectionData: TypeAlias = (
    RuntimeArrayData
    | MeasurementTable
    | ParentChildRelationshipPayload
    | ObjectRelationship
    | SparseIJVLabelRows
    | ObjectLabelSet
    | ObjectLabelPayload
    | RuntimeSliceAlignedValueSet
    | RuntimeProjectionPrimitive
    | RuntimeProjectionMapping
    | RuntimeProjectionSequence
)


@dataclass(frozen=True, slots=True)
class RuntimeProjectionSourceIdentityRequest:
    """Nominal request for runtime-slice projection with source identity rules."""

    value: RuntimeProjectionData
    source_description: str
    variable_components: Sequence[VariableComponents] = field(default_factory=tuple)

    def value_for_projection(self, slice_count: int) -> RuntimeProjectionData:
        """Return the value with declared source-identity stack axes projected."""
        variable_component_projection = VariableComponentAxisProjection.from_axes(
            component.value
            for component in self.variable_components
            if component.value is not None
        )
        if variable_component_projection.is_empty:
            return self.value
        metadata = image_payload_metadata(
            self.value
        ).with_indexed_source_plane_provenance(slice_count)
        if metadata.source_image_provenance_planes.has_values:
            return RuntimeImagePayloadContext(
                image_payload_data(self.value),
                image_payload_mask(self.value),
                metadata,
            ).payload()
        provenance_planes = variable_component_projection.provenance_planes(
            source_path=metadata.source_path,
            source_component_metadata=metadata.source_component_metadata,
            plane_count=slice_count,
        )
        if not provenance_planes.has_values:
            return self.value
        return RuntimeImagePayloadContext(
            image_payload_data(self.value),
            image_payload_mask(self.value),
            metadata.with_source_provenance(
                metadata.source_provenance.with_source_image_provenance_planes(
                    provenance_planes
                )
            ),
        ).payload()

    def slice_description(self, slice_index: int) -> str:
        """Return the diagnostic description for one projected runtime slice."""
        return f"{self.source_description} runtime slice {slice_index}"

@dataclass(frozen=True, slots=True)
class RuntimeProjectionPlaneMetadata:
    """Plane identity carried by one runtime-slice-projected payload item."""

    plane_indices: tuple[int, ...]
    plane_shape: tuple[int, ...]
    source_plane_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.plane_indices:
            raise ValueError(
                "RuntimeProjectionPlaneMetadata.plane_indices cannot be empty."
            )
        if len(self.plane_indices) != len(self.plane_shape):
            raise ValueError(
                "RuntimeProjectionPlaneMetadata plane index rank must match "
                "plane shape rank."
            )
        if any(index < 0 for index in self.plane_indices):
            raise ValueError(
                "RuntimeProjectionPlaneMetadata.plane_indices cannot be negative."
            )
        if any(size <= 0 for size in self.plane_shape):
            raise ValueError(
                "RuntimeProjectionPlaneMetadata.plane_shape values must be positive."
            )
        if any(index >= size for index, size in zip(self.plane_indices, self.plane_shape)):
            raise ValueError(
                "RuntimeProjectionPlaneMetadata.plane_indices must be within "
                f"plane_shape, got {self.plane_indices!r} for {self.plane_shape!r}."
            )
        if any(index < 0 for index in self.source_plane_indices):
            raise ValueError(
                "RuntimeProjectionPlaneMetadata.source_plane_indices cannot be negative."
            )

    @classmethod
    def from_runtime_axis(
        cls,
        value: "RuntimeProjectionData",
        context: "RuntimeProjectionAxis",
    ) -> "RuntimeProjectionPlaneMetadata | None":
        """Return display-plane metadata for a projected runtime axis."""
        if context.extent == 1:
            return None
        return cls(
            plane_indices=(context.slice_index,),
            plane_shape=(context.extent,),
            source_plane_indices=(
                RuntimeSliceProjection.source_plane_indices_for_slice(value, context)
                or ()
            ),
        )

    @property
    def roi_metadata(self) -> dict[str, tuple[int, ...]]:
        """Return the metadata payload understood by PolyStore ROI archives."""
        return {
            "plane_indices": self.plane_indices,
            "plane_shape": self.plane_shape,
        }

    @property
    def carries_source_plane_identity(self) -> bool:
        """Return whether the projected plane already selected source identity."""
        return bool(self.source_plane_indices)


class RuntimeProjectionSourceIdentityRequirement(str, Enum):
    """Source-identity requirement for runtime-slice projected payloads."""

    OPTIONAL = "optional"
    REQUIRED_COMPONENT_METADATA = "required_component_metadata"

    def project_payload_items(
        self,
        request: RuntimeProjectionSourceIdentityRequest,
    ) -> tuple["RuntimeProjectedPayloadItem", ...]:
        """Return runtime-slice-projected payload items under this identity policy."""
        value = request.value
        slice_count = RuntimeSliceProjection.slice_count_from_values((value,))
        if slice_count is None:
            return (
                RuntimeProjectedPayloadItem(
                    value=value,
                    source_description=request.source_description,
                ),
            )
        if slice_count == 1:
            context = RuntimeProjectionAxis(
                slice_index=0,
                extent=slice_count,
            )
            return (
                RuntimeProjectedPayloadItem(
                    value=RuntimeSliceProjection.value_for_slice(value, context),
                    source_description=request.slice_description(0),
                    runtime_plane_metadata=(
                        RuntimeProjectionPlaneMetadata.from_runtime_axis(
                            value,
                            context,
                        )
                    ),
                ),
            )
        projection_value = request.value_for_projection(slice_count)
        RuntimeProjectionSourceIdentityRequirementStrategy.for_requirement(
            self
        ).validate_stack_value(
            projection_value,
            slice_count,
            source_description=request.source_description,
        )
        items: list[RuntimeProjectedPayloadItem] = []
        for slice_index in range(slice_count):
            context = RuntimeProjectionAxis(
                slice_index=slice_index,
                extent=slice_count,
            )
            items.append(
                RuntimeProjectedPayloadItem(
                    value=RuntimeSliceProjection.value_for_slice(
                        projection_value,
                        context,
                    ),
                    source_description=request.slice_description(slice_index),
                    runtime_plane_metadata=(
                        RuntimeProjectionPlaneMetadata.from_runtime_axis(
                            projection_value,
                            context,
                        )
                    ),
                )
            )
        return tuple(items)

class RuntimeProjectionSourceIdentityError(ValueError):
    """Raised when runtime-slice projection cannot preserve source identity."""

class RuntimeProjectionSourceIdentityRequirementStrategy(
    EnumKeyedStrategyMixin[RuntimeProjectionSourceIdentityRequirement],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Validation policy for one source-identity requirement."""

    __registry_key__ = "requirement_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "requirement"
    __enum_label_attr__ = "requirement_label"

    requirement: ClassVar[RuntimeProjectionSourceIdentityRequirement | None] = None
    requirement_label: ClassVar[str | None] = None

    @classmethod
    def for_requirement(
        cls,
        requirement: RuntimeProjectionSourceIdentityRequirement,
    ) -> "RuntimeProjectionSourceIdentityRequirementStrategy":
        return cls.for_enum_member(requirement)

    @abstractmethod
    def validate_stack_value(
        self,
        value: "RuntimeProjectionData",
        slice_count: int,
        *,
        source_description: str,
    ) -> None:
        """Validate whether the requirement can address projected stack slices."""

class OptionalRuntimeProjectionSourceIdentityRequirementStrategy(
    RuntimeProjectionSourceIdentityRequirementStrategy
):
    """Validation policy for payloads that may omit per-slice source identity."""

    requirement = RuntimeProjectionSourceIdentityRequirement.OPTIONAL

    def validate_stack_value(
        self,
        value: "RuntimeProjectionData",
        slice_count: int,
        *,
        source_description: str,
    ) -> None:
        del value, slice_count, source_description

class ComponentMetadataRuntimeProjectionSourceIdentityRequirementStrategy(
    RuntimeProjectionSourceIdentityRequirementStrategy
):
    """Validation policy for payloads that must preserve source component metadata."""

    requirement = RuntimeProjectionSourceIdentityRequirement.REQUIRED_COMPONENT_METADATA

    def validate_stack_value(
        self,
        value: "RuntimeProjectionData",
        slice_count: int,
        *,
        source_description: str,
    ) -> None:
        validate_source_plane_component_metadata(
            value,
            slice_count,
            source_description=source_description,
        )

@dataclass(frozen=True, slots=True)
class RuntimeProjectionAxis:
    """Execution context for projecting one runtime slice from a value."""

    slice_index: int
    extent: int

    def __post_init__(self) -> None:
        if self.extent <= 0:
            raise ValueError(
                "RuntimeProjectionAxis.extent must be positive."
            )
        if self.slice_index < 0 or self.slice_index >= self.extent:
            raise ValueError(
                "RuntimeProjectionAxis.slice_index must be within "
                f"[0, {self.extent}), got {self.slice_index}."
            )

    def identity_projected(
        self,
        value: RuntimeSliceIdentityProjectableValue,
    ) -> RuntimeProjectionData:
        """Stamp a value with this context's runtime-axis identity."""
        return cast(
            RuntimeProjectionData,
            value.with_runtime_slice_identity(
                slice_index=self.slice_index,
                slice_count=self.extent,
            ),
        )

    def aligned_value(
        self,
        value: RuntimeSliceAlignedValueSet,
    ) -> RuntimeProjectionData:
        """Project a value already aligned to the context's runtime axis."""
        return value.value_for_aligned_slice(
            self.slice_index,
            self.extent,
        )

    def object_label_projection(
        self,
        value: ObjectLabelValue,
        *,
        plane_indices: tuple[int, ...] | None,
    ) -> RuntimeProjectionData:
        """Project object labels through this context's runtime axis."""
        return value.with_runtime_slice_projection(
            slice_index=self.slice_index,
            slice_count=self.extent,
            plane_indices=plane_indices,
        )

    def stack_view(
        self,
        value: RuntimeProjectionData,
    ) -> np.ndarray | None:
        """Return a stack view whose first axis is this context's runtime axis."""
        return RuntimeSliceProjection.stack_view(
            value,
            slice_count=self.extent,
        )

    def grayscale_stack_view(
        self,
        value: RuntimeProjectionData,
        *,
        flatten_high_rank: bool = False,
    ) -> np.ndarray | None:
        """Return a non-color stack view for object-label plane semantics."""
        return RuntimeSliceProjection.grayscale_plane_stack_view(
            value,
            slice_count=self.extent,
            flatten_high_rank=flatten_high_rank,
        )

    def axis_matches(self, axis_size: int) -> bool:
        """Return whether a first-axis length directly matches this context."""
        return axis_size == self.extent

    def axis_groups_runtime_positions(self, axis_size: int) -> bool:
        """Return whether a first axis contains grouped planes per runtime position."""
        return axis_size > self.extent and axis_size % self.extent == 0

    def axis_repeats_over_runtime_positions(self, axis_size: int) -> bool:
        """Return whether a shorter first axis repeats across runtime positions."""
        return axis_size < self.extent and self.extent % axis_size == 0

    def grouped_axis_indices(self, axis_size: int) -> tuple[int, ...]:
        """Return the grouped first-axis indexes represented by this context."""
        return tuple(range(self.slice_index, axis_size, self.extent))

    def contiguous_plane_indices(self, planes_per_position: int) -> tuple[int, ...]:
        """Return contiguous plane indexes represented by this context."""
        start = self.slice_index * planes_per_position
        return tuple(range(start, start + planes_per_position))

    def grouped_stack_slice(self, stack: np.ndarray) -> np.ndarray:
        """Return grouped first-axis planes represented by this context."""
        return stack[self.slice_index :: self.extent]

    def repeated_axis_index(self, axis_size: int) -> int:
        """Return the repeated first-axis index represented by this context."""
        return self.slice_index % axis_size

@dataclass(frozen=True, slots=True)
class RuntimeProjectedPayloadItem:
    """One runtime payload projected into an execution-addressable value."""

    value: RuntimeProjectionData
    source_description: str
    runtime_plane_metadata: RuntimeProjectionPlaneMetadata | None = None

    @property
    def data(self) -> RuntimeProjectionData:
        return image_payload_data(self.value)

    @property
    def metadata(self) -> ImagePayloadMetadata:
        return image_payload_metadata(self.value)

    @property
    def source_component_metadata(self) -> SourceComponentMetadata | None:
        return self.metadata.source_component_metadata

    def require_source_component_metadata(self) -> SourceComponentMetadata:
        metadata = self.source_component_metadata
        if metadata is None:
            raise ValueError(
                "Runtime payload projection requires source component metadata "
                f"for {self.source_description}."
            )
        return metadata

def validate_source_plane_component_metadata(
    value: RuntimeProjectionData,
    slice_count: int,
    *,
    source_description: str,
) -> None:
    """Require one source component-metadata record per projected slice."""
    plane_metadata = (
        image_payload_metadata(value)
        .with_indexed_source_plane_provenance(slice_count)
        .source_image_provenance_planes
        .component_metadata
    )
    if not plane_metadata or any(item is None for item in plane_metadata):
        raise RuntimeProjectionSourceIdentityError(
            "Runtime payload stack projection requires complete per-slice "
            f"component metadata for {source_description}; refusing to drop "
            "unaddressed slices."
        )
    if len(plane_metadata) == slice_count:
        return
    raise RuntimeProjectionSourceIdentityError(
        "Runtime payload stack metadata cardinality mismatch: "
        f"{len(plane_metadata)} metadata entries for {slice_count} runtime "
        f"slices in {source_description}."
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
    """Nominal strategy for projecting one runtime value family."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True
    value_type: ClassVar[type | tuple[type, ...] | None] = None
    value_type_label: ClassVar[str | None] = None

    @classmethod
    def strategy_for_value(
        cls,
        value: RuntimeProjectionData,
    ) -> "RuntimeSliceProjectionStrategy":
        strategy = cls.for_nominal_value(value)
        if strategy is None:
            return DefaultRuntimeSliceProjectionStrategy()
        return strategy

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        return RuntimeSliceProjection.default_value_for_slice(value, context)

    def identity_projected_value(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        del context
        return value

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        del value
        return None

    def stack_views(
        self,
        value: RuntimeProjectionData,
    ) -> tuple[np.ndarray, ...]:
        return RuntimeSliceProjection.default_stack_views(value)

class DefaultRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Projection strategy for image-like values without a narrower owner."""

    def identity_projected_value(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        if not carries_measurement_row_semantics(value):
            return value
        projected_rows = MeasurementRowsAxisProjection.from_rows(
            (value,),
        ).project_runtime_slice_index(context.slice_index)
        if len(projected_rows) != 1:
            raise ValueError(
                "Single measurement-row identity projection returned "
                f"{len(projected_rows)} rows."
            )
        return cast(RuntimeProjectionData, projected_rows[0])

class RuntimeSliceIdentityProjectionMixin:
    """Mixin for value families that own execution-slice identity stamping."""

    def identity_projected_value(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        identity_projectable = cast(RuntimeSliceIdentityProjectableValue, value)
        return context.identity_projected(identity_projectable)

class RuntimeSliceNoStackViewsMixin:
    """Mixin for value families whose slice contract is not stack-view based."""

    def stack_views(
        self,
        value: RuntimeProjectionData,
    ) -> tuple[np.ndarray, ...]:
        del value
        return ()

class RuntimeSliceRelationshipProjectionMixin:
    """Mixin for relationship payloads that carry the same slice-count contract."""

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        return RuntimeSliceProjection.relationship_slice_count(
            cast(ParentChildRelationshipPayload | ObjectRelationship, value)
        )

class RuntimeSliceIdentityProjectableValueProjectionStrategy(
    RuntimeSliceIdentityProjectionMixin,
    RuntimeSliceProjectionStrategy,
):
    """Projection strategy for identity-stamping values without slice projection."""

    value_type = RuntimeSliceIdentityProjectableValue

class RuntimeSliceAlignedValueProjectionStrategy(
    RuntimeSliceNoStackViewsMixin,
    RuntimeSliceProjectionStrategy,
):
    """Projection strategy for values already aligned to runtime slices."""

    value_type = RuntimeSliceAlignedValueSet

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        aligned = cast(RuntimeSliceAlignedValueSet, value)
        return context.aligned_value(aligned)

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        return cast(RuntimeSliceAlignedValueSet, value).slice_count

class MeasurementTableRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Projection strategy for measurement rows keyed by runtime-slice index."""

    value_type = MeasurementTable

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        return MeasurementTableAxisProjection(
            axis=MeasurementRowAxisField.SLICE_INDEX,
            value=context.slice_index,
            table=cast(MeasurementTable, value),
        ).apply()

class RuntimeSliceProjectableValueProjectionStrategy(
    RuntimeSliceNoStackViewsMixin,
    RuntimeSliceProjectionStrategy,
):
    """Projection strategy for values that implement the runtime-slice hook."""

    value_type = RuntimeSliceProjectableValue

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        projectable = cast(RuntimeSliceProjectableValue, value)
        return cast(
            RuntimeProjectionData,
            projectable.project_runtime_slice(context.slice_index),
        )

class ParentChildRelationshipRuntimeSliceProjectionStrategy(
    RuntimeSliceIdentityProjectionMixin,
    RuntimeSliceRelationshipProjectionMixin,
    RuntimeSliceProjectableValueProjectionStrategy
):
    """Projection strategy for generic parent-child relationship payloads."""

    value_type = ParentChildRelationshipPayload

class ObjectRelationshipRuntimeSliceProjectionStrategy(
    RuntimeSliceRelationshipProjectionMixin,
    RuntimeSliceProjectableValueProjectionStrategy
):
    """Projection strategy for named object relationship payloads."""

    value_type = ObjectRelationship

class SparseIJVLabelRowsRuntimeSliceProjectionStrategy(
    RuntimeSliceNoStackViewsMixin,
    RuntimeSliceProjectionStrategy,
):
    """Projection strategy for sparse IJV label rows."""

    value_type = SparseIJVLabelRows

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        return cast(SparseIJVLabelRows, value).slice(context.slice_index)

class ObjectLabelValueRuntimeSliceProjectionStrategy(
    RuntimeSliceIdentityProjectionMixin,
    RuntimeSliceProjectionStrategy,
):
    """Projection strategy for native object-label values."""

    value_type = ObjectLabelValue

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        return ObjectLabelRuntimeSliceStackContract.runtime_slice_count(
            cast(ObjectLabelValue, value)
        )

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        labels = cast(ObjectLabelValue, value)
        return context.object_label_projection(
            labels,
            plane_indices=RuntimeSliceProjection.object_label_plane_indices_for_slice(
                labels,
                context,
            ),
        )

    def stack_views(
        self,
        value: RuntimeProjectionData,
    ) -> tuple[np.ndarray, ...]:
        return RuntimeSliceProjection.object_label_stack_views(
            cast(ObjectLabelValue, value)
        )

class SequenceRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Projection strategy for tuple/list containers."""

    value_type = (tuple, list)

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        return RuntimeSliceProjection.sequence_value_for_slice(
            cast(RuntimeProjectionSequence, value),
            context,
        )

    def identity_projected_value(
        self,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        return cast(
            RuntimeProjectionData,
            RuntimeSliceProjection.sequence_identity_projected_value(
                cast(RuntimeProjectionSequence, value),
                context,
            ),
        )

    def stack_views(
        self,
        value: RuntimeProjectionData,
    ) -> tuple[np.ndarray, ...]:
        return RuntimeSliceProjection.sequence_stack_views(
            cast(RuntimeProjectionSequence, value)
        )

class RuntimeSliceProjection:
    """SSOT for runtime-slice count and value projection."""

    @classmethod
    def context_for_value(
        cls,
        value: RuntimeProjectionData,
        *,
        slice_index: int,
        slice_count: int | None = None,
        source_description: str,
    ) -> RuntimeProjectionAxis | None:
        effective_slice_count = (
            slice_count
            if slice_count is not None
            else cls.slice_count_from_values((value,))
        )
        if effective_slice_count is not None:
            return RuntimeProjectionAxis(
                slice_index=slice_index,
                extent=effective_slice_count,
            )
        stack_views = RuntimeSliceProjectionStrategy.strategy_for_value(
            value
        ).stack_views(value)
        if not stack_views:
            return None
        stack_counts = {
            int(stack.shape[0])
            for stack in stack_views
            if stack.shape[0] > slice_index
        }
        effective_slice_count = cls.single_slice_count(
            stack_counts,
            source_description=source_description,
        )
        if effective_slice_count is None:
            raise ValueError(
                "Cannot project runtime slice without a declared runtime slice "
                f"count for {source_description}."
            )
        return RuntimeProjectionAxis(
            slice_index=slice_index,
            extent=effective_slice_count,
        )

    @classmethod
    def value_for_slice(
        cls,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        normalized_value = cls.runtime_slice_normalized_value(value)
        return RuntimeSliceProjectionStrategy.strategy_for_value(
            normalized_value
        ).value_for_slice(normalized_value, context)

    @classmethod
    def sequence_identity_projected_value(
        cls,
        value: RuntimeProjectionSequence,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        """Project execution-slice identity recursively through nested outputs."""
        if cls.sequence_is_measurement_row_sequence(value):
            return MeasurementRowsAxisProjection.from_rows(
                value,
            ).project_runtime_slice_index(context.slice_index)
        projected_items = tuple(
            RuntimeSliceProjectionStrategy.strategy_for_value(
                item
            ).identity_projected_value(item, context)
            for item in value
        )
        if all(
            projected_item is item
            for projected_item, item in zip(projected_items, value, strict=True)
        ):
            return value
        if isinstance(value, tuple):
            return projected_items
        return list(projected_items)

    @staticmethod
    def sequence_is_measurement_row_sequence(
        value: RuntimeProjectionSequence,
    ) -> bool:
        return bool(value) and all(
            carries_measurement_row_semantics(item)
            for item in value
        )

    @classmethod
    def object_label_endpoint(
        cls,
        value: RuntimeProjectionData,
        *,
        context: RuntimeProjectionAxis | None = None,
    ) -> RuntimeProjectionData:
        """Resolve one object-label endpoint through runtime-slice semantics."""
        if context is None:
            return value
        return cls.value_for_slice(value, context)

    @classmethod
    def object_label_endpoint_dense_array(
        cls,
        value: RuntimeProjectionData,
        *,
        context: RuntimeProjectionAxis | None = None,
        dtype: RuntimeProjectionDType | None = None,
    ) -> np.ndarray:
        """Resolve and materialize one object-label endpoint as dense labels."""
        return object_label_dense_array(
            cls.object_label_endpoint(
                value,
                context=context,
            ),
            dtype=dtype,
        )

    @classmethod
    def kwargs_for_slice(
        cls,
        kwargs: Mapping[str, RuntimeProjectionData],
        context: RuntimeProjectionAxis,
        *,
        sequence_kwargs: frozenset[str] = frozenset(),
    ) -> dict[str, RuntimeProjectionData]:
        return {
            name: (
                tuple(
                    cls.value_for_slice(item, context)
                    for item in cls.runtime_slice_normalized_value(value)
                )
                if name in sequence_kwargs and isinstance(value, tuple)
                else cls.value_for_slice(
                    value,
                    context,
                )
            )
            for name, value in kwargs.items()
        }

    @classmethod
    def slice_count_from_kwargs(
        cls,
        kwargs: Mapping[str, RuntimeProjectionData],
        *,
        sequence_kwargs: frozenset[str] = frozenset(),
    ) -> int | None:
        return cls.slice_count_from_values(
            item
            for name, value in kwargs.items()
            for item in (
                cls.runtime_slice_normalized_value(value)
                if name in sequence_kwargs and isinstance(value, tuple)
                else (cls.runtime_slice_normalized_value(value),)
            )
        )

    @classmethod
    def runtime_slice_normalized_value(
        cls,
        value: RuntimeProjectionData,
    ) -> RuntimeProjectionData:
        if (
            isinstance(value, tuple)
            and value
            and all(isinstance(item, MeasurementTable) for item in value)
        ):
            return cls.measurement_tables_with_repeated_scalar_slice_offsets(value)
        return value

    @classmethod
    def slice_count_from_values(
        cls,
        values: Iterable[RuntimeProjectionData],
    ) -> int | None:
        values = tuple(values)
        declared_slice_counts = tuple(
            count
            for value in values
            for count in (
                RuntimeSliceProjectionStrategy.strategy_for_value(
                    value
                ).slice_count_for_value(value),
            )
            if count is not None
        )
        tensor_slice_counts = {
            count for count in declared_slice_counts if count > 1
        }
        tensor_slice_count = cls.compatible_runtime_slice_count(
            tensor_slice_counts,
            source_description="declared runtime-slice values",
        )
        if tensor_slice_count is not None:
            return tensor_slice_count

        tensor_slice_counts = {
            stack.shape[0]
            for value in values
            for stack in RuntimeSliceProjectionStrategy.strategy_for_value(
                value
            ).stack_views(value)
            if stack.shape[0] > 1
        }
        tensor_slice_counts.update(
            count
            for count in declared_slice_counts
            if count > 1
        )
        tensor_slice_count = cls.compatible_runtime_slice_count(
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
            for stack in RuntimeSliceProjectionStrategy.strategy_for_value(
                value
            ).stack_views(value)
        ) or any(count == 1 for count in declared_slice_counts):
            return 1
        return None

    @classmethod
    def default_stack_views(
        cls,
        value: RuntimeProjectionData,
    ) -> tuple[np.ndarray, ...]:
        """Return stack views for image-like array payloads."""
        metadata = image_payload_metadata(value)
        source_plane_count = metadata.source_provenance.source_plane_count
        if source_plane_count > 1:
            stack = cls.stack_view(
                value,
                slice_count=source_plane_count,
            )
            if stack is not None:
                return (stack,)
        stack = cls.stack_view(value)
        if stack is None:
            return ()
        return (stack,)

    @classmethod
    def sequence_stack_views(
        cls,
        value: RuntimeProjectionSequence,
    ) -> tuple[np.ndarray, ...]:
        stack = None
        if len(value) > 1:
            stack = cls.stack_view(value)
        if stack is not None:
            return (stack,)
        return tuple(
            stack
            for item in value
            for stack in RuntimeSliceProjectionStrategy.strategy_for_value(
                item
            ).stack_views(item)
        )

    @classmethod
    def object_label_stack_views(
        cls,
        value: ObjectLabelValue,
    ) -> tuple[np.ndarray, ...]:
        if not RuntimePlaneAxisSliceProjectionPolicy.for_enum_member(
            value.plane_axis
        ).supports_slice_projection():
            return ()

        stacks: list[np.ndarray] = []
        for item in value.runtime_slice_stack_view_sources():
            array = image_payload_data(item)
            if (
                isinstance(array, np.ndarray)
                and array.ndim > 3
                and not is_color_image_slice(array)
                and not is_color_image_stack(array)
            ):
                stack = cls.grayscale_plane_stack_view(
                    item,
                    slice_count=int(array.shape[0]),
                )
                if stack is not None:
                    stacks.append(stack)
                continue
            stack = cls.stack_view(item)
            if stack is not None:
                stacks.append(stack)
        return tuple(stacks)

    @classmethod
    def first_axis_slice_count_from_values(
        cls,
        values: Iterable[RuntimeProjectionData],
    ) -> int | None:
        """Return runtime count for high-rank values beside one 2D image plane."""
        slice_counts = {
            int(array.shape[0])
            for value in values
            for array in (image_payload_data(value),)
            if isinstance(array, np.ndarray)
            and array.ndim > 3
            and array.shape[0] > 1
            and not is_color_image_slice(array)
            and not is_color_image_stack(array)
        }
        return cls.single_slice_count(
            slice_counts,
            source_description="high-rank first-axis values",
        )

    @classmethod
    def stack_view(
        cls,
        value: RuntimeProjectionData,
        *,
        slice_count: int | None = None,
    ) -> np.ndarray | None:
        if isinstance(value, (str, bytes, bytearray, Mapping)):
            return None
        if isinstance(value, (tuple, list)):
            if cls._sequence_is_ragged(value):
                return None
            value = np.asarray(value)
        elif not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if (
            slice_count is not None
            and isinstance(value, np.ndarray)
            and is_color_image_stack(value)
            and value.shape[0] == slice_count
        ):
            return value
        return cls.grayscale_plane_stack_view(value, slice_count=slice_count)

    @staticmethod
    def _sequence_is_ragged(value: Sequence[RuntimeProjectionData]) -> bool:
        """Return whether a Python sequence lacks a rectangular array shape."""
        if not value:
            return False
        shapes = {tuple(np.shape(item)) for item in value}
        return len(shapes) > 1

    @classmethod
    def grayscale_plane_stack_view(
        cls,
        value: RuntimeProjectionData,
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
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData | None:
        array = image_payload_data(value)
        if is_color_image_slice(array) or is_color_image_stack(array):
            return None
        if not isinstance(array, np.ndarray):
            return None
        if array.ndim < 3 or not context.axis_matches(int(array.shape[0])):
            return None
        return array[context.slice_index]

    @classmethod
    def sequence_value_for_slice(
        cls,
        value: RuntimeProjectionSequence,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        """Project tuple/list containers through their item semantics."""
        stack = None
        if len(value) > 1:
            stack = cls.stack_view(value)
        if stack is not None:
            return cls.stack_slice_value(stack, context, value)
        projected = [
            cls.value_for_slice(
                item,
                context,
            )
            for item in value
        ]
        return tuple(projected) if isinstance(value, tuple) else projected

    @classmethod
    def object_label_plane_indices_for_slice(
        cls,
        value: ObjectLabelValue,
        context: RuntimeProjectionAxis,
    ) -> tuple[int, ...] | None:
        """Return source plane indexes represented by one object-label projection."""
        array = object_label_dense_array(value)
        if is_color_image_slice(array) or is_color_image_stack(array):
            return None
        if not isinstance(array, np.ndarray) or array.ndim < 3:
            return None
        if array.ndim > 3 and context.axis_matches(int(array.shape[0])):
            planes_per_slice = int(np.prod(array.shape[1:-2]))
            return context.contiguous_plane_indices(planes_per_slice)
        stack = cls.grayscale_plane_stack_view(
            array,
            slice_count=context.extent,
            flatten_high_rank=True,
        )
        if stack is None:
            return None
        if context.axis_matches(int(stack.shape[0])):
            return (context.slice_index,)
        if context.axis_groups_runtime_positions(int(stack.shape[0])):
            return context.grouped_axis_indices(int(stack.shape[0]))
        if context.axis_repeats_over_runtime_positions(int(stack.shape[0])):
            return (context.repeated_axis_index(int(stack.shape[0])),)
        return None

    @classmethod
    def source_plane_indices_for_slice(
        cls,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> tuple[int, ...] | None:
        """Return source-plane indexes represented by one runtime slice."""
        if isinstance(value, ObjectLabelValue):
            return cls.object_label_plane_indices_for_slice(value, context)
        metadata = image_payload_metadata(value).with_indexed_source_plane_provenance(
            context.extent
        )
        if not metadata.has_plane_specific_values:
            return None
        stack = context.grayscale_stack_view(value, flatten_high_rank=True)
        if stack is None:
            return None
        if context.axis_matches(int(stack.shape[0])):
            return (context.slice_index,)
        if context.axis_groups_runtime_positions(int(stack.shape[0])):
            return context.grouped_axis_indices(int(stack.shape[0]))
        if context.axis_repeats_over_runtime_positions(int(stack.shape[0])):
            return (context.repeated_axis_index(int(stack.shape[0])),)
        return None

    @classmethod
    def default_value_for_slice(
        cls,
        value: RuntimeProjectionData,
        context: RuntimeProjectionAxis,
    ) -> RuntimeProjectionData:
        """Default projection for image-like array payloads."""
        metadata = image_payload_metadata(value).with_indexed_source_plane_provenance(
            context.extent
        )
        mask = image_payload_mask(value)
        if mask is not None or metadata.has_values:
            data = cls.value_for_slice(
                image_payload_data(value),
                context,
            )
            source_plane_indices = cls.source_plane_indices_for_slice(
                value,
                context,
            )
            return RuntimeImagePayloadContext(
                data,
                project_image_mask_to_data_domain(mask, data),
                metadata.for_runtime_plane_projection(
                    source_plane_indices=source_plane_indices,
                    runtime_plane_index=context.slice_index,
                    runtime_plane_count=context.extent,
                ),
            ).payload()
        axis_sliced = cls.first_axis_slice_if_aligned(
            value,
            context,
        )
        if axis_sliced is not None:
            return axis_sliced
        stack = context.stack_view(value)
        if stack is None:
            return value
        return cls.stack_slice_value(
            stack,
            context,
            value,
        )

    @classmethod
    def stack_slice_value(
        cls,
        stack: np.ndarray,
        context: RuntimeProjectionAxis,
        original_value: RuntimeProjectionData,
    ) -> RuntimeProjectionData:
        """Project one runtime slice from a stack-shaped value."""
        if context.axis_matches(int(stack.shape[0])):
            return stack[context.slice_index]
        if stack.shape[0] == 1:
            return stack[0]
        if cls.is_grouped_label_stack(stack, context):
            return dense_label_stack_reduce_planes(
                context.grouped_stack_slice(stack)
            )
        if cls.is_repeated_label_stack(stack, context):
            return stack[context.repeated_axis_index(int(stack.shape[0]))]
        return original_value

    @staticmethod
    def is_grouped_label_stack(
        stack: np.ndarray,
        context: RuntimeProjectionAxis,
    ) -> bool:
        """Return whether stack planes are interleaved groups per runtime slice."""
        return (
            context.axis_groups_runtime_positions(int(stack.shape[0]))
            and dense_label_stack_supports_plane_reduction(stack)
        )

    @staticmethod
    def is_repeated_label_stack(
        stack: np.ndarray,
        context: RuntimeProjectionAxis,
    ) -> bool:
        """Return whether fewer label planes repeat across runtime slices."""
        return (
            context.axis_repeats_over_runtime_positions(int(stack.shape[0]))
            and dense_label_stack_supports_plane_reduction(stack)
        )

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
    def measurement_table_slice_count(value: RuntimeProjectionData) -> int | None:
        if isinstance(value, (tuple, list)):
            return RuntimeSliceProjection.measurement_table_collection_slice_count(value)
        if not isinstance(value, MeasurementTable):
            return None
        if not value.subject.scope.projects_runtime_slices:
            return None
        slice_indices = RuntimeSliceProjection.measurement_table_slice_indices(value)
        if not slice_indices:
            return None
        return RuntimeSliceProjection.zero_based_contiguous_slice_count(slice_indices)

    @staticmethod
    def measurement_table_effective_slice_count(value: MeasurementTable) -> int:
        """Return declared measurement-table slice count, treating scalar tables as one slice."""
        if RuntimeSliceProjection.measurement_table_declares_no_slice_index(value):
            return 1
        slice_indices = RuntimeSliceProjection.measurement_table_slice_indices(value)
        if not slice_indices:
            return 1
        expected_indices = set(range(min(slice_indices), max(slice_indices) + 1))
        if slice_indices != expected_indices:
            raise ValueError(
                f"MeasurementTable '{value.name}' has non-contiguous slice_index "
                f"values {sorted(slice_indices)}; expected "
                f"{sorted(expected_indices)}."
            )
        return len(expected_indices)

    @staticmethod
    def measurement_table_declares_no_slice_index(value: MeasurementTable) -> bool:
        """Return whether declared fields make row-level slice scanning unnecessary."""
        if not value.fields:
            return False
        slice_field = MeasurementRowAxisField.SLICE_INDEX.value
        return all(field.name != slice_field for field in value.fields)

    @staticmethod
    def measurement_table_collection_slice_count(
        values: Iterable[RuntimeProjectionData],
    ) -> int | None:
        values = RuntimeSliceProjection.runtime_slice_normalized_value(tuple(values))
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
            count = RuntimeSliceProjection.zero_based_contiguous_slice_count(slice_indices)
            if count is not None:
                slice_counts.add(count)
        return RuntimeSliceProjection.single_slice_count(
            slice_counts,
            source_description="measurement table collection values",
        )

    @staticmethod
    def zero_based_contiguous_slice_count(slice_indices: set[int]) -> int | None:
        """Return a runtime slice count only for dense zero-based slice domains."""
        if not slice_indices:
            return None
        expected_indices = set(range(max(slice_indices) + 1))
        if slice_indices != expected_indices:
            return None
        return len(expected_indices)

    @staticmethod
    def measurement_table_slice_indices(value: MeasurementTable) -> set[int]:
        return runtime_measurement_table_slice_indices(value)

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
        fields = RuntimeSliceProjection.measurement_table_slice_index_fields(table)
        rows = []
        for row in measurement_rows((table,)):
            row_mapping = dict(measurement_row_mapping(row))
            if "slice_index" in row_mapping:
                row_slice_index = int(row_mapping["slice_index"])
            else:
                row_slice_index = 0
            rows.append(
                {
                    **row_mapping,
                    "slice_index": row_slice_index + slice_offset,
                }
            )
        schema_validated = RuntimeSliceProjection.measurement_table_schema_matches_rows(
            fields,
            rows,
        )
        schema_loss_reasons = frozenset()
        if not schema_validated:
            schema_loss_reasons = frozenset(("row_layout",))
        return MeasurementTable(
            name=table.name,
            rows=rows,
            object_name=table.object_name,
            fields=fields,
            object_id_field=table.object_id_field,
            source_image_name=table.source_image_name,
            subject=table.subject,
            validated_runtime_schema=schema_validated,
            schema_loss_reasons=schema_loss_reasons,
            source_provenance=table.source_provenance,
        )

    @staticmethod
    def measurement_table_broadcast_to_slice_count(
        table: MeasurementTable,
        slice_count: int,
    ) -> MeasurementTable:
        """Return a scalar table repeated across every runtime slice."""
        if slice_count <= 1:
            return table
        fields = RuntimeSliceProjection.measurement_table_slice_index_fields(table)
        rows = [
            {
                **dict(measurement_row_mapping(row)),
                "slice_index": slice_index,
            }
            for slice_index in range(slice_count)
            for row in measurement_rows((table,))
        ]
        schema_validated = RuntimeSliceProjection.measurement_table_schema_matches_rows(
            fields,
            rows,
        )
        schema_loss_reasons = frozenset()
        if not schema_validated:
            schema_loss_reasons = frozenset(("row_layout",))
        return MeasurementTable(
            name=table.name,
            rows=rows,
            object_name=table.object_name,
            fields=fields,
            object_id_field=table.object_id_field,
            source_image_name=table.source_image_name,
            subject=table.subject,
            validated_runtime_schema=schema_validated,
            schema_loss_reasons=schema_loss_reasons,
            source_provenance=table.source_provenance,
        )

    @staticmethod
    def measurement_table_slice_index_fields(
        table: MeasurementTable,
    ) -> tuple[FieldSpec, ...]:
        """Return the declared schema after adding the canonical slice axis."""
        existing_fields = tuple(table.fields)
        if not existing_fields:
            return ()
        if any(field.name == MeasurementRowAxisField.SLICE_INDEX.value for field in existing_fields):
            return existing_fields
        return (
            FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, dtype="int"),
            *existing_fields,
        )

    @staticmethod
    def measurement_table_schema_matches_rows(
        fields: tuple[FieldSpec, ...],
        rows: Sequence[MeasurementTableSchemaRow],
    ) -> bool:
        """Return whether declared fields still describe projected row layout."""
        declared_layout = measurement_table_row_layout_from_fields(fields)
        if declared_layout is None:
            return False
        observed_layouts = measurement_table_row_layouts(rows)
        if not observed_layouts:
            observed_layout = MeasurementTableRowLayout.EMPTY
        elif len(observed_layouts) == SINGLE_OBSERVED_ROW_LAYOUT_COUNT:
            observed_layout = next(iter(observed_layouts))
        else:
            return False
        return observed_layout in (declared_layout, MeasurementTableRowLayout.EMPTY)

    @staticmethod
    def measurement_tables_with_repeated_scalar_slice_offsets(
        tables: tuple[MeasurementTable, ...],
    ) -> tuple[MeasurementTable, ...]:
        """Offset repeated scalar measurement tables onto consecutive slice indexes."""
        grouped: dict[tuple[str, str | None, str | None], list[int]] = {}
        for index, table in enumerate(tables):
            key = (
                table.name,
                measurement_table_object_name(table),
                table.source_image_name,
            )
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(index)

        aligned = list(tables)
        for indexes in grouped.values():
            if len(indexes) <= 1:
                continue
            group_slice_indices = tuple(
                RuntimeSliceProjection.measurement_table_slice_indices(tables[index])
                for index in indexes
            )
            if (
                all(len(slice_indices) == 1 for slice_indices in group_slice_indices)
                and len({next(iter(slice_indices)) for slice_indices in group_slice_indices})
                == len(indexes)
            ):
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

    @staticmethod
    def compatible_runtime_slice_count(
        slice_counts: set[int],
        *,
        source_description: str,
    ) -> int | None:
        """Return a runtime count when larger counts are grouped planes per slice."""
        if not slice_counts:
            return None
        if len(slice_counts) == 1:
            return next(iter(slice_counts))
        smallest = min(slice_counts)
        if smallest > 0 and all(count % smallest == 0 for count in slice_counts):
            return smallest
        return RuntimeSliceProjection.single_slice_count(
            slice_counts,
            source_description=source_description,
        )
