"""Nominal runtime-slice projection contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, TypeAlias, cast, overload

from arraybridge.decorators import DtypeConversionConfig
from metaclass_registry import AutoRegisterMeta
import numpy as np
import numpy.typing as npt

from openhcs.constants.constants import VariableComponents
from openhcs.core.aligned_image_payload import AlignedImageStack, ImageOutputBundle
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.runtime_artifact_queries import (
    MeasurementTableAxisProjection,
    MeasurementTableUnion,
)
from openhcs.core.measurement_row_materialization import MeasurementRowsAxisProjection
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
    RuntimeSliceIdentityProjectableValue,
    RuntimeSliceProjectableValue,
)
from openhcs.core.runtime_relationships import DirectedObjectRelationshipPayload
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    MaskedImagePayload,
    image_payload_data,
    image_payload_slice_context,
    image_payload_metadata,
)
from openhcs.core.runtime_array_values import RuntimeArrayData, is_array_payload
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.core.runtime_spatial_graph import SpatialGraph
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
)

RuntimeProjectionPrimitive: TypeAlias = str | bytes | int | float | bool | None
RuntimeProjectionMapping: TypeAlias = Mapping[str, "RuntimeProjectionData"]
RuntimeProjectionSequence: TypeAlias = Sequence["RuntimeProjectionData"]
RuntimeProjectionDType: TypeAlias = npt.DTypeLike
RuntimeProjectionData: TypeAlias = (
    RuntimeArrayData
    | MeasurementTable
    | ColumnarRows
    | DirectedObjectRelationshipPayload
    | ObjectRelationship
    | SpatialGraph
    | SparseIJVLabelRows
    | ObjectLabelSet
    | ObjectLabelPayload
    | AlignedImageStack
    | RuntimeSliceAlignedValueSet
    | RuntimeSliceProjectableValue
    | RuntimePlaneAxisValueProjection
    | DtypeConversionConfig
    | RuntimeProjectionPrimitive
    | Enum
    | RuntimeProjectionMapping
    | RuntimeProjectionSequence
)


class RuntimeSliceProjectionDeclarationError(ValueError):
    """Raised when runtime-slice behavior has no complete nominal declaration."""


@dataclass(frozen=True, slots=True)
class RuntimeProjectionSourceIdentityRequest:
    """Nominal request for runtime-slice projection with source identity rules."""

    value: RuntimeProjectionData
    source_description: str
    variable_components: Sequence[VariableComponents] = field(default_factory=tuple)
    plane_projection: RuntimePlaneAxisValueProjection | None = None

    def runtime_slice_count(self) -> int | None:
        """Return a count only when a nominal declaration selects stack semantics."""
        if self.plane_projection is not None:
            return self.plane_projection.axis_size
        declared_count = RuntimeSliceProjection.slice_count_from_values((self.value,))
        if declared_count is not None:
            return declared_count
        if not self.variable_components:
            return None
        raise RuntimeSliceProjectionDeclarationError(
            "Variable-component runtime projection requires a nominal payload "
            "with RuntimePlaneAxis.RUNTIME_SLICE; source provenance and ndarray "
            "shape cannot declare the runtime axis."
        )

    def projected_value(
        self, context: RuntimePlaneAxisValueProjection
    ) -> RuntimeProjectionData:
        """Project only through the axis declared by this request."""
        return RuntimeSliceProjection.value_for_slice(self.value, context)

    def source_plane_indices(
        self,
        context: RuntimePlaneAxisValueProjection,
    ) -> tuple[int, ...] | None:
        """Return exact provenance projection for the request-declared stack axis."""
        source_plane_count = image_payload_metadata(
            self.value
        ).source_provenance.source_plane_count
        if source_plane_count == 0:
            return None
        if source_plane_count != context.axis_size:
            raise ValueError(
                "Runtime stack source provenance must exactly match the declared "
                f"variable-component axis: {source_plane_count} != {context.axis_size}."
            )
        return (context.require_plane_index(),)

    def plane_metadata(
        self,
        context: RuntimePlaneAxisValueProjection,
    ) -> "RuntimeProjectionPlaneMetadata | None":
        """Return plane identity derived from this request's declared axis."""
        return RuntimeProjectionPlaneMetadata(
            plane_indices=(context.require_plane_index(),),
            plane_shape=(context.axis_size,),
            source_plane_indices=self.source_plane_indices(context) or (),
        )

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
        if any(
            index >= size for index, size in zip(self.plane_indices, self.plane_shape)
        ):
            raise ValueError(
                "RuntimeProjectionPlaneMetadata.plane_indices must be within "
                f"plane_shape, got {self.plane_indices!r} for {self.plane_shape!r}."
            )
        if any(index < 0 for index in self.source_plane_indices):
            raise ValueError(
                "RuntimeProjectionPlaneMetadata.source_plane_indices cannot be negative."
            )

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
        slice_count = request.runtime_slice_count()
        if slice_count is None:
            return (
                RuntimeProjectedPayloadItem(
                    value=value,
                    source_description=request.source_description,
                ),
            )
        RuntimeProjectionSourceIdentityRequirementStrategy.for_requirement(
            self
        ).validate_stack_value(
            value,
            slice_count,
            source_description=request.source_description,
        )
        items: list[RuntimeProjectedPayloadItem] = []
        for slice_index in range(slice_count):
            projection = (
                request.plane_projection
                or RuntimePlaneAxisValueProjection.preserve(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE,
                    axis_size=slice_count,
                )
            )
            context = projection.selected_plane(slice_index)
            items.append(
                RuntimeProjectedPayloadItem(
                    value=request.projected_value(context),
                    source_description=request.slice_description(slice_index),
                    runtime_plane_metadata=request.plane_metadata(context),
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
        .source_image_provenance_planes.component_metadata
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
        if strategy is not None:
            return strategy
        if is_array_payload(value):
            return ArrayBridgeArrayRuntimeSliceProjectionStrategy()
        raise RuntimeSliceProjectionDeclarationError(
            "Runtime-slice projection has no nominal strategy for "
            f"{type(value).__name__}."
        )

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        del context
        return value

    def identity_projected_value(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        del context
        return value

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        del value
        return None


class PassThroughRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Explicit pass-through declaration for non-projecting runtime value types."""

    value_type = (
        np.ndarray,
        Mapping,
        str,
        bytes,
        bytearray,
        int,
        float,
        bool,
        Enum,
        DtypeConversionConfig,
        type(None),
    )


class ArrayBridgeArrayRuntimeSliceProjectionStrategy(
    RuntimeSliceProjectionStrategy
):
    """Preserve external ArrayBridge arrays whose shape declares no runtime axis."""


class SpatialGraphRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Explicitly preserve one scalar spatial graph across slice projection."""

    value_type = SpatialGraph


class ImagePayloadRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project image payloads through their declared runtime plane axis."""

    value_type = (ImageMetadataPayload, MaskedImagePayload)

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        metadata = image_payload_metadata(value)
        if metadata.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return None
        data = np.asarray(image_payload_data(value))
        if data.ndim < 3:
            raise RuntimeSliceProjectionDeclarationError(
                "Runtime-slice image payload must expose its declared plane axis "
                f"as the leading dimension, got shape {tuple(data.shape)!r}."
            )
        return int(data.shape[0])

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        plane_axis = image_payload_metadata(value).plane_axis
        if plane_axis is not context.axis:
            return value
        data = np.asarray(image_payload_data(value))
        context.validate_shape(data.shape, value_name="Image payload")
        plane_index = context.require_plane_index()
        context.validate_plane_index(plane_index, data.shape)
        return image_payload_slice_context(
            value,
            data[plane_index],
            plane_index,
            plane_axis=context.axis,
        )


class AlignedImageStackRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project an aligned image stack through its declared outer or inner axis."""

    value_type = AlignedImageStack

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        aligned = cast(AlignedImageStack, value)
        if context.axis is RuntimePlaneAxis.RUNTIME_SLICE:
            return aligned.aligned_slice(
                context.require_plane_index(),
                context.axis_size,
            )
        projected_slices = tuple(
            RuntimeSliceProjection.value_for_slice(image_slice, context)
            for image_slice in aligned.slices
        )
        if all(
            projected is image_slice
            for projected, image_slice in zip(
                projected_slices,
                aligned.slices,
                strict=True,
            )
        ):
            return aligned
        return aligned.with_slices(projected_slices)

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        return len(cast(AlignedImageStack, value).slices)


class ImageOutputBundleRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project each named output through its shared declared runtime axis."""

    value_type = ImageOutputBundle

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        bundle = cast(ImageOutputBundle, value)
        inner_slice_count = RuntimeSliceProjection.slice_count_from_values(
            bundle.slices
        )
        if context.axis is RuntimePlaneAxis.RUNTIME_SLICE and inner_slice_count is None:
            return bundle.aligned_slice(
                context.require_plane_index(),
                context.axis_size,
            )
        projected_outputs = tuple(
            RuntimeSliceProjection.value_for_slice(output, context)
            for output in bundle.slices
        )
        if all(
            projected is output
            for projected, output in zip(
                projected_outputs,
                bundle.slices,
                strict=True,
            )
        ):
            return bundle
        return bundle.with_slices(projected_outputs)

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        bundle = cast(ImageOutputBundle, value)
        inner_slice_count = RuntimeSliceProjection.slice_count_from_values(
            bundle.slices
        )
        return len(bundle.slices) if inner_slice_count is None else inner_slice_count


class RuntimeSliceIdentityProjectionMixin:
    """Mixin for value families that own execution-slice identity stamping."""

    def identity_projected_value(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        if context.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return value
        identity_projectable = cast(RuntimeSliceIdentityProjectableValue, value)
        return cast(
            RuntimeProjectionData,
            identity_projectable.with_runtime_slice_identity(
                slice_index=context.require_plane_index(),
                slice_count=context.axis_size,
            ),
        )


class RuntimeSliceRelationshipProjectionMixin:
    """Mixin for relationship payloads that carry the same slice-count contract."""

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        return RuntimeSliceProjection.relationship_slice_count(
            cast(
                DirectedObjectRelationshipPayload | ObjectRelationship,
                value,
            )
        )


class RuntimeSliceIdentityProjectableValueProjectionStrategy(
    RuntimeSliceIdentityProjectionMixin,
    RuntimeSliceProjectionStrategy,
):
    """Projection strategy for identity-stamping values without slice projection."""

    value_type = RuntimeSliceIdentityProjectableValue


class RuntimeSliceAlignedValueProjectionStrategy(
    RuntimeSliceProjectionStrategy,
):
    """Projection strategy for values already aligned to runtime slices."""

    value_type = RuntimeSliceAlignedValueSet

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        if context.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return value
        aligned = cast(RuntimeSliceAlignedValueSet, value)
        return aligned.value_for_aligned_slice(
            context.require_plane_index(),
            context.axis_size,
        )

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
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        table = cast(MeasurementTable, value)
        if (
            context.axis is not RuntimePlaneAxis.RUNTIME_SLICE
            or self.row_axis_domain(table) is None
        ):
            return table
        return MeasurementTableAxisProjection(
            axis=MeasurementRowAxisField.SLICE_INDEX,
            value=context.require_plane_index(),
            table=table,
        ).apply()

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        table = cast(MeasurementTable, value)
        if self.row_axis_domain(table) is None:
            return None
        source_provenance = table.source_provenance
        if source_provenance.source_plane_count > 0:
            return source_provenance.source_plane_count
        if source_provenance.addressable:
            return None
        raise RuntimeSliceProjectionDeclarationError(
            "Runtime-slice measurement table requires declared source-plane "
            "provenance; row fields and row ordering cannot declare its axis."
        )

    @staticmethod
    def row_axis_domain(table: MeasurementTable) -> tuple[int, ...] | None:
        return MeasurementTableUnion(table.name, (table,)).row_axis_domain(
            MeasurementRowAxisField.SLICE_INDEX
        )


class ColumnarRowsRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Project schema-bearing measurement rows through their axis authority."""

    value_type = ColumnarRows

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        if context.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return value
        return MeasurementRowsAxisProjection.from_rows(
            cast(ColumnarRows, value)
        ).project_runtime_slice_index(context.require_plane_index())


class RuntimeSliceProjectableValueProjectionStrategy(
    RuntimeSliceProjectionStrategy,
):
    """Projection strategy for values that implement the runtime-slice hook."""

    value_type = RuntimeSliceProjectableValue

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        if context.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return value
        projectable = cast(RuntimeSliceProjectableValue, value)
        return cast(
            RuntimeProjectionData,
            projectable.project_runtime_slice(context.require_plane_index()),
        )


class DirectedObjectRelationshipRuntimeSliceProjectionStrategy(
    RuntimeSliceIdentityProjectionMixin,
    RuntimeSliceRelationshipProjectionMixin,
    RuntimeSliceProjectableValueProjectionStrategy,
):
    """Projection strategy for endpoint-neutral directed relationship payloads."""

    value_type = DirectedObjectRelationshipPayload


class ObjectRelationshipRuntimeSliceProjectionStrategy(
    RuntimeSliceRelationshipProjectionMixin,
    RuntimeSliceProjectableValueProjectionStrategy,
):
    """Projection strategy for named object relationship payloads."""

    value_type = ObjectRelationship


class SparseIJVLabelRowsRuntimeSliceProjectionStrategy(
    RuntimeSliceProjectionStrategy,
):
    """Projection strategy for sparse IJV label rows."""

    value_type = SparseIJVLabelRows

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        if context.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return value
        return cast(SparseIJVLabelRows, value).slice(context.require_plane_index())

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        return cast(SparseIJVLabelRows, value).slice_count


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
        labels = cast(ObjectLabelValue, value)
        if labels.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return None
        plane_count = labels.declared_plane_count()
        if plane_count is None:
            raise RuntimeSliceProjectionDeclarationError(
                "Runtime-slice object labels have no nominal plane-stack contract."
            )
        return plane_count

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        labels = cast(ObjectLabelValue, value)
        if labels.plane_axis is not context.axis:
            return labels
        plane_count = labels.declared_plane_count()
        if plane_count is None:
            raise RuntimeSliceProjectionDeclarationError(
                "Object labels have no nominal plane-stack contract for their "
                f"declared {context.axis.value!r} axis."
            )
        if plane_count != context.axis_size:
            raise ValueError(
                "Object-label runtime plane-axis cardinality mismatch: "
                f"declared {plane_count!r}, execution requires {context.axis_size}."
            )
        source_plane_count = labels.source_provenance.source_plane_count
        if source_plane_count not in (0, context.axis_size):
            raise ValueError(
                "Object-label source provenance must be absent or exactly match "
                f"the declared plane axis: {source_plane_count} != {context.axis_size}."
            )
        dense_labels = object_label_dense_array(labels)
        context.validate_shape(
            dense_labels.shape,
            value_name="Object-label payload",
        )
        plane_index = context.require_plane_index()
        context.validate_plane_index(plane_index, dense_labels.shape)
        return labels.with_source_plane_measurement_labels(
            dense_labels[plane_index],
            plane_index,
        )


class SequenceRuntimeSliceProjectionStrategy(RuntimeSliceProjectionStrategy):
    """Projection strategy for tuple/list containers."""

    value_type = (tuple, list)

    def value_for_slice(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        return RuntimeSliceProjection.sequence_value_for_slice(
            cast(RuntimeProjectionSequence, value),
            context,
        )

    def identity_projected_value(
        self,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        return cast(
            RuntimeProjectionData,
            RuntimeSliceProjection.sequence_identity_projected_value(
                cast(RuntimeProjectionSequence, value),
                context,
            ),
        )

    def slice_count_for_value(
        self,
        value: RuntimeProjectionData,
    ) -> int | None:
        return RuntimeSliceProjection.slice_count_from_values(
            cast(RuntimeProjectionSequence, value)
        )


class RuntimeSliceProjection:
    """SSOT for runtime-slice count and value projection."""

    @classmethod
    def preserved_context_for_value(
        cls,
        value: RuntimeProjectionData,
    ) -> RuntimePlaneAxisValueProjection | None:
        """Return the preserved runtime-slice coordinate declared by ``value``."""
        slice_count = cls.slice_count_from_values((value,))
        if slice_count is None:
            return None
        return RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=slice_count,
        )

    @classmethod
    def value_for_singleton_slice(
        cls,
        value: RuntimeProjectionData,
        *,
        source_description: str,
    ) -> RuntimeProjectionData:
        """Consume one payload-declared singleton runtime-slice axis."""

        projection = cls.preserved_context_for_value(value)
        if projection is None:
            raise RuntimeSliceProjectionDeclarationError(
                f"{source_description} has no declared runtime-slice axis."
            )
        if projection.axis_size != 1:
            raise ValueError(
                f"{source_description} must declare exactly one runtime slice "
                f"before its axis can be consumed, got {projection.axis_size}."
            )
        return cls.value_for_slice(value, projection.selected_plane(0))

    @classmethod
    def context_for_value(
        cls,
        value: RuntimeProjectionData,
        *,
        slice_index: int,
        slice_count: int | None = None,
        source_description: str,
    ) -> RuntimePlaneAxisValueProjection | None:
        effective_slice_count = (
            slice_count
            if slice_count is not None
            else cls.slice_count_from_values((value,))
        )
        if effective_slice_count is not None:
            return RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                plane_index=slice_index,
                axis_size=effective_slice_count,
            )
        del source_description
        return None

    @classmethod
    @overload
    def value_for_slice(
        cls,
        value: ObjectLabelValue,
        context: RuntimePlaneAxisValueProjection,
    ) -> ObjectLabelValue: ...

    @classmethod
    @overload
    def value_for_slice(
        cls,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData: ...

    @classmethod
    def value_for_slice(
        cls,
        value: RuntimeProjectionData,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        normalized_value = cls.runtime_slice_normalized_value(value)
        return RuntimeSliceProjectionStrategy.strategy_for_value(
            normalized_value
        ).value_for_slice(normalized_value, context)

    @classmethod
    def sequence_identity_projected_value(
        cls,
        value: RuntimeProjectionSequence,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        """Project execution-slice identity recursively through nested outputs."""
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

    @classmethod
    def object_label_endpoint(
        cls,
        value: RuntimeProjectionData,
        *,
        context: RuntimePlaneAxisValueProjection | None = None,
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
        context: RuntimePlaneAxisValueProjection | None = None,
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
        context: RuntimePlaneAxisValueProjection,
    ) -> dict[str, RuntimeProjectionData]:
        return {
            name: cls.value_for_slice(value, context) for name, value in kwargs.items()
        }

    @classmethod
    def slice_count_from_kwargs(
        cls,
        kwargs: Mapping[str, RuntimeProjectionData],
    ) -> int | None:
        return cls.slice_count_from_values(kwargs.values())

    @classmethod
    def runtime_slice_normalized_value(
        cls,
        value: RuntimeProjectionData,
    ) -> RuntimeProjectionData:
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
        declared_slice_count = cls.single_slice_count(
            set(declared_slice_counts),
            source_description="declared runtime-slice values",
        )
        return declared_slice_count

    @classmethod
    def sequence_value_for_slice(
        cls,
        value: RuntimeProjectionSequence,
        context: RuntimePlaneAxisValueProjection,
    ) -> RuntimeProjectionData:
        """Project tuple/list containers through their item semantics."""
        projected = [
            cls.value_for_slice(
                item,
                context,
            )
            for item in value
        ]
        return tuple(projected) if isinstance(value, tuple) else projected

    @staticmethod
    def relationship_slice_count(
        value: (DirectedObjectRelationshipPayload | ObjectRelationship),
    ) -> int | None:
        return value.slice_count

    @staticmethod
    def measurement_table_slice_count(value: RuntimeProjectionData) -> int | None:
        if isinstance(value, (tuple, list)):
            counts = {
                count
                for table in value
                if isinstance(table, MeasurementTable)
                for count in (
                    RuntimeSliceProjectionStrategy.strategy_for_value(
                        table
                    ).slice_count_for_value(table),
                )
                if count is not None
            }
            return RuntimeSliceProjection.single_slice_count(
                counts,
                source_description="declared measurement table values",
            )
        if not isinstance(value, MeasurementTable):
            return None
        return RuntimeSliceProjectionStrategy.strategy_for_value(
            value
        ).slice_count_for_value(value)

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
                f"Conflicting runtime slice counts from {source_description}."
            )
        return next(iter(slice_counts))
