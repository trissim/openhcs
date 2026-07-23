"""Typed runtime artifact contracts for OpenHCS.

Artifacts are named, non-primary-image values produced or consumed by function
invocations. They cover current side-channel I/O and provide the extension point for
objects, measurements, relationships, and other richer runtime state.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass, field, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, cast

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.component_set import ComponentSet

if TYPE_CHECKING:
    from openhcs.core.runtime_artifact_values import (
        RuntimeValue,
    )
    from openhcs.core.runtime_image_values import ImagePayloadMetadata
    from openhcs.core.runtime_measurements import MeasurementSubject
    from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet


class ArtifactPayloadShape(str, Enum):
    """Generic runtime payload shape required by an artifact kind."""

    ANY = "any"
    ARRAY = "array"
    TABLE = "table"
    MAPPING = "mapping"

    def accepts(self, value: object) -> bool:
        """Return whether a runtime value satisfies this nominal payload shape."""

        match self:
            case ArtifactPayloadShape.ANY:
                return True
            case ArtifactPayloadShape.ARRAY:
                from openhcs.core.runtime_array_values import is_array_payload

                return is_array_payload(value)
            case ArtifactPayloadShape.TABLE:
                from openhcs.core.runtime_tabular_values import is_table_payload

                return is_table_payload(value)
            case ArtifactPayloadShape.MAPPING:
                return isinstance(value, Mapping)


class NamedArtifactPayload:
    """Nominal payload whose declared name must match its compiled artifact."""

    __slots__ = ()
    name: str

    def validate_artifact_name(self, expected_name: str | None = None) -> None:
        """Require a nonempty name and, when supplied, its compiled identity."""

        if not self.name:
            raise ValueError(f"{type(self).__name__}.name cannot be empty.")
        if expected_name is not None and self.name != expected_name:
            raise ValueError(
                f"Runtime payload name {self.name!r} does not match planned "
                f"artifact {expected_name!r}."
            )


class ArtifactType(ABC, metaclass=AutoRegisterMeta):
    """Registered runtime artifact payload category."""

    __registry_key__ = "value"
    __skip_if_no_key__ = True

    value: ClassVar[str | None] = None
    payload_shape: ClassVar[ArtifactPayloadShape] = ArtifactPayloadShape.ANY
    participates_in_measurement_source_names: ClassVar[bool] = False
    participates_in_main_flow_output: ClassVar[bool] = False
    carries_source_image_context: ClassVar[bool] = False
    payload_description: ClassVar[str | None] = None

    @classmethod
    def runtime_parameter_types(cls) -> tuple[type, ...]:
        """Return nominal payload types accepted by callable artifact parameters."""

        return ()

    @classmethod
    def accepts_parameter_annotation(cls, annotation: object) -> bool:
        """Return whether an annotation accepts this artifact's runtime payload."""

        from openhcs.core.pipeline.function_contracts import (
            annotation_accepts_runtime_type,
        )

        return any(
            annotation_accepts_runtime_type(annotation, runtime_type)
            for runtime_type in cls.runtime_parameter_types()
        )

    @classmethod
    def coerce(cls, artifact_type: "ArtifactTypeValue") -> type["ArtifactType"]:
        """Return the registered artifact type for a class or wire value."""
        if isinstance(artifact_type, str):
            try:
                return cls.__registry__[artifact_type]
            except KeyError as exc:
                raise ValueError(f"Unknown artifact type {artifact_type!r}.") from exc
        if isinstance(artifact_type, type) and issubclass(artifact_type, cls):
            return artifact_type
        raise TypeError(
            "Artifact type must be an ArtifactType class or registered value, "
            f"got {type(artifact_type).__name__}."
        )

    @classmethod
    def require_value(cls) -> str:
        if cls.value is None:
            raise TypeError(f"{cls.__name__} does not declare an artifact type value.")
        return cls.value

    @classmethod
    def description(cls) -> str:
        if cls.payload_description is not None:
            return cls.payload_description
        return f"{cls.payload_shape} {cls.require_value()} payload"

    @classmethod
    def diagnostic_label(cls) -> str:
        """Return the stable artifact type label used in diagnostics."""
        return f"<{cls.__name__}: {cls.require_value()!r}>"

    @classmethod
    def normalize_group_scoped_payload(
        cls,
        output_plan: "ArtifactOutputPlan",
        value: object,
    ) -> object:
        """Normalize payload axes already represented by compiled group scope."""

        del output_plan
        return value

    @classmethod
    def normalize_runtime_payload(
        cls,
        name: str,
        value: object,
    ) -> object:
        """Convert one raw return into this artifact type's nominal payload."""

        del name
        return value

    @classmethod
    def normalize_source_payload(
        cls,
        data: object,
        channel_axis: int | None,
    ) -> tuple[object, int | None]:
        """Normalize source pixels according to this artifact's payload semantics."""

        return data, channel_axis

    @classmethod
    def compose_runtime_values(
        cls,
        values: Sequence["RuntimeValue"],
        producer_group_scope: ComponentGroupScope | None = None,
    ) -> object:
        """Compose producer records according to this artifact's semantics."""

        del producer_group_scope
        if len(values) != 1:
            raise ValueError(
                f"{cls.__name__} does not define grouped runtime composition for "
                f"{len(values)} values."
            )
        return values[0].data

    @classmethod
    def normalize_slice_aligned_value(
        cls,
        value: "RuntimeSliceAlignedValueSet[object]",
        output_plan: "ArtifactOutputPlan",
        *,
        axis_id: str,
    ) -> object | None:
        """Aggregate an owned slice-aligned payload before runtime storage."""

        del value, output_plan, axis_id
        return None

    @classmethod
    def accepts_runtime_payload(cls, data: object) -> bool:
        """Return whether data satisfies this artifact's nominal payload contract."""

        return cls.payload_shape.accepts(data)

    @classmethod
    def validate_runtime_payload(
        cls,
        name: str,
        data: object,
    ) -> None:
        """Validate one nominal payload against this artifact type."""

        from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet

        if isinstance(data, RuntimeSliceAlignedValueSet):
            invalid = tuple(
                type(value).__name__
                for value in (
                    data.value_for_slice(index) for index in range(data.slice_count)
                )
                if not cls.accepts_runtime_payload(value)
            )
            if not invalid:
                return
            raise TypeError(
                f"Artifact {name!r} expected slice-aligned {cls.description()}, "
                f"got invalid values {invalid!r}."
            )
        if cls.accepts_runtime_payload(data):
            if isinstance(data, NamedArtifactPayload):
                data.validate_artifact_name(name)
            return
        raise TypeError(
            f"Artifact {name!r} expected {cls.description()}, "
            f"got {type(data).__name__}."
        )

    @classmethod
    def materialization_payload(cls, value: "RuntimeValue") -> object:
        """Return the payload exposed to materializers for this artifact type."""
        return value.data

    @classmethod
    def runtime_semantic_id(cls, data: object) -> str | None:
        """Return artifact-owned subidentity for one nominal runtime value."""

        del data
        return None

    @classmethod
    def publishes_to_main_flow(
        cls,
        sidecar_role: "ArtifactSidecarRole | None",
    ) -> bool:
        """Return whether this artifact role participates in canonical image flow."""

        return cls.participates_in_main_flow_output and sidecar_role is None


ArtifactTypeValue = type[ArtifactType] | str


def artifact_type_strategy_key_from_class(name: str, cls: type[object]) -> str | None:
    """Return the nominal strategy key for a class declaring an ArtifactType member."""
    del name
    member = cls.__dict__.get("artifact_type")
    if isinstance(member, type) and issubclass(member, ArtifactType):
        return artifact_type_strategy_key(member)
    return None


def artifact_type_strategy_key(artifact_type: type[ArtifactType]) -> str:
    """Return the JSON-safe nominal key for an artifact-type strategy."""
    return f"{artifact_type.__module__}.{artifact_type.__qualname__}"


class ArtifactTypeStrategyMatchMixin:
    """MRO match hook for strategy roots selected by ArtifactType inheritance."""

    artifact_type: ClassVar[type[ArtifactType] | None] = None
    __key_extractor__ = staticmethod(artifact_type_strategy_key_from_class)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        member = cls.__dict__.get("artifact_type")
        if (
            isinstance(member, type)
            and issubclass(member, ArtifactType)
            and cls.__dict__.get("strategy_key") is None
        ):
            cls.strategy_key = artifact_type_strategy_key(member)

    @classmethod
    def for_artifact_type(cls, artifact_type: ArtifactTypeValue):
        return cls.for_context(ArtifactType.coerce(artifact_type))

    def matches(self, context: type[ArtifactType]) -> bool:
        artifact_type = type(self).artifact_type
        return artifact_type is not None and issubclass(
            ArtifactType.coerce(context), artifact_type
        )


class SpecialArtifactType(ArtifactType):
    """Generic artifact type for explicit side-channel payloads."""

    value = "special"


class ImageArtifactType(ArtifactType):
    """Image array artifact type."""

    value = "image"
    payload_shape = ArtifactPayloadShape.ARRAY
    participates_in_measurement_source_names = True
    participates_in_main_flow_output = True
    carries_source_image_context = True

    @classmethod
    def runtime_parameter_types(cls) -> tuple[type, ...]:
        import numpy as np

        from openhcs.core.runtime_array_values import RuntimeArrayPayload

        return (np.ndarray, RuntimeArrayPayload)

    @classmethod
    def normalize_runtime_payload(
        cls,
        name: str,
        value: object,
    ) -> object:
        """Apply the declared image identity without discarding payload context."""

        from openhcs.core.runtime_image_values import (
            image_payload_data,
            image_payload_mask,
            image_payload_metadata,
        )
        from openhcs.core.runtime_slice_alignment import (
            RuntimeSliceAlignedValues,
            RuntimeSliceAlignedValueSet,
        )

        def named_payload(payload: object) -> object:
            metadata = image_payload_metadata(payload)
            return metadata.with_source_provenance(
                metadata.source_provenance.with_derived_source_image_names((name,))
            ).payload_with(
                image_payload_data(payload),
                image_payload_mask(payload),
            )

        if isinstance(value, RuntimeSliceAlignedValueSet):
            return RuntimeSliceAlignedValues(
                tuple(
                    named_payload(value.value_for_slice(index))
                    for index in range(value.slice_count)
                )
            )
        return named_payload(value)

    @classmethod
    def materialization_payload(cls, value: "RuntimeValue") -> object:
        """Image payloads retain their metadata at runtime."""

        return value.data

    @classmethod
    def compose_runtime_values(
        cls,
        values: Sequence["RuntimeValue"],
        producer_group_scope: ComponentGroupScope | None = None,
    ) -> object:
        """Stack grouped image records while preserving payload context."""

        if producer_group_scope is None and len(values) == 1:
            return values[0].data

        from openhcs.core.aligned_image_payload import stack_image_payloads
        from openhcs.core.aligned_image_payload import stack_image_payload_context
        from openhcs.core.memory import detect_memory_type, stack_runtime_slices
        from openhcs.core.runtime_image_values import (
            ImagePayloadMetadataCompositionMode,
            image_payload_data,
        )
        from openhcs.core.runtime_plane_projection import RuntimePlaneAxis

        components = {value.key.scope.component for value in values}
        if None in components or len(components) != 1:
            raise ValueError(
                "Grouped image composition requires one exact declared component "
                f"axis, got {components!r}."
            )
        if producer_group_scope is not None:
            payloads = tuple(value.data for value in values)
            arrays = tuple(image_payload_data(payload) for payload in payloads)
            return stack_image_payload_context(
                payloads,
                stack_runtime_slices(arrays, detect_memory_type(arrays[0]), 0),
                metadata_mode=ImagePayloadMetadataCompositionMode.for_plane_axis(
                    RuntimePlaneAxis.RUNTIME_SLICE
                ),
            )
        return stack_image_payloads(
            tuple(value.data for value in values),
            metadata_mode=ImagePayloadMetadataCompositionMode.for_plane_axis(
                RuntimePlaneAxis.RUNTIME_SLICE
            ),
        )

    @classmethod
    def normalize_group_scoped_payload(
        cls,
        output_plan: "ArtifactOutputPlan",
        value: object,
    ) -> object:
        """Consume the singleton image axis represented by the artifact key."""

        from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

        projection = RuntimeSliceProjection.preserved_context_for_value(value)
        if projection is None:
            return value
        if projection.axis_size != 1:
            raise ValueError(
                f"Group-scoped scalar artifact {output_plan.name!r} for "
                f"{output_plan.group_component.value!r} cannot retain a declared "
                f"runtime-slice axis of size {projection.axis_size}."
            )
        return RuntimeSliceProjection.value_for_slice(
            value,
            projection.selected_plane(0),
        )


class ObjectLabelsArtifactType(ArtifactType):
    """Object-label array artifact type."""

    value = "object_labels"
    payload_shape = ArtifactPayloadShape.ARRAY
    participates_in_main_flow_output = True
    carries_source_image_context = True
    payload_description = "object_labels payload"

    @classmethod
    def runtime_parameter_types(cls) -> tuple[type, ...]:
        from openhcs.core.runtime_object_labels import ObjectLabelValue

        return (ObjectLabelValue,)

    @classmethod
    def _bind_compiled_name(cls, name: str, value: object) -> object:
        """Bind or validate the compiled identity of one nominal label value."""

        from openhcs.core.runtime_object_labels import ObjectLabelSet, ObjectLabelValue

        if isinstance(value, ObjectLabelSet):
            value.validate_artifact_name(name)
            return value
        if isinstance(value, ObjectLabelValue):
            return ObjectLabelSet.from_payload(name, value)
        return value

    @classmethod
    def normalize_runtime_payload(
        cls,
        name: str,
        value: object,
    ) -> object:
        """Bind scalar object labels to their compiled artifact identity."""

        from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet

        if isinstance(value, RuntimeSliceAlignedValueSet):
            return value
        return cls._bind_compiled_name(name, value)

    @classmethod
    def normalize_source_payload(
        cls,
        data: object,
        channel_axis: int | None,
    ) -> tuple[object, int | None]:
        """Convert color-coded source labels into canonical integer IDs."""

        from openhcs.core.runtime_object_labels import normalize_source_label_data

        return normalize_source_label_data(data, channel_axis), None

    @classmethod
    def compose_runtime_values(
        cls,
        values: Sequence["RuntimeValue"],
        producer_group_scope: ComponentGroupScope | None = None,
    ) -> object:
        """Aggregate grouped object-label records into one runtime stack."""

        from openhcs.core.memory import detect_memory_type
        from openhcs.core.runtime_object_label_aggregation import (
            ObjectLabelPure2DSliceAggregator,
        )
        from openhcs.core.runtime_plane_projection import RuntimePlaneAxis

        if producer_group_scope is None and len(values) == 1:
            return values[0].data
        labels = tuple(value.data for value in values)
        if len({value.representation for value in labels}) != 1:
            raise ValueError(
                "Cannot compose grouped object labels with mixed representations."
            )
        aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
            labels,
            detect_memory_type(labels[0].labels),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )
        return cls._bind_compiled_name(values[0].name, aggregated)

    @classmethod
    def normalize_slice_aligned_value(
        cls,
        value: "RuntimeSliceAlignedValueSet[object]",
        output_plan: "ArtifactOutputPlan",
        *,
        axis_id: str,
    ) -> object | None:
        """Aggregate object-label slices into one plane-scoped label domain."""

        from openhcs.core.memory import detect_memory_type
        from openhcs.core.runtime_object_label_aggregation import (
            ObjectLabelPure2DSliceAggregator,
        )
        from openhcs.core.runtime_object_labels import ObjectLabelValue
        from openhcs.core.runtime_plane_projection import RuntimePlaneAxis

        slices = tuple(
            value.value_for_slice(index) for index in range(value.slice_count)
        )
        if not slices or not all(isinstance(item, ObjectLabelValue) for item in slices):
            return None
        aggregated = ObjectLabelPure2DSliceAggregator.aggregate(
            slices,
            detect_memory_type(slices[0].labels),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )
        return cls._bind_compiled_name(output_plan.name, aggregated)

    @classmethod
    def accepts_runtime_payload(
        cls,
        data: object,
    ) -> bool:
        """Accept only nominal object-label values."""

        from openhcs.core.runtime_object_labels import ObjectLabelValue

        return isinstance(data, ObjectLabelValue)

    @classmethod
    def validate_runtime_payload(
        cls,
        name: str,
        data: object,
    ) -> None:
        """Accept nominal label aggregates that own their slice-aligned domain."""

        from openhcs.core.runtime_object_labels import ObjectLabelValue

        if isinstance(data, ObjectLabelValue):
            if isinstance(data, NamedArtifactPayload):
                data.validate_artifact_name(name)
            return
        super().validate_runtime_payload(name, data)

    @classmethod
    def materialization_payload(cls, value: "RuntimeValue") -> object:
        """Expose object-label storage without discarding its nominal metadata."""

        return value.data


class MeasurementsArtifactType(ArtifactType):
    """Measurement-table artifact type."""

    value = "measurements"
    payload_shape = ArtifactPayloadShape.TABLE

    @classmethod
    def runtime_parameter_types(cls) -> tuple[type, ...]:
        from openhcs.core.runtime_tabular_values import ColumnarRows

        return (ColumnarRows,)

    @classmethod
    def accepts_runtime_payload(cls, data: object) -> bool:
        from openhcs.core.runtime_measurements import MeasurementTable

        return isinstance(data, MeasurementTable)

    @classmethod
    def runtime_semantic_id(cls, data: object) -> str | None:
        from openhcs.core.runtime_measurements import MeasurementTable

        if not isinstance(data, MeasurementTable):
            return None
        return data.runtime_semantic_id

    @classmethod
    def compose_runtime_values(
        cls,
        values: Sequence["RuntimeValue"],
        producer_group_scope: ComponentGroupScope | None = None,
    ) -> object:
        """Concatenate exact producer-group measurement tables."""

        del producer_group_scope
        from openhcs.core.runtime_artifact_queries import MeasurementTableUnion

        return MeasurementTableUnion(
            values[0].name,
            tuple(value.data for value in values),
        ).as_table()

    @classmethod
    def materialization_payload(cls, value: "RuntimeValue") -> object:
        from openhcs.core.runtime_measurements import MeasurementTable

        table = value.data
        if not isinstance(table, MeasurementTable):
            cls.validate_runtime_payload(value.name, table)
        return table.rows


class ObjectLineageArtifactType(ArtifactType):
    """Internal directed object lineage used to project object measurements."""

    value = "object_lineage"
    payload_shape = ArtifactPayloadShape.TABLE

    @classmethod
    def accepts_runtime_payload(cls, data: object) -> bool:
        from openhcs.core.runtime_relationships import ObjectRelationship

        return isinstance(data, ObjectRelationship)

    @classmethod
    def compose_runtime_values(
        cls,
        values: Sequence["RuntimeValue"],
        producer_group_scope: ComponentGroupScope | None = None,
    ) -> object:
        """Reconstruct relationship records and retain grouped slice alignment."""

        from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues

        relationships = tuple(value.data for value in values)
        if producer_group_scope is None and len(relationships) == 1:
            return relationships[0]
        return RuntimeSliceAlignedValues(relationships)


class RelationshipsArtifactType(ObjectLineageArtifactType):
    """Explicitly recorded directed relationship exported to external consumers."""

    value = "relationships"

    @classmethod
    def materialization_payload(cls, value: "RuntimeValue") -> object:
        return value.data.as_table()


class TableArtifactType(ArtifactType):
    """Generic table artifact type."""

    value = "table"
    payload_shape = ArtifactPayloadShape.TABLE


class SpatialGridArtifactType(ArtifactType):
    """Spatial-grid mapping artifact type."""

    value = "spatial_grid"
    payload_shape = ArtifactPayloadShape.MAPPING
    payload_description = "spatial grid mapping"

    @classmethod
    def runtime_parameter_types(cls) -> tuple[type, ...]:
        from openhcs.core.runtime_spatial_grid import SpatialGrid

        return (SpatialGrid,)

    @classmethod
    def normalize_runtime_payload(
        cls,
        name: str,
        value: object,
    ) -> object:
        """Normalize scalar or runtime-slice grid returns to nominal values."""

        from openhcs.core.runtime_slice_alignment import (
            RuntimeSliceAlignedValues,
            RuntimeSliceAlignedValueSet,
        )
        from openhcs.core.runtime_spatial_grid import SpatialGrid

        if isinstance(value, RuntimeSliceAlignedValueSet):
            return RuntimeSliceAlignedValues(
                tuple(
                    SpatialGrid.from_runtime_value(
                        name,
                        value.value_for_slice(slice_index),
                    )
                    for slice_index in range(value.slice_count)
                )
            )
        if isinstance(value, (list, tuple)):
            return RuntimeSliceAlignedValues(
                tuple(SpatialGrid.from_runtime_value(name, item) for item in value)
            )
        return SpatialGrid.from_runtime_value(name, value)

    @classmethod
    def accepts_runtime_payload(cls, data: object) -> bool:
        from openhcs.core.runtime_spatial_grid import SpatialGrid

        return isinstance(data, SpatialGrid)

    @classmethod
    def compose_runtime_values(
        cls,
        values: Sequence["RuntimeValue"],
        producer_group_scope: ComponentGroupScope | None = None,
    ) -> object:
        """Compose only spatial grids with identical declared slice domains."""

        del producer_group_scope
        from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
        from openhcs.core.runtime_spatial_grid import SpatialGrid

        grids = tuple(value.data for value in values)
        if len(grids) == 1:
            return grids[0]
        aligned_grids = tuple(
            grid for grid in grids if isinstance(grid, RuntimeSliceAlignedValues)
        )
        if aligned_grids:
            if len(aligned_grids) != len(grids):
                raise ValueError(
                    "Grouped spatial-grid composition cannot mix scalar and "
                    "runtime-slice-aligned values."
                )
            slice_counts = {grid.slice_count for grid in aligned_grids}
            if len(slice_counts) != 1:
                raise ValueError(
                    "Grouped spatial-grid composition requires identical runtime "
                    f"slice counts, got {tuple(sorted(slice_counts))!r}."
                )
            slice_count = aligned_grids[0].slice_count
            composed_slices = []
            for slice_index in range(slice_count):
                candidates = tuple(
                    cast(SpatialGrid, grid.value_for_slice(slice_index))
                    for grid in aligned_grids
                )
                if any(
                    candidate.as_mapping() != candidates[0].as_mapping()
                    for candidate in candidates[1:]
                ):
                    raise ValueError(
                        "Grouped spatial-grid composition resolved non-identical "
                        f"grids at runtime slice {slice_index}."
                    )
                composed_slices.append(candidates[0])
            return RuntimeSliceAlignedValues(tuple(composed_slices))
        scalar_grids = tuple(cast(SpatialGrid, grid) for grid in grids)
        if any(
            grid.as_mapping() != scalar_grids[0].as_mapping()
            for grid in scalar_grids[1:]
        ):
            raise ValueError(
                "Grouped spatial-grid composition resolved non-identical grids."
            )
        return scalar_grids[0]

    @classmethod
    def materialization_payload(cls, value: "RuntimeValue") -> object:
        from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
        from openhcs.core.runtime_spatial_grid import SpatialGrid

        if isinstance(value.data, RuntimeSliceAlignedValueSet):
            return tuple(
                cast(
                    SpatialGrid,
                    value.data.value_for_slice(slice_index),
                ).as_mapping()
                for slice_index in range(value.data.slice_count)
            )
        return cast(SpatialGrid, value.data).as_mapping()


class SpatialGraphArtifactType(ArtifactType):
    """Spatial graph with edge paths, topology, and scalar edge features."""

    value = "spatial_graph"
    carries_source_image_context = True
    payload_description = "spatial graph payload"

    @classmethod
    def runtime_parameter_types(cls) -> tuple[type, ...]:
        from openhcs.core.runtime_spatial_graph import SpatialGraph

        return (SpatialGraph,)

    @classmethod
    def accepts_runtime_payload(cls, data: object) -> bool:
        from openhcs.core.runtime_spatial_graph import SpatialGraph

        return isinstance(data, SpatialGraph)


class MetadataArtifactType(ArtifactType):
    """Metadata mapping artifact type."""

    value = "metadata"
    payload_shape = ArtifactPayloadShape.MAPPING
    payload_description = "metadata mapping"


class ArtifactSidecarRole(str, Enum):
    """Named sidecar artifact roles derived from a primary artifact."""

    CROP_MASK = "crop_mask"
    MATERIALIZED_IMAGE_COPY = "materialized_image_copy"

    def name_for(
        self,
        primary_artifact_name: str,
        *,
        separator: str = "__",
    ) -> str:
        """Return the sidecar artifact name for one primary artifact."""
        if not separator:
            raise ValueError("ArtifactSidecarRole sidecar separator cannot be empty.")
        normalized = primary_artifact_name.strip()
        if not normalized:
            raise ValueError("primary_artifact_name cannot be empty.")
        return f"{normalized}{separator}{self.value}"


class ArtifactMaterializationPayload(ABC):
    """Nominal marker for rich artifact materialization metadata."""

    @abstractmethod
    def participates_in_runtime_export_observation(self) -> bool:
        """Return whether this materialization is a pipeline-declared export."""

    @abstractmethod
    def participates_in_persistent_materialization(self) -> bool:
        """Return whether this materialization writes to persistent targets."""

    @abstractmethod
    def uses_source_identity_filename(self) -> bool:
        """Return whether this materialization names files by source identity."""


def _coerce_artifact_plan_type(
    plan_type: type["ArtifactPlan"],
) -> type["ArtifactPlan"]:
    if isinstance(plan_type, type) and issubclass(plan_type, ArtifactPlan):
        return plan_type
    raise TypeError(
        "Artifact plan type must be an ArtifactPlan class, "
        f"got {type(plan_type).__name__}."
    )


def _require_registered_artifact_plan_type(
    plan_type: type["ArtifactPlan"],
    field_name: str,
) -> type["ArtifactPlan"]:
    resolved_plan_type = _coerce_artifact_plan_type(plan_type)
    if resolved_plan_type not in ArtifactPlan.__registry__.values():
        raise ValueError(
            f"{field_name} is not a registered ArtifactPlan type: "
            f"{resolved_plan_type.__name__}."
        )
    return resolved_plan_type


def _require_registered_artifact_type(
    artifact_type: ArtifactTypeValue,
    field_name: str,
) -> type[ArtifactType]:
    resolved_artifact_type = ArtifactType.coerce(artifact_type)
    if resolved_artifact_type not in ArtifactType.__registry__.values():
        raise ValueError(
            f"{field_name} is not a registered ArtifactType: "
            f"{resolved_artifact_type.__name__}."
        )
    return resolved_artifact_type


@dataclass(frozen=True)
class ArtifactSpecRef:
    """Scope-free identity for one declared artifact spec."""

    plan_type: type["ArtifactPlan"]
    artifact_type: type[ArtifactType]
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_type",
            _require_registered_artifact_plan_type(
                self.plan_type,
                "ArtifactSpecRef.plan_type",
            ),
        )
        object.__setattr__(
            self,
            "artifact_type",
            _require_registered_artifact_type(
                self.artifact_type,
                "ArtifactSpecRef.artifact_type",
            ),
        )
        if not self.name:
            raise ValueError("ArtifactSpecRef.name cannot be empty.")

    def for_plan_type(
        self,
        plan_type: type["ArtifactPlan"],
    ) -> "ArtifactSpecRef":
        """Project this artifact identity to another compiled plan role."""

        return replace(
            self,
            plan_type=_coerce_artifact_plan_type(plan_type),
        )


@dataclass(frozen=True)
class ObjectArtifactSubjectBinding:
    """Bind one artifact-local identity field to an exact object subject."""

    SUBJECT_FEATURE: ClassVar[str] = "__openhcs_object_subject__"
    SUBJECT_ID_FEATURE: ClassVar[str] = "__openhcs_object_subject_id__"

    source: ArtifactSpecRef
    id_field: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, ArtifactSpecRef):
            raise TypeError(
                "ObjectArtifactSubjectBinding.source must be an ArtifactSpecRef."
            )
        if self.source.artifact_type is not ObjectLabelsArtifactType:
            raise ValueError(
                "ObjectArtifactSubjectBinding requires an ObjectLabels source, got "
                f"{self.source.artifact_type.value}:{self.source.name}."
            )
        if not self.id_field:
            raise ValueError("ObjectArtifactSubjectBinding.id_field cannot be empty.")

    def subject_token(
        self,
        *,
        producer_step_scope_id: str | None,
        producer_step_index: int | str | None,
    ) -> str:
        """Return one scalar token shared by sibling outputs from this producer."""

        return json.dumps(
            (
                self.source.plan_type.plan_role,
                self.source.artifact_type.value,
                self.source.name,
                producer_step_scope_id,
                producer_step_index,
            ),
            separators=(",", ":"),
        )

    def feature_metadata(
        self,
        features: Mapping[str, object],
        *,
        producer_step_scope_id: str | None,
        producer_step_index: int | str | None,
    ) -> dict[str, object]:
        """Project the declared local member identity to framework metadata."""

        if self.id_field not in features:
            raise ValueError(
                f"Declared object-subject field {self.id_field!r} is absent from "
                f"artifact member features {tuple(features)!r}."
            )
        return {
            self.SUBJECT_FEATURE: self.subject_token(
                producer_step_scope_id=producer_step_scope_id,
                producer_step_index=producer_step_index,
            ),
            self.SUBJECT_ID_FEATURE: features[self.id_field],
        }


@dataclass(frozen=True)
class ArtifactSpecRelation(ABC, metaclass=AutoRegisterMeta):
    """Declare a semantic dependency on one source artifact.

    Specialized relations add projection or materialization behavior; the root
    relation preserves dependency provenance without changing runtime scope.
    """

    __registry_key__ = "relation_key"
    __skip_if_no_key__ = True

    relation_key: ClassVar[str | None] = None
    target_plan_type: ClassVar[type["ArtifactPlan"] | None] = None
    target_artifact_type: ClassVar[type[ArtifactType] | None] = None

    source: ArtifactSpecRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, ArtifactSpecRef):
            raise TypeError(
                "ArtifactSpecRelation.source must be an ArtifactSpecRef, "
                f"got {type(self.source).__name__}."
            )
        if self.target_plan_type is not None:
            _require_registered_artifact_plan_type(
                self.target_plan_type,
                f"{type(self).__name__}.target_plan_type",
            )
        if self.target_artifact_type is not None:
            _require_registered_artifact_type(
                self.target_artifact_type,
                f"{type(self).__name__}.target_artifact_type",
            )

    def require_target_spec(self, spec: "ArtifactSpec") -> None:
        if self.target_plan_type is not None:
            target_plan_type = _coerce_artifact_plan_type(self.target_plan_type)
            actual_plan_type = spec.require_plan_type()
            if actual_plan_type is not target_plan_type:
                raise ValueError(
                    f"{type(self).__name__} requires target plan role "
                    f"{target_plan_type.plan_role}, got {actual_plan_type.plan_role}."
                )
        if self.target_artifact_type is None:
            return
        target_artifact_type = ArtifactType.coerce(self.target_artifact_type)
        if spec.artifact_type is not target_artifact_type:
            raise ValueError(
                f"{type(self).__name__} requires target artifact type "
                f"{target_artifact_type.value}, got {spec.artifact_type.value}."
            )

    def bind_target_spec(self, spec: "ArtifactSpec") -> "ArtifactSpecRelation":
        """Bind target-owned relation state before validating the declaration."""

        del spec
        return self

    def group_scope_source(self) -> ArtifactSpecRef | None:
        """Return the source whose component group this relation propagates."""

        return None

    def materialization_source(self) -> ArtifactSpecRef | None:
        """Return the source whose identity names materialized output files."""

        return None

    def source_context_source(self) -> ArtifactSpecRef | None:
        """Return the source whose runtime context belongs to the target artifact."""

        return None

    def source_stack_scope_source(self) -> ArtifactSpecRef | None:
        """Return the source whose complete runtime stack the target preserves."""

        return None

    def measurement_subject(self) -> "MeasurementSubject | None":
        """Return the exact measurement subject declared by this relation."""

        return None

    def object_subject_binding(self) -> ObjectArtifactSubjectBinding | None:
        """Return the target-local identity bound to one object subject."""

        return None

    def stack_broadcast_source(self) -> ArtifactSpecRef | None:
        """Return the source whose runtime stack may broadcast this input."""

        return None

    def dependency_refs(self) -> tuple[ArtifactSpecRef, ...]:
        """Return every artifact that must exist before the relation target."""

        return (self.source,)

    def for_plan_type(
        self,
        plan_type: type["ArtifactPlan"],
    ) -> "ArtifactSpecRelation | None":
        """Project this relation to another artifact plan role."""

        target_plan_type = _coerce_artifact_plan_type(plan_type)
        if (
            self.target_plan_type is not None
            and self.target_plan_type is not target_plan_type
        ):
            return None
        return self


class ArtifactSidecarSourceRelation(ArtifactSpecRelation):
    """Declare the primary artifact whose identity selects one sidecar."""

    relation_key: ClassVar[str] = "artifact_sidecar_source"

    def bind_target_spec(self, spec: "ArtifactSpec") -> "ArtifactSpecRelation":
        if spec.sidecar_role is None:
            raise ValueError(
                f"{type(self).__name__} requires an artifact sidecar target."
            )
        return self


class ArtifactSourceContextSourceRelation(ArtifactSpecRelation, ABC):
    """Target artifact preserves runtime source context from a declared source."""

    def source_context_source(self) -> ArtifactSpecRef:
        """Return the artifact that supplies the target's runtime context."""

        return self.source


class ArtifactGroupScopeSourceRelation(ArtifactSourceContextSourceRelation, ABC):
    """Target artifact inherits group scope from a declared source artifact."""

    def group_scope_source(self) -> ArtifactSpecRef:
        """Return this relation's declared group-scope source."""

        return self.source


class GroupLineageSourceRelation(ArtifactGroupScopeSourceRelation):
    """Target artifact inherits grouping from a declared source artifact."""

    relation_key: ClassVar[str] = "group_lineage_source"


class SourceStackLineageSourceRelation(GroupLineageSourceRelation):
    """Target artifact preserves source-stack compatibility with a source artifact."""

    relation_key: ClassVar[str] = "source_stack_lineage_source"

    def source_stack_scope_source(self) -> ArtifactSpecRef:
        """Return the source whose complete runtime stack is preserved."""

        return self.source


class InputGroupLineageSourceRelation(ArtifactGroupScopeSourceRelation):
    """Input artifact inherits invocation grouping from a declared input source."""

    relation_key: ClassVar[str] = "input_group_lineage_source"


class InputStackBroadcastSourceRelation(ArtifactSourceContextSourceRelation):
    """Input artifact may broadcast across one declared source input stack."""

    relation_key: ClassVar[str] = "input_stack_broadcast_source"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source.artifact_type is not ImageArtifactType:
            raise ValueError(
                f"{type(self).__name__} requires an image source, got "
                f"{self.source.artifact_type.value}:{self.source.name}."
            )

    def stack_broadcast_source(self) -> ArtifactSpecRef:
        """Return the exact input whose runtime stack owns broadcast cardinality."""

        return self.source


class MaterializationSourceIdentityRelation(ArtifactSpecRelation):
    """Materialized output filenames inherit one declared source identity."""

    relation_key: ClassVar[str] = "materialization_source_identity"

    def materialization_source(self) -> ArtifactSpecRef:
        """Return the exact source artifact used for materialized filenames."""

        return self.source


@dataclass(frozen=True)
class ObjectMeasurementSubjectRelation(ArtifactSpecRelation):
    """Object-scoped measurements are owned by one ObjectLabels artifact."""

    relation_key: ClassVar[str] = "object_measurement_subject"
    id_field: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source.artifact_type is not ObjectLabelsArtifactType:
            raise ValueError(
                f"{type(self).__name__} requires an object-labels source, got "
                f"{self.source.artifact_type.value}:{self.source.name}."
            )

    def measurement_subject(self) -> "MeasurementSubject":
        from openhcs.core.runtime_measurements import (
            MeasurementScope,
            MeasurementSubject,
        )

        return MeasurementSubject(
            MeasurementScope.OBJECT,
            self.source.name,
            self.id_field,
        )

    def object_subject_binding(self) -> ObjectArtifactSubjectBinding | None:
        """Expose an explicit measurement-row identity for generic UI linkage."""

        if self.id_field is None:
            return None
        return ObjectArtifactSubjectBinding(self.source, self.id_field)


@dataclass(frozen=True)
class ObjectArtifactMemberSubjectRelation(ArtifactSpecRelation):
    """Artifact members are owned by objects through one declared local field."""

    relation_key: ClassVar[str] = "object_artifact_member_subject"
    source: ArtifactSpecRef | None = None
    member_id_field: str = "label"
    self_owned: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.member_id_field:
            raise ValueError(
                "ObjectArtifactMemberSubjectRelation.member_id_field cannot be empty."
            )
        if self.source is not None:
            super().__post_init__()
            if self.source.artifact_type is not ObjectLabelsArtifactType:
                raise ValueError(
                    f"{type(self).__name__} requires an object-labels source, got "
                    f"{self.source.artifact_type.value}:{self.source.name}."
                )

    def bind_target_spec(self, spec: "ArtifactSpec") -> "ArtifactSpecRelation":
        """Bind a source-less declaration to its owning ObjectLabels output."""

        if self.source is not None:
            return self
        if spec.artifact_type is not ObjectLabelsArtifactType:
            raise ValueError(
                f"{type(self).__name__} without an explicit source requires an "
                f"ObjectLabels target, got {spec.artifact_type.value}:{spec.name}."
            )
        return replace(self, source=spec.ref(), self_owned=True)

    def object_subject_binding(self) -> ObjectArtifactSubjectBinding:
        """Return the exact object subject and target-local member identity."""

        if self.source is None:
            raise RuntimeError(
                "ObjectArtifactMemberSubjectRelation must be bound to a target spec."
            )
        return ObjectArtifactSubjectBinding(self.source, self.member_id_field)

    def dependency_refs(self) -> tuple[ArtifactSpecRef, ...]:
        """A self-owned ObjectLabels member relation adds no dependency edge."""

        if self.source is None or self.self_owned:
            return ()
        return super().dependency_refs()


class ImageMeasurementSubjectRelation(ArtifactSpecRelation):
    """Image-scoped measurements are owned by one image artifact."""

    relation_key: ClassVar[str] = "image_measurement_subject"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source.artifact_type is not ImageArtifactType:
            raise ValueError(
                f"{type(self).__name__} requires an image source, got "
                f"{self.source.artifact_type.value}:{self.source.name}."
            )

    def measurement_subject(self) -> "MeasurementSubject":
        from openhcs.core.runtime_measurements import (
            MeasurementScope,
            MeasurementSubject,
        )

        return MeasurementSubject(MeasurementScope.IMAGE, self.source.name)


@dataclass(frozen=True)
class ArtifactMeasurementSubjectRelation(ArtifactSpecRelation):
    """Artifact-scoped measurements are owned by the measurement output itself."""

    relation_key: ClassVar[str] = "artifact_measurement_subject"
    source: ArtifactSpecRef | None = None

    def __post_init__(self) -> None:
        if self.source is None:
            return
        super().__post_init__()
        if self.source.artifact_type is not MeasurementsArtifactType:
            raise ValueError(
                f"{type(self).__name__} requires a measurements source, got "
                f"{self.source.artifact_type.value}:{self.source.name}."
            )

    def bind_target_spec(self, spec: "ArtifactSpec") -> "ArtifactSpecRelation":
        """Bind this self-subject relation to its exact target output."""

        target_ref = spec.ref()
        if self.source is None:
            return replace(self, source=target_ref)
        if self.source != target_ref:
            raise ValueError(
                f"{type(self).__name__} must reference its target output "
                f"{target_ref!r}, got {self.source!r}."
            )
        return self

    def measurement_subject(self) -> "MeasurementSubject":
        from openhcs.core.runtime_measurements import (
            MeasurementScope,
            MeasurementSubject,
        )

        return MeasurementSubject(MeasurementScope.ARTIFACT)

    def dependency_refs(self) -> tuple[ArtifactSpecRef, ...]:
        """Self-subject declaration does not introduce an external dependency."""

        return ()


@dataclass(frozen=True)
class ArtifactSpec:
    """Artifact term whose enclosing declaration may bind its plan role."""

    name: str
    artifact_type: type[ArtifactType]
    parameter_name: str | None = field(default=None, compare=False)
    materialization: ArtifactMaterializationPayload | None = None
    required: bool = True
    sidecar_role: ArtifactSidecarRole | None = None
    relations: tuple[ArtifactSpecRelation, ...] = ()
    plan_type: type["ArtifactPlan"] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.parameter_name is not None and not self.parameter_name:
            raise ValueError("ArtifactSpec.parameter_name cannot be empty.")
        object.__setattr__(
            self,
            "artifact_type",
            _require_registered_artifact_type(
                self.artifact_type,
                "ArtifactSpec.artifact_type",
            ),
        )
        for relation in self.relations:
            if not isinstance(relation, ArtifactSpecRelation):
                raise TypeError(
                    "ArtifactSpec.relations must contain ArtifactSpecRelation "
                    f"values, got {type(relation).__name__}."
                )
        if self.plan_type is None:
            return
        object.__setattr__(
            self,
            "plan_type",
            _require_registered_artifact_plan_type(
                self.plan_type,
                "ArtifactSpec.plan_type",
            ),
        )
        object.__setattr__(
            self,
            "relations",
            tuple(
                dict.fromkeys(
                    relation.bind_target_spec(self) for relation in self.relations
                )
            ),
        )
        for relation in self.relations:
            relation.require_target_spec(self)

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.artifact_type,
                _artifact_spec_hash_value(self.materialization),
                self.required,
                self.sidecar_role,
                self.relations,
                self.plan_type,
            )
        )

    def require_plan_type(self) -> type["ArtifactPlan"]:
        """Return the exact role after its declaration owner has bound it."""

        if self.plan_type is None:
            raise ValueError(
                f"ArtifactSpec {self.name!r} has no plan role outside an "
                "artifact input/output declaration."
            )
        return self.plan_type

    def ref(self) -> ArtifactSpecRef:
        """Return the scope-free identity for this declaration."""
        return ArtifactSpecRef(
            plan_type=self.require_plan_type(),
            artifact_type=self.artifact_type,
            name=self.name,
        )

    @property
    def participates_in_main_flow(self) -> bool:
        """Return whether this declaration is part of the canonical image flow."""
        return self.artifact_type.publishes_to_main_flow(self.sidecar_role)

    def with_group_scope_relation(
        self,
        relation: ArtifactGroupScopeSourceRelation,
    ) -> Self:
        """Replace this declaration's group-scope authority."""

        return replace(
            self,
            relations=(
                *(
                    existing
                    for existing in self.relations
                    if existing.group_scope_source() is None
                ),
                relation,
            ),
        )

    @classmethod
    def input(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
        **kwargs,
    ) -> "ArtifactSpec":
        return cls(
            name=name,
            plan_type=ArtifactInputPlan,
            artifact_type=ArtifactType.coerce(artifact_type),
            **kwargs,
        )

    @classmethod
    def output(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
        **kwargs,
    ) -> "ArtifactSpec":
        return cls(
            name=name,
            plan_type=ArtifactOutputPlan,
            artifact_type=ArtifactType.coerce(artifact_type),
            **kwargs,
        )

    def for_plan_type(self, plan_type: type["ArtifactPlan"]) -> "ArtifactSpec":
        """Return this declaration with the same payload term under another role."""
        target_plan_type = _coerce_artifact_plan_type(plan_type)
        return replace(
            self,
            plan_type=target_plan_type,
            parameter_name=(
                self.parameter_name if target_plan_type is ArtifactInputPlan else None
            ),
            relations=tuple(
                projected
                for relation in self.relations
                for projected in (relation.for_plan_type(target_plan_type),)
                if projected is not None
            ),
        )

    @classmethod
    def output_inheriting_group_scope(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
        source: ArtifactSpecRef | ArtifactSpec,
        **kwargs,
    ) -> ArtifactSpec:
        """Declare an output artifact whose group scope follows a source artifact."""
        source_ref = source.ref() if isinstance(source, ArtifactSpec) else source
        if not isinstance(source_ref, ArtifactSpecRef):
            raise TypeError(
                "ArtifactSpec.output_inheriting_group_scope source must be an "
                f"ArtifactSpec or ArtifactSpecRef, got {type(source).__name__}."
            )
        relations = tuple(kwargs.pop("relations", ()))
        return cls.output(
            name,
            artifact_type,
            relations=(
                *relations,
                GroupLineageSourceRelation(source=source_ref),
            ),
            **kwargs,
        )

    @classmethod
    def output_preserving_source_stack_scope(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
        source: ArtifactSpecRef | ArtifactSpec,
        **kwargs,
    ) -> ArtifactSpec:
        """Declare an output artifact that remains compatible with the source stack."""
        source_ref = source.ref() if isinstance(source, ArtifactSpec) else source
        if not isinstance(source_ref, ArtifactSpecRef):
            raise TypeError(
                "ArtifactSpec.output_preserving_source_stack_scope source must be "
                f"an ArtifactSpec or ArtifactSpecRef, got {type(source).__name__}."
            )
        relations = tuple(kwargs.pop("relations", ()))
        return cls.output(
            name,
            artifact_type,
            relations=(
                *relations,
                SourceStackLineageSourceRelation(source=source_ref),
            ),
            **kwargs,
        )

    def materialization_uses_source_identity_filename(self) -> bool:
        """Return whether this spec's materialized files require source identity."""
        if self.materialization is None:
            return False
        return self.materialization.uses_source_identity_filename()

    def group_scope_sources(self) -> tuple[ArtifactSpecRef, ...]:
        """Return the declared sources that own this artifact's group scope."""

        return tuple(
            dict.fromkeys(
                source
                for relation in self.relations
                for source in (relation.group_scope_source(),)
                if source is not None
            )
        )

    def source_context_sources(self) -> tuple[ArtifactSpecRef, ...]:
        """Return declared sources that carry this artifact's runtime context."""

        if not self.artifact_type.carries_source_image_context:
            return ()
        return tuple(
            dict.fromkeys(
                source
                for relation in self.relations
                for source in (relation.source_context_source(),)
                if source is not None
            )
        )

    def preserves_source_stack_scope(self) -> bool:
        """Return whether this output retains a declared source stack axis."""

        return bool(self.source_stack_scope_sources())

    def source_stack_scope_sources(self) -> tuple[ArtifactSpecRef, ...]:
        """Return exact sources whose complete runtime stacks are preserved."""

        return tuple(
            dict.fromkeys(
                source
                for relation in self.relations
                for source in (relation.source_stack_scope_source(),)
                if source is not None
            )
        )

    def source_stack_scope_identity(self) -> ArtifactSpecRef:
        """Return the single input-role identity owning this artifact's stack."""

        sources = self.source_stack_scope_sources()
        if len(sources) > 1:
            raise ValueError(
                f"Artifact {self.ref()!r} preserves multiple source stacks "
                f"{sources!r}; no singular stack identity exists."
            )
        return (sources[0] if sources else self.ref()).for_plan_type(ArtifactInputPlan)

    def stack_broadcast_sources(self) -> tuple[ArtifactSpecRef, ...]:
        """Return exact input-stack owners that may broadcast this artifact."""

        return tuple(
            dict.fromkeys(
                source
                for relation in self.relations
                for source in (relation.stack_broadcast_source(),)
                if source is not None
            )
        )

    def dependency_refs(self) -> tuple[ArtifactSpecRef, ...]:
        """Return exact artifacts required by this declaration's relations."""

        return tuple(
            dict.fromkeys(
                dependency
                for relation in self.relations
                for dependency in relation.dependency_refs()
            )
        )


@dataclass(frozen=True, slots=True)
class ArtifactSpecAccumulator:
    """Ordered artifact-spec merge authority for one producer/consumer role."""

    role: str
    specs: OrderedDict[ArtifactSpecRef, ArtifactSpec]

    @classmethod
    def empty(cls, role: str) -> "ArtifactSpecAccumulator":
        """Create an empty ordered accumulator for an artifact role."""
        return cls(role=role, specs=OrderedDict())

    def add(self, incoming: ArtifactSpec) -> None:
        """Merge an incoming spec into this accumulator."""
        ref = incoming.ref()
        if ref not in self.specs:
            self.specs[ref] = incoming
            return
        self.specs[ref] = self.merge_existing(
            existing=self.specs[ref],
            incoming=incoming,
        )

    def merge_existing(
        self,
        *,
        existing: ArtifactSpec,
        incoming: ArtifactSpec,
    ) -> ArtifactSpec:
        """Merge two declarations for the same exact artifact reference."""
        if existing.ref() != incoming.ref():
            raise ValueError(
                f"Cannot merge distinct {self.role} artifact refs: "
                f"{existing.ref()!r} and {incoming.ref()!r}."
            )
        if (
            existing.materialization is not None
            and incoming.materialization is not None
            and existing.materialization != incoming.materialization
        ):
            raise ValueError(
                f"Conflicting {self.role} artifact materialization for "
                f"'{incoming.name}'."
            )

        materialization = (
            existing.materialization
            if existing.materialization is not None
            else incoming.materialization
        )
        if (
            existing.sidecar_role is not None
            and incoming.sidecar_role is not None
            and existing.sidecar_role is not incoming.sidecar_role
        ):
            raise ValueError(
                f"Conflicting {self.role} artifact sidecar role for "
                f"'{incoming.name}'."
            )
        sidecar_role = (
            existing.sidecar_role
            if existing.sidecar_role is not None
            else incoming.sidecar_role
        )
        relations = tuple(dict.fromkeys((*existing.relations, *incoming.relations)))
        return replace(
            existing,
            materialization=materialization,
            required=existing.required or incoming.required,
            sidecar_role=sidecar_role,
            relations=relations,
        )


def _artifact_spec_hash_value(value) -> Hashable:
    """Project rich artifact metadata into a hashable equality-compatible value."""
    if is_dataclass(value):
        return (type(value), _artifact_spec_hash_value(astuple(value)))
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    (
                        _artifact_spec_hash_value(key),
                        _artifact_spec_hash_value(item),
                    )
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_artifact_spec_hash_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_artifact_spec_hash_value(item) for item in value)
    if isinstance(value, Hashable):
        return value
    raise TypeError(
        "ArtifactSpec materialization metadata contains unsupported "
        f"unhashable value {type(value).__name__}."
    )


@dataclass(frozen=True, slots=True)
class ArtifactSpecCollection(Sequence[ArtifactSpec]):
    """Ordered query surface over declared artifact specs."""

    specs: tuple[ArtifactSpec, ...]

    def __init__(self, specs: Iterable[ArtifactSpec]):
        normalized = tuple(specs)
        for spec in normalized:
            if not isinstance(spec, ArtifactSpec):
                raise TypeError(
                    "ArtifactSpecCollection requires ArtifactSpec values, "
                    f"got {type(spec).__name__}."
                )
        object.__setattr__(self, "specs", normalized)

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self):
        return iter(self.specs)

    def __getitem__(self, index):
        return self.specs[index]

    def of_artifact_type(
        self,
        artifact_type: ArtifactTypeValue,
    ) -> tuple[ArtifactSpec, ...]:
        """Return specs with the requested artifact type, preserving order."""
        resolved_artifact_type = ArtifactType.coerce(artifact_type)
        return tuple(
            spec for spec in self.specs if spec.artifact_type is resolved_artifact_type
        )

    def for_plan_type(
        self,
        plan_type: type["ArtifactPlan"],
    ) -> "ArtifactSpecCollection":
        """Return specs with the requested artifact plan role."""
        resolved_plan_type = _coerce_artifact_plan_type(plan_type)
        return ArtifactSpecCollection(
            spec for spec in self.specs if spec.plan_type is resolved_plan_type
        )

    def names_for_plan_type(
        self,
        plan_type: type["ArtifactPlan"],
    ) -> tuple[str, ...]:
        """Return names for specs with the requested artifact plan role."""
        return self.for_plan_type(plan_type).names()

    def for_artifact_type(
        self,
        artifact_type: ArtifactTypeValue,
    ) -> "ArtifactSpecCollection":
        """Return specs with the requested artifact type."""
        return ArtifactSpecCollection(self.of_artifact_type(artifact_type))

    def names(self) -> tuple[str, ...]:
        """Return artifact names in collection order."""
        return tuple(spec.name for spec in self.specs)

    def name_set(self) -> frozenset[str]:
        """Return artifact names as a set."""
        return frozenset(self.names())

    def names_of_artifact_type(
        self,
        artifact_type: ArtifactTypeValue,
    ) -> tuple[str, ...]:
        """Return names for specs with the requested artifact type."""
        return ArtifactSpecCollection(self.of_artifact_type(artifact_type)).names()

    def name_set_of_artifact_type(
        self,
        artifact_type: ArtifactTypeValue,
    ) -> frozenset[str]:
        """Return names for specs with the requested artifact type as a set."""
        return frozenset(self.names_of_artifact_type(artifact_type))

    def by_name(self, name: str) -> ArtifactSpec | None:
        """Return the first spec with a matching artifact name."""
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None

    def by_name_and_artifact_type(
        self,
        name: str,
        artifact_type: ArtifactTypeValue,
    ) -> ArtifactSpec | None:
        """Return the one semantic artifact matching name and type."""
        resolved_artifact_type = ArtifactType.coerce(artifact_type)
        matches = tuple(
            spec
            for spec in self.specs
            if spec.name == name and spec.artifact_type is resolved_artifact_type
        )
        if not matches:
            return None
        first = matches[0]
        if any(spec != first for spec in matches[1:]):
            raise ValueError(
                f"Conflicting active {resolved_artifact_type.require_value()} "
                f"artifact declarations are named {name!r}: {matches!r}."
            )
        return first

    def require_by_name_and_artifact_type(
        self,
        name: str,
        artifact_type: ArtifactTypeValue,
    ) -> ArtifactSpec:
        """Return the declared artifact with this identity or fail."""

        resolved_artifact_type = ArtifactType.coerce(artifact_type)
        spec = self.by_name_and_artifact_type(name, resolved_artifact_type)
        if spec is None:
            raise ValueError(
                f"No {resolved_artifact_type.require_value()} artifact named "
                f"{name!r} is declared."
            )
        return spec

    def ref_set(self) -> frozenset[ArtifactSpecRef]:
        """Return full declared artifact references."""
        return frozenset(spec.ref() for spec in self.specs)

    def select_refs(
        self,
        refs: Iterable[ArtifactSpecRef],
    ) -> "ArtifactSpecCollection":
        """Select exact declared identities while preserving declaration order."""

        selected_refs = tuple(refs)
        if any(not isinstance(ref, ArtifactSpecRef) for ref in selected_refs):
            raise TypeError(
                "ArtifactSpecCollection.select_refs requires ArtifactSpecRef values."
            )
        if len(frozenset(selected_refs)) != len(selected_refs):
            raise ValueError(
                "ArtifactSpecCollection.select_refs does not accept duplicate "
                f"identities: {selected_refs!r}."
            )
        selected_ref_set = frozenset(selected_refs)
        selected = ArtifactSpecCollection(
            spec for spec in self.specs if spec.ref() in selected_ref_set
        )
        resolved_refs = tuple(spec.ref() for spec in selected)
        if len(resolved_refs) != len(selected_refs):
            missing = tuple(ref for ref in selected_refs if ref not in resolved_refs)
            duplicate_declarations = tuple(
                ref
                for ref in selected_refs
                if resolved_refs.count(ref) > 1
            )
            raise ValueError(
                "Selected artifact identities must resolve to one exact declared "
                "occurrence: "
                f"missing={missing!r}, duplicate_declarations="
                f"{duplicate_declarations!r}."
            )
        return selected

    def rebind(self, replacements: Iterable[ArtifactSpec]) -> "ArtifactSpecCollection":
        """Replace active artifact bindings while preserving declaration order."""

        replacement_specs = tuple(replacements)
        replacement_refs = tuple(
            spec.ref().for_plan_type(ArtifactInputPlan) for spec in replacement_specs
        )
        if len(set(replacement_refs)) != len(replacement_refs):
            raise ValueError(
                "Artifact replacements contain duplicate active identities: "
                f"{replacement_refs!r}."
            )
        replaced = frozenset(replacement_refs)
        return ArtifactSpecCollection(
            (
                *(
                    spec
                    for spec in self.specs
                    if spec.ref().for_plan_type(ArtifactInputPlan) not in replaced
                ),
                *replacement_specs,
            )
        )

    def by_ref(self, ref: ArtifactSpecRef) -> ArtifactSpec | None:
        """Return one spec by full artifact reference."""
        matches = tuple(spec for spec in self.specs if spec.ref() == ref)
        if not matches:
            return None
        first = matches[0]
        if any(spec != first for spec in matches[1:]):
            raise ValueError(
                f"Conflicting artifact declarations for exact ref {ref!r}: {matches!r}."
            )
        return first

    def relation_refs(
        self,
        relation_type: type[ArtifactSpecRelation],
    ) -> tuple[tuple[ArtifactSpec, ArtifactSpecRelation], ...]:
        """Return specs and relation tags of the requested relation family."""
        if not isinstance(relation_type, type) or not issubclass(
            relation_type,
            ArtifactSpecRelation,
        ):
            raise TypeError(
                "relation_type must be an ArtifactSpecRelation type, "
                f"got {type(relation_type).__name__}."
            )
        return tuple(
            (spec, relation)
            for spec in self.specs
            for relation in spec.relations
            if isinstance(relation, relation_type)
        )

    def validate_registered_relation_refs(
        self,
        *,
        owner_name: str,
        relation_specs: Iterable[ArtifactSpec] | None = None,
    ) -> None:
        """Validate relations owned by selected specs against this collection."""

        owners = self.specs if relation_specs is None else tuple(relation_specs)
        for spec in owners:
            if not isinstance(spec, ArtifactSpec):
                raise TypeError(
                    "Artifact relation owners must be ArtifactSpec values, got "
                    f"{type(spec).__name__}."
                )
        refs = self.ref_set()
        unknown = tuple(
            relation.source
            for spec in owners
            for relation in spec.relations
            if relation.source not in refs
        )
        if unknown:
            raise ValueError(
                f"{owner_name} declares artifact relation references to unknown "
                f"artifact specs: {unknown!r}."
            )

    def unique(
        self, *, conflict_context: str = "artifact spec"
    ) -> tuple[ArtifactSpec, ...]:
        """Return specs de-duplicated by artifact identity, failing on conflicts."""
        unique_specs: dict[ArtifactSpecRef, ArtifactSpec] = {}
        for spec in self.specs:
            key = spec.ref()
            if key in unique_specs and (
                unique_specs[key] != spec
                or unique_specs[key].parameter_name != spec.parameter_name
            ):
                raise ValueError(
                    f"Conflicting {conflict_context} declarations for "
                    f"{spec.require_plan_type().plan_role}:"
                    f"{spec.artifact_type.value}:{spec.name}."
                )
            unique_specs[key] = spec
        return tuple(unique_specs.values())

    def select_declared_occurrences(
        self,
        selected_specs: Iterable[ArtifactSpec],
    ) -> "ArtifactSpecCollection":
        """Select exact occurrences while preserving declaration order."""

        remaining = list(selected_specs)
        for spec in remaining:
            if not isinstance(spec, ArtifactSpec):
                raise TypeError(
                    "ArtifactSpecCollection.select_declared_occurrences requires "
                    f"ArtifactSpec values, got {type(spec).__name__}."
                )
        selected: list[ArtifactSpec] = []
        for declared in self.specs:
            matching_index = next(
                (
                    index
                    for index, candidate in enumerate(remaining)
                    if candidate.ref() == declared.ref()
                ),
                None,
            )
            if matching_index is None:
                continue
            candidate = remaining.pop(matching_index)
            if (
                candidate != declared
                or candidate.parameter_name != declared.parameter_name
            ):
                raise ValueError(
                    "Selected artifact spec drifts from its declared occurrence: "
                    f"selected {candidate!r}, declared {declared!r}."
                )
            selected.append(declared)
        if remaining:
            raise ValueError(
                "Selected artifact specs are not declared with sufficient "
                f"occurrence cardinality: {tuple(remaining)!r}."
            )
        return ArtifactSpecCollection(selected)


@dataclass(frozen=True)
class ArtifactPlan(ABC, metaclass=AutoRegisterMeta):
    """Compiled storage plan shared by produced and consumed artifacts."""

    __registry_key__ = "plan_role"
    __skip_if_no_key__ = True

    plan_role: ClassVar[str | None] = None

    name: str
    path: str
    artifact_type: type[ArtifactType] = SpecialArtifactType
    group_keys: tuple[str | None, ...] = (None,)
    group_component: AllComponents | None = None
    variable_components: tuple[AllComponents, ...] = ()
    component_domains: tuple[ComponentGroupScope, ...] = ()
    paths_by_group: Mapping[str | None, str] | None = None
    sidecar_role: ArtifactSidecarRole | None = None

    _missing_group_uses_default_path: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_type",
            ArtifactType.coerce(self.artifact_type),
        )
        object.__setattr__(
            self,
            "variable_components",
            ComponentSet.coerce(self.variable_components).as_tuple(),
        )
        normalized_domains: dict[AllComponents, ComponentGroupScope] = {}
        for domain in self.component_domains:
            if not isinstance(domain, ComponentGroupScope):
                raise TypeError(
                    "ArtifactPlan.component_domains must contain "
                    "ComponentGroupScope values."
                )
            if domain.component is None:
                raise ValueError("Artifact plan component domains cannot be ungrouped.")
            canonical_domain = ComponentGroupScope.from_raw(
                domain.keys,
                component=domain.component,
            )
            existing = normalized_domains.get(domain.component)
            if existing is not None and existing != canonical_domain:
                raise ValueError(
                    f"Artifact plan {self.name!r} declares conflicting domains for "
                    f"component {domain.component.value!r}: {existing!r} and "
                    f"{canonical_domain!r}."
                )
            normalized_domains[domain.component] = canonical_domain
        object.__setattr__(
            self,
            "component_domains",
            tuple(normalized_domains.values()),
        )
        if self.group_component in self.variable_components:
            raise ValueError(
                f"Artifact plan {self.name!r} cannot group by "
                f"{self.group_component.value!r} while also retaining it as a "
                "variable component."
            )
        self.group_scope()

    def group_scope(self) -> ComponentGroupScope:
        """Return this plan's typed component-group identity."""

        return ComponentGroupScope.from_raw(
            self.group_keys or (None,),
            component=self.group_component,
        )

    @property
    def participates_in_main_flow(self) -> bool:
        """Return whether this compiled artifact publishes canonical image flow."""

        return self.artifact_type.publishes_to_main_flow(self.sidecar_role)

    def component_domain(
        self,
        component: AllComponents,
    ) -> ComponentGroupScope | None:
        """Return the compiled domain inherited for one component axis."""

        return next(
            (
                domain
                for domain in self.component_domains
                if domain.component is component
            ),
            None,
        )

    @property
    def single_group_key(self) -> str | None:
        group_keys = self.group_keys or (None,)
        if len(group_keys) == 1:
            return group_keys[0]
        return None

    @property
    def has_dynamic_group_scope(self) -> bool:
        """Return whether concrete runtime groups are discovered during execution."""
        return self.group_scope().is_dynamic

    def require_single_group_key(self) -> str | None:
        """Return the only artifact group key, failing for ambiguous groups."""
        group_keys = self.group_keys or (None,)
        if len(group_keys) == 1:
            return group_keys[0]
        raise RuntimeError(
            f"Artifact plan '{self.name}' requires one group key, got {group_keys!r}."
        )

    def ref(self) -> ArtifactSpecRef:
        """Return this compiled plan's scope-free declaration identity."""
        return ArtifactSpecRef(
            plan_type=type(self),
            artifact_type=self.artifact_type,
            name=self.name,
        )

    @classmethod
    def require_exact_map(
        cls,
        plans: Mapping[object, object],
        *,
        boundary: str,
    ) -> None:
        """Require a caller-owned map to preserve each plan's exact identity."""

        for plan_ref, artifact_plan in plans.items():
            if not isinstance(plan_ref, ArtifactSpecRef):
                raise TypeError(
                    f"{boundary} maps require ArtifactSpecRef keys, got "
                    f"{type(plan_ref).__name__}."
                )
            if not isinstance(artifact_plan, cls):
                raise TypeError(
                    f"{boundary} maps require {cls.__name__} values, got "
                    f"{type(artifact_plan).__name__} for {plan_ref!r}."
                )
            if plan_ref != artifact_plan.ref():
                raise ValueError(
                    f"{boundary} key {plan_ref!r} conflicts with plan ref "
                    f"{artifact_plan.ref()!r}."
                )

    def _path_for_group(self, group_key: str | None) -> str | None:
        if not self.paths_by_group:
            return self.path
        if group_key in self.paths_by_group:
            return self.paths_by_group[group_key]
        if (
            group_key is not None
            and self.group_component is not None
            and None in self.paths_by_group
        ):
            return grouped_artifact_path(self.paths_by_group[None], group_key)
        if None in self.paths_by_group:
            return self.paths_by_group[None]
        if self._missing_group_uses_default_path:
            return self.path
        return None

    def _plan_for_group(self, group_key: str | None) -> Self | None:
        group_path = self._path_for_group(group_key)
        if group_path is None:
            return None
        return cast(
            Self,
            replace(
                self,
                path=group_path,
                group_keys=(group_key,),
                paths_by_group={group_key: group_path},
            ),
        )


@dataclass(frozen=True)
class ArtifactOutputPlan(ArtifactPlan):
    """Compiled storage plan for one produced artifact."""

    plan_role: ClassVar[str] = "output"
    _missing_group_uses_default_path: ClassVar[bool] = True

    materialization: ArtifactMaterializationPayload | None = None
    relations: tuple[ArtifactSpecRelation, ...] = ()
    producer_step_index: int | str | None = None
    producer_step_scope_id: str | None = None
    producer_step_name: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.source_context_source()

    def normalize_payload(self, value: object, *, axis_id: str) -> object:
        """Normalize a produced payload through this compiled storage scope."""

        from openhcs.core.runtime_artifact_values import RuntimeValue
        from openhcs.core.runtime_slice_alignment import (
            RuntimeSliceAlignedValues,
            RuntimeSliceAlignedValueSet,
        )

        if isinstance(value, RuntimeSliceAlignedValueSet):
            value = RuntimeSliceAlignedValues(
                tuple(
                    item.data if isinstance(item, RuntimeValue) else item
                    for item in (
                        value.value_for_slice(index)
                        for index in range(value.slice_count)
                    )
                )
            )
        normalized = self.artifact_type.normalize_runtime_payload(self.name, value)
        if isinstance(normalized, RuntimeSliceAlignedValueSet):
            owned = self.artifact_type.normalize_slice_aligned_value(
                normalized,
                self,
                axis_id=axis_id,
            )
            normalized = (
                owned
                if owned is not None
                else RuntimeSliceAlignedValues(
                    tuple(
                        item.data if isinstance(item, RuntimeValue) else item
                        for item in (
                            normalized.value_for_slice(index)
                            for index in range(normalized.slice_count)
                        )
                    )
                )
            )
        if self.group_component is not None and not self.variable_components:
            normalized = self.artifact_type.normalize_group_scoped_payload(
                self,
                normalized,
            )
        return normalized

    def materialization_uses_source_identity_filename(self) -> bool:
        """Return whether this output's materialized files require source identity."""
        if self.materialization is None:
            return False
        return self.materialization.uses_source_identity_filename()

    def materialization_payload(self, value: "RuntimeValue") -> object:
        """Return this output's payload under its declared filename source context."""

        payload = value.materialization_payload()
        materialization_source = self.materialization_source()
        if (
            materialization_source is None
            or materialization_source == self.source_context_source()
        ):
            return payload
        return self.materialization_metadata(value).attach_to(payload)

    def materialization_metadata(
        self,
        value: "RuntimeValue",
    ) -> "ImagePayloadMetadata":
        """Return payload metadata with declared materialization-source provenance."""

        from openhcs.core.runtime_image_values import image_payload_metadata

        payload_metadata = image_payload_metadata(value.materialization_payload())
        materialization_source = self.materialization_source()
        if (
            materialization_source is not None
            and materialization_source != self.source_context_source()
        ):
            source_metadata = value.materialization_source_metadata
            if source_metadata is None:
                raise ValueError(
                    f"Artifact output {self.ref()!r} declares independent "
                    f"materialization source {materialization_source!r}, but its "
                    "runtime value carries no materialization-source metadata."
                )
            return payload_metadata.with_source_provenance(
                source_metadata.source_provenance
            )
        return payload_metadata

    def materialization_source(self) -> ArtifactSpecRef | None:
        """Return the sole declared materialization identity source, if present."""

        sources = tuple(
            dict.fromkeys(
                source
                for relation in self.relations
                for source in (relation.materialization_source(),)
                if source is not None
            )
        )
        if len(sources) > 1:
            raise ValueError(
                f"Artifact output {self.ref()!r} declares multiple materialization "
                f"identity sources: {sources!r}."
            )
        return sources[0] if sources else None

    def group_scope_sources(self) -> tuple[ArtifactSpecRef, ...]:
        """Return the compiled sources that own this output's group scope."""

        return tuple(
            dict.fromkeys(
                source
                for relation in self.relations
                for source in (relation.group_scope_source(),)
                if source is not None
            )
        )

    def source_context_source(self) -> ArtifactSpecRef | None:
        """Return the sole declared runtime-context source for this output."""

        if not self.artifact_type.carries_source_image_context:
            return None
        sources = tuple(
            dict.fromkeys(
                source
                for relation in self.relations
                for source in (relation.source_context_source(),)
                if source is not None
            )
        )
        if len(sources) > 1:
            raise ValueError(
                f"Artifact output {self.ref()!r} declares multiple runtime-context "
                f"sources: {sources!r}."
            )
        return sources[0] if sources else None

    def measurement_subject(self) -> "MeasurementSubject | None":
        """Return the sole measurement subject declared by this output."""

        subjects = tuple(
            dict.fromkeys(
                subject
                for relation in self.relations
                for subject in (relation.measurement_subject(),)
                if subject is not None
            )
        )
        if len(subjects) > 1:
            raise ValueError(
                f"Artifact output {self.ref()!r} declares multiple measurement "
                f"subjects: {subjects!r}."
            )
        return subjects[0] if subjects else None

    def object_subject_binding(self) -> ObjectArtifactSubjectBinding | None:
        """Return the sole object-subject binding declared by this output."""

        bindings = tuple(
            dict.fromkeys(
                binding
                for relation in self.relations
                for binding in (relation.object_subject_binding(),)
                if binding is not None
            )
        )
        if len(bindings) > 1:
            raise ValueError(
                f"Artifact output {self.ref()!r} declares multiple object-subject "
                f"bindings: {bindings!r}."
            )
        return bindings[0] if bindings else None

    def for_group(self, group_key: str | None) -> "ArtifactOutputPlan":
        """Return a group-specific output plan with the finalized path."""
        plan = self._plan_for_group(group_key)
        if plan is None:
            raise RuntimeError("ArtifactOutputPlan group resolution must be total.")
        return plan

    def for_invocation_group(
        self,
        group_key: str | None,
    ) -> "ArtifactOutputPlan":
        """Resolve storage scope from the invocation or this exact output plan."""

        output_scope = self.group_scope()
        if (
            group_key is None
            and not output_scope.is_ungrouped
            and not output_scope.is_dynamic
        ):
            group_key = output_scope.require_single_static_key()
        return self.for_group(output_scope.resolve_runtime_key(group_key))


@dataclass(frozen=True)
class ArtifactInputPlan(ArtifactPlan):
    """Producer-owned storage plan for one consumed artifact."""

    plan_role: ClassVar[str] = "input"

    source_step_id: int | str | None = None
    source_step_scope_id: str | None = None

    def for_group(self, group_key: str | None) -> "ArtifactInputPlan | None":
        """Return a group-specific input plan, or None if not available."""
        return self._plan_for_group(group_key)

    def producer_group_scope(self) -> ComponentGroupScope:
        """Return the compiler-resolved scope of this named producer."""

        return self.group_scope()

    def composes_producer_groups(
        self,
        consumer_variable_components: ComponentSet,
    ) -> bool:
        """Return whether this consumer reconstructs the producer group axis."""

        if not isinstance(consumer_variable_components, ComponentSet):
            raise TypeError(
                "ArtifactInputPlan.composes_producer_groups requires a ComponentSet."
            )
        return self.producer_group_scope().component in consumer_variable_components

    def retains_producer_stack(
        self,
        consumer_variable_components: ComponentSet,
    ) -> bool:
        """Return whether the producer's third axis remains the consumer stack."""

        if not isinstance(consumer_variable_components, ComponentSet):
            raise TypeError(
                "ArtifactInputPlan.retains_producer_stack requires a ComponentSet."
            )
        return (
            bool(self.variable_components)
            and bool(consumer_variable_components)
            and not self.composes_producer_groups(consumer_variable_components)
        )

    def runtime_variable_components(
        self,
        consumer_variable_components: ComponentSet,
    ) -> ComponentSet:
        """Return axes present after this input is reconstructed for a consumer."""

        if not isinstance(consumer_variable_components, ComponentSet):
            raise TypeError(
                "ArtifactInputPlan.runtime_variable_components requires a ComponentSet."
            )
        if self.composes_producer_groups(consumer_variable_components):
            producer_component = self.producer_group_scope().component
            if producer_component is None:
                raise RuntimeError(
                    "Grouped artifact composition lost its producer component."
                )
            return ComponentSet((producer_component,))
        if self.retains_producer_stack(consumer_variable_components):
            return consumer_variable_components
        return ComponentSet()

    def path_for_runtime_query(self, group_key: str | None) -> str:
        """Return the persisted input path addressed by a runtime query."""
        group_path = self._path_for_group(group_key)
        if group_path is not None:
            return group_path
        return self.path


@dataclass(frozen=True, slots=True)
class ArtifactInputProjectionPlan:
    """Invocation-edge projection of one exact producer-owned artifact input."""

    invocation_scope: ComponentGroupScope
    producer_selection_scope: ComponentGroupScope
    component_scopes: tuple[ComponentGroupScope, ...] = ()
    consumer_variable_components: tuple[AllComponents, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_scope, ComponentGroupScope):
            raise TypeError(
                "ArtifactInputProjectionPlan.invocation_scope must be a "
                "ComponentGroupScope."
            )
        if not isinstance(self.producer_selection_scope, ComponentGroupScope):
            raise TypeError(
                "ArtifactInputProjectionPlan.producer_selection_scope must be a "
                "ComponentGroupScope."
            )
        object.__setattr__(
            self,
            "invocation_scope",
            ComponentGroupScope.from_raw(
                self.invocation_scope.keys,
                component=self.invocation_scope.component,
            ),
        )
        object.__setattr__(
            self,
            "producer_selection_scope",
            ComponentGroupScope.from_raw(
                self.producer_selection_scope.keys,
                component=self.producer_selection_scope.component,
            ),
        )

        normalized_scopes: dict[AllComponents, ComponentGroupScope] = {}
        for scope in self.component_scopes:
            if not isinstance(scope, ComponentGroupScope):
                raise TypeError(
                    "ArtifactInputProjectionPlan.component_scopes must contain "
                    "ComponentGroupScope values."
                )
            if scope.component is None:
                raise ValueError(
                    "ArtifactInputProjectionPlan component coordinates cannot be "
                    "ungrouped."
                )
            canonical_scope = ComponentGroupScope.from_raw(
                scope.keys,
                component=scope.component,
            )
            existing = normalized_scopes.get(scope.component)
            if existing is not None and existing != canonical_scope:
                raise ValueError(
                    "Artifact input projection declares conflicting scopes for "
                    f"component {scope.component.value!r}: {existing!r} and "
                    f"{canonical_scope!r}."
                )
            normalized_scopes[scope.component] = canonical_scope
        object.__setattr__(
            self,
            "component_scopes",
            tuple(normalized_scopes.values()),
        )
        object.__setattr__(
            self,
            "consumer_variable_components",
            ComponentSet.coerce(self.consumer_variable_components).as_tuple(),
        )

    def component_scope(
        self,
        component: AllComponents,
    ) -> ComponentGroupScope | None:
        """Return the exact compiled coordinate for one projected component."""

        return next(
            (scope for scope in self.component_scopes if scope.component is component),
            None,
        )

    def projected_variable_components(
        self,
        storage_plan: ArtifactInputPlan,
    ) -> ComponentSet:
        """Return producer stack axes projected out for this consumer invocation."""

        consumer_components = ComponentSet.coerce(self.consumer_variable_components)
        if storage_plan.retains_producer_stack(consumer_components):
            return ComponentSet()
        return ComponentSet.coerce(storage_plan.variable_components)

    def validate_storage_plan(self, storage_plan: ArtifactInputPlan) -> None:
        """Require this projection to select within its producer storage plan."""

        artifact_ref = storage_plan.ref()
        producer_scope = storage_plan.producer_group_scope()
        selection_scope = self.producer_selection_scope
        if producer_scope.is_ungrouped:
            if not selection_scope.is_ungrouped:
                raise ValueError(
                    f"Ungrouped producer {artifact_ref!r} cannot use grouped "
                    f"selection {selection_scope!r}."
                )
        else:
            if selection_scope.component is not producer_scope.component:
                raise ValueError(
                    f"Artifact input {artifact_ref!r} producer component "
                    f"{producer_scope.component.value!r} does not match selection "
                    f"{selection_scope.component!r}."
                )
            if not selection_scope.is_dynamic:
                if not producer_scope.contains_scope(selection_scope):
                    raise ValueError(
                        f"Artifact input {artifact_ref!r} selection "
                        f"{selection_scope!r} is outside producer scope "
                        f"{producer_scope!r}."
                    )

    def validate_axis_projection(self, storage_plan: ArtifactInputPlan) -> None:
        """Require one exact axis invocation projection or retained stack axis."""

        self.validate_storage_plan(storage_plan)
        artifact_ref = storage_plan.ref()
        producer_scope = storage_plan.producer_group_scope()
        selection_scope = self.producer_selection_scope
        retained_components = ComponentSet.coerce(self.consumer_variable_components)
        if (
            not producer_scope.is_ungrouped
            and producer_scope.component in retained_components
        ):
            if selection_scope != producer_scope:
                raise ValueError(
                    f"Retained producer axis {producer_scope.component.value!r} for "
                    f"{artifact_ref!r} must select the complete producer scope."
                )
        elif not producer_scope.is_ungrouped:
            if selection_scope.is_dynamic:
                if (
                    self.invocation_scope.component is not selection_scope.component
                    or self.invocation_scope.is_ungrouped
                ):
                    raise ValueError(
                        f"Dynamic producer selection {selection_scope!r} for "
                        f"{artifact_ref!r} is not owned by invocation scope "
                        f"{self.invocation_scope!r}."
                    )

        for component in self.projected_variable_components(storage_plan):
            scope = self.component_scope(component)
            if scope is None:
                raise ValueError(
                    f"Artifact input {artifact_ref!r} projects producer stack "
                    f"component {component.value!r} without an exact coordinate."
                )
            if not scope.is_dynamic and len(scope.keys) != 1:
                raise ValueError(
                    f"Artifact input {artifact_ref!r} projection for component "
                    f"{component.value!r} is not a single exact coordinate: {scope!r}."
                )
            if scope.is_dynamic and self.invocation_scope.component is not component:
                raise ValueError(
                    f"Dynamic stack projection {scope!r} for {artifact_ref!r} "
                    f"is not owned by invocation scope {self.invocation_scope!r}."
                )

    def validate_complete_producer_projection(
        self,
        storage_plan: ArtifactInputPlan,
    ) -> None:
        """Require a plate invocation to select the complete producer scope."""

        self.validate_storage_plan(storage_plan)
        artifact_ref = storage_plan.ref()
        producer_scope = storage_plan.producer_group_scope()
        if self.producer_selection_scope != producer_scope:
            raise ValueError(
                f"Plate artifact input {artifact_ref!r} must select complete "
                f"producer scope {producer_scope!r}, got "
                f"{self.producer_selection_scope!r}."
            )


ArtifactSpecRelation.target_plan_type = ArtifactOutputPlan
InputGroupLineageSourceRelation.target_plan_type = ArtifactInputPlan
InputStackBroadcastSourceRelation.target_plan_type = ArtifactInputPlan
InputStackBroadcastSourceRelation.target_artifact_type = ImageArtifactType
MaterializationSourceIdentityRelation.target_artifact_type = ImageArtifactType
ObjectMeasurementSubjectRelation.target_artifact_type = MeasurementsArtifactType
ImageMeasurementSubjectRelation.target_artifact_type = MeasurementsArtifactType
ArtifactMeasurementSubjectRelation.target_artifact_type = MeasurementsArtifactType


def grouped_artifact_path(base_path: str, group_key: str) -> str:
    """Return the existing grouped artifact path form for a runtime group."""
    path = Path(base_path)
    filename = path.name
    if "_" not in filename:
        return str(path.with_name(f"{path.stem}_w{group_key}{path.suffix}"))
    axis_id, rest = filename.split("_", 1)
    return str(path.parent / f"{axis_id}_w{group_key}_{rest}")


@dataclass(frozen=True)
class NoMainFlowOutput:
    """Nominal return value for invocations that record artifacts only."""
