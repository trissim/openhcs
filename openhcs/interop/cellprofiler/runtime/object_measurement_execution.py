"""Object-measurement execution domain and label argument policies."""

from __future__ import annotations
from abc import abstractmethod
from typing import ClassVar
from metaclass_registry import AutoRegisterMeta
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    StrategyLabelRegistryMixin,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.runtime_object_label_domains import ObjectLabelDomainScope
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
class CellProfilerObjectMeasurementExecutionPolicy(
    EnumKeyedStrategyMixin,
    StrategyLabelRegistryMixin,
    metaclass=AutoRegisterMeta,
):
    """Apply the callable's declared object-measurement execution semantics."""

    __enum_member_attr__ = "execution_mode"
    execution_mode: ClassVar[ObjectLabelInputExecutionMode]

    @abstractmethod
    def semantic_label_payload(
        self,
        source_projected_payload: ObjectLabelValue,
        completion_payload: ObjectLabelValue,
    ) -> ObjectLabelValue:
        """Return the label payload that represents this execution domain."""

    @abstractmethod
    def image_execution_mode(
        self,
        labels: ObjectLabelValue,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        """Return the image execution mode for the semantic label payload."""


class SliceAlignedObjectMeasurementExecutionPolicy(
    CellProfilerObjectMeasurementExecutionPolicy
):
    """Slice-aligned measurement functions consume the dense execution plane."""

    execution_mode = ObjectLabelInputExecutionMode.SLICE_ALIGNED

    def semantic_label_payload(
        self,
        source_projected_payload: ObjectLabelValue,
        completion_payload: ObjectLabelValue,
    ) -> ObjectLabelValue:
        del source_projected_payload
        projection = completion_payload.declared_plane_projection()
        if projection is not None and projection.axis_size == 1:
            return RuntimeSliceProjection.value_for_singleton_slice(
                completion_payload,
                source_description="Slice-aligned object-label input",
            )
        return completion_payload

    def image_execution_mode(
        self,
        labels: ObjectLabelValue,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        if labels.object_label_domain().scope is ObjectLabelDomainScope.PAYLOAD:
            if (
                default is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
                and runtime_slice_count == 1
            ):
                return default
            return ImagePayloadExecutionMode.FULL_STACK
        if default is ImagePayloadExecutionMode.FULL_STACK:
            projection = labels.declared_plane_projection()
            raise ValueError(
                "Slice-aligned object measurement cannot execute a full-stack image "
                "with an unprojected object-label plane domain: "
                f"{projection!r}."
            )
        return default


class FullStackObjectMeasurementExecutionPolicy(
    CellProfilerObjectMeasurementExecutionPolicy
):
    """Full-stack measurement functions consume labels with semantic domains."""

    execution_mode = ObjectLabelInputExecutionMode.FULL_STACK

    def semantic_label_payload(
        self,
        source_projected_payload: ObjectLabelValue,
        completion_payload: ObjectLabelValue,
    ) -> ObjectLabelValue:
        del completion_payload
        return source_projected_payload

    def image_execution_mode(
        self,
        labels: ObjectLabelValue,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        del default, runtime_slice_count
        return ImagePayloadExecutionMode.FULL_STACK
