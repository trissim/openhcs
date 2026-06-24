"""Pure-2D CellProfiler output aggregation authorities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.aligned_image_payload import (
    ImageArrayShapeSemantics,
    ImagePayloadSliceProjector,
)
from openhcs.core.image_shapes import is_color_image_slice, is_color_image_stack
from openhcs.core.image_stack_layout import ImageStackLayout, ImageStackLayoutUnstackRequest
from openhcs.core.memory import convert_memory, detect_memory_type
from openhcs.core.registry_strategies import GeneratedLeafClassSpec
from openhcs.core.runtime_semantics import ParentChildRelationshipPayload, RuntimePlaneAxis
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_values import (
    ImageMetadataPayload,
    ImagePayloadMetadataCompositionRequest,
    MaskedImagePayload,
    ObjectLabelPayload,
    ObjectLabelPure2DSliceAggregator,
    ObjectLabelSet,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.runtime.image_payload_collapse import (
    SINGLETON_STACK_OUTPUT_COLLAPSE,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerClassAttributes,
    CellProfilerRuntimeType,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    CellProfilerRuntimeValueSequence,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    Pure2DAuxiliaryOutputAggregator,
)

class CellProfilerPure2DOutputAggregator(ABC, metaclass=AutoRegisterMeta):
    """Aggregate one per-slice CellProfiler output position."""

    __registry_key__ = "output_type"
    __registry__: ClassVar[dict[CellProfilerRuntimeValue, type["CellProfilerPure2DOutputAggregator"]]] = {}
    output_type: ClassVar[type[CellProfilerRuntimeValue] | None] = None

    @classmethod
    def aggregate(
        cls,
        slice_outputs: CellProfilerRuntimeValueSequence,
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis = RuntimePlaneAxis.RUNTIME_SLICE,
    ) -> CellProfilerRuntimeValue:
        request = CellProfilerPure2DOutputAggregationRequest(
            slice_outputs=tuple(slice_outputs),
            memory_type=memory_type,
            plane_axis=plane_axis,
        )
        for aggregator_type in cls.registered_aggregator_families():
            if aggregator_type.supports(request.slice_outputs):
                return aggregator_type().aggregate_outputs(request)
        return Pure2DAuxiliaryOutputAggregator.aggregate(
            list(request.slice_outputs),
            request.memory_type,
        )

    @classmethod
    def supports(cls, slice_outputs: CellProfilerRuntimeValueSequence) -> bool:
        """Return whether this aggregator owns the output payload type."""
        accepted_types = cls.accepted_output_types()
        return (
            bool(slice_outputs)
            and bool(accepted_types)
            and all(isinstance(output, accepted_types) for output in slice_outputs)
        )

    @classmethod
    def registered_aggregator_families(
        cls,
    ) -> tuple[type["CellProfilerPure2DOutputAggregator"], ...]:
        """Return registered aggregators plus typed family bases in MRO order."""
        family_types: list[type[CellProfilerPure2DOutputAggregator]] = []
        for aggregator_type in cls.__registry__.values():
            for candidate_type in aggregator_type.mro():
                if (
                    candidate_type is cls
                    or not isinstance(candidate_type, type)
                    or not issubclass(candidate_type, cls)
                    or candidate_type in family_types
                ):
                    continue
                family_types.append(candidate_type)
        return tuple(family_types)

    @classmethod
    def accepted_output_types(cls) -> tuple[type[CellProfilerRuntimeValue], ...]:
        """Return nominal output types owned by this aggregator family."""
        return tuple(
            aggregator_type.output_type
            for aggregator_type in CellProfilerPure2DOutputAggregator.__registry__.values()
            if (
                aggregator_type.output_type is not None
                and issubclass(aggregator_type, cls)
            )
        )

    @abstractmethod
    def aggregate_outputs(
        self,
        request: "CellProfilerPure2DOutputAggregationRequest",
    ) -> CellProfilerRuntimeValue:
        """Aggregate one output position across pure-2D slices."""


@dataclass(frozen=True, slots=True)
class CellProfilerPure2DOutputAggregationRequest:
    """Nominal aggregation context for per-slice CellProfiler outputs."""

    slice_outputs: CellProfilerRuntimeValues
    memory_type: str
    plane_axis: RuntimePlaneAxis = RuntimePlaneAxis.RUNTIME_SLICE


class ObjectLabelValuePure2DOutputAggregator(CellProfilerPure2DOutputAggregator):
    """Aggregate typed object-label outputs."""

    output_type = None

    def aggregate_outputs(
        self,
        request: CellProfilerPure2DOutputAggregationRequest,
    ) -> CellProfilerRuntimeValue:
        return ObjectLabelPure2DSliceAggregator.aggregate(
            request.slice_outputs,
            request.memory_type,
            plane_axis=request.plane_axis,
        )


class ImagePayloadPure2DOutputAggregator(CellProfilerPure2DOutputAggregator):
    """Aggregate typed image payload outputs."""

    output_type = None

    def aggregate_outputs(
        self,
        request: CellProfilerPure2DOutputAggregationRequest,
    ) -> CellProfilerRuntimeValue:
        return _stack_cellprofiler_slice_outputs(
            request.slice_outputs,
            request.memory_type,
        )


class ParentChildRelationshipPure2DOutputAggregator(CellProfilerPure2DOutputAggregator):
    """Aggregate typed parent-child relationship outputs."""

    output_type = ParentChildRelationshipPayload

    def aggregate_outputs(
        self,
        request: CellProfilerPure2DOutputAggregationRequest,
    ) -> CellProfilerRuntimeValue:
        slice_outputs = request.slice_outputs
        return ParentChildRelationshipPayload(
            parent_ids=tuple(
                parent_id for output in slice_outputs for parent_id in output.parent_ids
            ),
            child_ids=tuple(
                child_id for output in slice_outputs for child_id in output.child_ids
            ),
            slice_indices=tuple(
                slice_index
                for slice_index, output in enumerate(slice_outputs)
                for _child_id in output.child_ids
            ),
            slice_count=len(slice_outputs),
        )


@dataclass(frozen=True, slots=True)
class Pure2DOutputAggregatorSpec(GeneratedLeafClassSpec):
    """Declarative leaf spec for one pure-2D output aggregator."""

    output_type: CellProfilerRuntimeType

    def class_attributes(self) -> CellProfilerClassAttributes:
        return {"output_type": self.output_type}


for _pure_2d_output_aggregator_spec in (
    Pure2DOutputAggregatorSpec(
        "ObjectLabelPayloadPure2DOutputAggregator",
        ObjectLabelValuePure2DOutputAggregator,
        ObjectLabelPayload,
    ),
    Pure2DOutputAggregatorSpec(
        "ObjectLabelSetPure2DOutputAggregator",
        ObjectLabelValuePure2DOutputAggregator,
        ObjectLabelSet,
    ),
    Pure2DOutputAggregatorSpec(
        "MaskedImagePayloadPure2DOutputAggregator",
        ImagePayloadPure2DOutputAggregator,
        MaskedImagePayload,
    ),
    Pure2DOutputAggregatorSpec(
        "ImageMetadataPayloadPure2DOutputAggregator",
        ImagePayloadPure2DOutputAggregator,
        ImageMetadataPayload,
    ),
    Pure2DOutputAggregatorSpec(
        "NumPyImagePure2DOutputAggregator",
        ImagePayloadPure2DOutputAggregator,
        np.ndarray,
    ),
):
    _pure_2d_output_aggregator_spec.declare_in(globals())


def _stack_cellprofiler_slice_outputs(
    slice_outputs: CellProfilerRuntimeValueSequence,
    memory_type: str,
) -> CellProfilerRuntimeValue:
    normalized_outputs = tuple(
        SINGLETON_STACK_OUTPUT_COLLAPSE.collapse(output) for output in slice_outputs
    )
    output_masks = tuple(image_payload_mask(output) for output in normalized_outputs)
    output_data = tuple(image_payload_data(output) for output in normalized_outputs)
    try:
        stacked = ImageStackLayout.stack_slices_or_single_stack(
            slices=output_data,
            memory_type=memory_type,
            gpu_id=0,
        )
    except ValueError as exc:
        raise ValueError(
            "CellProfiler slice outputs must share a registered OpenHCS image "
            "stack layout; got shapes "
            f"{[output.shape if isinstance(output, np.ndarray) else None for output in output_data]!r}."
        ) from exc
    return _with_stacked_output_context(
        stacked,
        normalized_outputs,
        output_masks,
        memory_type,
    )


def _unstack_cellprofiler_image_slices(image: CellProfilerRuntimeValue, memory_type: str) -> CellProfilerRuntimeValues:
    image_data = image_payload_data(image)
    image_mask = image_payload_mask(image)
    image_metadata = image_payload_metadata(image)
    image_data_semantics = ImageArrayShapeSemantics(image_data)
    if image_data_semantics.is_pairwise_slice_grid:
        image_data = image_data_semantics.collapse_pairwise_slice_grid()
        if (
            image_mask is not None
            and isinstance(image_mask, np.ndarray)
            and ImageArrayShapeSemantics(image_mask).shares_pairwise_slice_grid_axes_with(
                image_data_semantics.value
            )
        ):
            image_mask = ImageArrayShapeSemantics(image_mask).collapse_pairwise_slice_grid()
    slice_projector = ImagePayloadSliceProjector(
        mask=image_mask,
        metadata=image_metadata,
    )
    if is_color_image_slice(image_data):
        return (image,)
    if is_color_image_stack(image_data):
        source_type = detect_memory_type(image_data)
        if source_type != memory_type:
            image_data = convert_memory(
                data=image_data,
                source_type=source_type,
                target_type=memory_type,
                gpu_id=0,
            )
        return tuple(
            slice_projector.payload_for_slice(image_data[index], index)
            for index in range(image_data.shape[0])
        )
    if (
        plane_stack := RuntimeSliceProjection.grayscale_plane_stack_view(
            image_data,
            flatten_high_rank=True,
        )
    ) is not None:
        return tuple(
            slice_projector.payload_for_slice(plane_stack[index], index)
            for index in range(plane_stack.shape[0])
        )
    return tuple(
        slice_projector.payload_for_slice(slice_data, index)
        for index, slice_data in enumerate(
            ImageStackLayoutUnstackRequest(image_data, memory_type, 0).slices()
        )
    )


def _with_stacked_output_context(
    stacked: CellProfilerRuntimeValue,
    slice_outputs: CellProfilerRuntimeValueSequence,
    masks: Sequence[CellProfilerRuntimeValue | None],
    memory_type: str,
) -> CellProfilerRuntimeValue:
    metadata = ImagePayloadMetadataCompositionRequest(slice_outputs).metadata()
    present_masks = tuple(mask for mask in masks if mask is not None)
    if not present_masks:
        return metadata.payload_with(stacked)
    if len(present_masks) != len(masks):
        raise ValueError("Cannot stack a mix of masked and unmasked image outputs.")
    stacked_mask = (
        present_masks[0]
        if len(present_masks) == 1
        else ImageStackLayout.stack_slices_or_single_stack(
            present_masks,
            memory_type=memory_type,
            gpu_id=0,
        )
    )
    return metadata.payload_with(stacked, mask=stacked_mask)
