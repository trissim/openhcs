"""Pure-2D CellProfiler output aggregation authorities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.aligned_image_payload import (
    ImageArrayShapeSemantics,
    ImagePayloadSliceProjector,
)
from openhcs.core.image_shapes import is_color_image_slice, is_color_image_stack
from openhcs.core.image_stack_layout import ImageStackLayout, ImageStackLayoutUnstackRequest
from openhcs.core.measurement_row_materialization import MeasurementRowsAxisProjection
from openhcs.core.memory import convert_memory, detect_memory_type
from openhcs.core.runtime_semantics import ParentChildRelationshipPayload, RuntimePlaneAxis
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_values import (
    ImageMetadataPayload,
    ImagePayloadMetadataCompositionRequest,
    MaskedImagePayload,
    ObjectLabelPayload,
    ObjectLabelPure2DSliceAggregator,
    ObjectLabelSet,
    SourceImagePlaneAxisPolicy,
    SourceImagePlaneAxisRequest,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.source_image_semantics import source_image_payload_role
from openhcs.interop.cellprofiler.runtime.image_payload_collapse import (
    SINGLETON_STACK_OUTPUT_COLLAPSE,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
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


@dataclass(frozen=True, slots=True)
class CellProfilerPure2DImagePlaneSemantics:
    """Source-plane semantics for CellProfiler PURE_2D image execution."""

    image: CellProfilerRuntimeValue

    @classmethod
    def from_image(
        cls,
        image: CellProfilerRuntimeValue,
    ) -> "CellProfilerPure2DImagePlaneSemantics":
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
                image_mask = ImageArrayShapeSemantics(
                    image_mask
                ).collapse_pairwise_slice_grid()
            image = image_metadata.payload_with(image_data, image_mask)
        return cls(image)

    @property
    def data(self) -> CellProfilerRuntimeValue:
        return image_payload_data(self.image)

    @property
    def mask(self) -> CellProfilerRuntimeValue | None:
        return image_payload_mask(self.image)

    @property
    def metadata(self):
        return image_payload_metadata(self.image)

    @property
    def source_role(self):
        return source_image_payload_role(self.image)

    @property
    def plane_axis(self) -> RuntimePlaneAxis | None:
        return SourceImagePlaneAxisPolicy.for_request(
            SourceImagePlaneAxisRequest(self.image)
        ).axis()

    def is_single_source_plane(self) -> bool:
        if is_color_image_slice(self.data):
            return True
        if self.plane_axis is not None:
            return False
        source_role = self.source_role
        return (
            source_role is not None
            and source_role.is_channel_last_source_plane(self.data)
        )

    def slices(self, memory_type: str) -> CellProfilerRuntimeValues:
        if self.is_single_source_plane():
            return (self.image,)
        slice_projector = ImagePayloadSliceProjector(
            mask=self.mask,
            metadata=self.metadata,
        )
        source_role = self.source_role
        if (
            is_color_image_stack(self.data)
            or (
                source_role is not None
                and source_role.is_channel_last_source_stack(self.data)
            )
        ):
            image_data = self.data
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
                self.data,
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
                ImageStackLayoutUnstackRequest(self.data, memory_type, 0).slices()
            )
        )


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


class MeasurementRowSequencePure2DOutputAggregator(CellProfilerPure2DOutputAggregator):
    """Aggregate per-slice measurement rows with the outer PURE_2D slice identity."""

    output_type = tuple

    @classmethod
    def supports(cls, slice_outputs: CellProfilerRuntimeValueSequence) -> bool:
        return bool(slice_outputs) and all(
            cls.row_sequence_for_value(output) is not None
            for output in slice_outputs
        )

    def aggregate_outputs(
        self,
        request: CellProfilerPure2DOutputAggregationRequest,
    ) -> CellProfilerRuntimeValue:
        rows: list[CellProfilerRuntimeValue] = []
        for slice_index, output in enumerate(request.slice_outputs):
            row_sequence = self.row_sequence_for_value(output)
            if row_sequence is None:
                raise TypeError(
                    f"{type(self).__name__} got non-row output "
                    f"{type(output).__name__}."
                )
            projection = MeasurementRowsAxisProjection.from_rows(row_sequence)
            rows.extend(projection.project_runtime_slice_index(slice_index))
        return rows

    @staticmethod
    def row_sequence_for_value(
        value: CellProfilerRuntimeValue,
    ) -> tuple[CellProfilerRuntimeValue, ...] | None:
        if isinstance(value, (str, bytes, np.ndarray)):
            return None
        if not isinstance(value, Sequence):
            return None
        rows = tuple(value)
        if not rows:
            return ()
        if all(is_dataclass(row) or isinstance(row, Mapping) for row in rows):
            return rows
        return None


class ObjectLabelPayloadPure2DOutputAggregator(
    ObjectLabelValuePure2DOutputAggregator
):
    output_type = ObjectLabelPayload


class ObjectLabelSetPure2DOutputAggregator(ObjectLabelValuePure2DOutputAggregator):
    output_type = ObjectLabelSet


class MaskedImagePayloadPure2DOutputAggregator(ImagePayloadPure2DOutputAggregator):
    output_type = MaskedImagePayload


class ImageMetadataPayloadPure2DOutputAggregator(ImagePayloadPure2DOutputAggregator):
    output_type = ImageMetadataPayload


class NumPyImagePure2DOutputAggregator(ImagePayloadPure2DOutputAggregator):
    output_type = np.ndarray


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
    return CellProfilerPure2DImagePlaneSemantics.from_image(image).slices(memory_type)


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
