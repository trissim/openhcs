"""Runtime plane projection authorities for CellProfiler adapter payloads."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import ClassVar, cast

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.aligned_image_payload import project_singleton_stack_image_domain
from openhcs.core.image_shapes import is_color_image_slice, is_image_stack
from openhcs.core.runtime_semantics import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
    bounded_runtime_plane_index,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImagePayloadMetadataInput,
    ObjectLabelSet,
    RuntimeImagePayloadContext,
    SourceImagePlaneAxisPolicy,
    SourceImagePlaneAxisRequest,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_slice_context,
    object_label_dense_array,
)
from openhcs.core.source_matching import SourceImageSetIdentity
from openhcs.interop.cellprofiler.runtime.payload_types import (
    ImagePayloadMaskValue,
    ImagePayloadValue,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerCurrentImage,
    CurrentSourcePayloadPlaneSelectionAuthority,
    CurrentSourcePayloadPlaneSelectionRequest,
    ParsedSourceMetadata,
    SourcePayloadPlaneIdentitySequence,
    SourcePlaneIdentitySequence,
    SourceScopedPayload,
    _SOURCE_PLANE_IDENTITY_POLICY,
)
@dataclass(frozen=True, slots=True)
class CurrentSourcePlaneProjectionBase(metaclass=AutoRegisterMeta):
    """Current-source plane identity authority for stack-like runtime payloads."""

    __registry_key__ = "projection_key"
    __skip_if_no_key__ = True
    projection_key: ClassVar[str | None] = None

    adapter: "CellProfilerRuntimeAdapter"
    current_image: CellProfilerCurrentImage

    @classmethod
    def registered_types(cls) -> tuple[type["CurrentSourcePlaneProjectionBase"], ...]:
        return tuple(cls.__registry__.values())

    def matching_plane_index(self, payload: SourceScopedPayload) -> int | None:
        plane_selection = self.select_matching_plane(payload)
        plane_selection.require_unambiguous()
        if not plane_selection.is_matched:
            return None
        return plane_selection.plane_index

    def payload_plane_identities(
        self,
        payload: SourceScopedPayload,
    ) -> SourcePlaneIdentitySequence:
        return SourcePayloadPlaneIdentitySequence(
            payload,
            _SOURCE_PLANE_IDENTITY_POLICY,
        ).identities()

    def payload_image_set_identities(
        self,
        payload: SourceScopedPayload,
    ) -> SourcePlaneIdentitySequence:
        """Return plane identities reduced to the source image-set axes."""
        return SourcePayloadPlaneIdentitySequence(
            payload,
            SourceImageSetIdentity.DEFAULT_POLICY,
        ).identities()

    def payload_has_plane_identity(self, payload: SourceScopedPayload) -> bool:
        return SourcePayloadPlaneIdentitySequence(
            payload,
            _SOURCE_PLANE_IDENTITY_POLICY,
        ).has_identity

    def select_matching_plane(
        self,
        payload: SourceScopedPayload,
    ) -> "CurrentSourcePayloadPlaneSelection":
        return CurrentSourcePayloadPlaneSelectionAuthority.select(
            CurrentSourcePayloadPlaneSelectionRequest(
                adapter=self.adapter,
                current_image=self.current_image,
                payload=payload,
            )
        )

@dataclass(frozen=True, slots=True)
class CurrentSourcePayloadPlaneSelector(CurrentSourcePlaneProjectionBase):
    """Select the stacked payload plane matching the current source image."""

    projection_key: ClassVar[str] = "payload_plane_selection"

@dataclass(frozen=True, slots=True)
class RuntimePlaneCurrentImageContext:
    """Current-image provenance context for runtime-plane projection."""

    current_image: ImagePayloadMetadataInput | None = None

    @property
    def has_image(self) -> bool:
        return self.current_image is not None

    def require_image(self) -> ImagePayloadMetadataInput:
        if self.current_image is None:
            raise RuntimeError("Runtime plane projection requires a current image.")
        return self.current_image

    def current_image_is_planar(self) -> bool:
        current_data = image_payload_data(
            project_singleton_stack_image_domain(self.require_image())
        )
        if is_color_image_slice(current_data):
            return True
        current_array = np.asarray(current_data)
        return current_array.ndim == 2

@dataclass(frozen=True, slots=True)
class RuntimePlaneSelectedPlaneIndex:
    """Selected runtime plane index carrier."""

    value: int | None

@dataclass(frozen=True, slots=True)
class RuntimePlaneSelectedImagePayloadPlane:
    """Selected runtime image plane data plus its source stack index."""

    data: np.ndarray
    index: int

@dataclass(frozen=True, slots=True)
class RuntimePlaneProjectionContext:
    """Runtime-plane projection context shared across projection requests."""

    adapter: "CellProfilerRuntimeAdapter"
    current_image_context: RuntimePlaneCurrentImageContext = field(
        default_factory=RuntimePlaneCurrentImageContext
    )

@dataclass(frozen=True, slots=True)
class RuntimePlaneImagePayloadProjection:
    """Project source-bound image stacks to the adapter's selected runtime plane."""

    context: RuntimePlaneProjectionContext

    def project(
        self,
        payload: ImagePayloadMetadataInput,
    ) -> ImagePayloadMetadataInput:
        if self.context.adapter.runtime_slice_plane_index() is None:
            return payload
        plane_stack = RuntimePlaneImagePayloadStack(payload)
        if not plane_stack.is_projectable:
            return payload
        selection = RuntimePlaneImagePayloadPlaneSelection(
            context=self.context,
            payload=payload,
            plane_count=plane_stack.plane_count,
        ).select()
        plane_index = selection.selected_plane_index.value
        if plane_index is None:
            return payload
        if plane_index >= plane_stack.plane_count:
            raise RuntimeError(
                "Runtime image plane projection produced an out-of-range plane "
                f"index {plane_index} for payload shape {plane_stack.shape!r}."
            )
        return RuntimePlaneImagePayloadSliceContext(
            payload=payload,
            context=self.context,
            selected_plane=RuntimePlaneSelectedImagePayloadPlane(
                data=plane_stack.plane(plane_index),
                index=plane_index,
            ),
            source_context=selection.source_context,
        ).project()

@dataclass(frozen=True, slots=True)
class RuntimePlaneImagePayloadStack:
    """Array-domain view used for runtime-plane image projection."""

    payload: ImagePayloadMetadataInput

    @property
    def array(self) -> np.ndarray:
        return np.asarray(image_payload_data(self.payload))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.array.shape)

    @property
    def is_projectable(self) -> bool:
        return is_image_stack(self.array) and self.plane_count > 1

    @property
    def plane_count(self) -> int:
        if self.array.ndim < 1:
            return 0
        return int(self.array.shape[0])

    def plane(self, plane_index: int) -> np.ndarray:
        return self.array[plane_index]

@dataclass(frozen=True, slots=True)
class RuntimePlaneProjectionRequest:
    """Shared runtime-plane projection request coordinates."""

    context: RuntimePlaneProjectionContext
    plane_count: int

@dataclass(frozen=True, slots=True)
class RuntimePlanePayloadProjectionRequest(RuntimePlaneProjectionRequest):
    """Runtime-plane projection request coordinates for one image payload."""

    payload: ImagePayloadMetadataInput

@dataclass(frozen=True, slots=True)
class RuntimePlaneImagePayloadPlaneSelection(RuntimePlanePayloadProjectionRequest):
    """Resolve a projection plane from source provenance or current image context."""

    def select(self) -> "RuntimePlaneImagePayloadPlaneSelectionResult":
        axis = SourceImagePlaneAxisPolicy.for_request(
            SourceImagePlaneAxisRequest(self.payload)
        ).axis()
        if axis is not None:
            return RuntimePlaneImagePayloadPlaneSelectionResult(
                selected_plane_index=RuntimePlaneSelectedPlaneIndex(
                    RuntimePlaneImagePayloadPlaneIndex(
                        context=self.context,
                        payload=self.payload,
                        plane_count=self.plane_count,
                        axis=axis,
                    ).value()
                ),
                source_context=RuntimePlaneImagePayloadSourceContext.PAYLOAD_PLANE,
            )
        return RuntimePlaneImagePayloadPlaneSelectionResult(
            selected_plane_index=RuntimePlaneSelectedPlaneIndex(
                RuntimePlaneCurrentImagePayloadPlaneIndex(
                    context=self.context,
                    plane_count=self.plane_count,
                ).value()
            ),
            source_context=RuntimePlaneImagePayloadSourceContext.CURRENT_IMAGE,
        )

@dataclass(frozen=True, slots=True)
class RuntimePlaneImagePayloadPlaneSelectionResult:
    """Plane selection plus the authority for the projected plane source context."""

    selected_plane_index: RuntimePlaneSelectedPlaneIndex
    source_context: "RuntimePlaneImagePayloadSourceContext"

class RuntimePlaneImagePayloadSourceContext(Enum):
    """Source-context authority for a projected runtime image plane."""

    PAYLOAD_PLANE = "payload_plane"
    CURRENT_IMAGE = "current_image"

@dataclass(frozen=True, slots=True)
class RuntimePlaneImagePayloadSourceContextRequest:
    """Shared current-image/source-context coordinates for metadata projection."""

    context: RuntimePlaneProjectionContext
    source_context: RuntimePlaneImagePayloadSourceContext

@dataclass(frozen=True, slots=True)
class RuntimePlaneCurrentImagePayloadPlaneIndex(RuntimePlaneProjectionRequest):
    """Fallback plane selection for stacks that lost per-plane source metadata."""

    def value(self) -> int | None:
        if not self.context.current_image_context.has_image:
            return None
        if not self.current_image_is_planar():
            return None
        return bounded_runtime_plane_index(
            self.plane_count,
            self.context.adapter.runtime_slice_plane_index(),
        )

    def current_image_is_planar(self) -> bool:
        return self.context.current_image_context.current_image_is_planar()

@dataclass(frozen=True, slots=True)
class RuntimePlaneImagePayloadSliceContext(RuntimePlaneImagePayloadSourceContextRequest):
    """Attach the best source context to a projected runtime image plane."""

    payload: ImagePayloadMetadataInput
    selected_plane: RuntimePlaneSelectedImagePayloadPlane

    def project(self) -> ImagePayloadMetadataInput:
        projected = image_payload_slice_context(
            self.payload,
            self.selected_plane.data,
            self.selected_plane.index,
        )
        if not self.context.current_image_context.has_image:
            return projected
        metadata = RuntimePlaneImagePayloadProjectedMetadata(
            projected=projected,
            context=self.context,
            source_context=self.source_context,
        ).metadata()
        return metadata.payload_with(
            image_payload_data(projected),
            mask=image_payload_mask(projected),
        )

@dataclass(frozen=True, slots=True)
class RuntimePlaneImagePayloadProjectedMetadata(RuntimePlaneImagePayloadSourceContextRequest):
    """Resolve metadata for a runtime-image plane after projection."""

    projected: ImagePayloadMetadataInput

    def metadata(self) -> ImagePayloadMetadata:
        projected_metadata = image_payload_metadata(self.projected)
        current_metadata = image_payload_metadata(
            self.context.current_image_context.require_image()
        )
        if self.source_context is RuntimePlaneImagePayloadSourceContext.CURRENT_IMAGE:
            metadata = replace(projected_metadata)
            metadata.source_path = current_metadata.source_path
            metadata.source_component_metadata = current_metadata.source_component_metadata
            metadata.source_image_provenance_planes = (
                current_metadata.source_image_provenance_planes
            )
            metadata.source_image_names = (
                current_metadata.source_image_names
                or projected_metadata.source_image_names
            )
            metadata.spatial_origin_yx = current_metadata.spatial_origin_yx
            metadata.source_spatial_shape_yx = current_metadata.source_spatial_shape_yx
            metadata.physical_border_edges_yx = current_metadata.physical_border_edges_yx
            metadata.mask_defines_border = current_metadata.mask_defines_border
            return metadata
        return projected_metadata.with_source_context_from(current_metadata)

@dataclass(frozen=True, slots=True)
class RuntimePlaneImagePayloadPlaneIndex(RuntimePlanePayloadProjectionRequest):
    """Resolve the selected plane for an image payload's semantic plane axis."""

    axis: RuntimePlaneAxis

    def value(self) -> int | None:
        if self.axis is RuntimePlaneAxis.RUNTIME_SLICE:
            return bounded_runtime_plane_index(
                self.plane_count,
                self.context.adapter.runtime_slice_plane_index(),
            )
        from openhcs.interop.cellprofiler.runtime.source_binding_runtime import (
            SourceBindingPayloadPlaneResolution,
        )

        return SourceBindingPayloadPlaneResolution(
            adapter=self.context.adapter,
            payload=self.payload,
            plane_count=self.plane_count,
        ).plane_index()

@dataclass(frozen=True, slots=True)
class CurrentSourceImagePayloadProjection(CurrentSourcePlaneProjectionBase):
    """Project a stacked runtime image payload to the plane matching current source."""

    projection_key: ClassVar[str] = "image_payload"

    def project(self, payload: ImagePayloadMetadataInput) -> ImagePayloadValue:
        data = self._projectable_data(payload)
        if not self.is_projectable_stack(data):
            return payload
        if not RuntimePlaneCurrentImageContext(
            self.current_image
        ).current_image_is_planar():
            return payload

        plane_index = self.matching_plane_index(payload)
        if plane_index is None:
            return payload
        mask = image_payload_mask(payload)
        mask_plane = self.project_mask_plane(
            mask,
            data=data,
            plane_index=plane_index,
        )
        return cast(
            ImagePayloadValue,
            RuntimeImagePayloadContext(
                cast(ImagePayloadValue, data[plane_index]),
                cast(ImagePayloadMaskValue, mask_plane),
                image_payload_metadata(payload).for_source_plane(plane_index),
            ).payload(),
        )

    def _projectable_data(self, payload: ImagePayloadMetadataInput) -> ImagePayloadValue:
        return image_payload_data(payload)

    def is_projectable_stack(self, data: ImagePayloadValue) -> bool:
        return isinstance(data, np.ndarray) and is_image_stack(data) and data.shape[0] > 1

    @staticmethod
    def project_mask_plane(
        mask: ImagePayloadMaskValue,
        *,
        data: np.ndarray,
        plane_index: int,
    ) -> ImagePayloadMaskValue:
        """Project a stack-aligned mask plane when the mask carries that axis."""
        if mask is None:
            return None
        mask_array = np.asarray(mask)
        if mask_array.ndim >= 3 and mask_array.shape[0] == data.shape[0]:
            return mask_array[plane_index]
        return mask

@dataclass(frozen=True, slots=True)
class CurrentSourceObjectLabelPayloadProjection(CurrentSourcePlaneProjectionBase):
    """Project stacked object labels to the plane matching current source."""

    projection_key: ClassVar[str] = "object_label_payload"

    def project(self, labels: ObjectLabelSet) -> ObjectLabelSet:
        data = self._projectable_data(labels)
        if not self.is_projectable_stack(data):
            return labels
        if not RuntimePlaneCurrentImageContext(
            self.current_image
        ).current_image_is_planar():
            return labels

        runtime_payload = labels.runtime_payload()
        plane_selection = self.select_matching_plane(runtime_payload)
        projection = self.runtime_axis_projection(
            labels,
            plane_count=int(data.shape[0]),
            selected_plane_index=plane_selection.plane_index,
        )
        if projection is not None:
            projected = projection.project(labels)
            if not isinstance(projected, ObjectLabelSet):
                raise TypeError(
                    "Current-source object-label projection must preserve "
                    f"ObjectLabelSet, got {type(projected).__name__}."
                )
            if projected is not labels:
                return projected

        if plane_selection.plane_index is None:
            plane_selection.require_unambiguous()
        return labels

    def _projectable_data(self, payload: ObjectLabelSet) -> np.ndarray:
        return object_label_dense_array(payload)

    def runtime_axis_projection(
        self,
        labels: ObjectLabelSet,
        *,
        plane_count: int,
        selected_plane_index: int | None,
    ) -> RuntimePlaneAxisValueProjection | None:
        if selected_plane_index is not None:
            return RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=labels.plane_axis,
                plane_index=selected_plane_index,
                axis_size=plane_count,
            )
        if (
            labels.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
            and not labels.source_alias_group
        ):
            from openhcs.interop.cellprofiler.runtime.source_binding_runtime import (
                SourceBindingAxisResolutionAuthority,
            )

            plane_index = (
                SourceBindingAxisResolutionAuthority.active_axis_plane_resolution(
                    self.adapter
                ).plane_index()
            )
            if plane_index is not None:
                return RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=labels.plane_axis,
                    plane_index=plane_index,
                    axis_size=plane_count,
                )
        return RuntimePlaneAxisValueProjection.from_projector(
            self.adapter,
            labels.plane_axis,
            labels.source_alias_group,
        )

    def is_projectable_stack(self, data: ImagePayloadValue) -> bool:
        return (
            isinstance(data, np.ndarray)
            and data.ndim >= 3
            and data.shape[0] > 1
        )
