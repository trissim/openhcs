"""Generic aligned image-payload composition for multi-source runtime inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta
from openhcs.core.alias_property import AliasProperty
import numpy as np

from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_color_volume_stack,
    is_grayscale_image_slice,
    is_grayscale_image_stack,
    is_grayscale_volume_slice,
    is_grayscale_volume_stack,
    is_image_stack,
)
from openhcs.core.image_stack_layout import (
    ImageStackLayoutUnstackRequest,
    NumpySliceConversion,
)
from openhcs.core.memory import MEMORY_TYPE_NUMPY, convert_memory, detect_memory_type
from openhcs.core.registry_strategies import (
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.source_image_provenance import SourcePlaneIndexedProvenanceExpansion
from openhcs.core.source_spatial_domain import (
    SourceSpatialDomain,
    SourceSpatialDomainAdapter,
)
from openhcs.core.runtime_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    ImagePayloadMetadataCarrier,
    ImagePayloadMetadataCompositionMode,
    ImagePayloadMetadataCompositionRequest,
    ImagePayloadMetadataInput,
    ImagePayloadSequence,
    MaskedImagePayload,
    ObjectLabelDenseDataStrategy,
    ObjectLabelPayload,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelSet,
    ObjectLabelValue,
    RuntimeArrayData,
    RuntimeImagePayloadContext,
    RuntimeSliceStackRequest,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    project_image_mask_to_data_domain,
)
from openhcs.core.source_image_semantics import source_image_payload_role


@dataclass(frozen=True, slots=True)
class ImagePayloadSourceSpatialDomainAdapter(SourceSpatialDomainAdapter):
    """Source-domain adapter for image payload data and masks."""

    value_type = ImagePayloadMetadataCarrier
    value_type_label = "image_payload"
    value: Any
    source_domain: SourceSpatialDomain
    array = AliasProperty[Any]("value")
    domain = AliasProperty[SourceSpatialDomain]("source_domain")

    @classmethod
    def for_value(
        cls,
        value: Any,
        *,
        source_shape_override_yx: tuple[int, int] | None = None,
    ) -> "ImagePayloadSourceSpatialDomainAdapter | None":
        if not isinstance(value, ImagePayloadMetadataCarrier):
            return None
        return cls(
            image_payload_data(value),
            cls.domain_from_metadata(
                image_payload_metadata(value),
                value_name="Image payload",
            ),
        )

    @classmethod
    def domain_from_metadata(
        cls,
        metadata: ImagePayloadMetadata,
        *,
        fill_value: Any = 0,
        source_domain: SourceSpatialDomain | None = None,
        value_name: str,
    ) -> SourceSpatialDomain:
        domain = metadata.source_spatial_domain
        if source_domain is not None:
            domain = domain.with_missing_from(source_domain)
        return (
            domain
            .with_missing_from(SourceSpatialDomain(origin_yx=(0, 0)))
            .with_fill_value(fill_value)
            .with_value_name(value_name)
        )

    @property
    def spatial_shape_yx(self) -> tuple[int, int]:
        array = np.asarray(self.array)
        if is_color_image_slice(self.array):
            return tuple(int(value) for value in array.shape[:2])
        if array.ndim < 2:
            raise ValueError(
                "Source-spatial image payloads require at least two dimensions, "
                f"got {array.ndim}."
            )
        return tuple(int(value) for value in array.shape[-2:])

    @classmethod
    def payloads_aligned_to_common_source_domain(
        cls,
        payloads: tuple[ImagePayloadMetadataInput, ...],
    ) -> tuple[ImagePayloadMetadataInput, ...]:
        adapters = cls.source_domain_adapters(payloads)
        source_domain = SourceSpatialDomainAdapter.common_source_domain(
            adapters,
            value_name="Image bundle source image",
        )
        if (
            source_domain is None
            or not SourceSpatialDomainAdapter.requires_source_domain_alignment(adapters)
        ):
            return payloads
        return tuple(
            cls.payload_in_source_domain(payload, source_domain)
            for payload in payloads
        )

    @classmethod
    def source_domain_adapters(
        cls,
        payloads: tuple[ImagePayloadMetadataInput, ...],
    ) -> tuple["ImagePayloadSourceSpatialDomainAdapter", ...]:
        adapters: list[ImagePayloadSourceSpatialDomainAdapter] = []
        for payload in payloads:
            adapter = SourceSpatialDomainAdapter.for_value(payload)
            if not isinstance(adapter, cls):
                raise TypeError(
                    "Image bundle alignment requires image payload adapters."
                )
            adapters.append(adapter)
        return tuple(adapters)

    @classmethod
    def payload_in_source_domain(
        cls,
        payload: ImagePayloadMetadataInput,
        source_domain: SourceSpatialDomain,
    ) -> ImagePayloadMetadataInput:
        metadata = image_payload_metadata(payload)
        source_metadata = metadata.with_materialized_source_domain(source_domain)
        data = cls(
            image_payload_data(payload),
            cls.domain_from_metadata(
                metadata,
                source_domain=source_domain,
                value_name="Image payload",
            ),
        ).materialize()
        return source_metadata.payload_with(
            data,
            cls.mask_in_source_domain(payload, metadata, source_domain),
        )

    @classmethod
    def mask_in_source_domain(
        cls,
        payload: ImagePayloadMetadataInput,
        metadata: ImagePayloadMetadata,
        source_domain: SourceSpatialDomain,
    ) -> RuntimeArrayData | None:
        mask = image_payload_mask(payload)
        if mask is None:
            return None
        return cls(
            mask,
            cls.domain_from_metadata(
                metadata,
                fill_value=False,
                source_domain=source_domain,
                value_name="Image mask",
            ),
        ).materialize()


class NumPyImagePayloadSourceSpatialDomainAdapter(ImagePayloadSourceSpatialDomainAdapter):
    """Source-domain adapter for raw NumPy image arrays."""

    value_type = np.ndarray
    value_type_label = "numpy_image"

    @classmethod
    def for_value(
        cls,
        value: Any,
        *,
        source_shape_override_yx: tuple[int, int] | None = None,
    ) -> "NumPyImagePayloadSourceSpatialDomainAdapter | None":
        if not isinstance(value, np.ndarray):
            return None
        return cls(
            value,
            SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=source_shape_override_yx,
                fill_value=0,
                value_name="NumPy image payload",
            ),
        )


@dataclass(frozen=True, slots=True)
class ObjectLabelPayloadSourceSpatialDomainAdapter(SourceSpatialDomainAdapter):
    """Source-domain adapter for object-label payload values."""

    value_type = ObjectLabelValue
    value_type_label = "object_label_payload"
    value: ObjectLabelValue
    source_shape_override_yx: tuple[int, int] | None = None

    @classmethod
    def for_value(
        cls,
        value: Any,
        *,
        source_shape_override_yx: tuple[int, int] | None = None,
    ) -> "ObjectLabelPayloadSourceSpatialDomainAdapter | None":
        if not isinstance(value, ObjectLabelValue):
            return None
        return cls(value, source_shape_override_yx=source_shape_override_yx)

    @property
    def array(self) -> Any:
        return ObjectLabelDenseDataStrategy.for_payload(self.value).data(self.value)

    @property
    def domain(self) -> SourceSpatialDomain:
        return (
            self.value.object_label_source_spatial_domain()
            .with_missing_from(
                SourceSpatialDomain(source_shape_yx=self.source_shape_override_yx)
            )
        )


@dataclass(frozen=True, slots=True)
class AlignedImageStackKwargResolver:
    """Materialize one kwarg for a specific aligned image-stack slice."""

    slice_index: int
    slice_count: int
    reference_payload: Any | None = None
    enable_payload_alignment: bool = True

    def resolve(self, value: Any) -> Any:
        strategy = AlignedImageStackKwargResolutionStrategy.for_value(value, self)
        return strategy.resolve(value, self)

    def without_slice_context(self) -> "AlignedImageStackKwargResolver":
        return type(self)(
            slice_index=self.slice_index,
            slice_count=self.slice_count,
            reference_payload=self.reference_payload,
            enable_payload_alignment=False,
        )

    def domain_adapter(self, value: Any) -> SourceSpatialDomainAdapter | None:
        if self.reference_payload is None:
            return None
        metadata = image_payload_metadata(self.reference_payload)
        if metadata.source_spatial_domain.source_shape_yx is None:
            return None
        return SourceSpatialDomainAdapter.for_value(
            value,
            source_shape_override_yx=metadata.source_spatial_domain.source_shape_yx,
        )

    def reference_domain(self) -> SourceSpatialDomainAdapter | None:
        if self.reference_payload is None:
            return None
        return SourceSpatialDomainAdapter.for_value(self.reference_payload)


@dataclass(frozen=True, slots=True)
class ImagePayloadSliceProjector:
    """Project payload context from a parent image into one child image slice."""

    mask: RuntimeArrayData | None
    metadata: ImagePayloadMetadata

    def payloads_for_slices(
        self,
        slices: Sequence[RuntimeArrayData],
    ) -> list[ImagePayloadMetadataInput]:
        """Return payloads for every child slice using one projection pass."""
        if self.mask is None:
            if not self.metadata.has_plane_specific_values:
                return [self.metadata.payload_with(slice_data) for slice_data in slices]
            return [
                self.metadata.for_source_plane(index).payload_with(slice_data)
                for index, slice_data in enumerate(slices)
            ]
        direct_payloads = self._direct_grayscale_plane_payloads(slices)
        if direct_payloads is not None:
            return direct_payloads
        return [
            self.payload_for_slice(slice_data, index)
            for index, slice_data in enumerate(slices)
        ]

    def _direct_grayscale_plane_payloads(
        self,
        slices: Sequence[RuntimeArrayData],
    ) -> list[ImagePayloadMetadataInput] | None:
        """Return direct per-plane masked payloads for ordinary 2D stack slices."""
        if self.mask is None:
            return None
        mask_array = np.asarray(self.mask, dtype=bool)
        if mask_array.ndim < 3 or mask_array.shape[0] < len(slices):
            return None
        payloads: list[ImagePayloadMetadataInput] = []
        for index, slice_data in enumerate(slices):
            if not is_grayscale_image_slice(slice_data):
                return None
            mask_slice = mask_array[index]
            if tuple(mask_slice.shape) != tuple(np.shape(slice_data)):
                return None
            payloads.append(
                self.metadata.for_source_plane(index).payload_with(
                    slice_data,
                    mask_slice,
                )
            )
        return payloads

    def payload_for_slice(
        self,
        data_slice: RuntimeArrayData,
        index: int,
    ) -> ImagePayloadMetadataInput:
        """Return a slice payload with mask and metadata in the slice domain."""
        metadata = self.metadata_for_slice(data_slice, index)
        mask = self.mask_for_slice(data_slice, index)
        payload: ImagePayloadMetadataInput = metadata.payload_with(data_slice, mask)
        return payload

    def metadata_for_slice(
        self,
        data_slice: RuntimeArrayData,
        index: int,
    ) -> ImagePayloadMetadata:
        """Return metadata for a child image slice or preserved source stack."""
        if self.slice_preserves_source_plane_axis(data_slice):
            return self.metadata_for_preserved_source_plane_axis(data_slice)
        return self.metadata.for_source_plane(index)

    def metadata_for_preserved_source_plane_axis(
        self,
        data_slice: RuntimeArrayData,
    ) -> ImagePayloadMetadata:
        """Return metadata for a child payload that still carries source planes."""
        plane_count = self.preserved_source_plane_count(data_slice)
        if plane_count is None:
            return self.metadata
        source_provenance = SourcePlaneIndexedProvenanceExpansion(
            self.metadata.source_provenance,
            expected_plane_count=plane_count,
        ).expanded()
        if source_provenance == self.metadata.source_provenance:
            return self.metadata
        return self.metadata.with_source_provenance(source_provenance)

    def mask_for_slice(
        self,
        data_slice: RuntimeArrayData,
        index: int,
    ) -> RuntimeArrayData | None:
        """Return the parent mask projected into ``data_slice``'s domain."""
        if self.mask is None:
            return None
        if self.slice_preserves_source_plane_axis(data_slice):
            preserved_mask = project_image_mask_to_data_domain(self.mask, data_slice)
            if preserved_mask is not None:
                return preserved_mask
        mask_array = np.asarray(self.mask)
        data_array = np.asarray(data_slice)
        data_shape = tuple(data_array.shape)
        spatial_shape = _payload_spatial_shape(data_slice)
        for candidate in self._mask_candidates(mask_array, index):
            candidate_shape = tuple(candidate.shape)
            if candidate_shape == data_shape or candidate_shape == spatial_shape:
                return candidate
            projected_mask = project_image_mask_to_data_domain(candidate, data_slice)
            if projected_mask is not None:
                return projected_mask
        raise ValueError(
            "Image payload mask cannot be projected into slice domain; "
            f"got mask {mask_array.shape!r} for slice {data_shape!r}."
        )

    def slice_preserves_source_plane_axis(self, data_slice: RuntimeArrayData) -> bool:
        """Return whether child data still carries every source plane."""
        return self.preserved_source_plane_count(data_slice) is not None

    def preserved_source_plane_count(
        self,
        data_slice: RuntimeArrayData,
    ) -> int | None:
        """Return preserved source-plane count for a child stack, if any."""
        if not is_image_stack(data_slice):
            return None
        data_plane_count = int(np.shape(data_slice)[0])
        source_plane_count = self.metadata.source_provenance.source_plane_count
        if source_plane_count == data_plane_count:
            return data_plane_count
        expanded_provenance = SourcePlaneIndexedProvenanceExpansion(
            self.metadata.source_provenance,
            expected_plane_count=data_plane_count,
        ).expanded()
        return (
            data_plane_count
            if expanded_provenance.source_plane_count == data_plane_count
            else None
        )

    @staticmethod
    def _mask_candidates(mask: np.ndarray, index: int) -> tuple[np.ndarray, ...]:
        candidates: list[np.ndarray] = []
        if mask.ndim >= 3 and mask.shape[0] > index:
            candidates.append(mask[index])
        from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

        plane_stack = RuntimeSliceProjection.grayscale_plane_stack_view(
            mask,
            flatten_high_rank=True,
        )
        if plane_stack is not None and plane_stack.shape[0] > index:
            candidates.append(plane_stack[index])
        if mask.ndim >= 3 and mask.shape[0] == 1:
            child = mask[0]
            candidates.append(child)
            if child.ndim >= 3 and child.shape[0] > index:
                candidates.append(child[index])
        candidates.append(mask)
        return tuple(candidates)


def stack_image_payload_context(
    image_payloads: Sequence[Any],
    stack: RuntimeArrayData,
) -> Any:
    """Attach composed image metadata and masks to a freshly stacked payload."""
    payloads = tuple(image_payloads)
    metadata = ImagePayloadMetadataCompositionRequest(payloads).metadata()
    return RuntimeImagePayloadContext(
        stack,
        _stack_image_payload_mask(payloads, stack),
        metadata,
    ).payload()


def stack_image_payload_context_from_metadata(
    image_payloads: Sequence[Any],
    stack: RuntimeArrayData,
    metadata_by_payload: Sequence[ImagePayloadMetadata],
) -> Any:
    """Attach composed image context using already resolved payload metadata."""
    payloads = tuple(image_payloads)
    metadata = ImagePayloadMetadataCompositionRequest(
        payloads,
        source_metadata_override=tuple(metadata_by_payload),
    ).metadata()
    return RuntimeImagePayloadContext(
        stack,
        _stack_image_payload_mask(payloads, stack),
        metadata,
    ).payload()


def _stack_image_payload_mask(
    image_payloads: Sequence[Any],
    stack: RuntimeArrayData,
) -> RuntimeArrayData | None:
    masks = tuple(image_payload_mask(payload) for payload in image_payloads)
    if not any(mask is not None for mask in masks):
        return None
    payloads = tuple(image_payloads)
    data_slices = tuple(image_payload_data(payload) for payload in payloads)
    stack_shape = tuple(np.shape(stack))
    if len(payloads) == 1 and stack_shape == tuple(np.shape(data_slices[0])):
        output_slice_domains = (stack,)
        compose_stack_axis = False
    else:
        if stack_shape[:1] != (len(payloads),):
            raise ValueError(
                "Image payload stack mask composition requires output stack "
                f"axis length {len(payloads)}, got stack shape {stack_shape!r}."
            )
        output_slice_domains = tuple(
            stack[slice_index]
            for slice_index in range(len(payloads))
        )
        compose_stack_axis = True
    resolved_masks = tuple(
        _complete_image_payload_mask(slice_domain, mask)
        for slice_domain, mask in zip(output_slice_domains, masks)
    )
    if compose_stack_axis:
        return np.stack(resolved_masks)
    return resolved_masks[0]


def _complete_image_payload_mask(
    payload_data: RuntimeArrayData,
    mask: RuntimeArrayData | None,
) -> RuntimeArrayData:
    if mask is not None:
        projected_mask = project_image_mask_to_data_domain(mask, payload_data)
        if projected_mask is None:
            raise ValueError(
                "Image payload mask cannot be projected into output slice "
                f"domain; got mask {tuple(np.shape(mask))!r} for slice "
                f"{tuple(np.shape(payload_data))!r}."
            )
        return projected_mask
    return np.ones(_payload_spatial_shape(payload_data), dtype=bool)


def unstack_image_payload_context(
    payload: Any,
    slices: Sequence[Any],
) -> list[Any]:
    """Attach one source plane of payload context to each unstacked image slice."""
    mask = image_payload_mask(payload)
    metadata = image_payload_metadata(payload)
    if mask is None and not metadata.has_values:
        return list(slices)
    projector = ImagePayloadSliceProjector(mask=mask, metadata=metadata)
    return projector.payloads_for_slices(slices)


class SingletonStackImageDomainStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project singleton OpenHCS image stacks into their contained image domain."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def project(cls, value: Any) -> Any:
        strategy = cls.for_nominal_value(value)
        if strategy is None:
            return value
        return strategy.project_value(value)

    @abstractmethod
    def project_value(self, value: Any) -> Any:
        """Return ``value`` with a leading singleton stack axis removed when present."""


class ArraySingletonStackImageDomainStrategy(SingletonStackImageDomainStrategy):
    value_type = np.ndarray

    def project_value(self, value: Any) -> Any:
        if not isinstance(value, np.ndarray):
            raise TypeError("Array singleton-stack projector requires ndarray.")
        if ImageArrayShapeSemantics(value).is_singleton_image_stack:
            return value[0]
        return value


class ContextualSingletonStackImageDomainStrategy(SingletonStackImageDomainStrategy):
    value_type = ImageMetadataPayload

    def project_value(self, value: Any) -> Any:
        if not isinstance(value, ImageMetadataPayload):
            raise TypeError(
                "Contextual singleton-stack projector requires ImageMetadataPayload."
            )
        projected_data = SingletonStackImageDomainStrategy.project(value.data)
        if projected_data is value.data:
            return value
        return RuntimeImagePayloadContext(
            projected_data,
            None,
            value.metadata.for_source_plane(0),
        ).payload()


class MaskedSingletonStackImageDomainStrategy(
    ContextualSingletonStackImageDomainStrategy
):
    value_type = MaskedImagePayload

    def project_value(self, value: Any) -> Any:
        if not isinstance(value, MaskedImagePayload):
            raise TypeError(
                "Masked singleton-stack projector requires MaskedImagePayload."
            )
        projected_data = SingletonStackImageDomainStrategy.project(value.data)
        if projected_data is value.data:
            return value
        return RuntimeImagePayloadContext(
            projected_data,
            SingletonStackImageDomainStrategy.project(value.mask),
            value.metadata.for_source_plane(0),
        ).payload()


def project_singleton_stack_image_domain(value: Any) -> Any:
    """Remove one leading singleton OpenHCS image-stack axis when present."""
    return SingletonStackImageDomainStrategy.project(value)


class AlignedImageStackKwargResolutionStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal strategy for resolving one slice-aligned runtime kwarg."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True
    value_type: ClassVar[type[Any] | None] = None
    value_type_label: ClassVar[str | None] = None

    @classmethod
    def for_value(
        cls,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> "AlignedImageStackKwargResolutionStrategy":
        for strategy_type in cls.strategy_types_for_nominal_value(value):
            strategy = strategy_type()
            if strategy.matches(value, resolver):
                return strategy
        raise TypeError(
            "No aligned image-stack kwarg resolution strategy accepted "
            f"{type(value).__name__}."
        )

    @abstractmethod
    def matches(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> bool:
        """Return whether this strategy owns the supplied runtime value."""

    @abstractmethod
    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        """Return the value in the current aligned slice context."""


class TupleAlignedKwargResolutionStrategy(AlignedImageStackKwargResolutionStrategy):
    """Resolve tuple-valued kwargs elementwise while preserving tuple structure."""

    value_type = tuple

    def matches(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> bool:
        del resolver
        return isinstance(value, tuple)

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        return tuple(resolver.resolve(item) for item in value)


class SourceSpatialAlignedKwargResolutionStrategy(
    AlignedImageStackKwargResolutionStrategy
):
    """Materialize source-spatial payloads in the current aligned slice domain."""

    def matches(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> bool:
        return resolver.domain_adapter(value) is not None

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        adapter = resolver.domain_adapter(value)
        if adapter is None:
            raise TypeError(
                "Source-spatial kwarg strategy requires a source-domain adapter."
            )
        resolved = adapter.materialize_for_slice(
            resolver.slice_index,
            resolver.slice_count,
        )
        reference_domain = resolver.reference_domain()
        if reference_domain is None:
            return resolved
        return reference_domain.extract_source_array(resolved)


class ImageMetadataPayloadAlignedKwargResolutionStrategy(
    SourceSpatialAlignedKwargResolutionStrategy
):
    """Resolve image payloads through their source-spatial metadata."""

    value_type = ImageMetadataPayload


class MaskedImagePayloadAlignedKwargResolutionStrategy(
    SourceSpatialAlignedKwargResolutionStrategy
):
    """Resolve masked image payloads through their source-spatial metadata."""

    value_type = MaskedImagePayload


class ObjectLabelAlignedKwargResolutionStrategy(
    SourceSpatialAlignedKwargResolutionStrategy
):
    """Resolve object labels by runtime-slice and source-spatial contracts."""

    def matches(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> bool:
        return (
            ObjectLabelRuntimeSliceStackContract.preserves_runtime_slice_stack(
                value,
                slice_count=resolver.slice_count,
            )
            or super().matches(value, resolver)
        )

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        if resolver.domain_adapter(value) is not None:
            return super().resolve(value, resolver)
        if ObjectLabelRuntimeSliceStackContract.preserves_runtime_slice_stack(
            value,
            slice_count=resolver.slice_count,
        ):
            from openhcs.core.runtime_slice_projection import (
                RuntimeSliceProjection,
                RuntimeProjectionAxis,
            )

            return RuntimeSliceProjection.value_for_slice(
                value,
                RuntimeProjectionAxis(
                    slice_index=resolver.slice_index,
                    extent=resolver.slice_count,
                ),
            )
        return super().resolve(value, resolver)


class ObjectLabelPayloadAlignedKwargResolutionStrategy(
    ObjectLabelAlignedKwargResolutionStrategy
):
    """Resolve object-label payloads through declared label-domain semantics."""

    value_type = ObjectLabelPayload


class ObjectLabelSetAlignedKwargResolutionStrategy(ObjectLabelAlignedKwargResolutionStrategy):
    """Resolve object-label sets through declared label-domain semantics."""

    value_type = ObjectLabelSet


class PayloadSliceAlignedKwargResolutionStrategy(
    AlignedImageStackKwargResolutionStrategy
):
    """Slice array-like payload kwargs when they share the aligned stack axis."""

    value_type = np.ndarray

    def matches(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> bool:
        return (
            resolver.enable_payload_alignment
            and len(payload_slices_for_alignment(value)) in {1, resolver.slice_count}
        )

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        slices = payload_slices_for_alignment(value)
        if len(slices) == resolver.slice_count:
            return resolver.without_slice_context().resolve(
                slices[resolver.slice_index]
            )
        return resolver.without_slice_context().resolve(slices[0])


class RuntimeSliceAlignedValueKwargResolutionStrategy(
    AlignedImageStackKwargResolutionStrategy
):
    """Select non-image values that explicitly declare runtime-slice alignment."""

    value_type = RuntimeSliceAlignedValueSet

    def matches(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> bool:
        del resolver
        return isinstance(value, RuntimeSliceAlignedValueSet)

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        if not isinstance(value, RuntimeSliceAlignedValueSet):
            raise TypeError(
                "RuntimeSliceAlignedValueKwargResolutionStrategy requires "
                "RuntimeSliceAlignedValueSet."
            )
        return value.value_for_aligned_slice(
            slice_index=resolver.slice_index,
            slice_count=resolver.slice_count,
        )


class AlwaysMatchingAlignedKwargResolutionMixin:
    """Resolution strategy applies whenever value-type selection reaches it."""

    def matches(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> bool:
        del value, resolver
        return True


class PassThroughAlignedKwargResolutionStrategy(
    AlwaysMatchingAlignedKwargResolutionMixin,
    AlignedImageStackKwargResolutionStrategy,
):
    """Leave non-slice-aligned kwargs in their native runtime domain."""

    value_type = object

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        del resolver
        return value


@dataclass(frozen=True, slots=True)
class ImageArrayShapeSemantics:
    """Nominal owner for OpenHCS/CellProfiler image-array shape conventions."""

    value: Any

    @property
    def ndim(self) -> int | None:
        shape = self.shape
        if shape is None:
            return None
        return len(shape)

    @property
    def shape(self) -> tuple[int, ...] | None:
        shape = tuple(int(axis) for axis in np.shape(self.value))
        if not shape:
            return None
        return shape

    @property
    def is_pairwise_slice_grid(self) -> bool:
        shape = self.shape
        return (
            self.ndim == 4
            and shape is not None
            and not is_color_image_stack(self.value)
            and shape[0] == shape[1]
        )

    @property
    def is_singleton_image_stack(self) -> bool:
        shape = self.shape
        return (
            self.ndim is not None
            and self.ndim > 0
            and shape is not None
            and shape[0] == 1
            and (
                is_grayscale_image_stack(self.value)
                or is_color_image_stack(self.value)
                or is_grayscale_volume_stack(self.value)
                or is_color_volume_stack(self.value)
            )
        )

    def collapse_pairwise_slice_grid(self) -> Any:
        """Collapse a square pairwise slice grid to its per-cycle diagonal stack."""
        array = np.asarray(self.value)
        if not type(self)(array).is_pairwise_slice_grid:
            raise ValueError(
                "Pairwise slice grid must be shaped (N, N, H, W); "
                f"got {self.shape!r}."
            )
        return np.stack(
            tuple(array[index, index] for index in range(array.shape[0])),
            axis=0,
        )

    def collapse_singleton_grayscale_plane_stack(self) -> Any:
        shape = self.shape
        if (
            self.ndim == 3
            and shape is not None
            and shape[0] == 1
            and not is_color_image_slice(self.value)
        ):
            return self.value[0]
        return self.value

    def shares_pairwise_slice_grid_axes_with(self, other: Any) -> bool:
        shape = self.shape
        other_shape = ImageArrayShapeSemantics(other).shape
        return shape is not None and other_shape is not None and shape[:2] == other_shape[:2]


class ImagePayloadExecutionMode(Enum):
    """How a runtime executor should interpret a resolved image payload."""

    NATURAL = "natural"
    FULL_STACK = "full_stack"
    ALIGNED_MULTI_IMAGE_STACK = "aligned_multi_image_stack"


@dataclass(frozen=True, slots=True)
class ImagePayloadComposition:
    """Resolved image payload plus its execution mode."""

    payload: Any
    execution_mode: ImagePayloadExecutionMode


@dataclass(slots=True)
class ImagePayloadBundleContext(ImagePayloadSequence):
    """Compose same-slice image bundle data, masks, and metadata together."""

    metadata_mode: ImagePayloadMetadataCompositionMode = (
        ImagePayloadMetadataCompositionMode.BUNDLE
    )

    @classmethod
    def from_payloads(
        cls,
        payloads: tuple[ImagePayloadMetadataInput, ...],
        *,
        metadata_mode: ImagePayloadMetadataCompositionMode = (
            ImagePayloadMetadataCompositionMode.BUNDLE
        ),
    ) -> "ImagePayloadBundleContext":
        normalized = tuple(
            _normalize_bundle_image_payload(payload) for payload in payloads
        )
        return cls(
            ImagePayloadSourceSpatialDomainAdapter
            .payloads_aligned_to_common_source_domain(normalized),
            metadata_mode=metadata_mode,
        )

    def compose(self) -> Any:
        composed = self.compose_unmasked(self.data_payloads)
        return (
            ImagePayloadMetadataCompositionRequest(
                self.payloads,
                mode=self.metadata_mode,
            )
            .metadata()
            .payload_with(
                composed,
                self.compose_mask(composed),
            )
        )

    def compose_mask(self, composed: Any) -> Any | None:
        combined = self.combined_mask()
        if combined is None:
            return None
        if self.mask_matches_composed_payload(combined, composed):
            return combined
        complete_masks = self.complete_masks
        if complete_masks is None:
            return combined
        return self.compose_unmasked(complete_masks).astype(bool, copy=False)

    def combined_mask(self) -> RuntimeArrayData | None:
        masks = self.present_masks
        if not masks:
            return None
        combined = np.asarray(masks[0], dtype=bool)
        for mask in masks[1:]:
            combined = np.logical_and(combined, np.asarray(mask, dtype=bool))
        return combined

    @staticmethod
    def mask_matches_composed_payload(mask: Any, composed: Any) -> bool:
        mask_shape = tuple(np.asarray(mask).shape)
        composed_shape = tuple(np.asarray(composed).shape)
        return mask_shape == composed_shape or mask_shape == composed_shape[-2:]

    @staticmethod
    def compose_unmasked(payloads: tuple[RuntimeArrayData, ...]) -> RuntimeArrayData:
        """Compose image payload arrays without mask/metadata wrapping."""
        memory_type = detect_memory_type(payloads[0])
        if _is_homogeneous_image_bundle(payloads):
            return RuntimeSliceStackRequest(
                slices=payloads,
                memory_type=memory_type,
            ).stack()
        return ImageBundleLayout.for_slices(payloads).stack(
            slices=payloads,
            memory_type=memory_type,
            gpu_id=0,
        )


@dataclass(slots=True)
class AlignedImageSliceContext:
    """Declared semantic context for one aligned image output slice."""

    MAIN_FLOW_OUTPUT_KIND: ClassVar[str] = "main"
    ANONYMOUS_MAIN_FLOW_OUTPUT_KEY: ClassVar[str] = "main"

    output_kind: str
    output_key: str
    artifact_kind: str | None = None

    @classmethod
    def main_flow(
        cls,
        output_key: str,
        *,
        artifact_kind: str | None = None,
    ) -> "AlignedImageSliceContext":
        """Return declared context for one main-flow output surface."""
        return cls(
            output_kind=cls.MAIN_FLOW_OUTPUT_KIND,
            output_key=output_key,
            artifact_kind=artifact_kind,
        )

    @classmethod
    def anonymous_main_flow(cls) -> "AlignedImageSliceContext":
        """Return context for ordinary unnamed main-flow output."""
        return cls.main_flow(cls.ANONYMOUS_MAIN_FLOW_OUTPUT_KEY)

    @property
    def is_anonymous_main_flow(self) -> bool:
        return (
            self.output_kind == self.MAIN_FLOW_OUTPUT_KIND
            and self.output_key == self.ANONYMOUS_MAIN_FLOW_OUTPUT_KEY
            and self.artifact_kind is None
        )

    def __post_init__(self) -> None:
        if not self.output_kind:
            raise ValueError("AlignedImageSliceContext.output_kind cannot be empty.")
        if not self.output_key:
            raise ValueError("AlignedImageSliceContext.output_key cannot be empty.")


@dataclass(slots=True)
class AlignedImageStack:
    """Per-slice multi-image bundles aligned to one OpenHCS stack."""

    slices: tuple[Any, ...]
    slice_contexts: tuple[AlignedImageSliceContext, ...] = ()

    def __post_init__(self) -> None:
        self.slices = tuple(self.slices)
        self.slice_contexts = tuple(self.slice_contexts)
        if not self.slices:
            raise ValueError("AlignedImageStack.slices cannot be empty.")
        if self.slice_contexts and len(self.slice_contexts) != len(self.slices):
            raise ValueError(
                "AlignedImageStack.slice_contexts must be empty or match slices; "
                f"got {len(self.slice_contexts)} context(s) for {len(self.slices)} slice(s)."
            )

    def slice_source_spatial_adapter(
        self,
        slice_index: int,
    ) -> SourceSpatialDomainAdapter | None:
        """Return the typed source-domain adapter for one execution slice."""
        return SourceSpatialDomainAdapter.for_value(self.slices[slice_index])

    def first_slice_source_spatial_adapter(self) -> SourceSpatialDomainAdapter | None:
        """Return the typed source-domain adapter for the first execution slice."""
        return self.slice_source_spatial_adapter(0)

    def aligned_slice(self, slice_index: int, slice_count: int) -> Any:
        """Return this aligned runtime value in an outer aligned slice context."""
        if len(self.slices) == slice_count:
            return self.slices[slice_index]
        if len(self.slices) == 1:
            return self.slices[0]
        raise ValueError(
            "Nested aligned image stack has incompatible slice count; "
            f"got {len(self.slices)} for outer count {slice_count}."
        )


class AlignedImageStackSingletonStackImageDomainStrategy(
    SingletonStackImageDomainStrategy
):
    """Project an aligned slice carrier into the stack payload it represents."""

    value_type = AlignedImageStack

    def project_value(self, value: Any) -> Any:
        if not isinstance(value, AlignedImageStack):
            raise TypeError(
                "Aligned-image stack projector requires AlignedImageStack."
            )
        slice_data = tuple(image_payload_data(payload) for payload in value.slices)
        memory_type = detect_memory_type(slice_data[0])
        stack = ImageBundleLayout.for_slices(slice_data).stack(
            slices=slice_data,
            memory_type=memory_type,
            gpu_id=0,
        )
        return stack_image_payload_context(value.slices, stack)


class NestedAlignedImageStackKwargResolutionStrategy(
    AlwaysMatchingAlignedKwargResolutionMixin,
    AlignedImageStackKwargResolutionStrategy
):
    """Select matching slices from kwargs that are already aligned stacks."""

    value_type = AlignedImageStack

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        return resolver.without_slice_context().resolve(
            value.aligned_slice(
                resolver.slice_index,
                resolver.slice_count,
            )
        )


class ImageBundleLayout(ABC, metaclass=AutoRegisterMeta):
    """Nominal layout for heterogeneous same-slice runtime image bundles."""

    __registry_key__ = "layout_key"
    __skip_if_no_key__ = True
    layout_key: ClassVar[str | None] = None

    @classmethod
    def for_slices(cls, slices: Sequence[Any]) -> "ImageBundleLayout":
        for layout_type in cls.__registry__.values():
            if layout_type.matches(slices):
                return layout_type()
        raise ValueError(
            "OpenHCS image bundles require 2D grayscale or HWC color slices; "
            f"got shapes {[ImageArrayShapeSemantics(slice_data).shape for slice_data in slices]!r}."
        )

    @classmethod
    @abstractmethod
    def matches(cls, slices: Sequence[Any]) -> bool:
        """Return whether this layout can compose the supplied slices."""

    @abstractmethod
    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        """Stack same-slice runtime images into one callable input bundle."""


class HomogeneousImageBundleLayout(ImageBundleLayout):
    """Stack same-kind grayscale or color image slices without promotion."""

    layout_key = "homogeneous"

    @classmethod
    def matches(cls, slices: Sequence[Any]) -> bool:
        if not all(_is_bundle_image_slice(slice_data) for slice_data in slices):
            return False
        shape_set = {tuple(np.shape(slice_data)) for slice_data in slices}
        if len(shape_set) != 1:
            return False
        return (
            all(is_grayscale_image_slice(slice_data) for slice_data in slices)
            or all(is_color_image_slice(slice_data) for slice_data in slices)
        )

    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        numpy_slices = tuple(
            NumpySliceConversion(slice_data, gpu_id).array()
            for slice_data in slices
        )
        stacked = np.stack(numpy_slices)
        if memory_type == MEMORY_TYPE_NUMPY:
            return stacked
        return convert_memory(
            data=stacked,
            source_type=MEMORY_TYPE_NUMPY,
            target_type=memory_type,
            gpu_id=gpu_id,
        )


class MixedColorImageBundleLayout(ImageBundleLayout):
    """Promote grayscale slices when a bundle mixes grayscale and color images."""

    layout_key = "mixed_color"

    @classmethod
    def matches(cls, slices: Sequence[Any]) -> bool:
        return (
            all(_is_bundle_image_slice(slice_data) for slice_data in slices)
            and any(is_color_image_slice(slice_data) for slice_data in slices)
            and any(is_grayscale_image_slice(slice_data) for slice_data in slices)
        )

    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        numpy_slices = tuple(
            NumpySliceConversion(slice_data, gpu_id).array()
            for slice_data in slices
        )
        spatial_shapes = {tuple(slice_data.shape[:2]) for slice_data in numpy_slices}
        if len(spatial_shapes) != 1:
            raise ValueError(
                "OpenHCS mixed color image bundles require stable spatial shape; "
                f"got {[slice_data.shape for slice_data in numpy_slices]!r}."
            )
        channel_counts = {
            int(slice_data.shape[-1])
            for slice_data in numpy_slices
            if is_color_image_slice(slice_data)
        }
        if len(channel_counts) != 1:
            raise ValueError(
                "OpenHCS mixed color image bundles require stable color channel "
                f"count; got {sorted(channel_counts)!r}."
            )
        channel_count = next(iter(channel_counts))
        stacked = np.stack(
            tuple(
                _promote_slice_to_color(slice_data, channel_count)
                for slice_data in numpy_slices
            )
        )
        if memory_type == MEMORY_TYPE_NUMPY:
            return stacked
        return convert_memory(
            data=stacked,
            source_type=MEMORY_TYPE_NUMPY,
            target_type=memory_type,
            gpu_id=gpu_id,
        )


@dataclass(frozen=True, slots=True)
class SingleSourceVolumePayload:
    """Predicate for one multi-plane source image that must remain whole."""

    data: Any
    metadata: ImagePayloadMetadata

    @property
    def should_preserve(self) -> bool:
        return (
            is_grayscale_volume_slice(self.data)
            and self.metadata.source_image_provenance_planes.has_values
            and self.has_single_source_identity
        )

    @property
    def has_single_source_identity(self) -> bool:
        paths = tuple(
            path
            for path in self.metadata.source_image_provenance_planes.paths
            if path is not None
        )
        return bool(paths) and len(set(paths)) == 1


@dataclass(frozen=True, slots=True)
class SliceCountAlignment:
    """Compatibility relation between one payload count and the shared maximum."""

    count: int
    max_slice_count: int

    @property
    def aligns_with_maximum(self) -> bool:
        return self.count > 0 and self.max_slice_count % self.count == 0


def compose_aligned_image_payload(
    owner_name: str,
    image_payloads: tuple[Any, ...],
    slice_contexts: Sequence[AlignedImageSliceContext] = (),
    metadata_mode: ImagePayloadMetadataCompositionMode = (
        ImagePayloadMetadataCompositionMode.BUNDLE
    ),
) -> ImagePayloadComposition:
    """Compose one or more image payloads into an executor-ready payload."""
    if not image_payloads:
        raise ValueError(f"{owner_name} cannot compose an empty image input set.")
    contexts = tuple(slice_contexts)
    if contexts:
        if len(contexts) != len(image_payloads):
            raise ValueError(
                f"{owner_name} declared {len(contexts)} slice context(s) for "
                f"{len(image_payloads)} image payload(s)."
            )
        return ImagePayloadComposition(
            payload=AlignedImageStack(
                slices=image_payloads,
                slice_contexts=contexts,
            ),
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        )
    if len(image_payloads) == 1:
        return ImagePayloadComposition(
            payload=image_payloads[0],
            execution_mode=ImagePayloadExecutionMode.NATURAL,
        )

    payload_slices = tuple(
        payload_slices_for_alignment(payload)
        for payload in image_payloads
    )
    slice_counts = tuple(len(slices) for slices in payload_slices)
    max_slice_count = max(slice_counts)
    invalid_counts = tuple(
        count
        for count in slice_counts
        if not SliceCountAlignment(count, max_slice_count).aligns_with_maximum
    )
    if invalid_counts:
        raise ValueError(
            f"{owner_name} cannot align multi-image inputs with incompatible "
            f"slice counts {slice_counts!r}."
        )

    if max_slice_count == 1:
        return ImagePayloadComposition(
            payload=ImagePayloadBundleContext.from_payloads(
                tuple(slices[0] for slices in payload_slices),
                metadata_mode=metadata_mode,
            ).compose(),
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
        )
    return ImagePayloadComposition(
        payload=AlignedImageStack(
            slices=tuple(
                ImagePayloadBundleContext.from_payloads(
                    tuple(
                        aligned_payload_slice(slices, slice_index)
                        for slices in payload_slices
                    ),
                    metadata_mode=metadata_mode,
                ).compose()
                for slice_index in range(max_slice_count)
            )
        ),
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )


def payload_slices_for_alignment(payload: Any) -> tuple[Any, ...]:
    """Return payload slices used for multi-source alignment."""
    if isinstance(payload, RuntimeSliceAlignedValueSet):
        return tuple(
            payload.value_for_slice(index)
            for index in range(payload.slice_count)
        )
    data = (
        ObjectLabelDenseDataStrategy.for_payload(payload).data(payload)
        if isinstance(payload, (ObjectLabelPayload, ObjectLabelSet))
        else image_payload_data(payload)
    )
    mask = image_payload_mask(payload)
    metadata = image_payload_metadata(payload)
    data_semantics = ImageArrayShapeSemantics(data)
    if data_semantics.is_pairwise_slice_grid:
        data = data_semantics.collapse_pairwise_slice_grid()
        if (
            mask is not None
            and ImageArrayShapeSemantics(mask).shares_pairwise_slice_grid_axes_with(
                image_payload_data(payload)
            )
        ):
            mask = ImageArrayShapeSemantics(mask).collapse_pairwise_slice_grid()
        payload = metadata.payload_with(data, mask)
    source_role = source_image_payload_role(payload)
    if ImageArrayShapeSemantics(data).ndim == 2:
        return (payload,)
    if (
        is_color_image_slice(data)
        or (
            source_role is not None
            and source_role.is_channel_last_source_plane(data)
        )
    ):
        return (payload,)
    if (
        source_role is None
        and SingleSourceVolumePayload(data, metadata).should_preserve
    ):
        return (payload,)
    if (
        source_role is not None
        and source_role.is_channel_last_source_stack(data)
    ):
        memory_type = detect_memory_type(data)
        slice_projector = ImagePayloadSliceProjector(mask=mask, metadata=metadata)
        return tuple(
            slice_projector.payload_for_slice(data_slice, index)
            for index, data_slice in enumerate(
                ImageStackLayoutUnstackRequest(data, memory_type, 0).slices()
            )
        )
    if is_image_stack(data):
        memory_type = detect_memory_type(data)
        slice_projector = ImagePayloadSliceProjector(mask=mask, metadata=metadata)
        return tuple(
            slice_projector.payload_for_slice(data_slice, index)
            for index, data_slice in enumerate(
                ImageStackLayoutUnstackRequest(data, memory_type, 0).slices()
            )
        )
    return (payload,)


def _payload_spatial_shape(payload: Any) -> tuple[int, ...]:
    array = np.asarray(payload)
    if is_color_image_slice(payload):
        return tuple(int(axis) for axis in array.shape[:2])
    if array.ndim < 2:
        raise ValueError(
            "Image payload slices require at least two spatial dimensions; "
            f"got {array.shape!r}."
        )
    return tuple(int(axis) for axis in array.shape[-2:])



def aligned_payload_slice(
    slices: tuple[Any, ...],
    slice_index: int,
) -> Any:
    """Return the payload slice for one aligned execution index."""
    if len(slices) == 1:
        return slices[0]
    return slices[slice_index % len(slices)]


def flatten_aligned_image_payload_slices(payload: Any) -> tuple[Any, ...]:
    """Return scalar image payload slices represented by an aligned output carrier."""
    if isinstance(payload, AlignedImageStack):
        return tuple(
            output_slice
            for aligned_slice in payload.slices
            for output_slice in payload_slices_for_alignment(aligned_slice)
        )
    return payload_slices_for_alignment(payload)


def flatten_aligned_image_slice_contexts(
    payload: Any,
) -> tuple[AlignedImageSliceContext, ...]:
    """Return per-output semantic context for flattened aligned image slices."""
    if not isinstance(payload, AlignedImageStack) or not payload.slice_contexts:
        return ()
    return tuple(
        slice_context
        for aligned_slice, slice_context in zip(
            payload.slices,
            payload.slice_contexts,
            strict=True,
        )
        for _output_slice in payload_slices_for_alignment(aligned_slice)
    )


def aligned_image_stack_kwargs(
    kwargs: Mapping[str, Any],
    slice_index: int,
    slice_count: int,
    reference_payload: Any | None = None,
) -> dict[str, Any]:
    """Slice runtime-array kwargs alongside an aligned image stack."""
    resolver = AlignedImageStackKwargResolver(
        slice_index=slice_index,
        slice_count=slice_count,
        reference_payload=reference_payload,
    )
    return {
        name: resolver.resolve(value)
        for name, value in kwargs.items()
    }


def _normalize_bundle_image_payload(payload: Any) -> Any:
    """Normalize one same-slice image payload before bundle composition."""
    data = image_payload_data(payload)
    mask = image_payload_mask(payload)
    metadata = image_payload_metadata(payload)
    data = ImageArrayShapeSemantics(data).collapse_singleton_grayscale_plane_stack()
    if mask is not None:
        mask = ImageArrayShapeSemantics(mask).collapse_singleton_grayscale_plane_stack()
    return metadata.payload_with(data, mask)


def payload_slice_count(payload: Any) -> int:
    """Return the number of aligned slices represented by one payload."""
    return len(payload_slices_for_alignment(payload))


def _is_bundle_image_slice(value: Any) -> bool:
    data = image_payload_data(value)
    return is_grayscale_image_slice(data) or is_color_image_slice(data)


def _is_homogeneous_image_bundle(slices: Sequence[Any]) -> bool:
    slices = tuple(image_payload_data(slice_data) for slice_data in slices)
    return (
        all(is_grayscale_image_slice(slice_data) for slice_data in slices)
        or all(is_grayscale_volume_slice(slice_data) for slice_data in slices)
        or all(is_color_image_slice(slice_data) for slice_data in slices)
    )


def _promote_slice_to_color(slice_data: np.ndarray, channel_count: int) -> np.ndarray:
    if is_color_image_slice(slice_data):
        return slice_data
    return np.repeat(slice_data[:, :, np.newaxis], channel_count, axis=2)
