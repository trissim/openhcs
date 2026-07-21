"""Generic aligned image-payload composition for multi-source runtime inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, ClassVar, Mapping

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.alias_property import AliasProperty
from openhcs.core.artifacts import ArtifactSpecRef
from openhcs.core.memory import (
    MEMORY_TYPE_NUMPY,
    convert_memory,
    detect_memory_type,
    stack_runtime_slices,
)
from openhcs.core.registry_strategies import (
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.runtime_image_values import ImagePayloadMetadata, ImagePayloadMetadataCarrier, ImagePayloadMetadataCompositionMode, image_payload_data, image_payload_mask, image_payload_mask_for_slice, image_payload_metadata, preserved_image_plane_projection, with_image_payload_data
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_object_labels import ObjectLabelRepresentation
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis, RuntimePlaneAxisProjector, RuntimePlaneAxisValueProjection, RuntimeSliceProjectableValue
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.source_spatial_domain import (
    SourceSpatialDomain,
    SourceSpatialDomainAdapter,
)


@dataclass(frozen=True, slots=True)
class ImagePayloadSourceSpatialDomainAdapter(SourceSpatialDomainAdapter):
    """Source-domain adapter for image payload data and masks."""

    value_type = ImagePayloadMetadataCarrier
    value_type_label = "image_payload"
    value: Any
    source_domain: SourceSpatialDomain
    domain = AliasProperty[SourceSpatialDomain]("source_domain")

    @property
    def array(self) -> Any:
        return image_payload_data(self.value)

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
            value,
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
            domain.with_missing_from(SourceSpatialDomain(origin_yx=(0, 0)))
            .with_fill_value(fill_value)
            .with_value_name(value_name)
        )

    @property
    def spatial_axes_yx(self) -> tuple[int, int]:
        axes = image_payload_metadata(self.value).spatial_axes_yx(self.value)
        if axes is None:
            raise ValueError(
                "Source-spatial image payload metadata does not declare two "
                f"spatial axes for shape {tuple(np.shape(self.array))!r}."
            )
        return axes

    @property
    def spatial_shape_yx(self) -> tuple[int, int]:
        shape = image_payload_metadata(self.value).spatial_shape_yx(self.value)
        if shape is None:
            raise ValueError(
                "Source-spatial image payloads require at least two dimensions, "
                f"got shape {tuple(np.shape(self.array))!r}."
            )
        return shape

    @classmethod
    def payloads_aligned_to_common_source_domain(
        cls,
        payloads: tuple[RuntimeArrayData, ...],
    ) -> tuple[RuntimeArrayData, ...]:
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
            cls.payload_in_source_domain(payload, source_domain) for payload in payloads
        )

    @classmethod
    def source_domain_adapters(
        cls,
        payloads: tuple[RuntimeArrayData, ...],
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
        payload: RuntimeArrayData,
        source_domain: SourceSpatialDomain,
    ) -> RuntimeArrayData:
        metadata = image_payload_metadata(payload)
        source_extent = source_domain.with_origin_yx(None)
        source_metadata = metadata.with_materialized_source_domain(source_extent)
        data = cls(
            payload,
            cls.domain_from_metadata(
                metadata,
                source_domain=source_extent,
                value_name="Image payload",
            ),
        ).materialize()
        return source_metadata.payload_with(
            data,
            cls.mask_in_source_domain(payload, metadata, source_extent),
        )

    @classmethod
    def mask_in_source_domain(
        cls,
        payload: RuntimeArrayData,
        metadata: ImagePayloadMetadata,
        source_domain: SourceSpatialDomain,
    ) -> RuntimeArrayData | None:
        mask = image_payload_mask(payload)
        if mask is None:
            return None
        return NumPyImagePayloadSourceSpatialDomainAdapter(
            mask,
            cls.domain_from_metadata(
                metadata,
                fill_value=False,
                source_domain=source_domain,
                value_name="Image mask",
            ),
        ).materialize()

    def value_in_payload_domain(
        self,
        target: SourceSpatialDomainAdapter,
    ) -> RuntimeArrayData:
        """Project this image payload into another declared payload domain."""
        materialized = self.payload_in_source_domain(self.value, target.domain)
        target_domain = SourceSpatialDomain(
            origin_yx=target.payload_domain.origin_yx,
            source_shape_yx=target.payload_domain.source_shape_yx,
            fill_value=self.domain.fill_value,
            value_name=self.domain.value_name,
        )
        metadata = replace(
            image_payload_metadata(materialized),
            source_spatial_domain=target_domain,
            physical_border_edges_yx=target_domain.physical_border_edges_for_shape(
                target.payload_domain.spatial_shape_yx
            ),
        )
        materialized_mask = image_payload_mask(materialized)
        return with_image_payload_data(
            materialized,
            target.extract_source_array(
                image_payload_data(materialized),
                spatial_axes_yx=self.spatial_axes_yx,
            ),
            mask=(
                None
                if materialized_mask is None
                else target.extract_source_array(
                    materialized_mask,
                    spatial_axes_yx=self.spatial_axes_yx,
                )
            ),
            metadata=metadata,
        )


class NumPyImagePayloadSourceSpatialDomainAdapter(
    ImagePayloadSourceSpatialDomainAdapter
):
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

    @property
    def array(self) -> Any:
        return self.value

    @property
    def spatial_shape_yx(self) -> tuple[int, int]:
        shape = ImagePayloadMetadata().spatial_shape_yx(self.value)
        if shape is None:
            raise ValueError(
                "NumPy source-spatial image payloads require at least two "
                f"dimensions, got shape {tuple(np.shape(self.value))!r}."
            )
        return shape

    def value_in_payload_domain(
        self,
        target: SourceSpatialDomainAdapter,
    ) -> Any:
        """Project a raw array without introducing a nominal payload carrier."""
        return target.extract_source_array(
            self.materialize(),
            spatial_axes_yx=self.spatial_axes_yx,
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
        return object_label_dense_array(self.value)

    @property
    def domain(self) -> SourceSpatialDomain:
        return self.value.object_label_source_spatial_domain().with_missing_from(
            SourceSpatialDomain(source_shape_yx=self.source_shape_override_yx)
        )

    @property
    def spatial_axes_yx(self) -> tuple[int, int]:
        array = np.asarray(self.array)
        if array.ndim < 2:
            raise ValueError(
                "Object-label source-spatial payloads require at least two "
                f"dimensions, got shape {array.shape!r}."
            )
        return array.ndim - 2, array.ndim - 1

    def dense_variant(self, labels: object) -> object:
        """Materialize one label variant through its nominal object-label carrier."""
        variant = self.value.with_labels(labels)
        return object_label_dense_array(variant)

    def value_in_payload_domain(
        self,
        target: SourceSpatialDomainAdapter,
    ) -> ObjectLabelValue:
        """Project every label variant into another declared payload domain."""
        target_domain = SourceSpatialDomain(
            origin_yx=target.payload_domain.origin_yx,
            source_shape_yx=target.payload_domain.source_shape_yx,
            fill_value=self.domain.fill_value,
            value_name=self.domain.value_name,
        )
        variants = self.value.variant_data.project(
            lambda labels: target.extract_source_array(
                self.domain.materialize(
                    self.dense_variant(labels),
                    spatial_axes_yx=self.spatial_axes_yx,
                ),
                spatial_axes_yx=self.spatial_axes_yx,
            )
        )
        return self.value.with_variants(
            variants,
            source_spatial_domain=target_domain,
            representation=ObjectLabelRepresentation.DENSE_LABELS,
        )


@dataclass(frozen=True, slots=True)
class AlignedImageStackKwargResolver:
    """Materialize one kwarg for a specific aligned image-stack slice."""

    projection_axis: "RuntimePlaneAxisValueProjection"
    reference_payload: Any | None = None

    def resolve(self, value: Any) -> Any:
        strategy = AlignedImageStackKwargResolutionStrategy.require_nominal_value(
            value,
            context="Aligned image-stack kwarg resolution",
        )
        return strategy.resolve(value, self)

    def resolve_source_spatial_value(self, value: Any) -> Any:
        """Project a nominal value into the declared reference payload domain."""
        if self.reference_payload is None:
            return value
        metadata = image_payload_metadata(self.reference_payload)
        source_shape = metadata.source_spatial_domain.source_shape_yx
        if source_shape is None:
            return value
        adapter = SourceSpatialDomainAdapter.for_value(
            value,
            source_shape_override_yx=source_shape,
        )
        reference = SourceSpatialDomainAdapter.for_value(self.reference_payload)
        if adapter is None or reference is None:
            return value
        return adapter.value_in_payload_domain(reference)


@dataclass(frozen=True, slots=True)
class ImagePayloadSliceProjector:
    """Project payload context from a parent image into one child image slice."""

    mask: RuntimeArrayData | None
    metadata: ImagePayloadMetadata

    def payloads_for_slices(
        self,
        slices: Sequence[RuntimeArrayData],
    ) -> list[RuntimeArrayData]:
        """Return payloads for every child slice using one projection pass."""
        if self.metadata.plane_axis is None:
            if len(slices) != 1:
                raise ValueError(
                    "Image payload produced multiple slices without a declared "
                    "plane axis."
                )
            return [self.metadata.payload_with(slices[0], self.mask)]
        metadata = self.metadata.with_indexed_source_plane_provenance(len(slices))
        masks = self._masks_for_slices(slices) if self.mask is not None else None
        return [
            metadata.for_leading_source_plane(index).payload_with(
                slice_data,
                None if masks is None else masks[index],
            )
            for index, slice_data in enumerate(slices)
        ]

    def _masks_for_slices(
        self,
        slices: Sequence[RuntimeArrayData],
    ) -> tuple[RuntimeArrayData, ...]:
        """Project masks after an explicit slice owner has fixed cardinality."""
        if self.mask is None:
            raise ValueError("Masked slice projection requires a mask payload.")
        mask_array = np.asarray(self.mask, dtype=bool)
        if mask_array.ndim == 0 or mask_array.shape[0] != len(slices):
            raise ValueError(
                "Image payload mask cardinality must exactly match the declared "
                f"plane axis: {mask_array.shape!r} for {len(slices)} slice(s)."
            )
        candidates = tuple(mask_array[index] for index in range(len(slices)))
        masks: list[RuntimeArrayData] = []
        for index, slice_data in enumerate(slices):
            metadata = self.metadata.for_leading_source_plane(index)
            if not metadata.mask_domain(slice_data).accepts(
                tuple(np.shape(candidates[index]))
            ):
                raise ValueError(
                    "Image payload mask shape must match the selected slice "
                    f"domain; got {tuple(np.shape(candidates[index]))!r} for "
                    f"{tuple(np.shape(slice_data))!r}."
                )
            masks.append(candidates[index])
        return tuple(masks)

    def payload_for_slice(
        self,
        data_slice: RuntimeArrayData,
        index: int,
    ) -> RuntimeArrayData:
        """Return a slice payload with mask and metadata in the slice domain."""
        metadata = self.metadata_for_slice(data_slice, index)
        mask = self.mask_for_slice(data_slice, index)
        payload: RuntimeArrayData = metadata.payload_with(data_slice, mask)
        return payload

    def metadata_for_slice(
        self,
        data_slice: RuntimeArrayData,
        index: int,
    ) -> ImagePayloadMetadata:
        """Return metadata for one explicitly selected child image slice."""
        del data_slice
        return self.metadata.for_leading_source_plane(index)

    def mask_for_slice(
        self,
        data_slice: RuntimeArrayData,
        index: int,
    ) -> RuntimeArrayData | None:
        """Return the parent mask projected into ``data_slice``'s domain."""
        return image_payload_mask_for_slice(
            mask=self.mask,
            metadata=self.metadata,
            data_slice=data_slice,
            plane_index=index,
        )


def stack_image_payload_context(
    image_payloads: Sequence[Any],
    stack: RuntimeArrayData,
    *,
    metadata_mode: ImagePayloadMetadataCompositionMode,
) -> Any:
    """Attach composed image metadata and masks to a freshly stacked payload."""
    payloads = tuple(image_payloads)
    metadata = ImagePayloadMetadata.compose(
        payloads,
        mode=metadata_mode,
    )
    return metadata.payload_with(stack, _stack_image_payload_mask(payloads, stack))


def stack_image_payloads(
    image_payloads: Sequence[Any],
    *,
    metadata_mode: ImagePayloadMetadataCompositionMode,
) -> Any:
    """Stack image payloads in their declared memory domain with full context."""

    payloads = tuple(image_payloads)
    if not payloads:
        raise ValueError("Cannot stack an empty image payload sequence.")
    arrays = tuple(image_payload_data(payload) for payload in payloads)
    memory_type = detect_memory_type(arrays[0])
    return stack_image_payload_context(
        payloads,
        stack_runtime_slices(arrays, memory_type, 0),
        metadata_mode=metadata_mode,
    )


def stack_image_payload_context_from_metadata(
    image_payloads: Sequence[Any],
    stack: RuntimeArrayData,
    metadata_by_payload: Sequence[ImagePayloadMetadata],
    *,
    metadata_mode: ImagePayloadMetadataCompositionMode,
) -> Any:
    """Attach composed image context using already resolved payload metadata."""
    payloads = tuple(image_payloads)
    metadata = ImagePayloadMetadata.compose(
        payloads,
        mode=metadata_mode,
        source_metadata=tuple(metadata_by_payload),
    )
    return metadata.payload_with(stack, _stack_image_payload_mask(payloads, stack))


def _stack_image_payload_mask(
    image_payloads: Sequence[Any],
    stack: RuntimeArrayData,
) -> RuntimeArrayData | None:
    masks = tuple(image_payload_mask(payload) for payload in image_payloads)
    if not any(mask is not None for mask in masks):
        return None
    payloads = tuple(image_payloads)
    stack_shape = tuple(np.shape(stack))
    if stack_shape[:1] != (len(payloads),):
        raise ValueError(
            "Image payload stack mask composition requires output stack "
            f"axis length {len(payloads)}, got stack shape {stack_shape!r}."
        )
    output_slice_domains = tuple(
        stack[slice_index] for slice_index in range(len(payloads))
    )
    resolved_masks = tuple(
        _complete_image_payload_mask(payload, slice_domain, mask)
        for payload, slice_domain, mask in zip(
            payloads,
            output_slice_domains,
            masks,
            strict=True,
        )
    )
    return stack_runtime_slices(
        resolved_masks,
        detect_memory_type(stack),
        0,
    )


def _complete_image_payload_mask(
    payload: Any,
    payload_data: RuntimeArrayData,
    mask: RuntimeArrayData | None,
) -> RuntimeArrayData:
    mask_domain = image_payload_metadata(payload).mask_domain(payload_data)
    if mask is not None:
        if not mask_domain.accepts(tuple(np.shape(mask))):
            raise ValueError(
                "Image payload mask must match the selected output slice "
                f"domain; got mask {tuple(np.shape(mask))!r} for slice "
                f"{tuple(np.shape(payload_data))!r}."
            )
        return mask
    return np.ones(mask_domain.default_mask_shape(), dtype=bool)


def unstack_image_payload_context(
    payload: Any,
    slices: Sequence[Any],
    *,
    default_plane_axis: RuntimePlaneAxis | None = None,
) -> list[Any]:
    """Attach one source plane of payload context to each unstacked image slice."""
    mask = image_payload_mask(payload)
    metadata = image_payload_metadata(payload)
    if mask is None and not metadata.has_values:
        return list(slices)
    if metadata.plane_axis is None and default_plane_axis is not None:
        metadata = replace(metadata, plane_axis=default_plane_axis)
    projector = ImagePayloadSliceProjector(mask=mask, metadata=metadata)
    return projector.payloads_for_slices(slices)


class AlignedImageStackKwargResolutionStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal strategy for resolving one slice-aligned runtime kwarg."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True
    __registry__: ClassVar[
        dict[str, type["AlignedImageStackKwargResolutionStrategy"]]
    ] = {}
    value_type: ClassVar[type[Any] | None] = None
    value_type_label: ClassVar[str | None] = None

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

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        return tuple(resolver.resolve(item) for item in value)


class ImagePayloadAlignedKwargResolutionStrategy(
    AlignedImageStackKwargResolutionStrategy
):
    """Resolve image payloads without discarding metadata or masks."""

    value_type = ImagePayloadMetadataCarrier

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        return resolver.resolve_source_spatial_value(value)


class ObjectLabelAlignedKwargResolutionStrategy(
    AlignedImageStackKwargResolutionStrategy
):
    """Resolve object labels by runtime-slice and source-spatial contracts."""

    value_type = ObjectLabelValue

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        if not isinstance(value, ObjectLabelValue):
            raise TypeError(
                "Object-label aligned kwarg resolution requires ObjectLabelValue."
            )
        slice_count = value.runtime_slice_plane_count()
        if slice_count is not None:
            if slice_count != resolver.projection_axis.axis_size:
                raise ValueError(
                    "Runtime-slice object-label cardinality must exactly match the "
                    f"declared projection axis: {slice_count} != "
                    f"{resolver.projection_axis.axis_size}."
                )
            from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

            projected = RuntimeSliceProjection.value_for_slice(
                value,
                resolver.projection_axis,
            )
        else:
            projected = value
        return resolver.resolve_source_spatial_value(projected)


class RuntimeSliceAlignedValueKwargResolutionStrategy(
    AlignedImageStackKwargResolutionStrategy
):
    """Select non-image values that explicitly declare runtime-slice alignment."""

    value_type = RuntimeSliceAlignedValueSet

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
        return resolver.projection_axis.aligned_value(value)


class RuntimeSliceProjectableAlignedKwargResolutionStrategy(
    AlignedImageStackKwargResolutionStrategy
):
    """Project values through their declared runtime-slice hook."""

    value_type = RuntimeSliceProjectableValue

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

        return RuntimeSliceProjection.value_for_slice(
            value,
            resolver.projection_axis,
        )


class PassThroughAlignedKwargResolutionStrategy(
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

    @property
    def plane_axis(self) -> RuntimePlaneAxis | None:
        """Return the axis declared by the composed payload owner."""
        if isinstance(self.payload, AlignedImageStack):
            return RuntimePlaneAxis.RUNTIME_SLICE
        return image_payload_metadata(self.payload).plane_axis

    def preserved_plane_projection(
        self,
        projector: RuntimePlaneAxisProjector,
        *,
        source_aliases: tuple[str, ...] = (),
    ) -> RuntimePlaneAxisValueProjection | None:
        """Return the complete projection owned by the composed payload axis."""

        axis = self.plane_axis
        if axis is None:
            return None
        return preserved_image_plane_projection(
            self.payload,
            projector,
            source_aliases,
        )


@dataclass(slots=True)
class ImagePayloadBundleContext:
    """Compose same-slice image bundle data, masks, and metadata together."""

    payloads: Sequence[RuntimeArrayData]
    metadata_mode: ImagePayloadMetadataCompositionMode = (
        ImagePayloadMetadataCompositionMode.BUNDLE
    )

    def __post_init__(self) -> None:
        self.payloads = tuple(self.payloads)
        if not self.payloads:
            raise ValueError("ImagePayloadBundleContext.payloads cannot be empty.")
        declared_axes = tuple(
            (
                index,
                metadata.plane_axis,
                metadata.source_image_names,
                tuple(np.shape(image_payload_data(payload))),
            )
            for index, (payload, metadata) in enumerate(
                zip(self.payloads, self.source_metadata, strict=True)
            )
            if metadata.plane_axis is not None
        )
        if declared_axes:
            raise ValueError(
                "Same-slice image bundles require every payload plane axis to be "
                f"projected before composition; got {declared_axes!r}."
            )

    @property
    def source_metadata(self) -> tuple[ImagePayloadMetadata, ...]:
        return tuple(image_payload_metadata(payload) for payload in self.payloads)

    @property
    def data_payloads(self) -> tuple[RuntimeArrayData, ...]:
        return tuple(image_payload_data(payload) for payload in self.payloads)

    @property
    def masks(self) -> tuple[RuntimeArrayData | None, ...]:
        return tuple(image_payload_mask(payload) for payload in self.payloads)

    @property
    def present_masks(self) -> tuple[RuntimeArrayData, ...]:
        return tuple(mask for mask in self.masks if mask is not None)

    @classmethod
    def from_payloads(
        cls,
        payloads: tuple[RuntimeArrayData, ...],
        *,
        metadata_mode: ImagePayloadMetadataCompositionMode = (
            ImagePayloadMetadataCompositionMode.BUNDLE
        ),
    ) -> "ImagePayloadBundleContext":
        return cls(
            ImagePayloadSourceSpatialDomainAdapter.payloads_aligned_to_common_source_domain(
                payloads
            ),
            metadata_mode=metadata_mode,
        )

    def compose(self) -> Any:
        composed = self.compose_unmasked(self.data_payloads)
        metadata = ImagePayloadMetadata.compose(
            self.payloads,
            mode=self.metadata_mode,
        )
        return metadata.payload_with(
            composed,
            self.compose_mask(composed, metadata),
        )

    def compose_mask(
        self,
        composed: Any,
        metadata: ImagePayloadMetadata,
    ) -> Any | None:
        masks = self.present_masks
        if not masks:
            return None
        shared_spatial_shape = metadata.mask_domain(composed).shared_spatial_mask_shape
        if shared_spatial_shape is not None and all(
            tuple(np.shape(mask)) == shared_spatial_shape for mask in masks
        ):
            return self.combined_mask()
        resolved_masks = tuple(
            _complete_image_payload_mask(payload, data, mask)
            for payload, data, mask in zip(
                self.payloads,
                self.data_payloads,
                self.masks,
                strict=True,
            )
        )
        return stack_runtime_slices(
            resolved_masks,
            detect_memory_type(resolved_masks[0]),
            0,
        ).astype(bool, copy=False)

    def combined_mask(self) -> RuntimeArrayData | None:
        masks = self.present_masks
        if not masks:
            return None
        mask_shapes = tuple(tuple(np.shape(mask)) for mask in masks)
        if any(shape != mask_shapes[0] for shape in mask_shapes[1:]):
            raise ValueError(
                "Image bundle mask intersection requires one exact declared "
                f"spatial mask shape; got {mask_shapes!r}."
            )
        combined = np.asarray(masks[0], dtype=bool)
        for mask in masks[1:]:
            combined = np.logical_and(combined, np.asarray(mask, dtype=bool))
        return combined

    def compose_unmasked(
        self,
        payloads: tuple[RuntimeArrayData, ...],
    ) -> RuntimeArrayData:
        """Compose image payload arrays without mask/metadata wrapping."""
        memory_type = detect_memory_type(payloads[0])
        channel_axes = tuple(
            metadata.normalized_source_channel_axis(payload)
            for payload, metadata in zip(
                payloads,
                self.source_metadata,
                strict=True,
            )
        )
        declared_channel_count = sum(axis is not None for axis in channel_axes)
        if declared_channel_count in {0, len(payloads)}:
            return stack_runtime_slices(payloads, memory_type, 0)
        return self.compose_mixed_channel_payloads(
            payloads,
            channel_axes=channel_axes,
            memory_type=memory_type,
        )

    @staticmethod
    def compose_mixed_channel_payloads(
        payloads: tuple[RuntimeArrayData, ...],
        *,
        channel_axes: tuple[int | None, ...],
        memory_type: str,
    ) -> RuntimeArrayData:
        """Promote channel-free payloads using declared channel-axis semantics."""
        numpy_payloads = tuple(
            np.asarray(
                convert_memory(
                    data=payload,
                    source_type=detect_memory_type(payload),
                    target_type=MEMORY_TYPE_NUMPY,
                    gpu_id=0,
                )
            )
            for payload in payloads
        )
        declared_axes = tuple(axis for axis in channel_axes if axis is not None)
        channel_axis = declared_axes[0]
        if any(axis != channel_axis for axis in declared_axes[1:]):
            raise ValueError(
                "Image bundle payloads declare conflicting source channel axes: "
                f"{channel_axes!r}."
            )
        channel_counts = tuple(
            int(payload.shape[channel_axis])
            for payload, axis in zip(numpy_payloads, channel_axes, strict=True)
            if axis is not None
        )
        channel_count = channel_counts[0]
        if any(count != channel_count for count in channel_counts[1:]):
            raise ValueError(
                "Image bundle payloads declare incompatible channel counts: "
                f"{channel_counts!r}."
            )
        source_shapes = tuple(
            (
                tuple(payload.shape)
                if axis is None
                else tuple(
                    dimension
                    for index, dimension in enumerate(payload.shape)
                    if index != axis
                )
            )
            for payload, axis in zip(numpy_payloads, channel_axes, strict=True)
        )
        if any(shape != source_shapes[0] for shape in source_shapes[1:]):
            raise ValueError(
                "Image bundle payloads must share one declared source image shape: "
                f"{source_shapes!r}."
            )
        promoted = tuple(
            (
                payload
                if axis is not None
                else np.repeat(
                    np.expand_dims(payload, axis=channel_axis),
                    channel_count,
                    axis=channel_axis,
                )
            )
            for payload, axis in zip(numpy_payloads, channel_axes, strict=True)
        )
        stacked = np.stack(promoted, axis=0)
        if memory_type == MEMORY_TYPE_NUMPY:
            return stacked
        return convert_memory(
            data=stacked,
            source_type=MEMORY_TYPE_NUMPY,
            target_type=memory_type,
            gpu_id=0,
        )


@dataclass(frozen=True, slots=True)
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

    def contextualize_image_payload(self, payload: RuntimeArrayData) -> RuntimeArrayData:
        """Attach this named main-flow identity to an image payload."""

        if self.is_anonymous_main_flow:
            return payload
        metadata = image_payload_metadata(payload)
        return metadata.with_source_provenance(
            metadata.source_provenance.with_derived_source_image_names(
                (self.output_key,)
            )
        ).payload_with(
            image_payload_data(payload),
            image_payload_mask(payload),
        )

    def matches_artifact_ref(self, artifact_ref: ArtifactSpecRef) -> bool:
        """Return whether this context carries one exact compiled artifact ref."""

        if not isinstance(artifact_ref, ArtifactSpecRef):
            raise TypeError(
                "Aligned image context lookup requires ArtifactSpecRef, got "
                f"{type(artifact_ref).__name__}."
            )
        return (
            self.output_kind == self.MAIN_FLOW_OUTPUT_KIND
            and self.output_key == artifact_ref.name
            and self.artifact_kind == artifact_ref.artifact_type.require_value()
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
        if len(self.slices) != slice_count:
            raise ValueError(
                "Nested aligned image stack cardinality must exactly match the "
                f"declared outer axis: {len(self.slices)} != {slice_count}."
            )
        return self.slices[slice_index]

    def with_slices(self, slices: Sequence[Any]) -> "AlignedImageStack":
        """Replace payload slices while preserving the concrete alignment owner."""

        return type(self)(tuple(slices), self.slice_contexts)

    def output_payload(
        self,
        artifact_ref: ArtifactSpecRef,
    ) -> Any | None:
        """Return the payload carried for one exact compiled artifact ref."""

        if not self.slice_contexts:
            return None
        matches = tuple(
            payload
            for payload, context in zip(
                self.slices,
                self.slice_contexts,
                strict=True,
            )
            if context.matches_artifact_ref(artifact_ref)
        )
        if len(matches) > 1:
            raise ValueError(
                "Aligned image stack carries duplicate main-flow output context "
                f"for {artifact_ref!r}."
            )
        return matches[0] if matches else None


@dataclass(slots=True)
class ImageOutputBundle(AlignedImageStack):
    """Named main-flow image outputs sharing one invocation context."""

    def __post_init__(self) -> None:
        AlignedImageStack.__post_init__(self)
        if not self.slice_contexts or any(
            context.is_anonymous_main_flow for context in self.slice_contexts
        ):
            raise ValueError(
                "ImageOutputBundle requires one named context per image output."
            )


def pack_aligned_image_outputs(
    outputs: Sequence[Any],
    *,
    slice_contexts: Sequence[AlignedImageSliceContext] = (),
) -> Any:
    """Pack one or more image outputs into the single canonical return slot."""

    packed = tuple(outputs)
    if not packed:
        raise ValueError("Canonical image output packing requires at least one output.")
    contexts = tuple(slice_contexts)
    if contexts and len(contexts) != len(packed):
        raise ValueError(
            "Canonical image output contexts must match output count; "
            f"got {len(contexts)} context(s) for {len(packed)} output(s)."
        )
    if contexts:
        packed = tuple(
            context.contextualize_image_payload(output)
            for output, context in zip(packed, contexts, strict=True)
        )
    if len(packed) == 1:
        return packed[0]
    if contexts:
        return ImageOutputBundle(packed, contexts)
    return AlignedImageStack(packed)


class NestedAlignedImageStackKwargResolutionStrategy(
    AlignedImageStackKwargResolutionStrategy
):
    """Select matching slices from kwargs that are already aligned stacks."""

    value_type = AlignedImageStack

    def resolve(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> Any:
        return resolver.resolve(
            value.aligned_slice(
                resolver.projection_axis.require_plane_index(),
                resolver.projection_axis.axis_size,
            )
        )


def compose_aligned_image_payload(
    owner_name: str,
    image_payloads: tuple[Any, ...],
    slice_contexts: Sequence[AlignedImageSliceContext] = (),
    stack_broadcast_source_indices: Sequence[int | None] = (),
    metadata_mode: ImagePayloadMetadataCompositionMode = (
        ImagePayloadMetadataCompositionMode.BUNDLE
    ),
) -> ImagePayloadComposition:
    """Compose one or more image payloads into an executor-ready payload."""
    if not image_payloads:
        raise ValueError(f"{owner_name} cannot compose an empty image input set.")
    broadcast_sources = tuple(stack_broadcast_source_indices)
    if broadcast_sources and len(broadcast_sources) != len(image_payloads):
        raise ValueError(
            f"{owner_name} declared {len(broadcast_sources)} stack-broadcast "
            f"source(s) for {len(image_payloads)} image payload(s)."
        )
    if not broadcast_sources:
        broadcast_sources = (None,) * len(image_payloads)
    for input_index, source_index in enumerate(broadcast_sources):
        if source_index is None:
            continue
        if type(source_index) is not int:
            raise TypeError(
                f"{owner_name} stack-broadcast source index for input "
                f"{input_index} must be an int or None, got "
                f"{type(source_index).__name__}."
            )
        if not 0 <= source_index < len(image_payloads):
            raise ValueError(
                f"{owner_name} stack-broadcast source index {source_index} for "
                f"input {input_index} is outside its {len(image_payloads)} inputs."
            )
        if source_index == input_index:
            raise ValueError(
                f"{owner_name} image input {input_index} cannot broadcast from itself."
            )
    contexts = tuple(slice_contexts)
    if contexts:
        if any(source is not None for source in broadcast_sources):
            raise ValueError(
                f"{owner_name} cannot combine explicit slice contexts with "
                "input-stack broadcast declarations."
            )
        if len(contexts) != len(image_payloads):
            raise ValueError(
                f"{owner_name} declared {len(contexts)} slice context(s) for "
                f"{len(image_payloads)} image payload(s)."
            )
        return ImagePayloadComposition(
            payload=ImageOutputBundle(
                slices=image_payloads,
                slice_contexts=contexts,
            ),
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        )
    aligned_payloads = tuple(
        payload
        for payload in image_payloads
        if isinstance(payload, AlignedImageStack)
    )
    if aligned_payloads:
        aligned_inputs = tuple(
            isinstance(payload, AlignedImageStack) for payload in image_payloads
        )
        slice_counts = tuple(len(payload.slices) for payload in aligned_payloads)
        if len(set(slice_counts)) != 1:
            raise ValueError(
                f"{owner_name} aligned image input cardinalities must match "
                f"exactly; got {slice_counts!r}."
            )
        slice_count = slice_counts[0]
        unowned_inputs = tuple(
            input_index
            for input_index, aligned in enumerate(aligned_inputs)
            if not aligned
            and (
                broadcast_sources[input_index] is None
                or not aligned_inputs[broadcast_sources[input_index]]
            )
        )
        if unowned_inputs and slice_count != 1:
            raise ValueError(
                f"{owner_name} cannot mix aligned and unaligned image payloads; "
                "unaligned inputs require an explicit aligned stack owner. "
                f"Unowned input indices: {unowned_inputs!r}."
            )
        if len(aligned_payloads) == 1:
            if len(image_payloads) == 1:
                return ImagePayloadComposition(
                    payload=aligned_payloads[0],
                    execution_mode=(
                        ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
                    ),
                )
        declared_contexts = tuple(
            payload.slice_contexts
            for payload in aligned_payloads
            if payload.slice_contexts
        )
        if declared_contexts and any(
            contexts != declared_contexts[0]
            for contexts in declared_contexts[1:]
        ):
            raise ValueError(
                f"{owner_name} aligned image inputs carry conflicting exact "
                f"slice contexts: {declared_contexts!r}."
            )
        return ImagePayloadComposition(
            payload=AlignedImageStack(
                slices=tuple(
                    ImagePayloadBundleContext.from_payloads(
                        tuple(
                            (
                                payload.slices[slice_index]
                                if isinstance(payload, AlignedImageStack)
                                else payload
                            )
                            for payload in image_payloads
                        ),
                        metadata_mode=metadata_mode,
                    ).compose()
                    for slice_index in range(slice_count)
                ),
                slice_contexts=(declared_contexts[0] if declared_contexts else ()),
            ),
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        )
    if len(image_payloads) == 1:
        return ImagePayloadComposition(
            payload=image_payloads[0],
            execution_mode=ImagePayloadExecutionMode.NATURAL,
        )
    from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

    runtime_slice_counts = tuple(
        RuntimeSliceProjection.slice_count_from_values((payload,))
        for payload in image_payloads
    )
    declared_runtime_slice_counts = tuple(
        count for count in runtime_slice_counts if count is not None
    )
    if declared_runtime_slice_counts:
        if len(set(declared_runtime_slice_counts)) != 1:
            raise ValueError(
                f"{owner_name} runtime-slice image input cardinalities must match "
                f"exactly; got {declared_runtime_slice_counts!r}."
            )
        slice_count = declared_runtime_slice_counts[0]
        unowned_inputs = tuple(
            input_index
            for input_index, count in enumerate(runtime_slice_counts)
            if count is None
            and (
                broadcast_sources[input_index] is None
                or runtime_slice_counts[broadcast_sources[input_index]] is None
            )
        )
        if unowned_inputs and slice_count != 1:
            raise ValueError(
                f"{owner_name} cannot mix runtime-slice-aligned and unaligned "
                "image payloads; unaligned inputs require an explicit "
                "runtime-slice owner. "
                f"Unowned input indices: {unowned_inputs!r}."
            )
        return ImagePayloadComposition(
            payload=AlignedImageStack(
                slices=tuple(
                    ImagePayloadBundleContext.from_payloads(
                        tuple(
                            (
                                RuntimeSliceProjection.value_for_slice(
                                    payload,
                                    RuntimePlaneAxisValueProjection.from_selected_plane(
                                        axis=RuntimePlaneAxis.RUNTIME_SLICE,
                                        plane_index=slice_index,
                                        axis_size=slice_count,
                                    ),
                                )
                                if runtime_slice_counts[input_index] is not None
                                else payload
                            )
                            for input_index, payload in enumerate(image_payloads)
                        ),
                        metadata_mode=metadata_mode,
                    ).compose()
                    for slice_index in range(slice_count)
                )
            ),
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        )
    return ImagePayloadComposition(
        payload=ImagePayloadBundleContext.from_payloads(
            image_payloads,
            metadata_mode=metadata_mode,
        ).compose(),
        execution_mode=ImagePayloadExecutionMode.FULL_STACK,
    )


def payload_slices_for_alignment(payload: Any) -> tuple[Any, ...]:
    """Return slices declared by a nominal runtime-alignment owner."""
    if isinstance(payload, AlignedImageStack):
        return payload.slices
    if isinstance(payload, RuntimeSliceAlignedValueSet):
        return tuple(
            payload.value_for_slice(index) for index in range(payload.slice_count)
        )
    if isinstance(payload, ImagePayloadMetadataCarrier):
        from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

        slice_count = RuntimeSliceProjection.slice_count_from_values((payload,))
        if slice_count is None:
            return (payload,)
        return tuple(
            RuntimeSliceProjection.value_for_slice(
                payload,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE,
                    plane_index=index,
                    axis_size=slice_count,
                ),
            )
            for index in range(slice_count)
        )
    if isinstance(payload, ObjectLabelValue):
        slice_count = payload.runtime_slice_plane_count()
        if slice_count is None:
            return (payload,)
        from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

        return tuple(
            RuntimeSliceProjection.value_for_slice(
                payload,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE,
                    plane_index=index,
                    axis_size=slice_count,
                ),
            )
            for index in range(slice_count)
        )
    return (payload,)


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
        projection_axis=RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=slice_index,
            axis_size=slice_count,
        ),
        reference_payload=reference_payload,
    )
    return {name: resolver.resolve(value) for name, value in kwargs.items()}


def payload_slice_count(payload: Any) -> int:
    """Return the number of aligned slices represented by one payload."""
    return len(payload_slices_for_alignment(payload))
