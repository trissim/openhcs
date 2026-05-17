"""Generic aligned image-payload composition for multi-source runtime inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
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
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import MEMORY_TYPE_NUMPY, convert_memory, detect_memory_type
from openhcs.core.registry_strategies import (
    GeneratedLeafClassSpec,
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_semantics import (
    SourceSpatialDomain,
    SourceSpatialDomainAdapter,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImageMetadataPayload,
    MaskedImagePayload,
    ObjectLabelDenseDataStrategy,
    ObjectLabelMeasurementPayloadStrategy,
    ObjectLabelPayload,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelSet,
    compose_image_payload_metadata,
    image_payload_metadata,
    image_payload_data,
    image_payload_mask,
    project_image_mask_to_data_domain,
    image_payload_with_context,
)


@dataclass(frozen=True, slots=True)
class ImagePayloadSourceSpatialDomainAdapter(SourceSpatialDomainAdapter):
    """Source-domain adapter for image payload data and masks."""

    value_type = (ImageMetadataPayload, MaskedImagePayload)
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
        if not isinstance(value, (ImageMetadataPayload, MaskedImagePayload)):
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
        value_name: str,
    ) -> SourceSpatialDomain:
        return SourceSpatialDomain(
            origin_yx=metadata.spatial_origin_yx or (0, 0),
            source_shape_yx=metadata.source_spatial_shape_yx,
            fill_value=fill_value,
            value_name=value_name,
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

    value_type = (ObjectLabelPayload, ObjectLabelSet)
    value_type_label = "object_label_payload"
    value: ObjectLabelPayload | ObjectLabelSet
    source_shape_override_yx: tuple[int, int] | None = None

    @classmethod
    def for_value(
        cls,
        value: Any,
        *,
        source_shape_override_yx: tuple[int, int] | None = None,
    ) -> "ObjectLabelPayloadSourceSpatialDomainAdapter | None":
        if not isinstance(value, (ObjectLabelPayload, ObjectLabelSet)):
            return None
        return cls(value, source_shape_override_yx=source_shape_override_yx)

    @property
    def array(self) -> Any:
        if isinstance(self.value, ObjectLabelSet):
            payload = self.value.runtime_payload()
            return payload.labels if isinstance(payload, ObjectLabelPayload) else payload
        return self.value.labels

    @property
    def domain(self) -> SourceSpatialDomain:
        return SourceSpatialDomain(
            origin_yx=self.value.spatial_origin_yx,
            source_shape_yx=(
                self.value.source_spatial_shape_yx
                or self.source_shape_override_yx
            ),
            value_name="Object-label",
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
        if metadata.source_spatial_shape_yx is None:
            return None
        return SourceSpatialDomainAdapter.for_value(
            value,
            source_shape_override_yx=metadata.source_spatial_shape_yx,
        )

    def reference_domain(self) -> SourceSpatialDomainAdapter | None:
        if self.reference_payload is None:
            return None
        return SourceSpatialDomainAdapter.for_value(self.reference_payload)


@dataclass(frozen=True, slots=True)
class ImagePayloadSliceProjector:
    """Project payload context from a parent image into one child image slice."""

    mask: Any | None
    metadata: ImagePayloadMetadata

    def payload_for_slice(self, data_slice: Any, index: int) -> Any:
        """Return a slice payload with mask and metadata in the slice domain."""
        return image_payload_with_context(
            data=data_slice,
            mask=self.mask_for_slice(data_slice, index),
            metadata=self.metadata.for_channel(index),
        )

    def mask_for_slice(self, data_slice: Any, index: int) -> Any | None:
        """Return the parent mask projected into ``data_slice``'s domain."""
        if self.mask is None:
            return None
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
        return image_payload_with_context(
            data=projected_data,
            metadata=value.metadata.for_channel(0),
        )


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
        return image_payload_with_context(
            data=projected_data,
            mask=SingletonStackImageDomainStrategy.project(value.mask),
            metadata=value.metadata.for_channel(0),
        )


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
        for value_type in type(value).__mro__:
            for strategy_type in cls.registered_strategy_types():
                if strategy_type.value_type is not value_type:
                    continue
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


for _source_spatial_aligned_kwarg_strategy in (
    GeneratedLeafClassSpec(
        "ImageMetadataPayloadAlignedKwargResolutionStrategy",
        SourceSpatialAlignedKwargResolutionStrategy,
        attributes={
            "__doc__": "Resolve image payloads through their source-spatial metadata.",
            "value_type": ImageMetadataPayload,
        },
    ),
    GeneratedLeafClassSpec(
        "MaskedImagePayloadAlignedKwargResolutionStrategy",
        SourceSpatialAlignedKwargResolutionStrategy,
        attributes={
            "__doc__": "Resolve masked image payloads through their source-spatial metadata.",
            "value_type": MaskedImagePayload,
        },
    ),
):
    _source_spatial_aligned_kwarg_strategy.declare_in(globals())


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
            labels = np.asarray(ObjectLabelDenseDataStrategy.for_payload(value).data(value))
            return ObjectLabelMeasurementPayloadStrategy.for_source(value).with_labels(
                value,
                labels[resolver.slice_index],
            )
        return super().resolve(value, resolver)


for _object_label_aligned_kwarg_strategy in (
    GeneratedLeafClassSpec(
        "ObjectLabelPayloadAlignedKwargResolutionStrategy",
        ObjectLabelAlignedKwargResolutionStrategy,
        attributes={
            "__doc__": "Resolve object-label payloads through declared label-domain semantics.",
            "value_type": ObjectLabelPayload,
        },
    ),
    GeneratedLeafClassSpec(
        "ObjectLabelSetAlignedKwargResolutionStrategy",
        ObjectLabelAlignedKwargResolutionStrategy,
        attributes={
            "__doc__": "Resolve object-label sets through declared label-domain semantics.",
            "value_type": ObjectLabelSet,
        },
    ),
):
    _object_label_aligned_kwarg_strategy.declare_in(globals())


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


class PassThroughAlignedKwargResolutionStrategy(AlignedImageStackKwargResolutionStrategy):
    """Leave non-slice-aligned kwargs in their native runtime domain."""

    value_type = object

    def matches(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> bool:
        del value, resolver
        return True

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
        return getattr(self.value, "ndim", None)

    @property
    def shape(self) -> tuple[int, ...] | None:
        shape = getattr(self.value, "shape", None)
        if shape is None:
            return None
        return tuple(int(axis) for axis in shape)

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


@dataclass(frozen=True, slots=True)
class ImageBundleSourceDomainAligner:
    """Align same-slice image bundle members into one source XY domain."""

    image_payloads: tuple[Any, ...]

    def align(self) -> tuple[Any, ...]:
        adapters = self.payload_adapters()
        source_shape = SourceSpatialDomainAdapter.common_source_shape_yx(adapters)
        if (
            source_shape is None
            or not SourceSpatialDomainAdapter.requires_source_domain_alignment(adapters)
        ):
            return self.image_payloads
        return tuple(
            self.payload_in_source_domain(payload, source_shape)
            for payload in self.image_payloads
        )

    def payload_adapters(self) -> tuple[SourceSpatialDomainAdapter, ...]:
        adapters: list[SourceSpatialDomainAdapter] = []
        for payload in self.image_payloads:
            adapter = SourceSpatialDomainAdapter.for_value(payload)
            if not isinstance(adapter, ImagePayloadSourceSpatialDomainAdapter):
                raise TypeError(
                    "Image bundle alignment requires image payload adapters."
                )
            adapters.append(adapter)
        return tuple(adapters)

    def payload_in_source_domain(
        self,
        payload: Any,
        source_shape_yx: tuple[int, int],
    ) -> Any:
        metadata = image_payload_metadata(payload)
        source_shape = metadata.source_spatial_shape_yx or source_shape_yx
        source_metadata = metadata.with_spatial_crop(
            input_shape_yx=source_shape,
            output_shape_yx=source_shape,
            offset_yx=(
                -metadata.spatial_origin_yx[0],
                -metadata.spatial_origin_yx[1],
            )
            if metadata.spatial_origin_yx is not None
            else (0, 0),
            physical_border_edges_yx=(True, True, True, True),
        )
        data = ImagePayloadSourceSpatialDomainAdapter(
            image_payload_data(payload),
            ImagePayloadSourceSpatialDomainAdapter.domain_from_metadata(
                metadata,
                value_name="Image payload",
            ),
        ).materialize()
        mask = self.mask_in_source_domain(payload, metadata)
        return image_payload_with_context(
            data=data,
            mask=mask,
            metadata=source_metadata,
        )

    def mask_in_source_domain(
        self,
        payload: Any,
        metadata: ImagePayloadMetadata,
    ) -> Any | None:
        mask = image_payload_mask(payload)
        if mask is None:
            return None
        return ImagePayloadSourceSpatialDomainAdapter(
            mask,
            ImagePayloadSourceSpatialDomainAdapter.domain_from_metadata(
                metadata,
                fill_value=False,
                value_name="Image mask",
            ),
        ).materialize()


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


@dataclass(frozen=True, slots=True)
class ImagePayloadBundleContext:
    """Compose same-slice image bundle data, masks, and metadata together."""

    image_payloads: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_payloads", tuple(self.image_payloads))
        if not self.image_payloads:
            raise ValueError("ImagePayloadBundleContext.image_payloads cannot be empty.")

    @classmethod
    def from_payloads(
        cls,
        image_payloads: tuple[Any, ...],
    ) -> "ImagePayloadBundleContext":
        normalized = tuple(
            _normalize_bundle_image_payload(payload) for payload in image_payloads
        )
        return cls(ImageBundleSourceDomainAligner(normalized).align())

    def compose(self) -> Any:
        composed = self.compose_unmasked(
            tuple(image_payload_data(payload) for payload in self.image_payloads)
        )
        return image_payload_with_context(
            composed,
            mask=self.compose_mask(composed),
            metadata=compose_image_payload_metadata(self.image_payloads),
        )

    def compose_mask(self, composed: Any) -> Any | None:
        combined = self.combined_mask()
        if combined is None:
            return None
        if self.mask_matches_composed_payload(combined, composed):
            return combined
        complete_masks = self.complete_masks()
        if complete_masks is None:
            return combined
        return self.compose_unmasked(complete_masks).astype(bool, copy=False)

    def combined_mask(self) -> Any | None:
        masks = tuple(
            mask
            for mask in (
                image_payload_mask(payload)
                for payload in self.image_payloads
            )
            if mask is not None
        )
        if not masks:
            return None
        combined = np.asarray(masks[0], dtype=bool)
        for mask in masks[1:]:
            combined = np.logical_and(combined, np.asarray(mask, dtype=bool))
        return combined

    def complete_masks(self) -> tuple[Any, ...] | None:
        masks = tuple(image_payload_mask(payload) for payload in self.image_payloads)
        if any(mask is None for mask in masks):
            return None
        if len(masks) != len(self.image_payloads):
            return None
        return tuple(np.asarray(mask, dtype=bool) for mask in masks)

    @staticmethod
    def mask_matches_composed_payload(mask: Any, composed: Any) -> bool:
        mask_shape = tuple(np.asarray(mask).shape)
        composed_shape = tuple(np.asarray(composed).shape)
        return mask_shape == composed_shape or mask_shape == composed_shape[-2:]

    @staticmethod
    def compose_unmasked(image_payloads: tuple[Any, ...]) -> Any:
        """Compose image payload arrays without mask/metadata wrapping."""
        memory_type = detect_memory_type(image_payloads[0])
        if _is_homogeneous_image_bundle(image_payloads):
            return ImageStackLayout.for_slices(image_payloads).stack(
                slices=image_payloads,
                memory_type=memory_type,
                gpu_id=0,
            )
        return ImageBundleLayout.for_slices(image_payloads).stack(
            slices=image_payloads,
            memory_type=memory_type,
            gpu_id=0,
        )


@dataclass(frozen=True, slots=True)
class AlignedImageStack:
    """Per-slice multi-image bundles aligned to one OpenHCS stack."""

    slices: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slices", tuple(self.slices))
        if not self.slices:
            raise ValueError("AlignedImageStack.slices cannot be empty.")

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


class NestedAlignedImageStackKwargResolutionStrategy(
    AlignedImageStackKwargResolutionStrategy
):
    """Select matching slices from kwargs that are already aligned stacks."""

    value_type = AlignedImageStack

    def matches(
        self,
        value: Any,
        resolver: AlignedImageStackKwargResolver,
    ) -> bool:
        del value, resolver
        return True

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
            f"got shapes {[getattr(slice_data, 'shape', None) for slice_data in slices]!r}."
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
            _as_numpy_slice(slice_data, gpu_id)
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


def compose_aligned_image_payload(
    owner_name: str,
    image_payloads: tuple[Any, ...],
) -> ImagePayloadComposition:
    """Compose one or more image payloads into an executor-ready payload."""
    if not image_payloads:
        raise ValueError(f"{owner_name} cannot compose an empty image input set.")
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
        if count not in {1, max_slice_count}
    )
    if invalid_counts:
        raise ValueError(
            f"{owner_name} cannot align multi-image inputs with incompatible "
            f"slice counts {slice_counts!r}."
        )

    if max_slice_count == 1:
        return ImagePayloadComposition(
            payload=ImagePayloadBundleContext.from_payloads(
                tuple(slices[0] for slices in payload_slices)
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
                    )
                ).compose()
                for slice_index in range(max_slice_count)
            )
        ),
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )


def payload_slices_for_alignment(payload: Any) -> tuple[Any, ...]:
    """Return payload slices used for multi-source alignment."""
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
        payload = image_payload_with_context(data=data, mask=mask, metadata=metadata)
    if hasattr(data, "ndim") and data.ndim == 2:
        return (payload,)
    if is_color_image_slice(data):
        return (payload,)
    if _is_single_source_volume_payload(data, metadata):
        return (payload,)
    if is_image_stack(data):
        memory_type = detect_memory_type(data)
        slice_projector = ImagePayloadSliceProjector(mask=mask, metadata=metadata)
        return tuple(
            slice_projector.payload_for_slice(data_slice, index)
            for index, data_slice in enumerate(
                ImageStackLayout.for_stack(data).unstack(
                    array=data,
                    memory_type=memory_type,
                    gpu_id=0,
                )
            )
        )
    return (payload,)


def _is_single_source_volume_payload(data: Any, metadata: Any) -> bool:
    """Return True for one multi-plane source image, not an OpenHCS stack."""
    return (
        is_grayscale_volume_slice(data)
        and metadata.source_path is not None
        and not metadata.channel_source_paths
    )


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
    return slices[slice_index]


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
    return image_payload_with_context(data=data, mask=mask, metadata=metadata)


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


def _as_numpy_slice(slice_data: Any, gpu_id: int) -> np.ndarray:
    slice_data = image_payload_data(slice_data)
    source_type = detect_memory_type(slice_data)
    if source_type == MEMORY_TYPE_NUMPY:
        return slice_data
    return convert_memory(
        data=slice_data,
        source_type=source_type,
        target_type=MEMORY_TYPE_NUMPY,
        gpu_id=gpu_id,
    )


def _promote_slice_to_color(slice_data: np.ndarray, channel_count: int) -> np.ndarray:
    if is_color_image_slice(slice_data):
        return slice_data
    return np.repeat(slice_data[:, :, np.newaxis], channel_count, axis=2)
