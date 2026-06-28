"""CellProfiler runtime execution-mode policy contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
import numpy as np

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_color_volume_slice,
    is_color_volume_stack,
    is_grayscale_volume_stack,
)
from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.runtime_values import (
    ImageMetadataPayload,
    MaskedImagePayload,
    ObjectLabelDenseDataStrategy,
    ObjectLabelValue,
    image_payload_data,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryKey,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    build_structuring_element,
)




class CellProfilerInvocationExecutionModePolicyMixin:
    """Declaration-owned execution-mode behavior for CellProfiler modules."""

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: RuntimeInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del image
        return default


class CellProfilerInvocationExecutionModePolicy(
    CellProfilerInvocationExecutionModePolicyMixin,
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Registered fallback policy root for CellProfiler execution mode."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    declaration_policy_bases = (CellProfilerInvocationExecutionModePolicyMixin,)




class DefaultInvocationExecutionModePolicy(CellProfilerInvocationExecutionModePolicy):
    """Use the execution mode implied by image payload composition."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value




class CellProfilerPayloadSpatialRankStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Resolve spatial rank from nominal runtime payload types."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def resolve_rank(cls, value: CellProfilerRuntimeValue) -> int | None:
        strategy = cls.for_nominal_value(value)
        if strategy is None:
            return None
        return strategy.spatial_rank(value)

    @abstractmethod
    def spatial_rank(self, value: CellProfilerRuntimeValue) -> int | None:
        """Return the spatial rank, excluding color channels, when known."""




class DenseArrayPayloadSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank for dense image arrays."""

    value_type = np.ndarray

    def spatial_rank(self, value: CellProfilerRuntimeValue) -> int | None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Dense array rank strategy requires ndarray.")
        if is_color_image_slice(value) or is_color_image_stack(value):
            return 2
        if (
            is_grayscale_volume_stack(value)
            or is_color_volume_slice(value)
            or is_color_volume_stack(value)
        ):
            return 3
        return int(value.ndim)




class DataBackedPayloadSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank through payload objects that expose image data."""

    def spatial_rank(self, value: CellProfilerRuntimeValue) -> int | None:
        value_type = type(self).value_type
        if value_type is None:
            expected_type_name = "declared value_type"
        else:
            expected_type_name = value_type.__name__
        if value_type is None or not isinstance(value, value_type):
            raise TypeError(
                f"{type(self).__name__} requires {expected_type_name}."
            )
        return CellProfilerPayloadSpatialRankStrategy.resolve_rank(value.data)




class MaskedImagePayloadSpatialRankStrategy(DataBackedPayloadSpatialRankStrategy):
    """Resolve spatial rank through masked-image payload data."""

    value_type = MaskedImagePayload




class ImageMetadataPayloadSpatialRankStrategy(DataBackedPayloadSpatialRankStrategy):
    """Resolve spatial rank through image metadata payload data."""

    value_type = ImageMetadataPayload




class ObjectLabelValueSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank for nominal object-label runtime values."""

    value_type = ObjectLabelValue

    def spatial_rank(self, value: CellProfilerRuntimeValue) -> int | None:
        if not isinstance(value, ObjectLabelValue):
            raise TypeError(
                "Object-label rank strategy requires an object-label runtime value."
            )
        return ObjectLabelDenseDataStrategy.spatial_rank(value)




@dataclass(frozen=True, slots=True)
class InvocationSpatialRankCandidates:
    """Spatial-rank observations available for one CellProfiler invocation."""

    ranks: tuple[int, ...]

    def max_rank_or_none(self) -> int | None:
        if not self.ranks:
            return None
        return max(self.ranks)




@dataclass(frozen=True, slots=True)
class StructuringElementFootprintRequest:
    """Typed morphology footprint request from CellProfiler kwargs."""

    shape_kwarg: ClassVar[str] = "structuring_element"
    size_kwarg: ClassVar[str] = "size"

    shape: CellProfilerRuntimeValue
    size: int

    @classmethod
    def from_kwargs(cls, kwargs: CellProfilerKwargs) -> "StructuringElementFootprintRequest":
        return cls(
            shape=kwargs[cls.shape_kwarg],
            size=int(kwargs[cls.size_kwarg]),
        )

    def footprint(self) -> np.ndarray:
        return build_structuring_element(self.shape, self.size)




class VolumetricInputExecutionModePolicy(CellProfilerInvocationExecutionModePolicyMixin):
    """Run full-stack when the nominal image payload contains a Z volume."""

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: RuntimeInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del kwargs, invocation_options
        if self.is_volumetric_payload(image):
            return ImagePayloadExecutionMode.FULL_STACK
        return default

    def is_volumetric_payload(self, image: CellProfilerRuntimeValue) -> bool:
        spatial_rank = self.spatial_rank(image)
        return spatial_rank is not None and spatial_rank >= 3

    def spatial_rank(self, image: CellProfilerRuntimeValue) -> int | None:
        data_rank = CellProfilerPayloadSpatialRankStrategy.resolve_rank(image)
        if data_rank is not None:
            return data_rank
        return CellProfilerPayloadSpatialRankStrategy.resolve_rank(
            image_payload_data(image)
        )

    def invocation_spatial_rank(
        self,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
    ) -> int | None:
        return InvocationSpatialRankCandidates(
            tuple(
                rank
                for rank in (
                    self.spatial_rank(image),
                    *(
                        CellProfilerPayloadSpatialRankStrategy.resolve_rank(value)
                        for value in kwargs.values()
                    ),
                )
                if rank is not None
            )
        ).max_rank_or_none()




class StructuringElementExecutionModePolicy(VolumetricInputExecutionModePolicy):
    """Match CellProfiler morphology dispatch from typed footprint rank."""

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: RuntimeInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del invocation_options
        spatial_rank = self.invocation_spatial_rank(image=image, kwargs=kwargs)
        if spatial_rank is None or spatial_rank < 3:
            return default
        footprint = StructuringElementFootprintRequest.from_kwargs(kwargs).footprint()
        if footprint.ndim == spatial_rank:
            return ImagePayloadExecutionMode.FULL_STACK
        return default
