"""OpenHCS main-flow carrier policy for CellProfiler module execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute

from openhcs.constants.constants import AllComponents, VariableComponents
from openhcs.core.aligned_image_payload import (
    compose_aligned_image_payload,
    flatten_aligned_image_payload_slices,
    payload_slice_count,
)
from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.image_shapes import is_image_stack
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import detect_memory_type
from openhcs.core.runtime_values import (
    DerivedImagePayloadContext,
    ImagePayloadMetadataCompositionMode,
    RuntimeImagePayloadContext,
    RuntimeImageSourceIdentityCompleteness,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    project_image_mask_to_data_domain,
)
from openhcs.core.steps.function_output_identity import (
    FunctionOutputIdentityAuthority,
    FunctionOutputIdentityCache,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerMeasurementImage,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerRuntimeValue
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryKey,
)
from openhcs.microscopes.microscope_interfaces import FilenameParser


class CellProfilerMainFlowReplacementPolicyMixin(ABC):
    """Declaration-owned main-flow replacement policy for image outputs."""

    @abstractmethod
    def replaces_main_flow(
        self,
        image_outputs: tuple[ArtifactSpec, ...],
    ) -> bool:
        """Return True when the declared module image output owns downstream flow."""


class CellProfilerMainFlowReplacementPolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Nominal policy for mapping declared CellProfiler image outputs to main flow."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    declaration_policy_bases = (CellProfilerMainFlowReplacementPolicyMixin,)


class ContractImageOutputMainFlowReplacementPolicy(
    CellProfilerMainFlowReplacementPolicy
):
    """Default CellProfiler modules publish their sole image output to main flow."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def replaces_main_flow(
        self,
        image_outputs: tuple[ArtifactSpec, ...],
    ) -> bool:
        return len(image_outputs) == 1


class MeasurementSourceImageCardinality(IntEnum):
    """Closed cardinality axis for source images measured by a CP module."""

    NONE = 0
    SINGLE = 1
    MULTIPLE = 2

    @classmethod
    def from_images(
        cls,
        source_images: tuple[CellProfilerMeasurementImage, ...],
    ) -> "MeasurementSourceImageCardinality":
        return cls(min(len(source_images), cls.MULTIPLE.value))


class CellProfilerMeasurementMainFlowSurface(ABC, metaclass=AutoRegisterMeta):
    """Nominal output policy for one measurement source-image cardinality."""

    __registry_key__ = "cardinality"
    __skip_if_no_key__ = True

    cardinality: ClassVar[MeasurementSourceImageCardinality | None] = None

    @classmethod
    def for_source_images(
        cls,
        source_images: tuple[CellProfilerMeasurementImage, ...],
    ) -> "CellProfilerMeasurementMainFlowSurface":
        cardinality = MeasurementSourceImageCardinality.from_images(source_images)
        return cls.__registry__[cardinality]()

    @abstractmethod
    def output_image(
        self,
        *,
        input_image: CellProfilerRuntimeValue,
        source_images: tuple[CellProfilerMeasurementImage, ...],
        variable_components: tuple[VariableComponents, ...],
        parser: FilenameParser | None,
        identity_cache: FunctionOutputIdentityCache,
    ) -> CellProfilerRuntimeValue:
        """Return the image carrier published to OpenHCS main flow."""


class NoSourceMeasurementMainFlowSurface(CellProfilerMeasurementMainFlowSurface):
    """Keep current OpenHCS main-flow image for object-only measurements."""

    cardinality = MeasurementSourceImageCardinality.NONE

    def output_image(
        self,
        *,
        input_image: CellProfilerRuntimeValue,
        source_images: tuple[CellProfilerMeasurementImage, ...],
        variable_components: tuple[VariableComponents, ...],
        parser: FilenameParser | None,
        identity_cache: FunctionOutputIdentityCache,
    ) -> CellProfilerRuntimeValue:
        del source_images, variable_components, parser, identity_cache
        return input_image


class SingleSourceMeasurementMainFlowSurface(CellProfilerMeasurementMainFlowSurface):
    """Publish the single source image surface measured by CellProfiler."""

    cardinality = MeasurementSourceImageCardinality.SINGLE

    def output_image(
        self,
        *,
        input_image: CellProfilerRuntimeValue,
        source_images: tuple[CellProfilerMeasurementImage, ...],
        variable_components: tuple[VariableComponents, ...],
        parser: FilenameParser | None,
        identity_cache: FunctionOutputIdentityCache,
    ) -> CellProfilerRuntimeValue:
        payload = source_images[0].payload
        if (
            image_payload_metadata(payload).source_image_provenance_planes.count > 1
            and not MainFlowSurfaceAddressability(
                payload,
                variable_components=variable_components,
                parser=parser,
                identity_cache=identity_cache,
            ).addressable
        ):
            return input_image
        return payload


class MultipleSourceMeasurementMainFlowSurface(CellProfilerMeasurementMainFlowSurface):
    """Publish the composed source image surface measured by CellProfiler."""

    cardinality = MeasurementSourceImageCardinality.MULTIPLE

    def output_image(
        self,
        *,
        input_image: CellProfilerRuntimeValue,
        source_images: tuple[CellProfilerMeasurementImage, ...],
        variable_components: tuple[VariableComponents, ...],
        parser: FilenameParser | None,
        identity_cache: FunctionOutputIdentityCache,
    ) -> CellProfilerRuntimeValue:
        composed = compose_aligned_image_payload(
            "CellProfilerMeasurementMainFlow",
            tuple(image.payload for image in source_images),
            metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
        ).payload
        if MainFlowSurfaceAddressability(
            composed,
            variable_components=variable_components,
            parser=parser,
            identity_cache=identity_cache,
        ).addressable:
            return composed
        return input_image


@dataclass(frozen=True, slots=True)
class MainFlowSurfaceAddressability:
    """Addressability contract for publishing source surfaces to main flow."""

    payload: CellProfilerRuntimeValue
    variable_components: tuple[VariableComponents, ...]
    parser: FilenameParser | None
    identity_cache: FunctionOutputIdentityCache

    @property
    def addressable(self) -> bool:
        if self.parser is not None:
            try:
                identity = FunctionOutputIdentityAuthority.identity_from_metadata_with_cache(
                    self.parser,
                    image_payload_metadata(self.payload),
                    variable_components=self.variable_components,
                    identity_cache=self.identity_cache,
                )
            except ValueError:
                return False
            return identity is not None and identity.extension is not None

        source_identities = tuple(
            image_payload_metadata(output_slice)
            .source_provenance
            .scalar_source_identity
            .identity
            for output_slice in flatten_aligned_image_payload_slices(self.payload)
        )
        if not all(
            path is not None or component_metadata is not None
            for path, component_metadata in source_identities
        ):
            return False
        component_metadatas = tuple(
            component_metadata
            for _path, component_metadata in source_identities
            if component_metadata is not None
        )
        if len(component_metadatas) != len(source_identities):
            return False
        return (
            self._component_metadata_is_stack_addressable(component_metadatas)
            and len(set(source_identities)) == len(source_identities)
        )

    def _component_metadata_is_stack_addressable(
        self,
        component_metadatas: tuple[tuple[tuple[str, str], ...], ...],
    ) -> bool:
        metadata_maps = tuple(dict(metadata) for metadata in component_metadatas)
        variable_component_values = frozenset(
            component.value
            for component in self.variable_components
            if component.value is not None
        )
        semantic_keys = (
            frozenset(key for metadata in metadata_maps for key in metadata)
            - variable_component_values
        )
        ordered_keys = tuple(
            component.value
            for component in AllComponents
            if component.value in semantic_keys
        )
        extra_keys = tuple(sorted(semantic_keys - frozenset(ordered_keys)))
        for key in (*ordered_keys, *extra_keys):
            values = tuple(metadata.get(key) for metadata in metadata_maps)
            if any(key not in metadata for metadata in metadata_maps):
                return False
            if len(frozenset(values)) != 1:
                return False
        return True


class CellProfilerMeasurementMainFlowPolicy:
    """Select the main-flow image surface for CellProfiler measurement modules."""

    def output_image(
        self,
        *,
        input_image: CellProfilerRuntimeValue,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
        variable_components: tuple[VariableComponents, ...],
        parser: FilenameParser | None,
        identity_cache: FunctionOutputIdentityCache,
    ) -> CellProfilerRuntimeValue:
        source_images = self.source_domain_images(measurement_images)
        return CellProfilerMeasurementMainFlowSurface.for_source_images(
            source_images
        ).output_image(
            input_image=input_image,
            source_images=source_images,
            variable_components=variable_components,
            parser=parser,
            identity_cache=identity_cache,
        )

    def source_domain_images(
        self,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
    ) -> tuple[CellProfilerMeasurementImage, ...]:
        return tuple(
            image
            for image in measurement_images
            if image.has_source_identity
        )


CELLPROFILER_MEASUREMENT_MAIN_FLOW = CellProfilerMeasurementMainFlowPolicy()


class CellProfilerSideEffectMainFlowPolicy:
    """Publish the source image set consumed by non-replacing image modules."""

    def output_image(
        self,
        *,
        current_image: CellProfilerRuntimeValue,
        image_request: CellProfilerImageRequest,
        variable_components: tuple[VariableComponents, ...],
        parser: FilenameParser | None,
        identity_cache: FunctionOutputIdentityCache,
    ) -> CellProfilerRuntimeValue:
        if (
            image_request.has_source_identity
            and image_request.publishes_side_effect_main_flow
        ):
            if (
                image_payload_metadata(
                    image_request.payload
                ).source_image_provenance_planes.count > 1
                and not MainFlowSurfaceAddressability(
                    image_request.payload,
                    variable_components=variable_components,
                    parser=parser,
                    identity_cache=identity_cache,
                ).addressable
            ):
                return current_image
            return image_request.payload
        return current_image


CELLPROFILER_SIDE_EFFECT_MAIN_FLOW = CellProfilerSideEffectMainFlowPolicy()


def cellprofiler_main_flow_output(
    input_image: CellProfilerRuntimeValue,
    output_image: CellProfilerRuntimeValue,
) -> CellProfilerRuntimeValue:
    """Apply OpenHCS image-stack layout and provenance to a CP output image."""
    input_data = image_payload_data(input_image)
    output_data = image_payload_data(output_image)
    output_mask = image_payload_mask(output_image)
    output_metadata = image_payload_metadata(output_image)
    if not is_image_stack(input_data):
        return output_image
    memory_type = detect_memory_type(input_data)
    stacked = ImageStackLayout.stack_function_result_for_input_stack(
        output_data,
        input_stack=input_data,
        memory_type=memory_type,
        gpu_id=0,
    )
    stacked_mask = (
        ImageStackLayout.stack_function_result_for_input_stack(
            output_mask,
            input_stack=input_data,
            memory_type=memory_type,
            gpu_id=0,
        )
        if output_mask is not None
        else output_mask
    )
    stacked_mask = project_image_mask_to_data_domain(stacked_mask, stacked)
    output_payload = RuntimeImagePayloadContext(
        data=stacked,
        mask=stacked_mask,
        metadata=output_metadata,
    ).payload()
    if RuntimeImageSourceIdentityCompleteness(output_payload).complete():
        return output_payload
    return DerivedImagePayloadContext(input_image, output_payload).payload()


def cellprofiler_recorded_image_main_flow_output(
    *,
    current_image: CellProfilerRuntimeValue,
    invocation_image: CellProfilerRuntimeValue,
    recorded_image: CellProfilerRuntimeValue,
) -> CellProfilerRuntimeValue:
    """Apply main-flow layout to a declared image output already recorded."""
    result = cellprofiler_main_flow_output(invocation_image, recorded_image)
    if is_image_stack(image_payload_data(result)):
        return result
    if (
        payload_slice_count(current_image) == 1
        and is_image_stack(image_payload_data(current_image))
    ):
        return cellprofiler_main_flow_output(current_image, result)
    return result
