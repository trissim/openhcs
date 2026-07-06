"""CellProfiler output value and source-context policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
import numpy as np

from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.runtime_semantics import ObjectLabelDomainScope
from openhcs.core.source_metadata import SourceVoxelSpacing
from openhcs.core.runtime_values import (
    DerivedImagePayloadContext,
    ImagePayloadMetadataCarrier,
    ImagePayloadMetadataInput,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectLabelValueConstructionContext,
    ObjectLabelValue,
    ObjectLabelVariantData,
    RuntimeArrayPayload,
    SourceImageObjectLabelBuildRequest,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.runtime.image_payload_collapse import (
    SINGLETON_STACK_OUTPUT_COLLAPSE,
)
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryKey,
)


class CellProfilerOutputContextSelectionMixin(ABC):
    """Shared nominal selection algorithm for output-context strategy roots."""

    @classmethod
    def for_value(cls, value: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
        strategy = cls.for_nominal_value(value)
        if strategy is not None:
            return strategy
        raise TypeError(cls.unsupported_value_message(value))

    @classmethod
    @abstractmethod
    def unsupported_value_message(cls, value: CellProfilerRuntimeValue) -> str:
        """Return the type error message for unsupported output payloads."""


class CellProfilerOutputContextMessageMixin:
    """Shared unsupported-value message for CellProfiler output context roots."""

    output_value_subject: ClassVar[str]
    supported_output_values: ClassVar[str] = "image payloads or numpy arrays"

    @classmethod
    def unsupported_value_message(cls, value: CellProfilerRuntimeValue) -> str:
        return (
            f"CellProfiler {cls.output_value_subject} outputs must be "
            f"{cls.supported_output_values}; got {type(value).__name__}."
        )


class CellProfilerImageOutputSourcePayloadPolicyMixin(ABC):
    """Declaration-owned source-payload policy for image outputs."""

    @abstractmethod
    def source_payload(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue | None:
        """Return source context for one image output artifact."""


class CellProfilerImageOutputSourcePayloadPolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Resolve metadata-bearing source payloads for declared image outputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    declaration_policy_bases = (CellProfilerImageOutputSourcePayloadPolicyMixin,)


class DefaultImageOutputSourcePayloadPolicy(CellProfilerImageOutputSourcePayloadPolicy):
    """Use the module's unique declared primary image as output provenance."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def source_payload(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue | None:
        return request.primary_image_output_source_payload()


class CellProfilerImageOutputValuePolicyMixin(ABC):
    """Declaration-owned output-value policy for image outputs."""

    @abstractmethod
    def output_value(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue:
        """Return the value to record for one image output artifact."""


class CellProfilerImageOutputValuePolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Normalize declared image output values before source context is attached."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    declaration_policy_bases = (CellProfilerImageOutputValuePolicyMixin,)


class DefaultImageOutputValuePolicy(CellProfilerImageOutputValuePolicy):
    """Record image outputs exactly as produced by CellProfiler."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def output_value(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue:
        return request.output_value


@dataclass(frozen=True, slots=True)
class CellProfilerObjectLabelOutputSourceContext:
    """Source provenance and CP parent-image context for one object-label output."""

    source_payload: CellProfilerRuntimeValue
    parent_image_payload: CellProfilerRuntimeValue | None

    @property
    def source_metadata(self):
        """Return metadata for the declared source payload."""
        return image_payload_metadata(self.source_payload)

    @property
    def parent_image_source_voxel_spacing(self) -> SourceVoxelSpacing:
        """Return spacing stamped from the CP parent image, or absence."""
        if self.parent_image_payload is None:
            return SourceVoxelSpacing()
        return image_payload_metadata(self.parent_image_payload).source_voxel_spacing


class CellProfilerObjectLabelOutputSourceContextPolicyMixin(ABC):
    """Declaration-owned source-payload policy for object-label outputs."""

    @abstractmethod
    def source_context(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerObjectLabelOutputSourceContext:
        """Return source and CP parent-image context for one object-label output."""


class CellProfilerObjectLabelOutputSourceContextPolicy(
    CellProfilerModulePolicyLookupMixin,
    CellProfilerObjectLabelOutputSourceContextPolicyMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Resolve metadata-bearing source payloads for declared object-label outputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    declaration_policy_bases = (CellProfilerObjectLabelOutputSourceContextPolicyMixin,)


class DefaultObjectLabelOutputSourceContextPolicy(
    CellProfilerObjectLabelOutputSourceContextPolicy
):
    """Use the default declared source context for object-label outputs."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def source_context(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerObjectLabelOutputSourceContext:
        source_payload = request.default_object_label_output_source_payload()
        parent_payload = request.metadata_bearing_primary_image_source_payload()
        return CellProfilerObjectLabelOutputSourceContext(
            source_payload,
            parent_payload or source_payload,
        )


class InputObjectLabelOutputSourceContextPolicyMixin(
    CellProfilerObjectLabelOutputSourceContextPolicyMixin,
    ABC,
):
    """Use input object-label context for object-label outputs derived from objects."""

    def source_context(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerObjectLabelOutputSourceContext:
        source_payload = request.input_object_label_output_source_payload()
        return CellProfilerObjectLabelOutputSourceContext(source_payload, source_payload)


class InputObjectLabelWithoutParentImageOutputSourceContextPolicyMixin(
    InputObjectLabelOutputSourceContextPolicyMixin,
    ABC,
):
    """Use input object-label provenance without declaring a CP parent image."""

    def source_context(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerObjectLabelOutputSourceContext:
        return CellProfilerObjectLabelOutputSourceContext(
            request.input_object_label_output_source_payload(),
            None,
        )


class CellProfilerImageOutputContextStrategy(
    CellProfilerOutputContextMessageMixin,
    CellProfilerOutputContextSelectionMixin,
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Attach runtime image context to declared image outputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)
    value_type: ClassVar[type[CellProfilerRuntimeValue] | None] = None
    output_value_subject = "image"

    @abstractmethod
    def runtime_image_value(
        self,
        value: ImagePayloadMetadataInput,
        source_image_payload: ImagePayloadMetadataInput | None,
    ) -> ImagePayloadMetadataInput:
        """Return the output in OpenHCS runtime image-payload form."""


class ContextualCellProfilerImageOutputStrategy(CellProfilerImageOutputContextStrategy):
    """Preserve outputs that already carry OpenHCS image context."""

    value_type = ImagePayloadMetadataCarrier

    def runtime_image_value(
        self,
        value: ImagePayloadMetadataInput,
        source_image_payload: ImagePayloadMetadataInput | None,
    ) -> ImagePayloadMetadataInput:
        if source_image_payload is None:
            return value
        return DerivedImagePayloadContext(source_image_payload, value).payload()


class NumpyCellProfilerImageOutputStrategy(CellProfilerImageOutputContextStrategy):
    """Attach source image context to raw CellProfiler numpy image outputs."""

    value_type = np.ndarray

    def runtime_image_value(
        self,
        value: ImagePayloadMetadataInput,
        source_image_payload: ImagePayloadMetadataInput | None,
    ) -> ImagePayloadMetadataInput:
        if not isinstance(value, np.ndarray):
            raise TypeError("Numpy image output strategy requires numpy.ndarray.")
        return DerivedImagePayloadContext(
            source_image_payload,
            SINGLETON_STACK_OUTPUT_COLLAPSE.collapse(value),
        ).payload()


class CellProfilerObjectLabelOutputContextStrategy(
    CellProfilerOutputContextMessageMixin,
    CellProfilerOutputContextSelectionMixin,
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Attach runtime source-image context to declared object-label outputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)
    value_type: ClassVar[type[CellProfilerRuntimeValue] | None] = None
    output_value_subject = "object-label"
    supported_output_values = "object-label payloads or numpy arrays"

    @abstractmethod
    def runtime_object_label_value(
        self,
        value: CellProfilerRuntimeValue,
        source_image_payload: CellProfilerRuntimeValue,
        domain_scope: ObjectLabelDomainScope | None = None,
    ) -> CellProfilerRuntimeValue:
        """Return the output in OpenHCS runtime object-label payload form."""


class ContextualCellProfilerObjectLabelOutputStrategy(
    CellProfilerObjectLabelOutputContextStrategy
):
    """Preserve object-label outputs that already carry OpenHCS context."""

    value_type = (ObjectLabelPayload, ObjectLabelSet)

    def runtime_object_label_value(
        self,
        value: CellProfilerRuntimeValue,
        source_image_payload: CellProfilerRuntimeValue,
        domain_scope: ObjectLabelDomainScope | None = None,
    ) -> CellProfilerRuntimeValue:
        if not isinstance(value, ObjectLabelValue):
            raise TypeError(
                "Contextual object-label output strategy requires an OpenHCS "
                "object-label payload."
            )
        output_value = value.with_source_image_context(source_image_payload)
        if domain_scope is None:
            return output_value
        output_domain = output_value.object_label_domain().with_scope(domain_scope)
        return ObjectLabelValueConstructionContext.from_value(
            output_value,
            domain=output_domain,
        ).value_from_variants(
            output_value,
            ObjectLabelVariantData.from_value(output_value),
        )


class NumpyCellProfilerObjectLabelOutputStrategy(
    CellProfilerObjectLabelOutputContextStrategy
):
    """Attach source image context to raw CellProfiler object-label arrays."""

    value_type = np.ndarray

    def runtime_object_label_value(
        self,
        value: CellProfilerRuntimeValue,
        source_image_payload: CellProfilerRuntimeValue,
        domain_scope: ObjectLabelDomainScope | None = None,
    ) -> CellProfilerRuntimeValue:
        if not isinstance(value, np.ndarray):
            raise TypeError("Numpy object-label output strategy requires numpy.ndarray.")
        return SourceImageObjectLabelBuildRequest(
            image=source_image_payload,
            labels=value,
            domain_scope=domain_scope,
        ).payload()


class OpaqueCellProfilerObjectLabelOutputStrategy(
    CellProfilerObjectLabelOutputContextStrategy
):
    """Preserve opaque object-label payloads for non-array runtime stores."""

    value_type = RuntimeArrayPayload

    def runtime_object_label_value(
        self,
        value: CellProfilerRuntimeValue,
        source_image_payload: CellProfilerRuntimeValue,
        domain_scope: ObjectLabelDomainScope | None = None,
    ) -> CellProfilerRuntimeValue:
        del source_image_payload, domain_scope
        return value
