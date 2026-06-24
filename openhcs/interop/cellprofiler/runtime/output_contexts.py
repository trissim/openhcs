"""CellProfiler output value and source-context policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
import numpy as np

from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.runtime_semantics import ObjectLabelDomainScope
from openhcs.core.runtime_values import (
    DerivedImagePayloadContext,
    ImagePayloadMetadataCarrier,
    ImagePayloadMetadataInput,
    ObjectLabelValue,
    RuntimeArrayPayload,
    SourceImageObjectLabelBuildRequest,
    image_payload_data,
    image_payload_metadata,
    image_payload_slice_context,
)
from openhcs.interop.cellprofiler.runtime.image_payload_collapse import (
    SINGLETON_STACK_OUTPUT_COLLAPSE,
)
from openhcs.interop.cellprofiler.runtime.module_names import (
    CELLPROFILER_CORRECT_ILLUMINATION_APPLY_MODULE,
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

_CORRECT_ILLUMINATION_APPLY_MODULE = CELLPROFILER_CORRECT_ILLUMINATION_APPLY_MODULE

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


class CellProfilerImageOutputSourcePayloadPolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Resolve metadata-bearing source payloads for declared image outputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    @abstractmethod
    def source_payload(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue | None:
        """Return source context for one image output artifact."""


class DefaultImageOutputSourcePayloadPolicy(CellProfilerImageOutputSourcePayloadPolicy):
    """Use the module's unique declared primary image as output provenance."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def source_payload(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue | None:
        if (
            isinstance(request.output_value, ImagePayloadMetadataCarrier)
            and image_payload_metadata(request.output_value).source_provenance.has_values
        ):
            return None
        return request.unique_primary_image_source_payload()


class CellProfilerImageOutputValuePolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Normalize declared image output values before source context is attached."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    @abstractmethod
    def output_value(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue:
        """Return the value to record for one image output artifact."""


class DefaultImageOutputValuePolicy(CellProfilerImageOutputValuePolicy):
    """Record image outputs exactly as produced by CellProfiler."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def output_value(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue:
        return request.output_value


class CorrectIlluminationApplyImageOutputSourcePayloadPolicy(
    CellProfilerImageOutputSourcePayloadPolicy
):
    """Use the corrected channel's original image as output provenance."""

    module_name = _CORRECT_ILLUMINATION_APPLY_MODULE

    def source_payload(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue | None:
        source_spec = request.correct_illumination_apply_source_spec()
        if source_spec is None:
            return request.source.payload
        return request.input_image_source_payload(source_spec)


@dataclass(frozen=True, slots=True)
class CorrectedImageOutputPlaneStack:
    """Detect corrected-image output stacks that represent one source plane."""

    value: CellProfilerRuntimeValue

    def plane_index(self) -> int | None:
        data = np.asarray(image_payload_data(self.value))
        if data.ndim < 3 or is_color_image_slice(data):
            return None
        if data.shape[0] == 1:
            return 0
        if self.has_duplicate_source_plane_identity(data.shape[0]):
            return 0
        return None

    def has_duplicate_source_plane_identity(self, plane_count: int) -> bool:
        identities = self.plane_identities(plane_count)
        if not identities:
            return False
        return len(frozenset(identities)) == 1

    def plane_identities(
        self,
        plane_count: int,
    ) -> tuple[tuple[str | None, tuple[tuple[str, str], ...] | None], ...]:
        provenance = image_payload_metadata(self.value).source_provenance
        if provenance.source_plane_count != plane_count:
            return ()
        return tuple(
            provenance.for_source_plane(plane_index).identity()
            for plane_index in range(plane_count)
        )


class CorrectIlluminationApplyImageOutputValuePolicy(CellProfilerImageOutputValuePolicy):
    """Collapse duplicate grouped-plane stacks emitted for one corrected source."""

    module_name = _CORRECT_ILLUMINATION_APPLY_MODULE

    def output_value(self, request: CellProfilerOutputRecordRequest) -> CellProfilerRuntimeValue:
        plane_index = CorrectedImageOutputPlaneStack(request.output_value).plane_index()
        if plane_index is None:
            return request.output_value
        output_data = np.asarray(image_payload_data(request.output_value))[plane_index]
        return image_payload_slice_context(request.output_value, output_data, plane_index)


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

    value_type = ObjectLabelValue

    def runtime_object_label_value(
        self,
        value: CellProfilerRuntimeValue,
        source_image_payload: CellProfilerRuntimeValue,
        domain_scope: ObjectLabelDomainScope | None = None,
    ) -> CellProfilerRuntimeValue:
        del domain_scope
        if not isinstance(value, ObjectLabelValue):
            raise TypeError(
                "Contextual object-label output strategy requires an OpenHCS "
                "object-label payload."
            )
        return value.with_source_image_context(source_image_payload)


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
