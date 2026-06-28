"""CellProfiler special-input binding policy contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import time
from typing import TYPE_CHECKING

from metaclass_registry import RegistryFamily, RegistryKeyAttribute
import numpy as np

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
from openhcs.core.runtime_values import (
    ImagePayloadMetadataInput,
    ObjectLabelData,
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeArtifactKindStrategy,
    RuntimeInputBindingRequestBase,
)
from openhcs.interop.cellprofiler.runtime.bound_parameters import (
    declared_runtime_bound_parameter_names,
)
from openhcs.interop.cellprofiler.runtime.object_label_source_projection import (
    CurrentImageObjectLabelPlaneAlignment,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryKey,
)
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    CellProfilerSpecialInputPayloadSemantics,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.interop.cellprofiler.runtime.runtime_special_values import (
    CellProfilerSpecialInputKwargs,
    CellProfilerSpecialInputValue,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.module_execution import (
        CellProfilerModuleRuntimePlan,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecialInputBindingRequest(RuntimeInputBindingRequestBase):
    """Authoritative runtime context for binding declared special_inputs."""

    registry_key = "special_input"

    parameter_names: tuple[str, ...]
    special_input_specs: tuple[ArtifactSpec, ...]
    runtime_inputs: tuple[ArtifactSpec, ...]
    project_object_labels_to_current_plane: bool = False

    @property
    def object_inputs(self) -> tuple[ArtifactSpec, ...]:
        return ArtifactSpecCollection(self.special_input_specs).of_kind(
            ArtifactKind.OBJECT_LABELS
        )

    @property
    def image_inputs(self) -> tuple[ArtifactSpec, ...]:
        return ArtifactSpecCollection(self.special_input_specs).of_kind(
            ArtifactKind.IMAGE
        )

    def runtime_value(
        self,
        spec: ArtifactSpec,
        parameter_name: str | None = None,
        semantics: CellProfilerSpecialInputPayloadSemantics = (
            CellProfilerSpecialInputPayloadSemantics.INTENSITY_IMAGE
        ),
    ) -> CellProfilerRuntimeValue:
        if spec.kind is ArtifactKind.OBJECT_LABELS:
            if parameter_name is not None:
                return self.label_argument_for(spec, parameter_name)
            return self.object_label_runtime_value(spec, semantics)
        request = self.artifact_input_request(spec)
        artifact_strategy = RuntimeArtifactKindStrategy.for_kind(spec.kind)
        if semantics.dense_label_domain:
            started_at = time.perf_counter()
            value = object_label_dense_array(
                artifact_strategy.raw_runtime_input_value(request),
                dtype=np.int32,
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "special_input_runtime_value",
                time.perf_counter() - started_at,
                module=self.module_name,
                spec=spec.name,
                kind=spec.kind.value,
                semantics=semantics.value,
            )
            return value
        started_at = time.perf_counter()
        value = artifact_strategy.runtime_input_value(request)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "special_input_runtime_value",
            time.perf_counter() - started_at,
            module=self.module_name,
            spec=spec.name,
            kind=spec.kind.value,
            semantics=semantics.value,
        )
        return value

    def runtime_value_without_current_image_projection(
        self,
        spec: ArtifactSpec,
    ) -> CellProfilerSpecialInputValue:
        """Return a runtime artifact input without ambient source-image narrowing."""
        request = replace(self.artifact_input_request(spec), current_image=None)
        return RuntimeArtifactKindStrategy.for_kind(spec.kind).runtime_input_value(request)

    def object_label_runtime_value(
        self,
        spec: ArtifactSpec,
        semantics: CellProfilerSpecialInputPayloadSemantics,
    ) -> ObjectLabelData:
        """Return an object-label special input in the invocation's artifact domain."""
        payload = RuntimeArtifactKindStrategy.for_kind(spec.kind).runtime_input_value(
            self.artifact_input_request(spec)
        )
        del semantics
        return object_label_dense_array(payload, dtype=np.int32)

    def current_plane_object_label_runtime_value(
        self,
        spec: ArtifactSpec,
    ) -> ObjectLabelData:
        """Return object labels projected into the invocation's current plane."""
        payload = self.current_plane_label_payload_for(spec)
        return object_label_dense_array(payload, dtype=np.int32)

    def object_label_payload(
        self,
        spec: ArtifactSpec,
    ) -> ObjectLabelValue:
        """Return object labels with provenance preserved for special inputs."""
        payload = self.label_payload_for(spec)
        if not isinstance(payload, ObjectLabelValue):
            raise TypeError(
                f"{self.module_name} special input {spec.name!r} resolved to "
                f"{type(payload).__name__}, expected ObjectLabelValue."
            )
        return payload

    def current_image_aligned_object_label_runtime_value(
        self,
        spec: ArtifactSpec,
        *,
        alignment_image: CellProfilerRuntimeValue | None = None,
    ) -> CellProfilerRuntimeValue:
        """Return object labels ordered to match the current image stack planes."""
        labels = self.adapter.get_objects(spec.name)
        aligned = CurrentImageObjectLabelPlaneAlignment(
            adapter=self.adapter,
            current_image=alignment_image if alignment_image is not None else self.current_image,
            labels=labels,
        ).aligned_dense_value()
        if aligned is not None:
            return aligned
        return self.current_plane_object_label_runtime_value(spec)

    def bind_positional_parameters(self) -> CellProfilerSpecialInputKwargs:
        """Bind declared special-input parameters to compiled runtime specs."""
        if len(self.parameter_names) != len(self.special_input_specs):
            raise NotImplementedError(
                f"{self.module_name} declares special_inputs "
                f"{list(self.parameter_names)}, but compiled runtime inputs are "
                f"{[spec.name for spec in self.special_input_specs]}."
            )
        return {
            parameter_name: self.runtime_value(spec, parameter_name=parameter_name)
            for parameter_name, spec in zip(
                self.parameter_names,
                self.special_input_specs,
                strict=True,
            )
        }


class CellProfilerSpecialInputPolicyMixin(ABC):
    """Declaration-owned binding behavior for CellProfiler special_inputs."""

    def special_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return trailing image specs consumed by special_inputs instead of primary image payload."""

        return _signature_special_image_inputs(module_name, func, declared_inputs)

    def binding_current_image(
        self,
        *,
        current_image: ImagePayloadMetadataInput,
        primary_image: ImagePayloadMetadataInput | None,
    ) -> ImagePayloadMetadataInput:
        """Return the source image context used to bind special inputs."""
        del primary_image
        return current_image

    def bound_parameter_names(
        self,
        plan: "CellProfilerModuleRuntimePlan",
    ) -> tuple[str, ...]:
        """Return callable parameters supplied by declared special inputs."""
        return tuple(
            dict.fromkeys(
                (
                    *plan.special_input_names,
                    *self.extra_bound_parameter_names(plan),
                )
            )
        )

    def extra_bound_parameter_names(
        self,
        plan: "CellProfilerModuleRuntimePlan",
    ) -> tuple[str, ...]:
        """Return additional runtime-bound parameters declared by policy roles."""
        del plan
        return declared_runtime_bound_parameter_names(type(self))

    @abstractmethod
    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerSpecialInputKwargs:
        """Return kwargs for a callable's declared special_inputs."""


class CellProfilerSpecialInputPolicy(
    CellProfilerSpecialInputPolicyMixin,
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Registered fallback policy root for CellProfiler special_inputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    declaration_policy_bases = (CellProfilerSpecialInputPolicyMixin,)


class PositionalSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Bind special_inputs positionally to compiled runtime artifact specs."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerSpecialInputKwargs:
        return request.bind_positional_parameters()


class TrailingImageSpecialInputPolicy(CellProfilerSpecialInputPolicyMixin):
    """Treat all image inputs after the primary image as special inputs."""

    def special_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func
        image_inputs = ArtifactSpecCollection(declared_inputs).of_kind(
            ArtifactKind.IMAGE
        )
        return image_inputs[1:]


class NoSpecialImageInputsMixin(CellProfilerSpecialInputPolicyMixin):
    """Declare that a special-input policy consumes no image artifacts."""

    def special_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func, declared_inputs
        return ()


def _signature_special_image_inputs(
    module_name: str,
    func: CellProfilerFunction,
    declared_inputs: tuple[ArtifactSpec, ...],
) -> tuple[ArtifactSpec, ...]:
    image_inputs = ArtifactSpecCollection(declared_inputs).of_kind(ArtifactKind.IMAGE)
    special_input_count = len(special_input_names_from_callable(func))
    non_image_count = len(
        tuple(spec for spec in declared_inputs if spec.kind is not ArtifactKind.IMAGE)
    )
    special_image_count = max(0, special_input_count - non_image_count)
    if special_image_count == 0:
        return ()
    if special_image_count > len(image_inputs):
        raise NotImplementedError(
            f"{module_name} declares {special_image_count} image special "
            f"input(s), but only has image inputs {[spec.name for spec in image_inputs]}."
        )
    return image_inputs[-special_image_count:]
