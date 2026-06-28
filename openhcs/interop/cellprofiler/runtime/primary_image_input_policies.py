"""CellProfiler primary-image input policy contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from metaclass_registry import RegistryFamily, RegistryKeyAttribute

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerImageRequest
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryKey,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.module_execution import (
        CellProfilerModuleRuntimePlan,
        CellProfilerSpecialInputPolicy,
    )


class CellProfilerPrimaryImageInputPolicyMixin(ABC):
    """Declaration-owned image-domain behavior for CellProfiler modules."""

    @abstractmethod
    def primary_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
        *,
        special_input_policy: "CellProfilerSpecialInputPolicy",
    ) -> tuple[ArtifactSpec, ...]:
        """Return image inputs that should drive function invocation slices."""

    def runtime_image_current_image(
        self,
        module_name: str,
        spec: ArtifactSpec,
        current_image: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue | None:
        """Return source context used when resolving a primary runtime image."""
        del module_name, spec
        return current_image

    def invocation_runtime_kwargs(
        self,
        *,
        module_name: str,
        plan: "CellProfilerModuleRuntimePlan",
        image_request: CellProfilerImageRequest,
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerRuntimeValue,
        runtime_kwargs: CellProfilerKwargs,
        object_input_source_image_name: Callable[[], str | None],
    ) -> CellProfilerKwargs:
        """Return module-owned runtime kwargs after generic input binding."""
        del (
            module_name,
            plan,
            image_request,
            adapter,
            current_image,
            object_input_source_image_name,
        )
        return runtime_kwargs


class CellProfilerPrimaryImageInputPolicy(
    CellProfilerPrimaryImageInputPolicyMixin,
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Registered fallback policy root for CellProfiler primary-image inputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    declaration_policy_bases = (CellProfilerPrimaryImageInputPolicyMixin,)


class DefaultPrimaryImageInputPolicy(CellProfilerPrimaryImageInputPolicy):
    """Use non-special image inputs as the algorithmic image domain."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def primary_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
        *,
        special_input_policy: "CellProfilerSpecialInputPolicy",
    ) -> tuple[ArtifactSpec, ...]:
        image_inputs = ArtifactSpecCollection(declared_inputs).of_kind(
            ArtifactKind.IMAGE
        )
        special_image_count = len(
            special_input_policy.special_image_inputs(
                module_name,
                func,
                declared_inputs,
            )
        )
        if special_image_count == 0:
            return image_inputs
        return image_inputs[: len(image_inputs) - special_image_count]


class ObjectLabelDrivenPrimaryImageInputPolicy(CellProfilerPrimaryImageInputPolicyMixin):
    """Treat declared images as carriers; object labels define the domain."""

    def primary_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
        *,
        special_input_policy: "CellProfilerSpecialInputPolicy",
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func, declared_inputs, special_input_policy
        return ()
