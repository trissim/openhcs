"""Object-label input binding policies for CellProfiler runtime modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from metaclass_registry import RegistryFamily, RegistryKeyAttribute

from openhcs.core.artifacts import ArtifactKind
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    ObjectInputBindingRequest,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargDict
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryKey,
)


class CellProfilerObjectInputPolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Nominal binding policy for CellProfiler object-label inputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    binds_without_declared_inputs: ClassVar[bool] = False
    supported_non_object_input_kinds: ClassVar[frozenset[ArtifactKind]] = frozenset()

    @abstractmethod
    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        """Return absorbed-function kwargs for object-label runtime inputs."""


class UnsupportedObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Reject undeclared object-input semantics instead of guessing."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        if not request.object_inputs:
            return {}
        raise NotImplementedError(
            f"{request.module_name} has object runtime inputs "
            f"{[spec.name for spec in request.object_inputs]}, but no nominal input "
            "binding policy has been declared for this CellProfiler module."
        )


class SingleObjectLabelInputPolicy(CellProfilerObjectInputPolicy):
    """Bind one object-label input into a module-specific parameter."""

    label_kwarg: ClassVar[str]

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        request.require_exact_object_count(1)
        return {
            self.label_kwarg: request.label_argument_for(
                request.object_inputs[0],
                self.label_kwarg,
            )
        }


class IdentifySecondaryObjectsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind primary objects with generic label-variant context when available."""

    module_name = "IdentifySecondaryObjects"

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        request.require_exact_object_count(1)
        return {
            "primary_labels": request.label_argument_for(
                request.object_inputs[0],
                "primary_labels",
            )
        }


class IdentifyTertiaryObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Bind smaller/larger labels to the absorbed tertiary-object signature."""

    module_name = "IdentifyTertiaryObjects"

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        request.require_exact_object_count(2)
        larger, smaller = request.object_inputs
        return {
            "primary_labels": request.label_argument_for(smaller, "primary_labels"),
            "secondary_labels": request.label_argument_for(larger, "secondary_labels"),
        }
