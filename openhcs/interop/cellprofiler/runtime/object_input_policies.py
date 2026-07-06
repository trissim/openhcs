"""Object-label input binding policies for CellProfiler runtime modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from metaclass_registry import RegistryFamily, RegistryKeyAttribute

from openhcs.core.artifacts import ArtifactSpec, ArtifactType
from openhcs.interop.cellprofiler.runtime.bound_parameters import (
    MeasurementTableCollectionParameterName,
    RuntimeBoundParameterName,
    RuntimeSliceSequenceParameterName,
    declared_measurement_table_parameter_names,
    declared_runtime_bound_parameter_names,
    declared_runtime_slice_sequence_parameter_names,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    ObjectInputBindingRequest,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargDict
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryKey,
)

class CellProfilerObjectInputPolicyMixin(ABC):
    """Declaration-owned object-label input binding behavior."""

    binds_without_declared_inputs: ClassVar[bool] = False
    supported_non_object_input_kinds: ClassVar[frozenset[ArtifactType]] = frozenset()

    @abstractmethod
    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        """Return absorbed-function kwargs for object-label runtime inputs."""

    def bound_parameter_names(
        self,
        plan: "CellProfilerModuleRuntimePlan",
    ) -> tuple[str, ...]:
        """Return callable parameters supplied by this runtime input policy."""
        names = self.declared_bound_parameter_names()
        if names:
            return names if plan.object_inputs or type(self).binds_without_declared_inputs else ()
        if plan.object_inputs or type(self).binds_without_declared_inputs:
            raise NotImplementedError(
                f"{type(self).__name__} must declare its runtime-bound callable "
                "parameters through an inherited binding contract."
            )
        return ()

    def validate_declared_object_inputs(
        self,
        *,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> None:
        """Validate object-input semantics while building the runtime plan."""

    def validate_runtime_plan_object_inputs(
        self,
        *,
        module_name: str,
        object_label_inputs: tuple[ArtifactSpec, ...],
        special_input_names: tuple[str, ...],
    ) -> None:
        """Validate the object inputs visible to this runtime plan."""
        self.validate_declared_object_inputs(
            module_name=module_name,
            object_inputs=self.runtime_plan_validation_object_inputs(
                object_label_inputs=object_label_inputs,
                special_input_names=special_input_names,
            ),
        )

    def runtime_plan_validation_object_inputs(
        self,
        *,
        object_label_inputs: tuple[ArtifactSpec, ...],
        special_input_names: tuple[str, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return object-label inputs owned by object-input validation."""
        if special_input_names:
            return ()
        return object_label_inputs

    def declared_bound_parameter_names(self) -> tuple[str, ...]:
        """Return bound parameter names from policy role declarations."""
        return declared_runtime_bound_parameter_names(type(self))

    def runtime_slice_sequence_parameter_names(
        self,
        plan: "CellProfilerModuleRuntimePlan",
    ) -> tuple[str, ...]:
        """Return bound tuple parameters that project item-wise per slice."""
        if plan.object_inputs or type(self).binds_without_declared_inputs:
            return declared_runtime_slice_sequence_parameter_names(type(self))
        return ()

    def measurement_table_parameter_names(
        self,
        plan: "CellProfilerModuleRuntimePlan",
    ) -> tuple[str, ...]:
        """Return bound parameters carrying measurement-table collections."""
        if plan.object_inputs or type(self).binds_without_declared_inputs:
            return declared_measurement_table_parameter_names(type(self))
        return ()


class CellProfilerObjectInputPolicy(
    CellProfilerObjectInputPolicyMixin,
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Registered fallback policy root for CellProfiler object-label inputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    declaration_policy_bases = (CellProfilerObjectInputPolicyMixin,)


class UnsupportedObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Reject undeclared object-input semantics instead of guessing."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def bound_parameter_names(
        self,
        plan: "CellProfilerModuleRuntimePlan",
    ) -> tuple[str, ...]:
        del plan
        return ()

    def validate_declared_object_inputs(
        self,
        *,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> None:
        if not object_inputs:
            return
        raise NotImplementedError(
            f"{module_name} has object runtime inputs "
            f"{[spec.name for spec in object_inputs]}, but no nominal input "
            "binding policy has been declared for this CellProfiler module."
        )

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        self.validate_declared_object_inputs(
            module_name=request.module_name,
            object_inputs=request.object_inputs,
        )
        return {}


class SingleObjectLabelInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind one object-label input into a module-specific parameter."""

    label_kwarg: ClassVar[RuntimeBoundParameterName]

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


class PrimaryObjectLabelInputPolicy(SingleObjectLabelInputPolicy):
    """Bind one primary-object label input."""

    label_kwarg = RuntimeBoundParameterName("primary_labels")


class PairedPrimarySecondaryObjectInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind two object-label inputs as primary/secondary label kwargs."""

    smaller_label_kwarg: ClassVar[RuntimeBoundParameterName] = (
        RuntimeBoundParameterName("primary_labels")
    )
    larger_label_kwarg: ClassVar[RuntimeBoundParameterName] = (
        RuntimeBoundParameterName("secondary_labels")
    )

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        request.require_exact_object_count(2)
        larger, smaller = request.object_inputs
        return {
            self.smaller_label_kwarg: request.label_argument_for(
                smaller,
                self.smaller_label_kwarg,
            ),
            self.larger_label_kwarg: request.label_argument_for(
                larger,
                self.larger_label_kwarg,
            ),
        }


class CroppingObjectLabelInputPolicy(SingleObjectLabelInputPolicy):
    """Bind one object-label input as a crop mask."""

    label_kwarg = RuntimeBoundParameterName("cropping_labels")


class LabelsObjectInputPolicy(SingleObjectLabelInputPolicy):
    """Bind one object-label input to the conventional labels kwarg."""

    label_kwarg = RuntimeBoundParameterName("labels")


class ObjectLabelsInputBindingMixin:
    """Bind object-label inputs under CellProfiler's object_labels kwarg."""

    object_labels_kwarg: ClassVar[RuntimeSliceSequenceParameterName] = (
        RuntimeSliceSequenceParameterName("object_labels")
    )

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        return {self.object_labels_kwarg: request.labels_for_inputs()}


class OverlayOutlinesInputPolicy(
    ObjectLabelsInputBindingMixin,
    CellProfilerObjectInputPolicyMixin,
):
    """Bind ordered object outline rows for the generic overlay runner."""


class ObjectRowsInputPolicy(
    ObjectLabelsInputBindingMixin,
    CellProfilerObjectInputPolicyMixin,
):
    """Bind ordered object rows to object-label payloads."""


class ObjectRowsWithMeasurementsInputPolicy(ObjectRowsInputPolicy):
    """Bind ordered object rows plus prior measurements for the primary object."""

    measurement_tables_kwarg: ClassVar[MeasurementTableCollectionParameterName] = (
        MeasurementTableCollectionParameterName("measurement_tables")
    )

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        bound = super().bind(request)
        bound[self.measurement_tables_kwarg] = (
            request.measurement_tables_for_primary_object()
        )
        return bound
