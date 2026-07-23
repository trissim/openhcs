"""Object-label input binding policies for CellProfiler runtime modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.callable_contract import KeywordRuntimeParameter
from openhcs.core.pipeline.function_contracts import (
    annotation_accepts_runtime_type,
    resolved_callable_parameter,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelSet,
    ObjectLabelValue,
)
from openhcs.interop.cellprofiler.module_measurement_features import (
    CellProfilerModuleAuthority,
)
from collections.abc import Callable
from openhcs.core.steps.function_runtime import RuntimeCallableArgument, RuntimeFunctionOutput
from openhcs.interop.cellprofiler.runtime.artifact_binding import RuntimeInputBindingRequest

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding


class CellProfilerObjectInputPolicyMixin(CellProfilerModuleAuthority):
    """Nominal owner for declared object-label input binding."""

    @classmethod
    def validate_declared_object_inputs(
        cls,
        *,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> None:
        del cls, module_name, object_inputs

    @classmethod
    def validate_object_label_parameter_abi(
        cls,
        func: Callable[..., RuntimeFunctionOutput],
    ) -> None:
        """Terminate object-label ABI validation for policies without bindings."""

        del cls, func

    @classmethod
    def _validate_object_label_parameters(
        cls,
        func: Callable[..., RuntimeFunctionOutput],
        parameter_names: tuple[str, ...],
    ) -> None:
        """Require policy-owned object-label parameters to expose their payload ABI."""

        for parameter_name in parameter_names:
            parameter = resolved_callable_parameter(func, parameter_name)
            if annotation_accepts_runtime_type(
                parameter.annotation,
                ObjectLabelSet,
            ):
                continue
            raise TypeError(
                f"{cls.module_name} object-label parameter {parameter.name!r} "
                "must explicitly accept ObjectLabelValue, "
                f"got {parameter.annotation!r}."
            )


class ObjectLabelsRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound ordered object-label inputs for one invocation."""

    parameter_name = "object_labels"
    annotation_type = tuple[ObjectLabelValue, ...]
    parameter_default = ()


class SingleObjectLabelInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind one object-label input into a module-specific parameter."""

    label_kwarg: ClassVar[str]

    @classmethod
    def validate_object_label_parameter_abi(
        cls,
        func: Callable[..., RuntimeFunctionOutput],
    ) -> None:
        cls._validate_object_label_parameters(func, (cls.label_kwarg,))

    @classmethod
    def validate_declared_object_inputs(
        cls,
        *,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> None:
        del cls
        if len(object_inputs) > 1:
            raise ValueError(
                f"{module_name} accepts at most one object runtime input, "
                f"got {[spec.name for spec in object_inputs]}."
            )

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        bound = super().bind_runtime_inputs(request)
        object_inputs = request.unbound_object_inputs
        if not object_inputs:
            return bound
        cls.validate_declared_object_inputs(
            module_name=request.adapter.request.require_callable_contract().module_name,
            object_inputs=object_inputs,
        )
        bound[cls.label_kwarg] = request.label_argument_for(
            object_inputs[0],
            cls.label_kwarg,
        )
        return bound


class PrimaryObjectLabelInputPolicy(SingleObjectLabelInputPolicy):
    """Bind one primary-object label input."""

    label_kwarg = "primary_labels"


class TwoObjectLabelInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Require the exact pair consumed by two-object binding policies."""

    @classmethod
    def validate_declared_object_inputs(
        cls,
        *,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> None:
        del cls
        if len(object_inputs) != 2:
            raise ValueError(
                f"{module_name} requires 2 object runtime inputs, "
                f"got {[spec.name for spec in object_inputs]}."
            )


class PairedPrimarySecondaryObjectInputPolicy(TwoObjectLabelInputPolicy):
    """Bind two object-label inputs as primary/secondary label kwargs."""

    smaller_label_kwarg: ClassVar[str] = "primary_labels"
    larger_label_kwarg: ClassVar[str] = "secondary_labels"
    larger_objects_binding: ClassVar["SettingToKeywordBinding"]

    @classmethod
    def primary_image_domain_input_binding(cls) -> "SettingToKeywordBinding":
        """Use the larger-object input as the paired-label invocation domain."""

        return cls.larger_objects_binding

    @classmethod
    def validate_object_label_parameter_abi(
        cls,
        func: Callable[..., RuntimeFunctionOutput],
    ) -> None:
        cls._validate_object_label_parameters(
            func,
            (cls.smaller_label_kwarg, cls.larger_label_kwarg),
        )

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        bound = super().bind_runtime_inputs(request)
        object_inputs = request.unbound_object_inputs
        if not object_inputs:
            return bound
        cls.validate_declared_object_inputs(
            module_name=request.adapter.request.require_callable_contract().module_name,
            object_inputs=object_inputs,
        )
        larger, smaller = object_inputs
        bound[cls.smaller_label_kwarg] = request.label_argument_for(
            smaller,
            cls.smaller_label_kwarg,
        )
        bound[cls.larger_label_kwarg] = request.label_argument_for(
            larger,
            cls.larger_label_kwarg,
        )
        return bound


class LabelsObjectInputPolicy(SingleObjectLabelInputPolicy):
    """Bind one object-label input to the conventional labels kwarg."""

    label_kwarg = "labels"
    input_objects_binding: ClassVar["SettingToKeywordBinding"]

    @classmethod
    def primary_image_domain_input_binding(cls) -> "SettingToKeywordBinding":
        """Use the conventional input-object binding as the invocation domain."""

        return cls.input_objects_binding


class ObjectLabelsInputBindingMixin(CellProfilerObjectInputPolicyMixin):
    """Bind object-label inputs under CellProfiler's object_labels kwarg."""

    @classmethod
    def validate_object_label_parameter_abi(
        cls,
        func: Callable[..., RuntimeFunctionOutput],
    ) -> None:
        cls._validate_object_label_parameters(
            func,
            (ObjectLabelsRuntimeParameter.require_parameter_name(),),
        )

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        bound = super().bind_runtime_inputs(request)
        object_inputs = request.unbound_object_inputs
        if not object_inputs:
            return bound
        cls.validate_declared_object_inputs(
            module_name=request.adapter.request.require_callable_contract().module_name,
            object_inputs=object_inputs,
        )
        parameter_name = ObjectLabelsRuntimeParameter.require_parameter_name()
        bound[parameter_name] = tuple(
            request.label_argument_for(spec, parameter_name)
            for spec in object_inputs
        )
        return bound
