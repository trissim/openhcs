"""CellProfiler callable ABI authority."""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from openhcs.constants.constants import VariableComponents
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract, FunctionStepExecutionScope
from openhcs.core.pipeline.function_contracts import (
    annotation_accepts_runtime_type,
    annotation_produces_runtime_type,
)
from openhcs.core.runtime_object_labels import ObjectLabelValue

if TYPE_CHECKING:
    from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
    from openhcs.core.runtime_array_values import RuntimeArrayData
    from openhcs.interop.cellprofiler.runtime.artifact_binding import (
        RuntimeInputBindingRequest,
    )
    from openhcs.interop.cellprofiler.runtime.invocation import (
        CellProfilerImageRequest,
    )
    from openhcs.interop.cellprofiler.runtime.output_contexts import (
        CellProfilerObjectLabelOutputSourceContext,
    )
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )
    from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding
    from collections.abc import Callable
    from openhcs.core.steps.function_runtime import (
        RuntimeCallableArgument,
        RuntimeCallableKwargs,
        RuntimeFunctionOutput,
    )


def _callable_return_slot_variants(annotation: Any) -> tuple[tuple[Any, ...], ...]:
    """Return statically declared positional return-slot variants."""

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return tuple(
            variant
            for member in get_args(annotation)
            for variant in _callable_return_slot_variants(member)
        )
    if origin is tuple:
        slots = get_args(annotation)
        return () if Ellipsis in slots else (slots,)
    if isinstance(annotation, type):
        from openhcs.core.runtime_output_matching import (
            RuntimeOutputBundle,
        )

        if issubclass(annotation, RuntimeOutputBundle):
            bundle_return = get_type_hints(annotation.as_runtime_tuple).get(
                "return",
                inspect.Signature.empty,
            )
            return _callable_return_slot_variants(bundle_return)
    if annotation in (inspect.Signature.empty, tuple):
        return ()
    return ((annotation,),)


class CellProfilerModuleCallableABI:
    binds_without_declared_inputs: ClassVar[bool] = False

    @classmethod
    def executes_per_object_measurements(
        cls,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> bool:
        """Return whether this module executes once per active object input."""
        del object_inputs
        return False

    @classmethod
    def executes_per_image_measurements(
        cls,
        func: "Callable[..., RuntimeFunctionOutput]",
        object_inputs: tuple[ArtifactSpec, ...],
        *,
        callable_contract: CallableContract,
    ) -> bool:
        """Return whether this module executes once per active image input."""
        del func, object_inputs, callable_contract
        return False

    @classmethod
    def execution_mode(
        cls,
        default: "ImagePayloadExecutionMode",
        *,
        image: "RuntimeCallableArgument",
        kwargs: "RuntimeCallableKwargs",
        variable_components: tuple[VariableComponents, ...],
    ) -> "ImagePayloadExecutionMode":
        """Return the default runtime image execution mode."""

        del cls, image, kwargs, variable_components
        return default

    @classmethod
    def source_payload(
        cls,
        request: "CellProfilerOutputRecordRequest",
    ) -> "RuntimeCallableArgument | None":
        """Return the default image-output source payload."""

        del cls
        return request.declared_source_payload()

    @classmethod
    def output_value(
        cls,
        request: "CellProfilerOutputRecordRequest",
    ) -> "RuntimeCallableArgument":
        """Return the default image-output value."""

        del cls
        return request.output_value

    @classmethod
    def source_context(
        cls,
        request: "CellProfilerOutputRecordRequest",
    ) -> "CellProfilerObjectLabelOutputSourceContext":
        """Return the default object-label output source context."""

        from openhcs.interop.cellprofiler.runtime.output_contexts import (
            CellProfilerObjectLabelOutputSourceContext,
        )

        del cls
        source_payload = replace(
            request,
            current_image=request.source.payload,
        ).declared_source_payload()
        return CellProfilerObjectLabelOutputSourceContext(
            source_payload,
            source_payload,
        )

    @classmethod
    def primary_image_inputs(
        cls,
        func: "Callable[..., RuntimeFunctionOutput]",
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return non-special image inputs that drive invocation slices."""

        if (
            CallableContract.from_callable(func).execution_scope
            is FunctionStepExecutionScope.PLATE
        ):
            return ()
        image_inputs = ArtifactSpecCollection(declared_inputs).of_artifact_type(
            ImageArtifactType
        )
        return tuple(
            spec
            for spec in image_inputs
            if spec.parameter_name is None and not spec.stack_broadcast_sources()
        )

    @classmethod
    def invocation_domain_inputs(
        cls,
        func: "Callable[..., RuntimeFunctionOutput]",
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return the inputs whose component scope owns one invocation."""

        primary_images = cls.primary_image_inputs(func, declared_inputs)
        if primary_images:
            return primary_images
        object_inputs = ArtifactSpecCollection(declared_inputs).of_artifact_type(
            ObjectLabelsArtifactType
        )
        if cls.executes_per_object_measurements(object_inputs):
            return object_inputs[:1]
        return tuple(
            artifact_input
            for artifact_input in declared_inputs
            if artifact_input.group_scope_sources()
        )

    @classmethod
    def primary_image_domain_input_binding(cls) -> "SettingToKeywordBinding":
        """Return the exact artifact binding that owns the invocation image domain."""

        raise TypeError(
            f"{cls.__name__} does not declare a primary image-domain input binding."
        )

    @classmethod
    def invocation_runtime_kwargs(
        cls,
        *,
        image_request: "CellProfilerImageRequest",
        runtime_kwargs: "RuntimeCallableKwargs",
    ) -> "RuntimeCallableKwargs":
        """Return runtime kwargs after generic input binding."""

        del (
            cls,
            image_request,
        )
        return runtime_kwargs

    @classmethod
    def project_invocation_image_request(
        cls,
        *,
        image_request: "CellProfilerImageRequest",
        runtime_kwargs: "RuntimeCallableKwargs",
    ) -> "CellProfilerImageRequest":
        """Project the invocation image through the module's declared input domain."""

        del cls, runtime_kwargs
        return image_request

    @classmethod
    def object_measurement_invocation_kwargs(
        cls,
        runtime_kwargs: "RuntimeCallableKwargs",
        *,
        include_image_measurements: bool,
    ) -> "RuntimeCallableKwargs":
        """Project kwargs for one object-input measurement invocation."""

        del cls, include_image_measurements
        return runtime_kwargs

    @classmethod
    def validate_callable_artifact_abi(
        cls,
        func: "Callable[..., RuntimeFunctionOutput]",
        contract: CallableContract,
    ) -> None:
        """Validate one raw callable ABI against its callable artifact declarations."""

        from openhcs.core.aligned_image_payload import AlignedImageStack
        module_name = contract.module_name
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("CellProfiler callable contract requires a module name.")
        declared_outputs = contract.artifact_outputs
        invalid_source_context_outputs = tuple(
            (spec.ref(), sources)
            for spec in declared_outputs
            for sources in (spec.source_context_sources(),)
            if spec.artifact_type.carries_source_image_context and len(sources) != 1
        )
        if invalid_source_context_outputs:
            raise ValueError(
                f"CellProfiler module {module_name!r} image/object outputs "
                "must declare exactly one runtime-context source: "
                f"{invalid_source_context_outputs!r}."
            )
        return_annotation = get_type_hints(func).get("return", inspect.Signature.empty)
        return_variants = _callable_return_slot_variants(return_annotation)
        canonical_outputs = contract.canonical_return_output_specs
        trailing_outputs = contract.trailing_return_output_specs
        expected_slot_count = len(trailing_outputs) + 1
        matching_variants = tuple(
            slots for slots in return_variants if len(slots) == expected_slot_count
        )
        if not matching_variants:
            raise ValueError(
                f"callable {func.__name__!r} return annotation "
                f"{return_annotation!r} does not declare the canonical return "
                f"followed by exactly {len(trailing_outputs)} trailing "
                "artifact-output slot(s)."
            )
        typed_variants = matching_variants
        if (
            len(canonical_outputs) > 1
            and not any(
                annotation_accepts_runtime_type(slots[0], AlignedImageStack)
                for slots in typed_variants
                if slots
            )
        ):
            raise ValueError(
                f"callable {func.__name__!r} carries {len(canonical_outputs)} "
                "canonical image outputs but its first return slot does not declare "
                "AlignedImageStack."
            )

        output_slot_specs: tuple[ArtifactSpec | None, ...] = (
            canonical_outputs[0] if len(canonical_outputs) == 1 else None,
            *trailing_outputs,
        )
        object_label_slot_indices = tuple(
            index
            for index, spec in enumerate(output_slot_specs)
            if spec is not None and spec.artifact_type is ObjectLabelsArtifactType
        )
        object_label_variants = tuple(
            slots
            for slots in typed_variants
            if all(index < len(slots) for index in object_label_slot_indices)
        )
        if (
            object_label_slot_indices
            and object_label_variants
            and not any(
                all(
                    annotation_produces_runtime_type(slots[index], ObjectLabelValue)
                    for index in object_label_slot_indices
                )
                for slots in object_label_variants
            )
        ):
            raise TypeError(
                f"callable {func.__name__!r} object-label return slot(s) "
                f"{object_label_slot_indices!r} must explicitly return "
                "ObjectLabelValue; raw arrays do not declare object identity or "
                "plane semantics."
            )
        measurement_slot_indices = tuple(
            index
            for index, spec in enumerate(output_slot_specs)
            if spec is not None and spec.artifact_type is MeasurementsArtifactType
        )
        if measurement_slot_indices:
            from openhcs.core.runtime_tabular_values import ColumnarRows

            measurement_variants = tuple(
                slots
                for slots in typed_variants
                if all(index < len(slots) for index in measurement_slot_indices)
            )
            if measurement_variants and not any(
                all(
                    annotation_produces_runtime_type(slots[index], ColumnarRows)
                    for index in measurement_slot_indices
                )
                for slots in measurement_variants
            ):
                raise TypeError(
                    f"callable {func.__name__!r} measurement return slot(s) "
                    f"{measurement_slot_indices!r} must explicitly return "
                    "ColumnarRows; raw records and sequences erase the declared "
                    "measurement schema."
                )
    @classmethod
    def binding_current_image(
        cls,
        *,
        current_image: "RuntimeArrayData",
        primary_image: "RuntimeArrayData | None",
    ) -> "RuntimeArrayData":
        """Return source image context used to bind special inputs."""

        del cls
        return primary_image if primary_image is not None else current_image

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: "RuntimeInputBindingRequest",
    ) -> "dict[str, RuntimeCallableArgument]":
        """Bind exact compiler-owned artifact input parameters."""

        return request.bind_parameters()

    @classmethod
    def validate_declared_object_inputs(
        cls,
        *,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> None:
        """Reject object inputs without a nominal input-policy override."""

        del cls
        if object_inputs:
            raise ValueError(
                f"{module_name} has object runtime inputs "
                f"{[spec.name for spec in object_inputs]}, but no nominal input "
                "binding policy has been declared for this CellProfiler module."
            )
