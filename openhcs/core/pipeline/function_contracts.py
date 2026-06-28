"""Function-level artifact contract decorators for the pipeline compiler."""

from collections import OrderedDict
from collections.abc import Mapping
from enum import Enum
import inspect
from typing import Callable, TypeVar

from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.runtime_invocation import RuntimeParameterDeclaration
from openhcs.core.runtime_batch_contracts import (
    RUNTIME_BATCH_EXECUTORS_ATTR,
    Pure2DSliceBatchExecutor,
    RuntimeBatchExecutionDomain,
    RuntimeBatchExecutor,
    RuntimePure2DSliceBatchRequest,
    measurement_image_batch_executor,
    pure_2d_batch_executor,
    runtime_batch_executor,
    runtime_batch_executors_from_callable,
)
from openhcs.core.special_output_declarations import (
    SpecialOutputDeclaration,
    SpecialOutputDeclarations,
)
from openhcs.core.variable_component_stack_requirement import (
    AlwaysRequiresVariableComponentStack,
    VariableComponentStackRequirement,
)
from openhcs.processing.materialization import MaterializationSpec

F = TypeVar("F", bound=Callable)


class ObjectLabelMeasurementExecution(str, Enum):
    """How object-measurement functions consume label domains."""

    SLICE_ALIGNED = "slice_aligned"
    FULL_STACK = "full_stack"


class ImagePayloadConsumption(str, Enum):
    """How a callable consumes its primary image payload."""

    NATURAL = "natural"
    COMPOSED = "composed"


def _artifact_spec_from_output_declaration(
    spec: str | ArtifactSpec | tuple[str, MaterializationSpec],
) -> ArtifactSpec:
    """Normalize one output declaration into an ArtifactSpec."""
    if isinstance(spec, ArtifactSpec):
        return spec

    if isinstance(spec, str):
        return ArtifactSpec(spec, ArtifactKind.SPECIAL)

    if isinstance(spec, tuple) and len(spec) == 2:
        key, mat_spec = spec
        if not isinstance(key, str):
            raise ValueError(f"Artifact output key must be string, got {type(key)}: {key}")
        if not isinstance(mat_spec, MaterializationSpec):
            raise ValueError(
                "Materialization spec must be a MaterializationSpec. "
                f"Got {type(mat_spec)} for key '{key}'."
            )
        return ArtifactSpec(
            key,
            ArtifactKind.SPECIAL,
            materialization=mat_spec,
        )

    raise ValueError(
        f"Invalid artifact output spec: {spec}. "
        "Must be string, ArtifactSpec, or (string, MaterializationSpec) tuple."
    )


def _artifact_spec_from_input_declaration(spec: str | ArtifactSpec) -> ArtifactSpec:
    """Normalize one input declaration into an ArtifactSpec."""
    if isinstance(spec, ArtifactSpec):
        return spec
    if isinstance(spec, str):
        return ArtifactSpec(spec, ArtifactKind.SPECIAL)
    raise ValueError(
        f"Invalid artifact input spec: {spec}. Must be string or ArtifactSpec."
    )


def artifact_outputs(
    *output_specs: str | ArtifactSpec | tuple[str, MaterializationSpec],
) -> Callable[[F], F]:
    """Declare named artifacts produced by a processing function."""

    def decorator(func: F) -> F:
        artifact_specs = OrderedDict()
        for spec in output_specs:
            artifact_spec = _artifact_spec_from_output_declaration(spec)
            artifact_specs[artifact_spec.name] = artifact_spec

        vars(func)[FunctionContractAttribute.artifact_outputs] = artifact_specs
        return func

    return decorator


def artifact_inputs(*input_specs: str | ArtifactSpec) -> Callable[[F], F]:
    """Declare named artifacts consumed by a processing function."""

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.artifact_inputs] = OrderedDict(
            (artifact_spec.name, artifact_spec)
            for artifact_spec in (
                _artifact_spec_from_input_declaration(spec)
                for spec in input_specs
            )
        )
        return func

    return decorator


def _special_parameter_names(
    parameter_names: tuple[str, ...],
    *,
    decorator_name: str,
) -> tuple[str, ...]:
    normalized = tuple(name.strip() for name in parameter_names if name.strip())
    if len(normalized) != len(parameter_names):
        raise ValueError(f"{decorator_name} parameter names cannot be empty.")
    return normalized


def special_inputs(*parameter_names: str) -> Callable[[F], F]:
    """Declare runtime-managed non-image parameters for compatibility loaders."""

    normalized = _special_parameter_names(
        parameter_names,
        decorator_name="special_inputs",
    )

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.special_inputs] = normalized
        return func

    return decorator


def _special_output_specs(
    output_specs: SpecialOutputDeclarations,
) -> SpecialOutputDeclarations:
    normalized: list[SpecialOutputDeclaration] = []
    for spec in output_specs:
        if isinstance(spec, str):
            if not spec.strip():
                raise ValueError("special_outputs names cannot be empty.")
            normalized.append(spec.strip())
            continue
        if (
            isinstance(spec, tuple)
            and len(spec) == 2
            and isinstance(spec[0], str)
            and spec[0].strip()
            and (spec[1] is None or isinstance(spec[1], MaterializationSpec))
        ):
            normalized.append((spec[0].strip(), spec[1]))
            continue
        raise ValueError(
            "special_outputs specs must be strings or "
            "(name, materialization_spec) tuples."
        )
    return tuple(normalized)


def special_outputs(*output_specs: SpecialOutputDeclaration) -> Callable[[F], F]:
    """Declare compatibility output names for absorbed CellProfiler functions."""

    normalized = _special_output_specs(output_specs)

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.special_outputs] = normalized
        return func

    return decorator


def runtime_bound_parameters(
    *parameter_types: type[RuntimeParameterDeclaration],
) -> Callable[[F], F]:
    """Declare callable parameters supplied by runtime execution infrastructure."""

    normalized = _runtime_parameter_declaration_types(
        parameter_types,
        decorator_name="runtime_bound_parameters",
    )

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.runtime_bound_parameters] = normalized
        return func

    return decorator


def _runtime_parameter_declaration_types(
    parameter_types: tuple[type[RuntimeParameterDeclaration], ...],
    *,
    decorator_name: str,
) -> tuple[type[RuntimeParameterDeclaration], ...]:
    normalized: list[type[RuntimeParameterDeclaration]] = []
    seen: set[str] = set()
    for parameter_type in parameter_types:
        if not isinstance(parameter_type, type):
            raise TypeError(
                f"{decorator_name} values must be parameter declaration types."
            )
        parameter = parameter_type.parameter()
        if not isinstance(parameter, inspect.Parameter):
            raise TypeError(
                f"{parameter_type.__name__}.parameter() must return inspect.Parameter."
            )
        parameter_name = parameter_type.require_parameter_name()
        if not isinstance(parameter_name, str) or not parameter_name.strip():
            raise TypeError(
                f"{parameter_type.__name__}.require_parameter_name() must return "
                "a non-empty string."
            )
        if parameter.name != parameter_name:
            raise TypeError(
                f"{parameter_type.__name__}.parameter() name {parameter.name!r} "
                f"does not match require_parameter_name() {parameter_name!r}."
            )
        if parameter_name in seen:
            raise ValueError(
                f"{decorator_name} declares duplicate runtime parameter "
                f"{parameter_name!r}."
            )
        normalized.append(parameter_type)
        seen.add(parameter_name)
    return tuple(normalized)


def required_variable_components(
    *components: VariableComponents,
) -> Callable[[F], F]:
    """Declare FunctionStep variable axes required by a callable."""
    normalized = tuple(
        component
        if isinstance(component, VariableComponents)
        else VariableComponents(component)
        for component in components
    )

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.required_variable_components] = normalized
        return func

    return decorator


def require_variable_component_stack(func: F) -> F:
    """Declare that a callable needs a real stacked variable-component axis."""
    vars(func)[FunctionContractAttribute.variable_component_stack_requirement] = (
        AlwaysRequiresVariableComponentStack()
    )
    return func


def variable_component_stack_requirement(
    requirement: VariableComponentStackRequirement,
) -> Callable[[F], F]:
    """Attach a typed stack-axis requirement to a callable."""
    if not isinstance(requirement, VariableComponentStackRequirement):
        raise TypeError(
            "variable_component_stack_requirement requires "
            "VariableComponentStackRequirement."
        )

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.variable_component_stack_requirement] = (
            requirement
        )
        return func

    return decorator


def allowed_group_by(*group_by_values: GroupBy) -> Callable[[F], F]:
    """Declare FunctionStep group_by values allowed by a callable."""
    normalized = tuple(
        group_by
        if isinstance(group_by, GroupBy)
        else GroupBy(group_by)
        for group_by in group_by_values
    )

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.allowed_group_by] = normalized
        return func

    return decorator


def object_label_measurement_execution(
    mode: ObjectLabelMeasurementExecution,
) -> Callable[[F], F]:
    """Declare whether object-measurement labels are slice-aligned or full-stack."""

    if not isinstance(mode, ObjectLabelMeasurementExecution):
        raise TypeError(
            "object_label_measurement_execution mode must be "
            "ObjectLabelMeasurementExecution."
        )

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.object_label_measurement_execution] = mode
        return func

    return decorator


def object_label_measurement_execution_from_callable(
    func: Callable,
) -> ObjectLabelMeasurementExecution:
    """Return the declared object-label measurement execution mode."""

    try:
        namespace = vars(func)
    except TypeError:
        return ObjectLabelMeasurementExecution.SLICE_ALIGNED
    if FunctionContractAttribute.object_label_measurement_execution not in namespace:
        return ObjectLabelMeasurementExecution.SLICE_ALIGNED
    declared = namespace[FunctionContractAttribute.object_label_measurement_execution]
    if not isinstance(declared, ObjectLabelMeasurementExecution):
        raise TypeError(
            f"{func} object-label measurement execution must be "
            "ObjectLabelMeasurementExecution."
        )
    return declared


def composed_image_payload(func: F) -> F:
    """Declare that a callable consumes its image input as a composed image set."""
    vars(func)[FunctionContractAttribute.image_payload_consumption] = (
        ImagePayloadConsumption.COMPOSED
    )
    return func


def image_payload_consumption_from_callable(
    func: Callable,
) -> ImagePayloadConsumption:
    """Return how a callable consumes its primary image payload."""
    try:
        namespace = vars(func)
    except TypeError:
        return ImagePayloadConsumption.NATURAL
    if FunctionContractAttribute.image_payload_consumption not in namespace:
        return ImagePayloadConsumption.NATURAL
    declared = namespace[FunctionContractAttribute.image_payload_consumption]
    if not isinstance(declared, ImagePayloadConsumption):
        raise TypeError(
            f"{func} image payload consumption must be ImagePayloadConsumption."
        )
    return declared


def special_input_names_from_callable(func: Callable) -> tuple[str, ...]:
    """Return declared special-input parameter names for one callable."""
    try:
        namespace = vars(func)
    except TypeError:
        return ()
    if FunctionContractAttribute.special_inputs not in namespace:
        return ()
    declared = namespace[FunctionContractAttribute.special_inputs]
    if not isinstance(declared, tuple):
        raise TypeError(
            f"{func}.{FunctionContractAttribute.special_inputs} must be a tuple."
        )
    return declared


def special_output_specs_from_callable(
    func: Callable,
) -> SpecialOutputDeclarations:
    """Return declared special-output specs for one callable."""
    try:
        namespace = vars(func)
    except TypeError:
        return ()
    if FunctionContractAttribute.special_outputs not in namespace:
        return ()
    declared = namespace[FunctionContractAttribute.special_outputs]
    if not isinstance(declared, tuple):
        raise TypeError(
            f"{func}.{FunctionContractAttribute.special_outputs} must be a tuple."
        )
    return _special_output_specs(declared)


def runtime_bound_parameter_names_from_callable(func: Callable) -> tuple[str, ...]:
    """Return callable parameters declared as runtime-supplied."""
    try:
        namespace = vars(func)
    except TypeError:
        return ()
    if FunctionContractAttribute.runtime_bound_parameters not in namespace:
        return ()
    declared = namespace[FunctionContractAttribute.runtime_bound_parameters]
    if not isinstance(declared, tuple):
        raise TypeError(
            f"{func}.{FunctionContractAttribute.runtime_bound_parameters} "
            "must be a tuple."
        )
    parameter_types = _runtime_parameter_declaration_types(
        declared,
        decorator_name="runtime_bound_parameters",
    )
    return tuple(
        parameter_type.require_parameter_name()
        for parameter_type in parameter_types
    )
