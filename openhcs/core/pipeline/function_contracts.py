"""Function-level artifact contract decorators for the pipeline compiler."""

from collections.abc import Sequence
from dataclasses import replace
from enum import Enum
from functools import lru_cache
import inspect
from types import UnionType
from typing import Annotated, Any, Callable, TypeVar, Union, get_args, get_origin, get_type_hints

from python_introspect import add_parameter_exclusions

from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    SpecialArtifactType,
)
from openhcs.core.callable_contract import (
    CallableContract,
    CallableMetadata,
    FunctionStepExecutionScope,
    ImagePayloadConsumption,
)
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.callable_contract import RuntimeParameterDeclaration
from openhcs.core.variable_component_stack_requirement import (
    AlwaysRequiresVariableComponentStack,
    VariableComponentStackRequirement,
)
from openhcs.processing.materialization import MaterializationSpec

F = TypeVar("F", bound=Callable)


@lru_cache(maxsize=256)
def resolved_callable_type_hints(func: Callable) -> dict[str, Any]:
    """Return the callable's resolved type contract or propagate its error."""

    return get_type_hints(func)


def annotation_accepts_runtime_type(annotation: object, value_type: type[Any]) -> bool:
    """Return whether a resolved annotation explicitly accepts a nominal type."""

    if annotation in (inspect.Signature.empty, Any):
        return False
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return any(
            annotation_accepts_runtime_type(member, value_type)
            for member in get_args(annotation)
        )
    if origin is Annotated:
        annotated_type, *_metadata = get_args(annotation)
        return annotation_accepts_runtime_type(annotated_type, value_type)
    if origin in (Sequence, tuple, list):
        return any(
            member is not Ellipsis
            and annotation_accepts_runtime_type(member, value_type)
            for member in get_args(annotation)
        )
    nominal_type = origin if isinstance(origin, type) else annotation
    return isinstance(nominal_type, type) and issubclass(value_type, nominal_type)


def annotation_produces_runtime_type(
    annotation: object,
    runtime_type: type[Any],
) -> bool:
    """Return whether every value declared by an annotation has a nominal type."""

    if annotation in (inspect.Signature.empty, Any):
        return False
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return all(
            annotation_produces_runtime_type(member, runtime_type)
            for member in get_args(annotation)
        )
    if origin is Annotated:
        annotated_type, *_metadata = get_args(annotation)
        return annotation_produces_runtime_type(annotated_type, runtime_type)
    nominal_type = origin if isinstance(origin, type) else annotation
    return isinstance(nominal_type, type) and issubclass(nominal_type, runtime_type)


def resolved_callable_parameter(
    func: Callable,
    parameter_name: str,
) -> inspect.Parameter:
    """Return one callable parameter with its runtime type hint resolved."""

    signature = inspect.signature(func)
    parameter = signature.parameters.get(parameter_name)
    if parameter is None:
        raise ValueError(
            f"Callable {func.__name__!r} does not declare parameter "
            f"{parameter_name!r}."
        )
    annotation = resolved_callable_type_hints(func).get(
        parameter_name,
        parameter.annotation,
    )
    return parameter.replace(annotation=annotation)


class ObjectLabelInputExecutionMode(str, Enum):
    """How a callable consumes object-label special-input domains."""

    SLICE_ALIGNED = "slice_aligned"
    FULL_STACK = "full_stack"
    MATCH_IMAGE_STACK = "match_image_stack"

    def preserves_full_stack(self, *, image_stack_required: bool) -> bool:
        """Return whether binding must preserve the declared label stack."""

        return self is self.FULL_STACK or (
            self is self.MATCH_IMAGE_STACK and image_stack_required
        )


def _artifact_spec_from_output_declaration(
    spec: str | ArtifactSpec | tuple[str, MaterializationSpec | None],
) -> ArtifactSpec:
    """Normalize one output declaration into an ArtifactSpec."""
    if isinstance(spec, ArtifactSpec):
        return spec.for_plan_type(ArtifactOutputPlan)

    if isinstance(spec, str):
        name = spec.strip()
        if not name:
            raise ValueError("Artifact output names cannot be empty.")
        return ArtifactSpec.output(name, SpecialArtifactType)

    if isinstance(spec, tuple) and len(spec) == 2:
        key, mat_spec = spec
        if not isinstance(key, str):
            raise ValueError(f"Artifact output key must be string, got {type(key)}: {key}")
        name = key.strip()
        if not name:
            raise ValueError("Artifact output names cannot be empty.")
        if mat_spec is not None and not isinstance(mat_spec, MaterializationSpec):
            raise ValueError(
                "Materialization spec must be a MaterializationSpec or None. "
                f"Got {type(mat_spec)} for key '{key}'."
            )
        return ArtifactSpec.output(
            name,
            SpecialArtifactType,
            materialization=mat_spec,
        )

    raise ValueError(
        f"Invalid artifact output spec: {spec}. "
        "Must be string, ArtifactSpec, or "
        "(string, MaterializationSpec) tuple."
    )


def _artifact_spec_from_input_declaration(
    spec: str | ArtifactSpec,
) -> ArtifactSpec:
    """Normalize one input declaration into an ArtifactSpec."""
    if isinstance(spec, ArtifactSpec):
        return spec.for_plan_type(ArtifactInputPlan)
    if isinstance(spec, str):
        return ArtifactSpec.input(
            spec,
            SpecialArtifactType,
            parameter_name=spec,
        )
    raise ValueError(
        f"Invalid artifact input spec: {spec}. "
        "Must be string or ArtifactSpec."
    )


def artifact_outputs(
    *output_specs: str | ArtifactSpec | tuple[str, MaterializationSpec | None],
) -> Callable[[F], F]:
    """Declare named artifacts produced by a processing function."""

    def decorator(func: F) -> F:
        artifact_specs = ArtifactSpecCollection(
            _artifact_spec_from_output_declaration(spec) for spec in output_specs
        ).unique(conflict_context=f"{func.__name__} artifact output")
        vars(func)[FunctionContractAttribute.artifact_outputs] = artifact_specs
        return func

    return decorator


# Persisted user functions may use the original public spelling. Both names
# intentionally expose the same decorator and therefore the same contract metadata.
special_outputs = artifact_outputs


def artifact_inputs(*input_specs: str | ArtifactSpec) -> Callable[[F], F]:
    """Declare named artifacts consumed by a processing function."""

    def decorator(func: F) -> F:
        artifact_specs = ArtifactSpecCollection(
            _artifact_spec_from_input_declaration(spec) for spec in input_specs
        ).specs
        vars(func)[FunctionContractAttribute.artifact_inputs] = artifact_specs
        add_parameter_exclusions(
            func,
            tuple(
                spec.parameter_name
                for spec in artifact_specs
                if spec.parameter_name is not None
            ),
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
        add_parameter_exclusions(func, normalized)
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
        add_parameter_exclusions(
            func,
            tuple(parameter_type.require_parameter_name() for parameter_type in normalized),
        )
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


def execution_scope(
    scope: FunctionStepExecutionScope,
) -> Callable[[F], F]:
    """Declare the lifecycle scope for a FunctionStep callable."""
    if not isinstance(scope, FunctionStepExecutionScope):
        raise TypeError("execution_scope requires FunctionStepExecutionScope.")

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.execution_scope] = scope
        return func

    return decorator


def object_label_input_execution_mode(
    mode: ObjectLabelInputExecutionMode,
) -> Callable[[F], F]:
    """Declare how a callable consumes object-label special inputs."""

    if not isinstance(mode, ObjectLabelInputExecutionMode):
        raise TypeError(
            "object_label_input_execution_mode mode must be "
            "ObjectLabelInputExecutionMode."
        )

    def decorator(func: F) -> F:
        vars(func)[FunctionContractAttribute.object_label_input_execution_mode] = mode
        return func

    return decorator


def object_label_input_execution_mode_from_callable(
    func: Callable,
) -> ObjectLabelInputExecutionMode:
    """Return the declared object-label special-input execution mode."""

    try:
        namespace = vars(func)
    except TypeError:
        return ObjectLabelInputExecutionMode.SLICE_ALIGNED
    if FunctionContractAttribute.object_label_input_execution_mode not in namespace:
        return ObjectLabelInputExecutionMode.SLICE_ALIGNED
    declared = namespace[FunctionContractAttribute.object_label_input_execution_mode]
    if not isinstance(declared, ObjectLabelInputExecutionMode):
        raise TypeError(
            f"{func} object-label input execution mode must be "
            "ObjectLabelInputExecutionMode."
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
    return CallableMetadata.from_callable(func).image_payload_consumption


def special_input_names_from_callable(func: Callable) -> tuple[str, ...]:
    """Return normalized artifact-fed names under the compatibility API."""

    return CallableMetadata.from_callable(func).artifact_input_parameter_names


def special_input_parameters_from_callable(
    func: Callable,
) -> tuple[inspect.Parameter, ...]:
    """Return special-input parameters in their canonical signature order."""

    declared_names = special_input_names_from_callable(func)
    signature = inspect.signature(func)
    missing = tuple(
        name for name in declared_names if name not in signature.parameters
    )
    if missing:
        raise ValueError(
            f"Callable {func.__name__!r} declares absent special-input "
            f"parameters {missing!r}."
        )
    ordered = tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.name in declared_names
    )
    if tuple(parameter.name for parameter in ordered) != declared_names:
        raise ValueError(
            f"Callable {func.__name__!r} special-input order {declared_names!r} "
            "conflicts with its canonical signature order."
        )
    return tuple(
        resolved_callable_parameter(func, parameter.name)
        for parameter in ordered
    )


def special_input_parameter_accepts_sequence(
    parameter: inspect.Parameter,
) -> bool:
    """Return whether one callable parameter declares an ordered value sequence."""

    return get_origin(parameter.annotation) in (Sequence, tuple, list)


def validate_artifact_input_parameter_bindings(
    func: Callable,
    specs: Sequence[ArtifactSpec],
    *,
    adapter_manages_inputs: bool,
) -> None:
    """Compatibility wrapper for generic contract-owned binding validation."""

    del adapter_manages_inputs
    contract = CallableContract.from_callable(func)
    contract = replace(
        contract,
        metadata=replace(contract.metadata, artifact_inputs=tuple(specs)),
    )
    contract.validate_artifact_input_parameter_bindings()


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
