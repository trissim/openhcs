"""Function-level artifact contract decorators for the pipeline compiler."""

from collections import OrderedDict
from collections.abc import Mapping
from enum import Enum
from typing import Callable, TypeVar

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
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
from openhcs.processing.materialization import MaterializationSpec

F = TypeVar("F", bound=Callable)


class ObjectLabelMeasurementExecution(str, Enum):
    """How object-measurement functions consume label domains."""

    SLICE_ALIGNED = "slice_aligned"
    FULL_STACK = "full_stack"


_OBJECT_LABEL_MEASUREMENT_EXECUTION_ATTR = "__object_label_measurement_execution__"


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

        func.__artifact_outputs__ = artifact_specs
        return func

    return decorator


def artifact_inputs(*input_specs: str | ArtifactSpec) -> Callable[[F], F]:
    """Declare named artifacts consumed by a processing function."""

    def decorator(func: F) -> F:
        func.__artifact_inputs__ = OrderedDict(
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
        func.__special_inputs__ = normalized
        return func

    return decorator


def _special_output_specs(output_specs: tuple[object, ...]) -> tuple[object, ...]:
    normalized: list[object] = []
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
        ):
            normalized.append((spec[0].strip(), spec[1]))
            continue
        raise ValueError(
            "special_outputs specs must be strings or "
            "(name, materialization_spec) tuples."
        )
    return tuple(normalized)


def special_outputs(*output_specs: object) -> Callable[[F], F]:
    """Declare compatibility output names for absorbed CellProfiler functions."""

    normalized = _special_output_specs(output_specs)

    def decorator(func: F) -> F:
        func.__special_outputs__ = normalized
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
        setattr(func, _OBJECT_LABEL_MEASUREMENT_EXECUTION_ATTR, mode)
        return func

    return decorator


def object_label_measurement_execution_from_callable(
    func: Callable,
) -> ObjectLabelMeasurementExecution:
    """Return the declared object-label measurement execution mode."""

    try:
        declared = vars(func).get(_OBJECT_LABEL_MEASUREMENT_EXECUTION_ATTR)
    except TypeError:
        declared = None
    if declared is None:
        return ObjectLabelMeasurementExecution.SLICE_ALIGNED
    if not isinstance(declared, ObjectLabelMeasurementExecution):
        raise TypeError(
            f"{func} object-label measurement execution must be "
            "ObjectLabelMeasurementExecution."
        )
    return declared


def special_input_names_from_callable(func: Callable) -> tuple[str, ...]:
    """Return declared special-input parameter names for one callable."""
    try:
        declared = vars(func).get("__special_inputs__", ())
    except TypeError:
        declared = ()
    if not isinstance(declared, tuple):
        raise TypeError(
            f"{func}.__special_inputs__ must be a tuple."
        )
    return declared
