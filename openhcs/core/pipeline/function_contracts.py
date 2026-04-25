"""Function-level artifact contract decorators for the pipeline compiler."""

from collections import OrderedDict
from typing import Callable, TypeVar

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.processing.materialization import MaterializationSpec

F = TypeVar("F", bound=Callable)


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
