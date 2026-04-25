"""Typed runtime artifact values and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openhcs.core.artifacts import (
    ArtifactKey,
    ArtifactKind,
    ArtifactOutputPlan,
    ArtifactScope,
)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One named field expected in a tabular runtime value."""

    name: str
    dtype: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Runtime value field name cannot be empty.")


@dataclass(frozen=True, slots=True)
class RuntimeValueSchema:
    """Semantic schema attached to a runtime artifact value."""

    kind: ArtifactKind
    fields: tuple[FieldSpec, ...] = ()
    dimensions: tuple[str, ...] = ()
    object_name: str | None = None
    source_image_name: str | None = None

    def __post_init__(self) -> None:
        if self.object_name == "":
            raise ValueError("RuntimeValueSchema.object_name cannot be empty.")
        if self.source_image_name == "":
            raise ValueError("RuntimeValueSchema.source_image_name cannot be empty.")


@dataclass(frozen=True, slots=True)
class RuntimeStoragePolicy:
    """Storage intent for a runtime value once stores/materializers consume it."""

    backend: str | None = None
    path: str | None = None
    materialize: bool = False

    def __post_init__(self) -> None:
        if self.path and not self.backend:
            raise ValueError("RuntimeStoragePolicy.path requires a backend.")


@dataclass(frozen=True, slots=True)
class RuntimeValue:
    """Artifact payload validated against compiled runtime semantics."""

    key: ArtifactKey
    data: Any
    schema: RuntimeValueSchema
    storage: RuntimeStoragePolicy | None = None

    def __post_init__(self) -> None:
        if self.key.kind is not self.schema.kind:
            raise ValueError(
                f"RuntimeValue key kind {self.key.kind.value} does not match "
                f"schema kind {self.schema.kind.value}."
            )

    @property
    def name(self) -> str:
        return self.key.name

    @property
    def kind(self) -> ArtifactKind:
        return self.key.kind


def normalize_artifact_value(
    output_plan: ArtifactOutputPlan,
    value: Any,
    *,
    axis_id: str,
) -> RuntimeValue:
    """Normalize a raw function artifact return into a validated RuntimeValue."""
    if isinstance(value, RuntimeValue):
        return validate_runtime_value(value, output_plan, axis_id=axis_id)

    runtime_value = RuntimeValue(
        key=ArtifactKey(
            name=output_plan.name,
            kind=output_plan.kind,
            scope=ArtifactScope(
                axis_id=axis_id,
                group_key=_single_group_key(output_plan),
            ),
        ),
        data=value,
        schema=RuntimeValueSchema(kind=output_plan.kind),
        storage=RuntimeStoragePolicy(
            backend="memory",
            path=output_plan.path,
            materialize=output_plan.materialization is not None,
        ),
    )
    return validate_runtime_value(runtime_value, output_plan, axis_id=axis_id)


def validate_runtime_value(
    value: RuntimeValue,
    output_plan: ArtifactOutputPlan,
    *,
    axis_id: str,
) -> RuntimeValue:
    """Validate a runtime value against the compiled output plan."""
    if value.key.name != output_plan.name:
        raise ValueError(
            f"RuntimeValue name '{value.key.name}' does not match planned "
            f"artifact '{output_plan.name}'."
        )
    if value.kind is not output_plan.kind:
        raise ValueError(
            f"Artifact '{output_plan.name}' expected {output_plan.kind.value}, "
            f"got {value.kind.value}."
        )
    if value.schema.kind is not output_plan.kind:
        raise ValueError(
            f"Artifact '{output_plan.name}' schema kind {value.schema.kind.value} "
            f"does not match planned kind {output_plan.kind.value}."
        )
    if value.key.scope.axis_id != axis_id:
        raise ValueError(
            f"Artifact '{output_plan.name}' belongs to axis "
            f"'{value.key.scope.axis_id}', not '{axis_id}'."
        )

    _validate_payload_kind(output_plan.name, value.kind, value.data)
    return value


def _validate_payload_kind(name: str, kind: ArtifactKind, data: Any) -> None:
    if kind is ArtifactKind.SPECIAL:
        return
    if kind in {ArtifactKind.IMAGE, ArtifactKind.OBJECT_LABELS}:
        if not (hasattr(data, "shape") or hasattr(data, "ndim")):
            raise TypeError(
                f"Artifact '{name}' expected {kind.value} payload with array-like "
                f"shape/ndim, got {type(data).__name__}."
            )
        return
    if kind is ArtifactKind.METADATA:
        if not isinstance(data, Mapping):
            raise TypeError(
                f"Artifact '{name}' expected metadata mapping, got {type(data).__name__}."
            )
        return
    if kind in {
        ArtifactKind.MEASUREMENTS,
        ArtifactKind.RELATIONSHIPS,
        ArtifactKind.TABLE,
    }:
        if not _is_table_like(data):
            raise TypeError(
                f"Artifact '{name}' expected table-like {kind.value} payload, "
                f"got {type(data).__name__}."
            )


def _is_table_like(data: Any) -> bool:
    return (
        hasattr(data, "columns")
        or isinstance(data, Mapping)
        or (
            isinstance(data, Sequence)
            and not isinstance(data, (str, bytes, bytearray))
        )
    )


def _single_group_key(output_plan: ArtifactOutputPlan) -> str | None:
    group_keys = output_plan.group_keys or (None,)
    if len(group_keys) == 1:
        return group_keys[0]
    return None
