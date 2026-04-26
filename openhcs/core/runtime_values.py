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
    parent_object_name: str | None = None
    child_object_name: str | None = None
    object_id_field: str | None = None

    def __post_init__(self) -> None:
        if self.object_name == "":
            raise ValueError("RuntimeValueSchema.object_name cannot be empty.")
        if self.source_image_name == "":
            raise ValueError("RuntimeValueSchema.source_image_name cannot be empty.")
        if self.parent_object_name == "":
            raise ValueError("RuntimeValueSchema.parent_object_name cannot be empty.")
        if self.child_object_name == "":
            raise ValueError("RuntimeValueSchema.child_object_name cannot be empty.")
        if self.object_id_field == "":
            raise ValueError("RuntimeValueSchema.object_id_field cannot be empty.")


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


@dataclass(frozen=True, slots=True)
class NamedImage:
    """Native OpenHCS named image value."""

    name: str
    data: Any
    dimensions: tuple[str, ...] = ()
    source_image_name: str | None = None

    def __post_init__(self) -> None:
        _require_name(self.name, "NamedImage.name")
        if self.source_image_name == "":
            raise ValueError("NamedImage.source_image_name cannot be empty.")
        if not _is_array_like(self.data):
            raise TypeError(
                f"NamedImage '{self.name}' requires array-like data with "
                f"shape/ndim, got {type(self.data).__name__}."
            )


@dataclass(frozen=True, slots=True)
class ObjectLabelSet:
    """Native OpenHCS object-label value."""

    name: str
    labels: Any
    source_image_name: str | None = None
    dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_name(self.name, "ObjectLabelSet.name")
        if self.source_image_name == "":
            raise ValueError("ObjectLabelSet.source_image_name cannot be empty.")
        if not _is_array_like(self.labels):
            raise TypeError(
                f"ObjectLabelSet '{self.name}' requires array-like labels with "
                f"shape/ndim, got {type(self.labels).__name__}."
            )


@dataclass(frozen=True, slots=True)
class MeasurementTable:
    """Native OpenHCS measurement table value."""

    name: str
    rows: Any
    object_name: str | None = None
    fields: tuple[FieldSpec, ...] = ()
    object_id_field: str | None = None
    source_image_name: str | None = None

    def __post_init__(self) -> None:
        _require_name(self.name, "MeasurementTable.name")
        if self.object_name == "":
            raise ValueError("MeasurementTable.object_name cannot be empty.")
        if self.object_id_field == "":
            raise ValueError("MeasurementTable.object_id_field cannot be empty.")
        if self.source_image_name == "":
            raise ValueError("MeasurementTable.source_image_name cannot be empty.")
        if not _is_table_like(self.rows):
            raise TypeError(
                f"MeasurementTable '{self.name}' requires table-like rows, "
                f"got {type(self.rows).__name__}."
            )


@dataclass(frozen=True, slots=True)
class ObjectRelationship:
    """Native OpenHCS parent-child object relationship value."""

    name: str
    parent_object_name: str
    child_object_name: str
    parent_ids: Any
    child_ids: Any

    def __post_init__(self) -> None:
        _require_name(self.name, "ObjectRelationship.name")
        _require_name(
            self.parent_object_name,
            "ObjectRelationship.parent_object_name",
        )
        _require_name(
            self.child_object_name,
            "ObjectRelationship.child_object_name",
        )
        _validate_relationship_ids(self.parent_ids, self.child_ids, self.name)

    def as_table(self) -> dict[str, Any]:
        """Return table-like relationship columns for materialization."""
        return {
            "parent_object": self.parent_object_name,
            "child_object": self.child_object_name,
            "parent_id": self.parent_ids,
            "child_id": self.child_ids,
        }


def normalize_artifact_value(
    output_plan: ArtifactOutputPlan,
    value: Any,
    *,
    axis_id: str,
) -> RuntimeValue:
    """Normalize a raw function artifact return into a validated RuntimeValue."""
    if isinstance(value, RuntimeValue):
        return validate_runtime_value(value, output_plan, axis_id=axis_id)

    native_value = _normalize_native_value(output_plan, value, axis_id=axis_id)
    if native_value is not None:
        return validate_runtime_value(native_value, output_plan, axis_id=axis_id)

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


def _normalize_native_value(
    output_plan: ArtifactOutputPlan,
    value: Any,
    *,
    axis_id: str,
) -> RuntimeValue | None:
    if isinstance(value, NamedImage):
        _validate_native_name(output_plan, value.name)
        return _runtime_value(
            output_plan,
            value.data,
            axis_id=axis_id,
            schema=RuntimeValueSchema(
                kind=ArtifactKind.IMAGE,
                dimensions=value.dimensions,
                source_image_name=value.source_image_name,
            ),
        )
    if isinstance(value, ObjectLabelSet):
        _validate_native_name(output_plan, value.name)
        return _runtime_value(
            output_plan,
            value.labels,
            axis_id=axis_id,
            schema=RuntimeValueSchema(
                kind=ArtifactKind.OBJECT_LABELS,
                dimensions=value.dimensions,
                object_name=value.name,
                source_image_name=value.source_image_name,
            ),
        )
    if isinstance(value, MeasurementTable):
        _validate_native_name(output_plan, value.name)
        return _runtime_value(
            output_plan,
            value.rows,
            axis_id=axis_id,
            schema=RuntimeValueSchema(
                kind=ArtifactKind.MEASUREMENTS,
                fields=value.fields or _infer_fields(value.rows),
                object_name=value.object_name,
                source_image_name=value.source_image_name,
                object_id_field=value.object_id_field,
            ),
        )
    if isinstance(value, ObjectRelationship):
        _validate_native_name(output_plan, value.name)
        return _runtime_value(
            output_plan,
            value.as_table(),
            axis_id=axis_id,
            schema=RuntimeValueSchema(
                kind=ArtifactKind.RELATIONSHIPS,
                fields=(
                    FieldSpec("parent_object"),
                    FieldSpec("child_object"),
                    FieldSpec("parent_id"),
                    FieldSpec("child_id"),
                ),
                parent_object_name=value.parent_object_name,
                child_object_name=value.child_object_name,
            ),
        )
    return None


def _runtime_value(
    output_plan: ArtifactOutputPlan,
    data: Any,
    *,
    axis_id: str,
    schema: RuntimeValueSchema,
) -> RuntimeValue:
    return RuntimeValue(
        key=ArtifactKey(
            name=output_plan.name,
            kind=output_plan.kind,
            scope=ArtifactScope(
                axis_id=axis_id,
                group_key=_single_group_key(output_plan),
            ),
        ),
        data=data,
        schema=schema,
        storage=RuntimeStoragePolicy(
            backend="memory",
            path=output_plan.path,
            materialize=output_plan.materialization is not None,
        ),
    )


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
        if not _is_array_like(data):
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


def _is_array_like(data: Any) -> bool:
    return hasattr(data, "shape") or hasattr(data, "ndim")


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")


def _validate_native_name(output_plan: ArtifactOutputPlan, name: str) -> None:
    if name != output_plan.name:
        raise ValueError(
            f"Native runtime value '{name}' does not match planned artifact "
            f"'{output_plan.name}'."
        )


def _infer_fields(rows: Any) -> tuple[FieldSpec, ...]:
    columns = getattr(rows, "columns", None)
    if columns is not None:
        return tuple(FieldSpec(str(column)) for column in columns)
    if isinstance(rows, Mapping):
        return tuple(FieldSpec(str(column)) for column in rows)
    if (
        isinstance(rows, Sequence)
        and rows
        and isinstance(rows[0], Mapping)
    ):
        return tuple(FieldSpec(str(column)) for column in rows[0])
    return ()


def _validate_relationship_ids(parent_ids: Any, child_ids: Any, name: str) -> None:
    if isinstance(parent_ids, Sequence) and isinstance(child_ids, Sequence):
        if (
            not isinstance(parent_ids, (str, bytes, bytearray))
            and not isinstance(child_ids, (str, bytes, bytearray))
            and len(parent_ids) != len(child_ids)
        ):
            raise ValueError(
                f"ObjectRelationship '{name}' parent_ids and child_ids must "
                f"have equal length, got {len(parent_ids)} and {len(child_ids)}."
            )


def _single_group_key(output_plan: ArtifactOutputPlan) -> str | None:
    group_keys = output_plan.group_keys or (None,)
    if len(group_keys) == 1:
        return group_keys[0]
    return None
