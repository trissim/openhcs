"""Typed runtime artifact values and validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from openhcs.core.artifacts import (
    ArtifactKey,
    ArtifactKind,
    ArtifactOutputPlan,
    ArtifactPayloadShape,
    ArtifactScope,
)
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementScope,
    MeasurementSubject,
    ObjectLabelRepresentation,
    RelationshipEndpoint,
    RelationshipSemantics,
    coerce_enum,
)


@dataclass(frozen=True, slots=True)
class RuntimeValueSchema:
    """Semantic schema attached to a runtime artifact value."""

    kind: ArtifactKind
    fields: tuple[FieldSpec, ...] = ()
    dimensions: tuple[str, ...] = ()
    label_representation: ObjectLabelRepresentation | None = None
    measurement_subject: MeasurementSubject | None = None
    relationship: RelationshipSemantics | None = None
    object_name: str | None = None
    source_image_name: str | None = None
    object_id_field: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            coerce_enum(ArtifactKind, self.kind, "RuntimeValueSchema.kind"),
        )
        if self.label_representation is not None:
            object.__setattr__(
                self,
                "label_representation",
                coerce_enum(
                    ObjectLabelRepresentation,
                    self.label_representation,
                    "RuntimeValueSchema.label_representation",
                ),
            )
        if self.object_name == "":
            raise ValueError("RuntimeValueSchema.object_name cannot be empty.")
        if self.source_image_name == "":
            raise ValueError("RuntimeValueSchema.source_image_name cannot be empty.")
        if self.object_id_field == "":
            raise ValueError("RuntimeValueSchema.object_id_field cannot be empty.")
        if (
            self.label_representation is not None
            and self.kind is not ArtifactKind.OBJECT_LABELS
        ):
            raise ValueError(
                "RuntimeValueSchema.label_representation requires "
                "OBJECT_LABELS kind."
            )
        if (
            self.measurement_subject is not None
            and self.kind is not ArtifactKind.MEASUREMENTS
        ):
            raise ValueError(
                "RuntimeValueSchema.measurement_subject requires "
                "MEASUREMENTS kind."
            )
        if (
            self.relationship is not None
            and self.kind is not ArtifactKind.RELATIONSHIPS
        ):
            raise ValueError(
                "RuntimeValueSchema.relationship requires RELATIONSHIPS kind."
            )


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
    representation: ObjectLabelRepresentation = ObjectLabelRepresentation.DENSE_LABELS

    def __post_init__(self) -> None:
        _require_name(self.name, "ObjectLabelSet.name")
        representation = coerce_enum(
            ObjectLabelRepresentation,
            self.representation,
            "ObjectLabelSet.representation",
        )
        object.__setattr__(self, "representation", representation)
        if self.source_image_name == "":
            raise ValueError("ObjectLabelSet.source_image_name cannot be empty.")
        validator = _PAYLOAD_VALIDATORS[representation.payload_shape]
        if validator is not None and not validator(self.labels):
            raise TypeError(
                f"ObjectLabelSet '{self.name}' requires "
                f"{representation.value} payload, got "
                f"{type(self.labels).__name__}."
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
    subject: MeasurementSubject | None = None

    def __post_init__(self) -> None:
        _require_name(self.name, "MeasurementTable.name")
        if self.object_name == "":
            raise ValueError("MeasurementTable.object_name cannot be empty.")
        if self.object_id_field == "":
            raise ValueError("MeasurementTable.object_id_field cannot be empty.")
        if self.source_image_name == "":
            raise ValueError("MeasurementTable.source_image_name cannot be empty.")
        subject = _resolve_measurement_subject(
            self.subject,
            artifact_name=self.name,
            object_name=self.object_name,
            object_id_field=self.object_id_field,
            source_image_name=self.source_image_name,
        )
        object.__setattr__(self, "subject", subject)
        if not _is_table_like(self.rows):
            raise TypeError(
                f"MeasurementTable '{self.name}' requires table-like rows, "
                f"got {type(self.rows).__name__}."
            )


@dataclass(frozen=True, slots=True)
class ObjectRelationship:
    """Native OpenHCS directed object relationship value."""

    name: str
    source: RelationshipEndpoint
    target: RelationshipEndpoint
    source_ids: Any
    target_ids: Any
    relationship_type: str = "related"

    def __post_init__(self) -> None:
        _require_name(self.name, "ObjectRelationship.name")
        if not isinstance(self.source, RelationshipEndpoint):
            raise TypeError(
                "ObjectRelationship.source must be RelationshipEndpoint, "
                f"got {type(self.source).__name__}."
            )
        if not isinstance(self.target, RelationshipEndpoint):
            raise TypeError(
                "ObjectRelationship.target must be RelationshipEndpoint, "
                f"got {type(self.target).__name__}."
            )
        _require_name(self.relationship_type, "ObjectRelationship.relationship_type")
        _validate_relationship_ids(self.source_ids, self.target_ids, self.name)

    @property
    def semantics(self) -> RelationshipSemantics:
        return RelationshipSemantics(
            source=self.source,
            target=self.target,
            relationship_type=self.relationship_type,
        )

    def as_table(self) -> dict[str, Any]:
        """Return table-like relationship columns for materialization."""
        return {
            "relationship_type": self.relationship_type,
            "source_role": self.source.role,
            "target_role": self.target.role,
            "source_object": self.source.name,
            "target_object": self.target.name,
            self.source.id_field: self.source_ids,
            self.target.id_field: self.target_ids,
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
                label_representation=value.representation,
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
                measurement_subject=value.subject,
                object_name=_measurement_object_name(value),
                source_image_name=_measurement_source_image_name(value),
                object_id_field=_measurement_object_id_field(value),
            ),
        )
    if isinstance(value, ObjectRelationship):
        _validate_native_name(output_plan, value.name)
        table = value.as_table()
        return _runtime_value(
            output_plan,
            table,
            axis_id=axis_id,
            schema=RuntimeValueSchema(
                kind=ArtifactKind.RELATIONSHIPS,
                fields=_infer_fields(table),
                relationship=value.semantics,
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

    _validate_payload_kind(output_plan.name, value.kind, value.data, value.schema)
    return value


def _validate_payload_kind(
    name: str,
    kind: ArtifactKind,
    data: Any,
    schema: RuntimeValueSchema,
) -> None:
    payload_shape = _payload_shape_for(kind, schema)
    validator = _PAYLOAD_VALIDATORS[payload_shape]
    if validator is None:
        return
    if validator(data):
        return
    raise TypeError(
        f"Artifact '{name}' expected {kind.payload_description}, "
        f"got {type(data).__name__}."
    )


def _payload_shape_for(
    kind: ArtifactKind,
    schema: RuntimeValueSchema,
) -> ArtifactPayloadShape:
    if kind.uses_label_representation_payload_shape:
        representation = (
            schema.label_representation or ObjectLabelRepresentation.DENSE_LABELS
        )
        return representation.payload_shape
    return kind.payload_shape


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


def _is_mapping_like(data: Any) -> bool:
    return isinstance(data, Mapping)


@dataclass(frozen=True, slots=True)
class _PayloadValidator:
    shape: ArtifactPayloadShape
    predicate: Callable[[Any], bool] | None


def _payload_validators(
    rows: tuple[_PayloadValidator, ...],
) -> Mapping[ArtifactPayloadShape, Callable[[Any], bool] | None]:
    validators = {row.shape: row.predicate for row in rows}
    if set(validators) != set(ArtifactPayloadShape):
        raise TypeError("Incomplete runtime payload validator table.")
    return MappingProxyType(validators)


_PAYLOAD_VALIDATORS = _payload_validators(
    (
        _PayloadValidator(ArtifactPayloadShape.ANY, None),
        _PayloadValidator(ArtifactPayloadShape.ARRAY, _is_array_like),
        _PayloadValidator(ArtifactPayloadShape.TABLE, _is_table_like),
        _PayloadValidator(ArtifactPayloadShape.MAPPING, _is_mapping_like),
    )
)


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")


def _validate_native_name(output_plan: ArtifactOutputPlan, name: str) -> None:
    if name != output_plan.name:
        raise ValueError(
            f"Native runtime value '{name}' does not match planned artifact "
            f"'{output_plan.name}'."
        )


def _resolve_measurement_subject(
    subject: MeasurementSubject | None,
    *,
    artifact_name: str,
    object_name: str | None,
    object_id_field: str | None,
    source_image_name: str | None,
) -> MeasurementSubject:
    if subject is None:
        if object_name is not None:
            return MeasurementSubject(
                MeasurementScope.OBJECT,
                object_name,
                object_id_field,
            )
        if source_image_name is not None:
            return MeasurementSubject(MeasurementScope.IMAGE, source_image_name)
        return MeasurementSubject(MeasurementScope.ARTIFACT, artifact_name)

    if object_name is not None and (
        subject.scope is not MeasurementScope.OBJECT or subject.name != object_name
    ):
        raise ValueError(
            "MeasurementTable.object_name conflicts with "
            "MeasurementTable.subject."
        )
    if object_id_field is not None and subject.id_field != object_id_field:
        raise ValueError(
            "MeasurementTable.object_id_field conflicts with "
            "MeasurementTable.subject."
        )
    if source_image_name is not None and (
        subject.scope is not MeasurementScope.IMAGE or subject.name != source_image_name
    ):
        raise ValueError(
            "MeasurementTable.source_image_name conflicts with "
            "MeasurementTable.subject."
        )
    return subject


def _measurement_object_name(value: MeasurementTable) -> str | None:
    if value.object_name is not None:
        return value.object_name
    if value.subject and value.subject.scope is MeasurementScope.OBJECT:
        return value.subject.name
    return None


def _measurement_object_id_field(value: MeasurementTable) -> str | None:
    if value.object_id_field is not None:
        return value.object_id_field
    if value.subject and value.subject.scope is MeasurementScope.OBJECT:
        return value.subject.id_field
    return None


def _measurement_source_image_name(value: MeasurementTable) -> str | None:
    if value.source_image_name is not None:
        return value.source_image_name
    if value.subject and value.subject.scope is MeasurementScope.IMAGE:
        return value.subject.name
    return None


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


def _validate_relationship_ids(source_ids: Any, target_ids: Any, name: str) -> None:
    if isinstance(source_ids, Sequence) and isinstance(target_ids, Sequence):
        if (
            not isinstance(source_ids, (str, bytes, bytearray))
            and not isinstance(target_ids, (str, bytes, bytearray))
            and len(source_ids) != len(target_ids)
        ):
            raise ValueError(
                f"ObjectRelationship '{name}' source_ids and target_ids must "
                f"have equal length, got {len(source_ids)} and {len(target_ids)}."
            )


def _single_group_key(output_plan: ArtifactOutputPlan) -> str | None:
    group_keys = output_plan.group_keys or (None,)
    if len(group_keys) == 1:
        return group_keys[0]
    return None
