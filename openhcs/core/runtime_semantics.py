"""Generic semantic contracts for typed runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from openhcs.core.artifacts import ArtifactKind, ArtifactPayloadShape


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One named field expected in a tabular runtime value."""

    name: str
    dtype: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Runtime value field name cannot be empty.")


class ObjectLabelRepresentation(str, Enum):
    """Storage representation used by an object-label artifact payload."""

    def __new__(cls, value: str, payload_shape: ArtifactPayloadShape):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._payload_shape = payload_shape
        return obj

    DENSE_LABELS = ("dense_labels", ArtifactPayloadShape.ARRAY)
    SPARSE_IJV = ("sparse_ijv", ArtifactPayloadShape.TABLE)

    @property
    def payload_shape(self) -> ArtifactPayloadShape:
        return self._payload_shape


class MeasurementScope(str, Enum):
    """Semantic entity scope for measurement rows."""

    def __new__(cls, value: str, requires_subject_name: bool = False):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._requires_subject_name = requires_subject_name
        return obj

    ARTIFACT = ("artifact", False)
    IMAGE = ("image", True)
    OBJECT = ("object", True)
    RELATIONSHIP = ("relationship", True)
    EXPERIMENT = ("experiment", False)

    @property
    def requires_subject_name(self) -> bool:
        return self._requires_subject_name


@dataclass(frozen=True, slots=True)
class MeasurementSubject:
    """Entity measured by a measurement table."""

    scope: MeasurementScope
    name: str | None = None
    id_field: str | None = None

    def __post_init__(self) -> None:
        scope = coerce_enum(MeasurementScope, self.scope, "MeasurementSubject.scope")
        object.__setattr__(self, "scope", scope)

        if self.name == "":
            raise ValueError("MeasurementSubject.name cannot be empty.")
        if self.id_field == "":
            raise ValueError("MeasurementSubject.id_field cannot be empty.")
        if scope.requires_subject_name and self.name is None:
            raise ValueError(
                f"MeasurementSubject.name is required for {scope.value} scope."
            )


@dataclass(frozen=True, slots=True)
class RelationshipEndpoint:
    """One endpoint in a directed relationship."""

    name: str
    role: str
    id_field: str
    kind: ArtifactKind = ArtifactKind.OBJECT_LABELS

    def __post_init__(self) -> None:
        _require_name(self.name, "RelationshipEndpoint.name")
        _require_name(self.role, "RelationshipEndpoint.role")
        _require_name(self.id_field, "RelationshipEndpoint.id_field")
        object.__setattr__(
            self,
            "kind",
            coerce_enum(ArtifactKind, self.kind, "RelationshipEndpoint.kind"),
        )


@dataclass(frozen=True, slots=True)
class RelationshipSemantics:
    """Directed relationship semantics between two named runtime entities."""

    source: RelationshipEndpoint
    target: RelationshipEndpoint
    relationship_type: str = "related"

    def __post_init__(self) -> None:
        _require_name(
            self.relationship_type,
            "RelationshipSemantics.relationship_type",
        )
        if not isinstance(self.source, RelationshipEndpoint):
            raise TypeError(
                "RelationshipSemantics.source must be RelationshipEndpoint, "
                f"got {type(self.source).__name__}."
            )
        if not isinstance(self.target, RelationshipEndpoint):
            raise TypeError(
                "RelationshipSemantics.target must be RelationshipEndpoint, "
                f"got {type(self.target).__name__}."
            )


def coerce_enum(enum_type: type[Enum], value: Any, field_name: str) -> Any:
    """Normalize string-backed enum inputs while keeping validation centralized."""
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of "
            f"{', '.join(member.value for member in enum_type)}; got {value!r}."
        ) from exc


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
