"""Generic semantic contracts for typed runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from openhcs.core.artifacts import ArtifactKind


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

    DENSE_LABELS = "dense_labels"
    SPARSE_IJV = "sparse_ijv"


class MeasurementScope(str, Enum):
    """Semantic entity scope for measurement rows."""

    ARTIFACT = "artifact"
    IMAGE = "image"
    OBJECT = "object"
    RELATIONSHIP = "relationship"
    EXPERIMENT = "experiment"


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
        if scope in {
            MeasurementScope.IMAGE,
            MeasurementScope.OBJECT,
            MeasurementScope.RELATIONSHIP,
        } and self.name is None:
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

    @property
    def source_name(self) -> str:
        return self.source.name

    @property
    def target_name(self) -> str:
        return self.target.name

    @property
    def source_role(self) -> str:
        return self.source.role

    @property
    def target_role(self) -> str:
        return self.target.role

    @property
    def source_id_field(self) -> str:
        return self.source.id_field

    @property
    def target_id_field(self) -> str:
        return self.target.id_field

    @property
    def source_kind(self) -> ArtifactKind:
        return self.source.kind

    @property
    def target_kind(self) -> ArtifactKind:
        return self.target.kind


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
