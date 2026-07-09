"""Typed runtime artifact contracts for OpenHCS.

Artifacts are named, non-primary-image values produced or consumed by function
invocations. They cover current side-channel I/O and provide the extension point for
objects, measurements, relationships, and other richer runtime state.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import astuple, asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, cast

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.component_set import ComponentSet

if TYPE_CHECKING:
    from openhcs.core.runtime_values import RuntimeValueSchema
    from openhcs.processing.materialization import MaterializationSpec


class ArtifactPayloadShape(str, Enum):
    """Generic runtime payload shape required by an artifact kind."""

    ANY = "any"
    ARRAY = "array"
    TABLE = "table"
    MAPPING = "mapping"


class ArtifactType(ABC, metaclass=AutoRegisterMeta):
    """Registered runtime artifact payload category."""

    __registry_key__ = "value"
    __skip_if_no_key__ = True

    value: ClassVar[str | None] = None
    payload_shape: ClassVar[ArtifactPayloadShape] = ArtifactPayloadShape.ANY
    uses_label_representation_payload_shape: ClassVar[bool] = False
    materialization_uses_source_identity_filename: ClassVar[bool] = False
    participates_in_measurement_source_names: ClassVar[bool] = False
    participates_in_main_flow_output: ClassVar[bool] = False
    participates_in_axis_plane_identity: ClassVar[bool] = False
    participates_in_object_domain_scope: ClassVar[bool] = False
    participates_in_pairwise_object_domain_input: ClassVar[bool] = False
    supports_inputless_artifact_only_execution: ClassVar[bool] = False
    runtime_record_uses_payload_slice_count: ClassVar[bool] = True
    payload_description: ClassVar[str | None] = None

    @classmethod
    def coerce(cls, artifact_type: "ArtifactTypeValue") -> type["ArtifactType"]:
        """Return the registered artifact type for a class or wire value."""
        if isinstance(artifact_type, str):
            try:
                return cls.__registry__[artifact_type]
            except KeyError as exc:
                raise ValueError(f"Unknown artifact type {artifact_type!r}.") from exc
        if isinstance(artifact_type, type) and issubclass(artifact_type, cls):
            return artifact_type
        raise TypeError(
            "Artifact type must be an ArtifactType class or registered value, "
            f"got {type(artifact_type).__name__}."
        )

    @classmethod
    def require_value(cls) -> str:
        if cls.value is None:
            raise TypeError(f"{cls.__name__} does not declare an artifact type value.")
        return cls.value

    @classmethod
    def description(cls) -> str:
        if cls.payload_description is not None:
            return cls.payload_description
        return f"{cls.payload_shape} {cls.require_value()} payload"

    @classmethod
    def diagnostic_label(cls) -> str:
        """Return the stable artifact type label used in diagnostics."""
        return f"<{cls.__name__}: {cls.require_value()!r}>"

    @classmethod
    def default_materialization_spec(
        cls,
        schema: "RuntimeValueSchema",
    ) -> "MaterializationSpec | None":
        """Return this artifact type's default materialization, if any."""
        del schema
        return None

    @classmethod
    def has_default_materialization(cls) -> bool:
        """Return whether this artifact type declares default materialization."""
        return any(
            base is not ArtifactType and "default_materialization_spec" in vars(base)
            for base in cls.__mro__
        )

    @classmethod
    def exports_as_table(cls) -> bool:
        """Return whether this artifact type materializes as a table export."""
        return cls.payload_shape is ArtifactPayloadShape.TABLE


ArtifactTypeValue = type[ArtifactType] | str


def artifact_type_strategy_key_from_class(name: str, cls: type[object]) -> str | None:
    """Return the nominal strategy key for a class declaring an ArtifactType member."""
    del name
    member = cls.__dict__.get("artifact_type")
    if isinstance(member, type) and issubclass(member, ArtifactType):
        return artifact_type_strategy_key(member)
    return None


def artifact_type_strategy_key(artifact_type: type[ArtifactType]) -> str:
    """Return the JSON-safe nominal key for an artifact-type strategy."""
    return f"{artifact_type.__module__}.{artifact_type.__qualname__}"


class ArtifactTypeStrategyMatchMixin:
    """MRO match hook for strategy roots selected by ArtifactType inheritance."""

    artifact_type: ClassVar[type[ArtifactType] | None] = None
    __key_extractor__ = staticmethod(artifact_type_strategy_key_from_class)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        member = cls.__dict__.get("artifact_type")
        if (
            isinstance(member, type)
            and issubclass(member, ArtifactType)
            and cls.__dict__.get("strategy_key") is None
        ):
            cls.strategy_key = artifact_type_strategy_key(member)

    @classmethod
    def for_artifact_type(cls, artifact_type: ArtifactTypeValue):
        return cls.for_context(ArtifactType.coerce(artifact_type))

    def matches(self, context: type[ArtifactType]) -> bool:
        artifact_type = type(self).artifact_type
        return artifact_type is not None and issubclass(
            ArtifactType.coerce(context), artifact_type
        )


class CsvMaterializedArtifactType(ArtifactType):
    """Artifact type with CSV table materialization."""

    runtime_record_uses_payload_slice_count = False

    @classmethod
    def default_materialization_spec(
        cls,
        schema: "RuntimeValueSchema",
    ) -> "MaterializationSpec":
        from openhcs.processing.materialization import csv_only

        field_names = [field.name for field in schema.fields]
        fields: list[str] | None = field_names or None
        return csv_only(suffix=".csv", fields=fields)


class JsonMaterializedArtifactType(ArtifactType):
    """Artifact type with JSON materialization."""

    @classmethod
    def default_materialization_spec(
        cls,
        schema: "RuntimeValueSchema",
    ) -> "MaterializationSpec":
        del schema
        from openhcs.processing.materialization import json_only

        return json_only(suffix=".json")


class ObjectLabelMaterializedArtifactType(ArtifactType):
    """Artifact type with object-label ROI materialization."""

    @classmethod
    def default_materialization_spec(
        cls,
        schema: "RuntimeValueSchema",
    ) -> "MaterializationSpec":
        del schema
        from openhcs.processing.materialization import segmentation_mask_rois

        return segmentation_mask_rois()


class SpecialArtifactType(ArtifactType):
    """Generic artifact type for explicit side-channel payloads."""

    value = "special"


class ImageArtifactType(ArtifactType):
    """Image array artifact type."""

    value = "image"
    payload_shape = ArtifactPayloadShape.ARRAY
    materialization_uses_source_identity_filename = True
    participates_in_measurement_source_names = True
    participates_in_main_flow_output = True


class ObjectLabelsArtifactType(ObjectLabelMaterializedArtifactType):
    """Object-label array artifact type."""

    value = "object_labels"
    payload_shape = ArtifactPayloadShape.ARRAY
    materialization_uses_source_identity_filename = True
    participates_in_main_flow_output = True
    participates_in_object_domain_scope = True
    participates_in_pairwise_object_domain_input = True
    payload_description = "object_labels payload"
    uses_label_representation_payload_shape = True


class MeasurementsArtifactType(CsvMaterializedArtifactType):
    """Measurement-table artifact type."""

    value = "measurements"
    payload_shape = ArtifactPayloadShape.TABLE
    participates_in_axis_plane_identity = True
    supports_inputless_artifact_only_execution = True


class RelationshipsArtifactType(CsvMaterializedArtifactType):
    """Relationship-table artifact type."""

    value = "relationships"
    payload_shape = ArtifactPayloadShape.TABLE
    participates_in_axis_plane_identity = True
    participates_in_object_domain_scope = True
    participates_in_pairwise_object_domain_input = True
    supports_inputless_artifact_only_execution = True
    materialized_fields: ClassVar[tuple[str, ...]] = (
        "relationship_type",
        "source_role",
        "target_role",
        "source_object",
        "target_object",
        "parent_id",
        "child_id",
        "slice_index",
        "slice_count",
    )

    @classmethod
    def default_materialization_spec(
        cls,
        schema: "RuntimeValueSchema",
    ) -> "MaterializationSpec":
        del schema
        from openhcs.processing.materialization import csv_only

        return csv_only(suffix=".csv", fields=list(cls.materialized_fields))


class TableArtifactType(CsvMaterializedArtifactType):
    """Generic table artifact type."""

    value = "table"
    payload_shape = ArtifactPayloadShape.TABLE


class SpatialGridArtifactType(JsonMaterializedArtifactType):
    """Spatial-grid mapping artifact type."""

    value = "spatial_grid"
    payload_shape = ArtifactPayloadShape.MAPPING
    payload_description = "spatial grid mapping"
    participates_in_pairwise_object_domain_input = True


class MetadataArtifactType(JsonMaterializedArtifactType):
    """Metadata mapping artifact type."""

    value = "metadata"
    payload_shape = ArtifactPayloadShape.MAPPING
    payload_description = "metadata mapping"


class ArtifactSidecarRole(str, Enum):
    """Named sidecar artifact roles derived from a primary artifact."""

    CROP_MASK = "crop_mask"

    def name_for(
        self,
        primary_artifact_name: str,
        *,
        separator: str = "__",
    ) -> str:
        """Return the sidecar artifact name for one primary artifact."""
        if not separator:
            raise ValueError("ArtifactSidecarRole sidecar separator cannot be empty.")
        normalized = primary_artifact_name.strip()
        if not normalized:
            raise ValueError("primary_artifact_name cannot be empty.")
        return f"{normalized}{separator}{self.value}"


class ArtifactMaterializationPayload(ABC):
    """Nominal marker for rich artifact materialization metadata."""

    def uses_source_identity_filename_for_artifact_type(
        self,
        artifact_type: type[ArtifactType],
    ) -> bool:
        """Return whether this materialization names files by source identity."""
        return artifact_type.materialization_uses_source_identity_filename


def _coerce_artifact_plan_type(
    plan_type: type["ArtifactPlan"],
) -> type["ArtifactPlan"]:
    if isinstance(plan_type, type) and issubclass(plan_type, ArtifactPlan):
        return plan_type
    raise TypeError(
        "Artifact plan type must be an ArtifactPlan class, "
        f"got {type(plan_type).__name__}."
    )


def _require_registered_artifact_plan_type(
    plan_type: type["ArtifactPlan"],
    field_name: str,
) -> type["ArtifactPlan"]:
    resolved_plan_type = _coerce_artifact_plan_type(plan_type)
    if resolved_plan_type not in ArtifactPlan.__registry__.values():
        raise ValueError(
            f"{field_name} is not a registered ArtifactPlan type: "
            f"{resolved_plan_type.__name__}."
        )
    return resolved_plan_type


def _require_registered_artifact_type(
    artifact_type: ArtifactTypeValue,
    field_name: str,
) -> type[ArtifactType]:
    resolved_artifact_type = ArtifactType.coerce(artifact_type)
    if resolved_artifact_type not in ArtifactType.__registry__.values():
        raise ValueError(
            f"{field_name} is not a registered ArtifactType: "
            f"{resolved_artifact_type.__name__}."
        )
    return resolved_artifact_type


@dataclass(frozen=True)
class ArtifactSpecRef:
    """Scope-free identity for one declared artifact spec."""

    plan_type: type["ArtifactPlan"]
    artifact_type: type[ArtifactType]
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_type",
            _require_registered_artifact_plan_type(
                self.plan_type,
                "ArtifactSpecRef.plan_type",
            ),
        )
        object.__setattr__(
            self,
            "artifact_type",
            _require_registered_artifact_type(
                self.artifact_type,
                "ArtifactSpecRef.artifact_type",
            ),
        )
        if not self.name:
            raise ValueError("ArtifactSpecRef.name cannot be empty.")

    def payload(self) -> dict[str, object]:
        return {
            "plan_role": self.plan_type.plan_role,
            "artifact_type": self.artifact_type.value,
            "name": self.name,
        }

    @classmethod
    def input(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
    ) -> "ArtifactSpecRef":
        return cls(
            plan_type=ArtifactInputPlan,
            artifact_type=ArtifactType.coerce(artifact_type),
            name=name,
        )

    @classmethod
    def output(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
    ) -> "ArtifactSpecRef":
        return cls(
            plan_type=ArtifactOutputPlan,
            artifact_type=ArtifactType.coerce(artifact_type),
            name=name,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ArtifactSpecRef":
        return cls(
            plan_type=ArtifactPlan.__registry__[str(payload["plan_role"])],
            artifact_type=ArtifactType.coerce(str(payload["artifact_type"])),
            name=str(payload["name"]),
        )


@dataclass(frozen=True)
class ArtifactSpecRelation(ABC, metaclass=AutoRegisterMeta):
    """Nominal relation tag attached to one declared artifact spec."""

    __registry_key__ = "relation_key"
    __skip_if_no_key__ = True

    relation_key: ClassVar[str | None] = None
    target_plan_type: ClassVar[type["ArtifactPlan"] | None] = None
    target_artifact_type: ClassVar[type[ArtifactType] | None] = None

    source: ArtifactSpecRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, ArtifactSpecRef):
            raise TypeError(
                "ArtifactSpecRelation.source must be an ArtifactSpecRef, "
                f"got {type(self.source).__name__}."
            )
        if self.target_plan_type is not None:
            _require_registered_artifact_plan_type(
                self.target_plan_type,
                f"{type(self).__name__}.target_plan_type",
            )
        if self.target_artifact_type is not None:
            _require_registered_artifact_type(
                self.target_artifact_type,
                f"{type(self).__name__}.target_artifact_type",
            )

    def require_target_spec(self, spec: "ArtifactSpec") -> None:
        if self.target_plan_type is not None:
            target_plan_type = _coerce_artifact_plan_type(self.target_plan_type)
            if spec.plan_type is not target_plan_type:
                raise ValueError(
                    f"{type(self).__name__} requires target plan role "
                    f"{target_plan_type.plan_role}, got {spec.plan_type.plan_role}."
                )
        if self.target_artifact_type is None:
            return
        target_artifact_type = ArtifactType.coerce(self.target_artifact_type)
        if spec.artifact_type is not target_artifact_type:
            raise ValueError(
                f"{type(self).__name__} requires target artifact type "
                f"{target_artifact_type.value}, got {spec.artifact_type.value}."
            )

    def payload(self) -> dict[str, object]:
        return {
            "relation_key": type(self).relation_key,
            "source": self.source.payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ArtifactSpecRelation":
        source_payload = payload["source"]
        if not isinstance(source_payload, Mapping):
            raise TypeError("Artifact relation source payload must be a mapping.")
        return cls(source=ArtifactSpecRef.from_payload(source_payload))


class ArtifactGroupScopeSourceRelation(ArtifactSpecRelation, ABC):
    """Target artifact inherits group scope from a declared source artifact."""


class GroupLineageSourceRelation(ArtifactGroupScopeSourceRelation):
    """Target artifact inherits grouping from a declared source artifact."""

    relation_key: ClassVar[str] = "group_lineage_source"


class SourceStackLineageSourceRelation(GroupLineageSourceRelation):
    """Target artifact preserves source-stack compatibility with a source artifact."""

    relation_key: ClassVar[str] = "source_stack_lineage_source"


@dataclass(frozen=True)
class ArtifactSpec:
    """Declared artifact contract for one plan role and one artifact type."""

    name: str
    plan_type: type["ArtifactPlan"]
    artifact_type: type[ArtifactType]
    materialization: ArtifactMaterializationPayload | None = None
    required: bool = True
    sidecar_role: ArtifactSidecarRole | None = None
    relations: tuple[ArtifactSpecRelation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_type",
            _require_registered_artifact_plan_type(
                self.plan_type,
                "ArtifactSpec.plan_type",
            ),
        )
        object.__setattr__(
            self,
            "artifact_type",
            _require_registered_artifact_type(
                self.artifact_type,
                "ArtifactSpec.artifact_type",
            ),
        )
        object.__setattr__(self, "relations", tuple(self.relations))
        for relation in self.relations:
            if not isinstance(relation, ArtifactSpecRelation):
                raise TypeError(
                    "ArtifactSpec.relations must contain ArtifactSpecRelation "
                    f"values, got {type(relation).__name__}."
                )
            relation.require_target_spec(self)

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.plan_type,
                self.artifact_type,
                _artifact_spec_hash_value(self.materialization),
                self.required,
                self.sidecar_role,
                self.relations,
            )
        )

    def ref(self) -> ArtifactSpecRef:
        """Return the scope-free identity for this declaration."""
        return ArtifactSpecRef(
            plan_type=self.plan_type,
            artifact_type=self.artifact_type,
            name=self.name,
        )

    @classmethod
    def input(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
        **kwargs,
    ) -> "ArtifactSpec":
        return cls(
            name=name,
            plan_type=ArtifactInputPlan,
            artifact_type=ArtifactType.coerce(artifact_type),
            **kwargs,
        )

    @classmethod
    def output(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
        **kwargs,
    ) -> "ArtifactSpec":
        return cls(
            name=name,
            plan_type=ArtifactOutputPlan,
            artifact_type=ArtifactType.coerce(artifact_type),
            **kwargs,
        )

    def for_plan_type(self, plan_type: type["ArtifactPlan"]) -> "ArtifactSpec":
        """Return this declaration with the same payload term under another role."""
        target_plan_type = _coerce_artifact_plan_type(plan_type)
        return replace(
            self,
            plan_type=target_plan_type,
            relations=tuple(
                relation
                for relation in self.relations
                if relation.target_plan_type is None
                or relation.target_plan_type is target_plan_type
            ),
        )

    @classmethod
    def output_inheriting_group_scope(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
        source: ArtifactSpecRef | ArtifactSpec,
        **kwargs,
    ) -> ArtifactSpec:
        """Declare an output artifact whose group scope follows a source artifact."""
        source_ref = source.ref() if isinstance(source, ArtifactSpec) else source
        if not isinstance(source_ref, ArtifactSpecRef):
            raise TypeError(
                "ArtifactSpec.output_inheriting_group_scope source must be an "
                f"ArtifactSpec or ArtifactSpecRef, got {type(source).__name__}."
            )
        relations = tuple(kwargs.pop("relations", ()))
        return cls.output(
            name,
            artifact_type,
            relations=(
                *relations,
                GroupLineageSourceRelation(source=source_ref),
            ),
            **kwargs,
        )

    @classmethod
    def output_preserving_source_stack_scope(
        cls,
        name: str,
        artifact_type: ArtifactTypeValue,
        source: ArtifactSpecRef | ArtifactSpec,
        **kwargs,
    ) -> ArtifactSpec:
        """Declare an output artifact that remains compatible with the source stack."""
        source_ref = source.ref() if isinstance(source, ArtifactSpec) else source
        if not isinstance(source_ref, ArtifactSpecRef):
            raise TypeError(
                "ArtifactSpec.output_preserving_source_stack_scope source must be "
                f"an ArtifactSpec or ArtifactSpecRef, got {type(source).__name__}."
            )
        relations = tuple(kwargs.pop("relations", ()))
        return cls.output(
            name,
            artifact_type,
            relations=(
                *relations,
                SourceStackLineageSourceRelation(source=source_ref),
            ),
            **kwargs,
        )

    def materialization_uses_source_identity_filename(self) -> bool:
        """Return whether this spec's materialized files require source identity."""
        if self.materialization is None:
            return self.artifact_type.materialization_uses_source_identity_filename
        return self.materialization.uses_source_identity_filename_for_artifact_type(
            self.artifact_type
        )


def _artifact_spec_hash_value(value) -> Hashable:
    """Project rich artifact metadata into a hashable equality-compatible value."""
    if is_dataclass(value):
        return (type(value), _artifact_spec_hash_value(astuple(value)))
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    (
                        _artifact_spec_hash_value(key),
                        _artifact_spec_hash_value(item),
                    )
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_artifact_spec_hash_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_artifact_spec_hash_value(item) for item in value)
    if isinstance(value, Hashable):
        return value
    raise TypeError(
        "ArtifactSpec materialization metadata contains unsupported "
        f"unhashable value {type(value).__name__}."
    )


@dataclass(slots=True)
class ArtifactSpecCollection:
    """Ordered query surface over declared artifact specs."""

    specs: tuple[ArtifactSpec, ...]

    def __init__(self, specs: Iterable[ArtifactSpec]):
        normalized = tuple(specs)
        for spec in normalized:
            if not isinstance(spec, ArtifactSpec):
                raise TypeError(
                    "ArtifactSpecCollection requires ArtifactSpec values, "
                    f"got {type(spec).__name__}."
                )
        self.specs = normalized

    def of_artifact_type(
        self,
        artifact_type: ArtifactTypeValue,
    ) -> tuple[ArtifactSpec, ...]:
        """Return specs with the requested artifact type, preserving order."""
        resolved_artifact_type = ArtifactType.coerce(artifact_type)
        return tuple(
            spec for spec in self.specs if spec.artifact_type is resolved_artifact_type
        )

    def for_plan_type(
        self,
        plan_type: type["ArtifactPlan"],
    ) -> "ArtifactSpecCollection":
        """Return specs with the requested artifact plan role."""
        resolved_plan_type = _coerce_artifact_plan_type(plan_type)
        return ArtifactSpecCollection(
            spec for spec in self.specs if spec.plan_type is resolved_plan_type
        )

    def names_for_plan_type(
        self,
        plan_type: type["ArtifactPlan"],
    ) -> tuple[str, ...]:
        """Return names for specs with the requested artifact plan role."""
        return self.for_plan_type(plan_type).names()

    def for_artifact_type(
        self,
        artifact_type: ArtifactTypeValue,
    ) -> "ArtifactSpecCollection":
        """Return specs with the requested artifact type."""
        return ArtifactSpecCollection(self.of_artifact_type(artifact_type))

    def names(self) -> tuple[str, ...]:
        """Return artifact names in collection order."""
        return tuple(spec.name for spec in self.specs)

    def name_set(self) -> frozenset[str]:
        """Return artifact names as a set."""
        return frozenset(self.names())

    def names_of_artifact_type(
        self,
        artifact_type: ArtifactTypeValue,
    ) -> tuple[str, ...]:
        """Return names for specs with the requested artifact type."""
        return ArtifactSpecCollection(self.of_artifact_type(artifact_type)).names()

    def name_set_of_artifact_type(
        self,
        artifact_type: ArtifactTypeValue,
    ) -> frozenset[str]:
        """Return names for specs with the requested artifact type as a set."""
        return frozenset(self.names_of_artifact_type(artifact_type))

    def by_name(self, name: str) -> ArtifactSpec | None:
        """Return the first spec with a matching artifact name."""
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None

    def by_name_and_artifact_type(
        self,
        name: str,
        artifact_type: ArtifactTypeValue,
    ) -> ArtifactSpec | None:
        """Return the first spec matching artifact name and type."""
        resolved_artifact_type = ArtifactType.coerce(artifact_type)
        for spec in self.specs:
            if spec.name == name and spec.artifact_type is resolved_artifact_type:
                return spec
        return None

    def ref_set(self) -> frozenset[ArtifactSpecRef]:
        """Return full declared artifact references."""
        return frozenset(spec.ref() for spec in self.specs)

    def by_ref(self, ref: ArtifactSpecRef) -> ArtifactSpec | None:
        """Return one spec by full artifact reference."""
        matches = tuple(spec for spec in self.specs if spec.ref() == ref)
        if len(matches) > 1:
            raise ValueError(f"Duplicate artifact spec ref {ref!r}.")
        if not matches:
            return None
        return matches[0]

    def relation_refs(
        self,
        relation_type: type[ArtifactSpecRelation],
    ) -> tuple[tuple[ArtifactSpec, ArtifactSpecRelation], ...]:
        """Return specs and relation tags of the requested relation family."""
        if not isinstance(relation_type, type) or not issubclass(
            relation_type,
            ArtifactSpecRelation,
        ):
            raise TypeError(
                "relation_type must be an ArtifactSpecRelation type, "
                f"got {type(relation_type).__name__}."
            )
        return tuple(
            (spec, relation)
            for spec in self.specs
            for relation in spec.relations
            if isinstance(relation, relation_type)
        )

    def validate_registered_relation_refs(self, *, owner_name: str) -> None:
        """Validate that registered relation sources target declared specs."""
        refs = self.ref_set()
        for relation_type in ArtifactSpecRelation.__registry__.values():
            unknown = tuple(
                relation.source
                for _spec, relation in self.relation_refs(relation_type)
                if relation.source not in refs
            )
            if unknown:
                raise ValueError(
                    f"{owner_name} declares {relation_type.__name__} references "
                    f"to unknown artifact specs: {unknown!r}."
                )

    def unique(
        self, *, conflict_context: str = "artifact spec"
    ) -> tuple[ArtifactSpec, ...]:
        """Return specs de-duplicated by artifact identity, failing on conflicts."""
        unique_specs: dict[ArtifactSpecRef, ArtifactSpec] = {}
        for spec in self.specs:
            key = spec.ref()
            if key in unique_specs and unique_specs[key] != spec:
                raise ValueError(
                    f"Conflicting {conflict_context} declarations for "
                    f"{spec.plan_type.plan_role}:{spec.artifact_type.value}:{spec.name}."
                )
            unique_specs[key] = spec
        return tuple(unique_specs.values())


@dataclass(frozen=True)
class ArtifactScope:
    """Execution scope for artifact identity."""

    axis_id: str
    group_key: str | None = None
    site: str | None = None
    channel: str | None = None
    z_index: str | None = None
    timepoint: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Return the transport representation for this artifact scope."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "ArtifactScope":
        return cls(
            **{
                field.name: data[field.name]
                for field in fields(cls)
                if field.name in data
            }
        )


@dataclass(frozen=True)
class ArtifactKey:
    """Stable identity for one artifact instance in an execution scope."""

    name: str
    artifact_type: type[ArtifactType]
    scope: ArtifactScope
    semantic_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_type",
            ArtifactType.coerce(self.artifact_type),
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return the transport representation for this artifact identity."""
        record = asdict(self)
        record["artifact_type"] = self.artifact_type.value
        return record

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "ArtifactKey":
        scope = data["scope"]
        if not isinstance(scope, Mapping):
            raise TypeError("ArtifactKey.scope must be a mapping.")
        semantic_id = data.get("semantic_id")
        return cls(
            name=str(data["name"]),
            artifact_type=ArtifactType.coerce(str(data["artifact_type"])),
            scope=ArtifactScope.from_json_dict(scope),
            semantic_id=None if semantic_id is None else str(semantic_id),
        )


@dataclass(frozen=True)
class ArtifactPlan(ABC, metaclass=AutoRegisterMeta):
    """Compiled storage plan shared by produced and consumed artifacts."""

    __registry_key__ = "plan_role"
    __skip_if_no_key__ = True

    plan_role: ClassVar[str | None] = None

    name: str
    path: str
    artifact_type: type[ArtifactType] = SpecialArtifactType
    group_keys: tuple[str | None, ...] = (None,)
    group_component: AllComponents | None = None
    paths_by_group: Mapping[str | None, str] | None = None
    sidecar_role: ArtifactSidecarRole | None = None

    _missing_group_uses_default_path: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_type",
            ArtifactType.coerce(self.artifact_type),
        )
        if self.group_component is not None:
            object.__setattr__(
                self,
                "group_component",
                ComponentSet.coerce_component(self.group_component),
            )

    @property
    def single_group_key(self) -> str | None:
        group_keys = self.group_keys or (None,)
        if len(group_keys) == 1:
            return group_keys[0]
        return None

    @property
    def has_dynamic_group_scope(self) -> bool:
        """Return whether concrete runtime groups are discovered during execution."""
        group_keys = self.group_keys or (None,)
        return self.group_component is not None and group_keys == (None,)

    def require_single_group_key(self) -> str | None:
        """Return the only artifact group key, failing for ambiguous groups."""
        group_keys = self.group_keys or (None,)
        if len(group_keys) == 1:
            return group_keys[0]
        raise RuntimeError(
            f"Artifact plan '{self.name}' requires one group key, "
            f"got {group_keys!r}."
        )

    def artifact_key(self, *, axis_id: str) -> ArtifactKey:
        return ArtifactKey(
            name=self.name,
            artifact_type=self.artifact_type,
            scope=ArtifactScope(
                axis_id=axis_id,
                group_key=self.single_group_key,
            ),
        )

    def _path_for_group(self, group_key: str | None) -> str | None:
        if not self.paths_by_group:
            return self.path
        if group_key in self.paths_by_group:
            return self.paths_by_group[group_key]
        if (
            group_key is not None
            and self.group_component is not None
            and None in self.paths_by_group
        ):
            return grouped_artifact_path(self.paths_by_group[None], group_key)
        if None in self.paths_by_group:
            return self.paths_by_group[None]
        if self._missing_group_uses_default_path:
            return self.path
        return None

    def _plan_for_group(self, group_key: str | None) -> Self | None:
        group_path = self._path_for_group(group_key)
        if group_path is None:
            return None
        return cast(
            Self,
            replace(
                self,
                path=group_path,
                group_keys=(group_key,),
                paths_by_group={group_key: group_path},
            ),
        )


@dataclass(frozen=True)
class ArtifactOutputPlan(ArtifactPlan):
    """Compiled storage plan for one produced artifact."""

    plan_role: ClassVar[str] = "output"
    _missing_group_uses_default_path: ClassVar[bool] = True

    materialization: ArtifactMaterializationPayload | None = None
    producer_step_index: int | str | None = None
    producer_step_scope_id: str | None = None
    producer_step_name: str | None = None

    def materialization_uses_source_identity_filename(self) -> bool:
        """Return whether this output's materialized files require source identity."""
        if self.materialization is None:
            return self.artifact_type.materialization_uses_source_identity_filename
        return self.materialization.uses_source_identity_filename_for_artifact_type(
            self.artifact_type
        )

    def for_group(self, group_key: str | None) -> "ArtifactOutputPlan":
        """Return a group-specific output plan with the finalized path."""
        plan = self._plan_for_group(group_key)
        if plan is None:
            raise RuntimeError("ArtifactOutputPlan group resolution must be total.")
        return plan

    def runtime_record_group_keys(
        self,
        *,
        requested_group_key: str | None,
        scoped_group_key: str | None = None,
        slice_count: int | None,
    ) -> tuple[str | None, ...]:
        """Return compiler-planned groups that should receive runtime records."""
        group_keys = tuple(self.group_keys or (None,))
        concrete_group_keys = tuple(
            group_key for group_key in group_keys if group_key is not None
        )
        if requested_group_key is not None:
            if requested_group_key in group_keys:
                return (requested_group_key,)
            if scoped_group_key is not None and scoped_group_key in group_keys:
                return (scoped_group_key,)
            if len(group_keys) == 1 and group_keys[0] is not None:
                return group_keys
            if not concrete_group_keys:
                return (requested_group_key,)
            return ()
        if scoped_group_key is not None:
            if scoped_group_key in group_keys:
                return (scoped_group_key,)
            if len(group_keys) == 1:
                return group_keys
            return ()
        if not concrete_group_keys:
            return (requested_group_key,)
        if not self.artifact_type.runtime_record_uses_payload_slice_count:
            return concrete_group_keys
        if slice_count is None:
            if len(concrete_group_keys) == 1:
                return concrete_group_keys
            return ()
        if len(concrete_group_keys) != slice_count:
            if len(concrete_group_keys) == 1:
                return concrete_group_keys
            return (requested_group_key,)
        return concrete_group_keys


@dataclass(frozen=True)
class ArtifactInputPlan(ArtifactPlan):
    """Compiled storage plan for one consumed artifact."""

    plan_role: ClassVar[str] = "input"

    source_step_id: int | str | None = None
    source_step_scope_id: str | None = None

    def for_group(self, group_key: str | None) -> "ArtifactInputPlan | None":
        """Return a group-specific input plan, or None if not available."""
        return self._plan_for_group(group_key)

    def path_for_runtime_query(self, group_key: str | None) -> str:
        """Return the persisted input path addressed by a runtime query."""
        group_path = self._path_for_group(group_key)
        if group_path is not None:
            return group_path
        return self.path


GroupLineageSourceRelation.target_plan_type = ArtifactOutputPlan


def grouped_artifact_path(base_path: str, group_key: str) -> str:
    """Return the existing grouped artifact path form for a runtime group."""
    path = Path(base_path)
    filename = path.name
    if "_" not in filename:
        return str(path.with_name(f"{path.stem}_w{group_key}{path.suffix}"))
    axis_id, rest = filename.split("_", 1)
    return str(path.parent / f"{axis_id}_w{group_key}_{rest}")


@dataclass(frozen=True)
class StepResult:
    """Function return envelope for image output plus named artifacts."""

    image: "StepResultImagePayload"
    artifacts: Mapping[str, "StepResultArtifactPayload"]


@dataclass(frozen=True)
class NoMainFlowOutput:
    """Nominal return value for invocations that record artifacts only."""


class StepResultImagePayload(ABC):
    """Nominal marker for StepResult primary image payloads."""


class StepResultArtifactPayload(ABC):
    """Nominal marker for StepResult named artifact payloads."""
