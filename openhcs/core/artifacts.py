"""Typed runtime artifact contracts for OpenHCS.

Artifacts are named, non-primary-image values produced or consumed by function
invocations. They cover current side-channel I/O and provide the extension point for
objects, measurements, relationships, and other richer runtime state.
"""

from abc import ABC
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import astuple, dataclass, is_dataclass, replace
from enum import Enum
from typing import ClassVar, Self, cast

from metaclass_registry import AutoRegisterMeta


class ArtifactPayloadShape(str, Enum):
    """Generic runtime payload shape required by an artifact kind."""

    ANY = "any"
    ARRAY = "array"
    TABLE = "table"
    MAPPING = "mapping"


@dataclass(frozen=True, slots=True)
class ArtifactKindOptions:
    """Optional semantic flags attached to one artifact kind."""

    uses_label_representation_payload_shape: bool = False
    participates_in_measurement_source_names: bool = False
    participates_in_main_flow_output: bool = False
    participates_in_axis_plane_identity: bool = False
    participates_in_object_domain_scope: bool = False
    participates_in_pairwise_object_domain_input: bool = False
    payload_description: str | None = None

    def description_for(
        self,
        *,
        value: str,
        payload_shape: ArtifactPayloadShape,
    ) -> str:
        if self.payload_description is not None:
            return self.payload_description
        return f"{payload_shape} {value} payload"


DEFAULT_ARTIFACT_KIND_OPTIONS = ArtifactKindOptions()


class ArtifactKind(str, Enum):
    """Closed family of runtime artifact categories."""

    def __new__(
        cls,
        value: str,
        payload_shape: "ArtifactPayloadShape",
        options: ArtifactKindOptions = DEFAULT_ARTIFACT_KIND_OPTIONS,
    ):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._payload_shape = payload_shape
        obj.uses_label_representation_payload_shape = (
            options.uses_label_representation_payload_shape
        )
        obj.participates_in_measurement_source_names = (
            options.participates_in_measurement_source_names
        )
        obj.participates_in_main_flow_output = options.participates_in_main_flow_output
        obj.participates_in_axis_plane_identity = (
            options.participates_in_axis_plane_identity
        )
        obj.participates_in_object_domain_scope = (
            options.participates_in_object_domain_scope
        )
        obj.participates_in_pairwise_object_domain_input = (
            options.participates_in_pairwise_object_domain_input
        )
        obj.payload_description = options.description_for(
            value=value,
            payload_shape=payload_shape,
        )
        return obj

    SPECIAL = ("special", ArtifactPayloadShape.ANY)
    IMAGE = (
        "image",
        ArtifactPayloadShape.ARRAY,
        ArtifactKindOptions(
            participates_in_measurement_source_names=True,
            participates_in_main_flow_output=True,
        ),
    )
    OBJECT_LABELS = (
        "object_labels",
        ArtifactPayloadShape.ARRAY,
        ArtifactKindOptions(
            participates_in_main_flow_output=True,
            participates_in_object_domain_scope=True,
            participates_in_pairwise_object_domain_input=True,
            payload_description="object_labels payload",
            uses_label_representation_payload_shape=True,
        ),
    )
    MEASUREMENTS = (
        "measurements",
        ArtifactPayloadShape.TABLE,
        ArtifactKindOptions(participates_in_axis_plane_identity=True),
    )
    RELATIONSHIPS = (
        "relationships",
        ArtifactPayloadShape.TABLE,
        ArtifactKindOptions(
            participates_in_axis_plane_identity=True,
            participates_in_object_domain_scope=True,
            participates_in_pairwise_object_domain_input=True,
        ),
    )
    TABLE = ("table", ArtifactPayloadShape.TABLE)
    SPATIAL_GRID = (
        "spatial_grid",
        ArtifactPayloadShape.MAPPING,
        ArtifactKindOptions(
            payload_description="spatial grid mapping",
            participates_in_pairwise_object_domain_input=True,
        ),
    )
    METADATA = (
        "metadata",
        ArtifactPayloadShape.MAPPING,
        ArtifactKindOptions(payload_description="metadata mapping"),
    )

    @property
    def payload_shape(self) -> "ArtifactPayloadShape":
        return ArtifactPayloadShape(self._payload_shape)


class ArtifactSidecarRole(str, Enum):
    """Named sidecar artifact roles derived from a primary artifact."""

    CROP_MASK = "crop_mask"


@dataclass(frozen=True, slots=True)
class ArtifactSidecarSpec:
    """Typed naming rule for a sidecar artifact derived from another artifact."""

    role: ArtifactSidecarRole
    separator: str = "__"

    def __post_init__(self) -> None:
        if not isinstance(self.role, ArtifactSidecarRole):
            raise TypeError(
                "ArtifactSidecarSpec.role must be an ArtifactSidecarRole, "
                f"got {type(self.role).__name__}."
            )
        if not self.separator:
            raise ValueError("ArtifactSidecarSpec.separator cannot be empty.")

    def name_for(self, primary_artifact_name: str) -> str:
        """Return the sidecar artifact name for one primary artifact."""
        normalized = primary_artifact_name.strip()
        if not normalized:
            raise ValueError("primary_artifact_name cannot be empty.")
        return f"{normalized}{self.separator}{self.role.value}"


CROP_MASK_ARTIFACT_SIDECAR = ArtifactSidecarSpec(ArtifactSidecarRole.CROP_MASK)


class ArtifactMaterializationPayload(ABC):
    """Nominal marker for rich artifact materialization metadata."""


@dataclass(frozen=True)
class ArtifactSpec:
    """Declared input or output artifact contract for a function invocation."""

    name: str
    kind: ArtifactKind = ArtifactKind.SPECIAL
    materialization: ArtifactMaterializationPayload | None = None
    required: bool = True
    sidecar_role: ArtifactSidecarRole | None = None

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.kind,
                _artifact_spec_hash_value(self.materialization),
                self.required,
                self.sidecar_role,
            )
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

    def of_kind(self, kind: ArtifactKind) -> tuple[ArtifactSpec, ...]:
        """Return specs with the requested artifact kind, preserving order."""
        resolved_kind = kind if isinstance(kind, ArtifactKind) else ArtifactKind(kind)
        return tuple(spec for spec in self.specs if spec.kind is resolved_kind)

    def by_name(self, name: str) -> ArtifactSpec | None:
        """Return the first spec with a matching artifact name."""
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None

    def by_name_and_kind(
        self,
        name: str,
        kind: ArtifactKind,
    ) -> ArtifactSpec | None:
        """Return the first spec matching both artifact name and kind."""
        resolved_kind = kind if isinstance(kind, ArtifactKind) else ArtifactKind(kind)
        for spec in self.specs:
            if spec.name == name and spec.kind is resolved_kind:
                return spec
        return None

    def unique(self, *, conflict_context: str = "artifact spec") -> tuple[ArtifactSpec, ...]:
        """Return specs de-duplicated by artifact identity, failing on conflicts."""
        unique_specs: dict[tuple[str, ArtifactKind], ArtifactSpec] = {}
        for spec in self.specs:
            key = (spec.name, spec.kind)
            existing = unique_specs.get(key)
            if existing is not None and existing != spec:
                raise ValueError(
                    f"Conflicting {conflict_context} declarations for "
                    f"{spec.kind.value}:{spec.name}."
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


@dataclass(frozen=True)
class ArtifactKey:
    """Stable identity for one artifact instance in an execution scope."""

    name: str
    kind: ArtifactKind
    scope: ArtifactScope
    semantic_id: str | None = None


@dataclass(frozen=True)
class ArtifactPlan(ABC, metaclass=AutoRegisterMeta):
    """Compiled storage plan shared by produced and consumed artifacts."""

    __registry_key__ = "plan_role"
    __skip_if_no_key__ = True

    plan_role: ClassVar[str | None] = None

    name: str
    path: str
    kind: ArtifactKind = ArtifactKind.SPECIAL
    group_keys: tuple[str | None, ...] = (None,)
    paths_by_group: Mapping[str | None, str] | None = None

    _missing_group_uses_default_path: ClassVar[bool] = False

    @classmethod
    def registered_plan_types(cls) -> tuple[type["ArtifactPlan"], ...]:
        """Return registered concrete artifact plan classes."""
        return tuple(cls.__registry__.values())

    @property
    def single_group_key(self) -> str | None:
        group_keys = self.group_keys or (None,)
        if len(group_keys) == 1:
            return group_keys[0]
        return None

    def artifact_key(self, *, axis_id: str) -> ArtifactKey:
        return ArtifactKey(
            name=self.name,
            kind=self.kind,
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

    def for_group(self, group_key: str | None) -> "ArtifactOutputPlan":
        """Return a group-specific output plan with the finalized path."""
        plan = self._plan_for_group(group_key)
        if plan is None:
            raise RuntimeError("ArtifactOutputPlan group resolution must be total.")
        return plan

    def runtime_slice_group_keys(
        self,
        *,
        requested_group_key: str | None,
        slice_count: int | None,
    ) -> tuple[str | None, ...]:
        """Return output groups that should receive projected runtime slices."""
        if requested_group_key not in (None, "default"):
            return (requested_group_key,)
        group_keys = tuple(self.group_keys or (None,))
        if (
            slice_count is not None
            and len(group_keys) == slice_count
            and all(group_key is not None for group_key in group_keys)
            and self.paths_by_group is not None
            and all(group_key in self.paths_by_group for group_key in group_keys)
        ):
            return group_keys
        return (requested_group_key,)


@dataclass(frozen=True)
class ArtifactInputPlan(ArtifactPlan):
    """Compiled storage plan for one consumed artifact."""

    plan_role: ClassVar[str] = "input"

    source_step_id: int | str | None = None
    source_step_scope_id: str | None = None

    def group_key_for_axis(
        self,
        *,
        axis_id: str,
        requested_group_key: str | None,
    ) -> str | None:
        """Return the input group key selected for one execution axis."""
        if requested_group_key not in (None, "default"):
            return requested_group_key
        axis_group_key = str(axis_id)
        if axis_group_key in self.group_keys and (
            self.paths_by_group is None
            or len(self.paths_by_group) == 0
            or axis_group_key in self.paths_by_group
        ):
            return axis_group_key
        return requested_group_key

    def for_group(self, group_key: str | None) -> "ArtifactInputPlan | None":
        """Return a group-specific input plan, or None if not available."""
        return self._plan_for_group(group_key)


@dataclass(frozen=True)
class StepResult:
    """Function return envelope for image output plus named artifacts."""

    image: "StepResultImagePayload"
    artifacts: Mapping[str, "StepResultArtifactPayload"]


class StepResultImagePayload(ABC):
    """Nominal marker for StepResult primary image payloads."""


class StepResultArtifactPayload(ABC):
    """Nominal marker for StepResult named artifact payloads."""
