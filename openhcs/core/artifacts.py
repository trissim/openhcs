"""Typed runtime artifact contracts for OpenHCS.

Artifacts are named, non-primary-image values produced or consumed by function
invocations. They cover current side-channel I/O and provide the extension point for
objects, measurements, relationships, and other richer runtime state.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, ClassVar, Mapping, Self, cast


class ArtifactPayloadShape(str, Enum):
    """Generic runtime payload shape required by an artifact kind."""

    ANY = "any"
    ARRAY = "array"
    TABLE = "table"
    MAPPING = "mapping"


class ArtifactKind(str, Enum):
    """Closed family of runtime artifact categories."""

    def __new__(
        cls,
        value: str,
        payload_shape: "ArtifactPayloadShape",
        options: Mapping[str, bool] | None = None,
    ):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._payload_shape = payload_shape
        obj._uses_label_representation_payload_shape = (
            bool((options or {}).get("uses_label_representation_payload_shape"))
        )
        obj._payload_description = (options or {}).get(
            "payload_description",
            f"{payload_shape} {value} payload",
        )
        return obj

    SPECIAL = ("special", ArtifactPayloadShape.ANY)
    IMAGE = ("image", ArtifactPayloadShape.ARRAY)
    OBJECT_LABELS = (
        "object_labels",
        ArtifactPayloadShape.ARRAY,
        {
            "payload_description": "object_labels payload",
            "uses_label_representation_payload_shape": True,
        },
    )
    MEASUREMENTS = ("measurements", ArtifactPayloadShape.TABLE)
    RELATIONSHIPS = ("relationships", ArtifactPayloadShape.TABLE)
    TABLE = ("table", ArtifactPayloadShape.TABLE)
    METADATA = (
        "metadata",
        ArtifactPayloadShape.MAPPING,
        {"payload_description": "metadata mapping"},
    )

    @property
    def payload_shape(self) -> "ArtifactPayloadShape":
        return ArtifactPayloadShape(self._payload_shape)

    @property
    def uses_label_representation_payload_shape(self) -> bool:
        return self._uses_label_representation_payload_shape

    @property
    def payload_description(self) -> str:
        return self._payload_description


@dataclass(frozen=True)
class ArtifactSpec:
    """Declared input or output artifact contract for a function invocation."""

    name: str
    kind: ArtifactKind = ArtifactKind.SPECIAL
    materialization: Any = None
    required: bool = True


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


@dataclass(frozen=True)
class ArtifactPlan:
    """Compiled storage plan shared by produced and consumed artifacts."""

    name: str
    path: str
    kind: ArtifactKind = ArtifactKind.SPECIAL
    group_keys: tuple[str | None, ...] = (None,)
    paths_by_group: Mapping[str | None, str] | None = None

    _missing_group_uses_default_path: ClassVar[bool] = False

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

    _missing_group_uses_default_path: ClassVar[bool] = True

    materialization: Any = None
    producer_step_index: int | str | None = None
    producer_step_scope_id: str | None = None
    producer_step_name: str | None = None

    def for_group(self, group_key: str | None) -> "ArtifactOutputPlan":
        """Return a group-specific output plan with the finalized path."""
        plan = self._plan_for_group(group_key)
        if plan is None:
            raise RuntimeError("ArtifactOutputPlan group resolution must be total.")
        return plan


@dataclass(frozen=True)
class ArtifactInputPlan(ArtifactPlan):
    """Compiled storage plan for one consumed artifact."""

    source_step_id: int | str | None = None
    source_step_scope_id: str | None = None

    def for_group(self, group_key: str | None) -> "ArtifactInputPlan | None":
        """Return a group-specific input plan, or None if not available."""
        return self._plan_for_group(group_key)


@dataclass(frozen=True)
class StepResult:
    """Function return envelope for image output plus named artifacts."""

    image: Any
    artifacts: Mapping[str, Any]
