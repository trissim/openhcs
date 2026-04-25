"""Typed runtime artifact contracts for OpenHCS.

Artifacts are named, non-primary-image values produced or consumed by function
invocations. They cover current side-channel I/O and provide the extension point for
objects, measurements, relationships, and other richer runtime state.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ArtifactKind(str, Enum):
    """Closed family of runtime artifact categories."""

    SPECIAL = "special"
    IMAGE = "image"
    OBJECT_LABELS = "object_labels"
    MEASUREMENTS = "measurements"
    RELATIONSHIPS = "relationships"
    TABLE = "table"
    METADATA = "metadata"


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
class ArtifactOutputPlan:
    """Compiled storage plan for one produced artifact."""

    name: str
    path: str
    kind: ArtifactKind = ArtifactKind.SPECIAL
    materialization: Any = None
    group_keys: tuple[str | None, ...] = (None,)
    paths_by_group: Mapping[str | None, str] | None = None
    producer_step_index: int | str | None = None
    producer_step_name: str | None = None

    def for_group(self, group_key: str | None) -> "ArtifactOutputPlan":
        """Return a group-specific output plan with the finalized path."""
        paths_by_group = self.paths_by_group or {None: self.path}
        if group_key in paths_by_group:
            group_path = paths_by_group[group_key]
        elif None in paths_by_group:
            group_path = paths_by_group[None]
        else:
            group_path = self.path

        return ArtifactOutputPlan(
            name=self.name,
            path=group_path,
            kind=self.kind,
            materialization=self.materialization,
            group_keys=(group_key,),
            paths_by_group={group_key: group_path},
            producer_step_index=self.producer_step_index,
            producer_step_name=self.producer_step_name,
        )


@dataclass(frozen=True)
class ArtifactInputPlan:
    """Compiled storage plan for one consumed artifact."""

    name: str
    path: str
    kind: ArtifactKind = ArtifactKind.SPECIAL
    paths_by_group: Mapping[str | None, str] | None = None
    group_keys: tuple[str | None, ...] = (None,)
    source_step_id: int | str | None = None

    def for_group(self, group_key: str | None) -> "ArtifactInputPlan | None":
        """Return a group-specific input plan, or None if not available."""
        if self.paths_by_group:
            if group_key in self.paths_by_group:
                path = self.paths_by_group[group_key]
            elif None in self.paths_by_group:
                path = self.paths_by_group[None]
            else:
                return None
        else:
            path = self.path

        return ArtifactInputPlan(
            name=self.name,
            path=path,
            kind=self.kind,
            paths_by_group={group_key: path},
            group_keys=(group_key,),
            source_step_id=self.source_step_id,
        )


@dataclass(frozen=True)
class StepResult:
    """Function return envelope for image output plus named artifacts."""

    image: Any
    artifacts: Mapping[str, Any]
