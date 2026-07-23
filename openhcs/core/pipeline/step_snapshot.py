"""Compiler identity for already-resolved pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from openhcs.core.steps.abstract import AbstractStep

if TYPE_CHECKING:
    from objectstate import ObjectState


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    """Bind compiler identity to one already-resolved pipeline step.

    The normal compiler path has already converted ObjectState to a resolved step
    object before this snapshot is built. ObjectState contributes only its scope
    identity; all step semantics remain owned by the resolved step.
    """

    index: int
    scope_id: str
    step: AbstractStep


def build_step_snapshots(
    steps: Sequence[AbstractStep],
    step_state_map: Mapping[int, "ObjectState"],
) -> tuple[StepSnapshot, ...]:
    """Build compiler snapshots for already-resolved steps."""
    snapshots: list[StepSnapshot] = []
    for index, step in enumerate(steps):
        try:
            step_state = step_state_map[index]
        except KeyError as exc:
            raise ValueError(
                f"Missing ObjectState for resolved step {index} ({step.name})."
            ) from exc
        snapshots.append(
            StepSnapshot(index=index, scope_id=step_state.scope_id, step=step)
        )
    return tuple(snapshots)
