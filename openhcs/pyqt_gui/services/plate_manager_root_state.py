"""Root-state helpers for PlateManager orchestrator scopes."""

from __future__ import annotations

from objectstate.object_state import ObjectState


def root_orchestrator_scope_ids(root_state: ObjectState) -> list[str]:
    """Return the formal root plate list from RootState parameters."""
    stored_scope_ids = root_state.parameters.get("orchestrator_scope_ids")
    if stored_scope_ids is None:
        return []
    return [str(scope_id) for scope_id in stored_scope_ids]
