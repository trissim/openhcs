"""ObjectState scope visibility policy shared by UI bridge projections."""

from __future__ import annotations

from dataclasses import dataclass, field

from openhcs.agent.dto.ui_bridge import UiObjectStateScopeVisibility


HIDDEN_OBJECT_STATE_SCOPES = frozenset(("", "__plates__"))


@dataclass(frozen=True, slots=True)
class ObjectStateScopeVisibility:
    """Agent-facing visibility policy for ObjectState scope ids."""

    request: UiObjectStateScopeVisibility = field(
        default_factory=UiObjectStateScopeVisibility
    )

    def includes_system_scopes(self) -> bool:
        return self.request.include_system_scopes

    def includes_scope_id(self, scope_id: str) -> bool:
        if self.includes_system_scopes():
            return True
        return scope_id not in HIDDEN_OBJECT_STATE_SCOPES
