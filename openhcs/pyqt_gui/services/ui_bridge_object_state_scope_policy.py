"""ObjectState scope visibility policy shared by UI bridge projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from openhcs.agent.dto.ui_bridge import UiObjectStateScopeVisibility
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId


@dataclass(frozen=True, slots=True)
class ObjectStateScopeVisibility:
    """Agent-facing visibility policy for ObjectState scope ids."""

    hidden_unaliased_system_scopes: ClassVar[frozenset[str]] = frozenset(
        ("__plates__",)
    )

    request: UiObjectStateScopeVisibility = field(
        default_factory=UiObjectStateScopeVisibility
    )

    def includes_system_scopes(self) -> bool:
        return self.request.include_system_scopes

    def includes_scope_id(self, scope_id: str) -> bool:
        if self.includes_system_scopes() or self.has_agent_alias(scope_id):
            return True
        return scope_id not in self.hidden_unaliased_system_scopes

    @staticmethod
    def has_agent_alias(scope_id: str) -> bool:
        return OpenHCSUiWindowId.agent_window_id_for_manager_scope(scope_id) != scope_id
