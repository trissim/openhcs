from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openhcs.agent.dto.ui_bridge import UiObjectStateFieldMutationRequest
from objectstate import ObjectState, ObjectStateRegistry
from openhcs.pyqt_gui.services.ui_bridge_object_state import (
    ObjectStateFieldMutationService,
)


@dataclass
class MutableConfig:
    threshold: int = 1


class InheritableMode(Enum):
    ENABLED = "enabled"
    INHERIT = None


@dataclass
class EnumConfig:
    mode: InheritableMode | None = InheritableMode.ENABLED


def test_object_state_field_authorization_rejects_before_write() -> None:
    ObjectStateRegistry.clear()
    state = ObjectState(MutableConfig(), scope_id="plate::functionstep_0")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    request = UiObjectStateFieldMutationRequest.from_fields(
        object_state_scope_id=state.scope_id,
        field_path="threshold",
        value=5,
    )
    service = ObjectStateFieldMutationService(
        before_mutation=lambda _scope_id: (_ for _ in ()).throw(
            RuntimeError("mutation rejected")
        )
    )

    try:
        result = service.mutate(request)
        assert not result.mutated
        assert result.errors[0].code == "ui_object_state_field_mutation_failed"
        assert state.parameters["threshold"] == 1
    finally:
        ObjectStateRegistry.clear()


def test_object_state_field_mutation_accepts_declared_enum_member_name() -> None:
    ObjectStateRegistry.clear()
    state = ObjectState(EnumConfig(), scope_id="enum-config")
    ObjectStateRegistry.register(state, _skip_snapshot=True)

    try:
        result = ObjectStateFieldMutationService().mutate(
            UiObjectStateFieldMutationRequest.from_fields(
                object_state_scope_id=state.scope_id,
                field_path="mode",
                value="INHERIT",
            )
        )

        assert result.errors == ()
        assert result.mutated
        assert state.parameters["mode"] is InheritableMode.INHERIT
    finally:
        ObjectStateRegistry.clear()
