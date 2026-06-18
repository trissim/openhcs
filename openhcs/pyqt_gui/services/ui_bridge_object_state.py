"""ObjectState scope projection provider for the PyQt UI bridge."""

from __future__ import annotations

from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiObjectStateScopeCatalog,
    UiObjectStateScopeIdentity,
    UiObjectStateScopeListRequest,
    UiObjectStateScopeSummary,
    UiTimeTravelRuntimeState,
)
from openhcs.config_framework import ObjectState, ObjectStateRegistry
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiObjectStateScopeProviderABC,
    UiObjectStateScopeProviderIdentity,
)
from openhcs.pyqt_gui.services.ui_bridge_object_state_scope_policy import (
    ObjectStateScopeVisibility,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
)


OBJECT_STATE_SCOPE_PROVIDER_ID = "object_state.scopes"


class ObjectStateScopeProjectionService:
    """Project ObjectState registry entries into bounded agent DTOs."""

    def catalog(
        self,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        visibility = ObjectStateScopeVisibility(request)
        scopes = tuple(
            self.summary(state)
            for state in ObjectStateRegistry.get_all()
            if visibility.includes_scope_id(state.scope_id)
        )
        return UiObjectStateScopeCatalog(
            schema_version=SCHEMA_VERSION,
            object_state_token=ObjectStateRegistry.get_token(),
            current_branch=ObjectStateRegistry.get_current_branch(),
            current_snapshot_index=ObjectStateRegistry.get_current_snapshot_index(),
            time_travel_state=UiTimeTravelRuntimeState(
                active=ObjectStateRegistry.is_time_traveling()
            ),
            scopes=scopes,
        )

    @staticmethod
    def summary(state: ObjectState) -> UiObjectStateScopeSummary:
        return UiObjectStateScopeSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiObjectStateScopeIdentity(
                object_state_scope_id=state.scope_id,
            ),
            object_type=type(state.object_instance).__name__,
            parameter_count=len(state.parameters),
            dirty_field_count=len(state.dirty_fields),
            signature_diff_field_count=len(state.signature_diff_fields),
            last_changed_field=state.last_changed_field,
        )


class ObjectStateScopeProvider(UiObjectStateScopeProviderABC):
    """ObjectState scope catalog provider."""

    identity = UiObjectStateScopeProviderIdentity(
        provider_id=OBJECT_STATE_SCOPE_PROVIDER_ID,
        title="ObjectState scopes",
    )

    def __init__(self) -> None:
        self._projection = ObjectStateScopeProjectionService()

    def catalog(
        self,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        return self._projection.catalog(request)


class ObjectStateBridgeProviderSet(UiBridgeProviderSetABC):
    """Provider set for ObjectState registry projections."""

    registry_key = OBJECT_STATE_SCOPE_PROVIDER_ID

    def register(self, context: UiBridgeRegistrationContext) -> None:
        context.registry.register_object_state_scope_provider(ObjectStateScopeProvider())
