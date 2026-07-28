"""Composition root for the PyQt UI bridge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from openhcs.pyqt_gui.services.ui_agent_bridge import (
    UiAgentBridgeService,
    UiBridgeOperationTracker,
    UiObjectStateSnapshotProvider,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    CompositeUiBridgeProviderSet,
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
    UiBridgeSurfaceRegistry,
)


@dataclass(frozen=True, slots=True)
class OpenHCSUiBridgeCompositionRoot:
    """Build a UI bridge service from registered provider sets."""

    provider_set: UiBridgeProviderSetABC
    object_state_mutation_authorizer: Callable[[str], None]
    snapshot_restore_authorizer: Callable[[], None]

    @classmethod
    def for_main_window(cls, main_window) -> "OpenHCSUiBridgeCompositionRoot":
        return cls(
            CompositeUiBridgeProviderSet(
                tuple(
                    provider_set_type.for_main_window(main_window)
                    for provider_set_type in UiBridgeProviderSetABC.__registry__.values()
                    if provider_set_type.compose_for_main_window
                )
            ),
            (
                main_window.plate_manager_widget.require_pipeline_definition_mutation_allowed_for_scope
            ),
            (
                main_window.plate_manager_widget.require_pipeline_definition_mutation_allowed
            ),
        )

    def build_service(self) -> UiAgentBridgeService:
        snapshot_provider = UiObjectStateSnapshotProvider(
            before_restore=self.snapshot_restore_authorizer,
        )
        operation_tracker = UiBridgeOperationTracker()
        registry = UiBridgeSurfaceRegistry()
        registry.register_live_overview_contributor(operation_tracker)
        self.provider_set.register(
            UiBridgeRegistrationContext(
                registry=registry,
                snapshot_provider=snapshot_provider,
            )
        )
        return UiAgentBridgeService(
            registry=registry,
            snapshot_provider=snapshot_provider,
            operation_tracker=operation_tracker,
            object_state_mutation_authorizer=(self.object_state_mutation_authorizer),
        )
