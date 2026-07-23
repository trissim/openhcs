"""Composition root for the PyQt UI bridge."""

from __future__ import annotations

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

    @classmethod
    def for_main_window(cls, main_window) -> "OpenHCSUiBridgeCompositionRoot":
        return cls(
            CompositeUiBridgeProviderSet(
                tuple(
                    provider_set_type.for_main_window(main_window)
                    for provider_set_type in UiBridgeProviderSetABC.__registry__.values()
                    if provider_set_type.compose_for_main_window
                )
            )
        )

    def build_service(self) -> UiAgentBridgeService:
        snapshot_provider = UiObjectStateSnapshotProvider()
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
        )
