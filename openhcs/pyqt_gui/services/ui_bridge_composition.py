"""Composition root for the PyQt UI bridge."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.pyqt_gui.services.ui_agent_bridge import (
    UiAgentBridgeService,
    UiObjectStateSnapshotProvider,
)
from openhcs.pyqt_gui.services.ui_bridge_object_state import ObjectStateBridgeProviderSet
from openhcs.pyqt_gui.services.ui_bridge_plate_manager import PlateManagerBridgeProviderSet
from openhcs.pyqt_gui.services.ui_bridge_windows import MainWindowBridgeProviderSet
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
                (
                    PlateManagerBridgeProviderSet(main_window.plate_manager_widget),
                    MainWindowBridgeProviderSet(main_window),
                    ObjectStateBridgeProviderSet(),
                )
            )
        )

    def build_service(self) -> UiAgentBridgeService:
        snapshot_provider = UiObjectStateSnapshotProvider()
        registry = UiBridgeSurfaceRegistry()
        self.provider_set.register(
            UiBridgeRegistrationContext(
                registry=registry,
                snapshot_provider=snapshot_provider,
            )
        )
        return UiAgentBridgeService(
            registry=registry,
            snapshot_provider=snapshot_provider,
        )
