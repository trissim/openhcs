"""Composition root for the PyQt UI bridge."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.pyqt_gui.services.ui_agent_bridge import (
    UiAgentBridgeService,
    UiBridgeOperationTracker,
    UiObjectStateSnapshotProvider,
)
from openhcs.pyqt_gui.services.ui_bridge_object_state import ObjectStateBridgeProviderSet
from openhcs.pyqt_gui.services.ui_bridge_plate_manager import PlateManagerBridgeProviderSet
from openhcs.pyqt_gui.services.ui_bridge_pipeline_editor import (
    PipelineEditorBridgeProviderSet,
)
from openhcs.pyqt_gui.services.ui_bridge_live_overview import (
    LiveOverviewBridgeProviderSet,
)
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
                    PipelineEditorBridgeProviderSet(main_window.pipeline_editor_widget),
                    MainWindowBridgeProviderSet(main_window),
                    ObjectStateBridgeProviderSet(),
                    LiveOverviewBridgeProviderSet(),
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
