from __future__ import annotations

from PyQt6.QtWidgets import QApplication
from pyqt_reactive.theming import ColorScheme

from openhcs.core.config import GlobalPipelineConfig
from openhcs.pyqt_gui.services.service_adapter import GlobalEventBus
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget


class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


class PlateManagerServiceStub:
    """Minimal service adapter surface needed by PlateManagerWidget construction."""

    def __init__(self) -> None:
        self.global_config = GlobalPipelineConfig()
        self.color_scheme = ColorScheme()
        self.event_bus = GlobalEventBus()

    def get_global_config(self) -> GlobalPipelineConfig:
        return self.global_config

    def get_current_color_scheme(self) -> ColorScheme:
        return self.color_scheme

    def get_event_bus(self) -> GlobalEventBus:
        return self.event_bus


def test_plate_manager_constructor_initializes_qobject_before_signal_use(monkeypatch) -> None:
    QtApplicationHarness.app()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)

    widget = PlateManagerWidget(PlateManagerServiceStub())

    assert widget.debug_snapshot_available is not None
    widget.cleanup()
    widget.close()
