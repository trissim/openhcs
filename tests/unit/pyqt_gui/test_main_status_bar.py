from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow

from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.services.main_window_workflows import MainWindowEmbeddedWidgets
from pyqt_reactive.theming import ColorScheme


class _MainWindowStatusBarHarness(QMainWindow):
    status_message = pyqtSignal(str)


def test_status_messages_remain_in_permanent_right_lane(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))

    app = QApplication.instance() or QApplication([])
    window = _MainWindowStatusBarHarness()
    window.window_color_scheme_services = SimpleNamespace(
        get_current_color_scheme=ColorScheme
    )
    window.plate_manager_widget = SimpleNamespace(
        require_pipeline_definition_mutation_allowed=lambda: None
    )
    window.embedded_widgets = MainWindowEmbeddedWidgets()
    window.floating_windows = {}
    window.ui_bridge_lifecycle = SimpleNamespace(close=lambda: None)

    OpenHCSMainWindow.setup_status_bar(window)
    window.resize(1024, 600)
    window.show()
    app.processEvents()

    try:
        message = "Imported pipeline successfully"
        window.status_message.emit(message)
        app.processEvents()

        assert window.statusBar().currentMessage() == ""
        assert window._status_message_label.text() == message
        assert window.bottom_control_panel.isVisible()
        assert (
            window.bottom_control_panel.geometry().right()
            < window._status_message_label.geometry().left()
        )
    finally:
        window.close()
        app.processEvents()
