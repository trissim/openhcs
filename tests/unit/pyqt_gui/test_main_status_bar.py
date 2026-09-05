from __future__ import annotations

import gc
from types import SimpleNamespace
from weakref import ref

from objectstate.object_state import ObjectStateRegistry
from PyQt6.QtCore import QCoreApplication, QEvent, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow
from pyqt_reactive.theming import ColorScheme

from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.services.main_window_workflows import MainWindowEmbeddedWidgets


class _MainWindowStatusBarHarness(QMainWindow):
    status_message = pyqtSignal(str)

    def _on_time_travel_complete(self, _dirty_states, _triggering_scope) -> None:
        pass

    def _on_object_state_unregistered(self, _scope_id, _state) -> None:
        pass


class _MainWindowLifecycleHarness(_MainWindowStatusBarHarness):
    _close_object_state_subscriptions = (
        OpenHCSMainWindow._close_object_state_subscriptions
    )

    def closeEvent(self, event) -> None:
        OpenHCSMainWindow.closeEvent(self, event)


def _configure_status_bar_harness(window) -> None:
    window.window_color_scheme_services = SimpleNamespace(
        get_current_color_scheme=ColorScheme
    )
    window.plate_manager_widget = SimpleNamespace(
        require_pipeline_definition_mutation_allowed=lambda: None
    )
    window.embedded_widgets = MainWindowEmbeddedWidgets()
    window.floating_windows = {}
    window.ui_bridge_lifecycle = SimpleNamespace(close=lambda: None)
    window.window_services = SimpleNamespace()
    OpenHCSMainWindow.setup_status_bar(window)


def test_status_messages_remain_in_permanent_right_lane(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))

    app = QApplication.instance() or QApplication([])
    window = _MainWindowStatusBarHarness()
    _configure_status_bar_harness(window)
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
        window.time_travel_widget.close()
        window.close()
        app.processEvents()


def test_reopened_main_window_has_one_live_status_subscription(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("PyQt6.QtCore.QTimer.singleShot", lambda *_args: None)
    app = QApplication.instance() or QApplication([])

    def build_window() -> _MainWindowLifecycleHarness:
        window = _MainWindowLifecycleHarness()
        _configure_status_bar_harness(window)
        OpenHCSMainWindow._connect_object_state_lifecycle(window)
        window.dock_layout_store = SimpleNamespace(save=lambda _window: None)
        window.shortcut_lifecycle = SimpleNamespace(close=lambda: None)
        window.lifecycle_workflow = SimpleNamespace(close=lambda: None)
        return window

    first = build_window()
    first_widget = first.time_travel_widget
    first_ref = ref(first)
    first_widget_ref = ref(first_widget)
    assert len(ObjectStateRegistry._on_history_changed_callbacks) == 1
    assert len(ObjectStateRegistry._on_time_travel_complete_callbacks) == 1
    assert len(ObjectStateRegistry._on_unregister_callbacks) == 1

    first.show()
    app.processEvents()
    first.close()
    app.processEvents()
    assert ObjectStateRegistry._on_history_changed_callbacks == []
    assert ObjectStateRegistry._on_time_travel_complete_callbacks == []
    assert ObjectStateRegistry._on_unregister_callbacks == []

    second = build_window()
    assert [
        callback.__self__
        for callback in ObjectStateRegistry._on_history_changed_callbacks
    ] == [second.time_travel_widget]
    assert [
        callback.__self__
        for callback in ObjectStateRegistry._on_time_travel_complete_callbacks
    ] == [second]
    assert [
        callback.__self__ for callback in ObjectStateRegistry._on_unregister_callbacks
    ] == [second]

    second.close()
    first.deleteLater()
    second.deleteLater()
    del first_widget
    del first
    del second
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()
    assert first_ref() is None
    assert first_widget_ref() is None
