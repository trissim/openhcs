from __future__ import annotations

from types import MethodType, SimpleNamespace

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QDialog, QTreeWidgetItem

from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.windows.managed_windows import LogViewerWindowWrapper
from pyqt_reactive.services.zmq_server_info import BaseServerInfo
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.widgets.shared.zmq_server_browser_widget import (
    ZMQServerBrowserWidgetABC,
)
from zmqruntime.messages import PongResponse, ServerRole


class _SignalHarness:
    def __init__(self) -> None:
        self._callback = None

    def connect(self, callback) -> None:
        self._callback = callback

    def emit(self, value: str) -> None:
        assert self._callback is not None
        self._callback(value)


class _LogViewerWindowHarness:
    def __init__(self) -> None:
        self.opened_path = None
        self.cleanup_count = 0

    def switch_to_log(self, log_file_path) -> None:
        self.opened_path = log_file_path

    def cleanup(self) -> None:
        self.cleanup_count += 1


class _LogViewerWrapperHarness(LogViewerWindowWrapper):
    def __init__(self, child) -> None:
        QDialog.__init__(self)
        self.widget = child


def test_show_window_preserves_window_manager_result(monkeypatch) -> None:
    managed_window = SimpleNamespace(hide=lambda: None)
    main_window = SimpleNamespace(
        _create_window_factory=lambda _window_id: lambda: managed_window,
        window_specs={
            "log_viewer": SimpleNamespace(initialize_on_startup=True),
        },
        _ensure_flash_overlay=lambda _window: None,
    )
    monkeypatch.setattr(
        WindowManager,
        "show_or_focus",
        classmethod(lambda _cls, _scope_id, _factory: managed_window),
    )

    result = OpenHCSMainWindow.show_window(
        main_window,
        "log_viewer",
        hide_if_startup=False,
    )

    assert result is managed_window


def test_zmq_server_log_double_click_routes_to_shown_log_window(tmp_path) -> None:
    """The visible server-row gesture routes through the managed log window."""
    log_file_path = tmp_path / "execution.log"
    log_file_path.write_text("ready\n")
    log_window = _LogViewerWindowHarness()
    main_window = SimpleNamespace(show_log_viewer=lambda: log_window)
    main_window._open_log_file_in_viewer = MethodType(
        OpenHCSMainWindow._open_log_file_in_viewer,
        main_window,
    )

    log_file_opened = _SignalHarness()
    log_file_opened.connect(main_window._open_log_file_in_viewer)
    browser = SimpleNamespace(log_file_opened=log_file_opened)
    server_row = QTreeWidgetItem()
    server_row.setData(
        0,
        Qt.ItemDataRole.UserRole,
        BaseServerInfo.from_response(
            PongResponse(
                port=5555,
                control_port=6555,
                ready=True,
                server="ExecutionServer",
                server_role=ServerRole.EXECUTION,
                log_file_path=str(log_file_path),
            )
        ),
    )

    ZMQServerBrowserWidgetABC._on_item_double_clicked(browser, server_row)

    assert log_window.opened_path == log_file_path


def test_log_viewer_wrapper_owns_child_log_switch(tmp_path) -> None:
    log_file_path = tmp_path / "execution.log"
    child = _LogViewerWindowHarness()
    wrapper = SimpleNamespace(widget=child)

    LogViewerWindowWrapper.switch_to_log(wrapper, log_file_path)

    assert child.opened_path == log_file_path


def test_log_viewer_wrapper_closes_child_lifecycle(qapp) -> None:
    child = _LogViewerWindowHarness()
    wrapper = _LogViewerWrapperHarness(child)

    wrapper.closeEvent(QCloseEvent())

    assert child.cleanup_count == 1
