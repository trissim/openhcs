from __future__ import annotations

from functools import partialmethod
from types import MethodType, SimpleNamespace

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QDialog, QTreeWidgetItem

from openhcs.core.execution_state import ManagerExecutionState
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.windows.managed_windows import LogViewerWindowWrapper
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
from pyqt_reactive.services.zmq_server_info import BaseServerInfo
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.widgets import StatusState
from pyqt_reactive.widgets.shared.zmq_server_browser_widget import (
    ZMQServerBrowserWidgetABC,
)
from zmqruntime.messages import PongResponse, ServerRole
from zmqruntime.startup import EndpointStartupPhase, EndpointStartupStatus


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


class _StatusIndicatorHarness:
    def __init__(self) -> None:
        self.state = None
        self.text = None
        self.tooltip = None

    def set_state(self, state, text) -> None:
        self.state = state
        self.text = text

    present_checking = partialmethod(set_state, StatusState.CHECKING)
    present_connected = partialmethod(set_state, StatusState.CONNECTED)
    present_disconnected = partialmethod(set_state, StatusState.DISCONNECTED)
    present_warning = partialmethod(set_state, StatusState.WARNING)

    def setToolTip(self, tooltip) -> None:
        self.tooltip = tooltip


def test_show_window_preserves_window_manager_result(monkeypatch) -> None:
    managed_window = SimpleNamespace(hide=lambda: None)
    startup_presentations = []
    main_window = SimpleNamespace(
        _create_window_factory=lambda _window_id: lambda: managed_window,
        window_specs={
            "log_viewer": SimpleNamespace(
                apply_startup_presentation=lambda window, requested: (
                    startup_presentations.append((window, requested))
                )
            ),
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
    assert startup_presentations == [(managed_window, False)]


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


def test_zmq_startup_status_projects_to_persistent_indicator() -> None:
    indicator = _StatusIndicatorHarness()
    messages = []
    main_window = SimpleNamespace(
        _zmq_status_indicator=indicator,
        status_message=SimpleNamespace(emit=messages.append),
    )
    status = EndpointStartupStatus(
        phase=EndpointStartupPhase.PREPARING_CAPABILITIES,
        message="Discovering functions in the execution process",
    )

    OpenHCSMainWindow._apply_zmq_connection_status(main_window, status)

    assert indicator.state is StatusState.WARNING
    assert indicator.text == "ZMQ: Discovering functions in the execution process"
    assert indicator.tooltip == status.message
    assert messages == [status.message]


def test_completed_batch_keeps_gui_owned_zmq_client_session() -> None:
    disconnect_calls = []
    messages = []
    refresh_calls = []
    manager = SimpleNamespace(
        execution_state=object(),
        current_execution_id="execution-1",
        _batch_workflow_service=SimpleNamespace(
            disconnect_async=lambda: disconnect_calls.append(True),
        ),
        global_config=SimpleNamespace(
            analysis_consolidation_config=SimpleNamespace(enabled=False),
        ),
        status_message=SimpleNamespace(emit=messages.append),
        refresh_execution_ui=lambda: refresh_calls.append(True),
    )

    PlateManagerWidget._finalize_all_plates_completed_ui(manager, 1, 0)

    assert manager.execution_state is ManagerExecutionState.IDLE
    assert manager.current_execution_id is None
    assert disconnect_calls == []
    assert messages == ["All done: 1 completed, 0 failed"]
    assert refresh_calls == [True]
