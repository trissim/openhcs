from __future__ import annotations

import asyncio
from concurrent.futures import Future
from functools import partialmethod
from types import MethodType, SimpleNamespace

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QDialog, QTreeWidgetItem
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.services.zmq_server_info import BaseServerInfo
from pyqt_reactive.services.zmq_server_scan_service import (
    EndpointObservationSnapshot,
)
from pyqt_reactive.widgets import StatusState
from pyqt_reactive.widgets.shared.zmq_server_browser_widget import (
    ZMQServerBrowserWidgetABC,
)
from zmqruntime import EndpointConnectionCancelledError
from zmqruntime.messages import PongResponse, ServerRole
from zmqruntime.startup import EndpointStartupPhase, EndpointStartupStatus

from openhcs.core.execution_state import (
    ManagerExecutionState,
)
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.services.zmq_version_restart import ZMQVersionRestartWorkflow
from openhcs.pyqt_gui.windows.managed_windows import LogViewerWindowWrapper


class _SignalHarness:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *values: object) -> None:
        assert self._callbacks
        for callback in self._callbacks:
            callback(*values)


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


def test_zmq_startup_status_only_commits_to_endpoint_authority() -> None:
    indicator = _StatusIndicatorHarness()
    messages = []
    refreshes = []
    observations = []
    main_window = SimpleNamespace(
        _zmq_status_indicator=indicator,
        runtime_context=SimpleNamespace(
            ui_config=SimpleNamespace(
                zmq=SimpleNamespace(default_port=7777),
            ),
        ),
        status_message=SimpleNamespace(emit=messages.append),
        zmq_manager_widget=SimpleNamespace(
            refresh_servers=lambda: refreshes.append(True),
            observe_endpoint_startup=lambda port, observed: observations.append(
                (port, observed)
            ),
        ),
    )
    status = EndpointStartupStatus(
        phase=EndpointStartupPhase.PREPARING_CAPABILITIES,
        message="Discovering functions in the execution process",
    )
    OpenHCSMainWindow._observe_zmq_startup_status(main_window, status)

    assert indicator.state is None
    assert observations == [(7777, status)]
    assert messages == []
    assert refreshes == [True]


def test_zmq_endpoint_snapshot_is_status_bar_presentation_authority() -> None:
    indicator = _StatusIndicatorHarness()
    messages = []
    main_window = SimpleNamespace(
        _zmq_status_indicator=indicator,
        status_message=SimpleNamespace(emit=messages.append),
        runtime_context=SimpleNamespace(
            ui_config=SimpleNamespace(
                zmq=SimpleNamespace(default_port=7777),
            ),
        ),
    )
    connected = EndpointObservationSnapshot.from_responses(
        (
            PongResponse(
                port=7777,
                control_port=8777,
                ready=True,
                server="ExecutionServer",
                server_role=ServerRole.EXECUTION,
            ),
        )
    )

    OpenHCSMainWindow._apply_zmq_endpoint_snapshot(main_window, connected)

    assert indicator.state is StatusState.CONNECTED
    assert indicator.text == "ZMQ: Connected"
    assert indicator.tooltip == "Execution endpoint 7777: Connected"
    assert messages == ["Execution endpoint 7777: Connected"]

    OpenHCSMainWindow._apply_zmq_endpoint_snapshot(
        main_window,
        EndpointObservationSnapshot(),
    )

    assert indicator.state is StatusState.DISCONNECTED
    assert indicator.text == "ZMQ: Not connected"
    assert indicator.tooltip == "Execution endpoint 7777: Not connected"
    assert messages == [
        "Execution endpoint 7777: Connected",
        "Execution endpoint 7777: Not connected",
    ]


def test_zmq_endpoint_termination_descends_to_client_lifecycle_owner() -> None:
    status_signal = _SignalHarness()
    compatibility_signal = _SignalHarness()
    execution_state_signal = _SignalHarness()
    endpoint_signal = _SignalHarness()
    snapshot_signal = _SignalHarness()
    received_statuses = []
    received_snapshots = []
    terminated_ports = []
    main_window = SimpleNamespace(
        plate_manager_widget=SimpleNamespace(
            zmq_connection_status_changed=status_signal,
            zmq_endpoint_compatibility_observed=compatibility_signal,
            manager_execution_state_changed=execution_state_signal,
            zmq_client_service=SimpleNamespace(
                endpoint_terminated=terminated_ports.append,
            ),
        ),
        zmq_manager_widget=SimpleNamespace(
            endpoint_terminated=endpoint_signal,
            endpoint_snapshot_changed=snapshot_signal,
        ),
        _observe_zmq_startup_status=received_statuses.append,
        zmq_version_restart_workflow=SimpleNamespace(
            observe_compatibility=received_statuses.append,
            observe_execution_state=received_statuses.append,
        ),
        _apply_zmq_endpoint_snapshot=received_snapshots.append,
    )

    OpenHCSMainWindow._connect_zmq_lifecycle(main_window)
    endpoint_signal.emit("termination")
    status_signal.emit("connected")
    compatibility_signal.emit("compatible")
    execution_state_signal.emit("idle")
    snapshot_signal.emit("snapshot")

    assert terminated_ports == ["termination"]
    assert received_statuses == [
        "connected",
        "compatible",
        "idle",
    ]
    assert received_snapshots == ["snapshot"]


def test_version_replacement_is_deferred_until_manager_is_idle(qapp) -> None:
    confirmations = []
    messages = []
    compatibility = SimpleNamespace(matches=False)
    execution_state = [ManagerExecutionState.RUNNING]
    main_window = QDialog()
    workflow = ZMQVersionRestartWorkflow(
        main_window=main_window,
        client_service=SimpleNamespace(),
        execution_state=lambda: execution_state[0],
        execute_async=lambda *_args: None,
        publish_status=messages.append,
        presenter=SimpleNamespace(
            confirm_restart=lambda compatibility: (
                confirmations.append(compatibility) or False
            ),
            show_failure=lambda _message: None,
        ),
    )

    workflow.observe_compatibility(compatibility)

    assert confirmations == []
    assert messages == [
        "ZMQ version replacement will be offered after the current operation finishes"
    ]

    execution_state[0] = ManagerExecutionState.IDLE
    workflow.observe_execution_state(ManagerExecutionState.IDLE)

    assert confirmations == [compatibility]


def test_background_initialization_prepares_execution_services() -> None:
    calls = []

    async def prepare_execution_services() -> None:
        return None

    window = SimpleNamespace(
        show_window=lambda window_id: calls.append(("show_window", window_id)),
        show_default_windows=lambda: calls.append(("show_default_windows",)),
        window_services=SimpleNamespace(
            execute_async_operation=lambda operation: calls.append(
                ("execute_async_operation", operation)
            ),
        ),
        _prepare_execution_services=prepare_execution_services,
        _start_ui_bridge_if_enabled=lambda: calls.append(("start_ui_bridge",)),
        _check_for_updates_on_startup=lambda: calls.append(("check_updates",)),
    )

    OpenHCSMainWindow.deferred_initialization(window)
    OpenHCSMainWindow.start_background_services(window)

    assert calls == [
        ("show_window", "log_viewer"),
        ("show_default_windows",),
        ("execute_async_operation", prepare_execution_services),
        ("start_ui_bridge",),
        ("check_updates",),
    ]


def test_execution_service_preparation_starts_endpoint_before_catalog() -> None:
    calls = []
    catalog_future = Future()
    catalog_future.set_result("catalog")

    async def ensure_execution_server() -> bool:
        calls.append("endpoint")
        return True

    window = SimpleNamespace(
        plate_manager_widget=SimpleNamespace(
            ensure_execution_server=ensure_execution_server,
        ),
        function_catalog_projection=SimpleNamespace(
            prepare=lambda: calls.append("catalog") or catalog_future,
        ),
    )

    asyncio.run(OpenHCSMainWindow._prepare_execution_services(window))

    assert calls == ["endpoint", "catalog"]


def test_execution_service_preparation_accepts_owned_teardown_cancellation() -> None:
    catalog_future = Future()
    catalog_future.set_result("catalog")

    async def ensure_execution_server() -> bool:
        raise EndpointConnectionCancelledError("closing")

    window = SimpleNamespace(
        plate_manager_widget=SimpleNamespace(
            ensure_execution_server=ensure_execution_server,
        ),
        function_catalog_projection=SimpleNamespace(prepare=lambda: catalog_future),
    )

    asyncio.run(OpenHCSMainWindow._prepare_execution_services(window))
