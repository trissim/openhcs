from types import SimpleNamespace

import pytest

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QTreeWidgetItem

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

from pyqt_reactive.services.zmq_server_info import (
    BaseServerInfo,
    ExecutionServerInfo,
)
from pyqt_reactive.services.zmq_server_scan_service import StartingEndpointObservation
from zmqruntime.messages import PongResponse
from zmqruntime.startup import EndpointStartupPhase, EndpointStartupStatus

from openhcs.core.streaming_config_declarations import ViewerType
from openhcs.pyqt_gui.widgets.shared.server_browser import live_tree_sync
from openhcs.pyqt_gui.widgets.shared.server_browser.live_tree_sync import (
    LaunchingViewerServerInfo,
    LiveServerTreeSync,
)
from openhcs.pyqt_gui.widgets.shared.zmq_server_manager import (
    ZMQServerManagerWidget,
)


class _EmptyViewerManager:
    def list_viewers(self):
        return ()


class _FakeTree:
    def __init__(self):
        self.items = []

    def addTopLevelItem(self, item):
        self.items.append(item)

    def topLevelItemCount(self):
        return len(self.items)

    def topLevelItem(self, index):
        return self.items[index]

    def takeTopLevelItem(self, index):
        return self.items.pop(index)


def _execution_server_info(running_executions=()) -> ExecutionServerInfo:
    response = PongResponse.from_dict(
        {
            "type": "pong",
            "port": 7777,
            "control_port": 8777,
            "ready": True,
            "server": "OpenHCSExecutionServer",
            "server_type": "execution",
            "server_role": "execution",
            "log_file_path": "/tmp/server.log",
            "workers": [],
            "running_executions": list(running_executions),
            "queued_executions": [],
        }
    )
    info = BaseServerInfo.from_response(response)
    assert isinstance(info, ExecutionServerInfo)
    return info


def _execution_server_row(info: ExecutionServerInfo):
    item = QTreeWidgetItem(["Port 7777 - Execution Server", "✅ Idle", ""])
    item.setData(0, Qt.ItemDataRole.UserRole, info)
    return item


def _sync(tree):
    return LiveServerTreeSync(
        tree=tree,
        find_item_by_port=lambda port: next(
            (
                item
                for item in tree.items
                if item.data(0, Qt.ItemDataRole.UserRole).port == port
            ),
            None,
        ),
        sync_server_item=lambda _server_info: None,
        sync_startup_endpoint=lambda _observation: None,
    )


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_connected_startup_observation_projects_connected_execution_row() -> None:
    tree = _FakeTree()
    manager = SimpleNamespace(
        server_tree=tree,
        _find_existing_server_item=lambda _port: None,
    )
    observation = StartingEndpointObservation(
        endpoint_port=7777,
        startup_status=EndpointStartupStatus(
            phase=EndpointStartupPhase.CONNECTED,
            message="Connected to server endpoint on port 7777",
        ),
    )

    ZMQServerManagerWidget._sync_startup_endpoint(manager, observation)

    row = tree.topLevelItem(0)
    assert row.text(0) == "Port 7777 - Execution Server"
    assert row.text(1) == "✅ Connected"
    assert row.text(2) == "Connected to server endpoint on port 7777"
    assert row.data(0, Qt.ItemDataRole.UserRole) is observation


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_execution_server_row_is_removed_by_authoritative_missing_scan(
    monkeypatch,
):
    monkeypatch.setattr(
        live_tree_sync.ViewerStateManager,
        "get_instance",
        staticmethod(lambda: _EmptyViewerManager()),
    )
    info = _execution_server_info()
    tree = _FakeTree()
    tree.addTopLevelItem(_execution_server_row(info))
    sync = _sync(tree)

    sync.populate_tree([])

    assert tree.topLevelItemCount() == 0


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_removed_execution_server_row_stays_absent_on_later_scans(
    monkeypatch,
):
    monkeypatch.setattr(
        live_tree_sync.ViewerStateManager,
        "get_instance",
        staticmethod(lambda: _EmptyViewerManager()),
    )
    info = _execution_server_info()
    tree = _FakeTree()
    tree.addTopLevelItem(_execution_server_row(info))
    sync = _sync(tree)

    sync.populate_tree([])
    sync.populate_tree([])

    assert tree.topLevelItemCount() == 0


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_launching_viewer_row_keeps_nominal_viewer_identity(monkeypatch):
    manager = SimpleNamespace(
        list_viewers=lambda: (
            SimpleNamespace(
                port=5555,
                viewer_type=ViewerType.NAPARI.wire_value,
                queued_images=3,
                state=live_tree_sync.ViewerState.LAUNCHING,
            ),
        )
    )
    monkeypatch.setattr(
        live_tree_sync.ViewerStateManager,
        "get_instance",
        staticmethod(lambda: manager),
    )
    tree = _FakeTree()
    sync = _sync(tree)

    sync.populate_tree([])

    payload = tree.topLevelItem(0).data(0, Qt.ItemDataRole.UserRole)
    assert isinstance(payload, LaunchingViewerServerInfo)
    assert payload.viewer_type is ViewerType.NAPARI


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_launching_viewer_row_tracks_current_viewer_snapshot(monkeypatch):
    viewer = SimpleNamespace(
        port=5555,
        viewer_type=ViewerType.NAPARI.wire_value,
        queued_images=3,
        state=live_tree_sync.ViewerState.LAUNCHING,
    )
    viewers = [viewer]
    manager = SimpleNamespace(list_viewers=lambda: tuple(viewers))
    monkeypatch.setattr(
        live_tree_sync.ViewerStateManager,
        "get_instance",
        staticmethod(lambda: manager),
    )
    tree = _FakeTree()
    sync = _sync(tree)

    sync.populate_tree([])
    viewer.queued_images = 7
    sync.populate_tree([])

    payload = tree.topLevelItem(0).data(0, Qt.ItemDataRole.UserRole)
    assert isinstance(payload, LaunchingViewerServerInfo)
    assert payload.queued_images == 7
    assert tree.topLevelItem(0).text(2) == "7 images queued"

    viewers.clear()
    sync.populate_tree([])

    assert tree.topLevelItemCount() == 0
