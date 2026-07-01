import pytest

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QTreeWidgetItem

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

from pyqt_reactive.services.zmq_server_info_parser import DefaultServerInfoParser

from openhcs.pyqt_gui.widgets.shared.server_browser import live_tree_sync
from openhcs.pyqt_gui.widgets.shared.server_browser.live_tree_sync import (
    LiveServerTreeSync,
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


def _execution_server_payload(running_executions=()):
    return {
        "port": 7777,
        "ready": True,
        "server": "OpenHCSExecutionServer",
        "log_file_path": "/tmp/server.log",
        "workers": [],
        "running_executions": list(running_executions),
        "queued_executions": [],
    }


def _execution_server_row(payload):
    item = QTreeWidgetItem(["Port 7777 - Execution Server", "✅ Idle", ""])
    item.setData(0, Qt.ItemDataRole.UserRole, payload)
    return item


def _sync(tree, parser, progress_execution_ids, last_known_servers):
    return LiveServerTreeSync(
        tree=tree,
        find_item_by_port=lambda port: next(
            (
                item
                for item in tree.items
                if item.data(0, Qt.ItemDataRole.UserRole).get("port") == port
            ),
            None,
        ),
        sync_server_item=lambda _server_info: None,
        progress_execution_ids=progress_execution_ids,
        parse_server_info=parser.parse,
        last_known_servers=last_known_servers,
        missing_port_counts={},
    )


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_execution_server_row_survives_transient_missing_scan_with_progress(monkeypatch):
    monkeypatch.setattr(
        live_tree_sync.ViewerStateManager,
        "get_instance",
        staticmethod(lambda: _EmptyViewerManager()),
    )
    parser = DefaultServerInfoParser()
    payload = _execution_server_payload()
    tree = _FakeTree()
    tree.addTopLevelItem(_execution_server_row(payload))
    sync = _sync(
        tree,
        parser,
        progress_execution_ids=lambda: {"compile-exec"},
        last_known_servers={7777: payload},
    )

    sync.populate_tree([])
    sync.populate_tree([])

    assert tree.topLevelItemCount() == 1
    assert parser.parse(tree.topLevelItem(0).data(0, Qt.ItemDataRole.UserRole)).port == 7777


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_execution_server_row_is_removed_after_repeated_missing_scans_without_progress(
    monkeypatch,
):
    monkeypatch.setattr(
        live_tree_sync.ViewerStateManager,
        "get_instance",
        staticmethod(lambda: _EmptyViewerManager()),
    )
    parser = DefaultServerInfoParser()
    payload = _execution_server_payload()
    tree = _FakeTree()
    tree.addTopLevelItem(_execution_server_row(payload))
    sync = _sync(
        tree,
        parser,
        progress_execution_ids=set,
        last_known_servers={7777: payload},
    )

    sync.populate_tree([])
    sync.populate_tree([])

    assert tree.topLevelItemCount() == 0
