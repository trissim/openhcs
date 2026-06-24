import pytest

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

from pyqt_reactive.widgets.shared import TreeStateAdapter


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_tree_state_adapter_restores_expansion_and_selection_by_key():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    tree = QTreeWidget()
    tree.setHeaderLabels(["Server", "Status", "Info"])

    server = QTreeWidgetItem(["Port 7777 - Execution Server", "✅", ""])
    server.setData(0, Qt.ItemDataRole.UserRole, {"port": 7777})
    tree.addTopLevelItem(server)

    worker = QTreeWidgetItem(["Worker worker_0", "⚙️", ""])
    worker.setData(
        0,
        Qt.ItemDataRole.UserRole,
        {"type": "worker", "node_id": "worker_0"},
    )
    server.addChild(worker)

    server.setExpanded(True)
    worker.setSelected(True)

    adapter = TreeStateAdapter.default()
    expansion_state = adapter.capture_expansion_state(tree)
    selected_keys = adapter.capture_selected_keys(tree)

    tree.clear()
    rebuilt_server = QTreeWidgetItem(["Port 7777 - Execution Server", "✅", ""])
    rebuilt_server.setData(0, Qt.ItemDataRole.UserRole, {"port": 7777})
    tree.addTopLevelItem(rebuilt_server)

    rebuilt_worker = QTreeWidgetItem(["Worker worker_0", "⚙️", ""])
    rebuilt_worker.setData(
        0,
        Qt.ItemDataRole.UserRole,
        {"type": "worker", "node_id": "worker_0"},
    )
    rebuilt_server.addChild(rebuilt_worker)

    adapter.restore_expansion_state(tree, expansion_state)
    adapter.restore_selected_keys(tree, selected_keys)

    assert rebuilt_server.isExpanded()
    assert rebuilt_worker.isSelected()
