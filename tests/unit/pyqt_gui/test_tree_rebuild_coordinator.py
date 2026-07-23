import pytest

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

from pyqt_reactive.widgets.shared import TreeRebuildCoordinator


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_tree_rebuild_coordinator_preserves_selection_and_expansion():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    tree = QTreeWidget()
    tree.setHeaderLabels(["Server", "Status", "Info"])

    old_server = QTreeWidgetItem(["Port 7777", "ok", ""])
    old_server.setData(0, Qt.ItemDataRole.UserRole, {"port": 7777})
    tree.addTopLevelItem(old_server)

    old_child = QTreeWidgetItem(["Worker", "ok", ""])
    old_child.setData(
        0,
        Qt.ItemDataRole.UserRole,
        {"type": "worker", "node_id": "worker_0"},
    )
    old_server.addChild(old_child)
    old_server.setExpanded(True)
    old_child.setSelected(True)

    coordinator = TreeRebuildCoordinator.default()

    def rebuild() -> None:
        new_server = QTreeWidgetItem(["Port 7777", "ok", ""])
        new_server.setData(0, Qt.ItemDataRole.UserRole, {"port": 7777})
        tree.addTopLevelItem(new_server)
        new_child = QTreeWidgetItem(["Worker", "ok", ""])
        new_child.setData(
            0,
            Qt.ItemDataRole.UserRole,
            {"type": "worker", "node_id": "worker_0"},
        )
        new_server.addChild(new_child)

    coordinator.rebuild(tree, rebuild)

    rebuilt_server = tree.topLevelItem(0)
    rebuilt_child = rebuilt_server.child(0)
    assert rebuilt_server.isExpanded()
    assert rebuilt_child.isSelected()
