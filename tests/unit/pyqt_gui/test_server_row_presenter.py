import pytest

try:
    from PyQt6.QtWidgets import QTreeWidgetItem

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

from pyqt_reactive.services.zmq_server_info_parser import DefaultServerInfoParser
from openhcs.pyqt_gui.widgets.shared.server_browser import ServerRowPresenter


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_server_row_presenter_renders_execution_server():
    parser = DefaultServerInfoParser()
    info = parser.parse(
        {
            "port": 7777,
            "ready": True,
            "server": "OpenHCSExecutionServer",
            "log_file_path": "/tmp/server.log",
            "workers": [],
            "running_executions": [],
            "queued_executions": [],
        }
    )

    created: list[tuple[str, str, str, dict]] = []
    presenter = ServerRowPresenter(
        create_tree_item=lambda display, status, extra, data: (
            created.append((display, status, extra, data)),
            QTreeWidgetItem([display, status, extra]),
        )[1],
        update_execution_server_item=lambda _item, _data: None,
        log_warning=lambda *_args, **_kwargs: None,
    )

    item = presenter.render_server(info, "✅")

    assert item.text(0) == "Port 7777 - Execution Server"
    assert item.text(1) == "✅ Idle"
    assert len(created) == 1


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_server_row_presenter_populates_execution_children():
    parser = DefaultServerInfoParser()
    info = parser.parse(
        {
            "port": 7777,
            "ready": True,
            "server": "OpenHCSExecutionServer",
            "log_file_path": "/tmp/server.log",
            "workers": [],
            "running_executions": [],
            "queued_executions": [],
        }
    )

    called = {"value": 0}
    presenter = ServerRowPresenter(
        create_tree_item=lambda display, status, extra, data: QTreeWidgetItem(
            [display, status, extra]
        ),
        update_execution_server_item=lambda item, data: (
            called.__setitem__("value", called["value"] + 1),
            item.addChild(QTreeWidgetItem(["child", "", ""])),
        ),
        log_warning=lambda *_args, **_kwargs: None,
    )

    item = QTreeWidgetItem(["server", "", ""])
    has_children = presenter.populate_server_children(info, item)

    assert called["value"] == 1
    assert has_children
