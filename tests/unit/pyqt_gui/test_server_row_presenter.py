import pytest

try:
    from PyQt6.QtWidgets import QTreeWidgetItem

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

from pyqt_reactive.services.zmq_server_info import BaseServerInfo
from openhcs.pyqt_gui.widgets.shared.server_browser import ServerRowPresenter
from zmqruntime.messages import PongResponse


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_server_row_presenter_renders_execution_server():
    info = BaseServerInfo.from_response(
        PongResponse.from_dict(
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
                "running_executions": [],
                "queued_executions": [],
            }
        )
    )

    created: list[tuple[str, str, str, BaseServerInfo]] = []
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
    info = BaseServerInfo.from_response(
        PongResponse.from_dict(
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
                "running_executions": [],
                "queued_executions": [],
            }
        )
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
