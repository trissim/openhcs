from pyqt_reactive.services.zmq_server_info_parser import DefaultServerInfoParser

from openhcs.pyqt_gui.widgets.shared.server_browser import ServerTreePopulation


class _FakeTree:
    def __init__(self):
        self.items = []

    def addTopLevelItem(self, item):
        self.items.append(item)


def _execution_server_info():
    parser = DefaultServerInfoParser()
    return parser.parse(
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


def test_server_tree_population_adds_scanned_servers():
    rendered = []
    populated = []
    tree = _FakeTree()

    population = ServerTreePopulation(
        create_tree_item=lambda display, status, info, data: {
            "display": display,
            "status": status,
            "info": info,
            "data": data,
            "children": [],
        },
        render_server=lambda info, status_icon: (
            rendered.append((info.port, status_icon)),
            {"port": info.port},
        )[1],
        populate_server_children=lambda info, item: (
            populated.append((info.port, item.get("port"))),
            False,
        )[1],
        log_info=lambda *_args, **_kwargs: None,
    )
    population._get_launching_viewers = lambda: {}

    population.populate_tree(tree=tree, parsed_servers=[_execution_server_info()])

    assert len(tree.items) == 1
    assert rendered == [(7777, "✅")]
    assert populated == [(7777, 7777)]
