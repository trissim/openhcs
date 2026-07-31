from types import SimpleNamespace

from zmqruntime.viewer_state import ViewerInstance, ViewerState

from openhcs.pyqt_gui.services.ui_bridge_live_overview import (
    ViewerSessionLiveOverviewContributor,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId


class _ViewerStateManagerStub:
    def __init__(self, viewers=()) -> None:
        self.viewers = list(viewers)

    def list_viewers(self):
        return list(self.viewers)


def _viewer(
    viewer_type: str,
    port: int,
    *,
    state: ViewerState,
    running: bool,
    queued_images: int = 0,
    processed_images: int = 0,
    error_message: str | None = None,
) -> ViewerInstance:
    return ViewerInstance(
        viewer_type=viewer_type,
        port=port,
        visualizer=SimpleNamespace(is_running=running),
        state=state,
        queued_images=queued_images,
        processed_images=processed_images,
        error_message=error_message,
    )


def test_viewer_session_overview_projects_current_manager_snapshots() -> None:
    manager = _ViewerStateManagerStub(
        (
            _viewer(
                "napari",
                5591,
                state=ViewerState.READY,
                running=True,
                queued_images=2,
                processed_images=24,
            ),
            _viewer(
                "fiji",
                5556,
                state=ViewerState.ERROR,
                running=False,
                error_message="viewer exited",
            ),
        )
    )
    contributor = ViewerSessionLiveOverviewContributor(manager)

    section = contributor.overview_sections()[0]

    assert section.section_id == contributor.overview_identity.section_id
    assert section.summary == "2 viewers"
    assert tuple((metric.key, metric.value) for metric in section.metrics) == (
        ("viewers", "2"),
        ("healthy", "1"),
        ("queued_images", "2"),
    )
    assert tuple(item.label for item in section.items) == (
        "fiji viewer",
        "napari viewer",
    )
    assert section.items[0].status == "error"
    assert section.items[0].severity == "error"
    assert "port=5556" in section.items[0].detail
    assert "error=viewer exited" in section.items[0].detail
    assert section.items[1].status == "ready"
    assert section.items[1].severity == "info"
    assert section.items[1].source_window_id == OpenHCSUiWindowId.zmq_server_manager

    manager.viewers.clear()
    refreshed = contributor.overview_sections()[0]

    assert refreshed.summary == "0 viewers"
    assert refreshed.items == ()
