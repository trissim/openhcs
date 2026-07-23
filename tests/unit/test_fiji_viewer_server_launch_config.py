from __future__ import annotations

from openhcs.runtime.fiji_viewer_server import (
    FijiViewerServer,
    FijiViewerServerLaunchConfig,
)
from zmqruntime.streaming import StreamingVisualizerServer


def test_fiji_server_retains_its_nominal_launch_config(monkeypatch) -> None:
    launch_config = FijiViewerServerLaunchConfig(
        port=20_321,
        fiji_viewer_title="Test Fiji",
        fiji_display_config=None,
        display_enabled=False,
    )
    monkeypatch.setattr(
        StreamingVisualizerServer,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    server = FijiViewerServer(launch_config)

    assert server.launch_config is launch_config
