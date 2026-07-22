"""Native viewer result-element navigation and projection contracts."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.viewer import ViewerWindowNavigationRequest
from openhcs.agent.services.viewer_window_service import (
    ViewerWindowGatewayABC,
    ViewerWindowService,
)
from openhcs.runtime.napari_streaming_handlers import (
    NapariComponentGroupStore,
    NapariLayerRouteStateStore,
)
from openhcs.runtime.napari_viewer_server import (
    NapariNavigationControlMessageAction,
)
from openhcs.runtime.viewer_controls import ViewerNavigationControlOptions
from openhcs.runtime.viewer_protocol import ViewerControlResponseField


class _NavigationResponseGateway(ViewerWindowGatewayABC):
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def snapshot_window(self, request):
        raise AssertionError(request)

    def window_state(self, request):
        raise AssertionError(request)

    def window_payloads(self, request):
        raise AssertionError(request)

    def navigate_window(self, request):
        return self.response


class _DimensionLabelOverlay:
    def __init__(self) -> None:
        self.routes: list[str] = []

    def setup_for_layer(self, route_key: str) -> None:
        self.routes.append(route_key)


def _viewer_server(viewer, layer, route_key: str = "result-rois"):
    route_state = NapariLayerRouteStateStore.empty()
    route_state.set_title(route_key, "Result ROIs")
    route_state.set_layer(route_key, layer)
    overlay = _DimensionLabelOverlay()
    server = SimpleNamespace(
        viewer=viewer,
        layer_route_state=route_state,
        component_groups=NapariComponentGroupStore(),
        display_pipeline=SimpleNamespace(dimension_label_overlay=overlay),
        napari_window_title="OpenHCS Napari Viewer",
    )
    return server, overlay


@pytest.mark.parametrize("invalid_index", (True, "1", -1))
def test_viewer_navigation_rejects_invalid_data_index(invalid_index: object) -> None:
    error_type = TypeError if invalid_index in (True, "1") else ValueError

    with pytest.raises(error_type, match="data_index"):
        ViewerNavigationControlOptions(
            route_key="result-rois",
            data_index=invalid_index,  # type: ignore[arg-type]
        )


def test_napari_navigation_selects_native_feature_row_and_projects_evidence(qtbot):
    from napari.components import ViewerModel
    from napari_builtins._qt.features_table import FeaturesTable

    viewer = ViewerModel()
    layer = viewer.add_shapes(
        [
            np.array([[0, 0], [0, 2], [2, 2]], dtype=float),
            np.array([[4, 4], [4, 6], [6, 6]], dtype=float),
        ],
        shape_type=["polygon", "polygon"],
        features={"label": [11, 12], "area": [3.0, 4.0]},
        name="Result ROIs",
    )
    feature_table = FeaturesTable(viewer)
    qtbot.addWidget(feature_table)
    layer.visible = False
    viewer.layers.selection.active = None
    server, overlay = _viewer_server(viewer, layer)
    request = ViewerNavigationControlOptions.from_overrides(
        route_key="result-rois",
        visible=True,
        selected=True,
        data_index=1,
    )

    response = NapariNavigationControlMessageAction().handle(
        server,
        {ViewerControlResponseField.PAYLOAD.value: request},
    )
    qtbot.wait(0)

    assert response["status"] == "success"
    assert layer.visible is True
    assert viewer.layers.selection.active is layer
    assert layer.selected_data == {1}
    assert feature_table.table.model().rowCount() == 2
    assert [
        index.row() for index in feature_table.table.selectionModel().selectedRows()
    ] == [1]
    assert overlay.routes == ["result-rois"]
    projected_layer = response["layers"][0]
    assert projected_layer["visible"] is True
    assert projected_layer["selected"] is True
    assert projected_layer["feature_row_count"] == 2
    assert projected_layer["selected_data_indices"] == (1,)

    result = ViewerWindowService(
        gateway=_NavigationResponseGateway(response)
    ).navigate_window(
        ViewerWindowNavigationRequest(
            connection=ExecutionConnectionSpec(port=5900),
            navigation=request,
        )
    )
    assert result.observed is True
    assert result.route_key == "result-rois"
    assert result.visible is True
    assert result.selected is True
    assert result.data_index == 1
    assert result.feature_row_count == 2
    assert result.selected_data_indices == (1,)


def test_napari_navigation_rejects_out_of_range_data_index(qtbot) -> None:
    from napari.components import ViewerModel

    viewer = ViewerModel()
    layer = viewer.add_points(
        np.array([[1, 2]], dtype=float),
        features={"score": [0.75]},
        name="Result Points",
    )
    server, _overlay = _viewer_server(viewer, layer, "result-points")

    response = NapariNavigationControlMessageAction().handle(
        server,
        {
            ViewerControlResponseField.PAYLOAD.value: ViewerNavigationControlOptions.from_overrides(
                route_key="result-points",
                data_index=1,
            )
        },
    )

    assert response["status"] == "error"
    assert "outside 1 populated feature row" in response["message"]


def test_napari_navigation_rejects_layer_without_native_data_selection(qtbot) -> None:
    from napari.components import ViewerModel

    viewer = ViewerModel()
    layer = viewer.add_image(np.zeros((4, 4), dtype=np.uint8), name="Image")
    server, _overlay = _viewer_server(viewer, layer, "result-image")

    response = NapariNavigationControlMessageAction().handle(
        server,
        {
            ViewerControlResponseField.PAYLOAD.value: ViewerNavigationControlOptions.from_overrides(
                route_key="result-image",
                data_index=0,
            )
        },
    )

    assert response["status"] == "error"
    assert "native feature-bearing data selection" in response["message"]


def test_navigate_viewer_cli_projects_data_index() -> None:
    import openhcs.mcp.dev_client as dev_client

    args = dev_client._build_parser().parse_args(
        ("navigate-viewer", "5900", "result-rois", "--data-index", "3")
    )

    call = dev_client._calls_from_args(args)[0]
    assert call.name == "openhcs_navigate_viewer_window"
    assert call.arguments["data_index"] == 3


def test_mcp_navigation_schema_explains_linked_result_selection() -> None:
    if importlib.util.find_spec("mcp") is None:
        return

    from openhcs.mcp import server

    built = server.build_server()
    listed_tools = built.list_tools()
    tools = (
        asyncio.run(listed_tools) if inspect.isawaitable(listed_tools) else listed_tools
    )
    navigation = next(
        tool for tool in tools if tool.name == "openhcs_navigate_viewer_window"
    )

    assert "data_index" in navigation.inputSchema["properties"]
    assert "native feature-bearing result layer" in navigation.description
    assert "selected_data_indices" in navigation.description
