"""Integration tests for the first-party native Napari ROI Manager."""

from pathlib import Path
import tomllib

import numpy as np
from packaging.requirements import Requirement
import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_existing_openhcs_manifest_owns_roi_manager_widget() -> None:
    manifest = yaml.safe_load(
        (REPOSITORY_ROOT / "openhcs" / "napari.yaml").read_text(encoding="utf-8")
    )

    assert manifest["name"] == "openhcs"
    commands = {
        command["id"]: command for command in manifest["contributions"]["commands"]
    }
    command = commands["openhcs.make_roi_manager_widget"]
    assert command["python_name"] == "openhcs.napari_roi_manager:QRoiManager"
    assert manifest["contributions"]["widgets"] == [
        {
            "command": "openhcs.make_roi_manager_widget",
            "display_name": "OpenHCS ROI Manager",
        }
    ]


@pytest.mark.unit
def test_roi_manager_has_no_mandatory_external_distribution() -> None:
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    requirements = {
        Requirement(requirement).name
        for extra in ("napari", "viz", "all")
        for requirement in project["optional-dependencies"][extra]
    }
    assert "napari-roi-manager" not in requirements
    assert "openhcs-napari-roi-manager" not in requirements
    assert project["license-files"] == ["LICENSE", "THIRD_PARTY_LICENSES/*"]
    notice = REPOSITORY_ROOT / "THIRD_PARTY_LICENSES" / "napari-roi-manager-LICENSE"
    assert "Copyright (c) 2024, Hanjin Liu" in notice.read_text(encoding="utf-8")


@pytest.mark.unit
def test_roi_feature_json_round_trip_preserves_native_values() -> None:
    pytest.importorskip("napari")
    import openhcs.napari_roi_manager as roi_manager
    from openhcs.napari_roi_manager._dataclasses import RoiData

    assert not hasattr(roi_manager, "__version__")

    rois = RoiData(
        data=[np.array(((0.0, 0.0), (1.0, 1.0)))],
        shape_type=["line"],
        names=["axon"],
        features={"area": [np.float32(4.5)], "position": [np.int64(7)]},
    )

    restored = RoiData.from_json_dict(rois.to_json_dict())

    assert restored.names == ["axon"]
    assert restored.features == {"area": [4.5], "position": [7]}


@pytest.mark.unit
def test_result_binding_lazily_reuses_one_roi_manager_for_native_layers() -> None:
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    events: list[object] = []

    class RoiManagerDock:
        def show(self) -> None:
            events.append("manager_show")

    class RoiManagerWidget:
        def connect_layer(self, layer) -> None:
            events.append(("manager_bind", layer))

    manager_dock = RoiManagerDock()
    manager_widget = RoiManagerWidget()

    class Window:
        @staticmethod
        def add_plugin_dock_widget(plugin_name: str, widget_name: str):
            events.append(("manager_open", plugin_name, widget_name))
            return manager_dock, manager_widget

    class ResultSelectionController:
        @staticmethod
        def bind(layer) -> None:
            events.append(("selection_bind", layer))

    server = napari_viewer_server.NapariViewerServer.__new__(
        napari_viewer_server.NapariViewerServer
    )
    server.viewer = type("Viewer", (), {"window": Window()})()
    server.result_selection_controller = ResultSelectionController()
    server.roi_manager_dock = None
    server.roi_manager_widget = None
    first_layer = object()
    second_layer = object()

    server.bind_result_selection_layer(first_layer)
    server.bind_result_selection_layer(second_layer)

    assert events == [
        ("selection_bind", first_layer),
        (
            "manager_open",
            "openhcs",
            "OpenHCS ROI Manager",
        ),
        ("manager_bind", first_layer),
        "manager_show",
        ("selection_bind", second_layer),
        ("manager_bind", second_layer),
        "manager_show",
    ]


@pytest.mark.unit
def test_installed_manager_binds_and_selects_native_shapes_without_copying(
    qtbot,
) -> None:
    napari = pytest.importorskip("napari")
    pytest.importorskip("openhcs.napari_roi_manager")
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)
    layer = viewer.add_shapes(
        [
            np.array([[0, 0], [0, 4], [4, 4]], dtype=float),
            np.array([[8, 8], [8, 12], [12, 12]], dtype=float),
        ],
        shape_type=("polygon", "polygon"),
        features={"name": ("first", "second"), "area": (8.0, 8.0)},
        name="Native ROIs",
    )

    class ResultSelectionController:
        @staticmethod
        def bind(_layer) -> None:
            return None

    server = napari_viewer_server.NapariViewerServer.__new__(
        napari_viewer_server.NapariViewerServer
    )
    server.viewer = viewer
    server.result_selection_controller = ResultSelectionController()
    server.roi_manager_dock = None
    server.roi_manager_widget = None
    original_layers = tuple(viewer.layers)

    server.bind_result_selection_layer(layer)

    manager = server.roi_manager_widget
    assert manager is not None
    assert tuple(viewer.layers) == original_layers
    assert manager._layer is layer
    assert manager._roilist.rowCount() == 2
    assert manager._roilist.get_column("name") == ["first", "second"]

    manager._roilist.selectRow(1)
    assert layer.selected_data == {1}

    second_layer = viewer.add_shapes(
        [np.array([[16, 16], [16, 20], [20, 20]], dtype=float)],
        shape_type="polygon",
        features={"name": ("third",)},
        name="Second ROI set",
    )
    original_manager = manager
    server.bind_result_selection_layer(second_layer)

    assert server.roi_manager_widget is original_manager
    assert manager._layer is second_layer
    assert manager._roilist.get_column("name") == ["third"]
    viewer.close()
