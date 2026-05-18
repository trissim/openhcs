from __future__ import annotations

import pytest

from polystore.streaming_constants import StreamingDataType

from openhcs.runtime.napari_streaming_handlers import (
    NapariLayerUpdateAuthority,
    NapariStreamingDataTypeHandler,
    build_napari_streaming_data_type_handlers,
    napari_streaming_data_type_handler,
)


def _handler_marker(name: str):
    def marker(*args, **kwargs):
        return name, args, kwargs

    return marker


def test_build_napari_streaming_data_type_handlers_declares_full_axis() -> None:
    build_image = _handler_marker("build_image")
    create_image = _handler_marker("create_image")
    build_shapes = _handler_marker("build_shapes")
    create_shapes = _handler_marker("create_shapes")
    build_points = _handler_marker("build_points")
    create_points = _handler_marker("create_points")

    handlers = build_napari_streaming_data_type_handlers(
        build_image_data=build_image,
        create_image_layer=create_image,
        build_shapes_data=build_shapes,
        create_shapes_layer=create_shapes,
        build_points_data=build_points,
        create_points_layer=create_points,
    )

    assert set(handlers) == {
        StreamingDataType.IMAGE,
        StreamingDataType.SHAPES,
        StreamingDataType.POINTS,
    }
    assert handlers[StreamingDataType.IMAGE] == NapariStreamingDataTypeHandler(
        data_type=StreamingDataType.IMAGE,
        build_nd_data=build_image,
        create_layer=create_image,
    )
    assert handlers[StreamingDataType.SHAPES].build_nd_data is build_shapes
    assert handlers[StreamingDataType.POINTS].create_layer is create_points


def test_napari_streaming_data_type_handler_accepts_string_data_type() -> None:
    handlers = build_napari_streaming_data_type_handlers(
        build_image_data=_handler_marker("build_image"),
        create_image_layer=_handler_marker("create_image"),
        build_shapes_data=_handler_marker("build_shapes"),
        create_shapes_layer=_handler_marker("create_shapes"),
        build_points_data=_handler_marker("build_points"),
        create_points_layer=_handler_marker("create_points"),
    )

    assert (
        napari_streaming_data_type_handler(handlers, "image").data_type
        is StreamingDataType.IMAGE
    )


def test_napari_streaming_data_type_handler_fails_loudly_for_missing_axis() -> None:
    with pytest.raises(ValueError, match="No Napari streaming handler registered"):
        napari_streaming_data_type_handler({}, StreamingDataType.IMAGE)


class _FakeLayerList(list):
    def remove(self, layer):
        super().remove(layer)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.dims = type("Dims", (), {"axis_labels": None})()
        self.calls = []

    def add_image(self, data, *, name, **kwargs):
        return self._add_layer("image", data, name, kwargs)

    def add_shapes(self, data, *, name, **kwargs):
        return self._add_layer("shapes", data, name, kwargs)

    def add_points(self, data, *, name, **kwargs):
        return self._add_layer("points", data, name, kwargs)

    def _add_layer(self, layer_type, data, name, kwargs):
        layer = type("Layer", (), {"name": name, "data": data, "kwargs": kwargs})()
        self.layers.append(layer)
        self.calls.append((layer_type, data, name, kwargs))
        return layer


def test_napari_layer_update_authority_replaces_existing_image_and_axis_labels():
    viewer = _FakeViewer()
    layers = {}
    authority = NapariLayerUpdateAuthority()

    first = authority.create_or_update_image(
        viewer,
        layers,
        "nuclei",
        "image-1",
        colormap=None,
        axis_labels=("z", "y", "x"),
    )
    second = authority.create_or_update_image(
        viewer,
        layers,
        "nuclei",
        "image-2",
        colormap="green",
        axis_labels=("c", "y", "x"),
    )

    assert first not in viewer.layers
    assert second in viewer.layers
    assert layers == {"nuclei": second}
    assert viewer.dims.axis_labels == ("c", "y", "x")
    assert viewer.calls[0] == ("image", "image-1", "nuclei", {"colormap": "gray"})
    assert viewer.calls[1] == ("image", "image-2", "nuclei", {"colormap": "green"})


def test_napari_layer_update_authority_declares_shapes_and_points_kwargs():
    viewer = _FakeViewer()
    layers = {}
    authority = NapariLayerUpdateAuthority()

    shapes = authority.create_or_update_shapes(
        viewer,
        layers,
        "rois",
        shapes_data=[[[0, 0], [1, 1]]],
        shape_types=["polygon"],
        properties={"id": [1]},
    )
    points = authority.create_or_update_points(
        viewer,
        layers,
        "spots",
        points_data=[[0, 0]],
        properties={"id": [2]},
    )

    assert layers == {"rois": shapes, "spots": points}
    assert viewer.calls[0][3] == {
        "shape_type": ["polygon"],
        "properties": {"id": [1]},
        "edge_color": "red",
        "face_color": "transparent",
        "edge_width": 2,
    }
    assert viewer.calls[1][3] == {
        "properties": {"id": [2]},
        "face_color": "green",
        "size": 3,
    }
