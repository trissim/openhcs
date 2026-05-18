from __future__ import annotations

import pytest

from polystore.streaming_constants import StreamingDataType

from openhcs.runtime.napari_streaming_handlers import (
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

