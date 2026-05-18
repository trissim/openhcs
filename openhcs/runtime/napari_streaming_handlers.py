"""Shared Napari streaming handler axis declarations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from polystore.streaming_constants import StreamingDataType


@dataclass(frozen=True, slots=True)
class NapariStreamingDataTypeHandler:
    """Layer-building behavior for one streaming data type."""

    data_type: StreamingDataType
    build_nd_data: Callable[..., object]
    create_layer: Callable[..., object]


NapariStreamingDataTypeHandlers = Mapping[
    StreamingDataType,
    NapariStreamingDataTypeHandler,
]


def build_napari_streaming_data_type_handlers(
    *,
    build_image_data: Callable[..., object],
    create_image_layer: Callable[..., object],
    build_shapes_data: Callable[..., object],
    create_shapes_layer: Callable[..., object],
    build_points_data: Callable[..., object],
    create_points_layer: Callable[..., object],
) -> dict[StreamingDataType, NapariStreamingDataTypeHandler]:
    """Build the canonical StreamingDataType handler table for Napari viewers."""
    return {
        StreamingDataType.IMAGE: NapariStreamingDataTypeHandler(
            data_type=StreamingDataType.IMAGE,
            build_nd_data=build_image_data,
            create_layer=create_image_layer,
        ),
        StreamingDataType.SHAPES: NapariStreamingDataTypeHandler(
            data_type=StreamingDataType.SHAPES,
            build_nd_data=build_shapes_data,
            create_layer=create_shapes_layer,
        ),
        StreamingDataType.POINTS: NapariStreamingDataTypeHandler(
            data_type=StreamingDataType.POINTS,
            build_nd_data=build_points_data,
            create_layer=create_points_layer,
        ),
    }


def napari_streaming_data_type_handler(
    handlers: NapariStreamingDataTypeHandlers,
    data_type: StreamingDataType | str,
) -> NapariStreamingDataTypeHandler:
    """Return the handler for one streaming data type, failing loudly if absent."""
    resolved_data_type = (
        StreamingDataType(data_type) if isinstance(data_type, str) else data_type
    )
    try:
        return handlers[resolved_data_type]
    except KeyError as error:
        raise ValueError(
            f"No Napari streaming handler registered for {resolved_data_type!r}."
        ) from error

