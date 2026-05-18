"""Shared Napari streaming handler axis declarations."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from polystore.streaming_constants import StreamingDataType

from openhcs.runtime.viewer_protocol import NapariLayerKind


logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class NapariLayerUpdateRequest:
    """Typed request for creating or replacing one Napari layer."""

    viewer: object
    layers: dict[str, object]
    layer_name: str
    layer_kind: NapariLayerKind
    data: object
    layer_kwargs: Mapping[str, object]


class NapariLayerUpdateAuthority:
    """Owns create-or-replace mechanics for Napari streaming layers."""

    def create_or_update(self, request: NapariLayerUpdateRequest) -> object:
        existing_layer = self._existing_layer(request.viewer, request.layer_name)
        if existing_layer is not None:
            request.viewer.layers.remove(existing_layer)
            request.layers.pop(request.layer_name, None)
            logger.info(
                "🔬 NAPARI PROCESS: Removed existing %s layer %s for recreation",
                request.layer_kind.value,
                request.layer_name,
            )

        add_method = getattr(request.viewer, f"add_{request.layer_kind.value}")
        new_layer = add_method(
            request.data,
            name=request.layer_name,
            **dict(request.layer_kwargs),
        )
        request.layers[request.layer_name] = new_layer
        self._log_created_layer(request)
        return new_layer

    def create_or_update_image(
        self,
        viewer: object,
        layers: dict[str, object],
        layer_name: str,
        image_data: object,
        colormap: object,
        axis_labels: object | None = None,
    ) -> object:
        layer = self.create_or_update(
            NapariLayerUpdateRequest(
                viewer=viewer,
                layers=layers,
                layer_name=layer_name,
                layer_kind=NapariLayerKind.IMAGE,
                data=image_data,
                layer_kwargs={"colormap": colormap or "gray"},
            )
        )
        if axis_labels is not None:
            viewer.dims.axis_labels = axis_labels
            logger.info("🔬 NAPARI PROCESS: Set viewer.dims.axis_labels=%s", axis_labels)
        return layer

    def create_or_update_shapes(
        self,
        viewer: object,
        layers: dict[str, object],
        layer_name: str,
        shapes_data: object,
        shape_types: object,
        properties: object,
    ) -> object:
        return self.create_or_update(
            NapariLayerUpdateRequest(
                viewer=viewer,
                layers=layers,
                layer_name=layer_name,
                layer_kind=NapariLayerKind.SHAPES,
                data=shapes_data,
                layer_kwargs={
                    "shape_type": shape_types,
                    "properties": properties,
                    "edge_color": "red",
                    "face_color": "transparent",
                    "edge_width": 2,
                },
            )
        )

    def create_or_update_points(
        self,
        viewer: object,
        layers: dict[str, object],
        layer_name: str,
        points_data: object,
        properties: object,
    ) -> object:
        return self.create_or_update(
            NapariLayerUpdateRequest(
                viewer=viewer,
                layers=layers,
                layer_name=layer_name,
                layer_kind=NapariLayerKind.POINTS,
                data=points_data,
                layer_kwargs={
                    "properties": properties,
                    "face_color": "green",
                    "size": 3,
                },
            )
        )

    def _existing_layer(self, viewer: object, layer_name: str) -> object | None:
        for layer in viewer.layers:
            if layer.name == layer_name:
                return layer
        return None

    def _log_created_layer(self, request: NapariLayerUpdateRequest) -> None:
        if request.layer_kind in {NapariLayerKind.SHAPES, NapariLayerKind.POINTS}:
            count = len(request.data) if hasattr(request.data, "__len__") else 0
            logger.info(
                "🔬 NAPARI PROCESS: Created %s layer %s with %d %s",
                request.layer_kind.value,
                request.layer_name,
                count,
                request.layer_kind.value,
            )
            return
        logger.info(
            "🔬 NAPARI PROCESS: Created %s layer %s",
            request.layer_kind.value,
            request.layer_name,
        )
