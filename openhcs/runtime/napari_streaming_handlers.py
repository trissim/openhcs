"""Shared Napari streaming handler axis declarations."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

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


@dataclass(frozen=True, slots=True)
class NapariLayerLogPolicy:
    """Creation-log policy for one Napari layer kind."""

    layer_kind: NapariLayerKind
    count_data: bool

    def log_created(self, request: NapariLayerUpdateRequest) -> None:
        if self.count_data:
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


def napari_layer_log_policies() -> dict[NapariLayerKind, NapariLayerLogPolicy]:
    """Build exhaustive creation-log policies for Napari layer kinds."""
    policies = {
        NapariLayerKind.IMAGE: NapariLayerLogPolicy(NapariLayerKind.IMAGE, False),
        NapariLayerKind.SHAPES: NapariLayerLogPolicy(NapariLayerKind.SHAPES, True),
        NapariLayerKind.POINTS: NapariLayerLogPolicy(NapariLayerKind.POINTS, True),
    }
    if set(policies) != set(NapariLayerKind):
        missing = set(NapariLayerKind) - set(policies)
        raise ValueError(f"Missing Napari layer log policies for {missing!r}.")
    return policies


class NapariLayerUpdateAuthority:
    """Owns create-or-replace mechanics for Napari streaming layers."""

    def __init__(self) -> None:
        self._log_policies = napari_layer_log_policies()

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
        self._log_policies[request.layer_kind].log_created(request)


@dataclass(slots=True)
class NapariLayerStateStore:
    """Own per-layer Napari runtime state that must stay keyed together."""

    layers: dict[str, object]
    dimension_labels: dict[str, object]
    pending_updates: dict[str, object]

    @classmethod
    def empty(cls) -> "NapariLayerStateStore":
        return cls(layers={}, dimension_labels={}, pending_updates={})

    def labels_for(self, layer_key: str) -> object:
        return self.dimension_labels.get(layer_key, {})

    def set_labels(self, layer_key: str, labels: object) -> None:
        self.dimension_labels[layer_key] = labels

    def cancel_pending_update(self, layer_key: str) -> bool:
        timer = self.pending_updates.get(layer_key)
        if timer is None:
            return False
        timer.stop()
        return True

    def set_pending_update(self, layer_key: str, timer: object) -> None:
        self.pending_updates[layer_key] = timer

    def pop_pending_update(self, layer_key: str) -> object | None:
        return self.pending_updates.pop(layer_key, None)

    def has_layer(self, layer_key: str) -> bool:
        return layer_key in self.layers

    def layer(self, layer_key: str) -> object:
        return self.layers[layer_key]

    def set_layer(self, layer_key: str, layer: object) -> None:
        self.layers[layer_key] = layer


@dataclass(slots=True)
class NapariBatchProcessorStore:
    """Own lazy NapariBatchProcessor instances by layer key."""

    debounce_delay_ms: int
    max_debounce_wait_ms: int = 5000
    processors: dict[str, object] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_create(
        self,
        *,
        layer_key: str,
        napari_server: object,
        batch_size: int | None = None,
    ) -> object:
        with self.lock:
            if layer_key not in self.processors:
                from polystore.streaming.receivers.napari import NapariBatchProcessor

                self.processors[layer_key] = NapariBatchProcessor(
                    napari_server=napari_server,
                    batch_size=batch_size,
                    debounce_delay_ms=self.debounce_delay_ms,
                    max_debounce_wait_ms=self.max_debounce_wait_ms,
                )
                logger.info(
                    "NapariViewerServer: Created batch processor for layer '%s' with batch_size=%s",
                    layer_key,
                    batch_size,
                )
            return self.processors[layer_key]


class NapariShapeKind(Enum):
    """Shape kinds accepted by the Napari ROI label rasterizer."""

    POLYGON = "polygon"
    PATH = "path"
    POINTS = "points"


@dataclass(frozen=True, slots=True)
class NapariShapePaintContext:
    """Mutable label-array position passed to shape painters."""

    labels_array: np.ndarray
    indices: tuple[int, ...]
    label_id: int


class NapariShapeLabelRasterizer:
    """Convert streamed Napari shape payloads into dense label arrays."""

    default_image_shape: tuple[int, int] = (512, 512)

    def __init__(self) -> None:
        self._paint_routes: Mapping[
            NapariShapeKind,
            Callable[[Mapping[str, object], NapariShapePaintContext], int],
        ] = {
            NapariShapeKind.POLYGON: self._paint_polygon,
            NapariShapeKind.PATH: self._paint_path,
            NapariShapeKind.POINTS: self._skip_points,
        }

    def rasterize(
        self,
        *,
        layer_items: list[Mapping[str, object]],
        stack_components: list[str],
        component_values: Mapping[str, list[object]],
    ) -> np.ndarray:
        logger.info("🔬 NAPARI PROCESS: Building ROI stack with global component values")
        for component, values in component_values.items():
            logger.info("🔬 NAPARI PROCESS:   %s: %s", component, values)

        first_shapes = layer_items[0]["data"]
        if not first_shapes:
            logger.warning(
                "🔬 NAPARI PROCESS: No shapes data, creating default 512x512 array"
            )
            return np.zeros((1, 1, *self.default_image_shape), dtype=np.uint16)

        image_shape = self._image_shape(first_shapes)
        nd_shape = [
            *(len(component_values[component]) for component in stack_components),
            *image_shape,
        ]
        labels_array = np.zeros(nd_shape, dtype=np.uint16)

        label_id = 1
        for item in layer_items:
            indices = self._component_indices(item, stack_components, component_values)
            for shape_dict in item["data"]:
                shape_kind = NapariShapeKind(str(shape_dict["type"]))
                label_id = self._paint_routes[shape_kind](
                    shape_dict,
                    NapariShapePaintContext(labels_array, indices, label_id),
                )

        logger.info(
            "🔬 NAPARI PROCESS: Created labels array with shape %s and %d labels",
            labels_array.shape,
            label_id - 1,
        )
        return labels_array

    def _image_shape(self, shapes: list[Mapping[str, object]]) -> tuple[int, int]:
        max_y, max_x = 0, 0
        for shape_dict in shapes:
            shape_kind = NapariShapeKind(str(shape_dict["type"]))
            bounds = self._coordinate_bounds(shape_dict, shape_kind)
            max_y = max(max_y, bounds[0])
            max_x = max(max_x, bounds[1])

        if max_y == 0 or max_x == 0:
            logger.warning(
                "🔬 NAPARI PROCESS: Invalid shape dimensions (y=%s, x=%s), using default 512x512",
                max_y,
                max_x,
            )
            return (max(max_y, self.default_image_shape[0]), max(max_x, self.default_image_shape[1]))
        return (max_y, max_x)

    def _coordinate_bounds(
        self,
        shape_dict: Mapping[str, object],
        shape_kind: NapariShapeKind,
    ) -> tuple[int, int]:
        coords = np.array(shape_dict["coordinates"])
        if len(coords) == 0:
            return (0, 0)
        return (int(np.max(coords[:, 0])) + 1, int(np.max(coords[:, 1])) + 1)

    def _component_indices(
        self,
        item: Mapping[str, object],
        stack_components: list[str],
        component_values: Mapping[str, list[object]],
    ) -> tuple[int, ...]:
        components = item["components"]
        return tuple(
            component_values[component].index(components.get(component, 0))
            for component in stack_components
        )

    def _paint_polygon(
        self,
        shape_dict: Mapping[str, object],
        context: NapariShapePaintContext,
    ) -> int:
        from skimage import draw

        coords = np.array(shape_dict["coordinates"])
        rr, cc = draw.polygon(
            coords[:, 0],
            coords[:, 1],
            shape=context.labels_array.shape[-2:],
        )
        full_indices = context.indices + (rr, cc)
        context.labels_array[full_indices] = context.label_id
        return context.label_id + 1

    def _paint_path(
        self,
        shape_dict: Mapping[str, object],
        context: NapariShapePaintContext,
    ) -> int:
        coords = np.array(shape_dict["coordinates"])
        if len(coords) < 1:
            return context.label_id

        rr = coords[:, 0].astype(int)
        cc = coords[:, 1].astype(int)
        valid = (
            (rr >= 0)
            & (rr < context.labels_array.shape[-2])
            & (cc >= 0)
            & (cc < context.labels_array.shape[-1])
        )
        rr, cc = rr[valid], cc[valid]
        full_indices = context.indices + (rr, cc)
        context.labels_array[full_indices] = context.label_id
        return context.label_id + 1

    def _skip_points(
        self,
        shape_dict: Mapping[str, object],
        context: NapariShapePaintContext,
    ) -> int:
        return context.label_id
