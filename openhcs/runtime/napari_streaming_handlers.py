"""Shared Napari streaming handler axis declarations."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, TypeAlias

import numpy as np

from metaclass_registry import AutoRegisterMeta
from polystore.streaming_constants import StreamingDataType

from openhcs.runtime.viewer_protocol import (
    NapariLayerKind,
)
from openhcs.runtime.viewer_component_system import (
    ComponentMap,
    ComponentValue,
    ComponentValues,
    ViewerComponentCoordinateAuthority,
    ViewerLayerAxisProjection,
)


logger = logging.getLogger(__name__)


LayerKwargValue: TypeAlias = str | int | float | bool | tuple | list | dict | None
LayerDataPayload: TypeAlias = np.ndarray | list | tuple | str | int | float | bool | None
DimensionLabelMap: TypeAlias = dict[str, list[str]]
ShapePayloadValue: TypeAlias = LayerDataPayload | dict
ShapePayloadMap: TypeAlias = Mapping[str, ShapePayloadValue]


class NapariLayerHandle(ABC):
    """Nominal marker for concrete layer objects returned by a Napari viewer."""


class NapariLayerSelectionController(ABC):
    """Minimal layer-selection contract used by OpenHCS streaming."""

    @property
    @abstractmethod
    def active(self) -> NapariLayerHandle | None:
        """Return the currently selected Napari layer."""

    @active.setter
    @abstractmethod
    def active(self, layer: NapariLayerHandle | None) -> None:
        """Set the currently selected Napari layer."""


class NapariLayerCollection(ABC):
    """Minimal layer collection contract used by OpenHCS streaming."""

    selection: NapariLayerSelectionController

    @abstractmethod
    def remove(self, layer: NapariLayerHandle) -> None:
        """Remove a concrete Napari layer from the viewer."""

    @abstractmethod
    def __contains__(self, layer: NapariLayerHandle) -> bool:
        """Return whether the concrete Napari layer is still mounted."""


class NapariDimsController(ABC):
    """Subset of napari dims state mutated by streaming updates."""

    axis_labels: tuple[str, ...]
    ndim: int


class NapariViewerLayerCreator(ABC):
    """Explicit layer creation contract instead of reflective add_* lookup."""

    layers: NapariLayerCollection
    dims: NapariDimsController

    @abstractmethod
    def add_image(
        self,
        data: LayerDataPayload,
        *,
        name: str,
        **kwargs: LayerKwargValue,
    ) -> NapariLayerHandle:
        """Create an image layer."""

    @abstractmethod
    def add_shapes(
        self,
        data: LayerDataPayload,
        *,
        name: str,
        **kwargs: LayerKwargValue,
    ) -> NapariLayerHandle:
        """Create a shapes layer."""

    @abstractmethod
    def add_points(
        self,
        data: LayerDataPayload,
        *,
        name: str,
        **kwargs: LayerKwargValue,
    ) -> NapariLayerHandle:
        """Create a points layer."""

    @abstractmethod
    def add_labels(
        self,
        data: LayerDataPayload,
        *,
        name: str,
        **kwargs: LayerKwargValue,
    ) -> NapariLayerHandle:
        """Create a labels layer."""


class NapariTimerHandle(ABC):
    """Timer contract stored by layer state."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the pending layer update."""


@dataclass(frozen=True, slots=True)
class NapariStreamLayerAddress:
    """Component/path/data-type identity for one staged stream payload."""

    components: ComponentMap
    path: str
    stream_layer_data_type: StreamingDataType

    def with_components(self, components: ComponentMap) -> "NapariStreamLayerAddress":
        """Return the same payload address after component normalization."""
        return NapariStreamLayerAddress(
            components=components,
            path=self.path,
            stream_layer_data_type=self.stream_layer_data_type,
        )

    def same_layer_slot(self, other: "NapariStreamLayerAddress") -> bool:
        """Return whether two payloads replace the same component/data-type slot."""
        return (
            self.components == other.components
            and self.stream_layer_data_type == other.stream_layer_data_type
        )


@dataclass(frozen=True, slots=True)
class NapariStreamLayerItem:
    """One component-addressed payload staged for a Napari layer update."""

    data: LayerDataPayload
    address: NapariStreamLayerAddress


@dataclass(frozen=True, slots=True)
class NapariStreamingDataTypeHandler:
    """Layer-building behavior for one streaming data type."""

    handled_stream_data_type: StreamingDataType
    build_nd_data: Callable[..., LayerDataPayload]
    create_layer: Callable[..., NapariLayerHandle]


NapariStreamingDataTypeHandlers = Mapping[
    StreamingDataType,
    NapariStreamingDataTypeHandler,
]


def build_napari_streaming_data_type_handlers(
    *,
    build_image_data: Callable[..., LayerDataPayload],
    create_image_layer: Callable[..., NapariLayerHandle],
    build_shapes_data: Callable[..., LayerDataPayload],
    create_shapes_layer: Callable[..., NapariLayerHandle],
    build_points_data: Callable[..., LayerDataPayload],
    create_points_layer: Callable[..., NapariLayerHandle],
) -> dict[StreamingDataType, NapariStreamingDataTypeHandler]:
    """Build the canonical StreamingDataType handler table for Napari viewers."""
    return {
        StreamingDataType.IMAGE: NapariStreamingDataTypeHandler(
            handled_stream_data_type=StreamingDataType.IMAGE,
            build_nd_data=build_image_data,
            create_layer=create_image_layer,
        ),
        StreamingDataType.SHAPES: NapariStreamingDataTypeHandler(
            handled_stream_data_type=StreamingDataType.SHAPES,
            build_nd_data=build_shapes_data,
            create_layer=create_shapes_layer,
        ),
        StreamingDataType.POINTS: NapariStreamingDataTypeHandler(
            handled_stream_data_type=StreamingDataType.POINTS,
            build_nd_data=build_points_data,
            create_layer=create_points_layer,
        ),
    }


def napari_streaming_data_type_handler(
    handlers: NapariStreamingDataTypeHandlers,
    stream_layer_data_type: StreamingDataType | str,
) -> NapariStreamingDataTypeHandler:
    """Return the handler for one streaming data type, failing loudly if absent."""
    resolved_data_type = (
        StreamingDataType(stream_layer_data_type)
        if isinstance(stream_layer_data_type, str)
        else stream_layer_data_type
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

    viewer: NapariViewerLayerCreator
    layers: dict[str, NapariLayerHandle]
    route_key: str
    layer_name: str
    data: LayerDataPayload
    layer_kwargs: Mapping[str, LayerKwargValue]


class NapariLayerCreatePolicy(ABC, metaclass=AutoRegisterMeta):
    """Create one concrete Napari layer for a typed layer kind."""

    __registry_key__ = "NAPARI_LAYER_KIND"
    __skip_if_no_key__ = True
    NAPARI_LAYER_KIND: ClassVar[NapariLayerKind | None] = None
    LOG_CREATED_ITEM_COUNT: ClassVar[bool] = False

    @classmethod
    def for_layer_kind(
        cls,
        layer_kind: NapariLayerKind,
    ) -> "NapariLayerCreatePolicy":
        return cls.__registry__[layer_kind]()

    @abstractmethod
    def create(self, request: NapariLayerUpdateRequest) -> NapariLayerHandle:
        """Create the layer described by request."""

    @classmethod
    def registered_layer_kind(cls) -> NapariLayerKind:
        """Return the registered Napari layer kind for this policy class."""
        if cls.NAPARI_LAYER_KIND is None:
            raise ValueError(f"{cls.__name__} is not registered for a Napari layer kind.")
        return cls.NAPARI_LAYER_KIND

    def log_created(self, request: NapariLayerUpdateRequest) -> None:
        """Log creation of the layer described by request."""
        layer_kind = self.registered_layer_kind()
        if self.LOG_CREATED_ITEM_COUNT:
            count = len(request.data)
            logger.info(
                "🔬 NAPARI PROCESS: Created %s layer %s with %d %s",
                layer_kind.value,
                request.layer_name,
                count,
                layer_kind.value,
            )
            return
        logger.info(
            "🔬 NAPARI PROCESS: Created %s layer %s",
            layer_kind.value,
            request.layer_name,
        )


@dataclass(frozen=True, slots=True)
class NapariImageLayerCreatePolicy(NapariLayerCreatePolicy):
    """Create image layers through the declared viewer API."""

    NAPARI_LAYER_KIND = NapariLayerKind.IMAGE

    def create(self, request: NapariLayerUpdateRequest) -> NapariLayerHandle:
        return request.viewer.add_image(
            request.data,
            name=request.layer_name,
            **dict(request.layer_kwargs),
        )


@dataclass(frozen=True, slots=True)
class NapariShapesLayerCreatePolicy(NapariLayerCreatePolicy):
    """Create shapes layers through the declared viewer API."""

    NAPARI_LAYER_KIND = NapariLayerKind.SHAPES
    LOG_CREATED_ITEM_COUNT = True

    def create(self, request: NapariLayerUpdateRequest) -> NapariLayerHandle:
        return request.viewer.add_shapes(
            request.data,
            name=request.layer_name,
            **dict(request.layer_kwargs),
        )


@dataclass(frozen=True, slots=True)
class NapariPointsLayerCreatePolicy(NapariLayerCreatePolicy):
    """Create points layers through the declared viewer API."""

    NAPARI_LAYER_KIND = NapariLayerKind.POINTS
    LOG_CREATED_ITEM_COUNT = True

    def create(self, request: NapariLayerUpdateRequest) -> NapariLayerHandle:
        return request.viewer.add_points(
            request.data,
            name=request.layer_name,
            **dict(request.layer_kwargs),
        )


def validate_napari_layer_create_policies() -> None:
    """Ensure every Napari layer kind has exactly one registered create policy."""
    registered = set(NapariLayerCreatePolicy.__registry__)
    expected = set(NapariLayerKind)
    if registered != expected:
        missing = expected - registered
        raise ValueError(f"Missing Napari layer create policies for {missing!r}.")


class NapariImageColormapPolicy:
    """Formal default for image layer colormap."""

    DEFAULT = "gray"
    COLOR_CHANNEL_COUNTS = frozenset({3, 4})

    @classmethod
    def colormap(cls, colormap: str | None) -> str:
        if colormap is None:
            return cls.DEFAULT
        return colormap

    @classmethod
    def layer_kwargs(
        cls,
        image_data: LayerDataPayload,
        colormap: str | None,
    ) -> dict[str, LayerKwargValue]:
        if cls.is_rgb(image_data):
            return {"rgb": True}
        return {"colormap": cls.colormap(colormap)}

    @classmethod
    def is_rgb(cls, image_data: LayerDataPayload) -> bool:
        shape = tuple(int(dimension) for dimension in np.shape(image_data))
        return len(shape) >= 3 and shape[-1] in cls.COLOR_CHANNEL_COUNTS


@dataclass(frozen=True, slots=True)
class NapariLayerSelectionSnapshot:
    """Selection state before an automatic layer replacement."""

    active_layer: NapariLayerHandle | None
    replacing_active_layer: bool


class NapariLayerSelectionAuthority:
    """Preserve user layer selection across automatic streaming updates."""

    @classmethod
    def capture(
        cls,
        viewer: NapariViewerLayerCreator,
        replaced_layer: NapariLayerHandle | None,
    ) -> NapariLayerSelectionSnapshot:
        active_layer = cls._active_layer(viewer)
        return NapariLayerSelectionSnapshot(
            active_layer=active_layer,
            replacing_active_layer=active_layer is not None
            and active_layer is replaced_layer,
        )

    @classmethod
    def restore(
        cls,
        viewer: NapariViewerLayerCreator,
        snapshot: NapariLayerSelectionSnapshot,
        replacement_layer: NapariLayerHandle,
    ) -> None:
        if snapshot.active_layer is None:
            return
        if snapshot.replacing_active_layer:
            cls._set_active_layer(viewer, replacement_layer)
            return
        if snapshot.active_layer in viewer.layers:
            cls._set_active_layer(viewer, snapshot.active_layer)

    @staticmethod
    def _active_layer(
        viewer: NapariViewerLayerCreator,
    ) -> NapariLayerHandle | None:
        return viewer.layers.selection.active

    @staticmethod
    def _set_active_layer(
        viewer: NapariViewerLayerCreator,
        layer: NapariLayerHandle,
    ) -> None:
        viewer.layers.selection.active = layer


class NapariLayerUpdateAuthority:
    """Owns create-or-replace mechanics for Napari streaming layers."""

    def __init__(self) -> None:
        validate_napari_layer_create_policies()

    def create_or_update(
        self,
        policy: NapariLayerCreatePolicy,
        request: NapariLayerUpdateRequest,
    ) -> NapariLayerHandle:
        existing_layer = self._existing_layer(request)
        selection = NapariLayerSelectionAuthority.capture(
            request.viewer,
            existing_layer,
        )
        layer_kind = policy.registered_layer_kind()
        if existing_layer is not None:
            request.viewer.layers.remove(existing_layer)
            request.layers.pop(request.route_key, None)
            logger.info(
                "🔬 NAPARI PROCESS: Removed existing %s layer %s for route %s",
                layer_kind.value,
                request.layer_name,
                request.route_key,
            )

        new_layer = policy.create(request)
        request.layers[request.route_key] = new_layer
        NapariLayerSelectionAuthority.restore(
            request.viewer,
            selection,
            new_layer,
        )
        policy.log_created(request)
        return new_layer

    def create_or_update_image(
        self,
        viewer: NapariViewerLayerCreator,
        layers: dict[str, NapariLayerHandle],
        route_key: str,
        layer_name: str,
        image_data: LayerDataPayload,
        colormap: str | None,
        axis_labels: tuple[str, ...] | None = None,
        translate: tuple[float, ...] | None = None,
    ) -> NapariLayerHandle:
        layer_kwargs = NapariImageColormapPolicy.layer_kwargs(
            image_data,
            colormap,
        )
        if axis_labels is not None:
            layer_kwargs["axis_labels"] = axis_labels
        if translate is not None:
            layer_kwargs["translate"] = translate

        layer = self.create_or_update(
            NapariLayerCreatePolicy.for_layer_kind(NapariLayerKind.IMAGE),
            NapariLayerUpdateRequest(
                viewer=viewer,
                layers=layers,
                route_key=route_key,
                layer_name=layer_name,
                data=image_data,
                layer_kwargs=layer_kwargs,
            )
        )
        if axis_labels is not None:
            logger.info(
                "🔬 NAPARI PROCESS: Route %s carries layer-local axis_labels=%s",
                route_key,
                axis_labels,
            )
        return layer

    def create_or_update_shapes(
        self,
        viewer: NapariViewerLayerCreator,
        layers: dict[str, NapariLayerHandle],
        route_key: str,
        layer_name: str,
        shapes_data: LayerDataPayload,
        shape_types: LayerDataPayload,
        properties: dict,
    ) -> NapariLayerHandle:
        return self.create_or_update(
            NapariLayerCreatePolicy.for_layer_kind(NapariLayerKind.SHAPES),
            NapariLayerUpdateRequest(
                viewer=viewer,
                layers=layers,
                route_key=route_key,
                layer_name=layer_name,
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
        viewer: NapariViewerLayerCreator,
        layers: dict[str, NapariLayerHandle],
        route_key: str,
        layer_name: str,
        points_data: LayerDataPayload,
        properties: dict,
        translate: tuple[float, ...] | None = None,
    ) -> NapariLayerHandle:
        layer_kwargs = {
            "properties": properties,
            "face_color": "green",
            "size": 3,
        }
        if translate is not None:
            layer_kwargs["translate"] = translate
        return self.create_or_update(
            NapariLayerCreatePolicy.for_layer_kind(NapariLayerKind.POINTS),
            NapariLayerUpdateRequest(
                viewer=viewer,
                layers=layers,
                route_key=route_key,
                layer_name=layer_name,
                data=points_data,
                layer_kwargs=layer_kwargs,
            )
        )

    def _existing_layer(
        self,
        request: NapariLayerUpdateRequest,
    ) -> NapariLayerHandle | None:
        if request.route_key not in request.layers:
            return None
        layer = request.layers[request.route_key]
        if layer in request.viewer.layers:
            return layer
        return None

@dataclass(slots=True)
class NapariDimensionLayerState:
    """Semantic dimension-label state for one streamed Napari layer."""

    labels: DimensionLabelMap
    presentation: "NapariAxisPresentation | None" = None

    @classmethod
    def empty(cls) -> "NapariDimensionLayerState":
        return cls(labels={})

    @property
    def stack_axes(self) -> tuple[str, ...]:
        if self.presentation is None:
            return ()
        return self.presentation.axis_projection.projected_axis_components

    @property
    def axis_labels(self) -> tuple[str, ...]:
        if self.presentation is None:
            return ()
        return self.presentation.axis_labels

    @property
    def axis_offsets(self) -> tuple[int, ...]:
        if self.presentation is None:
            return ()
        return self.presentation.axis_projection.axis_offsets


@dataclass(frozen=True, slots=True)
class NapariAxisPresentation:
    """Layer-local axis semantics projected into the viewer coordinate space."""

    layer_key: str
    axis_projection: ViewerLayerAxisProjection
    payload_axis_labels: tuple[str, ...] = ()

    @property
    def axis_labels(self) -> tuple[str, ...]:
        return tuple(
            [
                *self.axis_projection.projected_axis_components,
                *self.payload_axis_labels,
                "y",
                "x",
            ]
        )

    @classmethod
    def from_projection(
        cls,
        *,
        layer_key: str,
        projection: ViewerLayerAxisProjection,
        payload_axis_labels: tuple[str, ...] = (),
    ) -> "NapariAxisPresentation":
        return cls(
            layer_key=layer_key,
            axis_projection=projection,
            payload_axis_labels=payload_axis_labels,
        )

    def label_index(self, viewer_step: int, axis_index: int) -> int:
        return viewer_step - self.axis_projection.axis_offset(axis_index)


@dataclass(slots=True)
class NapariLayerRouteStateStore:
    """Own per-layer Napari runtime state that must stay keyed together."""

    layers: dict[str, NapariLayerHandle]
    layer_titles: dict[str, str]
    layer_dimension_states: dict[str, NapariDimensionLayerState]
    layer_pending_updates: dict[str, NapariTimerHandle]
    active_dimension_label_route: str | None

    @classmethod
    def empty(cls) -> "NapariLayerRouteStateStore":
        return cls(
            layers={},
            layer_titles={},
            layer_dimension_states={},
            layer_pending_updates={},
            active_dimension_label_route=None,
        )

    def set_title(self, layer_key: str, title: str) -> None:
        self.layer_titles[layer_key] = title

    def title_for(self, layer_key: str) -> str:
        return self.layer_titles[layer_key]

    def title_collides(self, layer_key: str, title: str) -> bool:
        return any(
            other_key != layer_key and other_title == title
            for other_key, other_title in self.layer_titles.items()
        )

    def purge_route(self, layer_key: str) -> None:
        self.layers.pop(layer_key, None)
        self.layer_titles.pop(layer_key, None)
        self.layer_dimension_states.pop(layer_key, None)
        self.layer_pending_updates.pop(layer_key, None)
        if self.active_dimension_label_route == layer_key:
            self.active_dimension_label_route = None

    def dimension_state_for(self, layer_key: str) -> NapariDimensionLayerState:
        if layer_key not in self.layer_dimension_states:
            return NapariDimensionLayerState.empty()
        return self.layer_dimension_states[layer_key]

    def set_dimension_state(
        self,
        layer_key: str,
        state: NapariDimensionLayerState,
    ) -> None:
        self.layer_dimension_states[layer_key] = state

    def set_active_dimension_label_route(self, layer_key: str) -> None:
        self.active_dimension_label_route = layer_key

    def cancel_pending_update(self, layer_key: str) -> bool:
        if layer_key not in self.layer_pending_updates:
            return False
        timer = self.layer_pending_updates[layer_key]
        timer.stop()
        return True

    def set_pending_update(self, layer_key: str, timer: NapariTimerHandle) -> None:
        self.layer_pending_updates[layer_key] = timer

    def pop_pending_update(self, layer_key: str) -> NapariTimerHandle | None:
        if layer_key not in self.layer_pending_updates:
            return None
        return self.layer_pending_updates.pop(layer_key)

    def has_layer(self, layer_key: str) -> bool:
        return layer_key in self.layers

    def layer(self, layer_key: str) -> NapariLayerHandle:
        return self.layers[layer_key]

    def set_layer(self, layer_key: str, layer: NapariLayerHandle) -> None:
        self.layers[layer_key] = layer


@dataclass(slots=True)
class NapariComponentGroupStore:
    """Own accumulated stream items by Napari layer route."""

    groups: dict[str, list["NapariStreamLayerItem"]] = field(default_factory=dict)

    def items_for(self, layer_key: str) -> list["NapariStreamLayerItem"]:
        if layer_key not in self.groups:
            self.groups[layer_key] = []
        return self.groups[layer_key]

    def existing_items_for(
        self,
        layer_key: str,
    ) -> list["NapariStreamLayerItem"] | None:
        if layer_key not in self.groups:
            return None
        return self.groups[layer_key]

    def item_count(self, layer_key: str) -> int:
        items = self.existing_items_for(layer_key)
        if items is None:
            return 0
        return len(items)

    def purge(self, layer_key: str) -> None:
        self.groups.pop(layer_key, None)

    def clear(self) -> None:
        self.groups.clear()

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)


@dataclass(slots=True)
class NapariLayerBatchDebouncePolicy:
    """Shared debounce policy for Napari layer updates and batch processors."""

    delay_ms: int = 1000
    max_wait_ms: int = 5000

    def start_timer(self, timer: NapariTimerHandle) -> None:
        timer.start(self.delay_ms)

    def create_processor(
        self,
        *,
        napari_server: "NapariServerDisplayProtocol",
        batch_size: int | None,
    ) -> "NapariBatchProcessor":
        from polystore.streaming.receivers.napari import NapariBatchProcessor

        return NapariBatchProcessor(
            napari_server=napari_server,
            batch_size=batch_size,
            debounce_delay_ms=self.delay_ms,
            max_debounce_wait_ms=self.max_wait_ms,
        )


@dataclass(slots=True)
class NapariBatchProcessorStore:
    """Own lazy NapariBatchProcessor instances by layer key."""

    debounce_policy: NapariLayerBatchDebouncePolicy = field(
        default_factory=NapariLayerBatchDebouncePolicy
    )
    processors: dict[str, "NapariBatchProcessor"] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_create(
        self,
        *,
        layer_key: str,
        napari_server: "NapariServerDisplayProtocol",
        batch_size: int | None = None,
    ) -> "NapariBatchProcessor":
        with self.lock:
            if layer_key not in self.processors:
                self.processors[layer_key] = self.debounce_policy.create_processor(
                    napari_server=napari_server,
                    batch_size=batch_size,
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
    """Mutable label volume position passed to shape painters."""

    target_label_volume: np.ndarray
    indices: tuple[int, ...]
    label_id: int

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return (
            int(self.target_label_volume.shape[-2]),
            int(self.target_label_volume.shape[-1]),
        )

    def paint_pixels(self, rows: np.ndarray, columns: np.ndarray) -> None:
        target_indices = self.indices + (rows, columns)
        self.target_label_volume[target_indices] = self.label_id


class NapariShapeLabelRasterizer:
    """Convert streamed Napari shape payloads into dense label arrays."""

    default_image_shape: tuple[int, int] = (512, 512)

    def __init__(self) -> None:
        self._paint_routes: Mapping[
            NapariShapeKind,
            Callable[[ShapePayloadMap, NapariShapePaintContext], int],
        ] = {
            NapariShapeKind.POLYGON: self._paint_polygon,
            NapariShapeKind.PATH: self._paint_path,
            NapariShapeKind.POINTS: self._skip_points,
        }

    def rasterize(
        self,
        *,
        layer_items: Sequence[NapariStreamLayerItem],
        axis_projection: ViewerLayerAxisProjection,
    ) -> np.ndarray:
        projected_axis_components = axis_projection.projected_axis_components
        component_values = axis_projection.component_values
        logger.info("🔬 NAPARI PROCESS: Building ROI stack with global component values")
        for component, values in component_values.items():
            logger.info("🔬 NAPARI PROCESS:   %s: %s", component, values)

        first_shapes = layer_items[0].data
        if not first_shapes:
            logger.warning(
                "🔬 NAPARI PROCESS: No shapes data, creating default 512x512 array"
            )
            return np.zeros((1, 1, *self.default_image_shape), dtype=np.uint16)

        image_shape = self._image_shape(layer_items)
        nd_shape = [
            *(len(component_values[component]) for component in projected_axis_components),
            *image_shape,
        ]
        label_volume = np.zeros(nd_shape, dtype=np.uint16)

        label_id = 1
        for item in layer_items:
            indices = self._component_indices(item, projected_axis_components, component_values)
            for shape_dict in item.data:
                shape_kind = NapariShapeKind(str(shape_dict["type"]))
                label_id = self._paint_routes[shape_kind](
                    shape_dict,
                    NapariShapePaintContext(label_volume, indices, label_id),
                )

        logger.info(
            "🔬 NAPARI PROCESS: Created labels array with shape %s and %d labels",
            label_volume.shape,
            label_id - 1,
        )
        return label_volume

    def _image_shape(
        self,
        layer_items: Sequence[NapariStreamLayerItem],
    ) -> tuple[int, int]:
        source_shape = self._source_spatial_shape(layer_items)
        if source_shape is not None:
            return source_shape

        max_y, max_x = 0, 0
        for item in layer_items:
            for shape_dict in item.data:
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

    def _source_spatial_shape(
        self,
        layer_items: Sequence[NapariStreamLayerItem],
    ) -> tuple[int, int] | None:
        for item in layer_items:
            for shape_dict in item.data:
                if "metadata" not in shape_dict:
                    continue
                metadata = shape_dict["metadata"]
                if not isinstance(metadata, Mapping):
                    continue
                if "source_spatial_shape_yx" not in metadata:
                    continue
                source_shape = metadata["source_spatial_shape_yx"]
                if len(source_shape) != 2:
                    raise ValueError(
                        "source_spatial_shape_yx metadata must have two values, "
                        f"got {source_shape!r}."
                    )
                return (int(source_shape[0]), int(source_shape[1]))
        return None

    def _coordinate_bounds(
        self,
        shape_dict: ShapePayloadMap,
        shape_kind: NapariShapeKind,
    ) -> tuple[int, int]:
        coords = np.array(shape_dict["coordinates"])
        if len(coords) == 0:
            return (0, 0)
        return (int(np.max(coords[:, 0])) + 1, int(np.max(coords[:, 1])) + 1)

    def _component_indices(
        self,
        item: NapariStreamLayerItem,
        projected_axis_components: Sequence[str],
        component_values: ComponentValues,
    ) -> tuple[int, ...]:
        components = item.address.components
        return tuple(
            self._component_index(component, components, component_values)
            for component in projected_axis_components
        )

    def _component_index(
        self,
        component: str,
        components: Mapping[str, ComponentValue],
        component_values: ComponentValues,
    ) -> int:
        return ViewerComponentCoordinateAuthority.index(
            components=components,
            component_values=component_values,
            component=component,
            context="Napari shape item",
        )

    def _paint_polygon(
        self,
        shape_dict: ShapePayloadMap,
        context: NapariShapePaintContext,
    ) -> int:
        from skimage import draw

        coords = np.array(shape_dict["coordinates"])
        rr, cc = draw.polygon(
            coords[:, 0],
            coords[:, 1],
            shape=context.spatial_shape,
        )
        context.paint_pixels(rr, cc)
        return context.label_id + 1

    def _paint_path(
        self,
        shape_dict: ShapePayloadMap,
        context: NapariShapePaintContext,
    ) -> int:
        coords = np.array(shape_dict["coordinates"])
        if len(coords) < 1:
            return context.label_id

        rr = coords[:, 0].astype(int)
        cc = coords[:, 1].astype(int)
        max_row, max_column = context.spatial_shape
        valid = (
            (rr >= 0)
            & (rr < max_row)
            & (cc >= 0)
            & (cc < max_column)
        )
        rr, cc = rr[valid], cc[valid]
        context.paint_pixels(rr, cc)
        return context.label_id + 1

    def _skip_points(
        self,
        shape_dict: ShapePayloadMap,
        context: NapariShapePaintContext,
    ) -> int:
        return context.label_id
