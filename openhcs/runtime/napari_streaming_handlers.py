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
    ViewerComponentValueOrdering,
)


logger = logging.getLogger(__name__)


ComponentValue: TypeAlias = str | int | float | bool | tuple | None
ComponentMap: TypeAlias = dict[str, ComponentValue]
ComponentValues: TypeAlias = dict[str, list[ComponentValue]]
ComponentDomainKey: TypeAlias = str | tuple[str, ...] | tuple[str, tuple[str, ...]]
LayerKwargValue: TypeAlias = str | int | float | bool | tuple | list | dict | None
LayerDataPayload: TypeAlias = np.ndarray | list | tuple | str | int | float | bool | None
DimensionLabelMap: TypeAlias = dict[str, list[str]]
ShapePayloadValue: TypeAlias = LayerDataPayload | dict
ShapePayloadMap: TypeAlias = Mapping[str, ShapePayloadValue]


class NapariLayerHandle(ABC):
    """Nominal marker for concrete layer objects returned by a Napari viewer."""


class NapariLayerCollection(ABC):
    """Minimal layer collection contract used by OpenHCS streaming."""

    @abstractmethod
    def remove(self, layer: NapariLayerHandle) -> None:
        """Remove a concrete Napari layer from the viewer."""

    @abstractmethod
    def __contains__(self, layer: NapariLayerHandle) -> bool:
        """Return whether the concrete Napari layer is still mounted."""


class NapariDimsController(ABC):
    """Subset of napari dims state mutated by streaming updates."""

    axis_labels: tuple[str, ...]


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
class NapariStreamLayerItem:
    """One component-addressed payload staged for a Napari layer update."""

    data: LayerDataPayload
    components: ComponentMap
    path: str
    data_type: StreamingDataType


@dataclass(frozen=True, slots=True)
class NapariStreamingDataTypeHandler:
    """Layer-building behavior for one streaming data type."""

    data_type: StreamingDataType
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

    viewer: NapariViewerLayerCreator
    layers: dict[str, NapariLayerHandle]
    route_key: str
    layer_name: str
    layer_kind: NapariLayerKind
    data: LayerDataPayload
    layer_kwargs: Mapping[str, LayerKwargValue]


@dataclass(frozen=True, slots=True)
class NapariLayerLogPolicy:
    """Creation-log policy for one Napari layer kind."""

    layer_kind: NapariLayerKind
    count_data: bool

    def log_created(self, request: NapariLayerUpdateRequest) -> None:
        if self.count_data:
            count = len(request.data)
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


class NapariLayerCreatePolicy(ABC, metaclass=AutoRegisterMeta):
    """Create one concrete Napari layer for a typed layer kind."""

    __registry_key__ = "layer_kind"
    __skip_if_no_key__ = True
    layer_kind: ClassVar[NapariLayerKind | None] = None

    @classmethod
    def for_layer_kind(
        cls,
        layer_kind: NapariLayerKind,
    ) -> "NapariLayerCreatePolicy":
        return cls.__registry__[layer_kind]()

    @abstractmethod
    def create(self, request: NapariLayerUpdateRequest) -> NapariLayerHandle:
        """Create the layer described by request."""


@dataclass(frozen=True, slots=True)
class NapariImageLayerCreatePolicy(NapariLayerCreatePolicy):
    """Create image layers through the declared viewer API."""

    layer_kind = NapariLayerKind.IMAGE

    def create(self, request: NapariLayerUpdateRequest) -> NapariLayerHandle:
        return request.viewer.add_image(
            request.data,
            name=request.layer_name,
            **dict(request.layer_kwargs),
        )


@dataclass(frozen=True, slots=True)
class NapariShapesLayerCreatePolicy(NapariLayerCreatePolicy):
    """Create shapes layers through the declared viewer API."""

    layer_kind = NapariLayerKind.SHAPES

    def create(self, request: NapariLayerUpdateRequest) -> NapariLayerHandle:
        return request.viewer.add_shapes(
            request.data,
            name=request.layer_name,
            **dict(request.layer_kwargs),
        )


@dataclass(frozen=True, slots=True)
class NapariPointsLayerCreatePolicy(NapariLayerCreatePolicy):
    """Create points layers through the declared viewer API."""

    layer_kind = NapariLayerKind.POINTS

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

    @classmethod
    def colormap(cls, colormap: str | None) -> str:
        if colormap is None:
            return cls.DEFAULT
        return colormap


class NapariLayerUpdateAuthority:
    """Owns create-or-replace mechanics for Napari streaming layers."""

    def __init__(self) -> None:
        self._log_policies = napari_layer_log_policies()
        validate_napari_layer_create_policies()

    def create_or_update(self, request: NapariLayerUpdateRequest) -> NapariLayerHandle:
        existing_layer = self._existing_layer(request)
        if existing_layer is not None:
            request.viewer.layers.remove(existing_layer)
            request.layers.pop(request.route_key, None)
            logger.info(
                "🔬 NAPARI PROCESS: Removed existing %s layer %s for route %s",
                request.layer_kind.value,
                request.layer_name,
                request.route_key,
            )

        new_layer = NapariLayerCreatePolicy.for_layer_kind(
            request.layer_kind
        ).create(request)
        request.layers[request.route_key] = new_layer
        self._log_created_layer(request)
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
    ) -> NapariLayerHandle:
        layer = self.create_or_update(
            NapariLayerUpdateRequest(
                viewer=viewer,
                layers=layers,
                route_key=route_key,
                layer_name=layer_name,
                layer_kind=NapariLayerKind.IMAGE,
                data=image_data,
                layer_kwargs={
                    "colormap": NapariImageColormapPolicy.colormap(colormap)
                },
            )
        )
        if axis_labels is not None:
            viewer.dims.axis_labels = axis_labels
            logger.info("🔬 NAPARI PROCESS: Set viewer.dims.axis_labels=%s", axis_labels)
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
            NapariLayerUpdateRequest(
                viewer=viewer,
                layers=layers,
                route_key=route_key,
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
        viewer: NapariViewerLayerCreator,
        layers: dict[str, NapariLayerHandle],
        route_key: str,
        layer_name: str,
        points_data: LayerDataPayload,
        properties: dict,
    ) -> NapariLayerHandle:
        return self.create_or_update(
            NapariLayerUpdateRequest(
                viewer=viewer,
                layers=layers,
                route_key=route_key,
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

    def _log_created_layer(self, request: NapariLayerUpdateRequest) -> None:
        self._log_policies[request.layer_kind].log_created(request)


@dataclass(slots=True)
class NapariLayerStateStore:
    """Own per-layer Napari runtime state that must stay keyed together."""

    layers: dict[str, NapariLayerHandle]
    layer_titles: dict[str, str]
    dimension_labels: dict[str, DimensionLabelMap]
    pending_updates: dict[str, NapariTimerHandle]

    @classmethod
    def empty(cls) -> "NapariLayerStateStore":
        return cls(layers={}, layer_titles={}, dimension_labels={}, pending_updates={})

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
        self.dimension_labels.pop(layer_key, None)
        self.pending_updates.pop(layer_key, None)

    def labels_for(self, layer_key: str) -> DimensionLabelMap:
        if layer_key not in self.dimension_labels:
            return {}
        return self.dimension_labels[layer_key]

    def set_labels(self, layer_key: str, labels: DimensionLabelMap) -> None:
        self.dimension_labels[layer_key] = labels

    def cancel_pending_update(self, layer_key: str) -> bool:
        if layer_key not in self.pending_updates:
            return False
        timer = self.pending_updates[layer_key]
        timer.stop()
        return True

    def set_pending_update(self, layer_key: str, timer: NapariTimerHandle) -> None:
        self.pending_updates[layer_key] = timer

    def pop_pending_update(self, layer_key: str) -> NapariTimerHandle | None:
        if layer_key not in self.pending_updates:
            return None
        return self.pending_updates.pop(layer_key)

    def has_layer(self, layer_key: str) -> bool:
        return layer_key in self.layers

    def layer(self, layer_key: str) -> NapariLayerHandle:
        return self.layers[layer_key]

    def set_layer(self, layer_key: str, layer: NapariLayerHandle) -> None:
        self.layers[layer_key] = layer


@dataclass(slots=True)
class NapariComponentMetadataNormalizer:
    """Normalize component metadata before Napari stack indexing."""

    indexed_components: frozenset[str] = frozenset(
        {"site", "channel", "z_index", "timepoint"}
    )

    def normalize(self, components: ComponentMap) -> ComponentMap:
        return {
            component: self.normalize_value(component, value)
            for component, value in components.items()
        }

    def normalize_value(self, component: str, value: ComponentValue) -> ComponentValue:
        if component not in self.indexed_components:
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped.lstrip("+-").isdigit():
                return int(stripped)
        return value


@dataclass(slots=True)
class NapariBatchProcessorStore:
    """Own lazy NapariBatchProcessor instances by layer key."""

    debounce_delay_ms: int
    max_debounce_wait_ms: int = 5000
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


@dataclass(slots=True)
class NapariComponentValueDomain:
    """Store observed component values for keyed streaming domains."""

    values_by_key: dict[ComponentDomainKey, dict[str, set[ComponentValue]]] = field(
        default_factory=dict
    )

    def update(
        self,
        domain_key: ComponentDomainKey,
        stack_components: Sequence[str],
        layer_items: Sequence[NapariStreamLayerItem],
    ) -> None:
        if domain_key not in self.values_by_key:
            self.values_by_key[domain_key] = {
                comp: set() for comp in stack_components
            }

        observed_values = self.values_by_key[domain_key]
        for item in layer_items:
            components = item.components
            for comp in stack_components:
                if comp in components:
                    observed_values[comp].add(components[comp])

    def values_for(
        self,
        domain_key: ComponentDomainKey,
        stack_components: Sequence[str],
    ) -> ComponentValues:
        if domain_key not in self.values_by_key:
            return {comp: [] for comp in stack_components}

        return {
            comp: sorted(values, key=ViewerComponentValueOrdering.key)
            for comp, values in self.values_by_key[domain_key].items()
        }


@dataclass(slots=True)
class NapariComponentValueTracker:
    """Track observed component values for one streamed Napari route."""

    domain: NapariComponentValueDomain = field(default_factory=NapariComponentValueDomain)

    def update(
        self,
        route_key: str,
        stack_components: Sequence[str],
        layer_items: Sequence[NapariStreamLayerItem],
    ) -> None:
        self.domain.update(
            self._domain_key(route_key, stack_components),
            stack_components,
            layer_items,
        )

    def values_for(self, route_key: str, stack_components: Sequence[str]) -> ComponentValues:
        return self.domain.values_for(
            self._domain_key(route_key, stack_components),
            stack_components,
        )

    @staticmethod
    def _domain_key(
        route_key: str,
        stack_components: Sequence[str],
    ) -> tuple[str, tuple[str, ...]]:
        return (route_key, tuple(stack_components))


@dataclass(slots=True)
class NapariDisplayAxisDomain:
    """Track the shared viewer axis domain for one stack-component layout."""

    domain: NapariComponentValueDomain = field(default_factory=NapariComponentValueDomain)

    def update(
        self,
        stack_components: Sequence[str],
        layer_items: Sequence[NapariStreamLayerItem],
    ) -> None:
        self.domain.update(tuple(stack_components), stack_components, layer_items)

    def values_for(self, stack_components: Sequence[str]) -> ComponentValues:
        return self.domain.values_for(tuple(stack_components), stack_components)


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
        stack_components: Sequence[str],
        component_values: ComponentValues,
    ) -> np.ndarray:
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
            *(len(component_values[component]) for component in stack_components),
            *image_shape,
        ]
        labels_array = np.zeros(nd_shape, dtype=np.uint16)

        label_id = 1
        for item in layer_items:
            indices = self._component_indices(item, stack_components, component_values)
            for shape_dict in item.data:
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
        stack_components: Sequence[str],
        component_values: ComponentValues,
    ) -> tuple[int, ...]:
        components = item.components
        return tuple(
            self._component_index(component, components, component_values)
            for component in stack_components
        )

    def _component_index(
        self,
        component: str,
        components: Mapping[str, ComponentValue],
        component_values: ComponentValues,
    ) -> int:
        values = component_values[component]
        if not values:
            return 0
        if component not in components:
            logger.warning(
                "🔬 NAPARI PROCESS: Shape item missing stack component %s; "
                "placing on first %s plane.",
                component,
                component,
            )
            return 0
        value = components[component]
        if value in values:
            return values.index(value)
        logger.warning(
            "🔬 NAPARI PROCESS: Shape item has %s=%r outside stack values %s; "
            "placing on first %s plane.",
            component,
            value,
            values,
            component,
        )
        return 0

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
            shape=context.labels_array.shape[-2:],
        )
        full_indices = context.indices + (rr, cc)
        context.labels_array[full_indices] = context.label_id
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
        shape_dict: ShapePayloadMap,
        context: NapariShapePaintContext,
    ) -> int:
        return context.label_id
