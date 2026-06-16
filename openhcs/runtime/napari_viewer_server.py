"""
Napari-based real-time visualization module.

This module provides the NapariStreamVisualizer class for real-time
visualization of tensors during pipeline execution.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import sys
import threading
import time
import zmq
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional, TypeAlias
from qtpy.QtCore import QTimer

from openhcs.core.config import TransportMode as OpenHCSTransportMode
from metaclass_registry import AutoRegisterMeta
from polystore.backend_registry import register_cleanup_callback
from zmqruntime.config import TransportMode, ZMQConfig
from polystore.streaming_constants import StreamingDataType
from polystore.streaming.identity import (
    StreamProducerDisplayNameAuthority,
    StreamProducerIdentity,
)
from polystore.streaming.receivers.napari import (
    normalize_component_layout,
    build_route_key,
)
from openhcs.runtime.viewer_protocol import (
    ChannelColormapPolicy,
    ComponentDimensionLabelPolicy,
    NAPARI_HEARTBEAT,
    NapariViewerServerRequest,
    ViewerComponentValueOrdering,
    ViewerQtEnvironmentPolicy,
    ViewerProtocolStatus,
)
from openhcs.runtime.napari_streaming_handlers import (
    ComponentMap,
    ComponentValue,
    ComponentValues,
    LayerDataPayload,
    NapariLayerHandle,
    NapariBatchProcessorStore,
    NapariComponentValueTracker,
    NapariDisplayAxisDomain,
    NapariComponentMetadataNormalizer,
    NapariLayerUpdateAuthority,
    NapariLayerStateStore,
    NapariStreamLayerItem,
    NapariShapeLabelRasterizer,
    NapariViewerLayerCreator,
    build_napari_streaming_data_type_handlers,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from zmqruntime.streaming import StreamingVisualizerServer, VisualizerProcessManager
from zmqruntime.transport import (
    coerce_transport_mode,
    get_control_url,
    get_zmq_transport_url,
    is_port_in_use,
    ping_control_port,
    wait_for_server_ready,
)

# Optional napari import - this module should only be imported if napari is available
try:
    import napari
except ImportError:
    napari = None

if napari is None:
    raise ImportError(
        "napari is required for NapariStreamVisualizer. "
        "Install it with: pip install 'openhcs[viz]' or pip install napari"
    )


logger = logging.getLogger(__name__)
_NAPARI_LAYER_UPDATES = NapariLayerUpdateAuthority()
DEFAULT_DIRECT_IMAGE_PATH = "<direct_napari_image>"
DEFAULT_IMAGE_DATA_TYPE = "image"
DEFAULT_IMAGE_COLORMAP = "viridis"

NapariWireValue: TypeAlias = (
    str | int | float | bool | np.ndarray | np.dtype | tuple | list | dict | None
)
NapariWirePayload: TypeAlias = dict[str, NapariWireValue]
NapariWirePayloads: TypeAlias = list[NapariWirePayload]
NapariDisplayConfigPayload: TypeAlias = NapariWirePayload
NapariComponentNameMetadata: TypeAlias = dict[str, dict[str, ComponentValue]]
NapariComponentModeMap: TypeAlias = dict[str, str]
NapariComponentGroups: TypeAlias = dict[str, list[NapariStreamLayerItem]]
NapariControlReplyPayload: TypeAlias = dict[str, NapariWireValue]


class NapariWireField(str, Enum):
    """Wire keys used by Napari stream and ROI payloads."""

    TYPE = "type"
    METADATA = "metadata"
    IMAGES = "images"
    DISPLAY_CONFIG = "display_config"
    COMPONENT_NAMES_METADATA = "component_names_metadata"
    PATH = "path"
    IMAGE_ID = "image_id"
    DATA_TYPE = "data_type"
    SHAPES = "shapes"
    SHAPE = "shape"
    DTYPE = "dtype"
    SHM_NAME = "shm_name"
    DATA = "data"
    PRODUCER_IDENTITY = "producer_identity"
    COORDINATES = "coordinates"


class VisualMetadataField(str, Enum):
    """Optional visual metadata fields attached to ROI payloads."""

    CENTROID = "centroid"
    LABEL = "label"
    AREA = "area"
    COMPONENT = "component"


@dataclass(frozen=True)
class PayloadMap:
    """Typed accessor for one JSON-derived payload mapping."""

    payload: NapariWirePayload
    context: str

    def required(
        self,
        field: NapariWireField,
    ) -> NapariWireValue:
        if field.value not in self.payload:
            raise ValueError(
                f"{self.context} missing required payload field '{field.value}'"
            )
        return self.payload[field.value]

    def optional(
        self,
        field: NapariWireField,
    ) -> NapariWireValue | None:
        if field.value in self.payload:
            return self.payload[field.value]
        return None

    def value_or_default(
        self,
        field: NapariWireField,
        default: NapariWireValue,
    ) -> NapariWireValue:
        if field.value in self.payload:
            return self.payload[field.value]
        return default

    def required_mapping(
        self,
        field: NapariWireField,
    ) -> NapariWirePayload:
        return self._mapping(field, self.required(field))

    def optional_mapping(
        self,
        field: NapariWireField,
    ) -> NapariWirePayload:
        if field.value not in self.payload:
            return {}
        return self._mapping(field, self.payload[field.value])

    def required_payloads(
        self,
        field: NapariWireField,
    ) -> NapariWirePayloads:
        value = self.required(field)
        if not isinstance(value, list):
            raise TypeError(
                f"{self.context} field '{field.value}' must be a list, "
                f"got {type(value).__name__}"
            )
        payloads = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise TypeError(
                    f"{self.context} field '{field.value}' item {index} must be a dict, "
                    f"got {type(item).__name__}"
                )
            payloads.append(item)
        return payloads

    def _mapping(
        self,
        field: NapariWireField,
        value: NapariWireValue,
    ) -> NapariWirePayload:
        if not isinstance(value, dict):
            raise TypeError(
                f"{self.context} field '{field.value}' must be a dict, "
                f"got {type(value).__name__}"
            )
        return value


class NapariComponentMetadataPayload:
    """Coerce wire metadata into the two semantic metadata maps used by Napari."""

    @classmethod
    def component_map(
        cls,
        payload: NapariWirePayload,
        context: str,
    ) -> ComponentMap:
        return {
            str(component): cls._component_value(value, context)
            for component, value in payload.items()
        }

    @classmethod
    def component_name_metadata(
        cls,
        payload: NapariWirePayload,
        context: str,
    ) -> NapariComponentNameMetadata:
        metadata: NapariComponentNameMetadata = {}
        for component, component_payload in payload.items():
            if not isinstance(component_payload, dict):
                raise TypeError(
                    f"{context} component metadata for {component!r} must be a dict, "
                    f"got {type(component_payload).__name__}"
                )
            metadata[str(component)] = {
                str(value): cls._component_value(name, context)
                for value, name in component_payload.items()
            }
        return metadata

    @staticmethod
    def _component_value(
        value: NapariWireValue,
        context: str,
    ) -> ComponentValue:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise TypeError(
            f"{context} component value must be scalar or tuple-like, "
            f"got {type(value).__name__}"
        )


class StackComponentAuthority:
    """Fail-loud lookup for semantic stack components."""

    @staticmethod
    def required(components: ComponentMap, component: str):
        if component not in components:
            raise ValueError(f"Streamed item missing stack component '{component}'")
        return components[component]


@dataclass(frozen=True)
class ShapePayload:
    """Typed view of one Napari ROI shape payload."""

    payload: NapariWirePayload

    @property
    def shape_type(self) -> str:
        return str(PayloadMap(self.payload, "Napari shape payload").required(
            NapariWireField.TYPE
        ))

    @property
    def coordinates(self) -> LayerDataPayload:
        return PayloadMap(self.payload, "Napari shape payload").required(
            NapariWireField.COORDINATES
        )

    @property
    def metadata(self) -> "VisualMetadata":
        return VisualMetadata(
            PayloadMap(self.payload, "Napari shape payload").optional_mapping(
                NapariWireField.METADATA
            )
        )


@dataclass(frozen=True)
class VisualMetadata:
    """Optional display metadata attached to one shape payload."""

    metadata: NapariWirePayload

    def value(
        self,
        field: VisualMetadataField,
        default: NapariWireValue,
    ) -> NapariWireValue:
        if field.value in self.metadata:
            return self.metadata[field.value]
        return default


@dataclass(frozen=True)
class ComponentLayout:
    """Normalized component layout for Napari layer grouping."""

    component_modes: NapariComponentModeMap
    component_order: tuple[str, ...]

    @classmethod
    def from_display_config(
        cls,
        display_config: NapariDisplayConfigPayload,
    ) -> "ComponentLayout":
        component_modes, component_order = normalize_component_layout(display_config)
        return cls(
            component_modes=component_modes,
            component_order=tuple(component_order),
        )

    @property
    def stack_components(self) -> tuple[str, ...]:
        return tuple(
            component
            for component in self.component_order
            if self.component_modes[component] == "stack"
        )


@dataclass(frozen=True)
class NapariBatchPayload:
    """Typed view of an incoming batch message."""

    msg_type: NapariWireValue | None
    images: NapariWirePayloads
    display_config: NapariDisplayConfigPayload
    component_names_metadata: NapariComponentNameMetadata

    @classmethod
    def from_json_payload(cls, data: NapariWirePayload) -> "NapariBatchPayload":
        payload = PayloadMap(data, "Napari batch message")
        return cls(
            msg_type=payload.optional(NapariWireField.TYPE),
            images=payload.required_payloads(NapariWireField.IMAGES),
            display_config=payload.required_mapping(NapariWireField.DISPLAY_CONFIG),
            component_names_metadata=NapariComponentMetadataPayload.component_name_metadata(
                payload.optional_mapping(NapariWireField.COMPONENT_NAMES_METADATA),
                "Napari component-name metadata",
            ),
        )


@dataclass(frozen=True)
class NapariImagePayload:
    """Typed view of one image/shapes message."""

    raw: NapariWirePayload
    display_config: NapariDisplayConfigPayload
    path: str
    image_id: NapariWireValue | None
    data_type: str
    component_metadata: ComponentMap
    producer_identity: StreamProducerIdentity

    @classmethod
    def from_payload(
        cls,
        image_info: NapariWirePayload,
        display_config: NapariDisplayConfigPayload,
    ) -> "NapariImagePayload":
        payload = PayloadMap(image_info, "Napari image message")
        return cls(
            raw=image_info,
            display_config=display_config,
            path=str(payload.required(NapariWireField.PATH)),
            image_id=payload.optional(NapariWireField.IMAGE_ID),
            data_type=str(payload.value_or_default(
                NapariWireField.DATA_TYPE,
                DEFAULT_IMAGE_DATA_TYPE,
            )),
            component_metadata=NapariComponentMetadataPayload.component_map(
                payload.optional_mapping(NapariWireField.METADATA),
                "Napari image component metadata",
            ),
            producer_identity=StreamProducerIdentity.from_payload(
                payload.required(NapariWireField.PRODUCER_IDENTITY)
            ),
        )

    @property
    def shapes(self) -> LayerDataPayload:
        return PayloadMap(self.raw, "Napari shapes/points message").required(
            NapariWireField.SHAPES
        )

    @property
    def image_shape(self) -> tuple[int, ...]:
        shape = PayloadMap(self.raw, "Napari image message").required(
            NapariWireField.SHAPE
        )
        if isinstance(shape, tuple):
            return tuple(int(dimension) for dimension in shape)
        if isinstance(shape, list):
            return tuple(int(dimension) for dimension in shape)
        raise TypeError(
            "Napari image message field 'shape' must be a sequence, "
            f"got {type(shape).__name__}"
        )

    @property
    def dtype(self) -> str | np.dtype:
        return PayloadMap(self.raw, "Napari image message").required(
            NapariWireField.DTYPE
        )

    @property
    def shm_name(self) -> str | None:
        shm_name = PayloadMap(self.raw, "Napari image message").optional(
            NapariWireField.SHM_NAME
        )
        if shm_name is None:
            return None
        return str(shm_name)

    @property
    def direct_data(self) -> LayerDataPayload:
        return PayloadMap(self.raw, "Napari image message").optional(
            NapariWireField.DATA
        )

    @property
    def colormap(self) -> str:
        if "colormap" in self.display_config:
            return self.display_config["colormap"]
        return DEFAULT_IMAGE_COLORMAP


_COMPONENT_DIMENSION_LABELS = ComponentDimensionLabelPolicy()
_NAPARI_SHAPE_RASTERIZER = NapariShapeLabelRasterizer()
_COMPONENT_METADATA_NORMALIZER = NapariComponentMetadataNormalizer()
_ACK_ERROR = ViewerProtocolStatus.ERROR.value
_ACK_SUCCESS = ViewerProtocolStatus.SUCCESS.value


@dataclass(frozen=True)
class NapariLoadedPayloadData:
    """Loaded display data and visual parameters for one stream payload."""

    data: LayerDataPayload
    colormap: str | None


class NapariPayloadDataLoader:
    """Load payload data by streaming data kind."""

    SHAPE_LIKE_DATA_TYPES = frozenset(
        {StreamingDataType.SHAPES, StreamingDataType.POINTS}
    )

    def load(self, payload: NapariImagePayload) -> NapariLoadedPayloadData:
        data_type = StreamingDataType(payload.data_type)
        if data_type in self.SHAPE_LIKE_DATA_TYPES:
            return NapariLoadedPayloadData(data=payload.shapes, colormap=None)
        return NapariLoadedPayloadData(
            data=self._image_data(payload),
            colormap=payload.colormap,
        )

    def _image_data(self, payload: NapariImagePayload) -> np.ndarray:
        if payload.shm_name:
            return self._shared_memory_image(payload)
        if payload.direct_data is not None:
            return np.array(payload.direct_data, dtype=payload.dtype).reshape(
                payload.image_shape
            )
        raise ValueError("No image data in message")

    def _shared_memory_image(self, payload: NapariImagePayload) -> np.ndarray:
        from multiprocessing import shared_memory

        try:
            shm = shared_memory.SharedMemory(name=payload.shm_name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Shared memory {payload.shm_name} not found"
            ) from exc
        try:
            return np.ndarray(
                payload.image_shape,
                dtype=payload.dtype,
                buffer=shm.buf,
            ).copy()
        finally:
            shm.close()
            self._unlink_shared_memory(payload.shm_name)

    @staticmethod
    def _unlink_shared_memory(shm_name: str) -> None:
        from multiprocessing import shared_memory

        shm = None
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.unlink()
        except FileNotFoundError:
            logger.debug(
                "🔬 NAPARI PROCESS: Shared memory %s already unlinked",
                shm_name,
            )
        finally:
            if shm is not None:
                shm.close()


@dataclass(frozen=True)
class NapariLayerRoute:
    """Resolved route identity for one component-aware display item."""

    layer_key: str
    layer_title: str
    component_info: ComponentMap
    component_layout: ComponentLayout
    data_type: StreamingDataType


@dataclass(frozen=True)
class NapariComponentAwareDisplayRequest:
    """Request for routing one loaded payload into a Napari layer group."""

    viewer: NapariViewerLayerCreator
    layers: dict[str, NapariLayerHandle]
    component_groups: NapariComponentGroups
    data: LayerDataPayload
    path: str
    colormap: str | None
    display_config: NapariDisplayConfigPayload
    replace_layers: bool
    component_metadata: ComponentMap
    producer_identity: StreamProducerIdentity
    data_type: StreamingDataType
    server: "NapariViewerServer"

    @classmethod
    def from_inputs(
        cls,
        *,
        viewer: NapariViewerLayerCreator,
        layers: dict[str, NapariLayerHandle],
        component_groups: NapariComponentGroups,
        data: LayerDataPayload,
        path: str,
        colormap: str | None,
        display_config: NapariDisplayConfigPayload,
        replace_layers: bool,
        component_metadata: ComponentMap,
        producer_identity: StreamProducerIdentity,
        data_type: StreamingDataType | str,
        server: "NapariViewerServer",
    ) -> "NapariComponentAwareDisplayRequest":
        if server is None:
            raise ValueError("Server instance required for debounced updates")
        if not component_metadata:
            raise ValueError(f"No component metadata available for path: {path}")
        return cls(
            viewer=viewer,
            layers=layers,
            component_groups=component_groups,
            data=data,
            path=path,
            colormap=colormap,
            display_config=display_config,
            replace_layers=replace_layers,
            component_metadata=component_metadata,
            producer_identity=producer_identity,
            data_type=StreamingDataType(data_type),
            server=server,
        )


class NapariLayerItemAuthority:
    """Own item identity and replacement inside one layer group."""

    @staticmethod
    def item(
        request: NapariComponentAwareDisplayRequest,
        route: NapariLayerRoute,
    ) -> NapariStreamLayerItem:
        return NapariStreamLayerItem(
            data=request.data,
            components=route.component_info,
            path=str(request.path),
            data_type=route.data_type,
        )

    @staticmethod
    def matching_index(
        group: list[NapariStreamLayerItem],
        route: NapariLayerRoute,
    ) -> int | None:
        for index, item in enumerate(group):
            if (
                item.components == route.component_info
                and item.data_type == route.data_type
            ):
                return index
        return None


class NapariComponentAwareDisplayCoordinator:
    """Route loaded payload data into debounced Napari layer updates."""

    item_authority = NapariLayerItemAuthority()

    def display(self, request: NapariComponentAwareDisplayRequest) -> None:
        route = self._route(request)
        self._log_route(route)
        self._reconcile_deleted_layer(request, route.layer_key)
        group = self._group_for(request, route.layer_key)
        self._clear_group_for_replace(request, route.layer_key, group)
        self._upsert_item(request, route, group)
        logger.info(
            "🔬 NAPARI PROCESS: Scheduling debounced update for %s (data_type=%s)",
            route.layer_key,
            route.data_type,
        )
        request.server.display_pipeline.schedule_layer_update(
            route.layer_key,
            route.data_type,
            route.component_layout,
        )

    def _route(self, request: NapariComponentAwareDisplayRequest) -> NapariLayerRoute:
        component_info = _COMPONENT_METADATA_NORMALIZER.normalize(
            request.component_metadata
        )
        component_layout = ComponentLayout.from_display_config(request.display_config)
        layer_key = build_route_key(
            producer_identity=request.producer_identity,
            component_info=component_info,
            component_modes=component_layout.component_modes,
            component_order=component_layout.component_order,
            data_type=request.data_type,
        )
        layer_title = NapariLayerTitleAuthority.disambiguate(
            title=NapariLayerTitleAuthority.title(
                producer=request.producer_identity,
                data_type=request.data_type,
                component_info=component_info,
                component_layout=component_layout,
            ),
            producer=request.producer_identity,
            route_key=layer_key,
            layer_state=request.server.layer_state,
        )
        request.server.layer_state.set_title(layer_key, layer_title)
        return NapariLayerRoute(
            layer_key=layer_key,
            layer_title=layer_title,
            component_info=component_info,
            component_layout=component_layout,
            data_type=request.data_type,
        )

    @staticmethod
    def _log_route(route: NapariLayerRoute) -> None:
        logger.info(
            "🔍 NAPARI PROCESS: component_modes=%s, layer_key='%s'",
            route.component_layout.component_modes,
            route.layer_key,
        )
        logger.info(
            "🔍 NAPARI PROCESS: layer_key='%s', component_info=%s",
            route.layer_key,
            route.component_info,
        )

    @staticmethod
    def _reconcile_deleted_layer(
        request: NapariComponentAwareDisplayRequest,
        layer_key: str,
    ) -> None:
        if layer_key in request.layers and request.layers[layer_key] not in request.viewer.layers:
            num_items = 0
            if layer_key in request.component_groups:
                num_items = len(request.component_groups[layer_key])
            request.server.layer_state.purge_route(layer_key)
            request.component_groups.pop(layer_key, None)
            logger.info(
                "🔬 NAPARI PROCESS: Reconciling state — '%s' was deleted from viewer; "
                "purged stale caches (had %d items in component_groups)",
                layer_key,
                num_items,
            )

    @staticmethod
    def _group_for(
        request: NapariComponentAwareDisplayRequest,
        layer_key: str,
    ) -> list[NapariStreamLayerItem]:
        if layer_key not in request.component_groups:
            request.component_groups[layer_key] = []
        return request.component_groups[layer_key]

    @staticmethod
    def _clear_group_for_replace(
        request: NapariComponentAwareDisplayRequest,
        layer_key: str,
        group: list[NapariStreamLayerItem],
    ) -> None:
        if request.replace_layers and group:
            logger.info(
                "🔬 NAPARI PROCESS: replace_layers=True, clearing %d existing items "
                "from layer '%s'",
                len(group),
                layer_key,
            )
            group.clear()

    def _upsert_item(
        self,
        request: NapariComponentAwareDisplayRequest,
        route: NapariLayerRoute,
        group: list[NapariStreamLayerItem],
    ) -> None:
        new_item = self.item_authority.item(request, route)
        existing_index = self.item_authority.matching_index(group, route)
        if existing_index is None:
            group.append(new_item)
            logger.info(
                "🔬 NAPARI PROCESS: Added %s to component_groups[%s], now has %d items",
                route.data_type,
                route.layer_key,
                len(group),
            )
            return

        old_data_type = group[existing_index].data_type
        group[existing_index] = new_item
        logger.info(
            "🔬 NAPARI PROCESS: Replaced %s item in component_groups[%s] at index %d, "
            "total items: %d",
            old_data_type,
            route.layer_key,
            existing_index,
            len(group),
        )


_NAPARI_PAYLOAD_DATA_LOADER = NapariPayloadDataLoader()
_NAPARI_COMPONENT_DISPLAY_COORDINATOR = NapariComponentAwareDisplayCoordinator()

# ZMQ connection delay (ms)
ZMQ_CONNECTION_DELAY_MS = 100  # Brief delay for ZMQ connection to establish

# Global process management for napari viewer
_global_viewer_process: Optional[multiprocessing.Process] = None
_global_viewer_port: Optional[int] = None
_global_process_lock = threading.Lock()

# Registry of data type handlers (will be populated after helper functions are defined)
_DATA_TYPE_HANDLERS = None


def _cleanup_global_viewer() -> None:
    """
    Clean up global napari viewer process for test mode.

    This forcibly terminates the napari viewer process to allow pytest to exit.
    Should only be called in test mode.
    """
    global _global_viewer_process

    with _global_process_lock:
        if _global_viewer_process and _global_viewer_process.is_alive():
            logger.info("🔬 VISUALIZER: Terminating napari viewer for test cleanup")
            _global_viewer_process.terminate()
            _global_viewer_process.join(timeout=3)

            if _global_viewer_process.is_alive():
                logger.warning("🔬 VISUALIZER: Force killing napari viewer process")
                _global_viewer_process.kill()
                _global_viewer_process.join(timeout=1)

            _global_viewer_process = None


register_cleanup_callback(_cleanup_global_viewer)


def _build_nd_shapes(
    layer_items: list[NapariStreamLayerItem],
    stack_components,
):
    """
    Build nD shapes by prepending stack component indices to 2D shape coordinates.

    Args:
        layer_items: List of items with 'data' (shapes_data) and 'components'
        stack_components: List of component names to stack

    Returns:
        Tuple of (all_shapes_nd, all_shape_types, all_properties)
    """
    from polystore.roi_converters import NapariROIConverter

    all_shapes_nd = []
    all_shape_types = []
    all_properties = {"label": [], "area": [], "centroid_y": [], "centroid_x": []}

    # Build component value to index mapping (same as _build_nd_image_array)
    component_values = {}
    for comp in stack_components:
        values = sorted(
            set(
                StackComponentAuthority.required(item.components, comp)
                for item in layer_items
            ),
            key=ViewerComponentValueOrdering.key,
        )
        component_values[comp] = values

    for item in layer_items:
        shapes_data = item.data  # List of shape dicts
        components = item.components

        # Get stack component INDICES to prepend (not values!)
        prepend_dims = [
            component_values[comp].index(StackComponentAuthority.required(components, comp))
            for comp in stack_components
        ]

        # Convert each shape to nD
        for shape_dict in shapes_data:
            # Use registry-based dimension handler
            nd_coords = NapariROIConverter.add_dimensions_to_shape(
                shape_dict, prepend_dims
            )
            all_shapes_nd.append(nd_coords)
            all_shape_types.append(shape_dict["type"])

            # Extract properties
            metadata = ShapePayload(shape_dict).metadata
            centroid = metadata.value(VisualMetadataField.CENTROID, (0, 0))
            all_properties["label"].append(
                metadata.value(VisualMetadataField.LABEL, "")
            )
            all_properties["area"].append(
                metadata.value(VisualMetadataField.AREA, 0)
            )
            all_properties["centroid_y"].append(centroid[0])
            all_properties["centroid_x"].append(centroid[1])

    return all_shapes_nd, all_shape_types, all_properties


def _build_nd_points(
    layer_items: list[NapariStreamLayerItem],
    stack_components,
    component_values: ComponentValues | None = None,
):
    """
    Build nD points by prepending stack component indices to 2D point coordinates.

    Args:
        layer_items: List of items with 'data' (list of point coordinate arrays) and 'components'
        stack_components: List of component names to stack
        component_values: Optional dict of {component: [sorted values]} to use for mapping.
                         If provided, uses this for building the stack dimensions.
                         If None, derives from layer_items.

    Returns:
        Tuple of (all_points_nd, all_properties)
    """
    all_points_nd = []
    all_properties = {"label": [], "component": []}

    # Build component value to index mapping (use global if provided)
    if component_values is None:
        component_values = {}
        for comp in stack_components:
            values = sorted(
                set(
                    StackComponentAuthority.required(item.components, comp)
                    for item in layer_items
                ),
                key=ViewerComponentValueOrdering.key,
            )
            component_values[comp] = values

    for item in layer_items:
        points_data = item.data  # List of shape dicts from ROI converter
        components = item.components

        # DEBUG: Log what we actually have
        logger.info(f"🐛 DEBUG: points_data type: {type(points_data)}")
        if isinstance(points_data, list) and len(points_data) > 0:
            logger.info(f"🐛 DEBUG: first element type: {type(points_data[0])}")
            logger.info(f"🐛 DEBUG: first element: {points_data[0]}")

        # Get stack component INDICES to prepend
        prepend_dims = [
            component_values[comp].index(StackComponentAuthority.required(components, comp))
            for comp in stack_components
        ]

        # Convert each shape dict to nD points
        # points_data is a list of dicts with 'type', 'coordinates', 'metadata'
        for shape_dict in points_data:
            shape_payload = ShapePayload(shape_dict)
            # Only process 'points' type entries
            if shape_payload.shape_type != "points":
                continue

            coordinates = shape_payload.coordinates
            metadata = shape_payload.metadata

            # coordinates is a list of [y, x] pairs
            # Prepend stack dimensions to each point: [y, x] -> [stack_idx, ..., y, x]
            for coord in coordinates:
                nd_coord = prepend_dims + list(coord)
                all_points_nd.append(nd_coord)

                # Track properties for this point
                all_properties["label"].append(
                    metadata.value(VisualMetadataField.LABEL, "")
                )
                all_properties["component"].append(
                    metadata.value(VisualMetadataField.COMPONENT, 0)
                )

    points_array = np.empty((0, 2 + len(stack_components)))
    if all_points_nd:
        points_array = np.array(all_points_nd)
    return points_array, all_properties


def _build_nd_image_array(
    layer_items: list[NapariStreamLayerItem],
    stack_components,
    component_values: ComponentValues | None = None,
):
    """
    Build nD image array by stacking images along stack component dimensions.

    Args:
        layer_items: List of items with 'data' (image arrays) and 'components'
        stack_components: List of component names to stack
        component_values: Optional dict of {component: [sorted values]} to use for mapping.
                         If provided, uses this for building the stack dimensions.
                         If None, derives from layer_items (old behavior).

    Returns:
        np.ndarray: Stacked image array
    """
    # When component_values is provided (global tracker), always build multi-dimensional array
    # This ensures ROIs at non-first indices get proper stack dimensions immediately
    if component_values is not None:
        # Using global component values - build proper multi-dimensional array
        # even if we only have one item currently
        pass  # Fall through to multi-dimensional logic below
    elif len(stack_components) == 1 and len(layer_items) > 1:
        # Old behavior: Single stack component with multiple items - simple 3D stack
        image_stack = [img.data for img in layer_items]
        from openhcs.core.memory import stack_slices

        return stack_slices(image_stack, memory_type="numpy", gpu_id=0)
    elif len(stack_components) == 1 and len(layer_items) == 1:
        # Single item, single component, no global values - just return as-is
        # (Will be wrapped in extra dimension if needed by caller)
        return layer_items[0].data

    # Multiple stack components OR using global component values - create multi-dimensional array
    if component_values is None:
        # Derive from layer items (old behavior when no global tracker)
        component_values = {}
        for comp in stack_components:
            values = sorted(
                set(
                    StackComponentAuthority.required(img.components, comp)
                    for img in layer_items
                ),
                key=ViewerComponentValueOrdering.key,
            )
            component_values[comp] = values

    # Log component values for debugging
    logger.info(
        f"🔬 NAPARI PROCESS: Building nD array with stack_components={stack_components}, component_values={component_values}"
    )

    # Create empty array with shape (comp1_size, comp2_size, ..., y, x)
    first_img = layer_items[0].data
    stack_shape = (
        tuple(len(component_values[comp]) for comp in stack_components)
        + first_img.shape
    )
    stacked_array = np.zeros(stack_shape, dtype=first_img.dtype)
    logger.info(
        f"🔬 NAPARI PROCESS: Created nD array with shape {stack_shape} from {len(layer_items)} items"
    )

    # Fill array
    for img in layer_items:
        # Get indices for this image
        indices = tuple(
            component_values[comp].index(
                StackComponentAuthority.required(img.components, comp)
            )
            for comp in stack_components
        )
        logger.debug(
            f"🔬 NAPARI PROCESS: Placing image at indices {indices}, components={img.components}"
        )
        stacked_array[indices] = img.data

    return stacked_array


def _create_or_update_image_layer(
    viewer, layers, route_key, layer_name, image_data, colormap, axis_labels=None
):
    """Create or update a Napari image layer."""
    return _NAPARI_LAYER_UPDATES.create_or_update_image(
        viewer,
        layers,
        route_key,
        layer_name,
        image_data,
        colormap,
        axis_labels,
    )


def _create_or_update_shapes_layer(
    viewer, layers, route_key, layer_name, shapes_data, shape_types, properties
):
    """Create or update a Napari shapes layer."""
    return _NAPARI_LAYER_UPDATES.create_or_update_shapes(
        viewer,
        layers,
        route_key,
        layer_name,
        shapes_data,
        shape_types,
        properties,
    )


def _create_or_update_points_layer(
    viewer, layers, route_key, layer_name, points_data, properties
):
    """Create or update a Napari points layer."""
    return _NAPARI_LAYER_UPDATES.create_or_update_points(
        viewer,
        layers,
        route_key,
        layer_name,
        points_data,
        properties,
    )


_DATA_TYPE_HANDLERS = build_napari_streaming_data_type_handlers(
    build_image_data=_build_nd_image_array,
    create_image_layer=_create_or_update_image_layer,
    build_shapes_data=_build_nd_shapes,
    create_shapes_layer=_create_or_update_shapes_layer,
    build_points_data=_build_nd_points,
    create_points_layer=_create_or_update_points_layer,
)


class NapariLayerAxisPolicy:
    """Choose displayed stack axes for each Napari layer kind."""

    def stack_components_for(
        self,
        *,
        stack_components: tuple[str, ...] | list[str],
        component_values: ComponentValues,
    ) -> tuple[str, ...]:
        return self._non_singleton_components(stack_components, component_values)

    @staticmethod
    def _non_singleton_components(
        stack_components: tuple[str, ...] | list[str],
        component_values: ComponentValues,
    ) -> tuple[str, ...]:
        return tuple(
            component
            for component in stack_components
            if len(component_values[component]) > 1
        )

class NapariLayerTitleAuthority:
    """Build visible layer titles from producer identity and real slice axes."""

    DATA_TYPE_SUFFIX = {
        StreamingDataType.IMAGE: "",
        StreamingDataType.SHAPES: "labels",
        StreamingDataType.POINTS: "points",
    }

    @classmethod
    def title(
        cls,
        *,
        producer: StreamProducerIdentity,
        data_type: StreamingDataType,
        component_info: ComponentMap,
        component_layout: ComponentLayout,
    ) -> str:
        parts = [StreamProducerDisplayNameAuthority.output_label(producer)]
        for component in component_layout.component_order:
            if component_layout.component_modes[component] != "slice":
                continue
            if component in component_info:
                parts.append(f"{component} {component_info[component]}")
        suffix = cls.DATA_TYPE_SUFFIX[data_type]
        if suffix:
            parts.append(suffix)
        return " ".join(str(part) for part in parts if part)

    @staticmethod
    def disambiguate(
        *,
        title: str,
        producer: StreamProducerIdentity,
        route_key: str,
        layer_state: NapariLayerStateStore,
    ) -> str:
        if not layer_state.title_collides(route_key, title):
            return title
        return f"{title} [{StreamProducerDisplayNameAuthority.disambiguation_label(producer)}]"


class NapariLayerDisplayPipeline:
    """Owns debounced Napari layer display and update routing."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server
        self.axis_policy = NapariLayerAxisPolicy()
        self.layer_update_routes = {
            StreamingDataType.IMAGE: self.update_image_layer,
            StreamingDataType.SHAPES: self.update_shapes_layer,
            StreamingDataType.POINTS: self.update_points_layer,
        }

    @staticmethod
    def component_values_for_components(
        component_values: ComponentValues,
        stack_components: tuple[str, ...] | list[str],
    ) -> ComponentValues:
        """Project component-value mapping onto the active displayed stack axes."""
        return {component: component_values[component] for component in stack_components}

    def apply_dimension_labels(
        self,
        layer_key: str,
        active_stack_components: tuple[str, ...],
        active_component_values: ComponentValues,
    ) -> tuple[str, ...] | None:
        """Store semantic labels for the active stack axes of one layer."""
        if not active_stack_components:
            self.server.layer_state.set_labels(layer_key, {})
            return None

        axis_labels = tuple([*active_stack_components, "y", "x"])
        logger.info(
            "🔬 NAPARI PROCESS: Built axis_labels=%s for stack_components=%s",
            axis_labels,
            active_stack_components,
        )

        dimension_labels = {}
        for component in active_stack_components:
            values = active_component_values[component]
            component_metadata = self.component_metadata_for(component)
            dimension_labels[component] = _COMPONENT_DIMENSION_LABELS.labels_for(
                component=component,
                values=values,
                metadata=component_metadata,
            )
        self.server.layer_state.set_labels(layer_key, dimension_labels)
        return axis_labels

    def component_metadata_for(self, component: str) -> dict[str, ComponentValue]:
        """Return optional display metadata for one component."""
        if component not in self.server.component_metadata:
            return {}
        return self.server.component_metadata[component]

    def schedule_layer_update(
        self,
        layer_key,
        data_type,
        component_layout: ComponentLayout,
    ) -> None:
        if self.server.layer_state.cancel_pending_update(layer_key):
            logger.debug(f"🔬 NAPARI PROCESS: Cancelled pending update for {layer_key}")

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(
            lambda: self.execute_layer_update(
                layer_key,
                data_type,
                component_layout,
            )
        )
        timer.start(self.server.update_delay_ms)
        self.server.layer_state.set_pending_update(layer_key, timer)
        logger.debug(
            f"🔬 NAPARI PROCESS: Scheduled update for {layer_key} in {self.server.update_delay_ms}ms"
        )

    def execute_layer_update(
        self,
        layer_key,
        data_type,
        component_layout: ComponentLayout,
    ) -> None:
        self.server.layer_state.pop_pending_update(layer_key)

        if layer_key not in self.server.component_groups:
            logger.warning(
                f"🔬 NAPARI PROCESS: No items found for {layer_key}, skipping update"
            )
            return
        layer_items = self.server.component_groups[layer_key]
        if not layer_items:
            logger.warning(
                f"🔬 NAPARI PROCESS: Empty item group for {layer_key}, skipping update"
            )
            return

        wells_in_layer = set(
            item.components["well"]
            for item in layer_items
            if "well" in item.components
        )
        logger.info(
            f"🔬 NAPARI PROCESS: layer_key='{layer_key}' has {len(layer_items)} items from wells: {sorted(wells_in_layer)}"
        )

        batch_processor = self.server.batch_processors.get_or_create(
            layer_key=layer_key,
            napari_server=self.server,
        )
        try:
            batch_processor.add_items(
                layer_key=layer_key,
                items=layer_items,
                display_payload=component_layout,
                component_names_metadata=self.server.component_metadata,
            )
        except Exception:
            logger.exception(
                "🔬 NAPARI PROCESS: Failed to update layer %s; viewer will keep processing messages",
                layer_key,
            )

    def display_layer_batch(
        self,
        *,
        layer_key: str,
        items: list[NapariStreamLayerItem],
        display_payload: ComponentLayout,
        component_names_metadata: NapariComponentNameMetadata,
    ) -> None:
        component_layout = display_payload
        if component_names_metadata:
            self.server.component_metadata.update(component_names_metadata)

        items_by_type: dict[StreamingDataType, list[NapariStreamLayerItem]] = {}
        for item in items:
            data_type = item.data_type
            if isinstance(data_type, str):
                data_type = StreamingDataType(data_type)
            if data_type not in items_by_type:
                items_by_type[data_type] = []
            items_by_type[data_type].append(item)

        for data_type, typed_items in items_by_type.items():
            update_route = self.layer_update_routes[data_type]
            update_route(
                layer_key,
                typed_items,
                component_layout.stack_components,
                component_layout.component_modes,
            )
            logger.info(
                "🔬 NAPARI PROCESS: Displayed %d %s item(s) in layer %s",
                len(typed_items),
                data_type.value,
                layer_key,
            )

    def setup_dimension_label_handler(self, layer_key, stack_components) -> None:
        if not self.server.viewer or not stack_components:
            return

        layer_labels = self.server.layer_state.labels_for(layer_key)
        if not layer_labels:
            return

        def update_dimension_label(event=None):
            try:
                current_step = self.server.viewer.dims.current_step
                label_parts = []
                for i, component in enumerate(stack_components):
                    if component in layer_labels:
                        labels = layer_labels[component]
                        if i < len(current_step):
                            idx = current_step[i]
                            if 0 <= idx < len(labels):
                                label = labels[idx]
                                if label and str(label).lower() != "none":
                                    label_parts.append(label)

                overlay_text = ""
                if label_parts:
                    overlay_text = " | ".join(label_parts)
                self.server.viewer.text_overlay.text = overlay_text

            except Exception as e:
                logger.debug(f"🔬 NAPARI PROCESS: Error updating dimension label: {e}")

        try:
            self.server.viewer.dims.events.current_step.connect(update_dimension_label)
            update_dimension_label()
            logger.info(
                f"🔬 NAPARI PROCESS: Set up dimension label handler for {layer_key}"
            )
        except Exception as e:
            logger.warning(
                f"🔬 NAPARI PROCESS: Failed to setup dimension label handler: {e}"
            )

    def update_image_layer(
        self,
        layer_key,
        layer_items,
        stack_components,
        component_modes,
    ) -> None:
        self.server.component_values.update(layer_key, stack_components, layer_items)
        self.server.display_axis_domain.update(stack_components, layer_items)
        component_values = self.server.display_axis_domain.values_for(stack_components)
        active_stack_components = self.axis_policy.stack_components_for(
            stack_components=stack_components,
            component_values=component_values,
        )
        active_component_values = self.component_values_for_components(
            component_values,
            active_stack_components,
        )

        shapes = [item.data.shape for item in layer_items]
        shape_ranks = {len(shape) for shape in shapes}
        if len(shape_ranks) > 1:
            raise ValueError(
                f"Layer {layer_key} contains mixed-rank image payloads: {sorted(set(shapes))}"
            )
        if len(set(shapes)) > 1:
            logger.info(
                f"🔬 NAPARI PROCESS: Images in layer {layer_key} have different shapes - padding to max size"
            )

            first_shape = shapes[0]
            max_shape = list(first_shape)
            for img_shape in shapes:
                for i, dim in enumerate(img_shape):
                    max_shape[i] = max(max_shape[i], dim)
            max_shape = tuple(max_shape)

            for item_index, img_info in enumerate(layer_items):
                img_data = img_info.data
                if img_data.shape != max_shape:
                    pad_width = []
                    for i, (current_dim, max_dim) in enumerate(
                        zip(img_data.shape, max_shape)
                    ):
                        pad_before = 0
                        pad_after = max_dim - current_dim
                        pad_width.append((pad_before, pad_after))

                    padded_data = np.pad(
                        img_data,
                        pad_width,
                        mode="constant",
                        constant_values=0,
                    )
                    layer_items[item_index] = NapariStreamLayerItem(
                        data=padded_data,
                        components=img_info.components,
                        path=img_info.path,
                        data_type=img_info.data_type,
                    )
                    logger.debug(
                        f"🔬 NAPARI PROCESS: Padded image from {img_data.shape} to {padded_data.shape}"
                    )

        logger.info(
            f"🔬 NAPARI PROCESS: Building nD data for {layer_key} from {len(layer_items)} items"
        )
        stacked_data = _build_nd_image_array(
            layer_items,
            active_stack_components,
            active_component_values,
        )

        colormap = None
        if "channel" in component_modes and component_modes["channel"] == "slice":
            first_item = layer_items[0]
            channel_value = StackComponentAuthority.required(
                first_item.components,
                "channel",
            )
            colormap = ChannelColormapPolicy().colormap(channel_value)

        axis_labels = self.apply_dimension_labels(
            layer_key,
            active_stack_components,
            active_component_values,
        )

        _create_or_update_image_layer(
            self.server.viewer,
            self.server.layer_state.layers,
            layer_key,
            self.server.layer_state.title_for(layer_key),
            stacked_data,
            colormap,
            axis_labels,
        )

        self.setup_dimension_label_handler(layer_key, active_stack_components)

    def update_shapes_layer(
        self,
        layer_key,
        layer_items,
        stack_components,
        component_modes,
    ) -> None:
        logger.info(
            f"🔬 NAPARI PROCESS: Converting shapes to labels for {layer_key} from {len(layer_items)} items"
        )

        self.server.component_values.update(layer_key, stack_components, layer_items)
        self.server.display_axis_domain.update(stack_components, layer_items)
        component_values = self.server.display_axis_domain.values_for(stack_components)
        active_stack_components = self.axis_policy.stack_components_for(
            stack_components=stack_components,
            component_values=component_values,
        )
        active_component_values = self.component_values_for_components(
            component_values,
            active_stack_components,
        )

        labels_data = _NAPARI_SHAPE_RASTERIZER.rasterize(
            layer_items=layer_items,
            stack_components=active_stack_components,
            component_values=active_component_values,
        )

        if self.server.layer_state.has_layer(layer_key):
            try:
                self.server.viewer.layers.remove(
                    self.server.layer_state.layer(layer_key)
                )
                logger.info(
                    f"🔬 NAPARI PROCESS: Removed existing labels layer {layer_key} for recreation"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to remove existing labels layer {layer_key}: {e}"
                )

        new_layer = self.server.viewer.add_labels(
            labels_data,
            name=self.server.layer_state.title_for(layer_key),
        )
        axis_labels = self.apply_dimension_labels(
            layer_key,
            active_stack_components,
            active_component_values,
        )
        if axis_labels is not None:
            self.server.viewer.dims.axis_labels = axis_labels
            logger.info(
                "🔬 NAPARI PROCESS: Set viewer.dims.axis_labels=%s",
                axis_labels,
            )
        self.server.layer_state.set_layer(layer_key, new_layer)
        self.setup_dimension_label_handler(layer_key, active_stack_components)
        logger.info(
            f"🔬 NAPARI PROCESS: Created labels layer {layer_key} with shape {labels_data.shape}"
        )

    def update_points_layer(
        self,
        layer_key,
        layer_items,
        stack_components,
        component_modes,
    ) -> None:
        points_items = [
            item
            for item in layer_items
            if item.data_type == StreamingDataType.POINTS
        ]

        if not points_items:
            logger.warning(
                f"🔬 NAPARI PROCESS: No POINTS items found for {layer_key}, skipping"
            )
            return

        logger.info(
            f"🔬 NAPARI PROCESS: Building points layer for {layer_key} from {len(points_items)} items (filtered from {len(layer_items)} total)"
        )

        self.server.component_values.update(layer_key, stack_components, layer_items)
        self.server.display_axis_domain.update(stack_components, layer_items)
        component_values = self.server.display_axis_domain.values_for(stack_components)
        active_stack_components = self.axis_policy.stack_components_for(
            stack_components=stack_components,
            component_values=component_values,
        )
        active_component_values = self.component_values_for_components(
            component_values,
            active_stack_components,
        )

        points_data, properties = _build_nd_points(
            points_items,
            active_stack_components,
            active_component_values,
        )

        _create_or_update_points_layer(
            self.server.viewer,
            self.server.layer_state.layers,
            layer_key,
            self.server.layer_state.title_for(layer_key),
            points_data,
            properties,
        )

        logger.info(
            f"🔬 NAPARI PROCESS: Created points layer {layer_key} with {len(points_data)} points"
        )


class NapariMessageTypeBase(ABC):
    """Shared class-level registry key contract for message handlers."""

    __registry_key__ = "message_type"
    __skip_if_no_key__ = True
    message_type: ClassVar[str | None] = None


class NapariControlMessageAction(NapariMessageTypeBase, metaclass=AutoRegisterMeta):
    """Registered handler for one Napari control message type."""

    @classmethod
    def for_message_type(cls, message_type: str | None) -> "NapariControlMessageAction":
        if message_type in cls.__registry__:
            return cls.__registry__[message_type]()
        return NapariUnknownControlMessageAction()

    @abstractmethod
    def handle(
        self,
        server: "NapariViewerServer",
        message: NapariWirePayload,
    ) -> NapariControlReplyPayload:
        """Handle a control message and return the control reply."""


class NapariShutdownControlMessageAction(NapariControlMessageAction):
    """Shared shutdown behavior for graceful and force shutdown requests."""

    message_type = None

    def handle(
        self,
        server: "NapariViewerServer",
        message: NapariWirePayload,
    ) -> NapariControlReplyPayload:
        del message
        logger.info("🔬 NAPARI SERVER: %s requested, closing viewer", self.message_type)
        server.request_shutdown()
        if server.viewer is not None:
            from qtpy import QtCore

            QtCore.QTimer.singleShot(100, server.viewer.close)
        return {
            "type": "shutdown_ack",
            "status": "success",
            "message": "Napari viewer shutting down",
        }


class NapariGracefulShutdownControlMessageAction(NapariShutdownControlMessageAction):
    """Registered graceful shutdown action."""

    message_type = "shutdown"


class NapariForceShutdownControlMessageAction(NapariShutdownControlMessageAction):
    """Registered force shutdown action."""

    message_type = "force_shutdown"


class NapariClearStateControlMessageAction(NapariControlMessageAction):
    """Registered action that clears accumulated streaming state."""

    message_type = "clear_state"

    def handle(
        self,
        server: "NapariViewerServer",
        message: NapariWirePayload,
    ) -> NapariControlReplyPayload:
        del message
        logger.info(
            "🔬 NAPARI SERVER: Clearing component groups (had %d groups)",
            len(server.component_groups),
        )
        server.component_groups.clear()
        return {
            "type": "clear_state_ack",
            "status": "success",
            "message": "Component groups cleared",
        }


class NapariUnknownControlMessageAction(NapariControlMessageAction):
    """Default no-op control action for unknown message types."""

    message_type = None

    def handle(
        self,
        server: "NapariViewerServer",
        message: NapariWirePayload,
    ) -> NapariControlReplyPayload:
        del server, message
        return {"status": "ok"}


class NapariStreamMessageHandler(NapariMessageTypeBase, metaclass=AutoRegisterMeta):
    """Registered handler for one Napari stream message type."""

    @classmethod
    def for_message_type(cls, message_type: str | None) -> "NapariStreamMessageHandler":
        if message_type not in cls.__registry__:
            raise ValueError(
                f"Napari stream messages must be registered message types, got {message_type!r}."
            )
        return cls.__registry__[message_type]()

    @abstractmethod
    def handle(self, server: "NapariViewerServer", data: NapariWirePayload) -> None:
        """Handle one decoded stream message."""


class NapariBatchStreamMessageHandler(NapariStreamMessageHandler):
    """Registered stream handler for batched Napari payloads."""

    message_type = "batch"

    def handle(self, server: "NapariViewerServer", data: NapariWirePayload) -> None:
        batch_payload = NapariBatchPayload.from_json_payload(data)
        if batch_payload.component_names_metadata:
            server.component_metadata.update(batch_payload.component_names_metadata)
            logger.info(
                "🔬 NAPARI PROCESS: Updated component metadata: %s",
                list(batch_payload.component_names_metadata.keys()),
            )

        for image_info in batch_payload.images:
            server._process_single_image(image_info, batch_payload.display_config)


class NapariViewerServer(StreamingVisualizerServer):
    """
    ZMQ server for Napari viewer that receives images from clients.

    Inherits from ZMQServer ABC to get ping/pong, port management, etc.
    Uses SUB socket to receive images from pipeline clients.
    """

    _server_type = "napari"  # Registration key for AutoRegisterMeta

    def __init__(
        self, request: NapariViewerServerRequest
    ):
        """
        Initialize Napari viewer server.

        Args:
            request: Typed Napari server construction request.
        """
        import zmq

        # Initialize with REP socket for receiving images (synchronous request/reply)
        # REP socket forces workers to wait for acknowledgment before closing shared memory
        super().__init__(
            request.port,
            viewer_type="napari",
            host="*",
            log_file_path=request.log_file_path,
            data_socket_type=zmq.REP,
            transport_mode=coerce_transport_mode(request.transport_mode),
            config=OPENHCS_ZMQ_CONFIG,
        )

        self.viewer_title = request.viewer_title
        self.replace_layers = request.replace_layers
        self.viewer = None
        self.layer_state = NapariLayerStateStore.empty()
        self.component_groups: NapariComponentGroups = {}
        self.component_metadata: NapariComponentNameMetadata = {}

        self.component_values = NapariComponentValueTracker()
        self.display_axis_domain = NapariDisplayAxisDomain()

        # Debouncing + locking for layer updates to prevent race conditions
        import threading

        self.layer_update_lock = threading.Lock()  # Prevent concurrent updates
        self.update_delay_ms = 1000  # Wait 200ms for more items before rebuilding
        self.batch_processors = NapariBatchProcessorStore(
            debounce_delay_ms=self.update_delay_ms,
        )
        self.display_pipeline = NapariLayerDisplayPipeline(self)

        # Ack socket handled by StreamingVisualizerServer

    def display_layer_batch(
        self,
        *,
        layer_key: str,
        items: list[NapariStreamLayerItem],
        display_payload: ComponentLayout,
        component_names_metadata: NapariComponentNameMetadata,
    ) -> None:
        """Display one debounced batch through the composed display pipeline."""
        self.display_pipeline.display_layer_batch(
            layer_key=layer_key,
            items=items,
            display_payload=display_payload,
            component_names_metadata=component_names_metadata,
        )

    def _send_ack(
        self,
        image_id: str,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        """Send acknowledgment that an image was processed.

        Args:
            image_id: UUID of the processed image
            status: 'success' or 'error'
            error: Error message if status='error'
        """
        self.send_ack(image_id, status=status, error=error)

    def _create_pong_response(self) -> NapariControlReplyPayload:
        """Override to add Napari-specific fields and memory usage."""
        return NAPARI_HEARTBEAT.apply_to(super()._create_pong_response())

    def handle_control_message(
        self,
        message: NapariWirePayload,
    ) -> NapariControlReplyPayload:
        """
        Handle control messages beyond ping/pong.

        Supported message types:
        - shutdown: Graceful shutdown (closes viewer)
        - force_shutdown: Force shutdown (same as shutdown for Napari)
        - clear_state: Clear accumulated component groups (for new pipeline runs)
        """
        msg_type = PayloadMap(message, "Napari control message").optional(
            NapariWireField.TYPE
        )
        return NapariControlMessageAction.for_message_type(msg_type).handle(
            self,
            message,
        )

    def display_image(self, image_data: np.ndarray, metadata: ComponentMap) -> None:
        """Display a single image payload (best-effort helper)."""
        image_info = {
            "path": DEFAULT_DIRECT_IMAGE_PATH,
            "data": image_data,
            "shape": image_data.shape,
            "dtype": image_data.dtype,
            "metadata": metadata,
            "producer_identity": StreamProducerIdentity.direct(
                "direct_image"
            ).to_payload(),
        }
        self._process_single_image(
            image_info,
            {"component_modes": {}, "component_order": []},
        )

    def process_image_message(self, message: bytes):
        """
        Process incoming image data message and send reply for REP socket.

        Args:
            message: Raw ZMQ message containing image data
        """
        import json

        # Parse JSON message
        data = json.loads(message.decode("utf-8"))

        msg_type = PayloadMap(data, "Napari message").optional(NapariWireField.TYPE)

        NapariStreamMessageHandler.for_message_type(msg_type).handle(self, data)

        # Send reply on REP socket (required pattern)
        try:
            reply = {"status": "success", "type": msg_type}
            self.data_socket.send_json(reply)
        except Exception as e:
            logger.error(f"🔬 NAPARI PROCESS: Failed to send reply: {e}")

    def _process_single_image(
        self,
        image_info: NapariWirePayload,
        display_config_dict: NapariDisplayConfigPayload,
    ) -> None:
        """Process a single image or shapes data and display in Napari."""
        payload = NapariImagePayload.from_payload(image_info, display_config_dict)
        logger.info(
            f"🔍 NAPARI PROCESS: Received {payload.data_type} with metadata: {payload.component_metadata} (path: {payload.path})"
        )

        try:
            loaded = _NAPARI_PAYLOAD_DATA_LOADER.load(payload)
            request = NapariComponentAwareDisplayRequest.from_inputs(
                viewer=self.viewer,
                layers=self.layer_state.layers,
                component_groups=self.component_groups,
                data=loaded.data,
                path=payload.path,
                colormap=loaded.colormap,
                display_config=payload.display_config,
                replace_layers=self.replace_layers,
                component_metadata=payload.component_metadata,
                producer_identity=payload.producer_identity,
                data_type=payload.data_type,
                server=self,
            )
            _NAPARI_COMPONENT_DISPLAY_COORDINATOR.display(request)
            if payload.image_id:
                self._send_ack(payload.image_id, status=_ACK_SUCCESS)

        except Exception as e:
            logger.error(
                f"🔬 NAPARI PROCESS: Failed to process {payload.data_type} {payload.path}: {e}",
                exc_info=True,
            )
            if payload.image_id:
                self._send_ack(payload.image_id, status=_ACK_ERROR, error=str(e))
            # Don't re-raise - continue processing other messages instead of crashing


def run_napari_viewer_process_from_legacy_signature(
    port: int,
    viewer_title: str,
    replace_layers: bool = False,
    log_file_path: str = None,
    transport_mode: OpenHCSTransportMode = OpenHCSTransportMode.IPC,
):
    """
    Napari viewer process entry point. Runs in a separate process.
    Listens for ZeroMQ messages with image data to display.

    Args:
        port: ZMQ port to listen on
        viewer_title: Title for the napari viewer window
        replace_layers: If True, replace existing layers; if False, add new layers with unique names
        log_file_path: Path to log file (for client discovery via ping/pong)
        transport_mode: ZMQ transport mode (IPC or TCP)
    """
    try:
        import zmq
        import napari

        request = NapariViewerServerRequest.from_legacy_signature(
            port,
            viewer_title,
            replace_layers,
            log_file_path,
            transport_mode,
        )

        # Create ZMQ server instance (inherits from ZMQServer ABC)
        server = NapariViewerServer(request)

        # Start the server (binds sockets)
        server.start()

        # Create napari viewer in this process (main thread)
        viewer = napari.Viewer(title=viewer_title, show=True)
        server.viewer = viewer

        # Initialize layers dictionary with existing layers (for reconnection scenarios)
        for layer in viewer.layers:
            server.layer_state.set_layer(layer.name, layer)

        # Enable text overlay for dimension labels
        viewer.text_overlay.visible = True
        viewer.text_overlay.color = "white"
        viewer.text_overlay.font_size = 14

        logger.info(
            f"🔬 NAPARI PROCESS: Viewer started on data port {port}, control port {server.control_port}"
        )

        # Add cleanup handler for when viewer is closed
        def cleanup_and_exit():
            logger.info("🔬 NAPARI PROCESS: Viewer closed, cleaning up and exiting...")
            try:
                server.stop()
            except:
                pass
            sys.exit(0)

        # Connect the viewer close event to cleanup
        viewer.window.qt_viewer.destroyed.connect(cleanup_and_exit)

        # Use proper Qt event loop integration
        import sys
        from qtpy import QtWidgets, QtCore

        ViewerQtEnvironmentPolicy().apply_to(os.environ)

        # Get the Qt application
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        # Ensure the application DOES quit when the napari window closes
        app.setQuitOnLastWindowClosed(True)

        # Set up a QTimer for message processing
        timer = QtCore.QTimer()

        def process_messages():
            # Process control messages (ping/pong handled by ABC)
            server.process_messages()

            # Process data messages (images) if ready
            # REP socket requires recv->send alternation, so process one at a time
            if server._ready:
                try:
                    message = server.data_socket.recv(zmq.NOBLOCK)
                    server.process_image_message(message)
                except zmq.Again:
                    # No message available
                    pass

        # Connect timer to message processing
        timer.timeout.connect(process_messages)
        timer.start(50)  # Process messages every 50ms

        logger.info("🔬 NAPARI PROCESS: Starting Qt event loop")

        # Run the Qt event loop - this keeps napari responsive
        app.exec_()

    except Exception as e:
        logger.error(f"🔬 NAPARI PROCESS: Fatal error: {e}")
    finally:
        logger.info("🔬 NAPARI PROCESS: Shutting down")
        if "server" in locals():
            server.stop()
