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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from typing import ClassVar, Optional, Sequence, TypeAlias
from qtpy.QtCore import QTimer

from openhcs.core.config import TransportMode as OpenHCSTransportMode
from openhcs.core.aligned_image_payload import project_singleton_stack_image_domain
from openhcs.core.image_shapes import ArrayShape
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from metaclass_registry import AutoRegisterMeta
from polystore.backend_registry import register_cleanup_callback
from zmqruntime.config import TransportMode, ZMQConfig
from zmqruntime.viewer_protocol import ViewerComponentMode, ViewerWireField
from polystore.streaming_constants import StreamingDataType
from polystore.streaming.identity import (
    FixedStreamProducerIdentityKind,
    StreamProducerDisplayNameAuthority,
    StreamProducerIdentity,
    StreamRouteKeyAuthority,
)
from polystore.streaming.receivers.napari import build_route_key
from openhcs.runtime.viewer_protocol import (
    ChannelColormapPolicy,
    NAPARI_HEARTBEAT,
    NapariLayerKind,
    NapariViewerServerRequest,
    ViewerBatchMessageType,
    ViewerBatchWireField,
    ViewerControlMessageType,
    ViewerPayloadControlOptions,
    ViewerControlResponseField,
    ViewerControlReplyHeader,
    ViewerControlReplyPayload,
    ViewerQtEnvironmentPolicy,
    ViewerComponentValueOrdering,
    ViewerProtocolStatus,
)
from openhcs.runtime.napari_streaming_handlers import (
    LayerData,
    LayerKwargValue,
    NapariLayerHandle,
    NapariAxisPresentation,
    NapariAggregateAxisBindingSet,
    NapariAggregateAxisBindingAuthority,
    NapariLayerBatchDebouncePolicy,
    NapariBatchProcessorStore,
    NapariComponentGroupStore,
    NapariDimensionLayerState,
    NapariImagePayloadAxisLabelPolicy,
    NapariImageLayerPresentationPolicy,
    NapariLayerUpdateAuthority,
    NapariPendingLayerUpdate,
    NapariLayerRouteStateStore,
    NapariStreamLayerAddress,
    NapariStreamLayerItem,
    NapariShapeLabelRasterizer,
    NapariViewerLayerCreator,
)
from openhcs.runtime.viewer_component_system import (
    ComponentMap,
    ComponentValue,
    ComponentValues,
    ViewerComponentCoordinateAuthority,
    ViewerComponentAxisSemantics,
    ViewerComponentAxisSemanticsAuthority,
    ViewerComponentLayout,
    ViewerComponentMetadataPayload,
    ViewerComponentMetadataNormalizer,
    ViewerComponentNameMetadata,
    ViewerComponentSemanticRole,
    ViewerBatchPayloadFields,
    ViewerDisplayBatchContext,
    ViewerDisplayAxisDomain,
    ViewerMappingDisplayConfigInput,
    ViewerLayerAxisProjection,
    ViewerLayerAxisProjectionRequest,
    ViewerLayerAxisProjector,
    ViewerRouteComponentValueTracker,
    ViewerStreamingDataTypeHandler,
    ViewerStreamingDataTypeHandlerMeta,
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

NapariWireValue: TypeAlias = (
    str | int | float | bool | np.ndarray | np.dtype | tuple | list | dict | None
)
NapariBatchImagePayloads: TypeAlias = list[Mapping[str, NapariWireValue]]
NapariComponentModeMap: TypeAlias = dict[str, str]
NapariComponentGroups: TypeAlias = dict[str, list[NapariStreamLayerItem]]


class NapariWireField(str, Enum):
    """Wire keys used by Napari stream and ROI payloads."""

    CENTER = "center"
    COORDINATES = "coordinates"
    METADATA = "metadata"
    RADII = "radii"
    SNAPSHOT = "snapshot"


class VisualMetadataField(str, Enum):
    """Optional visual metadata fields attached to ROI payloads."""

    CENTROID = "centroid"
    LABEL = "label"
    AREA = "area"
    COMPONENT = "component"


@dataclass(frozen=True)
class PayloadMap:
    """Typed accessor for one JSON-derived payload mapping."""

    payload: Mapping[str, NapariWireValue]
    context: str

    def required(
        self,
        field: NapariWireField | ViewerBatchWireField,
    ) -> NapariWireValue:
        if field.value not in self.payload:
            raise ValueError(
                f"{self.context} missing required payload field '{field.value}'"
            )
        return self.payload[field.value]

    def optional(
        self,
        field: NapariWireField | ViewerBatchWireField,
    ) -> NapariWireValue | None:
        if field.value in self.payload:
            return self.payload[field.value]
        return None

    def required_mapping(
        self,
        field: NapariWireField | ViewerBatchWireField,
    ) -> Mapping[str, NapariWireValue]:
        return self._mapping(field, self.required(field))

    def optional_mapping(
        self,
        field: NapariWireField | ViewerBatchWireField,
    ) -> Mapping[str, NapariWireValue]:
        if field.value not in self.payload:
            return {}
        return self._mapping(field, self.payload[field.value])

    def _mapping(
        self,
        field: NapariWireField | ViewerBatchWireField,
        value: NapariWireValue,
    ) -> Mapping[str, NapariWireValue]:
        if not isinstance(value, dict):
            raise TypeError(
                f"{self.context} field '{field.value}' must be a dict, "
                f"got {type(value).__name__}"
            )
        return value


def _napari_wire_str_mapping(
    payload: Mapping[str, NapariWireValue],
    field: NapariWireField,
    context: str,
) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(
                f"{context} field '{field.value}' must use string keys, "
                f"got {type(key).__name__}"
            )
        if not isinstance(value, str):
            raise TypeError(
                f"{context} field '{field.value}' values must be strings, "
                f"got {type(value).__name__}"
            )
        result[key] = value
    return result


@dataclass(frozen=True)
class ShapePayload:
    """Typed view of one Napari ROI shape payload."""

    payload: Mapping[str, NapariWireValue]

    @property
    def shape_type(self) -> str:
        return str(PayloadMap(self.payload, "Napari shape payload").required(
            ViewerBatchWireField.TYPE
        ))

    @property
    def coordinates(self) -> LayerData:
        return PayloadMap(self.payload, "Napari shape payload").required(
            NapariWireField.COORDINATES
        )

    @property
    def metadata(self) -> "VisualMetadata":
        return VisualMetadata(
            PayloadMap(self.payload, "Napari shape payload").optional_mapping(
                ViewerWireField.METADATA
            )
        )


@dataclass(frozen=True)
class VisualMetadata:
    """Optional display metadata attached to one shape payload."""

    metadata: Mapping[str, NapariWireValue]

    def value(
        self,
        field: VisualMetadataField,
        default: NapariWireValue,
    ) -> NapariWireValue:
        if field.value in self.metadata:
            return self.metadata[field.value]
        return default


@dataclass(frozen=True)
class NapariBatchPayload(ViewerDisplayBatchContext[Mapping[str, NapariWireValue]]):
    """Typed view of an incoming batch message."""

    images: NapariBatchImagePayloads
    msg_type: ViewerBatchMessageType

    @classmethod
    def from_json_payload(
        cls,
        data: Mapping[str, NapariWireValue],
    ) -> "NapariBatchPayload":
        fields = ViewerBatchPayloadFields(data, "Napari batch message")
        display_config = fields.required_mapping(ViewerBatchWireField.DISPLAY_CONFIG)
        component_axis_semantics = fields.component_axis_semantics(
            ViewerMappingDisplayConfigInput(display_config),
            context="Napari component value domain",
        )
        component_names_metadata = fields.optional_component_names_metadata(
            context="Napari component-name metadata",
        )
        return cls(
            msg_type=fields.message_type(),
            images=fields.required_mapping_items(ViewerBatchWireField.IMAGES),
            viewer_display_config=display_config,
            store=component_names_metadata.store,
            entries=component_axis_semantics.entries,
            layout=component_axis_semantics.layout,
        )


@dataclass(frozen=True)
class NapariStreamMessageReply:
    """Reply sent on the Napari REP socket for one stream message."""

    status: ViewerProtocolStatus
    msg_type: ViewerBatchMessageType | None
    error: str | None = None

    @classmethod
    def success(
        cls,
        msg_type: ViewerBatchMessageType | None,
    ) -> "NapariStreamMessageReply":
        return cls(ViewerProtocolStatus.SUCCESS, msg_type)

    @classmethod
    def failure(
        cls,
        msg_type: ViewerBatchMessageType | None,
        error: str,
    ) -> "NapariStreamMessageReply":
        return cls(ViewerProtocolStatus.ERROR, msg_type, error)

    def to_wire_mapping(self) -> dict[str, NapariWireValue]:
        reply: dict[str, NapariWireValue] = {
            ViewerControlResponseField.STATUS.value: self.status.value,
            ViewerBatchWireField.TYPE.value: (
                self.msg_type.value if self.msg_type is not None else None
            ),
        }
        if self.error is not None:
            reply[ViewerControlResponseField.MESSAGE.value] = self.error
        return reply


@dataclass(frozen=True)
class NapariStreamLayerContext(ViewerComponentAxisSemantics):
    """Wire-derived routing facts for one streamed Napari payload."""

    producer: StreamProducerIdentity
    address: NapariStreamLayerAddress

    @classmethod
    def from_payload_map(
        cls,
        payload: PayloadMap,
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> "NapariStreamLayerContext":
        data_type_value = payload.optional(ViewerWireField.DATA_TYPE)
        if data_type_value is None:
            data_type_value = DEFAULT_IMAGE_DATA_TYPE
        return cls(
            entries=layer_axis_projection_semantics.entries,
            layout=layer_axis_projection_semantics.layout,
            producer=StreamProducerIdentity.from_payload(
                payload.required(ViewerWireField.PRODUCER_IDENTITY)
            ),
            address=NapariStreamLayerAddress(
                components=ViewerComponentMetadataPayload.component_map(
                    payload.optional_mapping(ViewerWireField.METADATA),
                    context="Napari image component metadata",
                ),
                path=str(payload.required(ViewerWireField.PATH)),
                stream_layer_data_type=StreamingDataType(
                    str(data_type_value)
                ),
            ),
        )

    def layer_route(
        self,
        *,
        payload_shape_role: "NapariImagePayloadShapeRole | None",
        layer_route_state: "NapariLayerRouteStateStore",
    ) -> "NapariLayerRoute":
        component_info = _COMPONENT_METADATA_NORMALIZER.normalize(
            self.address.components
        )
        component_layout = self.layout
        base_layer_key = build_route_key(
            producer_identity=self.producer,
            component_info=component_info,
            display_layout=component_layout,
            data_type=self.address.stream_layer_data_type,
        )
        layer_key = (
            base_layer_key
            if payload_shape_role is None
            else payload_shape_role.route_key(base_layer_key)
        )
        layer_title = NapariLayerTitleAuthority.disambiguate(
            title=NapariLayerTitleAuthority.title(
                producer=self.producer,
                stream_layer_data_type=self.address.stream_layer_data_type,
                component_info=component_info,
                component_layout=component_layout,
                payload_shape_role=payload_shape_role,
            ),
            producer=self.producer,
            route_key=layer_key,
            layer_route_state=layer_route_state,
        )
        return NapariLayerRoute(
            entries=self.entries,
            layout=self.layout,
            route_key=layer_key,
            layer_title=layer_title,
            component_info=component_info,
            item_address=self.address.with_components(component_info),
            payload_shape_role=payload_shape_role,
        )


@dataclass(frozen=True)
class NapariImagePayload(NapariStreamLayerContext):
    """Typed view of one image/shapes message."""

    raw: Mapping[str, NapariWireValue]
    image_id: str | None

    @classmethod
    def from_payload(
        cls,
        image_info: Mapping[str, NapariWireValue],
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> "NapariImagePayload":
        payload = PayloadMap(image_info, "Napari image message")
        stream_layer_context = NapariStreamLayerContext.from_payload_map(
            payload,
            layer_axis_projection_semantics,
        )
        image_id = payload.optional(ViewerWireField.IMAGE_ID)
        return cls(
            raw=image_info,
            image_id=str(image_id) if image_id is not None else None,
            entries=stream_layer_context.entries,
            layout=stream_layer_context.layout,
            producer=stream_layer_context.producer,
            address=stream_layer_context.address,
        )

    @property
    def shapes(self) -> LayerData:
        return PayloadMap(self.raw, "Napari shapes/points message").required(
            ViewerWireField.SHAPES
        )

    @property
    def image_shape(self) -> tuple[int, ...]:
        shape = PayloadMap(self.raw, "Napari image message").required(
            ViewerWireField.SHAPE
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
            ViewerWireField.DTYPE
        )

    @property
    def shm_name(self) -> str | None:
        shm_name = PayloadMap(self.raw, "Napari image message").optional(
            ViewerWireField.SHM_NAME
        )
        if shm_name is None:
            return None
        return str(shm_name)

    @property
    def direct_data(self) -> LayerData:
        return PayloadMap(self.raw, "Napari image message").optional(
            ViewerWireField.DATA
        )

_NAPARI_SHAPE_RASTERIZER = NapariShapeLabelRasterizer()
_COMPONENT_METADATA_NORMALIZER = ViewerComponentMetadataNormalizer()
_ACK_ERROR = ViewerProtocolStatus.ERROR.value
_ACK_SUCCESS = ViewerProtocolStatus.SUCCESS.value


class NapariImagePayloadShapeRole(str, Enum):
    """Nominal display role for image payloads that cannot share one layer."""

    SCALAR_PLANE = "scalar_plane"
    SCALAR_STACK = "scalar_stack"
    COLOR_PLANE = "color_plane"
    COLOR_STACK = "color_stack"
    GENERIC_ND = "generic_nd"

    @classmethod
    def for_stream_layer_context(
        cls,
        stream_layer_context: NapariStreamLayerContext,
        data: LayerData,
    ) -> "NapariImagePayloadShapeRole | None":
        if stream_layer_context.address.stream_layer_data_type is not StreamingDataType.IMAGE:
            return None

        array_shape = ArrayShape.from_value(data)
        if array_shape is None:
            return cls.GENERIC_ND
        if array_shape.has_rank(2):
            return cls.SCALAR_PLANE
        if array_shape.ndim >= 3 and array_shape.has_channel_last():
            if array_shape.has_rank(3):
                return cls.COLOR_PLANE
            return cls.COLOR_STACK
        if array_shape.has_rank(3):
            return cls.SCALAR_STACK
        return cls.GENERIC_ND

    @property
    def route_suffix(self) -> str:
        return self.value

    @property
    def title_suffix(self) -> str:
        return {
            self.SCALAR_PLANE: "",
            self.SCALAR_STACK: "image stack",
            self.COLOR_PLANE: "RGB",
            self.COLOR_STACK: "RGB stack",
            self.GENERIC_ND: "nD image",
        }[self]

    @property
    def is_default_route(self) -> bool:
        return self is self.SCALAR_PLANE

    def route_key(self, base_route_key: str) -> str:
        if self.is_default_route:
            return base_route_key
        return StreamRouteKeyAuthority.join([base_route_key, self.route_suffix])

    def title(self, base_title: str) -> str:
        if not self.title_suffix:
            return base_title
        return f"{base_title} {self.title_suffix}"


class NapariPayloadDataLoader:
    """Load payload data by streaming data kind."""

    SHAPE_LIKE_DATA_TYPES = frozenset(
        {StreamingDataType.SHAPES, StreamingDataType.POINTS}
    )

    def load(self, payload: NapariImagePayload) -> LayerData:
        data_type = payload.address.stream_layer_data_type
        if data_type in self.SHAPE_LIKE_DATA_TYPES:
            return payload.shapes
        return self._image_data(payload)

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
class NapariLayerRoute(ViewerComponentAxisSemantics):
    """Resolved route identity for one component-aware display item."""

    route_key: str
    layer_title: str
    component_info: ComponentMap
    item_address: NapariStreamLayerAddress
    payload_shape_role: NapariImagePayloadShapeRole | None = None


class NapariComponentAwareDisplayCoordinator:
    """Route loaded payload data into debounced Napari layer updates."""

    def display(
        self,
        *,
        data: LayerData,
        stream_layer_context: NapariStreamLayerContext,
        server: "NapariViewerServer",
    ) -> None:
        if not stream_layer_context.address.components:
            raise ValueError(
                "No component metadata available for path: "
                f"{stream_layer_context.address.path}"
            )
        routed_data = self._data_for_routing(data, stream_layer_context)
        route = self._route(
            data=routed_data,
            stream_layer_context=stream_layer_context,
            server=server,
        )
        self._log_route(route)
        self._reconcile_deleted_layer(server, route.route_key)
        group = self._group_for(server, route.route_key)
        self._clear_group_for_replace(server, route.route_key, group)
        self._upsert_item(
            data=routed_data,
            route=route,
            group=group,
        )
        logger.info(
            "🔬 NAPARI PROCESS: Scheduling debounced update for %s (data_type=%s)",
            route.route_key,
            route.item_address.stream_layer_data_type,
        )
        server.display_pipeline.schedule_layer_update(
            route.route_key,
            route.item_address.stream_layer_data_type,
            route,
        )

    @staticmethod
    def _data_for_routing(
        data: LayerData,
        stream_layer_context: NapariStreamLayerContext,
    ) -> LayerData:
        if stream_layer_context.address.stream_layer_data_type is StreamingDataType.IMAGE:
            return project_singleton_stack_image_domain(data)
        return data

    @staticmethod
    def _route(
        *,
        data: LayerData,
        stream_layer_context: NapariStreamLayerContext,
        server: "NapariViewerServer",
    ) -> NapariLayerRoute:
        payload_shape_role = NapariImagePayloadShapeRole.for_stream_layer_context(
            stream_layer_context,
            data,
        )
        route = stream_layer_context.layer_route(
            payload_shape_role=payload_shape_role,
            layer_route_state=server.layer_route_state,
        )
        server.layer_route_state.set_title(route.route_key, route.layer_title)
        return route

    @staticmethod
    def _log_route(
        route: NapariLayerRoute,
    ) -> None:
        logger.info(
            "🔍 NAPARI PROCESS: component_modes=%s, shape_role=%s, layer_key='%s'",
            route.layout.component_modes,
            route.payload_shape_role,
            route.route_key,
        )
        logger.info(
            "🔍 NAPARI PROCESS: layer_key='%s', component_info=%s",
            route.route_key,
            route.component_info,
        )

    @staticmethod
    def _reconcile_deleted_layer(
        server: "NapariViewerServer",
        layer_key: str,
    ) -> None:
        if (
            layer_key in server.layer_route_state.layers
            and server.layer_route_state.layers[layer_key] not in server.viewer.layers
        ):
            num_items = server.component_groups.item_count(layer_key)
            server.layer_route_state.purge_route(layer_key)
            server.component_groups.purge(layer_key)
            logger.info(
                "🔬 NAPARI PROCESS: Reconciling state — '%s' was deleted from viewer; "
                "purged stale caches (had %d items in component_groups)",
                layer_key,
                num_items,
            )

    @staticmethod
    def _group_for(
        server: "NapariViewerServer",
        layer_key: str,
    ) -> list[NapariStreamLayerItem]:
        return server.component_groups.items_for(layer_key)

    @staticmethod
    def _clear_group_for_replace(
        server: "NapariViewerServer",
        layer_key: str,
        group: list[NapariStreamLayerItem],
    ) -> None:
        if server.replace_layers and group:
            logger.info(
                "🔬 NAPARI PROCESS: replace_layers=True, clearing %d existing items "
                "from layer '%s'",
                len(group),
                layer_key,
            )
            group.clear()

    def _upsert_item(
        self,
        *,
        data: LayerData,
        route: NapariLayerRoute,
        group: list[NapariStreamLayerItem],
    ) -> None:
        new_item = NapariStreamLayerItem(
            data=data,
            address=route.item_address,
        )
        existing_index = None
        for index, item in enumerate(group):
            if item.address.same_layer_slot(route.item_address):
                existing_index = index
                break
        if existing_index is None:
            group.append(new_item)
            logger.info(
                "🔬 NAPARI PROCESS: Added %s to component_groups[%s], now has %d items",
                route.item_address.stream_layer_data_type,
                route.route_key,
                len(group),
            )
            return

        old_data_type = group[existing_index].address.stream_layer_data_type
        group[existing_index] = new_item
        logger.info(
            "🔬 NAPARI PROCESS: Replaced %s item in component_groups[%s] at index %d, "
            "total items: %d",
            old_data_type,
            route.route_key,
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


def _build_nd_points(
    layer_items: list[NapariStreamLayerItem],
    axis_projection: ViewerLayerAxisProjection,
):
    """
    Build nD points by prepending stack component indices to 2D point coordinates.

    Args:
        layer_items: List of items with 'data' (list of point coordinate arrays) and 'components'
        axis_projection: Projected component axes and their value domains.

    Returns:
        Tuple of (all_points_nd, all_properties)
    """
    all_points_nd = []
    all_properties = {"label": [], "component": []}

    for item in layer_items:
        points_data = item.data
        prepend_dims = list(
            axis_projection.coordinate_index(
                item.address.components,
                context="Napari points item",
            )
        )

        for shape_dict in points_data:
            shape_payload = ShapePayload(shape_dict)
            if shape_payload.shape_type != "points":
                continue

            coordinates = shape_payload.coordinates
            metadata = shape_payload.metadata

            for coord in coordinates:
                nd_coord = prepend_dims + list(coord)
                all_points_nd.append(nd_coord)

                all_properties["label"].append(
                    metadata.value(VisualMetadataField.LABEL, "")
                )
                all_properties["component"].append(
                    metadata.value(VisualMetadataField.COMPONENT, 0)
                )

    points_array = np.empty((0, 2 + len(axis_projection.projected_axis_components)))
    if all_points_nd:
        points_array = np.array(all_points_nd)
    return points_array, all_properties


def _build_nd_image_array(
    layer_items: list[NapariStreamLayerItem],
    axis_projection: ViewerLayerAxisProjection,
    aggregate_axis_bindings: NapariAggregateAxisBindingSet | None = None,
):
    """
    Build nD image array by stacking images along stack component dimensions.

    Args:
        layer_items: List of items with 'data' (image arrays) and 'components'
        axis_projection: Projected component axes and their value domains.

    Returns:
        np.ndarray: Stacked image array
    """
    projected_axis_components = axis_projection.projected_axis_components
    component_values = axis_projection.component_values
    if aggregate_axis_bindings is None:
        aggregate_axis_bindings = NapariAggregateAxisBindingSet()
    logger.info(
        f"🔬 NAPARI PROCESS: Building nD array with axis_components={projected_axis_components}, component_values={component_values}"
    )

    first_img = layer_items[0].data
    payload_shape = tuple(int(axis) for axis in first_img.shape)
    residual_payload_shape = tuple(
        extent
        for index, extent in enumerate(payload_shape)
        if index not in aggregate_axis_bindings.payload_axes
    )
    stack_shape = axis_projection.axis_shape() + residual_payload_shape
    stacked_array = np.zeros(stack_shape, dtype=first_img.dtype)
    occupied_indices: set[tuple[int, ...]] = set()
    logger.info(
        f"🔬 NAPARI PROCESS: Created nD array with shape {stack_shape} from {len(layer_items)} items"
    )

    for img in layer_items:
        for payload_indices in product(
            *(range(binding.extent) for binding in aggregate_axis_bindings.bindings)
        ):
            components = aggregate_axis_bindings.item_component_values(
                img,
                payload_indices,
            )
            indices = axis_projection.coordinate_index(
                components,
                context="Napari image item",
            )
            if indices in occupied_indices:
                raise ValueError(
                    "Duplicate Napari image item for projected coordinate "
                    f"{indices!r} on axes {projected_axis_components!r}."
                )
            occupied_indices.add(indices)
            logger.debug(
                "🔬 NAPARI PROCESS: Placing image at indices %s, components=%s",
                indices,
                components,
            )
            if payload_indices:
                stacked_array[indices] = img.data[payload_indices]
            else:
                stacked_array[indices] = img.data

    missing_indices = axis_projection.expected_indices() - occupied_indices
    if missing_indices:
        raise ValueError(
            "Napari image stack missing routed image(s) for projected coordinate(s) "
            f"{sorted(missing_indices)!r} on axes {projected_axis_components!r}."
        )
    logger.info(
        "🔬 NAPARI PROCESS: Image route coverage complete for axes=%s positions=%d gaps=%d",
        projected_axis_components,
        len(occupied_indices),
        len(missing_indices),
    )
    return stacked_array


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
        stream_layer_data_type: StreamingDataType,
        component_info: ComponentMap,
        component_layout: ViewerComponentLayout,
        payload_shape_role: NapariImagePayloadShapeRole | None = None,
    ) -> str:
        parts = [StreamProducerDisplayNameAuthority.output_label(producer)]
        for component in component_layout.components_for_mode(ViewerComponentMode.SLICE):
            value = ViewerComponentCoordinateAuthority.required_value(
                component_info,
                component,
                context="Napari layer title",
            )
            parts.append(f"{component} {value}")
        suffix = cls.DATA_TYPE_SUFFIX[stream_layer_data_type]
        if suffix:
            parts.append(suffix)
        title = " ".join(str(part) for part in parts if part)
        if payload_shape_role is None:
            return title
        return payload_shape_role.title(title)

    @staticmethod
    def disambiguate(
        *,
        title: str,
        producer: StreamProducerIdentity,
        route_key: str,
        layer_route_state: NapariLayerRouteStateStore,
    ) -> str:
        if not layer_route_state.title_collides(route_key, title):
            return title
        return f"{title} [{StreamProducerDisplayNameAuthority.disambiguation_label(producer)}]"


class NapariDimensionLabelRouteSource(str, Enum):
    """Provenance for the dimension-label route currently applied to Napari."""

    SELECTED_OPENHCS_LAYER = "selected_openhcs_layer"
    UPDATED_OPENHCS_LAYER = "updated_openhcs_layer"
    ACTIVE_STREAM_ROUTE = "active_stream_layer_context"
    COMPATIBLE_STREAM_ROUTE = "compatible_stream_layer_context"
    ACTIVE_NON_OPENHCS_LAYER = "active_non_openhcs_layer"
    MISSING = "missing"


@dataclass(frozen=True)
class NapariDimensionLabelRouteResolution:
    """Resolved dimension-label route plus the reason it was selected."""

    route_key: str | None
    source: NapariDimensionLabelRouteSource


class NapariDimensionLabelStore:
    """Store semantic axis labels for one streamed layer route."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server

    def apply(
        self,
        presentation: NapariAxisPresentation,
    ) -> tuple[str, ...] | None:
        self._validate_payload_axis_labels(presentation.payload_axis_labels)
        axis_projection = presentation.projection
        self._validate_axis_offsets(axis_projection.axis_offsets)
        axis_labels = presentation.axis_labels
        scalar_labels = self._scalar_labels(
            axis_projection.scalar_component_values,
            self.server.component_name_metadata,
        )
        if not axis_projection.projected_axis_components:
            self.server.layer_route_state.set_dimension_state(
                presentation.route_key,
                NapariDimensionLayerState(
                    labels={},
                    scalar_labels=scalar_labels,
                    presentation=presentation,
                ),
            )
            return axis_labels

        logger.info(
            "🔬 NAPARI PROCESS: Built axis_labels=%s for projected_axis_components=%s",
            axis_labels,
            axis_projection.projected_axis_components,
        )
        self.server.layer_route_state.set_dimension_state(
            presentation.route_key,
            NapariDimensionLayerState(
                labels=self._dimension_labels(
                    axis_projection.projected_axis_components,
                    axis_projection.component_values,
                    self.server.component_name_metadata,
                ),
                scalar_labels=scalar_labels,
                presentation=presentation,
            ),
        )
        return axis_labels

    @staticmethod
    def _validate_payload_axis_labels(payload_axis_labels: tuple[str, ...]) -> None:
        invalid_labels = [
            label for label in payload_axis_labels if not isinstance(label, str)
        ]
        if invalid_labels:
            raise TypeError(
                "payload_axis_labels must contain semantic axis-name strings; "
                f"got {invalid_labels!r}."
            )

    @staticmethod
    def _validate_axis_offsets(axis_offsets: tuple[int, ...]) -> None:
        invalid_offsets = [
            offset for offset in axis_offsets if not isinstance(offset, int)
        ]
        if invalid_offsets:
            raise TypeError(
                "axis_offsets must contain integer coordinate offsets; "
                f"got {invalid_offsets!r}."
            )

    def _dimension_labels(
        self,
        active_projected_axis_components: tuple[str, ...],
        active_component_values: ComponentValues,
        component_names_metadata: ViewerComponentNameMetadata,
    ) -> DimensionLabelMap:
        dimension_labels = {}
        for component in active_projected_axis_components:
            values = active_component_values[component]
            dimension_labels[component] = component_names_metadata.axis_labels(
                component,
                values,
            )
        return dimension_labels

    def _scalar_labels(
        self,
        scalar_component_values: ComponentValues,
        component_names_metadata: ViewerComponentNameMetadata,
    ) -> tuple[str, ...]:
        labels = []
        for component, values in scalar_component_values.items():
            if len(values) != 1:
                raise ValueError(
                    f"Collapsed component {component!r} must have one value, "
                    f"got {values!r}."
                )
            value = values[0]
            if component_names_metadata.display_name(component, value) is None:
                continue
            labels.append(component_names_metadata.axis_label(component, value))
        return tuple(labels)


class NapariDimensionLabelRouteResolver:
    """Resolve which route currently owns Napari dimension labels."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server

    def resolve(
        self,
        updated_route_key: str | None = None,
    ) -> NapariDimensionLabelRouteResolution:
        viewer_ndim = self._viewer_ndim()
        current_step = self._viewer_current_step()
        active_layer = self.server.viewer.layers.selection.active
        if active_layer is not None:
            route_key = self._route_key_for_layer(active_layer)
            if route_key is None:
                return NapariDimensionLabelRouteResolution(
                    route_key=None,
                    source=NapariDimensionLabelRouteSource.ACTIVE_NON_OPENHCS_LAYER,
                )
            if self._route_matches_viewer_context(route_key, viewer_ndim, current_step):
                return NapariDimensionLabelRouteResolution(
                    route_key=route_key,
                    source=NapariDimensionLabelRouteSource.SELECTED_OPENHCS_LAYER,
                )

        if self._route_matches_viewer_context(
            updated_route_key,
            viewer_ndim,
            current_step,
        ):
            return NapariDimensionLabelRouteResolution(
                route_key=updated_route_key,
                source=NapariDimensionLabelRouteSource.UPDATED_OPENHCS_LAYER,
            )

        route_key = self.server.layer_route_state.active_dimension_label_route
        if self._route_matches_viewer_context(route_key, viewer_ndim, current_step):
            return NapariDimensionLabelRouteResolution(
                route_key=route_key,
                source=NapariDimensionLabelRouteSource.ACTIVE_STREAM_ROUTE,
            )

        route_key = self._latest_compatible_route(viewer_ndim, current_step)
        if route_key is not None:
            return NapariDimensionLabelRouteResolution(
                route_key=route_key,
                source=NapariDimensionLabelRouteSource.COMPATIBLE_STREAM_ROUTE,
            )

        return NapariDimensionLabelRouteResolution(
            route_key=None,
            source=NapariDimensionLabelRouteSource.MISSING,
        )

    def _route_key_for_layer(self, active_layer: NapariLayerHandle) -> str | None:
        for layer_key, layer in self.server.layer_route_state.layers.items():
            if layer is active_layer:
                return layer_key
        return None

    def _route_matches_viewer_context(
        self,
        route_key: str | None,
        viewer_ndim: int,
        current_step: tuple[int, ...],
    ) -> bool:
        if route_key is None:
            return False
        state = self.server.layer_route_state.dimension_state_for(route_key)
        axis_labels = state.axis_labels
        return (
            bool(axis_labels)
            and len(axis_labels) == viewer_ndim
            and state.describes_current_step(current_step)
        )

    def _latest_compatible_route(
        self,
        viewer_ndim: int,
        current_step: tuple[int, ...],
    ) -> str | None:
        route_keys = tuple(self.server.layer_route_state.layer_dimension_states)
        for route_key in reversed(route_keys):
            if self._route_matches_viewer_context(
                route_key,
                viewer_ndim,
                current_step,
            ):
                return route_key
        return None

    def _viewer_ndim(self) -> int:
        ndim = self.server.viewer.dims.ndim
        try:
            return int(ndim)
        except (TypeError, ValueError):
            raise TypeError(f"Napari viewer dims.ndim must be int-like, got {ndim!r}.")

    def _viewer_current_step(self) -> tuple[int, ...]:
        return tuple(int(step) for step in self.server.viewer.dims.current_step)


class NapariDimensionLabelOverlayController:
    """Apply resolved dimension labels and overlay text to the viewer."""

    def __init__(
        self,
        server: "NapariViewerServer",
        route_resolver: NapariDimensionLabelRouteResolver,
    ) -> None:
        self.server = server
        self.route_resolver = route_resolver
        self._dimension_label_handler_connected = False
        self._layer_selection_handler_connected = False

    def setup_for_layer(self, layer_key: str) -> None:
        if not self.server.viewer:
            return

        if not self.server.layer_route_state.dimension_state_for(layer_key).axis_labels:
            return

        self._connect_handlers()
        resolution = self.route_resolver.resolve(layer_key)
        if resolution.source is NapariDimensionLabelRouteSource.MISSING:
            return
        if resolution.route_key is not None:
            self.server.layer_route_state.set_active_dimension_label_route(
                resolution.route_key
            )
        self._update_overlay()
        logger.info(
            "🔬 NAPARI PROCESS: Active dimension label route resolved to %s (%s)",
            resolution.route_key,
            resolution.source.value,
        )

    def _connect_handlers(self) -> None:
        self._connect_dimension_label_handler()
        self._connect_layer_selection_handler()

    def _connect_dimension_label_handler(self) -> None:
        if self._dimension_label_handler_connected:
            return

        try:
            current_step_event = self.server.viewer.dims.events.current_step
            current_step_event.connect(self._update_overlay)
            self._dimension_label_handler_connected = True
        except Exception as e:
            logger.warning(
                f"🔬 NAPARI PROCESS: Failed to setup dimension label handler: {e}"
            )

    def _connect_layer_selection_handler(self) -> None:
        if self._layer_selection_handler_connected:
            return

        try:
            active_layer_event = self.server.viewer.layers.selection.events.active
            active_layer_event.connect(self._update_overlay)
            self._layer_selection_handler_connected = True
        except Exception as e:
            logger.debug(
                f"🔬 NAPARI PROCESS: Failed to setup active-layer label handler: {e}"
            )

    def _update_overlay(self, event=None) -> None:
        try:
            resolution = self.route_resolver.resolve()
            route_key = resolution.route_key
            overlay_text = ""
            if route_key is not None:
                self.server.layer_route_state.set_active_dimension_label_route(route_key)
                state = self.server.layer_route_state.dimension_state_for(route_key)
                self._apply_axis_labels(route_key, state)
                overlay_text = self._dimension_label_text(state)
            self.server.viewer.text_overlay.text = overlay_text
        except Exception as e:
            logger.debug(f"🔬 NAPARI PROCESS: Error updating dimension label: {e}")

    def _apply_axis_labels(
        self,
        layer_key: str,
        state: NapariDimensionLayerState,
    ) -> None:
        axis_labels = state.axis_labels
        if not axis_labels:
            return
        ndim = self._viewer_ndim()
        if len(axis_labels) != ndim:
            logger.warning(
                "🔬 NAPARI PROCESS: Refusing axis_labels=%s for route %s because "
                "active viewer ndim is %s.",
                axis_labels,
                layer_key,
                ndim,
            )
            return
        self.server.viewer.dims.axis_labels = axis_labels

    def _viewer_ndim(self) -> int:
        ndim = self.server.viewer.dims.ndim
        try:
            return int(ndim)
        except (TypeError, ValueError):
            raise TypeError(f"Napari viewer dims.ndim must be int-like, got {ndim!r}.")

    def _dimension_label_text(self, state: NapariDimensionLayerState) -> str:
        current_step = tuple(int(step) for step in self.server.viewer.dims.current_step)
        label_parts = state.label_parts_for_current_step(current_step)
        if label_parts is None:
            return ""
        return " | ".join(label_parts)


@dataclass(frozen=True, slots=True)
class NapariLayerDisplayRequest:
    """Typed request for displaying one Napari route with one stream data type."""

    pipeline: "NapariLayerDisplayPipeline"
    items: list[NapariStreamLayerItem]
    presentation: NapariAxisPresentation

    def create_or_update_layer(
        self,
        *,
        layer_kind: NapariLayerKind,
        data: LayerData,
        layer_kwargs: Mapping[str, LayerKwargValue],
    ) -> NapariLayerHandle:
        """Create or replace the concrete Napari layer for this display request."""
        route_key = self.presentation.route_key
        layer_route_state = self.pipeline.server.layer_route_state
        return _NAPARI_LAYER_UPDATES.create_or_update(
            layer_kind=layer_kind,
            viewer=self.pipeline.server.viewer,
            layers=layer_route_state.layers,
            route_key=route_key,
            layer_name=layer_route_state.title_for(route_key),
            data=data,
            layer_kwargs=layer_kwargs,
        )


class NapariLayerDisplayHandler(
    ViewerStreamingDataTypeHandler[NapariLayerDisplayRequest],
    metaclass=ViewerStreamingDataTypeHandlerMeta,
):
    """Executable display handler for one Napari stream data type."""


@dataclass(frozen=True, slots=True)
class NapariImageLayerDisplayHandler(NapariLayerDisplayHandler):
    """Build or update a Napari image layer from routed image payloads."""

    streaming_data_type: ClassVar[StreamingDataType] = StreamingDataType.IMAGE

    def handle(self, request: NapariLayerDisplayRequest) -> None:
        layer_items = self._project_layer_items_to_image_domain(request.items)
        presentation = request.presentation
        shapes = [item.data.shape for item in layer_items]
        shape_ranks = {len(shape) for shape in shapes}
        if len(shape_ranks) > 1:
            raise ValueError(
                f"Layer {presentation.route_key} contains mixed-rank image payloads: "
                f"{sorted(set(shapes))}"
            )
        if len(set(shapes)) > 1:
            logger.info(
                "🔬 NAPARI PROCESS: Images in layer %s have different shapes - "
                "padding to max size",
                presentation.route_key,
            )
            self._pad_to_max_shape(layer_items)

        logger.info(
            "🔬 NAPARI PROCESS: Building nD data for %s from %d items",
            presentation.route_key,
            len(layer_items),
        )
        stacked_data = _build_nd_image_array(
            layer_items,
            presentation.projection,
            presentation.aggregate_axis_bindings,
        )
        pipeline = request.pipeline
        payload_axis_labels = pipeline.payload_axis_policy.axis_labels(
            layer_items[0].data,
            presentation.aggregate_axis_bindings.payload_axes,
        )
        translate = presentation.projection.translate(payload_axis_labels)

        color_component = presentation.role_component_for_mode(
            role=ViewerComponentSemanticRole.COLOR,
            mode=ViewerComponentMode.SLICE,
        )
        colormap = None
        if color_component is not None:
            first_item = layer_items[0]
            color_value = ViewerComponentCoordinateAuthority.required_value(
                first_item.address.components,
                color_component,
                context="Napari image colormap",
            )
            colormap = ChannelColormapPolicy().colormap(color_value)

        axis_labels = pipeline.dimension_label_store.apply(
            replace(presentation, payload_axis_labels=payload_axis_labels)
        )

        layer_kwargs = NapariImageLayerPresentationPolicy.layer_kwargs(
            stacked_data,
            colormap,
        )
        if axis_labels is not None:
            layer_kwargs["axis_labels"] = axis_labels
        layer_kwargs["translate"] = translate

        request.create_or_update_layer(
            layer_kind=NapariLayerKind.IMAGE,
            data=stacked_data,
            layer_kwargs=layer_kwargs,
        )
        if axis_labels is not None:
            logger.info(
                "🔬 NAPARI PROCESS: Route %s carries layer-local axis_labels=%s",
                presentation.route_key,
                axis_labels,
            )

        pipeline.dimension_label_overlay.setup_for_layer(presentation.route_key)

    @staticmethod
    def _project_layer_items_to_image_domain(
        layer_items: list[NapariStreamLayerItem],
    ) -> list[NapariStreamLayerItem]:
        projected_items: list[NapariStreamLayerItem] = []
        for item in layer_items:
            projected_data = project_singleton_stack_image_domain(item.data)
            if projected_data is item.data:
                projected_items.append(item)
                continue
            projected_items.append(
                NapariStreamLayerItem(
                    data=projected_data,
                    address=item.address,
                )
            )
        return projected_items

    @staticmethod
    def _pad_to_max_shape(layer_items: list[NapariStreamLayerItem]) -> None:
        shapes = [item.data.shape for item in layer_items]
        max_shape = list(shapes[0])
        for img_shape in shapes:
            for index, dimension in enumerate(img_shape):
                max_shape[index] = max(max_shape[index], dimension)
        resolved_max_shape = tuple(max_shape)

        for item_index, img_info in enumerate(layer_items):
            img_data = img_info.data
            if img_data.shape == resolved_max_shape:
                continue
            pad_width = []
            for current_dim, max_dim in zip(img_data.shape, resolved_max_shape):
                pad_width.append((0, max_dim - current_dim))

            padded_data = np.pad(
                img_data,
                pad_width,
                mode="constant",
                constant_values=0,
            )
            layer_items[item_index] = NapariStreamLayerItem(
                data=padded_data,
                address=img_info.address,
            )
            logger.debug(
                "🔬 NAPARI PROCESS: Padded image from %s to %s",
                img_data.shape,
                padded_data.shape,
            )


@dataclass(frozen=True, slots=True)
class NapariShapesLayerDisplayHandler(NapariLayerDisplayHandler):
    """Build or update a Napari labels layer from routed shape payloads."""

    streaming_data_type: ClassVar[StreamingDataType] = StreamingDataType.SHAPES

    def handle(self, request: NapariLayerDisplayRequest) -> None:
        pipeline = request.pipeline
        presentation = request.presentation
        logger.info(
            "🔬 NAPARI PROCESS: Converting shapes to labels for %s from %d items",
            presentation.route_key,
            len(request.items),
        )

        labels_data = _NAPARI_SHAPE_RASTERIZER.rasterize(
            layer_items=request.items,
            axis_projection=presentation.projection,
            aggregate_axis_bindings=presentation.aggregate_axis_bindings,
        )

        axis_labels = pipeline.dimension_label_store.apply(presentation)
        if axis_labels is not None:
            logger.info(
                "🔬 NAPARI PROCESS: Labels route %s carries layer-local axis_labels=%s",
                presentation.route_key,
                axis_labels,
            )

        layer_kwargs = {"translate": presentation.projection.translate()}
        if axis_labels is not None:
            layer_kwargs["axis_labels"] = axis_labels

        request.create_or_update_layer(
            layer_kind=NapariLayerKind.LABELS,
            data=labels_data,
            layer_kwargs=layer_kwargs,
        )
        pipeline.dimension_label_overlay.setup_for_layer(presentation.route_key)
        logger.info(
            "🔬 NAPARI PROCESS: Created labels layer %s with shape %s",
            presentation.route_key,
            labels_data.shape,
        )


@dataclass(frozen=True, slots=True)
class NapariPointsLayerDisplayHandler(NapariLayerDisplayHandler):
    """Build or update a Napari points layer from routed point payloads."""

    streaming_data_type: ClassVar[StreamingDataType] = StreamingDataType.POINTS

    def handle(self, request: NapariLayerDisplayRequest) -> None:
        pipeline = request.pipeline
        presentation = request.presentation
        logger.info(
            "🔬 NAPARI PROCESS: Building points layer for %s from %d items",
            presentation.route_key,
            len(request.items),
        )

        points_data, properties = _build_nd_points(
            request.items,
            presentation.projection,
        )
        axis_labels = pipeline.dimension_label_store.apply(presentation)

        layer_kwargs = {
            "properties": properties,
            "face_color": "green",
            "size": 3,
            "translate": presentation.projection.translate(),
        }
        if axis_labels is not None:
            layer_kwargs["axis_labels"] = axis_labels
        request.create_or_update_layer(
            layer_kind=NapariLayerKind.POINTS,
            data=points_data,
            layer_kwargs=layer_kwargs,
        )
        pipeline.dimension_label_overlay.setup_for_layer(presentation.route_key)

        logger.info(
            "🔬 NAPARI PROCESS: Created points layer %s with %d points",
            presentation.route_key,
            len(points_data),
        )


class NapariLayerDisplayPipeline:
    """Owns debounced Napari layer display and update routing."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server
        self.axis_projector = ViewerLayerAxisProjector()
        self.payload_axis_policy = NapariImagePayloadAxisLabelPolicy()
        route_resolver = NapariDimensionLabelRouteResolver(server)
        self.dimension_label_store = NapariDimensionLabelStore(server)
        self.dimension_label_overlay = NapariDimensionLabelOverlayController(
            server,
            route_resolver,
        )

    def display_axis_projection(
        self,
        layer_key: str,
        component_axis_semantics: ViewerComponentAxisSemantics,
        layer_items: list[NapariStreamLayerItem],
        aggregate_axis_bindings: NapariAggregateAxisBindingSet | None = None,
    ) -> ViewerLayerAxisProjection:
        """Return route-local coordinates projected into the viewer axis domain."""
        if aggregate_axis_bindings is None:
            aggregate_axis_bindings = NapariAggregateAxisBindingSet()

        axis_components = component_axis_semantics.layout.components_for_mode(
            ViewerComponentMode.STACK
        )
        self.server.component_values.update(
            layer_key,
            axis_components,
            layer_items,
        )
        self.server.display_axis_domain.record_display_axis_values(
            axis_components,
            layer_items,
        )

        aggregate_component_values = aggregate_axis_bindings.component_values
        if aggregate_component_values:
            self.server.component_values.update_component_values(
                layer_key,
                axis_components,
                aggregate_component_values,
            )
            self.server.display_axis_domain.record_display_component_values(
                axis_components,
                aggregate_component_values,
            )

        projection_request = ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=axis_components,
            route_component_values=self.server.component_values.values_for(
                self.server.component_values.domain_key(layer_key, axis_components),
                axis_components,
            ),
            viewer_component_values=self.server.display_axis_domain.display_axis_values_for(
                axis_components
            ),
            declared_component_values=component_axis_semantics.required_component_values(
                axis_components
            ),
        )
        return self.axis_projector.project(projection_request)

    def schedule_layer_update(
        self,
        layer_key: str,
        data_type: StreamingDataType,
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> None:
        if self.server.layer_route_state.cancel_pending_update(layer_key):
            logger.debug(f"🔬 NAPARI PROCESS: Cancelled pending update for {layer_key}")

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(
            lambda: self.execute_layer_update(
                layer_key,
                data_type,
                layer_axis_projection_semantics,
            )
        )
        self.server.layer_batch_processor_debounce_policy.start_timer(timer)
        self.server.layer_route_state.set_pending_update(
            layer_key,
            NapariPendingLayerUpdate.from_semantics(
                timer=timer,
                data_type=data_type,
                semantics=layer_axis_projection_semantics,
            ),
        )
        logger.debug(
            "🔬 NAPARI PROCESS: Scheduled update for %s in %sms",
            layer_key,
            self.server.layer_batch_processor_debounce_policy.delay_ms,
        )

    def execute_layer_update(
        self,
        layer_key: str,
        data_type: StreamingDataType,
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> None:
        self.server.layer_route_state.pop_pending_update(layer_key)

        layer_items = self.server.component_groups.existing_items_for(layer_key)
        if layer_items is None:
            logger.warning(
                f"🔬 NAPARI PROCESS: No items found for {layer_key}, skipping update"
            )
            return
        if not layer_items:
            logger.warning(
                f"🔬 NAPARI PROCESS: Empty item group for {layer_key}, skipping update"
            )
            return

        component_values = {
            component: sorted(
                {
                    item.address.components[component]
                    for item in layer_items
                    if component in item.address.components
                },
                key=ViewerComponentValueOrdering.key,
            )
            for component in layer_axis_projection_semantics.component_order
        }
        logger.info(
            "🔬 NAPARI PROCESS: layer_key='%s' has %d items with components=%s",
            layer_key,
            len(layer_items),
            component_values,
        )

        batch_processor = self.server.batch_processors.get_or_create(
            layer_key=layer_key,
            napari_server=self.server,
        )
        try:
            batch_processor.add_items(
                layer_key=layer_key,
                items=layer_items,
                display_payload=layer_axis_projection_semantics,
                component_names_metadata=self.server.component_name_metadata,
            )
        except Exception:
            logger.exception(
                "🔬 NAPARI PROCESS: Failed to update layer %s; viewer will keep processing messages",
                layer_key,
            )

    def settle_pending_updates(self) -> int:
        """Synchronously execute queued debounced layer updates."""

        pending_updates = self.server.layer_route_state.drain_pending_updates()
        for layer_key, update in pending_updates:
            self.execute_layer_update(
                layer_key,
                update.data_type,
                update,
            )
        return len(pending_updates)

    def display_layer_batch(
        self,
        *,
        layer_key: str,
        items: list[NapariStreamLayerItem],
        display_payload: ViewerComponentAxisSemantics,
        component_names_metadata: ViewerComponentNameMetadata,
    ) -> None:
        if component_names_metadata:
            self.server.component_name_metadata.merge(component_names_metadata)

        if not items:
            raise ValueError(f"Napari display batch for {layer_key!r} has no items.")

        data_type = items[0].address.stream_layer_data_type
        for item in items:
            if item.address.stream_layer_data_type is not data_type:
                raise ValueError(
                    "Napari display route "
                    f"{layer_key!r} mixed data types: {data_type!r} and "
                    f"{item.address.stream_layer_data_type!r}."
                )

        aggregate_axis_bindings = NapariAggregateAxisBindingAuthority.bindings(
            items,
            display_payload,
        )
        axis_projection = self.display_axis_projection(
            layer_key,
            display_payload,
            items,
            aggregate_axis_bindings,
        )
        NapariLayerDisplayHandler.for_data_type(data_type).handle(
            NapariLayerDisplayRequest(
                pipeline=self,
                items=items,
                presentation=NapariAxisPresentation(
                    entries=display_payload.entries,
                    layout=display_payload.layout,
                    route_key=layer_key,
                    projection=axis_projection,
                    aggregate_axis_bindings=aggregate_axis_bindings,
                ),
            )
        )
        logger.info(
            "🔬 NAPARI PROCESS: Displayed %d %s item(s) in layer %s",
            len(items),
            data_type.value,
            layer_key,
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
        message: Mapping[str, NapariWireValue],
    ) -> dict[str, NapariWireValue]:
        """Handle a control message and return the control reply."""


class NapariShutdownControlMessageAction(NapariControlMessageAction):
    """Shared shutdown behavior for graceful and force shutdown requests."""

    message_type = None

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, NapariWireValue],
    ) -> dict[str, NapariWireValue]:
        del message
        logger.info("🔬 NAPARI SERVER: %s requested, closing viewer", self.message_type)
        server.request_shutdown()
        if server.viewer is not None:
            from qtpy import QtCore

            QtCore.QTimer.singleShot(100, server.viewer.close)
        return ViewerControlReplyPayload(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.SUCCESS,
                response_type="shutdown_ack",
                message="Napari viewer shutting down",
            )
        ).to_wire_mapping()


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
        message: Mapping[str, NapariWireValue],
    ) -> dict[str, NapariWireValue]:
        del message
        logger.info(
            "🔬 NAPARI SERVER: Clearing component groups (had %d groups)",
            len(server.component_groups),
        )
        server.clear_accumulated_stream_state()
        return ViewerControlReplyPayload(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.SUCCESS,
                response_type="clear_state_ack",
                message="Component groups cleared",
            )
        ).to_wire_mapping()


class NapariSettleControlMessageAction(NapariControlMessageAction):
    """Registered action that drains queued debounced layer updates."""

    message_type = ViewerControlMessageType.SETTLE.value

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, NapariWireValue],
    ) -> dict[str, NapariWireValue]:
        del message
        if server.viewer is None:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="settle_ack",
                    message="Napari viewer is not available.",
                )
            ).to_wire_mapping()

        flushed_count = server.display_pipeline.settle_pending_updates()
        return ViewerControlReplyPayload(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.SUCCESS,
                response_type="settle_ack",
                message=f"Flushed {flushed_count} pending layer update(s).",
            )
        ).to_wire_mapping()


@dataclass(frozen=True, slots=True)
class ShapeCoordinateBounds:
    """Aggregate YX coordinate bounds for Napari shape payload state."""

    min_y: float | None
    min_x: float | None
    max_y: float | None
    max_x: float | None
    coordinate_count: int = 0

    @classmethod
    def empty(cls) -> "ShapeCoordinateBounds":
        return cls(None, None, None, None, 0)

    @property
    def has_coordinates(self) -> bool:
        return self.coordinate_count > 0

    def union(self, other: "ShapeCoordinateBounds") -> "ShapeCoordinateBounds":
        if not self.has_coordinates:
            return other
        if not other.has_coordinates:
            return self
        return type(self)(
            min_y=min(float(self.min_y), float(other.min_y)),
            min_x=min(float(self.min_x), float(other.min_x)),
            max_y=max(float(self.max_y), float(other.max_y)),
            max_x=max(float(self.max_x), float(other.max_x)),
            coordinate_count=self.coordinate_count + other.coordinate_count,
        )

    def outside_source_shape(self, source_shape_yx: tuple[int, int]) -> bool:
        if not self.has_coordinates:
            return False
        height, width = source_shape_yx
        return (
            float(self.min_y) < 0
            or float(self.min_x) < 0
            or float(self.max_y) >= height
            or float(self.max_x) >= width
        )

    def to_wire_mapping(self) -> dict[str, NapariWireValue]:
        if not self.has_coordinates:
            return {}
        return {
            "min_yx": (float(self.min_y), float(self.min_x)),
            "max_yx": (float(self.max_y), float(self.max_x)),
            "coordinate_count": self.coordinate_count,
        }

    @classmethod
    def from_shape_payload(
        cls,
        payload: Mapping[str, NapariWireValue],
    ) -> "ShapeCoordinateBounds | None":
        coordinates = payload.get(NapariWireField.COORDINATES.value)
        if coordinates is not None:
            return cls.from_yx_coordinates(coordinates)
        center = payload.get(NapariWireField.CENTER.value)
        radii = payload.get(NapariWireField.RADII.value)
        if center is not None and radii is not None:
            return cls.from_center_radii(center, radii)
        return None

    @classmethod
    def from_yx_coordinates(
        cls,
        coordinates: NapariWireValue,
    ) -> "ShapeCoordinateBounds | None":
        array = np.asarray(coordinates, dtype=float)
        if array.ndim == 1 and array.size == 2:
            array = array.reshape(1, 2)
        if array.ndim < 2 or array.shape[-1] != 2:
            return None
        points = array.reshape(-1, 2)
        return cls.from_points(points)

    @classmethod
    def from_center_radii(
        cls,
        center: NapariWireValue,
        radii: NapariWireValue,
    ) -> "ShapeCoordinateBounds | None":
        center_array = np.asarray(center, dtype=float)
        radii_array = np.asarray(radii, dtype=float)
        if center_array.shape != (2,) or radii_array.shape != (2,):
            return None
        y, x = center_array
        radius_y, radius_x = radii_array
        return cls(
            min_y=float(y - radius_y),
            min_x=float(x - radius_x),
            max_y=float(y + radius_y),
            max_x=float(x + radius_x),
            coordinate_count=4,
        )

    @classmethod
    def from_points(cls, points: np.ndarray) -> "ShapeCoordinateBounds":
        return cls(
            min_y=float(points[:, 0].min()),
            min_x=float(points[:, 1].min()),
            max_y=float(points[:, 0].max()),
            max_x=float(points[:, 1].max()),
            coordinate_count=int(points.shape[0]),
        )


@dataclass(frozen=True, slots=True)
class NapariViewerStateProjection:
    """Project live Napari route/component stores into an agent-readable state."""

    server: "NapariViewerServer"
    viewer: NapariViewerLayerCreator

    def wire_envelope(
        self,
        *,
        response_type: str,
        layers: tuple[dict[str, NapariWireValue], ...],
    ) -> dict[str, NapariWireValue]:
        return {
            "type": response_type,
            "status": _ACK_SUCCESS,
            "viewer": {
                "type": "napari",
                "title": self.server.napari_window_title,
            },
            "layer_count": len(layers),
            "layers": layers,
        }

    def to_wire_mapping(self) -> dict[str, NapariWireValue]:
        route_keys = self.route_keys()
        layers = tuple(
            self.layer_state_for(route_key)
            for route_key in route_keys
        )
        wire_mapping = self.wire_envelope(response_type="state_ack", layers=layers)
        wire_mapping.update(
            {
                "active_dimension_label_route": (
                    self.server.layer_route_state.active_dimension_label_route
                ),
                "viewer_ndim": int(self.viewer.dims.ndim),
                "current_step": tuple(
                    int(step) for step in self.viewer.dims.current_step
                ),
                "axis_labels": tuple(
                    str(label) for label in self.viewer.dims.axis_labels
                ),
                "component_group_count": len(self.server.component_groups),
                "component_item_count": sum(
                    self.server.component_groups.item_count(route_key)
                    for route_key in route_keys
                ),
            }
        )
        return wire_mapping

    def route_keys(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.server.layer_route_state.layer_titles,
                    *self.server.component_groups.groups,
                )
            )
        )

    def layer_state_for(self, route_key: str) -> dict[str, NapariWireValue]:
        dimension_state = self.server.layer_route_state.dimension_state_for(route_key)
        items = self.server.component_groups.existing_items_for(route_key)
        layer = self.mounted_layer(route_key)
        if items is None:
            item_tuple: tuple[NapariStreamLayerItem, ...] = ()
        else:
            item_tuple = tuple(items)
        layer_visible = False
        layer_selected = False
        if layer is not None:
            layer_visible = bool(layer.visible)
            layer_selected = layer is self.viewer.layers.selection.active
        return {
            "route_key": route_key,
            "title": self.layer_title(route_key),
            "mounted": layer is not None,
            "item_count": len(item_tuple),
            "data_types": tuple(
                dict.fromkeys(
                    item.address.stream_layer_data_type.value
                    for item in item_tuple
                )
            ),
            "component_values": tuple(
                dict(item.address.components)
                for item in item_tuple
            ),
            "payload_summaries": tuple(
                self.payload_summary_for(
                    item,
                    (
                        dimension_state.presentation.aggregate_axis_bindings
                        if dimension_state.presentation is not None
                        else NapariAggregateAxisBindingSet()
                    ),
                )
                for item in item_tuple
            ),
            "axis_labels": dimension_state.axis_labels,
            "stack_axes": dimension_state.stack_axes,
            "axis_offsets": dimension_state.axis_offsets,
            "scalar_labels": dimension_state.scalar_labels,
            "labels": dimension_state.labels,
            "axis_component_values": self.axis_component_values(dimension_state),
            "routed_component_values": self.routed_component_values(dimension_state),
            "data_shape": self.layer_data_shape(layer),
            "translate": self.layer_translate(layer),
            "visible": layer_visible,
            "selected": layer_selected,
            "pending_update": (
                route_key in self.server.layer_route_state.layer_pending_updates
            ),
        }

    def layer_title(self, route_key: str) -> str | None:
        if route_key not in self.server.layer_route_state.layer_titles:
            return None
        return self.server.layer_route_state.title_for(route_key)

    def layer_is_mounted(self, route_key: str) -> bool:
        return self.mounted_layer(route_key) is not None

    def mounted_layer(self, route_key: str) -> NapariLayerHandle | None:
        if not self.server.layer_route_state.has_layer(route_key):
            return None
        layer = self.server.layer_route_state.layer(route_key)
        if layer not in self.viewer.layers:
            return None
        return layer

    @staticmethod
    def layer_data_shape(layer: NapariLayerHandle | None) -> tuple[int, ...]:
        if layer is None:
            return ()
        data = layer.data
        if isinstance(data, np.ndarray):
            return tuple(int(axis) for axis in data.shape)
        if isinstance(data, list):
            return (len(data),)
        raise TypeError(
            "Napari layer data must be an ndarray or vector-shape list, "
            f"got {type(data).__name__}."
        )

    @staticmethod
    def layer_translate(layer: NapariLayerHandle | None) -> tuple[float, ...]:
        if layer is None:
            return ()
        return tuple(float(offset) for offset in layer.translate)

    @staticmethod
    def axis_component_values(
        dimension_state: NapariDimensionLayerState,
    ) -> dict[str, list[ComponentValue]]:
        if dimension_state.presentation is None:
            return {}
        return {
            component: list(values)
            for component, values in (
                dimension_state.presentation.projection.component_values.items()
            )
        }

    @staticmethod
    def routed_component_values(
        dimension_state: NapariDimensionLayerState,
    ) -> dict[str, list[ComponentValue]]:
        if dimension_state.presentation is None:
            return {}
        return {
            component: list(values)
            for component, values in (
                dimension_state.presentation.projection.routed_component_values.items()
            )
        }

    @classmethod
    def payload_summary_for(
        cls,
        item: NapariStreamLayerItem,
        aggregate_axis_bindings: NapariAggregateAxisBindingSet | None = None,
    ) -> dict[str, NapariWireValue]:
        data = item.data
        summary: dict[str, NapariWireValue] = {
            "data_type": item.address.stream_layer_data_type.value,
            "path": item.address.path,
            "components": dict(item.address.components),
            "payload_type": type(data).__name__,
        }
        if aggregate_axis_bindings is not None and aggregate_axis_bindings.bindings:
            summary["aggregate_component_values"] = {
                binding.component: tuple(binding.values)
                for binding in aggregate_axis_bindings.bindings
            }
        if isinstance(data, np.ndarray):
            summary.update(cls.array_summary(data))
            return summary
        if isinstance(data, (list, tuple)):
            summary["item_count"] = len(data)
            summary["nonzero_count"] = len(data)
            summary.update(cls.shape_payload_summary(data))
            return summary
        return summary

    @classmethod
    def shape_payload_summary(
        cls,
        payloads: Sequence[NapariWireValue],
    ) -> dict[str, NapariWireValue]:
        shape_payload_count = 0
        missing_source_shape_count = 0
        source_shapes: set[tuple[int, int]] = set()
        coordinate_count = 0
        out_of_bounds_count = 0
        aggregate_bounds = ShapeCoordinateBounds.empty()
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            shape_payload_count += 1
            source_shape = cls.shape_source_spatial_shape(payload)
            if source_shape is None:
                missing_source_shape_count += 1
            else:
                source_shapes.add(source_shape)
            bounds = ShapeCoordinateBounds.from_shape_payload(payload)
            if bounds is None:
                continue
            coordinate_count += bounds.coordinate_count
            aggregate_bounds = aggregate_bounds.union(bounds)
            if source_shape is not None and bounds.outside_source_shape(source_shape):
                out_of_bounds_count += 1

        summary: dict[str, NapariWireValue] = {
            "shape_payload_count": shape_payload_count,
            "missing_source_spatial_shape_count": missing_source_shape_count,
            "source_spatial_shapes_yx": tuple(sorted(source_shapes)),
            "shape_coordinate_count": coordinate_count,
            "shape_out_of_source_bounds_count": out_of_bounds_count,
        }
        if aggregate_bounds.has_coordinates:
            summary["shape_coordinate_bounds_yx"] = aggregate_bounds.to_wire_mapping()
        return summary

    @staticmethod
    def shape_source_spatial_shape(
        payload: Mapping[str, NapariWireValue],
    ) -> tuple[int, int] | None:
        metadata = payload.get(NapariWireField.METADATA.value)
        if not isinstance(metadata, Mapping):
            return None
        return SourceSpatialDomain.from_viewer_wire_mapping(
            metadata,
            source_label="Napari shape payload state projection",
            value_name="Napari shape payload",
        ).source_shape_yx

    @staticmethod
    def array_summary(array: np.ndarray) -> dict[str, NapariWireValue]:
        summary: dict[str, NapariWireValue] = {
            "shape": tuple(int(axis) for axis in array.shape),
            "dtype": str(array.dtype),
            "size": int(array.size),
            "nonzero_count": int(np.count_nonzero(array)),
        }
        if array.size:
            summary["min"] = NapariViewerStateProjection.json_scalar(array.min())
            summary["max"] = NapariViewerStateProjection.json_scalar(array.max())
        return summary

    @staticmethod
    def json_scalar(value: np.generic | bool | int | float | str) -> NapariWireValue:
        if isinstance(value, np.generic):
            scalar = value.item()
        else:
            scalar = value
        if isinstance(scalar, (bool, int, float, str)):
            return scalar
        return str(scalar)


@dataclass(frozen=True, slots=True)
class NapariViewerPayloadProjection(NapariViewerStateProjection):
    """Project live Napari layer payloads with axis-coordinate evidence."""

    request: ViewerPayloadControlOptions

    def to_wire_mapping(self) -> dict[str, NapariWireValue]:
        layers = tuple(
            self.layer_payloads_for(route_key)
            for route_key in self.route_keys()
        )
        return self.wire_envelope(response_type="payloads_ack", layers=layers)

    def route_keys(self) -> tuple[str, ...]:
        route_keys = super().route_keys()
        if self.request.route_key is None:
            return route_keys
        return tuple(
            route_key for route_key in route_keys if route_key == self.request.route_key
        )

    def layer_payloads_for(self, route_key: str) -> dict[str, NapariWireValue]:
        layer_state = self.layer_state_for(route_key)
        items = self.server.component_groups.existing_items_for(route_key)
        item_tuple: tuple[NapariStreamLayerItem, ...] = ()
        if items is not None:
            item_tuple = tuple(items)
        dimension_state = self.server.layer_route_state.dimension_state_for(route_key)
        return {
            "route_key": route_key,
            "title": layer_state["title"],
            "mounted": layer_state["mounted"],
            "item_count": len(item_tuple),
            "axis_labels": layer_state["axis_labels"],
            "stack_axes": layer_state["stack_axes"],
            "pending_update": layer_state["pending_update"],
            "payloads": tuple(
                record
                for item in item_tuple
                for record in self.records_for_item(item, route_key, dimension_state)
            ),
        }

    def records_for_item(
        self,
        item: NapariStreamLayerItem,
        route_key: str,
        dimension_state: NapariDimensionLayerState,
    ) -> tuple[dict[str, NapariWireValue], ...]:
        if dimension_state.presentation is None:
            return (self.record_for_item(item, route_key, (), (), None),)

        aggregate_axis_bindings = dimension_state.presentation.aggregate_axis_bindings
        aggregate_index_tuples = self.aggregate_index_tuples(aggregate_axis_bindings)
        return tuple(
            self.record_for_item(
                item,
                route_key,
                aggregate_indices,
                dimension_state.presentation.projection.coordinate_index(
                    aggregate_axis_bindings.item_component_values(
                        item,
                        aggregate_indices,
                    ),
                    context="Napari payload inspection",
                ),
                aggregate_axis_bindings,
            )
            for aggregate_indices in aggregate_index_tuples
        )

    @staticmethod
    def aggregate_index_tuples(
        aggregate_axis_bindings: NapariAggregateAxisBindingSet,
    ) -> tuple[tuple[int, ...], ...]:
        if not aggregate_axis_bindings.bindings:
            return ((),)
        return tuple(
            tuple(indices)
            for indices in product(
                *(range(binding.extent) for binding in aggregate_axis_bindings.bindings)
            )
        )

    def record_for_item(
        self,
        item: NapariStreamLayerItem,
        route_key: str,
        aggregate_indices: tuple[int, ...],
        axis_indices: tuple[int, ...],
        aggregate_axis_bindings: NapariAggregateAxisBindingSet | None,
    ) -> dict[str, NapariWireValue]:
        components = dict(item.address.components)
        data = item.data
        if aggregate_axis_bindings is not None:
            components = dict(
                aggregate_axis_bindings.item_component_values(
                    item,
                    aggregate_indices,
                )
            )
            data = self.aggregate_data_slice(item.data, aggregate_indices)

        return {
            "route_key": route_key,
            "data_type": item.address.stream_layer_data_type.value,
            "path": item.address.path,
            "components": components,
            "axis_indices": axis_indices,
            "aggregate_axis_indices": aggregate_indices,
            "summary": self.payload_summary(item, components, data),
            "array_values": self.array_values(data),
            "shape_payloads": self.shape_payloads(data),
        }

    @staticmethod
    def aggregate_data_slice(
        data: LayerData,
        aggregate_indices: tuple[int, ...],
    ) -> LayerData:
        if not aggregate_indices:
            return data
        if isinstance(data, np.ndarray):
            return data[aggregate_indices]
        return data

    @classmethod
    def payload_summary(
        cls,
        item: NapariStreamLayerItem,
        components: Mapping[str, ComponentValue],
        data: LayerData,
    ) -> dict[str, NapariWireValue]:
        summary: dict[str, NapariWireValue] = {
            "data_type": item.address.stream_layer_data_type.value,
            "path": item.address.path,
            "components": dict(components),
            "payload_type": type(data).__name__,
        }
        if isinstance(data, np.ndarray):
            summary.update(NapariViewerStateProjection.array_summary(data))
            return summary
        if isinstance(data, (list, tuple)):
            summary["item_count"] = len(data)
            summary["nonzero_count"] = len(data)
            summary.update(NapariViewerStateProjection.shape_payload_summary(data))
            return summary
        return summary

    def array_values(self, data: LayerData) -> tuple[NapariWireValue, ...]:
        if not self.request.include_array_values:
            return ()
        if not isinstance(data, np.ndarray):
            return ()
        if data.size > self.request.max_array_elements:
            return ()
        value = self.wire_value(data)
        if isinstance(value, tuple):
            return value
        return (value,)

    def shape_payloads(self, data: LayerData) -> tuple[dict[str, NapariWireValue], ...]:
        if not self.request.include_shape_payloads:
            return ()
        if not isinstance(data, (list, tuple)):
            return ()
        records: list[dict[str, NapariWireValue]] = []
        for payload in data:
            if not isinstance(payload, Mapping):
                continue
            records.append(
                {
                    str(key): self.wire_value(value)
                    for key, value in payload.items()
                }
            )
            if len(records) >= self.request.max_shape_payloads:
                break
        return tuple(records)

    @classmethod
    def wire_value(cls, value: NapariWireValue) -> NapariWireValue:
        if isinstance(value, np.ndarray):
            return cls.wire_value(value.tolist())
        if isinstance(value, np.generic):
            return NapariViewerStateProjection.json_scalar(value)
        if isinstance(value, Mapping):
            return {
                str(key): cls.wire_value(mapping_value)
                for key, mapping_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return tuple(cls.wire_value(item) for item in value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


class NapariStateControlMessageAction(NapariControlMessageAction):
    """Registered action that reports live layer and axis state."""

    message_type = ViewerControlMessageType.STATE.value

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, NapariWireValue],
    ) -> dict[str, NapariWireValue]:
        del message
        if server.viewer is None:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="state_ack",
                    message="Napari viewer is not available.",
                )
            ).to_wire_mapping()

        return NapariViewerStateProjection(
            server=server,
            viewer=server.viewer,
        ).to_wire_mapping()


class NapariPayloadsControlMessageAction(NapariControlMessageAction):
    """Registered action that reports live payload records by layer and axis."""

    message_type = ViewerControlMessageType.PAYLOADS.value

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, NapariWireValue],
    ) -> dict[str, NapariWireValue]:
        if server.viewer is None:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="payloads_ack",
                    message="Napari viewer is not available.",
                )
            ).to_wire_mapping()

        request = ViewerPayloadControlOptions.from_wire_payload(message)
        return NapariViewerPayloadProjection(
            server=server,
            viewer=server.viewer,
            request=request,
        ).to_wire_mapping()


class NapariScreenshotControlMessageAction(NapariControlMessageAction):
    """Registered action that captures the Napari Qt window."""

    message_type = ViewerControlMessageType.SCREENSHOT.value

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, NapariWireValue],
    ) -> NapariControlReplyPayload:
        if server.viewer is None:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="screenshot_ack",
                    message="Napari viewer is not available.",
                )
            ).to_wire_mapping()

        from openhcs.runtime.qt_window_snapshot import (
            QtWindowSnapshotRequest,
            QtWindowSnapshotService,
        )
        from openhcs.runtime.window_snapshot import (
            WindowSnapshotCaptureSpec,
            WindowSnapshotWirePayload,
        )

        payload = PayloadMap(message, "Napari screenshot control message")
        capture_spec = WindowSnapshotCaptureSpec.from_wire_payload(
            WindowSnapshotWirePayload(
                _napari_wire_str_mapping(
                    payload.required_mapping(NapariWireField.SNAPSHOT),
                    NapariWireField.SNAPSHOT,
                    payload.context,
                )
            )
        )
        snapshot = QtWindowSnapshotService().capture(
            QtWindowSnapshotRequest(
                widget=server.viewer.window.qt_viewer.window(),
                capture=capture_spec,
                subject_id=f"napari_{server.port}",
                title=server.napari_window_title,
            )
        )
        return {
            "type": "screenshot_ack",
            "status": _ACK_SUCCESS,
            "viewer": {
                "type": "napari",
                "title": server.napari_window_title,
            },
            "resource": {
                "uri": snapshot.uri,
                "title": snapshot.title,
                "mime_type": snapshot.mime_type,
                "path": snapshot.path,
                "size_bytes": snapshot.size_bytes,
                "sha256": snapshot.sha256,
            },
            "width": snapshot.width,
            "height": snapshot.height,
            "snapshot": snapshot.capture.to_wire_payload().as_dict(),
        }


class NapariUnknownControlMessageAction(NapariControlMessageAction):
    """Default no-op control action for unknown message types."""

    message_type = None

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, NapariWireValue],
    ) -> dict[str, NapariWireValue]:
        del server, message
        return {"status": "ok"}


class NapariStreamMessageHandler(NapariMessageTypeBase, metaclass=AutoRegisterMeta):
    """Registered handler for one Napari stream message type."""

    @classmethod
    def for_message_type(
        cls,
        message_type: ViewerBatchMessageType,
    ) -> "NapariStreamMessageHandler":
        if message_type not in cls.__registry__:
            raise ValueError(
                f"Napari stream messages must be registered message types, got {message_type!r}."
            )
        return cls.__registry__[message_type]()

    @abstractmethod
    def handle(
        self,
        server: "NapariViewerServer",
        data: Mapping[str, NapariWireValue],
    ) -> None:
        """Handle one decoded stream message."""


class NapariBatchStreamMessageHandler(NapariStreamMessageHandler):
    """Registered stream handler for batched Napari payloads."""

    message_type = ViewerBatchMessageType.BATCH

    def handle(
        self,
        server: "NapariViewerServer",
        data: Mapping[str, NapariWireValue],
    ) -> None:
        batch_payload = NapariBatchPayload.from_json_payload(data)
        if batch_payload.store:
            server.component_name_metadata.merge(batch_payload)
            logger.info(
                "🔬 NAPARI PROCESS: Updated component metadata: %s",
                list(batch_payload),
            )

        for image_info in batch_payload.images:
            server._process_single_image(
                image_info,
                batch_payload,
            )


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

        self.napari_window_title = request.viewer_title
        self.replace_layers = request.replace_layers
        self.viewer = None
        self.layer_route_state = NapariLayerRouteStateStore.empty()
        self.component_groups = NapariComponentGroupStore()
        self.component_name_metadata = ViewerComponentNameMetadata.empty()

        self.component_values = ViewerRouteComponentValueTracker()
        self.display_axis_domain = ViewerDisplayAxisDomain()

        # Debouncing + locking for layer updates to prevent race conditions
        import threading

        self.layer_update_lock = threading.Lock()  # Prevent concurrent updates
        self.layer_batch_processor_debounce_policy = NapariLayerBatchDebouncePolicy()
        self.batch_processors = NapariBatchProcessorStore(
            debounce_policy=self.layer_batch_processor_debounce_policy,
        )
        self.display_pipeline = NapariLayerDisplayPipeline(self)

        # Ack socket handled by StreamingVisualizerServer

    def display_layer_batch(
        self,
        *,
        layer_key: str,
        items: list[NapariStreamLayerItem],
        display_payload: ViewerComponentAxisSemantics,
        component_names_metadata: ViewerComponentNameMetadata,
    ) -> None:
        """Display one debounced batch through the composed display pipeline."""
        self.display_pipeline.display_layer_batch(
            layer_key=layer_key,
            items=items,
            display_payload=display_payload,
            component_names_metadata=component_names_metadata,
        )

    def clear_accumulated_stream_state(self) -> None:
        """Reset stream domains that must not leak across pipeline executions."""
        self.component_groups.clear()
        self.component_values = ViewerRouteComponentValueTracker()
        self.display_axis_domain = ViewerDisplayAxisDomain()
        self.component_name_metadata.clear()
        self.batch_processors = NapariBatchProcessorStore(
            debounce_policy=self.layer_batch_processor_debounce_policy,
        )

    def _create_pong_response(self) -> dict[str, NapariWireValue]:
        """Override to add Napari-specific fields and memory usage."""
        return NAPARI_HEARTBEAT.apply_to(super()._create_pong_response())

    def handle_control_message(
        self,
        message: Mapping[str, NapariWireValue],
    ) -> dict[str, NapariWireValue]:
        """
        Handle control messages beyond ping/pong.

        Supported message types:
        - shutdown: Graceful shutdown (closes viewer)
        - force_shutdown: Force shutdown (same as shutdown for Napari)
        - clear_state: Clear accumulated component groups (for new pipeline runs)
        """
        msg_type = PayloadMap(message, "Napari control message").optional(
            ViewerBatchWireField.TYPE
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
            "producer_identity": StreamProducerIdentity.fixed_output(
                FixedStreamProducerIdentityKind.DIRECT,
                "direct_image"
            ).to_payload(),
        }
        self._process_single_image(
            image_info,
            ViewerComponentAxisSemanticsAuthority.empty(),
        )

    def process_image_message(self, message: bytes):
        """
        Process incoming image data message and send reply for REP socket.

        Args:
            message: Raw ZMQ message containing image data
        """
        import json

        msg_type: ViewerBatchMessageType | None = None
        reply_sent = False
        try:
            data = json.loads(message.decode("utf-8"))
            msg_type = ViewerBatchMessageType(
                str(PayloadMap(data, "Napari message").required(
                    ViewerBatchWireField.TYPE
                ))
            )
            reply = NapariStreamMessageReply.success(msg_type)
            self.data_socket.send_json(reply.to_wire_mapping())
            reply_sent = True
            NapariStreamMessageHandler.for_message_type(msg_type).handle(self, data)
        except Exception as e:
            logger.error(
                "🔬 NAPARI PROCESS: Failed to process stream message: %s",
                e,
                exc_info=True,
            )
            if not reply_sent:
                try:
                    reply = NapariStreamMessageReply.failure(msg_type, str(e))
                    self.data_socket.send_json(reply.to_wire_mapping())
                except Exception as reply_error:
                    logger.error(
                        "🔬 NAPARI PROCESS: Failed to send failure reply: %s",
                        reply_error,
                    )

    def _process_single_image(
        self,
        image_info: Mapping[str, NapariWireValue],
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> None:
        """Process a single image or shapes data and display in Napari."""
        payload = NapariImagePayload.from_payload(
            image_info,
            layer_axis_projection_semantics,
        )
        payload_address = payload.address
        logger.info(
            f"🔍 NAPARI PROCESS: Received {payload_address.stream_layer_data_type} with metadata: {payload_address.components} (path: {payload_address.path})"
        )

        try:
            loaded_data = _NAPARI_PAYLOAD_DATA_LOADER.load(payload)
            if isinstance(loaded_data, np.ndarray):
                logger.info(
                    "🔬 STREAM RECEIVE: path=%s components=%s %s",
                    payload_address.path,
                    payload_address.components,
                    NapariViewerStateProjection.array_summary(loaded_data),
                )
            _NAPARI_COMPONENT_DISPLAY_COORDINATOR.display(
                data=loaded_data,
                stream_layer_context=payload,
                server=self,
            )
            if payload.image_id:
                self.send_ack(payload.image_id, status=_ACK_SUCCESS)

        except Exception as e:
            logger.error(
                f"🔬 NAPARI PROCESS: Failed to process {payload_address.stream_layer_data_type} {payload_address.path}: {e}",
                exc_info=True,
            )
            if payload.image_id:
                self.send_ack(payload.image_id, status=_ACK_ERROR, error=str(e))
            # Don't re-raise - continue processing other messages instead of crashing


def run_napari_viewer_process(
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

        request = NapariViewerServerRequest(
            port=port,
            viewer_title=viewer_title,
            replace_layers=replace_layers,
            log_file_path=log_file_path,
            transport_mode=transport_mode,
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
            server.layer_route_state.set_layer(layer.name, layer)

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
