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
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional, Sequence, TypeAlias
from qtpy.QtCore import QTimer

from openhcs.core.config import TransportMode as OpenHCSTransportMode
from metaclass_registry import AutoRegisterMeta
from polystore.backend_registry import register_cleanup_callback
from zmqruntime.config import TransportMode, ZMQConfig
from polystore.streaming_constants import StreamingDataType
from polystore.streaming.identity import (
    StreamProducerDisplayNameAuthority,
    StreamProducerIdentity,
    StreamRouteKeyAuthority,
)
from polystore.streaming.receivers.napari import build_route_key
from openhcs.runtime.viewer_protocol import (
    ChannelColormapPolicy,
    NAPARI_HEARTBEAT,
    NapariViewerServerRequest,
    ViewerBatchWireField,
    ViewerControlReplyHeader,
    ViewerControlReplyPayload,
    ViewerQtEnvironmentPolicy,
    ViewerProtocolStatus,
)
from openhcs.runtime.napari_streaming_handlers import (
    LayerDataPayload,
    NapariLayerHandle,
    NapariAxisPresentation,
    NapariLayerBatchDebouncePolicy,
    NapariBatchProcessorStore,
    NapariComponentGroupStore,
    NapariDimensionLayerState,
    NapariLayerSelectionAuthority,
    NapariLayerUpdateAuthority,
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
    ViewerComponentAxisSemanticsCarrier,
    ViewerComponentAxisSemantics,
    ViewerComponentAxisSemanticsAuthority,
    ViewerComponentLabelAuthority,
    ViewerComponentLayout,
    ViewerComponentMetadataNormalizer,
    ViewerComponentNameMetadata,
    ViewerComponentNameMetadataWirePayload,
    ViewerComponentSemanticRole,
    ViewerComponentValueDomainPayload,
    ViewerComponentValueParser,
    ViewerDisplayBatchContext,
    ViewerDisplayAxisDomain,
    ViewerMappingDisplayConfigInput,
    ViewerLayerAxisProjection,
    ViewerLayerAxisProjectionRequest,
    ViewerLayerAxisProjector,
    ViewerRouteComponentValueTracker,
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

    METADATA = "metadata"
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

    def value_or_default(
        self,
        field: NapariWireField | ViewerBatchWireField,
        default: NapariWireValue,
    ) -> NapariWireValue:
        if field.value in self.payload:
            return self.payload[field.value]
        return default

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

    def required_payloads(
        self,
        field: NapariWireField | ViewerBatchWireField,
    ) -> NapariBatchImagePayloads:
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


class NapariComponentMetadataPayload:
    """Coerce wire metadata into the two semantic metadata maps used by Napari."""

    @classmethod
    def component_map(
        cls,
        payload: Mapping[str, NapariWireValue],
        context: str,
    ) -> ComponentMap:
        return {
            str(component): ViewerComponentValueParser.parse(value, context=context)
            for component, value in payload.items()
        }

    @classmethod
    def component_name_metadata(
        cls,
        payload: Mapping[str, NapariWireValue],
        context: str,
    ) -> ViewerComponentNameMetadata:
        return ViewerComponentNameMetadata.from_wire_payload(
            ViewerComponentNameMetadataWirePayload.from_mapping(
                payload,
                context=context,
            ),
            context=context,
        )


class NapariComponentNameMetadataMerge:
    """Merge component-name metadata by component value without dropping old values."""

    @staticmethod
    def merge_into(
        target: ViewerComponentNameMetadata,
        incoming: ViewerComponentNameMetadata,
    ) -> None:
        target.merge(incoming)

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
    msg_type: NapariWireValue | None

    @classmethod
    def from_json_payload(
        cls,
        data: Mapping[str, NapariWireValue],
    ) -> "NapariBatchPayload":
        payload = PayloadMap(data, "Napari batch message")
        display_config = payload.required_mapping(ViewerBatchWireField.DISPLAY_CONFIG)
        value_domain = ViewerComponentValueDomainPayload.from_wire_mapping(
            payload.required_mapping(ViewerBatchWireField.COMPONENT_VALUE_DOMAIN),
            context="Napari component value domain",
        )
        return cls(
            msg_type=payload.optional(ViewerBatchWireField.TYPE),
            images=payload.required_payloads(ViewerBatchWireField.IMAGES),
            viewer_display_config=display_config,
            component_names_metadata=NapariComponentMetadataPayload.component_name_metadata(
                payload.optional_mapping(ViewerBatchWireField.COMPONENT_NAMES_METADATA),
                "Napari component-name metadata",
            ),
            component_axis_semantics=ViewerComponentAxisSemanticsAuthority.from_display_config(
                ViewerMappingDisplayConfigInput(display_config),
                value_domain,
            ),
        )


@dataclass(frozen=True)
class NapariStreamLayerContext(ViewerComponentAxisSemanticsCarrier):
    """Wire-derived routing facts for one streamed Napari payload."""

    producer: StreamProducerIdentity
    address: NapariStreamLayerAddress

    @classmethod
    def from_payload_map(
        cls,
        payload: PayloadMap,
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> "NapariStreamLayerContext":
        return cls(
            component_axis_semantics=layer_axis_projection_semantics,
            producer=StreamProducerIdentity.from_payload(
                payload.required(NapariWireField.PRODUCER_IDENTITY)
            ),
            address=NapariStreamLayerAddress(
                components=NapariComponentMetadataPayload.component_map(
                    payload.optional_mapping(NapariWireField.METADATA),
                    "Napari image component metadata",
                ),
                path=str(payload.required(NapariWireField.PATH)),
                stream_layer_data_type=StreamingDataType(
                    str(
                        payload.value_or_default(
                            NapariWireField.DATA_TYPE,
                            DEFAULT_IMAGE_DATA_TYPE,
                        )
                    )
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
        component_layout = self.component_axis_semantics.layout
        base_layer_key = build_route_key(
            producer_identity=self.producer,
            component_info=component_info,
            component_modes=component_layout.component_modes,
            component_order=component_layout.component_order,
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
            layer_key=layer_key,
            layer_title=layer_title,
            component_info=component_info,
            component_layout=component_layout,
            layer_axis_projection_semantics=self.component_axis_semantics,
            item_address=self.address.with_components(component_info),
            payload_shape_role=payload_shape_role,
        )


@dataclass(frozen=True)
class NapariStreamLayerContextCarrier:
    """Shared carrier for data already associated with a Napari stream route."""

    stream_layer_context: NapariStreamLayerContext


@dataclass(frozen=True)
class NapariImagePayload(NapariStreamLayerContextCarrier):
    """Typed view of one image/shapes message."""

    raw: Mapping[str, NapariWireValue]
    image_id: NapariWireValue | None

    @classmethod
    def from_payload(
        cls,
        image_info: Mapping[str, NapariWireValue],
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> "NapariImagePayload":
        payload = PayloadMap(image_info, "Napari image message")
        return cls(
            raw=image_info,
            image_id=payload.optional(NapariWireField.IMAGE_ID),
            stream_layer_context=NapariStreamLayerContext.from_payload_map(
                payload,
                layer_axis_projection_semantics,
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

_NAPARI_SHAPE_RASTERIZER = NapariShapeLabelRasterizer()
_COMPONENT_METADATA_NORMALIZER = ViewerComponentMetadataNormalizer()
_ACK_ERROR = ViewerProtocolStatus.ERROR.value
_ACK_SUCCESS = ViewerProtocolStatus.SUCCESS.value


@dataclass(frozen=True)
class NapariLoadedPayloadData:
    """Loaded display data for one stream payload."""

    data: LayerDataPayload


class NapariImagePayloadShapeRole(str, Enum):
    """Nominal display role for image payloads that cannot share one layer."""

    SCALAR_PLANE = "scalar_plane"
    SCALAR_STACK = "scalar_stack"
    COLOR_PLANE = "color_plane"
    COLOR_STACK = "color_stack"
    GENERIC_ND = "generic_nd"

    @classmethod
    def for_display_request(
        cls,
        request: "NapariComponentAwareDisplayRequest",
    ) -> "NapariImagePayloadShapeRole | None":
        if request.stream_layer_context.address.stream_layer_data_type is not StreamingDataType.IMAGE:
            return None

        shape = tuple(int(dimension) for dimension in np.shape(request.data))
        if len(shape) == 2:
            return cls.SCALAR_PLANE
        if len(shape) >= 3 and shape[-1] in (3, 4):
            if len(shape) == 3:
                return cls.COLOR_PLANE
            return cls.COLOR_STACK
        if len(shape) == 3:
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

    def load(self, payload: NapariImagePayload) -> NapariLoadedPayloadData:
        data_type = payload.stream_layer_context.address.stream_layer_data_type
        if data_type in self.SHAPE_LIKE_DATA_TYPES:
            return NapariLoadedPayloadData(data=payload.shapes)
        return NapariLoadedPayloadData(data=self._image_data(payload))

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
    component_layout: ViewerComponentLayout
    layer_axis_projection_semantics: ViewerComponentAxisSemantics
    item_address: NapariStreamLayerAddress
    payload_shape_role: NapariImagePayloadShapeRole | None = None


@dataclass(frozen=True)
class NapariComponentAwareDisplayRequest(NapariStreamLayerContextCarrier):
    """Request for routing one loaded payload into a Napari layer group."""

    data: LayerDataPayload
    server: "NapariViewerServer"

    @classmethod
    def from_stream_layer_context(
        cls,
        *,
        data: LayerDataPayload,
        stream_layer_context: NapariStreamLayerContext,
        server: "NapariViewerServer",
    ) -> "NapariComponentAwareDisplayRequest":
        if server is None:
            raise ValueError("Server instance required for debounced updates")
        if not stream_layer_context.address.components:
            raise ValueError(
                f"No component metadata available for path: {stream_layer_context.address.path}"
            )
        return cls(
            data=data,
            stream_layer_context=stream_layer_context,
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
            address=route.item_address,
        )

    @staticmethod
    def matching_index(
        group: list[NapariStreamLayerItem],
        route: NapariLayerRoute,
    ) -> int | None:
        for index, item in enumerate(group):
            if item.address.same_layer_slot(route.item_address):
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
            route.item_address.stream_layer_data_type,
        )
        request.server.display_pipeline.schedule_layer_update(
            route.layer_key,
            route.item_address.stream_layer_data_type,
            route.layer_axis_projection_semantics,
        )

    def _route(self, request: NapariComponentAwareDisplayRequest) -> NapariLayerRoute:
        payload_shape_role = NapariImagePayloadShapeRole.for_display_request(request)
        route = request.stream_layer_context.layer_route(
            payload_shape_role=payload_shape_role,
            layer_route_state=request.server.layer_route_state,
        )
        request.server.layer_route_state.set_title(route.layer_key, route.layer_title)
        return route

    @staticmethod
    def _log_route(
        route: NapariLayerRoute,
    ) -> None:
        logger.info(
            "🔍 NAPARI PROCESS: component_modes=%s, shape_role=%s, layer_key='%s'",
            route.component_layout.component_modes,
            route.payload_shape_role,
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
        server = request.server
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
        request: NapariComponentAwareDisplayRequest,
        layer_key: str,
    ) -> list[NapariStreamLayerItem]:
        return request.server.component_groups.items_for(layer_key)

    @staticmethod
    def _clear_group_for_replace(
        request: NapariComponentAwareDisplayRequest,
        layer_key: str,
        group: list[NapariStreamLayerItem],
    ) -> None:
        if request.server.replace_layers and group:
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
                route.item_address.stream_layer_data_type,
                route.layer_key,
                len(group),
            )
            return

        old_data_type = group[existing_index].address.stream_layer_data_type
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
    projected_axis_components = axis_projection.projected_axis_components
    component_values = axis_projection.component_values

    for item in layer_items:
        points_data = item.data
        components = item.address.components

        prepend_dims = [
            ViewerComponentCoordinateAuthority.index(
                components=components,
                component_values=component_values,
                component=component,
                context="Napari points item",
            )
            for component in projected_axis_components
        ]

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

    points_array = np.empty((0, 2 + len(projected_axis_components)))
    if all_points_nd:
        points_array = np.array(all_points_nd)
    return points_array, all_properties


def _build_nd_image_array(
    layer_items: list[NapariStreamLayerItem],
    axis_projection: ViewerLayerAxisProjection,
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
    logger.info(
        f"🔬 NAPARI PROCESS: Building nD array with axis_components={projected_axis_components}, component_values={component_values}"
    )

    first_img = layer_items[0].data
    stack_shape = (
        tuple(len(component_values[comp]) for comp in projected_axis_components)
        + first_img.shape
    )
    stacked_array = np.zeros(stack_shape, dtype=first_img.dtype)
    logger.info(
        f"🔬 NAPARI PROCESS: Created nD array with shape {stack_shape} from {len(layer_items)} items"
    )

    for img in layer_items:
        indices = tuple(
            ViewerComponentCoordinateAuthority.index(
                components=img.address.components,
                component_values=component_values,
                component=component,
                context="Napari image item",
            )
            for component in projected_axis_components
        )
        logger.debug(
            f"🔬 NAPARI PROCESS: Placing image at indices {indices}, components={img.address.components}"
        )
        stacked_array[indices] = img.data

    return stacked_array


def _create_or_update_image_layer(
    viewer,
    layers,
    route_key,
    layer_name,
    image_data,
    colormap,
    axis_labels=None,
    translate=None,
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
        translate,
    )


def _create_or_update_points_layer(
    viewer,
    layers,
    route_key,
    layer_name,
    points_data,
    properties,
    translate=None,
):
    """Create or update a Napari points layer."""
    return _NAPARI_LAYER_UPDATES.create_or_update_points(
        viewer,
        layers,
        route_key,
        layer_name,
        points_data,
        properties,
        translate,
    )

class NapariImagePayloadAxisLabelPolicy:
    """Labels payload-local image stack axes before the spatial y/x axes."""

    SPATIAL_AXIS_COUNT = 2
    COLOR_CHANNEL_COUNTS = frozenset({3, 4})

    @classmethod
    def axis_labels(cls, data: LayerDataPayload) -> tuple[str, ...]:
        shape = tuple(int(dimension) for dimension in np.shape(data))
        local_axis_count = cls.local_axis_count(shape)
        return tuple(cls.axis_label(index) for index in range(local_axis_count))

    @classmethod
    def local_axis_count(cls, shape: tuple[int, ...]) -> int:
        if len(shape) <= cls.SPATIAL_AXIS_COUNT:
            return 0

        color_axis_count = 0
        if cls.has_color_axis(shape):
            color_axis_count = 1
        return max(0, len(shape) - cls.SPATIAL_AXIS_COUNT - color_axis_count)

    @classmethod
    def has_color_axis(cls, shape: tuple[int, ...]) -> bool:
        return len(shape) >= 3 and shape[-1] in cls.COLOR_CHANNEL_COUNTS

    @staticmethod
    def axis_label(index: int) -> str:
        if index == 0:
            return "plane"
        return f"plane_{index + 1}"


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
        for component in component_layout.component_order:
            if component_layout.component_modes[component] != "slice":
                continue
            if component in component_info:
                parts.append(f"{component} {component_info[component]}")
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
    FALLBACK_STREAM_ROUTE = "fallback_stream_layer_context"
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
        axis_projection = presentation.axis_projection
        self._validate_axis_offsets(axis_projection.axis_offsets)
        axis_labels = presentation.axis_labels
        if not axis_projection.projected_axis_components:
            self.server.layer_route_state.set_dimension_state(
                presentation.layer_key,
                NapariDimensionLayerState(
                    labels={},
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
            presentation.layer_key,
            NapariDimensionLayerState(
                labels=self._dimension_labels(
                    axis_projection.projected_axis_components,
                    axis_projection.component_values,
                ),
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
    ) -> DimensionLabelMap:
        dimension_labels = {}
        label_authority = ViewerComponentLabelAuthority(self.server.component_name_metadata)
        for component in active_projected_axis_components:
            values = active_component_values[component]
            dimension_labels[component] = label_authority.axis_labels(
                component,
                values,
            )
        return dimension_labels


class NapariDimensionLabelRouteResolver:
    """Resolve which route currently owns Napari dimension labels."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server

    def resolve(self) -> NapariDimensionLabelRouteResolution:
        active_layer = self.server.viewer.layers.selection.active
        if active_layer is None:
            return self._fallback_resolution()

        route_key = self._route_key_for_layer(active_layer)
        if route_key is None:
            return NapariDimensionLabelRouteResolution(
                route_key=None,
                source=NapariDimensionLabelRouteSource.ACTIVE_NON_OPENHCS_LAYER,
            )
        return NapariDimensionLabelRouteResolution(
            route_key=route_key,
            source=NapariDimensionLabelRouteSource.SELECTED_OPENHCS_LAYER,
        )

    def _fallback_resolution(self) -> NapariDimensionLabelRouteResolution:
        route_key = self.server.layer_route_state.active_dimension_label_route
        if route_key is None:
            return NapariDimensionLabelRouteResolution(
                route_key=None,
                source=NapariDimensionLabelRouteSource.MISSING,
            )
        return NapariDimensionLabelRouteResolution(
            route_key=route_key,
            source=NapariDimensionLabelRouteSource.FALLBACK_STREAM_ROUTE,
        )

    def _route_key_for_layer(self, active_layer: NapariLayerHandle) -> str | None:
        for layer_key, layer in self.server.layer_route_state.layers.items():
            if layer is active_layer:
                return layer_key
        return None


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
        resolution = self.route_resolver.resolve()
        if resolution.source is NapariDimensionLabelRouteSource.MISSING:
            self.server.layer_route_state.set_active_dimension_label_route(layer_key)
            resolution = NapariDimensionLabelRouteResolution(
                route_key=layer_key,
                source=NapariDimensionLabelRouteSource.FALLBACK_STREAM_ROUTE,
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
            route_key = self.route_resolver.resolve().route_key
            overlay_text = ""
            if route_key is not None:
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
        current_step = self.server.viewer.dims.current_step
        label_parts = []
        for index, component in enumerate(state.stack_axes):
            if component not in state.labels or index >= len(current_step):
                continue
            labels = state.labels[component]
            if state.presentation is None:
                continue
            label_index = state.presentation.label_index(current_step[index], index)
            if 0 <= label_index < len(labels):
                label = labels[label_index]
                if label and str(label).lower() != "none":
                    label_parts.append(label)
        return " | ".join(label_parts)


class NapariDimensionLabelController:
    """Facade for the dimension-label store and viewer overlay subsystems."""

    def __init__(self, server: "NapariViewerServer") -> None:
        route_resolver = NapariDimensionLabelRouteResolver(server)
        self.store = NapariDimensionLabelStore(server)
        self.overlay = NapariDimensionLabelOverlayController(server, route_resolver)

    def setup_for_layer(self, layer_key: str) -> None:
        self.overlay.setup_for_layer(layer_key)


@dataclass(frozen=True, slots=True)
class NapariLayerTypedUpdateRequest(ViewerComponentAxisSemanticsCarrier):
    """Shared request for one data-type-specific Napari layer update."""

    layer_key: str
    layer_items: list[NapariStreamLayerItem]


class NapariLayerDisplayPipeline:
    """Owns debounced Napari layer display and update routing."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server
        self.axis_projector = ViewerLayerAxisProjector()
        self.payload_axis_policy = NapariImagePayloadAxisLabelPolicy()
        self.dimension_labels = NapariDimensionLabelController(server)
        self.layer_update_routes = {
            StreamingDataType.IMAGE: self.update_image_layer,
            StreamingDataType.SHAPES: self.update_shapes_layer,
            StreamingDataType.POINTS: self.update_points_layer,
        }

    def display_axis_projection(
        self,
        layer_key: str,
        axis_components: tuple[str, ...] | list[str],
        layer_items: list[NapariStreamLayerItem],
        component_axis_semantics: ViewerComponentAxisSemantics,
    ) -> ViewerLayerAxisProjection:
        """Return route-local coordinates projected into the viewer axis domain."""
        self.server.component_values.update(layer_key, axis_components, layer_items)
        self.server.display_axis_domain.update(axis_components, layer_items)
        route_values = self.server.component_values.domain.values_for(
            self.server.component_values._domain_key(layer_key, axis_components),
            axis_components,
        )
        viewer_values = self.server.display_axis_domain.values_for(axis_components)
        declared_values = component_axis_semantics.value_domain.required_component_values(
            axis_components
        )
        request = ViewerLayerAxisProjectionRequest.from_component_values(
            axis_components=axis_components,
            route_component_values=route_values,
            viewer_component_values=viewer_values,
            declared_component_values=declared_values,
        )
        return self.axis_projector.project(request)

    def schedule_layer_update(
        self,
        layer_key,
        data_type,
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
        self.server.layer_route_state.set_pending_update(layer_key, timer)
        logger.debug(
            "🔬 NAPARI PROCESS: Scheduled update for %s in %sms",
            layer_key,
            self.server.layer_batch_processor_debounce_policy.delay_ms,
        )

    def execute_layer_update(
        self,
        layer_key,
        data_type,
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

        wells_in_layer = set(
            item.address.components["well"]
            for item in layer_items
            if "well" in item.address.components
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
                display_payload=layer_axis_projection_semantics,
                component_names_metadata=self.server.component_name_metadata,
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
        display_payload: ViewerComponentAxisSemantics,
        component_names_metadata: ViewerComponentNameMetadata,
    ) -> None:
        component_axis_semantics = display_payload
        if component_names_metadata:
            NapariComponentNameMetadataMerge.merge_into(
                self.server.component_name_metadata,
                component_names_metadata,
            )

        items_by_type: dict[StreamingDataType, list[NapariStreamLayerItem]] = {}
        for item in items:
            data_type = item.address.stream_layer_data_type
            if isinstance(data_type, str):
                data_type = StreamingDataType(data_type)
            if data_type not in items_by_type:
                items_by_type[data_type] = []
            items_by_type[data_type].append(item)

        for data_type, typed_items in items_by_type.items():
            update_route = self.layer_update_routes[data_type]
            update_route(
                NapariLayerTypedUpdateRequest(
                    layer_key=layer_key,
                    layer_items=typed_items,
                    component_axis_semantics=component_axis_semantics,
                )
            )
            logger.info(
                "🔬 NAPARI PROCESS: Displayed %d %s item(s) in layer %s",
                len(typed_items),
                data_type.value,
                layer_key,
            )

    def update_image_layer(
        self,
        request: NapariLayerTypedUpdateRequest,
    ) -> None:
        layer_key = request.layer_key
        layer_items = request.layer_items
        axis_projection = self.display_axis_projection(
            layer_key,
            request.component_axis_semantics.layout.components_for_mode(
                "stack"
            ),
            layer_items,
            request.component_axis_semantics,
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
                        address=img_info.address,
                    )
                    logger.debug(
                        f"🔬 NAPARI PROCESS: Padded image from {img_data.shape} to {padded_data.shape}"
                    )

        logger.info(
            f"🔬 NAPARI PROCESS: Building nD data for {layer_key} from {len(layer_items)} items"
        )
        stacked_data = _build_nd_image_array(
            layer_items,
            axis_projection,
        )
        payload_axis_labels = self.payload_axis_policy.axis_labels(layer_items[0].data)
        translate = axis_projection.translate(payload_axis_labels)

        colormap = None
        color_component = (
            request.component_axis_semantics.role_policy.role_component_for_mode(
                role=ViewerComponentSemanticRole.COLOR,
                layout=request.component_axis_semantics.layout,
                mode="slice",
            )
        )
        if color_component is not None:
            first_item = layer_items[0]
            color_value = ViewerComponentCoordinateAuthority.required_value(
                first_item.address.components,
                color_component,
                context="Napari image colormap",
            )
            colormap = ChannelColormapPolicy().colormap(color_value)

        axis_labels = self.dimension_labels.store.apply(
            NapariAxisPresentation.from_projection(
                layer_key=layer_key,
                projection=axis_projection,
                payload_axis_labels=payload_axis_labels,
            )
        )

        _create_or_update_image_layer(
            self.server.viewer,
            self.server.layer_route_state.layers,
            layer_key,
            self.server.layer_route_state.title_for(layer_key),
            stacked_data,
            colormap,
            axis_labels,
            translate,
        )

        self.dimension_labels.setup_for_layer(layer_key)

    def update_shapes_layer(
        self,
        request: NapariLayerTypedUpdateRequest,
    ) -> None:
        layer_key = request.layer_key
        layer_items = request.layer_items
        logger.info(
            f"🔬 NAPARI PROCESS: Converting shapes to labels for {layer_key} from {len(layer_items)} items"
        )

        axis_projection = self.display_axis_projection(
            layer_key,
            request.component_axis_semantics.layout.components_for_mode(
                "stack"
            ),
            layer_items,
            request.component_axis_semantics,
        )
        labels_data = _NAPARI_SHAPE_RASTERIZER.rasterize(
            layer_items=layer_items,
            axis_projection=axis_projection,
        )

        axis_labels = self.dimension_labels.store.apply(
            NapariAxisPresentation.from_projection(
                layer_key=layer_key,
                projection=axis_projection,
            )
        )
        if axis_labels is not None:
            logger.info(
                "🔬 NAPARI PROCESS: Labels route %s carries layer-local axis_labels=%s",
                layer_key,
                axis_labels,
            )

        existing_layer = None
        if self.server.layer_route_state.has_layer(layer_key):
            existing_layer = self.server.layer_route_state.layer(layer_key)
        selection = NapariLayerSelectionAuthority.capture(
            self.server.viewer,
            existing_layer,
        )

        if existing_layer is not None:
            try:
                self.server.viewer.layers.remove(existing_layer)
                logger.info(
                    f"🔬 NAPARI PROCESS: Removed existing labels layer {layer_key} for recreation"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to remove existing labels layer {layer_key}: {e}"
                )

        layer_kwargs = {}
        if axis_labels is not None:
            layer_kwargs["axis_labels"] = axis_labels
        layer_kwargs["translate"] = axis_projection.translate()
        new_layer = self.server.viewer.add_labels(
            labels_data,
            name=self.server.layer_route_state.title_for(layer_key),
            **layer_kwargs,
        )
        self.server.layer_route_state.set_layer(layer_key, new_layer)
        NapariLayerSelectionAuthority.restore(
            self.server.viewer,
            selection,
            new_layer,
        )
        self.dimension_labels.setup_for_layer(layer_key)
        logger.info(
            f"🔬 NAPARI PROCESS: Created labels layer {layer_key} with shape {labels_data.shape}"
        )

    def update_points_layer(
        self,
        request: NapariLayerTypedUpdateRequest,
    ) -> None:
        layer_key = request.layer_key
        layer_items = request.layer_items
        points_items = [
            item
            for item in layer_items
            if item.address.stream_layer_data_type == StreamingDataType.POINTS
        ]

        if not points_items:
            logger.warning(
                f"🔬 NAPARI PROCESS: No POINTS items found for {layer_key}, skipping"
            )
            return

        logger.info(
            f"🔬 NAPARI PROCESS: Building points layer for {layer_key} from {len(points_items)} items (filtered from {len(layer_items)} total)"
        )

        axis_projection = self.display_axis_projection(
            layer_key,
            request.component_axis_semantics.layout.components_for_mode(
                "stack"
            ),
            layer_items,
            request.component_axis_semantics,
        )
        points_data, properties = _build_nd_points(
            points_items,
            axis_projection,
        )

        _create_or_update_points_layer(
            self.server.viewer,
            self.server.layer_route_state.layers,
            layer_key,
            self.server.layer_route_state.title_for(layer_key),
            points_data,
            properties,
            axis_projection.translate(),
        )
        self.dimension_labels.store.apply(
            NapariAxisPresentation.from_projection(
                layer_key=layer_key,
                projection=axis_projection,
            )
        )
        self.dimension_labels.setup_for_layer(layer_key)

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


class NapariScreenshotControlMessageAction(NapariControlMessageAction):
    """Registered action that captures the Napari Qt window."""

    message_type = "screenshot"

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
    def for_message_type(cls, message_type: str | None) -> "NapariStreamMessageHandler":
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

    message_type = "batch"

    def handle(
        self,
        server: "NapariViewerServer",
        data: Mapping[str, NapariWireValue],
    ) -> None:
        batch_payload = NapariBatchPayload.from_json_payload(data)
        if batch_payload.component_names_metadata:
            NapariComponentNameMetadataMerge.merge_into(
                server.component_name_metadata,
                batch_payload.component_names_metadata,
            )
            logger.info(
                "🔬 NAPARI PROCESS: Updated component metadata: %s",
                list(batch_payload.component_names_metadata),
            )

        for image_info in batch_payload.images:
            server._process_single_image(
                image_info,
                batch_payload.component_axis_semantics,
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
            "producer_identity": StreamProducerIdentity.direct(
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

        # Parse JSON message
        data = json.loads(message.decode("utf-8"))

        msg_type = PayloadMap(data, "Napari message").optional(ViewerBatchWireField.TYPE)

        NapariStreamMessageHandler.for_message_type(msg_type).handle(self, data)

        # Send reply on REP socket (required pattern)
        try:
            reply = {"status": "success", "type": msg_type}
            self.data_socket.send_json(reply)
        except Exception as e:
            logger.error(f"🔬 NAPARI PROCESS: Failed to send reply: {e}")

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
        payload_address = payload.stream_layer_context.address
        logger.info(
            f"🔍 NAPARI PROCESS: Received {payload_address.stream_layer_data_type} with metadata: {payload_address.components} (path: {payload_address.path})"
        )

        try:
            loaded = _NAPARI_PAYLOAD_DATA_LOADER.load(payload)
            request = NapariComponentAwareDisplayRequest.from_stream_layer_context(
                data=loaded.data,
                stream_layer_context=payload.stream_layer_context,
                server=self,
            )
            _NAPARI_COMPONENT_DISPLAY_COORDINATOR.display(request)
            if payload.image_id:
                self._send_ack(payload.image_id, status=_ACK_SUCCESS)

        except Exception as e:
            logger.error(
                f"🔬 NAPARI PROCESS: Failed to process {payload_address.stream_layer_data_type} {payload_address.path}: {e}",
                exc_info=True,
            )
            if payload.image_id:
                self._send_ack(payload.image_id, status=_ACK_ERROR, error=str(e))
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
