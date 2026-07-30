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
import queue
import sys
import threading
import weakref
import zmq
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import product
from numbers import Integral
from typing import ClassVar, Optional, Protocol, Sequence, TypeAlias, cast
from qtpy.QtCore import Qt, QTimer

from openhcs.core.artifacts import ObjectArtifactSubjectBinding
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from metaclass_registry import AutoRegisterMeta
from polystore.backend_registry import register_cleanup_callback
from polystore.streaming import StreamingSharedMemoryAuthority
from zmqruntime.config import TransportMode
from zmqruntime.messages import ControlMessageType, ResponseType
from zmqruntime.viewer_protocol import ViewerComponentMode, ViewerWireField
from polystore.streaming_constants import StreamingDataType
from polystore.streaming.identity import (
    FixedStreamProducerIdentityKind,
    StreamProducerDisplayNameAuthority,
    StreamProducerIdentity,
    StreamRouteKeyAuthority,
)
from polystore.streaming.receivers.napari import build_route_key
from openhcs.core.streaming_config_declarations import ViewerType
from openhcs.runtime.viewer_protocol import (
    ChannelColormapPolicy,
    NapariLayerKind,
    NapariViewerServerRequest,
    ViewerBatchMessageType,
    ViewerBatchWireField,
    ViewerControlField,
    ViewerControlMessageType,
    ViewerNavigationControlOptions,
    ViewerPayloadControlOptions,
    ViewerStateControlOptions,
    ViewerControlResponseField,
    ViewerControlReplyHeader,
    ViewerControlReplyPayload,
    ViewerDescriptorField,
    ViewerLayerField,
    ViewerPayloadField,
    ViewerPayloadSummaryField,
    ViewerComponentValueOrdering,
    ViewerProtocolStatus,
    ViewerSettlePhase,
    ViewerSettleProgress,
    ViewerQtEnvironmentPolicy,
)
from openhcs.runtime.viewer_controls import ViewerResultElementCoordinateAuthority
from openhcs.runtime.napari_streaming_handlers import (
    DimensionLabelMap,
    LayerData,
    LayerKwargValue,
    NapariLayerHandle,
    NapariShapesLayerHandle,
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
    NapariLayerSettlementState,
    NapariPendingLayerUpdate,
    NapariLayerRouteStateStore,
    NapariShapeLayerPayload,
    NapariStreamLayerAddress,
    NapariStreamLayerItem,
    NapariViewerLayerCreator,
    VisualMetadataField,
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
    ViewerComponentValueDomainPayload,
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
from zmqruntime.streaming import StreamingVisualizerServer
from zmqruntime.transport import remove_ipc_socket

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
_DEFAULT_WINDOW_SCREEN_FRACTION = 0.9
_FEATURE_TABLE_HEIGHT_FRACTION = 0.32
_DEFAULT_RESULT_SELECTION_HIGHLIGHT_THICKNESS = 4
_NAPARI_STOCK_HIGHLIGHT_COLOR = (0.0, 0.6, 1.0, 1.0)
_DEFAULT_RESULT_SELECTION_HIGHLIGHT_COLOR = (1.0, 0.82, 0.0, 1.0)

NapariWireValue: TypeAlias = (
    str | int | float | bool | np.ndarray | np.dtype | tuple | list | dict | None
)
NapariBatchImagePayloads: TypeAlias = list[Mapping[str, NapariWireValue]]
NapariComponentModeMap: TypeAlias = dict[str, str]
NapariComponentGroups: TypeAlias = dict[str, list[NapariStreamLayerItem]]


class NapariQtWindowSurface(Protocol):
    """Typed Qt main-window surface used by native result selection."""

    def isMinimized(self) -> bool:
        """Return whether the window is currently minimized."""

    def showNormal(self) -> None:
        """Restore a minimized window."""

    def show(self) -> None:
        """Show the window."""

    def raise_(self) -> None:
        """Raise the window in its native window system."""

    def activateWindow(self) -> None:
        """Request native Qt window activation."""


class NapariFeatureTableDockSurface(Protocol):
    """Typed authoritative Napari Features table dock surface."""

    def show(self) -> None:
        """Show the dock."""

    def raise_(self) -> None:
        """Raise the dock, including within a tabified dock area."""

    def window(self) -> NapariQtWindowSurface:
        """Return the dock's owning Napari Qt main window."""


class NapariRoiManagerDockSurface(Protocol):
    """Public dock surface for the installed Napari ROI Manager."""

    def show(self) -> None:
        """Show the manager when an ROI result first becomes available."""


class NapariRoiManagerWidgetSurface(Protocol):
    """Public native-layer binding seam owned by the ROI Manager plugin."""

    def connect_layer(self, layer: NapariLayerHandle) -> None:
        """Bind the exact native Shapes owner without copying its state."""


class NapariHighlightEmitterSurface(Protocol):
    """Native Napari highlight event used by selectable Shapes layers."""

    def connect(self, callback: Callable[[object], None]) -> None:
        """Connect one selection-change callback."""

    def __call__(self) -> None:
        """Request a native selection-highlight redraw."""


class NapariSelectableLayerEventsSurface(Protocol):
    """Selection events exposed by a native selectable Napari layer."""

    highlight: NapariHighlightEmitterSurface


def _apply_default_window_layout(viewer, feature_table_dock) -> None:
    """Give the image canvas and authoritative feature table useful space."""

    qt_window = feature_table_dock.window()
    available_geometry = qt_window.screen().availableGeometry()
    target_width = min(
        available_geometry.width(),
        max(
            qt_window.width(),
            round(available_geometry.width() * _DEFAULT_WINDOW_SCREEN_FRACTION),
        ),
    )
    target_height = min(
        available_geometry.height(),
        max(
            qt_window.height(),
            round(available_geometry.height() * _DEFAULT_WINDOW_SCREEN_FRACTION),
        ),
    )

    bottom_area = Qt.DockWidgetArea.BottomDockWidgetArea
    qt_window.setCorner(Qt.Corner.BottomLeftCorner, bottom_area)
    qt_window.setCorner(Qt.Corner.BottomRightCorner, bottom_area)
    qt_window.addDockWidget(bottom_area, feature_table_dock)
    viewer.window.resize(target_width, target_height)
    qt_window.resizeDocks(
        [feature_table_dock],
        [round(target_height * _FEATURE_TABLE_HEIGHT_FRACTION)],
        Qt.Orientation.Vertical,
    )


def _apply_scope_accent_styling(feature_table_dock, scope_accent_color: str) -> None:
    """Mark one viewer with the exact accent projected by its owning UI scope."""

    from qtpy.QtGui import QColor

    accent = QColor(scope_accent_color)
    if not accent.isValid():
        raise ValueError(f"Invalid scope accent color: {scope_accent_color!r}")
    canonical_color = accent.name().lower()
    qt_window = feature_table_dock.window()
    qt_window.setProperty("openhcs_scope_accent_color", canonical_color)
    existing_style = qt_window.styleSheet()
    scope_style = f"QMainWindow {{ border: 6px solid {canonical_color}; }}"
    qt_window.setStyleSheet(f"{existing_style}\n{scope_style}".strip())


class NapariWireField(str, Enum):
    """Wire keys used by Napari stream and ROI payloads."""

    CENTER = "center"
    COORDINATES = "coordinates"
    METADATA = "metadata"
    RADII = "radii"
    SNAPSHOT = "snapshot"


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
        return str(
            PayloadMap(self.payload, "Napari shape payload").required(
                ViewerBatchWireField.TYPE
            )
        )

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
class NapariAcceptedStreamItem:
    """One stream item after receiver-owned payload materialization."""

    payload: "NapariImagePayload"
    data: LayerData


@dataclass(frozen=True)
class NapariAcceptedStreamBatch:
    """Immutable transport-to-Qt handoff for one accepted stream batch."""

    payload: NapariBatchPayload
    items: tuple[NapariAcceptedStreamItem, ...]

    def dispatch_to(self, server: "NapariViewerServer") -> None:
        """Apply copied payloads to viewer state on the Qt thread."""

        if self.payload.store:
            server.component_name_metadata.merge(self.payload)
            logger.info(
                "🔬 NAPARI PROCESS: Updated component metadata: %s",
                list(self.payload),
            )

        for item in self.items:
            server._process_loaded_image(item)


@dataclass(frozen=True, slots=True)
class NapariAcceptedControlRequest:
    """Qt-bound control request received by the socket-owning pump."""

    message: Mapping[str, object]
    response_queue: queue.Queue[bytes]


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
    image_metadata: ImagePayloadMetadata
    plane_component_domain: ViewerComponentValueDomainPayload

    @classmethod
    def from_payload_map(
        cls,
        payload: PayloadMap,
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> "NapariStreamLayerContext":
        data_type_value = payload.optional(ViewerWireField.DATA_TYPE)
        if data_type_value is None:
            data_type_value = DEFAULT_IMAGE_DATA_TYPE
        source_channel_axis = payload.optional(ViewerWireField.SOURCE_CHANNEL_AXIS)
        plane_axis = payload.optional(ViewerWireField.PLANE_AXIS)
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
                stream_layer_data_type=StreamingDataType(str(data_type_value)),
            ),
            image_metadata=ImagePayloadMetadata(
                source_channel_axis=source_channel_axis,
                plane_axis=plane_axis,
                source_spatial_domain=SourceSpatialDomain.from_viewer_wire_mapping(
                    payload.payload,
                    source_label="Napari image payload",
                    value_name="Napari image payload",
                ),
            ),
            plane_component_domain=ViewerComponentValueDomainPayload.from_wire_mapping(
                payload.optional_mapping(ViewerWireField.PLANE_COMPONENT_VALUES),
                context="Napari image plane component values",
            ),
        )

    def layer_route(
        self,
        *,
        payload_layout_role: "NapariImagePayloadLayoutRole | None",
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
            if payload_layout_role is None
            else payload_layout_role.route_key(base_layer_key)
        )
        layer_title = NapariLayerTitleAuthority.disambiguate(
            title=NapariLayerTitleAuthority.title(
                producer=self.producer,
                stream_layer_data_type=self.address.stream_layer_data_type,
                component_info=component_info,
                component_layout=component_layout,
                payload_layout_role=payload_layout_role,
            ),
            producer=self.producer,
            route_key=layer_key,
            layer_route_state=layer_route_state,
        )
        return NapariLayerRoute(
            entries=self.entries,
            layout=self.layout,
            producer=self.producer,
            route_key=layer_key,
            layer_title=layer_title,
            component_info=component_info,
            item_address=self.address.with_components(component_info),
            payload_layout_role=payload_layout_role,
            image_metadata=self.image_metadata,
            plane_component_domain=self.plane_component_domain,
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
            image_metadata=stream_layer_context.image_metadata,
            plane_component_domain=stream_layer_context.plane_component_domain,
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


_COMPONENT_METADATA_NORMALIZER = ViewerComponentMetadataNormalizer()
_ACK_ERROR = ViewerProtocolStatus.ERROR.value
_ACK_SUCCESS = ViewerProtocolStatus.SUCCESS.value
NAPARI_SETTLEMENT_UPDATE_YIELD_MS = 10


class NapariImagePayloadLayoutRole(str, Enum):
    """Declared image layouts that cannot share one Napari layer."""

    SCALAR_PLANE = "scalar_plane"
    SCALAR_STACK = "scalar_stack"
    COLOR_PLANE = "color_plane"
    COLOR_STACK = "color_stack"

    @classmethod
    def for_stream_layer_context(
        cls,
        stream_layer_context: NapariStreamLayerContext,
    ) -> "NapariImagePayloadLayoutRole | None":
        if (
            stream_layer_context.address.stream_layer_data_type
            is not StreamingDataType.IMAGE
        ):
            return None
        metadata = stream_layer_context.image_metadata
        has_channel_axis = metadata.source_channel_axis is not None
        has_plane_axis = metadata.plane_axis is not None
        if has_channel_axis:
            return cls.COLOR_STACK if has_plane_axis else cls.COLOR_PLANE
        return cls.SCALAR_STACK if has_plane_axis else cls.SCALAR_PLANE

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
        try:
            return StreamingSharedMemoryAuthority.copy_sender_owned_array(
                name=payload.shm_name,
                shape=payload.image_shape,
                dtype=payload.dtype,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Shared memory {payload.shm_name} not found"
            ) from exc


@dataclass(frozen=True)
class NapariLayerRoute(ViewerComponentAxisSemantics):
    """Resolved route identity for one component-aware display item."""

    producer: StreamProducerIdentity
    route_key: str
    layer_title: str
    component_info: ComponentMap
    item_address: NapariStreamLayerAddress
    payload_layout_role: NapariImagePayloadLayoutRole | None = None
    image_metadata: ImagePayloadMetadata = field(default_factory=ImagePayloadMetadata)
    plane_component_domain: ViewerComponentValueDomainPayload = field(
        default_factory=lambda: ViewerComponentValueDomainPayload(())
    )


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
        routed_data = data
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
    def _route(
        *,
        data: LayerData,
        stream_layer_context: NapariStreamLayerContext,
        server: "NapariViewerServer",
    ) -> NapariLayerRoute:
        payload_layout_role = NapariImagePayloadLayoutRole.for_stream_layer_context(
            stream_layer_context,
        )
        route = stream_layer_context.layer_route(
            payload_layout_role=payload_layout_role,
            layer_route_state=server.layer_route_state,
        )
        server.layer_route_state.set_title(route.route_key, route.layer_title)
        return route

    @staticmethod
    def _log_route(
        route: NapariLayerRoute,
    ) -> None:
        logger.info(
            "🔍 NAPARI PROCESS: component_modes=%s, layout_role=%s, layer_key='%s'",
            route.layout.component_modes,
            route.payload_layout_role,
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
            producer=route.producer,
            address=route.item_address,
            image_metadata=route.image_metadata,
            plane_component_domain=route.plane_component_domain,
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

    @classmethod
    def title(
        cls,
        *,
        producer: StreamProducerIdentity,
        stream_layer_data_type: StreamingDataType,
        component_info: ComponentMap,
        component_layout: ViewerComponentLayout,
        payload_layout_role: NapariImagePayloadLayoutRole | None = None,
    ) -> str:
        parts = [StreamProducerDisplayNameAuthority.output_label(producer)]
        for component in component_layout.components_for_mode(
            ViewerComponentMode.SLICE
        ):
            value = ViewerComponentCoordinateAuthority.required_value(
                component_info,
                component,
                context="Napari layer title",
            )
            parts.append(f"{component} {value}")
        suffix = NapariLayerDisplayHandler.for_data_type(
            stream_layer_data_type
        ).title_suffix
        if suffix:
            parts.append(suffix)
        title = " ".join(str(part) for part in parts if part)
        if payload_layout_role is None:
            return title
        return payload_layout_role.title(title)

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
        selected_route_key = None
        active_layer = self.server.viewer.layers.selection.active
        if active_layer is not None:
            selected_route_key = self._route_key_for_layer(active_layer)
            if selected_route_key is None:
                return NapariDimensionLabelRouteResolution(
                    route_key=None,
                    source=NapariDimensionLabelRouteSource.ACTIVE_NON_OPENHCS_LAYER,
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

        if selected_route_key is not None:
            if self._route_matches_viewer_context(
                selected_route_key,
                viewer_ndim,
                current_step,
            ):
                return NapariDimensionLabelRouteResolution(
                    route_key=selected_route_key,
                    source=NapariDimensionLabelRouteSource.SELECTED_OPENHCS_LAYER,
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
        viewer_axis_origins = self.server.layer_route_state.axis_origins_for(
            axis_labels
        )
        return (
            bool(axis_labels)
            and len(axis_labels) == viewer_ndim
            and state.describes_current_step(
                current_step,
                viewer_axis_origins,
            )
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
            state = self.server.layer_route_state.dimension_state_for(layer_key)
            if self._apply_axis_labels(layer_key, state):
                self.server.layer_route_state.set_active_dimension_label_route(
                    layer_key
                )
                self.server.viewer.text_overlay.text = self._dimension_label_text(state)
                logger.info(
                    "🔬 NAPARI PROCESS: Applied route-local axis labels for %s "
                    "without a current value-label overlay.",
                    layer_key,
                )
            return
        self._apply_resolution(resolution)
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
        del event
        try:
            self._apply_resolution(self.route_resolver.resolve())
        except Exception as e:
            logger.debug(f"🔬 NAPARI PROCESS: Error updating dimension label: {e}")

    def _apply_resolution(
        self,
        resolution: NapariDimensionLabelRouteResolution,
    ) -> None:
        route_key = resolution.route_key
        overlay_text = ""
        if route_key is not None:
            self.server.layer_route_state.set_active_dimension_label_route(route_key)
            state = self.server.layer_route_state.dimension_state_for(route_key)
            self._apply_axis_labels(route_key, state)
            overlay_text = self._dimension_label_text(state)
        self.server.viewer.text_overlay.text = overlay_text

    def _apply_axis_labels(
        self,
        layer_key: str,
        state: NapariDimensionLayerState,
    ) -> bool:
        axis_labels = state.axis_labels
        if not axis_labels:
            return False
        ndim = self._viewer_ndim()
        if len(axis_labels) != ndim:
            logger.warning(
                "🔬 NAPARI PROCESS: Refusing axis_labels=%s for route %s because "
                "active viewer ndim is %s.",
                axis_labels,
                layer_key,
                ndim,
            )
            return False
        self.server.viewer.dims.axis_labels = axis_labels
        return True

    def _viewer_ndim(self) -> int:
        ndim = self.server.viewer.dims.ndim
        try:
            return int(ndim)
        except (TypeError, ValueError):
            raise TypeError(f"Napari viewer dims.ndim must be int-like, got {ndim!r}.")

    def _dimension_label_text(self, state: NapariDimensionLayerState) -> str:
        current_step = tuple(int(step) for step in self.server.viewer.dims.current_step)
        label_parts = state.label_parts_for_current_step(
            current_step,
            self.server.layer_route_state.axis_origins_for(state.axis_labels),
        )
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


class NapariLayerDisplayWork(ABC):
    """One handler-owned display operation advanced in bounded Qt work units."""

    @abstractmethod
    def advance(self) -> bool:
        """Execute one bounded unit and return whether the operation is complete."""


@dataclass(slots=True)
class NapariImmediateLayerDisplayWork(NapariLayerDisplayWork):
    """Adapt an existing atomic display handler to the bounded-work contract."""

    callback: Callable[[], None]
    complete: bool = False

    def advance(self) -> bool:
        if not self.complete:
            self.callback()
            self.complete = True
        return True


class NapariLayerDisplayHandler(
    ViewerStreamingDataTypeHandler[NapariLayerDisplayRequest],
    metaclass=ViewerStreamingDataTypeHandlerMeta,
):
    """Executable display handler for one Napari stream data type."""

    title_suffix: ClassVar[str] = ""

    def display_work(
        self,
        request: NapariLayerDisplayRequest,
    ) -> NapariLayerDisplayWork:
        """Return the handler-owned work needed to display one route."""

        return NapariImmediateLayerDisplayWork(lambda: self.handle(request))


@dataclass(frozen=True, slots=True)
class NapariImageLayerDisplayHandler(NapariLayerDisplayHandler):
    """Build or update a Napari image layer from routed image payloads."""

    streaming_data_type: ClassVar[StreamingDataType] = StreamingDataType.IMAGE

    def handle(self, request: NapariLayerDisplayRequest) -> None:
        layer_items = self._materialize_source_domains(request.items)
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
            layer_items[0].image_metadata,
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
            layer_items[0].data,
            layer_items[0].image_metadata,
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
    def _materialize_source_domains(
        layer_items: list[NapariStreamLayerItem],
    ) -> list[NapariStreamLayerItem]:
        """Place images in their declared source-image coordinate domains."""

        materialized_items: list[NapariStreamLayerItem] = []
        for item in layer_items:
            spatial_axes = item.image_metadata.spatial_axes_yx(item.data)
            source_domain = item.image_metadata.source_spatial_domain
            if spatial_axes is None or not source_domain.has_values:
                materialized_items.append(item)
                continue
            materialized_data = source_domain.materialize(
                item.data,
                spatial_axes_yx=spatial_axes,
            )
            materialized_items.append(
                replace(item, data=materialized_data)
                if materialized_data is not item.data
                else item
            )
        return materialized_items

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
                producer=img_info.producer,
                address=img_info.address,
                image_metadata=img_info.image_metadata,
                plane_component_domain=img_info.plane_component_domain,
            )
            logger.debug(
                "🔬 NAPARI PROCESS: Padded image from %s to %s",
                img_data.shape,
                padded_data.shape,
            )


@dataclass(slots=True)
class NapariShapesLayerDisplayWork(NapariLayerDisplayWork):
    """Incrementally materialize one native Shapes layer without starving Qt."""

    request: NapariLayerDisplayRequest
    payload: NapariShapeLayerPayload
    chunks: tuple[NapariShapeLayerPayload, ...]
    member_colors: list[tuple[float, float, float, float]]
    color_cycle: list[tuple[float, float, float, float]]
    common_layer_kwargs: dict[str, LayerKwargValue]
    next_chunk_index: int = 0
    next_member_index: int = 0
    layer: NapariShapesLayerHandle | None = None

    def advance(self) -> bool:
        if self.next_chunk_index >= len(self.chunks):
            return True

        chunk = self.chunks[self.next_chunk_index]
        next_member_index = self.next_member_index + len(chunk.data)
        member_colors = self.member_colors[self.next_member_index : next_member_index]
        if self.layer is None:
            layer_kwargs = dict(self.common_layer_kwargs)
            layer_kwargs.update(
                {
                    "shape_type": chunk.shape_types,
                    "features": chunk.features,
                    "edge_color": VisualMetadataField.LABEL.value,
                    "face_color": VisualMetadataField.LABEL.value,
                }
            )
            self.layer = cast(
                NapariShapesLayerHandle,
                self.request.create_or_update_layer(
                    layer_kind=NapariLayerKind.SHAPES,
                    data=chunk.data,
                    layer_kwargs=layer_kwargs,
                ),
            )
        else:
            self.layer.add(
                chunk.data,
                shape_type=chunk.shape_types,
                edge_color=member_colors,
                face_color=member_colors,
            )

        self.next_member_index = next_member_index
        self.next_chunk_index += 1
        if self.next_chunk_index < len(self.chunks):
            logger.debug(
                "🔬 NAPARI PROCESS: Materialized ROI chunk %d/%d for %s",
                self.next_chunk_index,
                len(self.chunks),
                self.request.presentation.route_key,
            )
            return False

        if self.layer is None:
            raise RuntimeError("Napari Shapes display completed without a layer.")
        self.layer.features = self.payload.features
        self.layer.edge_color_cycle = self.color_cycle
        self.layer.face_color_cycle = self.color_cycle
        if self.layer.edge_color_mode != "cycle":
            self.layer.edge_color_mode = "cycle"
        if self.layer.face_color_mode != "cycle":
            self.layer.face_color_mode = "cycle"
        route_key = self.request.presentation.route_key
        self.request.pipeline.server.bind_result_selection_layer(self.layer)
        self.request.pipeline.dimension_label_overlay.setup_for_layer(route_key)
        self.layer.visible = True
        logger.info(
            "🔬 NAPARI PROCESS: Created ROI layer %s with %d shape members in "
            "%d bounded work unit(s)",
            route_key,
            len(self.payload.data),
            len(self.chunks),
        )
        return True


@dataclass(frozen=True, slots=True)
class NapariShapesLayerDisplayHandler(NapariLayerDisplayHandler):
    """Build or update a native N-D Napari Shapes layer from routed ROIs."""

    streaming_data_type: ClassVar[StreamingDataType] = StreamingDataType.SHAPES
    title_suffix: ClassVar[str] = "ROIs"
    # Native Shapes.add() emits model and canvas events for every work unit.
    # These profiled bounds keep Qt responsive while avoiding dozens of
    # increasingly expensive redraws for ordinary high-content ROI layers.
    MAX_SHAPES_PER_WORK_UNIT: ClassVar[int] = 2_048
    MAX_VERTICES_PER_WORK_UNIT: ClassVar[int] = 65_536

    def display_work(
        self,
        request: NapariLayerDisplayRequest,
    ) -> NapariLayerDisplayWork:
        pipeline = request.pipeline
        presentation = request.presentation
        logger.info(
            "🔬 NAPARI PROCESS: Building native ROIs for %s from %d items",
            presentation.route_key,
            len(request.items),
        )

        shape_payload = NapariShapeLayerPayload.build(
            layer_items=request.items,
            axis_projection=presentation.projection,
            aggregate_axis_bindings=presentation.aggregate_axis_bindings,
        )
        member_colors = shape_payload.label_colors
        color_cycle = shape_payload.label_color_cycle
        chunks = shape_payload.chunks(
            max_shape_count=self.MAX_SHAPES_PER_WORK_UNIT,
            max_vertex_count=self.MAX_VERTICES_PER_WORK_UNIT,
        )

        axis_labels = pipeline.dimension_label_store.apply(presentation)
        if axis_labels is not None:
            logger.info(
                "🔬 NAPARI PROCESS: ROI route %s carries layer-local axis_labels=%s",
                presentation.route_key,
                axis_labels,
            )

        layer_kwargs: dict[str, LayerKwargValue] = {
            "edge_color_cycle": color_cycle,
            "face_color_cycle": color_cycle,
            "opacity": 0.7,
            "ndim": shape_payload.ndim,
            "translate": presentation.projection.translate(),
            "visible": False,
        }
        if shape_payload.result_metadata:
            layer_kwargs["metadata"] = dict(shape_payload.result_metadata)
        if axis_labels is not None:
            layer_kwargs["axis_labels"] = axis_labels
        return NapariShapesLayerDisplayWork(
            request=request,
            payload=shape_payload,
            chunks=chunks,
            member_colors=member_colors,
            color_cycle=color_cycle,
            common_layer_kwargs=layer_kwargs,
        )

    def handle(self, request: NapariLayerDisplayRequest) -> None:
        """Synchronously exhaust work for direct handler callers and tests."""

        work = self.display_work(request)
        while not work.advance():
            pass


@dataclass(frozen=True, slots=True)
class NapariPointsLayerDisplayHandler(NapariLayerDisplayHandler):
    """Build or update a Napari points layer from routed point payloads."""

    streaming_data_type: ClassVar[StreamingDataType] = StreamingDataType.POINTS
    title_suffix: ClassVar[str] = "points"

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
        self._display_work_by_route: dict[
            str,
            tuple[NapariPendingLayerUpdate, NapariLayerDisplayWork],
        ] = {}

    def clear_display_work(self) -> None:
        """Discard every deferred display continuation for a reset or shutdown."""

        self._display_work_by_route.clear()

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
        self._display_work_by_route.pop(layer_key, None)

        timer = QTimer()
        timer.setSingleShot(True)
        update = NapariPendingLayerUpdate.from_semantics(
            timer=timer,
            data_type=data_type,
            semantics=layer_axis_projection_semantics,
        )
        timer.timeout.connect(
            lambda: self.execute_scheduled_layer_update(layer_key, update)
        )
        self.server.layer_batch_processor_debounce_policy.start_timer(timer)
        self.server.layer_route_state.set_pending_update(layer_key, update)
        logger.debug(
            "🔬 NAPARI PROCESS: Scheduled update for %s in %sms",
            layer_key,
            self.server.layer_batch_processor_debounce_policy.delay_ms,
        )

    def execute_scheduled_layer_update(
        self,
        layer_key: str,
        update: NapariPendingLayerUpdate,
    ) -> None:
        """Advance one debounced route in bounded Qt callbacks."""

        if self.server.layer_route_state.pending_update_for(layer_key) is not update:
            return
        try:
            work = self._work_for_update(
                layer_key=layer_key,
                update=update,
            )
            if work.advance():
                self._complete_scheduled_work(layer_key, update)
                return
            QTimer.singleShot(
                NAPARI_SETTLEMENT_UPDATE_YIELD_MS,
                lambda: self.execute_scheduled_layer_update(layer_key, update),
            )
        except Exception as error:
            self.server.layer_route_state.record_update_error(layer_key, error)
            self._display_work_by_route.pop(layer_key, None)
            if self.server.layer_route_state.pending_update_for(layer_key) is update:
                self.server.layer_route_state.pop_pending_update(layer_key)
            logger.exception(
                "🔬 NAPARI PROCESS: Failed scheduled layer work for %s",
                layer_key,
            )
            return

    def _complete_scheduled_work(
        self,
        layer_key: str,
        update: NapariPendingLayerUpdate,
    ) -> None:
        """Release one exact debounced update after all display work completes."""

        self._display_work_by_route.pop(layer_key, None)
        if self.server.layer_route_state.pending_update_for(layer_key) is update:
            self.server.layer_route_state.pop_pending_update(layer_key)
        self.server.layer_route_state.clear_update_error(layer_key)

    def _work_for_update(
        self,
        *,
        layer_key: str,
        update: NapariPendingLayerUpdate,
    ) -> NapariLayerDisplayWork:
        """Return or prepare the exact handler-owned work for one update."""

        existing = self._display_work_by_route.get(layer_key)
        if existing is not None and existing[0] is update:
            return existing[1]
        work = self.execute_layer_update(
            layer_key,
            update.data_type,
            update,
        )
        self._display_work_by_route[layer_key] = (update, work)
        return work

    def execute_layer_update(
        self,
        layer_key: str,
        data_type: StreamingDataType,
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> NapariLayerDisplayWork:
        try:
            layer_items = self.server.component_groups.existing_items_for(layer_key)
            if layer_items is None:
                logger.warning(
                    f"🔬 NAPARI PROCESS: No items found for {layer_key}, skipping update"
                )
                return NapariImmediateLayerDisplayWork(lambda: None)
            if not layer_items:
                logger.warning(
                    f"🔬 NAPARI PROCESS: Empty item group for {layer_key}, skipping update"
                )
                return NapariImmediateLayerDisplayWork(lambda: None)

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
            work = batch_processor.add_items(
                layer_key=layer_key,
                items=layer_items,
                display_payload=layer_axis_projection_semantics,
                component_names_metadata=self.server.component_name_metadata,
            )
            if not isinstance(work, NapariLayerDisplayWork):
                raise TypeError(
                    "Napari batch display adapter must return NapariLayerDisplayWork."
                )
            self.server.layer_route_state.clear_update_error(layer_key)
            return work
        except Exception as error:
            self.server.layer_route_state.record_update_error(layer_key, error)
            logger.exception(
                "🔬 NAPARI PROCESS: Failed to update layer %s",
                layer_key,
            )
            raise

    def settlement_progress(self) -> ViewerSettleProgress:
        """Begin or observe an incremental Qt-driven settlement cycle."""

        settlement = self.server.layer_route_state.begin_settlement()
        if settlement.active_route is None:
            self._schedule_next_settlement_update(settlement)
        return settlement.progress()

    def _schedule_next_settlement_update(
        self,
        settlement: NapariLayerSettlementState,
    ) -> None:
        if settlement.failed or settlement.active_route is not None:
            return
        if settlement.completed_update_count == len(settlement.updates):
            try:
                self.server.layer_route_state.require_updates_succeeded()
            except Exception:
                settlement.fail()
            return

        claimed_update = settlement.begin_next()
        if claimed_update is None:
            return
        route_key, update = claimed_update
        QTimer.singleShot(
            NAPARI_SETTLEMENT_UPDATE_YIELD_MS,
            lambda: self._execute_settlement_update(
                settlement,
                route_key,
                update,
            ),
        )

    def _execute_settlement_update(
        self,
        settlement: NapariLayerSettlementState,
        route_key: str,
        update: NapariPendingLayerUpdate,
    ) -> None:
        """Advance one bounded route work unit and publish genuine progress."""

        try:
            settlement.begin_active_work_unit(route_key)
            work = self._work_for_update(
                layer_key=route_key,
                update=update,
            )
            if not work.advance():
                settlement.complete_active_work_unit(route_key)
                QTimer.singleShot(
                    NAPARI_SETTLEMENT_UPDATE_YIELD_MS,
                    lambda: self._execute_settlement_update(
                        settlement,
                        route_key,
                        update,
                    ),
                )
                return
        except Exception as error:
            self.server.layer_route_state.record_update_error(route_key, error)
            self._display_work_by_route.pop(route_key, None)
            logger.exception(
                "🔬 NAPARI PROCESS: Failed settlement layer work for %s",
                route_key,
            )
            settlement.fail_active(route_key)
            return
        self._display_work_by_route.pop(route_key, None)
        self.server.layer_route_state.clear_update_error(route_key)
        settlement.complete_active(route_key)
        self._schedule_next_settlement_update(settlement)

    def display_layer_batch(
        self,
        *,
        layer_key: str,
        items: list[NapariStreamLayerItem],
        display_payload: ViewerComponentAxisSemantics,
        component_names_metadata: ViewerComponentNameMetadata,
    ) -> NapariLayerDisplayWork:
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
        work = NapariLayerDisplayHandler.for_data_type(data_type).display_work(
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
            "🔬 NAPARI PROCESS: Prepared %d %s item(s) for layer %s",
            len(items),
            data_type.value,
            layer_key,
        )
        return work


class NapariMessageTypeBase(ABC):
    """Shared class-level registry key contract for message handlers."""

    __registry_key__ = "message_type"
    __skip_if_no_key__ = True
    message_type: ClassVar[str | None] = None


class NapariControlMessageAction(NapariMessageTypeBase, metaclass=AutoRegisterMeta):
    """Registered handler for one Napari control message type."""

    __registry__: ClassVar[dict[str, type["NapariControlMessageAction"]]] = {}

    @classmethod
    def for_message_type(cls, message_type: str | None) -> "NapariControlMessageAction":
        if message_type in cls.__registry__:
            return cls.__registry__[message_type]()
        return NapariUnknownControlMessageAction()

    @abstractmethod
    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, object],
    ) -> dict[str, object]:
        """Handle a control message and return the control reply."""

    def transport_thread_response(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Return a socket-thread-safe reply, or defer this action to Qt."""

        del server, message
        return None


class NapariShutdownControlMessageAction(NapariControlMessageAction):
    """Shared shutdown behavior for graceful and force shutdown requests."""

    message_type = None

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, object],
    ) -> dict[str, object]:
        del message
        logger.info("🔬 NAPARI SERVER: %s requested, closing viewer", self.message_type)
        server.request_shutdown()
        if server.viewer is not None:
            from qtpy import QtCore

            QtCore.QTimer.singleShot(100, server.viewer.close)
        return ViewerControlReplyPayload(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.SUCCESS,
                response_type=ResponseType.SHUTDOWN_ACK.value,
                message="Napari viewer shutting down",
            )
        ).to_wire_mapping()


class NapariGracefulShutdownControlMessageAction(NapariShutdownControlMessageAction):
    """Registered graceful shutdown action."""

    message_type = ControlMessageType.SHUTDOWN.value


class NapariForceShutdownControlMessageAction(NapariShutdownControlMessageAction):
    """Registered force shutdown action."""

    message_type = ControlMessageType.FORCE_SHUTDOWN.value


class NapariClearStateControlMessageAction(NapariControlMessageAction):
    """Registered action that clears accumulated streaming state."""

    message_type = ViewerControlMessageType.CLEAR_STATE.value

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, object],
    ) -> dict[str, object]:
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
        message: Mapping[str, object],
    ) -> dict[str, object]:
        del message
        unavailable = self._unavailable_response(server)
        if unavailable is not None:
            return unavailable
        return self._progress_response(
            server,
            server.display_pipeline.settlement_progress(),
        )

    def transport_thread_response(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Snapshot an existing settlement without waiting for Qt rendering."""

        del message
        unavailable = self._unavailable_response(server)
        if unavailable is not None:
            return unavailable
        progress = server.layer_route_state.existing_settlement_progress()
        if progress is None:
            return None
        return self._progress_response(server, progress)

    @staticmethod
    def _unavailable_response(
        server: "NapariViewerServer",
    ) -> dict[str, object] | None:
        if server.transport_failure is not None:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="settle_ack",
                    message=(
                        "Viewer transport failed before settlement: "
                        f"{server.transport_failure}"
                    ),
                )
            ).to_wire_mapping()
        if server.viewer is None:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="settle_ack",
                    message="Napari viewer is not available.",
                )
            ).to_wire_mapping()
        return None

    @staticmethod
    def _progress_response(
        server: "NapariViewerServer",
        progress: ViewerSettleProgress,
    ) -> dict[str, object]:
        failed = progress.phase is ViewerSettlePhase.FAILED
        failure = server.layer_route_state.update_failure_message()
        return ViewerControlReplyPayload(
            ViewerControlReplyHeader(
                (
                    ViewerProtocolStatus.ERROR
                    if failed
                    else ViewerProtocolStatus.SUCCESS
                ),
                response_type="settle_ack",
                message=(
                    f"Viewer settlement failed: {failure}"
                    if failed
                    else (
                        "Viewer settlement progress: "
                        f"{progress.completed_update_count}/"
                        f"{progress.total_update_count}."
                    )
                ),
            ),
            fields=progress.to_wire_mapping(),
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
        # Mask-derived Napari polygons can trace image pixel edges at +/-0.5.
        min_y = -0.5
        min_x = -0.5
        max_y = height - 0.5
        max_x = width - 0.5
        return (
            float(self.min_y) < min_y
            or float(self.min_x) < min_x
            or float(self.max_y) > max_y
            or float(self.max_x) > max_x
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


class NapariFeatureRows(Protocol):
    """Native Napari feature-table rows attached to a selectable layer."""

    def __len__(self) -> int:
        """Return the number of feature rows."""


class NapariSelectableFeatureLayer(Protocol):
    """Structural native contract shared by Napari Shapes and Points layers."""

    features: NapariFeatureRows
    metadata: Mapping[str, object]
    selected_data: set[int]


class NapariSelectableResultLayer(NapariSelectableFeatureLayer, Protocol):
    """Native selectable layer carrying exact N-D element coordinates."""

    data: Sequence[object]
    edge_color: object
    visible: bool
    events: NapariSelectableLayerEventsSurface


@dataclass(frozen=True, slots=True)
class NapariResultElementSelectionState:
    """Observed native feature-row selection for one mounted Napari layer."""

    supported: bool
    feature_row_count: int = 0
    selected_data_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class NapariResultSelectionGroupBinding:
    """Bind native feature rows to one declared cross-layer object subject."""

    subject_token: object
    id_feature: str

    def __post_init__(self) -> None:
        if not self.id_feature:
            raise ValueError(
                "NapariResultSelectionGroupBinding.id_feature cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class NapariResultSelectionGroupState:
    """One object subject represented by one or more native feature rows."""

    subject_token: object
    subject_id: object
    member_indices: tuple[int, ...]


class NapariResultSelectionGroupAuthority:
    """Resolve object-owned ROI groups from framework-declared feature metadata."""

    @staticmethod
    def feature_values(
        layer: NapariLayerHandle,
        feature_name: str,
    ) -> tuple[object, ...] | None:
        try:
            values = cast(Mapping[str, Sequence[object]], layer.features)[feature_name]
        except (KeyError, TypeError):
            metadata = cast(NapariSelectableFeatureLayer, layer).metadata
            metadata_values = metadata.get(feature_name)
            if (
                metadata_values is None
                or isinstance(metadata_values, (str, bytes))
                or not isinstance(metadata_values, Sequence)
            ):
                return None
            values = metadata_values
        return tuple(
            value.item() if isinstance(value, np.generic) else value for value in values
        )

    @classmethod
    def declared_binding(
        cls,
        layer: NapariLayerHandle,
    ) -> NapariResultSelectionGroupBinding | None:
        layer_metadata = cast(NapariSelectableFeatureLayer, layer).metadata
        metadata_subject = layer_metadata.get(
            ObjectArtifactSubjectBinding.SUBJECT_FEATURE
        )
        metadata_ids = cls.feature_values(
            layer,
            ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE,
        )
        if metadata_subject is not None or metadata_ids is not None:
            if metadata_subject is None or metadata_ids is None:
                raise ValueError(
                    "OpenHCS result layer metadata requires both subject and ID values."
                )
            if len(metadata_ids) != len(layer.features):
                raise ValueError(
                    "OpenHCS result layer subject IDs do not align with feature rows."
                )
            return NapariResultSelectionGroupBinding(
                subject_token=metadata_subject,
                id_feature=ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE,
            )

        return None

    @classmethod
    def state(
        cls,
        layer: NapariLayerHandle,
        binding: NapariResultSelectionGroupBinding,
        data_index: int,
    ) -> NapariResultSelectionGroupState:
        values = cls.feature_values(layer, binding.id_feature)
        if values is None:
            raise ValueError(
                f"Bound result group feature {binding.id_feature!r} is absent."
            )
        if data_index < 0 or data_index >= len(values):
            raise ValueError(
                f"Result group data index {data_index} is outside {len(values)} row(s)."
            )
        subject_id = values[data_index]
        members = tuple(
            index for index, value in enumerate(values) if value == subject_id
        )
        if not members:
            raise RuntimeError("OpenHCS result group resolved no native members.")
        return NapariResultSelectionGroupState(
            subject_token=binding.subject_token,
            subject_id=subject_id,
            member_indices=members,
        )


class NapariResultElementSelectionAuthority:
    """Own native feature-row selection without layer-kind or assay dispatch."""

    @classmethod
    def state(
        cls,
        layer: NapariLayerHandle | None,
    ) -> NapariResultElementSelectionState:
        if layer is None:
            return NapariResultElementSelectionState(supported=False)
        selectable_layer = cast(NapariSelectableFeatureLayer, layer)
        try:
            feature_row_count = len(selectable_layer.features)
            native_selected_data = tuple(selectable_layer.selected_data)
        except AttributeError:
            return NapariResultElementSelectionState(supported=False)
        if feature_row_count < 0:
            raise ValueError("Napari feature row count must be nonnegative.")
        selected_data_indices: list[int] = []
        for index in native_selected_data:
            if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
                raise TypeError(
                    "Napari selected_data must contain nonnegative integer indices."
                )
            selected_data_indices.append(int(index))
        return NapariResultElementSelectionState(
            supported=True,
            feature_row_count=feature_row_count,
            selected_data_indices=tuple(sorted(selected_data_indices)),
        )

    @classmethod
    def require_data_index(
        cls,
        layer: NapariLayerHandle,
        data_index: int,
    ) -> NapariResultElementSelectionState:
        state = cls.state(layer)
        if not state.supported:
            raise ValueError(
                "Target layer does not support native feature-bearing data selection."
            )
        if data_index >= state.feature_row_count:
            raise ValueError(
                f"Viewer data_index {data_index} is outside "
                f"{state.feature_row_count} populated feature row(s)."
            )
        return state

    @classmethod
    def select(
        cls,
        layer: NapariLayerHandle,
        data_index: int,
    ) -> NapariResultElementSelectionState:
        cls.require_data_index(layer, data_index)
        selectable_layer = cast(NapariSelectableFeatureLayer, layer)
        selectable_layer.selected_data = {data_index}
        observed = cls.state(layer)
        if observed.selected_data_indices != (data_index,):
            raise RuntimeError(
                "Napari did not retain the requested native data selection."
            )
        return observed


class NapariResultSelectionController:
    """Turn native result selection into an unambiguous visible viewer state."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server
        self._callbacks: weakref.WeakKeyDictionary[
            object,
            Callable[[object], None],
        ] = weakref.WeakKeyDictionary()
        self._observed_indices: weakref.WeakKeyDictionary[
            object,
            tuple[int, ...],
        ] = weakref.WeakKeyDictionary()
        self._group_bindings: weakref.WeakKeyDictionary[
            object,
            NapariResultSelectionGroupBinding,
        ] = weakref.WeakKeyDictionary()
        self._selection_observers: list[Callable[[], None]] = []
        self._pending_generation = 0
        self._refreshing_highlights = False
        self._synchronizing_group_selection = False
        self.ensure_default_highlight_thickness()
        self.ensure_default_highlight_color()

    @staticmethod
    def _highlight_settings():
        """Return Napari's native, Preferences-backed highlight authority."""

        from napari.settings import get_settings

        return get_settings().appearance.highlight

    def ensure_default_highlight_thickness(self) -> int:
        """Replace Napari's nearly invisible stock width while preserving choices."""

        settings = self._highlight_settings()
        if settings.highlight_thickness == 1:
            settings.highlight_thickness = _DEFAULT_RESULT_SELECTION_HIGHLIGHT_THICKNESS
        return int(settings.highlight_thickness)

    def ensure_default_highlight_color(self) -> tuple[float, float, float, float]:
        """Replace Napari's stock cyan when it would blend into result layers."""

        settings = self._highlight_settings()
        color = tuple(float(component) for component in settings.highlight_color)
        if color == _NAPARI_STOCK_HIGHLIGHT_COLOR:
            settings.highlight_color = list(_DEFAULT_RESULT_SELECTION_HIGHLIGHT_COLOR)
            color = _DEFAULT_RESULT_SELECTION_HIGHLIGHT_COLOR
        return cast(tuple[float, float, float, float], color)

    def set_highlight_thickness(self, thickness: int) -> None:
        """Set the native selected-outline width and redraw current selections."""

        if isinstance(thickness, bool) or not isinstance(thickness, int):
            raise TypeError("Napari result highlight thickness must be an integer.")
        if thickness < 1 or thickness > 10:
            raise ValueError(
                "Napari result highlight thickness must be between 1 and 10."
            )
        settings = self._highlight_settings()
        settings.highlight_thickness = thickness
        self.refresh_highlights()

    def set_highlight_color(self, color: Sequence[float]) -> None:
        """Set the native selected-outline color and redraw current selections."""

        rgba = self._normalize_rgba(color, context="result highlight")
        settings = self._highlight_settings()
        settings.highlight_color = list(rgba)
        self.refresh_highlights()

    @staticmethod
    def _normalize_rgba(
        color: Sequence[float],
        *,
        context: str,
    ) -> tuple[float, float, float, float]:
        if isinstance(color, (str, bytes)) or len(color) != 4:
            raise ValueError(f"Napari {context} color must contain RGBA values.")
        rgba = tuple(float(component) for component in color)
        if any(
            not np.isfinite(component) or component < 0 or component > 1
            for component in rgba
        ):
            raise ValueError(
                f"Napari {context} RGBA values must be finite and between 0 and 1."
            )
        return cast(tuple[float, float, float, float], rgba)

    def is_bound_result_layer(self, layer: object) -> bool:
        """Return whether a native layer owns OpenHCS result selection."""

        return layer in self._callbacks

    def result_layer_color(
        self,
        layer: NapariLayerHandle,
    ) -> tuple[float, float, float, float]:
        """Return the first native edge color for a bound result layer."""

        if not self.is_bound_result_layer(layer):
            raise ValueError("Target layer is not a bound OpenHCS result layer.")
        colors = np.asarray(
            cast(NapariSelectableResultLayer, layer).edge_color,
            dtype=float,
        )
        if colors.ndim == 1:
            color = colors
        elif colors.ndim == 2 and len(colors):
            color = colors[0]
        else:
            raise ValueError("Bound OpenHCS result layer has no native edge color.")
        return self._normalize_rgba(color, context="result layer")

    def set_result_layer_color(
        self,
        layer: NapariLayerHandle,
        color: Sequence[float],
    ) -> None:
        """Apply one native edge color to every ROI in a bound result layer."""

        if not self.is_bound_result_layer(layer):
            raise ValueError("Target layer is not a bound OpenHCS result layer.")
        rgba = self._normalize_rgba(color, context="result layer")
        cast(NapariSelectableResultLayer, layer).edge_color = list(rgba)
        self._notify_selection_observers()

    def result_group_color(
        self,
        layer: NapariLayerHandle,
    ) -> tuple[float, float, float, float]:
        """Return the native edge color for the selected object-owned ROI group."""

        state = NapariResultElementSelectionAuthority.state(layer)
        if not state.selected_data_indices:
            raise ValueError("Bound OpenHCS result layer has no selected ROI group.")
        colors = np.asarray(
            cast(NapariSelectableResultLayer, layer).edge_color,
            dtype=float,
        )
        if colors.ndim == 1:
            color = colors
        elif colors.ndim == 2 and len(colors):
            color = colors[state.selected_data_indices[0]]
        else:
            raise ValueError("Bound OpenHCS result layer has no native edge color.")
        return self._normalize_rgba(color, context="result group")

    def set_result_group_color(
        self,
        layer: NapariLayerHandle,
        color: Sequence[float],
    ) -> None:
        """Recolor every native ROI member of the selected object subject."""

        state = NapariResultElementSelectionAuthority.state(layer)
        if not state.selected_data_indices:
            raise ValueError("Bound OpenHCS result layer has no selected ROI group.")
        rgba = self._normalize_rgba(color, context="result group")
        linked = self._linked_group_members(layer, state.selected_data_indices[0])
        for candidate, member_indices in linked:
            candidate_state = NapariResultElementSelectionAuthority.state(candidate)
            result_layer = cast(NapariSelectableResultLayer, candidate)
            colors = np.asarray(result_layer.edge_color, dtype=float)
            if colors.ndim == 1:
                colors = np.broadcast_to(
                    colors,
                    (candidate_state.feature_row_count, 4),
                ).copy()
            elif colors.ndim == 2 and len(colors) == candidate_state.feature_row_count:
                colors = colors.copy()
            else:
                raise ValueError(
                    "Bound OpenHCS result layer edge colors do not align with features."
                )
            colors[np.asarray(member_indices, dtype=int)] = rgba
            result_layer.edge_color = colors
        self._notify_selection_observers()

    def connect_selection_observer(self, callback: Callable[[], None]) -> None:
        """Observe native group selection or result color changes."""

        self._selection_observers.append(callback)

    def _notify_selection_observers(self) -> None:
        for callback in tuple(self._selection_observers):
            callback()

    def refresh_highlights(self) -> None:
        """Redraw selected native layers without scheduling navigation again."""

        self._refreshing_highlights = True
        try:
            for layer in tuple(self._callbacks):
                state = NapariResultElementSelectionAuthority.state(
                    cast(NapariLayerHandle, layer)
                )
                if state.selected_data_indices:
                    cast(NapariSelectableResultLayer, layer).events.highlight()
        finally:
            self._refreshing_highlights = False

    def bind(
        self,
        layer: NapariLayerHandle,
        group_binding: NapariResultSelectionGroupBinding | None = None,
    ) -> None:
        """Bind one authoritative streamed result layer exactly once."""

        result_layer = cast(NapariSelectableResultLayer, layer)
        NapariResultElementSelectionAuthority.state(layer)
        if layer in self._callbacks:
            return
        resolved_group_binding = (
            group_binding
            if group_binding is not None
            else NapariResultSelectionGroupAuthority.declared_binding(layer)
        )
        if resolved_group_binding is not None:
            self._group_bindings[layer] = resolved_group_binding
        layer_reference = weakref.ref(layer)

        def on_highlight(_event: object = None) -> None:
            bound_layer = layer_reference()
            if bound_layer is None or self._refreshing_highlights:
                return
            self._queue_selection(cast(NapariLayerHandle, bound_layer))

        result_layer.events.highlight.connect(on_highlight)
        self._callbacks[layer] = on_highlight
        self._observed_indices[layer] = NapariResultElementSelectionAuthority.state(
            layer
        ).selected_data_indices

    def _queue_selection(self, layer: NapariLayerHandle) -> None:
        if self._synchronizing_group_selection:
            return
        state = NapariResultElementSelectionAuthority.state(layer)
        previous_indices = self._observed_indices.get(layer, ())
        self._observed_indices[layer] = state.selected_data_indices
        if state.selected_data_indices == previous_indices:
            # Shapes highlight events also report redraws, layer activation,
            # and other presentation changes. Replaying an unchanged result
            # selection here steals focus and can make another layer impossible
            # to select.
            return
        self._pending_generation += 1
        generation = self._pending_generation
        if not state.selected_data_indices:
            self._notify_selection_observers()
            return
        newly_selected = tuple(
            index
            for index in state.selected_data_indices
            if index not in previous_indices
        )
        data_index = (
            newly_selected[-1] if newly_selected else state.selected_data_indices[-1]
        )
        self._synchronize_linked_group(layer, data_index)
        layer_reference = weakref.ref(layer)
        QTimer.singleShot(
            0,
            lambda: self._apply_selection(
                layer_reference,
                data_index,
                generation,
            ),
        )

    def _linked_group_members(
        self,
        layer: NapariLayerHandle,
        data_index: int,
    ) -> tuple[tuple[NapariLayerHandle, tuple[int, ...]], ...]:
        binding = self._group_bindings.get(layer)
        if binding is None:
            return ((layer, (data_index,)),)
        source_group = NapariResultSelectionGroupAuthority.state(
            layer,
            binding,
            data_index,
        )
        linked: list[tuple[NapariLayerHandle, tuple[int, ...]]] = []
        for candidate, candidate_binding in tuple(self._group_bindings.items()):
            if candidate_binding.subject_token != source_group.subject_token:
                continue
            values = NapariResultSelectionGroupAuthority.feature_values(
                cast(NapariLayerHandle, candidate),
                candidate_binding.id_feature,
            )
            if values is None:
                continue
            member_indices = tuple(
                index
                for index, value in enumerate(values)
                if value == source_group.subject_id
            )
            if member_indices:
                linked.append((cast(NapariLayerHandle, candidate), member_indices))
        return tuple(linked) or ((layer, source_group.member_indices),)

    def _synchronize_linked_group(
        self,
        layer: NapariLayerHandle,
        data_index: int,
    ) -> tuple[tuple[NapariLayerHandle, tuple[int, ...]], ...]:
        linked = self._linked_group_members(layer, data_index)
        self._synchronizing_group_selection = True
        try:
            for candidate, member_indices in linked:
                cast(NapariSelectableFeatureLayer, candidate).selected_data = set(
                    member_indices
                )
                self._observed_indices[candidate] = member_indices
        finally:
            self._synchronizing_group_selection = False
        self._notify_selection_observers()
        return linked

    def _apply_selection(
        self,
        layer_reference: weakref.ReferenceType[object],
        data_index: int,
        generation: int,
    ) -> None:
        if generation != self._pending_generation:
            return
        layer = layer_reference()
        if layer is None or layer not in self.server.viewer.layers:
            return
        native_layer = cast(NapariLayerHandle, layer)
        state = NapariResultElementSelectionAuthority.state(native_layer)
        if data_index not in state.selected_data_indices:
            return

        result_layer = cast(NapariSelectableResultLayer, layer)
        result_layer.visible = True
        layer_selection = self.server.viewer.layers.selection
        if layer not in layer_selection or len(layer_selection) <= 1:
            # Napari defines ``active`` as "select only". Preserve an existing
            # multi-layer selection so its combined native Features Table does
            # not collapse after every row click.
            layer_selection.active = layer

        route_key = self.server.layer_route_state.route_for_layer(native_layer)
        if route_key is None:
            logger.warning(
                "Selected Napari result layer is mounted without an OpenHCS route."
            )
            return
        try:
            navigation = NapariNavigationControlMessageAction()
            axis_indices = navigation.result_element_axis_indices(
                self.server,
                native_layer,
                route_key,
                data_index,
            )
            self._refreshing_highlights = True
            try:
                if axis_indices:
                    navigation.apply_axis_indices(
                        self.server,
                        native_layer,
                        ViewerNavigationControlOptions.from_overrides(
                            route_key=route_key,
                            axis_indices=axis_indices,
                        ),
                    )
            finally:
                # Napari clears Shapes.selected_data when a dims change slices the
                # selected geometry out of view. Re-assert the same authoritative
                # member even if navigation fails so a UI interaction cannot erase
                # an otherwise valid native selection.
                try:
                    self._synchronize_linked_group(native_layer, data_index)
                finally:
                    self._refreshing_highlights = False
        except Exception:
            logger.exception(
                "Failed to navigate to selected Napari result element %d on %s",
                data_index,
                route_key,
            )


def _install_result_selection_toolbar(
    feature_table_dock: NapariFeatureTableDockSurface,
    controller: NapariResultSelectionController,
):
    """Expose native selection thickness beside the result-table workflow."""

    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QColorDialog, QLabel, QPushButton, QSpinBox, QToolBar

    qt_window = feature_table_dock.window()
    toolbar = QToolBar("OpenHCS ROI selection", qt_window)
    toolbar.setObjectName("openhcs_roi_selection_toolbar")
    label = QLabel("Selected ROI outline:", toolbar)
    thickness = QSpinBox(toolbar)
    thickness.setObjectName("openhcs_roi_highlight_thickness")
    thickness.setRange(1, 10)
    thickness.setSuffix(" px")
    thickness.setToolTip(
        "Width of Napari's native selected-ROI outline. Color remains available "
        "under Napari Preferences > Appearance > Highlight."
    )
    thickness.setValue(controller.ensure_default_highlight_thickness())
    thickness.valueChanged.connect(controller.set_highlight_thickness)
    highlight_settings = controller._highlight_settings()
    color_button = QPushButton("Selection color", toolbar)
    color_button.setObjectName("openhcs_roi_highlight_color")
    layer_color_button = QPushButton("ROI layer color", toolbar)
    layer_color_button.setObjectName("openhcs_roi_layer_color")
    group_color_button = QPushButton("ROI group color", toolbar)
    group_color_button.setObjectName("openhcs_roi_group_color")

    def sync_from_preferences(_event: object = None) -> None:
        thickness.setValue(int(highlight_settings.highlight_thickness))

    def sync_color_from_preferences(_event: object = None) -> None:
        rgba = tuple(float(value) for value in highlight_settings.highlight_color)
        color = QColor.fromRgbF(*rgba)
        foreground = "#000000" if color.lightnessF() > 0.55 else "#ffffff"
        color_button.setStyleSheet(
            "QPushButton { "
            f"background-color: {color.name()}; color: {foreground}; "
            "padding-left: 8px; padding-right: 8px; }"
        )
        color_button.setToolTip(
            f"Selected ROI outline color ({color.name()}). Also synchronized "
            "with Napari Preferences > Appearance > Highlight."
        )

    def active_result_layer() -> NapariLayerHandle | None:
        viewer = controller.server.viewer
        if viewer is None:
            return None
        layer = viewer.layers.selection.active
        if layer is None or not controller.is_bound_result_layer(layer):
            return None
        return layer

    def sync_layer_color(_event: object = None) -> None:
        layer = active_result_layer()
        layer_color_button.setEnabled(layer is not None)
        if layer is None:
            layer_color_button.setStyleSheet("")
            layer_color_button.setToolTip(
                "Select one OpenHCS ROI layer to recolor every ROI in that layer."
            )
            return
        rgba = controller.result_layer_color(layer)
        color = QColor.fromRgbF(*rgba)
        foreground = "#000000" if color.lightnessF() > 0.55 else "#ffffff"
        layer_color_button.setStyleSheet(
            "QPushButton { "
            f"background-color: {color.name()}; color: {foreground}; "
            "padding-left: 8px; padding-right: 8px; }"
        )
        layer_color_button.setToolTip(
            "Apply one edge color to every ROI in the active OpenHCS result layer."
        )

    def sync_group_color(_event: object = None) -> None:
        layer = active_result_layer()
        state = NapariResultElementSelectionAuthority.state(layer)
        has_group = layer is not None and bool(state.selected_data_indices)
        group_color_button.setEnabled(has_group)
        if layer is None or not has_group:
            group_color_button.setStyleSheet("")
            group_color_button.setToolTip(
                "Select one ROI to recolor every ROI member of its declared object group."
            )
            return
        rgba = controller.result_group_color(layer)
        color = QColor.fromRgbF(*rgba)
        foreground = "#000000" if color.lightnessF() > 0.55 else "#ffffff"
        group_color_button.setStyleSheet(
            "QPushButton { "
            f"background-color: {color.name()}; color: {foreground}; "
            "padding-left: 8px; padding-right: 8px; }"
        )
        group_color_button.setToolTip(
            "Recolor the selected object group while preserving other ROI groups."
        )

    def choose_color() -> None:
        initial = QColor.fromRgbF(
            *(float(value) for value in highlight_settings.highlight_color)
        )
        chosen = QColorDialog.getColor(
            initial,
            qt_window,
            "Selected ROI outline color",
        )
        if chosen.isValid():
            controller.set_highlight_color(
                (chosen.redF(), chosen.greenF(), chosen.blueF(), 1.0)
            )

    def choose_layer_color() -> None:
        layer = active_result_layer()
        if layer is None:
            sync_layer_color()
            return
        initial = QColor.fromRgbF(*controller.result_layer_color(layer))
        chosen = QColorDialog.getColor(
            initial,
            qt_window,
            "ROI layer color",
        )
        if chosen.isValid():
            controller.set_result_layer_color(
                layer,
                (chosen.redF(), chosen.greenF(), chosen.blueF(), 1.0),
            )
            sync_layer_color()

    def choose_group_color() -> None:
        layer = active_result_layer()
        if layer is None:
            sync_group_color()
            return
        state = NapariResultElementSelectionAuthority.state(layer)
        if not state.selected_data_indices:
            sync_group_color()
            return
        initial = QColor.fromRgbF(*controller.result_group_color(layer))
        chosen = QColorDialog.getColor(
            initial,
            qt_window,
            "ROI group color",
        )
        if chosen.isValid():
            controller.set_result_group_color(
                layer,
                (chosen.redF(), chosen.greenF(), chosen.blueF(), 1.0),
            )
            sync_group_color()

    highlight_settings.events.highlight_thickness.connect(sync_from_preferences)
    highlight_settings.events.highlight_color.connect(sync_color_from_preferences)
    color_button.clicked.connect(choose_color)
    layer_color_button.clicked.connect(choose_layer_color)
    group_color_button.clicked.connect(choose_group_color)
    controller.connect_selection_observer(sync_group_color)
    if controller.server.viewer is not None:
        controller.server.viewer.layers.selection.events.active.connect(
            sync_layer_color
        )
        controller.server.viewer.layers.selection.events.active.connect(
            sync_group_color
        )
    toolbar.addWidget(label)
    toolbar.addWidget(thickness)
    toolbar.addWidget(color_button)
    toolbar.addWidget(group_color_button)
    toolbar.addWidget(layer_color_button)
    sync_color_from_preferences()
    sync_layer_color()
    sync_group_color()
    qt_window.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)
    return toolbar


@dataclass(frozen=True, slots=True)
class NapariViewerStateProjection:
    """Project live Napari route/component stores into an agent-readable state."""

    server: "NapariViewerServer"
    viewer: NapariViewerLayerCreator
    request: ViewerStateControlOptions = field(
        default_factory=ViewerStateControlOptions
    )

    def wire_envelope(
        self,
        *,
        response_type: str,
        layers: tuple[dict[str, NapariWireValue], ...],
    ) -> dict[str, NapariWireValue]:
        return {
            ViewerControlResponseField.TYPE.value: response_type,
            ViewerControlResponseField.STATUS.value: _ACK_SUCCESS,
            ViewerControlField.VIEWER.value: {
                ViewerDescriptorField.TYPE.value: ViewerType.NAPARI.value,
                ViewerDescriptorField.TITLE.value: self.server.napari_window_title,
            },
            ViewerControlField.LAYER_COUNT.value: len(layers),
            ViewerControlField.LAYERS.value: layers,
        }

    def to_wire_mapping(self) -> dict[str, NapariWireValue]:
        route_keys = self.route_keys()
        layers = tuple(self.layer_state_for(route_key) for route_key in route_keys)
        wire_mapping = self.wire_envelope(response_type="state_ack", layers=layers)
        wire_mapping.update(
            {
                ViewerControlField.ACTIVE_DIMENSION_LABEL_ROUTE.value: (
                    self.server.layer_route_state.active_dimension_label_route
                ),
                ViewerControlField.VIEWER_NDIM.value: int(self.viewer.dims.ndim),
                ViewerControlField.CURRENT_STEP.value: tuple(
                    int(step) for step in self.viewer.dims.current_step
                ),
                ViewerControlField.AXIS_LABELS.value: tuple(
                    str(label) for label in self.viewer.dims.axis_labels
                ),
                ViewerControlField.COMPONENT_GROUP_COUNT.value: len(
                    self.server.component_groups
                ),
                ViewerControlField.COMPONENT_ITEM_COUNT.value: sum(
                    self.server.component_groups.item_count(route_key)
                    for route_key in route_keys
                ),
            }
        )
        return wire_mapping

    def route_keys(self) -> tuple[str, ...]:
        route_keys = tuple(
            dict.fromkeys(
                (
                    *self.server.layer_route_state.layer_titles,
                    *self.server.component_groups.groups,
                )
            )
        )
        route_key = self._route_key_filter()
        if route_key is None:
            return route_keys
        return tuple(candidate for candidate in route_keys if candidate == route_key)

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
        result_selection = NapariResultElementSelectionAuthority.state(layer)
        component_values = self.component_values_for(dimension_state, item_tuple)
        payload_summaries = self.payload_summaries_for(dimension_state, item_tuple)
        producer_identities = tuple(
            sorted(
                {item.producer for item in item_tuple},
                key=lambda producer: producer.output_key,
            )
        )
        return {
            ViewerLayerField.ROUTE_KEY.value: route_key,
            ViewerLayerField.PRODUCER_IDENTITIES.value: tuple(
                producer.to_payload() for producer in producer_identities
            ),
            ViewerLayerField.TITLE.value: self.layer_title(route_key),
            ViewerLayerField.MOUNTED.value: layer is not None,
            ViewerLayerField.ITEM_COUNT.value: len(item_tuple),
            ViewerLayerField.DATA_TYPES.value: tuple(
                dict.fromkeys(
                    item.address.stream_layer_data_type.value for item in item_tuple
                )
            ),
            ViewerLayerField.COMPONENT_VALUES.value: component_values,
            ViewerLayerField.COMPONENT_VALUE_COUNT.value: len(item_tuple),
            ViewerLayerField.COMPONENT_VALUES_TRUNCATED.value: (
                len(component_values) < len(item_tuple)
            ),
            ViewerLayerField.PAYLOAD_SUMMARIES.value: payload_summaries,
            ViewerLayerField.PAYLOAD_SUMMARY_COUNT.value: len(item_tuple),
            ViewerLayerField.PAYLOAD_SUMMARIES_TRUNCATED.value: (
                len(payload_summaries) < len(item_tuple)
            ),
            ViewerLayerField.AXIS_LABELS.value: dimension_state.axis_labels,
            ViewerLayerField.STACK_AXES.value: dimension_state.stack_axes,
            ViewerLayerField.AXIS_OFFSETS.value: dimension_state.axis_offsets,
            ViewerLayerField.SCALAR_LABELS.value: dimension_state.scalar_labels,
            ViewerLayerField.LABELS.value: dimension_state.labels,
            ViewerLayerField.AXIS_COMPONENT_VALUES.value: self.axis_component_values(
                dimension_state
            ),
            ViewerLayerField.ROUTED_COMPONENT_VALUES.value: (
                self.routed_component_values(dimension_state)
            ),
            ViewerLayerField.DATA_SHAPE.value: self.layer_data_shape(layer),
            ViewerLayerField.TRANSLATE.value: self.layer_translate(layer),
            ViewerLayerField.VISIBLE.value: layer_visible,
            ViewerLayerField.SELECTED.value: layer_selected,
            ViewerLayerField.FEATURE_ROW_COUNT.value: (
                result_selection.feature_row_count
            ),
            ViewerLayerField.SELECTED_DATA_INDICES.value: (
                result_selection.selected_data_indices
            ),
            ViewerLayerField.PENDING_UPDATE.value: (
                route_key in self.server.layer_route_state.layer_pending_updates
            ),
        }

    def component_values_for(
        self,
        dimension_state: NapariDimensionLayerState,
        items: tuple[NapariStreamLayerItem, ...],
    ) -> tuple[dict[str, ComponentValue], ...]:
        controls = self.state_controls()
        if not controls.include_component_values:
            return ()
        aggregate_axis_bindings = (
            dimension_state.presentation.aggregate_axis_bindings
            if dimension_state.presentation is not None
            else NapariAggregateAxisBindingSet()
        )
        return tuple(
            aggregate_axis_bindings.item_scalar_components(item)
            for item in self._bounded_items(
                items,
                controls.max_component_values_per_layer,
            )
        )

    def payload_summaries_for(
        self,
        dimension_state: NapariDimensionLayerState,
        items: tuple[NapariStreamLayerItem, ...],
    ) -> tuple[dict[str, NapariWireValue], ...]:
        controls = self.state_controls()
        if not controls.include_payload_summaries:
            return ()
        aggregate_axis_bindings = (
            dimension_state.presentation.aggregate_axis_bindings
            if dimension_state.presentation is not None
            else NapariAggregateAxisBindingSet()
        )
        return tuple(
            self.payload_summary(
                item,
                aggregate_axis_bindings.item_scalar_components(item),
                item.data,
                aggregate_axis_bindings,
            )
            for item in self._bounded_items(
                items,
                controls.max_payload_summaries_per_layer,
            )
        )

    @staticmethod
    def _bounded_items(
        items: tuple[NapariStreamLayerItem, ...],
        limit: int | None,
    ) -> tuple[NapariStreamLayerItem, ...]:
        if limit is None:
            return items
        return items[:limit]

    def state_controls(self) -> ViewerStateControlOptions:
        return self.request

    def _route_key_filter(self) -> str | None:
        route_key = self.request.route_key
        if isinstance(route_key, str):
            return route_key
        return None

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
    def payload_summary(
        cls,
        item: NapariStreamLayerItem,
        components: Mapping[str, ComponentValue],
        data: LayerData,
        aggregate_axis_bindings: NapariAggregateAxisBindingSet | None = None,
    ) -> dict[str, NapariWireValue]:
        summary: dict[str, NapariWireValue] = {
            ViewerPayloadField.DATA_TYPE.value: item.address.stream_layer_data_type.value,
            ViewerPayloadField.PATH.value: item.address.path,
            ViewerPayloadField.COMPONENTS.value: dict(components),
            "payload_type": type(data).__name__,
        }
        summary.update(
            item.image_metadata.source_spatial_domain.to_viewer_wire_mapping()
        )
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
            summary[ViewerPayloadSummaryField.NONZERO_COUNT.value] = len(data)
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
            ViewerPayloadSummaryField.SHAPE.value: tuple(
                int(axis) for axis in array.shape
            ),
            "dtype": str(array.dtype),
            "size": int(array.size),
            ViewerPayloadSummaryField.NONZERO_COUNT.value: int(np.count_nonzero(array)),
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

    def state_controls(self) -> ViewerStateControlOptions:
        return ViewerStateControlOptions.from_overrides(
            route_key=self.request.route_key,
        )

    def to_wire_mapping(self) -> dict[str, NapariWireValue]:
        layers = tuple(
            self.layer_payloads_for(route_key) for route_key in self.route_keys()
        )
        return self.wire_envelope(response_type="payloads_ack", layers=layers)

    def route_keys(self) -> tuple[str, ...]:
        route_keys = NapariViewerStateProjection.route_keys(self)
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
            ViewerLayerField.ROUTE_KEY.value: route_key,
            ViewerLayerField.PRODUCER_IDENTITIES.value: layer_state[
                ViewerLayerField.PRODUCER_IDENTITIES.value
            ],
            ViewerLayerField.TITLE.value: layer_state[ViewerLayerField.TITLE.value],
            ViewerLayerField.MOUNTED.value: layer_state[ViewerLayerField.MOUNTED.value],
            ViewerLayerField.ITEM_COUNT.value: len(item_tuple),
            ViewerLayerField.AXIS_LABELS.value: layer_state[
                ViewerLayerField.AXIS_LABELS.value
            ],
            ViewerLayerField.STACK_AXES.value: layer_state[
                ViewerLayerField.STACK_AXES.value
            ],
            ViewerLayerField.PENDING_UPDATE.value: layer_state[
                ViewerLayerField.PENDING_UPDATE.value
            ],
            ViewerLayerField.PAYLOADS.value: tuple(
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
            if not self.axis_indices_match((), dimension_state):
                return ()
            return (self.record_for_item(item, route_key, (), (), None),)

        aggregate_axis_bindings = dimension_state.presentation.aggregate_axis_bindings
        aggregate_index_tuples = self.aggregate_index_tuples(aggregate_axis_bindings)
        records: list[dict[str, NapariWireValue]] = []
        for aggregate_indices in aggregate_index_tuples:
            components = aggregate_axis_bindings.item_component_values(
                item,
                aggregate_indices,
            )
            axis_indices = dimension_state.presentation.projection.coordinate_index(
                components,
                context="Napari payload inspection",
            )
            if not self.axis_indices_match(axis_indices, dimension_state):
                continue
            records.append(
                self.record_for_item(
                    item,
                    route_key,
                    aggregate_indices,
                    axis_indices,
                    aggregate_axis_bindings,
                )
            )
        return tuple(records)

    def axis_indices_match(
        self,
        axis_indices: tuple[int, ...],
        dimension_state: NapariDimensionLayerState,
    ) -> bool:
        if self.request.axis_indices is None:
            return True
        if isinstance(self.request.axis_indices, Mapping):
            return self.semantic_axis_indices_match(
                axis_indices,
                dimension_state,
                self.request.axis_indices,
            )
        return tuple(axis_indices) == self.request.axis_indices

    @staticmethod
    def semantic_axis_indices_match(
        axis_indices: tuple[int, ...],
        dimension_state: NapariDimensionLayerState,
        semantic_axis_indices: Mapping[str, int],
    ) -> bool:
        if dimension_state.presentation is None:
            return not semantic_axis_indices
        axis_labels = dimension_state.presentation.projection.projected_axis_components
        for axis_name, axis_index in semantic_axis_indices.items():
            if axis_name not in axis_labels:
                return False
            tuple_index = axis_labels.index(axis_name)
            if (
                tuple_index >= len(axis_indices)
                or axis_indices[tuple_index] != axis_index
            ):
                return False
        return True

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

        array_values, array_value_summary = self.array_value_projection(data)
        return {
            ViewerPayloadField.ROUTE_KEY.value: route_key,
            ViewerPayloadField.DATA_TYPE.value: (
                item.address.stream_layer_data_type.value
            ),
            ViewerPayloadField.PATH.value: item.address.path,
            ViewerPayloadField.COMPONENTS.value: components,
            ViewerPayloadField.AXIS_INDICES.value: axis_indices,
            ViewerPayloadField.AGGREGATE_AXIS_INDICES.value: aggregate_indices,
            ViewerPayloadField.SUMMARY.value: NapariViewerStateProjection.payload_summary(
                item,
                components,
                data,
            ),
            ViewerPayloadField.ARRAY_VALUES.value: array_values,
            ViewerPayloadField.ARRAY_VALUE_SUMMARY.value: array_value_summary,
            ViewerPayloadField.SHAPE_PAYLOADS.value: self.shape_payloads(data),
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

    def array_value_projection(
        self,
        data: LayerData,
    ) -> tuple[tuple[NapariWireValue, ...], dict[str, NapariWireValue]]:
        if not self.request.include_array_values:
            return (), {}
        summary: dict[str, NapariWireValue] = {
            "requested": True,
            "included": False,
        }
        if not isinstance(data, np.ndarray):
            summary["omitted_reason"] = "payload_not_ndarray"
            return (), summary

        sample, slice_summary = self.array_value_sample(data)
        summary.update(slice_summary)
        summary.update(
            {
                "dtype": str(sample.dtype),
                ViewerPayloadSummaryField.SHAPE.value: tuple(
                    int(axis_size) for axis_size in sample.shape
                ),
                "size": int(sample.size),
                ViewerPayloadSummaryField.NONZERO_COUNT.value: int(
                    np.count_nonzero(sample)
                ),
            }
        )
        if sample.size > 0:
            summary["min"] = self.json_scalar(np.min(sample))
            summary["max"] = self.json_scalar(np.max(sample))
        if "omitted_reason" in summary:
            return (), summary
        if sample.size > self.request.max_array_elements:
            summary["omitted_reason"] = "max_array_elements_exceeded"
            summary["max_array_elements"] = self.request.max_array_elements
            return (), summary

        summary["included"] = True
        value = self.wire_value(sample)
        if isinstance(value, tuple):
            return value, summary
        return (value,), summary

    def array_value_sample(
        self,
        data: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, NapariWireValue]]:
        if self.request.array_slices is None:
            return data, {
                "slice_ranges": tuple((0, int(axis_size)) for axis_size in data.shape)
            }
        if len(self.request.array_slices) > data.ndim:
            if data.ndim == 0:
                sample = data.reshape((1,))[0:0]
            else:
                sample = data[tuple(slice(0, 0) for _ in data.shape)]
            return sample, {
                "slice_ranges": (),
                "requested_slice_ranges": self.request.array_slices,
                "omitted_reason": "slice_rank_exceeds_array_rank",
            }

        leading_dimension_count = data.ndim - len(self.request.array_slices)
        slices: list[slice] = []
        applied_ranges: list[tuple[int, int]] = []
        for axis_index, axis_size in enumerate(data.shape):
            slice_index = axis_index - leading_dimension_count
            if slice_index >= 0:
                start, stop = self.request.array_slices[slice_index]
                bounded_start = min(start, int(axis_size))
                bounded_stop = min(stop, int(axis_size))
            else:
                bounded_start = 0
                bounded_stop = int(axis_size)
            slices.append(slice(bounded_start, bounded_stop))
            applied_ranges.append((bounded_start, bounded_stop))
        return data[tuple(slices)], {
            "slice_ranges": tuple(applied_ranges),
            "requested_slice_ranges": self.request.array_slices,
        }

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
                {str(key): self.wire_value(value) for key, value in payload.items()}
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
        message: Mapping[str, object],
    ) -> dict[str, object]:
        if server.viewer is None:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="state_ack",
                    message="Napari viewer is not available.",
                )
            ).to_wire_mapping()

        request = message.get(ViewerControlResponseField.PAYLOAD.value)
        if not isinstance(request, ViewerStateControlOptions):
            raise TypeError(
                "Napari state control payload must be ViewerStateControlOptions."
            )
        return NapariViewerStateProjection(
            server=server,
            viewer=server.viewer,
            request=request,
        ).to_wire_mapping()


class NapariPayloadsControlMessageAction(NapariControlMessageAction):
    """Registered action that reports live payload records by layer and axis."""

    message_type = ViewerControlMessageType.PAYLOADS.value

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, object],
    ) -> dict[str, object]:
        if server.viewer is None:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="payloads_ack",
                    message="Napari viewer is not available.",
                )
            ).to_wire_mapping()

        request = message.get(ViewerControlResponseField.PAYLOAD.value)
        if not isinstance(request, ViewerPayloadControlOptions):
            raise TypeError(
                "Napari payload control payload must be ViewerPayloadControlOptions."
            )
        return NapariViewerPayloadProjection(
            server=server,
            viewer=server.viewer,
            request=request,
        ).to_wire_mapping()


class NapariNavigationControlMessageAction(NapariControlMessageAction):
    """Registered action that selects viewer layers and semantic axis indices."""

    message_type = ViewerControlMessageType.NAVIGATE.value

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, object],
    ) -> dict[str, object]:
        if server.viewer is None:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="navigation_ack",
                    message="Napari viewer is not available.",
                )
            ).to_wire_mapping()

        try:
            request = message.get(ViewerControlResponseField.PAYLOAD.value)
            if not isinstance(request, ViewerNavigationControlOptions):
                raise TypeError(
                    "Napari navigation control payload must be "
                    "ViewerNavigationControlOptions."
                )
            self._apply(server, request)
        except Exception as exc:
            return ViewerControlReplyPayload(
                ViewerControlReplyHeader(
                    ViewerProtocolStatus.ERROR,
                    response_type="navigation_ack",
                    message=str(exc),
                )
            ).to_wire_mapping()

        response = NapariViewerStateProjection(
            server=server,
            viewer=server.viewer,
            request=ViewerStateControlOptions.from_overrides(
                route_key=request.route_key,
                include_component_values=False,
                include_payload_summaries=False,
            ),
        ).to_wire_mapping()
        response[ViewerControlResponseField.TYPE.value] = "navigation_ack"
        response[ViewerControlResponseField.MESSAGE.value] = (
            "Viewer navigation applied."
        )
        return response

    def _apply(
        self,
        server: "NapariViewerServer",
        request: ViewerNavigationControlOptions,
    ) -> None:
        layer = self._mounted_layer(server, request.route_key)
        if request.data_index is not None:
            NapariResultElementSelectionAuthority.require_data_index(
                layer,
                request.data_index,
            )
            target_visible = (
                bool(layer.visible) if request.visible is None else request.visible
            )
            target_selected = (
                server.viewer.layers.selection.active is layer
                if request.selected is None
                else request.selected
            )
            if not target_visible or not target_selected:
                raise ValueError(
                    "Viewer data_index requires the target result layer to be "
                    "visible and selected."
                )
            result_axis_indices = self.result_element_axis_indices(
                server,
                layer,
                request.route_key,
                request.data_index,
            )
            conflicts = {
                axis_name: (result_index, request.axis_indices[axis_name])
                for axis_name, result_index in result_axis_indices.items()
                if axis_name in request.axis_indices
                and request.axis_indices[axis_name] != result_index
            }
            if conflicts:
                raise ValueError(
                    "Viewer data_index axis coordinates conflict with explicit "
                    f"axis_indices: {conflicts!r}."
                )
            request = replace(
                request,
                axis_indices={**result_axis_indices, **request.axis_indices},
            )
        if request.visible is not None:
            layer.visible = request.visible
        if request.axis_indices:
            self.apply_axis_indices(server, layer, request)
        if request.selected is True:
            server.viewer.layers.selection.active = layer
        elif (
            request.selected is False and server.viewer.layers.selection.active is layer
        ):
            server.viewer.layers.selection.active = None
        if request.data_index is not None:
            NapariResultElementSelectionAuthority.select(
                layer,
                request.data_index,
            )

        server.display_pipeline.dimension_label_overlay.setup_for_layer(
            request.route_key
        )
        if request.data_index is not None:
            server.raise_result_selection_surface()

    def _mounted_layer(
        self,
        server: "NapariViewerServer",
        route_key: str,
    ) -> NapariLayerHandle:
        if not server.layer_route_state.has_layer(route_key):
            raise ValueError(
                f"No Napari layer is registered for route_key {route_key!r}."
            )
        layer = server.layer_route_state.layer(route_key)
        if layer not in server.viewer.layers:
            raise ValueError(
                f"Napari layer for route_key {route_key!r} is not mounted."
            )
        return layer

    def result_element_axis_indices(
        self,
        server: "NapariViewerServer",
        layer: NapariLayerHandle,
        route_key: str,
        data_index: int,
    ) -> dict[str, int]:
        """Derive one result element's route-local slice from native geometry."""

        NapariResultElementSelectionAuthority.require_data_index(layer, data_index)
        dimension_state = server.layer_route_state.dimension_state_for(route_key)
        if not dimension_state.axis_labels:
            return {}
        result_layer = cast(NapariSelectableResultLayer, layer)
        try:
            coordinates = result_layer.data[data_index]
        except IndexError as exc:
            raise ValueError(
                f"Viewer data_index {data_index} has no native layer geometry."
            ) from exc
        return ViewerResultElementCoordinateAuthority.axis_indices(
            coordinates=cast(Sequence[object], coordinates),
            axis_labels=dimension_state.axis_labels,
            displayed_axis_count=int(server.viewer.dims.ndisplay),
        )

    def apply_axis_indices(
        self,
        server: "NapariViewerServer",
        layer: NapariLayerHandle,
        request: ViewerNavigationControlOptions,
    ) -> None:
        dimension_state = server.layer_route_state.dimension_state_for(
            request.route_key
        )
        axis_labels = dimension_state.axis_labels
        if not axis_labels:
            raise ValueError(
                f"Route {request.route_key!r} has no semantic axis labels to navigate."
            )
        current_step = [int(step) for step in server.viewer.dims.current_step]
        local_shape = NapariViewerStateProjection.layer_data_shape(layer)
        for axis_name, local_axis_index in request.axis_indices.items():
            axis_position = self._axis_position(
                axis_labels,
                axis_name,
                request.route_key,
            )
            if axis_position >= len(current_step):
                raise ValueError(
                    f"Axis {axis_name!r} is outside viewer current_step "
                    f"for route {request.route_key!r}."
                )
            self._validate_local_axis_index(
                dimension_state,
                local_shape,
                axis_name,
                axis_position,
                local_axis_index,
            )
            presentation = dimension_state.presentation
            if presentation is None:
                raise ValueError(
                    f"Route {request.route_key!r} has no axis presentation "
                    "for semantic navigation."
                )
            viewer_axis_origins = server.layer_route_state.axis_origins_for(axis_labels)
            current_step[axis_position] = presentation.viewer_step(
                local_axis_index,
                axis_position,
                viewer_axis_origin=viewer_axis_origins[axis_position],
            )
        server.viewer.dims.current_step = tuple(current_step)

    @staticmethod
    def _axis_position(
        axis_labels: tuple[str, ...],
        axis_name: str,
        route_key: str,
    ) -> int:
        if axis_name not in axis_labels:
            raise ValueError(
                f"Route {route_key!r} has no axis {axis_name!r}; "
                f"available axes are {axis_labels!r}."
            )
        return axis_labels.index(axis_name)

    @staticmethod
    def _validate_local_axis_index(
        dimension_state: NapariDimensionLayerState,
        local_shape: tuple[int, ...],
        axis_name: str,
        axis_position: int,
        local_axis_index: int,
    ) -> None:
        if axis_name in dimension_state.labels:
            local_extent = len(dimension_state.labels[axis_name])
        elif axis_position < len(local_shape):
            local_extent = local_shape[axis_position]
        else:
            return
        if local_axis_index >= local_extent:
            raise ValueError(
                f"Axis {axis_name!r} index {local_axis_index} is outside "
                f"the route-local extent {local_extent}."
            )


class NapariScreenshotControlMessageAction(NapariControlMessageAction):
    """Registered action that captures the Napari Qt window."""

    message_type = ViewerControlMessageType.SCREENSHOT.value

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, object],
    ) -> dict[str, object]:
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
        from openhcs.runtime.window_snapshot import WindowSnapshotCaptureSpec

        capture_spec = message.get(ViewerControlResponseField.PAYLOAD.value)
        if not isinstance(capture_spec, WindowSnapshotCaptureSpec):
            raise TypeError(
                "Napari screenshot control payload must be WindowSnapshotCaptureSpec."
            )
        snapshot = QtWindowSnapshotService().capture(
            QtWindowSnapshotRequest(
                widget=server.viewer.window.qt_viewer.window(),
                capture=capture_spec,
                subject_id=f"{ViewerType.NAPARI.value}_{server.port}",
                title=server.napari_window_title,
            )
        )
        return {
            ViewerControlResponseField.TYPE.value: "screenshot_ack",
            ViewerControlResponseField.STATUS.value: _ACK_SUCCESS,
            ViewerControlField.VIEWER.value: {
                ViewerDescriptorField.TYPE.value: ViewerType.NAPARI.value,
                ViewerDescriptorField.TITLE.value: server.napari_window_title,
            },
            ViewerControlField.RESOURCE.value: {
                "uri": snapshot.uri,
                "title": snapshot.title,
                "mime_type": snapshot.mime_type,
                "path": snapshot.path,
                "size_bytes": snapshot.size_bytes,
                "sha256": snapshot.sha256,
            },
            ViewerControlField.WIDTH.value: snapshot.width,
            ViewerControlField.HEIGHT.value: snapshot.height,
            ViewerControlField.SNAPSHOT.value: snapshot.capture,
        }


class NapariUnknownControlMessageAction(NapariControlMessageAction):
    """Return a protocol error for an unregistered control message type."""

    message_type = None

    def handle(
        self,
        server: "NapariViewerServer",
        message: Mapping[str, object],
    ) -> dict[str, object]:
        del server
        requested_type = message.get(ViewerControlResponseField.TYPE.value)
        return ViewerControlReplyPayload(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.ERROR,
                response_type=ResponseType.ERROR.value,
                message=f"Unsupported Napari control message: {requested_type!r}.",
            )
        ).to_wire_mapping()


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
    def accept(
        self,
        server: "NapariViewerServer",
        data: Mapping[str, NapariWireValue],
    ) -> NapariAcceptedStreamBatch:
        """Copy one decoded stream message into receiver-owned memory."""


class NapariBatchStreamMessageHandler(NapariStreamMessageHandler):
    """Registered stream handler for batched Napari payloads."""

    message_type = ViewerBatchMessageType.BATCH

    def accept(
        self,
        server: "NapariViewerServer",
        data: Mapping[str, NapariWireValue],
    ) -> NapariAcceptedStreamBatch:
        batch_payload = NapariBatchPayload.from_json_payload(data)
        return NapariAcceptedStreamBatch(
            payload=batch_payload,
            items=tuple(
                server._accept_single_image(image_info, batch_payload)
                for image_info in batch_payload.images
            ),
        )


class NapariDataTransportPump:
    """Socket-owning stream intake that never waits for Qt rendering."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None

    def start(self) -> None:
        """Bind and serve the data endpoint from its dedicated owner thread."""

        if self._thread is not None:
            raise RuntimeError("Napari data transport pump is already started.")
        self._stop_event.clear()
        self._ready_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._serve,
            name=f"napari-data-transport-{self.server.port}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=10.0):
            raise TimeoutError(
                f"Napari data transport failed to bind port {self.server.port}."
            )
        if self._startup_error is not None:
            raise RuntimeError(
                f"Napari data transport failed to start on port {self.server.port}."
            ) from self._startup_error

    def stop(self) -> None:
        """Stop intake and wait for the socket-owning thread to release it."""

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(
                    "Napari data transport thread on port %s did not stop promptly",
                    self.server.port,
                )
        self._thread = None

    def _serve(self) -> None:
        context = zmq.Context()
        socket = None
        try:
            socket = self.server.bind_data_socket(context)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            self._ready_event.set()
            logger.info(
                "🔬 NAPARI PROCESS: Data transport pump bound %s",
                self.server.data_transport_url(),
            )
            while not self._stop_event.is_set() and self.server.is_running():
                if socket not in dict(poller.poll(timeout=50)):
                    continue
                message = socket.recv()
                reply = self.server.accept_stream_message(message)
                socket.send_json(reply.to_wire_mapping())
        except Exception as error:
            if not self._ready_event.is_set():
                self._startup_error = error
                self._ready_event.set()
            elif not self._stop_event.is_set():
                logger.exception("🔬 NAPARI PROCESS: Data transport pump failed")
                self.server.record_transport_failure(error)
        finally:
            if socket is not None:
                socket.close(linger=0)
            context.term()


class NapariControlTransportPump:
    """Socket owner that keeps typed settlement observable while Qt renders."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None

    def start(self) -> None:
        """Bind and serve the control endpoint from its dedicated owner thread."""

        if self._thread is not None:
            raise RuntimeError("Napari control transport pump is already started.")
        self._stop_event.clear()
        self._ready_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._serve,
            name=f"napari-control-transport-{self.server.control_port}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=10.0):
            raise TimeoutError(
                "Napari control transport failed to bind port "
                f"{self.server.control_port}."
            )
        if self._startup_error is not None:
            raise RuntimeError(
                "Napari control transport failed to start on port "
                f"{self.server.control_port}."
            ) from self._startup_error

    def stop(self) -> None:
        """Stop control intake and wait for the socket-owning thread."""

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(
                    "Napari control transport thread on port %s did not stop promptly",
                    self.server.control_port,
                )
        self._thread = None

    def _serve(self) -> None:
        context = zmq.Context()
        socket = None
        try:
            socket = self.server.bind_control_socket(context)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            self._ready_event.set()
            logger.info(
                "🔬 NAPARI PROCESS: Control transport pump bound %s",
                self.server.control_transport_url(),
            )
            while not self._stop_event.is_set() and self.server.is_running():
                if socket not in dict(poller.poll(timeout=50)):
                    continue
                payload = self._response_payload(socket.recv())
                socket.send(payload)
        except Exception as error:
            if not self._ready_event.is_set():
                self._startup_error = error
                self._ready_event.set()
            elif not self._stop_event.is_set():
                logger.exception("🔬 NAPARI PROCESS: Control transport pump failed")
                self.server.record_transport_failure(error)
        finally:
            if socket is not None:
                socket.close(linger=0)
            context.term()

    def _response_payload(self, request_payload: bytes) -> bytes:
        """Dispatch a control request to its declared thread owner."""

        try:
            message = pickle.loads(request_payload)
            if not isinstance(message, Mapping):
                raise TypeError("Napari control request must decode to a mapping.")
        except Exception as error:
            return self.server.serialize_control_response(
                self.server.control_error_response(error)
            )

        msg_type = message.get(ViewerControlResponseField.TYPE.value)
        if msg_type == ControlMessageType.PING.value:
            return self.server.control_response_payload(message)
        action = NapariControlMessageAction.for_message_type(
            msg_type if isinstance(msg_type, str) else None
        )
        transport_response = action.transport_thread_response(
            self.server,
            message,
        )
        if transport_response is not None:
            return self.server.serialize_control_response(transport_response)

        response_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.server.accepted_control_requests.put(
            NapariAcceptedControlRequest(message, response_queue)
        )
        while True:
            try:
                return response_queue.get(timeout=0.05)
            except queue.Empty:
                if self._stop_event.is_set() or not self.server.is_running():
                    return self.server.serialize_control_response(
                        self.server.control_error_response(
                            RuntimeError(
                                "Napari viewer stopped before completing its "
                                "Qt-bound control request."
                            )
                        )
                    )


class NapariViewerServer(StreamingVisualizerServer):
    """
    ZMQ server for Napari viewer that receives images from clients.

    Inherits from ZMQServer ABC to get ping/pong, port management, etc.
    Uses a REP socket so each streamed payload retains acknowledgement semantics.
    """

    _server_type = ViewerType.NAPARI.value

    def __init__(self, request: NapariViewerServerRequest):
        """
        Initialize Napari viewer server.

        Args:
            request: Typed Napari server construction request.
        """
        # Initialize with REP socket for receiving images (synchronous request/reply)
        # REP socket forces workers to wait for acknowledgment before closing shared memory
        super().__init__(
            request.port,
            viewer_type=ViewerType.NAPARI.value,
            host="*",
            log_file_path=request.log_file_path,
            data_socket_type=zmq.REP,
            transport_mode=request.transport_mode,
            config=OPENHCS_ZMQ_CONFIG,
        )

        self.napari_window_title = request.viewer_title
        self.replace_layers = request.replace_layers
        self.viewer = None
        self.feature_table_dock: NapariFeatureTableDockSurface | None = None
        self.roi_manager_dock: NapariRoiManagerDockSurface | None = None
        self.roi_manager_widget: NapariRoiManagerWidgetSurface | None = None
        self.result_selection_toolbar = None
        self.layer_route_state = NapariLayerRouteStateStore.empty()
        self.component_groups = NapariComponentGroupStore()
        self.component_name_metadata = ViewerComponentNameMetadata.empty()

        self.component_values = ViewerRouteComponentValueTracker()
        self.display_axis_domain = ViewerDisplayAxisDomain()
        # Debouncing + locking for layer updates to prevent race conditions
        self.layer_update_lock = threading.Lock()  # Prevent concurrent updates
        self.layer_batch_processor_debounce_policy = NapariLayerBatchDebouncePolicy()
        self.batch_processors = NapariBatchProcessorStore(
            debounce_policy=self.layer_batch_processor_debounce_policy,
        )
        self.display_pipeline = NapariLayerDisplayPipeline(self)
        self.result_selection_controller = NapariResultSelectionController(self)
        self.accepted_stream_batches: queue.Queue[NapariAcceptedStreamBatch] = (
            queue.Queue()
        )
        self.accepted_control_requests: queue.Queue[NapariAcceptedControlRequest] = (
            queue.Queue()
        )
        self.data_transport_pump = NapariDataTransportPump(self)
        self.control_transport_pump = NapariControlTransportPump(self)
        self.transport_failure: Exception | None = None

        # Ack socket handled by StreamingVisualizerServer

    def raise_result_selection_surface(self) -> None:
        """Make the authoritative feature table and Napari window prominent."""

        feature_table_dock = self.feature_table_dock
        if feature_table_dock is None:
            raise RuntimeError("Napari Features table dock is not available.")
        feature_table_dock.show()
        feature_table_dock.raise_()
        qt_window = feature_table_dock.window()
        if qt_window.isMinimized():
            qt_window.showNormal()
        qt_window.show()
        qt_window.raise_()
        qt_window.activateWindow()

    def bind_result_selection_layer(self, layer: NapariLayerHandle) -> None:
        """Bind native selection behavior to one authoritative streamed layer."""

        self.result_selection_controller.bind(layer)
        self.bind_roi_manager_layer(layer)

    def bind_roi_manager_layer(self, layer: NapariLayerHandle) -> None:
        """Lazily mount one Fiji-style manager on the native Shapes owner."""

        if self.viewer is None:
            raise RuntimeError("Napari viewer is not available.")
        if self.roi_manager_widget is None:
            dock, widget = self.viewer.window.add_plugin_dock_widget(
                "openhcs",
                "OpenHCS ROI Manager",
            )
            self.roi_manager_dock = dock
            self.roi_manager_widget = cast(NapariRoiManagerWidgetSurface, widget)
        self.roi_manager_widget.connect_layer(layer)
        if self.roi_manager_dock is None:
            raise RuntimeError("Napari ROI Manager dock is not available.")
        self.roi_manager_dock.show()

    def start(self) -> None:
        """Bind each ZMQ endpoint in its dedicated socket-owner thread."""

        with self._lock:
            if self._running:
                return
            self._running = True

        try:
            self.control_transport_pump.start()
            self.data_transport_pump.start()
        except Exception:
            self.stop()
            raise
        logger.info(
            "ZMQ Server started on %s (REP), control %s",
            self.data_transport_url(),
            self.control_transport_url(),
        )

    def stop(self) -> None:
        """Stop each socket from the same thread that created it."""

        with self._lock:
            self._running = False
        self.data_transport_pump.stop()
        self.control_transport_pump.stop()
        with self._lock:
            if self.transport_mode is TransportMode.IPC:
                remove_ipc_socket(self.port, self.config)
                remove_ipc_socket(self.control_port, self.config)
        logger.info("ZMQ Server stopped")

    def record_transport_failure(self, error: Exception) -> None:
        """Retain a terminal intake failure for control-plane diagnostics."""

        self.transport_failure = error

    def request_shutdown(self) -> None:
        """Stop accepting display work and cancel every deferred layer update."""

        super().request_shutdown()
        pending_updates = self.layer_route_state.drain_pending_updates()
        self.display_pipeline.clear_display_work()
        if pending_updates:
            logger.info(
                "🔬 NAPARI SERVER: Cancelled %d pending layer update(s) for shutdown",
                len(pending_updates),
            )

    def display_layer_batch(
        self,
        *,
        layer_key: str,
        items: list[NapariStreamLayerItem],
        display_payload: ViewerComponentAxisSemantics,
        component_names_metadata: ViewerComponentNameMetadata,
    ) -> NapariLayerDisplayWork:
        """Display one debounced batch through the composed display pipeline."""
        return self.display_pipeline.display_layer_batch(
            layer_key=layer_key,
            items=items,
            display_payload=display_payload,
            component_names_metadata=component_names_metadata,
        )

    def clear_accumulated_stream_state(self) -> None:
        """Reset stream domains that must not leak across pipeline executions."""
        self.layer_route_state.reset_settlement()
        pending_updates = self.layer_route_state.drain_pending_updates()
        self.display_pipeline.clear_display_work()
        if pending_updates:
            logger.info(
                "🔬 NAPARI SERVER: Cancelled %d pending layer update(s) before "
                "clearing stream state",
                len(pending_updates),
            )
        self.component_groups.clear()
        self.component_values = ViewerRouteComponentValueTracker()
        self.display_axis_domain = ViewerDisplayAxisDomain()
        self.component_name_metadata.clear()
        self.layer_route_state.clear_update_errors()
        self.batch_processors = NapariBatchProcessorStore(
            debounce_policy=self.layer_batch_processor_debounce_policy,
        )

    def handle_control_message(
        self,
        message: Mapping[str, object],
    ) -> dict[str, object]:
        """
        Handle control messages beyond ping/pong.

        Supported message types:
        - shutdown: Graceful shutdown (closes viewer)
        - force_shutdown: Force shutdown (same as shutdown for Napari)
        - clear_state: Clear accumulated component groups (for new pipeline runs)
        """
        msg_type = message.get(ViewerControlResponseField.TYPE.value)
        if not isinstance(msg_type, str):
            msg_type = None
        return NapariControlMessageAction.for_message_type(msg_type).handle(
            self,
            message,
        )

    def display_image(self, image_data: np.ndarray, metadata: ComponentMap) -> None:
        """Display a single image payload (best-effort helper)."""
        image_info = {
            ViewerWireField.PATH.value: DEFAULT_DIRECT_IMAGE_PATH,
            ViewerWireField.DATA.value: image_data,
            ViewerWireField.SHAPE.value: image_data.shape,
            ViewerWireField.DTYPE.value: image_data.dtype,
            ViewerWireField.METADATA.value: metadata,
            ViewerWireField.PRODUCER_IDENTITY.value: StreamProducerIdentity.fixed_output(
                FixedStreamProducerIdentityKind.DIRECT,
                "direct_image",
            ).to_payload(),
        }
        self._process_single_image(
            image_info,
            ViewerComponentAxisSemanticsAuthority.empty(),
        )

    def accept_stream_message(self, message: bytes) -> NapariStreamMessageReply:
        """Copy one wire batch and enqueue it before acknowledging ownership."""

        import json

        msg_type: ViewerBatchMessageType | None = None
        try:
            data = json.loads(message.decode("utf-8"))
            msg_type = ViewerBatchMessageType(
                str(
                    PayloadMap(data, "Napari message").required(
                        ViewerBatchWireField.TYPE
                    )
                )
            )
            accepted_batch = NapariStreamMessageHandler.for_message_type(
                msg_type
            ).accept(self, data)
            self.accepted_stream_batches.put(accepted_batch)
            return NapariStreamMessageReply.success(msg_type)
        except Exception as error:
            logger.error(
                "🔬 NAPARI PROCESS: Failed to accept stream message: %s",
                error,
                exc_info=True,
            )
            return NapariStreamMessageReply.failure(msg_type, str(error))

    def process_accepted_stream_messages(self) -> int:
        """Drain receiver-owned batches into viewer state on the Qt thread."""

        processed_count = 0
        while True:
            try:
                accepted_batch = self.accepted_stream_batches.get_nowait()
            except queue.Empty:
                return processed_count
            accepted_batch.dispatch_to(self)
            processed_count += 1

    def process_messages(self) -> None:
        """Drain control actions whose registered owners require the Qt thread."""

        while True:
            try:
                request = self.accepted_control_requests.get_nowait()
            except queue.Empty:
                return
            request.response_queue.put(self.control_response_payload(request.message))

    def _accept_single_image(
        self,
        image_info: Mapping[str, NapariWireValue],
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> NapariAcceptedStreamItem:
        """Materialize one payload without reading or mutating Qt state."""

        payload = NapariImagePayload.from_payload(
            image_info,
            layer_axis_projection_semantics,
        )
        loaded_data = _NAPARI_PAYLOAD_DATA_LOADER.load(payload)
        return NapariAcceptedStreamItem(payload=payload, data=loaded_data)

    def _process_single_image(
        self,
        image_info: Mapping[str, NapariWireValue],
        layer_axis_projection_semantics: ViewerComponentAxisSemantics,
    ) -> None:
        """Materialize and display one direct payload on the Qt thread."""

        self._process_loaded_image(
            self._accept_single_image(
                image_info,
                layer_axis_projection_semantics,
            )
        )

    def _process_loaded_image(self, item: NapariAcceptedStreamItem) -> None:
        """Route one receiver-owned payload into deferred Napari display."""

        payload = item.payload
        payload_address = item.payload.address
        logger.info(
            f"🔍 NAPARI PROCESS: Received {payload_address.stream_layer_data_type} with metadata: {payload_address.components} (path: {payload_address.path})"
        )

        try:
            if isinstance(item.data, np.ndarray):
                logger.info(
                    "🔬 STREAM RECEIVE: path=%s components=%s %s",
                    payload_address.path,
                    payload_address.components,
                    NapariViewerStateProjection.array_summary(item.data),
                )
            _NAPARI_COMPONENT_DISPLAY_COORDINATOR.display(
                data=item.data,
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
    log_file_path: str | None = None,
    transport_mode: TransportMode = TransportMode.IPC,
    scope_accent_color: str | None = None,
) -> None:
    """
    Napari viewer process entry point. Runs in a separate process.
    Listens for ZeroMQ messages with image data to display.

    Args:
        port: ZMQ port to listen on
        viewer_title: Title for the napari viewer window
        replace_layers: If True, replace existing layers; if False, add new layers with unique names
        log_file_path: Path to log file (for client discovery via ping/pong)
        transport_mode: ZMQ transport mode (IPC or TCP)
        scope_accent_color: Exact UI-owned scope accent used to frame this window
    """
    server: NapariViewerServer | None = None
    try:
        request = NapariViewerServerRequest(
            port=port,
            viewer_title=viewer_title,
            replace_layers=replace_layers,
            log_file_path=log_file_path,
            transport_mode=transport_mode,
        )

        # Create ZMQ server instance (inherits from ZMQServer ABC)
        server = NapariViewerServer(request)

        # OpenCV wheels can replace Qt's platform-plugin path when imported.
        # Reassert the active binding's authoritative path after module imports
        # and immediately before native Qt/Napari construction.
        ViewerQtEnvironmentPolicy().apply_to(os.environ)

        # Create napari viewer in this process (main thread)
        viewer = napari.Viewer(title=viewer_title, show=True)
        server.viewer = viewer
        feature_table_dock, _feature_table = viewer.window.add_plugin_dock_widget(
            "napari",
            "Features table widget",
        )
        server.feature_table_dock = feature_table_dock
        _apply_default_window_layout(viewer, feature_table_dock)
        server.result_selection_toolbar = _install_result_selection_toolbar(
            feature_table_dock,
            server.result_selection_controller,
        )
        if scope_accent_color is not None:
            _apply_scope_accent_styling(feature_table_dock, scope_accent_color)
        logger.info("🔬 NAPARI PROCESS: Qt viewer construction complete")

        # Initialize layers dictionary with existing layers (for reconnection scenarios)
        for layer in viewer.layers:
            server.layer_route_state.set_layer(layer.name, layer)

        # Enable text overlay for dimension labels
        viewer.text_overlay.visible = True
        viewer.text_overlay.color = "white"
        viewer.text_overlay.font_size = 14

        # Use proper Qt event loop integration
        from qtpy import QtWidgets

        # Get the Qt application
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        # Ensure the application DOES quit when the napari window closes
        app.setQuitOnLastWindowClosed(True)

        # Set up a QTimer for message processing
        message_service_started = False

        def process_messages() -> None:
            nonlocal message_service_started
            if not message_service_started:
                message_service_started = True
                logger.info("🔬 NAPARI PROCESS: First Qt message-service callback")

            # The socket-owning transport thread has already copied shared-memory
            # payloads. Drain those immutable batches before control settlement so
            # a SETTLE request cannot overtake accepted display work.
            server.process_accepted_stream_messages()

            # Process control messages (ping/pong handled by ABC) on Qt because
            # viewer state and navigation remain Qt-owned.
            server.process_messages()

        # Do not publish transport endpoints until Qt has dispatched at least one
        # event and the recurring service timer is live. Endpoint presence must
        # never advertise a server that the event loop cannot yet service.
        message_timer: QTimer | None = None
        message_service_start_error: Exception | None = None

        def start_message_service() -> None:
            nonlocal message_timer, message_service_start_error
            timer = QTimer(app)
            try:
                timer.timeout.connect(process_messages)
                timer.start(50)
                message_timer = timer
                server.start()
                logger.info("🔬 NAPARI PROCESS: ZMQ endpoints bound")
            except Exception as error:
                message_service_start_error = error
                timer.stop()
                message_timer = None
                logger.exception(
                    "🔬 NAPARI PROCESS: Failed to start Qt message service"
                )
                app.quit()

        QTimer.singleShot(0, start_message_service)

        logger.info("🔬 NAPARI PROCESS: Starting Qt event loop")

        # Run the Qt event loop - this keeps napari responsive
        app.exec_()
        if message_timer is not None:
            message_timer.stop()
        if message_service_start_error is not None:
            raise RuntimeError(
                "Napari Qt message service failed during startup."
            ) from message_service_start_error

    except Exception:
        logger.exception("🔬 NAPARI PROCESS: Fatal error")
        raise
    finally:
        logger.info("🔬 NAPARI PROCESS: Shutting down")
        if server is not None:
            server.stop()
