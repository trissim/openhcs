"""
Napari-based real-time visualization module.

This module provides the NapariStreamVisualizer class for real-time
visualization of tensors during pipeline execution.
"""

import logging
import multiprocessing
import os
import pickle
import sys
import threading
import time
import zmq
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from qtpy.QtCore import QTimer

from openhcs.core.config import TransportMode as OpenHCSTransportMode
from polystore.backend_registry import register_cleanup_callback
from zmqruntime.config import TransportMode, ZMQConfig
from polystore.streaming_constants import StreamingDataType
from polystore.streaming.receivers.napari import (
    normalize_component_layout,
    build_layer_key,
)
from openhcs.runtime.viewer_protocol import (
    ChannelColormapPolicy,
    ComponentDimensionLabelPolicy,
    NAPARI_HEARTBEAT,
    NapariViewerServerRequest,
    ViewerQtEnvironmentPolicy,
    ViewerProtocolStatus,
)
from openhcs.runtime.napari_streaming_handlers import (
    NapariBatchProcessorStore,
    NapariComponentValueTracker,
    NapariComponentMetadataNormalizer,
    NapariLayerUpdateAuthority,
    NapariLayerStateStore,
    NapariShapeLabelRasterizer,
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


class NapariPayloadField(str, Enum):
    """Wire keys in Napari stream payloads."""

    TYPE = "type"
    IMAGES = "images"
    DISPLAY_CONFIG = "display_config"
    COMPONENT_NAMES_METADATA = "component_names_metadata"
    PATH = "path"
    IMAGE_ID = "image_id"
    DATA_TYPE = "data_type"
    METADATA = "metadata"
    SHAPES = "shapes"
    SHAPE = "shape"
    DTYPE = "dtype"
    SHM_NAME = "shm_name"
    DATA = "data"


class ShapePayloadField(str, Enum):
    """Wire keys in Napari ROI shape payloads."""

    TYPE = "type"
    COORDINATES = "coordinates"
    METADATA = "metadata"


class VisualMetadataField(str, Enum):
    """Optional visual metadata fields attached to ROI payloads."""

    CENTROID = "centroid"
    LABEL = "label"
    AREA = "area"
    COMPONENT = "component"


@dataclass(frozen=True)
class PayloadMap:
    """Typed accessor for one JSON-derived payload mapping."""

    payload: dict[str, Any]
    context: str

    def required(self, field: NapariPayloadField | ShapePayloadField) -> Any:
        if field.value not in self.payload:
            raise ValueError(
                f"{self.context} missing required payload field '{field.value}'"
            )
        return self.payload[field.value]

    def optional(self, field: NapariPayloadField | ShapePayloadField) -> Any:
        if field.value in self.payload:
            return self.payload[field.value]
        return None

    def value_or_default(
        self,
        field: NapariPayloadField | ShapePayloadField,
        default: Any,
    ) -> Any:
        if field.value in self.payload:
            return self.payload[field.value]
        return default

    def optional_mapping(self, field: NapariPayloadField | ShapePayloadField) -> dict[str, Any]:
        if field.value not in self.payload:
            return {}
        value = self.payload[field.value]
        if not isinstance(value, dict):
            raise TypeError(
                f"{self.context} field '{field.value}' must be a dict, "
                f"got {type(value).__name__}"
            )
        return value


class StackComponentAuthority:
    """Fail-loud lookup for semantic stack components."""

    @staticmethod
    def required(components: dict[str, Any], component: str) -> Any:
        if component not in components:
            raise ValueError(f"Streamed item missing stack component '{component}'")
        return components[component]


@dataclass(frozen=True)
class ShapePayload:
    """Typed view of one Napari ROI shape payload."""

    payload: dict[str, Any]

    @property
    def shape_type(self) -> str:
        return PayloadMap(self.payload, "Napari shape payload").required(
            ShapePayloadField.TYPE
        )

    @property
    def coordinates(self) -> Any:
        return PayloadMap(self.payload, "Napari shape payload").required(
            ShapePayloadField.COORDINATES
        )

    @property
    def metadata(self) -> "VisualMetadata":
        return VisualMetadata(
            PayloadMap(self.payload, "Napari shape payload").optional_mapping(
                ShapePayloadField.METADATA
            )
        )


@dataclass(frozen=True)
class VisualMetadata:
    """Optional display metadata attached to one shape payload."""

    metadata: dict[str, Any]

    def value(self, field: VisualMetadataField, default: Any) -> Any:
        if field.value in self.metadata:
            return self.metadata[field.value]
        return default


@dataclass(frozen=True)
class ComponentLayout:
    """Normalized component layout for Napari layer grouping."""

    component_modes: dict[str, Any]
    component_order: tuple[str, ...]

    @classmethod
    def from_display_config(cls, display_config: dict[str, Any]) -> "ComponentLayout":
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
            if self.component_modes.get(component) == "stack"
        )


@dataclass(frozen=True)
class NapariBatchPayload:
    """Typed view of an incoming batch message."""

    msg_type: Any
    images: list[dict[str, Any]]
    display_config: dict[str, Any]
    component_names_metadata: dict[str, Any]

    @classmethod
    def from_json_payload(cls, data: dict[str, Any]) -> "NapariBatchPayload":
        payload = PayloadMap(data, "Napari batch message")
        return cls(
            msg_type=payload.optional(NapariPayloadField.TYPE),
            images=payload.required(NapariPayloadField.IMAGES),
            display_config=payload.required(NapariPayloadField.DISPLAY_CONFIG),
            component_names_metadata=payload.optional_mapping(
                NapariPayloadField.COMPONENT_NAMES_METADATA
            ),
        )


@dataclass(frozen=True)
class NapariImagePayload:
    """Typed view of one image/shapes message."""

    raw: dict[str, Any]
    display_config: dict[str, Any]
    path: str
    image_id: Any
    data_type: str
    component_metadata: dict[str, Any]

    @classmethod
    def from_payload(
        cls,
        image_info: dict[str, Any],
        display_config: dict[str, Any],
    ) -> "NapariImagePayload":
        payload = PayloadMap(image_info, "Napari image message")
        return cls(
            raw=image_info,
            display_config=display_config,
            path=payload.required(NapariPayloadField.PATH),
            image_id=payload.optional(NapariPayloadField.IMAGE_ID),
            data_type=payload.value_or_default(
                NapariPayloadField.DATA_TYPE,
                DEFAULT_IMAGE_DATA_TYPE,
            ),
            component_metadata=payload.optional_mapping(NapariPayloadField.METADATA),
        )

    @property
    def shapes(self) -> Any:
        return PayloadMap(self.raw, "Napari shapes/points message").required(
            NapariPayloadField.SHAPES
        )

    @property
    def image_shape(self) -> Any:
        return PayloadMap(self.raw, "Napari image message").required(
            NapariPayloadField.SHAPE
        )

    @property
    def dtype(self) -> Any:
        return PayloadMap(self.raw, "Napari image message").required(
            NapariPayloadField.DTYPE
        )

    @property
    def shm_name(self) -> Any:
        return PayloadMap(self.raw, "Napari image message").optional(
            NapariPayloadField.SHM_NAME
        )

    @property
    def direct_data(self) -> Any:
        return PayloadMap(self.raw, "Napari image message").optional(
            NapariPayloadField.DATA
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


def _build_nd_shapes(layer_items, stack_components):
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
            set(StackComponentAuthority.required(item["components"], comp) for item in layer_items)
        )
        component_values[comp] = values

    for item in layer_items:
        shapes_data = item["data"]  # List of shape dicts
        components = item["components"]

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


def _build_nd_points(layer_items, stack_components, component_values=None):
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
                set(StackComponentAuthority.required(item["components"], comp) for item in layer_items)
            )
            component_values[comp] = values

    for item in layer_items:
        points_data = item["data"]  # List of shape dicts from ROI converter
        components = item["components"]

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

    return np.array(all_points_nd) if all_points_nd else np.empty(
        (0, 2 + len(stack_components))
    ), all_properties


def _build_nd_image_array(layer_items, stack_components, component_values=None):
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
        image_stack = [img["data"] for img in layer_items]
        from openhcs.core.memory import stack_slices

        return stack_slices(image_stack, memory_type="numpy", gpu_id=0)
    elif len(stack_components) == 1 and len(layer_items) == 1:
        # Single item, single component, no global values - just return as-is
        # (Will be wrapped in extra dimension if needed by caller)
        return layer_items[0]["data"]

    # Multiple stack components OR using global component values - create multi-dimensional array
    if component_values is None:
        # Derive from layer items (old behavior when no global tracker)
        component_values = {}
        for comp in stack_components:
            values = sorted(
                set(StackComponentAuthority.required(img["components"], comp) for img in layer_items)
            )
            component_values[comp] = values

    # Log component values for debugging
    logger.info(
        f"🔬 NAPARI PROCESS: Building nD array with stack_components={stack_components}, component_values={component_values}"
    )

    # Create empty array with shape (comp1_size, comp2_size, ..., y, x)
    first_img = layer_items[0]["data"]
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
                StackComponentAuthority.required(img["components"], comp)
            )
            for comp in stack_components
        )
        logger.debug(
            f"🔬 NAPARI PROCESS: Placing image at indices {indices}, components={img['components']}"
        )
        stacked_array[indices] = img["data"]

    return stacked_array


def _create_or_update_image_layer(
    viewer, layers, layer_name, image_data, colormap, axis_labels=None
):
    """Create or update a Napari image layer."""
    return _NAPARI_LAYER_UPDATES.create_or_update_image(
        viewer,
        layers,
        layer_name,
        image_data,
        colormap,
        axis_labels,
    )


def _create_or_update_shapes_layer(
    viewer, layers, layer_name, shapes_data, shape_types, properties
):
    """Create or update a Napari shapes layer."""
    return _NAPARI_LAYER_UPDATES.create_or_update_shapes(
        viewer,
        layers,
        layer_name,
        shapes_data,
        shape_types,
        properties,
    )


def _create_or_update_points_layer(viewer, layers, layer_name, points_data, properties):
    """Create or update a Napari points layer."""
    return _NAPARI_LAYER_UPDATES.create_or_update_points(
        viewer,
        layers,
        layer_name,
        points_data,
        properties,
    )


# Populate registry now that helper functions are defined
from polystore.streaming_constants import StreamingDataType

_DATA_TYPE_HANDLERS = build_napari_streaming_data_type_handlers(
    build_image_data=_build_nd_image_array,
    create_image_layer=_create_or_update_image_layer,
    build_shapes_data=_build_nd_shapes,
    create_shapes_layer=_create_or_update_shapes_layer,
    build_points_data=_build_nd_points,
    create_points_layer=_create_or_update_points_layer,
)


def _handle_component_aware_display(
    viewer,
    layers,
    component_groups,
    data,
    path,
    colormap,
    display_config,
    replace_layers,
    component_metadata=None,
    data_type="image",
    server=None,
):
    """
    Handle component-aware display following OpenHCS stacking patterns.

    Components marked as SLICE create separate layers, components marked as STACK are stacked together.
    Layer naming follows canonical component order from display config.

    Args:
        data_type: 'image' for image data, 'shapes' for ROI/shapes data (string or StreamingDataType enum)
        server: NapariViewerServer instance (needed for debounced updates)
    """
    try:
        # Convert data_type to enum if needed (for backwards compatibility)
        if isinstance(data_type, str):
            data_type = StreamingDataType(data_type)

        # Use component metadata from ZMQ message - fail loud if not available
        if not component_metadata:
            raise ValueError(f"No component metadata available for path: {path}")
        component_info = _COMPONENT_METADATA_NORMALIZER.normalize(component_metadata)

        component_layout = ComponentLayout.from_display_config(display_config)
        layer_key = build_layer_key(
            component_info=component_info,
            component_modes=component_layout.component_modes,
            component_order=component_layout.component_order,
            data_type=data_type,
        )

        # Log component modes for debugging
        logger.info(
            f"🔍 NAPARI PROCESS: component_modes={component_layout.component_modes}, layer_key='{layer_key}'"
        )

        # Log layer key and component info for debugging
        logger.info(
            f"🔍 NAPARI PROCESS: layer_key='{layer_key}', component_info={component_info}"
        )

        # Reconcile cached layer/group state with live napari viewer after possible manual deletions
        # CRITICAL: Only purge if the layer WAS in our cache but is now missing from viewer
        # (user manually deleted it). Do NOT purge if layer was never created yet (debounced update pending).
        try:
            current_layer_names = {l.name for l in viewer.layers}
            if layer_key not in current_layer_names and layer_key in layers:
                # Layer was in our cache but is now missing from viewer - user deleted it
                # Drop stale references so we will recreate the layer
                num_items = len(component_groups.get(layer_key, []))
                layers.pop(layer_key, None)
                component_groups.pop(layer_key, None)
                logger.info(
                    f"🔬 NAPARI PROCESS: Reconciling state — '{layer_key}' was deleted from viewer; purged stale caches (had {num_items} items in component_groups)"
                )
        except Exception:
            # Fail-loud elsewhere; reconciliation is best-effort and must not mask display
            pass

        # Initialize layer group if needed
        if layer_key not in component_groups:
            component_groups[layer_key] = []

        # Handle replace_layers mode: clear all items for this layer_key
        if replace_layers and component_groups[layer_key]:
            logger.info(
                f"🔬 NAPARI PROCESS: replace_layers=True, clearing {len(component_groups[layer_key])} existing items from layer '{layer_key}'"
            )
            component_groups[layer_key] = []

        # Check if an item with the same component_info AND data_type already exists
        # If so, replace it instead of appending (prevents accumulation across runs)
        # CRITICAL: Must include 'well' in comparison even if it's in STACK mode,
        # otherwise images from different wells with same channel/z/field will be treated as duplicates
        # CRITICAL: Must also check data_type to prevent images and ROIs from being treated as duplicates
        existing_index = None
        for i, item in enumerate(component_groups[layer_key]):
            # Compare ALL components including well AND data_type
            if item["components"] == component_info and item["data_type"] == data_type:
                logger.info(
                    f"🔬 NAPARI PROCESS: Found duplicate - component_info: {component_info}, data_type: {data_type} at index {i}"
                )
                existing_index = i
                break

        new_item = {
            "data": data,
            "components": component_info,
            "path": str(path),
            "data_type": data_type,
        }

        if existing_index is not None:
            # Replace existing item with same components and data type
            old_data_type = component_groups[layer_key][existing_index]["data_type"]
            component_groups[layer_key][existing_index] = new_item
            logger.info(
                f"🔬 NAPARI PROCESS: Replaced {old_data_type} item in component_groups[{layer_key}] at index {existing_index}, total items: {len(component_groups[layer_key])}"
            )
        else:
            # Add new item
            component_groups[layer_key].append(new_item)
            logger.info(
                f"🔬 NAPARI PROCESS: Added {data_type} to component_groups[{layer_key}], now has {len(component_groups[layer_key])} items"
            )

        # Schedule debounced layer update instead of immediate update
        # This prevents race conditions when multiple items arrive rapidly
        if server is None:
            raise ValueError("Server instance required for debounced updates")
        logger.info(
            f"🔬 NAPARI PROCESS: Scheduling debounced update for {layer_key} (data_type={data_type})"
        )
        server.display_pipeline.schedule_layer_update(layer_key, data_type, component_layout)

    except Exception as e:
        import traceback

        logger.error(
            f"🔬 NAPARI PROCESS: Component-aware display failed for {path}: {e}"
        )
        logger.error(
            f"🔬 NAPARI PROCESS: Component-aware display traceback: {traceback.format_exc()}"
        )
        raise  # Fail loud - no fallback


class NapariLayerDisplayPipeline:
    """Owns debounced Napari layer display and update routing."""

    def __init__(self, server: "NapariViewerServer") -> None:
        self.server = server
        self.layer_update_routes = {
            StreamingDataType.IMAGE: self.update_image_layer,
            StreamingDataType.SHAPES: self.update_shapes_layer,
            StreamingDataType.POINTS: self.update_points_layer,
        }

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

        layer_items = self.server.component_groups.get(layer_key, [])
        if not layer_items:
            logger.warning(
                f"🔬 NAPARI PROCESS: No items found for {layer_key}, skipping update"
            )
            return

        wells_in_layer = set(
            item["components"].get("well", "unknown") for item in layer_items
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
        items: list[dict[str, Any]],
        display_payload: ComponentLayout,
        component_names_metadata: dict[str, Any],
    ) -> None:
        component_layout = display_payload
        if component_names_metadata:
            self.server.component_metadata.update(component_names_metadata)

        items_by_type: dict[StreamingDataType, list[dict[str, Any]]] = {}
        for item in items:
            data_type = item.get("data_type")
            if isinstance(data_type, str):
                data_type = StreamingDataType(data_type)
            items_by_type.setdefault(data_type, []).append(item)

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

                self.server.viewer.text_overlay.text = (
                    " | ".join(label_parts) if label_parts else ""
                )

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
        self.server.component_values.update(stack_components, layer_items)
        global_component_values = self.server.component_values.values_for(
            stack_components
        )

        shapes = [item["data"].shape for item in layer_items]
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

            for img_info in layer_items:
                img_data = img_info["data"]
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
                    img_info["data"] = padded_data
                    logger.debug(
                        f"🔬 NAPARI PROCESS: Padded image from {img_data.shape} to {padded_data.shape}"
                    )

        logger.info(
            f"🔬 NAPARI PROCESS: Building nD data for {layer_key} from {len(layer_items)} items"
        )
        stacked_data = _build_nd_image_array(
            layer_items,
            stack_components,
            global_component_values,
        )

        colormap = None
        if "channel" in component_modes and component_modes["channel"] == "slice":
            first_item = layer_items[0]
            channel_value = first_item["components"].get("channel")
            colormap = ChannelColormapPolicy().colormap(channel_value)

        axis_labels = None
        if stack_components:
            axis_labels = tuple(list(stack_components) + ["y", "x"])
            logger.info(
                f"🔬 NAPARI PROCESS: Built axis_labels={axis_labels} for stack_components={stack_components}"
            )

        dimension_labels = {}
        for comp in stack_components:
            values = global_component_values[comp]
            comp_metadata = self.server.component_metadata.get(comp, {})
            dimension_labels[comp] = _COMPONENT_DIMENSION_LABELS.labels_for(
                component=comp,
                values=values,
                metadata=comp_metadata,
            )

        self.server.layer_state.set_labels(layer_key, dimension_labels)

        _create_or_update_image_layer(
            self.server.viewer,
            self.server.layer_state.layers,
            layer_key,
            stacked_data,
            colormap,
            axis_labels,
        )

        self.setup_dimension_label_handler(layer_key, stack_components)

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

        self.server.component_values.update(stack_components, layer_items)
        global_component_values = self.server.component_values.values_for(
            stack_components
        )

        labels_data = _NAPARI_SHAPE_RASTERIZER.rasterize(
            layer_items=layer_items,
            stack_components=stack_components,
            component_values=global_component_values,
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

        new_layer = self.server.viewer.add_labels(labels_data, name=layer_key)
        self.server.layer_state.set_layer(layer_key, new_layer)
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
            if item.get("data_type") == StreamingDataType.POINTS
        ]

        if not points_items:
            logger.warning(
                f"🔬 NAPARI PROCESS: No POINTS items found for {layer_key}, skipping"
            )
            return

        logger.info(
            f"🔬 NAPARI PROCESS: Building points layer for {layer_key} from {len(points_items)} items (filtered from {len(layer_items)} total)"
        )

        self.server.component_values.update(stack_components, layer_items)
        global_component_values = self.server.component_values.values_for(
            stack_components
        )

        points_data, properties = _build_nd_points(
            points_items,
            stack_components,
            global_component_values,
        )

        _create_or_update_points_layer(
            self.server.viewer,
            self.server.layer_state.layers,
            layer_key,
            points_data,
            properties,
        )

        logger.info(
            f"🔬 NAPARI PROCESS: Created points layer {layer_key} with {len(points_data)} points"
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

        self.viewer_title = request.viewer_title
        self.replace_layers = request.replace_layers
        self.viewer = None
        self.layer_state = NapariLayerStateStore.empty()
        self.component_groups = {}
        self.component_metadata = {}  # Store component metadata from microscope handler: {component: {id: name}}

        self.component_values = NapariComponentValueTracker()

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
        items: list[dict[str, Any]],
        display_payload: ComponentLayout,
        component_names_metadata: dict[str, Any],
    ) -> None:
        """Display one debounced batch through the composed display pipeline."""
        self.display_pipeline.display_layer_batch(
            layer_key=layer_key,
            items=items,
            display_payload=display_payload,
            component_names_metadata=component_names_metadata,
        )

    def _send_ack(self, image_id: str, status: str = "success", error: str = None):
        """Send acknowledgment that an image was processed.

        Args:
            image_id: UUID of the processed image
            status: 'success' or 'error'
            error: Error message if status='error'
        """
        self.send_ack(image_id, status=status, error=error)

    def _create_pong_response(self) -> Dict[str, Any]:
        """Override to add Napari-specific fields and memory usage."""
        return NAPARI_HEARTBEAT.apply_to(super()._create_pong_response())

    def handle_control_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle control messages beyond ping/pong.

        Supported message types:
        - shutdown: Graceful shutdown (closes viewer)
        - force_shutdown: Force shutdown (same as shutdown for Napari)
        - clear_state: Clear accumulated component groups (for new pipeline runs)
        """
        msg_type = message.get("type")

        if msg_type == "shutdown" or msg_type == "force_shutdown":
            logger.info(f"🔬 NAPARI SERVER: {msg_type} requested, closing viewer")
            self.request_shutdown()

            # Schedule viewer close on Qt event loop to trigger application exit
            # This must be done after sending the response, so we use QTimer.singleShot
            if self.viewer is not None:
                from qtpy import QtCore

                QtCore.QTimer.singleShot(100, self.viewer.close)

            return {
                "type": "shutdown_ack",
                "status": "success",
                "message": "Napari viewer shutting down",
            }

        elif msg_type == "clear_state":
            # Clear accumulated component groups to prevent shape accumulation across runs
            logger.info(
                f"🔬 NAPARI SERVER: Clearing component groups (had {len(self.component_groups)} groups)"
            )
            self.component_groups.clear()
            return {
                "type": "clear_state_ack",
                "status": "success",
                "message": "Component groups cleared",
            }

        # Unknown message type
        return {"status": "ok"}

    def display_image(self, image_data: np.ndarray, metadata: dict) -> None:
        """Display a single image payload (best-effort helper)."""
        image_info = {
            "path": DEFAULT_DIRECT_IMAGE_PATH,
            "data": image_data,
            "shape": getattr(image_data, "shape", None),
            "dtype": getattr(image_data, "dtype", None),
            "metadata": metadata,
        }
        self._process_single_image(image_info, {})

    def process_image_message(self, message: bytes):
        """
        Process incoming image data message and send reply for REP socket.

        Args:
            message: Raw ZMQ message containing image data
        """
        import json

        # Parse JSON message
        data = json.loads(message.decode("utf-8"))

        msg_type = PayloadMap(data, "Napari message").optional(NapariPayloadField.TYPE)

        # Check message type
        if msg_type == "batch":
            batch_payload = NapariBatchPayload.from_json_payload(data)

            # Extract component names metadata for dimension labels (e.g., channel names)
            if batch_payload.component_names_metadata:
                # Update server's component metadata cache
                self.component_metadata.update(batch_payload.component_names_metadata)
                logger.info(
                    f"🔬 NAPARI PROCESS: Updated component metadata: {list(batch_payload.component_names_metadata.keys())}"
                )

            for image_info in batch_payload.images:
                self._process_single_image(image_info, batch_payload.display_config)

        else:
            # Handle single image (legacy)
            self._process_single_image(
                data,
                PayloadMap(data, "Napari image message").optional_mapping(
                    NapariPayloadField.DISPLAY_CONFIG
                ),
            )

        # Send reply on REP socket (required pattern)
        try:
            reply = {"status": "success", "type": msg_type}
            self.data_socket.send_json(reply)
        except Exception as e:
            logger.error(f"🔬 NAPARI PROCESS: Failed to send reply: {e}")

    def _process_single_image(
        self, image_info: Dict[str, Any], display_config_dict: Dict[str, Any]
    ):
        """Process a single image or shapes data and display in Napari."""
        import numpy as np

        payload = NapariImagePayload.from_payload(image_info, display_config_dict)

        # Log incoming metadata to debug well filtering issues
        logger.info(
            f"🔍 NAPARI PROCESS: Received {payload.data_type} with metadata: {payload.component_metadata} (path: {payload.path})"
        )

        try:
            # Check if this is shapes or points data
            if payload.data_type == "shapes" or payload.data_type == "points":
                # Handle shapes/ROIs/points - just pass the shapes data directly
                data = payload.shapes
                colormap = None  # Shapes/points don't use colormap
            else:
                # Handle image data - load from shared memory or direct data
                # Load image data
                if payload.shm_name:
                    from multiprocessing import shared_memory

                    try:
                        shm = shared_memory.SharedMemory(name=payload.shm_name)
                        data = np.ndarray(
                            payload.image_shape,
                            dtype=payload.dtype,
                            buffer=shm.buf,
                        ).copy()
                        shm.close()
                        # Unlink shared memory after copying - viewer is responsible for cleanup
                        try:
                            shm.unlink()
                        except FileNotFoundError:
                            # Already unlinked (race condition or duplicate message)
                            logger.debug(
                                f"🔬 NAPARI PROCESS: Shared memory {payload.shm_name} already unlinked"
                            )
                        except Exception as e:
                            logger.warning(
                                f"🔬 NAPARI PROCESS: Failed to unlink shared memory {payload.shm_name}: {e}"
                            )
                    except FileNotFoundError:
                        # Shared memory doesn't exist - likely already processed and unlinked
                        logger.error(
                            f"🔬 NAPARI PROCESS: Shared memory {payload.shm_name} not found - may have been already processed"
                        )
                        if payload.image_id:
                            self._send_ack(
                                payload.image_id,
                                status=_ACK_ERROR,
                                error=f"Shared memory {payload.shm_name} not found",
                            )
                        return
                    except Exception as e:
                        logger.error(
                            f"🔬 NAPARI PROCESS: Failed to open shared memory {payload.shm_name}: {e}"
                        )
                        if payload.image_id:
                            self._send_ack(
                                payload.image_id,
                                status=_ACK_ERROR,
                                error=f"Failed to open shared memory: {e}",
                            )
                        raise
                elif payload.direct_data is not None:
                    data = np.array(payload.direct_data, dtype=payload.dtype).reshape(
                        payload.image_shape
                    )
                else:
                    logger.warning("🔬 NAPARI PROCESS: No image data in message")
                    if payload.image_id:
                        self._send_ack(
                            payload.image_id,
                            status=_ACK_ERROR,
                            error="No image data in message",
                        )
                    return

                # Extract colormap
                colormap = payload.colormap

            # Component-aware layer management (handles both images and shapes)
            _handle_component_aware_display(
                self.viewer,
                self.layer_state.layers,
                self.component_groups,
                data,
                payload.path,
                colormap,
                payload.display_config,
                self.replace_layers,
                payload.component_metadata,
                payload.data_type,
                server=self,
            )

            # Send acknowledgment that data was successfully displayed
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
