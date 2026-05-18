"""
Napari-based real-time visualization module for OpenHCS.

This module provides the NapariStreamVisualizer class for real-time
visualization of tensors during pipeline execution.

Doctrinal Clauses:
- Clause 65 — No Fallback Logic
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 368 — Visualization Must Be Observer-Only
"""

import logging
import os
import pickle
import subprocess
import sys
import threading
import time
import zmq
import numpy as np
from typing import Any, Dict, Optional
from qtpy.QtCore import QTimer

from polystore.filemanager import FileManager
from openhcs.utils.import_utils import optional_import
from openhcs.core.config import (
    TransportMode as OpenHCSTransportMode,
    NapariStreamingConfig,
)
from openhcs.runtime.viewer_protocol import (
    ChannelColormapPolicy,
    ComponentDimensionLabelPolicy,
    ManagedViewerLifecycleMixin,
    NAPARI_HEARTBEAT,
    NapariDetachedProcessRequest,
    NapariViewerServerRequest,
    ViewerControlPingMode,
    ViewerControlPingRequest,
    ViewerQtEnvironmentPolicy,
    ViewerLifecycleState,
    ViewerProcessHandle,
    ViewerProtocolStatus,
)
from openhcs.runtime.napari_streaming_handlers import (
    NapariLayerUpdateAuthority,
    NapariLayerStateStore,
    NapariShapeLabelRasterizer,
    build_napari_streaming_data_type_handlers,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from zmqruntime.config import TransportMode as ZMQTransportMode
from zmqruntime.streaming import StreamingVisualizerServer, VisualizerProcessManager
from zmqruntime.transport import (
    coerce_transport_mode,
    get_control_url,
    get_zmq_transport_url,
    is_port_in_use,
    wait_for_server_ready,
)

# Optional napari import - this module should only be imported if napari is available
napari = optional_import("napari")
if napari is None:
    raise ImportError(
        "napari is required for NapariStreamVisualizer. "
        "Install it with: pip install 'openhcs[viz]' or pip install napari"
    )


logger = logging.getLogger(__name__)
_NAPARI_LAYER_UPDATES = NapariLayerUpdateAuthority()
_COMPONENT_DIMENSION_LABELS = ComponentDimensionLabelPolicy()
_NAPARI_SHAPE_RASTERIZER = NapariShapeLabelRasterizer()
_ACK_ERROR = ViewerProtocolStatus.ERROR.value
_ACK_SUCCESS = ViewerProtocolStatus.SUCCESS.value

# ZMQ connection delay (ms)
ZMQ_CONNECTION_DELAY_MS = 100  # Brief delay for ZMQ connection to establish

# Global process management for napari viewer
_global_viewer_process: Optional[subprocess.Popen] = None
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
        if (
            _global_viewer_process
            and ViewerProcessHandle.from_process(_global_viewer_process).is_alive()
        ):
            logger.info("🔬 VISUALIZER: Terminating napari viewer for test cleanup")
            killed = ViewerProcessHandle.from_process(_global_viewer_process).terminate(
                timeout=3,
                kill_timeout=1,
            )
            if killed:
                logger.warning("🔬 VISUALIZER: Force killing napari viewer process")

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
    from openhcs.runtime.roi_converters import NapariROIConverter

    all_shapes_nd = []
    all_shape_types = []
    all_properties = {"label": [], "area": [], "centroid_y": [], "centroid_x": []}

    # Build component value to index mapping (same as _build_nd_image_array)
    component_values = {}
    for comp in stack_components:
        values = sorted(set(item["components"].get(comp, 0) for item in layer_items))
        component_values[comp] = values

    for item in layer_items:
        shapes_data = item["data"]  # List of shape dicts
        components = item["components"]

        # Get stack component INDICES to prepend (not values!)
        prepend_dims = [
            component_values[comp].index(components.get(comp, 0))
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
            metadata = shape_dict.get("metadata", {})
            centroid = metadata.get("centroid", (0, 0))
            all_properties["label"].append(metadata.get("label", ""))
            all_properties["area"].append(metadata.get("area", 0))
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
                set(item["components"].get(comp, 0) for item in layer_items)
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
            component_values[comp].index(components.get(comp, 0))
            for comp in stack_components
        ]

        # Convert each shape dict to nD points
        # points_data is a list of dicts with 'type', 'coordinates', 'metadata'
        for shape_dict in points_data:
            # Only process 'points' type entries
            if shape_dict.get("type") != "points":
                continue

            coordinates = shape_dict.get("coordinates", [])
            metadata = shape_dict.get("metadata", {})

            # coordinates is a list of [y, x] pairs
            # Prepend stack dimensions to each point: [y, x] -> [stack_idx, ..., y, x]
            for coord in coordinates:
                nd_coord = prepend_dims + list(coord)
                all_points_nd.append(nd_coord)

                # Track properties for this point
                all_properties["label"].append(metadata.get("label", ""))
                all_properties["component"].append(metadata.get("component", 0))

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
            values = sorted(set(img["components"].get(comp, 0) for img in layer_items))
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
            component_values[comp].index(img["components"].get(comp, 0))
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
from polystore.streaming.receivers.napari import (
    normalize_component_layout,
    build_layer_key,
)

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
        component_info = component_metadata

        component_modes, component_order = normalize_component_layout(display_config)
        layer_key = build_layer_key(
            component_info=component_info,
            component_modes=component_modes,
            component_order=component_order,
            data_type=data_type,
        )

        # Log component modes for debugging
        logger.info(
            f"🔍 NAPARI PROCESS: component_modes={component_modes}, layer_key='{layer_key}'"
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
        server._schedule_layer_update(
            layer_key, data_type, component_modes, component_order
        )

    except Exception as e:
        import traceback

        logger.error(
            f"🔬 NAPARI PROCESS: Component-aware display failed for {path}: {e}"
        )
        logger.error(
            f"🔬 NAPARI PROCESS: Component-aware display traceback: {traceback.format_exc()}"
        )
        raise  # Fail loud - no fallback


def _old_immediate_update_logic_removed():
    """
    Old immediate update logic removed in favor of debounced updates.
    Kept as reference for the variable size handling logic that needs to be ported.
    """
    pass
    # Old code was here - removed to prevent race conditions
    # Now using _schedule_layer_update -> _execute_layer_update -> _update_image_layer/_update_shapes_layer


class NapariViewerServer(StreamingVisualizerServer):
    """
    ZMQ server for Napari viewer that receives images from clients.

    Inherits from ZMQServer ABC to get ping/pong, port management, etc.
    Uses SUB socket to receive images from pipeline clients.
    """

    _server_type = "napari"  # Registration key for AutoRegisterMeta

    def __init__(
        self,
        port: int,
        viewer_title: str,
        replace_layers: bool = False,
        log_file_path: str = None,
        transport_mode: OpenHCSTransportMode = OpenHCSTransportMode.IPC,
    ):
        """
        Initialize Napari viewer server.

        Args:
            port: Data port for receiving images (control port will be port + 1000)
            viewer_title: Title for the napari viewer window
            replace_layers: If True, replace existing layers; if False, add new layers
            log_file_path: Path to log file (for client discovery)
            transport_mode: ZMQ transport mode (IPC or TCP)
        """
        import zmq
        request = NapariViewerServerRequest.from_legacy_signature(
            port,
            viewer_title,
            replace_layers,
            log_file_path,
            transport_mode,
        )

        # Initialize with SUB socket for receiving images
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

        # Global component value tracker for shared dimension mapping
        # Maps tuple of stack_components -> {component: set of values}
        # All layers with the same stack_components share the same global mapping
        self.global_component_values = {}

        # Debouncing + locking for layer updates to prevent race conditions
        import threading

        self.layer_update_lock = threading.Lock()  # Prevent concurrent updates
        self.update_delay_ms = 1000  # Wait 200ms for more items before rebuilding
        self.layer_update_routes = {
            StreamingDataType.IMAGE: self._update_image_layer,
            StreamingDataType.SHAPES: self._update_shapes_layer,
            StreamingDataType.POINTS: self._update_points_layer,
        }

        # Ack socket handled by StreamingVisualizerServer

    def _setup_ack_socket(self):
        """Setup PUSH socket for sending acknowledgments."""
        super()._setup_ack_socket()

    def _update_global_component_values(self, stack_components, layer_items):
        """
        Update the global component value tracker with values from new items.

        All layers sharing the same stack_components will use the same global mapping,
        ensuring consistent component-to-index mapping across image and ROI layers.

        Args:
            stack_components: Tuple/list of component names (e.g., ['channel', 'well'])
            layer_items: List of items with 'components' dict
        """
        # Use tuple as dict key (lists aren't hashable)
        components_key = tuple(stack_components)

        # Initialize if needed
        if components_key not in self.global_component_values:
            self.global_component_values[components_key] = {
                comp: set() for comp in stack_components
            }

        # Add values from these items
        global_values = self.global_component_values[components_key]
        for item in layer_items:
            for comp in stack_components:
                value = item["components"].get(comp, 0)
                global_values[comp].add(value)

        logger.info(
            f"🔬 NAPARI PROCESS: Updated global component values for {stack_components}"
        )
        for comp, values in global_values.items():
            sorted_values = sorted(values)
            logger.info(f"🔬 NAPARI PROCESS:   {comp}: {sorted_values}")

    def _get_global_component_values(self, stack_components):
        """
        Get the global component values for a given set of stack components.

        For indexed components (channel, z_index, timepoint), expands to include
        all values from 1 to max. For example, if only channel 2 is seen, returns [1, 2].
        This ensures proper stack dimensions even when some indices aren't present.

        Returns a dict of {component: sorted list of values} for all components
        that have been seen across all layers sharing these stack components.
        """
        components_key = tuple(stack_components)

        if components_key not in self.global_component_values:
            return {comp: [] for comp in stack_components}

        # Convert sets to sorted lists and expand indexed components
        global_values = self.global_component_values[components_key]
        result = {}

        # Components that should be expanded from 1 to max (1-indexed)
        INDEXED_COMPONENTS = {"channel", "z_index", "timepoint"}

        for comp, values in global_values.items():
            sorted_values = sorted(values)

            if comp in INDEXED_COMPONENTS and sorted_values:
                # Expand to include all indices from 1 to max
                # E.g., if we have [2, 4], expand to [1, 2, 3, 4]
                max_value = max(sorted_values)
                if max_value > 1:
                    # Create range from 1 to max_value (inclusive)
                    expanded_values = list(range(1, max_value + 1))
                    result[comp] = expanded_values
                    logger.info(
                        f"🔬 NAPARI PROCESS: Expanded {comp} from {sorted_values} to {expanded_values}"
                    )
                else:
                    # Max is 1, no expansion needed
                    result[comp] = sorted_values
            else:
                # Non-indexed component (well, site, etc.) - use actual values
                result[comp] = sorted_values

        return result

    def _schedule_layer_update(
        self, layer_key, data_type, component_modes, component_order
    ):
        """
        Schedule a debounced layer update.

        Cancels any pending update for this layer and schedules a new one.
        This prevents race conditions when multiple items arrive rapidly.
        """
        # Cancel existing timer if any
        if self.layer_state.cancel_pending_update(layer_key):
            logger.debug(f"🔬 NAPARI PROCESS: Cancelled pending update for {layer_key}")

        # Create new timer
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(
            lambda: self._execute_layer_update(
                layer_key, data_type, component_modes, component_order
            )
        )
        timer.start(self.update_delay_ms)
        self.layer_state.set_pending_update(layer_key, timer)
        logger.debug(
            f"🔬 NAPARI PROCESS: Scheduled update for {layer_key} in {self.update_delay_ms}ms"
        )

    def _execute_layer_update(
        self, layer_key, data_type, component_modes, component_order
    ):
        """
        Execute the actual layer update after debounce delay.

        Uses a lock to prevent concurrent updates to different layers.
        """
        # Remove timer
        self.layer_state.pop_pending_update(layer_key)

        # Acquire lock to prevent concurrent updates
        with self.layer_update_lock:
            logger.info(
                f"🔬 NAPARI PROCESS: Executing debounced update for {layer_key}"
            )

            # Get current items for this layer
            layer_items = self.component_groups.get(layer_key, [])
            if not layer_items:
                logger.warning(
                    f"🔬 NAPARI PROCESS: No items found for {layer_key}, skipping update"
                )
                return

            # Log layer composition
            wells_in_layer = set(
                item["components"].get("well", "unknown") for item in layer_items
            )
            logger.info(
                f"🔬 NAPARI PROCESS: layer_key='{layer_key}' has {len(layer_items)} items from wells: {sorted(wells_in_layer)}"
            )

            # Determine stack components (axes) to use
            first_item = layer_items[0]
            component_info = first_item["components"]
            stack_components = [
                comp
                for comp, mode in component_modes.items()
                if mode == "stack" and comp in component_info
            ]

            logger.info(
                f"🔬 NAPARI PROCESS: Using stack components: {stack_components}"
            )

            try:
                route = self.layer_update_routes.get(data_type)
                if route is None:
                    logger.warning(
                        f"🔬 NAPARI PROCESS: Unknown data type {data_type} for {layer_key}"
                    )
                    return
                route(layer_key, layer_items, stack_components, component_modes)
            except Exception as e:
                logger.error(
                    f"🔬 NAPARI PROCESS: Failed to update layer {layer_key}: {e}",
                    exc_info=True,
                )
                # Continue running - don't crash the viewer

    def _setup_dimension_label_handler(self, layer_key, stack_components):
        """
        Set up event handler to update text overlay when dimensions change.

        This connects the viewer's dimension slider changes to text overlay updates,
        displaying categorical labels (like well IDs) instead of numeric indices.

        Args:
            layer_key: The layer to monitor for dimension changes
            stack_components: List of components that are stacked (e.g., ['well', 'channel'])
        """
        if not self.viewer or not stack_components:
            return

        # Get dimension label mappings for this layer
        layer_labels = self.layer_state.labels_for(layer_key)
        if not layer_labels:
            return

        def update_dimension_label(event=None):
            """Update text overlay with current dimension labels."""
            try:
                current_step = self.viewer.dims.current_step

                # Build label text from stacked components
                label_parts = []
                for i, component in enumerate(stack_components):
                    if component in layer_labels:
                        labels = layer_labels[component]
                        # Get current index for this dimension
                        if i < len(current_step):
                            idx = current_step[i]
                            if 0 <= idx < len(labels):
                                label = labels[idx]
                                # Don't show if label is None or "None"
                                if label and str(label).lower() != "none":
                                    label_parts.append(label)

                if label_parts:
                    self.viewer.text_overlay.text = " | ".join(label_parts)
                else:
                    self.viewer.text_overlay.text = ""

            except Exception as e:
                logger.debug(f"🔬 NAPARI PROCESS: Error updating dimension label: {e}")

        # Connect to dimension change events
        try:
            self.viewer.dims.events.current_step.connect(update_dimension_label)
            # Initial update
            update_dimension_label()
            logger.info(
                f"🔬 NAPARI PROCESS: Set up dimension label handler for {layer_key}"
            )
        except Exception as e:
            logger.warning(
                f"🔬 NAPARI PROCESS: Failed to setup dimension label handler: {e}"
            )

    def _update_image_layer(
        self, layer_key, layer_items, stack_components, component_modes
    ):
        """Update an image layer with the current items."""

        # Update global component tracker with values from these items
        self._update_global_component_values(stack_components, layer_items)

        # Get global component values (union of all layers with same stack_components)
        global_component_values = self._get_global_component_values(stack_components)

        # Check if images have different shapes and pad if needed
        shapes = [item["data"].shape for item in layer_items]
        if len(set(shapes)) > 1:
            logger.info(
                f"🔬 NAPARI PROCESS: Images in layer {layer_key} have different shapes - padding to max size"
            )

            # Find max dimensions
            first_shape = shapes[0]
            max_shape = list(first_shape)
            for img_shape in shapes:
                for i, dim in enumerate(img_shape):
                    max_shape[i] = max(max_shape[i], dim)
            max_shape = tuple(max_shape)

            # Pad all images to max shape
            for img_info in layer_items:
                img_data = img_info["data"]
                if img_data.shape != max_shape:
                    # Calculate padding for each dimension
                    pad_width = []
                    for i, (current_dim, max_dim) in enumerate(
                        zip(img_data.shape, max_shape)
                    ):
                        pad_before = 0
                        pad_after = max_dim - current_dim
                        pad_width.append((pad_before, pad_after))

                    # Pad with zeros
                    padded_data = np.pad(
                        img_data, pad_width, mode="constant", constant_values=0
                    )
                    img_info["data"] = padded_data
                    logger.debug(
                        f"🔬 NAPARI PROCESS: Padded image from {img_data.shape} to {padded_data.shape}"
                    )

        logger.info(
            f"🔬 NAPARI PROCESS: Building nD data for {layer_key} from {len(layer_items)} items"
        )
        stacked_data = _build_nd_image_array(
            layer_items, stack_components, global_component_values
        )

        # Determine colormap
        colormap = None
        if "channel" in component_modes and component_modes["channel"] == "slice":
            first_item = layer_items[0]
            channel_value = first_item["components"].get("channel")
            colormap = ChannelColormapPolicy().colormap(channel_value)

        # Build axis labels for stacked dimensions
        # Format: (component1_name, component2_name, ..., 'y', 'x')
        # The stack components appear in the same order as in stack_components list
        # Must be a tuple for Napari
        axis_labels = None
        if stack_components:
            axis_labels = tuple(list(stack_components) + ["y", "x"])
            logger.info(
                f"🔬 NAPARI PROCESS: Built axis_labels={axis_labels} for stack_components={stack_components}"
            )

        # Build dimension labels from component values
        # Use global component values to ensure consistency across all layers
        dimension_labels = {}

        for comp in stack_components:
            # Use global component values instead of just this layer's values
            values = global_component_values[comp]
            comp_metadata = self.component_metadata.get(comp, {})
            dimension_labels[comp] = _COMPONENT_DIMENSION_LABELS.labels_for(
                component=comp,
                values=values,
                metadata=comp_metadata,
            )

        # Store dimension labels for this layer
        self.layer_state.set_labels(layer_key, dimension_labels)

        # Create or update the layer
        _create_or_update_image_layer(
            self.viewer, self.layer_state.layers, layer_key, stacked_data, colormap, axis_labels
        )

        # Set up dimension label handler (connects dimension changes to text overlay)
        self._setup_dimension_label_handler(layer_key, stack_components)

    def _update_shapes_layer(
        self, layer_key, layer_items, stack_components, component_modes
    ):
        """Update a shapes layer - use labels instead of shapes for efficiency."""
        logger.info(
            f"🔬 NAPARI PROCESS: Converting shapes to labels for {layer_key} from {len(layer_items)} items"
        )

        # Update global component tracker with values from these items
        self._update_global_component_values(stack_components, layer_items)

        # Get global component values (union of all layers with same stack_components)
        global_component_values = self._get_global_component_values(stack_components)

        # Convert shapes to label masks (much faster than individual shapes)
        # This happens synchronously but is fast because we're just creating arrays
        labels_data = _NAPARI_SHAPE_RASTERIZER.rasterize(
            layer_items=layer_items,
            stack_components=stack_components,
            component_values=global_component_values,
        )

        # Remove existing layer if it exists
        if self.layer_state.has_layer(layer_key):
            try:
                self.viewer.layers.remove(self.layer_state.layer(layer_key))
                logger.info(
                    f"🔬 NAPARI PROCESS: Removed existing labels layer {layer_key} for recreation"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to remove existing labels layer {layer_key}: {e}"
                )

        # Create new labels layer
        new_layer = self.viewer.add_labels(labels_data, name=layer_key)
        self.layer_state.set_layer(layer_key, new_layer)
        logger.info(
            f"🔬 NAPARI PROCESS: Created labels layer {layer_key} with shape {labels_data.shape}"
        )

    def _update_points_layer(
        self, layer_key, layer_items, stack_components, component_modes
    ):
        """Update a points layer (for skeleton tracings and point-based ROIs)."""
        # Filter to only POINTS items (exclude IMAGE items that may share the same layer_key)
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

        # Update global component tracker with ALL items (images + points) to stay in sync
        self._update_global_component_values(stack_components, layer_items)

        # Get global component values (union of all layers with same stack_components)
        global_component_values = self._get_global_component_values(stack_components)

        # Build nD points data using ONLY the points items BUT with global component values
        points_data, properties = _build_nd_points(
            points_items, stack_components, global_component_values
        )

        # Create or update the points layer
        _create_or_update_points_layer(
            self.viewer, self.layer_state.layers, layer_key, points_data, properties
        )

        logger.info(
            f"🔬 NAPARI PROCESS: Created points layer {layer_key} with {len(points_data)} points"
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

    def handle_data_message(self, message: Dict[str, Any]):
        """Handle incoming image data - called by process_messages()."""
        # This will be called from the Qt timer
        pass

    def display_image(self, image_data: np.ndarray, metadata: dict) -> None:
        """Display a single image payload (best-effort helper)."""
        image_info = {
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

        msg_type = data.get("type")

        # Check message type
        if msg_type == "batch":
            # Handle batch of images/shapes
            images = data.get("images", [])
            display_config_dict = data.get("display_config")

            # Extract component names metadata for dimension labels (e.g., channel names)
            component_names_metadata = data.get("component_names_metadata", {})
            if component_names_metadata:
                # Update server's component metadata cache
                self.component_metadata.update(component_names_metadata)
                logger.info(
                    f"🔬 NAPARI PROCESS: Updated component metadata: {list(component_names_metadata.keys())}"
                )

            for image_info in images:
                self._process_single_image(image_info, display_config_dict)

        else:
            # Handle single image (legacy)
            self._process_single_image(data, data.get("display_config"))

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

        path = image_info.get("path", "unknown")
        image_id = image_info.get("image_id")  # UUID for acknowledgment
        data_type = image_info.get(
            "data_type", "image"
        )  # 'image', 'shapes', or 'points'
        component_metadata = image_info.get("metadata", {})

        # Log incoming metadata to debug well filtering issues
        logger.info(
            f"🔍 NAPARI PROCESS: Received {data_type} with metadata: {component_metadata} (path: {path})"
        )

        try:
            # Check if this is shapes or points data
            if data_type == "shapes" or data_type == "points":
                # Handle shapes/ROIs/points - just pass the shapes data directly
                shapes_data = image_info.get("shapes", [])
                data = shapes_data
                colormap = None  # Shapes/points don't use colormap
            else:
                # Handle image data - load from shared memory or direct data
                shape = image_info.get("shape")
                dtype = image_info.get("dtype")
                shm_name = image_info.get("shm_name")
                direct_data = image_info.get("data")

                # Load image data
                if shm_name:
                    from multiprocessing import shared_memory

                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        data = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
                        shm.close()
                        # Unlink shared memory after copying - viewer is responsible for cleanup
                        try:
                            shm.unlink()
                        except FileNotFoundError:
                            # Already unlinked (race condition or duplicate message)
                            logger.debug(
                                f"🔬 NAPARI PROCESS: Shared memory {shm_name} already unlinked"
                            )
                        except Exception as e:
                            logger.warning(
                                f"🔬 NAPARI PROCESS: Failed to unlink shared memory {shm_name}: {e}"
                            )
                    except FileNotFoundError:
                        # Shared memory doesn't exist - likely already processed and unlinked
                        logger.error(
                            f"🔬 NAPARI PROCESS: Shared memory {shm_name} not found - may have been already processed"
                        )
                        if image_id:
                            self._send_ack(
                                image_id,
                                status=_ACK_ERROR,
                                error=f"Shared memory {shm_name} not found",
                            )
                        return
                    except Exception as e:
                        logger.error(
                            f"🔬 NAPARI PROCESS: Failed to open shared memory {shm_name}: {e}"
                        )
                        if image_id:
                            self._send_ack(
                                image_id,
                                status=_ACK_ERROR,
                                error=f"Failed to open shared memory: {e}",
                            )
                        raise
                elif direct_data:
                    data = np.array(direct_data, dtype=dtype).reshape(shape)
                else:
                    logger.warning("🔬 NAPARI PROCESS: No image data in message")
                    if image_id:
                        self._send_ack(
                            image_id, status=_ACK_ERROR, error="No image data in message"
                        )
                    return

                # Extract colormap
                colormap = "viridis"
                if display_config_dict and "colormap" in display_config_dict:
                    colormap = display_config_dict["colormap"]

            # Component-aware layer management (handles both images and shapes)
            _handle_component_aware_display(
                self.viewer,
                self.layer_state.layers,
                self.component_groups,
                data,
                path,
                colormap,
                display_config_dict or {},
                self.replace_layers,
                component_metadata,
                data_type,
                server=self,
            )

            # Send acknowledgment that data was successfully displayed
            if image_id:
                self._send_ack(image_id, status=_ACK_SUCCESS)

        except Exception as e:
            logger.error(
                f"🔬 NAPARI PROCESS: Failed to process {data_type} {path}: {e}",
                exc_info=True,
            )
            if image_id:
                self._send_ack(image_id, status=_ACK_ERROR, error=str(e))
            # Don't re-raise - continue processing other messages instead of crashing


def _spawn_detached_napari_process(
    port: int,
    viewer_title: str,
    replace_layers: bool = False,
    transport_mode: OpenHCSTransportMode = OpenHCSTransportMode.IPC,
) -> subprocess.Popen:
    """
    Spawn a completely detached napari viewer process that survives parent termination.

    This creates a subprocess that runs independently and won't be terminated when
    the parent process exits, enabling true persistence across pipeline runs.

    Args:
        port: ZMQ port to listen on
        viewer_title: Title for the napari viewer window
        replace_layers: If True, replace existing layers; if False, add new layers
        transport_mode: ZMQ transport mode (IPC or TCP)
    """
    try:
        launch_request = NapariDetachedProcessRequest.from_legacy_signature(
            port,
            viewer_title,
            replace_layers,
            transport_mode,
        )
        process = launch_request.launch()

        logger.info(
            "🔬 VISUALIZER: Detached napari process started (PID: %s), logging to %s",
            process.pid,
            launch_request.log_file,
        )
        return process

    except Exception as e:
        logger.error(f"🔬 VISUALIZER: Failed to spawn detached napari process: {e}")
        raise e


class NapariStreamVisualizer(ManagedViewerLifecycleMixin, VisualizerProcessManager):
    """
    Manages a Napari viewer instance for real-time visualization of tensors
    streamed from the OpenHCS pipeline. Runs napari in a separate process
    for Qt compatibility and true persistence across pipeline runs.
    """

    viewer_process_label = "Napari"

    def __init__(
        self,
        filemanager: FileManager,
        visualizer_config,
        viewer_title: str = "OpenHCS Real-Time Visualization",
        persistent: bool = True,
        port: int = None,
        replace_layers: bool = False,
        display_config=None,
        transport_mode: OpenHCSTransportMode = OpenHCSTransportMode.IPC,
    ):
        self.filemanager = filemanager
        self.viewer_title = viewer_title
        self.persistent = (
            persistent  # If True, viewer process stays alive after pipeline completion
        )
        self.visualizer_config = visualizer_config
        # Use config class default if not specified
        self.port = (
            port
            if port is not None
            else NapariStreamingConfig.__dataclass_fields__["port"].default
        )
        super().__init__(port=self.port)
        self.replace_layers = (
            replace_layers  # If True, replace existing layers; if False, add new layers
        )
        self.display_config = display_config  # Configuration for display behavior
        self.transport_mode = (
            coerce_transport_mode(transport_mode) or ZMQTransportMode.IPC
        )  # ZMQ transport mode (IPC or TCP)
        self.process: Optional[subprocess.Popen] = None
        self.process_handle: Optional[ViewerProcessHandle] = None
        self.zmq_context: Optional[zmq.Context] = None
        self.zmq_socket: Optional[zmq.Socket] = None
        self.lifecycle_state = ViewerLifecycleState.stopped()
        self._lock = threading.Lock()

        # Clause 368: Visualization must be observer-only.
        # This class will only read data and display it.

    def check_connected_viewer(self) -> bool:
        """Quick ping check to verify viewer is responsive (for connected viewers)."""
        return ViewerControlPingRequest.from_mode(
            mode=ViewerControlPingMode.QUICK,
            port=self.port,
            transport_mode=self.transport_mode,
            config=OPENHCS_ZMQ_CONFIG,
        ).check()

    def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait for the viewer to be ready to receive images.

        This method blocks until the viewer is responsive or the timeout expires.
        Should be called after start_viewer() when using async_mode=True.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if viewer is ready, False if timeout
        """
        return self._wait_for_viewer_ready(timeout=timeout)

    def _find_free_port(self) -> int:
        """Find a free port for ZeroMQ communication."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def start_viewer(self, async_mode: bool = True):
        """
        Starts the Napari viewer in a separate process.

        Args:
            async_mode: If True, start viewer asynchronously in background thread.
                       If False, wait for viewer to be ready before returning (legacy behavior).
        """
        if async_mode:
            # Start viewer asynchronously in background thread
            thread = threading.Thread(target=self._start_viewer_sync, daemon=True)
            thread.start()
            logger.info(
                f"🔬 VISUALIZER: Starting napari viewer asynchronously on port {self.port}"
            )
        else:
            # Legacy synchronous mode
            self._start_viewer_sync()

    def _start_viewer_sync(self):
        """Internal synchronous viewer startup (called by start_viewer)."""
        global _global_viewer_process, _global_viewer_port

        with self._lock:
            # Check if there's already a napari viewer running on the configured port
            port_in_use = is_port_in_use(
                self.port, self.transport_mode, config=OPENHCS_ZMQ_CONFIG
            )
            logger.info(f"🔬 VISUALIZER: Port {self.port} in use: {port_in_use}")

            if port_in_use:
                # Try to connect to existing viewer first before killing it
                logger.info(
                    f"🔬 VISUALIZER: Port {self.port} is in use, attempting to connect to existing viewer..."
                )
                if self._try_connect_to_existing_viewer(self.port):
                    logger.info(
                        f"🔬 VISUALIZER: Successfully connected to existing viewer on port {self.port}"
                    )
                    self.lifecycle_state.mark_connected_external()
                    return
                else:
                    # Existing viewer is unresponsive - kill it and start fresh
                    logger.info(
                        f"🔬 VISUALIZER: Existing viewer on port {self.port} is unresponsive, killing and restarting..."
                    )
                    # Use shared method from ZMQServer ABC
                    from zmqruntime.server import ZMQServer
                    from openhcs.constants.constants import CONTROL_PORT_OFFSET

                    ZMQServer.kill_processes_on_port(self.port)
                    ZMQServer.kill_processes_on_port(self.port + CONTROL_PORT_OFFSET)
                    # Wait a moment for ports to be freed
                    import time

                    time.sleep(0.5)

            if self.lifecycle_state.is_active:
                logger.warning("Napari viewer is already running.")
                return

            # Port is already set in __init__
            logger.info(
                f"🔬 VISUALIZER: Starting napari viewer process on port {self.port}"
            )

            # ALL viewers (persistent and non-persistent) should be detached subprocess
            # so they don't block parent process exit. The difference is only whether
            # we terminate them during cleanup.
            logger.info(
                f"🔬 VISUALIZER: Creating {'persistent' if self.persistent else 'non-persistent'} napari viewer (detached)"
            )
            self.process = _spawn_detached_napari_process(
                self.port, self.viewer_title, self.replace_layers, self.transport_mode
            )

            # Only track non-persistent viewers in global variable for test cleanup
            if not self.persistent:
                with _global_process_lock:
                    _global_viewer_process = self.process
                    _global_viewer_port = self.port

            # Set up ZeroMQ client immediately after process spawn.
            # Readiness is owned by ViewerStateManager.wait_for_ready() to avoid
            # duplicate/competing wait loops with conflicting timeout behavior.
            self._setup_zmq_client()

            # Check if process is running (different methods for subprocess vs multiprocessing)
            self.process_handle = ViewerProcessHandle.from_process(self.process)
            process_alive = self.process_handle.is_alive()

            if process_alive:
                self.lifecycle_state.mark_owned_process()
                logger.info(
                    f"🔬 VISUALIZER: Napari viewer process started successfully (PID: {self.process_handle.pid_label})"
                )
            else:
                logger.error("🔬 VISUALIZER: Failed to start napari viewer process")

    def _try_connect_to_existing_viewer(self, port: int) -> bool:
        """
        Try to connect to an existing napari viewer and verify it's responsive.

        Returns True only if we can successfully handshake with the viewer.
        """
        try:
            if ViewerControlPingRequest.from_mode(
                mode=ViewerControlPingMode.EXISTING_VIEWER,
                port=port,
                transport_mode=self.transport_mode,
                config=OPENHCS_ZMQ_CONFIG,
            ).check():
                self._setup_zmq_client()
                return True
            return False
        except Exception as e:
            logger.debug(f"Failed to connect to existing viewer on port {port}: {e}")
            return False

    def _wait_for_viewer_ready(self, timeout: float = 10.0) -> bool:
        """Wait for the napari viewer to be ready using handshake protocol."""
        logger.info(
            f"🔬 VISUALIZER: Waiting for napari viewer to be ready on port {self.port}..."
        )
        ready = wait_for_server_ready(
            self.port,
            self.transport_mode,
            host="localhost",
            config=OPENHCS_ZMQ_CONFIG,
            timeout=timeout,
            require_ready=True,
        )
        if ready:
            logger.info(f"🔬 VISUALIZER: Napari viewer is ready on port {self.port}")
            return True

        logger.warning("🔬 VISUALIZER: Timeout waiting for napari viewer handshake")
        return False

    def _setup_zmq_client(self):
        """Set up ZeroMQ client to send data to viewer process."""
        if self.port is None:
            raise RuntimeError("Port not set - call start_viewer() first")

        data_url = get_zmq_transport_url(
            self.port,
            host="localhost",
            mode=self.transport_mode,
            config=OPENHCS_ZMQ_CONFIG,
        )

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.connect(data_url)

        # Brief delay for ZMQ connection to establish
        time.sleep(ZMQ_CONNECTION_DELAY_MS / 1000.0)
        logger.info(f"🔬 VISUALIZER: ZMQ client connected to {data_url}")

    def send_control_message(self, message_type: str, timeout: float = 2.0) -> bool:
        """
        Send a control message to the viewer.

        Args:
            message_type: Type of control message ('clear_state', 'shutdown', etc.)
            timeout: Timeout in seconds for waiting for response

        Returns:
            True if message was sent and acknowledged, False otherwise
        """
        if not self.is_running or self.port is None:
            logger.warning(
                f"🔬 VISUALIZER: Cannot send {message_type} - viewer not running"
            )
            return False

        control_url = get_control_url(
            self.port,
            self.transport_mode,
            host="localhost",
            config=OPENHCS_ZMQ_CONFIG,
        )
        control_context = None
        control_socket = None

        try:
            control_context = zmq.Context()
            control_socket = control_context.socket(zmq.REQ)
            control_socket.setsockopt(zmq.LINGER, 0)
            control_socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            control_socket.connect(control_url)

            # Send control message
            message = {"type": message_type}
            control_socket.send(pickle.dumps(message))

            # Wait for acknowledgment
            response = control_socket.recv()
            response_data = pickle.loads(response)

            if response_data.get("status") == "success":
                logger.info(f"🔬 VISUALIZER: {message_type} acknowledged by viewer")
                return True
            else:
                logger.warning(f"🔬 VISUALIZER: {message_type} failed: {response_data}")
                return False

        except zmq.Again:
            logger.warning(
                f"🔬 VISUALIZER: Timeout waiting for {message_type} acknowledgment"
            )
            return False
        except Exception as e:
            logger.warning(f"🔬 VISUALIZER: Failed to send {message_type}: {e}")
            return False
        finally:
            if control_socket:
                try:
                    control_socket.close()
                except Exception as e:
                    logger.debug(f"Failed to close control socket: {e}")
            if control_context:
                try:
                    control_context.term()
                except Exception as e:
                    logger.debug(f"Failed to terminate control context: {e}")

    def clear_viewer_state(self) -> bool:
        """
        Clear accumulated viewer state (component groups) for a new pipeline run.

        Returns:
            True if state was cleared successfully, False otherwise
        """
        return self.send_control_message("clear_state")

    def send_image_data(
        self, step_id: str, image_data: np.ndarray, axis_id: str = "unknown"
    ):
        """
        DISABLED: This method bypasses component-aware stacking.
        All visualization must go through the streaming backend.
        """
        raise RuntimeError(
            f"send_image_data() is disabled. Use streaming backend for component-aware display. "
            f"step_id: {step_id}, axis_id: {axis_id}, shape: {image_data.shape}"
        )

    def stop_viewer(self):
        """Stop the napari viewer process (only if not persistent)."""
        with self._lock:
            if not self.persistent:
                logger.info("🔬 VISUALIZER: Stopping non-persistent napari viewer")
                self._cleanup_zmq()
                if self.process:
                    killed = ViewerProcessHandle.from_process(self.process).terminate()
                    if killed:
                        logger.warning(
                            "🔬 VISUALIZER: Force killing napari viewer process"
                        )
                self.lifecycle_state.mark_stopped()
            else:
                logger.info("🔬 VISUALIZER: Keeping persistent napari viewer alive")
                # Just cleanup our ZMQ connection, leave process running
                self._cleanup_zmq()
                # DON'T set is_running = False for persistent viewers!
                # The process is still alive and should be reusable

    def _cleanup_zmq(self):
        """Clean up ZeroMQ resources."""
        if self.zmq_socket:
            self.zmq_socket.close()
            self.zmq_socket = None
        if self.zmq_context:
            self.zmq_context.term()
            self.zmq_context = None

    def visualize_path(
        self, step_id: str, path: str, backend: str, axis_id: Optional[str] = None
    ):
        """
        DISABLED: This method bypasses component-aware stacking.
        All visualization must go through the streaming backend.
        """
        raise RuntimeError(
            f"visualize_path() is disabled. Use streaming backend for component-aware display. "
            f"Path: {path}, step_id: {step_id}, axis_id: {axis_id}"
        )

    def get_launch_command(self) -> list[str]:
        import sys

        launch_request = NapariDetachedProcessRequest.from_legacy_signature(
            self.port,
            self.viewer_title,
            self.replace_layers,
            self.transport_mode,
        )
        python_code = launch_request.to_process_request().python_code

        return [sys.executable, "-c", python_code]

    def get_launch_env(self) -> dict:
        import os

        env = os.environ.copy()
        return ViewerQtEnvironmentPolicy().apply_to(env)

    def start(self, detached: bool = True):
        self.start_viewer(async_mode=False)
        return self.process

    def stop(self, timeout: float = 5.0):
        self.stop_viewer()
