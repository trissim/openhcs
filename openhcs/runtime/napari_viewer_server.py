"""
Napari-based real-time visualization module.

This module provides the NapariStreamVisualizer class for real-time
visualization of tensors during pipeline execution.
"""

import logging
import multiprocessing
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

from zmqruntime.config import TransportMode, ZMQConfig
from polystore.streaming_constants import StreamingDataType
from polystore.streaming.receivers.napari import (
    normalize_component_layout,
    build_layer_key,
)
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


def _parse_component_info_from_path(path_str: str):
    """
    Fallback component parsing from path (used when component metadata unavailable).

    Args:
        path_str: Path string like 'step_name/A01/s1_c2_z3.tif'

    Returns:
        Dict with basic component info extracted from filename
    """
    try:
        import os
        import re

        filename = os.path.basename(path_str)

        # Basic regex for common patterns
        pattern = r"(?:s(\d+))?(?:_c(\d+))?(?:_z(\d+))?"
        match = re.search(pattern, filename)

        components = {}
        if match:
            site, channel, z_index = match.groups()
            if site:
                components["site"] = site
            if channel:
                components["channel"] = channel
            if z_index:
                components["z_index"] = z_index

        return components
    except Exception:
        return {}


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


def _create_or_update_layer(
    viewer, layers, layer_name, layer_type, data, **layer_kwargs
):
    """
    Create or update a Napari layer of any type.

    All layers are handled identically: if layer exists, remove and recreate.
    This ensures consistent behavior across all layer types.

    Args:
        viewer: Napari viewer
        layers: Dict of existing layers
        layer_name: Name for the layer
        layer_type: Type of layer ('image', 'shapes', 'points', etc.)
        data: Data for the layer (format depends on layer_type)
        **layer_kwargs: Additional kwargs to pass to viewer.add_<layer_type>()

    Returns:
        The created layer
    """
    # Check if layer exists
    existing_layer = None
    for layer in viewer.layers:
        if layer.name == layer_name:
            existing_layer = layer
            break

    # Remove existing layer if present
    if existing_layer is not None:
        viewer.layers.remove(existing_layer)
        layers.pop(layer_name, None)
        logger.info(
            f"🔬 NAPARI PROCESS: Removed existing {layer_type} layer {layer_name} for recreation"
        )

    # Get the add_* method for this layer type and create new layer
    add_method = getattr(viewer, f"add_{layer_type}")
    new_layer = add_method(data, name=layer_name, **layer_kwargs)
    layers[layer_name] = new_layer

    # Log with appropriate count/info
    if layer_type == "shapes":
        count = len(data) if hasattr(data, "__len__") else 0
        logger.info(
            f"🔬 NAPARI PROCESS: Created {layer_type} layer {layer_name} with {count} shapes"
        )
    elif layer_type == "points":
        count = len(data) if hasattr(data, "__len__") else 0
        logger.info(
            f"🔬 NAPARI PROCESS: Created {layer_type} layer {layer_name} with {count} points"
        )
    else:
        logger.info(f"🔬 NAPARI PROCESS: Created {layer_type} layer {layer_name}")

    return new_layer


# Convenience wrappers that call the unified function
def _create_or_update_image_layer(
    viewer, layers, layer_name, image_data, colormap, axis_labels=None
):
    """Create or update a Napari image layer."""
    layer = _create_or_update_layer(
        viewer, layers, layer_name, "image", image_data, colormap=colormap or "gray"
    )
    # Set axis labels on viewer.dims (add_image axis_labels parameter doesn't work)
    if axis_labels is not None:
        viewer.dims.axis_labels = axis_labels
        logger.info(f"🔬 NAPARI PROCESS: Set viewer.dims.axis_labels={axis_labels}")
    return layer


def _create_or_update_shapes_layer(
    viewer, layers, layer_name, shapes_data, shape_types, properties
):
    """Create or update a Napari shapes layer."""
    return _create_or_update_layer(
        viewer,
        layers,
        layer_name,
        "shapes",
        shapes_data,
        shape_type=shape_types,
        properties=properties,
        edge_color="red",
        face_color="transparent",
        edge_width=2,
    )


def _create_or_update_points_layer(viewer, layers, layer_name, points_data, properties):
    """Create or update a Napari points layer."""
    return _create_or_update_layer(
        viewer,
        layers,
        layer_name,
        "points",
        points_data,
        properties=properties,
        face_color="green",
        size=3,
    )


# Populate registry now that helper functions are defined
from polystore.streaming_constants import StreamingDataType

_DATA_TYPE_HANDLERS = {
    StreamingDataType.IMAGE: {
        "build_nd_data": _build_nd_image_array,
        "create_layer": _create_or_update_image_layer,
    },
    StreamingDataType.SHAPES: {
        "build_nd_data": _build_nd_shapes,
        "create_layer": _create_or_update_shapes_layer,
    },
    StreamingDataType.POINTS: {
        "build_nd_data": _build_nd_points,
        "create_layer": _create_or_update_points_layer,
    },
}


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

        # Initialize with REP socket for receiving images (synchronous request/reply)
        # REP socket forces workers to wait for acknowledgment before closing shared memory
        super().__init__(
            port,
            viewer_type="napari",
            host="*",
            log_file_path=log_file_path,
            data_socket_type=zmq.REP,
            transport_mode=coerce_transport_mode(transport_mode),
            config=OPENHCS_ZMQ_CONFIG,
        )

        self.viewer_title = viewer_title
        self.replace_layers = replace_layers
        self.viewer = None
        self.layers = {}
        self.component_groups = {}
        self.dimension_labels = {}  # Store dimension label mappings: layer_key -> {component: [labels]}
        self.component_metadata = {}  # Store component metadata from microscope handler: {component: {id: name}}

        # Global component value tracker for shared dimension mapping
        # Maps tuple of stack_components -> {component: set of values}
        # All layers with the same stack_components share the same global mapping
        self.global_component_values = {}

        # Debouncing + locking for layer updates to prevent race conditions
        import threading

        self.layer_update_lock = threading.Lock()  # Prevent concurrent updates
        self.pending_updates = {}  # layer_key -> QTimer (debounce)
        self.update_delay_ms = 1000  # Wait 200ms for more items before rebuilding

        # Batch processors for efficient accumulation and display
        # Uses batch_size from config if available (defaults to None = wait for all)
        self._batch_processors = {}  # layer_key -> NapariBatchProcessor
        self._batch_processor_lock = threading.Lock()

        # Ack socket handled by StreamingVisualizerServer

    def _setup_ack_socket(self):
        """Setup PUSH socket for sending acknowledgments."""
        super()._setup_ack_socket()

    def _get_or_create_batch_processor(
        self, layer_key: str, batch_size: Optional[int] = None
    ):
        """Get or create a batch processor for the given layer key."""
        with self._batch_processor_lock:
            if layer_key not in self._batch_processors:
                # Import here to avoid circular imports
                from polystore.streaming.receivers.napari import NapariBatchProcessor

                processor = NapariBatchProcessor(
                    napari_server=self,
                    batch_size=batch_size,
                    debounce_delay_ms=self.update_delay_ms,
                    max_debounce_wait_ms=5000,  # 5 second max wait
                )
                self._batch_processors[layer_key] = processor
                logger.info(
                    f"NapariViewerServer: Created batch processor for layer '{layer_key}' "
                    f"with batch_size={batch_size}"
                )

            return self._batch_processors[layer_key]

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
        if layer_key in self.pending_updates:
            self.pending_updates[layer_key].stop()
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
        self.pending_updates[layer_key] = timer
        logger.debug(
            f"🔬 NAPARI PROCESS: Scheduled update for {layer_key} in {self.update_delay_ms}ms"
        )

    def _execute_layer_update(
        self, layer_key, data_type, component_modes, component_order
    ):
        """
        Execute the actual layer update after debounce delay.

        Uses batch processor for efficient accumulation and display.
        """
        # Remove timer
        self.pending_updates.pop(layer_key, None)

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

        # Use batch processor for efficient accumulation and display
        # This avoids recreating layers from scratch on every update
        batch_processor = self._get_or_create_batch_processor(layer_key)
        batch_processor.add_items(
            layer_key=layer_key,
            items=layer_items,
            display_config={"component_modes": component_modes},
            component_names_metadata=self.component_metadata,
        )

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
        layer_labels = self.dimension_labels.get(layer_key, {})
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
            if channel_value == 1:
                colormap = "green"
            elif channel_value == 2:
                colormap = "red"

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

        # Component abbreviation mapping
        COMPONENT_ABBREV = {
            "channel": "Ch",
            "z_index": "Z",
            "timepoint": "T",
            "site": "Site",
            "well": "Well",
        }

        for comp in stack_components:
            # Use global component values instead of just this layer's values
            values = global_component_values[comp]

            # Try to get human-readable labels from metadata if available
            labels = []

            # Check if we have metadata for this component type
            comp_metadata = self.component_metadata.get(comp, {})

            for v in values:
                # First try to get name from metadata (e.g., channel name)
                metadata_name = comp_metadata.get(str(v))

                if metadata_name and str(metadata_name).lower() != "none":
                    # Use metadata name with index for clarity
                    if comp == "channel":
                        labels.append(f"Ch{v}: {metadata_name}")
                    elif comp == "well":
                        labels.append(
                            f"{metadata_name}"
                        )  # Well names are already good (e.g., "A01")
                    else:
                        labels.append(f"{comp.title()} {v}: {metadata_name}")
                else:
                    # No metadata - use abbreviated component name + index
                    abbrev = COMPONENT_ABBREV.get(comp, comp)
                    labels.append(f"{abbrev} {v}")

            dimension_labels[comp] = labels

        # Store dimension labels for this layer
        self.dimension_labels[layer_key] = dimension_labels

        # Create or update the layer
        _create_or_update_image_layer(
            self.viewer, self.layers, layer_key, stacked_data, colormap, axis_labels
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
        labels_data = self._shapes_to_labels(
            layer_items, stack_components, global_component_values
        )

        # Remove existing layer if it exists
        if layer_key in self.layers:
            try:
                self.viewer.layers.remove(self.layers[layer_key])
                logger.info(
                    f"🔬 NAPARI PROCESS: Removed existing labels layer {layer_key} for recreation"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to remove existing labels layer {layer_key}: {e}"
                )

        # Create new labels layer
        new_layer = self.viewer.add_labels(labels_data, name=layer_key)
        self.layers[layer_key] = new_layer
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
            self.viewer, self.layers, layer_key, points_data, properties
        )

        logger.info(
            f"🔬 NAPARI PROCESS: Created points layer {layer_key} with {len(points_data)} points"
        )

    def _shapes_to_labels(self, layer_items, stack_components, component_values):
        """Convert shapes data to label masks.

        Args:
            layer_items: List of shape items to convert
            stack_components: List of component names for stack dimensions
            component_values: Dict of {component: [sorted values]} from global tracker
        """
        from skimage import draw

        # Use global component values passed in
        # This ensures ROIs and images share the same component-to-index mapping
        logger.info(
            f"🔬 NAPARI PROCESS: Building ROI stack with global component values"
        )
        for comp, values in component_values.items():
            logger.info(f"🔬 NAPARI PROCESS:   {comp}: {values}")

        # Determine output shape
        # Get image shape from first item's shapes data
        first_shapes = layer_items[0]["data"]
        if not first_shapes:
            # No shapes, return empty array with reasonable default size
            logger.warning(
                "🔬 NAPARI PROCESS: No shapes data, creating default 512x512 array"
            )
            return np.zeros((1, 1, 512, 512), dtype=np.uint16)

        # Estimate image size from shape coordinates
        max_y, max_x = 0, 0
        for shape_dict in first_shapes:
            if shape_dict["type"] == "polygon":
                coords = np.array(shape_dict["coordinates"])
                max_y = max(max_y, int(np.max(coords[:, 0])) + 1)
                max_x = max(max_x, int(np.max(coords[:, 1])) + 1)
            elif shape_dict["type"] == "path":
                # Handle path (polyline) type - get bounding box
                coords = np.array(shape_dict["coordinates"])
                if len(coords) > 0:
                    max_y = max(max_y, int(np.max(coords[:, 0])) + 1)
                    max_x = max(max_x, int(np.max(coords[:, 1])) + 1)
            elif shape_dict["type"] == "points":
                # Handle points type - get bounding box
                coords = np.array(shape_dict["coordinates"])
                if len(coords) > 0:
                    max_y = max(max_y, int(np.max(coords[:, 0])) + 1)
                    max_x = max(max_x, int(np.max(coords[:, 1])) + 1)

        # Ensure minimum valid dimensions (avoid 0x0 shapes)
        if max_y == 0 or max_x == 0:
            logger.warning(
                f"🔬 NAPARI PROCESS: Invalid shape dimensions (y={max_y}, x={max_x}), using default 512x512"
            )
            max_y = max(max_y, 512)
            max_x = max(max_x, 512)

        # Build nD shape
        nd_shape = []
        for comp in stack_components:
            nd_shape.append(len(component_values[comp]))
        nd_shape.extend([max_y, max_x])

        # Create empty label array
        labels_array = np.zeros(nd_shape, dtype=np.uint16)

        # Fill in labels for each item
        label_id = 1
        for item in layer_items:
            # Get indices for this item
            indices = []
            for comp in stack_components:
                comp_value = item["components"].get(comp, 0)
                idx = component_values[comp].index(comp_value)
                indices.append(idx)

            # Get shapes data
            shapes_data = item["data"]

            # Draw each shape into the label mask
            for shape_dict in shapes_data:
                if shape_dict["type"] == "polygon":
                    coords = np.array(shape_dict["coordinates"])
                    rr, cc = draw.polygon(
                        coords[:, 0], coords[:, 1], shape=labels_array.shape[-2:]
                    )

                    # Set label at the correct nD position
                    full_indices = tuple(indices) + (rr, cc)
                    labels_array[full_indices] = label_id
                    label_id += 1

                elif shape_dict["type"] == "path":
                    # Draw polyline (skeleton branches)
                    # Skeleton paths from skan already contain every pixel in the branch,
                    # so we just set the label at each coordinate directly (no line drawing needed)
                    coords = np.array(shape_dict["coordinates"])
                    if len(coords) >= 1:
                        # Extract row and column indices
                        rr = coords[:, 0].astype(int)
                        cc = coords[:, 1].astype(int)

                        # Clip to image bounds
                        valid = (
                            (rr >= 0)
                            & (rr < labels_array.shape[-2])
                            & (cc >= 0)
                            & (cc < labels_array.shape[-1])
                        )
                        rr, cc = rr[valid], cc[valid]

                        # Set label at the correct nD position
                        full_indices = tuple(indices) + (rr, cc)
                        labels_array[full_indices] = label_id
                        label_id += 1

        logger.info(
            f"🔬 NAPARI PROCESS: Created labels array with shape {labels_array.shape} and {label_id - 1} labels"
        )
        return labels_array

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
        response = super()._create_pong_response()
        response["viewer"] = "napari"
        response["openhcs"] = True
        response["server"] = "NapariViewer"

        # Add memory usage
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            response["memory_mb"] = process.memory_info().rss / 1024 / 1024
            response["cpu_percent"] = process.cpu_percent(interval=0)
        except Exception:
            pass

        return response

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
                                status="error",
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
                                status="error",
                                error=f"Failed to open shared memory: {e}",
                            )
                        raise
                elif direct_data:
                    data = np.array(direct_data, dtype=dtype).reshape(shape)
                else:
                    logger.warning("🔬 NAPARI PROCESS: No image data in message")
                    if image_id:
                        self._send_ack(
                            image_id, status="error", error="No image data in message"
                        )
                    return

                # Extract colormap
                colormap = "viridis"
                if display_config_dict and "colormap" in display_config_dict:
                    colormap = display_config_dict["colormap"]

            # Component-aware layer management (handles both images and shapes)
            _handle_component_aware_display(
                self.viewer,
                self.layers,
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
                self._send_ack(image_id, status="success")

        except Exception as e:
            logger.error(
                f"🔬 NAPARI PROCESS: Failed to process {data_type} {path}: {e}",
                exc_info=True,
            )
            if image_id:
                self._send_ack(image_id, status="error", error=str(e))
            # Don't re-raise - continue processing other messages instead of crashing


def _napari_viewer_process(
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

        # Create ZMQ server instance (inherits from ZMQServer ABC)
        server = NapariViewerServer(
            port, viewer_title, replace_layers, log_file_path, transport_mode
        )

        # Start the server (binds sockets)
        server.start()

        # Create napari viewer in this process (main thread)
        viewer = napari.Viewer(title=viewer_title, show=True)
        server.viewer = viewer

        # Initialize layers dictionary with existing layers (for reconnection scenarios)
        for layer in viewer.layers:
            server.layers[layer.name] = layer

        # Set up dimension label tracking for well names
        # This will be populated as metadata arrives and used to update text overlay
        server.dimension_labels = {}  # layer_key -> {component: [label1, label2, ...]}

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

        # Ensure Qt platform is properly set for detached processes
        import os
        import platform

        if "QT_QPA_PLATFORM" not in os.environ:
            if platform.system() == "Darwin":  # macOS
                os.environ["QT_QPA_PLATFORM"] = "cocoa"
            elif platform.system() == "Linux":
                os.environ["QT_QPA_PLATFORM"] = "xcb"
                os.environ["QT_X11_NO_MITSHM"] = "1"
            # Windows doesn't need QT_QPA_PLATFORM set
        elif platform.system() == "Linux":
            # Disable shared memory for X11 (helps with display issues in detached processes)
            os.environ["QT_X11_NO_MITSHM"] = "1"

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
    # Use a simpler approach: spawn python directly with the napari viewer module
    # This avoids temporary file issues and import problems

    # Create the command to run the napari viewer directly
    current_dir = os.getcwd()
    python_code = f"""
import sys
import os

# Detach from parent process group (Unix only)
if hasattr(os, "setsid"):
    try:
        os.setsid()
    except OSError:
        pass

# Add current working directory to Python path
sys.path.insert(0, {repr(current_dir)})

try:
    from openhcs.runtime.napari_stream_visualizer import _napari_viewer_process
    from openhcs.core.config import TransportMode
    transport_mode = TransportMode.{transport_mode.name}
    _napari_viewer_process({port}, {repr(viewer_title)}, {replace_layers}, {repr(current_dir + "/.napari_log_path_placeholder")}, transport_mode)
except Exception as e:
    import logging
    logger = logging.getLogger("openhcs.runtime.napari_detached")
    logger.error(f"Detached napari error: {{e}}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
"""

    try:
        # Create log file for detached process
        log_dir = os.path.expanduser("~/.local/share/openhcs/logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"napari_detached_port_{port}.log")

        # Replace placeholder with actual log file path in python code
        python_code = python_code.replace(
            repr(current_dir + "/.napari_log_path_placeholder"), repr(log_file)
        )

        # Use subprocess.Popen with detachment flags
        if sys.platform == "win32":
            # Windows: Use CREATE_NEW_PROCESS_GROUP to detach but preserve display environment
            env = os.environ.copy()  # Preserve environment variables
            with open(log_file, "w") as log_f:
                process = subprocess.Popen(
                    [sys.executable, "-c", python_code],
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.DETACHED_PROCESS,
                    env=env,
                    cwd=os.getcwd(),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
        else:
            # Unix: Use start_new_session to detach but preserve display environment
            env = os.environ.copy()  # Preserve DISPLAY and other environment variables

            # Ensure Qt platform is set for GUI display
            import platform

            if "QT_QPA_PLATFORM" not in env:
                if platform.system() == "Darwin":  # macOS
                    env["QT_QPA_PLATFORM"] = "cocoa"
                elif platform.system() == "Linux":
                    env["QT_QPA_PLATFORM"] = "xcb"
                    env["QT_X11_NO_MITSHM"] = "1"
                # Windows doesn't need QT_QPA_PLATFORM set
            elif platform.system() == "Linux":
                # Ensure Qt can find the display
                env["QT_X11_NO_MITSHM"] = (
                    "1"  # Disable shared memory for X11 (helps with some display issues)
                )

            # Redirect stdout/stderr to log file for debugging
            log_f = open(log_file, "w")
            process = subprocess.Popen(
                [sys.executable, "-c", python_code],
                env=env,
                cwd=os.getcwd(),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # CRITICAL: Detach from parent process group
            )

        logger.info(
            f"🔬 VISUALIZER: Detached napari process started (PID: {process.pid}), logging to {log_file}"
        )
        return process

    except Exception as e:
        logger.error(f"🔬 VISUALIZER: Failed to spawn detached napari process: {e}")
        raise e
