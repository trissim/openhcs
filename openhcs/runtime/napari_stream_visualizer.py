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

from polystore.backend_registry import register_cleanup_callback
from polystore.filemanager import FileManager
from polystore.streaming.identity import StreamProducerIdentity
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
    ViewerComponentValueOrdering,
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
from openhcs.runtime.napari_viewer_server import (
    ComponentLayout,
    NapariLayerTitleAuthority,
    NapariViewerServer,
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
        values = sorted(
            set(item["components"].get(comp, 0) for item in layer_items),
            key=ViewerComponentValueOrdering.key,
        )
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
                set(item["components"].get(comp, 0) for item in layer_items),
                key=ViewerComponentValueOrdering.key,
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
            values = sorted(
                set(img["components"].get(comp, 0) for img in layer_items),
                key=ViewerComponentValueOrdering.key,
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
            component_values[comp].index(img["components"].get(comp, 0))
            for comp in stack_components
        )
        logger.debug(
            f"🔬 NAPARI PROCESS: Placing image at indices {indices}, components={img['components']}"
        )
        stacked_array[indices] = img["data"]

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


# Populate registry now that helper functions are defined
from polystore.streaming_constants import StreamingDataType
from polystore.streaming.receivers.napari import (
    build_route_key,
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
    producer_identity=None,
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
        if server is None:
            raise ValueError("Server instance required for debounced updates")

        # Normalize wire data type to enum.
        if isinstance(data_type, str):
            data_type = StreamingDataType(data_type)
        producer = StreamProducerIdentity.from_payload(producer_identity)

        # Use component metadata from ZMQ message - fail loud if not available
        if not component_metadata:
            raise ValueError(f"No component metadata available for path: {path}")
        component_info = component_metadata

        component_layout = ComponentLayout.from_display_config(display_config)
        component_modes = component_layout.component_modes
        component_order = component_layout.component_order
        layer_key = build_route_key(
            producer_identity=producer,
            component_info=component_info,
            component_modes=component_modes,
            component_order=component_order,
            data_type=data_type,
        )
        layer_title = NapariLayerTitleAuthority.disambiguate(
            title=NapariLayerTitleAuthority.title(
                producer=producer,
                data_type=data_type,
                component_info=component_info,
                component_layout=component_layout,
            ),
            producer=producer,
            route_key=layer_key,
            layer_state=server.layer_state,
        )
        server.layer_state.set_title(layer_key, layer_title)

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
            if layer_key in layers and layers[layer_key] not in viewer.layers:
                # Layer was in our cache but is now missing from viewer - user deleted it
                # Drop stale references so we will recreate the layer
                num_items = len(component_groups.get(layer_key, []))
                server.layer_state.purge_route(layer_key)
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
