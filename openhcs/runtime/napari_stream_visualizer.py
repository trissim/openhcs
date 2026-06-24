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

from __future__ import annotations

import logging
import subprocess
import threading
import time
import zmq
import numpy as np
from pathlib import Path
from typing import Optional

from polystore.backend_registry import register_cleanup_callback
from polystore.filemanager import FileManager
from openhcs.utils.import_utils import optional_import
from openhcs.core.streaming_config_factory import StreamingViewerRuntimeConfig
from openhcs.runtime.viewer_protocol import (
    DetachedViewerPythonArguments,
    DetachedViewerPythonExpression,
    DetachedViewerServerEntrypointSpec,
    ManagedViewerLifecycleMixin,
    ViewerProcessHandle,
    ViewerType,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG

# Optional napari import - this module should only be imported if napari is available
napari = optional_import("napari")
if napari is None:
    raise ImportError(
        "napari is required for NapariStreamVisualizer. "
        "Install it with: pip install 'openhcs[viz]' or pip install napari"
    )


logger = logging.getLogger(__name__)

# ZMQ connection delay (ms)
ZMQ_CONNECTION_DELAY_MS = 100  # Brief delay for ZMQ connection to establish

# Global process management for napari viewer
_global_viewer_process: Optional[subprocess.Popen[bytes]] = None
_global_process_lock = threading.Lock()

NAPARI_VIEWER_ENTRYPOINT = DetachedViewerServerEntrypointSpec(
    viewer_type=ViewerType.NAPARI,
    module_name="openhcs.runtime.napari_viewer_server",
    function_name="run_napari_viewer_process",
)

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


class NapariStreamVisualizer(ManagedViewerLifecycleMixin):
    """
    Manages a Napari viewer instance for real-time visualization of tensors
    streamed from the OpenHCS pipeline. Runs napari in a separate process
    for Qt compatibility and true persistence across pipeline runs.
    """

    viewer_process_label = "Napari"
    detached_server_entrypoint = NAPARI_VIEWER_ENTRYPOINT

    def __init__(
        self,
        *,
        filemanager: FileManager,
        runtime_config: StreamingViewerRuntimeConfig,
        replace_layers: bool = False,
    ):
        super().__init__(
            runtime_config=runtime_config,
            transport_config=OPENHCS_ZMQ_CONFIG,
        )
        self.filemanager = filemanager
        self.replace_layers = replace_layers
        self.zmq_context: Optional[zmq.Context] = None
        self.zmq_socket: Optional[zmq.Socket] = None

        # Clause 368: Visualization must be observer-only.
        # This class will only read data and display it.

    def detached_server_arguments(
        self,
        *,
        log_file: Path,
    ) -> DetachedViewerPythonArguments:
        return DetachedViewerPythonArguments.from_literals(
            self.required_port,
            self.viewer_title,
            self.replace_layers,
            str(log_file),
        ).append(DetachedViewerPythonExpression.symbol("transport_mode"))

    def start_viewer(self, async_mode: bool = True):
        """
        Starts the Napari viewer in a separate process.

        Args:
            async_mode: If True, start viewer asynchronously in background thread.
                       If False, wait for viewer to be ready before returning.
        """
        if async_mode:
            # Start viewer asynchronously in background thread
            thread = threading.Thread(target=self._start_viewer_sync, daemon=True)
            thread.start()
            logger.info(
                f"🔬 VISUALIZER: Starting napari viewer asynchronously on port {self.required_port}"
            )
        else:
            # Legacy synchronous mode
            self._start_viewer_sync()

    def _start_viewer_sync(self):
        """Internal synchronous viewer startup (called by start_viewer)."""
        global _global_viewer_process

        with self._lock:
            port = self.required_port
            self.prepare_fresh_viewer_start()

            if self.lifecycle_state.is_active:
                logger.warning("Napari viewer is already running.")
                return

            # Port is already set in __init__
            logger.info(
                f"🔬 VISUALIZER: Starting napari viewer process on port {port}"
            )

            # ALL viewers (persistent and non-persistent) should be detached subprocess
            # so they don't block parent process exit. The difference is only whether
            # we terminate them during cleanup.
            logger.info(
                f"🔬 VISUALIZER: Creating {self.persistence_label} napari viewer (detached)"
            )
            self.process = self.launch_detached_viewer()

            # Only track non-persistent viewers in global variable for test cleanup
            if not self.persistent:
                with _global_process_lock:
                    _global_viewer_process = self.process

            # Set up ZeroMQ client immediately after process spawn.
            # Readiness is owned by ViewerStateManager.wait_for_ready() to avoid
            # duplicate/competing wait loops with conflicting timeout behavior.
            self._setup_zmq_client()

            # Check if process is running (different methods for subprocess vs multiprocessing)
            process_alive = ViewerProcessHandle.from_process(self.process).is_alive()

            if process_alive:
                self.lifecycle_state.mark_owned_process()
                logger.info(
                    f"🔬 VISUALIZER: Napari viewer process started successfully (PID: {self.process_pid_label})"
                )
            else:
                logger.error("🔬 VISUALIZER: Failed to start napari viewer process")

    def _setup_zmq_client(self):
        """Set up ZeroMQ client to send data to viewer process."""
        data_url = self.runtime_endpoint.data_url()

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.connect(data_url)

        # Brief delay for ZMQ connection to establish
        time.sleep(ZMQ_CONNECTION_DELAY_MS / 1000.0)
        logger.info(f"🔬 VISUALIZER: ZMQ client connected to {data_url}")

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

    def cleanup_viewer_client(self) -> None:
        self._cleanup_zmq()

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

    def stop(self, timeout: float = 5.0):
        self.stop_viewer()
