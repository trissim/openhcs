"""
Fiji stream visualizer for OpenHCS.

Manages Fiji viewer instances for real-time visualization via ZMQ.
Uses FijiViewerServer (inherits from ZMQServer) for PyImageJ-based display.
Follows same architecture as NapariStreamVisualizer.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

from polystore.filemanager import FileManager
from polystore.backend_registry import register_cleanup_callback
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

logger = logging.getLogger(__name__)

# Global process management for Fiji viewer
_global_fiji_process: Optional[subprocess.Popen[bytes]] = None
_global_fiji_lock = threading.Lock()

FIJI_VIEWER_ENTRYPOINT = DetachedViewerServerEntrypointSpec(
    viewer_type=ViewerType.FIJI,
    module_name="openhcs.runtime.fiji_viewer_server",
    function_name="fiji_viewer_server_process",
    extra_imports=("from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG",),
)


def _cleanup_global_fiji_viewer() -> None:
    """Clean up global Fiji viewer process for test mode."""
    global _global_fiji_process

    with _global_fiji_lock:
        if (
            _global_fiji_process
            and ViewerProcessHandle.from_process(_global_fiji_process).is_alive()
        ):
            logger.info("🔬 FIJI VISUALIZER: Terminating Fiji viewer for test cleanup")
            killed = ViewerProcessHandle.from_process(_global_fiji_process).terminate(
                timeout=3,
                kill_timeout=1,
            )
            if killed:
                logger.warning("🔬 FIJI VISUALIZER: Force killing Fiji viewer process")

            _global_fiji_process = None


register_cleanup_callback(_cleanup_global_fiji_viewer)


class FijiStreamVisualizer(ManagedViewerLifecycleMixin):
    """
    Manages Fiji viewer instance for real-time visualization via ZMQ.

    Follows same architecture as NapariStreamVisualizer.
    """

    viewer_process_label = "Fiji"
    detached_server_entrypoint = FIJI_VIEWER_ENTRYPOINT

    def __init__(
        self,
        *,
        filemanager: FileManager,
        runtime_config: StreamingViewerRuntimeConfig,
    ):
        super().__init__(
            runtime_config=runtime_config,
            transport_config=OPENHCS_ZMQ_CONFIG,
        )
        self.filemanager = filemanager

    def detached_server_arguments(
        self,
        *,
        log_file: Path,
    ) -> DetachedViewerPythonArguments:
        return DetachedViewerPythonArguments.from_literals(
            self.required_port,
            self.viewer_title,
            None,
            str(log_file),
        ).append(
            DetachedViewerPythonExpression.symbol("transport_mode"),
            DetachedViewerPythonExpression.symbol("OPENHCS_ZMQ_CONFIG"),
        )

    def start_viewer(self, async_mode: bool = False) -> None:
        """Start Fiji viewer server process."""
        global _global_fiji_process

        with self._lock:
            port = self.required_port
            self.prepare_fresh_viewer_start()

            if self.lifecycle_state.is_active:
                logger.warning("Fiji viewer is already running.")
                return

            logger.info(
                f"🔬 FIJI VISUALIZER: Starting Fiji viewer server on port {port} (persistent={self.persistent})"
            )

            # ALL viewers (persistent and non-persistent) should be detached subprocess
            # so they don't block parent process exit. The difference is only whether
            # we terminate them during cleanup.
            logger.info(
                f"🔬 FIJI VISUALIZER: Creating {self.persistence_label} Fiji viewer (detached)"
            )
            self.process = self.launch_detached_viewer()

            # Only track non-persistent viewers in global variable for test cleanup
            if not self.persistent:
                with _global_fiji_lock:
                    _global_fiji_process = self.process

            # Wait for server to be ready before setting is_running flag
            # This ensures the viewer is actually ready to receive messages
            if async_mode:
                # For async mode, wait in background thread
                def wait_and_set_ready():
                    if self._wait_for_server_ready(timeout=10.0):
                        self.lifecycle_state.mark_owned_process()
                        logger.info(
                            f"🔬 FIJI VISUALIZER: Fiji viewer server ready (PID: {self.process_pid_label})"
                        )
                    else:
                        logger.error(
                            "🔬 FIJI VISUALIZER: Fiji viewer server failed to become ready"
                        )

                thread = threading.Thread(target=wait_and_set_ready, daemon=True)
                thread.start()
            else:
                # For sync mode, wait immediately
                if self._wait_for_server_ready(timeout=10.0):
                    self.lifecycle_state.mark_owned_process()
                    logger.info(
                        f"🔬 FIJI VISUALIZER: Fiji viewer server ready (PID: {self.process_pid_label})"
                    )
                else:
                    logger.error(
                        "🔬 FIJI VISUALIZER: Fiji viewer server failed to become ready"
                    )

    def _wait_for_server_ready(self, timeout: float = 10.0) -> bool:
        """Wait for Fiji server to be ready via ping/pong."""
        logger.info(
            f"🔬 FIJI VISUALIZER: Waiting for server on port {self.required_port} to be ready..."
        )
        ready = self.runtime_endpoint.wait_ready(
            timeout=timeout,
            require_ready=True,
        )
        if ready:
            logger.info(f"🔬 FIJI VISUALIZER: Server ready on port {self.required_port}")
            return True

        logger.warning(
            f"🔬 FIJI VISUALIZER: Timeout waiting for server on port {self.required_port}"
        )
        return False

    def stop_viewer(self) -> None:
        """Stop Fiji viewer server (only if not persistent)."""
        global _global_fiji_process

        with self._lock:
            if not self.persistent:
                logger.info("🔬 FIJI VISUALIZER: Stopping non-persistent Fiji viewer")

                if self.process:
                    killed = ViewerProcessHandle.from_process(self.process).terminate()
                    if killed:
                        logger.warning("🔬 FIJI VISUALIZER: Force killing Fiji viewer")

                # Clear global reference
                with _global_fiji_lock:
                    if _global_fiji_process == self.process:
                        _global_fiji_process = None

                self.lifecycle_state.mark_stopped()
            else:
                logger.info("🔬 FIJI VISUALIZER: Keeping persistent Fiji viewer alive")
                # DON'T mark stopped for persistent viewers.
                # The process is still alive and should be reusable

    def is_viewer_running(self) -> bool:
        """Check if Fiji viewer process is running."""
        return (
            self.is_running
            and self.process is not None
            and ViewerProcessHandle.from_process(self.process).is_alive()
        )

    def stop(self, timeout: float = 5.0):
        self.stop_viewer()
