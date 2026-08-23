"""
Fiji stream visualizer for OpenHCS.

Manages Fiji viewer instances for real-time visualization via ZMQ.
Uses FijiViewerServer (inherits from ZMQServer) for PyImageJ-based display.
Follows same architecture as NapariStreamVisualizer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from polystore.filemanager import FileManager

from openhcs.core.streaming_config_declarations import ViewerType
from openhcs.core.streaming_config_factory import StreamingViewerRuntimeConfig
from openhcs.runtime.viewer_protocol import (
    DetachedViewerPythonArguments,
    DetachedViewerPythonExpression,
    DetachedViewerServerEntrypointSpec,
    ManagedViewerLifecycleMixin,
)

logger = logging.getLogger(__name__)

FIJI_VIEWER_ENTRYPOINT = DetachedViewerServerEntrypointSpec(
    viewer_type=ViewerType.FIJI,
    module_name="openhcs.runtime.fiji_viewer_server",
    function_name="fiji_viewer_server_process",
    extra_imports=("from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG",),
)


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
        super().__init__(runtime_config=runtime_config)
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
            self.display_enabled,
        ).append(
            DetachedViewerPythonExpression.symbol("transport_mode"),
            DetachedViewerPythonExpression.symbol("OPENHCS_ZMQ_CONFIG"),
        )

    def start_viewer(self, async_mode: bool = False) -> None:
        """Start Fiji viewer server process."""
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

            if self.owned_viewer_process_is_alive():
                self.lifecycle_state.mark_owned_process()
                logger.info(
                    "🔬 FIJI VISUALIZER: Fiji viewer process started "
                    f"(PID: {self.process_pid_label})"
                )
            else:
                logger.error("🔬 FIJI VISUALIZER: Failed to start Fiji viewer process")

    def stop_viewer(self) -> None:
        """Stop Fiji viewer server (only if not persistent)."""
        with self._lock:
            if not self.persistent:
                logger.info("🔬 FIJI VISUALIZER: Stopping non-persistent Fiji viewer")

                if self.process:
                    killed = self.terminate_owned_viewer_process()
                    if killed:
                        logger.warning("🔬 FIJI VISUALIZER: Force killing Fiji viewer")

                self.lifecycle_state.mark_stopped()
            else:
                logger.info("🔬 FIJI VISUALIZER: Keeping persistent Fiji viewer alive")
                # DON'T mark stopped for persistent viewers.
                # The process is still alive and should be reusable

    def is_viewer_running(self) -> bool:
        """Check if Fiji viewer process is running."""
        return self.is_running

    def stop(self, timeout: float = 5.0):
        self.stop_viewer()
