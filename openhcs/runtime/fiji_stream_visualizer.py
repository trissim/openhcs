"""
Fiji stream visualizer for OpenHCS.

Manages Fiji viewer instances for real-time visualization via ZMQ.
Uses FijiViewerServer (inherits from ZMQServer) for PyImageJ-based display.
Follows same architecture as NapariStreamVisualizer.
"""

import logging
import multiprocessing
import pickle
import subprocess
import threading
import time
from typing import Optional
from pathlib import Path

import zmq

from polystore.filemanager import FileManager
from polystore.backend_registry import register_cleanup_callback
from openhcs.core.config import (
    TransportMode as OpenHCSTransportMode,
    FijiStreamingConfig,
)
from openhcs.runtime.viewer_protocol import DetachedViewerProcessRequest
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from zmqruntime.config import TransportMode as ZMQTransportMode
from zmqruntime.streaming import VisualizerProcessManager
from zmqruntime.transport import (
    coerce_transport_mode,
    get_control_url,
    is_port_in_use,
    ping_control_port,
    wait_for_server_ready,
)

logger = logging.getLogger(__name__)

# Global process management for Fiji viewer
_global_fiji_process: Optional[multiprocessing.Process] = None
_global_fiji_lock = threading.Lock()


def _cleanup_global_fiji_viewer() -> None:
    """Clean up global Fiji viewer process for test mode."""
    global _global_fiji_process

    with _global_fiji_lock:
        if _global_fiji_process and _global_fiji_process.is_alive():
            logger.info("🔬 FIJI VISUALIZER: Terminating Fiji viewer for test cleanup")
            _global_fiji_process.terminate()
            _global_fiji_process.join(timeout=3)

            if _global_fiji_process.is_alive():
                logger.warning("🔬 FIJI VISUALIZER: Force killing Fiji viewer process")
                _global_fiji_process.kill()
                _global_fiji_process.join(timeout=1)

            _global_fiji_process = None


register_cleanup_callback(_cleanup_global_fiji_viewer)


def _spawn_detached_fiji_process(
    port: int,
    viewer_title: str,
    display_config,
    transport_mode: OpenHCSTransportMode = OpenHCSTransportMode.IPC,
) -> subprocess.Popen:
    """
    Spawn a completely detached Fiji viewer process that survives parent termination.

    This creates a subprocess that runs independently and won't be terminated when
    the parent process exits, enabling true persistence across pipeline runs.

    Args:
        port: ZMQ port to listen on
        viewer_title: Title for the Fiji viewer window
        display_config: Display configuration
        transport_mode: ZMQ transport mode (IPC or TCP)
    """
    import os

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
    from openhcs.runtime.fiji_viewer_server import _fiji_viewer_server_process
    from openhcs.core.config import TransportMode
    from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
    transport_mode = TransportMode.{transport_mode.name}
    _fiji_viewer_server_process({port}, {repr(viewer_title)}, None, {repr(current_dir + "/.fiji_log_path_placeholder")}, transport_mode, OPENHCS_ZMQ_CONFIG)
except Exception as e:
    import logging
    logger = logging.getLogger("openhcs.runtime.fiji_detached")
    logger.error(f"Detached Fiji error: {{e}}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
"""

    try:
        # Create log file for detached process
        log_dir = os.path.expanduser("~/.local/share/openhcs/logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"fiji_detached_port_{port}.log")

        # Replace placeholder with actual log file path
        python_code = python_code.replace(
            repr(current_dir + "/.fiji_log_path_placeholder"), repr(log_file)
        )
        # Remove incidental indentation and leading/trailing whitespace from the
        # embedded snippet so it runs with the expected top-level indentation when
        # passed to `python -c`.
        import textwrap

        python_code = textwrap.dedent(python_code).strip()

        process = DetachedViewerProcessRequest(
            python_code=python_code,
            log_file=Path(log_file),
            cwd=Path.cwd(),
        ).launch()

        logger.info(
            f"🔬 FIJI VISUALIZER: Detached Fiji process started (PID: {process.pid}), logging to {log_file}"
        )
        return process

    except Exception as e:
        logger.error(f"🔬 FIJI VISUALIZER: Failed to spawn detached Fiji process: {e}")
        raise


class FijiStreamVisualizer(VisualizerProcessManager):
    """
    Manages Fiji viewer instance for real-time visualization via ZMQ.

    Follows same architecture as NapariStreamVisualizer.
    """

    def __init__(
        self,
        filemanager: FileManager,
        visualizer_config,
        viewer_title: str = "OpenHCS Fiji Visualization",
        persistent: bool = True,
        port: int = None,
        display_config=None,
        transport_mode: OpenHCSTransportMode = OpenHCSTransportMode.IPC,
    ):
        self.filemanager = filemanager
        self.viewer_title = viewer_title
        self.persistent = persistent
        self.visualizer_config = visualizer_config
        # Use config class default if not specified
        self.port = (
            port
            if port is not None
            else FijiStreamingConfig.__dataclass_fields__["port"].default
        )
        super().__init__(port=self.port)
        self.display_config = display_config
        self.transport_mode = (
            coerce_transport_mode(transport_mode) or ZMQTransportMode.IPC
        )  # ZMQ transport mode (IPC or TCP)
        self.process: Optional[multiprocessing.Process] = None
        self._is_running = False
        self._connected_to_existing = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """
        Check if the Fiji viewer is actually running.

        This property checks the actual process state, not just a cached flag.
        Returns True only if the process exists and is alive.
        """
        if not self._is_running:
            return False

        # If we connected to an existing viewer, verify it's still responsive
        if self._connected_to_existing:
            # Quick ping check to verify viewer is still alive
            if not self._quick_ping_check():
                logger.debug(
                    f"🔬 FIJI VISUALIZER: Connected viewer on port {self.port} is no longer responsive"
                )
                self._is_running = False
                self._connected_to_existing = False
                return False
            return True

        if self.process is None:
            self._is_running = False
            return False

        # Check if process is actually alive
        try:
            if hasattr(self.process, "is_alive"):
                # multiprocessing.Process
                alive = self.process.is_alive()
            else:
                # subprocess.Popen
                alive = self.process.poll() is None

            if not alive:
                logger.debug(
                    f"🔬 FIJI VISUALIZER: Fiji process on port {self.port} is no longer alive"
                )
                self._is_running = False

            return alive
        except Exception as e:
            logger.warning(f"🔬 FIJI VISUALIZER: Error checking process status: {e}")
            self._is_running = False
            return False

    def _quick_ping_check(self) -> bool:
        """Quick ping check to verify viewer is responsive (for connected viewers)."""
        return ping_control_port(
            self.port,
            self.transport_mode,
            host="localhost",
            config=OPENHCS_ZMQ_CONFIG,
            timeout_ms=200,
            require_ready=False,
        )

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
        return self._wait_for_server_ready(timeout=timeout)

    def start_viewer(self, async_mode: bool = False) -> None:
        """Start Fiji viewer server process."""
        global _global_fiji_process

        with self._lock:
            # Check if there's already a viewer running on the configured port
            if is_port_in_use(
                self.port, self.transport_mode, config=OPENHCS_ZMQ_CONFIG
            ):
                # Try to connect to existing viewer first
                logger.info(
                    f"🔬 FIJI VISUALIZER: Port {self.port} is in use, attempting to connect to existing viewer..."
                )
                if self._try_connect_to_existing_viewer():
                    logger.info(
                        f"🔬 FIJI VISUALIZER: Successfully connected to existing viewer on port {self.port}"
                    )
                    self._is_running = True
                    self._connected_to_existing = True
                    return
                else:
                    # Existing viewer is unresponsive - kill it and start fresh
                    logger.info(
                        f"🔬 FIJI VISUALIZER: Existing viewer on port {self.port} is unresponsive, killing and restarting..."
                    )
                    from zmqruntime.server import ZMQServer

                    ZMQServer.kill_processes_on_port(self.port)
                    ZMQServer.kill_processes_on_port(self.port + 1000)
                    time.sleep(0.5)

            if self._is_running:
                logger.warning("Fiji viewer is already running.")
                return

            logger.info(
                f"🔬 FIJI VISUALIZER: Starting Fiji viewer server on port {self.port} (persistent={self.persistent})"
            )

            # ALL viewers (persistent and non-persistent) should be detached subprocess
            # so they don't block parent process exit. The difference is only whether
            # we terminate them during cleanup.
            logger.info(
                f"🔬 FIJI VISUALIZER: Creating {'persistent' if self.persistent else 'non-persistent'} Fiji viewer (detached)"
            )
            self.process = _spawn_detached_fiji_process(
                self.port, self.viewer_title, self.display_config, self.transport_mode
            )

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
                        self._is_running = True
                        logger.info(
                            f"🔬 FIJI VISUALIZER: Fiji viewer server ready (PID: {self.process.pid if hasattr(self.process, 'pid') else 'unknown'})"
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
                    self._is_running = True
                    logger.info(
                        f"🔬 FIJI VISUALIZER: Fiji viewer server ready (PID: {self.process.pid if hasattr(self.process, 'pid') else 'unknown'})"
                    )
                else:
                    logger.error(
                        "🔬 FIJI VISUALIZER: Fiji viewer server failed to become ready"
                    )

    def _try_connect_to_existing_viewer(self) -> bool:
        """Try to connect to an existing Fiji viewer and verify it's responsive."""
        return ping_control_port(
            self.port,
            self.transport_mode,
            host="localhost",
            config=OPENHCS_ZMQ_CONFIG,
            timeout_ms=500,
            require_ready=True,
        )

    def _wait_for_server_ready(self, timeout: float = 10.0) -> bool:
        """Wait for Fiji server to be ready via ping/pong."""
        logger.info(
            f"🔬 FIJI VISUALIZER: Waiting for server on port {self.port} to be ready..."
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
            logger.info(f"🔬 FIJI VISUALIZER: Server ready on port {self.port}")
            return True

        logger.warning(
            f"🔬 FIJI VISUALIZER: Timeout waiting for server on port {self.port}"
        )
        return False

    def send_control_message(self, message_type: str, timeout: float = 2.0) -> bool:
        """
        Send a control message to the Fiji viewer.

        Args:
            message_type: Type of control message ('clear_state', 'shutdown', etc.)
            timeout: Timeout in seconds for waiting for response

        Returns:
            True if message was sent and acknowledged, False otherwise
        """
        if not self.is_running or self.port is None:
            logger.warning(
                f"🔬 FIJI VISUALIZER: Cannot send {message_type} - viewer not running"
            )
            return False

        control_context = None
        control_socket = None

        try:
            control_context = zmq.Context()
            control_socket = control_context.socket(zmq.REQ)
            control_socket.setsockopt(zmq.LINGER, 0)
            control_socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            control_url = get_control_url(
                self.port,
                self.transport_mode,
                host="localhost",
                config=OPENHCS_ZMQ_CONFIG,
            )
            control_socket.connect(control_url)

            # Send control message
            message = {"type": message_type}
            control_socket.send(pickle.dumps(message))

            # Wait for acknowledgment
            response = control_socket.recv()
            response_data = pickle.loads(response)

            if response_data.get("status") == "success":
                logger.info(
                    f"🔬 FIJI VISUALIZER: {message_type} acknowledged by viewer"
                )
                return True
            else:
                logger.warning(
                    f"🔬 FIJI VISUALIZER: {message_type} failed: {response_data}"
                )
                return False

        except zmq.Again:
            logger.warning(
                f"🔬 FIJI VISUALIZER: Timeout waiting for {message_type} acknowledgment"
            )
            return False
        except Exception as e:
            logger.warning(f"🔬 FIJI VISUALIZER: Failed to send {message_type}: {e}")
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
        Clear accumulated viewer state (dimension values, hyperstack metadata) for a new pipeline run.

        Returns:
            True if state was cleared successfully, False otherwise
        """
        return self.send_control_message("clear_state")

    def stop_viewer(self) -> None:
        """Stop Fiji viewer server (only if not persistent)."""
        global _global_fiji_process

        with self._lock:
            if not self.persistent:
                logger.info("🔬 FIJI VISUALIZER: Stopping non-persistent Fiji viewer")

                if self.process:
                    # Handle both subprocess and multiprocessing process types
                    if hasattr(self.process, "is_alive"):
                        # multiprocessing.Process
                        if self.process.is_alive():
                            self.process.terminate()
                            self.process.join(timeout=5)
                            if self.process.is_alive():
                                logger.warning(
                                    "🔬 FIJI VISUALIZER: Force killing Fiji viewer"
                                )
                                self.process.kill()
                                self.process.join(timeout=2)
                    else:
                        # subprocess.Popen
                        if self.process.poll() is None:
                            self.process.terminate()
                            try:
                                self.process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                logger.warning(
                                    "🔬 FIJI VISUALIZER: Force killing Fiji viewer"
                                )
                                self.process.kill()

                # Clear global reference
                with _global_fiji_lock:
                    if _global_fiji_process == self.process:
                        _global_fiji_process = None

                self._is_running = False
            else:
                logger.info("🔬 FIJI VISUALIZER: Keeping persistent Fiji viewer alive")
                # DON'T set _is_running = False for persistent viewers!
                # The process is still alive and should be reusable

    def _cleanup_zmq(self) -> None:
        """Clean up ZMQ client connection (for persistent viewers)."""
        if self._zmq_client:
            try:
                self._zmq_client.close()
            except Exception as e:
                logger.warning(f"🔬 FIJI VISUALIZER: Failed to cleanup ZMQ client: {e}")
            self._zmq_client = None

    def is_viewer_running(self) -> bool:
        """Check if Fiji viewer process is running."""
        return self.is_running and self.process is not None and self.process.is_alive()

    def get_launch_command(self) -> list[str]:
        import os
        import sys

        current_dir = os.getcwd()
        log_dir = os.path.expanduser("~/.local/share/openhcs/logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"fiji_detached_port_{self.port}.log")

        python_code = f"""
import sys
import os
sys.path.insert(0, {repr(current_dir)})
from openhcs.runtime.fiji_viewer_server import _fiji_viewer_server_process
from openhcs.core.config import TransportMode
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
transport_mode = TransportMode.{self.transport_mode.name}
_fiji_viewer_server_process({self.port}, {repr(self.viewer_title)}, None, {repr(log_file)}, transport_mode, OPENHCS_ZMQ_CONFIG)
"""

        # Ensure snippet has no incidental indentation
        import textwrap

        python_code = textwrap.dedent(python_code).strip()

        return [sys.executable, "-c", python_code]

    def get_launch_env(self) -> dict:
        import os

        env = os.environ.copy()
        if "QT_QPA_PLATFORM" not in env:
            env["QT_QPA_PLATFORM"] = "xcb"
        env["QT_X11_NO_MITSHM"] = "1"
        return env

    def start(self, detached: bool = True):
        self.start_viewer(async_mode=False)
        return self.process

    def stop(self, timeout: float = 5.0):
        self.stop_viewer()
