"""
OMERO Instance Manager - Manages OMERO server instances for testing.

Follows the same pattern as Napari instance management:
1. Check if Docker daemon is running (auto-start if needed)
2. Check if OMERO is running
3. Connect to existing instance if available
4. Start new instance if needed (via docker-compose)
5. Provide cleanup utilities
"""

import logging
import platform
import subprocess
import time
from importlib.resources import files
from pathlib import Path

from polystore.omero_tables import (
    OMERO_TABLE_SERVICE,
    OMEROTableServiceUnavailableError,
)

logger = logging.getLogger(__name__)

# Default OMERO connection parameters
DEFAULT_OMERO_HOST = "localhost"
DEFAULT_OMERO_PORT = 4064
DEFAULT_OMERO_WEB_PORT = 4080
DEFAULT_OMERO_USER = "root"
DEFAULT_OMERO_PASSWORD = "openhcs"


class OMEROInstanceManager:
    """
    Manages OMERO server instances for testing.

    Follows the same pattern as NapariStreamVisualizer:
    - Check if OMERO is running
    - Connect to existing instance if available
    - Start new instance if needed
    """

    def __init__(
        self,
        host: str = DEFAULT_OMERO_HOST,
        port: int = DEFAULT_OMERO_PORT,
        user: str = DEFAULT_OMERO_USER,
        password: str = DEFAULT_OMERO_PASSWORD,
        docker_compose_path: Path | None = None,
    ):
        """
        Initialize OMERO instance manager.

        Args:
            host: OMERO server hostname
            port: OMERO server port
            user: OMERO username
            password: OMERO password
            docker_compose_path: Path to docker-compose.yml (auto-detected if None)
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.docker_compose_path = docker_compose_path or self._find_docker_compose()
        self.conn = None
        self._started_by_us = False

    def _find_docker_compose(self) -> Path | None:
        """Resolve the compose declaration shipped by the OpenHCS package."""

        compose_resource = files("openhcs").joinpath("omero", "docker-compose.yml")
        return Path(str(compose_resource)) if compose_resource.is_file() else None

    def _is_docker_running(self) -> bool:
        """
        Check if Docker daemon is running.

        Returns:
            True if Docker daemon is responsive
        """
        try:
            # Try without sudo first
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return True

            # If failed, try with sudo (Linux may require it)
            result = subprocess.run(
                ["sudo", "docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _start_docker_daemon(self) -> bool:
        """
        Start Docker daemon if not running.

        Platform-specific implementation:
        - Linux: Uses systemctl to start docker service
        - macOS: Starts Docker Desktop application
        - Windows: Starts Docker Desktop application

        Returns:
            True if Docker daemon was started successfully
        """
        system = platform.system()

        try:
            if system == "Linux":
                logger.info("Starting Docker daemon via systemctl...")
                result = subprocess.run(
                    ["sudo", "systemctl", "start", "docker"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )

                if result.returncode == 0:
                    logger.info("✓ Docker daemon started")
                    # Wait a moment for daemon to be ready
                    time.sleep(2)
                    return self._is_docker_running()
                else:
                    logger.warning(f"Failed to start Docker daemon: {result.stderr}")
                    return False

            elif system == "Darwin":  # macOS
                logger.info("Starting Docker Desktop...")
                subprocess.Popen(
                    ["open", "-a", "Docker"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("Waiting for Docker Desktop to start...")
                # Docker Desktop takes longer to start
                return self._wait_for_docker_ready(timeout=60)

            elif system == "Windows":
                logger.info("Starting Docker Desktop...")
                subprocess.Popen(
                    ["start", "Docker Desktop"],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("Waiting for Docker Desktop to start...")
                return self._wait_for_docker_ready(timeout=60)

            else:
                logger.warning(
                    f"Unsupported platform for auto-starting Docker: {system}"
                )
                return False

        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Failed to start Docker daemon: {e}")
            return False

    def _wait_for_docker_ready(self, timeout: int = 60) -> bool:
        """
        Wait for Docker daemon to be ready.

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            True if Docker is ready
        """
        logger.info(f"Waiting for Docker daemon to be ready (timeout: {timeout}s)...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_docker_running():
                elapsed = time.time() - start_time
                logger.info(f"✓ Docker daemon is ready (took {elapsed:.1f}s)")
                return True

            time.sleep(2)  # Check every 2 seconds

        logger.error(f"Timeout waiting for Docker daemon to be ready ({timeout}s)")
        return False

    def is_omero_running(self) -> bool:
        """
        Check if OMERO is running and responsive.

        Returns:
            True if OMERO server is running and we can connect
        """
        try:
            from omero.gateway import BlitzGateway

            # Try to connect with short timeout
            conn = BlitzGateway(
                self.user, self.password, host=self.host, port=self.port
            )

            if conn.connect():
                conn.close()
                return True
            return False

        except Exception as e:
            logger.warning(f"OMERO server not running or cannot connect: {e}")
            logger.debug("Full OMERO connection traceback", exc_info=True)
            return False

    def is_omero_web_running(self) -> bool:
        """
        Check if OMERO.web is running and responsive.

        Returns:
            True if OMERO.web is accessible
        """
        try:
            import urllib.error
            import urllib.request

            # Try to connect to OMERO.web with short timeout
            web_url = f"http://{self.host}:{DEFAULT_OMERO_WEB_PORT}"
            request = urllib.request.Request(web_url)

            with urllib.request.urlopen(request, timeout=5):
                # Any HTTP response (including redirects) means web is running
                return True

        except (OSError, TimeoutError) as e:
            logger.warning(f"OMERO.web not running or cannot connect: {e}")
            return False

    def is_omero_stack_running(self) -> bool:
        """
        Check if both OMERO server and OMERO.web are running.

        Returns:
            True if both OMERO server and OMERO.web are accessible
        """
        server_running = self.is_omero_running()
        web_running = self.is_omero_web_running()

        if server_running and web_running:
            logger.info("✓ Both OMERO server and OMERO.web are running")
            return True
        elif server_running:
            logger.warning("OMERO server is running but OMERO.web is not accessible")
            return False
        elif web_running:
            logger.warning("OMERO.web is accessible but OMERO server is not running")
            return False
        else:
            logger.info("Neither OMERO server nor OMERO.web are running")
            return False

    def connect(self, timeout: int = 30) -> bool:
        """
        Connect to OMERO, starting Docker and OMERO if necessary.

        Automatic startup sequence:
        1. Check if already connected
        2. Check if OMERO is running → connect if yes
        3. Check if Docker is running → start if no
        4. Start OMERO via docker-compose
        5. Wait for OMERO to be ready
        6. Connect

        Args:
            timeout: Maximum time to wait for OMERO to be ready (seconds)

        Returns:
            True if connected successfully
        """
        # Check if already connected
        if self.conn is not None:
            try:
                # Verify connection is still alive
                self.conn.getEventContext()
                OMERO_TABLE_SERVICE.wait_until_available(self.conn)
                logger.info("✓ Already connected to OMERO")
                return True
            except Exception:  # noqa: BLE001 - OMERO and Ice expose separate roots.
                # Connection is dead, close it
                try:
                    self.conn.close()
                except Exception:
                    logger.debug(
                        "Failed to close stale OMERO connection", exc_info=True
                    )
                self.conn = None

        # Try to connect to existing instance - check both server and web
        if self.is_omero_stack_running():
            logger.info(f"✓ Found existing OMERO stack at {self.host}:{self.port}")
            return self._connect_to_omero()

        # No existing instance or incomplete stack - check Docker and start OMERO
        logger.info(
            "OMERO stack not fully running, attempting to start all services..."
        )

        # Ensure Docker daemon is running
        if not self._is_docker_running():
            logger.info("Docker daemon not running, attempting to start...")
            if not self._start_docker_daemon():
                logger.error("Failed to start Docker daemon")
                return False

        # Start OMERO via docker-compose
        if self._start_omero_docker() and self._wait_for_omero_ready(timeout):
            return self._connect_to_omero()

        logger.error("Failed to connect to or start OMERO")
        return False

    def _connect_to_omero(self) -> bool:
        """Establish connection to OMERO."""
        try:
            from omero.gateway import BlitzGateway

            self.conn = BlitzGateway(
                self.user, self.password, host=self.host, port=self.port
            )

            if self.conn.connect():
                try:
                    OMERO_TABLE_SERVICE.wait_until_available(self.conn)
                except OMEROTableServiceUnavailableError:
                    logger.exception(
                        "Connected to OMERO, but its table service is unavailable"
                    )
                    self.close()
                    return False
                logger.info(f"✓ Connected to OMERO at {self.host}:{self.port}")
                return True
            else:
                logger.error("Failed to connect to OMERO")
                return False

        except Exception as e:  # noqa: BLE001 - OMERO and Ice expose separate roots.
            logger.error(f"Failed to connect to OMERO: {e}")
            return False

    def _start_omero_docker(self) -> bool:
        """
        Start OMERO using docker-compose.

        Automatically builds the OpenHCS-enabled OMERO.web image if needed.

        Returns:
            True if docker-compose started successfully
        """
        if self.docker_compose_path is None:
            logger.warning("No docker-compose.yml found, cannot start OMERO")
            return False

        if not self.docker_compose_path.exists():
            logger.warning(
                f"docker-compose.yml not found at {self.docker_compose_path}"
            )
            return False

        try:
            logger.info(
                f"Starting OMERO via docker compose at {self.docker_compose_path.parent}"
            )

            # Build images first (this will build the OpenHCS-enabled OMERO.web image)
            # Only builds if Dockerfile has changed or image doesn't exist
            logger.info("Building OMERO.web with OpenHCS plugin (if needed)...")
            build_result = subprocess.run(
                ["sudo", "docker", "compose", "build"],
                cwd=self.docker_compose_path.parent,
                stdout=None,  # Inherit stdout to show build output in real-time
                stderr=None,  # Inherit stderr to show build errors in real-time
                text=True,
                timeout=300,  # 5 minute timeout for build
                check=False,
            )

            if build_result.returncode != 0:
                logger.error(
                    "docker compose build failed with exit code %s",
                    build_result.returncode,
                )
                return False

            # Run docker compose up -d
            result = subprocess.run(
                ["sudo", "docker", "compose", "up", "-d"],
                cwd=self.docker_compose_path.parent,
                stdout=None,  # Inherit stdout to show startup output in real-time
                stderr=None,  # Inherit stderr to show startup errors in real-time
                text=True,
                timeout=600,  # 10 minute timeout for docker compose (OMERO server takes time to start)
                check=False,
            )

            if result.returncode == 0:
                logger.info("✓ docker compose up completed")
                self._started_by_us = True
                return True
            else:
                logger.error(f"docker compose failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("docker compose up timed out")
            return False
        except FileNotFoundError:
            logger.error("docker-compose command not found. Install Docker Compose.")
            return False
        except OSError as e:
            logger.error(f"Failed to start docker-compose: {e}")
            return False

    def _wait_for_omero_ready(self, timeout: int = 180) -> bool:
        """
        Wait for OMERO to be ready to accept connections.

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            True if OMERO is ready
        """
        logger.info(f"Waiting for OMERO to be ready (timeout: {timeout}s)...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_omero_running():
                elapsed = time.time() - start_time
                logger.info(f"✓ OMERO is ready (took {elapsed:.1f}s)")
                return True

            time.sleep(2)  # Check every 2 seconds

        logger.error(f"Timeout waiting for OMERO to be ready ({timeout}s)")
        return False

    def close(self):
        """Close OMERO connection."""
        if self.conn is not None:
            try:
                self.conn.close()
                logger.info("✓ Closed OMERO connection")
            except Exception:
                logger.debug("Failed to close OMERO connection", exc_info=True)
            self.conn = None

    def cleanup(self, stop_if_started: bool = True):
        """
        Cleanup OMERO resources.

        Args:
            stop_if_started: If True, stop OMERO if we started it
        """
        self.close()

        if stop_if_started and self._started_by_us:
            self.stop_omero_docker()

    def stop_omero_docker(self):
        """Stop OMERO docker-compose services."""
        if self.docker_compose_path is None or not self.docker_compose_path.exists():
            logger.warning("Cannot stop OMERO: docker-compose.yml not found")
            return

        try:
            logger.info("Stopping OMERO via docker-compose...")

            result = subprocess.run(
                ["docker-compose", "down"],
                cwd=self.docker_compose_path.parent,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            if result.returncode == 0:
                logger.info("✓ OMERO stopped")
            else:
                logger.warning(f"docker-compose down had issues: {result.stderr}")

        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Failed to stop OMERO: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup(stop_if_started=False)  # Don't stop OMERO on exit by default
