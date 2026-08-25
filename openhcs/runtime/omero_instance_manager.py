"""Connect to OMERO or manage the packaged local deployment."""

import logging
import subprocess
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml
from polystore.omero_tables import (
    OMERO_TABLE_SERVICE,
    OMEROTableServiceUnavailableError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _OMEROConnectionSettings:
    """Typed projection of the canonical Compose connection declaration."""

    host: str
    port: int
    web_port: int
    user: str
    password: str

    @classmethod
    def from_compose(cls, compose_path: Path) -> "_OMEROConnectionSettings":
        """Load the connection extension from a Compose declaration."""

        document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        return cls(**document["x-openhcs-connection"])


class OMEROInstanceManager:
    """Own an OMERO connection and the local Compose lifecycle when required."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        web_port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        docker_compose_path: Path | None = None,
    ):
        """
        Initialize OMERO instance manager.

        Args:
            host: OMERO server hostname
            port: OMERO server port
            web_port: OMERO.web port
            user: OMERO username
            password: OMERO password
            docker_compose_path: Path to docker-compose.yml (auto-detected if None)
        """
        resolved_compose_path = docker_compose_path or self._find_docker_compose()
        if resolved_compose_path is None:
            raise FileNotFoundError("OpenHCS OMERO Compose declaration is unavailable")
        self.docker_compose_path = resolved_compose_path
        local_settings = _OMEROConnectionSettings.from_compose(self.docker_compose_path)
        self.host = local_settings.host if host is None else host
        self.port = local_settings.port if port is None else port
        self.web_port = local_settings.web_port if web_port is None else web_port
        self.user = local_settings.user if user is None else user
        self.password = local_settings.password if password is None else password
        self.conn = None
        self._started_by_us = False

    def _find_docker_compose(self) -> Path | None:
        """Resolve the compose declaration shipped by the OpenHCS package."""

        compose_resource = files("openhcs").joinpath("omero", "docker-compose.yml")
        return Path(str(compose_resource)) if compose_resource.is_file() else None

    def _docker_command(self) -> tuple[str, ...] | None:
        """Return the usable Docker command prefix for this process."""

        for command in (("docker",), ("sudo", "-n", "docker")):
            try:
                result = subprocess.run(
                    [*command, "info"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0:
                    return command
            except (OSError, subprocess.SubprocessError):
                continue
        return None

    def _wait_for_docker_command(
        self,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> tuple[str, ...] | None:
        """Wait boundedly for an operator-started Docker daemon."""

        deadline = time.monotonic() + timeout
        command = self._docker_command()
        if command is not None:
            return command

        logger.info(
            "Waiting up to %.0fs for the Docker daemon to become responsive...",
            timeout,
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(poll_interval, remaining))
            command = self._docker_command()
            if command is not None:
                logger.info("Docker daemon is responsive")
                return command

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
            web_url = f"http://{self.host}:{self.web_port}"
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
        Connect to OMERO, starting the packaged stack when Docker is available.

        Automatic startup sequence:
        1. Check if already connected
        2. Check if OMERO is running → connect if yes
        3. Wait boundedly for an operator-started Docker daemon
        4. Start the packaged OMERO Compose stack
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

        # Docker daemon lifecycle remains outside the OpenHCS domain boundary.
        docker_command = self._wait_for_docker_command()
        if docker_command is None:
            logger.error(
                "Docker is unavailable; start Docker before requesting the local "
                "OMERO deployment"
            )
            return False

        # Start OMERO via docker-compose
        if self._start_omero_docker(docker_command) and self._wait_for_omero_ready(
            timeout
        ):
            return self._connect_to_omero()

        logger.error("Failed to connect to or start OMERO")
        return False

    def _connect_to_omero(self) -> bool:
        """Establish connection to OMERO."""
        try:
            from omero.gateway import BlitzGateway

            connection = BlitzGateway(
                self.user, self.password, host=self.host, port=self.port
            )

            if not connection.connect():
                logger.error("Failed to connect to OMERO")
                return False
            self.conn = connection
            try:
                OMERO_TABLE_SERVICE.wait_until_available(connection)
            except OMEROTableServiceUnavailableError:
                logger.exception(
                    "Connected to OMERO, but its table service is unavailable"
                )
                self.close()
                return False
            logger.info(f"✓ Connected to OMERO at {self.host}:{self.port}")
            return True

        except Exception as e:  # noqa: BLE001 - OMERO and Ice expose separate roots.
            self.close()
            logger.error(f"Failed to connect to OMERO: {e}")
            return False

    def _start_omero_docker(self, docker_command: tuple[str, ...]) -> bool:
        """
        Start OMERO from the packaged Compose declaration.

        Returns:
            True if Docker Compose started successfully
        """
        if not self.docker_compose_path.exists():
            logger.warning(
                f"docker-compose.yml not found at {self.docker_compose_path}"
            )
            return False

        try:
            logger.info(
                "Starting OMERO from %s",
                self.docker_compose_path,
            )
            result = subprocess.run(
                [
                    *docker_command,
                    "compose",
                    "--file",
                    str(self.docker_compose_path),
                    "up",
                    "-d",
                ],
                stdout=None,  # Inherit stdout to show startup output in real-time
                stderr=None,  # Inherit stderr to show startup errors in real-time
                text=True,
                timeout=600,
                check=False,
            )

            if result.returncode == 0:
                logger.info("✓ docker compose up completed")
                self._started_by_us = True
                return True
            else:
                logger.error(
                    "docker compose up failed with exit code %s",
                    result.returncode,
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("docker compose up timed out")
            return False
        except FileNotFoundError:
            logger.error("Docker Compose is unavailable")
            return False
        except OSError as e:
            logger.error(f"Failed to start Docker Compose: {e}")
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
        """Stop services from the packaged Compose declaration."""
        if not self.docker_compose_path.exists():
            logger.warning("Cannot stop OMERO: docker-compose.yml not found")
            return

        docker_command = self._docker_command()
        if docker_command is None:
            logger.error("Docker daemon is unavailable to the current process")
            return

        try:
            logger.info("Stopping OMERO via Docker Compose...")

            result = subprocess.run(
                [
                    *docker_command,
                    "compose",
                    "--file",
                    str(self.docker_compose_path),
                    "down",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            if result.returncode == 0:
                logger.info("✓ OMERO stopped")
            else:
                logger.warning(f"Docker Compose down had issues: {result.stderr}")

        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Failed to stop OMERO through Docker Compose: {e}")

    def __enter__(self):
        """Context manager entry."""
        if not self.connect():
            raise ConnectionError("OMERO stack is unavailable")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup(stop_if_started=False)  # Don't stop OMERO on exit by default
