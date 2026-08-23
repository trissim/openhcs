"""
Shared ZMQ client manager for compile/run flows.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, TypeAlias

from zmqruntime.client import (
    EndpointClientSession,
    EndpointConnectionAttempt,
    EndpointConnectionPolicy,
)
from zmqruntime.messages import EndpointApplicationCompatibility
from zmqruntime.startup import EndpointStartupStatus, EndpointStartupStatusCallback
from zmqruntime.transport import TransportEndpoint

if TYPE_CHECKING:
    from openhcs.runtime.zmq_config import OpenHCSZMQConfig
    from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

ProgressCallback: TypeAlias = Callable[[dict], None]

logger = logging.getLogger(__name__)


class ZMQClientService:
    """Create/connect/disconnect a ZMQ execution client."""

    def __init__(
        self,
        config: "OpenHCSZMQConfig",
        *,
        status_callback: EndpointStartupStatusCallback | None = None,
        compatibility_callback: (
            Callable[[EndpointApplicationCompatibility], None] | None
        ) = None,
    ):
        self._config = config
        self._status_callback = status_callback
        self._compatibility_callback = compatibility_callback
        self._session: EndpointClientSession[ZMQExecutionClient] | None = None
        self._connection_attempt: EndpointConnectionAttempt | None = None
        self._client_lock = threading.Lock()

    @property
    def config(self) -> OpenHCSZMQConfig:
        return self._config

    def set_config(self, config: "OpenHCSZMQConfig") -> None:
        if config == self.config:
            return
        self._cancel_connection_attempt()
        with self._client_lock:
            if config == self.config:
                return
            self._disconnect_sync_unlocked()
            self._config = config

    @property
    def zmq_client(self) -> ZMQExecutionClient | None:
        """Return the client only when its application identity is admitted."""

        session = self._session
        return None if session is None else session.admitted_client

    async def connect(
        self,
        progress_callback=None,
        persistent: bool | None = None,
        timeout: float | None = None,
    ) -> "ZMQExecutionClient":
        """Create a client and connect to the execution server."""
        client = await self._connect(
            policy=EndpointConnectionPolicy.ATTACH_OR_START,
            progress_callback=progress_callback,
            persistent=persistent,
            timeout=timeout,
        )
        if client is None:
            raise RuntimeError("Failed to connect to ZMQ execution server")
        return client

    async def connect_existing(
        self,
        progress_callback=None,
        persistent: bool | None = None,
        timeout: float | None = None,
    ) -> "ZMQExecutionClient | None":
        """Attach to a ready execution server without starting one."""

        return await self._connect(
            policy=EndpointConnectionPolicy.ATTACH_EXISTING,
            progress_callback=progress_callback,
            persistent=persistent,
            timeout=timeout,
        )

    async def _connect(
        self,
        *,
        policy: EndpointConnectionPolicy,
        progress_callback=None,
        persistent: bool | None = None,
        timeout: float | None = None,
    ) -> "ZMQExecutionClient | None":
        async with self._client_ownership():
            return await self._connect_unlocked(
                policy=policy,
                progress_callback=progress_callback,
                persistent=persistent,
                timeout=timeout,
            )

    @asynccontextmanager
    async def _client_ownership(self) -> AsyncIterator[None]:
        """Own the service lock without leaking it when the waiter is cancelled."""

        loop = asyncio.get_running_loop()
        acquisition = loop.run_in_executor(None, self._client_lock.acquire)
        acquired = False
        try:
            await asyncio.shield(acquisition)
            acquired = True
            yield
        finally:
            if acquired:
                self._client_lock.release()
            else:
                acquisition.add_done_callback(
                    lambda _completed: self._client_lock.release()
                )

    async def _connect_unlocked(
        self,
        *,
        policy: EndpointConnectionPolicy,
        progress_callback=None,
        persistent: bool | None = None,
        timeout: float | None = None,
    ) -> "ZMQExecutionClient | None":
        from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

        config = self.config
        existing_session = self._session
        if existing_session is not None and existing_session.client.is_connected():
            existing_client = existing_session.client
            existing_callback = existing_client.progress_callback
            if existing_callback == progress_callback:
                return existing_session.require_admitted_client()
        if existing_session is not None:
            await self._disconnect_unlocked()
        loop = asyncio.get_event_loop()

        client: ZMQExecutionClient
        session: EndpointClientSession[ZMQExecutionClient]

        def publish_status(status: EndpointStartupStatus) -> None:
            if self._session is session and self._status_callback is not None:
                self._status_callback(status)

        client = ZMQExecutionClient(
            config=config,
            persistent=persistent,
            progress_callback=progress_callback,
            connection_status_callback=publish_status,
        )
        session = EndpointClientSession(client)
        self._session = session
        connection_attempt = client.new_connection_attempt()
        self._connection_attempt = connection_attempt

        try:
            connected = await loop.run_in_executor(
                None,
                connection_attempt.connect,
                policy,
                (
                    config.client_connect_timeout_seconds
                    if timeout is None
                    else timeout
                ),
            )
        except Exception as connection_error:
            self._session = None
            try:
                client.disconnect()
            except Exception as cleanup_error:
                raise ExceptionGroup(
                    "ZMQ connection and cleanup both failed",
                    (connection_error, cleanup_error),
                ) from connection_error
            raise
        finally:
            self._connection_attempt = None
        if not connected:
            self._session = None
            client.disconnect()
            return None
        logger.info("✅ Connected to ZMQ execution server")
        compatibility = session.observe_compatibility()
        if self._compatibility_callback is not None:
            self._compatibility_callback(compatibility)
        return session.require_admitted_client()

    async def restart_endpoint(
        self,
        *,
        expected_compatibility: EndpointApplicationCompatibility | None = None,
        progress_callback=None,
        persistent: bool | None = None,
        timeout: float | None = None,
    ) -> "ZMQExecutionClient":
        """Replace the configured endpoint and establish a fresh connection."""

        from zmqruntime import EndpointShutdownMode

        from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

        loop = asyncio.get_running_loop()
        async with self._client_ownership():
            session = self._session
            current = None if session is None else session.client
            if expected_compatibility is not None and (
                current is None
                or current.endpoint_compatibility() != expected_compatibility
            ):
                raise RuntimeError("ZMQ endpoint compatibility changed before restart")
            config = self.config if current is None else current.config
            endpoint = config.client_endpoint() if current is None else current.endpoint
            if current is not None:
                if progress_callback is None:
                    progress_callback = current.progress_callback
                if persistent is None:
                    persistent = current.persistent
            await self._disconnect_unlocked()
            shutdown = await loop.run_in_executor(
                None,
                lambda: ZMQExecutionClient.shutdown_endpoint_on_port(
                    port=endpoint.port,
                    mode=EndpointShutdownMode.FORCE,
                    timeout=config.client_connect_timeout_seconds,
                    transport_mode=endpoint.transport_mode,
                    host=endpoint.host,
                    config=config,
                ),
            )
            if not shutdown.succeeded or not shutdown.endpoint_terminated:
                raise RuntimeError("The existing ZMQ execution server did not stop")
            client = await self._connect_unlocked(
                policy=EndpointConnectionPolicy.ATTACH_OR_START,
                progress_callback=progress_callback,
                persistent=persistent,
                timeout=timeout,
            )
            if client is None:
                raise RuntimeError("The replacement ZMQ execution server did not start")
            return client

    async def disconnect(self) -> None:
        """Disconnect the client (async-safe)."""
        self._cancel_connection_attempt()
        async with self._client_ownership():
            await self._disconnect_unlocked()

    async def _disconnect_unlocked(self) -> None:
        session = self._session
        self._session = None
        if session is None:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, session.client.disconnect)

    def disconnect_sync(self) -> None:
        """Disconnect the client (sync)."""

        self._cancel_connection_attempt()
        with self._client_lock:
            self._disconnect_sync_unlocked()

    def endpoint_terminated(
        self,
        endpoint: TransportEndpoint,
    ) -> bool:
        """Invalidate this service's client when its exact endpoint terminates."""

        session = self._session
        if session is None or session.client.endpoint != endpoint:
            return False
        self._cancel_connection_attempt()
        with self._client_lock:
            session = self._session
            if session is None or session.client.endpoint != endpoint:
                return False
            self._disconnect_sync_unlocked()
        return True

    def _disconnect_sync_unlocked(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            session.client.disconnect()

    def _cancel_connection_attempt(self) -> None:
        connection_attempt = self._connection_attempt
        if connection_attempt is not None:
            connection_attempt.cancel()

    def has_client(self) -> bool:
        return self.zmq_client is not None

    def require_client(self) -> "ZMQExecutionClient":
        session = self._session
        if session is None:
            raise RuntimeError("ZMQ client is not connected")
        return session.require_admitted_client()
