"""
Shared ZMQ client manager for compile/run flows.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from zmqruntime.startup import EndpointStartupStatusCallback
    from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
    from openhcs.runtime.zmq_config import OpenHCSZMQConfig

ProgressCallback: TypeAlias = Callable[[dict], None]

logger = logging.getLogger(__name__)


class ZMQClientService:
    """Create/connect/disconnect a ZMQ execution client."""

    def __init__(
        self,
        config: "OpenHCSZMQConfig",
        *,
        status_callback: "EndpointStartupStatusCallback | None" = None,
    ):
        self.config = config
        self._status_callback = status_callback
        self.zmq_client = None
        self._generation = 0
        self._client_lock = threading.Lock()

    def set_config(self, config: "OpenHCSZMQConfig") -> None:
        if config == self.config:
            return
        self.disconnect_sync()
        self.config = config

    async def connect(
        self,
        progress_callback=None,
        persistent: bool | None = None,
        timeout: float | None = None,
    ):
        """Create a client and connect to the execution server."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._client_lock.acquire)
        try:
            return await self._connect_unlocked(
                progress_callback=progress_callback,
                persistent=persistent,
                timeout=timeout,
            )
        finally:
            self._client_lock.release()

    async def _connect_unlocked(
        self,
        *,
        progress_callback=None,
        persistent: bool | None = None,
        timeout: float | None = None,
    ):
        from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

        if self.zmq_client is not None and self.zmq_client.is_connected():
            existing_callback = self.zmq_client.progress_callback
            if existing_callback == progress_callback:
                return self.zmq_client
            await self._disconnect_unlocked()
        loop = asyncio.get_event_loop()
        self._generation += 1
        generation = self._generation
        client = ZMQExecutionClient(
            config=self.config,
            persistent=persistent,
            progress_callback=progress_callback,
            connection_status_callback=self._status_callback,
        )
        self.zmq_client = client
        connected = await loop.run_in_executor(
            None,
            lambda: client.connect(
                timeout=(
                    self.config.client_connect_timeout_seconds
                    if timeout is None
                    else timeout
                )
            ),
        )
        if not connected:
            if self.zmq_client is client:
                self.zmq_client = None
            raise RuntimeError("Failed to connect to ZMQ execution server")
        if self.zmq_client is not client or generation != self._generation:
            client.disconnect()
            raise RuntimeError("ZMQ client connection was superseded before use")
        logger.info("✅ Connected to ZMQ execution server")
        return client

    async def disconnect(self) -> None:
        """Disconnect the client (async-safe)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._client_lock.acquire)
        try:
            await self._disconnect_unlocked()
        finally:
            self._client_lock.release()

    async def _disconnect_unlocked(self) -> None:
        if self.zmq_client is None:
            return
        loop = asyncio.get_event_loop()
        client = self.zmq_client
        self.zmq_client = None
        self._generation += 1
        await loop.run_in_executor(None, client.disconnect)

    def disconnect_sync(self) -> None:
        """Disconnect the client (sync)."""
        if self.zmq_client is None:
            return
        try:
            self.zmq_client.disconnect()
        finally:
            self.zmq_client = None
            self._generation += 1


@dataclass(frozen=True, slots=True)
class ZMQExecutionClientBoundary:
    """Nominal UI workflow boundary for the shared ZMQ execution client."""

    client: ZMQClientService

    @property
    def port(self) -> int:
        return self.client.config.default_port

    @property
    def config(self) -> "OpenHCSZMQConfig":
        return self.client.config

    @property
    def current_client(self) -> "ZMQExecutionClient | None":
        return self.client.zmq_client

    def has_client(self) -> bool:
        return self.client.zmq_client is not None

    def require_client(self) -> "ZMQExecutionClient":
        if self.client.zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        return self.client.zmq_client

    async def connect(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        persistent: bool | None = None,
        timeout: float | None = None,
    ) -> "ZMQExecutionClient":
        return await self.client.connect(
            progress_callback=progress_callback,
            persistent=persistent,
            timeout=timeout,
        )

    async def disconnect(self) -> None:
        await self.client.disconnect()

    def disconnect_sync(self) -> None:
        self.client.disconnect_sync()
