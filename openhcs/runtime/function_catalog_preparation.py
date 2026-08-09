"""Asynchronous preparation lifecycle for an execution endpoint's catalog."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from collections.abc import Callable
from typing import TYPE_CHECKING

from zmqruntime.startup import EndpointStartupPhase, EndpointStartupStatus

if TYPE_CHECKING:
    from openhcs.agent.services.function_catalog_service import FunctionCatalogService


class FunctionCatalogPreparation:
    """Own one lazily started endpoint catalog preparation operation."""

    def __init__(
        self,
        function_catalog: "FunctionCatalogService",
    ) -> None:
        self._lock = threading.RLock()
        self._future: Future[None] | None = None
        self._function_catalog = function_catalog
        self._snapshot = EndpointStartupStatus(
            sequence=0,
            phase=EndpointStartupPhase.PREPARING_CAPABILITIES,
            message="Function catalog has not been requested",
            timestamp=0.0,
        )

    def ensure_started(self) -> Future[None]:
        """Return the one preparation future, starting it when first requested."""

        with self._lock:
            if self._future is not None:
                return self._future
            future: Future[None] = Future()
            self._future = future
            self._set_message("Starting function catalog preparation")
            threading.Thread(
                target=self._prepare,
                args=(future,),
                name="openhcs-function-catalog-preparation",
                daemon=True,
            ).start()
            return future

    def wait_until_ready(
        self,
        status_callback: Callable[[EndpointStartupStatus], None] | None = None,
        *,
        observation_interval_seconds: float = 1.0,
    ) -> None:
        """Wait for the owned operation while projecting its latest status."""

        future = self.ensure_started()
        while not future.done():
            snapshot = self.snapshot()
            if status_callback is not None:
                status_callback(snapshot)
            try:
                future.result(timeout=observation_interval_seconds)
            except FutureTimeoutError:
                continue
        snapshot = self.snapshot()
        if status_callback is not None:
            status_callback(snapshot)
        future.result()

    def snapshot(self) -> EndpointStartupStatus:
        """Return the latest immutable preparation update."""

        with self._lock:
            return self._snapshot

    def _set_message(self, message: str) -> None:
        with self._lock:
            self._snapshot = EndpointStartupStatus(
                sequence=self._snapshot.sequence + 1,
                phase=EndpointStartupPhase.PREPARING_CAPABILITIES,
                message=message,
                timestamp=time.time(),
            )

    def _prepare(self, future: Future[None]) -> None:
        try:
            self._function_catalog.catalog(
                compact_signatures=True,
                status_callback=self._set_message,
            )
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(None)
