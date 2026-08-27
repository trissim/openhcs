"""Asynchronous preparation lifecycle for an execution endpoint's catalog."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING

from zmqruntime import OperationCancellation
from zmqruntime.startup import EndpointStartupPhase, EndpointStartupStatus

if TYPE_CHECKING:
    from openhcs.agent.services.function_catalog_service import (
        FunctionCatalogServiceABC,
    )


class FunctionCatalogPreparation:
    """Own one lazily started endpoint catalog preparation operation."""

    @staticmethod
    def prepare_persistent_catalog() -> None:
        """Build registry-owned persistent caches in the dedicated child process."""

        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        RegistryService.prepare_in_current_process()

    def __init__(
        self,
        function_catalog: "FunctionCatalogServiceABC",
    ) -> None:
        self._lock = threading.RLock()
        self._future: Future[None] | None = None
        self._thread: threading.Thread | None = None
        self._cancellation = OperationCancellation()
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
            if self._cancellation.requested():
                future.cancel()
                return future
            self._set_message("Starting function catalog preparation")
            thread = threading.Thread(
                target=self._prepare,
                args=(future,),
                name="openhcs-function-catalog-preparation",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return future

    def cancel_and_join(self) -> None:
        """Cancel and join the exact preparation operation owned here."""

        self._cancellation.cancel()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

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
                cancellation=self._cancellation,
            )
        except CancelledError:
            future.cancel()
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(None)
