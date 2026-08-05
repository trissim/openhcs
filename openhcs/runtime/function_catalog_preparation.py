"""Asynchronous preparation lifecycle for an execution endpoint's catalog."""

from __future__ import annotations

import threading
from concurrent.futures import Future


class FunctionCatalogPreparation:
    """Own one lazily started endpoint catalog preparation operation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._future: Future[None] | None = None

    def ensure_started(self) -> Future[None]:
        """Return the one preparation future, starting it when first requested."""

        with self._lock:
            if self._future is not None:
                return self._future
            future: Future[None] = Future()
            self._future = future
            threading.Thread(
                target=self._prepare,
                args=(future,),
                name="openhcs-function-catalog-preparation",
                daemon=True,
            ).start()
            return future

    def wait_until_ready(self) -> None:
        """Wait for the live preparation operation and propagate its failure."""

        self.ensure_started().result()

    @staticmethod
    def _prepare(future: Future[None]) -> None:
        try:
            from openhcs.processing.backends.lib_registry.registry_service import (
                RegistryService,
            )

            RegistryService.get_all_functions_with_metadata()
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(None)
