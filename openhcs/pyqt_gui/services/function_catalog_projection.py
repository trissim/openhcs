"""Endpoint-owned function catalog projection for OpenHCS Qt consumers."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from zmqruntime import OperationCancellation

from openhcs.agent.dto.functions import (
    DEFAULT_FUNCTION_DETAIL_DOC_CHARS,
    FunctionCatalogControlRequest,
    FunctionCatalogEntry,
    FunctionCatalogPage,
    FunctionDetail,
    FunctionDetailControlRequest,
    FunctionSearchRequest,
)
from openhcs.runtime.zmq_config import OpenHCSZMQConfig

if TYPE_CHECKING:
    from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

logger = logging.getLogger(__name__)


FunctionCatalogClientFactory = Callable[
    [OpenHCSZMQConfig],
    "ZMQExecutionClient",
]


class EndpointFunctionUnavailableError(RuntimeError):
    """A server function cannot yet be represented by the local editor."""

    def __init__(
        self,
        entry: FunctionCatalogEntry,
        endpoint: OpenHCSZMQConfig,
    ) -> None:
        self.entry = entry
        self.endpoint = endpoint
        super().__init__(
            f"{entry.name!r} is available on the connected execution server "
            f"({endpoint.client_host}:{endpoint.default_port}) but its callable "
            "cannot be imported by this UI process. Remote-only function authoring "
            "requires the server-owned function-reference transport."
        )


@dataclass(frozen=True, slots=True)
class FunctionCatalogProjection:
    """One exact catalog snapshot derived from one execution endpoint."""

    endpoint: OpenHCSZMQConfig
    page: FunctionCatalogPage
    entries_by_id: Mapping[str, FunctionCatalogEntry]

    @classmethod
    def from_page(
        cls,
        endpoint: OpenHCSZMQConfig,
        page: FunctionCatalogPage,
    ) -> FunctionCatalogProjection:
        entries_by_id = MappingProxyType(
            {entry.function_id: entry for entry in page.items}
        )
        if len(entries_by_id) != len(page.items):
            raise ValueError(
                "Execution endpoint returned duplicate function identities in one "
                "catalog revision."
            )
        return cls(endpoint=endpoint, page=page, entries_by_id=entries_by_id)

    @property
    def namespace(self) -> tuple[OpenHCSZMQConfig, str]:
        """Complete endpoint configuration plus server-owned catalog revision."""

        return self.endpoint, self.page.revision


class ZMQFunctionCatalogProjectionService:
    """Materialize the live server catalog without a local registry fallback."""

    def __init__(
        self,
        config_provider: Callable[[], OpenHCSZMQConfig],
        *,
        client_factory: FunctionCatalogClientFactory | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._client_factory = client_factory or self._new_client
        self._client_endpoint: OpenHCSZMQConfig | None = None
        self._client: ZMQExecutionClient | None = None
        self._projection: FunctionCatalogProjection | None = None
        self._projection_compact_signatures: bool | None = None
        self._preparation_future: Future[FunctionCatalogPage] | None = None
        self._preparation_key: tuple[OpenHCSZMQConfig, bool] | None = None
        self._preparation_cancellation: OperationCancellation | None = None
        self._preparation_thread: threading.Thread | None = None
        self._preparation_generation = 0
        self._closed = False
        self._state_lock = threading.RLock()

    def _new_client(self, config: OpenHCSZMQConfig) -> ZMQExecutionClient:
        from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

        return ZMQExecutionClient(config=config)

    @property
    def projection(self) -> FunctionCatalogProjection | None:
        with self._state_lock:
            return self._projection

    def prepare(
        self,
        *,
        compact_signatures: bool = True,
    ) -> Future[FunctionCatalogPage]:
        """Start one shared endpoint catalog read without blocking the caller."""

        endpoint = self._config_provider()
        key = (endpoint, compact_signatures)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Function catalog projection is closed")
            if (
                self._projection is not None
                and self._projection.endpoint == endpoint
                and self._projection_compact_signatures == compact_signatures
            ):
                ready: Future[FunctionCatalogPage] = Future()
                ready.set_result(self._projection.page)
                return ready
            if self._preparation_future is not None and self._preparation_key == key:
                return self._preparation_future

            self._preparation_generation += 1
            generation = self._preparation_generation
            future: Future[FunctionCatalogPage] = Future()
            cancellation = OperationCancellation()
            thread = threading.Thread(
                target=self._prepare_catalog,
                args=(
                    endpoint,
                    compact_signatures,
                    generation,
                    future,
                    cancellation,
                ),
                name="openhcs-function-catalog-projection",
            )
            self._preparation_future = future
            self._preparation_key = key
            self._preparation_cancellation = cancellation
            self._preparation_thread = thread
            thread.start()
        return future

    def catalog(
        self,
        *,
        compact_signatures: bool = True,
    ) -> FunctionCatalogPage:
        endpoint = self._config_provider()
        page = self._client_for(endpoint).get_function_catalog(
            FunctionCatalogControlRequest(
                compact_signatures=compact_signatures,
            )
        )
        with self._state_lock:
            self._preparation_generation += 1
            self._preparation_future = None
            self._preparation_key = None
            self._projection = FunctionCatalogProjection.from_page(endpoint, page)
            self._projection_compact_signatures = compact_signatures
        return page

    def search(
        self,
        *,
        query: str | None = None,
        library: str | None = None,
        limit: int = 50,
        compact_signatures: bool = True,
    ) -> FunctionCatalogPage:
        endpoint = self._config_provider()
        return self._client_for(endpoint).search_function_catalog(
            FunctionSearchRequest(
                query=query,
                library=library,
                limit=limit,
                compact_signatures=compact_signatures,
            )
        )

    def get(
        self,
        function_id: str,
        *,
        max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS,
        compact_signature: bool = True,
    ) -> FunctionDetail:
        projection = self._require_projection()
        return self._client_for(projection.endpoint).get_function_detail(
            FunctionDetailControlRequest(
                function_id=function_id,
                catalog_revision=projection.page.revision,
                max_doc_chars=max_doc_chars,
                compact_signature=compact_signature,
            )
        )

    def import_selected_callable(self, function_id: str) -> Callable:
        """Import only the chosen function, never the UI-side registry catalog."""

        from openhcs.core.function_reference import FunctionReferenceTransportAuthority

        projection = self._require_projection()
        entry = projection.entries_by_id[function_id]
        module_name, separator, attribute_name = entry.import_path.rpartition(".")
        try:
            if not separator:
                raise ImportError(
                    f"Function import path {entry.import_path!r} has no module path."
                )
            target = FunctionReferenceTransportAuthority.importable_function(
                module_name,
                attribute_name,
            )
            if target is None:
                raise ImportError(
                    f"Function import path {entry.import_path!r} is unavailable."
                )
        except (ImportError, TypeError) as exc:
            raise EndpointFunctionUnavailableError(
                entry,
                projection.endpoint,
            ) from exc
        return target

    def invalidate(self) -> None:
        """Discard the derived page; the next read requests the endpoint again."""

        self._cancel_preparation()
        with self._state_lock:
            self._preparation_generation += 1
            self._projection = None
            self._projection_compact_signatures = None

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._cancel_preparation()
        if self._client is not None:
            try:
                self._client.disconnect()
            finally:
                self._client = None
                self._client_endpoint = None
        with self._state_lock:
            self._preparation_generation += 1
            self._projection = None
            self._projection_compact_signatures = None

    def _require_projection(self) -> FunctionCatalogProjection:
        endpoint = self._config_provider()
        projection = self.projection
        if projection is None or projection.endpoint != endpoint:
            self.catalog()
            projection = self.projection
        if projection is None:
            raise RuntimeError("Function catalog endpoint returned no projection.")
        return projection

    def _prepare_catalog(
        self,
        endpoint: OpenHCSZMQConfig,
        compact_signatures: bool,
        generation: int,
        future: Future[FunctionCatalogPage],
        cancellation: OperationCancellation,
    ) -> None:
        """Read one catalog on its client-owning worker thread."""

        if not future.set_running_or_notify_cancel():
            return
        client: ZMQExecutionClient | None = None
        try:
            client = self._client_factory(endpoint)
            page = client.get_function_catalog(
                FunctionCatalogControlRequest(
                    compact_signatures=compact_signatures,
                ),
                cancellation=cancellation,
            )
            projection = FunctionCatalogProjection.from_page(endpoint, page)
            with self._state_lock:
                if generation == self._preparation_generation:
                    self._projection = projection
                    self._projection_compact_signatures = compact_signatures
                    self._preparation_future = None
                    self._preparation_key = None
            future.set_result(page)
        except CancelledError as error:
            future.set_exception(error)
        except Exception as error:
            with self._state_lock:
                if self._preparation_future is future:
                    self._preparation_future = None
                    self._preparation_key = None
            future.set_exception(error)
        finally:
            if client is not None:
                try:
                    client.disconnect()
                except Exception:
                    logger.exception("Failed to disconnect function catalog client")
            with self._state_lock:
                if self._preparation_thread is threading.current_thread():
                    self._preparation_future = None
                    self._preparation_key = None
                    self._preparation_cancellation = None
                    self._preparation_thread = None

    def _cancel_preparation(self) -> None:
        """Cancel and join the exact endpoint-catalog worker owned by this service."""

        with self._state_lock:
            cancellation = self._preparation_cancellation
            thread = self._preparation_thread
            self._preparation_future = None
            self._preparation_key = None
            self._preparation_cancellation = None
            self._preparation_thread = None
        if cancellation is not None:
            cancellation.cancel()
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _client_for(self, endpoint: OpenHCSZMQConfig) -> ZMQExecutionClient:
        if self._client is not None and self._client_endpoint == endpoint:
            return self._client
        if self._client is not None:
            self._client.disconnect()
        self._client = self._client_factory(endpoint)
        self._client_endpoint = endpoint
        with self._state_lock:
            if self._projection is not None and self._projection.endpoint != endpoint:
                self._projection = None
                self._projection_compact_signatures = None
        return self._client
