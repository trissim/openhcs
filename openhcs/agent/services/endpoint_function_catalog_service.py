"""Endpoint-owned function catalog service for OpenHCS authoring consumers."""

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
    CustomFunctionRegistrationRequest,
    CustomFunctionRegistrationResult,
    FunctionCatalogControlRequest,
    FunctionCatalogEntry,
    FunctionCatalogPage,
    FunctionDetail,
    FunctionDetailControlRequest,
    FunctionReferenceControlRequest,
    FunctionSearchRequest,
)
from openhcs.agent.services.function_catalog_service import FunctionCatalogServiceABC
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
            "cannot be materialized by this authoring process."
        )


@dataclass(frozen=True, slots=True)
class FunctionCatalogEndpointRevision:
    """One execution endpoint paired with its catalog membership revision."""

    endpoint: OpenHCSZMQConfig
    revision: str


@dataclass(frozen=True, slots=True)
class FunctionCatalogProjection:
    """One exact catalog snapshot derived from one execution endpoint."""

    endpoint: OpenHCSZMQConfig
    page: FunctionCatalogPage
    compact_signatures: bool
    entries_by_id: Mapping[str, FunctionCatalogEntry]

    @classmethod
    def from_page(
        cls,
        endpoint: OpenHCSZMQConfig,
        page: FunctionCatalogPage,
        *,
        compact_signatures: bool,
    ) -> FunctionCatalogProjection:
        entries_by_id = MappingProxyType(
            {entry.function_id: entry for entry in page.items}
        )
        if len(entries_by_id) != len(page.items):
            raise ValueError(
                "Execution endpoint returned duplicate function identities in one "
                "catalog revision."
            )
        return cls(
            endpoint=endpoint,
            page=page,
            compact_signatures=compact_signatures,
            entries_by_id=entries_by_id,
        )

    @property
    def namespace(self) -> FunctionCatalogEndpointRevision:
        """Complete endpoint configuration plus server-owned catalog revision."""

        return FunctionCatalogEndpointRevision(self.endpoint, self.page.revision)


FunctionCatalogEndpointState = (
    FunctionCatalogEndpointRevision | FunctionCatalogProjection
)


@dataclass(frozen=True, slots=True)
class FunctionCatalogClientSession:
    """One client paired with the exact endpoint used to construct it."""

    endpoint: OpenHCSZMQConfig
    client: ZMQExecutionClient

    def disconnect(self) -> None:
        self.client.disconnect()


class FunctionCatalogPreparation:
    """One cancellable endpoint-catalog read and all of its lifecycle handles."""

    def __init__(
        self,
        endpoint: OpenHCSZMQConfig,
        compact_signatures: bool,
        worker: Callable[[FunctionCatalogPreparation], None],
    ) -> None:
        self.endpoint = endpoint
        self.compact_signatures = compact_signatures
        self.future: Future[FunctionCatalogPage] = Future()
        self.cancellation = OperationCancellation()
        self.thread = threading.Thread(
            target=worker,
            args=(self,),
            name="openhcs-function-catalog-projection",
        )

    def matches(
        self,
        endpoint: OpenHCSZMQConfig,
        compact_signatures: bool,
    ) -> bool:
        """Return whether this operation already owns the requested read."""

        return (
            self.endpoint == endpoint and self.compact_signatures == compact_signatures
        )

    def cancel_and_join(self) -> None:
        """Stop this operation and wait for its client-owning worker."""

        self.cancellation.cancel()
        if self.thread is not threading.current_thread():
            self.thread.join()


class ZMQFunctionCatalogService(FunctionCatalogServiceABC):
    """Use one execution endpoint as the authoring callable-catalog authority."""

    def __init__(
        self,
        config_provider: Callable[[], OpenHCSZMQConfig],
        *,
        client_factory: FunctionCatalogClientFactory | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._client_factory = client_factory or self._new_client
        self._client_session: FunctionCatalogClientSession | None = None
        self._endpoint_state: FunctionCatalogEndpointState | None = None
        self._preparation: FunctionCatalogPreparation | None = None
        self._closed = False
        self._state_lock = threading.RLock()

    def _new_client(self, config: OpenHCSZMQConfig) -> ZMQExecutionClient:
        from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

        return ZMQExecutionClient(config=config)

    @property
    def projection(self) -> FunctionCatalogProjection | None:
        with self._state_lock:
            state = self._endpoint_state
        return state if isinstance(state, FunctionCatalogProjection) else None

    def prepare(
        self,
        *,
        compact_signatures: bool = True,
    ) -> Future[FunctionCatalogPage]:
        """Start one shared endpoint catalog read without blocking the caller."""

        endpoint = self._config_provider()
        self._cancel_mismatched_preparation(endpoint, compact_signatures)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Function catalog projection is closed")
            if (
                isinstance(self._endpoint_state, FunctionCatalogProjection)
                and self._endpoint_state.endpoint == endpoint
                and self._endpoint_state.compact_signatures == compact_signatures
            ):
                ready: Future[FunctionCatalogPage] = Future()
                ready.set_result(self._endpoint_state.page)
                return ready
            if self._preparation is not None and self._preparation.matches(
                endpoint,
                compact_signatures,
            ):
                return self._preparation.future

            preparation = FunctionCatalogPreparation(
                endpoint,
                compact_signatures,
                self._prepare_catalog,
            )
            self._preparation = preparation
            preparation.thread.start()
        return preparation.future

    def catalog(
        self,
        *,
        compact_signatures: bool = True,
        status_callback: Callable[[str], None] | None = None,
        cancellation: OperationCancellation | None = None,
    ) -> FunctionCatalogPage:
        self._cancel_preparation()
        endpoint = self._config_provider()
        if status_callback is not None:
            status_callback("Requesting the execution endpoint function catalog")
        page = self._client_for(endpoint).get_function_catalog(
            FunctionCatalogControlRequest(
                compact_signatures=compact_signatures,
            ),
            cancellation=cancellation,
        )
        with self._state_lock:
            self._endpoint_state = FunctionCatalogProjection.from_page(
                endpoint,
                page,
                compact_signatures=compact_signatures,
            )
        if status_callback is not None:
            status_callback(f"Function catalog ready ({page.total} functions)")
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
        page = self._client_for(endpoint).search_function_catalog(
            FunctionSearchRequest(
                query=query,
                library=library,
                limit=limit,
                compact_signatures=compact_signatures,
            )
        )
        with self._state_lock:
            state = self._endpoint_state
            if (
                state is None
                or state.endpoint != endpoint
                or self._state_revision(state) != page.revision
            ):
                self._endpoint_state = FunctionCatalogEndpointRevision(
                    endpoint,
                    page.revision,
                )
        return page

    def get(
        self,
        function_id: str,
        *,
        max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS,
        compact_signature: bool = True,
    ) -> FunctionDetail:
        endpoint_revision = self._require_endpoint_revision()
        return self._client_for(endpoint_revision.endpoint).get_function_detail(
            FunctionDetailControlRequest(
                function_id=function_id,
                catalog_revision=endpoint_revision.revision,
                max_doc_chars=max_doc_chars,
                compact_signature=compact_signature,
            )
        )

    def get_by_import_path(
        self,
        import_path: str,
        *,
        max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS,
        compact_signature: bool = True,
    ) -> FunctionDetail | None:
        requested_import_path = import_path.strip()
        if not requested_import_path:
            return None
        page = self.search(
            query=requested_import_path,
            limit=50,
            compact_signatures=True,
        )
        entry = next(
            (item for item in page.items if item.import_path == requested_import_path),
            None,
        )
        if entry is None:
            return None
        return self.get(
            entry.function_id,
            max_doc_chars=max_doc_chars,
            compact_signature=compact_signature,
        )

    def resolve(self, function_id: str) -> Callable:
        """Resolve only the selected endpoint reference in this process."""

        endpoint_revision = self._require_endpoint_revision()
        entry = self._entry(function_id)
        try:
            reference = self._client_for(
                endpoint_revision.endpoint
            ).get_function_reference(
                FunctionReferenceControlRequest(
                    function_id=function_id,
                    catalog_revision=endpoint_revision.revision,
                )
            )
            return reference.resolve()
        except (ImportError, RuntimeError, TypeError) as exc:
            raise EndpointFunctionUnavailableError(
                entry,
                endpoint_revision.endpoint,
            ) from exc

    def register_custom_function(
        self,
        request: CustomFunctionRegistrationRequest,
    ) -> CustomFunctionRegistrationResult:
        """Register source at the endpoint and project ephemeral source locally."""

        endpoint = self._config_provider()
        result = self._client_for(endpoint).register_custom_function(request)
        if not request.persist:
            from openhcs.processing.custom_functions.manager import (
                CustomFunctionManager,
            )

            CustomFunctionManager().register_from_code(
                request.source_code,
                persist=False,
                clear_caches=False,
                emit_signal=False,
            )
        self.invalidate()
        return result

    def invalidate(self) -> None:
        """Discard the derived page; the next read requests the endpoint again."""

        self._cancel_preparation()
        with self._state_lock:
            self._endpoint_state = None

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._cancel_preparation()
        if self._client_session is not None:
            try:
                self._client_session.disconnect()
            finally:
                self._client_session = None
        with self._state_lock:
            self._endpoint_state = None

    def _require_endpoint_revision(self) -> FunctionCatalogEndpointRevision:
        endpoint = self._config_provider()
        with self._state_lock:
            state = self._endpoint_state
        if state is None or state.endpoint != endpoint:
            self.catalog()
            with self._state_lock:
                state = self._endpoint_state
        if state is None:
            raise RuntimeError("Function catalog endpoint returned no revision.")
        if isinstance(state, FunctionCatalogProjection):
            return state.namespace
        return state

    @staticmethod
    def _state_revision(state: FunctionCatalogEndpointState) -> str:
        if isinstance(state, FunctionCatalogProjection):
            return state.page.revision
        return state.revision

    def _entry(self, function_id: str) -> FunctionCatalogEntry:
        projection = self.projection
        if projection is not None:
            entry = projection.entries_by_id.get(function_id)
            if entry is not None:
                return entry
        return self.get(
            function_id,
            max_doc_chars=0,
            compact_signature=True,
        ).entry

    def _prepare_catalog(
        self,
        preparation: FunctionCatalogPreparation,
    ) -> None:
        """Read one catalog on its client-owning worker thread."""

        if not preparation.future.set_running_or_notify_cancel():
            return
        client: ZMQExecutionClient | None = None
        try:
            client = self._client_factory(preparation.endpoint)
            page = client.get_function_catalog(
                FunctionCatalogControlRequest(
                    compact_signatures=preparation.compact_signatures,
                ),
                cancellation=preparation.cancellation,
            )
            projection = FunctionCatalogProjection.from_page(
                preparation.endpoint,
                page,
                compact_signatures=preparation.compact_signatures,
            )
            with self._state_lock:
                if self._preparation is preparation:
                    self._endpoint_state = projection
                    self._preparation = None
            preparation.future.set_result(page)
        except CancelledError as error:
            preparation.future.set_exception(error)
        except Exception as error:
            with self._state_lock:
                if self._preparation is preparation:
                    self._preparation = None
            preparation.future.set_exception(error)
        finally:
            if client is not None:
                try:
                    client.disconnect()
                except Exception:
                    logger.exception("Failed to disconnect function catalog client")
            with self._state_lock:
                if self._preparation is preparation:
                    self._preparation = None

    def _cancel_preparation(self) -> None:
        """Cancel and join the exact endpoint-catalog worker owned by this service."""

        with self._state_lock:
            preparation = self._preparation
            self._preparation = None
        if preparation is None:
            return
        preparation.cancel_and_join()

    def _cancel_mismatched_preparation(
        self,
        endpoint: OpenHCSZMQConfig,
        compact_signatures: bool,
    ) -> None:
        """Cancel an active read only when it cannot satisfy this request."""

        with self._state_lock:
            preparation = self._preparation
            if preparation is None or preparation.matches(
                endpoint,
                compact_signatures,
            ):
                return
            self._preparation = None
        preparation.cancel_and_join()

    def _client_for(self, endpoint: OpenHCSZMQConfig) -> ZMQExecutionClient:
        if self._client_session is not None:
            if self._client_session.endpoint == endpoint:
                return self._client_session.client
            self._client_session.disconnect()
        self._client_session = FunctionCatalogClientSession(
            endpoint,
            self._client_factory(endpoint),
        )
        with self._state_lock:
            if (
                self._endpoint_state is not None
                and self._endpoint_state.endpoint != endpoint
            ):
                self._endpoint_state = None
        return self._client_session.client
