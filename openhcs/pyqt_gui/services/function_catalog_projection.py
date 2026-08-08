"""Endpoint-owned function catalog projection for OpenHCS Qt consumers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

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
from zmqruntime.startup import EndpointStartupStatusCallback


class FunctionCatalogClient(Protocol):
    """Client operations consumed by the endpoint catalog projection."""

    def get_function_catalog(
        self,
        request: FunctionCatalogControlRequest,
    ) -> FunctionCatalogPage: ...

    def search_function_catalog(
        self,
        request: FunctionSearchRequest,
    ) -> FunctionCatalogPage: ...

    def get_function_detail(
        self,
        request: FunctionDetailControlRequest,
    ) -> FunctionDetail: ...

    def disconnect(self) -> None: ...


FunctionCatalogClientFactory = Callable[[OpenHCSZMQConfig], FunctionCatalogClient]


class FunctionCatalogProjectionReader(Protocol):
    """Endpoint catalog reads consumed by the Qt help surface."""

    def catalog(
        self,
        *,
        compact_signatures: bool = True,
    ) -> FunctionCatalogPage: ...

    def search(
        self,
        *,
        query: str | None = None,
        library: str | None = None,
        limit: int = 50,
        compact_signatures: bool = True,
    ) -> FunctionCatalogPage: ...

    def get(
        self,
        function_id: str,
        *,
        max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS,
        compact_signature: bool = True,
    ) -> FunctionDetail: ...


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
        status_callback: EndpointStartupStatusCallback | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._client_factory = client_factory or self._new_client
        self._status_callback = status_callback
        self._client_endpoint: OpenHCSZMQConfig | None = None
        self._client: FunctionCatalogClient | None = None
        self._projection: FunctionCatalogProjection | None = None

    def _new_client(self, config: OpenHCSZMQConfig) -> FunctionCatalogClient:
        from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

        return ZMQExecutionClient(
            config=config,
            connection_status_callback=self._status_callback,
        )

    def set_status_callback(
        self,
        callback: EndpointStartupStatusCallback | None,
    ) -> None:
        """Set the UI projection for future endpoint lifecycle updates."""

        if callback is self._status_callback:
            return
        self.close()
        self._status_callback = callback

    @property
    def projection(self) -> FunctionCatalogProjection | None:
        return self._projection

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
        self._projection = FunctionCatalogProjection.from_page(endpoint, page)
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

        self._projection = None

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.disconnect()
        finally:
            self._client = None
            self._client_endpoint = None
            self._projection = None

    def _require_projection(self) -> FunctionCatalogProjection:
        endpoint = self._config_provider()
        if self._projection is None or self._projection.endpoint != endpoint:
            self.catalog()
        if self._projection is None:
            raise RuntimeError("Function catalog endpoint returned no projection.")
        return self._projection

    def _client_for(self, endpoint: OpenHCSZMQConfig) -> FunctionCatalogClient:
        if self._client is not None and self._client_endpoint == endpoint:
            return self._client
        if self._client is not None:
            self._client.disconnect()
        self._client = self._client_factory(endpoint)
        self._client_endpoint = endpoint
        self._projection = None
        return self._client
