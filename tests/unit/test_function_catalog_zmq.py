from __future__ import annotations

import ast
import inspect
import textwrap
from concurrent.futures import CancelledError, Future
from dataclasses import replace

import pytest
from zmqruntime import OperationCancellation
from zmqruntime.execution import ExecutionServer
from zmqruntime.startup import EndpointStartupPhase, EndpointStartupStatus

from openhcs.agent.dto.functions import (
    FunctionCatalogControlPayload,
    FunctionCatalogControlRequest,
    FunctionCatalogControlResponse,
    FunctionCatalogEntry,
    FunctionCatalogPreparationControlResponse,
    FunctionDetail,
    FunctionDetailControlPayload,
    FunctionDetailControlRequest,
    FunctionDetailControlResponse,
    FunctionSearchControlPayload,
    FunctionSearchRequest,
    catalog_page,
)
from openhcs.agent.services.function_catalog_service import FunctionCatalogService
from openhcs.runtime.zmq_control import (
    ZMQControlMessageRouter,
    ZMQControlRequestContext,
)
from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
from openhcs.runtime.zmq_execution_server import ZMQExecutionServer


def test_gui_application_setup_does_not_initialize_execution_catalog() -> None:
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(OpenHCSPyQtApp.setup_application))
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "initialize_registry" not in called_names


def _entry(*, function_id: str = "cpu:sample") -> FunctionCatalogEntry:
    return FunctionCatalogEntry(
        function_id=function_id,
        import_path="example.sample",
        name="sample",
        module="example",
        library="cpu",
        signature="sample(image, sigma=1.0)",
        summary="Sample function.",
        backend_tags=("numpy",),
    )


def _catalog(entry: FunctionCatalogEntry | None = None):
    items = (_entry() if entry is None else entry,)
    return catalog_page(
        items=items,
        total=len(items),
        limit=len(items),
        query=None,
        library=None,
    )


def _detail(entry: FunctionCatalogEntry | None = None) -> FunctionDetail:
    return FunctionDetail(
        schema_version="openhcs.agent.v1",
        entry=_entry() if entry is None else entry,
        parameters=(),
        doc="Sample function.",
    )


class _StaticFunctionCatalogPreparation:
    def __init__(self, future: Future[None]) -> None:
        self.future = future

    def ensure_started(self) -> Future[None]:
        return self.future

    def snapshot(self) -> EndpointStartupStatus:
        return EndpointStartupStatus(
            sequence=1,
            phase=EndpointStartupPhase.PREPARING_CAPABILITIES,
            message="Discovering functions",
            timestamp=1.0,
        )


def _context(
    preparation_future: Future[None] | None = None,
) -> ZMQControlRequestContext:
    ready = preparation_future or Future()
    if preparation_future is None:
        ready.set_result(None)
    return ZMQControlRequestContext(
        compiled_artifacts={},
        function_catalog=FunctionCatalogService(),
        function_catalog_preparation=_StaticFunctionCatalogPreparation(ready),
    )


def test_catalog_revision_is_derived_from_entry_owned_membership_identity() -> None:
    catalog = _catalog()
    presentation_change = _catalog(
        replace(
            _entry(),
            signature="sample(image, sigma: float = 1.0)",
            summary="A longer presentation summary.",
        )
    )
    membership_change = _catalog(replace(_entry(), backend_tags=("cupy",)))

    assert presentation_change.revision == catalog.revision
    assert membership_change.revision != catalog.revision


def test_function_catalog_control_payload_roundtrip() -> None:
    request = FunctionCatalogControlRequest(compact_signatures=False)
    payload = FunctionCatalogControlPayload.from_request(request)

    assert FunctionCatalogControlPayload.from_dict(payload.to_dict()).request is request


def test_function_detail_control_payload_roundtrip() -> None:
    request = FunctionDetailControlRequest(
        function_id="cpu:sample",
        catalog_revision="revision",
        max_doc_chars=123,
        compact_signature=False,
    )
    payload = FunctionDetailControlPayload.from_request(request)

    assert FunctionDetailControlPayload.from_dict(payload.to_dict()).request is request


def test_function_search_control_payload_reuses_typed_request() -> None:
    request = FunctionSearchRequest(
        query="segment nuclei",
        library="cpu",
        limit=17,
        compact_signatures=False,
    )
    payload = FunctionSearchControlPayload(request=request)

    assert FunctionSearchControlPayload.from_dict(payload.to_dict()).request is request


def test_zmq_router_reports_catalog_preparation_without_blocking() -> None:
    preparing: Future[None] = Future()

    response = ZMQControlMessageRouter.handle(
        FunctionCatalogControlPayload.from_request(
            FunctionCatalogControlRequest()
        ).to_dict(),
        _context(preparing),
    )

    pending = FunctionCatalogPreparationControlResponse.from_control_response(response)
    assert pending is not None
    assert pending.retry_after_seconds > 0


def test_catalog_client_polls_typed_preparation_response(monkeypatch) -> None:
    pending = FunctionCatalogPreparationControlResponse(
        status=EndpointStartupStatus(
            phase=EndpointStartupPhase.PREPARING_CAPABILITIES,
            message="Discovering functions",
            sequence=1,
            timestamp=1.0,
        ),
        retry_after_seconds=0.25,
    ).to_control_response()
    ready = {"status": "ok", "catalog": _catalog()}
    responses = iter((pending, ready))
    waits: list[float] = []

    class _Cancellation:
        @staticmethod
        def requested() -> bool:
            return False

        @staticmethod
        def wait(timeout: float) -> bool:
            waits.append(timeout)
            return False

    client = ZMQExecutionClient(port=22319, persistent=True)
    monkeypatch.setattr(
        client, "_send_control_request", lambda request: next(responses)
    )
    response = client._send_function_catalog_control_request(
        FunctionCatalogControlPayload.from_request(
            FunctionCatalogControlRequest()
        ).to_dict(),
        cancellation=_Cancellation(),
    )

    assert response is ready
    assert waits == [0.25]


def test_catalog_client_cancellation_prevents_another_poll(monkeypatch) -> None:
    cancellation = OperationCancellation()
    cancellation.cancel()
    client = ZMQExecutionClient(port=22319, persistent=True)
    monkeypatch.setattr(
        client,
        "_send_control_request",
        lambda _request: (_ for _ in ()).throw(
            AssertionError("Cancelled catalog preparation must not poll")
        ),
    )

    with pytest.raises(CancelledError, match="catalog preparation"):
        client._send_function_catalog_control_request(
            FunctionCatalogControlPayload.from_request(
                FunctionCatalogControlRequest()
            ).to_dict(),
            cancellation=cancellation,
        )


def test_zmq_router_projects_catalog_and_revision_checked_detail(monkeypatch) -> None:
    catalog = _catalog()
    detail = _detail()
    monkeypatch.setattr(
        FunctionCatalogService,
        "catalog",
        lambda self, *, compact_signatures=False: catalog,
    )
    monkeypatch.setattr(
        FunctionCatalogService,
        "get",
        lambda self, function_id, **kwargs: detail,
    )

    catalog_response = ZMQControlMessageRouter.handle(
        FunctionCatalogControlPayload.from_request(
            FunctionCatalogControlRequest()
        ).to_dict(),
        _context(),
    )
    projected_catalog = FunctionCatalogControlResponse.from_control_response(
        catalog_response
    ).catalog

    detail_response = ZMQControlMessageRouter.handle(
        FunctionDetailControlPayload.from_request(
            FunctionDetailControlRequest(
                function_id=detail.entry.function_id,
                catalog_revision=projected_catalog.revision,
            )
        ).to_dict(),
        _context(),
    )

    assert projected_catalog is catalog
    assert (
        FunctionDetailControlResponse.from_control_response(detail_response).detail
        is detail
    )


def test_zmq_router_rejects_detail_from_stale_catalog_revision(monkeypatch) -> None:
    monkeypatch.setattr(
        FunctionCatalogService,
        "catalog",
        lambda self, *, compact_signatures=False: _catalog(),
    )

    response = ZMQControlMessageRouter.handle(
        FunctionDetailControlPayload.from_request(
            FunctionDetailControlRequest(
                function_id="cpu:sample",
                catalog_revision="stale",
            )
        ).to_dict(),
        _context(),
    )

    assert response["status"] == "error"
    assert "changed after it was read" in response["error"]


def test_zmq_router_delegates_search_to_catalog_owner(monkeypatch) -> None:
    catalog = _catalog()
    observed = []

    def _search(self, **kwargs):
        observed.append(kwargs)
        return catalog

    monkeypatch.setattr(FunctionCatalogService, "search", _search)
    request = FunctionSearchRequest(
        query="sample",
        library="cpu",
        limit=9,
        compact_signatures=True,
    )

    response = ZMQControlMessageRouter.handle(
        FunctionSearchControlPayload(request=request).to_dict(),
        _context(),
    )

    assert (
        FunctionCatalogControlResponse.from_control_response(response).catalog
        is catalog
    )
    assert observed == [
        {
            "query": "sample",
            "library": "cpu",
            "limit": 9,
            "compact_signatures": True,
        }
    ]


def test_execution_server_start_only_binds_endpoint(
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        FunctionCatalogService,
        "catalog",
        lambda self, *, compact_signatures=False: events.append("catalog"),
    )
    monkeypatch.setattr(
        ExecutionServer,
        "start",
        lambda self: events.append("bind"),
    )

    ZMQExecutionServer().start()

    assert events == ["bind"]


def test_execution_server_runtime_capability_preparation_uses_single_owner(
    monkeypatch,
) -> None:
    events = []
    server = ZMQExecutionServer()
    monkeypatch.setattr(
        server._function_catalog_preparation,
        "wait_until_ready",
        lambda callback=None: events.append(callback),
    )
    callback = object()

    server.prepare_runtime_capabilities(callback)

    assert events == [callback]


def test_persistent_capability_preparation_uses_registry_owner(
    monkeypatch,
) -> None:
    from openhcs.processing.backends.lib_registry.registry_service import (
        RegistryService,
    )
    from openhcs.runtime.function_catalog_preparation import (
        FunctionCatalogPreparation,
    )

    events: list[str] = []
    monkeypatch.setattr(
        RegistryService,
        "prepare_in_current_process",
        lambda: events.append("prepare"),
    )
    FunctionCatalogPreparation.prepare_persistent_catalog()

    assert events == ["prepare"]


def test_endpoint_catalog_reconciles_persisted_custom_function_sources(
    tmp_path,
    monkeypatch,
) -> None:
    """One running catalog reflects external add/delete source mutations."""

    import openhcs.processing.func_registry as func_registry
    from openhcs.processing.backends.lib_registry.registry_service import (
        RegistryService,
    )
    from openhcs.processing.custom_functions.manager import CustomFunctionManager
    from openhcs.processing.custom_functions.runtime_registry import (
        CustomFunctionRuntimeRegistry,
    )

    def registered_custom_metadata(
        cls,
        *,
        status_callback=None,
    ):
        """Project only metadata attached by the real custom-function owner."""

        del cls, status_callback
        metadata = CustomFunctionRuntimeRegistry.metadata_by_name().values()
        return {
            function_metadata.composite_key: function_metadata
            for function_metadata in metadata
        }

    with monkeypatch.context() as isolated_catalog:
        isolated_catalog.setenv("OPENHCS_CPU_ONLY", "true")
        isolated_catalog.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        isolated_catalog.setattr(func_registry, "_registry_initialized", False)
        isolated_catalog.setattr(
            CustomFunctionRuntimeRegistry,
            "_metadata_by_name",
            {},
        )
        isolated_catalog.setattr(
            CustomFunctionRuntimeRegistry,
            "_source_revision",
            None,
        )
        isolated_catalog.setattr(
            RegistryService,
            "get_all_functions_with_metadata",
            classmethod(registered_custom_metadata),
        )

        service = FunctionCatalogService()
        manager = CustomFunctionManager()
        before = {item.name for item in service.catalog().items}
        manager.register_from_code(
            "@numpy\ndef live_catalog_refresh_probe(image):\n    return image\n",
            persist=True,
            emit_signal=False,
        )
        after_add = {item.name for item in service.catalog().items}
        assert "live_catalog_refresh_probe" not in before
        assert "live_catalog_refresh_probe" in after_add
        assert manager.delete_custom_function("live_catalog_refresh_probe")
        after_delete = {item.name for item in service.catalog().items}
        assert "live_catalog_refresh_probe" not in after_delete
