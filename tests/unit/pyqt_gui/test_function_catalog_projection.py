"""Qt function discovery derives exclusively from the connected ZMQ endpoint."""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from dataclasses import replace

from openhcs.agent.dto.functions import (
    FunctionCatalogControlRequest,
    FunctionCatalogEntry,
    FunctionDetail,
    FunctionDetailControlRequest,
    FunctionReferenceControlRequest,
    FunctionSearchRequest,
    catalog_page,
)
from openhcs.agent.services.endpoint_function_catalog_service import (
    EndpointFunctionUnavailableError,
    FunctionCatalogEndpointRevision,
    ZMQFunctionCatalogService,
)
from openhcs.core.callable_contract import CallableImportIdentity
from openhcs.core.function_reference import ImportableFunctionReference
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.custom_functions.signals import CustomFunctionSignals
from openhcs.pyqt_gui.dialogs import function_selector_dialog as selector_module
from openhcs.pyqt_gui.dialogs.function_selector_dialog import FunctionSelectorDialog
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


def _entry(
    function_id: str,
    *,
    import_path: str | None = None,
    library: str = "cpu",
    backend_tags: tuple[str, ...] = (),
) -> FunctionCatalogEntry:
    name = function_id.rpartition(":")[2]
    return FunctionCatalogEntry(
        function_id=function_id,
        import_path=import_path or f"{__name__}._local_function",
        name=name,
        module=(import_path or __name__).rpartition(".")[0],
        library=library,
        signature=f"{name}()",
        summary=f"{name} summary",
        backend_tags=backend_tags,
    )


def _page(*entries: FunctionCatalogEntry):
    return catalog_page(
        items=entries,
        catalog_items=entries,
        total=len(entries),
        limit=len(entries),
        query=None,
        library=None,
    )


class _EndpointClient:
    def __init__(self, *catalogs) -> None:
        self.catalogs = list(catalogs)
        self.catalog_requests: list[FunctionCatalogControlRequest] = []
        self.search_requests: list[FunctionSearchRequest] = []
        self.detail_requests: list[FunctionDetailControlRequest] = []
        self.reference_requests: list[FunctionReferenceControlRequest] = []
        self.disconnected = False

    def get_function_catalog(
        self,
        request: FunctionCatalogControlRequest,
        *,
        cancellation=None,
    ):
        del cancellation
        self.catalog_requests.append(request)
        if len(self.catalogs) > 1:
            return self.catalogs.pop(0)
        return self.catalogs[0]

    def search_function_catalog(self, request: FunctionSearchRequest):
        self.search_requests.append(request)
        return self.catalogs[-1]

    def get_function_detail(self, request: FunctionDetailControlRequest):
        self.detail_requests.append(request)
        entry = next(
            entry
            for entry in self.catalogs[-1].items
            if entry.function_id == request.function_id
        )
        return FunctionDetail(
            schema_version="test",
            entry=entry,
            parameters=(),
            doc=entry.summary,
        )

    def get_function_reference(self, request: FunctionReferenceControlRequest):
        self.reference_requests.append(request)
        entry = next(
            entry
            for entry in self.catalogs[-1].items
            if entry.function_id == request.function_id
        )
        module_name, _, function_name = entry.import_path.rpartition(".")
        return ImportableFunctionReference(
            import_identity=CallableImportIdentity(module_name, function_name),
            composite_key=f"python:{module_name}:{function_name}",
        )

    def disconnect(self) -> None:
        self.disconnected = True


def _local_function():
    return "local"


def _wait_until(qapp, predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("Qt condition did not become true")
        qapp.processEvents()
        time.sleep(0.01)


def test_projection_replaces_membership_on_revision_and_endpoint_change() -> None:
    endpoint_one = replace(OPENHCS_ZMQ_CONFIG, default_port=17777)
    endpoint_two = replace(OPENHCS_ZMQ_CONFIG, default_port=27777)
    selected_endpoint = [endpoint_one]
    first_entry = _entry("cpu:first")
    custom_entry = _entry(
        "custom:remote_segmentation",
        import_path="remote_host.custom_functions.remote_segmentation",
        library="custom",
        backend_tags=("gpu", "custom"),
    )
    client_one = _EndpointClient(_page(first_entry), _page(custom_entry))
    second_entry = _entry("gpu:second", library="gpu", backend_tags=("gpu",))
    client_two = _EndpointClient(_page(second_entry))
    clients = {endpoint_one: client_one, endpoint_two: client_two}
    service = ZMQFunctionCatalogService(
        lambda: selected_endpoint[0],
        client_factory=clients.__getitem__,
    )

    first_page = service.catalog()
    first_namespace = service.projection.namespace
    assert tuple(service.projection.entries_by_id) == (first_entry.function_id,)

    custom_page = service.catalog()
    assert custom_page.revision != first_page.revision
    assert tuple(service.projection.entries_by_id) == (custom_entry.function_id,)
    assert service.projection.namespace == FunctionCatalogEndpointRevision(
        endpoint_one,
        custom_page.revision,
    )
    assert service.projection.namespace != first_namespace

    selected_endpoint[0] = endpoint_two
    service.catalog()
    assert client_one.disconnected
    assert service.projection.namespace == FunctionCatalogEndpointRevision(
        endpoint_two,
        client_two.catalogs[0].revision,
    )
    assert tuple(service.projection.entries_by_id) == (second_entry.function_id,)


def test_projection_sends_nominal_search_and_detail_requests_unchanged() -> None:
    entry = _entry("cpu:measure")
    page = _page(entry)
    client = _EndpointClient(page)
    service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: client,
    )

    service.catalog(compact_signatures=False)
    search_page = service.search(
        query="measure nuclei",
        library="cpu",
        limit=17,
        compact_signatures=False,
    )
    detail = service.get(entry.function_id, max_doc_chars=321)
    resolved = service.resolve(entry.function_id)

    assert client.catalog_requests == [
        FunctionCatalogControlRequest(compact_signatures=False)
    ]
    assert client.search_requests == [
        FunctionSearchRequest(
            query="measure nuclei",
            library="cpu",
            limit=17,
            compact_signatures=False,
        )
    ]
    assert client.detail_requests == [
        FunctionDetailControlRequest(
            function_id=entry.function_id,
            catalog_revision=page.revision,
            max_doc_chars=321,
            compact_signature=True,
        )
    ]
    assert search_page is page
    assert detail.entry is entry
    assert resolved is _local_function
    assert client.reference_requests == [
        FunctionReferenceControlRequest(
            function_id=entry.function_id,
            catalog_revision=page.revision,
        )
    ]


def test_search_result_revision_authorizes_followup_detail_without_full_read() -> None:
    entry = _entry("cpu:filtered")
    other = _entry("cpu:other")
    search_page = catalog_page(
        items=(entry,),
        catalog_items=(entry, other),
        total=1,
        limit=1,
        query="filtered",
        library=None,
    )
    client = _EndpointClient(search_page)
    service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: client,
    )

    page = service.search(query="filtered", limit=1)
    detail = service.get(entry.function_id)

    assert page is search_page
    assert detail.entry is entry
    assert client.catalog_requests == []
    assert client.detail_requests[0].catalog_revision == search_page.revision


def test_search_membership_change_replaces_complete_projection_with_revision() -> None:
    first = _entry("cpu:first")
    second = _entry("cpu:second")
    client = _EndpointClient(_page(first), _page(second))
    service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: client,
    )

    service.catalog()
    assert service.projection is not None

    search_page = service.search(query="second")

    assert service.projection is None
    detail = service.get(second.function_id)
    assert detail.entry is second
    assert client.detail_requests[-1].catalog_revision == search_page.revision


def test_projection_coalesces_nonblocking_catalog_preparation() -> None:
    entry = _entry("cpu:prepared")
    release = threading.Event()

    class _BlockingEndpointClient(_EndpointClient):
        def get_function_catalog(self, request, *, cancellation=None):
            del cancellation
            self.catalog_requests.append(request)
            if not release.wait(2):
                raise TimeoutError("test did not release catalog request")
            return self.catalogs[0]

    client = _BlockingEndpointClient(_page(entry))
    service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: client,
    )

    first = service.prepare()
    second = service.prepare()

    assert first is second
    assert not first.done()
    release.set()
    assert first.result(timeout=2).items == (entry,)
    assert len(client.catalog_requests) == 1
    assert service.projection is not None
    assert tuple(service.projection.entries_by_id) == (entry.function_id,)


def test_projection_close_cancels_and_joins_catalog_preparation() -> None:
    entry = _entry("cpu:cancelled")
    started = threading.Event()
    exited = threading.Event()

    class _CancellableEndpointClient(_EndpointClient):
        def get_function_catalog(self, request, *, cancellation=None):
            self.catalog_requests.append(request)
            if cancellation is None:
                raise AssertionError(
                    "Catalog preparation requires cancellation authority"
                )
            started.set()
            cancellation.wait()
            exited.set()
            raise CancelledError("catalog preparation cancelled")

    client = _CancellableEndpointClient(_page(entry))
    service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: client,
    )

    future = service.prepare()
    assert started.wait(1.0)

    service.close()

    assert exited.is_set()
    assert client.disconnected is True
    assert isinstance(future.exception(), CancelledError)
    assert not any(
        thread.name == "openhcs-function-catalog-projection"
        for thread in threading.enumerate()
    )


def test_prepared_projection_survives_same_endpoint_detail_client_creation() -> None:
    entry = _entry("cpu:prepared-detail")
    client = _EndpointClient(_page(entry))
    service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: client,
    )

    service.prepare().result(timeout=2)
    detail = service.get(entry.function_id)

    assert detail.entry is entry
    assert len(client.catalog_requests) == 1
    assert service.projection is not None
    assert service.projection.entries_by_id[entry.function_id] is entry


def test_selector_never_scans_local_registry_and_reports_remote_only_selection(
    qapp,
    monkeypatch,
) -> None:
    signals = CustomFunctionSignals()
    monkeypatch.setattr(selector_module, "custom_function_signals", signals)
    monkeypatch.setattr(
        RegistryService,
        "get_all_functions_with_metadata",
        classmethod(
            lambda cls: (_ for _ in ()).throw(
                AssertionError("UI must not scan the local function registry")
            )
        ),
    )
    remote_entry = _entry(
        "custom:remote_only",
        import_path="remote_host.custom_functions.remote_only",
        library="custom",
        backend_tags=("gpu", "custom"),
    )
    client = _EndpointClient(_page(remote_entry))
    service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: client,
    )
    dialog = FunctionSelectorDialog(service)

    try:
        _wait_until(
            qapp,
            lambda: remote_entry.function_id in dialog.catalog_entries,
        )
        row = dialog.catalog_entries[remote_entry.function_id]
        assert row is remote_entry
        dialog._on_function_selected(remote_entry.function_id, row)
        dialog.accept_selection()

        assert dialog.result() != dialog.DialogCode.Accepted
        assert not dialog.select_btn.isEnabled()
        assert "available on the connected execution server" in (
            dialog.function_table_browser.status_label.text()
        )
    finally:
        signals.functions_changed.disconnect(dialog._on_functions_changed)
        dialog.close()


def test_selector_construction_does_not_wait_for_endpoint_catalog(
    qapp,
) -> None:
    release = threading.Event()

    class _BlockingEndpointClient(_EndpointClient):
        def get_function_catalog(self, request, *, cancellation=None):
            del cancellation
            self.catalog_requests.append(request)
            if not release.wait(2):
                raise TimeoutError("test did not release catalog request")
            return self.catalogs[0]

    entry = _entry("cpu:later")
    client = _BlockingEndpointClient(_page(entry))
    service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: client,
    )
    started = time.monotonic()
    dialog = FunctionSelectorDialog(service)

    try:
        assert time.monotonic() - started < 0.25
        assert dialog.catalog_entries == {}
        assert "Loading function catalog" in (
            dialog.function_table_browser.status_label.text()
        )

        release.set()
        _wait_until(
            qapp,
            lambda: entry.function_id in dialog.catalog_entries,
        )
        assert dialog.function_table_browser.status_label.text() == "Functions: 1/1"
    finally:
        release.set()
        selector_module.custom_function_signals.functions_changed.disconnect(
            dialog._on_functions_changed
        )
        dialog.close()


def test_projection_imports_only_the_selected_declared_path() -> None:
    entry = _entry("cpu:local")
    service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: _EndpointClient(_page(entry)),
    )
    service.catalog()

    assert service.resolve(entry.function_id) is _local_function

    remote = _entry(
        "gpu:remote",
        import_path="remote_host.gpu.remote",
        library="gpu",
        backend_tags=("gpu",),
    )
    remote_service = ZMQFunctionCatalogService(
        lambda: OPENHCS_ZMQ_CONFIG,
        client_factory=lambda _config: _EndpointClient(_page(remote)),
    )
    remote_service.catalog()
    try:
        remote_service.resolve(remote.function_id)
    except EndpointFunctionUnavailableError as exc:
        assert exc.entry is remote
    else:
        raise AssertionError("Remote-only selection did not fail explicitly")
