"""Ownership checks for the installed GUI and packaged MCP smoke test."""

import hashlib
from concurrent.futures import Future
from pathlib import Path

import pytest
from pyqt_reactive.services.window_snapshot import WindowSnapshotCaptureScope
from zmqruntime import (
    DataControlPortPair,
    DataControlPortPairAuthority,
    OperationDeadline,
    OperationTimeoutError,
    TransportMode,
)

from openhcs.agent.dto.common import AgentResourceRef
from openhcs.agent.dto.execution import ExecutionConnectionSpec, RuntimeServerInfo
from openhcs.agent.dto.functions import FunctionCatalogEntry, FunctionCatalogPage
from openhcs.agent.dto.ui_bridge import UiWindowSnapshotResult
from openhcs.pyqt_gui.config import AgentUiBridgeConfig, UIConfig
from openhcs.runtime.zmq_config import OpenHCSZMQConfig
from scripts.smoke_installed_gui import (
    InstalledExecutionServerLogEvidence,
    InstalledGuiSmokeTiming,
    InstalledGuiSnapshotEvidenceAuthority,
    assert_not_source_checkout_import,
    verify_installed_function_catalog,
    with_isolated_runtime_topology,
)


def _catalog_page(*entries: FunctionCatalogEntry) -> FunctionCatalogPage:
    return FunctionCatalogPage(
        schema_version="test",
        revision="catalog-revision",
        items=entries,
        total=len(entries),
        limit=len(entries),
    )


def _catalog_entry() -> FunctionCatalogEntry:
    return FunctionCatalogEntry(
        function_id="test:function",
        import_path="test.module.function",
        name="function",
        module="test.module",
        library="test",
        signature="function()",
        summary="test function",
    )


class _FunctionCatalogService:
    def __init__(self, page: FunctionCatalogPage) -> None:
        self.page = page
        self.resolved_function_ids: list[str] = []

    def prepare(self) -> Future[FunctionCatalogPage]:
        future: Future[FunctionCatalogPage] = Future()
        future.set_result(self.page)
        return future

    def resolve(self, function_id: str):
        self.resolved_function_ids.append(function_id)
        return lambda: None


def test_installed_gui_smoke_timing_derives_process_ceiling() -> None:
    timing = InstalledGuiSmokeTiming(
        operation_timeout_seconds=120.0,
        process_exit_grace_seconds=15.0,
    )

    assert timing.subprocess_timeout_seconds == 135.0


@pytest.mark.parametrize(
    ("operation_timeout_seconds", "process_exit_grace_seconds"),
    ((0.0, 15.0), (120.0, 0.0)),
)
def test_installed_gui_smoke_timing_requires_positive_budgets(
    operation_timeout_seconds: float,
    process_exit_grace_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        InstalledGuiSmokeTiming(
            operation_timeout_seconds=operation_timeout_seconds,
            process_exit_grace_seconds=process_exit_grace_seconds,
        )


def test_gui_smoke_rejects_source_package(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"

    with pytest.raises(AssertionError, match="source checkout instead of the wheel"):
        assert_not_source_checkout_import(
            package_path=checkout / "openhcs" / "__init__.py",
            forbidden_root=checkout,
        )


def test_gui_smoke_allows_wheel_venv_inside_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    site_packages = checkout / "test_gui" / "lib" / "site-packages"

    assert_not_source_checkout_import(
        package_path=site_packages / "openhcs" / "__init__.py",
        forbidden_root=checkout,
    )


def test_gui_smoke_requires_prepared_catalog_and_resolvable_reference() -> None:
    entry = _catalog_entry()
    service = _FunctionCatalogService(_catalog_page(entry))

    result = verify_installed_function_catalog(
        service,
        deadline=OperationDeadline.after_milliseconds(
            2_000,
            operation="test installed function catalog",
        ),
    )

    assert result.resolved_function_id == entry.function_id
    assert result.resolved_import_path == entry.import_path
    assert result.revision == "catalog-revision"
    assert result.total == 1
    assert service.resolved_function_ids == [entry.function_id]


def test_gui_smoke_reports_its_operation_when_catalog_wait_expires() -> None:
    class PendingFunctionCatalogService:
        @staticmethod
        def prepare() -> Future[FunctionCatalogPage]:
            return Future()

    with pytest.raises(
        OperationTimeoutError,
        match="Timed out waiting for test installed function catalog",
    ):
        verify_installed_function_catalog(
            PendingFunctionCatalogService(),
            deadline=OperationDeadline.after_milliseconds(
                1,
                operation="test installed function catalog",
            ),
        )


def test_gui_smoke_preserves_catalog_owner_timeout() -> None:
    owner_error = TimeoutError("catalog owner timed out")

    class FailedFunctionCatalogService:
        @staticmethod
        def prepare() -> Future[FunctionCatalogPage]:
            future: Future[FunctionCatalogPage] = Future()
            future.set_exception(owner_error)
            return future

    with pytest.raises(TimeoutError, match="catalog owner timed out") as caught:
        verify_installed_function_catalog(
            FailedFunctionCatalogService(),
            deadline=OperationDeadline.after_milliseconds(
                2_000,
                operation="test installed function catalog",
            ),
        )

    assert caught.value is owner_error


def test_gui_smoke_allocates_topology_through_transport_authority(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_transport = OpenHCSZMQConfig(
        default_port=8123,
        transport_mode=TransportMode.TCP,
    )
    original_bridge = AgentUiBridgeConfig(
        port=8223,
        transport_mode=TransportMode.TCP,
    )
    ui_config = UIConfig(zmq=original_transport, agent_bridge=original_bridge)
    observed: list[dict[str, object]] = []
    allocated_pairs = iter(
        (
            DataControlPortPair(data_port=8124, control_port=9124),
            DataControlPortPair(data_port=8224, control_port=9224),
        )
    )

    def acquire(config, *, transport_mode, excluded=(), host):
        observed.append(
            {
                "config": config,
                "transport_mode": transport_mode,
                "excluded": excluded,
                "host": host,
            }
        )
        return next(allocated_pairs)

    monkeypatch.setattr(
        DataControlPortPairAuthority,
        "acquire",
        staticmethod(acquire),
    )

    descriptor_path = tmp_path / "ui_bridge.json"
    isolated = with_isolated_runtime_topology(
        ui_config,
        descriptor_path=descriptor_path,
    )

    assert isolated.zmq == OpenHCSZMQConfig(
        default_port=8124,
        transport_mode=TransportMode.TCP,
    )
    assert isolated.agent_bridge == AgentUiBridgeConfig(
        port=8224,
        transport_mode=TransportMode.TCP,
        descriptor_file_path=descriptor_path,
    )
    assert ui_config.zmq is original_transport
    assert ui_config.agent_bridge is original_bridge
    assert observed == [
        {
            "config": original_transport,
            "transport_mode": TransportMode.TCP,
            "excluded": (),
            "host": original_transport.client_host,
        },
        {
            "config": OpenHCSZMQConfig(
                default_port=8223,
                transport_mode=TransportMode.TCP,
            ),
            "transport_mode": TransportMode.TCP,
            "excluded": (8124, 9124),
            "host": original_bridge.host,
        },
    ]


def test_gui_smoke_validates_emitted_snapshot_content(tmp_path: Path) -> None:
    source_directory = tmp_path / "agent-output"
    source_directory.mkdir()
    snapshot_path = source_directory / "main_window.png"
    evidence_directory = tmp_path / "ci-evidence"
    image_bytes = b"native screenshot bytes"
    snapshot_path.write_bytes(image_bytes)
    digest = hashlib.sha256(image_bytes).hexdigest()

    retained_result = InstalledGuiSnapshotEvidenceAuthority.retain(
        UiWindowSnapshotResult(
            schema_version="test",
            output_dir_path=str(source_directory),
            capture_scope=WindowSnapshotCaptureScope.WINDOW,
            window_id="main_window",
            captured=True,
            resource=AgentResourceRef(
                uri=snapshot_path.as_uri(),
                title="OpenHCS",
                mime_type="image/png",
                path=str(snapshot_path),
                size_bytes=len(image_bytes),
                sha256=digest,
            ),
            width=640,
            height=480,
        ),
        evidence_directory=evidence_directory,
    )

    assert retained_result.window_id == "main_window"
    assert retained_result.output_dir_path == str(evidence_directory)
    assert retained_result.resource is not None
    assert retained_result.resource.path == str(evidence_directory / snapshot_path.name)
    assert Path(retained_result.resource.path).read_bytes() == image_bytes
    assert retained_result.resource.sha256 == digest
    assert retained_result.resource.size_bytes == len(image_bytes)
    assert retained_result.width == 640
    assert retained_result.height == 480
    assert retained_result.capture_scope is WindowSnapshotCaptureScope.WINDOW


def test_gui_smoke_validates_advertised_execution_server_log(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "execution.log"
    content = b"execution server ready\n"
    log_path.write_bytes(content)

    evidence = InstalledExecutionServerLogEvidence.from_runtime_server_info(
        RuntimeServerInfo(
            schema_version="test",
            connection=ExecutionConnectionSpec(),
            reachable=True,
            ready=True,
            log_file_path=str(log_path),
        )
    )

    assert evidence.path == str(log_path)
    assert evidence.observed_size_bytes == len(content)


def test_gui_smoke_rejects_empty_advertised_execution_server_log(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "execution.log"
    log_path.touch()

    with pytest.raises(AssertionError, match="execution server log is empty"):
        InstalledExecutionServerLogEvidence.from_runtime_server_info(
            RuntimeServerInfo(
                schema_version="test",
                connection=ExecutionConnectionSpec(),
                reachable=True,
                ready=True,
                log_file_path=str(log_path),
            )
        )


def test_window_snapshot_cli_projects_declared_output_directory(tmp_path: Path) -> None:
    from openhcs.mcp import dev_client

    parser = dev_client._build_parser()
    arguments = parser.parse_args(
        (
            "window-snapshot",
            "main_window",
            "--output-dir-path",
            str(tmp_path),
        )
    )

    call = dev_client._calls_from_args(arguments)[0]

    assert call.name == "openhcs_ui_snapshot_window"
    assert call.arguments["output_dir_path"] == str(tmp_path)
