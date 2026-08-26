"""Operate an installed OpenHCS GUI through MCP outside the source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from importlib.metadata import distribution
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, get_type_hints

if TYPE_CHECKING:
    from zmqruntime import OperationDeadline

    from openhcs.agent.dto.ui_bridge import UiWindowSnapshotResult
    from openhcs.agent.services.endpoint_function_catalog_service import (
        ZMQFunctionCatalogService,
    )
    from openhcs.mcp.dev_client import McpDevClient, McpDevCommandExecution
    from openhcs.pyqt_gui.config import UIConfig

JsonValue = str | int | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
AgentDtoT = TypeVar("AgentDtoT")


@dataclass(frozen=True, slots=True)
class InstalledGuiSmokeTiming:
    """One safety budget for the complete installed-GUI acceptance process."""

    operation_timeout_seconds: float = 180.0
    process_exit_grace_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.operation_timeout_seconds <= 0:
            raise ValueError("Installed GUI smoke timeout must be positive.")
        if self.process_exit_grace_seconds <= 0:
            raise ValueError("Installed GUI smoke exit grace must be positive.")

    @property
    def subprocess_timeout_seconds(self) -> float:
        """Allow the owned operation deadline plus bounded teardown time."""

        return self.operation_timeout_seconds + self.process_exit_grace_seconds


INSTALLED_GUI_SMOKE_TIMING = InstalledGuiSmokeTiming()


class InstalledGuiSnapshotEvidenceAuthority:
    """Retain capability-owned native screenshot evidence for CI."""

    @staticmethod
    def _require_content_identity(
        path: Path,
        *,
        sha256: str,
        size_bytes: int,
    ) -> None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise AssertionError(
                "Installed GUI snapshot SHA-256 does not match its file."
            )
        if path.stat().st_size != size_bytes:
            raise AssertionError("Installed GUI snapshot size does not match its file.")

    @classmethod
    def retain(
        cls,
        result: UiWindowSnapshotResult,
        *,
        evidence_directory: Path,
    ) -> UiWindowSnapshotResult:
        """Validate and retain one capability result without mirroring its fields."""

        if not result.captured or result.errors:
            raise AssertionError(f"Installed GUI snapshot failed: {result}")
        resource = result.resource
        if resource is None or resource.path is None:
            raise AssertionError("Installed GUI snapshot omitted its resource path.")
        if resource.sha256 is None or resource.size_bytes is None:
            raise AssertionError("Installed GUI snapshot omitted content identity.")
        if resource.mime_type != "image/png":
            raise AssertionError(
                f"Installed GUI snapshot has unexpected MIME type: {resource.mime_type}"
            )
        if result.width is None or result.width <= 0:
            raise AssertionError("Installed GUI snapshot omitted a valid width.")
        if result.height is None or result.height <= 0:
            raise AssertionError("Installed GUI snapshot omitted a valid height.")

        source_path = Path(resource.path).resolve()
        evidence_root = evidence_directory.resolve()
        if source_path.suffix.lower() != ".png" or not source_path.is_file():
            raise AssertionError(
                f"Installed GUI snapshot is not a readable PNG: {source_path}"
            )
        cls._require_content_identity(
            source_path,
            sha256=resource.sha256,
            size_bytes=resource.size_bytes,
        )

        evidence_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = evidence_root / source_path.name
        if source_path != snapshot_path:
            shutil.copy2(source_path, snapshot_path)
        cls._require_content_identity(
            snapshot_path,
            sha256=resource.sha256,
            size_bytes=resource.size_bytes,
        )
        return replace(
            result,
            output_dir_path=str(evidence_root),
            resource=replace(
                resource,
                uri=snapshot_path.as_uri(),
                path=str(snapshot_path),
            ),
        )


@dataclass(frozen=True, slots=True)
class InstalledFunctionCatalogSmokeResult:
    """Evidence that the GUI-prewarmed endpoint catalogue is usable."""

    resolved_function_id: str
    resolved_import_path: str
    revision: str
    total: int


@dataclass(frozen=True, slots=True)
class InstalledLiveMcpSmokeResult:
    """Typed evidence from the packaged MCP operating the live packaged GUI."""

    action_status: str
    bridge_count: int
    bridge_port: int
    bridge_reachable: bool
    bridge_transport: str
    function_catalog: InstalledFunctionCatalogSmokeResult
    health_status: str
    mcp_server_source_path: str
    mcp_server_version: str
    snapshots: tuple[UiWindowSnapshotResult, ...]
    window_count_after_action: int
    window_count_before_action: int


@dataclass(frozen=True, slots=True)
class InstalledGuiSmokeResult:
    """Validated installed-GUI and packaged-MCP lifecycle evidence."""

    execution_port: int
    execution_transport: str
    exit_code: int
    openhcs_version: str
    package_path: str
    qt_platform: str
    ready: bool
    startup_error: str | None
    timed_out: bool
    visible: bool
    live_mcp: InstalledLiveMcpSmokeResult

    def payload(self, *, working_directory: Path) -> dict[str, JsonValue]:
        """Project the typed result into the script's JSON boundary."""

        return {
            **asdict(self),
            "working_directory": str(working_directory),
        }


@dataclass(slots=True)
class _StartupObservation:
    """Mutable callback state finalized into one validated result."""

    ready: bool = False
    visible: bool = False
    startup_error: str | None = None
    timed_out: bool = False
    live_mcp: InstalledLiveMcpSmokeResult | None = None

    def record_ready(self, *, visible: bool) -> None:
        self.ready = True
        self.visible = visible

    def record_failure(self, error: BaseException) -> None:
        self.startup_error = f"{type(error).__name__}: {error}"

    def record_live_mcp(self, result: InstalledLiveMcpSmokeResult) -> None:
        self.live_mcp = result

    def record_timeout(self) -> None:
        self.timed_out = True

    def result(
        self,
        *,
        execution_port: int,
        execution_transport: str,
        exit_code: int,
        openhcs_version: str,
        package_path: Path,
        qt_platform: str,
        timeout_seconds: float,
    ) -> InstalledGuiSmokeResult:
        """Validate the observed lifecycle and freeze its evidence."""

        if self.startup_error is not None:
            raise AssertionError(
                f"Installed GUI/MCP smoke failed: {self.startup_error}"
            )
        if self.timed_out:
            raise TimeoutError(
                "Installed GUI/MCP smoke did not complete within "
                f"{timeout_seconds:g} seconds."
            )
        if not self.ready or not self.visible:
            raise AssertionError(
                "Installed GUI did not become visibly ready: "
                f"ready={self.ready}, visible={self.visible}"
            )
        if self.live_mcp is None:
            raise AssertionError("Installed MCP did not operate the live GUI.")
        if exit_code != 0:
            raise AssertionError(f"Installed GUI exited with {exit_code}.")
        return InstalledGuiSmokeResult(
            execution_port=execution_port,
            execution_transport=execution_transport,
            exit_code=exit_code,
            openhcs_version=openhcs_version,
            package_path=str(package_path),
            qt_platform=qt_platform,
            ready=self.ready,
            startup_error=self.startup_error,
            timed_out=self.timed_out,
            visible=self.visible,
            live_mcp=self.live_mcp,
        )


def assert_not_source_checkout_import(
    *,
    package_path: Path,
    forbidden_root: Path,
) -> None:
    """Reject a source-owned import without rejecting an in-tree wheel venv."""

    if package_path.is_relative_to(forbidden_root / "openhcs"):
        raise AssertionError(
            "Smoke test imported the source checkout instead of the wheel: "
            f"{package_path}"
        )


def with_isolated_runtime_topology(
    ui_config: UIConfig,
    *,
    descriptor_path: Path,
) -> UIConfig:
    """Allocate execution and bridge endpoints through their transport owner."""

    from zmqruntime import DataControlPortPairAuthority, resolve_transport_mode

    transport_config = ui_config.zmq
    execution_pair = DataControlPortPairAuthority.acquire(
        transport_config,
        transport_mode=transport_config.transport_mode,
        host=transport_config.client_host,
    )
    isolated_transport_config = replace(
        transport_config,
        default_port=execution_pair.data_port,
    )

    bridge_config = ui_config.agent_bridge
    bridge_transport_config = replace(
        isolated_transport_config,
        default_port=bridge_config.require_port("Installed GUI bridge smoke"),
    )
    bridge_mode = resolve_transport_mode(bridge_config.transport_mode)
    bridge_pair = DataControlPortPairAuthority.acquire(
        bridge_transport_config,
        transport_mode=bridge_mode,
        excluded=(execution_pair.data_port, execution_pair.control_port),
        host=bridge_config.host,
    )
    return replace(
        ui_config,
        zmq=isolated_transport_config,
        agent_bridge=replace(
            bridge_config,
            enabled=True,
            port=bridge_pair.data_port,
            transport_mode=bridge_mode,
            descriptor_file_path=descriptor_path,
        ),
    )


def _tool_payload(
    execution: McpDevCommandExecution,
    *,
    tool_name: str,
) -> dict[str, JsonValue]:
    """Require one successful MCP result and return its JSON object payload."""

    from openhcs.mcp.dev_client_rendering import McpDevPayloadProjection

    if execution.returncode != 0:
        diagnostics = execution.rendered_output
        if execution.server_stderr_tail:
            diagnostics += f"\nMCP stderr:\n{execution.server_stderr_tail}"
        raise AssertionError(diagnostics)
    payload = McpDevPayloadProjection.tool_payload(execution.payload, tool_name)
    if payload is None:
        raise AssertionError(f"MCP response omitted {tool_name!r} payload.")
    return dict(payload)


def _typed_tool_payload(
    execution: McpDevCommandExecution,
    *,
    tool_name: str,
    payload_type: type[AgentDtoT],
) -> AgentDtoT:
    """Hydrate one MCP result through the declared agent DTO contract."""

    from openhcs.agent.services.ui_bridge_transport import AgentDtoJsonCodec

    payload = _tool_payload(execution, tool_name=tool_name)
    return AgentDtoJsonCodec.dataclass_from_json(payload_type, dict(payload))


def _declared_catalog_items(
    execution: McpDevCommandExecution,
    *,
    tool_name: str,
    catalog_type: type,
    item_type: type,
) -> tuple[Mapping[str, JsonValue], ...]:
    """Read a flattened MCP catalog through its declared tuple field."""

    type_hints = get_type_hints(catalog_type)
    item_fields = tuple(
        catalog_field
        for catalog_field in fields(catalog_type)
        if type_hints.get(catalog_field.name) == tuple[item_type, ...]
    )
    if len(item_fields) != 1:
        raise TypeError(
            f"{catalog_type.__name__} must declare exactly one {item_type.__name__} "
            "tuple field."
        )
    payload = _tool_payload(execution, tool_name=tool_name)
    items = payload.get(item_fields[0].name)
    if not isinstance(items, list) or not all(
        isinstance(item, Mapping) for item in items
    ):
        raise AssertionError(
            f"MCP {catalog_type.__name__} payload has invalid catalog items."
        )
    return tuple(item for item in items if isinstance(item, Mapping))


def _declared_identity_values(
    items: tuple[Mapping[str, JsonValue], ...],
    *,
    identity_type: type,
) -> tuple[str, ...]:
    """Read one-field flattened identities through their nominal declaration."""

    identity_fields = fields(identity_type)
    if len(identity_fields) != 1:
        raise TypeError(
            f"{identity_type.__name__} must declare exactly one identity field."
        )
    field_name = identity_fields[0].name
    values = tuple(item.get(field_name) for item in items)
    if not all(isinstance(value, str) for value in values):
        raise AssertionError(
            f"MCP {identity_type.__name__} payload has invalid identity values."
        )
    return tuple(value for value in values if isinstance(value, str))


def _capture_installed_gui_snapshot(
    client: McpDevClient,
    *,
    descriptor_path: Path,
    evidence_directory: Path,
    timeout_seconds: float,
    window_id: str,
) -> UiWindowSnapshotResult:
    """Capture one live window through the production MCP capability."""

    from pyqt_reactive.services.window_snapshot import WindowSnapshotCaptureScope

    from openhcs.agent.capabilities import UiSnapshotWindowCapability
    from openhcs.agent.dto.ui_bridge import UiWindowSnapshotResult
    from openhcs.mcp.dev_client_commands.ui import WindowSnapshotCommandSpec

    execution = client.execute(
        (
            WindowSnapshotCommandSpec.command,
            window_id,
            "--capture-scope",
            WindowSnapshotCaptureScope.WINDOW.value,
            "--descriptor-file-path",
            str(descriptor_path),
            "--json",
        ),
        timeout_seconds=timeout_seconds,
    )
    result = _typed_tool_payload(
        execution,
        tool_name=UiSnapshotWindowCapability.name,
        payload_type=UiWindowSnapshotResult,
    )
    return InstalledGuiSnapshotEvidenceAuthority.retain(
        result,
        evidence_directory=evidence_directory,
    )


def verify_installed_function_catalog(
    service: ZMQFunctionCatalogService,
    *,
    deadline: OperationDeadline,
) -> InstalledFunctionCatalogSmokeResult:
    """Require the GUI's background endpoint catalogue and one usable reference."""

    future = service.prepare()
    try:
        page = future.result(timeout=deadline.remaining_seconds())
    except TimeoutError as error:
        if not future.cancelled() and future.done() and future.exception() is error:
            raise
        raise deadline.timeout_error() from error
    if not page.revision.strip():
        raise AssertionError("Installed function catalog omitted its revision.")
    if page.total <= 0 or not page.items:
        raise AssertionError("Installed function catalog is empty.")
    if page.total != len(page.items):
        raise AssertionError(
            "Installed function catalog read was incomplete: "
            f"total={page.total} items={len(page.items)}"
        )

    entry = page.items[0]
    resolved = service.resolve(entry.function_id)
    if not callable(resolved):
        raise TypeError(
            "Installed function catalog returned a non-callable reference: "
            f"{entry.function_id}"
        )
    return InstalledFunctionCatalogSmokeResult(
        resolved_function_id=entry.function_id,
        resolved_import_path=entry.import_path,
        revision=page.revision,
        total=page.total,
    )


def run_installed_live_mcp_smoke(
    *,
    descriptor_path: Path,
    evidence_directory: Path,
    forbidden_root: Path,
    function_catalog_service: ZMQFunctionCatalogService,
    deadline: OperationDeadline,
) -> InstalledLiveMcpSmokeResult:
    """Operate one live packaged GUI through one packaged desktop MCP session."""

    from openhcs.agent.capabilities import (
        DesktopLocalCapabilitySurfaceProfile,
        agent_capabilities,
    )
    from openhcs.agent.dto.mcp import McpServerHealthResult
    from openhcs.agent.dto.ui_bridge import (
        UiActionInvocationStatus,
        UiActionInvokeResult,
        UiBridgeCatalog,
        UiBridgeStatus,
        UiWindowCatalog,
        UiWindowIdentity,
        UiWindowSummary,
    )
    from openhcs.agent.ui_bridge_actions import PlateManagerAction
    from openhcs.agent.ui_bridge_identities import (
        MainWindowWidgetIdentity,
        PlateManagerWidgetIdentity,
    )
    from openhcs.mcp.dev_client import McpDevClient
    from openhcs.mcp.dev_client_commanding import CapabilityBackedCommandSpec
    from openhcs.mcp.dev_client_commands.ui import (
        InvokeActionCommandSpec,
        UiSmokeCommandSpec,
    )

    function_catalog = verify_installed_function_catalog(
        function_catalog_service,
        deadline=deadline,
    )
    windows_command = CapabilityBackedCommandSpec.for_capability_name(
        agent_capabilities.ui_list_windows.name
    )
    if windows_command is None:
        raise AssertionError("The desktop MCP surface has no window-list command.")

    with McpDevClient(
        sys.executable,
        surface_profile=DesktopLocalCapabilitySurfaceProfile(),
        initialize_timeout_seconds=deadline.remaining_seconds(),
    ) as client:
        timeout_text = f"{deadline.remaining_seconds():g}"
        smoke_execution = client.execute(
            (UiSmokeCommandSpec.command, "--timeout-seconds", timeout_text),
            timeout_seconds=deadline.remaining_seconds(),
        )
        health = _typed_tool_payload(
            smoke_execution,
            tool_name=agent_capabilities.health_check.name,
            payload_type=McpServerHealthResult,
        )
        bridge_status = _typed_tool_payload(
            smoke_execution,
            tool_name=agent_capabilities.ui_bridge_status.name,
            payload_type=UiBridgeStatus,
        )
        bridge_catalog = _typed_tool_payload(
            smoke_execution,
            tool_name=agent_capabilities.ui_list_bridges.name,
            payload_type=UiBridgeCatalog,
        )
        windows_before = _declared_catalog_items(
            smoke_execution,
            tool_name=agent_capabilities.ui_list_windows.name,
            catalog_type=UiWindowCatalog,
            item_type=UiWindowSummary,
        )

        installed_version = distribution("openhcs").version
        if health.status != "ok" or not health.packaged_resources_ready:
            raise AssertionError(f"Installed MCP health is incomplete: {health}")
        if health.openhcs_version != installed_version:
            raise AssertionError(
                "Installed GUI and MCP versions diverged: "
                f"gui={installed_version} mcp={health.openhcs_version}"
            )
        server_source_path = Path(health.server_source_path).resolve()
        assert_not_source_checkout_import(
            package_path=server_source_path,
            forbidden_root=forbidden_root.resolve(),
        )
        if not bridge_status.reachable:
            raise AssertionError(f"Installed UI bridge is unreachable: {bridge_status}")
        if Path(bridge_status.descriptor_file_path or "").resolve() != descriptor_path:
            raise AssertionError(
                "Installed MCP selected a different UI bridge descriptor: "
                f"{bridge_status.descriptor_file_path!r}"
            )
        if (
            len(bridge_catalog.bridges) != 1
            or Path(bridge_catalog.bridges[0].descriptor_file_path or "").resolve()
            != descriptor_path
        ):
            raise AssertionError(
                "Installed MCP did not discover exactly its declared live bridge: "
                f"{bridge_catalog}"
            )

        required_window_ids = {
            MainWindowWidgetIdentity.require_value(),
            PlateManagerWidgetIdentity.require_value(),
        }
        observed_window_ids = set(
            _declared_identity_values(
                windows_before,
                identity_type=UiWindowIdentity,
            )
        )
        if not required_window_ids <= observed_window_ids:
            raise AssertionError(
                "Installed MCP did not discover the required live windows: "
                f"missing={sorted(required_window_ids - observed_window_ids)}"
            )

        timeout_text = f"{deadline.remaining_seconds():g}"
        action_execution = client.execute(
            (
                InvokeActionCommandSpec.command,
                PlateManagerWidgetIdentity.require_value(),
                PlateManagerAction.CODE_PLATE.value,
                "--request-token",
                "installed-gui-mcp-smoke",
                "--timeout-seconds",
                timeout_text,
            ),
            timeout_seconds=deadline.remaining_seconds(),
        )
        action_result = _typed_tool_payload(
            action_execution,
            tool_name=agent_capabilities.ui_invoke_action.name,
            payload_type=UiActionInvokeResult,
        )
        if action_result.status != UiActionInvocationStatus.ACCEPTED.value:
            raise AssertionError(
                f"Installed UI action was not accepted: {action_result}"
            )

        timeout_text = f"{deadline.remaining_seconds():g}"
        windows_execution = client.execute(
            (windows_command.command, "--timeout-seconds", timeout_text),
            timeout_seconds=deadline.remaining_seconds(),
        )
        windows_after = _declared_catalog_items(
            windows_execution,
            tool_name=agent_capabilities.ui_list_windows.name,
            catalog_type=UiWindowCatalog,
            item_type=UiWindowSummary,
        )
        if len(windows_after) <= len(windows_before):
            raise AssertionError(
                "The declared code action did not expose its new live window: "
                f"before={len(windows_before)} after={len(windows_after)}"
            )

        window_ids_before = frozenset(
            _declared_identity_values(
                windows_before,
                identity_type=UiWindowIdentity,
            )
        )
        window_ids_after = frozenset(
            _declared_identity_values(
                windows_after,
                identity_type=UiWindowIdentity,
            )
        )
        new_window_ids = window_ids_after - window_ids_before
        if not new_window_ids:
            raise AssertionError("Installed UI action exposed no new window identity.")
        evidence_directory.mkdir(parents=True, exist_ok=True)
        snapshots = tuple(
            _capture_installed_gui_snapshot(
                client,
                descriptor_path=descriptor_path,
                evidence_directory=evidence_directory,
                timeout_seconds=deadline.remaining_seconds(),
                window_id=window_id,
            )
            for window_id in (
                MainWindowWidgetIdentity.require_value(),
                *sorted(new_window_ids),
            )
        )

    return InstalledLiveMcpSmokeResult(
        action_status=action_result.status,
        bridge_count=len(bridge_catalog.bridges),
        bridge_port=bridge_status.connection.require_port("Installed UI bridge smoke"),
        bridge_reachable=bridge_status.reachable,
        bridge_transport=(
            bridge_status.connection.transport_endpoint().transport_mode.value
        ),
        function_catalog=function_catalog,
        health_status=health.status,
        mcp_server_source_path=str(server_source_path),
        mcp_server_version=health.openhcs_version,
        snapshots=snapshots,
        window_count_after_action=len(windows_after),
        window_count_before_action=len(windows_before),
    )


def shutdown_isolated_execution_endpoint(ui_config: UIConfig) -> None:
    """Stop only the execution endpoint allocated for this smoke process."""

    from zmqruntime import EndpointShutdownMode
    from zmqruntime.shutdown import EndpointShutdownService
    from zmqruntime.transport import TransportEndpoint, resolve_transport_mode

    transport_config = ui_config.zmq
    endpoint = TransportEndpoint(
        host=transport_config.client_host,
        port=transport_config.default_port,
        transport_mode=resolve_transport_mode(transport_config.transport_mode),
    )
    outcome = EndpointShutdownService.for_endpoint(
        transport_config,
        endpoint,
    ).shutdown_ports(
        ports=[endpoint.port],
        mode=EndpointShutdownMode.FORCE,
    )
    if not outcome.succeeded or endpoint.port not in outcome.terminated_ports:
        raise AssertionError(outcome.failure_message)


def run_installed_gui_smoke(
    *,
    evidence_directory: Path,
    forbidden_root: Path,
    timeout_seconds: float,
) -> InstalledGuiSmokeResult:
    """Exercise installed GUI readiness, packaged MCP control, and shutdown."""

    if timeout_seconds <= 0:
        raise ValueError("GUI startup smoke timeout must be positive.")
    os.environ["OPENHCS_CPU_ONLY"] = "true"

    from PyQt6.QtCore import QObject, QTimer, pyqtSignal
    from zmqruntime import OperationDeadline

    import openhcs
    from openhcs.agent.ui_bridge_environment import UiBridgeDescriptorEnvironment
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp
    from openhcs.pyqt_gui.config import (
        PyQtGuiRuntimeContext,
        UIConfig,
    )

    package_path = Path(openhcs.__file__).resolve()
    assert_not_source_checkout_import(
        package_path=package_path,
        forbidden_root=forbidden_root.resolve(),
    )

    descriptor_path = (Path.cwd() / "ui_bridge.json").resolve()
    ui_config = with_isolated_runtime_topology(
        UIConfig(check_for_updates_on_startup=False),
        descriptor_path=descriptor_path,
    )
    os.environ[UiBridgeDescriptorEnvironment.descriptor_file_path_key] = str(
        descriptor_path
    )
    runtime_context = PyQtGuiRuntimeContext(
        ui_config=ui_config,
        pipeline_runtime=GlobalPipelineConfig(),
    )
    application = OpenHCSPyQtApp(
        ["openhcs-gui-installed-smoke", "--no-gpu"],
        runtime_context=runtime_context,
    )
    deadline = OperationDeadline.after_milliseconds(
        round(timeout_seconds * 1_000),
        operation="installed GUI and packaged MCP smoke",
    )
    observation = _StartupObservation()
    worker_thread: threading.Thread | None = None

    class McpSmokeCompletion(QObject):
        completed = pyqtSignal(object)

    completion = McpSmokeCompletion()

    def close_application() -> None:
        main_window = application.main_window
        if main_window is None:
            application.exit(1)
            return
        main_window.close()

    def startup_ready() -> None:
        nonlocal worker_thread
        main_window = application.main_window
        observation.record_ready(
            visible=bool(main_window is not None and main_window.isVisible())
        )

        def run_live_mcp() -> None:
            try:
                result: InstalledLiveMcpSmokeResult | Exception = (
                    run_installed_live_mcp_smoke(
                        descriptor_path=descriptor_path,
                        evidence_directory=evidence_directory,
                        forbidden_root=forbidden_root,
                        function_catalog_service=application.function_catalog_service,
                        deadline=deadline,
                    )
                )
            except (
                Exception  # noqa: BLE001 - thread boundary reports failures
            ) as error:
                result = error
            completion.completed.emit(result)

        worker_thread = threading.Thread(
            target=run_live_mcp,
            name="installed-live-mcp-smoke",
        )
        worker_thread.start()

    def startup_failed(error: BaseException) -> None:
        observation.record_failure(error)
        close_application()

    def live_mcp_completed(result: object) -> None:
        if isinstance(result, BaseException):
            observation.record_failure(result)
        elif isinstance(result, InstalledLiveMcpSmokeResult):
            observation.record_live_mcp(result)
        else:
            observation.record_failure(
                TypeError(f"Unexpected MCP smoke result: {type(result).__name__}")
            )
        close_application()

    def startup_timed_out() -> None:
        if observation.live_mcp is not None or observation.startup_error is not None:
            return
        observation.record_timeout()
        close_application()

    completion.completed.connect(live_mcp_completed)
    QTimer.singleShot(deadline.timeout_ms, startup_timed_out)
    try:
        exit_code = application.run(
            on_main_window_ready=startup_ready,
            on_startup_failure=startup_failed,
        )
    finally:
        if worker_thread is not None:
            worker_thread.join(timeout=deadline.remaining_seconds_or_zero())
            if worker_thread.is_alive():
                observation.record_timeout()
        shutdown_isolated_execution_endpoint(ui_config)
    if descriptor_path.exists():
        raise AssertionError(
            f"Installed UI bridge descriptor survived shutdown: {descriptor_path}"
        )
    return observation.result(
        execution_port=ui_config.zmq.default_port,
        execution_transport=ui_config.zmq.transport_mode.value,
        exit_code=exit_code,
        openhcs_version=distribution("openhcs").version,
        package_path=package_path,
        qt_platform=application.platformName(),
        timeout_seconds=timeout_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forbid-import-root",
        type=Path,
        required=True,
        help="Source checkout that must not own the imported openhcs package.",
    )
    parser.add_argument(
        "--evidence-directory",
        type=Path,
        required=True,
        help="Directory that receives validated native GUI screenshot evidence.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=INSTALLED_GUI_SMOKE_TIMING.operation_timeout_seconds,
        help="Maximum time for painted GUI readiness and live MCP operation.",
    )
    args = parser.parse_args(argv)
    original_working_directory = Path.cwd()
    evidence_directory = args.evidence_directory.resolve()
    with tempfile.TemporaryDirectory(prefix="openhcs-installed-gui-") as directory:
        working_directory = Path(directory).resolve()
        os.chdir(working_directory)
        try:
            result = run_installed_gui_smoke(
                evidence_directory=evidence_directory,
                forbidden_root=args.forbid_import_root,
                timeout_seconds=args.timeout_seconds,
            )
            payload = result.payload(working_directory=working_directory)
        finally:
            os.chdir(original_working_directory)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
