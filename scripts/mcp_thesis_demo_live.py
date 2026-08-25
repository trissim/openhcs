#!/usr/bin/env python3
"""Run the no-fallback MCP thesis live-demo rehearsal.

This script intentionally drives the real PyQt UI bridge and the real ZMQ
compiler/executor path. It is not a headless substitute for the demo.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# Importing the package activates the pinned source-checkout externals before
# this direct script entrypoint imports their public modules.
import openhcs  # noqa: F401

# isort: split

from pyqt_reactive.dialogs.group_by_selector_dialog import GroupBySelectorDialog
from pyqt_reactive.services.function_navigation import FunctionPatternField
from pyqt_reactive.widgets.shared import DetachableActionBar, ManagedWindowAction
from zmqruntime import (
    EndpointShutdownMode,
    TransportEndpoint,
    TransportMode,
    ZMQClient,
    ZMQConfig,
)
from zmqruntime.transport import resolve_transport_mode

from openhcs.agent.capabilities import (
    DescribeConfigSchemaCapability,
    GetAuthoringContextCapability,
    InspectPipelineSourceArtifactPlanCapability,
    InspectPlatePathCapability,
    SamplePlateImageCapability,
    SampleViewerWindowImageCapability,
    UiCloseWindowCapability,
    UiFocusWindowCapability,
    UiGetStateSurfaceCapability,
    UiGetWidgetTreeCapability,
    UiInspectSelectedPlateImagesCapability,
    UiInvokeActionCapability,
    UiInvokeWidgetActionCapability,
    UiListActionsCapability,
    UiListWindowsCapability,
    UiNavigateWindowCapability,
    UiSnapshotWindowCapability,
    ValidateViewerWindowStateCapability,
)
from openhcs.agent.ui_bridge_actions import PlateManagerAction
from openhcs.agent.ui_bridge_identities import (
    ManagedWindowWidgetIdentity,
    PipelineEditorStateSurfaceIdentityDeclaration,
    PlateManagerOrchestratorCodeDocumentIdentity,
    PlateManagerStateSurfaceIdentityDeclaration,
    PlateManagerWidgetIdentity,
)
from openhcs.constants.constants import AllComponents
from openhcs.core.config_cache import ConfigCacheSpec, save_config_sync
from openhcs.mcp.dev_client import McpDevClient
from openhcs.mcp.dev_client_core import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    MCP_TOOL_TIMEOUT_MARGIN_SECONDS,
    mcp_tool_timeout_seconds,
)
from openhcs.pyqt_gui.config import (
    AgentUiBridgeConfig,
    UIConfig,
    UIConfigCacheEnvironment,
    get_default_ui_config,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.widgets.artifact_plan_view import ArtifactPlanViewWidget
from openhcs.pyqt_gui.widgets.image_browser import ImageBrowserWidget
from openhcs.pyqt_gui.windows.live_measurements_window import LiveMeasurementsWindow
from openhcs.runtime.viewer_protocol import (
    ViewerControlMessageRequest,
    ViewerRuntimeEndpoint,
    ViewerTransportEndpoint,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_ROOT = ROOT / "mcp_outputs" / "thesis_demo" / "live"
DEFAULT_TEST_PLATE_DIR = (
    ROOT
    / "tests"
    / "integration"
    / "tests_data"
    / "imagexpress_pipeline"
    / "test_main[ImageXpress]"
    / "zstack_plate"
)
ORCHESTRATOR_DOCUMENT_ID = PlateManagerOrchestratorCodeDocumentIdentity.require_value()
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
FINAL_DEMO_STEP_DISPLAY_INDEX = 8
FINAL_DEMO_STEP_ROUTE_INDEX = FINAL_DEMO_STEP_DISPLAY_INDEX - 1
UI_BRIDGE_COMMAND_TIMEOUT_MS = 2_000
VIEWER_COMMAND_TIMEOUT_MS = 2000
OWNED_VIEWER_SHUTDOWN_TIMEOUT_SECONDS = 5.0
OWNED_EXECUTION_SHUTDOWN_TIMEOUT_SECONDS = 5.0
BASELINE_SOURCE_ALIAS = "MCP_DNA"
BASELINE_CHANNEL_IDENTITY = "MCP_DNA"
EDITED_SOURCE_ALIAS = "MCP_DNA_REBOUND"
EDITED_CHANNEL_IDENTITY = "MCP_DNA_REBOUND"
BOUNDED_SAMPLE_EDGE = 8
BOUNDED_SAMPLE_MAX_ELEMENTS = BOUNDED_SAMPLE_EDGE * BOUNDED_SAMPLE_EDGE
SOURCE_SAMPLE_MAX_AUTO_RESOLUTION_SIZE = 1024
SOURCE_RESOLUTION_STATISTICS_MAX_ELEMENTS = (
    SOURCE_SAMPLE_MAX_AUTO_RESOLUTION_SIZE * SOURCE_SAMPLE_MAX_AUTO_RESOLUTION_SIZE
)
AUTHORING_CONTEXT_MAX_CHARS = 16_000
AUTHORING_SCHEMA_ROOT_PROBES = (
    (
        "pipeline",
        (
            "processing_config",
            "source_bindings_config",
            "path_planning_config",
            "step_materialization_config",
            "step_well_filter_config",
            "sequential_processing_config",
            "napari_streaming_config",
            "fiji_streaming_config",
        ),
    ),
    (
        "step",
        (
            "dtype_config",
            "processing_config",
            "source_bindings",
            "step_well_filter_config",
            "step_materialization_config",
            "streaming_defaults",
            "napari_streaming_config",
            "fiji_streaming_config",
        ),
    ),
)
AUTHORING_SCHEMA_PROBES = (
    (
        "pipeline",
        "processing_config",
        (
            "processing_config.variable_components",
            "processing_config.group_by",
        ),
    ),
    (
        "pipeline",
        "source_bindings_config",
        (
            "source_bindings_config.bindings[].alias",
            "source_bindings_config.bindings[].selector.components[].component",
            "source_bindings_config.bindings[].component_identity[].component",
        ),
    ),
    (
        "pipeline",
        "path_planning_config",
        (
            "path_planning_config.global_output_folder",
            "path_planning_config.sub_dir",
        ),
    ),
    (
        "pipeline",
        "napari_streaming_config",
        (
            "napari_streaming_config.enabled",
            "napari_streaming_config.port",
            "napari_streaming_config.site_mode",
        ),
    ),
    (
        "pipeline",
        "fiji_streaming_config",
        (
            "fiji_streaming_config.enabled",
            "fiji_streaming_config.port",
            "fiji_streaming_config.site_mode",
        ),
    ),
    (
        "step",
        "processing_config",
        (
            "processing_config.variable_components",
            "processing_config.group_by",
        ),
    ),
    (
        "step",
        "source_bindings",
        (
            "source_bindings.enabled",
            "source_bindings.bindings[].alias",
            "source_bindings.bindings[].component_identity[].component",
        ),
    ),
    (
        "step",
        "step_materialization_config",
        (
            "step_materialization_config.enabled",
            "step_materialization_config.sub_dir",
        ),
    ),
    (
        "step",
        "step_well_filter_config",
        (
            "step_well_filter_config.well_filter",
            "step_well_filter_config.well_filter_mode",
        ),
    ),
    (
        "step",
        "napari_streaming_config",
        (
            "napari_streaming_config.enabled",
            "napari_streaming_config.port",
            "napari_streaming_config.site_mode",
        ),
    ),
    (
        "step",
        "fiji_streaming_config",
        (
            "fiji_streaming_config.enabled",
            "fiji_streaming_config.port",
            "fiji_streaming_config.site_mode",
        ),
    ),
)


class RehearsalFailure(RuntimeError):
    """Hard failure for an unmet live-demo acceptance criterion."""


@dataclass
class StepRecord:
    name: str
    elapsed_seconds: float
    ok: bool
    evidence_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DemoArtifactContracts:
    """Public function-contract outputs selected for the live rehearsal."""

    function_id: str
    measurement_name: str
    roi_name: str


@dataclass(frozen=True)
class DemoSourceBindingState:
    """One authored lazy source-binding state used by the save/rebuild gate."""

    phase: str
    source_alias: str
    channel_identity: str


BASELINE_SOURCE_BINDING = DemoSourceBindingState(
    phase="baseline",
    source_alias=BASELINE_SOURCE_ALIAS,
    channel_identity=BASELINE_CHANNEL_IDENTITY,
)
EDITED_SOURCE_BINDING = DemoSourceBindingState(
    phase="edited",
    source_alias=EDITED_SOURCE_ALIAS,
    channel_identity=EDITED_CHANNEL_IDENTITY,
)
REVERTED_SOURCE_BINDING = DemoSourceBindingState(
    phase="reverted",
    source_alias=BASELINE_SOURCE_ALIAS,
    channel_identity=BASELINE_CHANNEL_IDENTITY,
)


@dataclass
class RunContext:
    index: int
    run_id: str
    run_dir: Path
    plate_dir: Path
    output_plate_dir: Path
    source_path: Path
    descriptor_dir: Path
    napari_port: int
    zmq_port: int
    viewer_timeout_ms: int
    mcp_client: McpDevClient | None = None
    descriptor_path: Path | None = None
    ui_process: subprocess.Popen[bytes] | None = None
    owned_ui_bridge_endpoint: TransportEndpoint | None = None
    runtime_lock_file: Any | None = None
    owns_execution_endpoint: bool = False
    owns_napari_viewer: bool = False
    source_channel_values: tuple[str, str] | None = None
    steps: list[StepRecord] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenHCS MCP thesis live-demo rehearsal against the real UI, "
            "ZMQ compiler/executor, and Napari viewer."
        )
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--fresh-first-run", action="store_true", default=True)
    parser.add_argument("--fresh-each-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--plate-dir", type=Path, default=DEFAULT_TEST_PLATE_DIR)
    parser.add_argument("--max-run-seconds", type=float, default=240.0)
    parser.add_argument("--zmq-port", type=int, default=7777)
    parser.add_argument("--napari-port", type=int, default=5555)
    parser.add_argument("--ui-start-timeout", type=float, default=90.0)
    parser.add_argument(
        "--isolated-ui-bridge-port",
        type=int,
        default=None,
        help=(
            "Explicit exclusive bridge port for an owned UI that may run beside "
            "another PyQt UI. Runtime and viewer process conflicts remain forbidden."
        ),
    )
    parser.add_argument("--workflow-timeout", type=float, default=180.0)
    parser.add_argument("--viewer-timeout-ms", type=int, default=2000)
    parser.add_argument(
        "--official30-case",
        default="ExampleHuman",
        help=(
            "Official30 case name to retrieve through the public knowledge surface "
            "after its owner records the focused gate."
        ),
    )
    return parser.parse_args()


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolved_run_directory(session_dir: Path, index: int) -> Path:
    """Return the absolute run boundary required by PipelinePathPlanner."""

    return (session_dir / f"run_{index:02d}").resolve(strict=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_json(
    ctx: RunContext,
    label: str,
    args: list[str],
    *,
    timeout: float = 60.0,
    record: bool = True,
) -> dict[str, Any]:
    start = time.perf_counter()
    evidence_path = ctx.run_dir / "commands" / f"{len(ctx.steps):02d}_{label}.json"
    if ctx.mcp_client is None:
        raise RehearsalFailure("Persistent MCP dev client is not initialized.")
    execution = ctx.mcp_client.execute(args, timeout_seconds=timeout)
    elapsed = time.perf_counter() - start
    payload = execution.payload
    write_json(
        evidence_path,
        {
            "argv": args,
            "transport": "persistent_mcp_stdio",
            "returncode": execution.returncode,
            "stdout": payload,
            "server_stderr_tail": execution.server_stderr_tail,
        },
    )
    ok = execution.returncode == 0 and not response_has_errors(payload)
    if record:
        ctx.steps.append(
            StepRecord(
                label,
                elapsed,
                ok,
                str(evidence_path),
                {"returncode": execution.returncode},
            )
        )
    if not ok:
        raise RehearsalFailure(f"{label} failed; see {evidence_path}")
    return payload


def response_has_errors(payload: dict[str, Any]) -> bool:
    if payload.get("errors"):
        return True
    for result in payload.get("results") or ():
        if not isinstance(result, dict):
            return True
        if result.get("mcp_error") is True:
            return True
        for item in result.get("payloads") or ():
            if isinstance(item, dict) and item.get("errors"):
                return True
    return False


def first_payload(payload: dict[str, Any], tool: str | None = None) -> dict[str, Any]:
    for result in payload.get("results") or ():
        if not isinstance(result, dict):
            continue
        if tool is not None and result.get("tool") != tool:
            continue
        payloads = result.get("payloads")
        if isinstance(payloads, list) and payloads and isinstance(payloads[0], dict):
            return payloads[0]
    raise RehearsalFailure(f"No payload found for {tool or '<first tool>'}.")


def mcp_cmd(
    command: str, *args: str | Path, descriptor: Path | None = None
) -> list[str]:
    cmd = [command]
    cmd.extend(str(arg) for arg in args)
    if descriptor is not None:
        cmd.extend(["--descriptor-file-path", str(descriptor)])
        cmd.extend(["--timeout-ms", str(UI_BRIDGE_COMMAND_TIMEOUT_MS)])
    cmd.extend(["--json"])
    return cmd


def mcp_call_tool_cmd(
    tool_name: str,
    arguments: dict[str, Any],
) -> list[str]:
    return [
        "call",
        tool_name,
        "--arguments",
        json.dumps(arguments, sort_keys=True),
        "--json",
    ]


def assert_git_tracked_clean(allow_dirty: bool) -> None:
    if allow_dirty:
        return
    for cmd in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        proc = subprocess.run(cmd, cwd=ROOT, check=False)
        if proc.returncode != 0:
            raise RehearsalFailure(
                "Tracked worktree changes are present. Commit/stash them or pass --allow-dirty."
            )


def argv_contains_sequence(argv: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    """Return whether one exact contiguous argument sequence is present."""

    width = len(sequence)
    return width > 0 and any(
        argv[index : index + width] == sequence
        for index in range(len(argv) - width + 1)
    )


def matching_pids(argv_sequence: tuple[str, ...]) -> list[int]:
    pids: list[int] = []
    own_pid = os.getpid()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == own_pid:
            continue
        try:
            argv = tuple(
                argument.decode("utf-8", errors="replace")
                for argument in (proc_dir / "cmdline").read_bytes().split(b"\0")
                if argument
            )
        except OSError:
            continue
        if argv_contains_sequence(argv, argv_sequence):
            pids.append(pid)
    return pids


def isolated_ui_bridge_endpoint(port: int) -> TransportEndpoint:
    """Resolve the exact bridge endpoint declared for an isolated owned UI."""

    connection = replace(AgentUiBridgeConfig.from_environment(), port=port)
    return TransportEndpoint(
        host=connection.host,
        port=connection.require_port("isolated UI bridge port"),
        transport_mode=resolve_transport_mode(connection.transport_mode),
    )


def assert_isolated_ui_bridge_available(port: int) -> None:
    """Require an unused data/control endpoint pair for an isolated owned UI."""

    endpoint = isolated_ui_bridge_endpoint(port)
    assert_transport_endpoint_available(
        endpoint,
        OPENHCS_ZMQ_CONFIG,
        label="Isolated UI bridge",
        remediation="Choose another --isolated-ui-bridge-port.",
    )


def assert_transport_endpoint_available(
    endpoint: TransportEndpoint,
    config: ZMQConfig,
    *,
    label: str,
    remediation: str | None = None,
) -> None:
    """Clear proven stale residue, then require an unowned endpoint pair."""

    pair = endpoint.port_pair(config)
    invalid_ports = tuple(port for port in pair.ports if not 1 <= port <= 65_535)
    if invalid_ports:
        raise RehearsalFailure(
            f"{label} endpoint pair {pair.data_port}/{pair.control_port} contains "
            f"invalid ports: {invalid_ports}."
        )
    endpoint.cleanup_stale_addresses(config)
    occupied_ports = tuple(sorted(endpoint.occupied_ports(config)))
    if not occupied_ports:
        return
    suffix = f" {remediation}" if remediation else ""
    raise RehearsalFailure(
        f"{label} endpoints are already owned: {occupied_ports}.{suffix}"
    )


def assert_no_live_process_conflicts(
    ctx: RunContext,
    *,
    isolated_ui_bridge_port: int | None = None,
) -> None:
    """Require isolated requested endpoints and an unambiguous desktop owner."""

    if isolated_ui_bridge_port is None:
        ui_pids = matching_pids(("-m", "openhcs.pyqt_gui"))
        if ui_pids:
            raise RehearsalFailure(
                "Refusing to start a fresh rehearsal while another PyQt UI "
                f"owns the default bridge identity: {ui_pids}."
            )
    else:
        assert_isolated_ui_bridge_available(isolated_ui_bridge_port)
    assert_owned_execution_endpoint_available(ctx)
    assert_owned_viewer_endpoint_available(ctx)


def assert_no_live_runtime_conflicts(ctx: RunContext) -> None:
    """Recheck exact runtime/viewer ownership after taking the serialized lock."""

    assert_owned_execution_endpoint_available(ctx)
    assert_owned_viewer_endpoint_available(ctx)


def terminate_owned_process(process: subprocess.Popen[bytes] | None) -> None:
    """Terminate only the process group started by this rehearsal."""

    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=8.0)


def release_owned_ui_bridge_endpoint(ctx: RunContext) -> None:
    """Remove residue only after the exact owned UI process has terminated."""

    endpoint = ctx.owned_ui_bridge_endpoint
    process = ctx.ui_process
    if endpoint is None or process is None or process.poll() is None:
        return
    endpoint.cleanup(OPENHCS_ZMQ_CONFIG)
    ctx.owned_ui_bridge_endpoint = None


def owned_viewer_endpoint(ctx: RunContext) -> ViewerRuntimeEndpoint:
    """Project the requested viewer identity through the runtime declaration."""

    return ViewerRuntimeEndpoint(
        transport=ViewerTransportEndpoint(
            host="localhost",
            port=ctx.napari_port,
            transport_mode=TransportMode.IPC,
        ),
        config=OPENHCS_ZMQ_CONFIG,
    )


def assert_owned_viewer_endpoint_available(ctx: RunContext) -> None:
    """Require the exact viewer endpoint requested by this rehearsal to be free."""

    viewer_endpoint = owned_viewer_endpoint(ctx)
    assert_transport_endpoint_available(
        TransportEndpoint(
            host=viewer_endpoint.host,
            port=viewer_endpoint.port,
            transport_mode=viewer_endpoint.mode,
        ),
        viewer_endpoint.config,
        label="Napari viewer",
    )


def stop_owned_viewer(ctx: RunContext) -> None:
    if not ctx.owns_napari_viewer:
        return
    endpoint = owned_viewer_endpoint(ctx)
    if not endpoint.in_use():
        ctx.owns_napari_viewer = False
        return
    response = ViewerControlMessageRequest(
        endpoint=endpoint,
        message_type="force_shutdown",
        timeout=2.0,
    ).send()
    if not response.succeeded():
        raise RehearsalFailure("Owned Napari viewer rejected shutdown.")
    if not endpoint.wait_until_released(timeout=OWNED_VIEWER_SHUTDOWN_TIMEOUT_SECONDS):
        if endpoint.ping(timeout_ms=200, require_ready=False):
            raise RehearsalFailure("Owned Napari viewer remained responsive.")
        endpoint.force_release_addresses()
        if endpoint.in_use():
            raise RehearsalFailure("Owned Napari viewer did not release its endpoint.")
    ctx.owns_napari_viewer = False


def stop_owned_processes(ctx: RunContext) -> None:
    try:
        stop_owned_viewer(ctx)
    except Exception as exc:
        print(f"WARNING: failed to stop owned Napari viewer: {exc}", file=sys.stderr)
    terminate_owned_process(ctx.ui_process)
    release_owned_ui_bridge_endpoint(ctx)
    try:
        stop_owned_execution_endpoint(ctx)
    except Exception as exc:
        print(
            f"WARNING: failed to stop owned execution endpoint: {exc}",
            file=sys.stderr,
        )
    release_runtime_lock(ctx)


def acquire_runtime_lock(ctx: RunContext) -> None:
    """Acquire the canonical live-runtime lease without waiting on another owner."""

    lock_path = official_runtime_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RehearsalFailure(
            f"Official runtime lock is owned by another process: {lock_path}."
        ) from exc
    ctx.runtime_lock_file = lock_file


def official_runtime_lock_path() -> Path:
    """Return the documented user-scoped XDG lease path."""

    configured_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_home = (
        Path(configured_cache_home).expanduser()
        if configured_cache_home
        else Path.home() / ".cache"
    )
    if not cache_home.is_absolute():
        raise RehearsalFailure("XDG_CACHE_HOME must be an absolute path.")
    return (cache_home / "openhcs" / "official30-runtime.lock").resolve(strict=False)


def release_runtime_lock(ctx: RunContext) -> None:
    lock_file = ctx.runtime_lock_file
    if lock_file is None:
        return
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()
    ctx.runtime_lock_file = None


def owned_execution_config(ctx: RunContext) -> OpenHCSZMQConfig:
    """Project the requested endpoint through the UI's ZMQ declaration."""

    return replace(get_default_ui_config().zmq, default_port=ctx.zmq_port)


def owned_ui_config_cache_path(ctx: RunContext) -> Path:
    """Return the isolated persistence identity for one owned desktop."""

    return (ctx.run_dir / "config" / "ui_config.config").resolve(strict=False)


def prepare_owned_ui_config(ctx: RunContext) -> Path:
    """Persist the exact UI declaration consumed and restored by an owned UI."""

    cache_file = owned_ui_config_cache_path(ctx)
    base = get_default_ui_config()
    config = replace(
        base,
        check_for_updates_on_startup=False,
        zmq=owned_execution_config(ctx),
    )
    if not save_config_sync(
        config,
        ConfigCacheSpec(config_type=UIConfig, cache_file=cache_file),
    ):
        raise RehearsalFailure(
            f"Could not persist the owned UI configuration at {cache_file}."
        )
    return cache_file


def assert_owned_execution_endpoint_available(ctx: RunContext) -> None:
    """Require both declared ports to be free before claiming endpoint ownership."""

    config = owned_execution_config(ctx)
    assert_transport_endpoint_available(
        TransportEndpoint(
            host=config.client_host,
            port=ctx.zmq_port,
            transport_mode=resolve_transport_mode(config.transport_mode),
        ),
        config,
        label="Execution",
    )


def stop_owned_execution_endpoint(ctx: RunContext) -> None:
    """Release only the endpoint claimed before this rehearsal launched its UI."""

    if not ctx.owns_execution_endpoint:
        return
    config = owned_execution_config(ctx)
    result = ZMQClient.shutdown_endpoint_on_port(
        port=ctx.zmq_port,
        mode=EndpointShutdownMode.FORCE,
        timeout=OWNED_EXECUTION_SHUTDOWN_TIMEOUT_SECONDS,
        transport_mode=config.transport_mode,
        host=config.client_host,
        config=config,
    )
    if not result.succeeded or not result.endpoint_terminated:
        raise RehearsalFailure(
            f"Owned execution endpoint {ctx.zmq_port} did not terminate."
        )
    ctx.owns_execution_endpoint = False


def start_ui(ctx: RunContext, *, isolated_bridge_port: int | None = None) -> None:
    assert_owned_execution_endpoint_available(ctx)
    config_cache_file = prepare_owned_ui_config(ctx)
    log_path = ctx.run_dir / "processes" / "pyqt_ui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    env = os.environ.copy()
    env["OPENHCS_ENABLE_UI_BRIDGE"] = "true"
    env["OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR"] = str(ctx.descriptor_dir)
    env[UIConfigCacheEnvironment.cache_file_path_key] = str(config_cache_file)
    env["XDG_DATA_HOME"] = str((ctx.run_dir / "xdg-data").resolve())
    if isolated_bridge_port is not None:
        env["OPENHCS_UI_BRIDGE_PORT"] = str(isolated_bridge_port)
    ctx.descriptor_dir.mkdir(parents=True, exist_ok=True)
    for old_descriptor in ctx.descriptor_dir.glob("ui_bridge_*.json"):
        old_descriptor.unlink()
    ctx.ui_process = subprocess.Popen(
        [PYTHON, "-m", "openhcs.pyqt_gui", "--log-level", "WARNING"],
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if isolated_bridge_port is not None:
        ctx.owned_ui_bridge_endpoint = isolated_ui_bridge_endpoint(isolated_bridge_port)
    ctx.owns_execution_endpoint = True


def wait_for_runtime(ctx: RunContext, timeout: float) -> dict[str, Any]:
    started_at = time.perf_counter()
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            payload = command_json(
                ctx,
                "runtime_scan",
                mcp_cmd(
                    "runtime-scan",
                    str(ctx.zmq_port),
                    "--timeout-seconds",
                    "20",
                ),
                timeout=30,
                record=False,
            )
            servers = first_payload(payload).get("servers") or ()
            for server in servers:
                connection = (
                    server.get("connection") if isinstance(server, dict) else {}
                )
                if (
                    isinstance(connection, dict)
                    and connection.get("port") == ctx.zmq_port
                    and server.get("reachable") is True
                    and server.get("ready") is True
                ):
                    ctx.steps.append(
                        StepRecord(
                            "runtime_ready",
                            time.perf_counter() - started_at,
                            True,
                            None,
                            {"port": ctx.zmq_port},
                        )
                    )
                    return payload
        except Exception as exc:  # noqa: BLE001 - keep polling with recorded cause
            last_error = exc
        time.sleep(1.0)
    raise RehearsalFailure(f"Runtime server was not ready: {last_error}")


def wait_for_ui_bridge(ctx: RunContext, timeout: float) -> Path:
    started_at = time.perf_counter()
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if ctx.ui_process is not None and ctx.ui_process.poll() is not None:
            raise RehearsalFailure("PyQt UI process exited before bridge became ready.")
        descriptors = sorted(
            ctx.descriptor_dir.glob("ui_bridge_*.json"), key=lambda p: p.stat().st_mtime
        )
        if descriptors:
            candidate = descriptors[-1]
            try:
                payload = command_json(
                    ctx,
                    "ui_status",
                    mcp_cmd(
                        "ui-status", "--timeout-seconds", "20", descriptor=candidate
                    ),
                    timeout=30,
                    record=False,
                )
                status = first_payload(payload)
                if (
                    status.get("descriptor_status") == "ok"
                    and status.get("reachable") is True
                    and status.get("bridge_instance_id")
                ):
                    ctx.descriptor_path = candidate
                    ctx.steps.append(
                        StepRecord(
                            "ui_bridge_ready",
                            time.perf_counter() - started_at,
                            True,
                            None,
                            {"descriptor_path": str(candidate)},
                        )
                    )
                    return candidate
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        time.sleep(1.0)
    raise RehearsalFailure(f"UI bridge was not ready: {last_error}")


def verify_execution_runtime_discovery(ctx: RunContext) -> None:
    payload = command_json(
        ctx,
        "runtime_scan_declared_roles",
        mcp_cmd(
            "runtime-scan",
            str(ctx.zmq_port),
            "--transport-mode",
            "ipc",
            "--timeout-seconds",
            "20",
            "--timeout-ms",
            "1000",
        ),
        timeout=30,
    )
    servers = first_payload(payload).get("servers") or ()
    server_names = {
        server.get("server")
        for server in servers
        if isinstance(server, dict)
        and server.get("reachable") is True
        and server.get("ready") is True
    }
    if "ZMQExecutionServer" not in server_names:
        raise RehearsalFailure(
            "ZMQ execution server missing from explicit runtime scan."
        )


def managed_window_action_target(
    ctx: RunContext,
    action: ManagedWindowAction,
    *,
    window_id: str,
) -> tuple[dict[str, Any], str]:
    response = command_json(
        ctx,
        f"list_managed_{action.value}_action",
        mcp_cmd(
            UiListActionsCapability.cli_command,
            ManagedWindowWidgetIdentity.require_value(),
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    actions = first_payload(response, UiListActionsCapability.name).get("actions")
    if not isinstance(actions, list):
        raise RehearsalFailure("Managed-window action catalog is unavailable.")
    matches = [
        row
        for row in actions
        if isinstance(row, dict)
        and row.get("widget_id") == ManagedWindowWidgetIdentity.require_value()
        and row.get("action_id") == action.value
    ]
    if len(matches) != 1 or matches[0].get("enabled") is not True:
        raise RehearsalFailure(
            f"Managed-window action {action.value!r} is unavailable."
        )
    target_scope_ids = matches[0].get("target_scope_ids")
    window_scopes = frozenset(
        OpenHCSUiWindowId.manager_scopes_for_agent_window_id(window_id)
    )
    advertised_targets = [
        scope_id
        for scope_id in target_scope_ids or ()
        if isinstance(scope_id, str) and scope_id in window_scopes
    ]
    if len(advertised_targets) != 1:
        raise RehearsalFailure(
            f"Managed-window action {action.value!r} does not target {window_id!r}."
        )
    return matches[0], advertised_targets[0]


def save_managed_window(ctx: RunContext, *, window_id: str, label: str) -> None:
    action = ManagedWindowAction.SAVE_WITHOUT_CLOSE
    summary, target_scope_id = managed_window_action_target(
        ctx,
        action,
        window_id=window_id,
    )
    response = command_json(
        ctx,
        label,
        mcp_call_tool_cmd(
            UiInvokeActionCapability.name,
            {
                "widget_id": ManagedWindowWidgetIdentity.require_value(),
                "action_id": action.value,
                "target_scope_ids": [target_scope_id],
                "observed_selection_revision_token": summary.get(
                    "selection_revision_token"
                ),
                "require_confirmation": False,
                "connection": ui_connection_arguments(ctx),
            },
        ),
        timeout=30,
    )
    result = first_payload(response, UiInvokeActionCapability.name)
    if result.get("status") != "accepted":
        raise RehearsalFailure(f"Managed-window save was rejected for {window_id!r}.")
    require_ui_mutation_completed(
        ctx,
        result,
        action_label=f"Managed-window save for {window_id!r}",
        completed=False,
        expected_outcome="accepted",
    )


def navigate_window(
    ctx: RunContext,
    *,
    window_id: str,
    label: str,
    field_path: str | None = None,
) -> dict[str, Any]:
    """Open and focus one declared UI window, optionally revealing a field."""

    arguments: dict[str, Any] = {
        "window_id": window_id,
        "create_if_missing": True,
        "connection": ui_connection_arguments(ctx),
    }
    if field_path is not None:
        arguments["field_path"] = field_path
    response = command_json(
        ctx,
        label,
        mcp_call_tool_cmd(UiNavigateWindowCapability.name, arguments),
        timeout=30,
    )
    navigation = first_payload(response, UiNavigateWindowCapability.name)
    if navigation.get("focused") is not True:
        raise RehearsalFailure(f"Could not focus UI window {window_id!r}.")
    if field_path is not None and navigation.get("navigated") is not True:
        raise RehearsalFailure(f"Could not navigate to exact field {field_path!r}.")
    return navigation


def exact_config_document_source(
    ctx: RunContext,
    *,
    window_id: str,
    config_type: type[object],
) -> str:
    evidence_window_id = window_id.replace(":", "_")
    navigate_window(
        ctx,
        window_id=window_id,
        label=f"open_{evidence_window_id}_for_code_document",
    )
    tree = tree_for_window(
        ctx,
        window_id,
        label=f"inspect_{evidence_window_id}_config_tabs",
    )
    select_structured_tab(
        ctx,
        window_id=window_id,
        tree=tree,
        tab_label=config_type.__name__,
        evidence_label=f"select_{config_type.__name__}_tab",
    )
    documents_response = command_json(
        ctx,
        f"{evidence_window_id}_code_documents",
        mcp_cmd(
            "code-documents",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    documents = first_payload(documents_response).get("documents")
    if not isinstance(documents, list):
        raise RehearsalFailure("Code-document catalog is unavailable.")
    matches = [
        document
        for document in documents
        if isinstance(document, dict)
        and document.get("widget_id") == window_id
        and document.get("writable") is True
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("document_id"), str):
        raise RehearsalFailure(
            f"Expected one writable active config document for {window_id!r}."
        )
    document_id = matches[0]["document_id"]
    document_response = command_json(
        ctx,
        "inspect_ui_config_document",
        mcp_cmd(
            "code-document",
            document_id,
            "--selection-mode",
            "all",
            "--clean",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    source = first_payload(document_response).get("source")
    if not isinstance(source, str) or not source.strip():
        raise RehearsalFailure("Active UI config code document returned no source.")
    return source


def execution_config_from_ui_document(
    source: str,
    *,
    expected_port: int,
) -> OpenHCSZMQConfig:
    namespace: dict[str, Any] = {}
    exec(compile(source, "<live-ui-config>", "exec"), namespace)
    ui_config = namespace.get("config")
    if type(ui_config) is not UIConfig:
        raise RehearsalFailure("Active config document does not reconstruct UIConfig.")
    config = ui_config.zmq
    if type(config) is not OpenHCSZMQConfig or config.default_port != expected_port:
        raise RehearsalFailure(
            "Active UIConfig does not own the expected execution ZMQ endpoint."
        )
    return config


def verify_cold_runtime_configuration(
    ctx: RunContext,
    *,
    config: OpenHCSZMQConfig,
) -> dict[str, Any]:
    response = command_json(
        ctx,
        "cold_runtime_configured_endpoints",
        mcp_cmd(
            "runtime-info",
            str(ctx.zmq_port),
            "--host",
            config.client_host,
            "--transport-mode",
            config.transport_mode.value,
            "--timeout-ms",
            str(config.server_info_timeout_ms),
        ),
        timeout=30,
    )
    info = first_payload(response)
    connection = info.get("connection")
    expected_control_port = ctx.zmq_port + config.control_port_offset
    if (
        info.get("server") != "ZMQExecutionServer"
        or info.get("reachable") is not True
        or info.get("ready") is not True
        or info.get("control_port") != expected_control_port
        or not isinstance(connection, dict)
        or connection.get("port") != ctx.zmq_port
        or connection.get("transport_mode") != config.transport_mode.value
    ):
        raise RehearsalFailure(
            "Cold execution server did not report its configured endpoints and readiness."
        )
    return {
        "data_port": ctx.zmq_port,
        "control_port": expected_control_port,
        "host": connection.get("host"),
        "transport_mode": connection.get("transport_mode"),
        "persistent": connection.get("persistent"),
        "ready": info.get("ready"),
        "server": info.get("server"),
    }


def demo_source(
    ctx: RunContext,
    binding_state: DemoSourceBindingState = BASELINE_SOURCE_BINDING,
) -> str:
    if ctx.source_channel_values is None:
        raise RehearsalFailure(
            "Demo source requested before public plate inspection discovered channels."
        )
    primary_channel, secondary_channel = ctx.source_channel_values
    plate = str(ctx.plate_dir)
    output_root = str(ctx.output_plate_dir)
    return f"""# Edit this orchestrator configuration and save to apply changes

from pathlib import Path

from arraybridge.decorators import DtypeConversion
from openhcs.constants.constants import (
    AllComponents,
    GroupBy,
    VariableComponents,
)
from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyDtypeConfig,
    LazyNapariStreamingConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazySourceBindingsConfig,
    LazyStepMaterializationConfig,
    LazyStepWellFilterConfig,
    NapariVariableSizeHandling,
    PipelineConfig,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceSelector,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.cell_counting_cpu import (
    DetectionMethod,
    count_cells_single_channel,
)
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu
from openhcs.processing.backends.pos_gen.ashlar_main_cpu import ashlar_compute_tile_positions_cpu
from openhcs.processing.backends.processors.numpy_processor import (
    NumpyStackProjectionMethod,
    create_composite,
    create_projection,
    stack_percentile_normalize,
)

# MCP thesis demo live rehearsal run: {ctx.run_id}
# MCP source-binding state: {binding_state.phase}

plate_paths = [
    {plate!r}
]

global_config = GlobalPipelineConfig(
    auto_add_output_plate_to_plate_manager=True
)

per_plate_configs = {{
    {plate!r}: PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            global_output_folder=Path({output_root!r})
        ),
        source_bindings_config=LazySourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias={binding_state.source_alias!r},
                    selector=SourceSelector(
                        components=(
                            ComponentSelector(
                                component=AllComponents.CHANNEL,
                                value={primary_channel!r}
                            ),
                        )
                    ),
                    component_identity=(
                        ComponentSelector(
                            component=AllComponents.CHANNEL,
                            value={binding_state.channel_identity!r}
                        ),
                    )
                ),
                NamedSourceBinding(
                    alias='MCP_AGP',
                    selector=SourceSelector(
                        components=(
                            ComponentSelector(
                                component=AllComponents.CHANNEL,
                                value={secondary_channel!r}
                            ),
                        )
                    ),
                    component_identity=(
                        ComponentSelector(
                            component=AllComponents.CHANNEL,
                            value='MCP_AGP'
                        ),
                    )
                ),
            )
        ),
        napari_streaming_config=LazyNapariStreamingConfig(
            enabled=True,
            persistent=True,
            port={ctx.napari_port}
        )
    )
}}

pipeline_data = {{
    {plate!r}: [
        FunctionStep(
            func=(stack_percentile_normalize, {{
                    'low_percentile': 0.5,
                    'high_percentile': 99.5
                }}),
            name='Image Enhancement Processing {ctx.run_id}',
            step_well_filter_config=LazyStepWellFilterConfig(
                well_filter=4
            ),
            step_materialization_config=LazyStepMaterializationConfig()
        ),
        FunctionStep(
            func=create_composite,
            name='create_composite',
            processing_config=LazyProcessingConfig(
                variable_components=[
                    VariableComponents.CHANNEL
                ],
                group_by=GroupBy.NONE
            )
        ),
        FunctionStep(
            func=(create_projection, {{
                    'method': NumpyStackProjectionMethod.MAX
                }}),
            name='Z-Stack Flattening',
            processing_config=LazyProcessingConfig(
                variable_components=[
                    VariableComponents.Z_INDEX
                ]
            ),
            step_materialization_config=LazyStepMaterializationConfig()
        ),
        FunctionStep(
            func=ashlar_compute_tile_positions_cpu,
            name='Position Computation'
        ),
        FunctionStep(
            func=(stack_percentile_normalize, {{
                    'low_percentile': 0.5,
                    'high_percentile': 99.5
                }}),
            name='Secondary Enhancement',
            processing_config=LazyProcessingConfig(
                input_source=InputSource.PIPELINE_START
            )
        ),
        FunctionStep(
            func=assemble_stack_cpu,
            name='CPU Assembly'
        ),
        FunctionStep(
            func=(create_projection, {{
                    'method': NumpyStackProjectionMethod.MAX
                }}),
            name='Z-Stack Flattening',
            processing_config=LazyProcessingConfig(
                variable_components=[
                    VariableComponents.Z_INDEX
                ]
            )
        ),
        FunctionStep(
            func=(count_cells_single_channel, {{
                    'min_cell_area': 40,
                    'max_cell_area': 200,
                    'enable_preprocessing': False,
                    'detection_method': DetectionMethod.WATERSHED
                }}),
            name='Cell Counting',
            processing_config=LazyProcessingConfig(
                group_by=GroupBy.CHANNEL
            ),
            dtype_config=LazyDtypeConfig(
                default_dtype_conversion=DtypeConversion.UINT8
            ),
            napari_streaming_config=LazyNapariStreamingConfig(
                variable_size_handling=NapariVariableSizeHandling.PAD_TO_MAX
            )
        )
    ]
}}
"""


def pipeline_document_source(orchestrator_source: str) -> str:
    """Project one applied orchestrator document into the artifact-plan contract."""

    return f"""{orchestrator_source.rstrip()}

# Alias the exact UI-authored objects into the public PipelineDocument contract.
# These are references, not a copied configuration or second pipeline.
pipeline_config = per_plate_configs[plate_paths[0]]
pipeline_steps = pipeline_data[plate_paths[0]]
"""


def pipeline_document_source_path(ctx: RunContext) -> Path:
    """Return the run-local source consumed by public artifact-plan inspection."""

    return ctx.run_dir / "pipeline_document.py"


def ensure_demo_plate(ctx: RunContext) -> None:
    started_at = time.perf_counter()
    if not ctx.plate_dir.is_dir():
        raise RehearsalFailure(f"Existing test plate is missing: {ctx.plate_dir}")
    ctx.output_plate_dir.mkdir(parents=True, exist_ok=True)
    ctx.steps.append(
        StepRecord(
            "verify_existing_test_plate",
            time.perf_counter() - started_at,
            True,
            None,
            {
                "plate_dir": str(ctx.plate_dir),
                "output_root": str(ctx.output_plate_dir),
                "format_detection_owner": InspectPlatePathCapability.name,
            },
        )
    )


def inspect_authoring_context(
    ctx: RunContext,
    *,
    kind: str,
    label: str,
) -> dict[str, Any]:
    """Require one bounded, task-specific public context with a reachable next action."""

    response = command_json(
        ctx,
        label,
        mcp_cmd(
            GetAuthoringContextCapability.cli_command,
            kind,
            "--max-chars",
            str(AUTHORING_CONTEXT_MAX_CHARS),
        ),
        timeout=45,
    )
    context = first_payload(response, GetAuthoringContextCapability.name)
    content = context.get("content")
    if (
        context.get("kind") != kind
        or not isinstance(content, str)
        or not content.strip()
    ):
        raise RehearsalFailure(
            f"Authoring context {kind!r} did not return its exact public content."
        )
    if "...<truncated " in content:
        raise RehearsalFailure(
            f"Authoring context {kind!r} was truncated before its next action."
        )
    schema_version = context.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise RehearsalFailure(f"Authoring context {kind!r} has no schema version.")
    action_lines = tuple(
        line.strip()
        for line in content.splitlines()
        if "openhcs_" in line or line.strip().casefold().startswith("next:")
    )
    if not action_lines:
        raise RehearsalFailure(
            f"Authoring context {kind!r} exposes no public next action."
        )
    return {
        "kind": kind,
        "schema_version": schema_version,
        "character_count": len(content),
        "next_action": action_lines[-1],
        "truncated": False,
    }


def authoring_schema_evidence(
    schema: Mapping[str, Any],
    *,
    requested_config_type: str,
    path_prefix: str | None,
    required_paths: tuple[str, ...],
) -> dict[str, Any]:
    expected_config_type = (
        "PipelineConfig" if requested_config_type == "pipeline" else "FunctionStep"
    )
    expected_authoring_path = (
        "ConfigPatch.values"
        if requested_config_type == "pipeline"
        else "FunctionStepAddRequest.step_config_overrides"
    )
    if (
        schema.get("config_type") != expected_config_type
        or schema.get("authoring_path") != expected_authoring_path
        or schema.get("path_prefix") != path_prefix
    ):
        raise RehearsalFailure(
            f"Config schema {requested_config_type}:{path_prefix} returned the wrong owner."
        )
    fields = schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise RehearsalFailure(
            f"Config schema {requested_config_type}:{path_prefix} has no fields."
        )
    field_by_path = {
        field["path"]: field
        for field in fields
        if isinstance(field, dict)
        and isinstance(field.get("path"), str)
        and field["path"]
    }
    missing_paths = set(required_paths) - set(field_by_path)
    if missing_paths:
        raise RehearsalFailure(
            f"Config schema {requested_config_type}:{path_prefix} is missing "
            f"{sorted(missing_paths)}."
        )
    root_paths = required_paths if path_prefix is None else (path_prefix,)
    for field_path in root_paths:
        field = field_by_path[field_path]
        if (
            not all(
                isinstance(field.get(name), str) and field[name]
                for name in (
                    "type_repr",
                    "description",
                    "declaring_type",
                    "default_origin",
                )
            )
            or field.get("default_repr") is None
        ):
            raise RehearsalFailure(
                f"Config field {field_path!r} lacks owner/default/type documentation."
            )
        if field.get("nested_schema_path") != field_path:
            raise RehearsalFailure(
                f"Config field {field_path!r} lacks its exact nested follow-up path."
            )
    type_docs = schema.get("types")
    if not isinstance(type_docs, list) or not any(
        isinstance(type_doc, dict)
        and isinstance(type_doc.get("type_repr"), str)
        and type_doc["type_repr"]
        and isinstance(type_doc.get("description"), str)
        and type_doc["description"]
        for type_doc in type_docs
    ):
        raise RehearsalFailure(
            f"Config schema {requested_config_type}:{path_prefix} has no type documentation."
        )
    registries = schema.get("registries")
    if not isinstance(registries, list):
        raise RehearsalFailure("Config schema registry projection is not a list.")
    registry_owners = [
        registry["owner_type"]
        for registry in registries
        if isinstance(registry, dict)
        and isinstance(registry.get("owner_type"), str)
        and registry["owner_type"]
    ]
    if path_prefix in {"napari_streaming_config", "fiji_streaming_config"}:
        if "openhcs.core.config.StreamingConfig" not in registry_owners:
            raise RehearsalFailure(
                f"Streaming schema {path_prefix!r} lacks its live registry owner."
            )
    elif path_prefix == "processing_config" and registry_owners:
        raise RehearsalFailure(
            "Processing schema unexpectedly includes an unrelated nominal registry."
        )
    if requested_config_type == "step" and path_prefix == "source_bindings":
        artifact_kind_fields = [
            field
            for field_path, field in field_by_path.items()
            if field_path.endswith("artifact_kind")
        ]
        if not artifact_kind_fields or not any(
            field.get("registry_values") for field in artifact_kind_fields
        ):
            raise RehearsalFailure(
                "Step source-binding schema lacks ArtifactType registry values."
            )
    enum_paths = sorted(
        field_path
        for field_path, field in field_by_path.items()
        if field.get("enum_values")
    )
    return {
        "requested_config_type": requested_config_type,
        "config_type": schema.get("config_type"),
        "path_prefix": path_prefix,
        "authoring_path": schema.get("authoring_path"),
        "field_count": len(field_by_path),
        "required_paths": list(required_paths),
        "enum_paths": enum_paths,
        "registry_owners": registry_owners,
        "type_document_count": len(type_docs),
    }


def inspect_authoring_schemas(ctx: RunContext) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    probes = (
        tuple(
            (config_type, None, required_paths)
            for config_type, required_paths in AUTHORING_SCHEMA_ROOT_PROBES
        )
        + AUTHORING_SCHEMA_PROBES
    )
    for config_type, path_prefix, required_paths in probes:
        arguments: dict[str, Any] = {"config_type": config_type}
        if path_prefix is not None:
            arguments["path_prefix"] = path_prefix
        response = command_json(
            ctx,
            f"inspect_{config_type}_{path_prefix or 'root'}_schema",
            mcp_call_tool_cmd(DescribeConfigSchemaCapability.name, arguments),
            timeout=45,
        )
        evidence.append(
            authoring_schema_evidence(
                first_payload(response, DescribeConfigSchemaCapability.name),
                requested_config_type=config_type,
                path_prefix=path_prefix,
                required_paths=required_paths,
            )
        )
    return {
        "capability": DescribeConfigSchemaCapability.name,
        "probe_count": len(evidence),
        "probes": evidence,
    }


def bounded_element_count(shape: Any) -> int | None:
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    if any(not isinstance(value, int) or value < 0 for value in shape):
        return None
    count = 1
    for value in shape:
        count *= value
    return count


def source_statistics_element_count(sample: Mapping[str, Any]) -> int | None:
    """Return the exact pixel count covered by reported source statistics."""

    statistics_scope = sample.get("statistics_scope")
    if statistics_scope == "bounded_sample":
        return bounded_element_count(sample.get("sample_shape"))
    if statistics_scope == "source_resolution":
        return bounded_element_count(sample.get("resolution_shape"))
    return None


def inspect_source_plate_and_sample(ctx: RunContext) -> dict[str, Any]:
    """Inspect and sample a source through bounded public plate-data surfaces."""

    inspection_response = command_json(
        ctx,
        "inspect_source_plate",
        mcp_cmd(
            InspectPlatePathCapability.cli_command,
            ctx.plate_dir,
            "--max-sample-files",
            "8",
            "--max-component-values",
            "64",
            "--max-parse-failure-samples",
            "8",
            "--max-files-to-parse",
            "256",
            "--timeout-seconds",
            "40",
        ),
        timeout=45,
    )
    inspection = first_payload(inspection_response, InspectPlatePathCapability.name)
    if inspection.get("status") != "ok":
        raise RehearsalFailure("Source plate inspection did not return status='ok'.")
    image_files = inspection.get("image_files")
    parse_summary = inspection.get("parse_summary")
    workflow_advice = inspection.get("workflow_advice")
    if not isinstance(image_files, dict) or not isinstance(parse_summary, dict):
        raise RehearsalFailure("Source inspection lacks image and parse summaries.")
    records = image_files.get("sampled_records")
    if not isinstance(records, list) or not records or len(records) > 8:
        raise RehearsalFailure("Source inspection returned no bounded image records.")
    selected_record = next(
        (
            record
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("virtual_path"), str)
            and record["virtual_path"]
        ),
        None,
    )
    if selected_record is None:
        raise RehearsalFailure("Source inspection records have no virtual image path.")
    attempted = parse_summary.get("attempted_file_count")
    parsed = parse_summary.get("parsed_file_count")
    if (
        not isinstance(attempted, int)
        or attempted < 1
        or not isinstance(parsed, int)
        or parsed < 1
    ):
        raise RehearsalFailure("Source inspection did not parse any image identity.")
    if not isinstance(workflow_advice, dict) or not all(
        isinstance(workflow_advice.get(field_name), str) and workflow_advice[field_name]
        for field_name in (
            "ingestion_route",
            "ingestion_owner",
            "source_binding_role",
            "ui_code_document_id",
        )
    ):
        raise RehearsalFailure(
            "Source inspection did not expose typed ingestion and UI routing advice."
        )
    channel_values = component_value_keys(inspection, "channel")
    if len(channel_values) < 2:
        raise RehearsalFailure(
            "The representative multidimensional rehearsal requires two discovered channels."
        )

    sample_response = command_json(
        ctx,
        "sample_source_plate_image_bounded",
        mcp_cmd(
            SamplePlateImageCapability.cli_command,
            ctx.plate_dir,
            selected_record["virtual_path"],
            "--height",
            str(BOUNDED_SAMPLE_EDGE),
            "--width",
            str(BOUNDED_SAMPLE_EDGE),
            "--max-array-elements",
            str(BOUNDED_SAMPLE_MAX_ELEMENTS),
            "--max-auto-resolution-size",
            str(SOURCE_SAMPLE_MAX_AUTO_RESOLUTION_SIZE),
            "--include-array-values",
        ),
        timeout=45,
    )
    sample = first_payload(sample_response, SamplePlateImageCapability.name)
    sample_elements = bounded_element_count(sample.get("sample_shape"))
    statistics_elements = source_statistics_element_count(sample)
    statistics_scope = sample.get("statistics_scope")
    statistics_budget = (
        BOUNDED_SAMPLE_MAX_ELEMENTS
        if statistics_scope == "bounded_sample"
        else SOURCE_RESOLUTION_STATISTICS_MAX_ELEMENTS
    )
    if (
        sample.get("sample_included") is not True
        or sample_elements is None
        or sample_elements < 1
        or sample_elements > BOUNDED_SAMPLE_MAX_ELEMENTS
        or statistics_elements is None
        or statistics_elements < 1
        or statistics_elements > statistics_budget
    ):
        raise RehearsalFailure(
            "Source pixel sample did not remain inside the explicit bounded budget."
        )
    if not isinstance(sample.get("selected_resolution_index"), int):
        raise RehearsalFailure(
            "Source pixel sample has no native-resolution selection provenance."
        )
    if not isinstance(sample.get("source_path"), str) or not sample["source_path"]:
        raise RehearsalFailure("Source pixel sample has no physical provenance path.")
    return {
        "inspection": {
            "plate_path": inspection.get("plate_path"),
            "detected_microscope_type": inspection.get("detected_microscope_type"),
            "handler_class": inspection.get("handler_class"),
            "image_count": image_files.get("count"),
            "sampled_image_count": len(records),
            "parsed_image_count": parsed,
            "components": inspection.get("components"),
            "channel_values": list(channel_values),
            "source_diagnostics": inspection.get("source_diagnostics"),
            "workflow_advice": workflow_advice,
        },
        "sample": {
            "virtual_path": sample.get("virtual_path"),
            "source_path": sample.get("source_path"),
            "source_shape": sample.get("shape"),
            "resolution_shape": sample.get("resolution_shape"),
            "selected_resolution_index": sample.get("selected_resolution_index"),
            "resolution_count": sample.get("resolution_count"),
            "downsample_yx": sample.get("downsample_yx"),
            "statistics_scope": sample.get("statistics_scope"),
            "statistics_element_count": statistics_elements,
            "statistics_element_budget": statistics_budget,
            "sample_shape": sample.get("sample_shape"),
            "sample_element_count": sample_elements,
            "dtype": sample.get("dtype"),
            "minimum": sample.get("minimum"),
            "maximum": sample.get("maximum"),
            "mean": sample.get("mean"),
        },
    }


def inspect_and_apply_code_document(
    ctx: RunContext,
    binding_state: DemoSourceBindingState,
) -> dict[str, Any]:
    assert ctx.descriptor_path is not None
    ctx.source_path.write_text(
        demo_source(ctx, binding_state),
        encoding="utf-8",
    )
    phase = binding_state.phase
    docs = command_json(
        ctx,
        f"{phase}_code_documents",
        mcp_cmd(
            "code-documents", "--timeout-seconds", "20", descriptor=ctx.descriptor_path
        ),
        timeout=30,
    )
    documents = first_payload(docs).get("documents") or ()
    if not any(
        isinstance(doc, dict)
        and doc.get("document_id") == ORCHESTRATOR_DOCUMENT_ID
        and doc.get("writable") is True
        for doc in documents
    ):
        raise RehearsalFailure(
            "Writable plate_manager.orchestrator_config document missing."
        )

    source_payload = command_json(
        ctx,
        f"{phase}_inspect_orchestrator_document",
        mcp_cmd(
            "code-document",
            ORCHESTRATOR_DOCUMENT_ID,
            "--selection-mode",
            "all",
            "--clean",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    token = first_payload(source_payload).get("current_revision_token")
    if not isinstance(token, str) or not token:
        raise RehearsalFailure(
            "Orchestrator code document did not expose a revision token."
        )

    validation = command_json(
        ctx,
        f"{phase}_validate_orchestrator_document",
        mcp_cmd(
            "validate-code-document",
            ORCHESTRATOR_DOCUMENT_ID,
            "--source-file",
            ctx.source_path,
            "--base-revision-token",
            token,
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    validation_payload = first_payload(validation)
    if validation_payload.get("valid") is not True:
        raise RehearsalFailure(
            "Orchestrator code document validation did not return valid=True."
        )

    apply_payload = command_json(
        ctx,
        f"{phase}_apply_orchestrator_document",
        mcp_cmd(
            "apply-code-document",
            ORCHESTRATOR_DOCUMENT_ID,
            "--source-file",
            ctx.source_path,
            "--base-revision-token",
            token,
            "--no-confirmation",
            "--snapshot-label",
            f"MCP thesis demo live {phase} source {ctx.run_id}",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=60,
    )
    applied_payload = first_payload(apply_payload)
    if (
        applied_payload.get("applied") is not True
        or applied_payload.get("outcome") != "applied"
    ):
        operation_id = applied_payload.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            raise RehearsalFailure(
                "Orchestrator code document apply did not return an operation id."
            )
        wait_for_ui_operation(
            ctx,
            operation_id=operation_id,
            expected_outcome="applied",
            timeout=30,
        )

    reread = command_json(
        ctx,
        f"{phase}_reread_orchestrator_document",
        mcp_cmd(
            "code-document",
            ORCHESTRATOR_DOCUMENT_ID,
            "--selection-mode",
            "all",
            "--clean",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    source = first_payload(reread).get("source")
    if not isinstance(source, str):
        raise RehearsalFailure(
            "Applied orchestrator document did not return Python source."
        )
    evidence = assert_applied_document_state(ctx, source, binding_state)
    pipeline_document_source_path(ctx).write_text(
        pipeline_document_source(source),
        encoding="utf-8",
    )
    return evidence


def assert_applied_document_state(
    ctx: RunContext,
    source: str,
    binding_state: DemoSourceBindingState,
) -> dict[str, Any]:
    namespace: dict[str, object] = {}
    exec(compile(source, str(ctx.source_path), "exec"), namespace)
    per_plate_configs = namespace["per_plate_configs"]
    pipeline_data = namespace["pipeline_data"]
    if not isinstance(per_plate_configs, dict) or not isinstance(pipeline_data, dict):
        raise RehearsalFailure(
            "Applied document did not reconstruct pipeline mappings."
        )

    expected_plate = ctx.plate_dir.resolve(strict=False)
    config_matches = [
        config
        for plate_path, config in per_plate_configs.items()
        if Path(plate_path).resolve(strict=False) == expected_plate
    ]
    pipeline_matches = [
        steps
        for plate_path, steps in pipeline_data.items()
        if Path(plate_path).resolve(strict=False) == expected_plate
    ]
    if len(config_matches) != 1 or len(pipeline_matches) != 1:
        raise RehearsalFailure(
            "Applied document did not reconstruct exactly one demo plate configuration."
        )

    config = config_matches[0]
    observed_bindings = tuple(
        (
            binding.alias,
            tuple(
                (selector.component, selector.value)
                for selector in binding.component_identity
            ),
        )
        for binding in config.source_bindings_config.bindings
    )
    expected_bindings = (
        (
            binding_state.source_alias,
            ((AllComponents.CHANNEL, binding_state.channel_identity),),
        ),
        ("MCP_AGP", ((AllComponents.CHANNEL, "MCP_AGP"),)),
    )
    if observed_bindings != expected_bindings:
        raise RehearsalFailure(
            f"Applied document reconstructed the wrong {binding_state.phase} "
            "source-binding state."
        )

    steps = pipeline_matches[0]
    expected_first_step_name = f"Image Enhancement Processing {ctx.run_id}"
    if (
        not isinstance(steps, list)
        or not steps
        or steps[0].name != expected_first_step_name
    ):
        raise RehearsalFailure(
            "Applied document reconstructed the wrong demo pipeline."
        )
    processing_semantics = []
    for step_index, step in enumerate(steps):
        processing_config = step.processing_config
        processing_semantics.append(
            {
                "step_index": step_index,
                "group_by": processing_config.group_by.value,
                "variable_components": [
                    component.value
                    for component in processing_config.variable_components
                ],
            }
        )
    return {
        "phase": binding_state.phase,
        "plate_path": str(expected_plate),
        "step_count": len(steps),
        "step_names": [step.name for step in steps],
        "source_bindings": [
            {
                "alias": binding.alias,
                "selector_components": [
                    {
                        "component": selector.component.value,
                        "value": selector.value,
                    }
                    for selector in binding.selector.components
                ],
                "component_identity": [
                    {
                        "component": selector.component.value,
                        "value": selector.value,
                    }
                    for selector in binding.component_identity
                ],
            }
            for binding in config.source_bindings_config.bindings
        ],
        "config_families": {
            "pipeline": type(config).__name__,
            "path_planning": type(config.path_planning_config).__name__,
            "source_bindings": type(config.source_bindings_config).__name__,
            "napari_streaming": type(config.napari_streaming_config).__name__,
        },
        "output_folder": str(config.path_planning_config.global_output_folder),
        "viewer_streaming": {
            "enabled": config.napari_streaming_config.enabled,
            "persistent": config.napari_streaming_config.persistent,
            "port": config.napari_streaming_config.port,
        },
        "processing_semantics": processing_semantics,
    }


def inspect_compiled_artifact_plan(
    ctx: RunContext,
    contracts: DemoArtifactContracts,
    *,
    axis_id: str,
) -> dict[str, Any]:
    """Compile the exact UI-authored PipelineDocument into a bounded public plan."""

    response = command_json(
        ctx,
        "inspect_compiled_artifact_plan",
        mcp_cmd(
            InspectPipelineSourceArtifactPlanCapability.cli_command,
            ctx.plate_dir,
            "--source-file",
            pipeline_document_source_path(ctx),
            "--axis-filter",
            axis_id,
        ),
        timeout=90,
    )
    plan = first_payload(
        response,
        InspectPipelineSourceArtifactPlanCapability.name,
    )
    steps = plan.get("steps")
    source_workspace = plan.get("source_workspace")
    if (
        plan.get("axis_count") != 1
        or not isinstance(steps, list)
        or not steps
        or not isinstance(source_workspace, dict)
    ):
        raise RehearsalFailure(
            "Artifact-plan inspection did not return one bounded axis and its steps."
        )
    if plan.get("step_count") != len(steps):
        raise RehearsalFailure(
            "Artifact-plan step count disagrees with its structured step records."
        )
    source_files = source_workspace.get("files")
    source_file_count = source_workspace.get("file_count")
    truncated_file_count = source_workspace.get("truncated_file_count")
    if (
        not isinstance(source_files, list)
        or not source_files
        or not isinstance(source_file_count, int)
        or not isinstance(truncated_file_count, int)
        or truncated_file_count < 0
        or source_file_count != len(source_files) + truncated_file_count
    ):
        raise RehearsalFailure(
            "Artifact plan did not expose a coherent bounded source workspace."
        )
    if not all(
        isinstance(record, dict)
        and isinstance(record.get("virtual_path"), str)
        and record["virtual_path"]
        and isinstance(record.get("source_path"), str)
        and record["source_path"]
        for record in source_files
    ):
        raise RehearsalFailure(
            "Artifact-plan source files lack virtual and physical storage identities."
        )
    outputs = [
        output
        for step in steps
        if isinstance(step, dict)
        for output in (step.get("artifact_outputs") or ())
        if isinstance(output, dict)
    ]
    output_identities = {
        (output["name"], output["kind"])
        for output in outputs
        if (
            isinstance(output.get("name"), str)
            and output["name"]
            and isinstance(output.get("kind"), str)
            and output["kind"]
        )
    }
    required_outputs = {
        (contracts.measurement_name, "measurements"),
        (contracts.roi_name, "object_labels"),
    }
    if not required_outputs.issubset(output_identities):
        raise RehearsalFailure(
            "Compiled artifact plan does not contain the function-catalog contracts: "
            f"{sorted(required_outputs - output_identities)}."
        )
    materialization_records = [
        output["materialization"]
        for output in outputs
        if isinstance(output.get("materialization"), dict)
    ]
    if not materialization_records:
        raise RehearsalFailure(
            "Compiled artifact outputs expose no materialization contracts."
        )
    return {
        "axis_id": axis_id,
        "axis_count": plan.get("axis_count"),
        "step_count": plan.get("step_count"),
        "progress_event_count": plan.get("progress_event_count"),
        "source_workspace": {
            "file_count": source_file_count,
            "returned_file_count": len(source_files),
            "truncated_file_count": truncated_file_count,
            "virtual_paths": [record["virtual_path"] for record in source_files],
            "source_paths": [record["source_path"] for record in source_files],
        },
        "artifact_outputs": [
            {"name": name, "kind": kind} for name, kind in sorted(output_identities)
        ],
        "required_artifacts": [
            {"name": name, "kind": kind} for name, kind in sorted(required_outputs)
        ],
        "artifact_output_count": len(outputs),
        "materialization_contract_count": len(materialization_records),
        "materialization_backends": sorted(
            {
                backend
                for record in materialization_records
                if isinstance((backend := record.get("persistent_backend")), str)
                and backend
            }
        ),
    }


def ui_connection_arguments(ctx: RunContext) -> dict[str, Any]:
    if ctx.descriptor_path is None:
        raise RehearsalFailure(
            "UI bridge connection requested before descriptor discovery."
        )
    return {
        "descriptor_file_path": str(ctx.descriptor_path),
        "timeout_ms": UI_BRIDGE_COMMAND_TIMEOUT_MS,
    }


def records_from_query_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    query = payload.get("query", payload)
    if not isinstance(query, dict):
        raise RehearsalFailure(
            "Selected-plate file query did not return a query payload."
        )
    records = query.get("records")
    if not isinstance(records, list):
        raise RehearsalFailure("Selected-plate file query did not return records.")
    return [record for record in records if isinstance(record, dict)]


def artifact_name_from_function_detail(
    detail: Mapping[str, Any],
    *,
    artifact_kind: str,
) -> str:
    runtime_contract = detail.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        raise RehearsalFailure("Function detail has no runtime contract.")
    outputs = runtime_contract.get("artifact_outputs")
    if not isinstance(outputs, list):
        raise RehearsalFailure("Function detail has no artifact output declarations.")
    for output in outputs:
        if not isinstance(output, dict) or output.get("kind") != artifact_kind:
            continue
        name = output.get("name")
        if isinstance(name, str) and name:
            return name
    raise RehearsalFailure(
        f"Function detail has no declared {artifact_kind!r} artifact output."
    )


def discover_demo_contracts(ctx: RunContext) -> DemoArtifactContracts:
    functions = command_json(
        ctx,
        "tour_search_cell_counting_function",
        mcp_cmd("functions", "count_cells_single_channel", "--limit", "20"),
        timeout=30,
    )
    function_items = first_payload(functions).get("items")
    if not isinstance(function_items, list):
        raise RehearsalFailure("Function catalog search did not return items.")
    function_id = next(
        (
            item.get("function_id")
            for item in function_items
            if isinstance(item, dict)
            and item.get("name") == "count_cells_single_channel"
            and isinstance(item.get("function_id"), str)
        ),
        None,
    )
    if not isinstance(function_id, str):
        raise RehearsalFailure("Could not discover the Cell Counting function id.")

    detail_response = command_json(
        ctx,
        "tour_describe_cell_counting_function",
        mcp_cmd("function", function_id, "--max-doc-chars", "2000"),
        timeout=30,
    )
    detail = first_payload(detail_response)
    measurement_name = artifact_name_from_function_detail(
        detail,
        artifact_kind="measurements",
    )
    roi_name = artifact_name_from_function_detail(detail, artifact_kind="object_labels")
    return DemoArtifactContracts(
        function_id=function_id,
        measurement_name=measurement_name,
        roi_name=roi_name,
    )


def official30_python_source_target(
    hits: list[Any],
) -> tuple[str, str]:
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        document = hit.get("document")
        section = hit.get("section")
        document_id = (
            document.get("document_id") if isinstance(document, dict) else None
        )
        section_id = section.get("section_id") if isinstance(section, dict) else None
        section_title = section.get("title") if isinstance(section, dict) else None
        if (
            isinstance(document_id, str)
            and isinstance(section_id, str)
            and isinstance(section_title, str)
            and section_title.casefold().endswith("openhcs python")
        ):
            return document_id, section_id
    raise RehearsalFailure(
        "Official30 knowledge search did not expose a converted OpenHCS Python section."
    )


def run_guided_tour(
    ctx: RunContext,
    *,
    official30_case: str | None,
) -> tuple[DemoArtifactContracts, dict[str, Any]]:
    catalog = command_json(
        ctx,
        "tour_list_knowledge_documents",
        mcp_cmd("knowledge", "--limit", "100"),
        timeout=30,
    )
    documents = first_payload(catalog).get("documents")
    if not isinstance(documents, list) or not documents:
        raise RehearsalFailure("Knowledge catalog is empty.")

    search = command_json(
        ctx,
        "tour_search_workflow_guidance",
        mcp_cmd(
            "knowledge-search", "pipeline artifact measurement viewer", "--limit", "5"
        ),
        timeout=30,
    )
    hits = first_payload(search).get("hits")
    if not isinstance(hits, list) or not hits:
        raise RehearsalFailure("Knowledge search returned no workflow guidance.")
    first_hit = hits[0]
    document = first_hit.get("document") if isinstance(first_hit, dict) else None
    document_id = document.get("document_id") if isinstance(document, dict) else None
    if not isinstance(document_id, str) or not document_id:
        raise RehearsalFailure("Knowledge search hit has no document id.")
    document_response = command_json(
        ctx,
        "tour_read_workflow_guidance",
        mcp_cmd("knowledge-document", document_id, "--max-chars", "4000"),
        timeout=30,
    )
    content = first_payload(document_response).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RehearsalFailure("Knowledge document returned no readable content.")

    contracts = discover_demo_contracts(ctx)
    focused = command_json(
        ctx,
        "tour_focus_ui_window",
        mcp_call_tool_cmd(
            UiFocusWindowCapability.name,
            {
                "window_id": OpenHCSUiWindowId.plate_manager,
                "connection": ui_connection_arguments(ctx),
            },
        ),
        timeout=30,
    )
    if first_payload(focused, UiFocusWindowCapability.name).get("focused") is not True:
        raise RehearsalFailure("Guided tour could not focus the PlateManager window.")

    official_source_evidence: dict[str, Any] | None = None
    if official30_case:
        official_search = command_json(
            ctx,
            "tour_discover_official30_public_source",
            mcp_cmd("knowledge-search", official30_case, "--limit", "20"),
            timeout=30,
        )
        official_hits = first_payload(official_search).get("hits")
        if not isinstance(official_hits, list) or not official_hits:
            raise RehearsalFailure(
                f"Official30 knowledge surface has no discoverable case {official30_case!r}."
            )
        official_document_id, official_section_id = official30_python_source_target(
            official_hits
        )
        source_args: list[str | Path] = [
            "knowledge-document",
            official_document_id,
            "--section-id",
            official_section_id,
            "--max-chars",
            "20000",
        ]
        source_response = command_json(
            ctx,
            "tour_read_official30_public_source",
            mcp_cmd(*source_args),
            timeout=45,
        )
        source = first_payload(source_response).get("content")
        if (
            not isinstance(source, str)
            or "pipeline_config" not in source
            or "pipeline_steps" not in source
        ):
            raise RehearsalFailure(
                "Official30 knowledge section did not return converted public pipeline source."
            )
        official_source_evidence = {
            "case": official30_case,
            "document_id": official_document_id,
            "section_id": official_section_id,
            "character_count": len(source),
            "complete_pipeline_document": True,
        }
    return contracts, {
        "knowledge_document_count": len(documents),
        "workflow_document_id": document_id,
        "workflow_content_character_count": len(content),
        "function_contract_id": contracts.function_id,
        "official_source": official_source_evidence,
    }


def wait_for_ui_operation(
    ctx: RunContext,
    *,
    operation_id: str,
    expected_outcome: str,
    timeout: float,
) -> dict[str, Any]:
    assert ctx.descriptor_path is not None
    tool_timeout = mcp_tool_timeout_seconds(
        round(timeout * 1000),
        timeout_seconds=DEFAULT_CALL_TIMEOUT_SECONDS,
    )
    arguments = {
        "operation_id": operation_id,
        "timeout_seconds": timeout,
        "connection": ui_connection_arguments(ctx),
    }
    payload = command_json(
        ctx,
        f"ui_operation_wait_{operation_id}",
        [
            "--timeout-seconds",
            str(tool_timeout),
            *mcp_call_tool_cmd(
                "openhcs_ui_wait_for_operation_receipt",
                arguments,
            ),
        ],
        timeout=tool_timeout + MCP_TOOL_TIMEOUT_MARGIN_SECONDS,
    )
    status = first_payload(payload, "openhcs_ui_wait_for_operation_receipt")
    if status.get("status") == "completed":
        if status.get("outcome") != expected_outcome:
            raise RehearsalFailure(
                f"UI operation {operation_id} completed with outcome "
                f"{status.get('outcome')!r}, expected {expected_outcome!r}."
            )
        return status
    if status.get("status") == "failed":
        raise RehearsalFailure(f"UI operation {operation_id} failed: {status}")
    raise RehearsalFailure(
        f"UI operation {operation_id} did not reach a terminal status: {status}"
    )


def require_ui_mutation_completed(
    ctx: RunContext,
    payload: Mapping[str, Any],
    *,
    action_label: str,
    completed: bool,
    expected_outcome: str,
) -> None:
    if completed:
        return
    receipt = payload.get("receipt")
    if not isinstance(receipt, Mapping) or receipt.get("accepted") is not True:
        raise RehearsalFailure(f"{action_label} was rejected: {payload}")
    operation_id = receipt.get("bridge_operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise RehearsalFailure(
            f"{action_label} returned an accepted receipt without an operation id."
        )
    wait_for_ui_operation(
        ctx,
        operation_id=operation_id,
        expected_outcome=expected_outcome,
        timeout=30,
    )


def selected_workflow(ctx: RunContext, workflow: str, timeout: float) -> dict[str, Any]:
    assert ctx.descriptor_path is not None
    payload = command_json(
        ctx,
        f"workflow_{workflow}",
        mcp_cmd(
            "selected-workflow",
            workflow,
            "--poll-state",
            "--poll-selection-mode",
            "selected",
            "--poll-interval-seconds",
            "0.5",
            "--poll-timeout-seconds",
            str(timeout),
            "--timeout-seconds",
            str(timeout + 30),
            descriptor=ctx.descriptor_path,
        ),
        timeout=timeout + 40,
    )
    summary = first_payload(payload, "mcp_dev_selected_workflow_poll")
    if (
        summary.get("poll_completed") is not True
        or summary.get("poll_status") != "completed"
    ):
        raise RehearsalFailure(f"{workflow} did not reach completed terminal state.")
    return payload


def selected_plate_state(ctx: RunContext) -> dict[str, Any]:
    assert ctx.descriptor_path is not None
    payload = command_json(
        ctx,
        "selected_plate_state",
        mcp_cmd(
            UiGetStateSurfaceCapability.cli_command,
            PlateManagerStateSurfaceIdentityDeclaration.require_value(),
            "--selection-mode",
            "selected",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    state = first_payload(payload).get("payload")
    if not isinstance(state, dict):
        raise RehearsalFailure("Plate manager state payload missing.")
    rows = state.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RehearsalFailure("Expected one selected plate row.")
    row = rows[0]
    if not isinstance(row, dict):
        raise RehearsalFailure("Selected plate row has invalid shape.")
    if row.get("plate_root") != str(ctx.plate_dir):
        raise RehearsalFailure("Selected plate row does not point at the demo plate.")
    if row.get("terminal_status") != "complete":
        raise RehearsalFailure(
            "Selected plate did not finish with terminal_status=complete."
        )
    output_root = row.get("output_plate_root")
    if not isinstance(output_root, str) or not Path(output_root).exists():
        raise RehearsalFailure("Output plate root is missing.")
    return row


def selected_output_records(
    ctx: RunContext,
    *,
    artifact_name: str,
    label: str,
) -> list[dict[str, Any]]:
    assert ctx.descriptor_path is not None
    response = command_json(
        ctx,
        label,
        mcp_cmd(
            "selected-plate-files",
            "--target",
            "output",
            "--kind",
            "result",
            "--path-contains",
            artifact_name,
            "--limit",
            "100",
            "--include-previews",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    records = records_from_query_payload(first_payload(response))
    if not records:
        raise RehearsalFailure(
            f"Selected output inventory contains no record for declared artifact {artifact_name!r}."
        )
    return records


def final_workflow_state_revision(workflow_payload: Mapping[str, Any]) -> str:
    for result in reversed(tuple(workflow_payload.get("results") or ())):
        if not isinstance(result, dict):
            continue
        if result.get("tool") != UiGetStateSurfaceCapability.name:
            continue
        payloads = result.get("payloads")
        if not isinstance(payloads, list) or not payloads:
            continue
        state = payloads[0]
        if not isinstance(state, dict):
            continue
        revision = state.get("current_revision_token")
        if isinstance(revision, str) and revision:
            return revision
    raise RehearsalFailure(
        "Completed plate workflow has no final state revision token."
    )


def canonical_component_metadata(
    ctx: RunContext,
    *,
    phase: str,
    workflow_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume one typed selected-plate metadata projection after initialization."""

    assert ctx.descriptor_path is not None
    response = command_json(
        ctx,
        f"{phase}_canonical_rebuilt_metadata",
        mcp_cmd(
            UiInspectSelectedPlateImagesCapability.cli_command,
            "--target",
            "selected",
            "--max-component-values",
            "64",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    inspection = first_payload(
        response,
        UiInspectSelectedPlateImagesCapability.name,
    ).get("inspection")
    if not isinstance(inspection, dict):
        raise RehearsalFailure("Selected plate inspection did not return metadata.")
    components = inspection.get("components")
    if not isinstance(components, list) or not components:
        raise RehearsalFailure("Selected plate inspection has no component metadata.")
    microscope = inspection.get("detected_microscope_type")
    if not isinstance(microscope, str) or not microscope:
        raise RehearsalFailure("Selected plate inspection has no microscope identity.")
    return {
        "phase": phase,
        "plate_state_revision": final_workflow_state_revision(workflow_payload),
        "detected_microscope_type": microscope,
        "handler_class": inspection.get("handler_class"),
        "metadata_handler_class": inspection.get("metadata_handler_class"),
        "components": components,
    }


def component_value_keys(
    metadata: Mapping[str, Any], component_name: str
) -> tuple[str, ...]:
    for component in metadata.get("components") or ():
        if (
            not isinstance(component, dict)
            or component.get("component") != component_name
        ):
            continue
        values = component.get("values")
        if not isinstance(values, list):
            return ()
        return tuple(
            value["key"]
            for value in values
            if isinstance(value, dict) and isinstance(value.get("key"), str)
        )
    return ()


def tree_for_window(
    ctx: RunContext,
    window_id: str,
    *,
    label: str,
    max_nodes: int = 400,
) -> dict[str, Any]:
    response = command_json(
        ctx,
        label,
        mcp_cmd(
            UiGetWidgetTreeCapability.cli_command,
            window_id,
            "--json",
            "--include-tree",
            "--include-non-actionable",
            "--maximum-item-model-nodes",
            "8",
            "--max-depth",
            "32",
            "--max-nodes",
            str(max_nodes),
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    tree = first_payload(response)
    if not isinstance(tree, dict):
        raise RehearsalFailure(f"Widget tree for {window_id!r} is not a mapping.")
    return tree


def visible_window_ids(ctx: RunContext, *, label: str) -> frozenset[str]:
    response = command_json(
        ctx,
        label,
        mcp_cmd(
            UiListWindowsCapability.cli_command,
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    windows = first_payload(response, UiListWindowsCapability.name).get("windows")
    if not isinstance(windows, list):
        raise RehearsalFailure("UI window catalog is unavailable.")
    return window_ids_from_catalog(windows)


def window_ids_from_catalog(windows: list[Any]) -> frozenset[str]:
    return frozenset(
        window_id
        for window in windows
        if isinstance(window, dict)
        and isinstance((window_id := window.get("window_id")), str)
        and window_id
    )


def plate_action_summary(ctx: RunContext, action: PlateManagerAction) -> dict[str, Any]:
    response = command_json(
        ctx,
        f"list_{action.value}_action",
        mcp_cmd(
            UiListActionsCapability.cli_command,
            PlateManagerWidgetIdentity.require_value(),
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    actions = first_payload(response, UiListActionsCapability.name).get("actions")
    if not isinstance(actions, list):
        raise RehearsalFailure("PlateManager action catalog is unavailable.")
    matches = [
        row
        for row in actions
        if isinstance(row, dict)
        and row.get("widget_id") == PlateManagerWidgetIdentity.require_value()
        and row.get("action_id") == action.value
    ]
    if len(matches) != 1:
        raise RehearsalFailure(
            f"Expected one exact PlateManager action {action.value!r}, got {len(matches)}."
        )
    if matches[0].get("enabled") is not True:
        raise RehearsalFailure(f"PlateManager action {action.value!r} is disabled.")
    return matches[0]


def invoke_plate_action(ctx: RunContext, action: PlateManagerAction) -> dict[str, Any]:
    summary = plate_action_summary(ctx, action)
    response = command_json(
        ctx,
        f"invoke_{action.value}_action",
        mcp_call_tool_cmd(
            UiInvokeActionCapability.name,
            {
                "widget_id": PlateManagerWidgetIdentity.require_value(),
                "action_id": action.value,
                "target_scope_ids": summary.get("target_scope_ids") or [],
                "observed_selection_revision_token": summary.get(
                    "selection_revision_token"
                ),
                "require_confirmation": False,
                "connection": ui_connection_arguments(ctx),
            },
        ),
        timeout=30,
    )
    result = first_payload(response, UiInvokeActionCapability.name)
    if result.get("status") != "accepted":
        raise RehearsalFailure(
            f"PlateManager action {action.value!r} was not accepted."
        )
    require_ui_mutation_completed(
        ctx,
        result,
        action_label=f"PlateManager action {action.value!r}",
        completed=False,
        expected_outcome=result["status"],
    )
    return result


def snapshot_window(ctx: RunContext, window_id: str, *, label: str) -> dict[str, Any]:
    response = command_json(
        ctx,
        label,
        mcp_cmd(
            UiSnapshotWindowCapability.cli_command,
            window_id,
            "--capture-scope",
            "window",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    snapshot = first_payload(response, UiSnapshotWindowCapability.name)
    if snapshot.get("captured") is not True:
        raise RehearsalFailure(f"Window snapshot failed for {window_id!r}.")
    return snapshot


def close_window(ctx: RunContext, window_id: str, *, label: str) -> None:
    response = command_json(
        ctx,
        label,
        mcp_call_tool_cmd(
            UiCloseWindowCapability.name,
            {
                "window_id": window_id,
                "connection": ui_connection_arguments(ctx),
            },
        ),
        timeout=30,
    )
    closed = first_payload(response, UiCloseWindowCapability.name)
    if closed.get("closed") is not True:
        raise RehearsalFailure(f"Window {window_id!r} did not close normally.")


def invoke_action_created_window(
    ctx: RunContext,
    action: PlateManagerAction,
    *,
    phase: str,
) -> str:
    before = visible_window_ids(ctx, label=f"{phase}_{action.value}_windows_before")
    invoke_plate_action(ctx, action)
    after = visible_window_ids(ctx, label=f"{phase}_{action.value}_windows_after")
    created = after - before
    if len(created) != 1:
        raise RehearsalFailure(
            f"Exact action {action.value!r} created {len(created)} windows, expected one."
        )
    return next(iter(created))


def plate_viewer_tab_targets(tree: Mapping[str, Any]) -> tuple[str, int, int]:
    image_nodes = [
        node
        for node in nested_widget_nodes(tree)
        if node.get("class_name") == ImageBrowserWidget.__name__
        and node.get("visible") is True
    ]
    if len(image_nodes) != 1:
        raise RehearsalFailure(
            f"Expected one visible {ImageBrowserWidget.__name__}, got {len(image_nodes)}."
        )

    actions = tree.get("actionable_widgets")
    if not isinstance(actions, list):
        raise RehearsalFailure("Selected-plate viewer has no tab selector projection.")
    candidates: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("class_name") != "QTabBar":
            continue
        if "tab_selector" not in tuple(action.get("action_kinds") or ()):
            continue
        if (
            not isinstance(action.get("path_id"), str)
            or action.get("item_count") != 2
            or action.get("current_index") not in {0, 1}
        ):
            continue
        candidates.append(action)
    if len(candidates) != 1:
        raise RehearsalFailure("Selected-plate viewer tab identity is ambiguous.")
    image_index = candidates[0]["current_index"]
    return candidates[0]["path_id"], image_index, 1 - image_index


def snapshot_plate_viewer_consumers(
    ctx: RunContext,
    *,
    window_id: str,
    phase: str,
) -> dict[str, Any]:
    image_tree = tree_for_window(
        ctx,
        window_id,
        label=f"{phase}_inspect_selected_plate_image_browser",
    )
    tab_path_id, image_index, metadata_index = plate_viewer_tab_targets(image_tree)
    image_snapshot = snapshot_window(
        ctx,
        window_id,
        label=f"{phase}_selected_plate_image_browser_snapshot",
    )
    response = command_json(
        ctx,
        f"{phase}_select_metadata_browser_tab",
        mcp_call_tool_cmd(
            UiInvokeWidgetActionCapability.name,
            {
                "window_id": window_id,
                "path_id": tab_path_id,
                "action_kind": "tab_selector",
                "target_index": metadata_index,
                "connection": ui_connection_arguments(ctx),
            },
        ),
        timeout=30,
    )
    invoked = first_payload(response, UiInvokeWidgetActionCapability.name)
    require_ui_mutation_completed(
        ctx,
        invoked,
        action_label="Selected-plate Metadata Browser tab selection",
        completed=invoked.get("invoked") is True,
        expected_outcome="invoked",
    )
    metadata_snapshot = snapshot_window(
        ctx,
        window_id,
        label=f"{phase}_metadata_browser_snapshot",
    )
    close_window(
        ctx,
        window_id,
        label=f"{phase}_close_selected_plate_viewer",
    )
    return {
        "window_id": window_id,
        "tab_path_id": tab_path_id,
        "image_browser": {
            "tab_index": image_index,
            "snapshot": image_snapshot.get("resource"),
        },
        "metadata_browser": {
            "tab_index": metadata_index,
            "snapshot": metadata_snapshot.get("resource"),
        },
    }


def pipeline_step_scope_id(ctx: RunContext, step_index: int) -> str:
    response = command_json(
        ctx,
        "inspect_pipeline_editor_state",
        mcp_cmd(
            UiGetStateSurfaceCapability.cli_command,
            PipelineEditorStateSurfaceIdentityDeclaration.require_value(),
            "--selection-mode",
            "all",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    payload = first_payload(response, UiGetStateSurfaceCapability.name).get("payload")
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list):
        raise RehearsalFailure("Pipeline editor state has no step projections.")
    matches = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("index") == step_index
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("step_scope_id"), str):
        raise RehearsalFailure(f"Pipeline editor has no exact step index {step_index}.")
    return matches[0]["step_scope_id"]


def function_action_bar(tree: Mapping[str, Any]) -> dict[str, Any]:
    nodes = nested_widget_nodes(tree)
    action_bars = [
        node
        for node in nodes
        if node.get("class_name") == DetachableActionBar.__name__
        and node.get("object_name") == "func_action_buttons_container"
        and isinstance(node.get("path"), list)
        and node.get("visible") is True
    ]
    if len(action_bars) != 1:
        raise RehearsalFailure(
            "Function-list action bar identity is unavailable or ambiguous."
        )
    return action_bars[0]


def function_component_button_action(tree: Mapping[str, Any]) -> dict[str, Any]:
    action_bar_path = function_action_bar(tree)["path"]
    actions = tree.get("actionable_widgets")
    if not isinstance(actions, list):
        raise RehearsalFailure("Function-list editor has no actionable projections.")
    buttons = sorted(
        (
            action
            for action in actions
            if isinstance(action, dict)
            and action.get("class_name") == "QPushButton"
            and isinstance(action.get("path"), list)
            and action["path"][: len(action_bar_path)] == action_bar_path
            and "button" in tuple(action.get("action_kinds") or ())
        ),
        key=lambda action: tuple(action["path"]),
    )
    if len(buttons) < 3:
        raise RehearsalFailure(
            "Function-list action bar has no component selector button."
        )
    component_button = buttons[2]
    if component_button.get("enabled") is not True:
        raise RehearsalFailure("Function-list component selector is disabled.")
    if not isinstance(component_button.get("path_id"), str):
        raise RehearsalFailure("Component selector has no exact widget path identity.")
    return component_button


def require_component_selector_dialog(tree: Mapping[str, Any]) -> None:
    root = tree.get("root")
    if not isinstance(root, dict):
        raise RehearsalFailure("Component selector has no projected root widget.")
    if root.get("class_name") != GroupBySelectorDialog.__name__:
        raise RehearsalFailure(
            "Component selector action did not open a GroupBySelectorDialog."
        )


def navigate_exact_field(
    ctx: RunContext,
    *,
    scope_id: str,
    field_path: str,
    phase: str,
) -> None:
    navigate_window(
        ctx,
        window_id=scope_id,
        field_path=field_path,
        label=f"{phase}_navigate_{field_path.replace('.', '_')}",
    )


def exercise_semantic_field(
    ctx: RunContext,
    *,
    scope_id: str,
    field_path: str,
    phase: str,
) -> dict[str, Any]:
    navigate_exact_field(
        ctx,
        scope_id=scope_id,
        field_path=field_path,
        phase=phase,
    )
    snapshot = snapshot_window(
        ctx,
        scope_id,
        label=f"{phase}_snapshot_{field_path.replace('.', '_')}",
    )
    return {
        "window_id": scope_id,
        "field_path": field_path,
        "snapshot": snapshot.get("resource"),
    }


def exercise_function_editor(
    ctx: RunContext,
    *,
    scope_id: str,
    phase: str,
) -> dict[str, Any]:
    field_path = FunctionPatternField.parameter_name()
    navigate_exact_field(
        ctx,
        scope_id=scope_id,
        field_path=field_path,
        phase=phase,
    )
    tree = tree_for_window(
        ctx,
        scope_id,
        label=f"{phase}_inspect_{field_path}",
    )
    action_bar = function_action_bar(tree)
    snapshot = snapshot_window(
        ctx,
        scope_id,
        label=f"{phase}_snapshot_{field_path}",
    )
    return {
        "window_id": scope_id,
        "field_path": field_path,
        "path_id": action_bar["path_id"],
        "snapshot": snapshot.get("resource"),
    }


def exercise_component_selector(
    ctx: RunContext,
    *,
    scope_id: str,
    phase: str,
) -> dict[str, Any]:
    navigate_exact_field(
        ctx,
        scope_id=scope_id,
        field_path=FunctionPatternField.parameter_name(),
        phase=f"{phase}_component_selector",
    )
    tab_tree = tree_for_window(
        ctx,
        scope_id,
        label=f"{phase}_inspect_function_pattern_tab",
    )
    select_structured_tab(
        ctx,
        window_id=scope_id,
        tree=tab_tree,
        tab_label="Function Pattern",
        evidence_label=f"{phase}_select_function_pattern_tab",
    )
    tree = tree_for_window(
        ctx,
        scope_id,
        label=f"{phase}_inspect_component_selector_button",
        max_nodes=2000,
    )
    component_button = function_component_button_action(tree)
    before = visible_window_ids(ctx, label=f"{phase}_component_windows_before")
    response = command_json(
        ctx,
        f"{phase}_open_component_selector",
        mcp_call_tool_cmd(
            UiInvokeWidgetActionCapability.name,
            {
                "window_id": scope_id,
                "path_id": component_button["path_id"],
                "action_kind": "button",
                "connection": ui_connection_arguments(ctx),
            },
        ),
        timeout=30,
    )
    invoked = first_payload(response, UiInvokeWidgetActionCapability.name)
    require_ui_mutation_completed(
        ctx,
        invoked,
        action_label="Exact component-selector action",
        completed=invoked.get("invoked") is True,
        expected_outcome="invoked",
    )
    after = visible_window_ids(ctx, label=f"{phase}_component_windows_after")
    created = after - before
    if len(created) != 1:
        raise RehearsalFailure(
            f"Component-selector action created {len(created)} windows, expected one."
        )
    dialog_window_id = next(iter(created))
    dialog_tree = tree_for_window(
        ctx,
        dialog_window_id,
        label=f"{phase}_inspect_component_selector_dialog",
    )
    require_component_selector_dialog(dialog_tree)
    snapshot = snapshot_window(
        ctx,
        dialog_window_id,
        label=f"{phase}_component_selector_snapshot",
    )
    close_window(
        ctx,
        dialog_window_id,
        label=f"{phase}_close_component_selector",
    )
    return {
        "window_id": dialog_window_id,
        "path_id": component_button["path_id"],
        "snapshot": snapshot.get("resource"),
    }


def validate_rebuilt_metadata_views(
    ctx: RunContext,
    *,
    metadata: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    """Exercise exact consumer routes after one canonical rebuilt-state event."""

    revision = metadata.get("plate_state_revision")
    if not isinstance(revision, str) or not revision:
        raise RehearsalFailure(
            "Canonical rebuilt metadata has no plate-state revision."
        )

    navigate_window(
        ctx,
        window_id=OpenHCSUiWindowId.image_browser,
        label=f"{phase}_navigate_image_browser",
    )
    image_snapshot = snapshot_window(
        ctx,
        OpenHCSUiWindowId.image_browser,
        label=f"{phase}_image_browser_snapshot",
    )

    metadata_window_id = invoke_action_created_window(
        ctx,
        PlateManagerAction.VIEW_METADATA,
        phase=phase,
    )
    plate_viewer = snapshot_plate_viewer_consumers(
        ctx,
        window_id=metadata_window_id,
        phase=phase,
    )

    step_scope_id = pipeline_step_scope_id(ctx, FINAL_DEMO_STEP_ROUTE_INDEX)
    function_field = exercise_function_editor(
        ctx,
        scope_id=step_scope_id,
        phase=phase,
    )
    group_by_field = exercise_semantic_field(
        ctx,
        scope_id=step_scope_id,
        field_path="processing_config.group_by",
        phase=phase,
    )
    component_selector = exercise_component_selector(
        ctx,
        scope_id=step_scope_id,
        phase=phase,
    )
    return {
        "plate_state_revision": revision,
        "surfaces": {
            "image_browser": {
                "window_id": OpenHCSUiWindowId.image_browser,
                "snapshot": image_snapshot.get("resource"),
                "selected_plate": plate_viewer["image_browser"],
            },
            "metadata_browser": {
                "action_id": PlateManagerAction.VIEW_METADATA.value,
                "window_id": metadata_window_id,
                "tab_path_id": plate_viewer["tab_path_id"],
                **plate_viewer["metadata_browser"],
            },
            "function_list_editor": function_field,
            "group_by_component_selector": {
                "field": group_by_field,
                "dialog": component_selector,
            },
        },
    }


def metadata_semantic_projection(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Strip transition evidence while retaining the typed metadata payload."""

    return {
        "detected_microscope_type": metadata.get("detected_microscope_type"),
        "handler_class": metadata.get("handler_class"),
        "metadata_handler_class": metadata.get("metadata_handler_class"),
        "components": metadata.get("components"),
    }


def assert_rebuilt_metadata_cycle(
    baseline: Mapping[str, Any],
    edited: Mapping[str, Any],
    reverted: Mapping[str, Any],
) -> None:
    baseline_channels = component_value_keys(baseline, "channel")
    edited_channels = component_value_keys(edited, "channel")
    reverted_channels = component_value_keys(reverted, "channel")
    if not baseline_channels:
        raise RehearsalFailure(
            "Initialized baseline metadata has no physical channel identities."
        )
    if not (baseline_channels == edited_channels == reverted_channels):
        raise RehearsalFailure(
            "Source-binding edits changed physical plate component metadata."
        )
    revisions = tuple(
        metadata.get("plate_state_revision")
        for metadata in (baseline, edited, reverted)
    )
    if any(not isinstance(revision, str) or not revision for revision in revisions):
        raise RehearsalFailure(
            "Initialized metadata cycle lacks a typed plate-state revision."
        )
    if len(set(revisions)) != len(revisions):
        raise RehearsalFailure(
            "Source-binding save/reinitialize/revert did not rebuild plate state."
        )
    semantic_projections = tuple(
        metadata_semantic_projection(metadata)
        for metadata in (baseline, edited, reverted)
    )
    if any(
        projection != semantic_projections[0] for projection in semantic_projections[1:]
    ):
        raise RehearsalFailure(
            "Source-binding edits changed the physical microscope metadata projection."
        )


def validate_measurement_snapshot(
    ctx: RunContext,
    contracts: DemoArtifactContracts,
) -> dict[str, Any]:
    records = selected_output_records(
        ctx,
        artifact_name=contracts.measurement_name,
        label="inspect_cell_counting_measurement_snapshot",
    )
    snapshots: list[dict[str, Any]] = []
    for record in records:
        preview = record.get("preview")
        if not isinstance(preview, dict):
            continue
        columns = preview.get("csv_columns")
        rows = preview.get("csv_rows")
        if isinstance(columns, list) and columns and isinstance(rows, list) and rows:
            snapshots.append(record)
    if not snapshots:
        raise RehearsalFailure(
            "Cell Counting measurement artifact has no non-empty CSV snapshot."
        )
    preview = snapshots[0]["preview"]
    return {
        "artifact_name": contracts.measurement_name,
        "record_count": len(records),
        "snapshot_count": len(snapshots),
        "columns": preview["csv_columns"],
        "row_count": len(preview["csv_rows"]),
        "path": snapshots[0].get("full_path") or snapshots[0].get("relative_path"),
        "metadata": snapshots[0].get("metadata"),
    }


def results_measurement_projection(
    tree: Mapping[str, Any],
    *,
    measurement_name: str,
) -> dict[str, Any]:
    nodes = nested_widget_nodes(tree)
    roots = [
        node
        for node in nodes
        if node.get("class_name") == LiveMeasurementsWindow.__name__
    ]
    entry_lists = [
        node
        for node in nodes
        if node.get("object_name") == "LiveResultsEntryList"
        and node.get("visible") is True
    ]
    tables = [
        node
        for node in nodes
        if node.get("object_name") == "LiveResultsTable" and node.get("visible") is True
    ]
    if len(roots) != 1 or len(entry_lists) != 1 or len(tables) != 1:
        raise RehearsalFailure(
            "Results window does not expose its canonical measurement list and table."
        )

    text_values: list[str] = []
    for node in nodes:
        for key in ("text", "current_text"):
            value = node.get(key)
            if isinstance(value, str) and value:
                text_values.append(value)
        item_texts = node.get("item_texts")
        if isinstance(item_texts, list):
            text_values.extend(
                value for value in item_texts if isinstance(value, str) and value
            )
    if not any(measurement_name in value for value in text_values):
        raise RehearsalFailure(
            "Results window does not show the declared measurement artifact."
        )
    status_texts = [
        value for value in text_values if "row(s)" in value and "column(s)" in value
    ]
    if not status_texts:
        raise RehearsalFailure(
            "Results measurement table has no populated row/column status."
        )
    return {
        "measurement_name": measurement_name,
        "entry_count": entry_lists[0].get("item_count"),
        "current_entry": entry_lists[0].get("current_text"),
        "status": status_texts[0],
    }


def validate_results_window(
    ctx: RunContext,
    contracts: DemoArtifactContracts,
) -> dict[str, Any]:
    window_id = invoke_action_created_window(
        ctx,
        PlateManagerAction.VIEW_RESULTS,
        phase="runtime_measurements",
    )
    tree = tree_for_window(
        ctx,
        window_id,
        label="inspect_runtime_measurements_results_window",
        max_nodes=3000,
    )
    if tree.get("tree_truncated") is True:
        raise RehearsalFailure("Results window widget projection is truncated.")
    projection = results_measurement_projection(
        tree,
        measurement_name=contracts.measurement_name,
    )
    snapshot = snapshot_window(
        ctx,
        window_id,
        label="runtime_measurements_results_snapshot",
    )
    close_window(
        ctx,
        window_id,
        label="runtime_measurements_close_results_window",
    )
    return {
        "window_id": window_id,
        **projection,
        "snapshot": snapshot.get("resource"),
    }


def component_mapping(record: Mapping[str, Any]) -> dict[str, Any]:
    components = record.get("components")
    return dict(components) if isinstance(components, dict) else {}


def payload_components_align(
    image_records: list[dict[str, Any]],
    roi_records: list[dict[str, Any]],
) -> bool:
    image_components = [component_mapping(record) for record in image_records]
    for roi_record in roi_records:
        roi_components = component_mapping(roi_record)
        if not roi_components:
            continue
        for image_component in image_components:
            shared_keys = roi_components.keys() & image_component.keys()
            if shared_keys and all(
                roi_components[key] == image_component[key] for key in shared_keys
            ):
                return True
    return False


def validate_roi_streaming(
    ctx: RunContext,
    contracts: DemoArtifactContracts,
) -> dict[str, Any]:
    assert ctx.descriptor_path is not None
    response = command_json(
        ctx,
        "stream_cell_counting_roi",
        mcp_cmd(
            "selected-plate-stream",
            "--target",
            "output",
            "--kind",
            "result",
            "--path-contains",
            contracts.roi_name,
            "--limit",
            "100",
            "--viewer-port",
            str(ctx.napari_port),
            "--viewer-transport-mode",
            "ipc",
            "--timeout-seconds",
            "30",
            descriptor=ctx.descriptor_path,
        ),
        timeout=45,
    )
    stream = first_payload(response).get("stream")
    if not isinstance(stream, dict):
        raise RehearsalFailure("Selected-plate ROI stream returned no stream payload.")
    roi_paths = stream.get("streamed_roi_paths")
    if not isinstance(roi_paths, list) or not roi_paths:
        raise RehearsalFailure("Selected-plate ROI stream did not stream ROI paths.")

    viewer_timeout_ms = viewer_command_timeout_ms(ctx)
    state_response = command_json(
        ctx,
        "inspect_roi_viewer_state",
        mcp_cmd(
            "viewer-state",
            "--port",
            str(ctx.napari_port),
            "--transport-mode",
            "ipc",
            "--max-component-values-per-layer",
            "256",
            "--timeout-ms",
            str(viewer_timeout_ms),
        ),
        timeout=30,
    )
    layers = first_payload(state_response).get("layers")
    if not isinstance(layers, list):
        raise RehearsalFailure("Napari viewer state has no layer collection.")
    shape_layers = [
        layer
        for layer in layers
        if isinstance(layer, dict) and "shapes" in (layer.get("data_types") or ())
    ]
    if not shape_layers:
        raise RehearsalFailure("ROI stream did not create a Napari shapes layer.")
    route_key = shape_layers[0].get("route_key")
    if not isinstance(route_key, str) or not route_key:
        raise RehearsalFailure("Napari ROI layer has no route key.")
    roi_response = command_json(
        ctx,
        "summarize_cell_counting_rois",
        mcp_cmd(
            "viewer-rois",
            "--port",
            str(ctx.napari_port),
            route_key,
            "--max-rois",
            "100",
            "--max-examples",
            "10",
            "--transport-mode",
            "ipc",
            "--timeout-ms",
            str(viewer_timeout_ms),
        ),
        timeout=30,
    )
    roi_summary = first_payload(roi_response)
    if (
        not isinstance(roi_summary.get("total_roi_count"), int)
        or roi_summary["total_roi_count"] < 1
    ):
        raise RehearsalFailure("Napari ROI summary has no shapes.")

    payload_response = command_json(
        ctx,
        "inspect_roi_payload_provenance",
        mcp_cmd(
            "viewer-payloads",
            "--port",
            str(ctx.napari_port),
            "--transport-mode",
            "ipc",
            "--no-array-values",
            "--timeout-ms",
            str(viewer_timeout_ms),
        ),
        timeout=30,
    )
    payload_layers = first_payload(payload_response).get("layers")
    if not isinstance(payload_layers, list):
        raise RehearsalFailure("Napari payload inspection returned no layers.")
    all_records = [
        record
        for layer in payload_layers
        if isinstance(layer, dict)
        for record in (layer.get("payloads") or ())
        if isinstance(record, dict)
    ]
    roi_records = [
        record
        for record in all_records
        if record.get("route_key") == route_key and record.get("shape_payloads")
    ]
    image_records = [
        record
        for record in all_records
        if record.get("data_type") == "image" and record.get("components")
    ]
    if not roi_records or not all(
        record.get("path") and component_mapping(record) for record in roi_records
    ):
        raise RehearsalFailure("Napari ROI payloads lack component or path provenance.")
    if not payload_components_align(image_records, roi_records):
        raise RehearsalFailure(
            "Napari ROI payload components do not align with image payloads."
        )
    return {
        "artifact_name": contracts.roi_name,
        "route_key": route_key,
        "paths": roi_paths,
        "streamed_roi_count": len(roi_paths),
        "roi_count": roi_summary["total_roi_count"],
        "roi_payload_count": len(roi_records),
    }


def viewer_command_timeout_ms(ctx: RunContext) -> int:
    return min(
        ctx.viewer_timeout_ms,
        VIEWER_COMMAND_TIMEOUT_MS,
    )


def layer_producer_identities(layer: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    producers = layer.get("producer_identities")
    if not isinstance(producers, (list, tuple)):
        return ()
    return tuple(producer for producer in producers if isinstance(producer, dict))


def final_demo_layers(layers: list[Any]) -> list[dict[str, Any]]:
    layer_dicts = [layer for layer in layers if isinstance(layer, dict)]
    if not layer_dicts:
        raise RehearsalFailure("Napari viewer has no layers.")
    matches = []
    for layer in layer_dicts:
        if any(
            producer.get("origin") == "pipeline"
            and producer.get("pipeline_position") == FINAL_DEMO_STEP_ROUTE_INDEX
            for producer in layer_producer_identities(layer)
        ):
            matches.append(layer)
    if not matches:
        raise RehearsalFailure(
            "Napari viewer has no layer produced by pipeline position "
            f"{FINAL_DEMO_STEP_ROUTE_INDEX}."
        )
    return matches


def fixed_demo_axis_indices(
    layer: dict[str, Any], *, well_index: int | None = None
) -> dict[str, int]:
    axes = set(_layer_axis_names(layer))
    axis_indices: dict[str, int] = {}
    if "site" in axes:
        axis_indices["site"] = 0
    for z_axis in ("z_index", "z", "zstep"):
        if z_axis in axes:
            axis_indices[z_axis] = 0
            break
    if well_index is not None and "well" in axes:
        axis_indices["well"] = well_index
    return axis_indices


def _layer_axis_names(layer: dict[str, Any]) -> tuple[str, ...]:
    axis_names: list[str] = []
    for field_name in ("stack_axes", "axis_labels"):
        values = layer.get(field_name)
        if isinstance(values, (list, tuple)):
            axis_names.extend(str(value) for value in values if isinstance(value, str))
    axis_component_values = layer.get("axis_component_values")
    if isinstance(axis_component_values, dict):
        axis_names.extend(str(key) for key in axis_component_values)
    return tuple(dict.fromkeys(axis_names))


def well_axis_count(layer: dict[str, Any]) -> int:
    values = layer.get("axis_component_values")
    if isinstance(values, dict):
        well_values = values.get("well")
        if isinstance(well_values, (list, tuple)):
            return len(well_values)
    component_values = layer.get("component_values")
    if not isinstance(component_values, (list, tuple)):
        return 0
    wells = {
        str(record.get("well"))
        for record in component_values
        if isinstance(record, dict) and record.get("well") is not None
    }
    return len(wells)


def axis_index_args(axis_indices: Mapping[str, int]) -> list[str]:
    args: list[str] = []
    for axis_name, axis_index in axis_indices.items():
        args.extend(["--axis-index", f"{axis_name}={axis_index}"])
    return args


def sample_viewer_image_bounded(
    ctx: RunContext,
    *,
    route_key: str,
    axis_indices: Mapping[str, int],
    viewer_timeout_ms: int,
) -> dict[str, Any]:
    response = command_json(
        ctx,
        "sample_napari_final_image_bounded",
        mcp_cmd(
            SampleViewerWindowImageCapability.cli_command,
            "--port",
            str(ctx.napari_port),
            "--transport-mode",
            "ipc",
            "--route-key",
            route_key,
            *axis_index_args(axis_indices),
            "--height",
            str(BOUNDED_SAMPLE_EDGE),
            "--width",
            str(BOUNDED_SAMPLE_EDGE),
            "--include-array-values",
            "--max-array-elements",
            str(BOUNDED_SAMPLE_MAX_ELEMENTS),
            "--timeout-ms",
            str(viewer_timeout_ms),
        ),
        timeout=30,
    )
    sample = first_payload(response, SampleViewerWindowImageCapability.name)
    records = sample.get("records")
    if (
        sample.get("observed") is not True
        or sample.get("sample_protocol_supported") is not True
        or not isinstance(records, list)
        or not records
        or not isinstance(sample.get("sample_included_count"), int)
        or sample["sample_included_count"] < 1
    ):
        raise RehearsalFailure(
            "Final viewer image did not return a bounded protocol sample."
        )
    sampled_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        value_summary = record.get("array_value_summary")
        if (
            not isinstance(value_summary, dict)
            or value_summary.get("included") is not True
        ):
            continue
        element_count = bounded_element_count(value_summary.get("shape"))
        if (
            element_count is None
            or element_count < 1
            or element_count > BOUNDED_SAMPLE_MAX_ELEMENTS
        ):
            raise RehearsalFailure(
                "Viewer pixel sample exceeded its explicit element budget."
            )
        sampled_records.append(
            {
                "payload_route_key": record.get("payload_route_key"),
                "layer_route_key": record.get("layer_route_key"),
                "path": record.get("path"),
                "summary": record.get("summary"),
                "sample_shape": value_summary.get("shape"),
                "sample_element_count": element_count,
            }
        )
    if not sampled_records:
        raise RehearsalFailure(
            "Viewer sample reported included data without a bounded sample record."
        )
    return {
        "route_key": route_key,
        "axis_indices": dict(axis_indices),
        "record_count": sample.get("record_count"),
        "sample_included_count": sample.get("sample_included_count"),
        "records": sampled_records,
    }


def validate_viewer(ctx: RunContext) -> dict[str, Any]:
    viewer_timeout_ms = viewer_command_timeout_ms(ctx)
    state = command_json(
        ctx,
        "napari_viewer_state",
        mcp_cmd(
            "viewer-state",
            "--port",
            str(ctx.napari_port),
            "--transport-mode",
            "ipc",
            "--max-component-values-per-layer",
            "256",
            "--timeout-ms",
            str(viewer_timeout_ms),
        ),
        timeout=30,
    )
    state_payload = first_payload(state)
    layers = state_payload.get("layers")
    if not isinstance(layers, list) or not layers:
        raise RehearsalFailure("Napari viewer has no layers.")
    final_layers = final_demo_layers(layers)
    main_image_layers = [
        candidate
        for candidate in final_layers
        if any(
            producer.get("output_kind") == "main"
            for producer in layer_producer_identities(candidate)
        )
        and "image" in tuple(candidate.get("data_types") or ())
    ]
    if len(main_image_layers) != 1:
        raise RehearsalFailure(
            "Final pipeline step must expose exactly one main image layer, found "
            f"{len(main_image_layers)}."
        )
    layer = main_image_layers[0]
    route_key = layer.get("route_key")
    if not isinstance(route_key, str):
        raise RehearsalFailure("Final Napari viewer layer route key missing.")
    visible_route_keys = [
        route
        for final_layer in final_layers
        if isinstance(route := final_layer.get("route_key"), str)
    ]
    if not visible_route_keys:
        raise RehearsalFailure("Final Napari viewer layers have no route keys.")

    fixed_axes = fixed_demo_axis_indices(layer, well_index=0)
    isolation_args = [
        "isolate-viewer",
        "--port",
        str(ctx.napari_port),
        "--transport-mode",
        "ipc",
        "--selected-route-key",
        route_key,
        *axis_index_args(fixed_axes),
        "--timeout-ms",
        str(viewer_timeout_ms),
        *visible_route_keys,
    ]
    isolation = command_json(
        ctx,
        "isolate_napari_final_step_layer",
        mcp_cmd(*isolation_args),
        timeout=30,
    )
    isolation_payload = first_payload(isolation)
    if isolation_payload.get("applied") is not True:
        raise RehearsalFailure("Napari final layer isolation did not apply.")

    viewer_payload = command_json(
        ctx,
        "validate_napari_final_step_viewer",
        mcp_cmd(
            ValidateViewerWindowStateCapability.cli_command,
            "--port",
            str(ctx.napari_port),
            "--transport-mode",
            "ipc",
            "--route-key",
            route_key,
            "--expected-layer-count",
            "1",
            "--required-component-label",
            "well",
            "--required-component-label",
            "site",
            "--required-component-label",
            "channel",
            "--required-component-label",
            "z_index",
            "--require-nonzero-payloads",
            "--include-state",
            "--timeout-ms",
            str(viewer_timeout_ms),
        ),
        timeout=30,
    )
    validation = first_payload(viewer_payload)
    if validation.get("valid") is not True:
        raise RehearsalFailure(
            "Napari final step viewer validation did not return valid=True."
        )

    command_json(
        ctx,
        "navigate_napari_layer",
        mcp_cmd(
            "navigate-viewer",
            "--port",
            str(ctx.napari_port),
            "--transport-mode",
            "ipc",
            "--route-key",
            route_key,
            *axis_index_args(fixed_axes),
            "--visible",
            "--selected",
            "--timeout-ms",
            str(viewer_timeout_ms),
        ),
        timeout=30,
    )

    bounded_sample = sample_viewer_image_bounded(
        ctx,
        route_key=route_key,
        axis_indices=fixed_axes,
        viewer_timeout_ms=viewer_timeout_ms,
    )

    scrolled_well_count = scroll_well_axis(ctx, route_key, layer, viewer_timeout_ms)

    payloads = command_json(
        ctx,
        "napari_viewer_payloads",
        mcp_cmd(
            "viewer-payloads",
            "--port",
            str(ctx.napari_port),
            "--transport-mode",
            "ipc",
            "--route-key",
            route_key,
            "--no-array-values",
            "--timeout-ms",
            str(viewer_timeout_ms),
        ),
        timeout=30,
    )
    payload_records = []
    for layer in first_payload(payloads).get("layers") or ():
        if isinstance(layer, dict):
            payload_records.extend(
                item for item in layer.get("payloads") or () if isinstance(item, dict)
            )
    if not payload_records:
        raise RehearsalFailure("Napari payload inspection returned no payload records.")
    if not any(
        record.get("components") and record.get("path") for record in payload_records
    ):
        raise RehearsalFailure(
            "Napari payload records lack provenance/component context."
        )
    return {
        "route_key": route_key,
        "visible_route_keys": visible_route_keys,
        "layer_title": layer.get("title"),
        "fixed_axes": fixed_axes,
        "scrolled_well_count": scrolled_well_count,
        "payload_record_count": len(payload_records),
        "bounded_sample": bounded_sample,
    }


def scroll_well_axis(
    ctx: RunContext,
    route_key: str,
    layer: dict[str, Any],
    viewer_timeout_ms: int,
) -> int:
    well_count = well_axis_count(layer)
    if well_count <= 1:
        return 0
    scroll_count = min(well_count, 4)
    for well_index in range(scroll_count):
        command_json(
            ctx,
            f"scroll_napari_well_{well_index + 1:02d}",
            mcp_cmd(
                "navigate-viewer",
                "--port",
                str(ctx.napari_port),
                "--transport-mode",
                "ipc",
                "--route-key",
                route_key,
                *axis_index_args(fixed_demo_axis_indices(layer, well_index=well_index)),
                "--visible",
                "--selected",
                "--timeout-ms",
                str(viewer_timeout_ms),
            ),
            timeout=30,
        )
    return scroll_count


def nested_widget_nodes(tree: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    root = tree.get("root")
    if not isinstance(root, dict):
        raise RehearsalFailure("Widget projection did not include its structured tree.")
    nodes: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        nodes.append(node)
        for child in node.get("children") or ():
            if isinstance(child, dict):
                visit(child)

    visit(root)
    return tuple(nodes)


def structured_tab_target(
    tree: Mapping[str, Any],
    tab_label: str,
) -> tuple[str, int]:
    actions = tree.get("actionable_widgets")
    if not isinstance(actions, list):
        raise RehearsalFailure("Widget projection has no actionable tab selectors.")
    targets: set[tuple[str, int]] = set()
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("class_name") != "QTabBar":
            continue
        if "tab_selector" not in tuple(action.get("action_kinds") or ()):
            continue
        path_id = action.get("path_id")
        item_texts = action.get("item_texts")
        if (
            not isinstance(path_id, str)
            or not isinstance(item_texts, list)
            or item_texts.count(tab_label) != 1
        ):
            continue
        targets.add((path_id, item_texts.index(tab_label)))
    if not targets:
        raise RehearsalFailure(
            f"Window has no exact structured {tab_label!r} tab selector."
        )
    if len(targets) != 1:
        raise RehearsalFailure(
            f"Structured {tab_label!r} tab selector identity is ambiguous."
        )
    return next(iter(targets))


def select_structured_tab(
    ctx: RunContext,
    *,
    window_id: str,
    tree: Mapping[str, Any],
    tab_label: str,
    evidence_label: str,
) -> tuple[str, int]:
    tab_path_id, target_index = structured_tab_target(tree, tab_label)
    response = command_json(
        ctx,
        evidence_label,
        mcp_call_tool_cmd(
            UiInvokeWidgetActionCapability.name,
            {
                "window_id": window_id,
                "path_id": tab_path_id,
                "action_kind": "tab_selector",
                "target_index": target_index,
                "connection": ui_connection_arguments(ctx),
            },
        ),
        timeout=30,
    )
    invoked = first_payload(response, UiInvokeWidgetActionCapability.name)
    require_ui_mutation_completed(
        ctx,
        invoked,
        action_label=f"Structured {tab_label} tab selection",
        completed=invoked.get("invoked") is True,
        expected_outcome="invoked",
    )
    return tab_path_id, target_index


def validate_artifact_tab(
    ctx: RunContext,
    contracts: DemoArtifactContracts,
    *,
    require_runtime_provenance: bool,
    runtime_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    step_scope_id = pipeline_step_scope_id(ctx, FINAL_DEMO_STEP_ROUTE_INDEX)
    navigate_window(
        ctx,
        window_id=step_scope_id,
        field_path=FunctionPatternField.parameter_name(),
        label="navigate_artifact_step_editor",
    )

    tree = tree_for_window(ctx, step_scope_id, label="inspect_artifact_tab_structure")
    tab_path_id, target_index = select_structured_tab(
        ctx,
        window_id=step_scope_id,
        tree=tree,
        tab_label="Artifacts",
        evidence_label="select_artifact_tab",
    )

    selected_tree = tree_for_window(
        ctx,
        step_scope_id,
        label="inspect_selected_artifact_tab_structure",
    )
    visible_artifact_nodes = [
        node
        for node in nested_widget_nodes(selected_tree)
        if node.get("class_name") == ArtifactPlanViewWidget.__name__
        and node.get("visible") is True
    ]
    if len(visible_artifact_nodes) != 1:
        raise RehearsalFailure(
            "Structured Artifact tab target is not visible after selection."
        )
    if require_runtime_provenance and (
        not runtime_paths or not all(Path(path).exists() for path in runtime_paths)
    ):
        raise RehearsalFailure(
            "Typed runtime artifact records do not expose materialized provenance paths."
        )
    snapshot = snapshot_window(
        ctx,
        step_scope_id,
        label=(
            "runtime_artifact_tab_snapshot"
            if require_runtime_provenance
            else "compiled_artifact_tab_snapshot"
        ),
    )
    return {
        "window_id": step_scope_id,
        "tab_path_id": tab_path_id,
        "tab_index": target_index,
        "declared_output_names": [contracts.measurement_name, contracts.roi_name],
        "runtime_provenance": require_runtime_provenance,
        "runtime_paths": list(runtime_paths),
        "snapshot": snapshot.get("resource"),
    }


def snapshot_windows(ctx: RunContext) -> None:
    assert ctx.descriptor_path is not None
    viewer_timeout_ms = viewer_command_timeout_ms(ctx)
    command_json(
        ctx,
        "ui_windows",
        mcp_cmd(
            UiListWindowsCapability.cli_command,
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    command_json(
        ctx,
        "plate_manager_snapshot",
        mcp_cmd(
            UiSnapshotWindowCapability.cli_command,
            OpenHCSUiWindowId.plate_manager,
            "--capture-scope",
            "window",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=30,
    )
    command_json(
        ctx,
        "napari_snapshot",
        mcp_cmd(
            "snapshot-viewer",
            "--port",
            str(ctx.napari_port),
            "--transport-mode",
            "ipc",
            "--timeout-ms",
            str(viewer_timeout_ms),
        ),
        timeout=30,
    )


def run_one(
    args: argparse.Namespace,
    index: int,
    session_dir: Path,
    fresh: bool,
    owned_contexts: list[RunContext],
    ui_owner: RunContext | None,
) -> dict[str, Any]:
    run_id = f"run{index}_{now_id()}"
    run_dir = resolved_run_directory(session_dir, index)
    ctx = RunContext(
        index=index,
        run_id=run_id,
        run_dir=run_dir,
        plate_dir=args.plate_dir.expanduser().resolve(strict=False),
        output_plate_dir=run_dir / "outputs",
        source_path=run_dir / "orchestrator_config.py",
        descriptor_dir=run_dir / "ui_bridge",
        napari_port=args.napari_port,
        zmq_port=args.zmq_port,
        viewer_timeout_ms=args.viewer_timeout_ms,
    )
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    owned_contexts.append(ctx)

    started_at = time.perf_counter()
    mcp_client = McpDevClient(PYTHON)
    runtime_endpoint: dict[str, Any] | None = None
    try:
        step_start = time.perf_counter()
        ctx.mcp_client = mcp_client.start()
        ctx.steps.append(
            StepRecord(
                "initialize_persistent_mcp_session",
                time.perf_counter() - step_start,
                True,
            )
        )
        if fresh:
            if ui_owner is not None:
                raise RehearsalFailure(
                    "A fresh rehearsal run cannot reuse an existing UI owner."
                )
            step_start = time.perf_counter()
            assert_no_live_process_conflicts(
                ctx,
                isolated_ui_bridge_port=args.isolated_ui_bridge_port,
            )
            ctx.steps.append(
                StepRecord(
                    "verify_no_live_process_conflicts",
                    time.perf_counter() - step_start,
                    True,
                )
            )
            acquire_runtime_lock(ctx)
            assert_no_live_runtime_conflicts(ctx)
            start_ui(
                ctx,
                isolated_bridge_port=args.isolated_ui_bridge_port,
            )
            wait_for_ui_bridge(ctx, args.ui_start_timeout)
        else:
            descriptor = require_owned_ui_descriptor(ui_owner)
            ctx.descriptor_path = descriptor
            command_json(
                ctx,
                "ui_status_existing",
                mcp_cmd("ui-status", "--timeout-seconds", "20", descriptor=descriptor),
                timeout=30,
            )

        command_json(
            ctx, "mcp_health", mcp_cmd("health", "--timeout-seconds", "20"), timeout=30
        )
        authoring_contexts = {
            kind: inspect_authoring_context(
                ctx,
                kind=kind,
                label=f"tour_authoring_context_{kind}",
            )
            for kind in ("first_use", "ui_visible_workflow", "pipeline")
        }
        ensure_demo_plate(ctx)
        source_inspection = inspect_source_plate_and_sample(ctx)
        discovered_channels = source_inspection["inspection"]["channel_values"]
        ctx.source_channel_values = (
            discovered_channels[0],
            discovered_channels[1],
        )
        contracts, guided_tour = run_guided_tour(
            ctx,
            official30_case=args.official30_case,
        )
        authoring_schemas = inspect_authoring_schemas(ctx)
        if fresh:
            wait_for_runtime(ctx, 45)
            expected_zmq_config = execution_config_from_ui_document(
                exact_config_document_source(
                    ctx,
                    window_id=OpenHCSUiWindowId.global_config,
                    config_type=UIConfig,
                ),
                expected_port=ctx.zmq_port,
            )
            runtime_endpoint = verify_cold_runtime_configuration(
                ctx,
                config=expected_zmq_config,
            )
        else:
            wait_for_runtime(ctx, 20)
            runtime_endpoint = verify_cold_runtime_configuration(
                ctx,
                config=OPENHCS_ZMQ_CONFIG,
            )
        verify_execution_runtime_discovery(ctx)
        baseline_authoring = inspect_and_apply_code_document(
            ctx,
            BASELINE_SOURCE_BINDING,
        )
        baseline_init = selected_workflow(ctx, "init_plate", args.workflow_timeout)
        baseline_metadata = canonical_component_metadata(
            ctx,
            phase="baseline",
            workflow_payload=baseline_init,
        )

        edited_authoring = inspect_and_apply_code_document(
            ctx,
            EDITED_SOURCE_BINDING,
        )
        edited_init = selected_workflow(ctx, "init_plate", args.workflow_timeout)
        edited_metadata = canonical_component_metadata(
            ctx,
            phase="edited",
            workflow_payload=edited_init,
        )
        edited_views = validate_rebuilt_metadata_views(
            ctx,
            metadata=edited_metadata,
            phase="edited",
        )

        reverted_authoring = inspect_and_apply_code_document(
            ctx,
            REVERTED_SOURCE_BINDING,
        )
        reverted_init = selected_workflow(ctx, "init_plate", args.workflow_timeout)
        reverted_metadata = canonical_component_metadata(
            ctx,
            phase="reverted",
            workflow_payload=reverted_init,
        )
        assert_rebuilt_metadata_cycle(
            baseline_metadata,
            edited_metadata,
            reverted_metadata,
        )
        reverted_views = validate_rebuilt_metadata_views(
            ctx,
            metadata=reverted_metadata,
            phase="reverted",
        )
        rebuilt_metadata = {
            "baseline": baseline_metadata,
            "edited": edited_metadata,
            "reverted": reverted_metadata,
            "edited_views": edited_views,
            "reverted_views": reverted_views,
        }
        well_axes = component_value_keys(reverted_metadata, "well")
        if not well_axes:
            raise RehearsalFailure(
                "Rebuilt metadata has no well axis for bounded artifact planning."
            )
        artifact_plan = inspect_compiled_artifact_plan(
            ctx,
            contracts,
            axis_id=well_axes[0],
        )
        compile_workflow = selected_workflow(
            ctx,
            "compile_plate",
            args.workflow_timeout,
        )
        compile_state_revision = final_workflow_state_revision(compile_workflow)
        compiled_artifact_tab = validate_artifact_tab(
            ctx,
            contracts,
            require_runtime_provenance=False,
        )
        ctx.owns_napari_viewer = True
        run_workflow = selected_workflow(ctx, "run_plate", args.workflow_timeout)
        execution_state_revision = final_workflow_state_revision(run_workflow)
        row = selected_plate_state(ctx)
        measurements = validate_measurement_snapshot(ctx, contracts)
        results_window = validate_results_window(ctx, contracts)
        authoring_contexts["viewer_review"] = inspect_authoring_context(
            ctx,
            kind="viewer_review",
            label="tour_authoring_context_viewer_review",
        )
        rois = validate_roi_streaming(ctx, contracts)
        viewer = validate_viewer(ctx)
        artifact_tab = validate_artifact_tab(
            ctx,
            contracts,
            require_runtime_provenance=True,
            runtime_paths=tuple(
                path
                for path in (
                    measurements.get("path"),
                    *(rois.get("paths") or ()),
                )
                if isinstance(path, str) and path
            ),
        )
        snapshot_windows(ctx)
        step_start = time.perf_counter()
        stop_owned_viewer(ctx)
        ctx.steps.append(
            StepRecord(
                "shutdown_owned_napari_viewer",
                time.perf_counter() - step_start,
                True,
            )
        )

        elapsed = time.perf_counter() - started_at
        if elapsed > args.max_run_seconds:
            raise RehearsalFailure(
                f"Run {index} took {elapsed:.1f}s, exceeding {args.max_run_seconds:.1f}s."
            )
        report = {
            "run_id": run_id,
            "elapsed_seconds": elapsed,
            "fresh_processes": fresh,
            "descriptor_path": str(ctx.descriptor_path),
            "plate_dir": str(ctx.plate_dir),
            "output_plate_root": row.get("output_plate_root"),
            "napari_port": ctx.napari_port,
            "isolated_ui_bridge_port": args.isolated_ui_bridge_port,
            "authoring_contexts": authoring_contexts,
            "source_inspection": source_inspection,
            "guided_tour": guided_tour,
            "authoring_schemas": authoring_schemas,
            "contracts": contracts.__dict__,
            "runtime_endpoint": runtime_endpoint,
            "pipeline_authoring": {
                "baseline": baseline_authoring,
                "edited": edited_authoring,
                "reverted": reverted_authoring,
            },
            "rebuilt_metadata": rebuilt_metadata,
            "artifact_plan": artifact_plan,
            "execution": {
                "compile_state_revision": compile_state_revision,
                "execution_state_revision": execution_state_revision,
                "terminal_status": row.get("terminal_status"),
                "plate_root": row.get("plate_root"),
                "output_plate_root": row.get("output_plate_root"),
            },
            "measurement_snapshot": measurements,
            "results_window": results_window,
            "roi_streaming": rois,
            "compiled_artifact_tab": compiled_artifact_tab,
            "artifact_tab": artifact_tab,
            "viewer": viewer,
            "steps": [record.__dict__ for record in ctx.steps],
        }
        report["objective_evidence"] = build_objective_evidence(report)
        write_json(ctx.run_dir / "report.json", report)
        return report
    except Exception as exc:
        write_json(
            ctx.run_dir / "failure.json",
            {
                "run_id": run_id,
                "elapsed_seconds": time.perf_counter() - started_at,
                "fresh_processes": fresh,
                "descriptor_path": (
                    None if ctx.descriptor_path is None else str(ctx.descriptor_path)
                ),
                "error": str(exc),
                "steps": [record.__dict__ for record in ctx.steps],
            },
        )
        raise
    finally:
        mcp_client.close()


def require_owned_ui_descriptor(ui_owner: RunContext | None) -> Path:
    """Return the exact live descriptor owned by this rehearsal session."""

    if ui_owner is None or ui_owner.ui_process is None:
        raise RehearsalFailure("No rehearsal-owned UI is available for reuse.")
    if ui_owner.ui_process.poll() is not None:
        raise RehearsalFailure("The rehearsal-owned UI exited before reuse.")
    descriptor = ui_owner.descriptor_path
    if descriptor is None or not descriptor.is_file():
        raise RehearsalFailure(
            "The rehearsal-owned UI has no live bridge descriptor for reuse."
        )
    return descriptor


def required_report_mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    requirement: str,
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RehearsalFailure(
            f"Objective evidence {requirement!r} is missing mapping {key!r}."
        )
    return value


def build_objective_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless every thesis objective has structured public evidence."""

    contexts = required_report_mapping(
        report,
        "authoring_contexts",
        requirement="progressive agent onboarding",
    )
    required_contexts = (
        "first_use",
        "ui_visible_workflow",
        "pipeline",
        "viewer_review",
    )
    if set(contexts) != set(required_contexts) or any(
        not isinstance(context, Mapping)
        or context.get("kind") != kind
        or context.get("truncated") is not False
        or not context.get("next_action")
        for kind, context in contexts.items()
    ):
        raise RehearsalFailure(
            "Progressive authoring contexts do not expose every exact next action."
        )

    source = required_report_mapping(
        report,
        "source_inspection",
        requirement="source inspection",
    )
    source_summary = required_report_mapping(
        source,
        "inspection",
        requirement="source inspection",
    )
    source_sample = required_report_mapping(
        source,
        "sample",
        requirement="bounded source sampling",
    )
    if (
        not isinstance(source_summary.get("image_count"), int)
        or source_summary["image_count"] < 1
        or not isinstance(source_summary.get("parsed_image_count"), int)
        or source_summary["parsed_image_count"] < 1
        or not isinstance(source_sample.get("sample_element_count"), int)
        or not 1 <= source_sample["sample_element_count"] <= BOUNDED_SAMPLE_MAX_ELEMENTS
        or not isinstance(source_sample.get("statistics_element_count"), int)
        or not isinstance(source_sample.get("statistics_element_budget"), int)
        or not 1
        <= source_sample["statistics_element_count"]
        <= source_sample["statistics_element_budget"]
    ):
        raise RehearsalFailure(
            "Source discovery and bounded pixel evidence are incomplete."
        )

    guided_tour = required_report_mapping(
        report,
        "guided_tour",
        requirement="public discoverability",
    )
    official_source = guided_tour.get("official_source")
    if (
        not isinstance(guided_tour.get("knowledge_document_count"), int)
        or guided_tour["knowledge_document_count"] < 1
        or not isinstance(guided_tour.get("function_contract_id"), str)
        or not guided_tour["function_contract_id"]
        or not isinstance(official_source, Mapping)
        or official_source.get("complete_pipeline_document") is not True
    ):
        raise RehearsalFailure(
            "The guided tour lacks public knowledge, function, or example evidence."
        )

    authoring_schemas = required_report_mapping(
        report,
        "authoring_schemas",
        requirement="authoritative configuration schemas",
    )
    schema_probes = authoring_schemas.get("probes")
    expected_schema_probes = {
        (config_type, path_prefix)
        for config_type, path_prefix, _required_paths in (
            tuple(
                (config_type, None, required_paths)
                for config_type, required_paths in AUTHORING_SCHEMA_ROOT_PROBES
            )
            + AUTHORING_SCHEMA_PROBES
        )
    }
    observed_schema_probes = (
        {
            (probe.get("requested_config_type"), probe.get("path_prefix"))
            for probe in schema_probes
            if isinstance(probe, Mapping)
        }
        if isinstance(schema_probes, list)
        else set()
    )
    if (
        authoring_schemas.get("capability") != DescribeConfigSchemaCapability.name
        or authoring_schemas.get("probe_count") != len(expected_schema_probes)
        or observed_schema_probes != expected_schema_probes
    ):
        raise RehearsalFailure(
            "Authoritative pipeline/step schema coverage is incomplete."
        )

    authoring_cycle = required_report_mapping(
        report,
        "pipeline_authoring",
        requirement="UI-visible pipeline authoring",
    )
    authored_phases = [
        required_report_mapping(
            authoring_cycle,
            phase,
            requirement="UI-visible pipeline authoring",
        )
        for phase in ("baseline", "edited", "reverted")
    ]
    first_aliases = [
        (
            phase["source_bindings"][0]["alias"]
            if isinstance(phase.get("source_bindings"), list)
            and phase["source_bindings"]
            and isinstance(phase["source_bindings"][0], Mapping)
            else None
        )
        for phase in authored_phases
    ]
    if (
        first_aliases[0] == first_aliases[1]
        or first_aliases[0] != first_aliases[2]
        or any(
            not isinstance(phase.get("step_count"), int) or phase["step_count"] < 1
            for phase in authored_phases
        )
    ):
        raise RehearsalFailure(
            "Typed source-binding save/edit/revert evidence is incomplete."
        )
    reverted_authoring = authored_phases[-1]
    discovered_channels = source_summary.get("channel_values")
    authored_bindings = reverted_authoring.get("source_bindings")
    authored_source_channels = (
        [
            binding["selector_components"][0]["value"]
            for binding in authored_bindings
            if isinstance(binding, Mapping)
            and isinstance(binding.get("selector_components"), list)
            and binding["selector_components"]
            and isinstance(binding["selector_components"][0], Mapping)
        ]
        if isinstance(authored_bindings, list)
        else []
    )
    if (
        not isinstance(discovered_channels, list)
        or len(discovered_channels) < 2
        or authored_source_channels[:2] != discovered_channels[:2]
    ):
        raise RehearsalFailure(
            "Authored source selectors do not match publicly inspected channel identities."
        )
    processing = reverted_authoring.get("processing_semantics")
    if (
        not isinstance(processing, list)
        or not any(
            isinstance(step, Mapping) and step.get("variable_components")
            for step in processing
        )
        or not any(
            isinstance(step, Mapping)
            and step.get("group_by") not in (None, "none", "NONE")
            for step in processing
        )
    ):
        raise RehearsalFailure(
            "Pipeline authoring lacks multidimensional variable-component/grouping evidence."
        )
    config_families = reverted_authoring.get("config_families")
    if not isinstance(config_families, Mapping) or not all(
        isinstance(config_families.get(name), str) and config_families[name]
        for name in ("pipeline", "path_planning", "source_bindings", "napari_streaming")
    ):
        raise RehearsalFailure(
            "Pipeline authoring lacks hierarchical configuration evidence."
        )

    rebuilt = required_report_mapping(
        report,
        "rebuilt_metadata",
        requirement="UI-visible source configuration",
    )
    revisions = [
        required_report_mapping(
            rebuilt,
            phase,
            requirement="UI-visible source configuration",
        ).get("plate_state_revision")
        for phase in ("baseline", "edited", "reverted")
    ]
    if len(set(revisions)) != 3 or any(
        not isinstance(revision, str) or not revision for revision in revisions
    ):
        raise RehearsalFailure(
            "UI-visible configuration did not rebuild three exact plate states."
        )
    for view_phase in ("edited_views", "reverted_views"):
        surfaces = required_report_mapping(
            required_report_mapping(
                rebuilt,
                view_phase,
                requirement="UI-visible source configuration",
            ),
            "surfaces",
            requirement="UI-visible source configuration",
        )
        if not {
            "image_browser",
            "metadata_browser",
            "function_list_editor",
            "group_by_component_selector",
        }.issubset(surfaces):
            raise RehearsalFailure(
                "UI-visible configuration lacks one of its structured consumer surfaces."
            )

    contracts = required_report_mapping(
        report,
        "contracts",
        requirement="typed artifact contracts",
    )
    artifact_plan = required_report_mapping(
        report,
        "artifact_plan",
        requirement="compile planning",
    )
    required_outputs = {
        (contracts.get("measurement_name"), "measurements"),
        (contracts.get("roi_name"), "object_labels"),
    }
    planned_output_records = artifact_plan.get("artifact_outputs")
    planned_outputs = {
        (record.get("name"), record.get("kind"))
        for record in (
            planned_output_records if isinstance(planned_output_records, list) else ()
        )
        if isinstance(record, dict)
    }
    if (
        any(name is None for name, _kind in required_outputs)
        or not required_outputs.issubset(planned_outputs)
        or not isinstance(artifact_plan.get("materialization_contract_count"), int)
        or artifact_plan["materialization_contract_count"] < 1
    ):
        raise RehearsalFailure(
            "Compile plan does not prove the discovered typed artifact contracts."
        )
    source_workspace = required_report_mapping(
        artifact_plan,
        "source_workspace",
        requirement="storage-independent compile planning",
    )
    virtual_paths = source_workspace.get("virtual_paths")
    source_paths = source_workspace.get("source_paths")
    source_file_count = source_workspace.get("file_count")
    truncated_file_count = source_workspace.get("truncated_file_count")
    if (
        not isinstance(virtual_paths, list)
        or not virtual_paths
        or not isinstance(source_paths, list)
        or len(source_paths) != len(virtual_paths)
        or not isinstance(source_file_count, int)
        or not isinstance(truncated_file_count, int)
        or source_file_count != len(virtual_paths) + truncated_file_count
    ):
        raise RehearsalFailure(
            "Compile plan lacks exact virtual/physical source-storage evidence."
        )

    execution = required_report_mapping(
        report,
        "execution",
        requirement="compile and execution",
    )
    if execution.get("terminal_status") != "complete" or any(
        not isinstance(execution.get(field_name), str) or not execution[field_name]
        for field_name in (
            "compile_state_revision",
            "execution_state_revision",
            "output_plate_root",
        )
    ):
        raise RehearsalFailure(
            "UI-selected compile and execution lack terminal structured evidence."
        )

    measurement = required_report_mapping(
        report,
        "measurement_snapshot",
        requirement="result validation",
    )
    results_window = required_report_mapping(
        report,
        "results_window",
        requirement="result validation",
    )
    roi_streaming = required_report_mapping(
        report,
        "roi_streaming",
        requirement="result validation",
    )
    if (
        not isinstance(measurement.get("row_count"), int)
        or measurement["row_count"] < 1
        or not isinstance(results_window.get("entry_count"), int)
        or results_window["entry_count"] < 1
        or not isinstance(roi_streaming.get("roi_count"), int)
        or roi_streaming["roi_count"] < 1
    ):
        raise RehearsalFailure(
            "Structured measurement, Results, and ROI validation are incomplete."
        )

    viewer = required_report_mapping(
        report,
        "viewer",
        requirement="viewer streaming",
    )
    viewer_sample = required_report_mapping(
        viewer,
        "bounded_sample",
        requirement="bounded viewer sampling",
    )
    viewer_sample_records = viewer_sample.get("records")
    if (
        not isinstance(viewer.get("payload_record_count"), int)
        or viewer["payload_record_count"] < 1
        or not isinstance(viewer_sample_records, list)
        or not viewer_sample_records
        or any(
            not isinstance(record, Mapping)
            or not isinstance(record.get("sample_element_count"), int)
            or not 1 <= record["sample_element_count"] <= BOUNDED_SAMPLE_MAX_ELEMENTS
            for record in viewer_sample_records
        )
    ):
        raise RehearsalFailure(
            "Viewer validation lacks provenance and bounded pixel evidence."
        )

    requirements = [
        {
            "requirement": "progressive_task_specific_onboarding",
            "public_evidence": list(required_contexts),
            "satisfied": True,
        },
        {
            "requirement": "authoritative_hierarchical_authoring_schemas",
            "public_evidence": [DescribeConfigSchemaCapability.name],
            "satisfied": True,
        },
        {
            "requirement": "source_inspection_and_safe_sampling",
            "public_evidence": [
                InspectPlatePathCapability.name,
                SamplePlateImageCapability.name,
            ],
            "satisfied": True,
        },
        {
            "requirement": "ui_visible_hierarchical_configuration",
            "public_evidence": [
                ORCHESTRATOR_DOCUMENT_ID,
                PlateManagerStateSurfaceIdentityDeclaration.require_value(),
            ],
            "satisfied": True,
        },
        {
            "requirement": "typed_source_bindings_and_multidimensional_pipeline",
            "public_evidence": [
                "pipeline_authoring",
                "rebuilt_metadata",
            ],
            "satisfied": True,
        },
        {
            "requirement": "compile_plan_and_artifact_contracts",
            "public_evidence": [InspectPipelineSourceArtifactPlanCapability.name],
            "satisfied": True,
        },
        {
            "requirement": "storage_independent_execution_and_results",
            "public_evidence": [
                "virtual_paths",
                "source_paths",
                "measurement_snapshot",
                "roi_streaming",
            ],
            "satisfied": True,
        },
        {
            "requirement": "viewer_streaming_and_bounded_visual_validation",
            "public_evidence": [
                ValidateViewerWindowStateCapability.name,
                SampleViewerWindowImageCapability.name,
            ],
            "satisfied": True,
        },
        {
            "requirement": "no_hidden_repository_knowledge",
            "public_evidence": [
                "authoring_contexts",
                "knowledge_document",
                "function_contract",
                "official_source",
            ],
            "satisfied": True,
        },
    ]
    return {
        "complete": True,
        "requirement_count": len(requirements),
        "requirements": requirements,
    }


def write_markdown_report(session_dir: Path, reports: list[dict[str, Any]]) -> None:
    lines = ["# MCP Thesis Demo Live Rehearsal", ""]
    for report in reports:
        lines.append(
            f"- {report['run_id']}: {report['elapsed_seconds']:.2f}s, "
            f"fresh={report['fresh_processes']}, descriptor={report['descriptor_path']}"
        )
        lines.append(f"  output={report['output_plate_root']}")
        lines.append(
            f"  viewer_port={report['napari_port']} route={report['viewer']['route_key']} "
            f"payload_records={report['viewer']['payload_record_count']}"
        )
        objective_evidence = report["objective_evidence"]
        lines.append(
            f"  objective_requirements={objective_evidence['requirement_count']} "
            f"complete={objective_evidence['complete']}"
        )
        for requirement in objective_evidence["requirements"]:
            evidence = ", ".join(requirement["public_evidence"])
            lines.append(
                f"  - {requirement['requirement']}: "
                f"satisfied={requirement['satisfied']} evidence={evidence}"
            )
    lines.append("")
    lines.append("## Step Timings")
    for report in reports:
        lines.append("")
        lines.append(f"### {report['run_id']}")
        for step in report["steps"]:
            lines.append(
                f"- {step['name']}: {step['elapsed_seconds']:.2f}s "
                f"ok={step['ok']} evidence={step.get('evidence_path') or ''}"
            )
    (session_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise RehearsalFailure("--runs must be >= 1")
    assert_git_tracked_clean(args.allow_dirty)
    session_dir = args.demo_root / "rehearsals" / now_id()
    session_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    owned_contexts: list[RunContext] = []
    ui_owner: RunContext | None = None
    try:
        for index in range(1, args.runs + 1):
            fresh = args.fresh_each_run or (args.fresh_first_run and index == 1)
            reports.append(
                run_one(
                    args,
                    index,
                    session_dir,
                    fresh,
                    owned_contexts,
                    ui_owner,
                )
            )
            if fresh:
                ui_owner = owned_contexts[-1]
            if args.fresh_each_run:
                stop_owned_processes(owned_contexts[-1])
                ui_owner = None
        write_json(session_dir / "summary.json", {"reports": reports})
        write_markdown_report(session_dir, reports)
    except Exception as exc:
        write_json(
            session_dir / "failure.json",
            {
                "error": str(exc),
                "reports": reports,
            },
        )
        raise
    finally:
        for ctx in reversed(owned_contexts):
            stop_owned_processes(ctx)

    print(f"rehearsal_session={session_dir}")
    for report in reports:
        print(
            f"{report['run_id']} elapsed={report['elapsed_seconds']:.2f}s "
            f"viewer_port={report['napari_port']} output={report['output_plate_root']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RehearsalFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
