#!/usr/bin/env python3
"""Run the no-fallback MCP thesis live-demo rehearsal.

This script intentionally drives the real PyQt UI bridge and the real ZMQ
compiler/executor path. It is not a headless substitute for the demo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


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
ORCHESTRATOR_DOCUMENT_ID = "plate_manager.orchestrator_config"
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
FINAL_DEMO_STEP_DISPLAY_INDEX = 8
FINAL_DEMO_STEP_ROUTE_INDEX = FINAL_DEMO_STEP_DISPLAY_INDEX - 1
FINAL_DEMO_STEP_NAME = "Cell Counting"
VIEWER_COMMAND_TIMEOUT_MS = 2000


class RehearsalFailure(RuntimeError):
    """Hard failure for an unmet live-demo acceptance criterion."""


@dataclass
class StepRecord:
    name: str
    elapsed_seconds: float
    ok: bool
    evidence_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


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
    descriptor_path: Path | None = None
    ui_process: subprocess.Popen[bytes] | None = None
    zmq_process: subprocess.Popen[bytes] | None = None
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
    parser.add_argument("--reuse-processes", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--plate-dir", type=Path, default=DEFAULT_TEST_PLATE_DIR)
    parser.add_argument("--max-run-seconds", type=float, default=240.0)
    parser.add_argument("--zmq-port", type=int, default=7777)
    parser.add_argument("--napari-port", type=int, default=5555)
    parser.add_argument("--ui-start-timeout", type=float, default=90.0)
    parser.add_argument("--workflow-timeout", type=float, default=180.0)
    parser.add_argument("--viewer-timeout-ms", type=int, default=2000)
    return parser.parse_args()


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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
    env: dict[str, str] | None = None,
    record: bool = True,
) -> dict[str, Any]:
    start = time.perf_counter()
    evidence_path = ctx.run_dir / "commands" / f"{len(ctx.steps):02d}_{label}.json"
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    payload: dict[str, Any]
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        write_json(
            evidence_path,
            {
                "args": args,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "json_error": str(exc),
            },
        )
        if record:
            ctx.steps.append(
                StepRecord(
                    label,
                    elapsed,
                    False,
                    str(evidence_path),
                    {"returncode": proc.returncode},
                )
            )
        raise RehearsalFailure(f"{label} did not return JSON; see {evidence_path}") from exc

    write_json(
        evidence_path,
        {
            "args": args,
            "returncode": proc.returncode,
            "stdout": payload,
            "stderr": proc.stderr,
        },
    )
    ok = proc.returncode == 0 and not response_has_errors(payload)
    if record:
        ctx.steps.append(
            StepRecord(
                label,
                elapsed,
                ok,
                str(evidence_path),
                {"returncode": proc.returncode},
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


def mcp_cmd(command: str, *args: str | Path, descriptor: Path | None = None) -> list[str]:
    cmd = [PYTHON, "-m", "openhcs.mcp.dev_client", command]
    cmd.extend(str(arg) for arg in args)
    if descriptor is not None:
        cmd.extend(["--descriptor-file-path", str(descriptor)])
        cmd.extend(["--timeout-ms", "2000"])
    cmd.extend(["--json"])
    return cmd


def mcp_call_tool_cmd(
    tool_name: str,
    arguments: dict[str, Any],
) -> list[str]:
    return [
        PYTHON,
        "-m",
        "openhcs.mcp.dev_client",
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
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            raise RehearsalFailure(
                "Tracked worktree changes are present. Commit/stash them or pass --allow-dirty."
            )


def matching_pids(needles: tuple[str, ...]) -> list[int]:
    pids: list[int] = []
    own_pid = os.getpid()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == own_pid:
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8",
                errors="replace",
            )
        except OSError:
            continue
        if all(needle in cmdline for needle in needles):
            pids.append(pid)
    return pids


def terminate_processes(pids: list[int], *, timeout: float = 8.0) -> None:
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not [pid for pid in pids if Path(f"/proc/{pid}").exists()]:
            return
        time.sleep(0.2)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_existing_openhcs_processes() -> None:
    terminate_processes(matching_pids(("-m", "openhcs.pyqt_gui")))
    terminate_processes(matching_pids(("-m", "openhcs.runtime.zmq_execution_server_launcher")))
    terminate_processes(matching_pids(("openhcs.runtime.napari_viewer_server",)))


def start_zmq(ctx: RunContext) -> None:
    log_path = ctx.run_dir / "processes" / "zmq_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    ctx.zmq_process = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "openhcs.runtime.zmq_execution_server_launcher",
            "--port",
            str(ctx.zmq_port),
            "--transport-mode",
            "ipc",
            "--log-level",
            "WARNING",
        ],
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def start_ui(ctx: RunContext) -> None:
    log_path = ctx.run_dir / "processes" / "pyqt_ui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    env = os.environ.copy()
    env["OPENHCS_ENABLE_UI_BRIDGE"] = "true"
    env["OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR"] = str(ctx.descriptor_dir)
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


def wait_for_runtime(ctx: RunContext, timeout: float) -> dict[str, Any]:
    started_at = time.perf_counter()
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if ctx.zmq_process is not None and ctx.zmq_process.poll() is not None:
            raise RehearsalFailure("ZMQ server process exited before becoming ready.")
        try:
            payload = command_json(
                ctx,
                "runtime_scan",
                mcp_cmd("runtime-scan", "--timeout-seconds", "20"),
                timeout=30,
                record=False,
            )
            servers = first_payload(payload).get("servers") or ()
            for server in servers:
                connection = server.get("connection") if isinstance(server, dict) else {}
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
        descriptors = sorted(ctx.descriptor_dir.glob("ui_bridge_*.json"), key=lambda p: p.stat().st_mtime)
        if descriptors:
            candidate = descriptors[-1]
            try:
                payload = command_json(
                    ctx,
                    "ui_status",
                    mcp_cmd("ui-status", "--timeout-seconds", "20", descriptor=candidate),
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


def verify_ui_bridge_runtime_scan(ctx: RunContext) -> None:
    if ctx.descriptor_path is None:
        raise RehearsalFailure("Cannot scan UI bridge before descriptor discovery.")
    descriptor = read_json(ctx.descriptor_path)
    connection = descriptor.get("connection")
    if not isinstance(connection, dict) or not isinstance(connection.get("port"), int):
        raise RehearsalFailure("UI bridge descriptor does not expose a numeric port.")
    ui_bridge_port = connection["port"]
    payload = command_json(
        ctx,
        "runtime_scan_with_ui_bridge",
        mcp_cmd(
            "runtime-scan",
            str(ctx.zmq_port),
            str(ui_bridge_port),
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
        raise RehearsalFailure("ZMQ execution server missing from explicit runtime scan.")
    if "OpenHCSUiBridgeServer" not in server_names:
        raise RehearsalFailure("OpenHCS UI bridge server missing from explicit runtime scan.")


def demo_source(ctx: RunContext) -> str:
    plate = str(ctx.plate_dir)
    output_root = str(ctx.output_plate_dir)
    return f"""# Edit this orchestrator configuration and save to apply changes

from openhcs.constants.constants import (
    GroupBy,
    VariableComponents,
)
from openhcs.constants.input_source import InputSource
from openhcs.core.memory import DtypeConversion
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyDtypeConfig,
    LazyNapariStreamingConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyStepMaterializationConfig,
    LazyStepWellFilterConfig,
    NapariVariableSizeHandling,
    PipelineConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.cell_counting_cpu import (
    DetectionMethod,
    count_cells_single_channel,
)
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu
from openhcs.processing.backends.pos_gen.ashlar_main_cpu import ashlar_compute_tile_positions_cpu
from openhcs.processing.backends.processors.numpy_processor import (
    create_composite,
    create_projection,
    stack_percentile_normalize,
)

# MCP thesis demo live rehearsal run: {ctx.run_id}

plate_paths = [
    {plate!r}
]

global_config = GlobalPipelineConfig(
    auto_add_output_plate_to_plate_manager=True
)

per_plate_configs = {{
    {plate!r}: PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            global_output_folder={output_root!r}
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
                    'method': 'max_projection'
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
                    'method': 'max_projection'
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
                    'detection_method': DetectionMethod.WATERSHED,
                    'return_segmentation_mask': True
                }}),
            name='Cell Counting',
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


def ensure_demo_plate(ctx: RunContext) -> None:
    started_at = time.perf_counter()
    if not ctx.plate_dir.is_dir():
        raise RehearsalFailure(f"Existing test plate is missing: {ctx.plate_dir}")
    metadata_path = ctx.plate_dir / "openhcs_metadata.json"
    if not metadata_path.is_file():
        raise RehearsalFailure(f"Existing test plate has no openhcs_metadata.json: {ctx.plate_dir}")
    image_count = sum(
        1
        for path in ctx.plate_dir.rglob("*")
        if path.suffix.lower() in {".tif", ".tiff"}
    )
    if image_count < 1:
        raise RehearsalFailure(f"Existing test plate has no TIFF images: {ctx.plate_dir}")
    ctx.output_plate_dir.mkdir(parents=True, exist_ok=True)
    ctx.source_path.write_text(demo_source(ctx), encoding="utf-8")
    ctx.steps.append(
        StepRecord(
            "verify_existing_test_plate",
            time.perf_counter() - started_at,
            True,
            None,
            {
                "plate_dir": str(ctx.plate_dir),
                "metadata_path": str(metadata_path),
                "image_count": image_count,
                "output_root": str(ctx.output_plate_dir),
            },
        )
    )


def inspect_and_apply_code_document(ctx: RunContext) -> None:
    assert ctx.descriptor_path is not None
    docs = command_json(
        ctx,
        "code_documents",
        mcp_cmd("code-documents", "--timeout-seconds", "20", descriptor=ctx.descriptor_path),
        timeout=30,
    )
    documents = first_payload(docs).get("documents") or ()
    if not any(
        isinstance(doc, dict)
        and doc.get("document_id") == ORCHESTRATOR_DOCUMENT_ID
        and doc.get("writable") is True
        for doc in documents
    ):
        raise RehearsalFailure("Writable plate_manager.orchestrator_config document missing.")

    source_payload = command_json(
        ctx,
        "inspect_orchestrator_document",
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
        raise RehearsalFailure("Orchestrator code document did not expose a revision token.")

    validation = command_json(
        ctx,
        "validate_orchestrator_document",
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
        raise RehearsalFailure("Orchestrator code document validation did not return valid=True.")

    apply_payload = command_json(
        ctx,
        "apply_orchestrator_document",
        mcp_cmd(
            "apply-code-document",
            ORCHESTRATOR_DOCUMENT_ID,
            "--source-file",
            ctx.source_path,
            "--base-revision-token",
            token,
            "--no-confirmation",
            "--snapshot-label",
            f"MCP thesis demo live source {ctx.run_id}",
            "--timeout-seconds",
            "20",
            descriptor=ctx.descriptor_path,
        ),
        timeout=60,
    )
    applied_payload = first_payload(apply_payload)
    if applied_payload.get("applied") is not True or applied_payload.get("outcome") != "applied":
        operation_id = applied_payload.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            raise RehearsalFailure("Orchestrator code document apply did not return an operation id.")
        wait_for_ui_operation(
            ctx,
            operation_id=operation_id,
            expected_outcome="applied",
            timeout=30,
        )

    reread = command_json(
        ctx,
        "reread_orchestrator_document",
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
    if not isinstance(source, str) or ctx.run_id not in source:
        raise RehearsalFailure("Applied orchestrator document does not contain the demo run id.")


def wait_for_ui_operation(
    ctx: RunContext,
    *,
    operation_id: str,
    expected_outcome: str,
    timeout: float,
) -> dict[str, Any]:
    assert ctx.descriptor_path is not None
    deadline = time.monotonic() + timeout
    attempt = 1
    last_payload: dict[str, Any] | None = None
    arguments = {
        "operation_id": operation_id,
        "connection": {
            "descriptor_file_path": str(ctx.descriptor_path),
            "timeout_ms": 2000,
        },
    }
    while time.monotonic() < deadline:
        payload = command_json(
            ctx,
            f"ui_operation_status_{attempt:02d}",
            mcp_call_tool_cmd("openhcs_ui_get_operation_status", arguments),
            timeout=30,
        )
        status = first_payload(payload, "openhcs_ui_get_operation_status")
        last_payload = status
        if status.get("status") == "completed":
            if status.get("outcome") != expected_outcome:
                raise RehearsalFailure(
                    f"UI operation {operation_id} completed with outcome "
                    f"{status.get('outcome')!r}, expected {expected_outcome!r}."
                )
            return status
        if status.get("status") == "failed":
            raise RehearsalFailure(f"UI operation {operation_id} failed: {status}")
        attempt += 1
        time.sleep(0.25)
    raise RehearsalFailure(
        f"Timed out waiting for UI operation {operation_id}; last={last_payload}"
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
    if summary.get("poll_completed") is not True or summary.get("poll_status") != "completed":
        raise RehearsalFailure(f"{workflow} did not reach completed terminal state.")
    return payload


def selected_plate_state(ctx: RunContext) -> dict[str, Any]:
    assert ctx.descriptor_path is not None
    payload = command_json(
        ctx,
        "selected_plate_state",
        mcp_cmd(
            "state-surface",
            "plate_manager.state",
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
        raise RehearsalFailure("Selected plate did not finish with terminal_status=complete.")
    output_root = row.get("output_plate_root")
    if not isinstance(output_root, str) or not Path(output_root).exists():
        raise RehearsalFailure("Output plate root is missing.")
    image_count = len(list(Path(output_root).rglob("*.tif")))
    if image_count < 1:
        raise RehearsalFailure("Output plate does not contain TIFF outputs.")
    return row


def viewer_command_timeout_ms(ctx: RunContext) -> int:
    return min(
        int(
            ctx.viewer_timeout_ms
            if hasattr(ctx, "viewer_timeout_ms")
            else VIEWER_COMMAND_TIMEOUT_MS
        ),
        VIEWER_COMMAND_TIMEOUT_MS,
    )


def final_demo_layer(layers: list[Any]) -> dict[str, Any]:
    final_layers = final_demo_layers(layers)
    image_layers = [
        layer
        for layer in final_layers
        if not layer.get("data_types") or "image" in tuple(layer.get("data_types") or ())
    ]
    return (image_layers or final_layers)[0]


def final_demo_layers(layers: list[Any]) -> list[dict[str, Any]]:
    layer_dicts = [layer for layer in layers if isinstance(layer, dict)]
    image_layers = [
        layer
        for layer in layer_dicts
        if not layer.get("data_types") or "image" in tuple(layer.get("data_types") or ())
    ]
    candidates = image_layers or layer_dicts
    if not candidates:
        raise RehearsalFailure("Napari viewer has no layers.")

    exact_matches = [
        layer
        for layer in layer_dicts
        if _layer_matches_final_demo_step(layer)
    ]
    if exact_matches:
        return exact_matches

    indexed = [
        (step_index, layer)
        for layer in candidates
        if (step_index := _layer_step_index(layer)) is not None
    ]
    if indexed:
        max_step_index = max(step_index for step_index, _layer in indexed)
        return [layer for step_index, layer in indexed if step_index == max_step_index]
    return [candidates[-1]]


def _layer_matches_final_demo_step(layer: dict[str, Any]) -> bool:
    route_key = str(layer.get("route_key") or "")
    title = str(layer.get("title") or "")
    return (
        f"step_{FINAL_DEMO_STEP_ROUTE_INDEX}" in route_key
        or title.startswith(f"{FINAL_DEMO_STEP_DISPLAY_INDEX}. ")
        or FINAL_DEMO_STEP_NAME.lower() in title.lower()
    )


def _layer_step_index(layer: dict[str, Any]) -> int | None:
    route_key = str(layer.get("route_key") or "")
    route_match = re.search(r"(?:^|_)step_(\d+)(?:_|$)", route_key)
    if route_match is not None:
        return int(route_match.group(1))
    title = str(layer.get("title") or "")
    title_match = re.match(r"\s*(\d+)\.", title)
    if title_match is not None:
        return int(title_match.group(1)) - 1
    return None


def fixed_demo_axis_indices(layer: dict[str, Any], *, well_index: int | None = None) -> dict[str, int]:
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
    layer = final_demo_layer(final_layers)
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
            "validate-viewer",
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
        raise RehearsalFailure("Napari final step viewer validation did not return valid=True.")

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
            payload_records.extend(item for item in layer.get("payloads") or () if isinstance(item, dict))
    if not payload_records:
        raise RehearsalFailure("Napari payload inspection returned no payload records.")
    if not any(record.get("components") and record.get("path") for record in payload_records):
        raise RehearsalFailure("Napari payload records lack provenance/component context.")
    return {
        "route_key": route_key,
        "visible_route_keys": visible_route_keys,
        "layer_title": layer.get("title"),
        "fixed_axes": fixed_axes,
        "scrolled_well_count": scrolled_well_count,
        "payload_record_count": len(payload_records),
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
                *axis_index_args(
                    fixed_demo_axis_indices(layer, well_index=well_index)
                ),
                "--visible",
                "--selected",
                "--timeout-ms",
                str(viewer_timeout_ms),
            ),
            timeout=30,
        )
        time.sleep(0.35)
    return scroll_count


def snapshot_windows(ctx: RunContext) -> None:
    assert ctx.descriptor_path is not None
    viewer_timeout_ms = min(
        int(ctx.viewer_timeout_ms if hasattr(ctx, "viewer_timeout_ms") else 2000),
        2000,
    )
    command_json(
        ctx,
        "ui_windows",
        mcp_cmd("windows", "--timeout-seconds", "20", descriptor=ctx.descriptor_path),
        timeout=30,
    )
    command_json(
        ctx,
        "plate_manager_snapshot",
        mcp_cmd(
            "window-snapshot",
            "plate_manager",
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


def run_one(args: argparse.Namespace, index: int, session_dir: Path, fresh: bool) -> dict[str, Any]:
    run_id = f"run{index}_{now_id()}"
    run_dir = session_dir / f"run_{index:02d}"
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
    )
    setattr(ctx, "viewer_timeout_ms", args.viewer_timeout_ms)
    ctx.run_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    try:
        if fresh:
            step_start = time.perf_counter()
            stop_existing_openhcs_processes()
            ctx.steps.append(StepRecord("stop_existing_openhcs_processes", time.perf_counter() - step_start, True))
            start_zmq(ctx)
            wait_for_runtime(ctx, 45)
            start_ui(ctx)
            wait_for_ui_bridge(ctx, args.ui_start_timeout)
        else:
            wait_for_runtime(ctx, 20)
            descriptor = newest_existing_descriptor()
            if descriptor is None:
                raise RehearsalFailure("No existing UI bridge descriptor found.")
            ctx.descriptor_path = descriptor
            command_json(
                ctx,
                "ui_status_existing",
                mcp_cmd("ui-status", "--timeout-seconds", "20", descriptor=descriptor),
                timeout=30,
            )

        command_json(ctx, "mcp_health", mcp_cmd("health", "--timeout-seconds", "20"), timeout=30)
        verify_ui_bridge_runtime_scan(ctx)
        ensure_demo_plate(ctx)
        inspect_and_apply_code_document(ctx)
        selected_workflow(ctx, "init_plate", args.workflow_timeout)
        selected_workflow(ctx, "compile_plate", args.workflow_timeout)
        selected_workflow(ctx, "run_plate", args.workflow_timeout)
        row = selected_plate_state(ctx)
        viewer = validate_viewer(ctx)
        snapshot_windows(ctx)

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
            "viewer": viewer,
            "steps": [record.__dict__ for record in ctx.steps],
        }
        write_json(ctx.run_dir / "report.json", report)
        return report
    except Exception as exc:
        write_json(
            ctx.run_dir / "failure.json",
            {
                "run_id": run_id,
                "elapsed_seconds": time.perf_counter() - started_at,
                "fresh_processes": fresh,
                "descriptor_path": None if ctx.descriptor_path is None else str(ctx.descriptor_path),
                "error": str(exc),
                "steps": [record.__dict__ for record in ctx.steps],
            },
        )
        raise


def newest_existing_descriptor() -> Path | None:
    descriptors = sorted(
        (ROOT / "mcp_outputs").glob("**/ui_bridge_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return descriptors[-1] if descriptors else None


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
    if args.reuse_processes:
        args.fresh_first_run = False

    assert_git_tracked_clean(args.allow_dirty)
    session_dir = args.demo_root / "rehearsals" / now_id()
    session_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    try:
        for index in range(1, args.runs + 1):
            fresh = args.fresh_each_run or (args.fresh_first_run and index == 1)
            reports.append(run_one(args, index, session_dir, fresh))
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
