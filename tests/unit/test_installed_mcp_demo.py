"""Focused gates for the portable installed MCP/Napari demo."""

from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import sys

from openhcs.agent.capabilities import agent_capabilities
from openhcs.constants.constants import AllComponents
from openhcs.core.config import TransportMode
from openhcs.core.plate_file_inventory import PlateFileKind
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.mcp import installed_demo
from openhcs.mcp.dev_client import McpDevCommandExecution
from openhcs.processing.presets.pipelines import (
    loose_operaphenix_neurite_outgrowth as neurite_preset,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


def _records(tmp_path: Path) -> tuple[dict[str, object], ...]:
    records = []
    for channel in (1, 2):
        source_path = tmp_path / f"A01_s001_w{channel}_z001_t001.tif"
        source_path.touch()
        records.append(
            {
                "kind": PlateFileKind.IMAGE.value,
                "source_path": str(source_path),
                "metadata": {
                    AllComponents.WELL.value: "A01",
                    AllComponents.SITE.value: "1",
                    AllComponents.CHANNEL.value: str(channel),
                    AllComponents.Z_INDEX.value: "1",
                    AllComponents.TIMEPOINT.value: "1",
                },
            }
        )
    return tuple(records)


def test_portable_source_projects_authoritative_neurite_preset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    records = _records(tmp_path)
    original_builder = neurite_preset.build_loose_operaphenix_neurite_pipeline
    observed: dict[str, object] = {}

    def tracked_builder(inputs):
        pipeline_config, pipeline_steps = original_builder(inputs)
        observed.update(
            inputs=inputs,
            pipeline_config=pipeline_config,
            pipeline_steps=tuple(pipeline_steps),
        )
        return pipeline_config, pipeline_steps

    monkeypatch.setattr(
        neurite_preset,
        "build_loose_operaphenix_neurite_pipeline",
        tracked_builder,
    )
    source, endpoint = installed_demo.build_portable_neurite_source(
        plate_path=tmp_path,
        output_root=tmp_path / "analysis",
        viewer_port=43123,
        source_records=records,
        viewer=True,
    )

    document = PipelineDocumentAuthority.from_source(source)
    expected_steps = observed["pipeline_steps"]

    assert document.pipeline_config == observed["pipeline_config"]
    assert isinstance(expected_steps, tuple)
    assert tuple(step.name for step in document.pipeline_steps) == tuple(
        step.name for step in expected_steps
    )
    assert endpoint.port == 43123
    assert endpoint.mode is TransportMode.TCP
    assert all(
        step.napari_streaming_config.enabled
        and step.napari_streaming_config.persistent
        and step.napari_streaming_config.port == 43123
        and step.napari_streaming_config.transport_mode is TransportMode.TCP
        for step in document.pipeline_steps
    )


def test_installed_demo_import_defers_optional_neurite_preset(tmp_path: Path) -> None:
    blocked_module = (
        "openhcs.processing.presets.pipelines.loose_operaphenix_neurite_outgrowth"
    )
    source = f"""
import builtins
from importlib.metadata import distribution

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == {blocked_module!r}:
        raise AssertionError(f"optional preset imported eagerly: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
entry_point = next(
    item
    for item in distribution("openhcs").entry_points
    if item.group == "console_scripts" and item.name == "openhcs-mcp-demo"
)
assert callable(entry_point.load())
"""

    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_headless_portable_source_disables_every_viewer_config(
    tmp_path: Path,
) -> None:
    source, _endpoint = installed_demo.build_portable_neurite_source(
        plate_path=tmp_path,
        output_root=tmp_path / "analysis",
        viewer_port=43124,
        source_records=_records(tmp_path),
        viewer=False,
    )

    document = PipelineDocumentAuthority.from_source(source)

    assert all(
        not step.napari_streaming_config.enabled
        and not step.napari_streaming_config.persistent
        for step in document.pipeline_steps
    )


def test_tcp_allocator_returns_free_runtime_pair() -> None:
    port = installed_demo._free_tcp_port_pair()
    control_port = port + OPENHCS_ZMQ_CONFIG.control_port_offset

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as data_socket:
        data_socket.bind(("127.0.0.1", port))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as control_socket:
        control_socket.bind(("127.0.0.1", control_port))


def test_command_payload_selects_declaration_owned_tool_result() -> None:
    tool_name = agent_capabilities.validate_viewer_window_state.name
    execution = McpDevCommandExecution(
        argv=("validate-viewer", "43123"),
        payload={
            "results": [
                {
                    "tool": tool_name,
                    "mcp_error": False,
                    "payloads": [{"observed": True, "valid": True, "errors": []}],
                }
            ]
        },
        rendered_output="",
        returncode=0,
        server_stderr_tail=None,
    )

    payload = installed_demo._command_payload(execution, tool_name=tool_name)

    assert payload == {"observed": True, "valid": True, "errors": []}


def test_execute_pipeline_relies_on_declared_phase_timeouts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run_mcp(client, argv, *, tool_name, timeout_seconds):
        observed.update(
            client=client,
            argv=tuple(argv),
            tool_name=tool_name,
            timeout_seconds=timeout_seconds,
        )
        return {"status": "complete"}

    monkeypatch.setattr(installed_demo, "_run_mcp", fake_run_mcp)
    client = object()

    payload = installed_demo._execute_pipeline(
        client,
        plate_path=tmp_path / "plate",
        source_path=tmp_path / "pipeline.py",
        runtime_port=43125,
    )

    assert payload == {"status": "complete"}
    assert observed["client"] is client
    assert observed["tool_name"] == agent_capabilities.submit_pipeline_execution.name
    assert observed["timeout_seconds"] is None
    assert "--submit-timeout-ms" in observed["argv"]
    assert "--wait-timeout-ms" in observed["argv"]
