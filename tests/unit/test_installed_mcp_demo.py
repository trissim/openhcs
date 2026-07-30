"""Focused gates for the portable installed MCP/Napari demo."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from zmqruntime.config import TransportMode

from openhcs.agent.capabilities import agent_capabilities
from openhcs.constants.constants import AllComponents
from openhcs.core.plate_file_inventory import PlateFileKind
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.mcp import installed_demo
from openhcs.mcp.dev_client import McpDevCommandExecution
from openhcs.processing.presets.pipelines import (
    loose_operaphenix_neurite_outgrowth as neurite_preset,
)


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


def test_feature_enhancement_declaration_defers_native_image_runtimes(
    tmp_path: Path,
) -> None:
    source = """
import sys
from openhcs.processing.backends.cellprofiler.feature_enhancement import (
    enhance_or_suppress_features,
)
assert callable(enhance_or_suppress_features)
execution_modules = {
    "scipy.linalg",
    "scipy.ndimage",
    "scipy.special",
    "skimage",
}
unexpected = sorted(execution_modules.intersection(sys.modules))
assert not unexpected, f"execution runtimes imported by declaration: {unexpected}"
"""

    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_analysis_submodule_import_does_not_load_unrelated_backends(
    tmp_path: Path,
) -> None:
    source = """
import sys
from openhcs.processing.backends.analysis import region_properties
assert region_properties.AnalysisBackendProvider.NUMBA.value == "numba"
prefix = "openhcs.processing.backends.analysis."
unexpected = sorted(
    name
    for name in sys.modules
    if name.startswith(prefix) and name != f"{prefix}region_properties"
)
assert not unexpected, f"unrelated analysis backends imported: {unexpected}"
"""

    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_portable_source_normalization_defers_catalog_and_execution_runtimes(
    tmp_path: Path,
) -> None:
    pipeline_source, _endpoint = installed_demo.build_portable_neurite_source(
        plate_path=tmp_path,
        output_root=tmp_path / "analysis",
        viewer_port=43124,
        source_records=_records(tmp_path),
        viewer=False,
    )
    source_path = tmp_path / "portable_pipeline.py"
    source_path.write_text(pipeline_source, encoding="utf-8")
    probe = """
import sys
from pathlib import Path
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.processing.backends.lib_registry.registry_service import RegistryService

def forbidden_catalog_discovery(cls):
    raise AssertionError("pipeline normalization requested full catalog discovery")

RegistryService.get_all_functions_with_metadata = classmethod(
    forbidden_catalog_discovery
)
baseline_modules = frozenset(sys.modules)
PipelineDocumentAuthority.from_source(Path(sys.argv[1]).read_text(encoding="utf-8"))
execution_prefixes = (
    "centrosome.cpmorphology",
    "centrosome.zernike",
    "scipy.interpolate",
    "scipy.linalg",
    "scipy.ndimage",
    "scipy.special",
    "skimage",
)
unexpected = sorted(
    name
    for name in sys.modules
    if name not in baseline_modules and name.startswith(execution_prefixes)
)
assert not unexpected, f"execution runtimes imported by pipeline source: {unexpected}"
"""

    result = subprocess.run(
        [sys.executable, "-c", probe, str(source_path)],
        cwd=Path(__file__).resolve().parents[2],
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


def test_installed_demo_phase_reporting_preserves_json_stdout(capsys) -> None:
    installed_demo._report_phase("starting MCP session")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Installed demo phase: starting MCP session\n"


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


def test_execute_pipeline_submits_then_polls_declared_job_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {"submissions": []}

    def fake_run_mcp(client, argv, *, tool_name, timeout_seconds):
        observed["submissions"].append(
            {
                "client": client,
                "argv": tuple(argv),
                "tool_name": tool_name,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"status": "submitted", "job_id": "job-1"}

    def fake_poll(client, *, job_id, timeout_seconds=180.0):
        observed.update(
            poll_client=client,
            job_id=job_id,
            poll_timeout_seconds=timeout_seconds,
        )
        return {"status": "complete", "job_id": job_id}

    monkeypatch.setattr(installed_demo, "_run_mcp", fake_run_mcp)
    monkeypatch.setattr(installed_demo, "_poll_execution_job", fake_poll)
    client = object()

    payload = installed_demo._execute_pipeline(
        client,
        plate_path=tmp_path / "plate",
        source_path=tmp_path / "pipeline.py",
        runtime_port=43125,
    )

    assert payload == {"status": "complete", "job_id": "job-1"}
    submissions = observed["submissions"]
    assert isinstance(submissions, list) and len(submissions) == 1
    submission = submissions[0]
    assert submission["client"] is client
    assert submission["tool_name"] == agent_capabilities.submit_pipeline_execution.name
    assert submission["timeout_seconds"] is None
    assert "--submit-timeout-ms" in submission["argv"]
    assert "--no-wait" in submission["argv"]
    assert "--wait-timeout-ms" not in submission["argv"]
    assert observed["poll_client"] is client
    assert observed["job_id"] == "job-1"


def test_execution_status_call_uses_owned_job_request() -> None:
    tool_name = agent_capabilities.get_execution_status.name
    observed: dict[str, object] = {}

    class FakeClient:
        def execute(self, argv, *, timeout_seconds):
            observed.update(argv=tuple(argv), timeout_seconds=timeout_seconds)
            return McpDevCommandExecution(
                argv=tuple(argv),
                payload={
                    "results": [
                        {
                            "tool": tool_name,
                            "mcp_error": False,
                            "payloads": [{"status": "running", "job_id": "job-1"}],
                        }
                    ]
                },
                rendered_output="",
                returncode=0,
                server_stderr_tail=None,
            )

    payload = installed_demo._execution_status_payload(
        FakeClient(),
        request=installed_demo.ExecutionStatusRequest(job_id="job-1"),
    )

    argv = observed["argv"]
    assert payload == {"status": "running", "job_id": "job-1"}
    assert observed["timeout_seconds"] is None
    assert argv[:5] == (
        "--timeout-seconds",
        "10.0",
        "--allow-error-payloads",
        "call",
        tool_name,
    )
    arguments_index = argv.index("--arguments")
    assert json.loads(argv[arguments_index + 1]) == {
        "job_id": "job-1",
        "timeout_ms": 5000,
    }


def test_execution_poll_observes_progress_until_complete(monkeypatch) -> None:
    statuses = iter(("submitted", "running", "complete"))
    calls: list[str] = []

    def fake_status(_client, *, request):
        status = next(statuses)
        calls.append(status)
        return {"status": status, "job_id": request.job_id}

    monkeypatch.setattr(installed_demo, "_execution_status_payload", fake_status)
    monkeypatch.setattr(installed_demo.time, "sleep", lambda _seconds: None)

    payload = installed_demo._poll_execution_job(
        object(),
        job_id="job-1",
        timeout_seconds=1.0,
    )

    assert calls == ["submitted", "running", "complete"]
    assert payload == {"status": "complete", "job_id": "job-1"}
