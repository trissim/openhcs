from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from benchmark.adapters.cellprofiler import (
    CELLPROFILER_EXECUTABLE_ENV,
    CellProfilerAdapter,
)
from benchmark.contracts.tool_adapter import ToolNotInstalledError


def test_cellprofiler_adapter_requires_executable(monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmark.adapters.cellprofiler.shutil.which",
        lambda _name: None,
    )

    with pytest.raises(ToolNotInstalledError, match="CellProfiler executable"):
        CellProfilerAdapter().validate_installation()


def test_cellprofiler_adapter_accepts_executable_env(monkeypatch) -> None:
    commands: list[tuple[str, ...]] = []
    monkeypatch.setenv(CELLPROFILER_EXECUTABLE_ENV, "/opt/cellprofiler/bin/cellprofiler")

    def _run(
        command,
        *,
        capture_output: bool,
        text: bool,
        timeout: float | None,
        check: bool,
    ):
        commands.append(tuple(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="CellProfiler 4.2.8.1\n",
            stderr="",
        )

    monkeypatch.setattr("benchmark.adapters.cellprofiler.subprocess.run", _run)

    adapter = CellProfilerAdapter()
    adapter.validate_installation()

    assert adapter.version == "CellProfiler 4.2.8.1"
    assert commands == [("/opt/cellprofiler/bin/cellprofiler", "--version")]


def test_cellprofiler_adapter_runs_cppipe_headless(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "plate"
    dataset_path.mkdir()
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    commands: list[tuple[str, ...]] = []

    def _run(
        command,
        *,
        capture_output: bool,
        text: bool,
        timeout: float | None,
        check: bool,
    ):
        assert capture_output is True
        assert text is True
        assert check is False
        command = tuple(command)
        commands.append(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="CellProfiler 4.2.6\n",
                stderr="",
            )
        output_root = Path(command[command.index("-o") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "Image.csv").write_text("ImageNumber,Count\n1,2\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("benchmark.adapters.cellprofiler.subprocess.run", _run)

    adapter = CellProfilerAdapter(executable="/usr/bin/cellprofiler")
    adapter.validate_installation()
    result = adapter.run(
        dataset_path=dataset_path,
        pipeline_name="native_reference",
        pipeline_params={
            "dataset_id": "synthetic",
            "cppipe_path": str(cppipe_path),
            "cellprofiler_timeout_seconds": 12,
        },
        metrics=[],
        output_dir=tmp_path / "outputs",
    )

    assert result.success is True
    assert result.provenance["cellprofiler_version"] == "CellProfiler 4.2.6"
    assert result.provenance["pipeline_source"] == "native_cppipe"
    assert result.provenance["csv_output_count"] == 1
    assert {
        record["phase"] for record in result.provenance["phase_timing_records"]
    } == {"RESOLVE_SOURCE", "EXECUTE_NATIVE_CP", "SNAPSHOT_OUTPUTS"}
    assert commands[1] == (
        "/usr/bin/cellprofiler",
        "-c",
        "-r",
        "-p",
        str(cppipe_path),
        "-i",
        str(dataset_path),
        "-o",
        str(result.output_path),
    )
