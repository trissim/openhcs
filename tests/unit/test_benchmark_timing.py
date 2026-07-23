from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.timing import (
    BenchmarkPhase,
    PhaseTimingRecord,
    PhaseTimingTrace,
    write_phase_timing_csv,
    write_phase_timing_jsonl,
)
from openhcs.core.config import Backend


class FakeFileManager:
    def __init__(self) -> None:
        self.saved: dict[Path, tuple[str, str]] = {}
        self.directories: list[tuple[Path, str]] = []

    def ensure_directory(self, directory: Path, backend: str) -> str:
        self.directories.append((Path(directory), backend))
        return str(directory)

    def save(self, data: str, output_path: Path, backend: str) -> None:
        self.saved[Path(output_path)] = (data, backend)


def test_phase_timing_trace_records_typed_phase_payload() -> None:
    trace = PhaseTimingTrace(run_id="run-1", pipeline_name="pipe", tool="OpenHCS")

    trace.record(BenchmarkPhase.COMPILE_DIALECT, seconds=0.25)

    assert trace.payloads() == (
        {
            "run_id": "run-1",
            "pipeline_name": "pipe",
            "tool": "OpenHCS",
            "phase": "COMPILE_DIALECT",
            "seconds": 0.25,
            "cached": False,
        },
    )


def test_phase_timing_writers_can_use_filemanager_vfs() -> None:
    records = (
        PhaseTimingRecord(
            run_id="run-1",
            pipeline_name="pipe",
            tool="OpenHCS",
            phase=BenchmarkPhase.EXECUTE_OPENHCS,
            seconds=1.5,
        ),
    )
    filemanager = FakeFileManager()

    write_phase_timing_jsonl(
        Path("reports/phases.jsonl"),
        records,
        filemanager=filemanager,
        backend=Backend.DISK,
    )
    write_phase_timing_csv(
        Path("reports/phases.csv"),
        records,
        filemanager=filemanager,
        backend=Backend.DISK,
    )

    jsonl, jsonl_backend = filemanager.saved[Path("reports/phases.jsonl")]
    csv_text, csv_backend = filemanager.saved[Path("reports/phases.csv")]
    assert json.loads(jsonl)["phase"] == "EXECUTE_OPENHCS"
    assert csv.DictReader(csv_text.splitlines()).fieldnames == [
        "run_id",
        "pipeline_name",
        "tool",
        "phase",
        "seconds",
        "cached",
    ]
    assert jsonl_backend == Backend.DISK.value
    assert csv_backend == Backend.DISK.value
