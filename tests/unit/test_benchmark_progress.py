from __future__ import annotations

from pathlib import Path

from benchmark.progress import (
    BenchmarkProgressEventKind,
    iter_progress_events,
    summarize_progress,
)
from openhcs.core.config import Backend


class FakeFileManager:
    def __init__(self, content: str) -> None:
        self.content = content
        self.loaded: list[tuple[Path, str]] = []

    def load(self, file_path: Path, backend: str) -> str:
        self.loaded.append((Path(file_path), backend))
        return self.content


def test_progress_summary_reports_active_case_without_inferring_completion(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "CASE_START cp_tutorial_pixel_based_classification timeout=160",
                "RUNTIME_PROFILE step_filter_source_anchors step=0 step_name=Threshold before=120 after=60",
                "2026-05-06 Starting step 'Threshold' for axis A01",
                "FunctionStep 0 (Threshold) completed for axis A01 in 4.706s (execute=2.665s)",
                "2026-05-06 Starting step 'ColorToGray' for axis A02",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = summarize_progress(log_path)

    assert snapshot.parsed_until_line == 5
    assert snapshot.active_case_name == "cp_tutorial_pixel_based_classification"
    assert snapshot.active_case is not None
    assert snapshot.active_case.current_axis == "A02"
    assert snapshot.active_case.current_step_name == "ColorToGray"
    assert snapshot.active_case.completed_axes == ("A01",)
    assert snapshot.active_case.completed_step_count == 1
    assert snapshot.active_case.finished is False
    assert [
        event.kind
        for event in snapshot.events
        if event.kind is BenchmarkProgressEventKind.SOURCE_ANCHOR_FILTER
    ]


def test_progress_summary_reports_case_results_and_command_status(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "CASE_START cp_tutorial_translocation_final timeout=180",
                "FunctionStep 25 (ExportToDatabase) completed for axis D12 in 0.100s (execute=0.050s)",
                "CASE_RESULT cp_tutorial_translocation_final success=True metrics={'execution_time_seconds': 96.6269, 'peak_memory_mb': 6595.0} error=None",
                "status=0",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = summarize_progress(log_path)

    case = snapshot.cases["cp_tutorial_translocation_final"]
    assert snapshot.active_case_name is None
    assert snapshot.command_status == 0
    assert case.finished is True
    assert case.success is True
    assert case.metrics["execution_time_seconds"] == 96.6269
    assert case.error is None


def test_progress_summary_reports_gnu_time_exit_and_resource_usage(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "CASE_START slow_case timeout=520",
                "Starting step 'MeasureTexture' for axis C01",
                "Command exited with non-zero status 124",
                "\tMaximum resident set size (kbytes): 6936168",
                "\tExit status: 124",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = summarize_progress(log_path)

    assert snapshot.command_status == 124
    assert snapshot.max_rss_kb == 6936168
    assert snapshot.active_case_name == "slow_case"
    assert snapshot.active_case is not None
    assert snapshot.active_case.current_axis == "C01"
    assert snapshot.active_case.current_step_name == "MeasureTexture"
    assert snapshot.active_case.finished is False


def test_iter_progress_events_supports_incremental_line_polling(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "CASE_START one timeout=10",
                "FunctionStep 0 (Threshold) completed for axis A01 in 1.0s (execute=0.5s)",
                "CASE_RESULT one success=False metrics={} error=failed",
            ]
        ),
        encoding="utf-8",
    )

    events = tuple(iter_progress_events(log_path, start_line=1))

    assert [event.kind for event in events] == [
        BenchmarkProgressEventKind.STEP_COMPLETE,
        BenchmarkProgressEventKind.CASE_RESULT,
    ]
    assert events[-1].line_number == 3


def test_progress_summary_can_read_logs_through_filemanager_vfs() -> None:
    content = "CASE_START vfs_case timeout=10\nstatus=124\n"
    filemanager = FakeFileManager(content)

    snapshot = summarize_progress(
        Path("logs/run.log"),
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert filemanager.loaded == [(Path("logs/run.log"), "memory")]
    assert snapshot.command_status == 124
    assert snapshot.max_rss_kb is None
    assert snapshot.cases["vfs_case"].started is True
