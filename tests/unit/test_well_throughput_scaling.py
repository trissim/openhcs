from __future__ import annotations

import csv
from pathlib import Path

from benchmark.reports.cppipe_figures import LINEAR_AXIS_BREAK_POLICY
from benchmark.well_throughput_scaling import (
    NativeCellProfilerExecutionBaseline,
    WellThroughputBenchmarkPlan,
    WellThroughputMode,
    WellThroughputObservationKey,
    WellThroughputResult,
    WellThroughputPreset,
    WellThroughputStatus,
    generate_well_throughput_figures,
    native_execution_baselines_from_summary_csv,
    read_well_throughput_csv,
    run_well_throughput_suite,
    well_throughput_plan_from_manifest,
    write_well_throughput_csv,
)


def test_well_throughput_presets_are_paired_modes() -> None:
    plan = WellThroughputBenchmarkPlan.from_presets(
        (
            WellThroughputPreset.WELL_1_THREAD_1,
            WellThroughputPreset.WELLS_8_WORKERS_2,
            WellThroughputPreset.WELLS_12_WORKERS_3,
            WellThroughputPreset.WELLS_16_WORKERS_4,
        )
    )

    assert tuple((mode.well_count, mode.worker_count) for mode in plan.modes) == (
        (1, 1),
        (8, 2),
        (12, 3),
        (16, 4),
    )
    assert tuple(mode.name for mode in plan.modes) == (
        "1w_1t",
        "8w_2c",
        "12w_3c",
        "16w_4c",
    )


def test_well_throughput_axis_plan_preserves_legacy_cross_product() -> None:
    plan = WellThroughputBenchmarkPlan.from_axes(
        well_counts=(12, 8),
        worker_counts=(3, 2),
    )

    assert tuple((mode.well_count, mode.worker_count) for mode in plan.modes) == (
        (8, 2),
        (8, 3),
        (12, 2),
        (12, 3),
    )


def test_native_execution_baselines_from_summary_csv(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case_name", "median_native_execution_seconds"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_name": "Example",
                "median_native_execution_seconds": "2.5",
            }
        )

    baselines = native_execution_baselines_from_summary_csv(path)

    assert baselines == {
        "Example": NativeCellProfilerExecutionBaseline("Example", 2.5)
    }
    assert baselines["Example"].projected_execution_seconds(12) == 30.0


def test_well_throughput_plan_from_manifest_reads_declared_modes(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "path_roots": {},
          "cases": [],
          "well_throughput_modes": ["1w_1t", "8w_2c", "12w_3c", "16w_4c"]
        }
        """,
        encoding="utf-8",
    )

    plan = well_throughput_plan_from_manifest(manifest_path)

    assert plan is not None
    assert tuple(mode.name for mode in plan.modes) == (
        "1w_1t",
        "8w_2c",
        "12w_3c",
        "16w_4c",
    )


def test_requested_well_throughput_axes_override_manifest_modes(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "path_roots": {},
          "cases": [],
          "well_throughput_modes": ["1w_1t", "8w_2c", "12w_3c"]
        }
        """,
        encoding="utf-8",
    )

    plan = WellThroughputBenchmarkPlan.from_requested_modes(
        well_counts=(2,),
        worker_counts=(1,),
        manifest_path=manifest_path,
    )

    assert tuple((mode.name, mode.well_count, mode.worker_count) for mode in plan.modes) == (
        ("2w_1c", 2, 1),
    )


def test_requested_well_throughput_presets_override_axis_modes(
    tmp_path: Path,
) -> None:
    plan = WellThroughputBenchmarkPlan.from_requested_modes(
        presets=(WellThroughputPreset.WELL_1_THREAD_1,),
        well_counts=(2,),
        worker_counts=(1,),
        manifest_path=tmp_path / "missing.json",
    )

    assert tuple(mode.name for mode in plan.modes) == ("1w_1t",)


def test_well_throughput_csv_round_trip_preserves_resume_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "well_throughput.csv"
    row = WellThroughputResult(
        case_name="Example",
        mode_name="12w_3c",
        worker_count=3,
        well_count=12,
        compile_seconds=1.0,
        prepare_seconds=0.0,
        execute_seconds=2.0,
        total_seconds=3.0,
        wells_per_second=6.0,
        successful_wells=12,
            native_single_sample_execution_seconds=10.0,
            projected_native_execution_seconds=120.0,
            projected_execution_speedup=60.0,
            peak_memory_mb=512.0,
    )

    write_well_throughput_csv(path, (row,))

    restored = read_well_throughput_csv(path)
    assert restored == (row,)
    assert WellThroughputObservationKey(
        restored[0].case_name,
        restored[0].mode_name,
    ) == WellThroughputObservationKey("Example", "12w_3c")


def test_well_throughput_csv_reads_legacy_rows_without_status(
    tmp_path: Path,
) -> None:
    path = tmp_path / "well_throughput.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_name",
                "mode_name",
                "worker_count",
                "well_count",
                "compile_seconds",
                "prepare_seconds",
                "execute_seconds",
                "total_seconds",
                "wells_per_second",
                "successful_wells",
                "native_single_sample_execution_seconds",
                "projected_native_execution_seconds",
                "projected_execution_speedup",
                "peak_memory_mb",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_name": "Example",
                "mode_name": "8w_2c",
                "worker_count": "2",
                "well_count": "8",
                "compile_seconds": "1.0",
                "prepare_seconds": "0.0",
                "execute_seconds": "2.0",
                "total_seconds": "3.0",
                "wells_per_second": "4.0",
                "successful_wells": "8",
                "native_single_sample_execution_seconds": "",
                "projected_native_execution_seconds": "",
                "projected_execution_speedup": "",
                "peak_memory_mb": "",
            }
        )

    (row,) = read_well_throughput_csv(path)

    assert row.status is WellThroughputStatus.SUCCESS
    assert row.memory_limit_mb is None
    assert row.error_message is None


def test_memory_limited_result_records_guardrail() -> None:
    baseline = NativeCellProfilerExecutionBaseline("Example", 2.0)
    result = WellThroughputResult.memory_limited(
        case_name="Example",
        mode=WellThroughputMode("16w_4c", 16, 4),
        compile_seconds=1.0,
        prepare_seconds=0.0,
        execute_seconds=0.0,
        total_seconds=3.0,
        peak_memory_mb=2048.0,
        memory_limit_mb=1024.0,
        native_execution_baseline=baseline,
        error_message="worker terminated",
    )

    assert result.status is WellThroughputStatus.MEMORY_LIMIT_EXCEEDED
    assert not result.is_successful()
    assert result.successful_wells == 0
    assert result.projected_native_execution_seconds == 32.0
    assert result.projected_execution_speedup is None
    assert result.memory_limit_mb == 1024.0


def test_failed_result_records_error_without_speedup() -> None:
    baseline = NativeCellProfilerExecutionBaseline("Example", 2.0)
    result = WellThroughputResult.failed(
        case_name="Example",
        mode=WellThroughputMode("8w_2c", 8, 2),
        compile_seconds=1.0,
        prepare_seconds=0.0,
        execute_seconds=0.0,
        total_seconds=3.0,
        peak_memory_mb=1024.0,
        native_execution_baseline=baseline,
        error_message="shape mismatch",
    )

    assert result.status is WellThroughputStatus.ERROR
    assert not result.is_successful()
    assert result.successful_wells == 0
    assert result.projected_native_execution_seconds == 16.0
    assert result.projected_execution_speedup is None
    assert result.error_message == "shape mismatch"


def test_rerun_missing_memory_filters_completed_rows(monkeypatch, tmp_path: Path) -> None:
    from benchmark import well_throughput_scaling

    case = type(
        "Case",
        (),
        {
            "name": "Example",
            "dataset_path": tmp_path / "dataset",
            "cppipe_path": tmp_path / "pipeline.cppipe",
            "pipeline_params": {"openhcs_max_axis_count": 1},
        },
    )()
    completed = WellThroughputResult(
        case_name="Example",
        mode_name="8w_2c",
        worker_count=2,
        well_count=8,
        compile_seconds=1.0,
        prepare_seconds=0.0,
        execute_seconds=2.0,
        total_seconds=3.0,
        wells_per_second=4.0,
        successful_wells=8,
        peak_memory_mb=128.0,
    )
    missing_memory = WellThroughputResult(
        case_name="Example",
        mode_name="12w_3c",
        worker_count=3,
        well_count=12,
        compile_seconds=1.0,
        prepare_seconds=0.0,
        execute_seconds=2.0,
        total_seconds=3.0,
        wells_per_second=6.0,
        successful_wells=12,
        peak_memory_mb=None,
    )
    rerun = WellThroughputResult(
        case_name="Example",
        mode_name="12w_3c",
        worker_count=3,
        well_count=12,
        compile_seconds=1.0,
        prepare_seconds=0.0,
        execute_seconds=1.5,
        total_seconds=2.0,
        wells_per_second=8.0,
        successful_wells=12,
        peak_memory_mb=256.0,
    )

    monkeypatch.setattr(
        well_throughput_scaling,
        "load_comparison_cases",
        lambda _manifest_path: (case,),
    )
    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_run_case_well_throughput(**kwargs):
        calls.append(
            (
                kwargs["case_name"],
                kwargs["mode"].name,
                dict(kwargs["pipeline_params"]),
            )
        )
        return rerun

    monkeypatch.setattr(
        well_throughput_scaling,
        "run_case_well_throughput",
        fake_run_case_well_throughput,
    )

    rows = run_well_throughput_suite(
        tmp_path / "manifest.json",
        output_root=tmp_path / "out",
        well_counts=(),
        worker_counts=(),
        plan=WellThroughputBenchmarkPlan(
            (
                WellThroughputMode("8w_2c", 8, 2),
                WellThroughputMode("12w_3c", 12, 3),
            )
        ),
        existing_results=(completed, missing_memory),
        rerun_missing_memory=True,
    )

    assert calls == [("Example", "12w_3c", {"openhcs_max_axis_count": 1})]
    assert rows == (completed, rerun)


def test_run_suite_reruns_existing_error_rows(monkeypatch, tmp_path: Path) -> None:
    from benchmark import well_throughput_scaling

    case = type(
        "Case",
        (),
        {
            "name": "Example",
            "dataset_path": tmp_path / "dataset",
            "cppipe_path": tmp_path / "pipeline.cppipe",
            "pipeline_params": {"openhcs_max_axis_count": 1},
        },
    )()
    existing_error = WellThroughputResult(
        case_name="Example",
        mode_name="1w_1t",
        worker_count=1,
        well_count=1,
        compile_seconds=0.0,
        prepare_seconds=0.0,
        execute_seconds=0.0,
        total_seconds=0.0,
        wells_per_second=0.0,
        successful_wells=0,
        status=WellThroughputStatus.ERROR,
        error_message="old failure",
    )
    rerun = WellThroughputResult(
        case_name="Example",
        mode_name="1w_1t",
        worker_count=1,
        well_count=1,
        compile_seconds=1.0,
        prepare_seconds=0.0,
        execute_seconds=2.0,
        total_seconds=3.0,
        wells_per_second=1.0,
        successful_wells=1,
        peak_memory_mb=128.0,
    )
    monkeypatch.setattr(
        well_throughput_scaling,
        "load_comparison_cases",
        lambda _manifest_path: (case,),
    )
    monkeypatch.setattr(
        well_throughput_scaling,
        "run_case_well_throughput",
        lambda **_kwargs: rerun,
    )

    rows = run_well_throughput_suite(
        tmp_path / "manifest.json",
        output_root=tmp_path / "out",
        well_counts=(),
        worker_counts=(),
        plan=WellThroughputBenchmarkPlan((WellThroughputMode("1w_1t", 1, 1),)),
        existing_results=(existing_error,),
    )

    assert rows == (rerun,)


def test_run_suite_passes_memory_limit_to_case_runner(monkeypatch, tmp_path: Path) -> None:
    from benchmark import well_throughput_scaling

    case = type(
        "Case",
        (),
        {
            "name": "Example",
            "dataset_path": tmp_path / "dataset",
            "cppipe_path": tmp_path / "pipeline.cppipe",
            "pipeline_params": {"openhcs_max_axis_count": 1},
        },
    )()
    monkeypatch.setattr(
        well_throughput_scaling,
        "load_comparison_cases",
        lambda _manifest_path: (case,),
    )
    observed_limits: list[float | None] = []

    def fake_run_case_well_throughput(**kwargs):
        observed_limits.append(kwargs["max_memory_mb"])
        return WellThroughputResult(
            case_name="Example",
            mode_name=kwargs["mode"].name,
            worker_count=kwargs["mode"].worker_count,
            well_count=kwargs["mode"].well_count,
            compile_seconds=1.0,
            prepare_seconds=0.0,
            execute_seconds=2.0,
            total_seconds=3.0,
            wells_per_second=4.0,
            successful_wells=kwargs["mode"].well_count,
            peak_memory_mb=128.0,
        )

    monkeypatch.setattr(
        well_throughput_scaling,
        "run_case_well_throughput",
        fake_run_case_well_throughput,
    )

    run_well_throughput_suite(
        tmp_path / "manifest.json",
        output_root=tmp_path / "out",
        well_counts=(),
        worker_counts=(),
        plan=WellThroughputBenchmarkPlan((WellThroughputMode("8w_2c", 8, 2),)),
        max_memory_mb=4096.0,
    )

    assert observed_limits == [4096.0]


def test_generate_well_throughput_figures_writes_linear_log_and_points(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "well_throughput.csv"
    rows = (
        WellThroughputResult(
            case_name="CaseA",
            mode_name="1w_1t",
            worker_count=1,
            well_count=1,
            compile_seconds=0.1,
            prepare_seconds=0.0,
            execute_seconds=1.0,
            total_seconds=1.1,
            wells_per_second=1.0,
            successful_wells=1,
            projected_execution_speedup=4.0,
            peak_memory_mb=512.0,
        ),
        WellThroughputResult(
            case_name="CaseB",
            mode_name="16w_4c",
            worker_count=4,
            well_count=16,
            compile_seconds=0.1,
            prepare_seconds=0.0,
            execute_seconds=1.0,
            total_seconds=1.1,
            wells_per_second=16.0,
            successful_wells=16,
            projected_execution_speedup=500.0,
            peak_memory_mb=4096.0,
        ),
    )
    write_well_throughput_csv(csv_path, rows)

    outputs = generate_well_throughput_figures(
        csv_path,
        tmp_path / "figures",
        output_formats=("png",),
    )

    assert {output.name for output in outputs} == {
        "well_throughput_speedup.png",
        "well_throughput_speedup_log.png",
        "well_throughput_average_speedup_points.csv",
        "well_throughput_average_speedup_points.png",
        "well_throughput_average_speedup_points_log.png",
        "well_throughput_peak_memory.png",
        "well_throughput_peak_memory_log.png",
    }
    assert all(output.exists() for output in outputs)


def test_linear_axis_break_policy_handles_single_extreme_outlier() -> None:
    assert LINEAR_AXIS_BREAK_POLICY.range_for((4.0, 500.0)) is not None
