from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.cellprofiler_comparison import (
    CellProfilerComparisonCase,
    append_observations_jsonl,
    comparison_observation_from_result,
    load_comparison_cases,
    load_observations_jsonl,
    _discard_openhcs_benchmark_tree,
    write_observations_csv,
    write_phase_timing_csv,
    write_summary_csv,
)
import pytest
from benchmark.contracts.tool_adapter import ToolExecutionError
from benchmark.contracts.tool_adapter import BenchmarkResult
from benchmark.runner import CellProfilerCompatibilityResult


def test_comparison_observation_extracts_execution_only_speedup(
    tmp_path: Path,
) -> None:
    case = CellProfilerComparisonCase(
        name="ExampleHuman",
        dataset_path=tmp_path / "ExampleHuman",
        cppipe_path=tmp_path / "ExampleHuman.cppipe",
        dataset_id="example-human",
    )
    result = CellProfilerCompatibilityResult(
        native_cellprofiler=_benchmark_result(
            "CellProfiler",
            tmp_path / "native",
            "EXECUTE_NATIVE_CP",
            42.0,
        ),
        openhcs_converted=_benchmark_result(
            "OpenHCS",
            tmp_path / "openhcs",
            "EXECUTE_OPENHCS",
            6.0,
            provenance={"equivalence_difference_count": 0},
        ),
    )

    observation = comparison_observation_from_result(
        result,
        case=case,
        suite_id="suite-1",
        repetition=1,
    )

    assert observation.equivalent is True
    assert observation.difference_count == 0
    assert observation.native_cellprofiler.execution_seconds == 42.0
    assert observation.openhcs.execution_seconds == 6.0
    assert observation.speedup == 7.0
    assert observation.parity_accuracy == 1.0


def test_comparison_writers_emit_raw_phase_and_summary_tables(
    tmp_path: Path,
) -> None:
    case = CellProfilerComparisonCase(
        name="ExampleFly",
        dataset_path=tmp_path / "ExampleFly",
        cppipe_path=tmp_path / "ExampleFly.cppipe",
    )
    observation = comparison_observation_from_result(
        CellProfilerCompatibilityResult(
            native_cellprofiler=_benchmark_result(
                "CellProfiler",
                tmp_path / "native",
                "EXECUTE_NATIVE_CP",
                30.0,
            ),
            openhcs_converted=_benchmark_result(
                "OpenHCS",
                tmp_path / "openhcs",
                "EXECUTE_OPENHCS",
                5.0,
                success=False,
                error_message="semantic mismatch",
                provenance={"equivalence_difference_count": 4},
            ),
        ),
        case=case,
        suite_id="suite-1",
        repetition=1,
    )

    append_observations_jsonl(tmp_path / "observations.jsonl", (observation,))
    write_observations_csv(tmp_path / "observations.csv", (observation,))
    write_phase_timing_csv(tmp_path / "phase_timing.csv", (observation,))
    write_summary_csv(tmp_path / "summary.csv", (observation,))

    assert load_observations_jsonl(tmp_path / "observations.jsonl")[0]["speedup"] == 6.0
    observation_rows = _csv_rows(tmp_path / "observations.csv")
    phase_rows = _csv_rows(tmp_path / "phase_timing.csv")
    summary_rows = _csv_rows(tmp_path / "summary.csv")
    assert observation_rows[0]["case_name"] == "ExampleFly"
    assert observation_rows[0]["difference_count"] == "4"
    assert observation_rows[0]["openhcs_error_message"] == "semantic mismatch"
    assert observation_rows[0]["parity_accuracy"] == "0.0"
    assert observation_rows[0]["total_phase_speedup"] == "6.0"
    assert {row["phase"] for row in phase_rows} == {
        "EXECUTE_NATIVE_CP",
        "EXECUTE_OPENHCS",
    }
    assert summary_rows[0]["median_speedup"] == "6.0"
    assert summary_rows[0]["median_total_phase_speedup"] == "6.0"
    assert summary_rows[0]["speedup_target"] == "5.0"
    assert summary_rows[0]["meets_speedup_target"] == "True"
    assert summary_rows[0]["equivalent_count"] == "0"


def test_load_comparison_cases_from_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "ExampleHuman",
                        "dataset_path": "data/human",
                        "cppipe_path": "pipes/human.cppipe",
                        "dataset_id": "human",
                        "microscope_type": "imagexpress",
                        "value_only": True,
                        "equivalence_reference_output_dir": "native/human",
                        "cellprofiler_timeout_seconds": 120,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_comparison_cases(manifest)

    assert cases == (
        CellProfilerComparisonCase(
            name="ExampleHuman",
            dataset_path=Path("data/human"),
            cppipe_path=Path("pipes/human.cppipe"),
            dataset_id="human",
            microscope_type="imagexpress",
            value_only=True,
            equivalence_reference_output_dir=Path("native/human"),
            cellprofiler_timeout_seconds=120.0,
        ),
    )


def test_discard_openhcs_benchmark_tree_requires_marker_and_suite_containment(
    tmp_path: Path,
) -> None:
    suite_root = tmp_path / "suite"
    marked_tree = suite_root / "tool_outputs" / "OpenHCS_case"
    nested_output = marked_tree / "nested" / "result"
    nested_output.mkdir(parents=True)
    (marked_tree / ".openhcs_benchmark_cache.json").write_text(
        "{}",
        encoding="utf-8",
    )

    _discard_openhcs_benchmark_tree(nested_output, suite_output_root=suite_root)

    assert not marked_tree.exists()


def test_discard_openhcs_benchmark_tree_refuses_unmarked_directory(
    tmp_path: Path,
) -> None:
    suite_root = tmp_path / "suite"
    unmarked = suite_root / "tool_outputs" / "OpenHCS_case"
    unmarked.mkdir(parents=True)

    with pytest.raises(ToolExecutionError):
        _discard_openhcs_benchmark_tree(unmarked, suite_output_root=suite_root)

    assert unmarked.exists()


def _benchmark_result(
    tool_name: str,
    output_path: Path,
    execution_phase: str,
    execution_seconds: float,
    *,
    success: bool = True,
    error_message: str | None = None,
    provenance: dict[str, object] | None = None,
) -> BenchmarkResult:
    output_path.mkdir(parents=True)
    return BenchmarkResult(
        tool_name=tool_name,
        dataset_id="dataset",
        pipeline_name="pipeline",
        metrics={"execution_time_seconds": execution_seconds},
        output_path=output_path,
        success=success,
        error_message=error_message,
        provenance={
            **(provenance or {}),
            "phase_timing_records": (
                {
                    "phase": execution_phase,
                    "seconds": execution_seconds,
                },
            ),
        },
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
