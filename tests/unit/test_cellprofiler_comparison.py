from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.cellprofiler_comparison import (
    CellProfilerComparisonCase,
    NativeCellProfilerReferenceScope,
    append_observations_jsonl,
    comparison_observation_from_result,
    load_comparison_cases,
    load_observations_jsonl,
    run_comparison_suite,
    _discard_openhcs_benchmark_tree,
    _discard_successful_openhcs_benchmark_tree,
    _failed_comparison_observation,
    write_module_coverage_artifacts,
    write_observations_csv,
    write_phase_timing_csv,
    write_summary_csv,
)
from benchmark.cellprofiler_benchmark_cli import _filter_cases_by_name
import pytest
from benchmark.contracts.tool_adapter import ToolExecutionError
from benchmark.contracts.tool_adapter import BenchmarkResult
from benchmark.runner import CellProfilerCompatibilityResult
from openhcs.core.source_schema_workspace import SourceSchemaImageSetSelection


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


def test_cached_native_reference_uses_timeout_as_conservative_speed_lower_bound(
    tmp_path: Path,
) -> None:
    case = CellProfilerComparisonCase(
        name="ExampleTimeoutReference",
        dataset_path=tmp_path / "ExampleTimeoutReference",
        cppipe_path=tmp_path / "ExampleTimeoutReference.cppipe",
        cellprofiler_timeout_seconds=900.0,
    )
    native_output = tmp_path / "native"
    native_output.mkdir()
    result = CellProfilerCompatibilityResult(
        native_cellprofiler=BenchmarkResult(
            tool_name="CellProfiler",
            dataset_id="dataset",
            pipeline_name="pipeline",
            metrics={},
            output_path=native_output,
            success=True,
            provenance={"reused_reference_output": True},
        ),
        openhcs_converted=_benchmark_result(
            "OpenHCS",
            tmp_path / "openhcs",
            "EXECUTE_OPENHCS",
            2.0,
            provenance={"equivalence_difference_count": 0},
        ),
    )

    observation = comparison_observation_from_result(
        result,
        case=case,
        suite_id="suite-1",
        repetition=1,
    )

    assert observation.native_cellprofiler.execution_seconds == 900.0
    assert observation.native_cellprofiler.total_metric_seconds == 900.0
    assert observation.speedup == 450.0
    assert observation.total_phase_speedup == 450.0


def test_required_native_reference_does_not_rerun_cellprofiler(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ExampleFly" / "images"
    dataset_path.mkdir(parents=True)
    cppipe_path = tmp_path / "ExampleFly" / "ExampleFly.cppipe"
    cppipe_path.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    case = CellProfilerComparisonCase(
        name="ExampleFly",
        dataset_path=dataset_path,
        cppipe_path=cppipe_path,
        dataset_id="ExampleFly",
    )

    observations = run_comparison_suite(
        (case,),
        output_root=tmp_path / "suite",
        suite_id="suite-1",
        native_reference_root=tmp_path / "native_refs",
        require_native_reference=True,
        continue_on_error=True,
    )

    assert len(observations) == 1
    assert observations[0].native_cellprofiler.cached is False
    assert observations[0].native_cellprofiler.success is False
    assert "Required cached native CellProfiler reference" in (
        observations[0].native_cellprofiler.error_message or ""
    )


def test_failed_comparison_observation_records_traceback(tmp_path: Path) -> None:
    case = CellProfilerComparisonCase(
        name="ExampleFailure",
        dataset_path=tmp_path / "images",
        cppipe_path=tmp_path / "pipeline.cppipe",
    )

    try:
        raise TypeError("broken metadata")
    except TypeError as exc:
        observation = _failed_comparison_observation(
            case,
            suite_id="suite-1",
            repetition=1,
            error=exc,
        )

    assert observation.native_cellprofiler.error_message is not None
    assert "Traceback (most recent call last)" in (
        observation.native_cellprofiler.error_message
    )
    assert "TypeError: broken metadata" in observation.native_cellprofiler.error_message


def test_native_reference_scope_discovers_completed_reference_with_dataset_alias(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "tutorial" / "translocation_axis1_20260512"
    dataset_path.mkdir(parents=True)
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "Metadata:[module_num:2|enabled:True]",
                "    Extract metadata?:Yes",
                "    Metadata source:File name",
                "    Regular expression to extract from file name:^(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9]+)_w(?P<Channel>[0-9]+)",
                "NamesAndTypes:[module_num:3|enabled:True]",
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                "    Image set matching method:Order",
                "    Assignments count:1",
                "    Single images count:0",
                "    Select the rule criteria:and (file does contain \"\")",
                "    Name to assign these images:DNA",
            ]
        ),
        encoding="utf-8",
    )
    case = CellProfilerComparisonCase(
        name="cp_tutorial_translocation_final",
        dataset_path=dataset_path,
        cppipe_path=cppipe_path,
        dataset_id="CellProfiler_tutorials",
    )
    scope = NativeCellProfilerReferenceScope(
        case=case,
        native_reference_root=tmp_path / "native_refs",
        pipeline_params={},
        source_schema_image_set_selection=SourceSchemaImageSetSelection(
            max_image_set_count=1,
        ),
    )
    reference = (
        scope.output_dir
        / "TranslocationData_cp_tutorial_translocation_final_native_cellprofiler"
    )
    reference.mkdir(parents=True)
    (reference / ".cellprofiler_benchmark_reference.json").write_text(
        json.dumps({"schema_version": 1, "provenance": {}}),
        encoding="utf-8",
    )

    location = scope.resolve()

    assert location.output_dir == scope.output_dir
    assert location.reference_output_dir == reference


def test_native_reference_scope_resolves_relative_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_path = tmp_path / "tutorial" / "images"
    dataset_path.mkdir(parents=True)
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
            ]
        ),
        encoding="utf-8",
    )
    case = CellProfilerComparisonCase(
        name="ExampleRelativeRoot",
        dataset_path=dataset_path,
        cppipe_path=cppipe_path,
        dataset_id="ExampleDataset",
    )

    scope = NativeCellProfilerReferenceScope(
        case=case,
        native_reference_root=Path("native_refs"),
        pipeline_params={},
    )

    assert scope.output_dir.is_absolute()
    assert scope.output_dir.parent == (tmp_path / "native_refs").resolve()


def test_comparison_writers_emit_raw_phase_and_summary_tables(
    tmp_path: Path,
) -> None:
    case = CellProfilerComparisonCase(
        name="ExampleFly",
        dataset_path=tmp_path / "ExampleFly",
        cppipe_path=tmp_path / "ExampleFly.cppipe",
        assay_category="Tissue/object morphology",
        module_category="Segmentation + object measurement",
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
    assert observation_rows[0]["assay_category"] == "Tissue/object morphology"
    assert (
        observation_rows[0]["module_category"]
        == "Segmentation + object measurement"
    )
    assert observation_rows[0]["difference_count"] == "4"
    assert observation_rows[0]["openhcs_error_message"] == "semantic mismatch"
    assert observation_rows[0]["parity_accuracy"] == "0.0"
    assert observation_rows[0]["total_phase_speedup"] == "6.0"
    assert {row["phase"] for row in phase_rows} == {
        "EXECUTE_NATIVE_CP",
        "EXECUTE_OPENHCS",
    }
    assert summary_rows[0]["median_speedup"] == "6.0"
    assert summary_rows[0]["assay_category"] == "Tissue/object morphology"
    assert summary_rows[0]["module_category"] == "Segmentation + object measurement"
    assert summary_rows[0]["median_total_phase_speedup"] == "6.0"
    assert summary_rows[0]["speedup_target"] == "5.0"
    assert summary_rows[0]["meets_execution_speedup_target"] == "True"
    assert summary_rows[0]["meets_total_phase_speedup_target"] == "True"
    assert summary_rows[0]["meets_speedup_target"] == "True"
    assert summary_rows[0]["equivalent_count"] == "0"


def test_summary_speedup_target_uses_execution_time_only(
    tmp_path: Path,
) -> None:
    case = CellProfilerComparisonCase(
        name="ExampleExecutionFast",
        dataset_path=tmp_path / "ExampleExecutionFast",
        cppipe_path=tmp_path / "ExampleExecutionFast.cppipe",
    )
    native_output = tmp_path / "native"
    openhcs_output = tmp_path / "openhcs"
    native_output.mkdir()
    openhcs_output.mkdir()
    observation = comparison_observation_from_result(
        CellProfilerCompatibilityResult(
            native_cellprofiler=BenchmarkResult(
                tool_name="CellProfiler",
                dataset_id="dataset",
                pipeline_name="pipeline",
                metrics={"execution_time_seconds": 60.0},
                output_path=native_output,
                success=True,
                provenance={
                    "phase_timing_records": (
                        {"phase": "EXECUTE_NATIVE_CP", "seconds": 60.0},
                        {"phase": "COMPARE_NATIVE", "seconds": 1.0},
                    ),
                },
            ),
            openhcs_converted=BenchmarkResult(
                tool_name="OpenHCS",
                dataset_id="dataset",
                pipeline_name="pipeline",
                metrics={"execution_time_seconds": 10.0},
                output_path=openhcs_output,
                success=True,
                provenance={
                    "equivalence_difference_count": 0,
                    "phase_timing_records": (
                        {"phase": "COMPILE_OPENHCS", "seconds": 15.0},
                        {"phase": "EXECUTE_OPENHCS", "seconds": 10.0},
                        {"phase": "COMPARE_OPENHCS", "seconds": 15.0},
                    ),
                },
            ),
        ),
        case=case,
        suite_id="suite-1",
        repetition=1,
    )

    write_summary_csv(tmp_path / "summary.csv", (observation,))
    summary_row = _csv_rows(tmp_path / "summary.csv")[0]

    assert summary_row["median_speedup"] == "6.0"
    assert summary_row["median_total_phase_speedup"] == "1.525"
    assert summary_row["meets_execution_speedup_target"] == "True"
    assert summary_row["meets_total_phase_speedup_target"] == "False"
    assert summary_row["meets_speedup_target"] == "True"


def test_load_comparison_cases_from_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "default_pipeline_params": {"compare_image_outputs": True},
                "cases": [
                    {
                        "name": "ExampleHuman",
                        "dataset_path": "data/human",
                        "cppipe_path": "pipes/human.cppipe",
                        "dataset_id": "human",
                        "microscope_type": "imagexpress",
                        "assay_category": "Cell morphology",
                        "module_category": "Segmentation + intensity measurement",
                        "value_only": True,
                        "equivalence_reference_output_dir": "native/human",
                        "cellprofiler_timeout_seconds": 120,
                        "pipeline_params": {"compare_image_outputs": False},
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
            assay_category="Cell morphology",
            module_category="Segmentation + intensity measurement",
            value_only=True,
            equivalence_reference_output_dir=Path("native/human"),
            cellprofiler_timeout_seconds=120.0,
            pipeline_params={
                "compare_image_outputs": False,
            },
        ),
    )


def test_filter_comparison_cases_by_exact_name_preserves_merged_params(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "default_pipeline_params": {"custom_param": 1},
                "cases": [
                    {
                        "name": "first",
                        "dataset_path": "data/first",
                        "cppipe_path": "pipes/first.cppipe",
                    },
                    {
                        "name": "second",
                        "dataset_path": "data/second",
                        "cppipe_path": "pipes/second.cppipe",
                        "pipeline_params": {"compare_image_outputs": False},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    selected = _filter_cases_by_name(load_comparison_cases(manifest), ("second",))

    assert tuple(case.name for case in selected) == ("second",)
    assert selected[0].pipeline_params == {
        "custom_param": 1,
        "compare_image_outputs": False,
    }


def test_filter_comparison_cases_by_exact_name_reports_unknown_and_available(
    tmp_path: Path,
) -> None:
    cases = (
        CellProfilerComparisonCase(
            name="first",
            dataset_path=tmp_path / "first",
            cppipe_path=tmp_path / "first.cppipe",
        ),
        CellProfilerComparisonCase(
            name="second",
            dataset_path=tmp_path / "second",
            cppipe_path=tmp_path / "second.cppipe",
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        _filter_cases_by_name(cases, ("missing", "other"))

    message = str(exc_info.value)
    assert "Unknown benchmark case name(s): missing, other" in message
    assert "Available case name(s): first, second" in message


def test_load_comparison_cases_resolves_declared_path_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    monkeypatch.setenv("CP_EXAMPLES_ROOT", str(tmp_path / "examples"))
    manifest.write_text(
        json.dumps(
            {
                "path_roots": {
                    "examples": {"env": "CP_EXAMPLES_ROOT"},
                    "cache": {"default": str(tmp_path / "cache")},
                },
                "cases": [
                    {
                        "name": "ExampleHuman",
                        "dataset_path_root": "examples",
                        "dataset_path": "ExampleHuman/images",
                        "cppipe_path_root": "cache",
                        "cppipe_path": "ExampleHuman/ExampleHuman.cppipe",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_comparison_cases(manifest)

    assert cases[0].dataset_path == tmp_path / "examples" / "ExampleHuman/images"
    assert cases[0].cppipe_path == tmp_path / "cache" / "ExampleHuman/ExampleHuman.cppipe"


def test_write_module_coverage_artifacts_for_manifest(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "coverage.cppipe"
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "IdentifyPrimaryObjects:[module_num:2|enabled:True]",
                "    Select the input image:DNA",
                "    Name the primary objects to be identified:Nuclei",
            )
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "CoverageCase",
                        "dataset_path": str(tmp_path / "images"),
                        "cppipe_path": str(cppipe_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    write_module_coverage_artifacts(tmp_path / "artifacts", manifest_path=manifest)

    summary = json.loads(
        (tmp_path / "artifacts" / "module_coverage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    cppipe_rows = _csv_rows(
        tmp_path / "artifacts" / "module_coverage_cppipe_modules.csv"
    )
    setting_rows = _csv_rows(
        tmp_path / "artifacts" / "module_coverage_cppipe_settings.csv"
    )
    semantic_rows = _csv_rows(
        tmp_path / "artifacts" / "module_coverage_semantic_families.csv"
    )

    assert summary["cppipe_case_count"] == 1
    assert summary["missing_processing_cppipe_module_count"] == 0
    assert summary["cppipe_setting_row_count"] == 3
    assert summary["covered_cppipe_setting_row_count"] == 3
    assert summary["unmapped_cppipe_setting_row_count"] == 0
    assert "IdentifyPrimaryObjects" in summary["supported_absorbed_processing_modules"]
    assert {row["module_name"] for row in cppipe_rows} == {
        "IdentifyPrimaryObjects",
        "Images",
    }
    assert {
        (row["module_name"], row["setting_name"], row["coverage"])
        for row in setting_rows
    } == {
        ("Images", "Filter images?", "infrastructure"),
        ("IdentifyPrimaryObjects", "Select the input image", "bound"),
        (
            "IdentifyPrimaryObjects",
            "Name the primary objects to be identified",
            "bound",
        ),
    }
    semantic_rows_by_module = {row["module_name"]: row for row in semantic_rows}
    assert (
        semantic_rows_by_module["IdentifyPrimaryObjects"]["family_coverage"]
        == "direct_supported"
    )
    assert (
        "IdentifyPrimaryObjects"
        in semantic_rows_by_module["IdentifyPrimaryObjects"][
            "family_supported_modules"
        ]
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


def test_discard_successful_openhcs_benchmark_tree_preserves_failed_outputs(
    tmp_path: Path,
) -> None:
    case = CellProfilerComparisonCase(
        name="ExampleFailed",
        dataset_path=tmp_path / "ExampleFailed",
        cppipe_path=tmp_path / "ExampleFailed.cppipe",
    )
    output_tree = tmp_path / "suite" / "tool_outputs" / "OpenHCS_failed"
    observation = comparison_observation_from_result(
        CellProfilerCompatibilityResult(
            native_cellprofiler=_benchmark_result(
                "CellProfiler",
                tmp_path / "native",
                "EXECUTE_NATIVE_CP",
                1.0,
            ),
            openhcs_converted=_benchmark_result(
                "OpenHCS",
                output_tree,
                "EXECUTE_OPENHCS",
                1.0,
                success=False,
                error_message="semantic mismatch",
                provenance={"equivalence_difference_count": 1},
            ),
        ),
        case=case,
        suite_id="suite-1",
        repetition=1,
    )

    _discard_successful_openhcs_benchmark_tree(
        observation,
        suite_output_root=tmp_path / "suite",
    )

    assert output_tree.exists()


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
