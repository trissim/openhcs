import ast
import json
from pathlib import Path

import pytest

from benchmark.converter.cppipe_corpus import CPPipeCorpusCase, CPPipeCorpusStatus
from benchmark.converter.compatibility_matrix import (
    CPPipeModuleAbsorptionCoverage,
    ModuleCorpusCoverage,
    SourceModuleCoverage,
    build_cellprofiler_compatibility_report,
    build_cellprofiler_compatibility_report_for_manifest,
    build_cellprofiler_compatibility_report_for_manifests,
)
from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.interop.cellprofiler.module_settings import (
    ModuleSettingCoverageStatus,
)


def test_compatibility_matrix_accounts_for_absorbed_modules() -> None:
    report = build_cellprofiler_compatibility_report(corpus_cases=())

    assert len(report.modules) == 91
    assert all(
        module.importable or module.is_infrastructure for module in report.modules
    )


def test_registered_modules_have_processing_contract_coverage() -> None:
    report = build_cellprofiler_compatibility_report(corpus_cases=())

    assert report.supported_corpus_processing_contract_gaps == ()
    assert report.missing_cppipe_processing_modules == ()


def test_compatibility_matrix_has_no_unresolved_processing_contracts() -> None:
    report = build_cellprofiler_compatibility_report(corpus_cases=())

    assert report.unresolved_processing_contracts == ()


def test_compatibility_matrix_reads_registered_declaration_facts() -> None:
    report = build_cellprofiler_compatibility_report(corpus_cases=())
    modules_by_name = {module.module_name: module for module in report.modules}

    assert (
        modules_by_name["IdentifyPrimaryObjects"].corpus_coverage
        is ModuleCorpusCoverage.NOT_IN_CORPUS
    )
    watershed = modules_by_name["Watershed"]
    assert watershed.function_names == (
        "watershed_library",
        "watershed_cellprofiler4",
    )
    assert watershed.execution_scope is FunctionStepExecutionScope.AXIS
    assert watershed.processing_contract is not None
    assert watershed.respects_masks is True
    assert not hasattr(report, "semantic_families")


def test_compatibility_matrix_accepts_explicit_cppipe_corpus(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "official_identify.cppipe"
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "NamesAndTypes:[module_num:1|enabled:True]",
                "    Assignments count:1",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                '    Select the rule criteria:and (file does contain "DNA")',
                "IdentifyPrimaryObjects:[module_num:2|enabled:True]",
                "    Select the input image:DNA",
                "    Name the primary objects to be identified:Nuclei",
            )
        )
    )

    report = build_cellprofiler_compatibility_report(
        corpus_cases=(
            CPPipeCorpusCase(
                name="OfficialIdentify",
                cppipe_path=cppipe_path,
                status=CPPipeCorpusStatus.SUPPORTED,
            ),
        )
    )
    modules_by_name = {module.module_name: module for module in report.modules}

    assert (
        modules_by_name["IdentifyPrimaryObjects"].corpus_coverage
        is ModuleCorpusCoverage.SUPPORTED_CORPUS
    )


def test_compatibility_matrix_summarizes_benchmark_coverage(
    tmp_path: Path,
) -> None:
    supported_cppipe_path = tmp_path / "supported.cppipe"
    supported_cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "NamesAndTypes:[module_num:1|enabled:True]",
                "    Assignments count:1",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                '    Select the rule criteria:and (file does contain "DNA")',
                "IdentifyPrimaryObjects:[module_num:2|enabled:True]",
                "    Select the input image:DNA",
                "    Name the primary objects to be identified:Nuclei",
            )
        )
    )
    known_invalid_cppipe_path = tmp_path / "known_invalid.cppipe"
    known_invalid_cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "GaussianFilter:[module_num:1|enabled:True]",
                "    Select the input image:Input",
                "NotAbsorbedModule:[module_num:2|enabled:True]",
                "    Setting:Value",
            )
        )
    )

    report = build_cellprofiler_compatibility_report(
        corpus_cases=(
            CPPipeCorpusCase(
                name="Supported",
                cppipe_path=supported_cppipe_path,
                status=CPPipeCorpusStatus.SUPPORTED,
            ),
            CPPipeCorpusCase(
                name="KnownInvalid",
                cppipe_path=known_invalid_cppipe_path,
                status=CPPipeCorpusStatus.KNOWN_INVALID,
            ),
        )
    )
    coverage = report.benchmark_coverage

    assert coverage.cppipe_case_count == 2
    assert coverage.supported_cppipe_case_count == 1
    assert coverage.known_invalid_cppipe_case_count == 1
    assert coverage.module_instance_count == 4
    assert coverage.unique_cppipe_module_count == 4
    assert "IdentifyPrimaryObjects" in coverage.supported_absorbed_processing_modules
    assert "GaussianFilter" in coverage.known_invalid_absorbed_processing_modules
    assert "Watershed" in coverage.untested_absorbed_processing_modules
    assert coverage.infrastructure_cppipe_modules == ("NamesAndTypes",)
    assert coverage.missing_processing_cppipe_modules == ("NotAbsorbedModule",)
    assert {
        (setting.module_name, setting.setting_name, setting.coverage)
        for setting in report.cppipe_settings
    } == {
        (
            "IdentifyPrimaryObjects",
            "Select the input image",
            ModuleSettingCoverageStatus.BOUND,
        ),
        (
            "IdentifyPrimaryObjects",
            "Name the primary objects to be identified",
            ModuleSettingCoverageStatus.BOUND,
        ),
    }


def test_compatibility_matrix_accepts_benchmark_manifest_corpus(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "manifest_only.cppipe"
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "NamesAndTypes:[module_num:1|enabled:True]",
                "    Assignments count:1",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                '    Select the rule criteria:and (file does contain "DNA")',
                "MedianFilter:[module_num:2|enabled:True]",
                "    Select the input image:DNA",
                "    Name the output image:FilteredDNA",
                "    Window:5",
                "SaveImages:[module_num:3|enabled:True]",
                "    Select the image to save:FilteredDNA",
            )
        )
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, {"manifest_only": cppipe_path})

    report = build_cellprofiler_compatibility_report_for_manifest(manifest_path)
    tested = set(report.benchmark_coverage.supported_absorbed_processing_modules)
    cppipe_modules = {module.module_name: module for module in report.cppipe_modules}

    assert {"MedianFilter", "SaveImages"} <= tested
    assert cppipe_modules["MedianFilter"].cppipe_case_names == ("manifest_only",)
    assert any(
        setting.case_name == "manifest_only"
        and setting.module_name == "MedianFilter"
        and setting.setting_name == "Window"
        for setting in report.cppipe_settings
    )


def test_compatibility_matrix_combines_multiple_benchmark_manifests(
    tmp_path: Path,
) -> None:
    median_cppipe_path = tmp_path / "median.cppipe"
    median_cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "NamesAndTypes:[module_num:1|enabled:True]",
                "    Assignments count:1",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                '    Select the rule criteria:and (file does contain "DNA")',
                "MedianFilter:[module_num:2|enabled:True]",
                "    Select the input image:DNA",
                "    Name the output image:FilteredDNA",
                "    Window:5",
            )
        )
    )
    save_cppipe_path = tmp_path / "save.cppipe"
    save_cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "NamesAndTypes:[module_num:1|enabled:True]",
                "    Assignments count:1",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                '    Select the rule criteria:and (file does contain "DNA")',
                "SaveImages:[module_num:2|enabled:True]",
                "    Select the image to save:DNA",
            )
        )
    )
    first_manifest_path = tmp_path / "first_manifest.json"
    second_manifest_path = tmp_path / "second_manifest.json"
    _write_manifest(first_manifest_path, {"median_case": median_cppipe_path})
    _write_manifest(second_manifest_path, {"save_case": save_cppipe_path})

    report = build_cellprofiler_compatibility_report_for_manifests(
        (first_manifest_path, second_manifest_path)
    )
    tested = set(report.benchmark_coverage.supported_absorbed_processing_modules)

    assert {"MedianFilter", "SaveImages"} <= tested
    assert report.benchmark_coverage.cppipe_case_count == 2


def _write_manifest(manifest_path: Path, cases: dict[str, Path]) -> None:
    payload = {
        "cases": [
            {
                "name": name,
                "cppipe_path": str(cppipe_path),
                "dataset_path": "/tmp/example",
            }
            for name, cppipe_path in cases.items()
        ]
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_compatibility_matrix_distinguishes_infrastructure_from_missing_processing(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "module_coverage.cppipe"
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "NotAbsorbedModule:[module_num:2|enabled:True]",
                "    Setting:Value",
            )
        )
    )

    report = build_cellprofiler_compatibility_report(
        corpus_cases=(
            CPPipeCorpusCase(
                name="ModuleCoverage",
                cppipe_path=cppipe_path,
                status=CPPipeCorpusStatus.KNOWN_INVALID,
            ),
        )
    )
    cppipe_modules = {module.module_name: module for module in report.cppipe_modules}

    assert (
        cppipe_modules["Images"].absorption_coverage
        is CPPipeModuleAbsorptionCoverage.INFRASTRUCTURE
    )
    assert report.missing_cppipe_processing_modules == (
        cppipe_modules["NotAbsorbedModule"],
    )


def test_export_modules_are_processing_declarations_not_infrastructure(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "file_processing.cppipe"
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "CreateBatchFiles:[module_num:1|enabled:True]",
                "    Store batch files in default output folder?:Yes",
                "ExportToDatabase:[module_num:2|enabled:True]",
                "    Database type:SQLite",
            )
        )
    )

    report = build_cellprofiler_compatibility_report(
        corpus_cases=(
            CPPipeCorpusCase(
                name="FileProcessing",
                cppipe_path=cppipe_path,
                status=CPPipeCorpusStatus.KNOWN_INVALID,
            ),
        )
    )
    cppipe_modules = {module.module_name: module for module in report.cppipe_modules}

    assert (
        cppipe_modules["CreateBatchFiles"].absorption_coverage
        is CPPipeModuleAbsorptionCoverage.MISSING_PROCESSING
    )
    assert (
        cppipe_modules["ExportToDatabase"].absorption_coverage
        is CPPipeModuleAbsorptionCoverage.ABSORBED_PROCESSING
    )


def test_compatibility_matrix_tracks_checked_in_source_module_coverage(
    tmp_path: Path,
) -> None:
    source_modules_root = tmp_path / "modules"
    source_modules_root.mkdir()
    (source_modules_root / "identifyprimaryobjects.py").write_text("")
    (source_modules_root / "exporttospreadsheet.py").write_text("")
    (source_modules_root / "notabsorbed.py").write_text("")
    (source_modules_root / "__init__.py").write_text("")

    report = build_cellprofiler_compatibility_report(
        corpus_cases=(),
        source_modules_root=source_modules_root,
    )
    source_modules = {module.module_name: module for module in report.source_modules}

    assert (
        source_modules["identifyprimaryobjects"].coverage
        is SourceModuleCoverage.ABSORBED
    )
    assert (
        source_modules["exporttospreadsheet"].coverage is SourceModuleCoverage.ABSORBED
    )
    assert report.missing_source_modules == (source_modules["notabsorbed"],)


def test_supported_corpus_rejects_an_enabled_unregistered_module(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "unsupported.cppipe"
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "NotAbsorbedModule:[module_num:1|enabled:True]",
                "    Setting:Value",
            )
        )
    )

    with pytest.raises(KeyError, match="No CellProfiler module declaration"):
        build_cellprofiler_compatibility_report(
            corpus_cases=(
                CPPipeCorpusCase(
                    name="Unsupported",
                    cppipe_path=cppipe_path,
                    status=CPPipeCorpusStatus.SUPPORTED,
                ),
            )
        )


def test_compatibility_matrix_has_no_generator_or_semantics_facade() -> None:
    source_path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "converter"
        / "compatibility_matrix.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert "CellProfilerModule" in names
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "__registry__"
        and isinstance(node.value, ast.Name)
        and node.value.id == "CellProfilerModule"
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Name) and node.id == "import_cellprofiler_pipeline"
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "bind_settings"
        for node in ast.walk(tree)
    )
