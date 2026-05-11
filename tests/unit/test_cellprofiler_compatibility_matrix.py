from pathlib import Path
import json

from benchmark.converter.cppipe_corpus import CPPipeCorpusCase, CPPipeCorpusStatus
from benchmark.converter.compatibility_matrix import (
    ArtifactContractCoverage,
    CPPipeModuleAbsorptionCoverage,
    CPPipeSettingCoverage,
    ModuleCorpusCoverage,
    SourceModuleCoverage,
    build_cellprofiler_compatibility_report,
    build_cellprofiler_compatibility_report_for_manifest,
    build_cellprofiler_compatibility_report_for_manifests,
)


def test_compatibility_matrix_accounts_for_absorbed_modules() -> None:
    report = build_cellprofiler_compatibility_report()

    assert len(report.modules) == 89
    assert all(module.importable for module in report.modules)


def test_supported_corpus_has_processing_contract_coverage() -> None:
    report = build_cellprofiler_compatibility_report()

    assert report.supported_corpus_processing_contract_gaps == ()
    assert report.missing_cppipe_processing_modules == ()
    assert report.missing_source_modules == ()


def test_compatibility_matrix_has_no_unresolved_processing_contracts() -> None:
    report = build_cellprofiler_compatibility_report()

    assert report.unresolved_processing_contracts == ()


def test_compatibility_matrix_tracks_artifact_and_corpus_coverage() -> None:
    report = build_cellprofiler_compatibility_report()
    modules_by_name = {module.module_name: module for module in report.modules}

    assert (
        modules_by_name["IdentifyPrimaryObjects"].artifact_contract_coverage
        is ArtifactContractCoverage.DECLARED_BUILDER
    )
    assert (
        modules_by_name["IdentifyPrimaryObjects"].corpus_coverage
        is ModuleCorpusCoverage.SUPPORTED_CORPUS
    )
    assert (
        modules_by_name["GaussianFilter"].artifact_contract_coverage
        is ArtifactContractCoverage.GENERIC_INFERENCE
    )
    assert (
        modules_by_name["Align"].artifact_contract_coverage
        is ArtifactContractCoverage.DECLARED_BUILDER
    )
    assert modules_by_name["Watershed"].semantics is not None
    assert modules_by_name["Watershed"].semantics.supports_3d is True
    assert modules_by_name["Watershed"].semantics.respects_masks is True
    family_rows = {row.module_name: row for row in report.semantic_families}
    assert family_rows["IdentifyPrimaryObjects"].family_name == "ObjectProcessingMasked2D"
    assert (
        family_rows["IdentifyPrimaryObjects"].family_coverage
        == "direct_supported"
    )
    assert "IdentifyPrimaryObjects" in family_rows[
        "IdentifySecondaryObjects"
    ].family_supported_modules


def test_compatibility_matrix_accepts_explicit_cppipe_corpus(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "official_trackobjects.cppipe"
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                (
                    "TrackObjects:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input objects:Cells",
            )
        )
    )

    report = build_cellprofiler_compatibility_report(
        corpus_cases=(
            CPPipeCorpusCase(
                name="OfficialTrackObjects",
                cppipe_path=cppipe_path,
                status=CPPipeCorpusStatus.SUPPORTED,
            ),
        )
    )
    modules_by_name = {module.module_name: module for module in report.modules}

    assert (
        modules_by_name["TrackObjects"].corpus_coverage
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
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "IdentifyPrimaryObjects:[module_num:2|enabled:True]",
                "    Select the input image:DNA",
                "    Name the primary objects to be identified:Nuclei",
                "NotAbsorbedModule:[module_num:3|enabled:True]",
                "    Setting:Value",
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
    assert coverage.infrastructure_cppipe_modules == ("Images",)
    assert coverage.missing_processing_cppipe_modules == ("NotAbsorbedModule",)
    assert {
        (setting.module_name, setting.setting_name, setting.coverage)
        for setting in report.cppipe_settings
    } == {
        ("Images", "Filter images?", CPPipeSettingCoverage.INFRASTRUCTURE),
        (
            "IdentifyPrimaryObjects",
            "Select the input image",
            CPPipeSettingCoverage.BOUND,
        ),
        (
            "IdentifyPrimaryObjects",
            "Name the primary objects to be identified",
            CPPipeSettingCoverage.BOUND,
        ),
        (
            "NotAbsorbedModule",
            "Setting",
            CPPipeSettingCoverage.MODULE_NOT_ABSORBED,
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
                "Watershed:[module_num:1|enabled:True]",
                "    Select the input image:Input",
                "RescaleIntensity:[module_num:2|enabled:True]",
                "    Select the input image:Input",
                "Medianfilter:[module_num:3|enabled:True]",
                "    Select the input image:Input",
            )
        )
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, {"manifest_only": cppipe_path})

    report = build_cellprofiler_compatibility_report_for_manifest(manifest_path)
    tested = set(report.benchmark_coverage.supported_absorbed_processing_modules)
    cppipe_modules = {module.module_name: module for module in report.cppipe_modules}

    assert {"Watershed", "RescaleIntensity", "Medianfilter"} <= tested
    assert cppipe_modules["Watershed"].cppipe_case_names == ("manifest_only",)
    assert any(
        setting.case_name == "manifest_only"
        and setting.module_name == "Watershed"
        and setting.setting_name == "Select the input image"
        for setting in report.cppipe_settings
    )


def test_compatibility_matrix_combines_multiple_benchmark_manifests(
    tmp_path: Path,
) -> None:
    watershed_cppipe_path = tmp_path / "watershed.cppipe"
    watershed_cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "Watershed:[module_num:1|enabled:True]",
                "    Select the input image:Input",
            )
        )
    )
    rescale_cppipe_path = tmp_path / "rescale.cppipe"
    rescale_cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "RescaleIntensity:[module_num:1|enabled:True]",
                "    Select the input image:Input",
            )
        )
    )
    first_manifest_path = tmp_path / "first_manifest.json"
    second_manifest_path = tmp_path / "second_manifest.json"
    _write_manifest(first_manifest_path, {"watershed_case": watershed_cppipe_path})
    _write_manifest(second_manifest_path, {"rescale_case": rescale_cppipe_path})

    report = build_cellprofiler_compatibility_report_for_manifests(
        (first_manifest_path, second_manifest_path)
    )
    tested = set(report.benchmark_coverage.supported_absorbed_processing_modules)

    assert {"Watershed", "RescaleIntensity"} <= tested
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
                status=CPPipeCorpusStatus.SUPPORTED,
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


def test_manual_file_processing_modules_are_infrastructure(
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
                status=CPPipeCorpusStatus.SUPPORTED,
            ),
        )
    )
    cppipe_modules = {module.module_name: module for module in report.cppipe_modules}

    assert (
        cppipe_modules["CreateBatchFiles"].absorption_coverage
        is CPPipeModuleAbsorptionCoverage.INFRASTRUCTURE
    )
    assert (
        cppipe_modules["ExportToDatabase"].absorption_coverage
        is CPPipeModuleAbsorptionCoverage.INFRASTRUCTURE
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
        source_modules["exporttospreadsheet"].coverage
        is SourceModuleCoverage.INFRASTRUCTURE
    )
    assert report.missing_source_modules == (source_modules["notabsorbed"],)


def test_checked_in_source_modules_have_nominal_semantics() -> None:
    report = build_cellprofiler_compatibility_report()

    assert all(module.semantics is not None for module in report.source_modules)
