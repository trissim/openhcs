from pathlib import Path

from benchmark.converter.cppipe_corpus import CPPipeCorpusCase, CPPipeCorpusStatus
from benchmark.converter.compatibility_matrix import (
    ArtifactContractCoverage,
    CPPipeModuleAbsorptionCoverage,
    ModuleCorpusCoverage,
    build_cellprofiler_compatibility_report,
)


def test_compatibility_matrix_accounts_for_absorbed_modules() -> None:
    report = build_cellprofiler_compatibility_report()

    assert len(report.modules) == 89
    assert all(module.importable for module in report.modules)


def test_supported_corpus_has_processing_contract_coverage() -> None:
    report = build_cellprofiler_compatibility_report()

    assert report.supported_corpus_processing_contract_gaps == ()
    assert report.missing_cppipe_processing_modules == ()


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
