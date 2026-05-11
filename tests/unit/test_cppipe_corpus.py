from pathlib import Path
import json
import re

import pytest

from benchmark.converter.cppipe_corpus import (
    CPPipeCorpusStatus,
    comparison_manifest_cppipe_corpus,
    comparison_manifests_cppipe_corpus,
    in_tree_cppipe_corpus,
)
from openhcs.interop.cellprofiler.runtime_pipeline import prepare_generated_pipeline
from openhcs.interop.cellprofiler import CellProfilerModuleRole
from openhcs.interop.cellprofiler import CellProfilerPipelineImportResult


def test_in_tree_cppipe_corpus_accounts_for_all_shipped_cppipes() -> None:
    corpus = in_tree_cppipe_corpus()
    declared_paths = {case.cppipe_path.resolve() for case in corpus}
    actual_paths = {
        path.resolve()
        for path in (
            Path(__file__).resolve().parents[2] / "benchmark" / "cellprofiler_pipelines"
        ).glob("*.cppipe")
    }

    assert declared_paths == actual_paths


def test_comparison_manifest_cppipe_corpus_projects_benchmark_cases(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, {"manifest_case": cppipe_path})

    corpus = comparison_manifest_cppipe_corpus(manifest_path)

    assert len(corpus) == 1
    case = corpus[0]
    assert case.name == "manifest_case"
    assert case.cppipe_path == cppipe_path
    assert case.status is CPPipeCorpusStatus.SUPPORTED


def test_comparison_manifest_cppipe_corpus_resolves_declared_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    monkeypatch.setenv("CPPIPE_ROOT", str(root))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "path_roots": {"cppipe": {"env": "CPPIPE_ROOT"}},
                "cases": [
                    {
                        "name": "rooted_case",
                        "cppipe_path_root": "cppipe",
                        "cppipe_path": "pipeline.cppipe",
                        "dataset_path": "/tmp/example",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    corpus = comparison_manifest_cppipe_corpus(manifest_path)

    assert corpus[0].cppipe_path == root / "pipeline.cppipe"


def test_comparison_manifests_cppipe_corpus_combines_manifests(
    tmp_path: Path,
) -> None:
    first_cppipe_path = tmp_path / "first.cppipe"
    second_cppipe_path = tmp_path / "second.cppipe"
    first_manifest_path = tmp_path / "first.json"
    second_manifest_path = tmp_path / "second.json"

    _write_manifest(first_manifest_path, {"first": first_cppipe_path})
    _write_manifest(second_manifest_path, {"second": second_cppipe_path})

    corpus = comparison_manifests_cppipe_corpus(
        (first_manifest_path, second_manifest_path)
    )

    assert tuple(case.name for case in corpus) == ("first", "second")
    assert tuple(case.cppipe_path for case in corpus) == (
        first_cppipe_path,
        second_cppipe_path,
    )


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


def test_in_tree_cppipe_corpus_prepare_expectations(tmp_path: Path) -> None:
    for case in in_tree_cppipe_corpus():
        output_path = tmp_path / f"{case.name}_generated.py"
        if case.status is CPPipeCorpusStatus.SUPPORTED:
            prepared = prepare_generated_pipeline(case.cppipe_path, output_path=output_path)
            assert prepared.processing_modules
            import_result = prepared.import_result
            assert isinstance(import_result, CellProfilerPipelineImportResult)
            assert import_result.pipeline is prepared.pipeline
            assert import_result.source_schema is prepared.source_schema
            assert import_result.generated_module_path == output_path
            assert import_result.provenance.cppipe_path == case.cppipe_path
            assert {
                module.role for module in import_result.provenance.processing_modules
            } == {CellProfilerModuleRole.PROCESSING}
            assert len(import_result.artifact_contracts) == len(
                prepared.generated_pipeline.artifact_contracts
            )
            continue

        assert case.expected_error_substring is not None
        with pytest.raises(ValueError, match=re.escape(case.expected_error_substring)):
            prepare_generated_pipeline(case.cppipe_path, output_path=output_path)
