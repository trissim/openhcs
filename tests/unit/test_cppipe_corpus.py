from pathlib import Path
import re

import pytest

from benchmark.converter.cppipe_corpus import (
    CPPipeCorpusStatus,
    in_tree_cppipe_corpus,
)
from benchmark.converter.runtime_pipeline import prepare_generated_pipeline
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
