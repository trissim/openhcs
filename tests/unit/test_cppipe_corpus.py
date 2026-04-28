from pathlib import Path
import re

import pytest

from benchmark.converter.cppipe_corpus import (
    CPPipeCorpusStatus,
    in_tree_cppipe_corpus,
)
from benchmark.converter.runtime_pipeline import prepare_generated_pipeline


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
            continue

        assert case.expected_error_substring is not None
        with pytest.raises(ValueError, match=re.escape(case.expected_error_substring)):
            prepare_generated_pipeline(case.cppipe_path, output_path=output_path)
