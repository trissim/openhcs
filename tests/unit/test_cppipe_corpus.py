import json
from itertools import groupby
from pathlib import Path
import re

import pytest

from benchmark.converter.cppipe_corpus import (
    CPPipeCorpusCase,
    CPPipeCorpusStatus,
    comparison_manifest_cppipe_corpus,
    comparison_manifests_cppipe_corpus,
    in_tree_cppipe_corpus,
)
from openhcs.core.config import PipelineConfig
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.source_bindings import SourceBindingsConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler import import_cellprofiler_pipeline
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.parser import CPPipeParser


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
    assert case.source_root == Path("/tmp/example")
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
    assert corpus[0].source_root == Path("/tmp/example")


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


@pytest.mark.parametrize(
    "case",
    in_tree_cppipe_corpus(),
    ids=lambda case: case.name,
)
def test_in_tree_cppipe_corpus_import_expectations(
    case: CPPipeCorpusCase,
) -> None:
    if case.status is not CPPipeCorpusStatus.SUPPORTED:
        assert case.expected_error_substring is not None
        with pytest.raises(
            KeyError,
            match=re.escape(case.expected_error_substring),
        ):
            import_cellprofiler_pipeline(case.cppipe_path)
        return

    parser = CPPipeParser()
    parsed_modules = tuple(parser.parse(case.cppipe_path))
    enabled_declarations = tuple(
        (module, CellProfilerModule.require_module(module.name))
        for module in parsed_modules
        if module.enabled
    )
    executable_modules = tuple(
        module
        for module, declaration in enabled_declarations
        if declaration.emits_function_step()
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(case.cppipe_path)

    assert type(pipeline_steps) is list
    assert pipeline_steps
    assert all(isinstance(step, FunctionStep) for step in pipeline_steps)
    assert isinstance(pipeline_config, PipelineConfig)
    executable_name_runs = tuple(
        name for name, _group in groupby(module.name for module in executable_modules)
    )
    public_step_name_runs = tuple(
        name for name, _group in groupby(step.name for step in pipeline_steps)
    )
    assert public_step_name_runs == executable_name_runs
    assert all(
        declaration in CellProfilerModule.__registry__.values()
        for _module, declaration in enabled_declarations
    )
    assert (
        pipeline_config.source_bindings_config.to_base_config()
        == CellProfilerModule.source_bindings_for_modules(
            parsed_modules,
            SourceBindingsConfig(image_plane_sources=parser.image_plane_sources),
        )
    )

    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
    namespace: dict[str, object] = {}
    exec(compile(source, f"{case.name}_pipeline.py", "exec"), namespace)
    reconstructed_steps = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )
    assert (
        FunctionStepTransportAuthority.source_from_pipeline(reconstructed_steps)
        == source
    )
    assert [step.name for step in reconstructed_steps] == [
        step.name for step in pipeline_steps
    ]


def test_cppipe_import_rejects_unknown_module_at_nominal_boundary(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "unsupported.cppipe"
    cppipe_path.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
UnsupportedModule:[module_num:1|enabled:True]
""",
        encoding="utf-8",
    )

    with pytest.raises(
        KeyError,
        match="No CellProfiler module declaration is registered",
    ):
        import_cellprofiler_pipeline(cppipe_path)


def test_cppipe_import_rejects_pipeline_without_modules(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "empty.cppipe"
    cppipe_path.write_text(
        "CellProfiler Pipeline: https://cellprofiler.org\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains no modules"):
        import_cellprofiler_pipeline(cppipe_path)
