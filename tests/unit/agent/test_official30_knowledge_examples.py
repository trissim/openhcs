"""MCP-facing official30 converted-source knowledge gates."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from textwrap import dedent

import pytest

import openhcs.agent.services.knowledge_base_service as knowledge_base_module
from openhcs.agent.capabilities import (
    DescribeFunctionCapability,
    ExplainArchitectureCapability,
    GetKnowledgeDocumentCapability,
    ListArchitectureTopicsCapability,
    SearchFunctionsCapability,
    SearchKnowledgeCapability,
)
from openhcs.agent.dto.functions import FunctionDetailRequest, FunctionSearchRequest
from openhcs.agent.dto.knowledge import (
    KnowledgeBaseDocumentRequest,
    KnowledgeBaseDocumentSummary,
    KnowledgeBaseSearchRequest,
)
from openhcs.agent.services.knowledge_base_service import (
    KnowledgeBaseDocumentSpec,
    KnowledgeBaseService,
)
from openhcs.core.config import PipelineConfig
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.core.steps.function_step import FunctionStep
from openhcs.mcp.context import OpenHCSAgentContext
from openhcs.mcp.dev_client_renderers.knowledge import KnowledgeDocumentRenderer
from openhcs.serialization.json import to_jsonable


OFFICIAL30_DOCUMENT_ID = "openhcs_official30_benchmark_recipes"


def _source_from_document_content(content: str) -> str:
    marker = ".. code-block:: python\n\n"
    assert marker in content
    return dedent(content.split(marker, 1)[1])


def test_official30_catalog_and_search_do_not_eagerly_convert_cases(monkeypatch):
    def unexpected_conversion(*_args: object) -> str:
        raise AssertionError("official30 conversion must be selected-section-only")

    monkeypatch.setattr(
        knowledge_base_module,
        "_official30_public_source",
        unexpected_conversion,
    )
    service = KnowledgeBaseService()

    catalog = service.list_documents()
    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id=OFFICIAL30_DOCUMENT_ID,
            section_id="examplehuman",
        )
    )
    search = service.search(
        KnowledgeBaseSearchRequest(query="ExampleHuman OpenHCS Python", limit=5)
    )

    assert any(item.document_id == OFFICIAL30_DOCUMENT_ID for item in catalog.documents)
    assert "Generated only when this exact section is requested" in document.content
    assert any(
        hit.section is not None
        and hit.section.section_id == "examplehuman-openhcs-python"
        for hit in search.hits
    )
    source_sections = tuple(
        section
        for section in document.sections
        if section.title.endswith(" OpenHCS Python")
    )
    case_titles = {section.title for section in document.sections if section.level == 3}
    assert len(source_sections) == 30
    assert {
        section.title.removesuffix(" OpenHCS Python") for section in source_sections
    } == case_titles


def test_official30_conversion_has_one_uncached_lazy_importer_boundary():
    source_path = Path(knowledge_base_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def called_names(function_name: str) -> set[str]:
        return {
            node.func.id
            for node in ast.walk(functions[function_name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    conversion_callers = {
        function_name
        for function_name in functions
        if "_official30_public_source" in called_names(function_name)
    }
    assert conversion_callers == {"_official30_source_section_lines"}
    for function_name in (
        "_display_lines",
        "_official30_recipe_projection",
        "list_documents",
        "search",
    ):
        assert "_official30_public_source" not in called_names(function_name)
    assert "import_cellprofiler_pipeline" in called_names("_official30_public_source")
    assert functions["_official30_public_source"].decorator_list == []
    assert "ExampleHuman" not in source


def test_official30_source_selection_carries_exact_manifest_case_identity(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    for filename in ("first.cppipe", "second.cppipe"):
        (source_root / filename).write_text("CellProfiler Pipeline", encoding="utf-8")
    manifest_path = tmp_path / "official30.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "path_roots": {"source": {"path": str(source_root)}},
                "cases": [
                    {
                        "name": "Case A",
                        "dataset_path_root": "source",
                        "dataset_path": ".",
                        "cppipe_path_root": "source",
                        "cppipe_path": "first.cppipe",
                    },
                    {
                        "name": "Case-A",
                        "dataset_path_root": "source",
                        "dataset_path": ".",
                        "cppipe_path_root": "source",
                        "cppipe_path": "second.cppipe",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[Path] = []

    def projected_source(cppipe_path: str, _source_root: str) -> str:
        calls.append(Path(cppipe_path))
        return "pipeline_config = None\npipeline_steps = []"

    monkeypatch.setattr(
        knowledge_base_module,
        "_official30_public_source",
        projected_source,
    )
    service = KnowledgeBaseService(
        repo_root=tmp_path,
        document_specs=(
            KnowledgeBaseDocumentSpec(
                KnowledgeBaseDocumentSummary(
                    document_id=OFFICIAL30_DOCUMENT_ID,
                    title="Official30",
                    summary="Official30 exact identity test.",
                    source_path=manifest_path.name,
                    tags=("official30",),
                    section_count=0,
                )
            ),
        ),
    )
    catalog_document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id=OFFICIAL30_DOCUMENT_ID,
        )
    )
    second_section = next(
        section
        for section in catalog_document.sections
        if section.title == "Case-A OpenHCS Python"
    )

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id=OFFICIAL30_DOCUMENT_ID,
            section_id=second_section.section_id,
        )
    )

    assert document.errors == ()
    assert calls == [(source_root / "second.cppipe").resolve()]


def test_official30_manifest_requires_canonical_case_name(tmp_path):
    manifest_path = tmp_path / "official30.json"
    manifest_path.write_text(
        json.dumps({"manifest_version": 1, "cases": [{"dataset_id": "missing"}]}),
        encoding="utf-8",
    )
    service = KnowledgeBaseService(
        repo_root=tmp_path,
        document_specs=(
            KnowledgeBaseDocumentSpec(
                KnowledgeBaseDocumentSummary(
                    document_id=OFFICIAL30_DOCUMENT_ID,
                    title="Official30",
                    summary="Official30 missing identity test.",
                    source_path=manifest_path.name,
                    tags=("official30",),
                    section_count=0,
                )
            ),
        ),
    )

    with pytest.raises(ValueError, match="nonempty, trimmed string name"):
        service.list_documents()


def test_requested_official30_source_is_importable_public_openhcs_python():
    service = KnowledgeBaseService()

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id=OFFICIAL30_DOCUMENT_ID,
            section_id="examplehuman-openhcs-python",
            max_chars=50_000,
        )
    )
    source = _source_from_document_content(document.content)
    pipeline_document = PipelineDocumentAuthority.from_source(source)

    assert document.errors == ()
    assert document.truncated is False
    assert isinstance(pipeline_document.pipeline_config, PipelineConfig)
    assert pipeline_document.pipeline_config != PipelineConfig()
    assert pipeline_document.pipeline_config.source_bindings_config is not None
    assert pipeline_document.pipeline_steps
    assert all(
        isinstance(step, FunctionStep) for step in pipeline_document.pipeline_steps
    )
    assert ".cppipe" not in source
    assert "pipeline_config = PipelineConfig(" in source
    assert "pipeline_steps = [" in source


def test_missing_official30_source_is_an_explicit_document_error(tmp_path):
    manifest_path = tmp_path / "official30.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "path_roots": {"missing": {"path": str(tmp_path / "missing")}},
                "cases": [
                    {
                        "name": "MissingCase",
                        "dataset_id": "missing",
                        "dataset_path_root": "missing",
                        "dataset_path": "images",
                        "cppipe_path_root": "missing",
                        "cppipe_path": "MissingCase.cppipe",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    service = KnowledgeBaseService(
        repo_root=tmp_path,
        document_specs=(
            KnowledgeBaseDocumentSpec(
                KnowledgeBaseDocumentSummary(
                    document_id=OFFICIAL30_DOCUMENT_ID,
                    title="Official30",
                    summary="Official30 test manifest.",
                    source_path="official30.json",
                    tags=("official30",),
                    section_count=0,
                )
            ),
        ),
    )

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id=OFFICIAL30_DOCUMENT_ID,
            section_id="missingcase-openhcs-python",
        )
    )

    assert document.content == ""
    assert document.errors[0].code == "official30_source_missing"
    assert "Raw .cppipe text is not returned as a fallback" in (
        document.errors[0].hint or ""
    )


def test_official30_conversion_failure_is_an_explicit_document_error(monkeypatch):
    def failed_conversion(*_args: object) -> str:
        raise RuntimeError("focused importer failure")

    monkeypatch.setattr(
        knowledge_base_module,
        "_official30_public_source",
        failed_conversion,
    )

    document = KnowledgeBaseService().get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id=OFFICIAL30_DOCUMENT_ID,
            section_id="examplehuman-openhcs-python",
        )
    )

    assert document.content == ""
    assert document.errors[0].code == "official30_conversion_failed"
    assert "RuntimeError: focused importer failure" in document.errors[0].message


def test_official30_source_document_renders_through_mcp_dev_renderer():
    document = KnowledgeBaseService().get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id=OFFICIAL30_DOCUMENT_ID,
            section_id="examplehuman-openhcs-python",
            max_chars=50_000,
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_get_knowledge_document",
                "mcp_error": False,
                "payloads": [to_jsonable(document)],
            }
        ],
    }

    rendered = KnowledgeDocumentRenderer.render(response)

    assert "Selected section: examplehuman-openhcs-python" in rendered
    assert "pipeline_config = PipelineConfig(" in rendered
    assert "pipeline_steps = [" in rendered


def test_mcp_capabilities_discover_examples_architecture_and_function_docs():
    context = OpenHCSAgentContext()

    example_search = SearchKnowledgeCapability.execute_request(
        context,
        KnowledgeBaseSearchRequest(query="ExampleHuman OpenHCS Python", limit=5),
    )
    example_hit = next(
        hit
        for hit in example_search.hits
        if hit.section is not None and hit.section.title.endswith("OpenHCS Python")
    )
    example_document = GetKnowledgeDocumentCapability.execute_request(
        context,
        KnowledgeBaseDocumentRequest.from_fields(
            document_id=example_hit.document.document_id,
            section_id=example_hit.section.section_id,
            max_chars=50_000,
        ),
    )

    architecture_search = SearchKnowledgeCapability.execute_request(
        context,
        KnowledgeBaseSearchRequest(query="nominal ownership registry MRO", limit=10),
    )
    architecture_hit = next(
        hit for hit in architecture_search.hits if hit.section is not None
    )
    architecture_document = GetKnowledgeDocumentCapability.execute_request(
        context,
        KnowledgeBaseDocumentRequest.from_fields(
            document_id=architecture_hit.document.document_id,
            section_id=architecture_hit.section.section_id,
            max_chars=8_000,
        ),
    )

    topics = ListArchitectureTopicsCapability.execute_no_argument(context)
    cellprofiler_topic_id = next(
        topic.topic_id for topic in topics.topics if "CellProfiler" in topic.title
    )
    cellprofiler_topic = ExplainArchitectureCapability.execute_scalar(
        context,
        cellprofiler_topic_id,
    )

    functions = SearchFunctionsCapability.execute_request(
        context,
        FunctionSearchRequest(query="count cells simple", limit=5),
    )
    function_entry = next(
        entry for entry in functions.items if entry.name == "count_cells_simple"
    )
    function_detail = DescribeFunctionCapability.execute_request(
        context,
        FunctionDetailRequest(
            function_id=function_entry.function_id,
            max_doc_chars=6_000,
            compact_signature=False,
        ),
    )

    assert isinstance(example_document.document, KnowledgeBaseDocumentSummary)
    assert "pipeline_steps = [" in example_document.content
    assert architecture_document.content
    assert cellprofiler_topic.internal_symbols
    assert any(
        symbol.title == "import_cellprofiler_pipeline"
        for symbol in cellprofiler_topic.internal_symbols
    )
    assert function_detail.doc
    assert "count_cells_simple(" in function_detail.entry.signature
    assert function_detail.runtime_contract is not None
    assert {
        artifact.name for artifact in function_detail.runtime_contract.artifact_outputs
    } == {"cell_counts", "segmentation_masks"}
