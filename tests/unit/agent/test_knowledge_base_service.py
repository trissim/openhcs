
from openhcs.agent.dto.knowledge import (
    KnowledgeBaseDocumentRequest,
    KnowledgeBaseDocumentSummary,
    KnowledgeBaseSearchRequest,
)
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.serialization import to_jsonable
from openhcs.agent.services.knowledge_base_service import (
    KnowledgeBaseDocumentSpec,
    KnowledgeBaseService,
)


def test_knowledge_base_catalog_lists_source_backed_documents():
    service = KnowledgeBaseService()

    catalog = service.list_documents()
    documents = {document.document_id: document for document in catalog.documents}

    assert catalog.schema_version == "openhcs.agent.v1"
    assert "openhcs_agent_mcp_overview" in documents
    assert "openhcs_domain_expert_onboarding" in documents
    assert "openhcs_example_corpus_map" in documents
    assert "openhcs_complete_examples" in documents
    assert "openhcs_production_examples" in documents
    assert "openhcs_configuration_framework" in documents
    assert "openhcs_pipeline_compilation_system" in documents
    assert "openhcs_custom_functions" in documents
    assert "openhcs_viewer_management" in documents
    assert documents["openhcs_agent_mcp_overview"].source_path == (
        "docs/plans/openhcs_mcp_agent_knowledge_base_20260625.md"
    )
    assert documents["openhcs_agent_mcp_overview"].section_count > 0
    assert "mcp" in documents["openhcs_agent_mcp_overview"].tags


def test_knowledge_base_document_read_is_bounded_and_sectioned():
    service = KnowledgeBaseService()

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_agent_mcp_overview",
            max_chars=500,
        )
    )
    first_section = document.sections[0]
    selected = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_agent_mcp_overview",
            section_id=first_section.section_id,
        )
    )

    assert document.schema_version == "openhcs.agent.v1"
    assert document.document is not None
    assert document.truncated is True
    assert len(document.content) == document.max_chars
    assert "OpenHCS" in document.content
    assert selected.selected_section_id == first_section.section_id
    assert selected.content.startswith("# ")


def test_knowledge_base_search_returns_source_sections():
    service = KnowledgeBaseService()

    result = service.search(KnowledgeBaseSearchRequest(query="ObjectState", limit=5))
    payload = to_jsonable(result)

    assert result.schema_version == "openhcs.agent.v1"
    assert result.hits
    assert payload["hits"][0]["document"]["source_path"].startswith("docs/")
    assert "ObjectState" in result.hits[0].snippet


def test_knowledge_base_search_covers_domain_expert_onboarding_terms():
    service = KnowledgeBaseService()
    expected_documents_by_query = {
        "microscopy": "openhcs_domain_expert_onboarding",
        "getting started": "openhcs_domain_expert_onboarding",
        "plate well site channel": "openhcs_data_dimensions",
        "segmentation fluorescence": "openhcs_domain_expert_onboarding",
        "plate layout": "openhcs_data_dimensions",
    }

    for query, expected_document_id in expected_documents_by_query.items():
        result = service.search(KnowledgeBaseSearchRequest(query=query, limit=10))
        document_ids = {hit.document.document_id for hit in result.hits}

        assert expected_document_id in document_ids


def test_knowledge_base_search_covers_example_corpus_terms():
    service = KnowledgeBaseService()
    expected_documents_by_query = {
        "CellProfiler examples cppipe": "openhcs_example_corpus_map",
        "CellProfiler mental model OpenHCS translation": (
            "openhcs_domain_expert_onboarding"
        ),
        "CellProfiler Images Objects Measurements FunctionStep artifacts": (
            "openhcs_example_corpus_map"
        ),
        "native OpenHCS examples": "openhcs_example_corpus_map",
        "ExampleHuman ExampleFly BBBC021": "openhcs_example_corpus_map",
        "official30 native CellProfiler reference": "openhcs_example_corpus_map",
        "official30 benchmark recipe": "openhcs_official30_benchmark_recipes",
        "30 benchmark pipeline recipes": "openhcs_official30_benchmark_recipes",
        "recipe": "openhcs_official30_benchmark_recipes",
        "benchmark pipelines preset pipelines": "openhcs_example_corpus_map",
        "complete examples FunctionStep dictionary GPU zarr": (
            "openhcs_complete_examples"
        ),
        "production examples cell analysis neurite tracing": (
            "openhcs_production_examples"
        ),
    }

    for query, expected_document_id in expected_documents_by_query.items():
        result = service.search(KnowledgeBaseSearchRequest(query=query, limit=10))
        document_ids = {hit.document.document_id for hit in result.hits}

        assert expected_document_id in document_ids


def test_example_corpus_document_points_to_live_plate_inspection_tools():
    service = KnowledgeBaseService()

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_example_corpus_map",
            max_chars=8_000,
        )
    )

    assert "Live Data Inspection" in document.content
    assert "inspect-plate" in document.content
    assert "query-plate-files" in document.content
    assert "selected-plate-images" in document.content
    assert "selected-plate-files" in document.content
    assert "Until that tool exists" not in document.content


def test_official30_recipe_document_renders_generated_case_index():
    service = KnowledgeBaseService()

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_official30_benchmark_recipes",
            max_chars=6_000,
        )
    )

    assert "Recipe count: 30" in document.content
    assert (
        "1. ExampleColocalization: dataset=ExampleColocalization "
        "cppipe=ExampleColocalization/ExampleColocalization.cppipe"
    ) in document.content
    assert (
        "30. cp_tutorial_translocation_start: "
        "dataset=CellProfiler_tutorials"
    ) in document.content
    assert "Default pipeline params: openhcs_max_axis_count=1" in document.content
    assert "Raw manifest:" not in document.content
    assert len(document.sections) == 32
    assert any(
        section.section_id == "examplehuman"
        for section in document.sections
    )


def test_official30_recipe_document_supports_case_sections():
    service = KnowledgeBaseService()

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_official30_benchmark_recipes",
            section_id="examplehuman",
            max_chars=2_000,
        )
    )

    assert document.selected_section_id == "examplehuman"
    assert "ExampleHuman" in document.content
    assert "dataset_id: ExampleHuman" in document.content
    assert "dataset_path: ExampleHuman/images" in document.content
    assert "cppipe_path: ExampleHuman/ExampleHuman.cppipe" in document.content
    assert "ExampleFly" not in document.content


def test_knowledge_base_search_covers_official30_case_sections():
    service = KnowledgeBaseService()

    result = service.search(
        KnowledgeBaseSearchRequest(
            query="ExampleHuman cppipe",
            limit=5,
        )
    )

    assert any(
        hit.document.document_id == "openhcs_official30_benchmark_recipes"
        and hit.section is not None
        and hit.section.section_id == "examplehuman"
        for hit in result.hits
    )


def test_knowledge_base_search_handles_broad_biology_workflow_query():
    service = KnowledgeBaseService()

    result = service.search(
        KnowledgeBaseSearchRequest(
            query=(
                "I have fluorescence plate images. Segment nuclei, expand to "
                "cells, measure cell intensity, and maybe colocalization."
            ),
            limit=5,
        )
    )

    assert result.hits
    document_ids = {hit.document.document_id for hit in result.hits}
    assert "openhcs_domain_expert_onboarding" in document_ids
    assert "openhcs_for_biologists_intro" in document_ids
    assert any(
        {"fluorescence", "plate", "segment", "nuclei", "cell", "intensity"}
        & set(hit.matched_terms)
        for hit in result.hits
    )
    assert all(hit.score > 0 for hit in result.hits)


def test_knowledge_base_search_covers_technical_operator_terms():
    service = KnowledgeBaseService()
    expected_documents_by_query = {
        "GlobalPipelineConfig registered step config": (
            "openhcs_configuration_framework"
        ),
        "ObjectState semantics": "openhcs_agent_mcp_overview",
        "MaterializationSpec writer": "openhcs_agent_mcp_overview",
        "InputSource PIPELINE_START": "openhcs_agent_mcp_overview",
        "how to add custom functions": "openhcs_custom_functions",
        "inspect in Napari": "openhcs_viewer_management",
    }

    for query, expected_document_id in expected_documents_by_query.items():
        result = service.search(KnowledgeBaseSearchRequest(query=query, limit=10))
        document_ids = {hit.document.document_id for hit in result.hits}

        assert expected_document_id in document_ids


def test_knowledge_base_unknown_document_returns_structured_error():
    service = KnowledgeBaseService()

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(document_id="missing_document")
    )

    assert document.document is None
    assert document.errors[0].code == "knowledge_document_unknown"


def test_knowledge_base_uses_path_policy_root_for_active_checkout(tmp_path):
    active_root = tmp_path / "active"
    doc_path = active_root / "docs" / "kb.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("# Active KB\n\nCurrent checkout docs.\n", encoding="utf-8")
    specs = (
        _document_spec(
            document_id="active_kb",
            title="Active KB",
            source_path="docs/kb.md",
        ),
    )
    policy = AgentPathPolicy.with_roots(
        readable_roots=(active_root,),
        writable_roots=(tmp_path,),
    )

    service = KnowledgeBaseService.from_path_policy(
        policy,
        document_specs=specs,
    )
    catalog = service.list_documents()

    assert catalog.documents[0].document_id == "active_kb"
    assert catalog.documents[0].section_count == 1
    assert catalog.warnings == ()


def test_knowledge_base_path_policy_does_not_fallback_to_installed_checkout(tmp_path):
    active_root = tmp_path / "active"
    active_root.mkdir()
    policy = AgentPathPolicy.with_roots(
        readable_roots=(active_root,),
        writable_roots=(tmp_path,),
    )
    specs = (
        _document_spec(
            document_id="installed_only",
            title="Installed Only",
            source_path="docs/source/guide_for_biologists/domain_expert_onboarding.rst",
        ),
    )

    service = KnowledgeBaseService.from_path_policy(
        policy,
        document_specs=specs,
    )
    catalog = service.list_documents()

    assert catalog.documents == ()
    assert catalog.warnings[0].code == "knowledge_document_missing"
    assert catalog.warnings[0].hint == (
        "docs/source/guide_for_biologists/domain_expert_onboarding.rst"
    )


def test_knowledge_base_reports_missing_allowlisted_documents(tmp_path):
    existing_doc = tmp_path / "docs" / "present.md"
    existing_doc.parent.mkdir(parents=True)
    existing_doc.write_text("# Present\n", encoding="utf-8")
    specs = (
        _document_spec("present", "Present", "docs/present.md"),
        _document_spec("missing", "Missing", "docs/missing.md"),
    )
    service = KnowledgeBaseService(repo_root=tmp_path, document_specs=specs)

    catalog = service.list_documents()
    search = service.search(KnowledgeBaseSearchRequest(query="Present"))
    missing = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(document_id="missing")
    )

    assert [document.document_id for document in catalog.documents] == ["present"]
    assert catalog.warnings[0].code == "knowledge_document_missing"
    assert catalog.warnings[0].hint == "docs/missing.md"
    assert search.warnings[0].code == "knowledge_document_missing"
    assert missing.errors[0].path == "docs/missing.md"


def test_knowledge_base_parent_section_includes_child_sections(tmp_path):
    doc_path = tmp_path / "docs" / "hierarchy.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(
        "# Root\n\n## Parent\nParent body.\n\n### Child\nChild body.\n\n## Next\nNext body.\n",
        encoding="utf-8",
    )
    service = KnowledgeBaseService(
        repo_root=tmp_path,
        document_specs=(
            _document_spec("hierarchy", "Hierarchy", "docs/hierarchy.md"),
        ),
    )

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="hierarchy",
            section_id="parent",
        )
    )

    assert "### Child" in document.content
    assert "Child body." in document.content
    assert "## Next" not in document.content


def _document_spec(
    document_id: str,
    title: str,
    source_path: str,
) -> KnowledgeBaseDocumentSpec:
    return KnowledgeBaseDocumentSpec(
        KnowledgeBaseDocumentSummary(
            document_id=document_id,
            title=title,
            summary=f"{title} summary.",
            source_path=source_path,
            tags=("test",),
            section_count=0,
        )
    )
