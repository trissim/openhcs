import pytest

from openhcs.agent import knowledge_manifest
from openhcs.agent.dto.knowledge import (
    KnowledgeBaseDocumentRequest,
    KnowledgeBaseDocumentSummary,
    KnowledgeBaseSearchRequest,
)
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.serialization.json import to_jsonable
from openhcs.agent.services.knowledge_base_service import (
    KnowledgeBaseDocumentSpec,
    KnowledgeBaseService,
)
from openhcs.agent.services import (
    knowledge_base_service as knowledge_base_service_module,
)


def test_knowledge_base_catalog_lists_source_backed_documents():
    service = KnowledgeBaseService()

    catalog = service.list_documents()
    documents = {document.document_id: document for document in catalog.documents}

    assert catalog.schema_version == "openhcs.agent.v1"
    assert "openhcs_core_model" in documents
    assert "openhcs_architecture_quick_start" in documents
    assert "openhcs_configuration_reference" in documents
    assert "openhcs_domain_expert_onboarding" in documents
    assert "openhcs_example_corpus_map" in documents
    assert "openhcs_complete_examples" in documents
    assert "openhcs_system_overview" in documents
    assert "openhcs_nominal_ownership" in documents
    assert "openhcs_streaming_boundary_and_wrappers" in documents
    assert "openhcs_image_sources" in documents
    assert "openhcs_pipeline_compilation_system" in documents
    assert "openhcs_custom_functions" in documents
    assert "openhcs_viewer_management" in documents
    assert documents["openhcs_configuration_reference"].source_path == (
        "docs/source/guide_for_biologists/configuration_reference.rst"
    )
    assert documents["openhcs_architecture_quick_start"].section_count > 0
    assert "MCP quick start" in documents["openhcs_architecture_quick_start"].tags
    assert documents["openhcs_core_model"].source_path == (
        "docs/source/concepts/core_model.rst"
    )
    assert "core model" in documents["openhcs_core_model"].tags
    assert (
        "Live UI/code biconversion"
        in documents["openhcs_code_ui_interconversion"].summary
    )
    assert "live UI code" in documents["openhcs_code_ui_interconversion"].tags
    assert "code mode" in documents["openhcs_code_ui_interconversion"].tags
    assert documents["openhcs_image_sources"].source_path == (
        "docs/source/guide_for_biologists/image_sources.rst"
    )
    source_paths = tuple(document.source_path for document in catalog.documents)
    assert len(source_paths) == len(set(source_paths))
    assert {
        "openhcs_production_examples",
        "openhcs_configuration_framework",
        "openhcs_context_system",
        "openhcs_special_io_system",
        "openhcs_pattern_grouping_special_outputs",
        "openhcs_napari_streaming_system",
        "openhcs_fiji_streaming_system",
        "openhcs_function_registry_system",
    }.isdisjoint(documents)


def test_knowledge_base_document_read_is_bounded_and_sectioned():
    service = KnowledgeBaseService()

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_architecture_quick_start",
            max_chars=500,
        )
    )
    first_section = document.sections[0]
    selected = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_architecture_quick_start",
            section_id=first_section.section_id,
        )
    )

    assert document.schema_version == "openhcs.agent.v1"
    assert document.document is not None
    assert document.truncated is True
    assert len(document.content) == document.max_chars
    assert "OpenHCS" in document.content
    assert selected.selected_section_id == first_section.section_id
    assert selected.content.startswith("Architecture quick start")


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
        "plate well site channel": "openhcs_core_model",
        "segmentation fluorescence": "openhcs_domain_expert_onboarding",
        "plate layout": "openhcs_data_dimensions",
    }

    for query, expected_document_id in expected_documents_by_query.items():
        result = service.search(KnowledgeBaseSearchRequest(query=query, limit=10))
        document_ids = {hit.document.document_id for hit in result.hits}

        assert expected_document_id in document_ids


def test_knowledge_base_retrieves_canonical_image_source_guide():
    service = KnowledgeBaseService()
    queries = (
        "how do I load CZI images",
        "open an OME-TIFF plate",
        "load OME-Zarr NGFF",
        "combine mixed image stores TIFF PNG",
        "assign named channels with source bindings",
        "interpret well site channel Z timepoint component metadata",
    )

    for query in queries:
        result = service.search(KnowledgeBaseSearchRequest(query=query, limit=10))
        assert "openhcs_image_sources" in {
            hit.document.document_id for hit in result.hits
        }

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_image_sources",
            max_chars=12_000,
        )
    )

    assert document.truncated is False
    assert "PipelineConfig.source_bindings_config" in document.content
    assert "CZI" in document.content
    assert "OME-TIFF" in document.content
    assert "OME-Zarr" in document.content
    assert "WELL" in document.content
    assert "SITE" in document.content
    assert "CHANNEL" in document.content
    assert "Z_INDEX" in document.content
    assert "TIMEPOINT" in document.content


def test_knowledge_base_search_covers_example_corpus_terms():
    service = KnowledgeBaseService()
    expected_documents_by_query = {
        "CellProfiler examples cppipe": "openhcs_example_corpus_map",
        "CellProfiler compatibility OpenHCS runtime official30": ("openhcs_core_model"),
        "runtime artifact sidecar source universe source binding": (
            "openhcs_core_model"
        ),
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
            "openhcs_complete_examples"
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


def test_example_corpus_document_exposes_native_python_source():
    service = KnowledgeBaseService()

    index = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_example_corpus_map",
            section_id="native-example-source-index",
            max_chars=12_000,
        )
    )
    source = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_example_corpus_map",
            section_id="openhcs-processing-presets-mfd-specs-py",
            max_chars=40_000,
        )
    )

    assert index.selected_section_id == "native-example-source-index"
    assert "Generated from the Python paths declared" in index.content
    assert "openhcs/processing/presets/mfd_specs.py" in index.content
    assert "openhcs/processing/presets/pipelines/10x_mfd_crop_analyze.py" in (
        index.content
    )
    assert "benchmark/pipelines/" not in index.content
    assert "openhcs/debug/example_export.py" not in index.content
    assert source.selected_section_id == ("openhcs-processing-presets-mfd-specs-py")
    assert "Source path: openhcs/processing/presets/mfd_specs.py" in source.content
    assert ".. code-block:: python" in source.content
    assert "from openhcs.core.steps.function_step import FunctionStep" in (
        source.content
    )
    assert "class MfdPresetKey" in source.content
    assert "def build_mfd_preset" in source.content


def test_knowledge_base_search_covers_native_python_source():
    service = KnowledgeBaseService()

    result = service.search(
        KnowledgeBaseSearchRequest(
            query="MfdPresetKey build_mfd_preset FunctionStep",
            limit=5,
        )
    )

    assert any(
        hit.document.document_id == "openhcs_example_corpus_map"
        and hit.section is not None
        and hit.section.section_id == "openhcs-processing-presets-mfd-specs-py"
        for hit in result.hits
    )


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
        "30. cp_tutorial_translocation_start: dataset=CellProfiler_tutorials"
    ) in document.content
    assert "Module Usage Index" in document.content
    assert "Default source-schema selection" not in document.content
    assert "Raw manifest:" not in document.content
    assert len(document.sections) == 63
    assert any(section.section_id == "examplehuman" for section in document.sections)
    assert any(
        section.section_id == "examplehuman-openhcs-python"
        for section in document.sections
    )
    assert any(
        section.section_id == "module-usage-index" for section in document.sections
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
    assert "modules: Images, Metadata, NamesAndTypes, Groups" in document.content
    assert "IdentifyPrimaryObjects" in document.content
    assert "MeasureObjectIntensity" in document.content
    assert "resolved_cppipe_path:" in document.content
    assert "ExampleFly" not in document.content


def test_official30_recipe_document_supports_module_usage_section():
    service = KnowledgeBaseService()

    document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_official30_benchmark_recipes",
            section_id="module-usage-index",
            max_chars=10_000,
        )
    )

    assert document.selected_section_id == "module-usage-index"
    assert "Module Usage Index" in document.content
    assert "ExampleTrackObjects:" in document.content
    assert "TrackObjects" in document.content
    assert "IdentifyPrimaryObjects" in document.content
    assert "MeasureObjectIntensity" in document.content


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


def test_knowledge_base_search_covers_official30_module_usage():
    service = KnowledgeBaseService()

    result = service.search(
        KnowledgeBaseSearchRequest(
            query="IdentifyPrimaryObjects MeasureObjectIntensity TrackObjects",
            limit=5,
        )
    )

    assert any(
        hit.document.document_id == "openhcs_official30_benchmark_recipes"
        and hit.section is not None
        and hit.section.section_id == "module-usage-index"
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
        "what is openhcs core model": "openhcs_core_model",
        "compiler runtime FunctionStep ObjectState code documents": (
            "openhcs_core_model"
        ),
        "UI code interconversion revision tokens": ("openhcs_code_ui_interconversion"),
        "GlobalPipelineConfig registered step config": (
            "openhcs_configuration_reference"
        ),
        "ObjectState semantics": "openhcs_code_ui_interconversion",
        "MaterializationSpec writer": "openhcs_artifact_contract_system",
        "InputSource PIPELINE_START": "openhcs_architecture_quick_start",
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


def test_default_knowledge_root_uses_packaged_projection_without_checkout_docs(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "site-packages"
    packaged_root = source_root / "openhcs" / "agent" / "resources" / "knowledge"
    packaged_manifest = (
        packaged_root / knowledge_manifest.DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH
    )
    packaged_manifest.parent.mkdir(parents=True)
    packaged_manifest.write_text('{"documents": []}\n', encoding="utf-8")
    monkeypatch.setattr(
        knowledge_manifest,
        "source_checkout_root",
        lambda: source_root,
    )
    monkeypatch.setattr(
        knowledge_manifest,
        "packaged_knowledge_base_root",
        lambda: packaged_root,
    )

    assert knowledge_manifest.default_repo_root() == packaged_root
    assert knowledge_manifest.default_knowledge_base_manifest_path() == (
        packaged_manifest
    )


def test_knowledge_base_accepts_packaged_projection_within_readable_install_root(
    tmp_path,
    monkeypatch,
):
    install_root = tmp_path / "site-packages"
    packaged_root = install_root / "openhcs" / "agent" / "resources" / "knowledge"
    document_path = packaged_root / "docs" / "kb.md"
    document_path.parent.mkdir(parents=True)
    document_path.write_text("# Packaged KB\n\nInstalled docs.\n", encoding="utf-8")
    monkeypatch.setattr(
        knowledge_base_service_module,
        "default_repo_root",
        lambda: packaged_root,
    )
    policy = AgentPathPolicy.with_roots(
        readable_roots=(install_root,),
        writable_roots=(tmp_path,),
    )

    service = KnowledgeBaseService.from_path_policy(
        policy,
        document_specs=(
            _document_spec(
                document_id="packaged_kb",
                title="Packaged KB",
                source_path="docs/kb.md",
            ),
        ),
    )
    catalog = service.list_documents()

    assert catalog.documents[0].document_id == "packaged_kb"
    assert catalog.warnings == ()


def test_packaged_knowledge_remains_available_with_explicit_data_roots(
    tmp_path,
    monkeypatch,
):
    packaged_root = (
        tmp_path / "site-packages" / "openhcs" / "agent" / "resources" / "knowledge"
    )
    document_path = packaged_root / "docs" / "kb.md"
    document_path.parent.mkdir(parents=True)
    document_path.write_text("# Packaged KB\n\nInstalled docs.\n", encoding="utf-8")
    monkeypatch.setattr(
        knowledge_base_service_module,
        "default_repo_root",
        lambda: packaged_root,
    )
    monkeypatch.setattr(
        knowledge_base_service_module,
        "packaged_knowledge_base_root",
        lambda: packaged_root,
    )
    tenant_root = tmp_path / "tenant"
    tenant_root.mkdir()
    policy = AgentPathPolicy.with_roots(
        readable_roots=(tenant_root,),
        writable_roots=(tenant_root,),
    )

    service = KnowledgeBaseService.from_path_policy(
        policy,
        document_specs=(
            _document_spec(
                document_id="packaged_kb",
                title="Packaged KB",
                source_path="docs/kb.md",
            ),
        ),
    )

    assert service.list_documents().documents[0].document_id == "packaged_kb"


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


def test_knowledge_base_rejects_parallel_document_source_authorities(tmp_path):
    specs = (
        _document_spec("canonical", "Canonical", "docs/architecture.rst"),
        _document_spec("compatibility", "Compatibility", "docs/architecture.rst"),
    )

    with pytest.raises(ValueError, match="Duplicate knowledge-base document source"):
        KnowledgeBaseService(repo_root=tmp_path, document_specs=specs)


def test_knowledge_base_parent_section_includes_child_sections(tmp_path):
    doc_path = tmp_path / "docs" / "hierarchy.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(
        "# Root\n\n## Parent\nParent body.\n\n### Child\nChild body.\n\n## Next\nNext body.\n",
        encoding="utf-8",
    )
    service = KnowledgeBaseService(
        repo_root=tmp_path,
        document_specs=(_document_spec("hierarchy", "Hierarchy", "docs/hierarchy.md"),),
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
