from __future__ import annotations

from pathlib import Path

from openhcs.agent.authoring_contexts import (
    AuthoringContextDeclaration,
    AuthoringContextRoute,
)
from openhcs.agent.capabilities import agent_capabilities
from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.dto.config import ConfigFieldSchema, ConfigSchema
from openhcs.agent.dto.knowledge import (
    KnowledgeBaseCatalog,
    KnowledgeBaseDocumentSummary,
    KnowledgeBaseDocumentTarget,
)
from openhcs.agent.services.knowledge_base_service import KnowledgeBaseService
from openhcs.agent.services.llm_context_service import AgentAuthoringContextService


class _UnexpectedFunctionCatalog:
    def search(self, **_kwargs):
        raise AssertionError("progressive contexts must not enumerate arbitrary functions")


class _ReflectedConfigService:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def describe_schema(self, config_type: str) -> ConfigSchema:
        self.requests.append(config_type)
        return ConfigSchema(
            schema_version=SCHEMA_VERSION,
            config_type=config_type,
            fields=(
                ConfigFieldSchema(
                    path=f"{config_type}_field",
                    type_repr="int",
                    default_repr="1",
                    required=False,
                    description=f"Reflected {config_type} field.",
                ),
            ),
        )


class _DeclaredKnowledgeBase:
    def __init__(self) -> None:
        document_ids = tuple(
            dict.fromkeys(
                target.document_id
                for declaration in AuthoringContextDeclaration.__registry__.values()
                for target in declaration.require_route().knowledge_targets
            )
        )
        self._catalog = KnowledgeBaseCatalog(
            schema_version=SCHEMA_VERSION,
            documents=tuple(
                KnowledgeBaseDocumentSummary(
                    document_id=document_id,
                    title=f"Title for {document_id}",
                    summary=f"Summary for {document_id}",
                    source_path=f"docs/{document_id}.rst",
                    tags=(),
                    section_count=1,
                )
                for document_id in document_ids
            ),
        )

    def list_documents(self) -> KnowledgeBaseCatalog:
        return self._catalog


def _service() -> tuple[AgentAuthoringContextService, _ReflectedConfigService]:
    config_service = _ReflectedConfigService()
    return (
        AgentAuthoringContextService(
            function_catalog=_UnexpectedFunctionCatalog(),
            config_service=config_service,
            knowledge_base=_DeclaredKnowledgeBase(),
        ),
        config_service,
    )


def test_context_registry_orders_ui_ownership_before_headless_routes() -> None:
    assert AuthoringContextDeclaration.allowed_values() == (
        "first_use",
        "ui_visible_workflow",
        "domain_expert_assisted_setup",
        "folder_onboarding",
        "pipeline",
        "custom_function",
        "headless_execution",
        "debugging",
        "viewer_review",
        "objectstate_editing",
        "cellprofiler_translation",
    )


def test_first_use_projects_new_routes_from_the_nominal_registry() -> None:
    class _TemporaryAuthoringContext(AuthoringContextDeclaration):
        kind = "temporary_progressive_test"
        route = AuthoringContextRoute(
            title="Temporary registry route",
            use_when="a test proves registry projection",
            knowledge_targets=(
                KnowledgeBaseDocumentTarget("openhcs_architecture_quick_start"),
            ),
        )

    try:
        service, _ = _service()
        content = service.get_authoring_context("first_use").content
        assert 'kind="temporary_progressive_test"' in content
        assert "Temporary registry route" in content
    finally:
        AuthoringContextDeclaration.__registry__.pop("temporary_progressive_test")


def test_every_context_is_complete_within_the_progressive_bound() -> None:
    service, config_service = _service()

    for declaration in AuthoringContextDeclaration.__registry__.values():
        route = declaration.require_route()
        context = service.get_authoring_context(declaration.require_kind())

        assert context.content.startswith(f"=== {route.title.upper()} ===")
        assert len(context.content) < 16_000
        assert "...<truncated" not in context.content
        assert "=== DEEPEN ONLY WHEN NEEDED ===" in context.content
        for target in route.knowledge_targets:
            assert target.document_id in context.content
            assert f"Summary for {target.document_id}" in context.content

    assert config_service.requests == ["global", "pipeline"]


def test_task_contexts_expose_only_the_next_relevant_boundary() -> None:
    service, _ = _service()
    first_use = service.get_authoring_context("first_use").content
    ui = service.get_authoring_context("ui_visible_workflow").content
    folder = service.get_authoring_context("folder_onboarding").content
    pipeline = service.get_authoring_context("pipeline").content
    custom = service.get_authoring_context("custom_function").content
    debugging = service.get_authoring_context("debugging").content
    viewer = service.get_authoring_context("viewer_review").content
    cellprofiler = service.get_authoring_context("cellprofiler_translation").content

    assert "CHOOSE ONE TASK ROUTE" in first_use
    assert "SOURCE-BINDING WORKFLOW" not in first_use
    assert "UI-VISIBLE WORKFLOW" not in first_use
    assert "REGISTERED OPENHCS FUNCTIONS" not in first_use

    assert "UI-VISIBLE WORKFLOW" in ui
    assert "FOLDER ONBOARDING WORKFLOW" not in ui
    assert "HEADLESS EXECUTION WORKFLOW" not in ui
    assert "'view_results' action relates" in ui
    assert "Do not select a surface by title matching" in ui
    assert "object/source identity" in ui
    assert "not a mirror of dialog tabs or table cells" in ui
    assert "reconciling its object identifiers and row cardinality" in ui
    assert "do not scrape the widget tree" in ui

    assert "FOLDER ONBOARDING WORKFLOW" in folder
    assert "openhcs_sample_plate_image" in folder
    assert "REGISTERED INGESTION OWNERS (LIVE REGISTRY)" in folder
    assert "opera_phenix: role=format_specific" in folder
    assert "bioformats: role=broad_structured_store" in folder
    assert "source_bindings: role=declared_file_fallback" in folder
    assert "format_specific_handler_candidates" in folder
    assert "declaration-owned detection/subset policy" in folder
    assert "complete detection contract" in folder
    assert "separate ordinary 2-D files" in folder
    assert "processing_config.variable_components" in folder
    assert "full named physical source universe and the inputs consumed" in folder
    assert "nominal handler projection" in folder
    assert "resolved step-local subset and order" in folder
    assert "Keep three layers separate" in folder
    assert "zero-based positions in that per-step assembled stack" in folder
    assert "Source provenance remains channel 4 and channel 1" in folder
    assert "route provenance retains those declared values when" in folder
    assert "openhcs_get_viewer_window_payloads" in folder
    assert "unexpectedly empty compiled step plan" in folder
    assert "MAP2 channel 2 is valid in the source workspace" in folder
    assert "leaked source only on a current route" in folder
    assert "route from another submission or a step that selects MAP2" in folder
    assert "simplest canonical MetaXpress neurite step" in folder
    assert "nuclear index 0, cell-body index 1, and neurite index 2" in folder
    assert "only the legacy shared-signal model" in folder
    assert "not equivalent to MAP2-seeded analysis" in folder
    assert "generic compiler regression protects any explicit ordered" in folder
    assert "variable_components=[CHANNEL] assembles that stack" in folder
    assert "normalizes group_by to GroupBy.NONE" in folder
    assert "do not reinterpret a previous-step output" in folder
    assert "Follow current artifact provenance" in folder
    assert "Use SourceBindingsHandler only when an arbitrary image folder" not in folder
    assert "openhcs_ui_sample_selected_plate_image" not in folder
    assert "CONFIG SCHEMA HINTS" not in folder

    assert "PIPELINE AUTHORING WORKFLOW" in pipeline
    assert "CONFIG SCHEMA HINTS" in pipeline
    assert "REGISTERED OPENHCS FUNCTIONS" not in pipeline
    assert "RUNTIME AND UI COORDINATION" not in pipeline
    assert "Five independent questions govern each step" in pipeline
    assert "not a third InputSource value" in pipeline
    assert "runtime values are only available during execution" in pipeline
    assert "incremental add-step request does not express a list chain" in pipeline
    assert "Pipeline composition is registry- and contract-driven" in pipeline
    assert "Any registered callable" in pipeline
    assert "reflected CallableContract is compatible" in pipeline
    assert "CLAHE, median denoising, background subtraction" in pipeline
    assert "preprocessing method selection validation" in pipeline
    assert "assay expert still owns expected biology" in pipeline
    assert "openhcs_search_functions" in pipeline
    assert "openhcs_describe_function" in pipeline
    assert "request the custom_function context" in pipeline
    assert "do not infer that BaSiCPy is usable" in pipeline
    assert "MetaXpress neurite outgrowth can still follow" in pipeline
    assert "more explicit module-by-module reference" in pipeline

    assert "A reviewed custom function becomes an ordinary registry-described" in custom
    assert "openhcs_search_functions" in custom
    assert "example preprocessing intents, not a promised built-in catalog" in custom
    assert "memory decorator, ProcessingContract, and artifact declarations" in custom
    assert "Path-bearing topology belongs in one SpatialGraphArtifactType" in custom
    assert "do not reduce graph semantics to a mask" in custom

    assert "DEBUGGING WORKFLOW" in debugging
    assert "pipeline_debug_toolbar.session" in debugging
    assert "compiled intent, not proof" in debugging
    assert "paused worker directly" in debugging
    assert "openhcs_inspect_debug_runtime_values" in debugging
    assert "Pin one current evidence scope" in debugging
    assert "execution_id, debug_session_id" in debugging
    assert "resolved/compiled source-binding order" in debugging
    assert "never merge persistent layers or artifacts" in debugging
    assert "Before changing biological thresholds" in debugging
    assert "openhcs_sample_plate_image" in debugging
    assert "openhcs_sample_viewer_window_image" in debugging
    assert "Pixel values are omitted by default from viewer sampling" in debugging
    assert "include_array_values=true" in debugging
    assert "max_array_elements" in debugging
    assert "openhcs_get_viewer_window_payloads" in debugging
    assert "openhcs_summarize_viewer_window_rois" in debugging
    assert "schema-bearing per-object measurement rows" in debugging
    assert "follow the 'view_results' action's `related_state_surface_ids`" in debugging
    assert "never guess from a title substring" in debugging
    assert "bounded retained-table authority" in debugging
    assert "row/object cardinality" in debugging
    assert "visual interpretation with the biologist" in debugging
    assert "Screenshots are secondary presentation evidence" in debugging

    assert "VIEWER REVIEW WORKFLOW" in viewer
    assert "PIPELINE AUTHORING WORKFLOW" not in viewer
    assert "SOURCE-BINDING WORKFLOW" not in viewer
    assert "view one specific step" in viewer
    assert "step_materialization_config persists" in viewer
    assert "pipeline_config.well_filter_config" in viewer
    assert "path_planning_config.well_filter=0" in viewer
    assert "Start from the user's scientific question" in viewer
    assert "same stable object or label identity" in viewer
    assert "derive matching display colors from that identity" in viewer
    assert "final SpatialGraph path Shapes layer" in viewer
    assert "SWC is the persistent morphology projection" in viewer
    assert "OpenHCS Napari reader or Fiji SNT" in viewer
    assert "standard SWC cannot retain arbitrary edge measurements" in viewer
    assert "Layer existence and nonzero pixels prove transport" in viewer
    assert "user-controlled presentation state" in viewer
    assert "raw route payloads, label identities" in viewer
    assert "Review one current execution in raw-evidence order" in viewer
    assert "structural evidence only" in viewer
    assert "cannot establish pixel-level segmentation or tracing completeness" in viewer
    assert "explicit array slices and array values" in viewer
    assert "Do not wait for the user to find a missed region by zooming" in viewer
    assert "rank strong unassigned residual components" in viewer
    assert "exact raw-versus-result pixels or rasterized shapes" in viewer
    assert "reconcile schema-bearing per-object measurement rows" in viewer
    assert "declared 'view_results' action's `related_state_surface_ids`" in viewer
    assert "Do not select by title substring" in viewer
    assert "Plate Manager Results action" in viewer
    assert "widget-tree cell scraping" in viewer
    assert "interpret the visualization with the biologist" in viewer
    assert "compact opinionated analysis" in viewer
    assert "modular CellProfiler-derived pipeline" in viewer
    assert "do not establish output equivalence or result validation" in viewer
    assert "accepted same-field executions" in viewer

    assert "derives stack axes, post-stack grouping" in cellprofiler
    assert "does not choose native OpenHCS viewer" in cellprofiler


def test_viewer_array_capabilities_expose_value_opt_in_and_bounded_tiling() -> None:
    payload_description = agent_capabilities.get_viewer_window_payloads.description
    sample_description = agent_capabilities.sample_viewer_window_image.description

    assert "Array values are omitted by default" in payload_description
    assert "include_array_values=true" in payload_description
    assert "max_array_elements" in payload_description
    assert "Pixel values are omitted by default" in sample_description
    assert "height*width within max_array_elements" in sample_description
    assert "tile a field" in sample_description


def test_declaration_owned_knowledge_targets_exist_in_the_live_catalog() -> None:
    known_document_ids = {
        document.document_id
        for document in KnowledgeBaseService().list_documents().documents
    }

    for declaration in AuthoringContextDeclaration.__registry__.values():
        targets = declaration.require_route().knowledge_targets
        assert targets
        assert len(targets) == len(set(targets))
        assert {target.document_id for target in targets} <= known_document_ids


def test_context_service_has_no_deep_document_or_function_catalog_mirror() -> None:
    import openhcs.agent.services.llm_context_service as context_module

    source = Path(context_module.__file__).read_text(encoding="utf-8")

    assert "KnowledgeBackedAuthoringContextSection" not in source
    assert "FunctionCatalogSection" not in source
    assert "CoreImportsSection" not in source
    assert "REGISTERED OPENHCS FUNCTIONS" not in source
    assert ".get_document(" not in source
