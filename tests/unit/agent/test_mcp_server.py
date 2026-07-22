import asyncio
import importlib.util
import inspect
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tomllib

import pytest

import openhcs
from openhcs.agent.capabilities import (
    AuthoringLocalCapabilitySurfaceProfile,
    CapabilityKind,
    CapabilityTransport,
    CoreLocalCapabilitySurfaceProfile,
    DesktopLocalCapabilitySurfaceProfile,
    FullLocalCapabilitySurfaceProfile,
    agent_capabilities,
    get_capability_registry,
)
from openhcs.agent.authoring_contexts import AuthoringContextDeclaration
from openhcs.agent.dto.common import AgentError, SCHEMA_VERSION
from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.dto.functions import (
    CustomFunctionRegistrationResult,
    FunctionCatalogEntry,
)
from openhcs.agent.dto.knowledge import KnowledgeBaseDocumentRequest
from openhcs.agent.dto.plate import (
    PlateFileQueryResult,
    PlateFileStreamResult,
    PlateImageSampleResult,
    PlateInspectionConfidence,
    PlateInspectionImageFileSummary,
    PlateInspectionImageRecordSummary,
    PlateInspectionStatus,
    PlatePathInspectionResult,
    SyntheticPlateGenerationResult,
)
from openhcs.agent.dto.ui_bridge import (
    UiCatalogPageMetadata,
    UiObjectStateFieldFilter,
    UiObjectStateFieldHelpRequest,
    UiObjectStateFieldMutationResult,
    UiObjectStateFieldSummary,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeIdentity,
    UiObjectStateScopeSummary,
    UiObjectStateValuePreview,
    UiMutationReceipt,
    UiSemanticAddress,
    UiStateSurfaceDocument,
    UiStateSurfaceIdentity,
    UiStateSurfaceSummary,
    UiWindowIdentity,
    UiWindowSummary,
    UiWindowSnapshotResult,
    UiWidgetActionSummary,
    UiWidgetRect,
    UiWidgetTreeNode,
    UiWidgetTreeResult,
)
from openhcs.agent.dto.viewer import (
    ViewerWindowDescriptor,
    ViewerWindowLayerPayloads,
    ViewerWindowLayerState,
    ViewerWindowNavigationResult,
    ViewerWindowPayloadRecord,
    ViewerWindowPayloadResult,
    ViewerWindowSnapshotResult,
    ViewerWindowStateResult,
)
from openhcs.agent.services.object_state_field_help_service import (
    ObjectStateFieldHelpService,
)
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.object_state_field_projection import (
    ObjectStateFieldListProjector,
)
from openhcs.agent.services.selected_plate_service import SelectedPlateService
from openhcs.agent.services.viewer_window_service import ViewerWindowService
from openhcs.runtime.window_snapshot import WindowSnapshotCaptureScope

import openhcs.mcp.bootstrap as bootstrap
import openhcs.mcp.server as server
from openhcs.mcp.context import OpenHCSAgentContext


def _direct_tool_text(result) -> str:
    """Return first text block from FastMCP structured or legacy direct calls."""
    if hasattr(result, "content"):
        content = result.content
    else:
        content = result[0] if isinstance(result, tuple) else result
    return content[0].text


def _direct_tool_structured(result):
    return result[1] if isinstance(result, tuple) else result.structuredContent


def _selected_plate_mcp_context(
    *,
    ui_bridge_service,
    plate_inspection_service=None,
    plate_streaming_service=None,
):
    plate_inspection_service = plate_inspection_service or SimpleNamespace()
    plate_streaming_service = plate_streaming_service or SimpleNamespace()
    return SimpleNamespace(
        ui_bridge_service=ui_bridge_service,
        plate_inspection_service=plate_inspection_service,
        plate_streaming_service=plate_streaming_service,
        selected_plate_service=SelectedPlateService(
            ui_bridge_service,
            plate_inspection_service,
            plate_streaming_service,
        ),
    )


class _ProjectedViewerWindowService(ViewerWindowService):
    def __init__(self, delegate):
        self._delegate = delegate

    def window_payloads(self, request):
        return self._delegate.window_payloads(request)

    def window_state(self, request):
        return self._delegate.window_state(request)

    def navigate_window(self, request):
        return self._delegate.navigate_window(request)


def _viewer_mcp_context(viewer_window_service):
    return SimpleNamespace(
        viewer_window_service=_ProjectedViewerWindowService(viewer_window_service)
    )


def mcp_help_threshold_function(image=None, threshold: float = 1.0):
    """Apply a threshold.

    Parameters
    ----------
    image
        Input image.
    threshold
        Numeric cutoff used for segmentation.
    """
    return image


def test_mcp_server_module_import_does_not_require_mcp_dependency():
    assert callable(server.build_server)
    assert "main" not in vars(server)


def test_installed_mcp_script_uses_fail_soft_bootstrap_entrypoint():
    with (Path(__file__).resolve().parents[3] / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    assert pyproject["project"]["scripts"]["openhcs-mcp"] == (
        "openhcs.mcp.bootstrap:main"
    )
    assert pyproject["project"]["scripts"]["openhcs-mcp-dev"] == (
        "openhcs.mcp.dev_client:main"
    )


def test_development_extra_installs_mcp_dependency():
    with (Path(__file__).resolve().parents[3] / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    assert "mcp>=1.28,<2" in pyproject["project"]["optional-dependencies"]["dev"]
    assert "mcp>=1.28,<2" in pyproject["project"]["optional-dependencies"]["mcp"]


def test_agent_dto_package_exports_mcp_debugging_contracts():
    import openhcs.agent.dto as dto

    exported_names = (
        "McpServerHealthResult",
        "KnowledgeBaseCatalog",
        "KnowledgeBaseDocument",
        "KnowledgeBaseDocumentRequest",
        "KnowledgeBaseSearchRequest",
        "KnowledgeBaseSearchResult",
        "CustomFunctionRegistrationResult",
        "PlateInspectionBounds",
        "PlateInspectionComponentSummary",
        "PlateInspectionComponentValue",
        "PlateInspectionConfidence",
        "PlateInspectionImageFileSummary",
        "PlateInspectionParseFailure",
        "PlateInspectionParseSummary",
        "PlateInspectionStatus",
        "PlateInspectionValueSource",
        "PlateInspectionWorkspacePreparation",
        "PlateFileQueryRecordSummary",
        "PlateFileQueryRequest",
        "PlateFileQueryResult",
        "PlateFileStreamRequest",
        "PlateFileStreamResult",
        "SelectedPlateFileQueryTarget",
        "PlateImageSampleRequest",
        "PlateImageSampleResult",
        "PlatePathInspectionRequest",
        "PlatePathInspectionResult",
        "PlateWorkspacePreparationOperation",
        "SelectedPlateFileQueryResult",
        "SelectedPlateFileStreamResult",
        "SelectedPlateImageInspectionResult",
        "SelectedPlateImageSampleResult",
        "SyntheticPlateGenerationRequest",
        "SyntheticPlateGenerationResult",
        "SourceWorkspaceFileRecord",
        "SourceWorkspaceSummary",
        "UiSelectedPlateWorkflowKind",
        "UiSelectedPlateWorkflowRequest",
        "UiSelectedPlateWorkflowResult",
        "UiObjectStateFieldHelpRequest",
        "UiObjectStateFieldHelpResult",
        "UiObjectStateFieldMutationRequest",
        "UiObjectStateFieldMutationResult",
        "UiWidgetRect",
        "UiWidgetTreeNode",
        "UiWidgetTreeRequest",
        "UiWidgetTreeResult",
        "UiWidgetActionSummary",
        "ViewerWindowLayerPayloads",
        "ViewerWindowPayloadRecord",
        "ViewerWindowPayloadRequest",
        "ViewerWindowPayloadResult",
    )

    for exported_name in exported_names:
        assert exported_name in vars(dto)


def test_mcp_server_builds_when_optional_dependency_is_installed():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    assert built is not None


def test_mcp_server_publishes_canonical_instructions():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    assert built.instructions == server.MCP_SERVER_INSTRUCTIONS
    assert "openhcs_health_check" in built.instructions
    assert "kind='first_use'" in built.instructions
    assert "compact orientation and intent router" in built.instructions
    assert "instead of loading every guide" in built.instructions
    assert "read it completely before advising" not in built.instructions
    assert "expert curriculum" not in built.instructions
    for kind in AuthoringContextDeclaration.allowed_values():
        assert kind in built.instructions
    assert "PipelineDocument containing PipelineConfig" in built.instructions
    assert "SourceBindingsConfig" in built.instructions
    assert "SourceBindingsHandler is the fallback ingestion owner" in built.instructions
    assert "CZI, OME-TIFF" in built.instructions
    assert "bounded representative samples" in built.instructions
    assert "openhcs_list_capabilities" in built.instructions
    assert "surface profile, workflow groups, target contexts" in built.instructions
    health_index = built.instructions.index("openhcs_health_check")
    first_use_index = built.instructions.index("kind='first_use'")
    capability_index = built.instructions.index("openhcs_list_capabilities")
    assert health_index < first_use_index < capability_index
    assert "kind='first_use' before choosing tools" in built.instructions
    assert "openhcs_search_knowledge" in built.instructions
    assert "openhcs_describe_config_schema" in built.instructions
    assert "already-running OpenHCS GUI" in built.instructions
    assert "same typed declarations" in built.instructions
    assert "names begin" not in built.instructions
    assert "compile before running" in built.instructions
    assert "structured execution results" in built.instructions


def test_mcp_server_factory_receives_canonical_identity_and_binds_tools():
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp.server.fastmcp import FastMCP

    construction_calls = []

    def factory(name, *, instructions):
        construction_calls.append((name, instructions))
        return FastMCP(name, instructions=instructions)

    built = server.build_server(fastmcp_factory=factory)
    listed_tools = built.list_tools()
    tools = (
        asyncio.run(listed_tools) if inspect.isawaitable(listed_tools) else listed_tools
    )

    assert construction_calls == [("OpenHCS", server.MCP_SERVER_INSTRUCTIONS)]
    assert "openhcs_health_check" in {tool.name for tool in tools}


def test_mcp_hosted_surface_is_generated_from_transport_availability():
    if importlib.util.find_spec("mcp") is None:
        return

    hosted_registry = get_capability_registry(
        CapabilityTransport.HOSTED_STREAMABLE_HTTP
    )
    all_registry = get_capability_registry()
    built = server.build_server(
        capability_transport=CapabilityTransport.HOSTED_STREAMABLE_HTTP
    )
    listed_tools = built.list_tools()
    tools = (
        asyncio.run(listed_tools) if inspect.isawaitable(listed_tools) else listed_tools
    )
    listed_resources = built.list_resources()
    resources = (
        asyncio.run(listed_resources)
        if inspect.isawaitable(listed_resources)
        else listed_resources
    )

    expected_tool_names = {
        capability.name
        for capability in hosted_registry.capabilities
        if capability.kind is CapabilityKind.TOOL
    }
    expected_resource_names = {
        capability.name
        for capability in hosted_registry.capabilities
        if capability.kind is CapabilityKind.RESOURCE
    }
    local_only_names = {
        capability.name
        for capability in all_registry.capabilities
        if not capability.supports_transport(CapabilityTransport.HOSTED_STREAMABLE_HTTP)
    }

    assert {tool.name for tool in tools} == expected_tool_names
    assert {str(resource.uri) for resource in resources} == expected_resource_names
    assert expected_tool_names.isdisjoint(local_only_names)
    assert all(tool.annotations.readOnlyHint is True for tool in tools)
    assert built.instructions == server.MCP_HOSTED_SERVER_INSTRUCTIONS


def test_mcp_hosted_capability_discovery_matches_registered_surface():
    if importlib.util.find_spec("mcp") is None:
        return

    hosted_registry = get_capability_registry(
        CapabilityTransport.HOSTED_STREAMABLE_HTTP
    )
    expected_names = {capability.name for capability in hosted_registry.capabilities}

    async def call_capability_discovery():
        built = server.build_server(
            capability_transport=CapabilityTransport.HOSTED_STREAMABLE_HTTP
        )
        return (
            await asyncio.wait_for(
                built.call_tool("openhcs_list_capabilities", {}),
                timeout=2,
            ),
            await asyncio.wait_for(
                built.read_resource("openhcs://capabilities"),
                timeout=2,
            ),
        )

    result, resource_result = asyncio.run(call_capability_discovery())
    payload = json.loads(_direct_tool_text(result))
    resource_payload = json.loads(resource_result[0].content)

    assert {
        capability["name"] for capability in payload["capabilities"]
    } == expected_names
    assert {
        capability["name"] for capability in resource_payload["capabilities"]
    } == expected_names
    assert all(
        capability["transport_availability"]
        == [
            CapabilityTransport.LOCAL_STDIO.value,
            CapabilityTransport.HOSTED_STREAMABLE_HTTP.value,
        ]
        for capability in payload["capabilities"]
    )


def test_mcp_invocation_observer_receives_nominal_capability_outcome():
    if importlib.util.find_spec("mcp") is None:
        return

    observed = []

    async def call_capability_discovery():
        built = server.build_server(
            capability_transport=CapabilityTransport.HOSTED_STREAMABLE_HTTP,
            invocation_observer=lambda capability, outcome: observed.append(
                (capability, outcome)
            ),
        )
        tool_result = await asyncio.wait_for(
            built.call_tool("openhcs_list_capabilities", {}),
            timeout=2,
        )
        resource_result = await asyncio.wait_for(
            built.read_resource("openhcs://capabilities"),
            timeout=2,
        )
        return tool_result, resource_result

    asyncio.run(call_capability_discovery())

    assert observed == [
        (
            agent_capabilities.list_capabilities,
            server.McpInvocationOutcome.SUCCEEDED,
        ),
        (
            agent_capabilities.capabilities,
            server.McpInvocationOutcome.SUCCEEDED,
        ),
    ]


def test_mcp_tool_annotations_are_derived_from_capability_registry():
    if importlib.util.find_spec("mcp") is None:
        return

    capabilities = {
        capability.name: capability
        for capability in get_capability_registry().capabilities
        if capability.kind.value == "tool"
    }
    built = server.build_server()
    listed_tools = built.list_tools()
    tools = (
        asyncio.run(listed_tools) if inspect.isawaitable(listed_tools) else listed_tools
    )

    assert {tool.name for tool in tools} == set(capabilities)
    for tool in tools:
        capability = capabilities[tool.name]
        read_only = not capability.mutating and not capability.side_effects
        open_world = bool(
            capability.side_effects
            or capability.requires_network
            or capability.data_exposure
            or capability.security_requirements
        )
        assert tool.annotations.title == capability.title
        assert tool.annotations.readOnlyHint is read_only
        assert tool.annotations.destructiveHint is (not read_only)
        assert tool.annotations.idempotentHint is read_only
        assert tool.annotations.openWorldHint is open_world


@pytest.mark.parametrize(
    "profile",
    (
        FullLocalCapabilitySurfaceProfile(),
        DesktopLocalCapabilitySurfaceProfile(),
        AuthoringLocalCapabilitySurfaceProfile(),
        CoreLocalCapabilitySurfaceProfile(),
    ),
)
def test_mcp_local_surface_profiles_bind_exact_selected_registry(profile):
    if importlib.util.find_spec("mcp") is None:
        return

    registry = get_capability_registry(capability_surface_profile=profile)
    built = server.build_server(capability_surface_profile=profile)
    tools = asyncio.run(built.list_tools())
    resources = asyncio.run(built.list_resources())

    assert {tool.name for tool in tools} == {
        capability.name
        for capability in registry.capabilities
        if capability.kind is CapabilityKind.TOOL
    }
    assert {str(resource.uri) for resource in resources} == {
        capability.name
        for capability in registry.capabilities
        if capability.kind is CapabilityKind.RESOURCE
    }
    discovery_result = asyncio.run(built.call_tool("openhcs_list_capabilities", {}))
    discovery_payload = _direct_tool_structured(discovery_result)
    assert discovery_payload["surface_profile"] == profile.name
    assert {capability["name"] for capability in discovery_payload["capabilities"]} == {
        capability.name for capability in registry.capabilities
    }


def test_mcp_tools_advertise_structured_json_and_nominal_output_contract():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server(
        capability_surface_profile=CoreLocalCapabilitySurfaceProfile()
    )
    tools = asyncio.run(built.list_tools())
    capabilities = {
        capability.name: capability
        for capability in get_capability_registry(
            capability_surface_profile=CoreLocalCapabilitySurfaceProfile()
        ).capabilities
        if capability.kind is CapabilityKind.TOOL
    }

    assert tools
    assert all(tool.outputSchema is not None for tool in tools)
    assert all(
        tool.meta == {"openhcs/outputContract": capabilities[tool.name].output_type}
        for tool in tools
    )

    result = asyncio.run(built.call_tool("openhcs_health_check", {}))
    assert json.loads(_direct_tool_text(result)) == _direct_tool_structured(result)


def test_mcp_dev_client_prefers_structured_content_and_preserves_tool_metadata():
    import openhcs.mcp.dev_client as dev_client
    from openhcs.mcp.dev_client_core import mcp_tool_metadata_from_wire

    result = server.to_jsonable(
        server.McpToolErrorResult(
            schema_version=SCHEMA_VERSION,
            ok=False,
            tool="openhcs_test",
            errors=(AgentError(code="test", message="structured"),),
        )
    )
    projected = dev_client.McpDevToolResult.from_payload(
        "openhcs_test",
        {
            "isError": False,
            "content": [{"type": "text", "text": '{"legacy": true}'}],
            "structuredContent": result,
        },
    )
    metadata = mcp_tool_metadata_from_wire(
        {
            "name": "openhcs_test",
            "title": "Test",
            "description": "Test tool",
            "inputSchema": {"type": "object"},
            "outputSchema": {"type": "object"},
            "annotations": {"readOnlyHint": True},
            "_meta": {"openhcs/outputContract": "TestResult"},
        }
    )

    assert projected.payloads == (result,)
    assert metadata.title == "Test"
    assert metadata.output_schema == {"type": "object"}
    assert metadata.meta == {"openhcs/outputContract": "TestResult"}


def test_mcp_bootstrap_defaults_to_desktop_surface():
    if importlib.util.find_spec("mcp") is None:
        return

    built = bootstrap.build_bootstrapped_server()
    tools = asyncio.run(built.list_tools())
    desktop_registry = get_capability_registry(
        capability_surface_profile=DesktopLocalCapabilitySurfaceProfile()
    )

    assert {tool.name for tool in tools} == {
        capability.name
        for capability in desktop_registry.capabilities
        if capability.kind is CapabilityKind.TOOL
    }
    assert bootstrap._build_parser().parse_args(()).surface == "desktop"


@pytest.mark.parametrize(
    ("metadata", "expected_open_world"),
    (
        ({}, False),
        ({"side_effects": ("writes_state",)}, True),
        ({"requires_network": True}, True),
        ({"data_exposure": ("local_path",)}, True),
        ({"security_requirements": ("user_consent",)}, True),
    ),
)
def test_mcp_open_world_annotation_projects_every_authoritative_signal(
    metadata,
    expected_open_world,
):
    if importlib.util.find_spec("mcp") is None:
        return

    from openhcs.agent.capabilities import AgentCapabilitySpec, CapabilityKind

    capability = AgentCapabilitySpec(
        name="openhcs_test_projection",
        kind=CapabilityKind.TOOL,
        title="Projection test",
        description="Test capability metadata projection.",
        service="test",
        **metadata,
    )

    annotations = server._mcp_tool_annotations(capability)

    assert annotations.openWorldHint is expected_open_world


def test_mcp_tools_have_blind_agent_descriptions():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools

    assert tools
    assert all(tool.description for tool in tools)


def test_mcp_create_pipeline_uses_optional_config_reference_in_full_document():
    if importlib.util.find_spec("mcp") is None:
        return

    config_service = ConfigService()
    config_ref = config_service.create(
        "pipeline",
        ConfigPatch(
            config_type="PipelineConfig",
            values={"well_filter_config": {"well_filter": 2}},
        ),
    )
    built = server.build_server(OpenHCSAgentContext(config_service=config_service))
    listed_tools = built.list_tools()
    tools = (
        asyncio.run(listed_tools) if inspect.isawaitable(listed_tools) else listed_tools
    )
    create_schema = {tool.name: tool.inputSchema for tool in tools}[
        "openhcs_create_pipeline"
    ]

    async def create_and_render():
        created = await built.call_tool(
            "openhcs_create_pipeline",
            {"pipeline_config_id": config_ref.config_id},
        )
        pipeline_id = json.loads(_direct_tool_text(created))["pipeline_id"]
        return await built.call_tool(
            "openhcs_render_pipeline_source",
            {"pipeline_id": pipeline_id},
        )

    rendered = asyncio.run(create_and_render())
    source = json.loads(_direct_tool_text(rendered))["source"]

    assert "pipeline_config_id" in create_schema["properties"]
    assert "pipeline_config_id" not in create_schema.get("required", ())
    assert "pipeline_config = PipelineConfig(" in source
    assert "well_filter=2" in source
    assert "pipeline_steps = []" in source


def test_mcp_tool_descriptions_expose_debugging_result_contracts():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools

    descriptions = {tool.name: tool.description for tool in tools}
    schemas = {tool.name: tool.inputSchema for tool in tools}

    assert "per-axis" in descriptions["openhcs_get_viewer_window_payloads"]
    assert "arrays and shapes" in descriptions["openhcs_get_viewer_window_payloads"]
    viewer_payload_properties = schemas["openhcs_get_viewer_window_payloads"][
        "properties"
    ]
    assert "include_response" in viewer_payload_properties
    assert "axis_indices" in viewer_payload_properties
    assert "array_slices" in viewer_payload_properties
    assert "bounded image records" in descriptions[
        "openhcs_sample_viewer_window_image"
    ]
    viewer_sample_properties = schemas["openhcs_sample_viewer_window_image"][
        "properties"
    ]
    assert "route_key" in viewer_sample_properties
    assert "route_key" not in schemas["openhcs_sample_viewer_window_image"].get(
        "required",
        [],
    )
    assert "axis_indices" in viewer_sample_properties
    assert "height" in viewer_sample_properties
    assert "width" in viewer_sample_properties
    assert "include_array_values" in viewer_sample_properties
    assert "ROI counts" in descriptions["openhcs_summarize_viewer_window_rois"]
    roi_summary_properties = schemas["openhcs_summarize_viewer_window_rois"][
        "properties"
    ]
    assert "route_key" in roi_summary_properties
    assert "route_key" not in schemas["openhcs_summarize_viewer_window_rois"].get(
        "required",
        [],
    )
    assert "axis_indices" in roi_summary_properties
    assert "max_examples" in roi_summary_properties
    assert "bounded" in descriptions["openhcs_get_viewer_window_state"]
    viewer_state_properties = schemas["openhcs_get_viewer_window_state"]["properties"]
    assert "route_key" in viewer_state_properties
    assert "include_component_values" in viewer_state_properties
    assert "max_component_values_per_layer" in viewer_state_properties
    assert "include_payload_summaries" in viewer_state_properties
    assert "max_payload_summaries_per_layer" in viewer_state_properties
    assert "include_response" in viewer_state_properties
    assert "bounded" in descriptions["openhcs_describe_function"]
    assert "max_doc_chars" in schemas["openhcs_describe_function"]["properties"]
    assert "compact_signature" in schemas["openhcs_describe_function"]["properties"]
    assert "kind='first_use'" in descriptions["openhcs_get_authoring_context"]
    assert "before choosing tools" in descriptions["openhcs_get_authoring_context"]
    assert "Diagnostic-only" in descriptions["openhcs_inspect_plate_path"]
    assert "PlateManager code document" in descriptions["openhcs_inspect_plate_path"]
    assert "CustomFunctionManager" in descriptions["openhcs_register_custom_function"]
    custom_function_properties = schemas["openhcs_register_custom_function"][
        "properties"
    ]
    assert "source_code" in custom_function_properties
    assert "persist" in custom_function_properties
    assert "revision" in descriptions["openhcs_ui_apply_code_document"]
    assert "virtual path" in descriptions["openhcs_stream_plate_files_to_viewer"]
    assert "managed viewer" in descriptions["openhcs_stream_plate_files_to_viewer"]
    assert (
        "current PlateManager selection"
        in descriptions["openhcs_ui_stream_selected_plate_files_to_viewer"]
    )
    assert "snapshot" in descriptions["openhcs_ui_apply_code_document"]
    assert "undo" in descriptions["openhcs_ui_apply_code_document"]
    assert "request_token" in schemas["openhcs_ui_apply_code_document"]["properties"]
    assert "flat document_id" in descriptions["openhcs_ui_list_code_documents"]
    assert "flat surface_id" in descriptions["openhcs_ui_list_state_surfaces"]
    assert "flat widget_id/action_id" in descriptions["openhcs_ui_list_actions"]
    assert "flat window_id" in descriptions["openhcs_ui_list_windows"]
    state_surface_properties = schemas["openhcs_ui_get_state_surface"]["properties"]
    assert "base_revision_token" in state_surface_properties
    assert "revision_token" not in state_surface_properties
    validate_properties = schemas["openhcs_ui_validate_code_document"]["properties"]
    assert "base_revision_token" in validate_properties
    assert "revision_token" not in validate_properties
    apply_properties = schemas["openhcs_ui_apply_code_document"]["properties"]
    assert "base_revision_token" in apply_properties
    assert "revision_token" not in apply_properties
    assert "clean=False" in descriptions["openhcs_ui_get_code_document"]
    assert "full" in descriptions["openhcs_ui_get_code_document"]
    assert "read-only inspection" in descriptions["openhcs_inspect_plate_path"]
    assert "microscope metadata" in descriptions["openhcs_inspect_plate_path"]
    assert "max_files_to_parse" in schemas["openhcs_inspect_plate_path"]["properties"]
    assert "bounded synthetic" in descriptions["openhcs_generate_synthetic_plate"]
    synthetic_plate_properties = schemas["openhcs_generate_synthetic_plate"][
        "properties"
    ]
    assert "output_dir" in synthetic_plate_properties
    assert "overlap_percent" in synthetic_plate_properties
    assert "wavelengths" in synthetic_plate_properties
    assert "wells" in synthetic_plate_properties
    assert "image/result file records" in descriptions["openhcs_query_plate_files"]
    query_plate_files_properties = schemas["openhcs_query_plate_files"]["properties"]
    assert "kind" in query_plate_files_properties
    assert "path_contains" in query_plate_files_properties
    assert "well" in query_plate_files_properties
    assert "include_previews" in query_plate_files_properties
    assert "virtual/source path" in descriptions["openhcs_sample_plate_image"]
    assert "bounded pixels" in descriptions["openhcs_sample_plate_image"]
    assert "image_path" in schemas["openhcs_sample_plate_image"]["properties"]
    assert "resolution_index" in schemas["openhcs_sample_plate_image"]["properties"]
    assert (
        "max_auto_resolution_size"
        in schemas["openhcs_sample_plate_image"]["properties"]
    )
    assert (
        "current PlateManager selection"
        in descriptions["openhcs_ui_inspect_selected_plate_images"]
    )
    selected_plate_image_properties = schemas[
        "openhcs_ui_inspect_selected_plate_images"
    ]["properties"]
    assert "max_sample_files" in selected_plate_image_properties
    assert "connection" in selected_plate_image_properties
    assert (
        "current PlateManager selection"
        in descriptions["openhcs_ui_query_selected_plate_files"]
    )
    selected_plate_file_properties = schemas["openhcs_ui_query_selected_plate_files"][
        "properties"
    ]
    assert "kind" in selected_plate_file_properties
    assert "target" in selected_plate_file_properties
    assert "path_contains" in selected_plate_file_properties
    assert "include_previews" in selected_plate_file_properties
    assert "connection" in selected_plate_file_properties
    assert (
        "Sample a selected-plate image"
        in descriptions["openhcs_ui_sample_selected_plate_image"]
    )
    selected_plate_sample_properties = schemas[
        "openhcs_ui_sample_selected_plate_image"
    ]["properties"]
    assert "image_path" in selected_plate_sample_properties
    assert "target" in selected_plate_sample_properties
    assert "include_array_values" in selected_plate_sample_properties
    assert "connection" in selected_plate_sample_properties
    assert "clickable geometry" in descriptions["openhcs_ui_get_widget_tree"]
    assert "action kinds" in descriptions["openhcs_ui_get_widget_tree"]
    assert "route-local axis indices" in descriptions["openhcs_navigate_viewer_window"]
    navigate_properties = schemas["openhcs_navigate_viewer_window"]["properties"]
    assert "route_key" in navigate_properties
    assert "axis_indices" in navigate_properties
    assert (
        "only selected viewer layers"
        in descriptions["openhcs_isolate_viewer_window_layers"]
    )
    isolate_properties = schemas["openhcs_isolate_viewer_window_layers"]["properties"]
    assert "visible_route_keys" in isolate_properties
    assert "selected_route_key" in isolate_properties
    assert "axis_indices" in isolate_properties
    viewer_validation_properties = schemas["openhcs_validate_viewer_window_state"][
        "properties"
    ]
    assert "route_key" in viewer_validation_properties
    assert "include_state" in viewer_validation_properties
    assert "compact_actions" in schemas["openhcs_ui_get_widget_tree"]["properties"]
    assert (
        "maximum_item_model_nodes"
        in schemas["openhcs_ui_get_widget_tree"]["properties"]
    )
    assert (
        "include_field_values"
        in schemas["openhcs_ui_list_object_state_scopes"]["properties"]
    )
    assert (
        "include_system_scopes"
        in schemas["openhcs_ui_list_object_state_scopes"]["properties"]
    )
    assert (
        "scope_visibility"
        not in schemas["openhcs_ui_list_object_state_scopes"]["properties"]
    )
    assert (
        "include_field_values"
        in schemas["openhcs_ui_get_object_state_fields"]["properties"]
    )
    assert "ObjectState field" in descriptions["openhcs_ui_describe_object_state_field"]
    object_state_help_properties = schemas["openhcs_ui_describe_object_state_field"][
        "properties"
    ]
    assert "object_state_scope_id" in object_state_help_properties
    assert "field_path" in object_state_help_properties
    assert "field_path" in schemas["openhcs_ui_describe_object_state_field"]["required"]
    assert (
        "object_state_scope_id"
        not in schemas["openhcs_ui_describe_object_state_field"]["required"]
    )
    assert "max_description_chars" in object_state_help_properties
    add_step_properties = schemas["openhcs_add_function_step"]["properties"]
    assert "step_config_overrides" in add_step_properties
    assert "dtype_config" not in add_step_properties
    assert "napari_streaming_config" not in add_step_properties


def test_mcp_widget_tree_projection_compacts_empty_action_fields():
    result = UiWidgetTreeResult(
        schema_version=SCHEMA_VERSION,
        window_id="plate_manager",
        projected=True,
        actionable_widgets=(
            UiWidgetActionSummary(
                path=(0,),
                path_id="root/0",
                child_index=0,
                class_name="QPushButton",
                object_name="",
                accessible_name="",
                accessible_description="",
                label="Compile",
                visible=True,
                enabled=True,
                geometry=UiWidgetRect(x=8, y=160, width=72, height=24),
                global_geometry=UiWidgetRect(x=18, y=180, width=72, height=24),
                action_kinds=("click",),
                clickable=True,
                checkable=False,
                checked=False,
                current_index=None,
                current_text=None,
                item_count=None,
                tool_tip="",
            ),
        ),
    )

    payload = server.McpWidgetTreePayloadProjection().project(result)
    action = payload["actionable_widgets"][0]

    assert set(action) == {
        "path",
        "path_id",
        "child_index",
        "class_name",
        "label",
        "visible",
        "enabled",
        "geometry",
        "global_geometry",
        "action_kinds",
        "clickable",
    }
    assert action["label"] == "Compile"
    assert action["geometry"] == {"x": 8, "y": 160, "width": 72, "height": 24}


def test_mcp_widget_tree_projection_preserves_semantic_action_values():
    payload = {
        "schema_version": SCHEMA_VERSION,
        "window_id": "global_config",
        "projected": True,
        "actionable_widgets": [
            {
                "path": [0, 1],
                "path_id": "root/0/1",
                "child_index": 1,
                "class_name": "NoScrollComboBox",
                "object_name": "",
                "accessible_name": "",
                "accessible_description": "",
                "label": "auto",
                "visible": True,
                "enabled": True,
                "geometry": {"x": 4, "y": 8, "width": 120, "height": 22},
                "global_geometry": {"x": 14, "y": 18, "width": 120, "height": 22},
                "action_kinds": ["choice"],
                "clickable": True,
                "checkable": True,
                "checked": False,
                "current_index": 0,
                "current_text": "",
                "item_count": 0,
                "tool_tip": "",
                "context_label": "Microscope",
                "action_role": None,
                "semantic_address": {"field_path": "microscope"},
                "object_state_scope_id": "global_config",
                "field_path": "microscope",
                "dirty": False,
                "signature_diff": True,
                "last_changed": False,
                "semantic_markers": ["_"],
                "raw_value": None,
                "resolved_value": 0,
                "raw_value_preview": {
                    "type_name": "None",
                    "is_none": True,
                    "text": "None",
                    "truncated": False,
                },
                "resolved_value_preview": {
                    "type_name": "int",
                    "is_none": False,
                    "text": "0",
                    "truncated": False,
                },
                "raw_value_is_none": True,
                "resolved_value_is_none": False,
                "inherited_value": True,
                "provenance": {"source": "default"},
            },
        ],
    }

    compact = server.McpWidgetTreePayloadProjection.compact_payload(payload)
    action = compact["actionable_widgets"][0]

    assert action["checkable"] is True
    assert action["checked"] is False
    assert action["current_index"] == 0
    assert action["item_count"] == 0
    assert action["context_label"] == "Microscope"
    assert action["semantic_address"] == {"field_path": "microscope"}
    assert action["signature_diff"] is True
    assert "dirty" not in action
    assert action["semantic_markers"] == ["_"]
    assert "raw_value" not in action
    assert action["raw_value_is_none"] is True
    assert "resolved_value" not in action
    assert action["raw_value_preview"]["text"] == "None"
    assert action["resolved_value_preview"]["text"] == "0"
    assert "resolved_value_is_none" not in action
    assert action["inherited_value"] is True
    assert action["provenance"] == {"source": "default"}


def test_mcp_ui_catalog_projection_flattens_identity_ids():
    documents = server.McpUiCatalogPayloadProjection("documents").compact_payload(
        {
            "documents": [
                {
                    "identity": {"document_id": "plate_manager.orchestrator_config"},
                    "title": "Plate manager orchestrator config",
                }
            ]
        }
    )
    actions = server.McpUiCatalogPayloadProjection("actions").compact_payload(
        {
            "actions": [
                {
                    "identity": {
                        "widget_id": "plate_manager",
                        "action_id": "add_plate",
                    },
                    "enabled": True,
                }
            ]
        }
    )
    windows = server.McpUiCatalogPayloadProjection("windows").compact_payload(
        {
            "windows": [
                {
                    "identity": {"window_id": "global_config"},
                    "title": "Configuration - GlobalPipelineConfig",
                }
            ]
        }
    )

    assert documents["documents"][0]["document_id"] == (
        "plate_manager.orchestrator_config"
    )
    assert "identity" not in documents["documents"][0]
    assert actions["actions"][0]["widget_id"] == "plate_manager"
    assert actions["actions"][0]["action_id"] == "add_plate"
    assert "identity" not in actions["actions"][0]
    assert windows["windows"][0]["window_id"] == "global_config"
    assert "identity" not in windows["windows"][0]


def test_mcp_widget_tree_projection_can_return_full_action_fields():
    result = UiWidgetTreeResult(
        schema_version=SCHEMA_VERSION,
        window_id="plate_manager",
        projected=True,
        actionable_widgets=(
            UiWidgetActionSummary(
                path=(0,),
                path_id="root/0",
                child_index=0,
                class_name="QPushButton",
                object_name="",
                accessible_name="",
                accessible_description="",
                label="Compile",
                visible=True,
                enabled=True,
                geometry=UiWidgetRect(x=8, y=160, width=72, height=24),
                global_geometry=UiWidgetRect(x=18, y=180, width=72, height=24),
                action_kinds=("click",),
                clickable=True,
                checkable=False,
                checked=False,
                current_index=None,
                current_text=None,
                item_count=None,
                tool_tip="",
            ),
        ),
    )

    payload = server.McpWidgetTreePayloadProjection(compact_actions=False).project(
        result
    )
    action = payload["actionable_widgets"][0]

    assert "object_name" in action
    assert "accessible_name" in action
    assert "raw_value" in action
    assert action["raw_value"] is None


def test_mcp_widget_tree_binding_projects_request_and_compact_actions():
    if importlib.util.find_spec("mcp") is None:
        return

    class _UiBridgeService:
        def __init__(self):
            self.connections = []
            self.widget_tree_requests = []

        def connection_from_fields(self, fields):
            self.connections.append(fields)
            return fields

        def widget_tree(self, request, connection):
            self.widget_tree_requests.append(request)
            action = UiWidgetActionSummary(
                path=(0,),
                path_id="root/0",
                child_index=0,
                class_name="QPushButton",
                object_name="compile_button",
                accessible_name="Compile",
                accessible_description="Compile selected plate",
                label="Compile",
                visible=True,
                enabled=True,
                geometry=UiWidgetRect(x=8, y=160, width=72, height=24),
                global_geometry=UiWidgetRect(x=18, y=180, width=72, height=24),
                action_kinds=("button",),
                clickable=True,
                checkable=False,
                checked=False,
                current_index=None,
                current_text=None,
                item_count=None,
                tool_tip="Compile selected plate",
            )
            return UiWidgetTreeResult(
                schema_version=SCHEMA_VERSION,
                window_id=request.window_id,
                projected=True,
                actionable_widgets=(action,),
                root=UiWidgetTreeNode(
                    path=(),
                    path_id="root",
                    child_index=None,
                    class_name="QWidget",
                    object_name="main",
                    visible=True,
                    enabled=True,
                    geometry=UiWidgetRect(x=0, y=0, width=320, height=200),
                    global_geometry=UiWidgetRect(x=10, y=20, width=320, height=200),
                    tool_tip="",
                    status_tip="",
                    whats_this="",
                    window_title="Main window",
                    accessible_name="",
                    accessible_description="",
                    text=None,
                    text_truncated=False,
                    title=None,
                    action_kinds=(),
                    clickable=False,
                    actionable=False,
                    checkable=None,
                    checked=None,
                    current_index=None,
                    current_text=None,
                    item_count=None,
                    children=(),
                ),
                summary=UiWindowSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiWindowIdentity(window_id=request.window_id),
                    title="Main window",
                    window_kind="embedded",
                    visible=True,
                    focusable=True,
                ),
                widget_count=1,
                actionable_count=1,
                returned_actionable_count=1,
                include_tree=request.include_tree,
            )

    ui_bridge_service = _UiBridgeService()
    built = server.build_server(SimpleNamespace(ui_bridge_service=ui_bridge_service))

    async def call_widget_tree(arguments):
        return await asyncio.wait_for(
            built.call_tool("openhcs_ui_get_widget_tree", arguments),
            timeout=2,
        )

    compact_result = asyncio.run(
        call_widget_tree(
            {
                "window_id": "plate_manager",
                "maximum_item_model_nodes": 256,
                "include_tree": True,
                "connection": {"timeout_ms": 1234},
            }
        )
    )
    full_result = asyncio.run(
        call_widget_tree(
            {
                "window_id": "plate_manager",
                "compact_actions": False,
                "connection": {"timeout_ms": 1234},
            }
        )
    )
    compact_payload = json.loads(_direct_tool_text(compact_result))
    full_payload = json.loads(_direct_tool_text(full_result))

    assert compact_payload["projected"] is True
    compact_action = compact_payload["actionable_widgets"][0]
    assert compact_action["path_id"] == "root/0"
    assert "raw_value" not in compact_action
    assert full_payload["actionable_widgets"][0]["object_name"] == "compile_button"
    assert full_payload["actionable_widgets"][0]["raw_value"] is None
    assert ui_bridge_service.connections[0].timeout_ms == 1234
    first_request = ui_bridge_service.widget_tree_requests[0]
    assert first_request.window_id == "plate_manager"
    assert first_request.maximum_item_model_nodes == 256
    assert first_request.include_tree is True
    assert first_request.open_policy.create_if_missing is False


def test_ui_bridge_tools_expose_structured_connection_schema():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools

    schemas = {tool.name: tool.inputSchema for tool in tools}
    status_schema = schemas["openhcs_ui_bridge_status"]
    focus_schema = schemas["openhcs_ui_focus_window"]
    connection_schema = status_schema["$defs"]["McpUiBridgeConnectionRequest"]

    assert status_schema["properties"]["connection"]["anyOf"][0] == {
        "$ref": "#/$defs/McpUiBridgeConnectionRequest"
    }
    assert focus_schema["properties"]["connection"]["anyOf"][0] == {
        "$ref": "#/$defs/McpUiBridgeConnectionRequest"
    }
    assert {
        "host",
        "port",
        "transport_mode",
        "persistent",
        "timeout_ms",
        "auth_token",
        "descriptor_file_path",
        "bridge_instance_id",
    } <= set(connection_schema["properties"])
    assert "connection_fields" not in connection_schema["properties"]


def test_ui_bridge_connection_tool_args_accepts_mcp_connection_request():
    request = server.McpUiBridgeConnectionRequest(
        port=7888,
        transport_mode="ipc",
        timeout_ms=1200,
        descriptor_file_path="/tmp/bridge.json",
        bridge_instance_id="ui-1",
    )

    args = server.UiBridgeConnectionToolArgs.from_mapping(request)

    assert args._request.port == 7888
    assert args._request.transport_mode == "ipc"
    assert args._request.timeout_ms == 1200
    assert args._request.descriptor_file_path == "/tmp/bridge.json"
    assert args._request.bridge_instance_id == "ui-1"
    assert "port" in args._request.connection_fields
    assert "host" not in args._request.connection_fields


def test_selected_plate_workflow_tool_schema_exposes_workflow_enum():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools

    schemas = {tool.name: tool.inputSchema for tool in tools}
    schema = schemas["openhcs_ui_selected_plate_workflow"]

    assert schema["$defs"]["UiSelectedPlateWorkflowKind"]["enum"] == [
        "init_plate",
        "compile_plate",
        "run_plate",
    ]
    assert schema["properties"]["workflow"] == {
        "$ref": "#/$defs/UiSelectedPlateWorkflowKind"
    }
    assert schema["properties"]["require_confirmation"]["default"] is False


def test_selected_plate_workflow_rejects_unknown_workflow_before_dispatch():
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp.server.fastmcp.exceptions import ToolError

    async def call_unknown_workflow():
        built = server.build_server()
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_selected_plate_workflow",
                {"workflow": "not_a_workflow"},
            ),
            timeout=2,
        )

    with pytest.raises(ToolError, match="not_a_workflow"):
        asyncio.run(call_unknown_workflow())


def test_mcp_health_check_tool_returns_promptly():
    if importlib.util.find_spec("mcp") is None:
        return

    async def call_health_check():
        built = server.build_server()
        return await asyncio.wait_for(
            built.call_tool("openhcs_health_check", {}),
            timeout=2,
        )

    result = asyncio.run(call_health_check())
    payload = json.loads(_direct_tool_text(result))

    assert payload["status"] == "ok"
    assert payload["service"] == "openhcs.mcp"
    assert payload["openhcs_version"] == openhcs.__version__
    assert payload["packaged_resources_ready"] is True
    assert payload["packaged_resource_count"] == len(
        server.MCP_SERVER_PACKAGED_RESOURCE_PATHS
    )
    assert payload["missing_packaged_resource_paths"] == []
    assert isinstance(payload["server_process_id"], int)
    assert isinstance(payload["started_at_unix"], float)
    assert payload["server_source_path"].endswith("openhcs/mcp/server.py")
    assert isinstance(payload["server_import_mtime_ns"], int)
    assert isinstance(payload["server_current_mtime_ns"], int)
    assert payload["server_source_changed_since_import"] is False
    assert payload["stale_source_paths"] == []
    assert payload["restart_required"] is False
    assert payload["restart_command"] == []
    assert payload["restart_hint"] is None


def test_mcp_health_check_reports_missing_packaged_resources(monkeypatch, tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    missing_resource = tmp_path / "missing-knowledge-resource.json"
    monkeypatch.setattr(
        server,
        "MCP_SERVER_PACKAGED_RESOURCE_PATHS",
        (missing_resource,),
    )

    async def call_health_check():
        built = server.build_server()
        return await asyncio.wait_for(
            built.call_tool("openhcs_health_check", {}),
            timeout=2,
        )

    result = asyncio.run(call_health_check())
    payload = json.loads(_direct_tool_text(result))

    assert payload["packaged_resources_ready"] is False
    assert payload["packaged_resource_count"] == 1
    assert payload["missing_packaged_resource_paths"] == [str(missing_resource)]


def test_mcp_stale_watchlist_includes_agent_contract_sources():
    watched_paths = {
        source_path.as_posix() for source_path in server.MCP_SERVER_SOURCE_PATHS
    }

    assert any(path.endswith("openhcs/mcp/server.py") for path in watched_paths)
    assert any(path.endswith("openhcs/mcp/context.py") for path in watched_paths)
    assert any(path.endswith("openhcs/agent/capabilities.py") for path in watched_paths)
    assert any(path.endswith("openhcs/agent/path_policy.py") for path in watched_paths)
    assert any(path.endswith("openhcs/agent/dto/mcp.py") for path in watched_paths)
    assert any(
        path.endswith("openhcs/agent/dto/knowledge.py") for path in watched_paths
    )
    assert any(path.endswith("openhcs/agent/dto/plate.py") for path in watched_paths)
    assert any(
        path.endswith("openhcs/agent/services/knowledge_base_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/plate_inspection_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/synthetic_plate_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/core/plate_image_inventory.py") for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/stdio.py") for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/processing/custom_functions/manager.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("docs/source/development/mcp_knowledge_base_manifest.json")
        for path in watched_paths
    )
    assert any(
        path.endswith("docs/source/concepts/core_model.rst") for path in watched_paths
    )
    assert any(
        path.endswith("docs/source/guide_for_biologists/domain_expert_onboarding.rst")
        for path in watched_paths
    )
    assert any(
        path.endswith("docs/source/guides/example_corpus_map.rst")
        for path in watched_paths
    )
    assert any(
        path.endswith("docs/source/guides/complete_examples.rst")
        for path in watched_paths
    )
    assert any(
        path.endswith("docs/source/architecture/system_overview.rst")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/dto/ui_bridge.py") for path in watched_paths
    )
    assert any(path.endswith("openhcs/agent/dto/viewer.py") for path in watched_paths)
    assert any(
        path.endswith("openhcs/agent/services/execution_session_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/ui_bridge_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/ui_bridge_transport.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/viewer_window_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/runtime_server_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/runtime/viewer_protocol.py") for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/runtime/window_snapshot.py") for path in watched_paths
    )


def test_mcp_tools_fail_fast_when_server_source_is_stale(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    monkeypatch.setattr(
        server,
        "_mcp_server_stale_source_paths",
        lambda: (server.MCP_SERVER_SOURCE_PATH,),
    )

    async def call_stale_server():
        built = server.build_server()
        blocked_tool = await asyncio.wait_for(
            built.call_tool("openhcs_list_capabilities", {}),
            timeout=2,
        )
        health_tool = await asyncio.wait_for(
            built.call_tool("openhcs_health_check", {}),
            timeout=2,
        )
        return blocked_tool, health_tool

    blocked_result, health_result = asyncio.run(call_stale_server())
    blocked_payload = json.loads(_direct_tool_text(blocked_result))
    health_payload = json.loads(_direct_tool_text(health_result))

    assert blocked_payload["schema_version"] == "openhcs.agent.v1"
    assert blocked_payload["ok"] is False
    assert blocked_payload["tool"] == "openhcs_list_capabilities"
    assert blocked_payload["errors"][0]["code"] == "mcp_server_stale"
    assert blocked_payload["errors"][0]["path"].endswith("openhcs/mcp/server.py")
    assert (
        "Restart the MCP client/server process" in blocked_payload["errors"][0]["hint"]
    )
    assert blocked_payload["server_process_id"] == server.MCP_SERVER_PROCESS_ID
    assert (
        blocked_payload["server_started_at_unix"] == server.MCP_SERVER_IMPORTED_AT_UNIX
    )
    assert blocked_payload["stale_source_paths"][0].endswith("openhcs/mcp/server.py")
    assert blocked_payload["restart_required"] is True
    assert blocked_payload["restart_command"][0] == sys.executable
    assert blocked_payload["restart_command"][-2:] == ["-m", "openhcs.mcp"]
    assert "restart_command" in blocked_payload["restart_hint"]

    assert health_payload["status"] == "ok"
    assert health_payload["server_source_changed_since_import"] is True
    assert health_payload["stale_source_paths"][0].endswith("openhcs/mcp/server.py")
    assert health_payload["restart_required"] is True
    assert health_payload["restart_command"][0] == sys.executable
    assert health_payload["restart_command"][-2:] == ["-m", "openhcs.mcp"]
    assert "restart_command" in health_payload["restart_hint"]


def test_mcp_tools_fail_fast_when_agent_source_is_stale(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    stale_agent_path = next(
        source_path
        for source_path in server.MCP_SERVER_SOURCE_PATHS
        if source_path.name == "capabilities.py"
    )
    monkeypatch.setattr(
        server,
        "_mcp_server_stale_source_paths",
        lambda: (stale_agent_path,),
    )

    async def call_stale_server():
        built = server.build_server()
        blocked_tool = await asyncio.wait_for(
            built.call_tool("openhcs_search_functions", {}),
            timeout=2,
        )
        health_tool = await asyncio.wait_for(
            built.call_tool("openhcs_health_check", {}),
            timeout=2,
        )
        return blocked_tool, health_tool

    blocked_result, health_result = asyncio.run(call_stale_server())
    blocked_payload = json.loads(_direct_tool_text(blocked_result))
    health_payload = json.loads(_direct_tool_text(health_result))

    assert blocked_payload["schema_version"] == "openhcs.agent.v1"
    assert blocked_payload["ok"] is False
    assert blocked_payload["tool"] == "openhcs_search_functions"
    assert blocked_payload["errors"][0]["code"] == "mcp_server_stale"
    assert blocked_payload["errors"][0]["path"].endswith(
        "openhcs/agent/capabilities.py"
    )
    assert blocked_payload["stale_source_paths"][0].endswith(
        "openhcs/agent/capabilities.py"
    )
    assert blocked_payload["restart_required"] is True
    assert blocked_payload["restart_command"][-2:] == ["-m", "openhcs.mcp"]
    assert health_payload["server_source_changed_since_import"] is True
    assert health_payload["stale_source_paths"][0].endswith(
        "openhcs/agent/capabilities.py"
    )
    assert health_payload["restart_required"] is True
    assert health_payload["restart_command"][-2:] == ["-m", "openhcs.mcp"]


def test_mcp_tool_error_classifies_pipeline_authoring_mistakes():
    from openhcs.agent.path_policy import AgentPathPolicyError
    from openhcs.agent.services.execution_session_service import (
        UnknownExecutionJobIdError,
        UnknownExecutionSessionIdError,
    )
    from openhcs.agent.services.function_catalog_service import UnknownFunctionIdError
    from openhcs.agent.services.pipeline_authoring_service import (
        DuplicatePipelineStepIdError,
        InvalidFunctionKwargsError,
        MissingFunctionKwargsError,
        UnknownPipelineIdError,
    )

    unknown_function = server._mcp_tool_error(
        "openhcs_add_function_step",
        UnknownFunctionIdError("openhcs:nope"),
    )
    unknown_pipeline = server._mcp_tool_error(
        "openhcs_validate_pipeline",
        UnknownPipelineIdError("pipeline-404"),
    )
    duplicate_step = server._mcp_tool_error(
        "openhcs_add_function_step",
        DuplicatePipelineStepIdError("step-1"),
    )
    invalid_kwargs = server._mcp_tool_error(
        "openhcs_validate_pipeline",
        InvalidFunctionKwargsError(
            "test:sample_processing_function",
            invalid_kwargs=("not_a_parameter",),
            accepted_kwargs=("sigma",),
        ),
    )
    missing_kwargs = server._mcp_tool_error(
        "openhcs_validate_pipeline",
        MissingFunctionKwargsError(
            "test:sample_required_parameter_function",
            missing_kwargs=("threshold",),
        ),
    )
    unknown_session = server._mcp_tool_error(
        "openhcs_submit_compile",
        UnknownExecutionSessionIdError("session-404"),
    )
    unknown_job = server._mcp_tool_error(
        "openhcs_get_execution_status",
        UnknownExecutionJobIdError("job-404"),
    )
    path_error = server._mcp_tool_error(
        "openhcs_inspect_pipeline_source_artifact_plan",
        AgentPathPolicyError("Readable path does not exist: /tmp/nope"),
    )
    generic = server._mcp_tool_error(
        "openhcs_probe_viewer_window",
        ValueError("bad timeout"),
    )

    assert unknown_function["errors"][0]["code"] == "unknown_function_id"
    assert unknown_function["errors"][0]["message"] == (
        "Unknown OpenHCS function_id: openhcs:nope"
    )
    assert "openhcs_search_functions" in unknown_function["errors"][0]["hint"]
    assert unknown_pipeline["errors"][0]["code"] == "unknown_pipeline_id"
    assert unknown_pipeline["errors"][0]["message"] == (
        "Unknown OpenHCS pipeline_id: pipeline-404"
    )
    assert "openhcs_create_pipeline" in unknown_pipeline["errors"][0]["hint"]
    assert duplicate_step["errors"][0]["code"] == "duplicate_pipeline_step_id"
    assert "unique" in duplicate_step["errors"][0]["hint"]
    assert invalid_kwargs["errors"][0]["code"] == "invalid_function_kwargs"
    assert "not_a_parameter" in invalid_kwargs["errors"][0]["message"]
    assert "openhcs_describe_function" in invalid_kwargs["errors"][0]["hint"]
    assert missing_kwargs["errors"][0]["code"] == "missing_function_kwargs"
    assert "threshold" in missing_kwargs["errors"][0]["message"]
    assert "openhcs_describe_function" in missing_kwargs["errors"][0]["hint"]
    assert unknown_session["errors"][0]["code"] == "unknown_execution_session_id"
    assert "session_id" in unknown_session["errors"][0]["hint"]
    assert unknown_job["errors"][0]["code"] == "unknown_execution_job_id"
    assert "job_id" in unknown_job["errors"][0]["hint"]
    assert path_error["errors"][0]["code"] == "agent_path_policy_rejected"
    assert "OPENHCS_AGENT_READ_ROOTS" in path_error["errors"][0]["hint"]
    assert generic["errors"][0]["code"] == "mcp_tool_failed"


def test_mcp_register_custom_function_delegates_to_function_catalog_service():
    if importlib.util.find_spec("mcp") is None:
        return

    class _FunctionCatalog:
        def __init__(self):
            self.requests = []

        def register_custom_function(self, request):
            self.requests.append(request)
            return CustomFunctionRegistrationResult(
                schema_version=SCHEMA_VERSION,
                registered_count=1,
                persisted=request.persist,
                storage_dir="/tmp/custom_functions",
                source_file_paths=(),
                functions=(
                    FunctionCatalogEntry(
                        function_id="openhcs:agent_registered_custom",
                        import_path=(
                            "openhcs.processing.custom_functions."
                            "agent_registered_custom"
                        ),
                        name="agent_registered_custom",
                        module="openhcs.processing.custom_functions",
                        library="openhcs",
                        signature="agent_registered_custom(gain=1.0)",
                        summary="Agent custom function.",
                        backend_tags=("openhcs", "custom"),
                    ),
                ),
                next_steps=("Call openhcs_describe_function(...)",),
            )

    function_catalog = _FunctionCatalog()
    context = SimpleNamespace(function_catalog=function_catalog)

    async def call_register_tool():
        built = server.build_server(context)
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_register_custom_function",
                {
                    "source_code": "source",
                    "persist": False,
                    "compact_signature": False,
                },
            ),
            timeout=2,
        )
        return result

    result = asyncio.run(call_register_tool())
    payload = json.loads(_direct_tool_text(result))

    assert len(function_catalog.requests) == 1
    request = function_catalog.requests[0]
    assert request.source_code == "source"
    assert request.persist is False
    assert request.compact_signature is False
    assert payload["registered_count"] == 1
    assert payload["persisted"] is False
    assert payload["source_file_paths"] == []
    assert payload["functions"][0]["function_id"] == "openhcs:agent_registered_custom"
    assert "openhcs_describe_function" in payload["next_steps"][0]


def test_mcp_stale_watchlist_reports_deleted_watched_docs(monkeypatch, tmp_path):
    watched_doc = tmp_path / "watched.md"
    watched_doc.write_text("# watched\n", encoding="utf-8")
    import_snapshot = server.McpSourceSnapshot.from_path(watched_doc)
    watched_doc.unlink()
    monkeypatch.setattr(
        server,
        "MCP_SERVER_IMPORT_SOURCE_SNAPSHOTS",
        {watched_doc: import_snapshot},
    )

    stale_paths = server._mcp_server_stale_source_paths()

    assert stale_paths == (watched_doc,)


def test_mcp_knowledge_tools_are_registered_and_callable():
    if importlib.util.find_spec("mcp") is None:
        return

    async def call_knowledge_tools():
        built = server.build_server()
        listed_tools = built.list_tools()
        if inspect.isawaitable(listed_tools):
            tools = await listed_tools
        else:
            tools = listed_tools
        catalog = await asyncio.wait_for(
            built.call_tool("openhcs_list_knowledge_documents", {}),
            timeout=2,
        )
        document = await asyncio.wait_for(
            built.call_tool(
                "openhcs_get_knowledge_document",
                {
                    "document_id": "openhcs_architecture_quick_start",
                    "max_chars": 300,
                },
            ),
            timeout=2,
        )
        search = await asyncio.wait_for(
            built.call_tool(
                "openhcs_search_knowledge",
                {"query": "MCP", "limit": 2},
            ),
            timeout=2,
        )
        return tools, catalog, document, search

    tools, catalog, document, search = asyncio.run(call_knowledge_tools())
    tool_names = {tool.name for tool in tools}
    catalog_payload = json.loads(_direct_tool_text(catalog))
    document_payload = json.loads(_direct_tool_text(document))
    search_payload = json.loads(_direct_tool_text(search))

    assert "openhcs_list_knowledge_documents" in tool_names
    assert "openhcs_get_knowledge_document" in tool_names
    assert "openhcs_search_knowledge" in tool_names
    assert catalog_payload["schema_version"] == "openhcs.agent.v1"
    assert any(
        item["document_id"] == "openhcs_architecture_quick_start"
        for item in catalog_payload["documents"]
    )
    assert (
        document_payload["document"]["document_id"]
        == "openhcs_architecture_quick_start"
    )
    assert document_payload["truncated"] is True
    assert search_payload["hits"]


def test_mcp_plate_inspection_tool_is_registered_and_callable(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    missing_plate = tmp_path / "missing-plate"

    async def call_plate_tool():
        built = server.build_server()
        listed_tools = built.list_tools()
        if inspect.isawaitable(listed_tools):
            tools = await listed_tools
        else:
            tools = listed_tools
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_inspect_plate_path",
                {"plate_path": str(missing_plate)},
            ),
            timeout=2,
        )
        return tools, result

    tools, result = asyncio.run(call_plate_tool())
    tool_names = {tool.name for tool in tools}
    payload = json.loads(_direct_tool_text(result))

    assert "openhcs_inspect_plate_path" in tool_names
    assert payload["schema_version"] == "openhcs.agent.v1"
    assert payload["status"] == "error"
    assert payload["errors"][0]["code"] == "plate_path_policy_rejected"


def test_mcp_synthetic_plate_generation_tool_projects_request(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    class _SyntheticPlateService:
        request = None

        def generate(self, request):
            self.request = request
            return SyntheticPlateGenerationResult(
                schema_version=SCHEMA_VERSION,
                output_dir=request.output_dir,
                requested_format=request.format,
                grid_size=(request.grid_rows, request.grid_cols),
                tile_size=(request.tile_width, request.tile_height),
                overlap_percent=request.overlap_percent,
                stage_error_px=request.stage_error_px,
                wells=request.wells,
                wavelengths=request.wavelengths,
                z_stack_levels=request.z_stack_levels,
                num_cells=request.num_cells,
                shared_cell_fraction=request.shared_cell_fraction,
                image_count=8,
                sampled_image_files=("TimePoint_1/A01_s1_w1_z1_t1.tif",),
                metadata_file_path=f"{request.output_dir}/plate.HTD",
                detected_microscope_type="imagexpress",
                handler_class="ImageXpressHandler",
            )

    synthetic_plate_service = _SyntheticPlateService()
    output_dir = tmp_path / "synthetic"
    context = SimpleNamespace(synthetic_plate_service=synthetic_plate_service)

    async def call_synthetic_plate_tool():
        built = server.build_server(context)
        listed_tools = built.list_tools()
        if inspect.isawaitable(listed_tools):
            tools = await listed_tools
        else:
            tools = listed_tools
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_generate_synthetic_plate",
                {
                    "output_dir": str(output_dir),
                    "grid_rows": 1,
                    "grid_cols": 2,
                    "tile_width": 64,
                    "tile_height": 48,
                    "overlap_percent": 10,
                    "wavelengths": 2,
                    "wells": ["A01"],
                    "random_seed": 11,
                },
            ),
            timeout=2,
        )
        return tools, result

    tools, result = asyncio.run(call_synthetic_plate_tool())
    tool_names = {tool.name for tool in tools}
    payload = json.loads(_direct_tool_text(result))

    assert "openhcs_generate_synthetic_plate" in tool_names
    assert synthetic_plate_service.request.output_dir == str(output_dir)
    assert synthetic_plate_service.request.grid_rows == 1
    assert synthetic_plate_service.request.grid_cols == 2
    assert synthetic_plate_service.request.tile_width == 64
    assert synthetic_plate_service.request.tile_height == 48
    assert synthetic_plate_service.request.overlap_percent == 10
    assert synthetic_plate_service.request.wavelengths == 2
    assert synthetic_plate_service.request.wells == ("A01",)
    assert synthetic_plate_service.request.random_seed == 11
    assert payload["schema_version"] == "openhcs.agent.v1"
    assert payload["output_dir"] == str(output_dir)
    assert payload["image_count"] == 8


def test_mcp_plate_file_query_tool_is_registered_and_callable(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    missing_plate = tmp_path / "missing-plate"

    async def call_plate_query_tool():
        built = server.build_server()
        listed_tools = built.list_tools()
        if inspect.isawaitable(listed_tools):
            tools = await listed_tools
        else:
            tools = listed_tools
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_query_plate_files",
                {
                    "plate_path": str(missing_plate),
                    "kind": "image",
                    "limit": 5,
                },
            ),
            timeout=2,
        )
        return tools, result

    tools, result = asyncio.run(call_plate_query_tool())
    tool_names = {tool.name for tool in tools}
    payload = json.loads(_direct_tool_text(result))

    assert "openhcs_query_plate_files" in tool_names
    assert payload["schema_version"] == "openhcs.agent.v1"
    assert payload["errors"][0]["code"] == "plate_path_policy_rejected"


def test_mcp_object_state_field_search_supports_exact_paths_and_leaf_default():
    if importlib.util.find_spec("mcp") is None:
        return

    def field_summary(field_path: str) -> UiObjectStateFieldSummary:
        return UiObjectStateFieldSummary(
            schema_version=SCHEMA_VERSION,
            address=UiSemanticAddress(
                object_state_scope_id="global_config",
                field_path=field_path,
            ),
            field_name=field_path.rsplit(".", 1)[-1],
            container_path=(field_path.rsplit(".", 1)[0] if "." in field_path else ""),
            object_state_path_type="openhcs.core.config.NapariStreamingConfig",
            raw_value_type="None",
            resolved_value_type="str",
            dirty=False,
            signature_diff=False,
            last_changed=False,
            raw_value_preview=UiObjectStateValuePreview(
                type_name="None",
                is_none=True,
                text="None",
            ),
            resolved_value_preview=UiObjectStateValuePreview(
                type_name="str",
                is_none=False,
                text="'resolved'",
            ),
            raw_value_is_none=True,
            resolved_value_is_none=False,
            inherited_value=True,
        )

    class _UiBridgeService:
        def __init__(self):
            self.requests = []

        def connection_from_fields(self, fields):
            return "ui-connection"

        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            self.requests.append(request)
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=12,
                current_branch="main",
                current_snapshot_index=-1,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="global_config",
                        ),
                        object_type="GlobalPipelineConfig",
                        parameter_count=5,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(
                            field_summary("napari_streaming_config"),
                            field_summary("napari_streaming_config.enabled"),
                            field_summary("napari_streaming_config.port"),
                            field_summary("well_filter_config.well_filter"),
                            field_summary("well_filter_config.well_filter_mode"),
                        ),
                    ),
                ),
            )

        def get_object_state_fields(self, query, connection):
            catalog = self.list_object_state_scopes(
                query.scope_list_request(),
                connection,
            )
            return ObjectStateFieldListProjector.project_catalog(query, catalog)

    ui_bridge_service = _UiBridgeService()
    context = SimpleNamespace(ui_bridge_service=ui_bridge_service)

    async def call_object_state_tools():
        built = server.build_server(context)
        exact_result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_get_object_state_fields",
                {
                    "scope_ids": ["global_config"],
                    "field_paths": ["well_filter_config.well_filter"],
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )
        broad_result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_get_object_state_fields",
                {
                    "scope_ids": ["global_config"],
                    "field_path_contains": ["napari_streaming_config"],
                },
            ),
            timeout=2,
        )
        container_result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_get_object_state_fields",
                {
                    "scope_ids": ["global_config"],
                    "field_path_contains": ["napari_streaming_config"],
                    "include_container_fields": True,
                },
            ),
            timeout=2,
        )
        return exact_result, broad_result, container_result

    exact_result, broad_result, container_result = asyncio.run(
        call_object_state_tools()
    )
    exact_payload = json.loads(_direct_tool_text(exact_result))
    broad_payload = json.loads(_direct_tool_text(broad_result))
    container_payload = json.loads(_direct_tool_text(container_result))

    exact_fields = exact_payload["scopes"][0]["fields"]
    assert [field["field_path"] for field in exact_fields] == [
        "well_filter_config.well_filter",
    ]
    assert exact_payload["field_paths"] == ["well_filter_config.well_filter"]
    assert exact_payload["include_container_fields"] is False
    assert ui_bridge_service.requests[0].field_paths == (
        "well_filter_config.well_filter",
    )
    assert ui_bridge_service.requests[1].field_paths == ()
    assert ui_bridge_service.requests[2].field_paths == ()

    broad_field_paths = [
        field["field_path"] for field in broad_payload["scopes"][0]["fields"]
    ]
    assert broad_field_paths == [
        "napari_streaming_config.enabled",
        "napari_streaming_config.port",
    ]

    container_field_paths = [
        field["field_path"] for field in container_payload["scopes"][0]["fields"]
    ]
    assert container_field_paths == [
        "napari_streaming_config",
        "napari_streaming_config.enabled",
        "napari_streaming_config.port",
    ]


def test_mcp_list_object_state_scopes_filters_scope_ids(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    monkeypatch.setattr(server, "_mcp_server_stale_source_paths", tuple)

    class _UiBridgeService:
        def connection_from_fields(self, fields):
            assert fields.timeout_ms == 1234
            return "ui-connection"

        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            assert request.scope_ids == ("/tmp/plate",)
            assert request.include_fields is True
            assert request.field_filter is UiObjectStateFieldFilter.SEMANTIC
            return request.filtered_catalog(
                UiObjectStateScopeCatalog(
                    schema_version=SCHEMA_VERSION,
                    object_state_token=12,
                    current_branch="main",
                    current_snapshot_index=-1,
                    scopes=(
                        UiObjectStateScopeSummary(
                            schema_version=SCHEMA_VERSION,
                            identity=UiObjectStateScopeIdentity(
                                object_state_scope_id="global_config",
                            ),
                            object_type="GlobalPipelineConfig",
                            parameter_count=1,
                            dirty_field_count=0,
                            signature_diff_field_count=0,
                        ),
                        UiObjectStateScopeSummary(
                            schema_version=SCHEMA_VERSION,
                            identity=UiObjectStateScopeIdentity(
                                object_state_scope_id="/tmp/plate",
                            ),
                            object_type="PipelineOrchestrator",
                            parameter_count=1,
                            dirty_field_count=0,
                            signature_diff_field_count=0,
                        ),
                    ),
                )
            )

    context = SimpleNamespace(ui_bridge_service=_UiBridgeService())

    async def call_scope_tool():
        built = server.build_server(context)
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_list_object_state_scopes",
                {
                    "scope_ids": ["/tmp/plate"],
                    "include_fields": True,
                    "field_filter": "semantic",
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_scope_tool())
    payload = json.loads(_direct_tool_text(result))

    assert [
        scope["identity"]["object_state_scope_id"] for scope in payload["scopes"]
    ] == ["/tmp/plate"]


def test_mcp_describe_object_state_field_tool_projects_request(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    monkeypatch.setattr(server, "_mcp_server_stale_source_paths", tuple)

    class _UiBridgeService:
        def __init__(self):
            self.requests = []

        def connection_from_fields(self, fields):
            assert fields.timeout_ms == 1234
            return "ui-connection"

        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            self.requests.append(request)
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=12,
                current_branch="main",
                current_snapshot_index=-1,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="global_config",
                        ),
                        object_type="GlobalPipelineConfig",
                        parameter_count=1,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(
                            UiObjectStateFieldSummary(
                                schema_version=SCHEMA_VERSION,
                                address=UiSemanticAddress(
                                    object_state_scope_id="global_config",
                                    field_path="napari_display_config.colormap",
                                    window_id="global_config",
                                ),
                                field_name="colormap",
                                container_path="napari_display_config",
                                object_state_path_type=(
                                    "openhcs.core.config.NapariDisplayConfig"
                                ),
                                raw_value_type="NapariColormap",
                                resolved_value_type="NapariColormap",
                                dirty=False,
                                signature_diff=False,
                                last_changed=False,
                                raw_value_preview=UiObjectStateValuePreview(
                                    type_name="NapariColormap",
                                    is_none=False,
                                    text="NapariColormap.GRAY",
                                ),
                                resolved_value_preview=UiObjectStateValuePreview(
                                    type_name="NapariColormap",
                                    is_none=False,
                                    text="NapariColormap.GRAY",
                                ),
                            ),
                        ),
                    ),
                ),
            )

    ui_bridge_service = _UiBridgeService()
    context = SimpleNamespace(
        ui_bridge_service=ui_bridge_service,
        object_state_field_help_service=ObjectStateFieldHelpService(ui_bridge_service),
    )

    async def call_describe_tool():
        built = server.build_server(context)
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_describe_object_state_field",
                {
                    "object_state_scope_id": "global_config",
                    "field_path": "napari_display_config.colormap",
                    "window_id": "global_config",
                    "max_description_chars": 500,
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_describe_tool())
    payload = json.loads(_direct_tool_text(result))

    assert len(ui_bridge_service.requests) == 1
    request = ui_bridge_service.requests[0]
    assert request.include_fields is True
    assert request.include_field_descriptions is True
    assert request.field_limit == 1
    assert request.field_paths == ("napari_display_config.colormap",)
    assert payload["help_target_type"] == "openhcs.core.config.NapariDisplayConfig"
    assert payload["summary"] == "• colormap (NapariColormap)"


def test_mcp_describe_object_state_field_tool_infers_unique_scope(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    monkeypatch.setattr(server, "_mcp_server_stale_source_paths", tuple)
    target_name = (
        f"{mcp_help_threshold_function.__module__}."
        f"{mcp_help_threshold_function.__qualname__}"
    )

    class _UiBridgeService:
        def __init__(self):
            self.requests = []

        def connection_from_fields(self, fields):
            assert fields.timeout_ms == 1234
            return "ui-connection"

        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            self.requests.append(request)
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=12,
                current_branch="main",
                current_snapshot_index=-1,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="plate::step::function_0",
                        ),
                        object_type=target_name,
                        parameter_count=1,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(
                            UiObjectStateFieldSummary(
                                schema_version=SCHEMA_VERSION,
                                address=UiSemanticAddress(
                                    object_state_scope_id=("plate::step::function_0"),
                                    field_path="threshold",
                                ),
                                field_name="threshold",
                                container_path="",
                                object_state_path_type=target_name,
                                raw_value_type="float",
                                resolved_value_type="float",
                                dirty=False,
                                signature_diff=False,
                                last_changed=False,
                                parameter_description=(
                                    "Numeric cutoff used for segmentation."
                                ),
                                raw_value_preview=UiObjectStateValuePreview(
                                    type_name="float",
                                    is_none=False,
                                    text="1.0",
                                ),
                                resolved_value_preview=UiObjectStateValuePreview(
                                    type_name="float",
                                    is_none=False,
                                    text="1.0",
                                ),
                            ),
                        ),
                    ),
                ),
            )

    ui_bridge_service = _UiBridgeService()
    context = SimpleNamespace(
        ui_bridge_service=ui_bridge_service,
        object_state_field_help_service=ObjectStateFieldHelpService(ui_bridge_service),
    )

    async def call_describe_tool():
        built = server.build_server(context)
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_describe_object_state_field",
                {
                    "field_path": "threshold",
                    "max_description_chars": 500,
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_describe_tool())
    payload = json.loads(_direct_tool_text(result))

    assert len(ui_bridge_service.requests) == 2
    assert ui_bridge_service.requests[0].include_field_descriptions is False
    assert ui_bridge_service.requests[1].include_field_descriptions is True
    assert payload["address"]["object_state_scope_id"] == "plate::step::function_0"
    assert payload["help_target_type"] == target_name
    assert payload["description"] == "Numeric cutoff used for segmentation."


def test_mcp_describe_object_state_field_tool_reports_ambiguous_scope(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    monkeypatch.setattr(server, "_mcp_server_stale_source_paths", tuple)
    target_name = (
        f"{mcp_help_threshold_function.__module__}."
        f"{mcp_help_threshold_function.__qualname__}"
    )

    def field_summary(scope_id: str) -> UiObjectStateFieldSummary:
        return UiObjectStateFieldSummary(
            schema_version=SCHEMA_VERSION,
            address=UiSemanticAddress(
                object_state_scope_id=scope_id,
                field_path="threshold",
            ),
            field_name="threshold",
            container_path="",
            object_state_path_type=target_name,
            raw_value_type="float",
            resolved_value_type="float",
            dirty=False,
            signature_diff=False,
            last_changed=False,
        )

    class _UiBridgeService:
        def __init__(self):
            self.requests = []

        def connection_from_fields(self, fields):
            assert fields.timeout_ms == 1234
            return "ui-connection"

        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            self.requests.append(request)
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=12,
                current_branch="main",
                current_snapshot_index=-1,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="plate::step::function_0",
                        ),
                        object_type=target_name,
                        parameter_count=1,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(field_summary("plate::step::function_0"),),
                    ),
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="plate::step::function_1",
                        ),
                        object_type=target_name,
                        parameter_count=1,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(field_summary("plate::step::function_1"),),
                    ),
                ),
            )

    ui_bridge_service = _UiBridgeService()
    context = SimpleNamespace(
        ui_bridge_service=ui_bridge_service,
        object_state_field_help_service=ObjectStateFieldHelpService(ui_bridge_service),
    )

    async def call_describe_tool():
        built = server.build_server(context)
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_describe_object_state_field",
                {
                    "field_path": "threshold",
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_describe_tool())
    payload = json.loads(_direct_tool_text(result))

    assert len(ui_bridge_service.requests) == 1
    assert payload["errors"][0]["code"] == "ambiguous_ui_object_state_field"
    assert "plate::step::function_0" in payload["errors"][0]["hint"]
    assert "plate::step::function_1" in payload["errors"][0]["hint"]


def test_object_state_field_help_service_describes_function_parameter_target():
    target_name = (
        f"{mcp_help_threshold_function.__module__}."
        f"{mcp_help_threshold_function.__qualname__}"
    )

    class _UiBridgeService:
        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            assert request.include_field_descriptions is True
            assert request.field_paths == ("threshold",)
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=12,
                current_branch="main",
                current_snapshot_index=-1,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="plate::step::function_0",
                        ),
                        object_type=target_name,
                        parameter_count=1,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(
                            UiObjectStateFieldSummary(
                                schema_version=SCHEMA_VERSION,
                                address=UiSemanticAddress(
                                    object_state_scope_id=("plate::step::function_0"),
                                    field_path="threshold",
                                ),
                                field_name="threshold",
                                container_path="",
                                object_state_path_type=target_name,
                                raw_value_type="float",
                                resolved_value_type="float",
                                dirty=False,
                                signature_diff=False,
                                last_changed=False,
                                parameter_description=(
                                    "Numeric cutoff used for segmentation."
                                ),
                                raw_value_preview=UiObjectStateValuePreview(
                                    type_name="float",
                                    is_none=False,
                                    text="1.0",
                                ),
                                resolved_value_preview=UiObjectStateValuePreview(
                                    type_name="float",
                                    is_none=False,
                                    text="1.0",
                                ),
                            ),
                        ),
                    ),
                ),
            )

    result = ObjectStateFieldHelpService(_UiBridgeService()).describe(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id="plate::step::function_0",
            field_path="threshold",
            max_description_chars=500,
        ),
        "ui-connection",
    )

    assert result.errors == ()
    assert result.help_target_type == target_name
    assert result.parameter_name == "threshold"
    assert result.summary == "• threshold (float)"
    assert result.description == "Numeric cutoff used for segmentation."
    assert result.target_summary == "Apply a threshold."


def test_mcp_mutate_object_state_field_tool_projects_request(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    monkeypatch.setattr(server, "_mcp_server_stale_source_paths", tuple)

    class _UiBridgeService:
        def __init__(self):
            self.requests = []

        def connection_from_fields(self, fields):
            assert fields.timeout_ms == 1234
            return "ui-connection"

        def mutate_object_state_field(self, request, connection):
            assert connection == "ui-connection"
            self.requests.append(request)
            return UiObjectStateFieldMutationResult(
                schema_version=SCHEMA_VERSION,
                address=UiSemanticAddress(
                    object_state_scope_id=request.object_state_scope_id,
                    field_path=request.field_path,
                    window_id=request.window_id,
                ),
                mutated=True,
                reset=request.reset,
                receipt=UiMutationReceipt.accepted_for(request.request_token),
            )

    ui_bridge_service = _UiBridgeService()
    context = SimpleNamespace(ui_bridge_service=ui_bridge_service)

    async def call_mutation_tool():
        built = server.build_server(context)
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_mutate_object_state_field",
                {
                    "object_state_scope_id": "global_config",
                    "field_path": "well_filter_config.well_filter",
                    "value": "A01",
                    "request_token": "req-1",
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_mutation_tool())
    payload = json.loads(_direct_tool_text(result))

    assert payload["mutated"] is True
    assert payload["receipt"]["accepted"] is True
    assert payload["receipt"]["request_token"]["value"] == "req-1"
    assert len(ui_bridge_service.requests) == 1
    request = ui_bridge_service.requests[0]
    assert request.object_state_scope_id == "global_config"
    assert request.field_path == "well_filter_config.well_filter"
    assert request.value == "A01"
    assert request.reset is False
    assert request.request_token.value == "req-1"


def test_mcp_object_state_field_search_filters_before_paging():
    if importlib.util.find_spec("mcp") is None:
        return

    def field_summary(field_path: str, *, signature_diff: bool = False):
        return UiObjectStateFieldSummary(
            schema_version=SCHEMA_VERSION,
            address=UiSemanticAddress(
                object_state_scope_id="global_config",
                field_path=field_path,
            ),
            field_name=field_path,
            container_path="",
            object_state_path_type="openhcs.core.config.GlobalPipelineConfig",
            raw_value_type="int",
            resolved_value_type="int",
            dirty=False,
            signature_diff=signature_diff,
            last_changed=False,
            raw_value_preview=UiObjectStateValuePreview(
                type_name="int",
                is_none=False,
                text="1",
            ),
            resolved_value_preview=UiObjectStateValuePreview(
                type_name="int",
                is_none=False,
                text="1",
            ),
            raw_value_is_none=False,
            resolved_value_is_none=False,
            inherited_value=False,
        )

    class _UiBridgeService:
        def __init__(self):
            self.requests = []

        def connection_from_fields(self, fields):
            return "ui-connection"

        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            self.requests.append(request)
            clean_fields = tuple(field_summary(f"clean_{index}") for index in range(5))
            semantic_fields = tuple(
                field_summary(f"changed_{index}", signature_diff=True)
                for index in range(5)
            )
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=12,
                current_branch="main",
                current_snapshot_index=-1,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="global_config",
                        ),
                        object_type="GlobalPipelineConfig",
                        parameter_count=10,
                        dirty_field_count=0,
                        signature_diff_field_count=5,
                        fields=(*clean_fields, *semantic_fields),
                        field_page=UiCatalogPageMetadata(
                            limit=request.field_limit,
                            offset=request.field_offset,
                            returned_count=10,
                            total_count=10,
                            truncated=False,
                            next_offset=None,
                        ),
                    ),
                ),
            )

        def get_object_state_fields(self, query, connection):
            catalog = self.list_object_state_scopes(
                query.scope_list_request(),
                connection,
            )
            return ObjectStateFieldListProjector.project_catalog(query, catalog)

    ui_bridge_service = _UiBridgeService()
    context = SimpleNamespace(ui_bridge_service=ui_bridge_service)

    async def call_object_state_tool():
        built = server.build_server(context)
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_get_object_state_fields",
                {
                    "scope_ids": ["global_config"],
                    "field_filter": "semantic",
                    "field_limit": 3,
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_object_state_tool())
    payload = json.loads(_direct_tool_text(result))

    request = ui_bridge_service.requests[0]
    assert request.field_offset == 0
    assert request.field_limit >= 10
    assert payload["matched_field_count"] == 5
    assert payload["returned_field_count"] == 3
    assert payload["field_offset"] == 0
    assert payload["field_limit"] == 3
    assert payload["next_offset"] == 3
    assert payload["truncated"] is True
    assert [field["field_path"] for field in payload["scopes"][0]["fields"]] == [
        "changed_0",
        "changed_1",
        "changed_2",
    ]


def test_mcp_selected_plate_image_inspection_composes_ui_state_and_plate_service():
    if importlib.util.find_spec("mcp") is None:
        return

    selected_plate_root = "/tmp/selected-plate"

    class _UiBridgeService:
        def __init__(self):
            self.state_surface_request = None
            self.connection_fields = None

        def connection_from_fields(self, fields):
            self.connection_fields = fields
            return "ui-connection"

        def get_state_surface(self, request, connection):
            self.state_surface_request = request
            assert connection == "ui-connection"
            return UiStateSurfaceDocument(
                schema_version=SCHEMA_VERSION,
                summary=UiStateSurfaceSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiStateSurfaceIdentity(surface_id="plate_manager.state"),
                    widget_id="plate_manager",
                    title="Plate manager state",
                    readable=True,
                    supported_selection_modes=("selected", "all"),
                    current_selection_count=1,
                    total_scope_count=1,
                ),
                payload_schema="openhcs.ui.plate_manager_state.v1",
                payload={
                    "rows": [
                        {
                            "plate_scope_id": selected_plate_root,
                            "name": "selected-plate",
                            "plate_root": selected_plate_root,
                            "selected": True,
                        }
                    ]
                },
                selected_scope_ids=(selected_plate_root,),
            )

    class _PlateInspectionService:
        def __init__(self):
            self.request = None

        def inspect(self, request):
            self.request = request
            return PlatePathInspectionResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_microscope_type=request.microscope_type,
                status=PlateInspectionStatus.OK,
                confidence=PlateInspectionConfidence.HIGH,
            )

    ui_bridge_service = _UiBridgeService()
    plate_inspection_service = _PlateInspectionService()
    context = _selected_plate_mcp_context(
        ui_bridge_service=ui_bridge_service,
        plate_inspection_service=plate_inspection_service,
    )

    async def call_selected_plate_tool():
        built = server.build_server(context)
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_inspect_selected_plate_images",
                {
                    "microscope_type": "openhcsdata",
                    "max_sample_files": 3,
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )
        return result

    result = asyncio.run(call_selected_plate_tool())
    payload = json.loads(_direct_tool_text(result))

    assert ui_bridge_service.state_surface_request.surface_id == "plate_manager.state"
    assert ui_bridge_service.state_surface_request.selection_mode == "selected"
    assert ui_bridge_service.connection_fields.timeout_ms == 1234
    assert plate_inspection_service.request.plate_path == selected_plate_root
    assert plate_inspection_service.request.microscope_type == "openhcsdata"
    assert plate_inspection_service.request.bounds.max_sample_files == 3
    assert payload["schema_version"] == "openhcs.agent.v1"
    assert payload["selected_plate"]["plate_root"] == selected_plate_root
    assert payload["target"] == "selected"
    assert payload["inspection"]["plate_path"] == selected_plate_root
    assert payload["inspection"]["status"] == "ok"


def test_mcp_selected_plate_image_inspection_targets_output_plate():
    if importlib.util.find_spec("mcp") is None:
        return

    selected_plate_root = "/tmp/selected-plate"
    output_plate_root = "/tmp/selected-plate_openhcs"

    class _UiBridgeService:
        def connection_from_fields(self, fields):
            return "ui-connection"

        def get_state_surface(self, request, connection):
            assert connection == "ui-connection"
            return UiStateSurfaceDocument(
                schema_version=SCHEMA_VERSION,
                summary=UiStateSurfaceSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiStateSurfaceIdentity(surface_id="plate_manager.state"),
                    widget_id="plate_manager",
                    title="Plate manager state",
                    readable=True,
                    current_selection_count=1,
                    total_scope_count=1,
                ),
                payload_schema="openhcs.ui.plate_manager_state.v1",
                payload={
                    "rows": [
                        {
                            "plate_scope_id": selected_plate_root,
                            "name": "selected-plate",
                            "plate_root": selected_plate_root,
                            "output_plate_root": output_plate_root,
                            "selected": True,
                        }
                    ]
                },
                selected_scope_ids=(selected_plate_root,),
            )

    class _PlateInspectionService:
        def __init__(self):
            self.request = None

        def inspect(self, request):
            self.request = request
            return PlatePathInspectionResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_microscope_type=request.microscope_type,
                status=PlateInspectionStatus.OK,
                confidence=PlateInspectionConfidence.HIGH,
            )

    plate_inspection_service = _PlateInspectionService()
    context = _selected_plate_mcp_context(
        ui_bridge_service=_UiBridgeService(),
        plate_inspection_service=plate_inspection_service,
    )

    async def call_selected_plate_images_tool():
        built = server.build_server(context)
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_inspect_selected_plate_images",
                {"target": "output"},
            ),
            timeout=2,
        )
        return result

    result = asyncio.run(call_selected_plate_images_tool())
    payload = json.loads(_direct_tool_text(result))

    assert plate_inspection_service.request.plate_path == output_plate_root
    assert plate_inspection_service.request.microscope_type == "auto"
    assert payload["target"] == "output"
    assert payload["inspection"]["plate_path"] == output_plate_root

    async def call_selected_plate_images_tool_with_explicit_type():
        built = server.build_server(context)
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_inspect_selected_plate_images",
                {"target": "output", "microscope_type": "imagexpress"},
            ),
            timeout=2,
        )
        return result

    asyncio.run(call_selected_plate_images_tool_with_explicit_type())
    assert plate_inspection_service.request.plate_path == output_plate_root
    assert plate_inspection_service.request.microscope_type == "imagexpress"


def test_mcp_selected_plate_file_query_composes_ui_state_and_plate_service():
    if importlib.util.find_spec("mcp") is None:
        return

    selected_plate_root = "/tmp/selected-plate"
    output_plate_root = "/tmp/selected-plate_openhcs"

    class _UiBridgeService:
        def __init__(self):
            self.state_surface_request = None

        def connection_from_fields(self, fields):
            return "ui-connection"

        def get_state_surface(self, request, connection):
            self.state_surface_request = request
            assert connection == "ui-connection"
            return UiStateSurfaceDocument(
                schema_version=SCHEMA_VERSION,
                summary=UiStateSurfaceSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiStateSurfaceIdentity(surface_id="plate_manager.state"),
                    widget_id="plate_manager",
                    title="Plate manager state",
                    readable=True,
                    supported_selection_modes=("selected", "all"),
                    current_selection_count=1,
                    total_scope_count=1,
                ),
                payload_schema="openhcs.ui.plate_manager_state.v1",
                payload={
                    "rows": [
                        {
                            "plate_scope_id": selected_plate_root,
                            "name": "selected-plate",
                            "plate_root": selected_plate_root,
                            "output_plate_root": output_plate_root,
                            "selected": True,
                        }
                    ]
                },
                selected_scope_ids=(selected_plate_root,),
            )

    class _PlateInspectionService:
        def __init__(self):
            self.request = None

        def query_files(self, request):
            self.request = request
            return PlateFileQueryResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_microscope_type=request.microscope_type,
                total_count=0,
            )

    ui_bridge_service = _UiBridgeService()
    plate_inspection_service = _PlateInspectionService()
    context = _selected_plate_mcp_context(
        ui_bridge_service=ui_bridge_service,
        plate_inspection_service=plate_inspection_service,
    )

    async def call_selected_plate_query_tool():
        built = server.build_server(context)
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_query_selected_plate_files",
                {
                    "kind": "result",
                    "target": "output",
                    "path_contains": "roi",
                    "limit": 7,
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )
        return result

    result = asyncio.run(call_selected_plate_query_tool())
    payload = json.loads(_direct_tool_text(result))

    assert ui_bridge_service.state_surface_request.surface_id == "plate_manager.state"
    assert ui_bridge_service.state_surface_request.selection_mode == "selected"
    assert plate_inspection_service.request.plate_path == output_plate_root
    assert plate_inspection_service.request.microscope_type == "auto"
    assert plate_inspection_service.request.kind.value == "result"
    assert plate_inspection_service.request.path_contains == "roi"
    assert plate_inspection_service.request.limit == 7
    assert payload["selected_plate"]["plate_root"] == selected_plate_root
    assert payload["target"] == "output"
    assert payload["query"]["plate_path"] == output_plate_root

    async def call_selected_plate_source_query_tool():
        built = server.build_server(context)
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_query_selected_plate_files",
                {
                    "kind": "image",
                    "target": "source",
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )
        return result

    result = asyncio.run(call_selected_plate_source_query_tool())
    payload = json.loads(_direct_tool_text(result))

    assert plate_inspection_service.request.plate_path == selected_plate_root
    assert plate_inspection_service.request.kind.value == "image"
    assert payload["target"] == "source"
    assert payload["query"]["plate_path"] == selected_plate_root


def test_mcp_selected_plate_result_stream_uses_output_context_for_output_target():
    if importlib.util.find_spec("mcp") is None:
        return

    selected_plate_root = "/tmp/selected-plate"
    output_plate_root = "/tmp/selected-plate_openhcs"

    class _UiBridgeService:
        def connection_from_fields(self, fields):
            return "ui-connection"

        def get_state_surface(self, request, connection):
            assert connection == "ui-connection"
            return UiStateSurfaceDocument(
                schema_version=SCHEMA_VERSION,
                summary=UiStateSurfaceSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiStateSurfaceIdentity(surface_id="plate_manager.state"),
                    widget_id="plate_manager",
                    title="Plate manager state",
                    readable=True,
                    current_selection_count=1,
                    total_scope_count=1,
                ),
                payload_schema="openhcs.ui.plate_manager_state.v1",
                payload={
                    "rows": [
                        {
                            "plate_scope_id": selected_plate_root,
                            "name": "selected-plate",
                            "plate_root": selected_plate_root,
                            "output_plate_root": output_plate_root,
                            "selected": True,
                        }
                    ]
                },
                selected_scope_ids=(selected_plate_root,),
            )

    class _PlateStreamingService:
        def __init__(self):
            self.request = None

        def stream_files(self, request):
            self.request = request
            return PlateFileStreamResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_microscope_type=request.microscope_type,
            )

    plate_streaming_service = _PlateStreamingService()
    context = _selected_plate_mcp_context(
        ui_bridge_service=_UiBridgeService(),
        plate_streaming_service=plate_streaming_service,
    )

    async def call_selected_plate_stream_tool():
        built = server.build_server(context)
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_stream_selected_plate_files_to_viewer",
                {
                    "kind": "result",
                    "target": "output",
                    "path_contains": "rois",
                    "limit": 2,
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_selected_plate_stream_tool())
    payload = json.loads(_direct_tool_text(result))

    assert plate_streaming_service.request.plate_path == output_plate_root
    assert plate_streaming_service.request.context_plate_path == selected_plate_root
    assert plate_streaming_service.request.microscope_type == "auto"
    assert plate_streaming_service.request.kind.value == "result"
    assert plate_streaming_service.request.path_contains == "rois"
    assert plate_streaming_service.request.limit == 2
    assert payload["selected_plate"]["output_plate_root"] == output_plate_root
    assert payload["target"] == "output"
    assert payload["stream"]["plate_path"] == output_plate_root


def test_mcp_selected_plate_image_sample_composes_ui_state_and_plate_service():
    if importlib.util.find_spec("mcp") is None:
        return

    selected_plate_root = "/tmp/selected-plate"
    selected_image_path = "./A01_s001_w1_z001_t001.tif"

    class _UiBridgeService:
        def __init__(self):
            self.state_surface_request = None

        def connection_from_fields(self, fields):
            return "ui-connection"

        def get_state_surface(self, request, connection):
            self.state_surface_request = request
            assert connection == "ui-connection"
            return UiStateSurfaceDocument(
                schema_version=SCHEMA_VERSION,
                summary=UiStateSurfaceSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiStateSurfaceIdentity(surface_id="plate_manager.state"),
                    widget_id="plate_manager",
                    title="Plate manager state",
                    readable=True,
                    supported_selection_modes=("selected", "all"),
                    current_selection_count=1,
                    total_scope_count=1,
                ),
                payload_schema="openhcs.ui.plate_manager_state.v1",
                payload={
                    "rows": [
                        {
                            "plate_scope_id": selected_plate_root,
                            "name": "selected-plate",
                            "plate_root": selected_plate_root,
                            "selected": True,
                        }
                    ]
                },
                selected_scope_ids=(selected_plate_root,),
            )

    class _PlateInspectionService:
        def __init__(self):
            self.inspect_request = None
            self.sample_request = None

        def inspect(self, request):
            self.inspect_request = request
            return PlatePathInspectionResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_microscope_type=request.microscope_type,
                status=PlateInspectionStatus.OK,
                confidence=PlateInspectionConfidence.HIGH,
                image_files=PlateInspectionImageFileSummary(
                    count=1,
                    sampled_records=(
                        PlateInspectionImageRecordSummary(
                            virtual_path=selected_image_path,
                            full_virtual_path=(
                                f"{selected_plate_root}/A01_s001_w1_z001_t001.tif"
                            ),
                            source_path="/tmp/source/A01_w1.tif",
                        ),
                    ),
                ),
            )

        def sample_image(self, request):
            self.sample_request = request
            return PlateImageSampleResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_image_path=request.image_path,
                virtual_path=request.image_path,
                source_path="/tmp/source/A01_w1.tif",
                shape=(1, 2, 2),
                dtype="uint16",
                minimum=1,
                maximum=4,
                mean=2.5,
                sample_origin_yx=(request.y, request.x),
                sample_shape=(1, request.height, request.width),
                sample_included=True,
                sample_values=(((1, 2), (3, 4)),),
            )

    ui_bridge_service = _UiBridgeService()
    plate_inspection_service = _PlateInspectionService()
    context = _selected_plate_mcp_context(
        ui_bridge_service=ui_bridge_service,
        plate_inspection_service=plate_inspection_service,
    )

    async def call_selected_plate_sample_tool():
        built = server.build_server(context)
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_sample_selected_plate_image",
                {
                    "image_path": selected_image_path,
                    "microscope_type": "openhcsdata",
                    "height": 2,
                    "width": 2,
                    "resolution_index": 0,
                    "max_auto_resolution_size": 512,
                },
            ),
            timeout=2,
        )
        return result

    result = asyncio.run(call_selected_plate_sample_tool())
    payload = json.loads(_direct_tool_text(result))

    assert ui_bridge_service.state_surface_request.surface_id == "plate_manager.state"
    assert ui_bridge_service.state_surface_request.selection_mode == "selected"
    assert plate_inspection_service.inspect_request is None
    assert plate_inspection_service.sample_request.plate_path == selected_plate_root
    assert plate_inspection_service.sample_request.image_path == selected_image_path
    assert plate_inspection_service.sample_request.height == 2
    assert plate_inspection_service.sample_request.resolution_index == 0
    assert plate_inspection_service.sample_request.max_auto_resolution_size == 512
    assert payload["selected_plate"]["plate_root"] == selected_plate_root
    assert payload["image_path"] == selected_image_path
    assert payload["auto_selected_image_path"] is False
    assert payload["sample"]["virtual_path"] == selected_image_path


def test_mcp_selected_plate_image_sample_auto_selects_first_inventory_record():
    if importlib.util.find_spec("mcp") is None:
        return

    selected_plate_root = "/tmp/selected-plate"
    selected_image_path = "./A01_s001_w1_z001_t001.tif"

    class _UiBridgeService:
        def connection_from_fields(self, fields):
            return "ui-connection"

        def get_state_surface(self, request, connection):
            assert connection == "ui-connection"
            return UiStateSurfaceDocument(
                schema_version=SCHEMA_VERSION,
                summary=UiStateSurfaceSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiStateSurfaceIdentity(surface_id="plate_manager.state"),
                    widget_id="plate_manager",
                    title="Plate manager state",
                    readable=True,
                    current_selection_count=1,
                    total_scope_count=1,
                ),
                payload_schema="openhcs.ui.plate_manager_state.v1",
                payload={
                    "rows": [
                        {
                            "plate_scope_id": selected_plate_root,
                            "name": "selected-plate",
                            "plate_root": selected_plate_root,
                            "selected": True,
                        }
                    ]
                },
                selected_scope_ids=(selected_plate_root,),
            )

    class _PlateInspectionService:
        def __init__(self):
            self.inspect_request = None
            self.sample_request = None

        def inspect(self, request):
            self.inspect_request = request
            return PlatePathInspectionResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_microscope_type=request.microscope_type,
                status=PlateInspectionStatus.OK,
                confidence=PlateInspectionConfidence.HIGH,
                image_files=PlateInspectionImageFileSummary(
                    count=1,
                    sampled_records=(
                        PlateInspectionImageRecordSummary(
                            virtual_path=selected_image_path,
                            full_virtual_path=(
                                f"{selected_plate_root}/A01_s001_w1_z001_t001.tif"
                            ),
                            source_path="/tmp/source/A01_w1.tif",
                        ),
                    ),
                ),
            )

        def sample_image(self, request):
            self.sample_request = request
            return PlateImageSampleResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_image_path=request.image_path,
                virtual_path=request.image_path,
                source_path="/tmp/source/A01_w1.tif",
                shape=(1, 1, 1),
                dtype="uint16",
                sample_included=False,
                sample_omitted_reason="include_array_values is false",
            )

    plate_inspection_service = _PlateInspectionService()
    context = _selected_plate_mcp_context(
        ui_bridge_service=_UiBridgeService(),
        plate_inspection_service=plate_inspection_service,
    )

    async def call_selected_plate_sample_tool():
        built = server.build_server(context)
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_sample_selected_plate_image",
                {"microscope_type": "openhcsdata", "include_array_values": False},
            ),
            timeout=2,
        )
        return result

    result = asyncio.run(call_selected_plate_sample_tool())
    payload = json.loads(_direct_tool_text(result))

    assert plate_inspection_service.inspect_request.plate_path == selected_plate_root
    assert plate_inspection_service.inspect_request.bounds.max_sample_files == 1
    assert plate_inspection_service.sample_request.image_path == selected_image_path
    assert plate_inspection_service.sample_request.include_array_values is False
    assert payload["image_path"] == selected_image_path
    assert payload["auto_selected_image_path"] is True
    assert payload["sample"]["requested_image_path"] == selected_image_path


def test_mcp_selected_plate_image_sample_targets_output_plate():
    if importlib.util.find_spec("mcp") is None:
        return

    selected_plate_root = "/tmp/selected-plate"
    output_plate_root = "/tmp/selected-plate_openhcs"
    selected_image_path = "./A01_s001_w1_z001_t001.tif"

    class _UiBridgeService:
        def connection_from_fields(self, fields):
            return "ui-connection"

        def get_state_surface(self, request, connection):
            assert connection == "ui-connection"
            return UiStateSurfaceDocument(
                schema_version=SCHEMA_VERSION,
                summary=UiStateSurfaceSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiStateSurfaceIdentity(surface_id="plate_manager.state"),
                    widget_id="plate_manager",
                    title="Plate manager state",
                    readable=True,
                    current_selection_count=1,
                    total_scope_count=1,
                ),
                payload_schema="openhcs.ui.plate_manager_state.v1",
                payload={
                    "rows": [
                        {
                            "plate_scope_id": selected_plate_root,
                            "name": "selected-plate",
                            "plate_root": selected_plate_root,
                            "output_plate_root": output_plate_root,
                            "selected": True,
                        }
                    ]
                },
                selected_scope_ids=(selected_plate_root,),
            )

    class _PlateInspectionService:
        def __init__(self):
            self.sample_request = None

        def sample_image(self, request):
            self.sample_request = request
            return PlateImageSampleResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_image_path=request.image_path,
                virtual_path=request.image_path,
                source_path="/tmp/output/A01_w1.tif",
                shape=(1, 1, 1),
                dtype="uint16",
            )

    plate_inspection_service = _PlateInspectionService()
    context = _selected_plate_mcp_context(
        ui_bridge_service=_UiBridgeService(),
        plate_inspection_service=plate_inspection_service,
    )

    async def call_selected_plate_sample_tool():
        built = server.build_server(context)
        result = await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_sample_selected_plate_image",
                {
                    "image_path": selected_image_path,
                    "target": "output",
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )
        return result

    result = asyncio.run(call_selected_plate_sample_tool())
    payload = json.loads(_direct_tool_text(result))

    assert plate_inspection_service.sample_request.plate_path == output_plate_root
    assert plate_inspection_service.sample_request.microscope_type == "auto"
    assert plate_inspection_service.sample_request.image_path == selected_image_path
    assert payload["target"] == "output"
    assert payload["sample"]["plate_path"] == output_plate_root


def test_mcp_stdio_server_roundtrip_returns_errors_as_payloads():
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def call_stdio_server():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=("-m", "openhcs.mcp"),
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=5)
                health = await asyncio.wait_for(
                    session.call_tool("openhcs_health_check", {}),
                    timeout=5,
                )
                bad_viewer_call = await asyncio.wait_for(
                    session.call_tool(
                        "openhcs_probe_viewer_window",
                        {"port": 1, "timeout_ms": 120_000},
                    ),
                    timeout=5,
                )
                return health, bad_viewer_call

    health, bad_viewer_call = asyncio.run(call_stdio_server())
    health_payload = json.loads(_direct_tool_text(health))
    bad_payload = json.loads(_direct_tool_text(bad_viewer_call))

    assert health_payload["status"] == "ok"
    assert health_payload["server_source_path"].endswith("openhcs/mcp/server.py")
    assert isinstance(health_payload["server_process_id"], int)
    assert isinstance(health_payload["started_at_unix"], float)
    assert isinstance(health_payload["server_import_mtime_ns"], int)
    assert isinstance(health_payload["server_current_mtime_ns"], int)
    assert health_payload["server_source_changed_since_import"] is False
    assert bad_payload["ok"] is False
    assert bad_payload["tool"] == "openhcs_probe_viewer_window"
    assert bad_payload["errors"][0]["code"] == "mcp_tool_failed"


def test_mcp_stdio_validation_error_keeps_session_alive():
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def call_stdio_server():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=("-m", "openhcs.mcp"),
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=5)
                invalid_workflow = await asyncio.wait_for(
                    session.call_tool(
                        "openhcs_ui_selected_plate_workflow",
                        {"workflow": "not_a_workflow"},
                    ),
                    timeout=5,
                )
                health = await asyncio.wait_for(
                    session.call_tool("openhcs_health_check", {}),
                    timeout=5,
                )
                return invalid_workflow, health

    invalid_workflow, health = asyncio.run(call_stdio_server())
    invalid_text = _direct_tool_text(invalid_workflow)
    health_payload = json.loads(_direct_tool_text(health))

    assert invalid_workflow.isError is True
    assert "not_a_workflow" in invalid_text
    assert "init_plate" in invalid_text
    assert health_payload["status"] == "ok"


def test_mcp_dev_client_rejects_non_object_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    assert dev_client.parse_json_object('{"port": 5565}') == {"port": 5565}

    with pytest.raises(ValueError, match="JSON object"):
        dev_client.parse_json_object("[5565]")


def test_mcp_dev_client_accepts_common_flags_after_subcommands():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()

    after_command = parser.parse_args(("ui-smoke", "--allow-error-payloads"))
    before_command = parser.parse_args(("--allow-error-payloads", "ui-smoke"))

    assert after_command.allow_error_payloads is True
    assert before_command.allow_error_payloads is True


def test_mcp_dev_client_generated_profiles_cover_capability_connection_profiles():
    if importlib.util.find_spec("mcp") is None:
        return

    from openhcs.agent.capabilities import CapabilityCliConnectionProfile
    import openhcs.mcp.dev_client as dev_client

    assert set(dev_client.GeneratedMcpDevCommandProfile.__registry__) == set(
        CapabilityCliConnectionProfile
    )


def test_mcp_dev_client_knowledge_commands_project_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()

    list_args = parser.parse_args(("knowledge",))
    document_args = parser.parse_args(
        (
            "knowledge-document",
            "openhcs_architecture_quick_start",
            "--section-id",
            "first-mcp-session",
            "--max-chars",
            "4096",
        )
    )
    inline_section_args = parser.parse_args(
        (
            "knowledge-document",
            "openhcs_architecture_quick_start#first-mcp-session",
        )
    )
    search_args = parser.parse_args(
        (
            "knowledge-search",
            "ObjectState",
            "field",
            "help",
            "--limit",
            "3",
        )
    )

    list_call = dev_client._calls_from_args(list_args)[0]
    document_call = dev_client._calls_from_args(document_args)[0]
    inline_section_call = dev_client._calls_from_args(inline_section_args)[0]
    search_call = dev_client._calls_from_args(search_args)[0]

    assert list_call.name == "openhcs_list_knowledge_documents"
    assert list_call.arguments == {}
    assert document_call.name == "openhcs_get_knowledge_document"
    assert document_call.arguments == {
        "document_id": "openhcs_architecture_quick_start",
        "section_id": "first-mcp-session",
        "max_chars": 4096,
    }
    assert inline_section_call.arguments == {
        "document_id": "openhcs_architecture_quick_start",
        "section_id": "first-mcp-session",
        "max_chars": KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_architecture_quick_start"
        ).bounds.max_chars,
    }
    assert search_call.name == "openhcs_search_knowledge"
    assert search_call.arguments == {
        "query": "ObjectState field help",
        "limit": 3,
    }


def test_mcp_dev_client_architecture_commands_project_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()

    list_args = parser.parse_args(("architecture", "--contains", "source"))
    topic_args = parser.parse_args(("architecture-topic", "source_semantics"))
    topic_alias_args = parser.parse_args(("explain-architecture", "pipeline_model"))
    symbol_args = parser.parse_args(("internal-symbol", "core.FunctionStep"))
    symbol_alias_args = parser.parse_args(
        ("architecture-symbol", "source.StepSourceBindingsConfig")
    )

    list_call = dev_client._calls_from_args(list_args)[0]
    topic_call = dev_client._calls_from_args(topic_args)[0]
    topic_alias_call = dev_client._calls_from_args(topic_alias_args)[0]
    symbol_call = dev_client._calls_from_args(symbol_args)[0]
    symbol_alias_call = dev_client._calls_from_args(symbol_alias_args)[0]

    assert list_call.name == "openhcs_list_architecture_topics"
    assert list_call.arguments == {}
    assert topic_call.name == "openhcs_explain_architecture"
    assert topic_call.arguments == {"topic_id": "source_semantics"}
    assert topic_alias_call.arguments == {"topic_id": "pipeline_model"}
    assert symbol_call.name == "openhcs_describe_internal_symbol"
    assert symbol_call.arguments == {"symbol_id": "core.FunctionStep"}
    assert symbol_alias_call.arguments == {
        "symbol_id": "source.StepSourceBindingsConfig"
    }


def test_mcp_dev_client_tools_command_renders_compact_filtered_list():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("tools", "--contains", "viewer", "--limit", "1"))
    response = {
        "errors": [],
        "tool_count": 3,
        "tools": [
            {
                "name": "openhcs_get_viewer_window_state",
                "description": "Read bounded viewer state.",
                "input_schema": {},
            },
            {
                "name": "openhcs_validate_viewer_window_state",
                "description": "Validate viewer axes and payloads.",
                "input_schema": {},
            },
            {
                "name": "openhcs_health_check",
                "description": "Report health.",
                "input_schema": {},
            },
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("tools").render_response(
        response,
        args,
    )

    assert "Tools: matched=2 total=3 shown=1" in rendered
    assert "Filter: contains=viewer" in rendered
    assert "[Viewer Review]" in rendered
    assert "- openhcs_get_viewer_window_state: Read bounded viewer state." in rendered
    assert "...<truncated 1 tools>" in rendered
    assert "input_schema" not in rendered


def test_mcp_dev_client_tools_command_can_render_flat_list():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("tools", "--flat"))
    response = {
        "errors": [],
        "tool_count": 1,
        "tools": [
            {
                "name": "openhcs_get_viewer_window_state",
                "description": "Read bounded viewer state.",
                "input_schema": {},
            },
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("tools").render_response(
        response,
        args,
    )

    assert "Tools: matched=1 total=1 shown=1" in rendered
    assert "[Viewer Review]" not in rendered
    assert "- openhcs_get_viewer_window_state: Read bounded viewer state." in rendered


def test_mcp_dev_client_knowledge_command_renders_compact_catalog():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("knowledge", "--contains", "domain", "--limit", "1"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_list_knowledge_documents",
                "mcp_error": False,
                "payloads": [
                    {
                        "documents": [
                            {
                                "document_id": "openhcs_domain_expert_onboarding",
                                "title": "OpenHCS domain expert onboarding",
                                "summary": "Scientist-facing onboarding.",
                                "section_count": 7,
                                "source_path": "docs/domain.rst",
                                "tags": [
                                    "segmentation",
                                    "first pipeline",
                                    "plate",
                                    "channel",
                                    "well",
                                    "site",
                                    "extra",
                                ],
                            },
                            {
                                "document_id": "openhcs_viewer_management",
                                "title": "Viewer management",
                                "summary": "Napari and Fiji.",
                                "section_count": 3,
                                "source_path": "docs/viewer.rst",
                                "tags": ["viewer"],
                            },
                        ],
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("knowledge").render_response(
        response,
        args,
    )

    assert "Knowledge documents: matched=1 shown=1" in rendered
    assert "Filter: contains=domain" in rendered
    assert (
        '- openhcs_domain_expert_onboarding: title="OpenHCS domain expert onboarding" '
        "sections=7 path=docs/domain.rst "
        "tags=segmentation,first pipeline,plate,channel,well,site,+1"
    ) in rendered
    assert "openhcs_viewer_management" not in rendered


def test_mcp_dev_client_knowledge_search_renders_compact_hits():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("knowledge-search", "count cells", "--limit", "1"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_search_knowledge",
                "mcp_error": False,
                "payloads": [
                    {
                        "query": "count cells",
                        "hits": [
                            {
                                "document": {
                                    "document_id": "openhcs_real_time_visualization",
                                    "title": "Real-time visualization",
                                },
                                "section": {
                                    "section_id": "roi-streaming",
                                    "title": "ROI Streaming",
                                },
                                "score": 60,
                                "line_number": 244,
                                "matched_terms": ["roi", "cells"],
                                "snippet": "OpenHCS automatically streams ROIs.",
                            }
                        ],
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "knowledge-search"
    ).render_response(
        response,
        args,
    )

    assert 'Knowledge search: query="count cells" hits=1' in rendered
    assert (
        "- openhcs_real_time_visualization#roi-streaming: score=60 line=244 "
        'title="ROI Streaming" terms=roi,cells'
    ) in rendered
    assert "OpenHCS automatically streams ROIs." in rendered


def test_mcp_dev_client_knowledge_search_accepts_query_option():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "knowledge-search",
            "--query",
            "source bindings",
            "--query",
            "roi",
            "--limit",
            "3",
        )
    )
    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_search_knowledge"
    assert call.arguments["query"] == "source bindings roi"
    assert call.arguments["limit"] == 3

    missing_query_args = parser.parse_args(("knowledge-search",))
    with pytest.raises(dev_client.McpDevCliUsageError, match="requires a query"):
        dev_client._calls_from_args(missing_query_args)


def test_mcp_dev_client_knowledge_document_renders_compact_content():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "knowledge-document",
            "openhcs_domain_expert_onboarding",
            "--section-id",
            "first-workflow",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_get_knowledge_document",
                "mcp_error": False,
                "payloads": [
                    {
                        "document": {
                            "document_id": "openhcs_domain_expert_onboarding",
                            "title": "OpenHCS domain expert onboarding",
                            "source_path": "docs/domain.rst",
                        },
                        "sections": [{"section_id": "first-workflow"}],
                        "selected_section_id": "first-workflow",
                        "max_chars": 1200,
                        "content": "First Workflow\n--------------\nRun one well first.",
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "knowledge-document"
    ).render_response(response, args)

    assert (
        "Knowledge document: id=openhcs_domain_expert_onboarding "
        'title="OpenHCS domain expert onboarding" path=docs/domain.rst '
        "sections=1 max_chars=1200"
    ) in rendered
    assert "Selected section: first-workflow" in rendered
    assert "Sections:" not in rendered
    assert "Content:\nFirst Workflow" in rendered


def test_mcp_dev_client_knowledge_document_renders_truncation_warning():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "knowledge-document",
            "openhcs_official30_benchmark_recipes",
            "--max-chars",
            "3000",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_get_knowledge_document",
                "mcp_error": False,
                "payloads": [
                    {
                        "document": {
                            "document_id": "openhcs_official30_benchmark_recipes",
                            "title": "OpenHCS official30 benchmark pipeline recipes",
                            "source_path": (
                                "benchmark/manifests/official30_portable_axis1.json"
                            ),
                        },
                        "sections": [
                            {
                                "section_id": "official30-benchmark-pipeline-recipes",
                                "title": "Official30 Benchmark Pipeline Recipes",
                            },
                            {
                                "section_id": "case-index",
                                "title": "Case Index",
                            },
                            {
                                "section_id": "examplehuman",
                                "title": "ExampleHuman",
                            },
                        ],
                        "max_chars": 3000,
                        "content": "Recipe count: 30\n\nRecipes:\n1. ExampleHuman",
                        "truncated": True,
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "knowledge-document"
    ).render_response(response, args)

    assert "Recipe count: 30" in rendered
    assert "Sections:" in rendered
    assert "- examplehuman: ExampleHuman" in rendered
    assert (
        "Content truncated; rerun with a larger --max-chars or a narrower --section-id."
    ) in rendered


def test_mcp_dev_client_architecture_commands_render_compact_context():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    list_args = parser.parse_args(
        ("architecture", "--contains", "source", "--limit", "1")
    )
    topic_args = parser.parse_args(("architecture-topic", "source_semantics"))
    symbol_args = parser.parse_args(("internal-symbol", "core.FunctionStep"))
    call_args = parser.parse_args(
        (
            "call",
            "openhcs_explain_architecture",
            "--arguments",
            '{"topic_id": "source_semantics"}',
        )
    )

    list_response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_list_architecture_topics",
                "mcp_error": False,
                "payloads": [
                    {
                        "topics": [
                            {
                                "topic_id": "source_semantics",
                                "title": "Source schema and semantic image names",
                                "summary": "Filenames, metadata, source bindings.",
                            },
                            {
                                "topic_id": "execution_runtime",
                                "title": "Compile and execution runtime",
                                "summary": "How pipelines run.",
                            },
                        ]
                    }
                ],
            }
        ],
    }
    topic_response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_explain_architecture",
                "mcp_error": False,
                "payloads": [
                    {
                        "topic_id": "source_semantics",
                        "title": "Source schema and semantic image names",
                        "summary": "Preserve image/object semantics.",
                        "concepts": [
                            "MetadataExtractionRule owns filename semantics.",
                        ],
                        "cellprofiler_translation_notes": [],
                        "internal_symbols": [
                            {
                                "symbol_id": "source.StepSourceBindingsConfig",
                                "title": "StepSourceBindingsConfig",
                                "role": "Step-local semantic input bindings.",
                                "symbol_kind": "class",
                                "import_path": "openhcs.core.source_bindings.StepSourceBindingsConfig",
                                "source_path": "openhcs/core/source_bindings.py",
                                "line_number": 607,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    symbol_response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_describe_internal_symbol",
                "mcp_error": False,
                "payloads": [
                    {
                        "symbol_id": "core.FunctionStep",
                        "title": "FunctionStep",
                        "role": "Pipeline declaration authority.",
                        "symbol_kind": "class",
                        "signature": "(func=[], **kwargs)",
                        "doc_summary": "Pipeline step that delegates execution.",
                        "import_path": "openhcs.core.steps.function_step.FunctionStep",
                        "source_path": "openhcs/core/steps/function_step.py",
                        "line_number": 18,
                    }
                ],
            }
        ],
    }

    list_rendered = dev_client.McpDevCommandSpec.for_name(
        "architecture"
    ).render_response(list_response, list_args)
    topic_rendered = dev_client.McpDevCommandSpec.for_name(
        "architecture-topic"
    ).render_response(topic_response, topic_args)
    symbol_rendered = dev_client.McpDevCommandSpec.for_name(
        "internal-symbol"
    ).render_response(symbol_response, symbol_args)
    call_rendered = dev_client.McpDevCommandSpec.for_name("call").render_response(
        topic_response,
        call_args,
    )

    assert "Architecture topics: matched=1 shown=1" in list_rendered
    assert "- source_semantics:" in list_rendered
    assert "execution_runtime" not in list_rendered
    assert "Architecture topic: id=source_semantics" in topic_rendered
    assert "- MetadataExtractionRule owns filename semantics." in topic_rendered
    assert "- source.StepSourceBindingsConfig:" in topic_rendered
    assert "role=Step-local semantic input bindings." in topic_rendered
    assert (
        'Internal symbol: id=core.FunctionStep title="FunctionStep" kind=class'
        in symbol_rendered
    )
    assert "Source: openhcs/core/steps/function_step.py:18" in symbol_rendered
    assert "Signature: (func=[], **kwargs)" in symbol_rendered
    assert "Architecture topic: id=source_semantics" in call_rendered
    assert "payloads" not in call_rendered


def test_mcp_dev_client_function_commands_project_tool_arguments(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    search_args = parser.parse_args(
        (
            "functions",
            "--query",
            "count cells",
            "--library",
            "openhcs",
            "--limit",
            "5",
            "--full-signatures",
        )
    )
    detail_args = parser.parse_args(
        (
            "function",
            "openhcs:analysis_count_cells_simple_count_cells_simple",
            "--max-doc-chars",
            "1200",
            "--full-signature",
        )
    )
    custom_source = tmp_path / "custom.py"
    custom_source.write_text(
        "from openhcs.core.memory import numpy\n", encoding="utf-8"
    )
    register_args = parser.parse_args(
        (
            "register-custom-function",
            str(custom_source),
            "--no-persist",
            "--full-signature",
        )
    )
    authoring_args = parser.parse_args(
        ("authoring-context", "--kind", "pipeline", "--max-chars", "1234")
    )

    search_call = dev_client._calls_from_args(search_args)[0]
    detail_call = dev_client._calls_from_args(detail_args)[0]
    register_call = dev_client._calls_from_args(register_args)[0]
    authoring_call = dev_client._calls_from_args(authoring_args)[0]

    assert search_call.name == "openhcs_search_functions"
    assert search_call.arguments == {
        "query": "count cells",
        "library": "openhcs",
        "limit": 5,
        "compact_signatures": False,
    }
    assert detail_call.name == "openhcs_describe_function"
    assert detail_call.arguments == {
        "function_id": "openhcs:analysis_count_cells_simple_count_cells_simple",
        "max_doc_chars": 1200,
        "compact_signature": False,
    }
    assert register_call.name == "openhcs_register_custom_function"
    assert register_call.arguments == {
        "source_code": "from openhcs.core.memory import numpy\n",
        "persist": False,
        "compact_signature": False,
    }
    assert authoring_call.name == "openhcs_get_authoring_context"
    assert authoring_call.arguments == {
        "kind": "pipeline",
        "max_chars": 1234,
    }
    assert authoring_args.max_chars == 1234

    topic_args = parser.parse_args(("authoring-context", "--topic", "pipeline"))
    topic_call = dev_client._calls_from_args(topic_args)[0]
    assert topic_call.arguments["kind"] == "pipeline"

    positional_args = parser.parse_args(("authoring-context", "first_use"))
    positional_call = dev_client._calls_from_args(positional_args)[0]
    assert positional_call.arguments["kind"] == "first_use"

    conflicting_args = parser.parse_args(
        ("authoring-context", "first_use", "--kind", "pipeline")
    )
    with pytest.raises(dev_client.McpDevCliUsageError, match="different values"):
        dev_client._calls_from_args(conflicting_args)


def test_mcp_dev_client_authoring_context_kind_choices_are_explicit(capsys):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(("authoring-context", "--help"))
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    for kind in (
        "pipeline",
        "custom_function",
        "first_use",
        "folder_onboarding",
        "domain_expert_assisted_setup",
        "ui_visible_workflow",
        "headless_execution",
        "viewer_review",
        "objectstate_editing",
        "cellprofiler_translation",
    ):
        assert kind in help_text

    with pytest.raises(SystemExit):
        parser.parse_args(("authoring-context", "--kind", "function"))


def test_mcp_dev_client_registry_context_commands_use_cold_start_timeout():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    default_args = parser.parse_args(("authoring-context",))
    default_call = dev_client._calls_from_args(default_args)[0]
    explicit_args = parser.parse_args(
        ("functions", "--timeout-seconds", "60", "--limit", "1")
    )

    assert [
        dev_client.McpDevCommandSpec.for_name("authoring-context").timeout_seconds(
            default_args
        ),
        dev_client.McpDevCommandSpec.for_name("functions").timeout_seconds(
            explicit_args
        ),
    ] == [
        dev_client.DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS,
        60.0,
    ]
    assert default_call.arguments["kind"] == "first_use"
    assert default_call.arguments["max_chars"] == 16_000


def test_mcp_dev_client_functions_command_renders_compact_search_results():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("functions", "count cells", "--limit", "1"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_search_functions",
                "mcp_error": False,
                "payloads": [
                    {
                        "query": "count cells",
                        "library": None,
                        "total": 29,
                        "items": [
                            {
                                "function_id": (
                                    "openhcs:analysis_count_cells_simple_count_cells_simple"
                                ),
                                "signature": "count_cells_simple(image, ...)",
                                "backend_tags": [
                                    "openhcs",
                                    "analysis",
                                    "count_cells_simple",
                                ],
                                "summary": "Count cells with simple thresholding.",
                            }
                        ],
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("functions").render_response(
        response,
        args,
    )

    assert (
        'Function search: query="count cells" library=<none> shown=1 total=29'
        in rendered
    )
    assert (
        "- openhcs:analysis_count_cells_simple_count_cells_simple: "
        "count_cells_simple(image, ...) tags=openhcs,analysis,count_cells_simple"
    ) in rendered
    assert "Count cells with simple thresholding." in rendered


def test_mcp_dev_client_register_custom_function_renders_next_steps():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        ("register-custom-function", "--source-code", "source", "--no-persist")
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_register_custom_function",
                "mcp_error": False,
                "payloads": [
                    {
                        "registered_count": 1,
                        "persisted": False,
                        "storage_dir": "/tmp/custom_functions",
                        "source_file_paths": [],
                        "functions": [
                            {
                                "function_id": "openhcs:agent_registered_custom",
                                "signature": ("agent_registered_custom(gain=1.0)"),
                                "backend_tags": ["openhcs", "custom"],
                                "summary": "Agent custom function.",
                            }
                        ],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "register-custom-function"
    ).render_response(response, args)

    assert (
        "Custom function registration: registered=1 persisted=False "
        "storage=/tmp/custom_functions"
    ) in rendered
    assert "Lifetime: process-local only" in rendered
    assert (
        "- openhcs:agent_registered_custom: agent_registered_custom(gain=1.0)"
        in rendered
    )
    assert "- function openhcs:agent_registered_custom" in rendered
    assert (
        "- draft-pipeline-step openhcs:agent_registered_custom --name <step_name>"
        in rendered
    )


def test_mcp_dev_client_function_command_renders_parameters_and_artifacts():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        ("function", "openhcs:analysis_count_cells_simple_count_cells_simple")
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_describe_function",
                "mcp_error": False,
                "payloads": [
                    {
                        "entry": {
                            "function_id": (
                                "openhcs:analysis_count_cells_simple_count_cells_simple"
                            ),
                            "name": "count_cells_simple",
                            "library": "openhcs",
                            "signature": "count_cells_simple(image, min_size, max_size)",
                            "summary": "Count cells.",
                        },
                        "parameters": [
                            {
                                "name": "image",
                                "required": True,
                                "annotation": None,
                                "default_repr": None,
                                "supplied_by": "runtime_primary_input",
                                "description": (
                                    "Supplied by OpenHCS from the FunctionStep "
                                    "input image payload; do not pass this as a "
                                    "function kwarg."
                                ),
                            },
                            {
                                "name": "min_size",
                                "required": False,
                                "annotation": "int",
                                "default_repr": "20",
                                "supplied_by": "agent",
                            },
                        ],
                        "runtime_contract": {
                            "artifact_outputs": [
                                {
                                    "name": "segmentation_masks",
                                    "kind": "special",
                                    "required": True,
                                }
                            ]
                        },
                        "doc": "Count cells in a 3D image.",
                        "doc_chars": 27,
                        "doc_truncated": False,
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("function").render_response(
        response,
        args,
    )

    assert (
        "Function: id=openhcs:analysis_count_cells_simple_count_cells_simple"
        in rendered
    )
    assert "Agent parameters:" in rendered
    assert "- min_size: required=False type=int default=20" in rendered
    assert "Runtime inputs:" in rendered
    assert "- image: supplied_by=runtime_primary_input" in rendered
    assert "- segmentation_masks: kind=special required=True" in rendered
    assert "Doc: chars=27 truncated=False" in rendered


def test_mcp_dev_client_function_command_suggests_full_doc_retry():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "function",
            "openhcs:analysis_cell_counting_cpu_count_cells_multi_channel",
            "--max-doc-chars",
            "800",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_describe_function",
                "mcp_error": False,
                "payloads": [
                    {
                        "entry": {
                            "function_id": (
                                "openhcs:analysis_cell_counting_cpu_"
                                "count_cells_multi_channel"
                            ),
                            "name": "count_cells_multi_channel",
                            "library": "openhcs",
                            "signature": "count_cells_multi_channel(chan_1, chan_2)",
                            "summary": "Count cells.",
                        },
                        "parameters": [],
                        "runtime_contract": {},
                        "doc": "Long doc preview.",
                        "doc_chars": 3254,
                        "doc_truncated": True,
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("function").render_response(
        response,
        args,
    )

    assert "Doc: chars=3254 truncated=True" in rendered
    assert (
        "Doc truncated; rerun: function "
        "openhcs:analysis_cell_counting_cpu_count_cells_multi_channel "
        "--max-doc-chars 3254"
    ) in rendered


def test_mcp_dev_client_authoring_context_renders_bounded_content():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("authoring-context", "--max-chars", "12"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_get_authoring_context",
                "mcp_error": False,
                "payloads": [
                    {
                        "kind": "pipeline",
                        "content": "12345678901234567890",
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "authoring-context"
    ).render_response(response, args)

    assert "Authoring context: kind=pipeline" in rendered
    assert "123456789012" in rendered
    assert "...<truncated 8 chars>" in rendered


def test_mcp_dev_client_draft_pipeline_step_command_renders_composite_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "draft-pipeline-step",
            "openhcs:analysis_count_cells_simple_count_cells_simple",
            "--name",
            "Count cells",
            "--kwargs",
            '{"min_size": 5}',
            "--step-config-overrides",
            '{"napari_streaming_config": {"enabled": true}}',
            "--max-source-chars",
            "40",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_create_pipeline",
                "mcp_error": False,
                "payloads": [
                    {
                        "pipeline_id": "pipeline-1",
                        "uri": "openhcs://pipelines/pipeline-1",
                    }
                ],
            },
            {
                "tool": "openhcs_add_function_step",
                "mcp_error": False,
                "payloads": [
                    {
                        "pipeline_id": "pipeline-1",
                        "steps": [
                            {
                                "step_id": "step-1",
                                "name": "Count cells",
                                "enabled": True,
                                "functions": [
                                    {
                                        "function_id": (
                                            "openhcs:analysis_count_cells_simple_count_cells_simple"
                                        )
                                    }
                                ],
                            }
                        ],
                        "errors": [],
                    }
                ],
            },
            {
                "tool": "openhcs_validate_pipeline",
                "mcp_error": False,
                "payloads": [
                    {
                        "valid": True,
                        "warnings": [{"code": "note", "message": "Pipeline is small."}],
                        "errors": [],
                    }
                ],
            },
            {
                "tool": "openhcs_render_pipeline_source",
                "mcp_error": False,
                "payloads": [
                    {
                        "title": "Pipeline",
                        "source": "pipeline_steps = [\\n    FunctionStep(...)\\n]\\n",
                        "errors": [],
                    }
                ],
            },
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "draft-pipeline-step"
    ).render_response(response, args)

    assert "Pipeline draft: id=pipeline-1 valid=True steps=1" in rendered
    assert "Ref: uri=openhcs://pipelines/pipeline-1" in rendered
    assert (
        '- step-1: name="Count cells" enabled=True '
        "functions=openhcs:analysis_count_cells_simple_count_cells_simple"
    ) in rendered
    assert "Validate warnings:" in rendered
    assert "- note: Pipeline is small." in rendered
    assert 'Source: title="Pipeline" bytes=46' in rendered
    assert "...<truncated 6 chars>" in rendered


def test_mcp_dev_client_draft_pipeline_step_suggests_missing_kwargs_repair():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "draft-pipeline-step",
            "openhcs:analysis_cell_counting_cpu_count_cells_multi_channel",
            "--name",
            "Count cells",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_create_pipeline",
                "mcp_error": False,
                "payloads": [
                    {
                        "pipeline_id": "pipeline-1",
                        "uri": "openhcs://pipelines/pipeline-1",
                    }
                ],
            },
            {
                "tool": "openhcs_add_function_step",
                "mcp_error": False,
                "payloads": [
                    {
                        "pipeline_id": "pipeline-1",
                        "steps": [
                            {
                                "step_id": "step-1",
                                "name": "Count cells",
                                "enabled": True,
                                "functions": [
                                    {
                                        "function_id": (
                                            "openhcs:analysis_cell_counting_cpu_"
                                            "count_cells_multi_channel"
                                        )
                                    }
                                ],
                            }
                        ],
                        "errors": [],
                    }
                ],
            },
            {
                "tool": "openhcs_validate_pipeline",
                "mcp_error": False,
                "payloads": [
                    {
                        "valid": False,
                        "warnings": [],
                        "errors": [
                            {
                                "code": "missing_function_kwargs",
                                "message": (
                                    "Missing required kwargs for OpenHCS function "
                                    "openhcs:analysis_cell_counting_cpu_"
                                    "count_cells_multi_channel: chan_1, chan_2."
                                ),
                                "hint": (
                                    "Call openhcs_describe_function for this "
                                    "function_id and provide required agent kwargs: "
                                    "chan_1, chan_2."
                                ),
                            }
                        ],
                    }
                ],
            },
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "draft-pipeline-step"
    ).render_response(response, args)

    assert "Pipeline draft: id=pipeline-1 valid=False steps=1" in rendered
    assert "Validate errors:" in rendered
    assert (
        "Next: function openhcs:analysis_cell_counting_cpu_count_cells_multi_channel"
    ) in rendered
    assert (
        "Retry shape: draft-pipeline-step "
        "openhcs:analysis_cell_counting_cpu_count_cells_multi_channel "
        '--kwargs \'{"chan_1": <value>, "chan_2": <value>}\''
    ) in rendered


def test_mcp_dev_client_artifact_plan_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    pipeline_source = (
        "from openhcs.core.config import PipelineConfig\n"
        "pipeline_config = PipelineConfig()\n"
        "pipeline_steps = []"
    )
    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "artifact-plan",
            "/tmp/example-plate",
            "--source-text",
            pipeline_source,
            "--well-filter",
            "A01,A02",
            "--global-config-id",
            "global-1",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_inspect_pipeline_source_artifact_plan"
    assert call.arguments == {
        "plate_path": "/tmp/example-plate",
        "pipeline_source": pipeline_source,
        "axis_filter": ["A01", "A02"],
        "global_config_id": "global-1",
    }


def test_mcp_dev_client_artifact_plan_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "artifact-plan",
            "/tmp/example-plate",
            "--source-text",
            "pipeline_steps = []",
            "--axis-filter",
            "A01",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_inspect_pipeline_source_artifact_plan",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "axes": ["A01"],
                        "axis_count": 1,
                        "axis_filter": ["A01"],
                        "progress_event_count": 1,
                        "step_count": 1,
                        "source_workspace": {
                            "file_count": 2,
                            "truncated_file_count": 0,
                            "axis_file_counts": {"A01": 2},
                            "files": [
                                {
                                    "virtual_path": "A01_s001_w1_z001_t001.tif",
                                    "source_path": "/tmp/source/A01_w1.tif",
                                    "source_metadata": {
                                        "well": "A01",
                                        "channel": 1,
                                    },
                                }
                            ],
                        },
                        "worker_assignments": {"worker_0": ["A01"]},
                        "steps": [
                            {
                                "step_index": 0,
                                "step_name": "Count cells",
                                "axis_id": "A01",
                                "execution_groups": [None],
                                "artifact_inputs": [
                                    {
                                        "name": "positions",
                                        "kind": "special",
                                        "path": (
                                            "/tmp/example-plate_openhcs/results/"
                                            "A01_positions_step0.pkl"
                                        ),
                                        "group_keys": [None],
                                        "source_step_id": 0,
                                        "source_step_scope_id": "step-find-positions",
                                    }
                                ],
                                "artifact_outputs": [
                                    {
                                        "name": "cell_counts",
                                        "kind": "special",
                                        "path": (
                                            "/tmp/example-plate_openhcs/results/"
                                            "A01_cell_counts_step0.pkl"
                                        ),
                                        "group_keys": [None],
                                        "materialization": {
                                            "persistent_enabled": True,
                                            "persistent_backend": "disk",
                                            "analysis_output_dir": (
                                                "/tmp/example-plate_openhcs/"
                                                "images_results"
                                            ),
                                            "paths": [
                                                {
                                                    "group_key": None,
                                                    "base_path": (
                                                        "/tmp/example-plate_openhcs/"
                                                        "images_results/"
                                                        "A01_cell_counts_step0.roi.zip"
                                                    ),
                                                    "candidate_paths": [
                                                        "/tmp/example-plate_openhcs/"
                                                        "images_results/"
                                                        "A01_cell_counts_step0_details.csv"
                                                    ],
                                                }
                                            ],
                                            "runtime_resolved": False,
                                            "disabled": False,
                                            "filename_uses_source_identity": False,
                                            "runtime_metadata_can_refine_paths": True,
                                            "note": None,
                                        },
                                    }
                                ],
                                "truncated_artifact_input_count": 0,
                                "truncated_artifact_output_count": 0,
                            }
                        ],
                        "warnings": [],
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("artifact-plan").render_response(
        response,
        args,
    )

    assert (
        "Artifact plan: plate=/tmp/example-plate axes=1 steps=1 progress_events=1"
        in rendered
    )
    assert "Axes: A01" in rendered
    assert "Source workspace (source-bound files): files=2 truncated=0" in rendered
    assert "axis files: A01=2" in rendered
    assert (
        "- A01_s001_w1_z001_t001.tif -> /tmp/source/A01_w1.tif "
        "components=well=A01, channel=1"
    ) in rendered
    assert "Workers: worker_0=[A01]" in rendered
    assert "- 0: Count cells axis=A01 groups=None" in rendered
    assert (
        "artifact input positions: kind=special "
        "path=/tmp/example-plate_openhcs/results/A01_positions_step0.pkl "
        "groups=None source_step=0 source_scope=step-find-positions"
    ) in rendered
    assert (
        "artifact cell_counts: kind=special path=/tmp/example-plate_openhcs/results/A01_cell_counts_step0.pkl groups=None"
        in rendered
    )
    assert (
        "materialization: explicit persistent=True backend=disk "
        "analysis_dir=/tmp/example-plate_openhcs/images_results "
        "runtime-metadata-filenames"
    ) in rendered
    assert (
        "candidates group=<none>: "
        "/tmp/example-plate_openhcs/images_results/A01_cell_counts_step0_details.csv"
    ) in rendered


def test_mcp_dev_client_artifact_plan_explains_empty_source_workspace():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "artifact-plan",
            "/tmp/example-plate",
            "--source-text",
            "pipeline_steps = []",
            "--axis-filter",
            "A01",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_inspect_pipeline_source_artifact_plan",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "axes": ["A01"],
                        "axis_count": 1,
                        "axis_filter": ["A01"],
                        "progress_event_count": 1,
                        "step_count": 1,
                        "source_workspace": {
                            "file_count": 0,
                            "truncated_file_count": 0,
                            "axis_file_counts": {"A01": 0},
                            "files": [],
                        },
                        "worker_assignments": {"worker_0": ["A01"]},
                        "steps": [],
                        "warnings": [],
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("artifact-plan").render_response(
        response,
        args,
    )

    assert "Source workspace (source-bound files): files=0 truncated=0" in rendered
    assert (
        "note: no source-bound virtual files were compiled. Standard microscope "
        "input may still be available through the plate handler"
    ) in rendered
    assert "use inspect-plate or selected-plate-images" in rendered


def test_mcp_dev_client_execute_source_composes_session_and_submit(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client
    import openhcs.mcp.dev_client_commands.knowledge_pipeline as knowledge_pipeline

    calls: list[dev_client.McpDevToolCall] = []
    timeouts: list[float] = []

    async def fake_call_tool(session, call, timeout_seconds):
        del session
        calls.append(call)
        timeouts.append(timeout_seconds)
        if call.name == "openhcs_create_orchestrator_session_from_pipeline_source":
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "schema_version": "openhcs.agent.v1",
                        "session_id": "session-1",
                        "uri": "openhcs://execution/sessions/session-1",
                    },
                ),
            )
        return dev_client.McpDevToolResult(
            tool=call.name,
            mcp_error=False,
            payloads=(
                {
                    "schema_version": "openhcs.agent.v1",
                    "session_id": "session-1",
                    "job_id": "job-1",
                    "kind": "execute",
                    "status": "complete",
                    "server_execution_id": "exec-1",
                    "response": {"status": "complete", "completed": True},
                },
            ),
        )

    monkeypatch.setattr(knowledge_pipeline, "call_mcp_tool", fake_call_tool)

    pipeline_source = (
        "from openhcs.core.config import PipelineConfig\n"
        "pipeline_config = PipelineConfig()\n"
        "pipeline_steps = []"
    )
    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "execute-source",
            "/tmp/plate",
            "--source-text",
            pipeline_source,
            "--host",
            "127.0.0.1",
            "--port",
            "5557",
            "--transport-mode",
            "ipc",
            "--wait-timeout-ms",
            "12000",
        )
    )

    response = asyncio.run(
        dev_client.McpDevCommandSpec.for_name("execute-source").run_session(
            SimpleNamespace(server_spec=dev_client.McpDevServerSpec(sys.executable)),
            args,
        )
    )

    assert [call.name for call in calls] == [
        "openhcs_create_orchestrator_session_from_pipeline_source",
        "openhcs_submit_pipeline_execution",
    ]
    assert calls[0].arguments["plate_path"] == "/tmp/plate"
    assert calls[0].arguments["pipeline_source"] == pipeline_source
    assert calls[0].arguments["host"] == "127.0.0.1"
    assert calls[0].arguments["port"] == 5557
    assert calls[0].arguments["transport_mode"] == "ipc"
    assert calls[1].arguments["session_id"] == "session-1"
    assert calls[1].arguments["wait"] is True
    assert calls[1].arguments["wait_timeout_ms"] == 12000
    assert timeouts[1] >= 22

    rendered = dev_client.McpDevCommandSpec.for_name("execute-source").render_response(
        dev_client.to_jsonable(response),
        args,
    )

    assert "Headless source execution:" in rendered
    assert "Session: id=session-1" in rendered
    assert "Job: id=job-1 kind=execute status=complete" in rendered
    assert "Response: status=complete completed=True" in rendered


def test_mcp_dev_client_inspect_plate_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "inspect-plate",
            "/tmp/example-plate",
            "--microscope-type",
            "imagexpress",
            "--pattern-format",
            "auto",
            "--max-sample-files",
            "3",
            "--max-component-values",
            "4",
            "--max-parse-failure-samples",
            "5",
            "--max-files-to-parse",
            "6",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_inspect_plate_path"
    assert call.arguments == {
        "plate_path": "/tmp/example-plate",
        "microscope_type": "imagexpress",
        "pattern_format": "auto",
        "max_sample_files": 3,
        "max_component_values": 4,
        "max_parse_failure_samples": 5,
        "max_files_to_parse": 6,
    }


def test_mcp_dev_client_query_plate_files_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "query-plate-files",
            "/tmp/example-plate",
            "--microscope-type",
            "openhcsdata",
            "--kind",
            "all",
            "--path-contains",
            "A01",
            "--well",
            "A01",
            "--offset",
            "2",
            "--limit",
            "3",
            "--max-preview-lines",
            "4",
            "--max-preview-bytes",
            "512",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_query_plate_files"
    assert call.arguments == {
        "plate_path": "/tmp/example-plate",
        "microscope_type": "openhcsdata",
        "pattern_format": None,
        "kind": "all",
        "path_contains": "A01",
        "well": "A01",
        "offset": 2,
        "limit": 3,
        "include_previews": True,
        "max_preview_lines": 4,
        "max_preview_bytes": 512,
    }

    preview_args = parser.parse_args(
        (
            "query-plate-files",
            "/tmp/example-plate",
            "--no-previews",
            "--include-previews",
        )
    )
    preview_call = dev_client._calls_from_args(preview_args)[0]
    assert preview_call.arguments["include_previews"] is True


def test_mcp_dev_client_generate_synthetic_plate_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "generate-synthetic-plate",
            "/tmp/example-synthetic",
            "--grid-rows",
            "1",
            "--grid-cols",
            "2",
            "--tile-width",
            "64",
            "--tile-height",
            "48",
            "--overlap-percent",
            "10",
            "--wavelengths",
            "2",
            "--well",
            "A01",
            "--random-seed",
            "11",
            "--sample-file-limit",
            "3",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_generate_synthetic_plate"
    assert call.arguments == {
        "output_dir": "/tmp/example-synthetic",
        "grid_rows": 1,
        "grid_cols": 2,
        "tile_width": 64,
        "tile_height": 48,
        "overlap_percent": 10,
        "stage_error_px": 2,
        "wavelengths": 2,
        "z_stack_levels": 1,
        "num_cells": 80,
        "shared_cell_fraction": 0.95,
        "wells": ["A01"],
        "format": "ImageXpress",
        "openhcs_format": False,
        "include_all_components": True,
        "random_seed": 11,
        "sample_file_limit": 3,
    }


def test_mcp_dev_client_generate_synthetic_plate_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("generate-synthetic-plate", "/tmp/example-synthetic"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_generate_synthetic_plate",
                "mcp_error": False,
                "payloads": [
                    {
                        "schema_version": "openhcs.agent.v1",
                        "output_dir": "/tmp/example-synthetic",
                        "requested_format": "ImageXpress",
                        "grid_size": [1, 2],
                        "tile_size": [64, 48],
                        "overlap_percent": 10,
                        "stage_error_px": 2,
                        "wells": ["A01"],
                        "wavelengths": 2,
                        "z_stack_levels": 1,
                        "num_cells": 80,
                        "shared_cell_fraction": 0.95,
                        "image_count": 4,
                        "sampled_image_files": [
                            "TimePoint_1/A01_s1_w1_z1_t1.tif",
                            "TimePoint_1/A01_s2_w2_z1_t1.tif",
                        ],
                        "truncated_image_count": 0,
                        "metadata_file_path": (
                            "/tmp/example-synthetic/example-synthetic.HTD"
                        ),
                        "detected_microscope_type": "imagexpress",
                        "handler_class": "ImageXpressHandler",
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "generate-synthetic-plate"
    ).render_response(response, args)

    assert "Synthetic plate: /tmp/example-synthetic" in rendered
    assert "Geometry: grid=1x2 tile=64x48 overlap=10%" in rendered
    assert "Content: wells=A01 channels=2 z=1" in rendered
    assert "Files: images=4 sampled=2 truncated=0" in rendered
    assert "Metadata: file=/tmp/example-synthetic/example-synthetic.HTD" in rendered
    assert "Next: inspect-plate /tmp/example-synthetic" in rendered


def test_mcp_dev_client_inspect_plate_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("inspect-plate", "/tmp/example-plate"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_inspect_plate_path",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "status": "ok",
                        "confidence": "high",
                        "detected_microscope_type": "openhcsdata",
                        "handler_class": "OpenHCSMicroscopeHandler",
                        "parser_class": "ImageXpressFilenameParser",
                        "image_files": {
                            "count": 2,
                            "sampled_files": [
                                "./A01_s001_w1_z001_t001.tif",
                                "./A01_s001_w2_z001_t001.tif",
                            ],
                            "sampled_records": [
                                {
                                    "virtual_path": "./A01_s001_w1_z001_t001.tif",
                                    "full_virtual_path": "/tmp/example-plate/A01_s001_w1_z001_t001.tif",
                                    "source_path": "/tmp/source-plate/TimePoint_1/A01_s001_w1_z001_t001.tif",
                                    "metadata": {
                                        "well": "A01",
                                        "channel": "1",
                                    },
                                },
                                {
                                    "virtual_path": "./A01_s001_w2_z001_t001.tif",
                                    "full_virtual_path": "/tmp/example-plate/A01_s001_w2_z001_t001.tif",
                                    "source_path": "/tmp/source-plate/TimePoint_1/A01_s001_w2_z001_t001.tif",
                                    "metadata": {
                                        "well": "A01",
                                        "channel": "2",
                                    },
                                },
                            ],
                            "truncated_file_count": 0,
                        },
                        "result_files": {
                            "count": 1,
                            "scanned_file_count": 1,
                            "sampled_files": ["results/A01_counts.csv"],
                            "sampled_records": [
                                {
                                    "relative_path": "results/A01_counts.csv",
                                    "full_path": "/tmp/example-plate/results/A01_counts.csv",
                                    "file_format": "CSV",
                                    "metadata": {"well": "A01"},
                                    "preview": {
                                        "text_lines": [
                                            "slice_index,cell_count",
                                            "0,11",
                                        ],
                                        "csv_columns": [
                                            "slice_index",
                                            "cell_count",
                                        ],
                                        "csv_rows": [
                                            {
                                                "slice_index": "0",
                                                "cell_count": "11",
                                            }
                                        ],
                                        "truncated": False,
                                    },
                                }
                            ],
                            "truncated_file_count": 0,
                        },
                        "parse_summary": {
                            "attempted_file_count": 2,
                            "parsed_file_count": 2,
                            "failed_file_count": 0,
                            "skipped_file_count": 0,
                        },
                        "grid_dimensions": [1, 1],
                        "pixel_size": 0.65,
                        "workspace_preparation": {
                            "operation": "none",
                            "read_only_inspection": True,
                            "required_before_execution": False,
                        },
                        "workflow_advice": {
                            "workflow_scope": "diagnostic",
                            "ingestion_route": "detected_handler",
                            "ingestion_owner": "bioformats",
                            "source_binding_role": "semantic_selection",
                            "ui_code_document_id": "plate_manager.orchestrator_config",
                            "ui_operation": "init",
                            "knowledge_query": (
                                "source model CZI Bio-Formats source bindings"
                            ),
                            "message": (
                                "Keep Bio-Formats as the ingestion owner; source "
                                "bindings select its emitted planes."
                            ),
                        },
                        "format_specific_handler_candidates": [
                            {
                                "microscope_type": "opera_phenix",
                                "handler_class": "OperaPhenixHandler",
                                "parser_class": "OperaPhenixFilenameParser",
                                "root_dir": "Images",
                                "tested_file_count": 3,
                                "recognized_file_count": 3,
                                "recognizes_all_tested_files": True,
                                "files_under_expected_root": True,
                                "metadata_detected": False,
                                "metadata_file_path": None,
                                "metadata_diagnostic": "Index.xml not found",
                            }
                        ],
                        "components": [
                            {
                                "component": "well",
                                "count": 1,
                                "source": "metadata_and_parsed_filenames",
                                "values": [{"key": "A01", "label": "None"}],
                                "truncated_value_count": 0,
                            },
                            {
                                "component": "channel",
                                "count": 2,
                                "source": "metadata_and_parsed_filenames",
                                "values": [
                                    {"key": "1", "label": "W1"},
                                    {"key": "2", "label": "W2"},
                                ],
                                "truncated_value_count": 0,
                            },
                        ],
                        "source_diagnostics": [
                            {
                                "diagnostic_type": (
                                    "bioformats_packed_rgb_series_exclusion"
                                ),
                                "message": "Packed RGB label series was excluded.",
                                "series_index": 7,
                            }
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("inspect-plate").render_response(
        response,
        args,
    )

    assert "Plate: /tmp/example-plate" in rendered
    assert "Status: ok confidence=high microscope=openhcsdata" in rendered
    assert "Images: count=2 sampled=2 truncated=0" in rendered
    assert "Results: count=1 sampled=1 scanned=1 truncated=0" in rendered
    assert (
        "Routing: scope=diagnostic ingestion=detected_handler owner=bioformats "
        "source_bindings=semantic_selection"
    ) in rendered
    assert (
        "UI next: document=plate_manager.orchestrator_config operation=init"
    ) in rendered
    assert "Advice: Keep Bio-Formats as the ingestion owner" in rendered
    assert "Format-specific handler candidates:" in rendered
    assert (
        "opera_phenix parser=OperaPhenixFilenameParser recognized=3/3 "
        "root=Images metadata_detected=False diagnostic=Index.xml not found"
    ) in rendered
    assert (
        "Axis sizes: wells=1 sites=<unknown> channels=2 z=<unknown> "
        "timepoints=<unknown> profile=unknown-site,multi-channel,unknown-z_index,"
        "unknown-timepoint"
    ) in rendered
    assert (
        "Metadata sources: well=metadata_and_parsed_filenames, "
        "channel=metadata_and_parsed_filenames"
    ) in rendered
    assert "- well: count=1 source=metadata_and_parsed_filenames values=A01" in rendered
    assert (
        "- channel: count=2 source=metadata_and_parsed_filenames values=1 (W1), 2 (W2)"
        in rendered
    )
    assert "Source diagnostics: 1" in rendered
    assert (
        "- bioformats_packed_rgb_series_exclusion: Packed RGB label series was excluded."
        in rendered
    )
    assert "Sample records:" in rendered
    assert (
        "- ./A01_s001_w1_z001_t001.tif -> "
        "/tmp/source-plate/TimePoint_1/A01_s001_w1_z001_t001.tif"
    ) in rendered
    assert (
        "Next: sample-plate-image /tmp/example-plate "
        "./A01_s001_w1_z001_t001.tif --height 8 --width 8 --no-array-values"
    ) in rendered
    assert "- results/A01_counts.csv type=CSV" in rendered
    assert "csv columns: slice_index, cell_count" in rendered
    assert "csv row: slice_index=0, cell_count=11" in rendered


def test_mcp_dev_client_query_plate_files_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("query-plate-files", "/tmp/example-plate"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_query_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "handler_class": "OpenHCSMicroscopeHandler",
                        "parser_class": "ImageXpressFilenameParser",
                        "total_count": 2,
                        "returned_count": 2,
                        "offset": 0,
                        "limit": 50,
                        "truncated_count": 0,
                        "records": [
                            {
                                "kind": "image",
                                "key": "./A01_s001_w1_z001_t001.tif",
                                "virtual_path": "./A01_s001_w1_z001_t001.tif",
                                "source_path": "/tmp/source/A01_w1.tif",
                                "metadata": {
                                    "well": "A01",
                                    "modified": "2026-06-27T16:50:37",
                                    "size": "12 KiB",
                                },
                            },
                            {
                                "kind": "result",
                                "key": "images_results/A01_counts.csv",
                                "relative_path": "images_results/A01_counts.csv",
                                "full_path": "/tmp/example-plate/images_results/A01_counts.csv",
                                "file_format": "CSV",
                                "metadata": {
                                    "well": "A01",
                                    "modified": "2026-06-27T15:42:42",
                                    "size": "48 B",
                                },
                                "preview": {
                                    "text_lines": [
                                        "slice_index,cell_count",
                                        "0,11",
                                    ],
                                    "csv_columns": ["slice_index", "cell_count"],
                                    "csv_rows": [
                                        {"slice_index": "0", "cell_count": "11"}
                                    ],
                                    "truncated": False,
                                },
                            },
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "query-plate-files"
    ).render_response(response, args)

    assert "Plate file query: /tmp/example-plate" in rendered
    assert "Result: returned=2 total=2 offset=0 limit=50 truncated=0" in rendered
    assert (
        "Returned records modified: mixed latest=2026-06-27T16:50:37 "
        "earliest=2026-06-27T15:42:42 distinct=2 older_records=1"
    ) in rendered
    assert (
        "Potential stale results: 1 result artifact(s) are older than the "
        "latest image record (2026-06-27T16:50:37); confirm they belong to "
        "the current pipeline/run before using them."
    ) in rendered
    assert (
        "- image ./A01_s001_w1_z001_t001.tif -> /tmp/source/A01_w1.tif "
        "modified=2026-06-27T16:50:37 size=12 KiB"
    ) in rendered
    assert (
        "- result images_results/A01_counts.csv type=CSV -> "
        "/tmp/example-plate/images_results/A01_counts.csv "
        "modified=2026-06-27T15:42:42 size=48 B"
    ) in rendered
    assert "csv columns: slice_index, cell_count" in rendered
    assert "csv row: slice_index=0, cell_count=11" in rendered


def test_mcp_dev_client_query_plate_files_renders_csv_table_after_preamble_preview():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("query-plate-files", "/tmp/example-plate"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_query_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "handler_class": "OpenHCSMicroscopeHandler",
                        "parser_class": None,
                        "total_count": 1,
                        "returned_count": 1,
                        "offset": 0,
                        "limit": 50,
                        "truncated_count": 0,
                        "records": [
                            {
                                "kind": "result",
                                "key": "images_results/metaxpress_style_summary.csv",
                                "relative_path": "images_results/metaxpress_style_summary.csv",
                                "full_path": "/tmp/example-plate/images_results/metaxpress_style_summary.csv",
                                "file_format": "CSV",
                                "preview": {
                                    "text_lines": [
                                        "Barcode,OpenHCS-images_results,,,,,",
                                        "Plate Name,images_results,,,,,",
                                        "Well,Mean Cell Count (W1),Total Cell Count (W1),Mean Cell Count (W2),Total Cell Count (W2)",
                                        "A01,11.0,11,10.0,10",
                                    ],
                                    "csv_columns": [
                                        "Well",
                                        "Mean Cell Count (W1)",
                                        "Total Cell Count (W1)",
                                        "Mean Cell Count (W2)",
                                        "Total Cell Count (W2)",
                                    ],
                                    "csv_rows": [
                                        {
                                            "Well": "A01",
                                            "Mean Cell Count (W1)": "11.0",
                                            "Total Cell Count (W1)": "11",
                                            "Mean Cell Count (W2)": "10.0",
                                            "Total Cell Count (W2)": "10",
                                        }
                                    ],
                                    "truncated": False,
                                },
                            }
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "query-plate-files"
    ).render_response(response, args)

    assert (
        "csv columns: Well, Mean Cell Count (W1), "
        "Total Cell Count (W1), Mean Cell Count (W2), Total Cell Count (W2)"
    ) in rendered
    assert (
        "csv row: Well=A01, Mean Cell Count (W1)=11.0, "
        "Total Cell Count (W1)=11, Mean Cell Count (W2)=10.0, "
        "Total Cell Count (W2)=10"
    ) in rendered


def test_mcp_dev_client_query_plate_files_reports_hidden_csv_preview_rows():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("query-plate-files", "/tmp/example-plate"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_query_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "handler_class": "OpenHCSMicroscopeHandler",
                        "parser_class": "ImageXpressFilenameParser",
                        "total_count": 1,
                        "returned_count": 1,
                        "offset": 0,
                        "limit": 50,
                        "truncated_count": 0,
                        "records": [
                            {
                                "kind": "result",
                                "key": "images_results/A01_counts.csv",
                                "relative_path": "images_results/A01_counts.csv",
                                "full_path": "/tmp/example-plate/images_results/A01_counts.csv",
                                "file_format": "CSV",
                                "preview": {
                                    "csv_columns": ["slice_index", "cell_count"],
                                    "csv_rows": [
                                        {"slice_index": "0", "cell_count": "25"},
                                        {"slice_index": "1", "cell_count": "33"},
                                        {"slice_index": "2", "cell_count": "22"},
                                        {"slice_index": "3", "cell_count": "25"},
                                    ],
                                    "truncated": False,
                                },
                            },
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "query-plate-files"
    ).render_response(response, args)

    assert "csv row: slice_index=0, cell_count=25" in rendered
    assert "csv row: slice_index=2, cell_count=22" in rendered
    assert "csv row: slice_index=3, cell_count=25" not in rendered
    assert "csv preview: showing 3/4 rows; 1 more in payload" in rendered


def test_mcp_dev_client_query_plate_files_prioritizes_compact_csv_cells():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("query-plate-files", "/tmp/example-plate"))
    wide_result = "{" + ("cell_positions: [(1, 2), (3, 4)], " * 20) + "}"
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_query_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "handler_class": "OpenHCSMicroscopeHandler",
                        "parser_class": "ImageXpressFilenameParser",
                        "total_count": 1,
                        "returned_count": 1,
                        "offset": 0,
                        "limit": 50,
                        "truncated_count": 0,
                        "records": [
                            {
                                "kind": "result",
                                "key": "results/A01_counts.csv",
                                "relative_path": "results/A01_counts.csv",
                                "full_path": "/tmp/example-plate/results/A01_counts.csv",
                                "file_format": "CSV",
                                "preview": {
                                    "csv_columns": [
                                        "slice_index",
                                        "chan_1_results",
                                        "colocalized_count",
                                        "chan_1_only_count",
                                        "overlap_positions",
                                    ],
                                    "csv_rows": [
                                        {
                                            "slice_index": "0",
                                            "chan_1_results": wide_result,
                                            "colocalized_count": "13",
                                            "chan_1_only_count": "186",
                                            "overlap_positions": wide_result,
                                        }
                                    ],
                                    "truncated": False,
                                },
                            },
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "query-plate-files"
    ).render_response(response, args)

    assert (
        "csv row: slice_index=0, colocalized_count=13, "
        "chan_1_only_count=186; omitted wide cells: "
        "chan_1_results, overlap_positions"
    ) in rendered
    assert "cell_positions: [(1, 2)" not in rendered


def test_mcp_dev_client_query_plate_files_omits_multiline_csv_cells():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("query-plate-files", "/tmp/example-plate"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_query_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "handler_class": "OpenHCSMicroscopeHandler",
                        "parser_class": "ImageXpressFilenameParser",
                        "total_count": 1,
                        "returned_count": 1,
                        "offset": 0,
                        "limit": 50,
                        "truncated_count": 0,
                        "records": [
                            {
                                "kind": "result",
                                "key": "results/A01_counts.csv",
                                "relative_path": "results/A01_counts.csv",
                                "full_path": "/tmp/example-plate/results/A01_counts.csv",
                                "file_format": "CSV",
                                "preview": {
                                    "csv_columns": [
                                        "slice_index",
                                        "details",
                                        "cell_count",
                                    ],
                                    "csv_rows": [
                                        {
                                            "slice_index": "0",
                                            "details": "first line\nsecond line",
                                            "cell_count": "11",
                                        }
                                    ],
                                    "truncated": False,
                                },
                            },
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "query-plate-files"
    ).render_response(response, args)

    assert (
        "csv row: slice_index=0, cell_count=11; omitted wide cells: details"
    ) in rendered
    assert "first line" not in rendered
    assert "second line" not in rendered


def test_mcp_dev_client_query_plate_files_renders_empty_records():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("query-plate-files", "/tmp/example-plate"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_query_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "detected_microscope_type": "openhcsdata",
                        "total_count": 0,
                        "returned_count": 0,
                        "offset": 0,
                        "limit": 50,
                        "truncated_count": 0,
                        "records": [],
                        "warnings": [
                            {
                                "code": "plate_result_files_available",
                                "message": "No image records, but 3 analysis results were found.",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "query-plate-files"
    ).render_response(response, args)

    assert "Records: <none>" in rendered
    assert (
        "Next: query-plate-files /tmp/example-plate --microscope-type openhcsdata "
        "--kind result --include-previews" in rendered
    )


def test_mcp_dev_client_query_plate_files_renders_next_page_hint():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("query-plate-files", "/tmp/example-plate"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_query_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "total_count": 7,
                        "returned_count": 3,
                        "offset": 0,
                        "limit": 3,
                        "truncated_count": 4,
                        "records": [
                            {
                                "kind": "result",
                                "key": "images_results/A01_counts.csv",
                                "relative_path": "images_results/A01_counts.csv",
                                "full_path": "/tmp/example-plate/images_results/A01_counts.csv",
                                "file_format": "CSV",
                            }
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "query-plate-files"
    ).render_response(response, args)

    assert "Next page: rerun with --offset 3 --limit 3" in rendered


def test_mcp_dev_client_query_plate_files_groups_repeated_warning_messages():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("query-plate-files", "/tmp/example-plate"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_query_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate",
                        "total_count": 1,
                        "returned_count": 1,
                        "offset": 0,
                        "limit": 50,
                        "truncated_count": 0,
                        "records": [
                            {
                                "kind": "result",
                                "key": "images_results/A01_counts.csv",
                                "relative_path": "images_results/A01_counts.csv",
                                "full_path": "/tmp/example-plate/images_results/A01_counts.csv",
                                "file_format": "CSV",
                            }
                        ],
                        "warnings": [
                            {
                                "code": "plate_parser_unavailable",
                                "message": "metadata missing",
                                "hint": "No parser metadata.",
                            },
                            {
                                "code": "plate_image_file_listing_failed",
                                "message": "metadata missing",
                                "hint": "No image metadata.",
                            },
                            {
                                "code": "plate_result_file_listing_failed",
                                "message": "metadata missing",
                                "hint": "Recovered output artifacts.",
                            },
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "query-plate-files"
    ).render_response(response, args)

    assert (
        "- plate_parser_unavailable, plate_image_file_listing_failed, "
        "plate_result_file_listing_failed: metadata missing "
        "hints=3 distinct; pass --json for details"
    ) in rendered
    assert rendered.count("metadata missing") == 1


def test_mcp_dev_client_inspect_plate_renders_result_only_warning_compactly():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("inspect-plate", "/tmp/example-plate-openhcs"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_inspect_plate_path",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/example-plate-openhcs",
                        "status": "partial",
                        "confidence": "none",
                        "detected_microscope_type": None,
                        "handler_class": None,
                        "parser_class": None,
                        "image_files": {
                            "count": 0,
                            "sampled_files": [],
                            "sampled_records": [],
                            "truncated_file_count": 0,
                        },
                        "result_files": {
                            "count": 1,
                            "scanned_file_count": 1,
                            "sampled_records": [
                                {
                                    "relative_path": "images_results/A01_counts.csv",
                                    "full_path": "/tmp/example-plate-openhcs/images_results/A01_counts.csv",
                                    "file_format": "CSV",
                                    "metadata": {
                                        "modified": "2026-06-27T15:42:42",
                                        "size": "48 B",
                                    },
                                    "preview": {
                                        "text_lines": [
                                            "slice_index,cell_count",
                                            "0,11",
                                        ],
                                        "csv_columns": ["slice_index", "cell_count"],
                                        "csv_rows": [
                                            {
                                                "slice_index": "0",
                                                "cell_count": "11",
                                            }
                                        ],
                                    },
                                }
                            ],
                            "truncated_file_count": 0,
                        },
                        "parse_summary": {
                            "attempted_file_count": 0,
                            "parsed_file_count": 0,
                            "failed_file_count": 0,
                            "skipped_file_count": 0,
                        },
                        "workspace_preparation": {
                            "operation": "none",
                            "read_only_inspection": True,
                            "required_before_execution": False,
                        },
                        "components": [],
                        "warnings": [
                            {
                                "code": "plate_handler_detection_failed",
                                "message": (
                                    "Microscope handler detection failed, but "
                                    "OpenHCS analysis result artifacts were found."
                                ),
                                "hint": "metadata missing",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("inspect-plate").render_response(
        response,
        args,
    )

    assert "Status: partial confidence=none microscope=<none>" in rendered
    assert "Results: count=1 sampled=1 scanned=1 truncated=0" in rendered
    assert (
        "- images_results/A01_counts.csv type=CSV "
        "modified=2026-06-27T15:42:42 size=48 B"
    ) in rendered
    assert "Warnings:" in rendered
    assert (
        "- plate_handler_detection_failed: Microscope handler detection failed, "
        'but OpenHCS analysis result artifacts were found. hint="metadata missing"'
    ) in rendered
    assert "{'code':" not in rendered
    assert "'modified':" not in rendered


def test_mcp_dev_client_sample_plate_image_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "sample-plate-image",
            "/tmp/example-plate",
            "A01_s001_w1_z001_t001.tif",
            "--microscope-type",
            "openhcsdata",
            "--pattern-format",
            "auto",
            "--y",
            "1",
            "--x",
            "2",
            "--height",
            "3",
            "--width",
            "4",
            "--resolution-index",
            "0",
            "--max-auto-resolution-size",
            "512",
            "--max-array-elements",
            "12",
            "--no-array-values",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_sample_plate_image"
    assert call.arguments == {
        "plate_path": "/tmp/example-plate",
        "image_path": "A01_s001_w1_z001_t001.tif",
        "microscope_type": "openhcsdata",
        "pattern_format": "auto",
        "y": 1,
        "x": 2,
        "height": 3,
        "width": 4,
        "resolution_index": 0,
        "max_auto_resolution_size": 512,
        "include_array_values": False,
        "max_array_elements": 12,
    }


def test_mcp_dev_client_sample_plate_image_accepts_include_array_values_alias():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    include_args = parser.parse_args(
        (
            "sample-plate-image",
            "/tmp/example-plate",
            "A01_s001_w1_z001_t001.tif",
            "--include-array-values",
        )
    )
    override_args = parser.parse_args(
        (
            "sample-plate-image",
            "/tmp/example-plate",
            "A01_s001_w1_z001_t001.tif",
            "--no-array-values",
            "--include-array-values",
        )
    )

    include_call = dev_client._calls_from_args(include_args)[0]
    override_call = dev_client._calls_from_args(override_args)[0]

    assert include_call.arguments["include_array_values"] is True
    assert override_call.arguments["include_array_values"] is True


def test_mcp_dev_client_stream_plate_files_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "stream-plate-files",
            "/tmp/example-plate-openhcs",
            "images/A01_s001_w1_z001_t001.tif",
            "images_results/A01_w1_segmentation_masks_step0_rois.roi.zip",
            "--viewer-port",
            "5555",
            "--viewer-transport-mode",
            "ipc",
            "--fresh-viewer",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_stream_plate_files_to_viewer"
    assert call.arguments == {
        "file_paths": [
            "images/A01_s001_w1_z001_t001.tif",
            "images_results/A01_w1_segmentation_masks_step0_rois.roi.zip",
        ],
        "microscope_type": "auto",
        "pattern_format": None,
        "kind": "all",
        "path_contains": None,
        "well": None,
        "limit": 1,
        "viewer_config_key": "napari_streaming_config",
        "host": "localhost",
        "port": 5555,
        "transport_mode": "ipc",
        "persistent": True,
        "fresh_viewer": True,
        "plate_path": "/tmp/example-plate-openhcs",
    }

    query_args = parser.parse_args(("stream-plate-files", "/tmp/example-plate-openhcs"))
    query_call = dev_client._calls_from_args(query_args)[0]

    assert query_call.arguments["file_paths"] is None
    assert query_call.arguments["kind"] == "image"

    alias_args = parser.parse_args(
        (
            "stream-plate-files",
            "/tmp/example-plate-openhcs",
            "--host",
            "127.0.0.1",
            "--port",
            "5556",
            "--transport-mode",
            "ipc",
        )
    )
    alias_call = dev_client._calls_from_args(alias_args)[0]

    assert alias_call.arguments["host"] == "127.0.0.1"
    assert alias_call.arguments["port"] == 5556
    assert alias_call.arguments["transport_mode"] == "ipc"


def test_mcp_dev_client_stream_commands_allow_fresh_viewer_startup_time():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    assert (
        dev_client.McpDevCommandSpec.for_name(
            "stream-plate-files"
        ).default_timeout_seconds
        == 60.0
    )
    assert (
        dev_client.McpDevCommandSpec.for_name(
            "selected-plate-stream"
        ).default_timeout_seconds
        == 60.0
    )


def test_mcp_dev_client_sample_plate_image_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "sample-plate-image",
            "/tmp/example-plate",
            "A01_s001_w1_z001_t001.tif",
            "--microscope-type",
            "openhcsdata",
            "--height",
            "2",
            "--width",
            "2",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_sample_plate_image",
                "mcp_error": False,
                "payloads": [
                    {
                        "virtual_path": "images/A01_s001_w1_z001_t001.tif",
                        "source_path": "/tmp/example-plate/images/A01_s001_w1_z001_t001.tif",
                        "shape": [1, 96, 96],
                        "resolution_shape": [1, 24, 24],
                        "dtype": "uint16",
                        "minimum": 0,
                        "maximum": 65535,
                        "mean": 123.4567,
                        "selected_resolution_index": 2,
                        "resolution_count": 3,
                        "downsample_yx": [4.0, 4.0],
                        "statistics_scope": "bounded_sample",
                        "sample_origin_yx": [0, 0],
                        "sample_shape": [1, 2, 2],
                        "sample_included": True,
                        "sample_values": [[[1, 2], [3, 4]]],
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "sample-plate-image"
    ).render_response(response, args)

    assert "Image: images/A01_s001_w1_z001_t001.tif" in rendered
    assert (
        "Resolution: selected=2 count=3 source_shape=1x96x96 "
        "resolution_shape=1x24x24 downsample_yx=4.0x4.0"
    ) in rendered
    assert (
        "Statistics: scope=bounded_sample dtype=uint16 min=0 max=65535 "
        "mean=123.457"
    ) in rendered
    assert "Sample: origin_yx=0x0 shape=1x2x2 included=True" in rendered
    assert "[\n  [\n    [\n      1," in rendered


def test_mcp_dev_client_sample_plate_image_omission_suggests_element_budget():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "sample-plate-image",
            "/tmp/example-plate",
            "A01_s001_w1_z001_t001.tif",
            "--include-array-values",
            "--max-array-elements",
            "40",
            "--height",
            "8",
            "--width",
            "8",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_sample_plate_image",
                "mcp_error": False,
                "payloads": [
                    {
                        "virtual_path": "images/A01_s001_w1_z001_t001.tif",
                        "source_path": "/tmp/example-plate/images/A01_s001_w1_z001_t001.tif",
                        "shape": [1, 96, 96],
                        "dtype": "uint16",
                        "minimum": 0,
                        "maximum": 65535,
                        "mean": 123.4567,
                        "sample_origin_yx": [0, 0],
                        "sample_shape": [1, 8, 8],
                        "sample_included": False,
                        "sample_omitted_reason": "max_array_elements_exceeded",
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "sample-plate-image"
    ).render_response(response, args)

    assert (
        "Sample values omitted: max_array_elements_exceeded; "
        "rerun with --max-array-elements 64 or smaller --width/--height"
    ) in rendered


def test_mcp_dev_client_sample_plate_image_omission_suggests_include_arrays():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "sample-plate-image",
            "/tmp/example-plate",
            "A01_s001_w1_z001_t001.tif",
            "--height",
            "8",
            "--width",
            "8",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_sample_plate_image",
                "mcp_error": False,
                "payloads": [
                    {
                        "virtual_path": "images/A01_s001_w1_z001_t001.tif",
                        "source_path": "/tmp/example-plate/images/A01_s001_w1_z001_t001.tif",
                        "shape": [1, 96, 96],
                        "dtype": "uint16",
                        "minimum": 0,
                        "maximum": 65535,
                        "mean": 123.4567,
                        "sample_origin_yx": [0, 0],
                        "sample_shape": [1, 8, 8],
                        "sample_included": False,
                        "sample_omitted_reason": "array_values_not_requested",
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "sample-plate-image"
    ).render_response(response, args)

    assert (
        "Sample values omitted: array_values_not_requested; "
        "rerun with --include-array-values --max-array-elements 64"
    ) in rendered


def test_mcp_dev_client_selected_plate_files_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-plate-files",
            "--microscope-type",
            "openhcsdata",
            "--kind",
            "all",
            "--target",
            "output",
            "--path-contains",
            "A01",
            "--well",
            "A01",
            "--limit",
            "3",
            "--no-previews",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_query_selected_plate_files"
    assert call.arguments == {
        "microscope_type": "openhcsdata",
        "pattern_format": None,
        "kind": "all",
        "target": "output",
        "path_contains": "A01",
        "well": "A01",
        "offset": 0,
        "limit": 3,
        "include_previews": False,
        "max_preview_lines": 8,
        "max_preview_bytes": 64 * 1024,
        "connection": {"timeout_ms": 1234},
    }

    preview_args = parser.parse_args(
        (
            "selected-plate-files",
            "--no-previews",
            "--include-previews",
        )
    )
    preview_call = dev_client._calls_from_args(preview_args)[0]
    assert preview_call.arguments["include_previews"] is True


def test_mcp_dev_client_selected_plate_files_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("selected-plate-files",))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_query_selected_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "selected_plate": {
                            "name": "selected-plate",
                            "plate_root": "/tmp/selected-plate",
                        },
                        "target": "selected",
                        "query": {
                            "plate_path": "/tmp/selected-plate",
                            "handler_class": "OpenHCSMicroscopeHandler",
                            "parser_class": "ImageXpressFilenameParser",
                            "total_count": 1,
                            "returned_count": 1,
                            "offset": 0,
                            "limit": 50,
                            "truncated_count": 0,
                            "records": [
                                {
                                    "kind": "image",
                                    "key": "./A01_s001_w1_z001_t001.tif",
                                    "virtual_path": "./A01_s001_w1_z001_t001.tif",
                                    "source_path": "/tmp/source/A01_w1.tif",
                                    "metadata": {"well": "A01"},
                                }
                            ],
                            "warnings": [],
                        },
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "selected-plate-files"
    ).render_response(response, args)

    assert (
        "Selected plate: selected-plate root=/tmp/selected-plate target=selected"
        in rendered
    )
    assert "Plate file query: /tmp/selected-plate" in rendered
    assert "Result: returned=1 total=1 offset=0 limit=50 truncated=0" in rendered
    assert "- image ./A01_s001_w1_z001_t001.tif -> /tmp/source/A01_w1.tif" in rendered


def test_mcp_dev_client_selected_plate_files_hints_related_output_for_empty_results():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("selected-plate-files", "--kind", "result"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_query_selected_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "selected_plate": {
                            "name": "selected-plate",
                            "plate_root": "/tmp/selected-plate",
                            "output_plate_root": "/tmp/selected-plate_openhcs",
                        },
                        "target": "selected",
                        "query": {
                            "plate_path": "/tmp/selected-plate",
                            "total_count": 0,
                            "returned_count": 0,
                            "offset": 0,
                            "limit": 50,
                            "truncated_count": 0,
                            "records": [],
                            "warnings": [],
                        },
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "selected-plate-files"
    ).render_response(response, args)

    assert "Records: <none>" in rendered
    assert "Related output: /tmp/selected-plate_openhcs" in rendered
    assert "Next: selected-plate-files --target output --kind result" in rendered


def test_mcp_dev_client_selected_plate_files_renders_materialized_record_root():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("selected-plate-files", "--kind", "result"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_query_selected_plate_files",
                "mcp_error": False,
                "payloads": [
                    {
                        "selected_plate": {
                            "name": "selected-plate",
                            "plate_root": "/tmp/selected-plate",
                            "output_plate_root": "/tmp/selected-plate_openhcs",
                        },
                        "target": "selected",
                        "query": {
                            "plate_path": "/tmp/selected-plate",
                            "total_count": 1,
                            "returned_count": 1,
                            "offset": 0,
                            "limit": 50,
                            "truncated_count": 0,
                            "records": [
                                {
                                    "kind": "result",
                                    "key": "images_results/A01_summary.csv",
                                    "relative_path": "images_results/A01_summary.csv",
                                    "full_path": (
                                        "/tmp/selected-plate_openhcs/"
                                        "images_results/A01_summary.csv"
                                    ),
                                    "file_format": "CSV",
                                    "metadata": {},
                                }
                            ],
                            "warnings": [],
                        },
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "selected-plate-files"
    ).render_response(response, args)

    assert "Plate file query: /tmp/selected-plate" in rendered
    assert (
        "Record file roots: /tmp/selected-plate_openhcs "
        "(differs from query root; inventory may expose materialized outputs)"
    ) in rendered


def test_mcp_dev_client_selected_plate_images_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-plate-images",
            "--microscope-type",
            "openhcsdata",
            "--target",
            "output",
            "--max-sample-files",
            "3",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_inspect_selected_plate_images"
    assert call.arguments == {
        "microscope_type": "openhcsdata",
        "pattern_format": None,
        "target": "output",
        "max_sample_files": 3,
        "max_component_values": 25,
        "max_parse_failure_samples": 10,
        "max_files_to_parse": 50_000,
        "connection": {"timeout_ms": 1234},
    }


def test_mcp_dev_client_selected_plate_images_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("selected-plate-images",))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_inspect_selected_plate_images",
                "mcp_error": False,
                "payloads": [
                    {
                        "selected_plate": {
                            "name": "selected-plate",
                            "plate_root": "/tmp/selected-plate",
                        },
                        "target": "selected",
                        "inspection": {
                            "plate_path": "/tmp/selected-plate",
                            "status": "ok",
                            "confidence": "high",
                            "detected_microscope_type": "openhcsdata",
                            "handler_class": "OpenHCSMicroscopeHandler",
                            "parser_class": "ImageXpressFilenameParser",
                            "image_files": {
                                "count": 2,
                                "sampled_files": [
                                    "checkpoints_step1/A01_s001_w1_z001_t001.tif",
                                    "images/A01_s001_w1_z001_t001.tif",
                                ],
                                "sampled_records": [
                                    {
                                        "virtual_path": "checkpoints_step1/A01_s001_w1_z001_t001.tif",
                                        "full_virtual_path": (
                                            "/tmp/selected-plate/"
                                            "checkpoints_step1/A01_s001_w1_z001_t001.tif"
                                        ),
                                        "source_path": (
                                            "/tmp/selected-plate/"
                                            "checkpoints_step1/A01_s001_w1_z001_t001.tif"
                                        ),
                                        "metadata": {
                                            "well": "A01",
                                            "modified": "2026-06-27T14:30:06",
                                            "size": "12 KiB",
                                        },
                                    },
                                    {
                                        "virtual_path": "images/A01_s001_w1_z001_t001.tif",
                                        "full_virtual_path": (
                                            "/tmp/selected-plate/images/A01_s001_w1_z001_t001.tif"
                                        ),
                                        "source_path": (
                                            "/tmp/selected-plate/"
                                            "images/A01_s001_w1_z001_t001.tif"
                                        ),
                                        "metadata": {
                                            "well": "A01",
                                            "modified": "2026-06-27T16:50:37",
                                            "size": "12 KiB",
                                        },
                                    },
                                ],
                                "truncated_file_count": 0,
                            },
                            "result_files": {
                                "count": 1,
                                "scanned_file_count": 1,
                                "sampled_files": ["results/A01.roi.zip"],
                                "sampled_records": [
                                    {
                                        "relative_path": "results/A01.roi.zip",
                                        "full_path": "/tmp/selected-plate/results/A01.roi.zip",
                                        "file_format": "ROI",
                                        "metadata": {
                                            "well": "A01",
                                            "modified": "2026-06-27T15:42:42",
                                            "size": "48 B",
                                        },
                                        "preview": {
                                            "roi_count": 2,
                                            "roi_member_count": 3,
                                            "roi_duplicate_member_count": 1,
                                            "roi_area_min": 4.0,
                                            "roi_area_mean": 5.5,
                                            "roi_area_max": 7.0,
                                            "roi_examples": [
                                                {
                                                    "label": 1,
                                                    "area": 4.0,
                                                    "bbox": [0, 0, 2, 2],
                                                    "centroid": [1.0, 1.0],
                                                }
                                            ],
                                            "truncated": True,
                                        },
                                    }
                                ],
                                "truncated_file_count": 0,
                            },
                            "parse_summary": {
                                "attempted_file_count": 1,
                                "parsed_file_count": 1,
                                "failed_file_count": 0,
                                "skipped_file_count": 0,
                            },
                            "workspace_preparation": {
                                "operation": "none",
                                "read_only_inspection": True,
                                "required_before_execution": False,
                            },
                            "components": [],
                            "warnings": [],
                            "errors": [],
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "selected-plate-images"
    ).render_response(response, args)

    assert (
        "Selected plate: selected-plate root=/tmp/selected-plate target=selected"
        in rendered
    )
    assert "Plate: /tmp/selected-plate" in rendered
    assert "Images: count=2 sampled=2 truncated=0" in rendered
    assert "Results: count=1 sampled=1 scanned=1 truncated=0" in rendered
    assert (
        "Sampled artifacts modified: mixed latest=2026-06-27T16:50:37 "
        "earliest=2026-06-27T14:30:06 distinct=3 older_records=2"
    ) in rendered
    assert (
        "- images/A01_s001_w1_z001_t001.tif modified=2026-06-27T16:50:37 size=12 KiB"
    ) in rendered
    assert (
        "- results/A01.roi.zip type=ROI modified=2026-06-27T15:42:42 size=48 B"
    ) in rendered
    assert (
        "roi preview: count=2 members=3 duplicate_members=1 "
        "area=min=4.0,mean=5.500,max=7.0"
    ) in rendered
    assert (
        "roi example: label=1 area=4.0 bbox=[0, 0, 2, 2] centroid=[1.0, 1.0]"
        in rendered
    )
    assert (
        "Next: selected-plate-sample images/A01_s001_w1_z001_t001.tif "
        "--height 8 --width 8 --no-array-values" in rendered
    )


def test_mcp_dev_client_selected_plate_sample_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-plate-sample",
            "./A01_s001_w1_z001_t001.tif",
            "--microscope-type",
            "openhcsdata",
            "--target",
            "output",
            "--height",
            "2",
            "--width",
            "3",
            "--no-array-values",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_sample_selected_plate_image"
    assert call.arguments == {
        "image_path": "./A01_s001_w1_z001_t001.tif",
        "microscope_type": "openhcsdata",
        "pattern_format": None,
        "target": "output",
        "y": 0,
        "x": 0,
        "height": 2,
        "width": 3,
        "resolution_index": None,
        "max_auto_resolution_size": 1024,
        "include_array_values": False,
        "max_array_elements": 4096,
        "connection": {"timeout_ms": 1234},
    }


def test_mcp_dev_client_selected_plate_sample_accepts_include_array_values_alias():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-plate-sample",
            "./A01_s001_w1_z001_t001.tif",
            "--no-array-values",
            "--include-array-values",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_sample_selected_plate_image"
    assert call.arguments["include_array_values"] is True


def test_mcp_dev_client_selected_plate_stream_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-plate-stream",
            "images/A01_s001_w1_z001_t001.tif",
            "images_results/A01_w1_segmentation_masks_step0_rois.roi.zip",
            "--target",
            "output",
            "--viewer-port",
            "5555",
            "--viewer-transport-mode",
            "ipc",
            "--fresh-viewer",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_stream_selected_plate_files_to_viewer"
    assert call.arguments == {
        "file_paths": [
            "images/A01_s001_w1_z001_t001.tif",
            "images_results/A01_w1_segmentation_masks_step0_rois.roi.zip",
        ],
        "microscope_type": "auto",
        "pattern_format": None,
        "kind": "all",
        "target": "output",
        "path_contains": None,
        "well": None,
        "limit": 1,
        "viewer_config_key": "napari_streaming_config",
        "host": "localhost",
        "port": 5555,
        "transport_mode": "ipc",
        "persistent": True,
        "fresh_viewer": True,
        "connection": {"timeout_ms": 1234},
    }

    query_args = parser.parse_args(("selected-plate-stream", "--target", "output"))
    query_call = dev_client._calls_from_args(query_args)[0]

    assert query_call.arguments["file_paths"] is None
    assert query_call.arguments["kind"] == "image"


def test_mcp_dev_client_selected_plate_sample_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("selected-plate-sample",))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_sample_selected_plate_image",
                "mcp_error": False,
                "payloads": [
                    {
                        "selected_plate": {
                            "name": "selected-plate",
                            "plate_root": "/tmp/selected-plate",
                        },
                        "target": "selected",
                        "image_path": "./A01_s001_w1_z001_t001.tif",
                        "auto_selected_image_path": True,
                        "sample": {
                            "virtual_path": "./A01_s001_w1_z001_t001.tif",
                            "source_path": "/tmp/source/A01_w1.tif",
                            "shape": [1, 2, 2],
                            "resolution_shape": [1, 2, 2],
                            "dtype": "uint16",
                            "minimum": 1,
                            "maximum": 4,
                            "mean": 2.5,
                            "selected_resolution_index": 0,
                            "resolution_count": 1,
                            "downsample_yx": [1.0, 1.0],
                            "statistics_scope": "source_resolution",
                            "sample_origin_yx": [0, 0],
                            "sample_shape": [1, 2, 2],
                            "sample_included": True,
                            "sample_values": [[[1, 2], [3, 4]]],
                            "errors": [],
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "selected-plate-sample"
    ).render_response(response, args)

    assert (
        "Selected plate: selected-plate root=/tmp/selected-plate target=selected"
        in rendered
    )
    assert "Selected image: ./A01_s001_w1_z001_t001.tif auto=True" in rendered
    assert "Image: ./A01_s001_w1_z001_t001.tif" in rendered
    assert (
        "Resolution: selected=0 count=1 source_shape=1x2x2 "
        "resolution_shape=1x2x2 downsample_yx=1.0x1.0"
    ) in rendered
    assert (
        "Statistics: scope=source_resolution dtype=uint16 min=1 max=4 mean=2.500"
        in rendered
    )
    assert "[\n  [\n    [\n      1," in rendered


def test_mcp_dev_client_selected_plate_sample_omission_suggests_element_budget():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("selected-plate-sample", "--target", "output"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_sample_selected_plate_image",
                "mcp_error": False,
                "payloads": [
                    {
                        "selected_plate": {
                            "name": "selected-plate",
                            "plate_root": "/tmp/selected-plate",
                        },
                        "target": "output",
                        "image_path": "./A01_s001_w1_z001_t001.tif",
                        "auto_selected_image_path": False,
                        "sample": {
                            "virtual_path": "./A01_s001_w1_z001_t001.tif",
                            "source_path": "/tmp/source/A01_w1.tif",
                            "shape": [1, 96, 96],
                            "dtype": "uint16",
                            "minimum": 0,
                            "maximum": 65535,
                            "mean": 12.0,
                            "sample_origin_yx": [0, 0],
                            "sample_shape": [1, 16, 16],
                            "sample_included": False,
                            "sample_omitted_reason": (
                                "sample has 256 elements, above max_array_elements=40"
                            ),
                            "errors": [],
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "selected-plate-sample"
    ).render_response(response, args)

    assert (
        "Selected plate: selected-plate root=/tmp/selected-plate target=output"
        in rendered
    )
    assert (
        "Sample values omitted: sample has 256 elements, above max_array_elements=40; "
        "rerun with --max-array-elements 256 or smaller --width/--height"
    ) in rendered


def test_mcp_dev_client_selected_plate_sample_error_keeps_target_context():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("selected-plate-sample", "--target", "output"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_sample_selected_plate_image",
                "mcp_error": False,
                "payloads": [
                    {
                        "selected_plate": {
                            "name": "selected-plate",
                            "plate_root": "/tmp/selected-plate",
                        },
                        "target": "output",
                        "errors": [
                            {
                                "code": "ui_selected_plate_no_sample_image",
                                "message": "The selected plate did not report any image files to sample.",
                            }
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "selected-plate-sample"
    ).render_response(response, args)

    assert "Selected plate sample: failed" in rendered
    assert (
        "Selected plate: selected-plate root=/tmp/selected-plate target=output"
        in rendered
    )
    assert "ui_selected_plate_no_sample_image" in rendered


def test_mcp_dev_client_call_renders_selected_plate_stream_compactly():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "call",
            "openhcs_ui_stream_selected_plate_files_to_viewer",
            "--arguments",
            '{"target":"output","kind":"all","limit":2}',
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_stream_selected_plate_files_to_viewer",
                "mcp_error": False,
                "payloads": [
                    {
                        "selected_plate": {
                            "name": "selected-plate",
                            "plate_root": "/tmp/selected-plate",
                            "output_plate_root": "/tmp/selected-plate_openhcs",
                        },
                        "target": "output",
                        "stream": {
                            "plate_path": "/tmp/selected-plate_openhcs",
                            "viewer_type": "napari",
                            "viewer_config_key": "napari_streaming_config",
                            "connection": {
                                "host": "localhost",
                                "port": 5555,
                                "transport_mode": "ipc",
                                "persistent": True,
                            },
                            "requested_paths": [
                                "images/A01_s001_w1_z001_t001.tif",
                                "results/A01_s001.roi.zip",
                            ],
                            "resolved_records": [
                                {
                                    "kind": "image",
                                    "key": "images/A01_s001_w1_z001_t001.tif",
                                    "source_path": "/tmp/source/A01_w1.tif",
                                },
                                {
                                    "kind": "result",
                                    "key": "results/A01_s001.roi.zip",
                                    "full_path": (
                                        "/tmp/selected-plate_openhcs/"
                                        "results/A01_s001.roi.zip"
                                    ),
                                    "file_format": "ROI",
                                },
                            ],
                            "skipped_records": [],
                            "streamed_image_paths": ["/tmp/source/A01_w1.tif"],
                            "streamed_roi_paths": [
                                "/tmp/selected-plate_openhcs/results/A01_s001.roi.zip"
                            ],
                            "handler_class": "OpenHCSMicroscopeHandler",
                            "parser_class": "ImageXpressFilenameParser",
                            "status_messages": ["streamed 2 files to napari"],
                            "errors": [],
                            "warnings": [],
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("call").render_response(
        response,
        args,
    )

    assert (
        "Selected plate: selected-plate root=/tmp/selected-plate target=output"
        in rendered
    )
    assert "Plate file stream: /tmp/selected-plate_openhcs" in rendered
    assert "Viewer: napari config=napari_streaming_config" in rendered
    assert "Files: requested=2 resolved=2 images=1 rois=1 skipped=0" in rendered
    assert "Images:\n- /tmp/source/A01_w1.tif" in rendered
    assert "ROIs:\n- /tmp/selected-plate_openhcs/results/A01_s001.roi.zip" in rendered
    assert "Status:\n- streamed 2 files to napari" in rendered
    assert (
        "Next:\n"
        "- validate-viewer --port 5555 --transport-mode ipc "
        "--require-nonzero-payloads\n"
        "- viewer-state --port 5555 --transport-mode ipc\n"
        "- viewer-rois --port 5555 --transport-mode ipc --limit 5"
    ) in rendered
    assert '"stream"' not in rendered


def test_mcp_dev_client_runtime_scan_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "runtime-scan",
            "5555",
            "7777",
            "--host",
            "127.0.0.1",
            "--timeout-ms",
            "300",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_scan_runtime_servers"
    assert call.arguments["ports"] == [5555, 7777]
    assert call.arguments["host"] == "127.0.0.1"
    assert call.arguments["timeout_ms"] == 300


def test_mcp_dev_client_runtime_scan_accepts_comma_separated_ports():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("runtime-scan", "5555,5565", "7777"))

    call = dev_client._calls_from_args(args)[0]

    assert call.arguments["ports"] == [5555, 5565, 7777]


def test_mcp_dev_client_runtime_scan_accepts_ports_alias():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "runtime-scan",
            "4444",
            "--ports",
            "5555,5565",
            "--ports",
            "7777",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.arguments["ports"] == [4444, 5555, 5565, 7777]


def test_mcp_dev_client_runtime_scan_command_renders_endpoint_kinds():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("runtime-scan", "5555", "7777"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_scan_runtime_servers",
                "mcp_error": False,
                "payloads": [
                    {
                        "ports": [5555, 7777],
                        "timeout_ms": 300,
                        "servers": [
                            {
                                "connection": {"port": 5555},
                                "server": "NapariViewer",
                                "reachable": True,
                                "ready": True,
                                "control_port": 6555,
                                "active_executions": None,
                                "running_executions": [],
                                "queued_executions": [],
                                "workers": [],
                                "uptime": None,
                                "log_file_path": "/tmp/napari.log",
                                "errors": [],
                            },
                            {
                                "connection": {"port": 7777},
                                "server": "ZMQExecutionServer",
                                "reachable": True,
                                "ready": True,
                                "control_port": 8777,
                                "active_executions": 1,
                                "running_executions": [{"id": "run-1"}],
                                "queued_executions": [],
                                "workers": [{"id": "worker-1"}],
                                "uptime": 12.34,
                                "log_file_path": "/tmp/exec.log",
                                "errors": [],
                            },
                        ],
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("runtime-scan").render_response(
        response,
        args,
    )

    assert "Runtime scan: ports=5555,7777 timeout_ms=300 servers=2" in rendered
    assert (
        "port=5555 server=NapariViewer reachable=True ready=True control=6555"
        in rendered
    )
    assert (
        "port=7777 server=ZMQExecutionServer reachable=True ready=True control=8777"
        in rendered
    )
    assert (
        "active=1 running=1 queued=0 workers=1 uptime=12.3s log=/tmp/exec.log"
        in rendered
    )


def test_mcp_dev_client_runtime_info_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "runtime-info",
            "7777",
            "--host",
            "127.0.0.1",
            "--non-persistent",
            "--timeout-ms",
            "600",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_get_runtime_server_info"
    assert call.arguments["port"] == 7777
    assert call.arguments["host"] == "127.0.0.1"
    assert call.arguments["persistent"] is False
    assert call.arguments["timeout_ms"] == 600


def test_mcp_dev_client_runtime_status_command_renders_execution_counts():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("runtime-status", "7777", "--execution-id", "run-1"))
    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_get_runtime_server_execution_status"
    assert call.arguments["port"] == 7777
    assert call.arguments["execution_id"] == "run-1"
    assert call.arguments["timeout_ms"] == 500

    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_get_runtime_server_execution_status",
                "mcp_error": False,
                "payloads": [
                    {
                        "connection": {"port": 7777},
                        "execution_id": "run-1",
                        "status": "ok",
                        "response": {
                            "active_executions": 1,
                            "executions": ["run-1", "run-2"],
                            "running_executions": [{"id": "run-1"}],
                            "queued_executions": [],
                            "uptime": 12.34,
                        },
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("runtime-status").render_response(
        response,
        args,
    )

    assert (
        "Runtime execution status: status=ok execution_id=run-1 port=7777" in rendered
    )
    assert "Executions: known=2 active=1 running=1 queued=0 uptime=12.3s" in rendered


def test_mcp_dev_client_ui_commands_project_connection_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "window-snapshot",
            "plate_manager",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_snapshot_window"
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }
    assert "timeout_ms" not in call.arguments

    workflow_args = parser.parse_args(
        (
            "selected-workflow",
            "init_plate",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    workflow_call = dev_client._calls_from_args(workflow_args)[0]

    assert workflow_call.name == "openhcs_ui_selected_plate_workflow"
    assert workflow_call.arguments["workflow"] == "init_plate"
    assert workflow_call.arguments["require_confirmation"] is False
    assert workflow_call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    state_surfaces_args = parser.parse_args(
        (
            "state-surfaces",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    state_surfaces_call = dev_client._calls_from_args(state_surfaces_args)[0]

    assert state_surfaces_call.name == "openhcs_ui_list_state_surfaces"
    assert state_surfaces_call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    state_args = parser.parse_args(
        (
            "state-surface",
            "plate_manager.state",
            "--selection-mode",
            "selected",
            "--base-revision-token",
            "rev-1",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    state_call = dev_client._calls_from_args(state_args)[0]

    assert state_call.name == "openhcs_ui_get_state_surface"
    assert state_call.arguments["surface_id"] == "plate_manager.state"
    assert state_call.arguments["selection_mode"] == "selected"
    assert state_call.arguments["base_revision_token"] == "rev-1"
    assert state_call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    widget_args = parser.parse_args(
        (
            "widget-tree",
            "plate_manager",
            "--create-if-missing",
            "--full-actions",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    widget_call = dev_client._calls_from_args(widget_args)[0]

    assert widget_call.name == "openhcs_ui_get_widget_tree"
    assert widget_call.arguments["window_id"] == "plate_manager"
    assert widget_call.arguments["create_if_missing"] is True
    assert widget_call.arguments["compact_actions"] is False
    assert widget_call.arguments["maximum_item_model_nodes"] == 512
    assert widget_call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    default_widget_args = parser.parse_args(("widget-tree", "plate_manager"))
    default_widget_call = dev_client._calls_from_args(default_widget_args)[0]

    assert default_widget_call.name == "openhcs_ui_get_widget_tree"
    assert default_widget_call.arguments["compact_actions"] is True
    assert default_widget_call.arguments["include_tree"] is True
    assert default_widget_call.arguments["actionable_only"] is False
    assert default_widget_call.arguments["max_depth"] == 8
    assert default_widget_call.arguments["max_nodes"] == 800

    json_widget_args = parser.parse_args(
        ("widget-tree", "plate_manager", "--output", "json")
    )
    json_widget_call = dev_client._calls_from_args(json_widget_args)[0]

    assert json_widget_call.arguments["include_tree"] is False
    assert json_widget_call.arguments["actionable_only"] is True
    assert json_widget_call.arguments["max_nodes"] == 40

    json_alias_widget_args = parser.parse_args(
        ("widget-tree", "plate_manager", "--json")
    )
    json_alias_widget_call = dev_client._calls_from_args(json_alias_widget_args)[0]

    assert json_alias_widget_args.output == "json"
    assert json_alias_widget_call.arguments["include_tree"] is False
    assert json_alias_widget_call.arguments["actionable_only"] is True

    actionable_widget_args = parser.parse_args(
        ("widget-tree", "plate_manager", "--actionable-only")
    )
    actionable_widget_call = dev_client._calls_from_args(actionable_widget_args)[0]

    assert actionable_widget_call.arguments["include_tree"] is True
    assert actionable_widget_call.arguments["actionable_only"] is True

    outline_widget_args = parser.parse_args(
        ("widget-tree", "global_config", "--output", "outline")
    )
    outline_widget_call = dev_client._calls_from_args(outline_widget_args)[0]

    assert outline_widget_call.arguments["include_tree"] is True
    assert outline_widget_call.arguments["actionable_only"] is False
    assert outline_widget_call.arguments["max_depth"] == 8
    assert outline_widget_call.arguments["max_nodes"] == 800

    rendered = dev_client.McpDevCommandSpec.for_name("widget-tree").render_response(
        {
            "errors": [],
            "results": [
                {
                    "tool": "openhcs_ui_get_widget_tree",
                    "mcp_error": False,
                    "payloads": [
                        {
                            "summary": {
                                "title": "* Configuration - GlobalPipelineConfig",
                                "dirty": True,
                                "dirty_field_count": 7,
                                "signature_diff": True,
                                "signature_diff_field_count": 2,
                                "semantic_markers": ["*", "_"],
                            },
                            "root": {
                                "class_name": "ConfigWindow",
                                "children": [
                                    {
                                        "class_name": "QTreeWidget",
                                        "current_text": (
                                            "selected plate with a very long "
                                            "configuration preview that should be "
                                            "shortened before it reaches the "
                                            "terminal renderer output"
                                        ),
                                        "children": [
                                            {
                                                "class_name": "QModelIndex",
                                                "text": "* WellFilter",
                                                "children": [],
                                            }
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
            "server": {"command": "python", "module": "openhcs.mcp"},
        },
        outline_widget_args,
    )

    assert "Window: * Configuration - GlobalPipelineConfig" in rendered
    assert 'QModelIndex "* WellFilter"' in rendered
    assert "shortened before it reaches the terminal renderer output" not in rendered
    assert 'current="selected plate with a very long configuration preview' in rendered

    outline_subtree_args = parser.parse_args(
        (
            "widget-tree",
            "global_config",
            "--output",
            "outline",
            "--outline-root-class",
            "QTreeWidget",
        )
    )
    rendered_subtree = dev_client.McpDevCommandSpec.for_name(
        "widget-tree"
    ).render_response(
        {
            "errors": [],
            "results": [
                {
                    "tool": "openhcs_ui_get_widget_tree",
                    "mcp_error": False,
                    "payloads": [
                        {
                            "summary": {"title": "Configuration"},
                            "root": {
                                "class_name": "ConfigWindow",
                                "children": [
                                    {"class_name": "QLabel", "text": "noise"},
                                    {
                                        "class_name": "QTreeWidget",
                                        "children": [
                                            {
                                                "class_name": "QModelIndex",
                                                "text": "WellFilter",
                                                "children": [],
                                            },
                                            {
                                                "class_name": "QScrollBar",
                                                "children": [],
                                            },
                                        ],
                                    },
                                ],
                            },
                        }
                    ],
                }
            ],
            "server": {"command": "python", "module": "openhcs.mcp"},
        },
        outline_subtree_args,
    )

    assert "QTreeWidget" in rendered_subtree
    assert 'QModelIndex "WellFilter"' in rendered_subtree
    assert "QLabel" not in rendered_subtree
    assert "QScrollBar" not in rendered_subtree

    rendered_semantic_row = dev_client.McpDevCommandSpec.for_name(
        "widget-tree"
    ).render_response(
        {
            "errors": [],
            "results": [
                {
                    "tool": "openhcs_ui_get_widget_tree",
                    "mcp_error": False,
                    "payloads": [
                        {
                            "summary": {"title": "Plate Manager"},
                            "root": {
                                "class_name": "PlateManagerWidget",
                                "children": [
                                    {
                                        "class_name": "QModelIndex",
                                        "text": (
                                            "✅ Completeplate | /tmp/plate | root | "
                                            "{ | num_workers:1 | } | noisy config preview"
                                        ),
                                        "current_text": (
                                            "✅ Completeplate | /tmp/plate | root | "
                                            "{ | num_workers:1 | } | noisy current preview"
                                        ),
                                        "path_id": "1.1.3",
                                        "actionable": True,
                                        "children": [],
                                    }
                                ],
                            },
                            "actionable_widgets": [
                                {
                                    "path_id": "1.1.3",
                                    "object_state_scope_id": "/tmp/plate",
                                    "field_path": "napari_streaming_config.enabled",
                                    "semantic_markers": ["*", "_"],
                                }
                            ],
                        }
                    ],
                }
            ],
            "server": {"command": "python", "module": "openhcs.mcp"},
        },
        outline_widget_args,
    )

    assert "QModelIndex path=1.1.3 [*_] scope=/tmp/plate" in rendered_semantic_row
    assert "field=napari_streaming_config.enabled" in rendered_semantic_row
    assert "noisy config preview" not in rendered_semantic_row
    assert "noisy current preview" not in rendered_semantic_row


def test_mcp_dev_client_widget_tree_outline_shows_disabled_action_paths():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("widget-tree", "pipeline_editor", "--output", "outline"))
    rendered = dev_client.McpDevCommandSpec.for_name("widget-tree").render_response(
        {
            "errors": [],
            "results": [
                {
                    "tool": "openhcs_ui_get_widget_tree",
                    "mcp_error": False,
                    "payloads": [
                        {
                            "summary": {"title": "Pipeline Editor"},
                            "root": {
                                "class_name": "PipelineEditorWidget",
                                "children": [
                                    {
                                        "class_name": "QPushButton",
                                        "text": "Edit",
                                        "path_id": "1.0.2",
                                        "action_kinds": ["button"],
                                        "visible": True,
                                        "enabled": False,
                                        "clickable": False,
                                        "actionable": False,
                                        "children": [],
                                    },
                                    {
                                        "class_name": "QPushButton",
                                        "text": "Generate",
                                        "path_id": "1.0.3",
                                        "action_kinds": ["button"],
                                        "visible": False,
                                        "enabled": False,
                                        "clickable": False,
                                        "actionable": False,
                                        "children": [],
                                    },
                                ],
                            },
                            "actionable_widgets": [],
                        }
                    ],
                }
            ],
            "server": {"command": "python", "module": "openhcs.mcp"},
        },
        args,
    )

    assert 'QPushButton "Edit" path=1.0.2 disabled' in rendered
    assert "not-clickable" not in rendered
    assert "Generate" not in rendered


def test_mcp_dev_client_call_renders_widget_tree_outline_for_pipeline_editor():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "call",
            "openhcs_ui_get_widget_tree",
            "--arguments",
            '{"window_id":"pipeline_editor","include_tree":true}',
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_get_widget_tree",
                "mcp_error": False,
                "payloads": [
                    {
                        "summary": {
                            "title": "Pipeline Editor",
                            "dirty": False,
                            "dirty_field_count": 0,
                            "signature_diff": True,
                            "signature_diff_field_count": 1,
                            "semantic_markers": ["_"],
                        },
                        "root": {
                            "class_name": "PipelineEditorWidget",
                            "children": [
                                {
                                    "class_name": "QPushButton",
                                    "text": "Compile",
                                    "path_id": "0.1",
                                    "actionable": True,
                                    "children": [],
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("call").render_response(
        response,
        args,
    )

    assert "Window: Pipeline Editor" in rendered
    assert "Status: dirty=False dirty_fields=0 default_diff=True" in rendered
    assert "PipelineEditorWidget" in rendered
    assert 'QPushButton "Compile" path=0.1' in rendered
    assert '"results"' not in rendered

    json_args = parser.parse_args(
        (
            "call",
            "openhcs_ui_get_widget_tree",
            "--arguments",
            '{"window_id":"pipeline_editor","include_tree":true}',
            "--json",
        )
    )
    raw_rendered = dev_client.McpDevCommandSpec.for_name("call").render_response(
        response,
        json_args,
    )

    assert '"results": [' in raw_rendered


def test_mcp_dev_client_code_documents_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "code-documents",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_list_code_documents"
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }


def test_mcp_dev_client_code_documents_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        ("code-documents", "--contains", "plate", "--limit", "1")
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_list_code_documents",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "warnings": [],
                        "documents": [
                            {
                                "document_id": "plate_manager.orchestrator_config",
                                "widget_id": "plate_manager",
                                "readable": True,
                                "writable": True,
                                "supported_selection_modes": ["selected", "all"],
                                "current_selection_count": 1,
                                "total_scope_count": 2,
                                "title": "Plate manager orchestrator config",
                            },
                            {
                                "document_id": "plate_manager.pipeline",
                                "widget_id": "plate_manager",
                                "readable": True,
                                "writable": True,
                                "supported_selection_modes": ["selected", "all"],
                                "current_selection_count": 1,
                                "total_scope_count": 2,
                                "title": "Plate manager pipeline",
                            },
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("code-documents").render_response(
        response, args
    )

    assert "Code documents: count=2 matched=2 shown=1" in rendered
    assert "Filter: contains=plate" in rendered
    assert (
        "- plate_manager.orchestrator_config: widget=plate_manager "
        "readable=True writable=True selection=1/2 modes=selected,all "
        'title="Plate manager orchestrator config"'
    ) in rendered
    assert "plate_manager.pipeline" not in rendered
    assert "...<truncated 1 documents>" in rendered

    call_args = parser.parse_args(
        (
            "call",
            "openhcs_ui_list_code_documents",
            "--arguments",
            "{}",
        )
    )
    call_rendered = dev_client.McpDevCommandSpec.for_name("call").render_response(
        response,
        call_args,
    )

    assert "Code documents: count=2 matched=2 shown=2" in call_rendered


def test_mcp_dev_client_code_document_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "code-document",
            "object_state_scope:/tmp/plate::function_0",
            "--selection-mode",
            "all",
            "--full",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_get_code_document"
    assert call.arguments["document_id"] == "object_state_scope:/tmp/plate::function_0"
    assert call.arguments["selection_mode"] == "all"
    assert call.arguments["clean"] is False
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    default_args = parser.parse_args(
        ("code-document", "object_state_scope:/tmp/plate::function_0")
    )

    assert default_args.max_source_chars == dev_client.DEFAULT_CODE_DOCUMENT_MAX_CHARS

    alias_args = parser.parse_args(
        ("get-code-document", "object_state_scope:/tmp/plate::function_0")
    )
    alias_call = dev_client._calls_from_args(alias_args)[0]
    assert alias_call.name == "openhcs_ui_get_code_document"
    assert (
        alias_call.arguments["document_id"]
        == "object_state_scope:/tmp/plate::function_0"
    )


def test_mcp_dev_client_code_document_command_renders_source_and_revision():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "code-document",
            "object_state_scope:/tmp/plate::function_0",
            "--max-source-chars",
            "20",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_get_code_document",
                "mcp_error": False,
                "payloads": [
                    {
                        "current_revision_token": "rev-123",
                        "current_snapshot": {
                            "branch": "main",
                            "index": 9,
                            "is_head": True,
                        },
                        "selection_mode": "selected",
                        "selected_scope_ids": ["/tmp/plate::function_0"],
                        "sha256": "sha-abc",
                        "size_bytes": 42,
                        "source": "pattern = (some_function, {'scale': 1.0})",
                        "summary": {
                            "identity": {
                                "document_id": (
                                    "object_state_scope:/tmp/plate::function_0"
                                )
                            },
                            "title": "Edit Function",
                            "widget_id": "object_state_scope",
                            "writable": True,
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("code-document").render_response(
        response, args
    )

    assert (
        "Code document: id=object_state_scope:/tmp/plate::function_0 "
        'title="Edit Function" widget=object_state_scope writable=True '
        "mode=selected scopes=/tmp/plate::function_0"
    ) in rendered
    assert (
        "Revision: token=rev-123 sha256=sha-abc bytes=42 snapshot=main@9 head=True"
        in rendered
    )
    assert "Source:\npattern = (some_func" in rendered
    assert "...<truncated" in rendered


def test_mcp_dev_client_validate_code_document_projects_file_source(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    source_path = tmp_path / "document.py"
    source_path.write_text("pattern = (func, {})\n", encoding="utf-8")

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "validate-code-document",
            "object_state_scope:/tmp/plate::function_0",
            "--source-file",
            str(source_path),
            "--base-revision-token",
            "rev-123",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_validate_code_document"
    assert call.arguments["document_id"] == "object_state_scope:/tmp/plate::function_0"
    assert call.arguments["source"] == "pattern = (func, {})\n"
    assert call.arguments["base_revision_token"] == "rev-123"
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }


def test_mcp_dev_client_validate_code_document_renders_errors():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "validate-code-document",
            "object_state_scope:/tmp/plate::function_0",
            "--source-text",
            "bad = True",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_validate_code_document",
                "mcp_error": False,
                "payloads": [
                    {
                        "document_id": "object_state_scope:/tmp/plate::function_0",
                        "valid": False,
                        "normalized_scope_ids": [],
                        "errors": [
                            {
                                "code": "ui_code_document_validation_failed",
                                "message": "Document must define pattern.",
                            }
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "validate-code-document"
    ).render_response(response, args)

    assert (
        "Code document validation: "
        "id=object_state_scope:/tmp/plate::function_0 valid=False "
        "normalized_scopes=<none>"
    ) in rendered
    assert "Errors:" in rendered
    assert (
        "- ui_code_document_validation_failed: Document must define pattern."
        in rendered
    )


def test_mcp_dev_client_apply_code_document_projects_guarded_mutation(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    source_path = tmp_path / "document.py"
    source_path.write_text("pattern = (func, {'enabled': True})\n", encoding="utf-8")

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "apply-code-document",
            "object_state_scope:/tmp/plate::function_0",
            "--source-file",
            str(source_path),
            "--base-revision-token",
            "rev-123",
            "--no-confirmation",
            "--snapshot-label",
            "agent edit",
            "--request-token",
            "request-1",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_apply_code_document"
    assert call.arguments["document_id"] == "object_state_scope:/tmp/plate::function_0"
    assert call.arguments["source"] == "pattern = (func, {'enabled': True})\n"
    assert call.arguments["base_revision_token"] == "rev-123"
    assert call.arguments["require_confirmation"] is False
    assert call.arguments["snapshot_label"] == "agent edit"
    assert call.arguments["request_token"] == "request-1"
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }


def test_mcp_dev_client_apply_code_document_defaults_to_confirmation_guard():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "apply-code-document",
            "object_state_scope:/tmp/plate::function_0",
            "--source-text",
            "pattern = (func, {})",
            "--base-revision-token",
            "rev-123",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.arguments["require_confirmation"] is True


def test_mcp_dev_client_apply_code_document_renders_receipt_and_snapshots():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "apply-code-document",
            "object_state_scope:/tmp/plate::function_0",
            "--source-text",
            "pattern = (func, {})",
            "--base-revision-token",
            "rev-123",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_apply_code_document",
                "mcp_error": False,
                "payloads": [
                    {
                        "document_id": "object_state_scope:/tmp/plate::function_0",
                        "applied": True,
                        "outcome": "applied",
                        "operation_id": "operation-1",
                        "base_revision_token": "rev-123",
                        "current_revision_token": "rev-456",
                        "new_revision_token": "rev-456",
                        "receipt": {
                            "accepted": True,
                            "request_token": {"value": "request-1"},
                            "bridge_operation_id": "operation-1",
                        },
                        "current_snapshot": {
                            "branch": "main",
                            "index": 10,
                            "snapshot_id": "snapshot-10",
                        },
                        "undo_snapshot": {
                            "branch": "main",
                            "index": 9,
                            "snapshot_id": "snapshot-9",
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "apply-code-document"
    ).render_response(response, args)

    assert (
        "Code document apply: id=object_state_scope:/tmp/plate::function_0 "
        "applied=True outcome=applied operation=operation-1"
    ) in rendered
    assert "Revision: base=rev-123 current=rev-456 new=rev-456" in rendered
    assert (
        "Receipt: accepted=True request_token=request-1 bridge_operation=operation-1"
    ) in rendered
    assert "Snapshots: current=main@10:snapshot-10 undo=main@9:snapshot-9" in rendered


def test_mcp_dev_client_apply_code_document_renders_unchanged_result():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "apply-code-document",
            "object_state_scope:/tmp/plate::function_0",
            "--source-text",
            "pattern = (func, {})",
            "--base-revision-token",
            "rev-123",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_apply_code_document",
                "mcp_error": False,
                "payloads": [
                    {
                        "document_id": "object_state_scope:/tmp/plate::function_0",
                        "applied": False,
                        "outcome": "unchanged",
                        "operation_id": "operation-1",
                        "base_revision_token": "rev-123",
                        "current_revision_token": "rev-123",
                        "new_revision_token": "rev-123",
                        "receipt": {
                            "accepted": True,
                            "request_token": {"value": "request-1"},
                            "bridge_operation_id": "operation-1",
                        },
                        "current_snapshot": {
                            "branch": "main",
                            "index": 10,
                            "snapshot_id": "snapshot-10",
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "apply-code-document"
    ).render_response(response, args)

    assert (
        "Code document apply: id=object_state_scope:/tmp/plate::function_0 "
        "applied=False outcome=unchanged operation=operation-1"
    ) in rendered
    assert "Revision: base=rev-123 current=rev-123 new=rev-123" in rendered
    assert "Snapshots: current=main@10:snapshot-10 undo=<none>" in rendered


def test_mcp_dev_client_actions_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "actions",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_list_actions"
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    filtered_args = parser.parse_args(("actions", "plate_manager"))
    filtered_call = dev_client._calls_from_args(filtered_args)[0]

    assert filtered_args.widget_id == "plate_manager"
    assert filtered_call.name == "openhcs_ui_list_actions"


def test_mcp_dev_client_actions_command_renders_side_effects_and_tokens():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("actions",))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_list_actions",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "warnings": [
                            {
                                "code": "plate_path_setup_uses_code_document",
                                "message": "Prefer code document setup.",
                            }
                        ],
                        "actions": [
                            {
                                "widget_id": "plate_manager",
                                "action_id": "compile_plate",
                                "title": "Compile",
                                "enabled": True,
                                "confirmation_required": True,
                                "invocation_mode": "sync",
                                "current_selection_count": 1,
                                "target_scope_ids": ["/tmp/plate"],
                                "selection_revision_token": "selection-rev",
                                "selection_mode": "selected_plate",
                                "related_state_surface_ids": ["plate_manager.state"],
                                "side_effects": ["starts_compile_workflow"],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("actions").render_response(
        response,
        args,
    )

    assert "UI actions: count=1" in rendered
    assert "Warnings:" in rendered
    assert (
        "- plate_path_setup_uses_code_document: Prefer code document setup." in rendered
    )
    assert (
        '- plate_manager/compile_plate: title="Compile" enabled=True '
        "confirm=True mode=sync selection=1 targets=/tmp/plate "
        "selection_rev=selection-rev effects=starts_compile_workflow"
    ) in rendered
    assert "selection_mode=selected_plate surfaces=plate_manager.state" in rendered


def test_mcp_dev_client_actions_command_can_filter_by_widget_id():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("actions", "plate_manager"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_list_actions",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "warnings": [],
                        "actions": [
                            {
                                "widget_id": "plate_manager",
                                "action_id": "compile_plate",
                                "title": "Compile",
                                "enabled": False,
                                "confirmation_required": True,
                                "invocation_mode": "sync",
                                "current_selection_count": 1,
                                "disabled_error": {
                                    "code": "orchestrator_not_initialized",
                                    "message": "Initialize the plate first.",
                                    "hint": "Run init_plate before compile_plate.",
                                },
                            },
                            {
                                "widget_id": "image_browser",
                                "action_id": "open_file",
                                "title": "Open File",
                                "enabled": True,
                            },
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("actions").render_response(
        response,
        args,
    )

    assert "UI actions: count=1 widget=plate_manager" in rendered
    assert "plate_manager/compile_plate" in rendered
    assert "Disabled hints:" in rendered
    assert (
        'plate_manager/compile_plate: "Run init_plate before compile_plate."'
        in rendered
    )
    assert "image_browser/open_file" not in rendered


def test_mcp_dev_client_actions_filter_suppresses_unrelated_global_warnings():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_list_actions",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "warnings": [
                            {
                                "code": "plate_path_setup_uses_code_document",
                                "message": (
                                    "Use plate_manager.orchestrator_config; "
                                    "the add_plate UI action opens a dialog."
                                ),
                            }
                        ],
                        "actions": [
                            {
                                "widget_id": "plate_manager",
                                "action_id": "add_plate",
                                "title": "Add",
                                "enabled": True,
                            }
                        ],
                    }
                ],
            }
        ],
    }

    plate_args = parser.parse_args(("actions", "plate_manager"))
    plate_rendered = dev_client.McpDevCommandSpec.for_name("actions").render_response(
        response,
        plate_args,
    )
    pipeline_args = parser.parse_args(("actions", "pipeline_editor"))
    pipeline_rendered = dev_client.McpDevCommandSpec.for_name(
        "actions"
    ).render_response(
        response,
        pipeline_args,
    )

    assert "Warnings:" in plate_rendered
    assert "- plate_path_setup_uses_code_document:" in plate_rendered
    assert "UI actions: count=0 widget=pipeline_editor" in pipeline_rendered
    assert "Warnings:" not in pipeline_rendered
    assert "widget-tree pipeline_editor" in pipeline_rendered


def test_mcp_dev_client_invoke_action_projects_guarded_call():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "invoke-action",
            "plate_manager",
            "compile_plate",
            "--target-scope-id",
            "/tmp/plate",
            "--observed-selection-revision-token",
            "selection-rev",
            "--request-token",
            "request-1",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_invoke_action"
    assert call.arguments["widget_id"] == "plate_manager"
    assert call.arguments["action_id"] == "compile_plate"
    assert call.arguments["target_scope_ids"] == ["/tmp/plate"]
    assert call.arguments["observed_selection_revision_token"] == "selection-rev"
    assert call.arguments["request_token"] == "request-1"
    assert call.arguments["require_confirmation"] is True
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }


def test_mcp_dev_client_invoke_action_allows_explicit_no_confirmation():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "invoke-action",
            "plate_manager",
            "compile_plate",
            "--no-confirmation",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.arguments["require_confirmation"] is False


def test_mcp_dev_client_invoke_action_renders_receipt_and_polling():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("invoke-action", "plate_manager", "compile_plate"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_invoke_action",
                "mcp_error": False,
                "payloads": [
                    {
                        "status": "rejected",
                        "receipt": {
                            "accepted": False,
                            "request_token": {"value": "request-1"},
                            "bridge_operation_id": "operation-1",
                        },
                        "target_scope_ids": ["/tmp/plate"],
                        "selection_revision_token": "selection-rev",
                        "workflow_status_surface_ids": ["plate_manager.state"],
                        "recommended_poll_interval_ms": 500,
                        "errors": [
                            {
                                "code": "confirmation_required",
                                "message": "UI confirmation is required.",
                                "hint": "Call openhcs_ui_navigate_window for direct ObjectState navigation.",
                            }
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("invoke-action").render_response(
        response,
        args,
    )

    assert (
        "UI action invoke: action=plate_manager/compile_plate status=rejected"
        in rendered
    )
    assert (
        "Receipt: accepted=False request_token=request-1 bridge_operation=operation-1"
    ) in rendered
    assert "Selection: targets=/tmp/plate selection_rev=selection-rev" in rendered
    assert "Polling: surfaces=plate_manager.state interval_ms=500" in rendered
    assert "- confirmation_required: UI confirmation is required." in rendered
    assert 'hint="Call openhcs_ui_navigate_window' in rendered


def test_mcp_dev_client_invoke_widget_action_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    default_args = parser.parse_args(("invoke-widget-action", "plate_manager", "1.0.6"))
    default_call = dev_client._calls_from_args(default_args)[0]

    assert default_call.arguments["action_kind"] == "auto"
    assert default_call.arguments["target_index"] is None

    args = parser.parse_args(
        (
            "invoke-widget-action",
            "plate_manager",
            "1.0.6",
            "--action-kind",
            "button",
            "--target-index",
            "2",
            "--create-if-missing",
            "--request-token",
            "request-1",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_invoke_widget_action"
    assert call.arguments["window_id"] == "plate_manager"
    assert call.arguments["path_id"] == "1.0.6"
    assert call.arguments["action_kind"] == "button"
    assert call.arguments["target_index"] == 2
    assert call.arguments["create_if_missing"] is True
    assert call.arguments["request_token"] == "request-1"
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }


def test_mcp_dev_client_invoke_widget_action_renders_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("invoke-widget-action", "plate_manager", "1.0.6"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_invoke_widget_action",
                "mcp_error": False,
                "payloads": [
                    {
                        "window_id": "plate_manager",
                        "path_id": "1.0.6",
                        "action_kind": "button",
                        "invoked": True,
                        "receipt": {
                            "accepted": True,
                            "request_token": {"value": "request-1"},
                            "bridge_operation_id": "operation-1",
                        },
                        "summary": {
                            "label": "Code",
                            "enabled": True,
                            "clickable": True,
                            "action_kinds": ["button"],
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "invoke-widget-action"
    ).render_response(response, args)

    assert (
        "Widget action invoke: window=plate_manager path=1.0.6 kind=button invoked=True"
    ) in rendered
    assert (
        "Receipt: accepted=True request_token=request-1 bridge_operation=operation-1"
    ) in rendered
    assert 'Widget: label="Code" enabled=True clickable=True actions=button' in rendered


def test_mcp_dev_client_object_state_scope_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "object-state-scopes",
            "--scope-id",
            "global_config",
            "--include-system-scopes",
            "--include-fields",
            "--changed-only",
            "--include-field-values",
            "--field-limit",
            "12",
            "--field-offset",
            "4",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_list_object_state_scopes"
    assert call.arguments["scope_ids"] == ["global_config"]
    assert call.arguments["include_system_scopes"] is True
    assert call.arguments["include_fields"] is True
    assert call.arguments["field_filter"] == "semantic"
    assert call.arguments["include_field_values"] is True
    assert call.arguments["field_limit"] == 12
    assert call.arguments["field_offset"] == 4
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    alias_args = parser.parse_args(
        (
            "object-state-scopes",
            "--max-fields",
            "7",
        )
    )
    alias_call = dev_client._calls_from_args(alias_args)[0]
    assert alias_call.arguments["field_limit"] == 7

    positional_args = parser.parse_args(
        (
            "object-state-scopes",
            "global_config",
            "/tmp/plate",
        )
    )
    positional_call = dev_client._calls_from_args(positional_args)[0]
    assert positional_call.arguments["scope_ids"] == [
        "global_config",
        "/tmp/plate",
    ]


def test_mcp_dev_client_object_state_scope_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("object-state-scopes",))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_list_object_state_scopes",
                "mcp_error": False,
                "payloads": [
                    {
                        "object_state_token": 12,
                        "current_branch": "main",
                        "current_snapshot_index": -1,
                        "active": False,
                        "errors": [],
                        "warnings": [],
                        "scopes": [
                            {
                                "identity": {
                                    "object_state_scope_id": "global_config",
                                },
                                "object_type": "GlobalPipelineConfig",
                                "parameter_count": 136,
                                "field_page": {
                                    "returned_count": 1,
                                    "total_count": 136,
                                    "next_offset": 1,
                                    "limit": 1,
                                },
                                "dirty_field_count": 0,
                                "signature_diff_field_count": 1,
                                "has_unsaved_changes": False,
                                "has_default_overrides": True,
                                "last_changed_field": "num_workers",
                                "fields": [
                                    {
                                        "address": {
                                            "field_path": (
                                                "analysis_consolidation_config.enabled"
                                            ),
                                            "object_state_scope_id": "global_config",
                                        },
                                        "object_state_path_type": (
                                            "openhcs.core.config.AnalysisConsolidationConfig"
                                        ),
                                        "dirty": False,
                                        "signature_diff": False,
                                        "semantic_markers": [],
                                        "raw_value": True,
                                        "raw_value_preview": {
                                            "text": "True",
                                            "type_name": "bool",
                                            "is_none": False,
                                            "truncated": False,
                                        },
                                        "resolved_value": True,
                                        "resolved_value_preview": {
                                            "text": "True",
                                            "type_name": "bool",
                                            "is_none": False,
                                            "truncated": False,
                                        },
                                        "inherited_value": False,
                                        "provenance": None,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "object-state-scopes"
    ).render_response(response, args)

    assert (
        "ObjectState scopes: scopes=1 token=12 branch=main snapshot=-1 active=False"
        in rendered
    )
    assert "Markers: [*]=unsaved/dirty [_]=differs-from-defaults [-]=clean" in rendered
    assert (
        "- [_] scope=global_config: type=GlobalPipelineConfig params=136 dirty=0 "
        "default_diff=1 unsaved=False overrides=True changed=num_workers "
        "fields=1/136 next=1"
    ) in rendered
    assert (
        "[-] analysis_consolidation_config.enabled: "
        "target=AnalysisConsolidationConfig raw=True -> resolved=True "
        "inherited=False provenance=<none>"
    ) in rendered
    assert "<none>: raw=True" not in rendered
    assert (
        "Next field page: rerun with --include-fields --field-offset 1 --field-limit 1"
    ) in rendered


def test_mcp_dev_client_object_state_fields_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "object-state-fields",
            "--scope-id",
            "global_config",
            "--contains",
            "napari_streaming_config",
            "--query",
            "streaming_defaults",
            "--field-path-contains",
            "well_filter_config",
            "--path-contains",
            "processing_config",
            "--field-path",
            "napari_streaming_config.enabled",
            "--dirty-only",
            "--include-field-values",
            "--include-container-fields",
            "--max-fields",
            "20",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_get_object_state_fields"
    assert call.arguments["scope_ids"] == ["global_config"]
    assert call.arguments["field_paths"] == ["napari_streaming_config.enabled"]
    assert call.arguments["field_path_contains"] == [
        "napari_streaming_config",
        "streaming_defaults",
        "well_filter_config",
        "processing_config",
    ]
    assert call.arguments["include_clean_fields"] is False
    assert call.arguments["include_container_fields"] is True
    assert call.arguments["field_filter"] == "dirty"
    assert call.arguments["include_field_values"] is True
    assert call.arguments["max_fields"] == 20
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    filter_args = parser.parse_args(
        (
            "object-state-fields",
            "--field-filter",
            "semantic",
        )
    )
    filter_call = dev_client._calls_from_args(filter_args)[0]
    assert filter_call.arguments["field_filter"] == "semantic"
    assert filter_call.arguments["include_clean_fields"] is False

    positional_args = parser.parse_args(
        (
            "object-state-fields",
            "global_config",
            "/tmp/plate::pipeline",
            "--scope-id",
            "global_config",
            "--changed-only",
            "--include-values",
        )
    )
    positional_call = dev_client._calls_from_args(positional_args)[0]

    assert positional_call.arguments["scope_ids"] == [
        "global_config",
        "/tmp/plate::pipeline",
    ]
    assert positional_call.arguments["include_clean_fields"] is False
    assert positional_call.arguments["field_filter"] == "semantic"
    assert positional_call.arguments["include_field_values"] is True


def test_mcp_dev_client_object_state_field_help_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "object-state-help",
            "global_config",
            "napari_streaming_config.enabled",
            "--max-description-chars",
            "1200",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_describe_object_state_field"
    assert call.arguments["object_state_scope_id"] == "global_config"
    assert call.arguments["field_path"] == "napari_streaming_config.enabled"
    assert call.arguments["max_description_chars"] == 1200
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    option_args = parser.parse_args(
        (
            "field-help",
            "--scope-id",
            "global_config",
            "--field-path",
            "streaming_defaults.enabled",
        )
    )
    option_call = dev_client._calls_from_args(option_args)[0]

    assert option_call.name == "openhcs_ui_describe_object_state_field"
    assert option_call.arguments["object_state_scope_id"] == "global_config"
    assert option_call.arguments["field_path"] == "streaming_defaults.enabled"

    inferred_args = parser.parse_args(
        (
            "object-state-help",
            "napari_streaming_config.enabled",
            "--max-description-chars",
            "1200",
        )
    )
    inferred_call = dev_client._calls_from_args(inferred_args)[0]

    assert inferred_call.name == "openhcs_ui_describe_object_state_field"
    assert "object_state_scope_id" not in inferred_call.arguments
    assert inferred_call.arguments["field_path"] == "napari_streaming_config.enabled"
    assert inferred_call.arguments["max_description_chars"] == 1200

    inferred_option_args = parser.parse_args(
        (
            "field-help",
            "--field-path",
            "streaming_defaults.enabled",
        )
    )
    inferred_option_call = dev_client._calls_from_args(inferred_option_args)[0]

    assert inferred_option_call.name == "openhcs_ui_describe_object_state_field"
    assert "object_state_scope_id" not in inferred_option_call.arguments
    assert inferred_option_call.arguments["field_path"] == "streaming_defaults.enabled"


def test_mcp_dev_client_object_state_set_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "object-state-set",
            "global_config",
            "napari_streaming_config.enabled",
            "--value",
            "true",
            "--request-token",
            "req-1",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_ui_mutate_object_state_field"
    assert call.arguments["object_state_scope_id"] == "global_config"
    assert call.arguments["field_path"] == "napari_streaming_config.enabled"
    assert call.arguments["value"] is True
    assert call.arguments["reset"] is False
    assert call.arguments["include_field_values"] is True
    assert call.arguments["request_token"] == "req-1"
    assert call.arguments["connection"] == {
        "bridge_instance_id": "ui-test",
        "timeout_ms": 1234,
    }

    reset_args = parser.parse_args(
        (
            "object-state-edit",
            "--scope-id",
            "global_config",
            "--field-path",
            "well_filter_config.well_filter",
            "--reset",
            "--no-field-values",
        )
    )
    reset_call = dev_client._calls_from_args(reset_args)[0]

    assert reset_call.name == "openhcs_ui_mutate_object_state_field"
    assert reset_call.arguments["object_state_scope_id"] == "global_config"
    assert reset_call.arguments["field_path"] == "well_filter_config.well_filter"
    assert reset_call.arguments["value"] is None
    assert reset_call.arguments["reset"] is True
    assert reset_call.arguments["include_field_values"] is False


def test_mcp_dev_client_object_state_set_command_renders_before_after():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "object-state-set",
            "global_config",
            "well_filter_config.well_filter",
            "--value",
            "A01",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_mutate_object_state_field",
                "mcp_error": False,
                "payloads": [
                    {
                        "address": {
                            "object_state_scope_id": "global_config",
                            "field_path": "well_filter_config.well_filter",
                        },
                        "mutated": True,
                        "reset": False,
                        "receipt": {
                            "request_token": {"value": "req-1"},
                            "bridge_operation_id": "op-1",
                            "accepted": True,
                        },
                        "before": {
                            "address": {
                                "object_state_scope_id": "global_config",
                                "field_path": "well_filter_config.well_filter",
                            },
                            "object_state_path_type": (
                                "openhcs.core.config.WellFilterConfig"
                            ),
                            "dirty": False,
                            "signature_diff": False,
                            "raw_value_preview": {
                                "text": "None",
                                "type_name": "None",
                                "is_none": True,
                            },
                            "resolved_value_preview": {
                                "text": "None",
                                "type_name": "None",
                                "is_none": True,
                            },
                            "inherited_value": False,
                        },
                        "after": {
                            "address": {
                                "object_state_scope_id": "global_config",
                                "field_path": "well_filter_config.well_filter",
                            },
                            "object_state_path_type": (
                                "openhcs.core.config.WellFilterConfig"
                            ),
                            "dirty": True,
                            "signature_diff": True,
                            "semantic_markers": ["*", "_"],
                            "raw_value_preview": {
                                "text": "'A01'",
                                "type_name": "str",
                                "is_none": False,
                            },
                            "resolved_value_preview": {
                                "text": "'A01'",
                                "type_name": "str",
                                "is_none": False,
                            },
                            "inherited_value": False,
                        },
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "object-state-set"
    ).render_response(response, args)

    assert (
        "ObjectState field mutation: scope=global_config "
        "field=well_filter_config.well_filter mutated=True reset=False"
    ) in rendered
    assert "Receipt: accepted=True operation=op-1" in rendered
    assert "Before:" in rendered
    assert (
        "[-] well_filter_config.well_filter: target=WellFilterConfig "
        "raw=None -> resolved=None inherited=False"
    ) in rendered
    assert "After:" in rendered
    assert (
        "[*_] well_filter_config.well_filter: target=WellFilterConfig "
        "raw='A01' -> resolved='A01' inherited=False"
    ) in rendered
    assert '"payloads"' not in rendered


def test_mcp_dev_client_object_state_fields_command_renders_resolved_values():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("object-state-fields", "--contains", "napari"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_get_object_state_fields",
                "mcp_error": False,
                "payloads": [
                    {
                        "object_state_token": 12,
                        "current_branch": "main",
                        "current_snapshot_index": -1,
                        "requested_scope_ids": [],
                        "field_paths": ["napari_streaming_config.enabled"],
                        "field_path_contains": ["napari"],
                        "field_filter": "semantic",
                        "include_container_fields": True,
                        "matched_scope_count": 1,
                        "matched_field_count": 1,
                        "returned_field_count": 1,
                        "field_offset": 0,
                        "field_limit": 100,
                        "next_offset": None,
                        "truncated": False,
                        "errors": [],
                        "warnings": [],
                        "scopes": [
                            {
                                "scope_id": "global_config",
                                "object_type": "GlobalPipelineConfig",
                                "dirty_field_count": 0,
                                "signature_diff_field_count": 1,
                                "has_unsaved_changes": False,
                                "has_default_overrides": True,
                                "fields": [
                                    {
                                        "field_path": "napari_streaming_config.enabled",
                                        "object_state_path_type": (
                                            "openhcs.core.config.NapariStreamingConfig"
                                        ),
                                        "dirty": False,
                                        "signature_diff": False,
                                        "semantic_markers": [],
                                        "raw_value": None,
                                        "raw_value_preview": {
                                            "text": "None",
                                            "type_name": "None",
                                            "is_none": True,
                                            "truncated": False,
                                        },
                                        "resolved_value": None,
                                        "resolved_value_preview": {
                                            "text": "False",
                                            "type_name": "bool",
                                            "is_none": False,
                                            "truncated": False,
                                        },
                                        "inherited_value": True,
                                        "provenance": {
                                            "source_scope_id": "global_config",
                                            "source_type": (
                                                "openhcs.core.config.StreamingDefaults"
                                            ),
                                            "source_field_path": (
                                                "streaming_defaults.enabled"
                                            ),
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "object-state-fields"
    ).render_response(response, args)

    assert (
        "ObjectState fields: scopes=1 fields=1 returned=1 offset=0 limit=100 "
        "truncated=False token=12 branch=main snapshot=-1"
    ) in rendered
    assert "Markers: [*]=unsaved/dirty [_]=differs-from-defaults [-]=clean" in rendered
    assert (
        "Returned semantics: dirty=0 default_diff=0 inherited=1 "
        "raw_none_resolved=1 resolved_none_raw=0 plain=0"
    ) in rendered
    assert (
        "Filters: field_paths=napari_streaming_config.enabled "
        "contains=napari field_filter=semantic include_container_fields=True"
    ) in rendered
    assert (
        "Scope [_] scope=global_config: type=GlobalPipelineConfig dirty=0 "
        "default_diff=1 unsaved=False overrides=True"
    ) in rendered
    assert (
        "[-] napari_streaming_config.enabled: "
        "target=NapariStreamingConfig raw=None -> resolved=False inherited=True "
        "provenance=global_config:streaming_defaults.enabled (StreamingDefaults)"
    ) in rendered


def test_mcp_dev_client_call_renders_object_state_field_help_compactly():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "call",
            "openhcs_ui_describe_object_state_field",
            "--arguments",
            (
                '{"object_state_scope_id":"global_config",'
                '"field_path":"napari_display_config.colormap"}'
            ),
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_describe_object_state_field",
                "mcp_error": False,
                "payloads": [
                    {
                        "address": {
                            "object_state_scope_id": "global_config",
                            "field_path": "napari_display_config.colormap",
                        },
                        "object_type": "openhcs.core.config.GlobalPipelineConfig",
                        "help_target_type": "openhcs.core.config.NapariDisplayConfig",
                        "parameter_name": "colormap",
                        "summary": "• colormap (NapariColormap)",
                        "description": "No description available",
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("call").render_response(
        response,
        args,
    )

    assert (
        "ObjectState field help: scope=global_config "
        "field=napari_display_config.colormap"
    ) in rendered
    assert "help_target=NapariDisplayConfig parameter=colormap" in rendered
    assert "Summary: • colormap (NapariColormap)" in rendered
    assert "Description:\nNo description available" in rendered
    assert '"payloads"' not in rendered


def test_mcp_dev_client_object_state_field_help_command_renders_compactly():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "object-state-field-help",
            "global_config",
            "napari_streaming_config.enabled",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_describe_object_state_field",
                "mcp_error": False,
                "payloads": [
                    {
                        "address": {
                            "object_state_scope_id": "global_config",
                            "field_path": "napari_streaming_config.enabled",
                        },
                        "field": {
                            "address": {
                                "object_state_scope_id": "global_config",
                                "field_path": "napari_streaming_config.enabled",
                            },
                            "field_name": "enabled",
                            "container_path": "napari_streaming_config",
                            "object_state_path_type": (
                                "openhcs.core.config.NapariStreamingConfig"
                            ),
                            "raw_value_type": "NoneType",
                            "resolved_value_type": "bool",
                            "dirty": False,
                            "signature_diff": False,
                            "last_changed": False,
                            "raw_value_preview": {
                                "text": "None",
                                "type_name": "None",
                                "is_none": True,
                                "truncated": False,
                            },
                            "resolved_value_preview": {
                                "text": "False",
                                "type_name": "bool",
                                "is_none": False,
                                "truncated": False,
                            },
                            "inherited_value": True,
                            "provenance": {
                                "source_scope_id": "global_config",
                                "source_type": (
                                    "openhcs.core.config.StreamingDefaults"
                                ),
                                "source_field_path": "streaming_defaults.enabled",
                            },
                        },
                        "object_type": "openhcs.core.config.GlobalPipelineConfig",
                        "help_target_type": "openhcs.core.config.NapariStreamingConfig",
                        "parameter_name": "enabled",
                        "target_summary": "Napari streaming settings.",
                        "summary": "• enabled (bool)",
                        "description": "Whether this streaming config is enabled.",
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "object-state-field-help"
    ).render_response(response, args)

    assert (
        "ObjectState field help: scope=global_config "
        "field=napari_streaming_config.enabled"
    ) in rendered
    assert "help_target=NapariStreamingConfig parameter=enabled" in rendered
    assert "Target summary: Napari streaming settings." in rendered
    assert (
        "[-] napari_streaming_config.enabled: "
        "target=NapariStreamingConfig raw=None -> resolved=False inherited=True"
    ) in rendered
    assert "Description:\nWhether this streaming config is enabled." in rendered
    assert '"payloads"' not in rendered


def test_mcp_dev_client_object_state_field_help_truncates_long_target_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "object-state-field-help",
            "global_config",
            "napari_streaming_config.enabled",
        )
    )
    long_target_summary = (
        "NapariStreamingConfig("
        + ", ".join(
            f"field_{index}: SomeVeryLongTypeName = None" for index in range(20)
        )
        + ")"
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_describe_object_state_field",
                "mcp_error": False,
                "payloads": [
                    {
                        "address": {
                            "object_state_scope_id": "global_config",
                            "field_path": "napari_streaming_config.enabled",
                        },
                        "object_type": "openhcs.core.config.GlobalPipelineConfig",
                        "help_target_type": "openhcs.core.config.NapariStreamingConfig",
                        "parameter_name": "enabled",
                        "target_summary": long_target_summary,
                        "summary": "• enabled (bool)",
                        "description": "Whether this streaming config is enabled.",
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "object-state-field-help"
    ).render_response(response, args)

    assert "Target summary: NapariStreamingConfig(" in rendered
    assert "Target summary: " + long_target_summary not in rendered
    assert "field_19" not in rendered
    assert "..." in rendered
    assert "Summary: • enabled (bool)" in rendered


def test_mcp_dev_client_ui_status_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("ui-status",))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_bridge_status",
                "mcp_error": False,
                "payloads": [
                    {
                        "reachable": True,
                        "descriptor_status": "ok",
                        "bridge_instance_id": "ui-test",
                        "descriptor_file_path": "/tmp/ui.json",
                        "connection": {
                            "transport_mode": "ipc",
                            "host": "127.0.0.1",
                            "port": 7888,
                        },
                        "descriptors": [{"pid": 123}],
                        "supported_operations": ["status", "list_windows"],
                        "bridge_features": ["ui_windows"],
                        "errors": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("ui-status").render_response(
        response,
        args,
    )

    assert "UI bridge: reachable=True descriptor=ok" in rendered
    assert "Instance: ui-test pid=123" in rendered
    assert "Connection: ipc 127.0.0.1:7888" in rendered
    assert "Capabilities: 2 operations, 1 features" in rendered


def test_mcp_dev_client_ui_status_renders_unavailable_hint():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("ui-status",))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_bridge_status",
                "mcp_error": False,
                "payloads": [
                    {
                        "reachable": False,
                        "errors": [
                            {
                                "code": "ui_bridge_unavailable",
                                "message": "No running OpenHCS UI bridge gateway is configured.",
                                "hint": (
                                    "Pass descriptor_file_path, set "
                                    "OPENHCS_UI_BRIDGE_DESCRIPTOR, set "
                                    "OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR, or restart the UI."
                                ),
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("ui-status").render_response(
        response,
        args,
    )

    assert "UI bridge: unavailable" in rendered
    assert "ui_bridge_unavailable" in rendered
    assert "OPENHCS_UI_BRIDGE_DESCRIPTOR" in rendered


def test_mcp_dev_client_windows_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("windows",))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_list_windows",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "windows": [
                            {
                                "window_id": "qt_top_level:1",
                                "window_kind": "qt_top_level",
                                "visible": True,
                                "dirty": False,
                                "signature_diff": False,
                                "title": "OpenHCS",
                            },
                            {
                                "window_id": "qt_top_level:2",
                                "window_kind": "qt_top_level",
                                "visible": True,
                                "dirty": False,
                                "signature_diff": False,
                                "title": "Error",
                            },
                            {
                                "window_id": "plate_manager",
                                "window_kind": "embedded",
                                "visible": True,
                                "dirty": False,
                                "signature_diff": False,
                                "title": "Plate Manager",
                            },
                            {
                                "window_id": "image_browser",
                                "window_kind": "managed",
                                "visible": False,
                                "dirty": True,
                                "signature_diff": True,
                                "title": "Image Browser",
                            },
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("windows").render_response(
        response,
        args,
    )

    assert "Windows: 4" in rendered
    assert (
        'Attention: 1 visible top-level window(s): qt_top_level:2 title="Error"'
    ) in rendered
    assert (
        "- qt_top_level:1 [qt_top_level] visible=True dirty=False diff=False "
        'title="OpenHCS"'
    ) in rendered
    assert (
        "- qt_top_level:2 [qt_top_level] visible=True dirty=False diff=False "
        'title="Error"'
    ) in rendered
    assert (
        "- plate_manager [embedded] visible=True dirty=False diff=False "
        'title="Plate Manager"'
    ) in rendered
    assert (
        "- image_browser [managed] visible=False dirty=True diff=True "
        'title="Image Browser"'
    ) in rendered


def test_mcp_dev_client_ui_smoke_renders_compact_window_attention():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("ui-smoke",))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_health_check",
                "mcp_error": False,
                "payloads": [
                    {
                        "status": "ok",
                        "restart_required": False,
                        "stale_source_paths": [],
                    }
                ],
            },
            {
                "tool": "openhcs_ui_bridge_status",
                "mcp_error": False,
                "payloads": [
                    {
                        "reachable": True,
                        "descriptor_status": "ok",
                        "bridge_instance_id": "ui-test",
                        "descriptor_file_path": "/tmp/ui.json",
                        "connection": {
                            "transport_mode": "ipc",
                            "host": "127.0.0.1",
                            "port": 7888,
                        },
                        "descriptors": [{"pid": 123}],
                        "supported_operations": ["status", "list_windows"],
                        "bridge_features": ["ui_windows"],
                        "errors": [],
                    }
                ],
            },
            {
                "tool": "openhcs_ui_list_bridges",
                "mcp_error": False,
                "payloads": [
                    {
                        "bridges": [{"bridge_instance_id": "ui-test"}],
                        "errors": [],
                    }
                ],
            },
            {
                "tool": "openhcs_ui_list_windows",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "windows": [
                            {
                                "window_id": "qt_top_level:1",
                                "window_kind": "qt_top_level",
                                "visible": True,
                                "dirty": False,
                                "signature_diff": False,
                                "title": "OpenHCS",
                            },
                            {
                                "window_id": "qt_top_level:2",
                                "window_kind": "qt_top_level",
                                "visible": True,
                                "dirty": False,
                                "signature_diff": False,
                                "title": "Error",
                            },
                        ],
                    }
                ],
            },
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("ui-smoke").render_response(
        response,
        args,
    )

    assert "UI smoke: results=4 mcp_errors=0" in rendered
    assert "Health: status=ok restart_required=False stale_paths=0" in rendered
    assert "UI bridge: reachable=True descriptor=ok" in rendered
    assert "Bridges: live=1 errors=0" in rendered
    assert "Windows: 2" in rendered
    assert (
        'Attention: 1 visible top-level window(s): qt_top_level:2 title="Error"'
    ) in rendered


def test_mcp_dev_client_state_surfaces_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        ("state-surfaces", "--contains", "plate", "--limit", "1")
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_list_state_surfaces",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "warnings": [],
                        "surfaces": [
                            {
                                "surface_id": "plate_manager.state",
                                "widget_id": "plate_manager",
                                "readable": True,
                                "supported_selection_modes": ["selected", "all"],
                                "current_selection_count": 1,
                                "total_scope_count": 2,
                                "title": "Plate manager state",
                            },
                            {
                                "surface_id": "plate_manager.debug",
                                "widget_id": "plate_manager",
                                "readable": True,
                                "supported_selection_modes": ["selected", "all"],
                                "current_selection_count": 1,
                                "total_scope_count": 2,
                                "title": "Plate manager debug",
                            },
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("state-surfaces").render_response(
        response, args
    )

    assert "State surfaces: count=2 matched=2 shown=1" in rendered
    assert "Filter: contains=plate" in rendered
    assert (
        "- plate_manager.state: widget=plate_manager readable=True "
        'selection=1/2 modes=selected,all title="Plate manager state"'
    ) in rendered
    assert "plate_manager.debug" not in rendered
    assert "...<truncated 1 surfaces>" in rendered


def test_mcp_dev_client_unavailable_state_surface_renders_errors_and_next_step():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("state-surface", "plate_manager"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_get_state_surface",
                "mcp_error": False,
                "payloads": [
                    {
                        "current_revision_token": None,
                        "errors": [
                            {
                                "code": "ui_bridge_request_failed",
                                "message": "'Unknown UI state surface: plate_manager'",
                            }
                        ],
                        "payload": {},
                        "selection_mode": "all",
                        "summary": {
                            "identity": {"surface_id": "plate_manager"},
                            "readable": False,
                            "title": "Unavailable UI state surface",
                        },
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("state-surface").render_response(
        response, args
    )

    assert "Surface: plate_manager" in rendered
    assert "Readable: False" in rendered
    assert "Errors:" in rendered
    assert (
        "- ui_bridge_request_failed: 'Unknown UI state surface: plate_manager'"
        in rendered
    )
    assert "Next: state-surfaces" in rendered


def test_mcp_dev_client_plate_manager_state_surface_renders_compact_rows():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("state-surface", "plate_manager.state"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_get_state_surface",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "summary": {
                            "identity": {
                                "surface_id": "plate_manager.state",
                            },
                        },
                        "payload": {
                            "current_revision_token": "rev-a",
                            "manager_execution_state": "idle",
                            "current_snapshot": {
                                "index": 3,
                                "label": "auto-add output plate [__plates__]",
                            },
                            "summary": {
                                "current_selection_count": 1,
                            },
                            "rows": [
                                {
                                    "name": "plate-a",
                                    "orchestrator_state": "completed",
                                    "status_prefix": "Complete",
                                    "initialized": True,
                                    "compiled": True,
                                    "execution_active": False,
                                    "terminal_status": "complete",
                                    "selected": True,
                                    "plate_root": "/tmp/plate-a",
                                    "output_plate_root": "/tmp/plate-a_openhcs",
                                },
                                {
                                    "name": "plate-a_openhcs",
                                    "orchestrator_state": "created",
                                    "status_prefix": "",
                                    "initialized": False,
                                    "compiled": False,
                                    "execution_active": False,
                                    "terminal_status": None,
                                    "selected": False,
                                    "plate_root": "/tmp/plate-a_openhcs",
                                    "source_plate_root": "/tmp/plate-a",
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("state-surface").render_response(
        response, args
    )

    assert "Plate manager: rows=2 selected=1 manager=idle" in rendered
    assert 'Snapshot: 3 "auto-add output plate [__plates__]"' in rendered
    assert (
        "- plate-a: state=completed, status=Complete, init=True, "
        "compiled=True, active=False, terminal=complete, selected=True, "
        "root=/tmp/plate-a, output=/tmp/plate-a_openhcs"
    ) in rendered
    assert (
        "- plate-a_openhcs: state=created, status=<none>, init=False, "
        "compiled=False, active=False, terminal=<none>, selected=False, "
        "root=/tmp/plate-a_openhcs, source=/tmp/plate-a"
    ) in rendered


def test_mcp_dev_client_pipeline_editor_state_surface_renders_compact_steps():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("state-surface", "pipeline_editor.state"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_get_state_surface",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "summary": {
                            "identity": {
                                "surface_id": "pipeline_editor.state",
                            },
                        },
                        "payload": {
                            "current_revision_token": "rev-pipeline",
                            "current_plate_scope_id": "/tmp/plate-a",
                            "pipeline_scope_id": "/tmp/plate-a::pipeline",
                            "selected_scope_ids": [
                                "/tmp/plate-a::functionstep_1",
                            ],
                            "current_snapshot": {
                                "index": 4,
                                "label": "edit pipeline",
                            },
                            "summary": {
                                "current_selection_count": 1,
                                "total_scope_count": 2,
                            },
                            "steps": [
                                {
                                    "index": 0,
                                    "name": "Normalize",
                                    "enabled": True,
                                    "selected": False,
                                    "dirty": False,
                                    "default_diff": True,
                                    "debug_pause": False,
                                    "function_names": ["normalize"],
                                    "function_ids": ["openhcs:analysis_normalize"],
                                    "step_scope_id": "/tmp/plate-a::functionstep_0",
                                },
                                {
                                    "index": 1,
                                    "name": "Count nuclei",
                                    "enabled": True,
                                    "selected": True,
                                    "dirty": True,
                                    "default_diff": True,
                                    "debug_pause": True,
                                    "function_names": ["threshold", "count_objects"],
                                    "function_ids": [
                                        "openhcs:analysis_threshold",
                                        "openhcs:analysis_count_objects",
                                    ],
                                    "step_scope_id": "/tmp/plate-a::functionstep_1",
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("state-surface").render_response(
        response, args
    )

    assert (
        "Pipeline editor: steps=2 selected=1/2 plate=/tmp/plate-a "
        "pipeline=/tmp/plate-a::pipeline"
    ) in rendered
    assert 'Snapshot: 4 "edit pipeline"' in rendered
    assert "Selected scopes: /tmp/plate-a::functionstep_1" in rendered
    assert (
        "- 0. [_] Normalize: enabled=True, selected=False, funcs=normalize, "
        "ids=openhcs:analysis_normalize, "
        "scope=/tmp/plate-a::functionstep_0"
    ) in rendered
    assert (
        "- 1. [*_] Count nuclei: enabled=True, selected=True, "
        "funcs=threshold,count_objects, "
        "ids=openhcs:analysis_threshold,openhcs:analysis_count_objects, "
        "debug_pause=True, "
        "scope=/tmp/plate-a::functionstep_1"
    ) in rendered


def test_mcp_dev_client_pipeline_debug_session_surface_renders_runtime_frame():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("state-surface", "pipeline_debug_toolbar.session"))
    frame = {
        "debug_session_id": "debug-1",
        "snapshot_store_ref": "/debug",
        "snapshot_store_backend": None,
        "progress_identity": {
            "execution_id": "exec-1",
            "plate_id": "/tmp/plate-a",
            "axis_id": "A01",
            "step_name": "IdentifyPrimaryObjects",
        },
        "cursor": {
            "step_index": 1,
            "step_scope_id": "/tmp/plate-a::functionstep_1",
            "group_key": "default",
            "invocation_key": "default:0:IdentifyPrimaryObjects",
            "pattern_group_identity": None,
            "dirty": False,
        },
        "event_type": "after_invocation",
        "step_name": "IdentifyPrimaryObjects",
        "callable_name": "identify_primary_objects",
        "snapshot_id": "snapshot-1",
        "timestamp": 123.0,
    }
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_get_state_surface",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "summary": {
                            "identity": {
                                "surface_id": "pipeline_debug_toolbar.session",
                            },
                        },
                        "payload": {
                            "current_revision_token": "rev-debug",
                            "current_plate_scope_id": "/tmp/plate-a",
                            "pipeline_scope_id": "/tmp/plate-a::pipeline",
                            "manager_execution_state": "running",
                            "initialized": True,
                            "compiled": True,
                            "phase": "active_session",
                            "active_session_id": "debug-1",
                            "execution_id": "exec-1",
                            "axis_id": "A01",
                            "selected_source_group": None,
                            "snapshot_store_ref": "/debug",
                            "snapshot_store_backend": None,
                            "terminal_status": None,
                            "cursor": frame["cursor"],
                            "terminal_summary": None,
                            "current_frame": frame,
                            "last_frame": frame,
                            "actions": [],
                        },
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("state-surface").render_response(
        response, args
    )

    assert "Pipeline debug: phase=active_session" in rendered
    assert (
        "Current frame: event=after_invocation step=IdentifyPrimaryObjects "
        "callable=identify_primary_objects axis=A01 snapshot=snapshot-1 "
        "invocation=default:0:IdentifyPrimaryObjects"
    ) in rendered


def test_mcp_dev_client_pipeline_editor_state_omits_missing_function_ids():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("state-surface", "pipeline_editor.state"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_get_state_surface",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [],
                        "summary": {
                            "identity": {
                                "surface_id": "pipeline_editor.state",
                            },
                        },
                        "payload": {
                            "current_revision_token": "rev-pipeline",
                            "current_plate_scope_id": "/tmp/plate-a",
                            "pipeline_scope_id": "/tmp/plate-a::pipeline",
                            "selected_scope_ids": [],
                            "summary": {
                                "current_selection_count": 0,
                                "total_scope_count": 1,
                            },
                            "steps": [
                                {
                                    "index": 0,
                                    "name": "Normalize",
                                    "enabled": True,
                                    "selected": False,
                                    "dirty": False,
                                    "default_diff": False,
                                    "function_names": ["normalize"],
                                    "step_scope_id": "/tmp/plate-a::functionstep_0",
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("state-surface").render_response(
        response, args
    )

    assert "funcs=normalize" in rendered
    assert "ids=<none>" not in rendered


def test_mcp_dev_client_selected_workflow_poll_arguments_are_projected():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-workflow",
            "compile_plate",
            "--poll-state",
            "--poll-selection-mode",
            "all",
            "--poll-interval-seconds",
            "0.25",
            "--poll-timeout-seconds",
            "7.5",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    workflow_call = dev_client._calls_from_args(args)[0]
    state_arguments = dev_client.plate_manager_state_surface_tool_arguments(
        args,
        selection_mode=args.poll_selection_mode,
    )

    assert args.poll_state is True
    assert args.poll_selection_mode == "all"
    assert args.poll_interval_seconds == 0.25
    assert args.poll_timeout_seconds == 7.5
    assert workflow_call.name == "openhcs_ui_selected_plate_workflow"
    assert workflow_call.arguments == {
        "workflow": "compile_plate",
        "require_confirmation": False,
        "connection": {
            "bridge_instance_id": "ui-test",
            "timeout_ms": 1234,
        },
    }
    assert state_arguments == {
        "surface_id": "plate_manager.state",
        "selection_mode": "all",
        "connection": {
            "bridge_instance_id": "ui-test",
            "timeout_ms": 1234,
        },
    }


def test_mcp_dev_client_selected_workflow_wait_alias_preserves_poll_controls():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-workflow",
            "compile_plate",
            "--wait",
            "--wait-selection-mode",
            "all",
            "--wait-interval-seconds",
            "0.25",
            "--wait-timeout-seconds",
            "7.5",
        )
    )

    assert args.poll_state is True
    assert args.poll_selection_mode == "all"
    assert args.poll_interval_seconds == 0.25
    assert args.poll_timeout_seconds == 7.5

    selected_workflow_parser = next(
        action for action in parser._actions if action.dest == "command"
    ).choices["selected-workflow"]
    help_text = selected_workflow_parser.format_help()
    compact_help_text = " ".join(help_text.split())
    assert "--wait, --poll-state" in help_text
    assert "not the wait for a UI operation receipt" in compact_help_text


def test_mcp_dev_client_workflow_poll_terminal_state_policy():
    import openhcs.mcp.dev_client as dev_client

    init_policy = dev_client.WorkflowStatePollPolicy.from_workflow_text("init_plate")
    compile_policy = dev_client.WorkflowStatePollPolicy.from_workflow_text(
        "compile_plate"
    )
    run_policy = dev_client.WorkflowStatePollPolicy.from_workflow_text("run_plate")

    assert init_policy.terminal_for_row(
        dev_client.WorkflowPollRowState.from_mapping(
            {"init_pending": False, "initialized": True}
        )
    )
    assert not init_policy.terminal_for_row(
        dev_client.WorkflowPollRowState.from_mapping(
            {"init_pending": True, "initialized": True}
        )
    )
    assert compile_policy.terminal_for_row(
        dev_client.WorkflowPollRowState.from_mapping(
            {"compile_pending": False, "compiled": True}
        )
    )
    assert not compile_policy.terminal_for_row(
        dev_client.WorkflowPollRowState.from_mapping(
            {"compile_pending": False, "compiled": False}
        )
    )
    assert run_policy.terminal_for_row(
        dev_client.WorkflowPollRowState.from_mapping(
            {
                "execution_active": False,
                "queue_position": None,
                "terminal_status": "complete",
            }
        )
    )
    assert not run_policy.terminal_for_row(
        dev_client.WorkflowPollRowState.from_mapping(
            {
                "execution_active": False,
                "queue_position": 1,
                "terminal_status": "complete",
            }
        )
    )


def test_mcp_dev_client_workflow_poll_filters_target_scope_ids():
    import openhcs.mcp.dev_client as dev_client

    result = dev_client.McpDevToolResult(
        tool="openhcs_ui_get_state_surface",
        mcp_error=False,
        payloads=(
            {
                "payload": {
                    "rows": [
                        {
                            "plate_scope_id": "scope-a",
                            "compile_pending": False,
                            "compiled": True,
                        },
                        {
                            "plate_scope_id": "scope-b",
                            "compile_pending": True,
                            "compiled": False,
                        },
                    ]
                }
            },
        ),
    )
    policy = dev_client.WorkflowStatePollPolicy.from_workflow_text("compile_plate")

    assert dev_client.workflow_poll_has_reached_terminal_state(
        result,
        target_scope_ids=("scope-a",),
        policy=policy,
    )
    assert not dev_client.workflow_poll_has_reached_terminal_state(
        result,
        target_scope_ids=("scope-missing",),
        policy=policy,
    )
    assert not dev_client.workflow_poll_has_reached_terminal_state(
        result,
        target_scope_ids=(),
        policy=policy,
    )


def test_mcp_dev_client_workflow_poll_reports_failed_terminal_rows():
    import openhcs.mcp.dev_client as dev_client

    run_policy = dev_client.WorkflowStatePollPolicy.from_workflow_text("run_plate")
    compile_policy = dev_client.WorkflowStatePollPolicy.from_workflow_text(
        "compile_plate"
    )
    failed_run_result = dev_client.McpDevToolResult(
        tool="openhcs_ui_get_state_surface",
        mcp_error=False,
        payloads=(
            {
                "payload": {
                    "rows": [
                        {
                            "plate_scope_id": "scope-a",
                            "execution_active": False,
                            "queue_position": None,
                            "terminal_status": "failed",
                        }
                    ]
                }
            },
        ),
    )
    failed_compile_result = dev_client.McpDevToolResult(
        tool="openhcs_ui_get_state_surface",
        mcp_error=False,
        payloads=(
            {
                "payload": {
                    "rows": [
                        {
                            "plate_scope_id": "scope-a",
                            "compile_pending": False,
                            "compiled": False,
                            "orchestrator_state": "compile_failed",
                        }
                    ]
                }
            },
        ),
    )

    assert (
        dev_client.workflow_poll_terminal_status(
            failed_run_result,
            target_scope_ids=("scope-a",),
            policy=run_policy,
        )
        is dev_client.WorkflowPollSummaryStatus.FAILED
    )
    assert (
        dev_client.workflow_poll_terminal_status(
            failed_compile_result,
            target_scope_ids=("scope-a",),
            policy=compile_policy,
        )
        is dev_client.WorkflowPollSummaryStatus.FAILED
    )


def test_mcp_dev_client_selected_workflow_poll_composes_followup_state_calls(
    monkeypatch,
):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client
    import openhcs.mcp.dev_client_commands.ui as ui_commands

    calls: list[dev_client.McpDevToolCall] = []
    state_call_count = 0

    async def fake_call_tool(session, call, timeout_seconds):
        nonlocal state_call_count
        calls.append(call)
        if call.name == "openhcs_ui_selected_plate_workflow":
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "action_result": {
                            "status": "accepted",
                            "target_scope_ids": ["scope-a"],
                        }
                    },
                ),
            )
        state_call_count += 1
        compiled = state_call_count > 1
        return dev_client.McpDevToolResult(
            tool=call.name,
            mcp_error=False,
            payloads=(
                {
                    "current_revision_token": f"rev-{state_call_count}",
                    "payload": {
                        "object_state_token": state_call_count,
                        "rows": [
                            {
                                "plate_scope_id": "scope-a",
                                "compile_pending": False,
                                "compiled": compiled,
                            }
                        ],
                    },
                },
            ),
        )

    monkeypatch.setattr(ui_commands, "call_mcp_tool", fake_call_tool)

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-workflow",
            "compile_plate",
            "--poll-state",
            "--poll-selection-mode",
            "all",
            "--poll-timeout-seconds",
            "1",
            "--poll-interval-seconds",
            "0",
            "--bridge-instance-id",
            "ui-test",
            "--timeout-ms",
            "1234",
        )
    )

    response = asyncio.run(
        dev_client.McpDevCommandSpec.for_name("selected-workflow").run_session(
            SimpleNamespace(server_spec=dev_client.McpDevServerSpec(sys.executable)),
            args,
        )
    )

    assert [call.name for call in calls] == [
        "openhcs_ui_get_state_surface",
        "openhcs_ui_selected_plate_workflow",
        "openhcs_ui_get_state_surface",
    ]
    assert calls[0].arguments == {
        "surface_id": "plate_manager.state",
        "selection_mode": "all",
        "connection": {
            "bridge_instance_id": "ui-test",
            "timeout_ms": 1234,
        },
    }
    summary = response.results[-1]
    assert summary.tool == "mcp_dev_selected_workflow_poll"
    assert summary.mcp_error is False
    assert summary.payloads[0] == {
        "poll_status": "completed",
        "poll_requested": True,
        "poll_completed": True,
        "poll_count": 1,
        "target_scope_ids": ["scope-a"],
        "workflow": "compile_plate",
        "action_status": "accepted",
    }


def test_mcp_dev_client_selected_workflow_poll_recovers_from_transient_read_timeout(
    monkeypatch,
):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client
    import openhcs.mcp.dev_client_commands.ui as ui_commands
    from openhcs.agent.services.ui_bridge_service import UiBridgeGatewayTimeoutError

    calls: list[dev_client.McpDevToolCall] = []
    state_call_count = 0

    async def fake_call_tool(session, call, timeout_seconds):
        nonlocal state_call_count
        calls.append(call)
        if call.name == "openhcs_ui_selected_plate_workflow":
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "action_result": {
                            "status": "accepted",
                            "target_scope_ids": ["scope-a"],
                        }
                    },
                ),
            )
        state_call_count += 1
        if state_call_count == 2:
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "errors": [
                            {
                                "code": UiBridgeGatewayTimeoutError.agent_error_code,
                                "message": "The busy UI did not answer yet.",
                            }
                        ]
                    },
                ),
            )
        compiled = state_call_count > 2
        return dev_client.McpDevToolResult(
            tool=call.name,
            mcp_error=False,
            payloads=(
                {
                    "current_revision_token": f"rev-{state_call_count}",
                    "payload": {
                        "object_state_token": state_call_count,
                        "rows": [
                            {
                                "plate_scope_id": "scope-a",
                                "compile_pending": False,
                                "compiled": compiled,
                            }
                        ],
                    },
                },
            ),
        )

    monkeypatch.setattr(ui_commands, "call_mcp_tool", fake_call_tool)

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-workflow",
            "compile_plate",
            "--poll-state",
            "--poll-timeout-seconds",
            "1",
            "--poll-interval-seconds",
            "0",
        )
    )

    response = asyncio.run(
        dev_client.McpDevCommandSpec.for_name("selected-workflow").run_session(
            SimpleNamespace(server_spec=dev_client.McpDevServerSpec(sys.executable)),
            args,
        )
    )

    assert [call.name for call in calls].count(
        "openhcs_ui_selected_plate_workflow"
    ) == 1
    assert state_call_count == 3
    assert not any(
        UiBridgeGatewayTimeoutError.agent_error_code in str(result.payloads)
        for result in response.results
    )
    summary = response.results[-1]
    assert summary.mcp_error is False
    assert summary.payloads[0] == {
        "poll_status": "completed",
        "poll_requested": True,
        "poll_completed": True,
        "poll_count": 2,
        "target_scope_ids": ["scope-a"],
        "workflow": "compile_plate",
        "action_status": "accepted",
        "transient_poll_error_count": 1,
    }


def test_mcp_dev_client_selected_workflow_poll_exhausts_transient_read_timeout(
    monkeypatch,
):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client
    import openhcs.mcp.dev_client_commands.ui as ui_commands
    from openhcs.agent.services.ui_bridge_service import UiBridgeGatewayTimeoutError

    workflow_call_count = 0
    state_call_count = 0

    async def fake_call_tool(session, call, timeout_seconds):
        nonlocal state_call_count, workflow_call_count
        if call.name == "openhcs_ui_selected_plate_workflow":
            workflow_call_count += 1
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "action_result": {
                            "status": "accepted",
                            "target_scope_ids": ["scope-a"],
                        }
                    },
                ),
            )
        state_call_count += 1
        if state_call_count == 1:
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "current_revision_token": "baseline",
                        "payload": {"object_state_token": 1, "rows": []},
                    },
                ),
            )
        return dev_client.McpDevToolResult(
            tool=call.name,
            mcp_error=False,
            payloads=(
                {
                    "errors": [
                        {
                            "code": UiBridgeGatewayTimeoutError.agent_error_code,
                            "message": "The busy UI did not answer before deadline.",
                        }
                    ]
                },
            ),
        )

    monkeypatch.setattr(ui_commands, "call_mcp_tool", fake_call_tool)

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-workflow",
            "compile_plate",
            "--poll-state",
            "--poll-timeout-seconds",
            "0",
            "--poll-interval-seconds",
            "0",
        )
    )

    response = asyncio.run(
        dev_client.McpDevCommandSpec.for_name("selected-workflow").run_session(
            SimpleNamespace(server_spec=dev_client.McpDevServerSpec(sys.executable)),
            args,
        )
    )

    assert workflow_call_count == 1
    assert state_call_count == 2
    assert response.results[-2].has_errors()
    summary = response.results[-1]
    assert summary.mcp_error is True
    assert summary.payloads[0] == {
        "poll_status": "timeout",
        "poll_requested": True,
        "poll_completed": False,
        "poll_count": 1,
        "target_scope_ids": ["scope-a"],
        "workflow": "compile_plate",
        "action_status": "accepted",
        "transient_poll_error_count": 1,
    }


def test_mcp_dev_client_selected_workflow_poll_summary_reports_failure(
    monkeypatch,
):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client
    import openhcs.mcp.dev_client_commands.ui as ui_commands

    state_call_count = 0

    async def fake_call_tool(session, call, timeout_seconds):
        nonlocal state_call_count
        if call.name == "openhcs_ui_selected_plate_workflow":
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "action_result": {
                            "status": "accepted",
                            "target_scope_ids": ["scope-a"],
                        }
                    },
                ),
            )
        state_call_count += 1
        failed = state_call_count > 1
        return dev_client.McpDevToolResult(
            tool=call.name,
            mcp_error=False,
            payloads=(
                {
                    "current_revision_token": f"rev-{state_call_count}",
                    "payload": {
                        "object_state_token": state_call_count,
                        "rows": [
                            {
                                "plate_scope_id": "scope-a",
                                "execution_active": False,
                                "queue_position": None,
                                "terminal_status": ("failed" if failed else None),
                                "orchestrator_state": (
                                    "exec_failed" if failed else "compiled"
                                ),
                            }
                        ],
                    },
                },
            ),
        )

    monkeypatch.setattr(ui_commands, "call_mcp_tool", fake_call_tool)

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-workflow",
            "run_plate",
            "--poll-state",
            "--poll-timeout-seconds",
            "1",
            "--poll-interval-seconds",
            "0",
        )
    )

    response = asyncio.run(
        dev_client.McpDevCommandSpec.for_name("selected-workflow").run_session(
            SimpleNamespace(server_spec=dev_client.McpDevServerSpec(sys.executable)),
            args,
        )
    )

    summary = response.results[-1]
    assert summary.tool == "mcp_dev_selected_workflow_poll"
    assert summary.mcp_error is True
    assert summary.payloads[0] == {
        "poll_status": "failed",
        "poll_requested": True,
        "poll_completed": False,
        "poll_count": 1,
        "target_scope_ids": ["scope-a"],
        "workflow": "run_plate",
        "action_status": "accepted",
    }


def test_mcp_dev_client_selected_workflow_poll_stops_on_agent_error(
    monkeypatch,
):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client
    import openhcs.mcp.dev_client_commands.ui as ui_commands

    state_call_count = 0

    async def fake_call_tool(session, call, timeout_seconds):
        nonlocal state_call_count
        if call.name == "openhcs_ui_selected_plate_workflow":
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "action_result": {
                            "status": "accepted",
                            "target_scope_ids": ["scope-a"],
                        }
                    },
                ),
            )
        state_call_count += 1
        if state_call_count == 1:
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "current_revision_token": "rev-1",
                        "payload": {
                            "object_state_token": 1,
                            "rows": [],
                        },
                    },
                ),
            )
        return dev_client.McpDevToolResult(
            tool=call.name,
            mcp_error=False,
            payloads=(
                {
                    "errors": [
                        {
                            "code": "mcp_server_stale",
                            "message": "The MCP server source changed.",
                        }
                    ]
                },
            ),
        )

    monkeypatch.setattr(ui_commands, "call_mcp_tool", fake_call_tool)

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-workflow",
            "run_plate",
            "--poll-state",
            "--poll-timeout-seconds",
            "60",
            "--poll-interval-seconds",
            "10",
        )
    )

    response = asyncio.run(
        dev_client.McpDevCommandSpec.for_name("selected-workflow").run_session(
            SimpleNamespace(server_spec=dev_client.McpDevServerSpec(sys.executable)),
            args,
        )
    )

    assert state_call_count == 2
    assert response.results[-2].has_errors()
    summary = response.results[-1]
    assert summary.tool == "mcp_dev_selected_workflow_poll"
    assert summary.mcp_error is True
    assert summary.payloads[0] == {
        "poll_status": "failed",
        "poll_requested": True,
        "poll_completed": False,
        "poll_count": 1,
        "target_scope_ids": ["scope-a"],
        "workflow": "run_plate",
        "action_status": "accepted",
    }


def test_mcp_dev_client_selected_workflow_poll_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-workflow",
            "run_plate",
            "--poll-state",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_selected_plate_workflow",
                "mcp_error": False,
                "payloads": [
                    {
                        "action_result": {
                            "status": "accepted",
                            "target_scope_ids": ["scope-a"],
                        }
                    }
                ],
            },
            {
                "tool": "openhcs_ui_get_state_surface",
                "mcp_error": False,
                "payloads": [
                    {
                        "payload": {
                            "rows": [
                                {
                                    "name": "plate-a",
                                    "orchestrator_state": "completed",
                                    "status_prefix": "Complete",
                                    "terminal_status": "complete",
                                    "selected": True,
                                },
                                {
                                    "name": "plate-a_openhcs",
                                    "orchestrator_state": "created",
                                    "status_prefix": "",
                                    "terminal_status": None,
                                    "selected": False,
                                },
                            ]
                        }
                    }
                ],
            },
            {
                "tool": "mcp_dev_selected_workflow_poll",
                "mcp_error": False,
                "payloads": [
                    {
                        "workflow": "run_plate",
                        "action_status": "accepted",
                        "poll_status": "completed",
                        "poll_count": 4,
                        "target_scope_ids": ["scope-a"],
                    }
                ],
            },
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "selected-workflow"
    ).render_response(response, args)

    assert "Workflow: run_plate" in rendered
    assert "Action: accepted poll=completed count=4" in rendered
    assert "Targets: scope-a" in rendered
    assert (
        '- plate-a: state=completed, status="Complete", terminal=complete, selected=True'
        in rendered
    )
    assert '- plate-a_openhcs: state=created, status="", terminal=<none>' in rendered


def test_mcp_dev_client_selected_workflow_poll_summarizes_rejection(
    monkeypatch,
):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client
    import openhcs.mcp.dev_client_commands.ui as ui_commands

    calls: list[dev_client.McpDevToolCall] = []

    async def fake_call_tool(session, call, timeout_seconds):
        calls.append(call)
        if call.name == "openhcs_ui_get_state_surface":
            return dev_client.McpDevToolResult(
                tool=call.name,
                mcp_error=False,
                payloads=(
                    {
                        "current_revision_token": "rev-1",
                        "payload": {
                            "object_state_token": 1,
                            "rows": [],
                        },
                    },
                ),
            )
        return dev_client.McpDevToolResult(
            tool=call.name,
            mcp_error=False,
            payloads=(
                {
                    "action_result": {
                        "status": "rejected",
                        "target_scope_ids": ["scope-a"],
                        "errors": [
                            {
                                "code": "empty_pipeline_definition",
                                "message": "Selected plate has no pipeline definition to compile.",
                                "hint": "Create a pipeline before compile_plate.",
                            }
                        ],
                    },
                    "errors": [
                        {
                            "code": "empty_pipeline_definition",
                            "message": "Selected plate has no pipeline definition to compile.",
                            "hint": "Create a pipeline before compile_plate.",
                        }
                    ],
                },
            ),
        )

    monkeypatch.setattr(ui_commands, "call_mcp_tool", fake_call_tool)

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "selected-workflow",
            "compile_plate",
            "--poll-state",
            "--bridge-instance-id",
            "ui-test",
        )
    )

    response = asyncio.run(
        dev_client.McpDevCommandSpec.for_name("selected-workflow").run_session(
            SimpleNamespace(server_spec=dev_client.McpDevServerSpec(sys.executable)),
            args,
        )
    )

    assert [call.name for call in calls] == [
        "openhcs_ui_get_state_surface",
        "openhcs_ui_selected_plate_workflow",
    ]
    summary = response.results[-1]
    assert summary.tool == "mcp_dev_selected_workflow_poll"
    assert summary.mcp_error is True
    assert summary.payloads[0] == {
        "poll_status": "skipped",
        "poll_requested": True,
        "poll_completed": False,
        "poll_count": 0,
        "target_scope_ids": ["scope-a"],
        "workflow": "compile_plate",
        "skip_reason": "workflow_not_accepted",
        "action_status": "rejected",
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "selected-workflow"
    ).render_response(dev_client.to_jsonable(response), args)

    assert "Workflow: compile_plate" in rendered
    assert "Skip reason: workflow_not_accepted" in rendered
    assert "Errors:" in rendered
    assert "empty_pipeline_definition" in rendered
    assert "Selected plate has no pipeline definition to compile." in rendered


def test_mcp_dev_client_workflow_poll_timeout_summary_is_error():
    import openhcs.mcp.dev_client as dev_client

    summary = dev_client.workflow_poll_summary_result(
        workflow="compile_plate",
        status=dev_client.WorkflowPollSummaryStatus.TIMEOUT,
        poll_requested=True,
        poll_completed=False,
        poll_count=2,
        target_scope_ids=("scope-a",),
    )

    assert summary.tool == "mcp_dev_selected_workflow_poll"
    assert summary.mcp_error is True
    assert summary.payloads[0] == {
        "poll_status": "timeout",
        "poll_requested": True,
        "poll_completed": False,
        "poll_count": 2,
        "target_scope_ids": ["scope-a"],
        "workflow": "compile_plate",
    }


def test_mcp_dev_client_command_failed_detects_poll_summary_errors():
    import openhcs.mcp.dev_client as dev_client

    assert dev_client._command_failed(
        {
            "results": [
                {
                    "tool": "mcp_dev_selected_workflow_poll",
                    "mcp_error": True,
                    "payloads": [
                        {
                            "poll_status": "skipped",
                            "skip_reason": "workflow_not_accepted",
                        }
                    ],
                }
            ]
        }
    )


def test_mcp_dev_client_viewer_payloads_use_protocol_defaults():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client
    from openhcs.runtime.viewer_protocol import ViewerPayloadControlOptions

    parser = dev_client._build_parser()
    args = parser.parse_args(("viewer-payloads", "5565"))
    defaults = ViewerPayloadControlOptions()

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_get_viewer_window_payloads"
    assert call.arguments["port"] == 5565
    assert call.arguments["include_array_values"] is defaults.include_array_values
    assert call.arguments["include_shape_payloads"] is defaults.include_shape_payloads
    assert call.arguments["max_array_elements"] == defaults.max_array_elements
    assert call.arguments["max_shape_payloads"] == defaults.max_shape_payloads
    assert "timeout_ms" not in call.arguments


def test_mcp_dev_client_viewer_payloads_projects_axis_indices():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "viewer-payloads",
            "5565",
            "--route-key",
            "image-layer",
            "--axis-indices",
            "0,1",
            "--control-timeout-ms",
            "1000",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_get_viewer_window_payloads"
    assert call.arguments["route_key"] == "image-layer"
    assert call.arguments["axis_indices"] == [0, 1]
    assert call.arguments["timeout_ms"] == 1000


def test_mcp_dev_client_viewer_payloads_accepts_negative_payload_flags():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "viewer-payloads",
            "5565",
            "--include-array-values",
            "--no-array-values",
            "--include-shape-payloads",
            "--no-shape-payloads",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_get_viewer_window_payloads"
    assert call.arguments["include_array_values"] is False
    assert call.arguments["include_shape_payloads"] is False


def test_mcp_dev_client_viewer_payloads_accepts_timeout_ms_alias():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("viewer-payloads", "5565", "--timeout-ms", "1000"))

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_get_viewer_window_payloads"
    assert call.arguments["timeout_ms"] == 1000


def test_mcp_dev_client_viewer_payloads_projects_connection_options():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "viewer-payloads",
            "5565",
            "--host",
            "127.0.0.1",
            "--transport-mode",
            "ipc",
            "--timeout-ms",
            "1000",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_get_viewer_window_payloads"
    assert call.arguments["host"] == "127.0.0.1"
    assert call.arguments["transport_mode"] == "ipc"
    assert call.arguments["timeout_ms"] == 1000


def test_mcp_dev_client_viewer_commands_accept_port_option_alias():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()

    payload_args = parser.parse_args(("viewer-payloads", "--port", "5565"))
    payload_call = dev_client._calls_from_args(payload_args)[0]
    assert payload_call.arguments["port"] == 5565

    state_args = parser.parse_args(("viewer-state", "--port", "5555"))
    state_call = dev_client._calls_from_args(state_args)[0]
    assert state_call.arguments["port"] == 5555

    roi_args = parser.parse_args(
        ("viewer-rois", "--port", "5555", "--route-key", "roi-layer")
    )
    roi_call = dev_client._calls_from_args(roi_args)[0]
    assert roi_call.arguments["port"] == 5555
    assert roi_call.arguments["route_key"] == "roi-layer"

    roi_positional_args = parser.parse_args(
        ("viewer-rois", "--port", "5555", "roi-layer")
    )
    roi_positional_call = dev_client._calls_from_args(roi_positional_args)[0]
    assert roi_positional_call.arguments["port"] == 5555
    assert roi_positional_call.arguments["route_key"] == "roi-layer"

    sample_args = parser.parse_args(
        ("sample-viewer-image", "--port", "5555", "image-layer")
    )
    sample_call = dev_client._calls_from_args(sample_args)[0]
    assert sample_call.arguments["port"] == 5555
    assert sample_call.arguments["route_key"] == "image-layer"

    navigate_args = parser.parse_args(
        ("navigate-viewer", "--port", "5555", "roi-layer")
    )
    navigate_call = dev_client._calls_from_args(navigate_args)[0]
    assert navigate_call.arguments["port"] == 5555
    assert navigate_call.arguments["route_key"] == "roi-layer"

    isolate_args = parser.parse_args(
        ("isolate-viewer", "--port", "5555", "image-layer", "roi-layer")
    )
    isolate_call = dev_client._calls_from_args(isolate_args)[0]
    assert isolate_call.arguments["port"] == 5555
    assert isolate_call.arguments["visible_route_keys"] == [
        "image-layer",
        "roi-layer",
    ]


def test_mcp_dev_client_viewer_commands_reject_conflicting_ports():
    if importlib.util.find_spec("mcp") is None:
        return

    import pytest
    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("viewer-state", "5555", "--port", "5556"))

    with pytest.raises(ValueError, match="positional port and --port"):
        dev_client._calls_from_args(args)


def test_mcp_dev_client_viewer_missing_port_reports_usage_without_traceback(capsys):
    if importlib.util.find_spec("mcp") is None:
        return

    import pytest
    import openhcs.mcp.dev_client as dev_client

    with pytest.raises(SystemExit) as exc_info:
        dev_client.main(("viewer-state",))

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Viewer command requires a port argument or --port." in stderr
    assert "Traceback" not in stderr


def test_mcp_dev_client_viewer_payloads_projects_semantic_axis_indices():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "viewer-payloads",
            "5565",
            "--route-key",
            "image-layer",
            "--axis-index",
            "channel=0",
            "--axis-indices",
            "site=1",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_get_viewer_window_payloads"
    assert call.arguments["axis_indices"] == {"site": 1, "channel": 0}


def test_mcp_dev_client_viewer_payloads_renders_compact_summary(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("viewer-payloads", "5555"))
    streamed_path = tmp_path / "streamed_A01_w1.tif"
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_get_viewer_window_payloads",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": True,
                        "layer_count": 1,
                        "layers": [
                            {
                                "route_key": "image-layer",
                                "title": "Images",
                                "mounted": True,
                                "item_count": 2,
                                "axis_labels": ["channel", "y", "x"],
                                "stack_axes": ["channel"],
                                "pending_update": False,
                                "payloads": [
                                    {
                                        "data_type": "image",
                                        "axis_indices": [0],
                                        "aggregate_axis_indices": [],
                                        "components": {"channel": 1, "well": "A01"},
                                        "path": str(streamed_path),
                                        "summary": {
                                            "shape": [96, 96],
                                            "dtype": "uint16",
                                            "nonzero_count": 9216,
                                        },
                                        "array_value_summary": {
                                            "included": False,
                                            "shape": [96, 96],
                                            "omitted_reason": "max_array_elements_exceeded",
                                        },
                                    }
                                ],
                            },
                            {
                                "route_key": "roi-layer",
                                "title": "ROIs",
                                "mounted": True,
                                "item_count": 1,
                                "axis_labels": ["y", "x"],
                                "stack_axes": [],
                                "pending_update": False,
                                "payloads": [
                                    {
                                        "data_type": "shapes",
                                        "axis_indices": [],
                                        "aggregate_axis_indices": [],
                                        "components": {
                                            "channel": 2,
                                            "well": "A01",
                                        },
                                        "path": "/tmp/A01_w2.roi.zip",
                                        "summary": {
                                            "shape_payload_count": 7,
                                            "nonzero_count": 7,
                                        },
                                        "shape_payloads": [
                                            {"metadata": {"label": 1}},
                                            {"metadata": {"label": 2}},
                                        ],
                                    }
                                ],
                            },
                        ],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("viewer-payloads").render_response(
        response, args
    )

    assert "Viewer payloads: observed=True layers=1" in rendered
    assert (
        '- image-layer: title="Images" mounted=True items=2 axes=channel,y,x'
        in rendered
    )
    assert "payload type=image axis=0 aggregate_axis=<none>" in rendered
    assert "components=channel=1, well=A01" in rendered
    assert f"path={streamed_path} (streamed/non-materialized)" in rendered
    assert "payload type=shapes axis=<none> aggregate_axis=<none>" in rendered
    assert "shape_members=7 returned_shapes=2 semantic_rois=use-viewer-rois" in rendered
    assert "shape=[96, 96] dtype=uint16 nonzero=9216" in rendered
    assert (
        "array=included=False:shape=[96, 96]:reason=max_array_elements_exceeded"
        in rendered
    )


def test_mcp_dev_client_probe_viewer_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("probe-viewer", "5555", "--timeout-ms", "1000"))

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_probe_viewer_window"
    assert call.arguments["port"] == 5555
    assert call.arguments["host"] == "localhost"
    assert call.arguments["timeout_ms"] == 1000


def test_mcp_dev_client_probe_viewer_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("probe-viewer", "5555"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_probe_viewer_window",
                "mcp_error": False,
                "payloads": [
                    {
                        "reachable": True,
                        "observed": True,
                        "viewer": {
                            "viewer_type": "napari",
                            "title": "OpenHCS Napari Visualization",
                        },
                        "connection": {"port": 5555},
                        "layer_count": 2,
                        "component_group_count": 2,
                        "component_item_count": 4,
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("probe-viewer").render_response(
        response,
        args,
    )

    assert (
        "Viewer probe: reachable=True observed=True type=napari "
        'title="OpenHCS Napari Visualization"'
    ) in rendered
    assert "Window: port=5555 layers=2 component_groups=2 component_items=4" in rendered


def test_mcp_dev_client_window_snapshot_command_renders_resource():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("window-snapshot", "global_config"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_snapshot_window",
                "mcp_error": False,
                "payloads": [
                    {
                        "captured": True,
                        "window_id": "global_config",
                        "capture_scope": "window",
                        "width": 550,
                        "height": 600,
                        "resource": {
                            "path": "/tmp/snapshots/global_config.png",
                            "uri": "file:///tmp/snapshots/global_config.png",
                            "mime_type": "image/png",
                            "size_bytes": 12345,
                            "sha256": "abc123",
                        },
                        "summary": {
                            "title": "Configuration - GlobalPipelineConfig",
                            "window_kind": "scope",
                            "visible": True,
                            "dirty": False,
                            "dirty_field_count": 0,
                            "signature_diff": True,
                            "signature_diff_field_count": 3,
                            "semantic_markers": ["_"],
                            "object_state_scope_id": "global_config",
                            "managed_action_ids": [
                                "save_and_close",
                                "save_without_close",
                            ],
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("window-snapshot").render_response(
        response, args
    )

    assert (
        "Window snapshot: captured=True window=global_config "
        'title="Configuration - GlobalPipelineConfig" kind=scope scope=window'
    ) in rendered
    assert (
        "Status: visible=True dirty=False dirty_fields=0 "
        "default_diff=True default_diff_fields=3 markers=_"
    ) in rendered
    assert "Image: size=550x600 bytes=12345 mime=image/png" in rendered
    assert (
        "Resource: path=/tmp/snapshots/global_config.png "
        "uri=file:///tmp/snapshots/global_config.png sha256=abc123"
    ) in rendered
    assert "ObjectState: scope=global_config" in rendered
    assert "Actions: save_and_close,save_without_close" in rendered
    assert '"payloads"' not in rendered


def test_mcp_dev_client_call_renders_window_snapshot_compactly():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "call",
            "openhcs_ui_snapshot_window",
            "--arguments",
            '{"window_id":"global_config"}',
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_ui_snapshot_window",
                "mcp_error": False,
                "payloads": [
                    {
                        "captured": True,
                        "window_id": "global_config",
                        "capture_scope": "window",
                        "width": 550,
                        "height": 600,
                        "resource": {
                            "path": "/tmp/snapshots/global_config.png",
                            "uri": "file:///tmp/snapshots/global_config.png",
                            "mime_type": "image/png",
                            "size_bytes": 12345,
                            "sha256": "abc123",
                        },
                        "summary": {
                            "title": "Configuration - GlobalPipelineConfig",
                            "window_kind": "scope",
                            "visible": True,
                            "dirty": False,
                            "dirty_field_count": 0,
                            "signature_diff": True,
                            "signature_diff_field_count": 3,
                            "semantic_markers": ["_"],
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("call").render_response(
        response,
        args,
    )

    assert "Window snapshot: captured=True window=global_config" in rendered
    assert "Resource: path=/tmp/snapshots/global_config.png" in rendered
    assert '"payloads"' not in rendered


def test_mcp_dev_client_snapshot_viewer_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "snapshot-viewer",
            "5555",
            "--output-dir-path",
            "/tmp/snapshots",
            "--capture-scope",
            "widget",
            "--timeout-ms",
            "1000",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_viewer_snapshot_window"
    assert call.arguments["port"] == 5555
    assert call.arguments["output_dir_path"] == "/tmp/snapshots"
    assert call.arguments["capture_scope"] == "widget"
    assert call.arguments["timeout_ms"] == 1000


def test_mcp_viewer_snapshot_binding_projects_request():
    if importlib.util.find_spec("mcp") is None:
        return

    class _ViewerWindowService:
        def __init__(self):
            self.snapshot_requests = []

        def snapshot_window(self, request):
            self.snapshot_requests.append(request)
            return ViewerWindowSnapshotResult(
                schema_version=SCHEMA_VERSION,
                connection=request.connection,
                output_dir_path=request.output_dir_path,
                capture_scope=request.capture_scope,
                captured=True,
            )

    viewer_window_service = _ViewerWindowService()
    built = server.build_server(
        SimpleNamespace(viewer_window_service=viewer_window_service)
    )

    async def call_snapshot_tool():
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_viewer_snapshot_window",
                {
                    "port": 5555,
                    "output_dir_path": "/tmp/snapshots",
                    "capture_scope": "window",
                    "timeout_ms": 1000,
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_snapshot_tool())
    payload = json.loads(_direct_tool_text(result))

    assert payload["captured"] is True
    assert payload["output_dir_path"] == "/tmp/snapshots"
    assert payload["capture_scope"] == "window"
    assert viewer_window_service.snapshot_requests
    request = viewer_window_service.snapshot_requests[0]
    assert request.connection.port == 5555
    assert request.timeout_ms == 1000
    assert request.output_dir_path == "/tmp/snapshots"
    assert request.capture_scope is WindowSnapshotCaptureScope.WINDOW


def test_mcp_ui_snapshot_binding_projects_request_and_connection():
    if importlib.util.find_spec("mcp") is None:
        return

    class _UiBridgeService:
        def __init__(self):
            self.connections = []
            self.snapshot_requests = []

        def connection_from_fields(self, fields):
            self.connections.append(fields)
            return fields

        def snapshot_window(self, request, connection):
            self.snapshot_requests.append(request)
            return UiWindowSnapshotResult(
                schema_version=SCHEMA_VERSION,
                window_id=request.window_id,
                output_dir_path=request.output_dir_path,
                capture_scope=request.capture_scope,
                captured=True,
            )

    ui_bridge_service = _UiBridgeService()
    built = server.build_server(SimpleNamespace(ui_bridge_service=ui_bridge_service))

    async def call_snapshot_tool():
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_snapshot_window",
                {
                    "window_id": "global_config",
                    "output_dir_path": "/tmp/snapshots",
                    "capture_scope": "window",
                    "create_if_missing": True,
                    "connection": {"timeout_ms": 1234},
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_snapshot_tool())
    payload = json.loads(_direct_tool_text(result))

    assert payload["captured"] is True
    assert payload["window_id"] == "global_config"
    assert payload["output_dir_path"] == "/tmp/snapshots"
    assert payload["capture_scope"] == "window"
    assert ui_bridge_service.connections[0].timeout_ms == 1234
    request = ui_bridge_service.snapshot_requests[0]
    assert request.window_id == "global_config"
    assert request.open_policy.create_if_missing is True
    assert request.output_dir_path == "/tmp/snapshots"
    assert request.capture_scope is WindowSnapshotCaptureScope.WINDOW


def test_mcp_snapshot_capture_scope_uses_nominal_enum_validation():
    assert WindowSnapshotCaptureScope("widget").value == "widget"

    with pytest.raises(
        ValueError,
        match="'canvas' is not a valid WindowSnapshotCaptureScope",
    ):
        WindowSnapshotCaptureScope("canvas")


def test_mcp_dev_client_snapshot_viewer_rejects_invalid_capture_scope():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(("snapshot-viewer", "5555", "--capture-scope", "canvas"))


def test_mcp_dev_client_snapshot_viewer_command_renders_resource():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("snapshot-viewer", "5555"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_viewer_snapshot_window",
                "mcp_error": False,
                "payloads": [
                    {
                        "captured": True,
                        "capture_scope": "widget",
                        "viewer": {"viewer_type": "napari", "title": "Viewer"},
                        "width": 640,
                        "height": 480,
                        "resource": {
                            "path": "/tmp/snapshots/viewer.png",
                            "uri": "file:///tmp/snapshots/viewer.png",
                            "mime_type": "image/png",
                            "size_bytes": 12345,
                            "sha256": "abc123",
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("snapshot-viewer").render_response(
        response, args
    )

    assert (
        'Viewer snapshot: captured=True type=napari title="Viewer" scope=widget'
        in rendered
    )
    assert "Image: size=640x480 bytes=12345 mime=image/png" in rendered
    assert (
        "Resource: path=/tmp/snapshots/viewer.png uri=file:///tmp/snapshots/viewer.png sha256=abc123"
        in rendered
    )


def test_mcp_dev_client_viewer_state_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "viewer-state",
            "5555",
            "--route-key",
            "image-layer",
            "--max-component-values-per-layer",
            "4",
            "--max-payload-summaries-per-layer",
            "3",
            "--timeout-ms",
            "1000",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_get_viewer_window_state"
    assert call.arguments["port"] == 5555
    assert call.arguments["route_key"] == "image-layer"
    assert call.arguments["include_component_values"] is True
    assert call.arguments["max_component_values_per_layer"] == 4
    assert call.arguments["include_payload_summaries"] is True
    assert call.arguments["max_payload_summaries_per_layer"] == 3
    assert call.arguments["timeout_ms"] == 1000


def test_mcp_dev_client_viewer_state_command_renders_component_metadata(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("viewer-state", "5555"))
    streamed_path = tmp_path / "streamed_out.tif"
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_get_viewer_window_state",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": True,
                        "viewer": {
                            "viewer_type": "napari",
                            "title": "OpenHCS Napari Visualization",
                        },
                        "layer_count": 1,
                        "viewer_ndim": 3,
                        "axis_labels": ["channel", "y", "x"],
                        "current_step": [0, 47, 47],
                        "active_dimension_label_route": "image-layer",
                        "component_group_count": 1,
                        "component_item_count": 2,
                        "layers": [
                            {
                                "route_key": "image-layer",
                                "title": "1. Agent invert",
                                "visible": True,
                                "selected": True,
                                "item_count": 2,
                                "data_types": ["image"],
                                "axis_labels": ["channel", "y", "x"],
                                "stack_axes": ["channel"],
                                "data_shape": [2, 96, 96],
                                "component_values": [
                                    {
                                        "well": "A01",
                                        "channel": 1,
                                        "site": 1,
                                        "timepoint": 1,
                                        "z_index": 1,
                                    },
                                    {
                                        "well": "A01",
                                        "channel": 2,
                                        "site": 1,
                                        "timepoint": 1,
                                        "z_index": 1,
                                    },
                                ],
                                "axis_component_values": {"channel": [1, 2]},
                                "routed_component_values": {"channel": [1, 2]},
                                "payload_summary_count": 2,
                                "payload_summaries_truncated": False,
                                "payload_summaries": [
                                    {
                                        "data_type": "image",
                                        "shape": [96, 96],
                                        "dtype": "uint16",
                                        "min": 0,
                                        "max": 25647,
                                        "components": {
                                            "well": "A01",
                                            "channel": 1,
                                            "site": 1,
                                        },
                                        "path": str(streamed_path),
                                    }
                                ],
                            }
                        ],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("viewer-state").render_response(
        response,
        args,
    )

    assert (
        'Viewer state: observed=True type=napari title="OpenHCS Napari Visualization"'
        in rendered
    )
    assert (
        "Window: layers=1 ndim=3 axes=channel,y,x current_step=[0, 47, 47]" in rendered
    )
    assert (
        '- image-layer: title="1. Agent invert" visible=True selected=True' in rendered
    )
    assert (
        "components: channel=1,2, site=1, timepoint=1, well=A01, z_index=1" in rendered
    )
    assert "axis values: channel=1,2" in rendered
    assert "payload type=image shape=[96, 96] dtype=uint16 min=0 max=25647" in rendered
    assert f"path={streamed_path} (streamed/non-materialized)" in rendered


def test_mcp_dev_client_validate_viewer_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "validate-viewer",
            "5555",
            "--route-key",
            "image-layer",
            "--expected-layer-count",
            "3",
            "--required-axis-label",
            "well",
            "--required-axis-label",
            "channel",
            "--required-component-label",
            "site,well",
            "--allow-zero-payloads",
            "--include-state",
            "--host",
            "127.0.0.1",
            "--transport-mode",
            "ipc",
            "--timeout-ms",
            "1234",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_validate_viewer_window_state"
    assert call.arguments == {
        "port": 5555,
        "host": "127.0.0.1",
        "transport_mode": "ipc",
        "timeout_ms": 1234,
        "route_key": "image-layer",
        "expected_layer_count": 3,
        "required_axis_labels": ["well", "channel"],
        "required_component_labels": ["site", "well"],
        "require_nonzero_payloads": False,
        "include_state": True,
    }


def test_mcp_dev_client_validate_viewer_command_splits_axis_label_groups():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "validate-viewer",
            "5555",
            "--required-axis-label",
            "channel/y/x",
            "--required-axis-label",
            "well,site",
            "--required-axis-label",
            "channel",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.arguments["required_axis_labels"] == [
        "channel",
        "y",
        "x",
        "well",
        "site",
    ]

    require_nonzero_args = parser.parse_args(
        ("validate-viewer", "5555", "--require-nonzero-payloads")
    )
    require_nonzero_call = dev_client._calls_from_args(require_nonzero_args)[0]

    assert require_nonzero_call.arguments["require_nonzero_payloads"] is True


def test_mcp_dev_client_validate_viewer_command_accepts_component_aliases():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "validate-viewer",
            "5555",
            "--require-axis",
            "channel/y/x",
            "--require-component",
            "well",
            "--require-components",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.arguments["required_axis_labels"] == ["channel", "y", "x"]
    assert call.arguments["required_component_labels"].count("well") == 1
    assert set(call.arguments["required_component_labels"]) == {
        component.value for component in dev_client.AllComponents
    }


def test_mcp_dev_client_validate_viewer_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("validate-viewer", "5555"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_validate_viewer_window_state",
                "mcp_error": False,
                "payloads": [
                    {
                        "valid": True,
                        "observed": True,
                        "layer_count": 2,
                        "mounted_layer_count": 2,
                        "pending_update_count": 0,
                        "payload_count": 4,
                        "nonzero_payload_count": 4,
                        "zero_payload_count": 0,
                        "missing_payload_coordinate_count": 0,
                        "duplicate_payload_coordinate_count": 0,
                        "validation_policy": {
                            "expected_layer_count": 2,
                            "required_axis_labels": ["well", "channel"],
                            "required_component_labels": ["site"],
                            "require_nonzero_payloads": True,
                        },
                        "connection": {
                            "host": "localhost",
                            "port": 5555,
                            "transport_mode": "ipc",
                        },
                        "layer_summaries": [
                            {
                                "route_key": "image-layer",
                                "title": "Image",
                                "valid": True,
                                "mounted": True,
                                "item_count": 1,
                                "axis_labels": ["well", "channel", "y", "x"],
                                "stack_axes": ["well", "channel"],
                                "component_labels": ["channel", "site", "well"],
                                "payload_count": 4,
                                "nonzero_payload_count": 4,
                                "coordinate_gap_count": 0,
                                "missing_required_axis_labels": [],
                                "missing_required_component_labels": [],
                                "axis_labels_present_as_components": [],
                            }
                        ],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("validate-viewer").render_response(
        response,
        args,
    )

    assert "Viewer validation: valid=True observed=True layers=2 mounted=2" in rendered
    assert "Payloads: total=4 nonzero=4 zero=0 missing=0" in rendered
    assert (
        "Policy: expected_layers=2 required_axes=well,channel required_components=site"
    ) in rendered
    assert (
        "- image-layer: valid=True mounted=True items=1 axes=well,channel,y,x"
        in rendered
    )
    assert "components=channel,site,well missing_components=<none>" in rendered


def test_mcp_dev_client_validate_viewer_renders_axis_component_hint():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("validate-viewer", "5555"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_validate_viewer_window_state",
                "mcp_error": False,
                "payloads": [
                    {
                        "valid": False,
                        "observed": True,
                        "layer_count": 1,
                        "mounted_layer_count": 1,
                        "pending_update_count": 0,
                        "payload_count": 1,
                        "nonzero_payload_count": 1,
                        "zero_payload_count": 0,
                        "missing_payload_coordinate_count": 0,
                        "duplicate_payload_coordinate_count": 0,
                        "validation_policy": {
                            "required_axis_labels": ["site", "well", "channel"],
                            "required_component_labels": [],
                            "require_nonzero_payloads": True,
                        },
                        "connection": {
                            "host": "localhost",
                            "port": 5555,
                            "transport_mode": "ipc",
                        },
                        "layer_summaries": [
                            {
                                "route_key": "image-layer",
                                "title": "Image",
                                "valid": False,
                                "mounted": True,
                                "item_count": 1,
                                "axis_labels": ["channel", "y", "x"],
                                "stack_axes": ["channel"],
                                "component_labels": ["channel", "site", "well"],
                                "payload_count": 1,
                                "nonzero_payload_count": 1,
                                "coordinate_gap_count": 0,
                                "missing_required_axis_labels": ["site", "well"],
                                "missing_required_component_labels": [],
                                "axis_labels_present_as_components": ["site", "well"],
                            }
                        ],
                        "warnings": [
                            {
                                "code": "viewer_required_axis_labels_missing",
                                "message": (
                                    "Viewer layer 'Image' is missing required "
                                    "axis labels: site, well."
                                ),
                                "hint": (
                                    "These labels are present as component metadata "
                                    "rather than mounted axes: site, well."
                                ),
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("validate-viewer").render_response(
        response,
        args,
    )

    assert "axis_as_components=site,well" in rendered
    assert (
        'hint="These labels are present as component metadata rather than mounted axes: site, well."'
        in rendered
    )


def test_mcp_dev_client_validate_viewer_renders_tool_boundary_error():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("validate-viewer", "5555"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_validate_viewer_window_state",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [
                            {
                                "code": "mcp_tool_failed",
                                "message": "Viewer MCP timeout must not exceed 2000ms.",
                            }
                        ]
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("validate-viewer").render_response(
        response,
        args,
    )

    assert rendered == (
        "Viewer validation: failed\n"
        "- mcp_tool_failed: Viewer MCP timeout must not exceed 2000ms."
    )


def test_mcp_dev_client_viewer_rois_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "viewer-rois",
            "5555",
            "roi-layer",
            "--axis-indices",
            "0,1",
            "--max-rois",
            "12",
            "--max-examples",
            "2",
            "--timeout-ms",
            "500",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_summarize_viewer_window_rois"
    assert call.arguments["port"] == 5555
    assert call.arguments["route_key"] == "roi-layer"
    assert call.arguments["axis_indices"] == [0, 1]
    assert call.arguments["max_rois"] == 12
    assert call.arguments["max_examples"] == 2
    assert call.arguments["timeout_ms"] == 500

    limit_args = parser.parse_args(("viewer-rois", "5555", "--limit", "7"))
    limit_call = dev_client._calls_from_args(limit_args)[0]

    assert limit_call.arguments["max_rois"] == 7


def test_mcp_dev_client_viewer_rois_command_projects_semantic_axis_indices():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "viewer-rois",
            "5555",
            "roi-layer",
            "--axis-index",
            "channel=0",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_summarize_viewer_window_rois"
    assert call.arguments["axis_indices"] == {"channel": 0}


def test_mcp_dev_client_viewer_rois_command_accepts_route_key_option():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "viewer-rois",
            "5555",
            "--route-key",
            "roi-layer",
            "--axis-index",
            "channel=1",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_summarize_viewer_window_rois"
    assert call.arguments["route_key"] == "roi-layer"
    assert call.arguments["axis_indices"] == {"channel": 1}


def test_mcp_dev_client_viewer_rois_command_rejects_conflicting_route_keys():
    if importlib.util.find_spec("mcp") is None:
        return

    import pytest
    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "viewer-rois",
            "5555",
            "roi-layer",
            "--route-key",
            "other-layer",
        )
    )

    with pytest.raises(ValueError, match="positional route_key"):
        dev_client._calls_from_args(args)


def test_mcp_dev_client_viewer_rois_command_allows_route_discovery():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("viewer-rois", "5555"))

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_summarize_viewer_window_rois"
    assert call.arguments["port"] == 5555
    assert call.arguments["route_key"] is None
    assert call.arguments["axis_indices"] is None


def test_mcp_dev_client_viewer_rois_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("viewer-rois", "5555", "roi-layer"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_summarize_viewer_window_rois",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": True,
                        "route_key": "roi-layer",
                        "axis_indices": [0, 1],
                        "layer_count": 1,
                        "payload_record_count": 1,
                        "payload_type_counts": {"shapes": 1},
                        "roi_payload_count": 1,
                        "total_roi_count": 3,
                        "returned_roi_count": 3,
                        "roi_count_exact": False,
                        "total_roi_member_count": 7,
                        "returned_roi_member_count": 3,
                        "roi_payloads_truncated": True,
                        "payloads": [
                            {
                                "payload_route_key": "roi-layer:0",
                                "layer_route_key": "roi-layer",
                                "layer_title": "ROIs",
                                "axis_indices": [0, 1],
                                "components": {"well": "A01", "channel": 1},
                                "roi_count": 3,
                                "returned_roi_count": 3,
                                "roi_count_exact": False,
                                "roi_member_count": 7,
                                "returned_roi_member_count": 3,
                                "roi_duplicate_member_count": 0,
                                "roi_payloads_truncated": True,
                                "area": {
                                    "min": 10.0,
                                    "median": 42.0,
                                    "mean": 40.0,
                                    "max": 80.0,
                                },
                                "perimeter": {
                                    "min": 8.0,
                                    "median": 20.0,
                                    "mean": 21.0,
                                    "max": 44.0,
                                },
                                "bounds_yx": [[1, 2], [10, 11]],
                                "coordinate_count": 128,
                                "spatial_origin_yx": [0, 0],
                                "source_spatial_shape_yx": [96, 96],
                                "out_of_source_bounds_count": 0,
                                "example_rois": [
                                    {
                                        "label": "cell-1",
                                        "area": 42,
                                        "centroid_yx": [5, 6],
                                        "bbox_yxyx": [1, 2, 9, 10],
                                    }
                                ],
                            }
                        ],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("viewer-rois").render_response(
        response,
        args,
    )

    assert "Viewer ROIs: observed=True route=roi-layer axis=0,1" in rendered
    assert "ROIs: total=3 returned=3 exact=False members=7/3 truncated=True" in rendered
    assert "Payload types: shapes=1" in rendered
    assert (
        '- title="ROIs" layer_route=roi-layer payload_route=roi-layer:0 axis=0,1 '
        "components=channel=1, well=A01 "
        "roi_count=3 returned=3 exact=False members=7/3 duplicate_members=0"
    ) in rendered
    assert "area=min=10.0,median=42.0,mean=40.0,max=80.0" in rendered
    assert "perimeter=min=8.0,median=20.0,mean=21.0,max=44.0" in rendered
    assert (
        "coords=128 source_origin=[0, 0] source_shape=[96, 96] out_of_bounds=0"
        in rendered
    )
    assert "example label=cell-1 area=42 centroid=[5, 6]" in rendered


def test_mcp_dev_client_viewer_rois_explains_missing_roi_payloads():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("viewer-rois", "5555"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_summarize_viewer_window_rois",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": True,
                        "route_key": None,
                        "axis_indices": None,
                        "layer_count": 7,
                        "payload_record_count": 32,
                        "payload_type_counts": {"image": 32},
                        "roi_payload_count": 0,
                        "total_roi_count": 0,
                        "returned_roi_count": 0,
                        "roi_count_exact": True,
                        "total_roi_member_count": 0,
                        "returned_roi_member_count": 0,
                        "roi_payloads_truncated": False,
                        "payloads": [],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("viewer-rois").render_response(
        response,
        args,
    )

    assert (
        "Viewer ROIs: observed=True route=<none> axis=<none> "
        "layers=7 records=32 payloads=0"
    ) in rendered
    assert "Payload types: image=32" in rendered
    assert (
        "Interpretation: no ROI/shapes payloads were found for the requested viewer."
        in rendered
    )
    assert (
        "- If `viewer-state` shows a shapes layer, rerun "
        "`viewer-rois <port> <route_key>` for that route."
    ) in rendered
    assert "raw" not in rendered.lower()


def test_mcp_viewer_rois_collapses_duplicate_member_metadata():
    if importlib.util.find_spec("mcp") is None:
        return

    class _ViewerWindowService:
        def window_payloads(self, request):
            return ViewerWindowPayloadResult(
                schema_version=SCHEMA_VERSION,
                connection=request.connection,
                observed=True,
                viewer=ViewerWindowDescriptor(
                    viewer_type="napari",
                    title="OpenHCS Napari Visualization",
                ),
                layer_count=1,
                layers=(
                    ViewerWindowLayerPayloads(
                        route_key="roi-layer",
                        title="ROIs",
                        mounted=True,
                        item_count=1,
                        payloads=(
                            ViewerWindowPayloadRecord(
                                route_key="roi-layer:0",
                                data_type="shapes",
                                path="/tmp/A01_w2_rois.roi.zip",
                                components={"well": "A01", "channel": 2},
                                summary={
                                    "shape_payload_count": 3,
                                    "shape_coordinate_count": 12,
                                },
                                shape_payloads=(
                                    {
                                        "type": "polygon",
                                        "metadata": {
                                            "label": 4,
                                            "area": 136.0,
                                            "bbox": [23, 7, 39, 22],
                                            "centroid": [30.6, 14.0],
                                        },
                                    },
                                    {
                                        "type": "polygon",
                                        "metadata": {
                                            "label": 4,
                                            "area": 136.0,
                                            "bbox": [23, 7, 39, 22],
                                            "centroid": [30.6, 14.0],
                                        },
                                    },
                                    {
                                        "type": "polygon",
                                        "metadata": {
                                            "label": 5,
                                            "area": 91.0,
                                            "bbox": [37, 37, 48, 48],
                                            "centroid": [42.0, 42.0],
                                        },
                                    },
                                ),
                            ),
                        ),
                    ),
                ),
            )

    built = server.build_server(_viewer_mcp_context(_ViewerWindowService()))

    async def call_viewer_rois():
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_summarize_viewer_window_rois",
                {"port": 5555},
            ),
            timeout=2,
        )

    result = asyncio.run(call_viewer_rois())
    payload = json.loads(_direct_tool_text(result))

    assert payload["total_roi_count"] == 2
    assert payload["returned_roi_count"] == 2
    assert payload["roi_count_exact"] is True
    assert payload["total_roi_member_count"] == 3
    assert payload["returned_roi_member_count"] == 3
    assert payload["payload_record_count"] == 1
    assert payload["payload_type_counts"] == {"shapes": 1}
    roi_payload = payload["payloads"][0]
    assert roi_payload["roi_count"] == 2
    assert roi_payload["roi_member_count"] == 3
    assert roi_payload["roi_duplicate_member_count"] == 1
    assert roi_payload["area"]["mean"] == 113.5


def test_mcp_sample_viewer_image_auto_selects_single_image_layer():
    if importlib.util.find_spec("mcp") is None:
        return

    class _ViewerWindowService:
        def __init__(self):
            self.payload_requests = []

        def window_payloads(self, request):
            self.payload_requests.append(request)
            return ViewerWindowPayloadResult(
                schema_version=SCHEMA_VERSION,
                connection=request.connection,
                observed=True,
                viewer=ViewerWindowDescriptor(
                    viewer_type="napari",
                    title="OpenHCS Napari Visualization",
                ),
                layer_count=2,
                layers=(
                    ViewerWindowLayerPayloads(
                        route_key="image-layer",
                        title="Image",
                        mounted=True,
                        item_count=5,
                        payloads=tuple(
                            ViewerWindowPayloadRecord(
                                route_key=f"image-layer:{index}",
                                data_type="image",
                                path=f"/tmp/image-{index}.tif",
                                components={"well": "A01", "channel": 1},
                                axis_indices=(index,),
                                summary={
                                    "shape": [96, 96],
                                    "dtype": "uint16",
                                    "min": 1,
                                    "max": 4,
                                    "nonzero_count": 10,
                                },
                                array_value_summary={
                                    "requested": True,
                                    "included": False,
                                },
                            )
                            for index in range(5)
                        ),
                    ),
                    ViewerWindowLayerPayloads(
                        route_key="roi-layer",
                        title="ROI",
                        mounted=True,
                        item_count=1,
                        payloads=(),
                    ),
                ),
            )

    viewer_window_service = _ViewerWindowService()
    built = server.build_server(_viewer_mcp_context(viewer_window_service))

    async def call_sample_tool():
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_sample_viewer_window_image",
                {"port": 5555},
            ),
            timeout=2,
        )

    result = asyncio.run(call_sample_tool())
    payload = json.loads(_direct_tool_text(result))

    assert viewer_window_service.payload_requests
    request = viewer_window_service.payload_requests[0]
    assert request.payload_controls.route_key is None
    assert payload["requested_route_key"] is None
    assert payload["route_key"] == "image-layer"
    assert payload["auto_selected_route_key"] == "image-layer"
    assert payload["candidate_image_route_keys"] == ["image-layer"]
    assert payload["record_count"] == 5
    assert payload["returned_record_count"] == 3
    assert payload["records_truncated_count"] == 2
    assert len(payload["records"]) == 3
    assert {warning["code"] for warning in payload["warnings"]} == {
        "viewer_image_route_auto_selected"
    }


def test_mcp_sample_viewer_image_ambiguous_route_returns_no_records():
    if importlib.util.find_spec("mcp") is None:
        return

    class _ViewerWindowService:
        def window_payloads(self, request):
            return ViewerWindowPayloadResult(
                schema_version=SCHEMA_VERSION,
                connection=request.connection,
                observed=True,
                viewer=ViewerWindowDescriptor(
                    viewer_type="napari",
                    title="OpenHCS Napari Visualization",
                ),
                layer_count=2,
                layers=(
                    ViewerWindowLayerPayloads(
                        route_key="first-image-layer",
                        title="First",
                        mounted=True,
                        item_count=1,
                        payloads=(
                            ViewerWindowPayloadRecord(
                                route_key="first-image-layer:0",
                                data_type="image",
                                path="/tmp/first.tif",
                                components={"well": "A01"},
                                axis_indices=(0,),
                                summary={"shape": [8, 8]},
                                array_value_summary={
                                    "requested": True,
                                    "included": False,
                                },
                            ),
                        ),
                    ),
                    ViewerWindowLayerPayloads(
                        route_key="second-image-layer",
                        title="Second",
                        mounted=True,
                        item_count=1,
                        payloads=(
                            ViewerWindowPayloadRecord(
                                route_key="second-image-layer:0",
                                data_type="image",
                                path="/tmp/second.tif",
                                components={"well": "A01"},
                                axis_indices=(0,),
                                summary={"shape": [8, 8]},
                                array_value_summary={
                                    "requested": True,
                                    "included": False,
                                },
                            ),
                        ),
                    ),
                ),
            )

    built = server.build_server(_viewer_mcp_context(_ViewerWindowService()))

    async def call_sample_tool():
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_sample_viewer_window_image",
                {"port": 5555},
            ),
            timeout=2,
        )

    result = asyncio.run(call_sample_tool())
    payload = json.loads(_direct_tool_text(result))

    assert payload["candidate_image_route_keys"] == [
        "first-image-layer",
        "second-image-layer",
    ]
    assert payload["record_count"] == 0
    assert payload["raw_image_record_count"] == 2
    assert payload["filtered_out_image_record_count"] == 2
    assert payload["sample_omitted_count"] == 0
    assert payload["records"] == []
    assert {error["code"] for error in payload["errors"]} == {
        "viewer_image_route_ambiguous"
    }


def test_mcp_sample_viewer_image_axis_filter_preserves_route_filter():
    if importlib.util.find_spec("mcp") is None:
        return

    class _ViewerWindowService:
        def window_payloads(self, request):
            return ViewerWindowPayloadResult(
                schema_version=SCHEMA_VERSION,
                connection=request.connection,
                observed=True,
                viewer=ViewerWindowDescriptor(
                    viewer_type="napari",
                    title="OpenHCS Napari Visualization",
                ),
                layer_count=2,
                layers=(
                    ViewerWindowLayerPayloads(
                        route_key="selected-layer",
                        title="Selected",
                        mounted=True,
                        item_count=1,
                        payloads=(
                            ViewerWindowPayloadRecord(
                                route_key="selected-layer:0",
                                data_type="image",
                                path="/tmp/selected.tif",
                                components={"well": "A01"},
                                axis_indices=(0,),
                                summary={"shape": [8, 8]},
                                array_value_summary={
                                    "requested": True,
                                    "included": False,
                                },
                            ),
                        ),
                    ),
                    ViewerWindowLayerPayloads(
                        route_key="other-layer",
                        title="Other",
                        mounted=True,
                        item_count=1,
                        payloads=(
                            ViewerWindowPayloadRecord(
                                route_key="other-layer:0",
                                data_type="image",
                                path="/tmp/other.tif",
                                components={"well": "A01"},
                                axis_indices=(0,),
                                summary={"shape": [8, 8]},
                                array_value_summary={
                                    "requested": True,
                                    "included": False,
                                },
                            ),
                        ),
                    ),
                ),
            )

    built = server.build_server(_viewer_mcp_context(_ViewerWindowService()))

    async def call_sample_tool():
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_sample_viewer_window_image",
                {
                    "port": 5555,
                    "route_key": "selected-layer",
                    "axis_indices": [0],
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_sample_tool())
    payload = json.loads(_direct_tool_text(result))

    assert payload["record_count"] == 1
    assert payload["raw_image_record_count"] == 2
    assert payload["records"][0]["layer_route_key"] == "selected-layer"
    assert payload["records"][0]["payload_route_key"] == "selected-layer:0"


def test_mcp_dev_client_sample_viewer_image_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "sample-viewer-image",
            "5555",
            "image-layer",
            "--axis-indices",
            "0,1",
            "--y",
            "2",
            "--x",
            "3",
            "--height",
            "4",
            "--width",
            "5",
            "--max-array-elements",
            "20",
            "--max-records",
            "2",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_sample_viewer_window_image"
    assert call.arguments["port"] == 5555
    assert call.arguments["route_key"] == "image-layer"
    assert call.arguments["axis_indices"] == [0, 1]
    assert call.arguments["y"] == 2
    assert call.arguments["x"] == 3
    assert call.arguments["height"] == 4
    assert call.arguments["width"] == 5
    assert call.arguments["include_array_values"] is False
    assert call.arguments["max_array_elements"] == 20
    assert call.arguments["max_records"] == 2


def test_mcp_dev_client_sample_viewer_image_command_allows_omitted_route_key():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("sample-viewer-image", "5555"))

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_sample_viewer_window_image"
    assert call.arguments["port"] == 5555
    assert call.arguments["route_key"] is None


def test_mcp_dev_client_sample_viewer_image_command_projects_semantic_axis_indices():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "sample-viewer-image",
            "5555",
            "image-layer",
            "--axis-index",
            "channel=0",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_sample_viewer_window_image"
    assert call.arguments["axis_indices"] == {"channel": 0}
    assert call.arguments["include_array_values"] is False


def test_mcp_dev_client_sample_viewer_image_command_can_include_array_values():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "sample-viewer-image",
            "5555",
            "image-layer",
            "--include-array-values",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_sample_viewer_window_image"
    assert call.arguments["include_array_values"] is True


def test_mcp_dev_client_sample_viewer_image_command_renders_compact_summary(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("sample-viewer-image", "5555", "image-layer"))
    streamed_path = tmp_path / "sampled_virtual_image.tif"
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_sample_viewer_window_image",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": True,
                        "route_key": "image-layer",
                        "axis_indices": [0, 1],
                        "array_slices": [[2, 6], [3, 8]],
                        "record_count": 1,
                        "returned_record_count": 1,
                        "records_truncated_count": 0,
                        "raw_image_record_count": 1,
                        "total_payload_record_count": 2,
                        "sample_protocol_supported": True,
                        "sample_included_count": 1,
                        "sample_omitted_count": 0,
                        "records": [
                            {
                                "payload_route_key": "image-layer:0",
                                "layer_route_key": "image-layer",
                                "axis_indices": [0, 1],
                                "path": str(streamed_path),
                                "summary": {
                                    "shape": [64, 64],
                                    "dtype": "uint16",
                                    "min": 1,
                                    "max": 4,
                                    "nonzero_count": 20,
                                },
                                "array_value_summary": {
                                    "included": True,
                                    "shape": [4, 5],
                                },
                                "array_values": [[1, 2], [3, 4]],
                            }
                        ],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "sample-viewer-image"
    ).render_response(response, args)

    assert "Viewer image sample: observed=True route=image-layer axis=0,1" in rendered
    assert (
        "Records: matched=1 returned=1 truncated=0 image=1 "
        "total_payloads=2 sample_supported=True" in rendered
    )
    assert (
        "- image-layer:0: layer=image-layer axis=0,1 "
        f"path={streamed_path} (streamed/non-materialized)"
    ) in rendered
    assert "shape=[64, 64] dtype=uint16 min=1 max=4 nonzero=20" in rendered
    assert "included=True sample_shape=[4, 5]" in rendered
    assert "sample values: [[1, 2], [3, 4]]" in rendered


def test_mcp_dev_client_sample_viewer_image_renders_omitted_reason():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("sample-viewer-image", "5555", "image-layer"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_sample_viewer_window_image",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": True,
                        "route_key": "image-layer",
                        "axis_indices": {"channel": 1},
                        "array_slices": [[0, 8], [0, 8]],
                        "record_count": 1,
                        "raw_image_record_count": 1,
                        "total_payload_record_count": 1,
                        "sample_protocol_supported": True,
                        "sample_included_count": 0,
                        "sample_omitted_count": 1,
                        "records": [
                            {
                                "payload_route_key": "image-layer",
                                "layer_route_key": "image-layer",
                                "axis_indices": [1],
                                "path": "virtual/image.tif",
                                "summary": {
                                    "shape": [96, 96],
                                    "dtype": "uint16",
                                    "min": 23,
                                    "max": 10755,
                                    "nonzero_count": 9216,
                                },
                                "array_value_summary": {
                                    "included": False,
                                    "shape": [8, 8],
                                    "omitted_reason": "max_array_elements_exceeded",
                                    "max_array_elements": 0,
                                },
                                "array_values": [],
                            }
                        ],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "sample-viewer-image"
    ).render_response(response, args)

    assert "axis=channel=1" in rendered
    assert "axis={'channel': 1}" not in rendered
    assert "included=False sample_shape=[8, 8]" in rendered
    assert "reason=array_values_not_requested max_elements=0" in rendered
    assert "rerun_with=--include-array-values --max-array-elements 64" in rendered


def test_mcp_dev_client_sample_viewer_image_omission_suggests_element_budget():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "sample-viewer-image",
            "5555",
            "image-layer",
            "--include-array-values",
            "--max-array-elements",
            "40",
        )
    )
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_sample_viewer_window_image",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": True,
                        "route_key": "image-layer",
                        "axis_indices": {"channel": 1},
                        "array_slices": [[0, 8], [0, 8]],
                        "record_count": 1,
                        "raw_image_record_count": 1,
                        "total_payload_record_count": 1,
                        "sample_protocol_supported": True,
                        "sample_included_count": 0,
                        "sample_omitted_count": 1,
                        "records": [
                            {
                                "payload_route_key": "image-layer",
                                "layer_route_key": "image-layer",
                                "axis_indices": [1],
                                "path": "virtual/image.tif",
                                "summary": {
                                    "shape": [96, 96],
                                    "dtype": "uint16",
                                    "min": 23,
                                    "max": 10755,
                                    "nonzero_count": 9216,
                                },
                                "array_value_summary": {
                                    "included": False,
                                    "shape": [8, 8],
                                    "omitted_reason": "max_array_elements_exceeded",
                                    "max_array_elements": 40,
                                },
                                "array_values": [],
                            }
                        ],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name(
        "sample-viewer-image"
    ).render_response(response, args)

    assert "reason=max_array_elements_exceeded max_elements=40" in rendered
    assert "rerun_max_elements=64" in rendered


def test_mcp_dev_client_navigate_viewer_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "navigate-viewer",
            "5555",
            "roi-layer",
            "--axis-index",
            "channel=1",
            "--axis-index",
            "well=0",
            "--hidden",
            "--deselected",
            "--timeout-ms",
            "500",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_navigate_viewer_window"
    assert call.arguments["port"] == 5555
    assert call.arguments["route_key"] == "roi-layer"
    assert call.arguments["axis_indices"] == {"channel": 1, "well": 0}
    assert call.arguments["visible"] is False
    assert call.arguments["selected"] is False
    assert call.arguments["timeout_ms"] == 500


def test_mcp_dev_client_navigate_viewer_command_can_avoid_state_toggles():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "navigate-viewer",
            "5555",
            "roi-layer",
            "--no-visible-change",
            "--no-selection-change",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.arguments["visible"] is None
    assert call.arguments["selected"] is None
    assert call.arguments["axis_indices"] == {}


def test_mcp_dev_client_navigate_viewer_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("navigate-viewer", "5555", "roi-layer"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_navigate_viewer_window",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": True,
                        "route_key": "roi-layer",
                        "visible": True,
                        "selected": True,
                        "axis_labels": ["channel", "y", "x"],
                        "current_step": [1, 0, 0],
                        "active_dimension_label_route": "roi-layer",
                        "errors": [],
                        "warnings": [
                            {
                                "code": "viewer_axis_missing",
                                "message": "z_index is not mounted.",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("navigate-viewer").render_response(
        response,
        args,
    )

    assert (
        "Viewer navigation: observed=True route=roi-layer visible=True selected=True"
        in rendered
    )
    assert "Position: axes=channel,y,x current_step=[1, 0, 0]" in rendered
    assert "active_route=roi-layer" in rendered
    assert "Warnings:" in rendered
    assert "- viewer_axis_missing: z_index is not mounted." in rendered


def test_mcp_dev_client_viewer_navigation_groups_repeated_errors():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("isolate-viewer", "5555", "roi-layer"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_isolate_viewer_window_layers",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": False,
                        "applied": False,
                        "selected_route_key": "roi-layer",
                        "changed_route_count": 0,
                        "layer_count": 0,
                        "axis_labels": [],
                        "current_step": [],
                        "active_dimension_label_route": None,
                        "visible_route_keys": [],
                        "hidden_route_keys": [],
                        "errors": [
                            {
                                "code": "viewer_window_navigation_failed",
                                "message": "Viewer control request timed out.",
                            },
                            {
                                "code": "viewer_window_navigation_failed",
                                "message": "Viewer control request timed out.",
                            },
                            {
                                "code": "viewer_window_state_failed",
                                "message": "Viewer control request timed out.",
                            },
                        ],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("isolate-viewer").render_response(
        response,
        args,
    )

    assert (
        "- viewer_window_navigation_failed, viewer_window_state_failed: "
        "Viewer control request timed out."
    ) in rendered
    assert rendered.count("Viewer control request timed out.") == 1


def test_mcp_dev_client_isolate_viewer_command_projects_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(
        (
            "isolate-viewer",
            "5555",
            "image-layer",
            "roi-layer",
            "--selected-route-key",
            "roi-layer",
            "--axis-index",
            "channel=1",
            "--timeout-ms",
            "500",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_isolate_viewer_window_layers"
    assert call.arguments["port"] == 5555
    assert call.arguments["visible_route_keys"] == ["image-layer", "roi-layer"]
    assert call.arguments["selected_route_key"] == "roi-layer"
    assert call.arguments["axis_indices"] == {"channel": 1}
    assert call.arguments["timeout_ms"] == 500


def test_mcp_dev_client_isolate_viewer_command_renders_compact_summary():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    args = parser.parse_args(("isolate-viewer", "5555", "roi-layer"))
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_isolate_viewer_window_layers",
                "mcp_error": False,
                "payloads": [
                    {
                        "observed": True,
                        "applied": True,
                        "selected_route_key": "roi-layer",
                        "changed_route_count": 2,
                        "layer_count": 2,
                        "axis_labels": ["channel", "y", "x"],
                        "current_step": [1, 0, 0],
                        "active_dimension_label_route": "roi-layer",
                        "visible_route_keys": ["roi-layer"],
                        "hidden_route_keys": ["image-layer"],
                        "visible_layers": [
                            {
                                "route_key": "roi-layer",
                                "title": "ROIs",
                                "visible": True,
                                "selected": True,
                            }
                        ],
                        "errors": [],
                        "warnings": [],
                    }
                ],
            }
        ],
    }

    rendered = dev_client.McpDevCommandSpec.for_name("isolate-viewer").render_response(
        response,
        args,
    )

    assert (
        "Viewer isolation: observed=True applied=True selected=roi-layer "
        "changed=2 layers=2"
    ) in rendered
    assert "Position: axes=channel,y,x current_step=[1, 0, 0]" in rendered
    assert "Visible: roi-layer" in rendered
    assert "Hidden: image-layer" in rendered
    assert '- roi-layer: visible=True selected=True title="ROIs"' in rendered


def test_mcp_dev_client_launches_fresh_current_source_server():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    async def call_health_through_dev_client():
        args = dev_client._build_parser().parse_args(
            ("call", "openhcs_health_check", "--json", "--timeout-seconds", "5")
        )
        return await dev_client.McpDevCommandSpec.for_name("call").run(
            dev_client.McpDevServerSpec(sys.executable),
            args,
        )

    payload = asyncio.run(call_health_through_dev_client())
    payload = dev_client.to_jsonable(payload)
    result = payload["results"][0]
    health_payload = result["payloads"][0]

    assert result["tool"] == "openhcs_health_check"
    assert result["mcp_error"] is False
    assert health_payload["status"] == "ok"
    assert health_payload["server_source_changed_since_import"] is False


def test_mcp_dev_client_server_spec_preserves_gui_session_environment(monkeypatch):
    import openhcs.mcp.dev_client as dev_client

    for key in dev_client.McpDevServerSpec.gui_environment_keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XAUTHORITY", "/tmp/test.Xauthority")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/dbus-test")
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/test-data-home")
    monkeypatch.setenv("OPENHCS_UNRELATED_TEST_VALUE", "ignored")

    environment = dev_client.McpDevServerSpec(sys.executable).environment()

    assert environment["DISPLAY"] == ":0"
    assert environment["XAUTHORITY"] == "/tmp/test.Xauthority"
    assert environment["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/tmp/dbus-test"
    assert environment["XDG_DATA_HOME"] == "/tmp/test-data-home"
    assert "OPENHCS_UNRELATED_TEST_VALUE" not in environment
    assert dev_client.McpDevServerSpec(sys.executable).process_args() == (
        "-m",
        "openhcs.mcp",
        "--surface",
        "full",
    )


def test_mcp_dev_client_reports_startup_transport_failure():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    async def call_missing_server_module():
        args = dev_client._build_parser().parse_args(
            ("call", "openhcs_health_check", "--json", "--timeout-seconds", "2")
        )
        return await dev_client.McpDevCommandSpec.for_name("call").run(
            dev_client.McpDevServerSpec(
                sys.executable,
                module_name="openhcs.mcp_missing",
            ),
            args,
        )

    payload = asyncio.run(call_missing_server_module())
    payload = dev_client.to_jsonable(payload)
    error = payload["errors"][0]

    assert payload["server"]["module"] == "openhcs.mcp_missing"
    assert payload["results"] == []
    assert error["code"] == "mcp_transport_failed"
    assert error["phase"] == "initialize"
    assert error["causes"]


def test_mcp_dev_client_transport_failure_projects_leaf_causes():
    import openhcs.mcp.dev_client as dev_client

    failure = dev_client.McpDevTransportFailure.from_exception(
        dev_client.McpDevClientPhase.INITIALIZE,
        ExceptionGroup("outer", (TimeoutError("init timed out"),)),
        server_stderr_tail="captured server log",
    )

    error = dev_client.to_jsonable(failure)

    assert error["exception_type"] == "ExceptionGroup"
    assert error["causes"] == [
        {
            "exception_type": "TimeoutError",
            "message": "init timed out",
        }
    ]
    assert error["server_stderr_tail"] == "captured server log"


def test_mcp_stdio_bootstrap_failure_keeps_transport_open(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    (tmp_path / "sitecustomize.py").write_text(
        "\n".join(
            (
                "import openhcs.mcp.server as openhcs_mcp_server",
                "",
                "def fail_build_server(**kwargs):",
                "    del kwargs",
                "    raise RuntimeError('stdio construction failed')",
                "",
                "openhcs_mcp_server.build_server = fail_build_server",
            )
        )
    )
    pythonpath_parts = [str(tmp_path)]
    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath is not None:
        pythonpath_parts.append(current_pythonpath)

    async def call_stdio_server():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=("-m", "openhcs.mcp"),
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            },
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=5)
                health = await asyncio.wait_for(
                    session.call_tool("openhcs_health_check", {}),
                    timeout=5,
                )
                failure = await asyncio.wait_for(
                    session.call_tool("openhcs_bootstrap_failure", {}),
                    timeout=5,
                )
                return health, failure

    health, failure = asyncio.run(call_stdio_server())
    health_payload = json.loads(_direct_tool_text(health))
    failure_payload = json.loads(_direct_tool_text(failure))

    assert health_payload["schema_version"] == "openhcs.mcp.bootstrap.v1"
    assert health_payload["ok"] is False
    assert health_payload["status"] == "unavailable"
    assert health_payload["phase"] == "build_server"
    assert health_payload["message"] == "stdio construction failed"
    assert failure_payload == health_payload


def test_mcp_bootstrap_failure_server_reports_startup_exception():
    if importlib.util.find_spec("mcp") is None:
        return

    async def call_bootstrap_failure_tool():
        built = bootstrap.build_bootstrap_failure_server(RuntimeError("startup failed"))
        return await asyncio.wait_for(
            built.call_tool("openhcs_health_check", {}),
            timeout=2,
        )

    result = asyncio.run(call_bootstrap_failure_tool())
    payload = json.loads(_direct_tool_text(result))

    assert payload["schema_version"] == "openhcs.mcp.bootstrap.v1"
    assert payload["ok"] is False
    assert payload["status"] == "unavailable"
    assert payload["service"] == "openhcs.mcp"
    assert payload["phase"] == "build_server"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["message"] == "startup failed"


def test_mcp_bootstrap_wraps_server_construction_failure(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    def fail_build_server(**kwargs):
        del kwargs
        raise RuntimeError("construction failed")

    monkeypatch.setattr(server, "build_server", fail_build_server)

    async def call_bootstrap_failure_tool():
        built = bootstrap.build_bootstrapped_server()
        return await asyncio.wait_for(
            built.call_tool("openhcs_bootstrap_failure", {}),
            timeout=2,
        )

    result = asyncio.run(call_bootstrap_failure_tool())
    payload = json.loads(_direct_tool_text(result))

    assert payload["schema_version"] == "openhcs.mcp.bootstrap.v1"
    assert payload["ok"] is False
    assert payload["status"] == "unavailable"
    assert payload["service"] == "openhcs.mcp"
    assert payload["phase"] == "build_server"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["message"] == "construction failed"


def test_mcp_bootstrap_wraps_server_run_failure(monkeypatch):
    phases: list[bootstrap.McpBootstrapFailurePhase] = []
    messages: list[str] = []
    failure_server_runs: list[bool] = []
    transports: list[str] = []

    class FailingRunServer:
        def run(self, *, transport: str) -> None:
            transports.append(transport)
            raise RuntimeError("run failed")

    class FailureReportServer:
        def run(self, *, transport: str) -> None:
            transports.append(transport)
            failure_server_runs.append(True)

    def build_failure_server(
        exception: BaseException,
        phase: bootstrap.McpBootstrapFailurePhase = (
            bootstrap.McpBootstrapFailurePhase.BUILD_SERVER
        ),
    ) -> FailureReportServer:
        phases.append(phase)
        messages.append(str(exception))
        return FailureReportServer()

    monkeypatch.setattr(
        bootstrap,
        "build_bootstrapped_server",
        FailingRunServer,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_bootstrap_failure_server",
        build_failure_server,
    )

    bootstrap.run_bootstrapped_server()

    assert phases == [bootstrap.McpBootstrapFailurePhase.RUN_SERVER]
    assert messages == ["run failed"]
    assert failure_server_runs == [True]
    assert transports == ["stdio", "stdio"]


def test_mcp_bootstrap_main_quiets_info_logs_and_restores_logging(monkeypatch):
    observed_disable_levels: list[int] = []
    original_disable_level = logging.root.manager.disable

    def record_run(_surface_profile) -> None:
        observed_disable_levels.append(logging.root.manager.disable)

    monkeypatch.delenv(bootstrap.MCP_VERBOSE_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(bootstrap, "run_bootstrapped_server", record_run)

    bootstrap.main(["--surface", "core"])

    assert observed_disable_levels == [logging.INFO]
    assert logging.root.manager.disable == original_disable_level


def test_mcp_tool_adapter_returns_error_payload_instead_of_raising(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    monkeypatch.setattr(server, "_mcp_server_stale_source_paths", tuple)

    async def call_bad_tools():
        built = server.build_server()
        calls = (
            (
                "openhcs_ui_bridge_status",
                {"connection": {"timeout_ms": 120_000}},
            ),
            (
                "openhcs_probe_viewer_window",
                {"port": 1, "timeout_ms": 120_000},
            ),
        )
        results = []
        for tool_name, arguments in calls:
            result = await asyncio.wait_for(
                built.call_tool(tool_name, arguments),
                timeout=2,
            )
            results.append((tool_name, result))
        return tuple(results)

    results = asyncio.run(call_bad_tools())

    for tool_name, result in results:
        payload = json.loads(_direct_tool_text(result))
        assert payload["schema_version"] == "openhcs.agent.v1"
        assert payload["ok"] is False
        assert payload["tool"] == tool_name
        assert payload["errors"][0]["code"] == "mcp_tool_failed"
        assert payload["errors"][0]["exception_type"] == "ValueError"


def test_mcp_server_exposes_execution_session_tools():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools
    tool_names = {tool.name for tool in tools}

    assert "openhcs_create_orchestrator_session" in tool_names
    assert "openhcs_create_orchestrator_session_from_pipeline_source" in tool_names
    assert "openhcs_inspect_pipeline_source_artifact_plan" in tool_names
    assert "openhcs_submit_compile" in tool_names
    assert "openhcs_submit_pipeline_execution" in tool_names
    assert "openhcs_get_execution_status" in tool_names
    assert "openhcs_viewer_snapshot_window" in tool_names
    assert "openhcs_get_viewer_window_state" in tool_names
    assert "openhcs_get_viewer_window_payloads" in tool_names
    assert "openhcs_sample_viewer_window_image" in tool_names
    assert "openhcs_summarize_viewer_window_rois" in tool_names
    assert "openhcs_navigate_viewer_window" in tool_names
    assert "openhcs_isolate_viewer_window_layers" in tool_names
    assert "openhcs_probe_viewer_window" in tool_names
    assert "openhcs_validate_viewer_window_state" in tool_names
    assert "openhcs_ui_get_object_state_fields" in tool_names


def test_execution_capabilities_distinguish_headless_and_ui_owned_runs():
    capabilities = {
        capability.name: capability
        for capability in get_capability_registry().capabilities
    }

    assert (
        "headless execution session"
        in capabilities["openhcs_create_orchestrator_session"].description
    )
    assert (
        "does not update the running UI PlateManager"
        in capabilities["openhcs_submit_pipeline_execution"].description
    )
    assert (
        "ObjectState snapshots"
        in capabilities["openhcs_ui_selected_plate_workflow"].description
    )


def test_viewer_capabilities_advertise_payload_coordinate_validation():
    capabilities = {
        capability.name: capability
        for capability in get_capability_registry().capabilities
    }

    state_capability = capabilities["openhcs_get_viewer_window_state"]
    payload_capability = capabilities["openhcs_get_viewer_window_payloads"]
    sample_capability = capabilities["openhcs_sample_viewer_window_image"]
    roi_capability = capabilities["openhcs_summarize_viewer_window_rois"]
    navigation_capability = capabilities["openhcs_navigate_viewer_window"]
    isolate_capability = capabilities["openhcs_isolate_viewer_window_layers"]
    validation_capability = capabilities["openhcs_validate_viewer_window_state"]

    assert "viewer_payload_summaries" in state_capability.data_exposure
    assert "viewer_shape_bounds" in state_capability.data_exposure
    assert "per-axis image and shape payload records" in payload_capability.description
    assert "viewer_payload_records" in payload_capability.data_exposure
    assert payload_capability.output_type == "ViewerWindowPayloadResult"
    assert "bounded pixel samples" in sample_capability.description
    assert "viewer_array_values" in sample_capability.data_exposure
    assert "ROI counts" in roi_capability.description
    assert "viewer_roi_statistics" in roi_capability.data_exposure
    assert navigation_capability.output_type == "ViewerWindowNavigationResult"
    assert "mutates_viewer_window_state" in navigation_capability.side_effects
    assert isolate_capability.output_type == "ViewerWindowLayerIsolationResult"
    assert "mutates_viewer_window_state" in isolate_capability.side_effects
    assert "viewer_coordinate_coverage" in validation_capability.data_exposure
    assert "viewer_payload_spatial_compatibility" in validation_capability.data_exposure
    assert "routed coordinate coverage" in validation_capability.description


def test_object_state_capabilities_advertise_resolved_previews():
    capabilities = {
        capability.name: capability
        for capability in get_capability_registry().capabilities
    }

    widget_tree = capabilities["openhcs_ui_get_widget_tree"]
    scope_list = capabilities["openhcs_ui_list_object_state_scopes"]
    field_list = capabilities["openhcs_ui_get_object_state_fields"]
    field_help = capabilities["openhcs_ui_describe_object_state_field"]

    assert "object_state_resolved_value_previews" in widget_tree.data_exposure
    assert "object_state_resolved_value_previews" in scope_list.data_exposure
    assert "object_state_field_provenance" in field_list.data_exposure
    assert "dirty/default markers" in field_list.description
    assert "parameter_descriptions" in field_help.data_exposure
    assert "docstrings" in field_help.data_exposure


def test_mcp_server_exposes_ui_bridge_tools():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools
    tool_names = {tool.name for tool in tools}

    assert "openhcs_ui_bridge_status" in tool_names
    assert "openhcs_ui_list_state_surfaces" in tool_names
    assert "openhcs_ui_get_state_surface" in tool_names
    assert "openhcs_ui_get_code_document" in tool_names
    assert "openhcs_ui_apply_code_document" in tool_names
    assert "openhcs_ui_close_window" in tool_names
    assert "openhcs_ui_snapshot_window" in tool_names
    assert "openhcs_ui_get_widget_tree" in tool_names
    assert "openhcs_ui_describe_object_state_field" in tool_names
    assert "openhcs_ui_selected_plate_workflow" in tool_names
    assert "openhcs_ui_restore_snapshot" in tool_names
    assert "openhcs_ui_get_operation_status" in tool_names
    assert "openhcs_ui_wait_for_operation" in tool_names
    wait_tool = next(
        tool for tool in tools if tool.name == "openhcs_ui_wait_for_operation"
    )
    wait_properties = wait_tool.inputSchema["properties"]
    assert set(wait_properties) == {
        "operation_id",
        "timeout_seconds",
        "poll_interval_seconds",
        "connection",
    }
    assert wait_properties["timeout_seconds"]["default"] == 30.0
    assert wait_properties["poll_interval_seconds"]["default"] == 0.5


def test_mcp_ui_bridge_timeout_policy_is_fail_fast():
    assert server.McpUiBridgeTimeoutPolicy.resolve(None) == 750
    assert server.McpUiBridgeTimeoutPolicy.resolve(2000) == 2000

    try:
        server.McpUiBridgeTimeoutPolicy.resolve(120_000)
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("large UI bridge MCP timeout was accepted")


def test_mcp_ui_bridge_command_timeout_policy_uses_existing_control_cap():
    assert server.McpUiBridgeCommandTimeoutPolicy.resolve(None) == 2000
    assert server.McpUiBridgeCommandTimeoutPolicy.resolve(750) == 750

    try:
        server.McpUiBridgeCommandTimeoutPolicy.resolve(120_000)
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("large UI bridge command MCP timeout was accepted")


def test_mcp_dev_client_ui_connection_timeout_fails_before_mcp_call():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    valid_args = parser.parse_args(("selected-plate-files", "--timeout-ms", "2000"))
    valid_call = dev_client._calls_from_args(valid_args)[0]

    assert valid_call.arguments["connection"]["timeout_ms"] == 2000

    invalid_args = parser.parse_args(("selected-plate-files", "--timeout-ms", "2001"))
    with pytest.raises(dev_client.McpDevCliUsageError, match="--timeout-ms: .*2000ms"):
        dev_client._calls_from_args(invalid_args)


def test_mcp_viewer_timeout_policy_is_fail_fast():
    assert server.McpViewerTimeoutPolicy.resolve(None) == 750
    assert server.McpViewerTimeoutPolicy.resolve(2000) == 2000

    try:
        server.McpViewerTimeoutPolicy.resolve(120_000)
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("large viewer MCP timeout was accepted")


def test_mcp_dev_client_viewer_connection_timeout_fails_before_mcp_call():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    valid_args = parser.parse_args(("validate-viewer", "5555", "--timeout-ms", "2000"))
    valid_call = dev_client._calls_from_args(valid_args)[0]

    assert valid_call.arguments["timeout_ms"] == 2000

    invalid_args = parser.parse_args(
        ("validate-viewer", "5555", "--timeout-ms", "2001")
    )
    with pytest.raises(dev_client.McpDevCliUsageError, match="--timeout-ms: .*2000ms"):
        dev_client._calls_from_args(invalid_args)


def test_mcp_dev_client_viewer_payload_timeout_fails_before_mcp_call():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()
    valid_args = parser.parse_args(("viewer-payloads", "5555", "--timeout-ms", "2000"))
    valid_call = dev_client._calls_from_args(valid_args)[0]

    assert valid_call.arguments["timeout_ms"] == 2000

    invalid_args = parser.parse_args(
        ("viewer-payloads", "5555", "--timeout-ms", "2001")
    )
    with pytest.raises(dev_client.McpDevCliUsageError, match="--timeout-ms: .*2000ms"):
        dev_client._calls_from_args(invalid_args)


def test_mcp_viewer_command_timeout_policy_uses_existing_control_cap():
    assert server.McpViewerCommandTimeoutPolicy.resolve(None) == 2000
    assert server.McpViewerCommandTimeoutPolicy.resolve(750) == 750

    try:
        server.McpViewerCommandTimeoutPolicy.resolve(120_000)
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("large viewer command MCP timeout was accepted")


def test_mcp_viewer_connection_fields_project_timeout_policy():
    fields = server.McpViewerConnectionToolFields(
        port=5555,
        host="localhost",
        transport_mode=None,
        timeout_ms=None,
    )

    assert fields.to_control_args().timeout_ms == 750
    assert (
        fields.to_control_args(server.McpViewerCommandTimeoutPolicy).timeout_ms == 2000
    )


def test_mcp_viewer_mutation_tools_default_to_command_timeout():
    if importlib.util.find_spec("mcp") is None:
        return

    class _ViewerWindowService:
        def __init__(self):
            self.navigation_requests = []
            self.state_requests = []

        def navigate_window(self, request):
            self.navigation_requests.append(request)
            return ViewerWindowNavigationResult(
                schema_version=SCHEMA_VERSION,
                connection=request.connection,
                observed=True,
                viewer=ViewerWindowDescriptor(
                    viewer_type="napari",
                    title="OpenHCS Napari Visualization",
                ),
                route_key=request.navigation.route_key,
                visible=request.navigation.visible,
                selected=request.navigation.selected,
            )

        def window_state(self, request):
            self.state_requests.append(request)
            return ViewerWindowStateResult(
                schema_version=SCHEMA_VERSION,
                connection=request.connection,
                observed=True,
                viewer=ViewerWindowDescriptor(
                    viewer_type="napari",
                    title="OpenHCS Napari Visualization",
                ),
                layer_count=2,
                layers=(
                    ViewerWindowLayerState(
                        route_key="image-layer",
                        title="Image",
                        mounted=True,
                        item_count=1,
                        visible=True,
                        selected=False,
                    ),
                    ViewerWindowLayerState(
                        route_key="roi-layer",
                        title="ROI",
                        mounted=True,
                        item_count=1,
                        visible=True,
                        selected=True,
                    ),
                ),
            )

    viewer_window_service = _ViewerWindowService()
    built = server.build_server(_viewer_mcp_context(viewer_window_service))

    async def call_viewer_mutation_tools():
        await asyncio.wait_for(
            built.call_tool(
                "openhcs_navigate_viewer_window",
                {"port": 5555, "route_key": "roi-layer"},
            ),
            timeout=2,
        )
        await asyncio.wait_for(
            built.call_tool(
                "openhcs_isolate_viewer_window_layers",
                {
                    "port": 5555,
                    "visible_route_keys": ["image-layer", "roi-layer"],
                    "selected_route_key": "roi-layer",
                },
            ),
            timeout=2,
        )

    asyncio.run(call_viewer_mutation_tools())

    assert viewer_window_service.navigation_requests
    assert all(
        request.timeout_ms == 2000
        for request in viewer_window_service.navigation_requests
    )
    assert viewer_window_service.state_requests
    assert all(
        request.timeout_ms == 2000 for request in viewer_window_service.state_requests
    )


def test_mcp_isolate_viewer_reports_applied_when_final_state_times_out():
    if importlib.util.find_spec("mcp") is None:
        return

    class _ViewerWindowService:
        def __init__(self):
            self.state_calls = 0
            self.navigation_requests = []

        def navigate_window(self, request):
            self.navigation_requests.append(request)
            return ViewerWindowNavigationResult(
                schema_version=SCHEMA_VERSION,
                connection=request.connection,
                observed=True,
                viewer=ViewerWindowDescriptor(
                    viewer_type="napari",
                    title="OpenHCS Napari Visualization",
                ),
                route_key=request.navigation.route_key,
                visible=request.navigation.visible,
                selected=request.navigation.selected,
            )

        def window_state(self, request):
            self.state_calls += 1
            if self.state_calls > 1:
                return ViewerWindowStateResult.from_error(
                    connection=request.connection,
                    error=AgentError(
                        code="viewer_window_state_failed",
                        message="Viewer control request timed out.",
                    ),
                )
            return ViewerWindowStateResult(
                schema_version=SCHEMA_VERSION,
                connection=request.connection,
                observed=True,
                viewer=ViewerWindowDescriptor(
                    viewer_type="napari",
                    title="OpenHCS Napari Visualization",
                ),
                layer_count=2,
                layers=(
                    ViewerWindowLayerState(
                        route_key="image-layer",
                        title="Image",
                        mounted=True,
                        item_count=1,
                        visible=True,
                        selected=True,
                    ),
                    ViewerWindowLayerState(
                        route_key="roi-layer",
                        title="ROI",
                        mounted=True,
                        item_count=1,
                        visible=True,
                        selected=False,
                    ),
                ),
            )

    viewer_window_service = _ViewerWindowService()
    built = server.build_server(_viewer_mcp_context(viewer_window_service))

    async def call_isolate_viewer():
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_isolate_viewer_window_layers",
                {
                    "port": 5555,
                    "visible_route_keys": ["roi-layer"],
                    "selected_route_key": "roi-layer",
                },
            ),
            timeout=2,
        )

    result = asyncio.run(call_isolate_viewer())
    payload = json.loads(_direct_tool_text(result))

    assert payload["applied"] is True
    assert payload["observed"] is False
    assert payload["selected_route_key"] == "roi-layer"
    assert payload["visible_route_keys"] == ["roi-layer"]
    assert payload["hidden_route_keys"] == ["image-layer"]
    assert payload["changed_route_count"] == 2
    assert payload["layer_count"] == 2
    assert payload["errors"][0]["code"] == "viewer_window_state_failed"
    assert len(viewer_window_service.navigation_requests) == 2
