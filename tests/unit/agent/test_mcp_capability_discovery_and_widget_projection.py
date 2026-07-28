import asyncio
from dataclasses import fields

from pyqt_reactive.services.widget_tree_projection_config import (
    COMPACT_FIELD_PROJECTION_METADATA_KEY,
    CompactFieldProjection,
    WidgetNodeIdentity,
)

from openhcs.agent.capabilities import (
    AgentCapabilitySearchRequest,
    AgentCapabilitySummary,
    CapabilityKind,
    CapabilityRole,
    CapabilityTargetContext,
    CapabilityTransport,
    CapabilityVisibility,
    CapabilityWorkflowGroup,
    CapabilityWorkflowStage,
    DesktopLocalCapabilitySurfaceProfile,
    get_capability_registry,
)
from openhcs.agent.dto.ui_bridge import UiWidgetActionSummary
import openhcs.mcp.server as server


def _structured_result(result) -> dict:
    return result[1] if isinstance(result, tuple) else result.structuredContent


def test_capability_search_filters_registry_metadata_before_bounded_paging():
    registry = get_capability_registry()
    request = AgentCapabilitySearchRequest(
        workflow_group=CapabilityWorkflowGroup.UI_CONTROL,
        workflow_stage=CapabilityWorkflowStage.CONTROL,
        target_context=CapabilityTargetContext.UI_WINDOW,
        visibility=CapabilityVisibility.STANDARD,
        role=CapabilityRole.PRIMARY,
        has_side_effects=True,
        text="window",
        offset=0,
        limit=2,
    )

    result = registry.search(request)
    matching_specs = tuple(
        capability
        for capability in registry.capabilities
        if request.matches(capability)
    )

    assert result.matched_count == len(matching_specs)
    assert result.returned_count == min(2, len(matching_specs))
    assert result.capabilities == tuple(
        capability.compact_summary() for capability in matching_specs[:2]
    )
    assert all(
        isinstance(capability, AgentCapabilitySummary)
        for capability in result.capabilities
    )
    assert all(capability.side_effects for capability in result.capabilities)
    assert result.next_offset == (
        2 if len(matching_specs) > result.returned_count else None
    )


def test_capability_search_uses_declared_side_effect_and_text_metadata():
    result = get_capability_registry().search(
        AgentCapabilitySearchRequest(
            kind=CapabilityKind.TOOL,
            side_effect_contains="mutates_viewer_window_state",
            text="viewer layer",
        )
    )

    assert result.capabilities
    assert all(
        "mutates_viewer_window_state" in capability.side_effects
        for capability in result.capabilities
    )
    assert all(
        capability.target_context is CapabilityTargetContext.VIEWER_WINDOW
        for capability in result.capabilities
    )


def test_capability_search_rejects_unbounded_pages():
    try:
        AgentCapabilitySearchRequest(
            limit=AgentCapabilitySearchRequest.MAXIMUM_LIMIT + 1
        )
    except ValueError as exc:
        assert str(AgentCapabilitySearchRequest.MAXIMUM_LIMIT) in str(exc)
    else:
        raise AssertionError("Capability search accepted an unbounded page.")


def test_mcp_capability_search_schema_and_results_follow_selected_surface():
    built = server.build_server(
        capability_surface_profile=DesktopLocalCapabilitySurfaceProfile()
    )
    tools = asyncio.run(built.list_tools())
    search_tool = next(
        tool for tool in tools if tool.name == "openhcs_search_capabilities"
    )

    workflow_group_schema = search_tool.inputSchema["properties"]["workflow_group"]
    workflow_group_reference = workflow_group_schema["anyOf"][0]["$ref"]
    workflow_group_definition = workflow_group_reference.rsplit("/", 1)[-1]
    assert search_tool.inputSchema["$defs"][workflow_group_definition]["enum"] == [
        workflow_group.value for workflow_group in CapabilityWorkflowGroup
    ]

    result = asyncio.run(
        built.call_tool(
            "openhcs_search_capabilities",
            {
                "workflow_group": CapabilityWorkflowGroup.VIEWER_REVIEW.value,
                "text": "roi",
                "limit": 4,
            },
        )
    )
    payload = _structured_result(result)

    assert payload["surface_profile"] == DesktopLocalCapabilitySurfaceProfile.name
    assert payload["returned_count"] <= 4
    assert payload["capabilities"]
    assert all(
        capability["workflow_group"] == CapabilityWorkflowGroup.VIEWER_REVIEW.value
        for capability in payload["capabilities"]
    )
    assert all("roi" in capability["description"].casefold() for capability in payload["capabilities"])
    selected_registry = get_capability_registry(
        capability_surface_profile=DesktopLocalCapabilitySurfaceProfile()
    )
    selected_specs = {
        capability.name: capability
        for capability in selected_registry.capabilities
    }
    for capability in payload["capabilities"]:
        declared = selected_specs[capability["name"]]
        assert capability["requires_network"] is declared.requires_network
        assert capability["required_extras"] == list(declared.required_extras)
        assert capability["data_exposure"] == list(declared.data_exposure)
        assert capability["security_requirements"] == list(
            declared.security_requirements
        )


def test_hosted_capability_search_cannot_expose_local_only_declarations():
    hosted_registry = get_capability_registry(
        CapabilityTransport.HOSTED_STREAMABLE_HTTP
    )
    built = server.build_server(
        capability_transport=CapabilityTransport.HOSTED_STREAMABLE_HTTP
    )
    result = asyncio.run(
        built.call_tool(
            "openhcs_search_capabilities",
            {"limit": AgentCapabilitySearchRequest.MAXIMUM_LIMIT},
        )
    )
    payload = _structured_result(result)

    assert {capability["name"] for capability in payload["capabilities"]} == {
        capability.name for capability in hosted_registry.capabilities
    }
    assert payload["matched_count"] == len(hosted_registry.capabilities)
    assert payload["next_offset"] is None


def test_widget_compaction_is_declaration_owned_without_transport_field_tables():
    assert "McpWidgetTreePayloadProjection" not in vars(server)
    assert not hasattr(AgentCapabilitySummary, "from_capability")
    assert [field.name for field in fields(CompactFieldProjection)] == ["includes"]

    declared_projection_fields = tuple(
        declared_field
        for owner in (WidgetNodeIdentity, UiWidgetActionSummary)
        for declared_field in fields(owner)
        if COMPACT_FIELD_PROJECTION_METADATA_KEY in declared_field.metadata
    )
    assert declared_projection_fields
    assert all(
        callable(
            declared_field.metadata[
                COMPACT_FIELD_PROJECTION_METADATA_KEY
            ].includes
        )
        for declared_field in declared_projection_fields
    )
