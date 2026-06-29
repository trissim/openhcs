from openhcs.agent.capabilities import (
    AgentCapabilitySpec,
    CapabilityCliConnectionProfile,
    CapabilityKind,
    get_capability_registry,
    validate_capability_registry,
)


def test_capability_registry_declares_schema_and_unique_names():
    registry = get_capability_registry()

    names = [capability.name for capability in registry.capabilities]

    assert registry.schema_version == "openhcs.agent.v1"
    assert len(names) == len(set(names))


def test_health_capability_declares_mcp_reliability_contract():
    registry = get_capability_registry()
    capabilities = {capability.name: capability for capability in registry.capabilities}

    health = capabilities["openhcs_health_check"]

    assert health.service == "capability_registry"
    assert "process identity" in health.description
    assert "source freshness" in health.description
    assert health.data_exposure == (
        "mcp_process_identity",
        "mcp_source_freshness",
    )
    assert health.output_type == "McpServerHealthResult"


def test_mutating_tools_must_declare_side_effects():
    mutating_tool = AgentCapabilitySpec(
        name="openhcs_create_something",
        kind=CapabilityKind.TOOL,
        title="Create something",
        description="Mutation without side-effect metadata.",
        service="test",
        mutating=True,
    )

    try:
        validate_capability_registry((mutating_tool,))
    except ValueError as exc:
        assert "side_effects" in str(exc)
    else:
        raise AssertionError("mutating tools without side effects must fail")


def test_ui_bridge_capabilities_declare_runtime_security_and_data_exposure():
    registry = get_capability_registry()
    capabilities = {capability.name: capability for capability in registry.capabilities}

    status = capabilities["openhcs_ui_bridge_status"]
    read_document = capabilities["openhcs_ui_get_code_document"]
    apply_document = capabilities["openhcs_ui_apply_code_document"]
    selected_workflow = capabilities["openhcs_ui_selected_plate_workflow"]
    widget_tree = capabilities["openhcs_ui_get_widget_tree"]
    restore_snapshot = capabilities["openhcs_ui_restore_snapshot"]
    operation_status = capabilities["openhcs_ui_get_operation_status"]

    assert status.runtime_requirements == ("running_openhcs_ui_bridge",)
    assert read_document.runtime_requirements == ("running_openhcs_ui_bridge",)
    assert read_document.data_exposure == ("local_paths_in_source",)
    assert read_document.security_requirements == ("ui_bridge_auth_token",)
    assert apply_document.side_effects == ("mutates_running_ui_state",)
    assert apply_document.output_type == "UiCodeDocumentApplyResult"
    assert apply_document.data_exposure == (
        "local_paths_in_source",
        "ui_revision_tokens",
        "object_state_snapshot_refs",
        "object_state_undo_targets",
    )
    assert selected_workflow.side_effects == (
        "may_mutate_running_ui_state",
        "may_start_ui_workflow",
    )
    assert widget_tree.output_type == "UiWidgetTreeResult"
    assert "ui_clickable_geometry" in widget_tree.data_exposure
    assert "ui_widget_enabled_state" in widget_tree.data_exposure
    assert "ui_action_kinds" in widget_tree.data_exposure
    assert "action kinds" in widget_tree.description
    assert restore_snapshot.side_effects == (
        "mutates_running_ui_state",
        "time_travels_ui_state",
    )
    assert operation_status.side_effects == ()


def test_viewer_probe_capability_declares_compact_liveness_contract():
    registry = get_capability_registry()
    capabilities = {capability.name: capability for capability in registry.capabilities}

    probe = capabilities["openhcs_probe_viewer_window"]

    assert probe.service == "viewer_window"
    assert probe.runtime_requirements == ("running_openhcs_viewer_server",)
    assert probe.data_exposure == ("viewer_identity", "viewer_layer_counts")
    assert probe.output_type == "ViewerWindowProbeResult"


def test_cli_connection_profiles_are_declared_on_capabilities():
    registry = get_capability_registry()
    profiles = {
        capability.name: capability.cli_connection_profile
        for capability in registry.capabilities
        if capability.cli_command is not None
    }

    assert profiles["openhcs_ui_bridge_status"] is (
        CapabilityCliConnectionProfile.UI_BRIDGE
    )
    assert profiles["openhcs_ui_get_widget_tree"] is (
        CapabilityCliConnectionProfile.UI_BRIDGE
    )
    assert profiles["openhcs_get_viewer_window_payloads"] is (
        CapabilityCliConnectionProfile.VIEWER_WINDOW
    )
    assert profiles["openhcs_validate_viewer_window_state"] is (
        CapabilityCliConnectionProfile.VIEWER_WINDOW
    )
    assert profiles["openhcs_get_runtime_server_info"] is (
        CapabilityCliConnectionProfile.RUNTIME_SERVER
    )
    assert profiles["openhcs_scan_runtime_servers"] is (
        CapabilityCliConnectionProfile.RUNTIME_SERVER
    )
    assert profiles["openhcs_health_check"] is CapabilityCliConnectionProfile.DIRECT
