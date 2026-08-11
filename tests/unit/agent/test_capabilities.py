from openhcs.agent.capabilities import (
    AgentCapabilitySpec,
    AgentDataclassRequestServiceInvocation,
    CapabilityCliConnectionProfile,
    CapabilityKind,
    CapabilityRole,
    CapabilityTargetContext,
    CapabilityVisibility,
    CapabilityWorkflowGroup,
    CapabilityWorkflowStage,
    CapabilityTransport,
    AuthoringLocalCapabilitySurfaceProfile,
    CoreLocalCapabilitySurfaceProfile,
    DesktopLocalCapabilitySurfaceProfile,
    FullLocalCapabilitySurfaceProfile,
    HeadlessExecutionCapability,
    HostedTransportCapabilityMixin,
    PipelineDraftCapability,
    PlatePathCapability,
    RuntimeServerCliConnectionCapability,
    UiBridgeCapability,
    ViewerWindowCliConnectionCapability,
    agent_capability_declarations,
    get_capability_registry,
    validate_capability_registry,
)
from openhcs.agent.dto.pipeline import CreatePipelineRequest


def test_capability_registry_declares_schema_and_unique_names():
    registry = get_capability_registry()

    names = [capability.name for capability in registry.capabilities]

    assert registry.schema_version == "openhcs.agent.v1"
    assert len(names) == len(set(names))


def test_source_session_capability_owns_progress_heartbeat_policy():
    capabilities = {
        capability.name: capability
        for capability in get_capability_registry().capabilities
    }

    source_session = capabilities[
        "openhcs_create_orchestrator_session_from_pipeline_source"
    ]

    assert source_session.progress_heartbeat_seconds == 10.0
    assert source_session.progress_worker_thread_safe is False
    assert source_session.as_jsonable()["progress_heartbeat_seconds"] == 10.0
    assert source_session.as_jsonable()["progress_worker_thread_safe"] is False

    function_search = capabilities["openhcs_search_functions"]
    assert function_search.progress_heartbeat_seconds == 5.0
    assert function_search.progress_worker_thread_safe is True


def test_create_pipeline_capability_accepts_optional_pipeline_config_reference():
    declarations = {
        declaration.name: declaration for declaration in agent_capability_declarations()
    }
    create_pipeline = declarations["openhcs_create_pipeline"]

    assert create_pipeline.input_contract is CreatePipelineRequest
    assert isinstance(
        create_pipeline.request_invocation,
        AgentDataclassRequestServiceInvocation,
    )


def test_capability_transport_defaults_to_local_stdio():
    capability = AgentCapabilitySpec(
        name="openhcs_local_default",
        kind=CapabilityKind.TOOL,
        title="Local default",
        description="Transport default test.",
        service="test",
    )

    assert capability.transport_availability == (CapabilityTransport.LOCAL_STDIO,)
    assert capability.supports_transport(CapabilityTransport.LOCAL_STDIO)
    assert not capability.supports_transport(CapabilityTransport.HOSTED_STREAMABLE_HTTP)


def test_hosted_capabilities_are_nominal_opt_ins_and_read_only():
    declarations = agent_capability_declarations()
    hosted_declarations = tuple(
        declaration
        for declaration in declarations
        if CapabilityTransport.HOSTED_STREAMABLE_HTTP
        in declaration.transport_availability
    )

    assert hosted_declarations
    assert all(
        issubclass(declaration, HostedTransportCapabilityMixin)
        for declaration in hosted_declarations
    )
    assert all(declaration.to_spec().read_only for declaration in hosted_declarations)
    assert all(not declaration.requires_network for declaration in hosted_declarations)
    assert all(
        not declaration.security_requirements for declaration in hosted_declarations
    )


def test_local_runtime_capability_families_are_not_hosted():
    local_runtime_roots = (
        PlatePathCapability,
        PipelineDraftCapability,
        HeadlessExecutionCapability,
        UiBridgeCapability,
        ViewerWindowCliConnectionCapability,
        RuntimeServerCliConnectionCapability,
    )

    for declaration in agent_capability_declarations():
        if issubclass(declaration, local_runtime_roots):
            assert CapabilityTransport.HOSTED_STREAMABLE_HTTP not in (
                declaration.transport_availability
            )


def test_transport_filtered_registry_rebuilds_groups_from_eligible_declarations():
    hosted = get_capability_registry(CapabilityTransport.HOSTED_STREAMABLE_HTTP)

    assert hosted.capabilities
    assert all(
        capability.supports_transport(CapabilityTransport.HOSTED_STREAMABLE_HTTP)
        for capability in hosted.capabilities
    )
    grouped_names = {
        capability_name
        for group in hosted.groups
        for capability_name in group.capability_names
    }
    assert grouped_names == {capability.name for capability in hosted.capabilities}


def test_local_surface_profiles_filter_declaration_metadata_without_name_lists():
    profiles = (
        FullLocalCapabilitySurfaceProfile(),
        DesktopLocalCapabilitySurfaceProfile(),
        AuthoringLocalCapabilitySurfaceProfile(),
        CoreLocalCapabilitySurfaceProfile(),
    )
    full_registry = get_capability_registry()

    for profile in profiles:
        registry = get_capability_registry(
            capability_surface_profile=profile,
        )
        assert registry.surface_profile == profile.name
        assert {capability.name for capability in registry.capabilities} == {
            capability.name
            for capability in full_registry.capabilities
            if profile.includes(capability)
        }
        assert all(profile.includes(capability) for capability in registry.capabilities)
        assert {
            capability_name
            for group in registry.groups
            for capability_name in group.capability_names
        } == {capability.name for capability in registry.capabilities}

    desktop = get_capability_registry(
        capability_surface_profile=DesktopLocalCapabilitySurfaceProfile(),
    )
    desktop_names = {capability.name for capability in desktop.capabilities}
    assert "openhcs_ui_bridge_status" in desktop_names
    assert "openhcs_get_viewer_window_state" in desktop_names
    assert "openhcs_describe_config_schema" in desktop_names
    assert "openhcs_register_custom_function" in desktop_names
    recovery_names = {
        "openhcs_ui_list_snapshots",
        "openhcs_ui_restore_snapshot",
        "openhcs_ui_time_travel_head",
        "openhcs_ui_list_branches",
        "openhcs_ui_switch_branch",
    }
    assert recovery_names <= desktop_names
    assert "openhcs_ui_wait_for_operation_receipt" in desktop_names
    assert "openhcs_ui_wait_for_operation" not in desktop_names
    assert "openhcs_create_orchestrator_session" in desktop_names
    assert "openhcs_submit_pipeline_execution" in desktop_names
    assert "openhcs_ui_invoke_widget_action" not in desktop_names

    desktop_capabilities = {
        capability.name: capability for capability in desktop.capabilities
    }
    for recovery_name in recovery_names:
        recovery = desktop_capabilities[recovery_name]
        assert recovery.visibility is CapabilityVisibility.STANDARD
        assert recovery.role is CapabilityRole.PRIMARY

    core = get_capability_registry(
        capability_surface_profile=CoreLocalCapabilitySurfaceProfile(),
    )
    core_names = {capability.name for capability in core.capabilities}
    assert "openhcs_create_orchestrator_session" in core_names
    assert "openhcs_stream_plate_files_to_viewer" not in core_names
    assert "openhcs_ui_bridge_status" not in core_names


def test_desktop_distribution_extras_derive_from_selected_capabilities():
    profile = DesktopLocalCapabilitySurfaceProfile()
    registry = get_capability_registry(capability_surface_profile=profile)
    required_extras = tuple(
        dict.fromkeys(
            extra
            for capability in registry.capabilities
            for extra in capability.required_extras
        )
    )

    assert profile.distribution_extras(registry.capabilities) == (
        "gui",
        "mcp",
        "viz",
    )
    assert "viz" in required_extras


def test_custom_function_registration_is_a_desktop_authoring_mutation():
    full_capabilities = {
        capability.name: capability
        for capability in get_capability_registry().capabilities
    }
    registration = full_capabilities["openhcs_register_custom_function"]

    assert registration.workflow_group is CapabilityWorkflowGroup.FUNCTION_AUTHORING
    assert registration.workflow_stage is CapabilityWorkflowStage.AUTHORING
    assert registration.target_context is CapabilityTargetContext.FUNCTION_REGISTRY
    assert registration.visibility is CapabilityVisibility.STANDARD
    assert registration.role is CapabilityRole.PRIMARY
    assert registration.mutating is True
    assert registration.side_effects == (
        "writes_custom_function_file",
        "updates_function_registry",
    )
    assert registration.read_only is False
    assert registration.supports_transport(CapabilityTransport.LOCAL_STDIO)
    assert not registration.supports_transport(
        CapabilityTransport.HOSTED_STREAMABLE_HTTP
    )
    assert registration.supports_surface_profile(
        DesktopLocalCapabilitySurfaceProfile()
    )
    assert registration.supports_surface_profile(
        AuthoringLocalCapabilitySurfaceProfile()
    )


def test_health_capability_declares_mcp_reliability_contract():
    registry = get_capability_registry()
    capabilities = {capability.name: capability for capability in registry.capabilities}

    health = capabilities["openhcs_health_check"]

    assert health.service == "capability_registry"
    assert "process identity" in health.description
    assert "source freshness" in health.description
    assert "installation-generation freshness" in health.description
    assert "client-owned reconnect contract" in health.description
    assert health.data_exposure == (
        "installed_openhcs_version",
        "packaged_resource_readiness",
        "packaged_resource_paths",
        "mcp_process_identity",
        "mcp_source_freshness",
        "mcp_installation_generation",
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


def test_capability_registry_projects_non_read_only_tools_from_declarations():
    registry = get_capability_registry()

    assert registry.non_read_only_tools
    assert all(
        capability.kind is CapabilityKind.TOOL and not capability.read_only
        for capability in registry.non_read_only_tools
    )


def test_tool_capabilities_declare_group_target_and_role_metadata():
    registry = get_capability_registry()

    tools = [
        capability
        for capability in registry.capabilities
        if capability.kind is CapabilityKind.TOOL
    ]

    assert tools
    assert all(capability.workflow_group is not None for capability in tools)
    assert all(capability.workflow_stage is not None for capability in tools)
    assert all(capability.target_context is not None for capability in tools)
    assert all(capability.visibility is not None for capability in tools)
    assert all(capability.role is not None for capability in tools)


def test_capability_registry_groups_are_generated_from_declarations():
    registry = get_capability_registry()

    grouped_names = {
        capability_name
        for group in registry.groups
        for capability_name in group.capability_names
    }
    registry_names = {capability.name for capability in registry.capabilities}

    assert grouped_names == registry_names
    assert [group.workflow_group for group in registry.groups] == [
        workflow_group
        for workflow_group in CapabilityWorkflowGroup
        if any(
            capability.workflow_group is workflow_group
            for capability in registry.capabilities
        )
    ]


def test_similar_mcp_tool_names_are_disambiguated_by_target_context_and_role():
    registry = get_capability_registry()
    capabilities = {capability.name: capability for capability in registry.capabilities}

    assert capabilities["openhcs_sample_plate_image"].target_context is (
        CapabilityTargetContext.PLATE_PATH
    )
    assert capabilities["openhcs_ui_sample_selected_plate_image"].target_context is (
        CapabilityTargetContext.UI_SELECTED_PLATE
    )
    assert capabilities["openhcs_ui_sample_selected_plate_image"].role is (
        CapabilityRole.MODE_VARIANT
    )
    assert capabilities["openhcs_get_execution_status"].target_context is (
        CapabilityTargetContext.SUBMITTED_JOB
    )
    assert (
        capabilities["openhcs_get_runtime_server_execution_status"].target_context
        is CapabilityTargetContext.RUNTIME_SERVER
    )
    assert capabilities["openhcs_ui_get_operation_status"].target_context is (
        CapabilityTargetContext.UI_BRIDGE
    )
    assert capabilities["openhcs_ui_wait_for_operation_receipt"].target_context is (
        CapabilityTargetContext.UI_BRIDGE
    )
    assert (
        capabilities["openhcs_ui_wait_for_operation_receipt"].role
        is CapabilityRole.PRIMARY
    )
    assert capabilities["openhcs_ui_invoke_action"].role is CapabilityRole.PRIMARY
    assert capabilities["openhcs_ui_invoke_widget_action"].role is (
        CapabilityRole.FALLBACK
    )
    assert capabilities["openhcs_ui_get_code_document"].workflow_group is (
        CapabilityWorkflowGroup.UI_STATE_EDITING
    )
    assert capabilities["openhcs_ui_get_code_document"].target_context is (
        CapabilityTargetContext.UI_CODE_DOCUMENT
    )
    assert capabilities["openhcs_validate_pipeline"].workflow_stage is (
        CapabilityWorkflowStage.VALIDATION
    )
    assert capabilities["openhcs_ui_mutate_object_state_field"].visibility is (
        CapabilityVisibility.EXPERT
    )


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
    operation_wait = capabilities["openhcs_ui_wait_for_operation_receipt"]

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
    assert operation_wait.side_effects == ()
    assert operation_wait.input_type == "UiBridgeOperationWaitRequest"
    assert operation_wait.output_type == "UiBridgeOperationRef"
    assert operation_wait.title == "Wait for UI bridge mutation receipt"
    assert "dispatch processing only" in operation_wait.description
    assert "does not wait for a compile, run, viewer" in operation_wait.description


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
