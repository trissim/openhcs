from openhcs.agent.capabilities import (
    AgentCapabilitySpec,
    CapabilityKind,
    get_capability_registry,
    validate_capability_registry,
)


def test_capability_registry_declares_schema_and_unique_names():
    registry = get_capability_registry()

    names = [capability.name for capability in registry.capabilities]

    assert registry.schema_version == "openhcs.agent.v1"
    assert len(names) == len(set(names))


def test_mutating_tools_must_declare_side_effects():
    mutating_tool = AgentCapabilitySpec(
        name="openhcs_create_something",
        kind=CapabilityKind.TOOL,
        title="Create something",
        description="Mutation without side-effect metadata.",
        service="test",
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
    restore_snapshot = capabilities["openhcs_ui_restore_snapshot"]
    operation_status = capabilities["openhcs_ui_get_operation_status"]

    assert status.runtime_requirements == ("running_openhcs_ui_bridge",)
    assert read_document.runtime_requirements == ("running_openhcs_ui_bridge",)
    assert read_document.data_exposure == ("local_paths_in_source",)
    assert read_document.security_requirements == ("ui_bridge_auth_token",)
    assert apply_document.side_effects == ("mutates_running_ui_state",)
    assert apply_document.data_exposure == ("local_paths_in_source",)
    assert restore_snapshot.side_effects == (
        "mutates_running_ui_state",
        "time_travels_ui_state",
    )
    assert operation_status.side_effects == ()
