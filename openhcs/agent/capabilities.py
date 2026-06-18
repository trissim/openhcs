"""Capability registry for OpenHCS agent integrations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openhcs.agent.dto.common import SCHEMA_VERSION


class CapabilityKind(Enum):
    RESOURCE = "resource"
    TOOL = "tool"
    PROMPT = "prompt"


class AgentContractName(Enum):
    PIPELINE_REF = "PipelineRef"
    ORCHESTRATOR_SESSION_REF = "OrchestratorSessionRef"
    EXECUTION_JOB_REF = "ExecutionJobRef"
    UI_BRIDGE_CATALOG = "UiBridgeCatalog"
    UI_ACTION_CATALOG = "UiActionCatalog"
    UI_ACTION_INVOKE_RESULT = "UiActionInvokeResult"
    UI_WINDOW_CATALOG = "UiWindowCatalog"
    UI_WINDOW_FOCUS_RESULT = "UiWindowFocusResult"
    UI_OBJECT_STATE_SCOPE_CATALOG = "UiObjectStateScopeCatalog"
    UI_SNAPSHOT_RESTORE_RESULT = "UiSnapshotRestoreResult"


class MutatingCapabilityNamePolicy:
    mutation_tokens = frozenset(
        (
            "add",
            "create",
            "delete",
            "execute",
            "compile",
            "mutate",
            "remove",
            "reorder",
            "run",
            "save",
            "submit",
            "update",
            "write",
        )
    )

    def matches(self, name: str) -> bool:
        name_tokens = CapabilityNameTokens(name)
        return any(name_tokens.contains(token) for token in self.mutation_tokens)


@dataclass(frozen=True, slots=True)
class CapabilityNameTokens:
    name: str

    def contains(self, token: str) -> bool:
        return token in self.tokens()

    def tokens(self) -> tuple[str, ...]:
        normalized = (
            self.name.replace("://", "_")
            .replace("/", "_")
            .replace("-", "_")
            .lower()
        )
        return tuple(part for part in normalized.split("_") if part)


MUTATING_CAPABILITY_NAME_POLICY = MutatingCapabilityNamePolicy()


@dataclass(frozen=True, slots=True)
class AgentCapabilitySpec:
    name: str
    kind: CapabilityKind
    title: str
    description: str
    service: str
    side_effects: tuple[str, ...] = ()
    requires_network: bool = False
    required_extras: tuple[str, ...] = ()
    runtime_requirements: tuple[str, ...] = ()
    data_exposure: tuple[str, ...] = ()
    security_requirements: tuple[str, ...] = ()
    input_type: str | None = None
    output_type: str | None = None


@dataclass(frozen=True, slots=True)
class AgentCapabilityRegistry:
    schema_version: str
    capabilities: tuple[AgentCapabilitySpec, ...]


CAPABILITIES: tuple[AgentCapabilitySpec, ...] = (
    AgentCapabilitySpec(
        name="openhcs://capabilities",
        kind=CapabilityKind.RESOURCE,
        title="OpenHCS agent capability registry",
        description="Lists the resources, tools, side effects, and extras exposed by this server.",
        service="capability_registry",
        output_type="AgentCapabilityRegistry",
    ),
    AgentCapabilitySpec(
        name="openhcs_health_check",
        kind=CapabilityKind.TOOL,
        title="Health check",
        description="Reports basic OpenHCS agent server health and schema version.",
        service="capability_registry",
        output_type="dict",
    ),
    AgentCapabilitySpec(
        name="openhcs_list_capabilities",
        kind=CapabilityKind.TOOL,
        title="List capabilities",
        description="Returns the canonical agent capability registry.",
        service="capability_registry",
        output_type="AgentCapabilityRegistry",
    ),
    AgentCapabilitySpec(
        name="openhcs_search_functions",
        kind=CapabilityKind.TOOL,
        title="Search processing functions",
        description="Searches the OpenHCS function registry by name, module, library, tag, or doc text.",
        service="function_catalog",
        input_type="FunctionCatalogQuery",
        output_type="FunctionCatalogPage",
    ),
    AgentCapabilitySpec(
        name="openhcs_describe_function",
        kind=CapabilityKind.TOOL,
        title="Describe processing function",
        description="Returns signature, parameter, and documentation details for one registry function.",
        service="function_catalog",
        input_type="function_id",
        output_type="FunctionDetail",
    ),
    AgentCapabilitySpec(
        name="openhcs_get_authoring_context",
        kind=CapabilityKind.TOOL,
        title="Get authoring context",
        description="Returns bounded prompt/context text for agents authoring OpenHCS code.",
        service="llm_context",
        input_type="AuthoringContextRequest",
        output_type="AuthoringContext",
    ),
    AgentCapabilitySpec(
        name="openhcs://architecture/topics",
        kind=CapabilityKind.RESOURCE,
        title="Architecture topics",
        description="Lists read-only architecture topics backed by real OpenHCS internal symbols.",
        service="architecture_projection",
        output_type="ArchitectureTopicPage",
    ),
    AgentCapabilitySpec(
        name="openhcs_list_architecture_topics",
        kind=CapabilityKind.TOOL,
        title="List architecture topics",
        description="Lists architecture topics available to agents.",
        service="architecture_projection",
        output_type="ArchitectureTopicPage",
    ),
    AgentCapabilitySpec(
        name="openhcs_explain_architecture",
        kind=CapabilityKind.TOOL,
        title="Explain architecture topic",
        description="Explains one OpenHCS architecture topic using source-backed internal API symbols.",
        service="architecture_projection",
        input_type="topic_id",
        output_type="ArchitectureTopic",
    ),
    AgentCapabilitySpec(
        name="openhcs_describe_internal_symbol",
        kind=CapabilityKind.TOOL,
        title="Describe internal symbol",
        description="Returns read-only signature/doc/source-location facts for one internal OpenHCS symbol.",
        service="architecture_projection",
        input_type="symbol_id",
        output_type="InternalApiSymbol",
    ),
    AgentCapabilitySpec(
        name="openhcs_describe_config_schema",
        kind=CapabilityKind.TOOL,
        title="Describe configuration schema",
        description="Reflects GlobalPipelineConfig or PipelineConfig fields without materializing lazy values.",
        service="config",
        input_type="config_type",
        output_type="ConfigSchema",
    ),
    AgentCapabilitySpec(
        name="openhcs_create_config",
        kind=CapabilityKind.TOOL,
        title="Create configuration",
        description="Creates a draft config reference from a typed config patch.",
        service="config",
        side_effects=("creates_in_memory_config_ref",),
        input_type="ConfigPatch",
        output_type="ConfigRef",
    ),
    AgentCapabilitySpec(
        name="openhcs_validate_config_patch",
        kind=CapabilityKind.TOOL,
        title="Validate configuration patch",
        description="Validates that a config patch can instantiate the target OpenHCS config class.",
        service="config",
        input_type="ConfigPatch",
        output_type="ConfigValidationResult",
    ),
    AgentCapabilitySpec(
        name="openhcs_render_config_source",
        kind=CapabilityKind.TOOL,
        title="Render configuration source",
        description="Renders a draft config reference as Python source using OpenHCS pycodify formatters.",
        service="config",
        input_type="ConfigRef",
        output_type="RenderedSource",
    ),
    AgentCapabilitySpec(
        name="openhcs_create_pipeline",
        kind=CapabilityKind.TOOL,
        title="Create draft pipeline",
        description="Creates an in-memory agent-authored OpenHCS pipeline draft.",
        service="pipeline_authoring",
        side_effects=("creates_in_memory_pipeline_ref",),
        output_type=AgentContractName.PIPELINE_REF.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_add_function_step",
        kind=CapabilityKind.TOOL,
        title="Add FunctionStep",
        description="Adds a FunctionStepSpec resolved through the OpenHCS function registry.",
        service="pipeline_authoring",
        side_effects=("mutates_in_memory_pipeline_ref",),
        input_type="FunctionStepSpec",
        output_type="PipelineSpec",
    ),
    AgentCapabilitySpec(
        name="openhcs_validate_pipeline",
        kind=CapabilityKind.TOOL,
        title="Validate draft pipeline",
        description="Validates function references and converts the draft into OpenHCS FunctionStep objects.",
        service="pipeline_authoring",
        input_type=AgentContractName.PIPELINE_REF.value,
        output_type="PipelineValidationResult",
    ),
    AgentCapabilitySpec(
        name="openhcs_render_pipeline_source",
        kind=CapabilityKind.TOOL,
        title="Render pipeline source",
        description="Renders an authored pipeline as Python source using the OpenHCS FunctionStep serializer.",
        service="pipeline_authoring",
        input_type=AgentContractName.PIPELINE_REF.value,
        output_type="RenderedSource",
    ),
    AgentCapabilitySpec(
        name="openhcs_create_orchestrator_session",
        kind=CapabilityKind.TOOL,
        title="Create orchestrator session",
        description="Creates an opaque execution session from a plate path and pipeline draft.",
        service="execution_session",
        side_effects=("creates_in_memory_execution_session",),
        input_type="OrchestratorSessionRequest",
        output_type=AgentContractName.ORCHESTRATOR_SESSION_REF.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_get_orchestrator_session",
        kind=CapabilityKind.TOOL,
        title="Get orchestrator session",
        description="Returns the stored plate, pipeline, config, and ZMQ connection identity for a session.",
        service="execution_session",
        input_type=AgentContractName.ORCHESTRATOR_SESSION_REF.value,
        output_type="OrchestratorSession",
    ),
    AgentCapabilitySpec(
        name="openhcs_submit_compile",
        kind=CapabilityKind.TOOL,
        title="Submit compile job",
        description="Submits a compile-only ZMQ execution job for an execution session.",
        service="execution_session",
        side_effects=("submits_zmq_compile_job",),
        input_type=AgentContractName.ORCHESTRATOR_SESSION_REF.value,
        output_type=AgentContractName.EXECUTION_JOB_REF.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_submit_pipeline_execution",
        kind=CapabilityKind.TOOL,
        title="Submit pipeline execution",
        description="Submits a ZMQ pipeline execution job for an execution session.",
        service="execution_session",
        side_effects=("submits_zmq_execution_job",),
        input_type=AgentContractName.ORCHESTRATOR_SESSION_REF.value,
        output_type=AgentContractName.EXECUTION_JOB_REF.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_get_execution_status",
        kind=CapabilityKind.TOOL,
        title="Get execution status",
        description="Polls the ZMQ server for one submitted compile or execution job.",
        service="execution_session",
        input_type=AgentContractName.EXECUTION_JOB_REF.value,
        output_type="ExecutionJobStatus",
    ),
    AgentCapabilitySpec(
        name="openhcs_scan_runtime_servers",
        kind=CapabilityKind.TOOL,
        title="Scan runtime servers",
        description="Scans candidate ports for running OpenHCS ZMQ execution servers.",
        service="runtime_server",
        input_type="RuntimeServerScanRequest",
        output_type="RuntimeServerScanResult",
    ),
    AgentCapabilitySpec(
        name="openhcs_get_runtime_server_info",
        kind=CapabilityKind.TOOL,
        title="Get runtime server info",
        description="Returns a read-only server snapshot from a running OpenHCS ZMQ execution server.",
        service="runtime_server",
        input_type="ExecutionConnectionSpec",
        output_type="RuntimeServerInfo",
    ),
    AgentCapabilitySpec(
        name="openhcs_get_runtime_server_execution_status",
        kind=CapabilityKind.TOOL,
        title="Get runtime execution status",
        description="Returns raw ZMQ execution status from a running OpenHCS runtime server.",
        service="runtime_server",
        input_type="RuntimeExecutionStatusRequest",
        output_type="RuntimeExecutionStatus",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_list_bridges",
        kind=CapabilityKind.TOOL,
        title="List UI bridges",
        description="Lists local OpenHCS PyQt UI bridge descriptor summaries visible to this user.",
        service="ui_bridge",
        data_exposure=("local_ui_bridge_descriptor_paths",),
        output_type=AgentContractName.UI_BRIDGE_CATALOG.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_bridge_status",
        kind=CapabilityKind.TOOL,
        title="Get UI bridge status",
        description="Reports whether a local running OpenHCS PyQt UI bridge is reachable.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        output_type="UiBridgeStatus",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_list_code_documents",
        kind=CapabilityKind.TOOL,
        title="List UI code documents",
        description="Lists code documents exposed by a running OpenHCS UI bridge.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        output_type="UiCodeDocumentCatalog",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_list_state_surfaces",
        kind=CapabilityKind.TOOL,
        title="List UI state surfaces",
        description="Lists pollable typed state surfaces exposed by a running OpenHCS UI bridge.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        output_type="UiStateSurfaceCatalog",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_get_state_surface",
        kind=CapabilityKind.TOOL,
        title="Get UI state surface",
        description="Reads or polls one typed UI state surface such as plate-manager status rows.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        data_exposure=("local_paths",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiStateSurfaceRequest",
        output_type="UiStateSurfaceDocument",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_list_actions",
        kind=CapabilityKind.TOOL,
        title="List UI actions",
        description="Lists invokable actions exposed by a running OpenHCS UI bridge.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        output_type=AgentContractName.UI_ACTION_CATALOG.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_invoke_action",
        kind=CapabilityKind.TOOL,
        title="Invoke UI action",
        description="Dispatches one running-UI action and returns a receipt; workflow progress is polled through related state surfaces.",
        service="ui_bridge",
        side_effects=("may_mutate_running_ui_state", "may_start_ui_workflow"),
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiActionInvokeRequest",
        output_type=AgentContractName.UI_ACTION_INVOKE_RESULT.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_list_windows",
        kind=CapabilityKind.TOOL,
        title="List UI windows",
        description="Lists visible and focusable windows exposed by a running OpenHCS UI bridge.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        output_type=AgentContractName.UI_WINDOW_CATALOG.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_focus_window",
        kind=CapabilityKind.TOOL,
        title="Focus UI window",
        description="Focuses one running UI window by stable window id or open ObjectState scope id.",
        service="ui_bridge",
        side_effects=("changes_running_ui_focus",),
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiWindowFocusRequest",
        output_type=AgentContractName.UI_WINDOW_FOCUS_RESULT.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_list_object_state_scopes",
        kind=CapabilityKind.TOOL,
        title="List ObjectState scopes",
        description="Lists ObjectState scopes visible to the running OpenHCS UI bridge.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        data_exposure=("object_state_scope_ids", "object_type_names"),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiObjectStateScopeListRequest",
        output_type=AgentContractName.UI_OBJECT_STATE_SCOPE_CATALOG.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_get_code_document",
        kind=CapabilityKind.TOOL,
        title="Get UI code document",
        description="Reads a bounded UI-owned code document such as the plate-manager orchestrator source.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        data_exposure=("local_paths_in_source",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiCodeDocumentRequest",
        output_type="UiCodeDocument",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_validate_code_document",
        kind=CapabilityKind.TOOL,
        title="Validate UI code document",
        description="Validates an edited UI code document through the bridge source policy without mutating UI state.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        data_exposure=("local_paths_in_source",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiCodeDocumentValidationRequest",
        output_type="UiCodeDocumentValidationResult",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_apply_code_document",
        kind=CapabilityKind.TOOL,
        title="Apply UI code document",
        description="Applies an edited UI code document through the running PyQt workflow with revision protection.",
        service="ui_bridge",
        side_effects=("mutates_running_ui_state",),
        runtime_requirements=("running_openhcs_ui_bridge",),
        data_exposure=("local_paths_in_source",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiCodeDocumentApplyRequest",
        output_type="UiCodeDocumentApplyResult",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_list_snapshots",
        kind=CapabilityKind.TOOL,
        title="List UI snapshots",
        description="Lists ObjectState snapshots visible to the running UI bridge.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiSnapshotListRequest",
        output_type="UiSnapshotCatalog",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_restore_snapshot",
        kind=CapabilityKind.TOOL,
        title="Restore UI snapshot",
        description="Restores the running UI to a selected ObjectState snapshot through the bridge.",
        service="ui_bridge",
        side_effects=("mutates_running_ui_state", "time_travels_ui_state"),
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiSnapshotRestoreRequest",
        output_type=AgentContractName.UI_SNAPSHOT_RESTORE_RESULT.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_time_travel_head",
        kind=CapabilityKind.TOOL,
        title="Return UI to current head",
        description="Returns the running UI from ObjectState time travel to the current branch head.",
        service="ui_bridge",
        side_effects=("mutates_running_ui_state", "time_travels_ui_state"),
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiTimeTravelHeadRequest",
        output_type=AgentContractName.UI_SNAPSHOT_RESTORE_RESULT.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_list_branches",
        kind=CapabilityKind.TOOL,
        title="List UI snapshot branches",
        description="Lists ObjectState branches visible to the running UI bridge.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        output_type="UiBranchCatalog",
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_switch_branch",
        kind=CapabilityKind.TOOL,
        title="Switch UI snapshot branch",
        description="Switches the running UI to another ObjectState branch through the bridge.",
        service="ui_bridge",
        side_effects=("mutates_running_ui_state", "time_travels_ui_state"),
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="UiBranchSwitchRequest",
        output_type=AgentContractName.UI_SNAPSHOT_RESTORE_RESULT.value,
    ),
    AgentCapabilitySpec(
        name="openhcs_ui_get_operation_status",
        kind=CapabilityKind.TOOL,
        title="Get UI bridge operation status",
        description="Returns status for an active or recent running-UI bridge operation.",
        service="ui_bridge",
        runtime_requirements=("running_openhcs_ui_bridge",),
        security_requirements=("ui_bridge_auth_token",),
        input_type="operation_id",
        output_type="UiBridgeOperationRef",
    ),
)


def get_capability_registry() -> AgentCapabilityRegistry:
    validate_capability_registry(CAPABILITIES)
    return AgentCapabilityRegistry(
        schema_version=SCHEMA_VERSION,
        capabilities=CAPABILITIES,
    )


def validate_capability_registry(
    capabilities: tuple[AgentCapabilitySpec, ...] = CAPABILITIES,
) -> None:
    """Assert static capability metadata is complete enough for policy checks."""
    seen: set[str] = set()
    for capability in capabilities:
        if capability.name in seen:
            raise ValueError(f"Duplicate OpenHCS agent capability: {capability.name}")
        seen.add(capability.name)
        if (
            capability.kind is CapabilityKind.TOOL
            and MUTATING_CAPABILITY_NAME_POLICY.matches(capability.name)
        ):
            if not capability.side_effects:
                raise ValueError(
                    f"Mutating tool {capability.name!r} must declare side_effects."
                )
