"""DTOs for the OpenHCS running-UI bridge agent API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pyqt_reactive.services.widget_tree_projection_config import (
    WidgetNodeIdentity,
    WidgetTreeProjectionControls,
)

from openhcs.core.selection import (
    SelectedAllSelectionMode as UiCodeDocumentSelectionMode,
    SelectedScopeIdsCarrier,
    SelectionModeCarrier,
)
from openhcs.agent.dto.common import (
    AgentError,
    AgentResourceRef,
    AgentResultEnvelope,
    AgentTimedStatusEnvelope,
    AgentWarning,
    JsonObject,
    JsonValue,
)
from openhcs.runtime.window_snapshot import WindowSnapshotCaptureSpec
from openhcs.agent.dto.execution import (
    ExecutionConnectionProjection,
    ExecutionConnectionSpec,
)


UI_BRIDGE_DEFAULT_HOST = "localhost"
UI_BRIDGE_UNKNOWN_OPERATION = "unknown"
UI_BRIDGE_UNKNOWN_WIDGET = "unknown"


class UiCodeDocumentId(str, Enum):
    PLATE_MANAGER_ORCHESTRATOR = "plate_manager.orchestrator_config"


class UiStateSurfaceId(str, Enum):
    PLATE_MANAGER = "plate_manager.state"


class UiWidgetId(str, Enum):
    PLATE_MANAGER = "plate_manager"


class UiSelectedPlateWorkflowKind(str, Enum):
    """Agent-facing workflow commands for the current PlateManager selection."""

    INIT = "init_plate"
    COMPILE = "compile_plate"
    RUN = "run_plate"


class UiBridgeOperationStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


class UiBridgeConnectionDefault:
    """Formal defaults for sparse UI bridge connection overlays."""

    @staticmethod
    def host(value: str | None) -> str:
        if value is None:
            return UI_BRIDGE_DEFAULT_HOST
        return value

    @staticmethod
    def persistent(value: bool | None) -> bool:
        if value is None:
            return True
        return value


@dataclass(frozen=True, kw_only=True)
class UiBridgeInstanceIdentity:
    bridge_instance_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class UiBridgeDescriptorFileRef:
    descriptor_file_path: str | None = None


@dataclass(frozen=True, kw_only=True)
class UiCodeDocumentIdentity:
    document_id: str

    def as_document_identity(self) -> "UiCodeDocumentIdentity":
        if type(self) is UiCodeDocumentIdentity:
            return self
        return UiCodeDocumentIdentity(document_id=self.document_id)


@dataclass(frozen=True, kw_only=True)
class UiStateSurfaceIdentity:
    surface_id: str

    def as_surface_identity(self) -> "UiStateSurfaceIdentity":
        if type(self) is UiStateSurfaceIdentity:
            return self
        return UiStateSurfaceIdentity(surface_id=self.surface_id)


@dataclass(frozen=True, kw_only=True)
class UiWidgetIdentity:
    widget_id: str


@dataclass(frozen=True, slots=True)
class UiSemanticAddress:
    """Stable semantic address shared by ObjectState, windows, and code surfaces."""

    object_state_scope_id: str
    field_path: str
    window_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class UiActionIdentity(UiWidgetIdentity):
    action_id: str


@dataclass(frozen=True, kw_only=True)
class UiWindowIdentity:
    window_id: str


@dataclass(frozen=True, kw_only=True)
class UiObjectStateScopeIdentity:
    object_state_scope_id: str


@dataclass(frozen=True, slots=True)
class UiObjectStateScopeVisibility:
    include_system_scopes: bool = False


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldListOptions:
    include_fields: bool = False
    field_limit: int = 200
    field_offset: int = 0


@dataclass(frozen=True, kw_only=True)
class UiSnapshotIdentity:
    snapshot_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class UiCurrentSnapshotState:
    current_snapshot: "UiSnapshotRef | None" = None


@dataclass(frozen=True, kw_only=True)
class UiCodeDocumentCurrentRevision:
    current_revision_token: str | None = None


@dataclass(frozen=True, kw_only=True)
class UiCodeDocumentBaseRevision:
    base_revision_token: str


@dataclass(frozen=True, kw_only=True)
class UiCodeDocumentOptionalBaseRevision:
    base_revision_token: str | None = None


@dataclass(frozen=True, kw_only=True)
class UiTimeTravelRuntimeState:
    active: bool = False


@dataclass(frozen=True, slots=True)
class UiBridgeConfirmationRequirement:
    """Explicit confirmation guard state for UI bridge mutations."""

    required: bool = True

    @classmethod
    def from_flag(cls, required: bool) -> "UiBridgeConfirmationRequirement":
        return cls(required=required)


@dataclass(frozen=True, kw_only=True)
class UiBridgeConfirmationRequirementCarrier:
    """Shared confirmation policy behavior for UI bridge mutations."""

    confirmation_requirement: UiBridgeConfirmationRequirement = (
        UiBridgeConfirmationRequirement()
    )

    def confirmation_is_required(self) -> bool:
        return self.confirmation_requirement.required


@dataclass(frozen=True, slots=True)
class UiBridgeConnectionFields(
    ExecutionConnectionSpec,
    UiBridgeInstanceIdentity,
    UiBridgeDescriptorFileRef,
):
    """Sparse connection-field projection for descriptor/env/MCP inputs."""

    connection_fields: tuple[str, ...] = ()
    timeout_ms: int | None = None
    auth_token: str | None = None

    @classmethod
    def from_values(
        cls,
        *,
        host: str | None = None,
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool | None = None,
        timeout_ms: int | None = None,
        auth_token: str | None = None,
        descriptor_file_path: str | None = None,
        bridge_instance_id: str | None = None,
    ) -> "UiBridgeConnectionFields":
        connection_fields = tuple(
            name
            for name, value in (
                ("host", host),
                ("port", port),
                ("transport_mode", transport_mode),
                ("persistent", persistent),
            )
            if value is not None
        )
        return cls(
            host=UiBridgeConnectionDefault.host(host),
            port=port,
            transport_mode=transport_mode,
            persistent=UiBridgeConnectionDefault.persistent(persistent),
            connection_fields=connection_fields,
            timeout_ms=timeout_ms,
            auth_token=auth_token,
            descriptor_file_path=descriptor_file_path,
            bridge_instance_id=bridge_instance_id,
        )

    @classmethod
    def from_descriptor(
        cls,
        descriptor: "UiBridgeDescriptorFile",
    ) -> "UiBridgeConnectionFields":
        return cls.from_values(
            host=descriptor.host,
            port=descriptor.port,
            transport_mode=descriptor.transport_mode,
            auth_token=descriptor.auth_token,
            descriptor_file_path=descriptor.descriptor_file_path,
            bridge_instance_id=descriptor.bridge_instance_id,
        )


@dataclass(frozen=True, slots=True)
class UiBridgeConnectionRequest(UiBridgeConnectionFields):
    """MCP-facing sparse connection request for a running UI bridge."""


@dataclass(frozen=True, slots=True)
class UiBridgeConnectionSpec(
    ExecutionConnectionSpec,
    UiBridgeInstanceIdentity,
    UiBridgeDescriptorFileRef,
):
    timeout_ms: int = 5000
    auth_token: str | None = None

    @classmethod
    def from_fields(
        cls,
        fields: UiBridgeConnectionFields,
        defaults: "UiBridgeConnectionSpec | None" = None,
    ) -> "UiBridgeConnectionSpec":
        if defaults is None:
            defaults = cls()
        connection_fields = set(fields.connection_fields)

        def connection_field(field_name: str, field_value, default_value):
            if field_name in connection_fields:
                return field_value
            return default_value

        return cls(
            host=connection_field(
                "host",
                fields.host,
                defaults.host,
            ),
            port=connection_field(
                "port",
                fields.port,
                defaults.port,
            ),
            transport_mode=connection_field(
                "transport_mode",
                fields.transport_mode,
                defaults.transport_mode,
            ),
            persistent=connection_field(
                "persistent",
                fields.persistent,
                defaults.persistent,
            ),
            timeout_ms=(
                fields.timeout_ms
                if fields.timeout_ms is not None
                else defaults.timeout_ms
            ),
            auth_token=(
                fields.auth_token
                if fields.auth_token is not None
                else defaults.auth_token
            ),
            descriptor_file_path=(
                fields.descriptor_file_path
                if fields.descriptor_file_path is not None
                else defaults.descriptor_file_path
            ),
            bridge_instance_id=(
                fields.bridge_instance_id
                if fields.bridge_instance_id is not None
                else defaults.bridge_instance_id
            ),
        )


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorEnvelope(UiBridgeInstanceIdentity):
    """Shared token-bearing descriptor metadata for UI bridge descriptor forms."""

    schema_version: str
    bridge_protocol_version: str
    pid: int
    started_at_unix: float
    auth_token: str


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorFile(
    ExecutionConnectionProjection,
    UiBridgeDescriptorEnvelope,
    UiBridgeDescriptorFileRef,
):
    """Internal token-bearing descriptor-file payload."""


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorWirePayload(
    ExecutionConnectionProjection,
    UiBridgeDescriptorEnvelope,
):
    """JSON descriptor file payload written by a running UI bridge."""

    @classmethod
    def from_descriptor(
        cls,
        descriptor: UiBridgeDescriptorFile,
    ) -> "UiBridgeDescriptorWirePayload":
        return cls(
            schema_version=descriptor.schema_version,
            bridge_protocol_version=descriptor.bridge_protocol_version,
            bridge_instance_id=descriptor.bridge_instance_id,
            pid=descriptor.pid,
            started_at_unix=descriptor.started_at_unix,
            connection=descriptor.connection,
            auth_token=descriptor.auth_token,
        )


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorSummary(
    AgentTimedStatusEnvelope,
    ExecutionConnectionProjection,
    UiBridgeInstanceIdentity,
    UiBridgeDescriptorFileRef,
):
    pid: int


@dataclass(frozen=True, slots=True)
class UiBridgeCatalog:
    schema_version: str
    bridges: tuple[UiBridgeDescriptorSummary, ...] = ()
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiBridgeStatus(
    ExecutionConnectionProjection,
    UiBridgeInstanceIdentity,
    UiBridgeDescriptorFileRef,
):
    schema_version: str
    reachable: bool
    service: str = "openhcs.ui_bridge"
    auth_required: bool = True
    supported_operations: tuple[str, ...] = ()
    provider_catalog_schema_versions: tuple[str, ...] = ()
    bridge_features: tuple[str, ...] = ()
    descriptor_status: str | None = None
    descriptors: tuple[UiBridgeDescriptorSummary, ...] = ()
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiSnapshotRef(UiSnapshotIdentity):
    schema_version: str
    index: int
    branch: str
    parent_snapshot_id: str | None
    timestamp_unix: float
    timestamp: str
    label: str
    num_states: int
    is_current: bool
    is_head: bool
    uri: str


@dataclass(frozen=True, slots=True)
class UiBranchRef:
    schema_version: str
    name: str
    head_snapshot_id: str
    base_snapshot_id: str
    description: str
    is_current: bool


@dataclass(frozen=True, slots=True)
class UiCodeDocumentSummary(UiWidgetIdentity):
    schema_version: str
    identity: UiCodeDocumentIdentity
    title: str
    readable: bool
    writable: bool
    supported_selection_modes: tuple[str, ...] = ()
    current_selection_count: int = 0
    total_scope_count: int = 0

    @property
    def document_id(self) -> str:
        return self.identity.document_id


@dataclass(frozen=True, slots=True)
class UiCodeDocumentCatalog:
    schema_version: str
    documents: tuple[UiCodeDocumentSummary, ...]
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiStateSurfaceSummary(UiWidgetIdentity):
    schema_version: str
    identity: UiStateSurfaceIdentity
    title: str
    readable: bool
    supported_selection_modes: tuple[str, ...] = ()
    current_selection_count: int = 0
    total_scope_count: int = 0

    @property
    def surface_id(self) -> str:
        return self.identity.surface_id


@dataclass(frozen=True, slots=True)
class UiStateSurfaceCatalog:
    schema_version: str
    surfaces: tuple[UiStateSurfaceSummary, ...]
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiCatalogPageMetadata:
    """Bounded catalog page metadata for growing UI bridge listings."""

    limit: int
    offset: int = 0
    returned_count: int = 0
    total_count: int | None = None
    truncated: bool = False
    next_offset: int | None = None


@dataclass(frozen=True, kw_only=True)
class UiCatalogPageCarrier:
    """Inherited carrier for catalog responses that expose a nested page object."""

    page: UiCatalogPageMetadata | None = None


@dataclass(frozen=True, kw_only=True)
class UiFieldCatalogPageCarrier:
    """Inherited carrier for field-list pagination nested under a scope summary."""

    field_page: UiCatalogPageMetadata | None = None


@dataclass(frozen=True, slots=True)
class UiPageRequest:
    """Shared page request for UI bridge catalogs."""

    limit: int = 100
    offset: int = 0


@dataclass(frozen=True, slots=True)
class UiCodeDocumentRequest(UiCodeDocumentIdentity, SelectionModeCarrier):
    clean: bool = True


@dataclass(frozen=True, slots=True)
class UiStateSurfaceRequest(
    UiStateSurfaceIdentity,
    SelectionModeCarrier,
    UiCodeDocumentOptionalBaseRevision,
):
    pass


@dataclass(frozen=True, slots=True)
class UiCodeDocument(
    UiCodeDocumentCurrentRevision,
    UiCurrentSnapshotState,
    SelectedScopeIdsCarrier,
):
    schema_version: str
    summary: UiCodeDocumentSummary
    source: str
    mime_type: str
    size_bytes: int
    sha256: str
    warnings: tuple[AgentWarning, ...] = ()
    errors: tuple[AgentError, ...] = ()


@dataclass(frozen=True, slots=True)
class UiPlateManagerRowState:
    plate_scope_id: str
    name: str
    plate_root: str
    cppipe_path: str | None
    selected: bool
    initialized: bool
    compiled: bool
    init_pending: bool
    compile_pending: bool
    execution_active: bool
    status_prefix: str
    orchestrator_state: str | None
    execution_id: str | None
    terminal_status: str | None
    runtime_state: str | None
    runtime_percent: float | None
    queue_position: int | None


@dataclass(frozen=True, kw_only=True)
class UiStateSurfaceEnvelope(AgentResultEnvelope):
    """Shared envelope for typed UI state-surface payloads."""

    summary: UiStateSurfaceSummary
    unchanged: bool = False


@dataclass(frozen=True, slots=True)
class UiPlateManagerState(
    UiStateSurfaceEnvelope,
    UiCodeDocumentCurrentRevision,
    UiCurrentSnapshotState,
    SelectedScopeIdsCarrier,
):
    object_state_token: int
    manager_execution_state: str
    rows: tuple[UiPlateManagerRowState, ...]


@dataclass(frozen=True, slots=True)
class UiStateSurfaceDocument(
    UiStateSurfaceEnvelope,
    UiCodeDocumentCurrentRevision,
    UiCurrentSnapshotState,
    SelectedScopeIdsCarrier,
):
    payload_schema: str
    payload: JsonObject


class UiActionInvocationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class UiMutationRequestToken:
    value: str | None = None


@dataclass(frozen=True, kw_only=True)
class UiMutationRequestTokenCarrier:
    """Shared request-token carrier for bridge mutation requests."""

    request_token: UiMutationRequestToken = UiMutationRequestToken()


@dataclass(frozen=True, slots=True)
class UiMutationReceipt:
    request_token: UiMutationRequestToken
    bridge_operation_id: str | None = None
    accepted: bool = False


@dataclass(frozen=True, slots=True)
class UiActionSummary(SelectionModeCarrier):
    schema_version: str
    identity: UiActionIdentity
    title: str
    enabled: bool
    invocation_mode: str
    side_effects: tuple[str, ...] = ()
    confirmation_required: bool = False
    current_selection_count: int = 0
    target_scope_ids: tuple[str, ...] = ()
    selection_revision_token: str | None = None
    related_state_surface_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UiActionCatalog(UiCatalogPageCarrier):
    schema_version: str
    actions: tuple[UiActionSummary, ...]
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiActionInvokeRequest(
    UiActionIdentity,
    SelectedScopeIdsCarrier,
    UiBridgeConfirmationRequirementCarrier,
    UiMutationRequestTokenCarrier,
):
    observed_selection_revision_token: str | None = None


@dataclass(frozen=True, slots=True)
class UiActionInvokeResult:
    schema_version: str
    identity: UiActionIdentity
    status: str
    receipt: UiMutationReceipt
    target_scope_ids: tuple[str, ...] = ()
    selection_revision_token: str | None = None
    workflow_status_surface_ids: tuple[str, ...] = ()
    recommended_poll_interval_ms: int = 500
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiWindowManagerScope:
    """WindowManager scope token associated with a UI window identity."""

    value: str

    @classmethod
    def from_identity(cls, identity: UiWindowIdentity) -> "UiWindowManagerScope":
        return cls(value=identity.window_id)


@dataclass(frozen=True, slots=True)
class UiWindowSummary:
    schema_version: str
    identity: UiWindowIdentity
    title: str
    window_kind: str
    visible: bool
    focusable: bool
    manager_scope: UiWindowManagerScope | None = None

    @property
    def window_id(self) -> str:
        return self.identity.window_id


@dataclass(frozen=True, slots=True)
class UiWindowCatalog(AgentResultEnvelope, UiCatalogPageCarrier):
    windows: tuple[UiWindowSummary, ...]


@dataclass(frozen=True, slots=True)
class UiWindowOpenPolicy:
    """Policy for materializing a UI window before operating on it."""

    create_if_missing: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWindowOperationRequest(UiWindowIdentity):
    """Shared request contract for operations against a UI window."""

    open_policy: UiWindowOpenPolicy


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWindowFocusRequest(UiWindowOperationRequest):
    pass


@dataclass(frozen=True, slots=True)
class UiWindowFocusResult(AgentResultEnvelope, UiWindowIdentity):
    focused: bool
    summary: UiWindowSummary | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWindowNavigateRequest(UiWindowOperationRequest):
    item_id: str | None = None
    field_path: str | None = None


@dataclass(frozen=True, slots=True)
class UiWindowNavigateResult(AgentResultEnvelope, UiWindowIdentity):
    focused: bool
    navigated: bool
    created: bool
    summary: UiWindowSummary | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWindowCloseRequest(UiWindowIdentity):
    """Request a normal close for one currently open UI window."""

    pass


@dataclass(frozen=True, slots=True)
class UiWindowCloseResult(AgentResultEnvelope, UiWindowIdentity):
    closed: bool
    summary: UiWindowSummary | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWindowSnapshotRequest(
    WindowSnapshotCaptureSpec,
    UiWindowOperationRequest,
):
    pass


@dataclass(frozen=True, slots=True)
class UiWindowSnapshotResult(
    WindowSnapshotCaptureSpec,
    AgentResultEnvelope,
    UiWindowIdentity,
):
    captured: bool
    resource: AgentResourceRef | None = None
    summary: UiWindowSummary | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWidgetTreeRequest(WidgetTreeProjectionControls, UiWindowOperationRequest):
    pass


@dataclass(frozen=True, slots=True)
class UiWidgetRect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class UiWidgetTreeNode(WidgetNodeIdentity):
    visible: bool
    enabled: bool
    geometry: UiWidgetRect
    global_geometry: UiWidgetRect
    tool_tip: str
    status_tip: str
    whats_this: str
    window_title: str
    text: str | None
    text_truncated: bool
    title: str | None
    action_kinds: tuple[str, ...]
    clickable: bool
    actionable: bool
    checkable: bool | None
    checked: bool | None
    current_index: int | None
    current_text: str | None
    item_count: int | None
    children: tuple["UiWidgetTreeNode", ...]


@dataclass(frozen=True, slots=True)
class UiWidgetTreeResult(AgentResultEnvelope, UiWindowIdentity):
    projected: bool
    root: UiWidgetTreeNode | None = None
    summary: UiWindowSummary | None = None
    widget_count: int = 0
    actionable_count: int = 0


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldProvenance:
    """Resolved source address for an inherited ObjectState field."""

    source_scope_id: str | None
    source_type: str | None
    source_field_path: str | None = None


@dataclass(frozen=True, kw_only=True)
class UiObjectStateFieldProvenanceCarrier:
    """Inherited carrier for field summaries with resolved provenance."""

    provenance: UiObjectStateFieldProvenance | None = None


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldSummary(UiObjectStateFieldProvenanceCarrier):
    """Field-level ObjectState semantics without exposing raw field values."""

    schema_version: str
    address: UiSemanticAddress
    field_name: str
    container_path: str
    raw_value_type: str
    resolved_value_type: str | None
    dirty: bool
    signature_diff: bool
    last_changed: bool


@dataclass(frozen=True, slots=True)
class UiObjectStateScopeSummary(UiFieldCatalogPageCarrier):
    schema_version: str
    identity: UiObjectStateScopeIdentity
    object_type: str
    parameter_count: int
    dirty_field_count: int
    signature_diff_field_count: int
    last_changed_field: str | None = None
    registered: bool = True
    fields: tuple[UiObjectStateFieldSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class UiObjectStateScopeCatalog(
    AgentResultEnvelope,
    UiTimeTravelRuntimeState,
    UiCatalogPageCarrier,
):
    object_state_token: int
    current_branch: str
    current_snapshot_index: int
    scopes: tuple[UiObjectStateScopeSummary, ...]


@dataclass(frozen=True, slots=True)
class UiObjectStateScopeListRequest(UiObjectStateScopeVisibility):
    include_fields: bool = False
    field_limit: int = 200
    field_offset: int = 0

    @classmethod
    def from_visibility_options(
        cls,
        visibility: UiObjectStateScopeVisibility,
        field_options: UiObjectStateFieldListOptions,
    ) -> "UiObjectStateScopeListRequest":
        return cls(
            include_system_scopes=visibility.include_system_scopes,
            include_fields=field_options.include_fields,
            field_limit=field_options.field_limit,
            field_offset=field_options.field_offset,
        )


@dataclass(frozen=True, slots=True)
class UiCodeDocumentValidationRequest(
    UiCodeDocumentIdentity,
    UiCodeDocumentOptionalBaseRevision,
):
    source: str


@dataclass(frozen=True, slots=True)
class UiCodeDocumentValidationResult(UiCodeDocumentIdentity):
    schema_version: str
    valid: bool
    normalized_scope_ids: tuple[str, ...] = ()
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, kw_only=True)
class UiBridgeConfirmationPolicy(UiBridgeConfirmationRequirementCarrier):
    """Shared mutation confirmation policy for UI bridge requests."""


@dataclass(frozen=True, kw_only=True)
class UiBridgeBranchMutationPolicy(UiBridgeConfirmationPolicy):
    """Shared branch creation policy for ObjectState movement requests."""

    allow_auto_branch: bool = False


@dataclass(frozen=True, slots=True)
class UiCodeDocumentApplyRequest(
    UiCodeDocumentIdentity,
    UiCodeDocumentBaseRevision,
    UiBridgeConfirmationPolicy,
):
    source: str
    snapshot_label: str | None = None
    apply_if_time_traveling: bool = False


@dataclass(frozen=True, slots=True)
class UiCodeDocumentApplyResult(UiCodeDocumentIdentity):
    schema_version: str
    applied: bool
    base_revision_token: str
    outcome: str = "not_applied"
    operation_id: str | None = None
    new_revision_token: str | None = None
    current_revision_token: str | None = None
    current_snapshot: UiSnapshotRef | None = None
    undo_snapshot: UiSnapshotRef | None = None
    pre_apply_snapshot: UiSnapshotRef | None = None
    post_apply_snapshot: UiSnapshotRef | None = None
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiSelectedPlateWorkflowRequest(
    SelectedScopeIdsCarrier,
    UiBridgeConfirmationRequirementCarrier,
    UiMutationRequestTokenCarrier,
):
    workflow: UiSelectedPlateWorkflowKind
    observed_selection_revision_token: str | None = None


@dataclass(frozen=True, slots=True)
class UiSelectedPlateWorkflowResult:
    schema_version: str
    workflow: UiSelectedPlateWorkflowKind
    action_result: UiActionInvokeResult
    state_surface_id: str = UiStateSurfaceId.PLATE_MANAGER.value
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiSnapshotCatalog(UiTimeTravelRuntimeState):
    schema_version: str
    current_branch: str
    current_snapshot_index: int
    object_state_token: int
    snapshots: tuple[UiSnapshotRef, ...]
    branches: tuple[UiBranchRef, ...]
    warnings: tuple[AgentWarning, ...] = ()
    errors: tuple[AgentError, ...] = ()


@dataclass(frozen=True, slots=True)
class UiSnapshotListRequest(UiObjectStateScopeVisibility):
    pass


@dataclass(frozen=True, slots=True)
class UiSnapshotRestoreRequest(
    UiSnapshotIdentity,
    UiBridgeBranchMutationPolicy,
    UiObjectStateScopeVisibility,
):
    index: int | None = None
    branch: str | None = None


@dataclass(frozen=True, slots=True)
class UiTimeTravelHeadRequest(UiBridgeConfirmationPolicy):
    pass


@dataclass(frozen=True, slots=True)
class UiSnapshotRestoreResult(UiCurrentSnapshotState):
    schema_version: str
    restored: bool
    target_snapshot: UiSnapshotRef | None
    catalog: UiSnapshotCatalog | None = None
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiBranchCatalog:
    schema_version: str
    current_branch: str
    branches: tuple[UiBranchRef, ...]
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class UiBranchSwitchRequest(UiBridgeBranchMutationPolicy):
    branch: str

    def as_restore_request(self) -> UiSnapshotRestoreRequest:
        return UiSnapshotRestoreRequest(
            branch=self.branch,
            confirmation_requirement=self.confirmation_requirement,
            allow_auto_branch=self.allow_auto_branch,
        )


@dataclass(frozen=True, slots=True)
class UiBridgeOperationRoute:
    operation_name: str
    request_id: str | None = None
    target_id: str | None = None


UNKNOWN_UI_BRIDGE_OPERATION_ROUTE = UiBridgeOperationRoute(
    operation_name=UI_BRIDGE_UNKNOWN_OPERATION,
)


@dataclass(frozen=True, slots=True)
class UiBridgeOperationIdentity:
    operation_id: str
    route: UiBridgeOperationRoute

    @property
    def operation_name(self) -> str:
        return self.route.operation_name

    @property
    def request_id(self) -> str | None:
        return self.route.request_id

    @property
    def target_id(self) -> str | None:
        return self.route.target_id


@dataclass(frozen=True, slots=True)
class UiBridgeOperationStatusRequest:
    operation_id: str


@dataclass(frozen=True, slots=True)
class UiBridgeOperationRef(AgentTimedStatusEnvelope):
    identity: UiBridgeOperationIdentity
    completed_at_unix: float | None = None
    outcome: str | None = None


@dataclass(frozen=True, slots=True)
class UiBridgeEnvelopeBase:
    schema_version: str
    bridge_protocol_version: str
    request_id: str
    payload: JsonObject


@dataclass(frozen=True, slots=True)
class UiBridgeRequestEnvelope(UiBridgeEnvelopeBase):
    operation: str
    auth_token: str | None


@dataclass(frozen=True, slots=True)
class UiBridgeResponseEnvelope(UiBridgeEnvelopeBase):
    ok: bool
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()
