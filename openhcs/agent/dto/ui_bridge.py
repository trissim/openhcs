"""DTOs for the OpenHCS running-UI bridge agent API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum, nonmember
from math import isfinite
from typing import ClassVar, TypeVar, cast

from objectstate import DottedFieldPath
from pyqt_reactive.services.widget_tree_projection_config import (
    COMPACT_FIELD_PROJECTION_METADATA_KEY,
    CompactFieldProjection,
    WidgetNodeIdentity,
    WidgetTreeProjectionControls,
    always_project_compact_field,
    compact_dataclass_projection,
)
from pyqt_reactive.services.window_snapshot import (
    WindowSnapshotCaptureScope,
    WindowSnapshotCaptureSpec,
)
from python_introspect import (
    overlay_non_none_dataclass,
    project_dataclass,
    validate_annotated_dataclass,
)
from zmqruntime import EndpointApplication
from zmqruntime.config import (
    NonBlankString,
    PositiveInteger,
    SocketPort,
    TransportMode,
)

from openhcs.agent.dto.common import (
    AGENT_PARAMETER_DESCRIPTION_METADATA_KEY,
    AGENT_PARAMETER_PRODUCER_OUTPUT_CONTRACT_METADATA_KEY,
    AgentCliRequest,
    AgentError,
    AgentResourceRef,
    AgentResultEnvelope,
    AgentTimedStatusEnvelope,
    AgentWarning,
    JsonObject,
    JsonValue,
)
from openhcs.agent.dto.execution import (
    ExecutionConnectionProjection,
    ExecutionConnectionSpec,
)
from openhcs.agent.path_policy import DEFAULT_AGENT_WINDOW_SNAPSHOT_DIR
from openhcs.agent.ui_bridge_actions import PlateManagerAction
from openhcs.agent.ui_bridge_identities import (
    MainWindowWidgetIdentity as MainWindowWidgetIdentity,
)
from openhcs.agent.ui_bridge_identities import (
    ManagedWindowWidgetIdentity as ManagedWindowWidgetIdentity,
)
from openhcs.agent.ui_bridge_identities import (
    PipelineDebugSessionStateSurfaceIdentityDeclaration as PipelineDebugSessionStateSurfaceIdentityDeclaration,
)
from openhcs.agent.ui_bridge_identities import (
    PipelineEditorStateSurfaceIdentityDeclaration as PipelineEditorStateSurfaceIdentityDeclaration,
)
from openhcs.agent.ui_bridge_identities import (
    PipelineEditorWidgetIdentity as PipelineEditorWidgetIdentity,
)
from openhcs.agent.ui_bridge_identities import (
    PlateManagerOrchestratorCodeDocumentIdentity as PlateManagerOrchestratorCodeDocumentIdentity,
)
from openhcs.agent.ui_bridge_identities import (
    PlateManagerStateSurfaceIdentityDeclaration as PlateManagerStateSurfaceIdentityDeclaration,
)
from openhcs.agent.ui_bridge_identities import (
    PlateManagerWidgetIdentity as PlateManagerWidgetIdentity,
)
from openhcs.agent.ui_bridge_identities import (
    UiBridgeIdentityDeclaration as UiBridgeIdentityDeclaration,
)
from openhcs.agent.ui_bridge_identities import (
    UiCodeDocumentIdentityDeclaration as UiCodeDocumentIdentityDeclaration,
)
from openhcs.agent.ui_bridge_identities import (
    UiLiveOverviewStateSurfaceIdentityDeclaration as UiLiveOverviewStateSurfaceIdentityDeclaration,
)
from openhcs.agent.ui_bridge_identities import (
    UiOwnedByWidgetIdentityDeclaration as UiOwnedByWidgetIdentityDeclaration,
)
from openhcs.agent.ui_bridge_identities import (
    UiStateSurfaceIdentityDeclarationBase as UiStateSurfaceIdentityDeclarationBase,
)
from openhcs.agent.ui_bridge_identities import (
    UiWidgetIdentityDeclaration as UiWidgetIdentityDeclaration,
)
from openhcs.core.progress.live_measurements import LiveMeasurementTablePreview
from openhcs.core.selection import (
    SelectedAllSelectionMode as UiCodeDocumentSelectionMode,
)
from openhcs.core.selection import (
    SelectedScopeIdsArgument,
    SelectedScopeIdsCarrier,
    SelectionModeCarrier,
)
from openhcs.serialization.json import to_jsonable

UI_BRIDGE_UNKNOWN_OPERATION = "unknown"
UI_BRIDGE_UNKNOWN_WIDGET = "unknown"
_UiBridgeOperationSelectionT = TypeVar("_UiBridgeOperationSelectionT")


class UiLiveOverviewSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _identity_enum(
    enum_name: str,
    identity_type: type[UiBridgeIdentityDeclaration],
) -> type[Enum]:
    """Project one public string enum from registered UI identity declarations."""
    members = {
        declaration.enum_member_name: declaration.value
        for declaration in UiBridgeIdentityDeclaration.__registry__.values()
        if issubclass(declaration, identity_type)
        and declaration.enum_member_name is not None
        and declaration.value is not None
    }
    return Enum(enum_name, members, type=str)


def _plate_manager_workflow_enum() -> type[Enum]:
    """Project selected-plate workflows from PlateManager action declarations."""
    members = {
        action.plate_operation.name: action.value
        for action in PlateManagerAction
        if action.plate_operation is not None
    }
    return Enum("UiSelectedPlateWorkflowKind", members, type=str)


UiCodeDocumentId = _identity_enum("UiCodeDocumentId", UiCodeDocumentIdentityDeclaration)
UiStateSurfaceId = _identity_enum(
    "UiStateSurfaceId",
    UiStateSurfaceIdentityDeclarationBase,
)
UiWidgetId = _identity_enum("UiWidgetId", UiWidgetIdentityDeclaration)
UiSelectedPlateWorkflowKind = _plate_manager_workflow_enum()


class UiBridgeOperationStatus(str, Enum):
    """Closed operation state with declaration-owned completion semantics."""

    is_terminal: bool
    live_overview_severity: str
    _completion_selector: Callable[[object, object, object], object]

    _select_active = nonmember(lambda active, _succeeded, _failed: active)
    _select_succeeded = nonmember(lambda _active, succeeded, _failed: succeeded)
    _select_failed = nonmember(lambda _active, _succeeded, failed: failed)

    RUNNING = (
        "running",
        False,
        UiLiveOverviewSeverity.INFO.value,
        _select_active,
    )
    COMPLETED = (
        "completed",
        True,
        UiLiveOverviewSeverity.INFO.value,
        _select_succeeded,
    )
    FAILED = (
        "failed",
        True,
        UiLiveOverviewSeverity.ERROR.value,
        _select_failed,
    )
    NOT_FOUND = (
        "not_found",
        True,
        UiLiveOverviewSeverity.ERROR.value,
        _select_failed,
    )
    UNAVAILABLE = (
        "unavailable",
        True,
        UiLiveOverviewSeverity.ERROR.value,
        _select_failed,
    )

    def __new__(
        cls,
        value: str,
        is_terminal: bool,
        live_overview_severity: str,
        completion_selector: Callable[[object, object, object], object],
    ) -> "UiBridgeOperationStatus":
        member = str.__new__(cls, value)
        member._value_ = value
        member.is_terminal = is_terminal
        member.live_overview_severity = live_overview_severity
        member._completion_selector = completion_selector
        return member

    def select_completion(
        self,
        *,
        active: _UiBridgeOperationSelectionT,
        succeeded: _UiBridgeOperationSelectionT,
        failed: _UiBridgeOperationSelectionT,
    ) -> _UiBridgeOperationSelectionT:
        """Select caller-owned output through this status member's leaf."""

        return cast(
            _UiBridgeOperationSelectionT,
            self._completion_selector(active, succeeded, failed),
        )


@dataclass(frozen=True, kw_only=True)
class UiBridgeInstanceIdentity:
    bridge_instance_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class UiBridgeApplicationIdentity:
    """Application identity observed at a UI bridge boundary."""

    application: EndpointApplication | None = None


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

    def as_identity(self) -> "UiWindowIdentity":
        if type(self) is UiWindowIdentity:
            return self
        return UiWindowIdentity(window_id=self.window_id)


@dataclass(frozen=True, kw_only=True)
class UiObjectStateScopeIdentity:
    object_state_scope_id: str


@dataclass(frozen=True, slots=True)
class UiObjectStateScopeVisibility:
    include_system_scopes: bool = False

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, JsonValue] | None,
    ) -> "UiObjectStateScopeVisibility":
        if value is None:
            return cls()
        include_system_scopes = value.get("include_system_scopes")
        if isinstance(include_system_scopes, bool):
            return cls(include_system_scopes=include_system_scopes)
        return cls()


class UiObjectStateFieldFilter(str, Enum):
    """Agent-facing filters over existing ObjectState field semantics."""

    ALL = "all"
    DIRTY = "dirty"
    DEFAULT_DIFF = "default_diff"
    INHERITED = "inherited"
    RAW_RESOLVED = "raw_resolved"
    SEMANTIC = "semantic"


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldListOptions:
    include_fields: bool = False
    field_limit: int = 200
    field_offset: int = 0
    include_field_values: bool = False
    include_field_descriptions: bool = False
    field_paths: tuple[str, ...] = ()
    field_filter: UiObjectStateFieldFilter = UiObjectStateFieldFilter.ALL


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
    base_revision_token: str = field(
        metadata={
            AGENT_PARAMETER_DESCRIPTION_METADATA_KEY: (
                "Base revision returned by the corresponding state/code-document "
                "read operation; this is not an action selection token."
            )
        }
    )


@dataclass(frozen=True, kw_only=True)
class UiCodeDocumentOptionalBaseRevision:
    base_revision_token: str | None = field(
        default=None,
        metadata={
            AGENT_PARAMETER_DESCRIPTION_METADATA_KEY: (
                "Optional base state/code-document revision returned by the "
                "corresponding read operation; this is not an action selection "
                "token."
            )
        },
    )


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


@dataclass(frozen=True, slots=True, kw_only=True)
class UiBridgeConnectionFields(
    UiBridgeInstanceIdentity,
    UiBridgeDescriptorFileRef,
):
    """Sparse connection-field projection for descriptor/env/MCP inputs."""

    host: NonBlankString | None = None
    port: SocketPort | None = None
    transport_mode: TransportMode | None = None
    persistent: bool | None = None
    timeout_ms: PositiveInteger | None = None
    auth_token: NonBlankString | None = None

    def __post_init__(self) -> None:
        validate_annotated_dataclass(self)


@dataclass(frozen=True, slots=True)
class UiBridgeConnectionRequest(UiBridgeConnectionFields):
    """MCP-facing sparse connection request for a running UI bridge."""


@dataclass(frozen=True, slots=True)
class UiBridgeConnectionSpec(
    ExecutionConnectionSpec,
    UiBridgeInstanceIdentity,
    UiBridgeDescriptorFileRef,
):
    timeout_ms: PositiveInteger = 5000
    auth_token: NonBlankString | None = None

    @classmethod
    def from_fields(
        cls,
        fields: UiBridgeConnectionFields,
        defaults: "UiBridgeConnectionSpec | None" = None,
    ) -> "UiBridgeConnectionSpec":
        if defaults is None:
            defaults = cls()
        return overlay_non_none_dataclass(defaults, fields)


@dataclass(frozen=True, slots=True)
class UiBridgeEndpointIdentity(
    ExecutionConnectionProjection,
    UiBridgeInstanceIdentity,
    UiBridgeDescriptorFileRef,
):
    """Exact public identity of one descriptor-backed UI bridge endpoint."""

    @classmethod
    def from_value(
        cls,
        value: ExecutionConnectionProjection,
    ) -> UiBridgeEndpointIdentity:
        """Project identity from any declared UI bridge endpoint carrier."""

        return project_dataclass(
            cls,
            value,
            connection=value.connection.public_connection(),
        )


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorEnvelope(
    UiBridgeApplicationIdentity,
    UiBridgeInstanceIdentity,
):
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

    def resolve_connection(
        self,
        defaults: UiBridgeConnectionSpec,
    ) -> UiBridgeConnectionSpec:
        """Overlay this descriptor's declared endpoint and bridge fields."""

        return overlay_non_none_dataclass(
            overlay_non_none_dataclass(defaults, self.connection),
            self,
        )

    def public_summary(self, status: str) -> UiBridgeDescriptorSummary:
        """Project this descriptor to its token-free public representation."""

        return project_dataclass(UiBridgeDescriptorSummary, self, status=status)


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorWirePayload(
    ExecutionConnectionProjection,
    UiBridgeDescriptorEnvelope,
):
    """JSON descriptor file payload written by a running UI bridge."""


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorSummary(
    AgentTimedStatusEnvelope,
    ExecutionConnectionProjection,
    UiBridgeApplicationIdentity,
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
class UiCodeDocumentRequest(
    UiCodeDocumentIdentity,
    SelectionModeCarrier,
    AgentCliRequest,
):
    clean: bool = True

    @classmethod
    def from_fields(
        cls,
        *,
        document_id: str,
        selection_mode: str = UiCodeDocumentSelectionMode.SELECTED.value,
        clean: bool = True,
    ) -> "UiCodeDocumentRequest":
        return cls(
            document_id=document_id,
            selection_mode=selection_mode,
            clean=clean,
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "document_id": self.document_id,
            "selection_mode": self.selection_mode,
            "clean": self.clean,
        }


@dataclass(frozen=True, slots=True)
class UiStateSurfaceRequest(
    UiStateSurfaceIdentity,
    SelectionModeCarrier,
    UiCodeDocumentOptionalBaseRevision,
    AgentCliRequest,
):
    @classmethod
    def from_fields(
        cls,
        *,
        surface_id: str = PlateManagerStateSurfaceIdentityDeclaration.require_value(),
        selection_mode: str = UiCodeDocumentSelectionMode.ALL.value,
        base_revision_token: str | None = None,
    ) -> "UiStateSurfaceRequest":
        return cls(
            surface_id=surface_id,
            selection_mode=selection_mode,
            base_revision_token=base_revision_token,
        )

    def as_tool_arguments(self) -> JsonObject:
        """Project the declared state-surface factory fields to MCP arguments."""

        return {
            "surface_id": self.surface_id,
            "selection_mode": self.selection_mode,
            "base_revision_token": self.base_revision_token,
        }


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
    output_plate_scope_id: str | None = None
    output_plate_root: str | None = None
    source_plate_scope_id: str | None = None
    source_plate_root: str | None = None
    debug_phase: str | None = None
    debug_session_id: str | None = None
    scope_accent_color: str | None = None


@dataclass(frozen=True, slots=True)
class UiPipelineEditorStepState:
    step_scope_id: str | None
    index: int
    name: str
    enabled: bool
    selected: bool
    dirty: bool
    default_diff: bool
    description: str | None = None
    debug_pause: bool = False
    function_names: tuple[str, ...] = ()
    function_ids: tuple[str, ...] = ()


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
class UiLiveMeasurementEntryState:
    """One retained progress event paired with its authoritative table preview."""

    sequence_id: int
    execution_id: str
    plate_id: str
    axis_id: str
    step_name: str
    preview: LiveMeasurementTablePreview
    truncated_preview_group: bool


@dataclass(frozen=True, slots=True)
class UiLiveMeasurementsState(
    UiStateSurfaceEnvelope,
    UiCodeDocumentCurrentRevision,
    UiCurrentSnapshotState,
    SelectedScopeIdsCarrier,
):
    """Bounded live measurement tables retained by the running Plate Manager."""

    object_state_token: int
    retained_entry_count: int
    visible_entry_count: int
    total_row_count: int
    latest_sequence_id: int | None
    entries: tuple[UiLiveMeasurementEntryState, ...]


@dataclass(frozen=True, slots=True)
class UiPipelineEditorState(
    UiStateSurfaceEnvelope,
    UiCodeDocumentCurrentRevision,
    UiCurrentSnapshotState,
    SelectedScopeIdsCarrier,
):
    object_state_token: int
    current_plate_scope_id: str | None
    pipeline_scope_id: str | None
    steps: tuple[UiPipelineEditorStepState, ...]


@dataclass(frozen=True, slots=True)
class UiDebugActionState(SelectedScopeIdsCarrier):
    action_id: str
    label: str
    placement: str
    enabled: bool
    side_effects: tuple[str, ...]
    confirmation_required: bool
    requires_active_debug_session: bool
    disabled_error: AgentError | None = None


@dataclass(frozen=True, slots=True)
class UiDebugCursorState:
    step_index: int
    step_scope_id: str | None
    group_key: str | None
    invocation_key: str | None
    pattern_group_identity: str | None
    dirty: bool = False


@dataclass(frozen=True, slots=True)
class UiDebugTerminalSummaryState:
    debug_session_id: str
    plate_scope_id: str
    terminal_status: str
    command_type: str | None
    axis_id: str | None
    snapshot_id: str | None
    snapshot_store_ref: str | None
    snapshot_store_backend: str | None
    step_name: str | None
    callable_name: str | None
    cursor: UiDebugCursorState | None
    completed_at_unix: float | None


@dataclass(frozen=True)
class UiDebugSessionIdentityCarrier:
    debug_session_id: str


@dataclass(frozen=True)
class UiDebugSnapshotStoreRefCarrier:
    snapshot_store_ref: str | None
    snapshot_store_backend: str | None


@dataclass(frozen=True, slots=True)
class UiProgressIdentityState:
    execution_id: str
    plate_id: str
    axis_id: str
    step_name: str


@dataclass(frozen=True, slots=True)
class UiDebugRuntimeFrameState(
    UiDebugSessionIdentityCarrier,
    UiDebugSnapshotStoreRefCarrier,
):
    progress_identity: UiProgressIdentityState
    cursor: UiDebugCursorState
    event_type: str
    step_name: str
    callable_name: str | None
    snapshot_id: str | None
    timestamp: float


@dataclass(frozen=True, slots=True)
class UiPipelineDebugSessionState(
    UiStateSurfaceEnvelope,
    UiCodeDocumentCurrentRevision,
    UiCurrentSnapshotState,
    SelectedScopeIdsCarrier,
):
    object_state_token: int
    current_plate_scope_id: str | None
    pipeline_scope_id: str | None
    manager_execution_state: str
    initialized: bool
    compiled: bool
    phase: str
    active_session_id: str | None
    execution_id: str | None
    axis_id: str | None
    selected_source_group: str | None
    snapshot_store_ref: str | None
    snapshot_store_backend: str | None
    terminal_status: str | None
    cursor: UiDebugCursorState | None
    terminal_summary: UiDebugTerminalSummaryState | None
    actions: tuple[UiDebugActionState, ...]
    current_frame: UiDebugRuntimeFrameState | None = None
    last_frame: UiDebugRuntimeFrameState | None = None


@dataclass(frozen=True, slots=True)
class UiLiveOverviewMetric:
    key: str
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class UiLiveOverviewItem:
    label: str
    status: str | None = None
    detail: str | None = None
    severity: str = UiLiveOverviewSeverity.INFO.value
    source_surface_id: str | None = None
    source_window_id: str | None = None
    source_widget_id: str | None = None


@dataclass(frozen=True, slots=True)
class UiLiveOverviewSection:
    section_id: str
    title: str
    summary: str
    metrics: tuple[UiLiveOverviewMetric, ...] = ()
    items: tuple[UiLiveOverviewItem, ...] = ()


@dataclass(frozen=True, slots=True)
class UiLiveOverviewState(
    UiStateSurfaceEnvelope,
    UiCodeDocumentCurrentRevision,
    UiCurrentSnapshotState,
    SelectedScopeIdsCarrier,
):
    object_state_token: int
    sections: tuple[UiLiveOverviewSection, ...]


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

    @classmethod
    def accepted_for(
        cls,
        request_token: UiMutationRequestToken,
        *,
        bridge_operation_id: str | None = None,
    ) -> "UiMutationReceipt":
        return cls(
            request_token=request_token,
            bridge_operation_id=bridge_operation_id,
            accepted=True,
        )

    @classmethod
    def rejected_for(
        cls,
        request_token: UiMutationRequestToken,
        *,
        bridge_operation_id: str | None = None,
    ) -> "UiMutationReceipt":
        return cls(
            request_token=request_token,
            bridge_operation_id=bridge_operation_id,
            accepted=False,
        )


@dataclass(frozen=True, slots=True)
class UiActionSummary(SelectionModeCarrier):
    schema_version: str
    identity: UiActionIdentity
    title: str
    enabled: bool
    invocation_mode: str
    side_effects: tuple[str, ...] = ()
    disabled_error: AgentError | None = None
    confirmation_required: bool = False
    current_selection_count: int = 0
    target_scope_ids: tuple[str, ...] = ()
    selection_revision_token: str | None = None
    related_state_surface_ids: tuple[str, ...] = ()
    required_target_count: int | None = None

    def invocation_guard_error(
        self,
        request: UiActionInvokeRequest,
    ) -> AgentError | None:
        """Validate an invocation against this action's authoritative summary."""
        if not self.enabled:
            return self.disabled_error or AgentError(
                code="ui_action_disabled",
                message=f"UI action {request.action_id!r} is disabled.",
            )
        if self.confirmation_required and request.confirmation_is_required():
            return AgentError(
                code="confirmation_required",
                message=(
                    "This UI action mutates state; set require_confirmation=False "
                    "to dispatch it."
                ),
            )
        if (
            self.required_target_count is not None
            and len(request.selected_scope_ids) != self.required_target_count
        ):
            return AgentError(
                code="ui_action_target_required",
                message=(
                    f"UI action {request.action_id!r} requires exactly "
                    f"{self.required_target_count} target scope(s)."
                ),
            )
        if self.required_target_count is not None:
            unavailable_targets = set(request.selected_scope_ids) - set(
                self.target_scope_ids
            )
            if unavailable_targets:
                return AgentError(
                    code="unknown_ui_action_target",
                    message="Requested target scope is not available for this UI action.",
                )
        elif (
            request.selected_scope_ids
            and request.selected_scope_ids != self.target_scope_ids
        ):
            return AgentError(
                code="stale_ui_action_selection",
                message="Requested target scopes do not match current UI action targets.",
                hint=self.selection_bound_hint(request),
            )
        if (
            request.observed_selection_revision_token is not None
            and self.selection_revision_token is not None
            and request.observed_selection_revision_token
            != self.selection_revision_token
        ):
            return AgentError(
                code="stale_ui_action_revision",
                message="UI action selection changed after the action was planned.",
                hint=(
                    "Call openhcs_ui_list_actions again for this widget and action, "
                    "then pass its selection_revision_token as "
                    "observed_selection_revision_token. State-surface "
                    "base_revision_token values belong to a different token family."
                ),
            )
        return None

    def selection_bound_hint(self, request: UiActionInvokeRequest) -> str:
        """Explain how an invocation relates to the action's current targets."""
        requested_targets = ", ".join(request.selected_scope_ids)
        current_targets = ", ".join(self.target_scope_ids) or "<none>"
        parts = [
            (
                f"{request.widget_id}/{request.action_id} is selection-bound "
                f"(selection_mode={self.selection_mode or '<none>'}); "
                "--target-scope-id is a staleness guard for the current UI "
                "selection, not direct ObjectState navigation."
            ),
            f"Current action targets: {current_targets}.",
        ]
        if self.related_state_surface_ids:
            parts.append(
                "Read related state surfaces before invoking: "
                f"{', '.join(self.related_state_surface_ids)}."
            )
        if len(request.selected_scope_ids) == 1:
            parts.append(
                "To open the requested ObjectState scope directly, call "
                "openhcs_ui_navigate_window with "
                f"window_id={requested_targets!r}."
            )
        else:
            parts.append(
                "To open a known ObjectState scope directly, call "
                "openhcs_ui_navigate_window with window_id=<object_state_scope_id>."
            )
        return " ".join(parts)


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
    AgentCliRequest,
):
    observed_selection_revision_token: str | None = field(
        default=None,
        metadata={
            AGENT_PARAMETER_DESCRIPTION_METADATA_KEY: (
                "Action-selection revision returned as selection_revision_token by "
                "the action-catalog producer for the same widget, action, and target "
                "selection; do not pass a state-surface revision token."
            ),
            AGENT_PARAMETER_PRODUCER_OUTPUT_CONTRACT_METADATA_KEY: UiActionCatalog,
        },
    )

    @classmethod
    def from_fields(
        cls,
        *,
        widget_id: str,
        action_id: str,
        target_scope_ids: list[str] | None = None,
        observed_selection_revision_token: str | None = None,
        request_token: str | None = None,
        require_confirmation: bool = True,
    ) -> "UiActionInvokeRequest":
        selected_scope_ids = SelectedScopeIdsArgument.from_optional_iterable(
            target_scope_ids
        )
        return cls(
            widget_id=widget_id,
            action_id=action_id,
            selected_scope_ids=selected_scope_ids.selected_scope_ids,
            observed_selection_revision_token=observed_selection_revision_token,
            request_token=UiMutationRequestToken(value=request_token),
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                require_confirmation
            ),
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "widget_id": self.widget_id,
            "action_id": self.action_id,
            "target_scope_ids": list(self.selected_scope_ids),
            "observed_selection_revision_token": (
                self.observed_selection_revision_token
            ),
            "request_token": self.request_token.value,
            "require_confirmation": self.confirmation_is_required(),
        }


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


class UiWindowSemanticMarker(str, Enum):
    ERROR_DIALOG = "error_dialog"


@dataclass(frozen=True, slots=True)
class UiWindowSummary:
    schema_version: str
    identity: UiWindowIdentity
    title: str
    window_kind: str
    visible: bool
    focusable: bool
    manager_scope: UiWindowManagerScope | None = None
    object_state_scope_id: str | None = None
    dirty: bool = False
    signature_diff: bool = False
    dirty_field_count: int = 0
    signature_diff_field_count: int = 0
    semantic_markers: tuple[str, ...] = ()
    managed_action_ids: tuple[str, ...] = ()

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
    @classmethod
    def from_fields(
        cls,
        *,
        window_id: str,
        create_if_missing: bool = True,
    ) -> "UiWindowFocusRequest":
        return cls(
            window_id=window_id,
            open_policy=UiWindowOpenPolicy(create_if_missing=create_if_missing),
        )


@dataclass(frozen=True, slots=True)
class UiWindowFocusResult(AgentResultEnvelope, UiWindowIdentity):
    focused: bool
    summary: UiWindowSummary | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWindowNavigateRequest(UiWindowOperationRequest):
    item_id: str | None = None
    field_path: str | None = None

    @classmethod
    def from_fields(
        cls,
        *,
        window_id: str,
        field_path: str | None = None,
        item_id: str | None = None,
        create_if_missing: bool = True,
    ) -> "UiWindowNavigateRequest":
        return cls(
            window_id=window_id,
            field_path=field_path,
            item_id=item_id,
            open_policy=UiWindowOpenPolicy(create_if_missing=create_if_missing),
        )


@dataclass(frozen=True, slots=True)
class UiWindowNavigateResult(AgentResultEnvelope, UiWindowIdentity):
    focused: bool
    navigated: bool
    created: bool
    summary: UiWindowSummary | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWindowCloseRequest(UiWindowIdentity):
    """Request a normal close for one currently open UI window."""

    @classmethod
    def from_fields(
        cls,
        *,
        window_id: str,
    ) -> "UiWindowCloseRequest":
        return cls(window_id=window_id)


@dataclass(frozen=True, slots=True)
class UiWindowCloseResult(AgentResultEnvelope, UiWindowIdentity):
    closed: bool
    summary: UiWindowSummary | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWindowSnapshotRequest(
    WindowSnapshotCaptureSpec,
    UiWindowOperationRequest,
    AgentCliRequest,
):
    @classmethod
    def from_fields(
        cls,
        *,
        window_id: str,
        output_dir_path: str | None = None,
        capture_scope: str = WindowSnapshotCaptureScope.WIDGET.value,
        create_if_missing: bool = False,
    ) -> "UiWindowSnapshotRequest":
        if output_dir_path is None:
            output_dir_path = str(DEFAULT_AGENT_WINDOW_SNAPSHOT_DIR)
        return cls(
            window_id=window_id,
            output_dir_path=output_dir_path,
            capture_scope=WindowSnapshotCaptureScope(capture_scope),
            open_policy=UiWindowOpenPolicy(create_if_missing=create_if_missing),
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "window_id": self.window_id,
            "output_dir_path": self.output_dir_path,
            "capture_scope": self.capture_scope.value,
            "create_if_missing": self.open_policy.create_if_missing,
        }


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
class UiWidgetTreeRequest(
    WidgetTreeProjectionControls,
    UiWindowOperationRequest,
    AgentCliRequest,
):
    actionable_only: bool = True
    include_tree: bool = False
    max_depth: int | None = 8
    max_nodes: int | None = 80

    @classmethod
    def from_fields(
        cls,
        *,
        window_id: str,
        create_if_missing: bool = False,
        maximum_text_length: int = (
            WidgetTreeProjectionControls.default_maximum_text_length()
        ),
        maximum_item_model_nodes: int | None = (
            WidgetTreeProjectionControls.default_maximum_item_model_nodes()
        ),
        truncation_suffix: str = (
            WidgetTreeProjectionControls.default_truncation_suffix()
        ),
        actionable_only: bool = True,
        include_tree: bool = False,
        max_depth: int | None = 8,
        max_nodes: int | None = 80,
    ) -> "UiWidgetTreeRequest":
        return cls(
            window_id=window_id,
            open_policy=UiWindowOpenPolicy(create_if_missing=create_if_missing),
            maximum_text_length=maximum_text_length,
            maximum_item_model_nodes=maximum_item_model_nodes,
            truncation_suffix=truncation_suffix,
            actionable_only=actionable_only,
            include_tree=include_tree,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

    @classmethod
    def default_include_tree(cls) -> bool:
        return False

    @classmethod
    def default_max_depth(cls) -> int:
        return 8

    @classmethod
    def default_max_nodes(cls) -> int:
        return 80

    def __post_init__(self) -> None:
        WidgetTreeProjectionControls.__post_init__(self)
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if self.max_nodes is not None and self.max_nodes < 1:
            raise ValueError("max_nodes must be positive or None")

    def as_tool_arguments(self) -> JsonObject:
        return {
            "window_id": self.window_id,
            "create_if_missing": self.open_policy.create_if_missing,
            "maximum_text_length": self.maximum_text_length,
            "maximum_item_model_nodes": self.maximum_item_model_nodes,
            "truncation_suffix": self.truncation_suffix,
            "actionable_only": self.actionable_only,
            "include_tree": self.include_tree,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
        }


@dataclass(frozen=True, slots=True)
class UiWidgetRect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class UiObjectStateValuePreview:
    """Bounded display preview for an ObjectState field value."""

    type_name: str
    is_none: bool
    text: str
    truncated: bool = False


@dataclass(frozen=True, kw_only=True)
class UiObjectStateFieldValuePreviewCarrier:
    """Shared raw/resolved value previews for ObjectState-aware UI DTOs."""

    raw_value_is_none: bool = False
    resolved_value_is_none: bool = False
    raw_value_preview: UiObjectStateValuePreview | None = field(
        default=None,
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=lambda owner, field_value: (
                    field_value is not None or owner.raw_value_is_none
                )
            )
        },
    )
    resolved_value_preview: UiObjectStateValuePreview | None = field(
        default=None,
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=lambda owner, field_value: (
                    field_value is not None or owner.resolved_value_is_none
                )
            )
        },
    )


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
    item_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UiWidgetActionSummary(UiObjectStateFieldValuePreviewCarrier, WidgetNodeIdentity):
    label: str | None = field(
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=always_project_compact_field
            )
        }
    )
    visible: bool = field(
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=always_project_compact_field
            )
        }
    )
    enabled: bool = field(
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=always_project_compact_field
            )
        }
    )
    geometry: UiWidgetRect = field(
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=always_project_compact_field
            )
        }
    )
    global_geometry: UiWidgetRect = field(
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=always_project_compact_field
            )
        }
    )
    action_kinds: tuple[str, ...] = field(
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=always_project_compact_field
            )
        }
    )
    clickable: bool = field(
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=always_project_compact_field
            )
        }
    )
    checkable: bool | None
    checked: bool | None = field(
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=lambda owner, field_value: (
                    isinstance(field_value, bool)
                    and (field_value or owner.checkable is True)
                )
            )
        }
    )
    current_index: int | None
    current_text: str | None
    item_count: int | None
    tool_tip: str
    item_texts: tuple[str, ...] = ()
    context_label: str | None = None
    action_role: str | None = None
    semantic_address: UiSemanticAddress | None = None
    object_state_scope_id: str | None = None
    field_path: str | None = None
    dirty: bool = False
    signature_diff: bool = False
    last_changed: bool = False
    semantic_markers: tuple[str, ...] = ()
    raw_value: JsonValue | None = field(
        default=None,
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=lambda owner, field_value: (
                    owner.raw_value_preview is None
                    and (field_value is not None or owner.raw_value_is_none)
                )
            )
        },
    )
    resolved_value: JsonValue | None = field(
        default=None,
        metadata={
            COMPACT_FIELD_PROJECTION_METADATA_KEY: CompactFieldProjection(
                includes=lambda owner, field_value: (
                    owner.resolved_value_preview is None
                    and (field_value is not None or owner.resolved_value_is_none)
                )
            )
        },
    )
    inherited_value: bool = False
    provenance: "UiObjectStateFieldProvenance | None" = None

    def compact_projection(self) -> JsonObject:
        """Return the field-declaration-owned compact action projection."""

        return {
            field_name: to_jsonable(field_value)
            for field_name, field_value in compact_dataclass_projection(self).items()
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class UiWidgetActionInvokeRequest(
    UiWindowOperationRequest,
    UiMutationRequestTokenCarrier,
    AgentCliRequest,
):
    path_id: str
    action_kind: str = "auto"
    target_index: int | None = None

    @classmethod
    def from_fields(
        cls,
        *,
        window_id: str,
        path_id: str,
        action_kind: str = "button",
        target_index: int | None = None,
        create_if_missing: bool = False,
        request_token: str | None = None,
    ) -> "UiWidgetActionInvokeRequest":
        return cls(
            window_id=window_id,
            open_policy=UiWindowOpenPolicy(create_if_missing=create_if_missing),
            path_id=path_id,
            action_kind=action_kind,
            target_index=target_index,
            request_token=UiMutationRequestToken(value=request_token),
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "window_id": self.window_id,
            "path_id": self.path_id,
            "action_kind": self.action_kind,
            "target_index": self.target_index,
            "create_if_missing": self.open_policy.create_if_missing,
            "request_token": self.request_token.value,
        }


class UiWidgetActionIssueCode(str, Enum):
    """Stable issue identities for projected widget action invocation."""

    RESOLUTION_FAILED = "ui_widget_action_resolution_failed"
    WIDGET_UNKNOWN = "unknown_ui_widget"
    ACTION_UNSUPPORTED = "ui_widget_action_unsupported"
    INDEX_INVALID = "ui_widget_index_invalid"
    NOT_VISIBLE = "ui_widget_not_visible"
    DISABLED = "ui_widget_disabled"
    NOT_CLICKABLE = "ui_widget_not_clickable"
    ACTION_KIND_UNAVAILABLE = "ui_widget_action_kind_unavailable"


@dataclass(frozen=True, slots=True)
class UiWidgetActionInvokeResult(
    AgentResultEnvelope,
    UiWindowIdentity,
):
    path_id: str
    action_kind: str
    invoked: bool
    receipt: UiMutationReceipt
    summary: UiWidgetActionSummary | None = None


@dataclass(frozen=True, slots=True)
class UiWidgetTreeResult(AgentResultEnvelope, UiWindowIdentity):
    projected: bool
    root: UiWidgetTreeNode | None = None
    actionable_widgets: tuple[UiWidgetActionSummary, ...] = ()
    summary: UiWindowSummary | None = None
    widget_count: int = 0
    actionable_count: int = 0
    returned_widget_count: int = 0
    returned_actionable_count: int = 0
    tree_truncated: bool = False
    actionable_widgets_truncated: bool = False
    actionable_only: bool = True
    include_tree: bool = False
    max_depth: int | None = None
    max_nodes: int | None = None

    def as_jsonable(self, *, compact_actions: bool = True) -> JsonObject:
        """Serialize the typed result with action compaction owned by its DTO."""

        payload = to_jsonable(self)
        if not isinstance(payload, dict):
            raise TypeError("Widget tree serialization did not produce an object.")
        if compact_actions:
            payload["actionable_widgets"] = [
                action.compact_projection() for action in self.actionable_widgets
            ]
        return payload


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
class UiObjectStateFieldSummary(
    UiObjectStateFieldProvenanceCarrier,
    UiObjectStateFieldValuePreviewCarrier,
):
    """Field-level ObjectState semantics and bounded raw/resolved values."""

    schema_version: str
    address: UiSemanticAddress
    field_name: str
    container_path: str
    object_state_path_type: str
    raw_value_type: str
    resolved_value_type: str | None
    dirty: bool
    signature_diff: bool
    last_changed: bool
    parameter_description: str | None = None
    semantic_markers: tuple[str, ...] = ()
    raw_value: JsonValue | None = None
    resolved_value: JsonValue | None = None
    inherited_value: bool = False


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldPathIndex:
    """Field-path relationship index for ObjectState field projections."""

    fields: tuple[UiObjectStateFieldSummary, ...]

    @property
    def container_field_paths(self) -> frozenset[str]:
        field_paths = tuple(field.address.field_path for field in self.fields)
        return frozenset(
            field_path
            for field_path in field_paths
            if any(
                DottedFieldPath(field_path).contains_path(other_path)
                for other_path in field_paths
                if other_path != field_path
            )
        )


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldHelpRequest(UiSemanticAddress):
    """Request docs/help for one live ObjectState field address."""

    max_description_chars: int = 4_000


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldHelpQuery(AgentCliRequest):
    """Public query for field help; scope may be inferred when unique."""

    field_path: str
    object_state_scope_id: str | None = None
    window_id: str | None = None
    max_description_chars: int = 4_000

    @classmethod
    def from_fields(
        cls,
        *,
        field_path: str,
        object_state_scope_id: str | None = None,
        window_id: str | None = None,
        max_description_chars: int = 4_000,
    ) -> "UiObjectStateFieldHelpQuery":
        return cls(
            field_path=field_path,
            object_state_scope_id=object_state_scope_id,
            window_id=window_id,
            max_description_chars=max_description_chars,
        )

    def __post_init__(self) -> None:
        if self.max_description_chars < 0:
            raise ValueError("max_description_chars must be nonnegative.")

    def concrete_request(
        self,
        object_state_scope_id: str,
    ) -> UiObjectStateFieldHelpRequest:
        return UiObjectStateFieldHelpRequest(
            object_state_scope_id=object_state_scope_id,
            field_path=self.field_path,
            window_id=self.window_id,
            max_description_chars=self.max_description_chars,
        )

    def error_address(self) -> UiSemanticAddress:
        return UiSemanticAddress(
            object_state_scope_id=self.object_state_scope_id or "",
            field_path=self.field_path,
            window_id=self.window_id,
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "field_path": self.field_path,
            "object_state_scope_id": self.object_state_scope_id,
            "window_id": self.window_id,
            "max_description_chars": self.max_description_chars,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class UiObjectStateFieldHelpResult(AgentResultEnvelope):
    """Display-ready help for one ObjectState field from Python introspection."""

    address: UiSemanticAddress
    field: UiObjectStateFieldSummary | None = None
    object_type: str | None = None
    help_target_type: str | None = None
    parameter_name: str | None = None
    target_summary: str | None = None
    target_description: str | None = None
    summary: str | None = None
    description: str | None = None
    enum_values: tuple[str, ...] = ()
    description_truncated: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class UiObjectStateFieldMutationRequest(
    UiMutationRequestTokenCarrier,
    UiSemanticAddress,
    AgentCliRequest,
):
    """Request an ObjectState-owned field update or reset."""

    value: JsonValue | None = None
    reset: bool = False
    include_field_values: bool = True

    @classmethod
    def from_fields(
        cls,
        *,
        object_state_scope_id: str,
        field_path: str,
        value: dict | list | str | int | float | bool | None = None,
        reset: bool = False,
        window_id: str | None = None,
        include_field_values: bool = True,
        request_token: str | None = None,
    ) -> "UiObjectStateFieldMutationRequest":
        return cls(
            object_state_scope_id=object_state_scope_id,
            field_path=field_path,
            window_id=window_id,
            value=value,
            reset=reset,
            include_field_values=include_field_values,
            request_token=UiMutationRequestToken(value=request_token),
        )

    def __post_init__(self) -> None:
        if self.reset and self.value is not None:
            raise ValueError("reset=True cannot be combined with value.")

    def as_tool_arguments(self) -> JsonObject:
        return {
            "object_state_scope_id": self.object_state_scope_id,
            "field_path": self.field_path,
            "value": self.value,
            "reset": self.reset,
            "window_id": self.window_id,
            "include_field_values": self.include_field_values,
            "request_token": self.request_token.value,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class UiObjectStateFieldMutationResult(AgentResultEnvelope):
    """Before/after projection for one ObjectState-owned field mutation."""

    address: UiSemanticAddress
    mutated: bool
    reset: bool
    receipt: UiMutationReceipt
    before: UiObjectStateFieldSummary | None = None
    after: UiObjectStateFieldSummary | None = None
    current_snapshot: "UiSnapshotRef | None" = None


@dataclass(frozen=True, slots=True)
class UiObjectStateScopeSummary(UiFieldCatalogPageCarrier):
    schema_version: str
    identity: UiObjectStateScopeIdentity
    object_type: str
    parameter_count: int
    dirty_field_count: int
    signature_diff_field_count: int
    has_unsaved_changes: bool = False
    has_default_overrides: bool = False
    dirty_marker: str = "*"
    signature_diff_marker: str = "_"
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
class UiObjectStateScopeListRequest(
    UiObjectStateScopeVisibility,
    AgentCliRequest,
):
    scope_ids: tuple[str, ...] = ()
    include_fields: bool = False
    field_limit: int = 200
    field_offset: int = 0
    include_field_values: bool = False
    include_field_descriptions: bool = False
    field_paths: tuple[str, ...] = ()
    field_filter: UiObjectStateFieldFilter = UiObjectStateFieldFilter.ALL

    @classmethod
    def from_visibility_options(
        cls,
        visibility: UiObjectStateScopeVisibility,
        field_options: UiObjectStateFieldListOptions,
        *,
        scope_ids: tuple[str, ...] = (),
    ) -> "UiObjectStateScopeListRequest":
        return cls(
            scope_ids=scope_ids,
            include_system_scopes=visibility.include_system_scopes,
            include_fields=field_options.include_fields,
            field_limit=field_options.field_limit,
            field_offset=field_options.field_offset,
            include_field_values=field_options.include_field_values,
            include_field_descriptions=field_options.include_field_descriptions,
            field_paths=field_options.field_paths,
            field_filter=field_options.field_filter,
        )

    @classmethod
    def from_fields(
        cls,
        *,
        scope_ids: tuple[str, ...] = (),
        include_system_scopes: bool = False,
        include_fields: bool = False,
        include_field_values: bool = False,
        field_filter: str = UiObjectStateFieldFilter.ALL.value,
        field_limit: int = 200,
        field_offset: int = 0,
    ) -> "UiObjectStateScopeListRequest":
        return cls.from_visibility_options(
            UiObjectStateScopeVisibility(
                include_system_scopes=include_system_scopes,
            ),
            UiObjectStateFieldListOptions(
                include_fields=include_fields,
                include_field_values=include_field_values,
                field_limit=field_limit,
                field_offset=field_offset,
                field_filter=UiObjectStateFieldFilter(field_filter),
            ),
            scope_ids=tuple(scope_ids),
        )

    def filtered_catalog(
        self,
        catalog: UiObjectStateScopeCatalog,
    ) -> UiObjectStateScopeCatalog:
        if not self.scope_ids:
            return catalog
        requested_scope_ids = set(self.scope_ids)
        return replace(
            catalog,
            scopes=tuple(
                scope
                for scope in catalog.scopes
                if scope.identity.object_state_scope_id in requested_scope_ids
            ),
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "scope_ids": list(self.scope_ids),
            "include_system_scopes": self.include_system_scopes,
            "include_fields": self.include_fields,
            "include_field_values": self.include_field_values,
            "field_filter": self.field_filter.value,
            "field_limit": self.field_limit,
            "field_offset": self.field_offset,
        }


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldProjection:
    """Compact ObjectState field row for agent-facing field-list queries."""

    field_path: str
    field_name: str
    container_path: str
    object_state_path_type: str
    dirty: bool
    signature_diff: bool
    last_changed: bool
    semantic_markers: tuple[str, ...]
    raw_value_type: str
    resolved_value_type: str | None
    raw_value_preview: UiObjectStateValuePreview | None
    resolved_value_preview: UiObjectStateValuePreview | None
    raw_value: JsonValue | None
    resolved_value: JsonValue | None
    raw_value_is_none: bool
    resolved_value_is_none: bool
    inherited_value: bool
    provenance: UiObjectStateFieldProvenance | None


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldScopeProjection:
    """Compact ObjectState scope row for field-list query results."""

    scope_id: str
    object_type: str
    dirty_field_count: int
    signature_diff_field_count: int
    has_unsaved_changes: bool
    has_default_overrides: bool
    fields: tuple[UiObjectStateFieldProjection, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class UiObjectStateFieldListResult(AgentResultEnvelope):
    """Filtered, paged ObjectState field projection for agent tools."""

    object_state_token: int
    current_branch: str
    current_snapshot_index: int
    requested_scope_ids: tuple[str, ...]
    field_paths: tuple[str, ...]
    field_path_contains: tuple[str, ...]
    field_filter: str
    include_container_fields: bool
    matched_scope_count: int
    matched_field_count: int
    returned_field_count: int
    field_limit: int
    field_offset: int
    next_offset: int | None
    truncated: bool
    scopes: tuple[UiObjectStateFieldScopeProjection, ...]


@dataclass(frozen=True, slots=True)
class UiObjectStateFieldListQuery(AgentCliRequest):
    """Public ObjectState field query projected from the scope-list bridge ABI."""

    source_query_scan_limit: ClassVar[int] = 1_000

    scope_ids: tuple[str, ...] = ()
    field_paths: tuple[str, ...] = ()
    field_path_contains: tuple[str, ...] = ()
    include_system_scopes: bool = False
    include_clean_fields: bool = True
    include_container_fields: bool = False
    field_filter: UiObjectStateFieldFilter = UiObjectStateFieldFilter.ALL
    include_field_values: bool = False
    field_limit: int = 200
    field_offset: int = 0
    max_fields: int = 100
    max_value_items: int = 20
    max_value_chars: int = 1000

    @classmethod
    def from_fields(
        cls,
        *,
        scope_ids: tuple[str, ...] = (),
        field_paths: tuple[str, ...] = (),
        field_path_contains: tuple[str, ...] = (),
        include_system_scopes: bool = False,
        include_clean_fields: bool = True,
        include_container_fields: bool = False,
        field_filter: str = UiObjectStateFieldFilter.ALL.value,
        include_field_values: bool = False,
        field_limit: int = 200,
        field_offset: int = 0,
        max_fields: int = 100,
        max_value_items: int = 20,
        max_value_chars: int = 1000,
    ) -> "UiObjectStateFieldListQuery":
        return cls(
            scope_ids=tuple(scope_ids),
            field_paths=tuple(field_paths),
            field_path_contains=tuple(field_path_contains),
            include_system_scopes=include_system_scopes,
            include_clean_fields=include_clean_fields,
            include_container_fields=include_container_fields,
            field_filter=UiObjectStateFieldFilter(field_filter),
            include_field_values=include_field_values,
            field_limit=field_limit,
            field_offset=field_offset,
            max_fields=max_fields,
            max_value_items=max_value_items,
            max_value_chars=max_value_chars,
        )

    def __post_init__(self) -> None:
        if self.field_limit < 0 or self.field_offset < 0:
            raise ValueError("field_limit and field_offset must be nonnegative.")
        if self.max_fields < 0 or self.max_value_items < 0 or self.max_value_chars < 0:
            raise ValueError(
                "max_fields, max_value_items, and max_value_chars must be nonnegative."
            )

    @property
    def contains_terms(self) -> tuple[str, ...]:
        return tuple(term.lower() for term in self.field_path_contains)

    @property
    def exact_source_field_paths(self) -> tuple[str, ...]:
        if (
            self.field_paths
            and not self.contains_terms
            and not self.include_container_fields
        ):
            return self.field_paths
        return ()

    @property
    def source_field_limit(self) -> int:
        if self.exact_source_field_paths:
            return max(1, len(self.exact_source_field_paths))
        return max(
            self.source_query_scan_limit,
            self.field_limit + self.field_offset,
            self.max_fields,
        )

    @property
    def return_limit(self) -> int:
        if self.max_fields:
            return min(self.field_limit, self.max_fields)
        return self.field_limit

    def scope_list_request(self) -> UiObjectStateScopeListRequest:
        return UiObjectStateScopeListRequest.from_visibility_options(
            UiObjectStateScopeVisibility(
                include_system_scopes=self.include_system_scopes,
            ),
            UiObjectStateFieldListOptions(
                include_fields=True,
                include_field_values=self.include_field_values,
                field_limit=self.source_field_limit,
                field_offset=0,
                field_paths=self.exact_source_field_paths,
                field_filter=self.field_filter,
            ),
            scope_ids=self.scope_ids,
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "scope_ids": list(self.scope_ids),
            "field_paths": list(self.field_paths),
            "field_path_contains": list(self.field_path_contains),
            "include_system_scopes": self.include_system_scopes,
            "include_clean_fields": self.include_clean_fields,
            "include_container_fields": self.include_container_fields,
            "field_filter": self.field_filter.value,
            "include_field_values": self.include_field_values,
            "field_limit": self.field_limit,
            "field_offset": self.field_offset,
            "max_fields": self.max_fields,
            "max_value_items": self.max_value_items,
            "max_value_chars": self.max_value_chars,
        }


@dataclass(frozen=True, slots=True)
class UiCodeDocumentValidationRequest(
    UiCodeDocumentIdentity,
    UiCodeDocumentOptionalBaseRevision,
    AgentCliRequest,
):
    source: str

    @classmethod
    def from_fields(
        cls,
        *,
        document_id: str,
        source: str,
        base_revision_token: str | None = None,
    ) -> "UiCodeDocumentValidationRequest":
        return cls(
            document_id=document_id,
            source=source,
            base_revision_token=base_revision_token,
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "base_revision_token": self.base_revision_token,
        }


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
    SelectedScopeIdsCarrier,
    UiBridgeConfirmationPolicy,
    UiMutationRequestTokenCarrier,
    AgentCliRequest,
):
    source: str
    snapshot_label: str | None = None
    apply_if_time_traveling: bool = False

    @classmethod
    def from_fields(
        cls,
        *,
        document_id: str,
        source: str,
        base_revision_token: str,
        selection_mode: str = UiCodeDocumentSelectionMode.SELECTED.value,
        selected_scope_ids: tuple[str, ...] = (),
        require_confirmation: bool = True,
        snapshot_label: str | None = None,
        apply_if_time_traveling: bool = False,
        request_token: str | None = None,
    ) -> "UiCodeDocumentApplyRequest":
        return cls(
            document_id=document_id,
            source=source,
            base_revision_token=base_revision_token,
            selection_mode=selection_mode,
            selected_scope_ids=tuple(selected_scope_ids),
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                require_confirmation
            ),
            request_token=UiMutationRequestToken(value=request_token),
            snapshot_label=snapshot_label,
            apply_if_time_traveling=apply_if_time_traveling,
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "base_revision_token": self.base_revision_token,
            "selection_mode": self.selection_mode,
            "selected_scope_ids": list(self.selected_scope_ids),
            "require_confirmation": self.confirmation_is_required(),
            "snapshot_label": self.snapshot_label,
            "apply_if_time_traveling": self.apply_if_time_traveling,
            "request_token": self.request_token.value,
        }


@dataclass(frozen=True, slots=True)
class UiCodeDocumentApplyResult(UiCodeDocumentIdentity):
    schema_version: str
    applied: bool
    base_revision_token: str
    receipt: UiMutationReceipt
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
    AgentCliRequest,
):
    workflow: UiSelectedPlateWorkflowKind
    observed_selection_revision_token: str | None = field(
        default=None,
        metadata={
            AGENT_PARAMETER_DESCRIPTION_METADATA_KEY: (
                "Action-selection revision returned as selection_revision_token by "
                "the action-catalog producer for the same selected-plate workflow; "
                "do not pass a PlateManager state revision token."
            ),
            AGENT_PARAMETER_PRODUCER_OUTPUT_CONTRACT_METADATA_KEY: UiActionCatalog,
        },
    )

    @classmethod
    def from_fields(
        cls,
        *,
        workflow: UiSelectedPlateWorkflowKind,
        target_scope_ids: list[str] | None = None,
        observed_selection_revision_token: str | None = None,
        request_token: str | None = None,
        require_confirmation: bool = False,
    ) -> "UiSelectedPlateWorkflowRequest":
        selected_scope_ids = SelectedScopeIdsArgument.from_optional_iterable(
            target_scope_ids
        )
        return cls(
            workflow=workflow,
            selected_scope_ids=selected_scope_ids.selected_scope_ids,
            observed_selection_revision_token=observed_selection_revision_token,
            request_token=UiMutationRequestToken(value=request_token),
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                require_confirmation
            ),
        )

    def as_tool_arguments(self) -> JsonObject:
        """Project the declared workflow factory fields to MCP arguments."""

        return {
            "workflow": self.workflow.value,
            "target_scope_ids": list(self.selected_scope_ids),
            "observed_selection_revision_token": (
                self.observed_selection_revision_token
            ),
            "request_token": self.request_token.value,
            "require_confirmation": self.confirmation_is_required(),
        }


@dataclass(frozen=True, slots=True)
class UiSelectedPlateWorkflowResult:
    schema_version: str
    workflow: UiSelectedPlateWorkflowKind
    action_result: UiActionInvokeResult
    state_surface_id: str = PlateManagerStateSurfaceIdentityDeclaration.require_value()
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
    @classmethod
    def from_fields(
        cls,
        *,
        scope_visibility: dict | None = None,
    ) -> "UiSnapshotListRequest":
        visibility = UiObjectStateScopeVisibility.from_mapping(scope_visibility)
        return cls(include_system_scopes=visibility.include_system_scopes)


@dataclass(frozen=True, slots=True)
class UiSnapshotRestoreRequest(
    UiSnapshotIdentity,
    UiBridgeBranchMutationPolicy,
    UiObjectStateScopeVisibility,
):
    index: int | None = None
    branch: str | None = None

    @classmethod
    def from_fields(
        cls,
        *,
        snapshot_id: str | None = None,
        index: int | None = None,
        branch: str | None = None,
        scope_visibility: dict | None = None,
        require_confirmation: bool = True,
        allow_auto_branch: bool = False,
    ) -> "UiSnapshotRestoreRequest":
        visibility = UiObjectStateScopeVisibility.from_mapping(scope_visibility)
        return cls(
            snapshot_id=snapshot_id,
            index=index,
            branch=branch,
            include_system_scopes=visibility.include_system_scopes,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                require_confirmation
            ),
            allow_auto_branch=allow_auto_branch,
        )

    @property
    def operation_target_id(self) -> str | None:
        if self.snapshot_id is not None:
            return self.snapshot_id
        if self.branch is not None:
            return self.branch
        if self.index is not None:
            return str(self.index)
        return None


@dataclass(frozen=True, slots=True)
class UiTimeTravelHeadRequest(UiBridgeConfirmationPolicy):
    @classmethod
    def from_fields(
        cls,
        *,
        require_confirmation: bool = True,
    ) -> "UiTimeTravelHeadRequest":
        return cls(
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                require_confirmation
            )
        )


@dataclass(frozen=True, slots=True)
class UiSnapshotRestoreResult(UiCurrentSnapshotState):
    schema_version: str
    restored: bool
    target_snapshot: UiSnapshotRef | None
    catalog: UiSnapshotCatalog | None = None
    operation_id: str | None = None
    receipt: UiMutationReceipt | None = None
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

    @classmethod
    def from_fields(
        cls,
        *,
        branch: str,
        require_confirmation: bool = True,
        allow_auto_branch: bool = False,
    ) -> "UiBranchSwitchRequest":
        return cls(
            branch=branch,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                require_confirmation
            ),
            allow_auto_branch=allow_auto_branch,
        )

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
class UiBridgeOperationWaitRequest(AgentCliRequest):
    """Bounded terminal wait for one accepted UI bridge operation."""

    operation_id: str
    timeout_seconds: float = 30.0
    poll_interval_seconds: float = 0.5

    @classmethod
    def from_fields(
        cls,
        operation_id: str,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.5,
    ) -> "UiBridgeOperationWaitRequest":
        return cls(
            operation_id=operation_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must not be empty.")
        if (
            not isfinite(self.timeout_seconds)
            or not 0.0 <= self.timeout_seconds <= 120.0
        ):
            raise ValueError("timeout_seconds must be between 0 and 120.")
        if (
            not isfinite(self.poll_interval_seconds)
            or not 0.05 <= self.poll_interval_seconds <= 5.0
        ):
            raise ValueError("poll_interval_seconds must be between 0.05 and 5.")

    def as_tool_arguments(self) -> JsonObject:
        """Project the declared operation-wait factory fields to MCP arguments."""

        return {
            "operation_id": self.operation_id,
            "timeout_seconds": self.timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
        }


@dataclass(frozen=True, slots=True)
class UiBridgeOperationRef(AgentTimedStatusEnvelope):
    identity: UiBridgeOperationIdentity
    completed_at_unix: float | None = None
    outcome: str | None = None


@dataclass(frozen=True, slots=True)
class UiBridgeEnvelopeBase(UiBridgeApplicationIdentity):
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
