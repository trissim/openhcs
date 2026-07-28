"""Capability registry for OpenHCS agent integrations."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import ClassVar, Generic, TypeAlias, TypeVar

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.dto.architecture import (
    ArchitectureTopic,
    ArchitectureTopicPage,
    InternalApiSymbol,
)
from openhcs.agent.dto.authoring import (
    AuthoringContext,
    AuthoringContextRequest,
)
from openhcs.agent.dto.common import (
    RenderedSource,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.config import (
    ConfigPatch,
    ConfigRef,
    ConfigSchema,
    ConfigSchemaRequest,
    ConfigSourceRenderRequest,
    ConfigValidationResult,
)
from openhcs.agent.dto.execution import (
    ArtifactPlanInspection,
    CompileSubmissionRequest,
    ExecutionJobRef,
    ExecutionJobStatus,
    ExecutionStatusRequest,
    OrchestratorSession,
    OrchestratorSessionCreationRequest,
    OrchestratorSessionRef,
    OrchestratorSessionRequest,
    PipelineExecutionSubmissionRequest,
    PipelineSourceArtifactPlanInspectionRequest,
    PipelineSourceOrchestratorSessionRequest,
    RuntimeDebugInspectionRequest,
    RuntimeDebugInspectionResult,
    RuntimeExecutionStatus,
    RuntimeServerExecutionStatusRequest,
    RuntimeServerInfo,
    RuntimeServerInfoRequest,
    RuntimeServerScanRequest,
    RuntimeServerScanResult,
)
from openhcs.agent.dto.functions import (
    CustomFunctionRegistrationRequest,
    CustomFunctionRegistrationResult,
    FunctionCatalogPage,
    FunctionDetail,
    FunctionDetailRequest,
    FunctionSearchRequest,
)
from openhcs.agent.dto.knowledge import (
    KnowledgeBaseCatalog,
    KnowledgeBaseDocument,
    KnowledgeBaseDocumentRequest,
    KnowledgeBaseSearchRequest,
    KnowledgeBaseSearchResult,
)
from openhcs.agent.dto.mcp import McpServerHealthResult
from openhcs.agent.dto.pipeline import (
    CreatePipelineRequest,
    FunctionStepAddRequest,
    PipelineRef,
    PipelineSourceRenderRequest,
    PipelineSpec,
    PipelineValidationRequest,
    PipelineValidationResult,
)
from openhcs.agent.dto.plate import (
    PlateFileQueryRequest,
    PlateFileQueryResult,
    PlateFileStreamRequest,
    PlateFileStreamResult,
    PlateImageSampleRequest,
    PlateImageSampleResult,
    PlatePathInspectionRequest,
    PlatePathInspectionResult,
    SelectedPlateFileQueryRequest,
    SelectedPlateFileQueryResult,
    SelectedPlateFileStreamRequest,
    SelectedPlateFileStreamResult,
    SelectedPlateImageInspectionRequest,
    SelectedPlateImageInspectionResult,
    SelectedPlateImageSampleRequest,
    SelectedPlateImageSampleResult,
    SyntheticPlateGenerationRequest,
    SyntheticPlateGenerationResult,
)
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiBranchCatalog,
    UiBranchSwitchRequest,
    UiBridgeCatalog,
    UiBridgeOperationRef,
    UiBridgeOperationWaitRequest,
    UiBridgeStatus,
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentCatalog,
    UiCodeDocumentRequest,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiObjectStateFieldHelpQuery,
    UiObjectStateFieldHelpResult,
    UiObjectStateFieldListQuery,
    UiObjectStateFieldListResult,
    UiObjectStateFieldMutationRequest,
    UiObjectStateFieldMutationResult,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeListRequest,
    UiSelectedPlateWorkflowRequest,
    UiSelectedPlateWorkflowResult,
    UiSnapshotCatalog,
    UiSnapshotListRequest,
    UiSnapshotRestoreRequest,
    UiSnapshotRestoreResult,
    UiStateSurfaceCatalog,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
    UiTimeTravelHeadRequest,
    UiWidgetActionInvokeRequest,
    UiWidgetActionInvokeResult,
    UiWidgetTreeRequest,
    UiWidgetTreeResult,
    UiWindowCatalog,
    UiWindowCloseRequest,
    UiWindowCloseResult,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowNavigateRequest,
    UiWindowNavigateResult,
    UiWindowSnapshotRequest,
    UiWindowSnapshotResult,
)
from openhcs.agent.dto.viewer import (
    ViewerWindowImageSampleRequest,
    ViewerWindowImageSampleResult,
    ViewerWindowLayerIsolationRequest,
    ViewerWindowLayerIsolationResult,
    ViewerWindowNavigationRequest,
    ViewerWindowNavigationResult,
    ViewerWindowPayloadRequest,
    ViewerWindowPayloadResult,
    ViewerWindowProbeResult,
    ViewerWindowRoiSummaryRequest,
    ViewerWindowRoiSummaryResult,
    ViewerWindowSnapshotRequest,
    ViewerWindowSnapshotResult,
    ViewerWindowStateRequest,
    ViewerWindowStateResult,
    ViewerWindowValidationRequest,
    ViewerWindowValidationSummaryResult,
)
from openhcs.serialization.json import to_jsonable


class CapabilityKind(Enum):
    RESOURCE = "resource"
    TOOL = "tool"
    PROMPT = "prompt"


class CapabilityTransport(Enum):
    """MCP exposure boundary on which a capability may be registered."""

    LOCAL_STDIO = "local_stdio"
    HOSTED_STREAMABLE_HTTP = "hosted_streamable_http"


class CapabilityCliConnectionProfile(Enum):
    """CLI connection mechanics required by a capability command."""

    DIRECT = "direct"
    UI_BRIDGE = "ui_bridge"
    VIEWER_WINDOW = "viewer_window"
    RUNTIME_SERVER = "runtime_server"


class CapabilityWorkflowGroup(Enum):
    """Agent-facing workflow group for capability exposition."""

    DISCOVERY = "discovery"
    KNOWLEDGE = "knowledge"
    FUNCTION_AUTHORING = "function_authoring"
    PIPELINE_AUTHORING = "pipeline_authoring"
    PLATE_DATA = "plate_data"
    UI_SELECTED_PLATE = "ui_selected_plate"
    HEADLESS_EXECUTION = "headless_execution"
    RUNTIME_DIAGNOSTICS = "runtime_diagnostics"
    UI_CONTROL = "ui_control"
    UI_STATE_EDITING = "ui_state_editing"
    VIEWER_REVIEW = "viewer_review"

    @property
    def title(self) -> str:
        return _enum_member_title(self)


class CapabilityWorkflowStage(Enum):
    """Workflow stage occupied by one agent capability."""

    DISCOVERY = "discovery"
    CONTEXT = "context"
    AUTHORING = "authoring"
    DATA_PREPARATION = "data_preparation"
    VALIDATION = "validation"
    EXECUTION = "execution"
    STATUS = "status"
    INSPECTION = "inspection"
    CONTROL = "control"
    STATE_EDITING = "state_editing"
    DIAGNOSTIC = "diagnostic"


class CapabilityTargetContext(Enum):
    """Runtime or data authority targeted by an agent capability."""

    SERVER = "server"
    KNOWLEDGE_BASE = "knowledge_base"
    ARCHITECTURE_MODEL = "architecture_model"
    FUNCTION_REGISTRY = "function_registry"
    CONFIG_DRAFT = "config_draft"
    PIPELINE_DRAFT = "pipeline_draft"
    PLATE_PATH = "plate_path"
    UI_SELECTED_PLATE = "ui_selected_plate"
    HEADLESS_SESSION = "headless_session"
    SUBMITTED_JOB = "submitted_job"
    RUNTIME_SERVER = "runtime_server"
    UI_BRIDGE = "ui_bridge"
    UI_WINDOW = "ui_window"
    UI_OBJECT_STATE = "ui_object_state"
    UI_CODE_DOCUMENT = "ui_code_document"
    VIEWER_WINDOW = "viewer_window"


class CapabilityVisibility(Enum):
    """Default audience visibility for grouped capability projections."""

    BEGINNER = "beginner"
    STANDARD = "standard"
    EXPERT = "expert"


class CapabilityRole(Enum):
    """Capability role inside an agent-facing workflow group."""

    PRIMARY = "primary"
    MODE_VARIANT = "mode_variant"
    FALLBACK = "fallback"
    DIAGNOSTIC = "diagnostic"
    EXPERT = "expert"


class LocalCapabilitySurfaceProfile(ABC, metaclass=AutoRegisterMeta):
    """Registered local MCP surface policy over declared exposition metadata."""

    __registry__: ClassVar[dict[str, type["LocalCapabilitySurfaceProfile"]]] = {}
    __registry_key__ = "name"
    __skip_if_no_key__ = True

    name: ClassVar[str | None] = None
    title: ClassVar[str]

    @classmethod
    def for_name(cls, name: str) -> "LocalCapabilitySurfaceProfile":
        try:
            return cls.__registry__[name]()
        except KeyError as exc:
            raise ValueError(f"Unknown local MCP surface profile: {name!r}.") from exc

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(cls.__registry__)

    def includes(self, capability: "AgentCapabilitySpec") -> bool:
        """Return whether this profile includes one nominal capability."""
        del capability
        return True


class NonExpertCapabilitySurfaceMixin:
    """Exclude declarations intentionally marked expert-only or fallback."""

    def includes(self, capability: "AgentCapabilitySpec") -> bool:
        return (
            capability.visibility is not CapabilityVisibility.EXPERT
            and capability.role not in (CapabilityRole.EXPERT, CapabilityRole.FALLBACK)
            and super().includes(capability)
        )


class WorkflowGroupCapabilitySurfaceMixin:
    """Restrict a surface to authoritative workflow-group declarations."""

    workflow_groups: ClassVar[frozenset[CapabilityWorkflowGroup]]

    def includes(self, capability: "AgentCapabilitySpec") -> bool:
        return capability.workflow_group in self.workflow_groups and super().includes(
            capability
        )


class SelfContainedCapabilitySurfaceMixin:
    """Exclude capabilities requiring a separately running external runtime."""

    def includes(self, capability: "AgentCapabilitySpec") -> bool:
        return not capability.runtime_requirements and super().includes(capability)


class FullLocalCapabilitySurfaceProfile(LocalCapabilitySurfaceProfile):
    name: ClassVar[str] = "full"
    title = "Full local development surface"


class DesktopLocalCapabilitySurfaceProfile(
    WorkflowGroupCapabilitySurfaceMixin,
    NonExpertCapabilitySurfaceMixin,
    LocalCapabilitySurfaceProfile,
):
    name: ClassVar[str] = "desktop"
    title = "Desktop user surface"
    workflow_groups = frozenset(
        (
            CapabilityWorkflowGroup.DISCOVERY,
            CapabilityWorkflowGroup.KNOWLEDGE,
            CapabilityWorkflowGroup.FUNCTION_AUTHORING,
            CapabilityWorkflowGroup.PIPELINE_AUTHORING,
            CapabilityWorkflowGroup.PLATE_DATA,
            CapabilityWorkflowGroup.UI_SELECTED_PLATE,
            CapabilityWorkflowGroup.UI_CONTROL,
            CapabilityWorkflowGroup.UI_STATE_EDITING,
            CapabilityWorkflowGroup.VIEWER_REVIEW,
        )
    )


class AuthoringLocalCapabilitySurfaceProfile(
    WorkflowGroupCapabilitySurfaceMixin,
    NonExpertCapabilitySurfaceMixin,
    LocalCapabilitySurfaceProfile,
):
    name: ClassVar[str] = "authoring"
    title = "Authoring surface"
    workflow_groups = frozenset(
        (
            CapabilityWorkflowGroup.DISCOVERY,
            CapabilityWorkflowGroup.KNOWLEDGE,
            CapabilityWorkflowGroup.FUNCTION_AUTHORING,
            CapabilityWorkflowGroup.PIPELINE_AUTHORING,
        )
    )


class CoreLocalCapabilitySurfaceProfile(
    SelfContainedCapabilitySurfaceMixin,
    WorkflowGroupCapabilitySurfaceMixin,
    NonExpertCapabilitySurfaceMixin,
    LocalCapabilitySurfaceProfile,
):
    name: ClassVar[str] = "core"
    title = "Core local workflow surface"
    workflow_groups = frozenset(
        (
            *AuthoringLocalCapabilitySurfaceProfile.workflow_groups,
            CapabilityWorkflowGroup.PLATE_DATA,
            CapabilityWorkflowGroup.HEADLESS_EXECUTION,
        )
    )


@dataclass(frozen=True, slots=True)
class AgentCapabilityExposition:
    """Complete nominal exposition contract for one agent capability."""

    workflow_group: CapabilityWorkflowGroup
    workflow_stage: CapabilityWorkflowStage
    target_context: CapabilityTargetContext
    visibility: CapabilityVisibility
    role: CapabilityRole = CapabilityRole.PRIMARY

    def refine(
        self,
        *,
        workflow_group: CapabilityWorkflowGroup | None = None,
        workflow_stage: CapabilityWorkflowStage | None = None,
        target_context: CapabilityTargetContext | None = None,
        visibility: CapabilityVisibility | None = None,
        role: CapabilityRole | None = None,
    ) -> "AgentCapabilityExposition":
        """Return a typed refinement owned by an inherited capability family."""
        return AgentCapabilityExposition(
            workflow_group=(
                self.workflow_group if workflow_group is None else workflow_group
            ),
            workflow_stage=(
                self.workflow_stage if workflow_stage is None else workflow_stage
            ),
            target_context=(
                self.target_context if target_context is None else target_context
            ),
            visibility=self.visibility if visibility is None else visibility,
            role=self.role if role is None else role,
        )


class CapabilityViewerControlTimeoutProfile(Enum):
    """Viewer-window control timeout profile required by a capability."""

    DEFAULT = "default"
    COMMAND = "command"


class CapabilityUiBridgeTimeoutProfile(Enum):
    """UI-bridge timeout profile required by a capability."""

    DEFAULT = "default"
    COMMAND = "command"


@dataclass(frozen=True, slots=True)
class AgentScalarInputContract:
    """Nominal contract for a scalar transport field without a request DTO."""

    field_name: str
    default_value: str | None = None

    @property
    def schema_name(self) -> str:
        return self.field_name


AgentContract: TypeAlias = type | AgentScalarInputContract
AgentContextT = TypeVar("AgentContextT")
AgentServiceT = TypeVar("AgentServiceT")
AgentRequestT = TypeVar("AgentRequestT")
AgentConnectionT = TypeVar("AgentConnectionT")
AgentResultT = TypeVar("AgentResultT")


def _enum_json_value(value: Enum | None) -> str | None:
    if value is None:
        return None
    return str(value.value)


def _enum_member_title(value: Enum) -> str:
    return " ".join(
        token if token.isupper() and len(token) <= 3 else token.lower().title()
        for token in value.name.split("_")
    )


def _contract_schema_name(contract: AgentContract | None) -> str | None:
    if contract is None:
        return None
    if isinstance(contract, AgentScalarInputContract):
        return contract.schema_name
    return contract.__name__


def require_agent_type_contract(contract: AgentContract | None) -> type:
    if not isinstance(contract, type):
        raise TypeError(f"Expected agent type contract, got {contract!r}.")
    return contract


class AgentCapabilityRequestInvocationABC(
    ABC,
    Generic[AgentContextT, AgentRequestT, AgentResultT],
):
    """Nominal execution binding owned by a capability declaration."""

    def execute(
        self,
        context: AgentContextT,
        request: AgentRequestT,
    ) -> AgentResultT:
        raise NotImplementedError


class AgentCapabilityConnectionInvocationABC(
    ABC,
    Generic[AgentContextT, AgentConnectionT, AgentResultT],
):
    """Nominal connection-only execution binding owned by a capability."""

    def execute(
        self,
        context: AgentContextT,
        connection: AgentConnectionT,
    ) -> AgentResultT:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AgentConnectionServiceInvocation(
    AgentCapabilityConnectionInvocationABC[
        AgentContextT,
        AgentConnectionT,
        AgentResultT,
    ],
    Generic[AgentContextT, AgentServiceT, AgentConnectionT, AgentResultT],
):
    """Capability execution through a context service and connection."""

    service: Callable[[AgentContextT], AgentServiceT]
    method: Callable[[AgentServiceT, AgentConnectionT], AgentResultT]

    def execute(
        self,
        context: AgentContextT,
        connection: AgentConnectionT,
    ) -> AgentResultT:
        return self.method(self.service(context), connection)


class AgentCapabilityConnectionRequestInvocationABC(
    ABC,
    Generic[AgentContextT, AgentRequestT, AgentConnectionT, AgentResultT],
):
    """Nominal request+connection execution binding owned by a capability."""

    def execute(
        self,
        context: AgentContextT,
        request: AgentRequestT,
        connection: AgentConnectionT,
    ) -> AgentResultT:
        raise NotImplementedError


class AgentCapabilityConnectionScalarInvocationABC(
    ABC,
    Generic[AgentContextT, AgentConnectionT, AgentResultT],
):
    """Nominal scalar+connection execution binding owned by a capability."""

    def execute(
        self,
        context: AgentContextT,
        value: str,
        connection: AgentConnectionT,
    ) -> AgentResultT:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AgentConnectionScalarServiceInvocation(
    AgentCapabilityConnectionScalarInvocationABC[
        AgentContextT,
        AgentConnectionT,
        AgentResultT,
    ],
    Generic[AgentContextT, AgentServiceT, AgentConnectionT, AgentResultT],
):
    """Capability execution through a context service, scalar, and connection."""

    service: Callable[[AgentContextT], AgentServiceT]
    method: Callable[[AgentServiceT, str, AgentConnectionT], AgentResultT]

    def execute(
        self,
        context: AgentContextT,
        value: str,
        connection: AgentConnectionT,
    ) -> AgentResultT:
        return self.method(self.service(context), value, connection)


@dataclass(frozen=True, slots=True)
class AgentConnectionRequestServiceInvocation(
    AgentCapabilityConnectionRequestInvocationABC[
        AgentContextT,
        AgentRequestT,
        AgentConnectionT,
        AgentResultT,
    ],
    Generic[
        AgentContextT, AgentServiceT, AgentRequestT, AgentConnectionT, AgentResultT
    ],
):
    """Capability execution through a context service, request, and connection."""

    service: Callable[[AgentContextT], AgentServiceT]
    method: Callable[[AgentServiceT, AgentRequestT, AgentConnectionT], AgentResultT]
    timeout_profile: CapabilityUiBridgeTimeoutProfile = (
        CapabilityUiBridgeTimeoutProfile.DEFAULT
    )

    def execute(
        self,
        context: AgentContextT,
        request: AgentRequestT,
        connection: AgentConnectionT,
    ) -> AgentResultT:
        return self.method(self.service(context), request, connection)


class AgentCapabilityNoArgumentInvocationABC(
    ABC,
    Generic[AgentContextT, AgentResultT],
):
    """Nominal no-argument execution binding owned by a capability declaration."""

    def execute(self, context: AgentContextT) -> AgentResultT:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AgentNoArgumentFunctionInvocation(
    AgentCapabilityNoArgumentInvocationABC[AgentContextT, AgentResultT],
    Generic[AgentContextT, AgentResultT],
):
    """Capability execution through a no-argument function."""

    function: Callable[[], AgentResultT]

    def execute(self, context: AgentContextT) -> AgentResultT:
        del context
        return self.function()


class AgentCapabilityScalarInvocationABC(
    ABC,
    Generic[AgentContextT, AgentResultT],
):
    """Nominal scalar execution binding owned by a capability declaration."""

    def execute(self, context: AgentContextT, value: str) -> AgentResultT:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AgentScalarServiceInvocation(
    AgentCapabilityScalarInvocationABC[AgentContextT, AgentResultT],
    Generic[AgentContextT, AgentServiceT, AgentResultT],
):
    """Capability execution through a context service and scalar value."""

    service: Callable[[AgentContextT], AgentServiceT]
    method: Callable[[AgentServiceT, str], AgentResultT]

    def execute(self, context: AgentContextT, value: str) -> AgentResultT:
        return self.method(self.service(context), value)


@dataclass(frozen=True, slots=True)
class AgentNoArgumentServiceInvocation(
    AgentCapabilityNoArgumentInvocationABC[AgentContextT, AgentResultT],
    Generic[AgentContextT, AgentServiceT, AgentResultT],
):
    """Capability execution through a context service with no request DTO."""

    service: Callable[[AgentContextT], AgentServiceT]
    method: Callable[[AgentServiceT], AgentResultT]

    def execute(self, context: AgentContextT) -> AgentResultT:
        return self.method(self.service(context))


@dataclass(frozen=True, slots=True)
class AgentRequestServiceInvocation(
    AgentCapabilityRequestInvocationABC[
        AgentContextT,
        AgentRequestT,
        AgentResultT,
    ],
    Generic[AgentContextT, AgentServiceT, AgentRequestT, AgentResultT],
):
    """Capability execution through a context service and request DTO."""

    service: Callable[[AgentContextT], AgentServiceT]
    method: Callable[[AgentServiceT, AgentRequestT], AgentResultT]

    def execute(
        self,
        context: AgentContextT,
        request: AgentRequestT,
    ) -> AgentResultT:
        return self.method(self.service(context), request)


class AgentFromFieldsServiceInvocation(
    AgentRequestServiceInvocation[
        AgentContextT,
        AgentServiceT,
        AgentRequestT,
        AgentResultT,
    ],
    Generic[AgentContextT, AgentServiceT, AgentRequestT, AgentResultT],
):
    """Marker for request DTOs whose MCP signature comes from from_fields()."""


class AgentDataclassRequestServiceInvocation(
    AgentRequestServiceInvocation[
        AgentContextT,
        AgentServiceT,
        AgentRequestT,
        AgentResultT,
    ],
    Generic[AgentContextT, AgentServiceT, AgentRequestT, AgentResultT],
):
    """Marker for dataclass request DTOs exposed as direct MCP parameters."""


class AgentConfigPatchServiceInvocation(
    AgentRequestServiceInvocation[
        AgentContextT,
        AgentServiceT,
        AgentRequestT,
        AgentResultT,
    ],
    Generic[AgentContextT, AgentServiceT, AgentRequestT, AgentResultT],
):
    """Marker for ConfigPatch DTOs with MCP JSON-object value coercion."""


@dataclass(frozen=True, slots=True)
class AgentViewerWindowRequestServiceInvocation(
    AgentRequestServiceInvocation[
        AgentContextT,
        AgentServiceT,
        AgentRequestT,
        AgentResultT,
    ],
    Generic[AgentContextT, AgentServiceT, AgentRequestT, AgentResultT],
):
    """Marker for viewer-window request DTOs exposed through control options."""

    timeout_profile: CapabilityViewerControlTimeoutProfile = (
        CapabilityViewerControlTimeoutProfile.DEFAULT
    )


@dataclass(frozen=True, slots=True)
class AgentCapabilitySpec:
    name: str
    kind: CapabilityKind
    title: str
    description: str
    service: str
    cli_command: str | None = None
    cli_aliases: tuple[str, ...] = ()
    cli_connection_profile: CapabilityCliConnectionProfile = (
        CapabilityCliConnectionProfile.DIRECT
    )
    transport_availability: tuple[CapabilityTransport, ...] = (
        CapabilityTransport.LOCAL_STDIO,
    )
    mutating: bool = False
    side_effects: tuple[str, ...] = ()
    requires_network: bool = False
    required_extras: tuple[str, ...] = ()
    runtime_requirements: tuple[str, ...] = ()
    data_exposure: tuple[str, ...] = ()
    security_requirements: tuple[str, ...] = ()
    progress_heartbeat_seconds: float | None = None
    progress_worker_thread_safe: bool = True
    input_contract: AgentContract | None = None
    output_contract: AgentContract | None = None
    exposition: AgentCapabilityExposition | None = None

    def __post_init__(self) -> None:
        if self.progress_heartbeat_seconds is not None and (
            not isfinite(self.progress_heartbeat_seconds)
            or self.progress_heartbeat_seconds <= 0
        ):
            raise ValueError("progress_heartbeat_seconds must be positive and finite.")

    @property
    def input_type(self) -> str | None:
        return _contract_schema_name(self.input_contract)

    @property
    def output_type(self) -> str | None:
        return _contract_schema_name(self.output_contract)

    def supports_transport(self, transport: CapabilityTransport) -> bool:
        """Return whether this declaration permits registration on ``transport``."""
        return transport in self.transport_availability

    def supports_surface_profile(self, profile: LocalCapabilitySurfaceProfile) -> bool:
        """Return whether the profile permits this declared visibility tier."""
        return profile.includes(self)

    @property
    def read_only(self) -> bool:
        """Return the declaration-owned mutation classification."""
        return not self.mutating and not self.side_effects

    @property
    def workflow_group(self) -> CapabilityWorkflowGroup | None:
        if self.exposition is None:
            return None
        return self.exposition.workflow_group

    @property
    def workflow_stage(self) -> CapabilityWorkflowStage | None:
        if self.exposition is None:
            return None
        return self.exposition.workflow_stage

    @property
    def target_context(self) -> CapabilityTargetContext | None:
        if self.exposition is None:
            return None
        return self.exposition.target_context

    @property
    def visibility(self) -> CapabilityVisibility | None:
        if self.exposition is None:
            return None
        return self.exposition.visibility

    @property
    def role(self) -> CapabilityRole | None:
        if self.exposition is None:
            return None
        return self.exposition.role

    def as_jsonable(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "title": self.title,
            "description": self.description,
            "service": self.service,
            "cli_command": self.cli_command,
            "cli_aliases": list(self.cli_aliases),
            "cli_connection_profile": self.cli_connection_profile.value,
            "transport_availability": [
                transport.value for transport in self.transport_availability
            ],
            "mutating": self.mutating,
            "side_effects": list(self.side_effects),
            "requires_network": self.requires_network,
            "required_extras": list(self.required_extras),
            "runtime_requirements": list(self.runtime_requirements),
            "data_exposure": list(self.data_exposure),
            "security_requirements": list(self.security_requirements),
            "progress_heartbeat_seconds": self.progress_heartbeat_seconds,
            "progress_worker_thread_safe": self.progress_worker_thread_safe,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "workflow_group": _enum_json_value(self.workflow_group),
            "workflow_stage": _enum_json_value(self.workflow_stage),
            "target_context": _enum_json_value(self.target_context),
            "visibility": _enum_json_value(self.visibility),
            "role": _enum_json_value(self.role),
        }


@dataclass(frozen=True, slots=True)
class AgentCapabilitySurfaceSelection:
    """Transport and local-profile policy used by every capability consumer."""

    transport: CapabilityTransport | None = None
    local_profile: LocalCapabilitySurfaceProfile = field(
        default_factory=FullLocalCapabilitySurfaceProfile
    )

    def includes(self, capability: AgentCapabilitySpec) -> bool:
        return (
            self.transport is None or capability.supports_transport(self.transport)
        ) and capability.supports_surface_profile(self.local_profile)


class AgentCapabilityDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Registered declaration for one agent-facing capability."""

    __registry__: ClassVar[dict[str, type["AgentCapabilityDeclaration"]]] = {}
    __registry_key__ = "name"
    __skip_if_no_key__ = True

    name: ClassVar[str | None] = None
    kind: ClassVar[CapabilityKind]
    title: ClassVar[str]
    description: ClassVar[str]
    service: ClassVar[str]
    cli_command: ClassVar[str | None] = None
    cli_aliases: ClassVar[tuple[str, ...]] = ()
    cli_connection_profile: ClassVar[CapabilityCliConnectionProfile] = (
        CapabilityCliConnectionProfile.DIRECT
    )
    transport_availability: ClassVar[tuple[CapabilityTransport, ...]] = (
        CapabilityTransport.LOCAL_STDIO,
    )
    mutating: ClassVar[bool] = False
    side_effects: ClassVar[tuple[str, ...]] = ()
    requires_network: ClassVar[bool] = False
    required_extras: ClassVar[tuple[str, ...]] = ()
    runtime_requirements: ClassVar[tuple[str, ...]] = ()
    data_exposure: ClassVar[tuple[str, ...]] = ()
    security_requirements: ClassVar[tuple[str, ...]] = ()
    progress_heartbeat_seconds: ClassVar[float | None] = None
    progress_worker_thread_safe: ClassVar[bool] = True
    input_contract: ClassVar[AgentContract | None] = None
    output_contract: ClassVar[AgentContract | None] = None
    exposition: ClassVar[AgentCapabilityExposition | None] = None
    no_argument_invocation: ClassVar[AgentCapabilityNoArgumentInvocationABC | None] = (
        None
    )
    connection_invocation: ClassVar[AgentCapabilityConnectionInvocationABC | None] = (
        None
    )
    connection_request_invocation: ClassVar[
        AgentCapabilityConnectionRequestInvocationABC | None
    ] = None
    connection_scalar_invocation: ClassVar[
        AgentCapabilityConnectionScalarInvocationABC | None
    ] = None
    scalar_invocation: ClassVar[AgentCapabilityScalarInvocationABC | None] = None
    request_invocation: ClassVar[AgentCapabilityRequestInvocationABC | None] = None

    @classmethod
    def execute_no_argument(
        cls,
        context: AgentContextT,
    ) -> AgentResultT:
        if cls.no_argument_invocation is None:
            raise TypeError(f"{cls.__name__} does not declare no-argument invocation.")
        return cls.no_argument_invocation.execute(context)

    @classmethod
    def execute_connection(
        cls,
        context: AgentContextT,
        connection: AgentConnectionT,
    ) -> AgentResultT:
        if cls.connection_invocation is None:
            raise TypeError(f"{cls.__name__} does not declare connection invocation.")
        return cls.connection_invocation.execute(context, connection)

    @classmethod
    def execute_scalar(
        cls,
        context: AgentContextT,
        value: str,
    ) -> AgentResultT:
        if cls.scalar_invocation is None:
            raise TypeError(f"{cls.__name__} does not declare scalar invocation.")
        return cls.scalar_invocation.execute(context, value)

    @classmethod
    def execute_connection_scalar(
        cls,
        context: AgentContextT,
        value: str,
        connection: AgentConnectionT,
    ) -> AgentResultT:
        if cls.connection_scalar_invocation is None:
            raise TypeError(
                f"{cls.__name__} does not declare connection scalar invocation."
            )
        return cls.connection_scalar_invocation.execute(context, value, connection)

    @classmethod
    def execute_connection_request(
        cls,
        context: AgentContextT,
        request: AgentRequestT,
        connection: AgentConnectionT,
    ) -> AgentResultT:
        if cls.connection_request_invocation is None:
            raise TypeError(
                f"{cls.__name__} does not declare connection request invocation."
            )
        return cls.connection_request_invocation.execute(context, request, connection)

    @classmethod
    def execute_request(
        cls,
        context: AgentContextT,
        request: AgentRequestT,
    ) -> AgentResultT:
        if cls.request_invocation is None:
            raise TypeError(f"{cls.__name__} does not declare request invocation.")
        return cls.request_invocation.execute(context, request)

    @classmethod
    def to_spec(cls) -> AgentCapabilitySpec:
        if cls.name is None:
            raise ValueError(f"{cls.__name__} must declare a capability name.")
        return AgentCapabilitySpec(
            name=cls.name,
            kind=cls.kind,
            title=cls.title,
            description=cls.description,
            service=cls.service,
            cli_command=cls.cli_command,
            cli_aliases=cls.cli_aliases,
            cli_connection_profile=cls.cli_connection_profile,
            transport_availability=cls.transport_availability,
            mutating=cls.mutating,
            side_effects=cls.side_effects,
            requires_network=cls.requires_network,
            required_extras=cls.required_extras,
            runtime_requirements=cls.runtime_requirements,
            data_exposure=cls.data_exposure,
            security_requirements=cls.security_requirements,
            progress_heartbeat_seconds=cls.progress_heartbeat_seconds,
            progress_worker_thread_safe=cls.progress_worker_thread_safe,
            input_contract=cls.input_contract,
            output_contract=cls.output_contract,
            exposition=cls.exposition,
        )


class HostedTransportCapabilityMixin:
    """Nominal opt-in for capabilities audited as safe on a hosted server."""

    transport_availability: ClassVar[tuple[CapabilityTransport, ...]] = (
        CapabilityTransport.LOCAL_STDIO,
        CapabilityTransport.HOSTED_STREAMABLE_HTTP,
    )


class DiscoveryCapability(AgentCapabilityDeclaration):
    """Capability exposed in the initial server/capability discovery lane."""

    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.DISCOVERY,
        workflow_stage=CapabilityWorkflowStage.DISCOVERY,
        target_context=CapabilityTargetContext.SERVER,
        visibility=CapabilityVisibility.BEGINNER,
    )


class KnowledgeCapability(AgentCapabilityDeclaration):
    """Capability that exposes bounded documentation or authoring context."""

    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.KNOWLEDGE,
        workflow_stage=CapabilityWorkflowStage.CONTEXT,
        target_context=CapabilityTargetContext.KNOWLEDGE_BASE,
        visibility=CapabilityVisibility.BEGINNER,
    )


class ArchitectureCapability(
    HostedTransportCapabilityMixin,
    KnowledgeCapability,
):
    """Capability that exposes source-backed OpenHCS architecture facts."""

    exposition = KnowledgeCapability.exposition.refine(
        target_context=CapabilityTargetContext.ARCHITECTURE_MODEL,
    )


class FunctionCatalogCapability(AgentCapabilityDeclaration):
    """Capability that reads or extends the processing-function catalog."""

    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.FUNCTION_AUTHORING,
        workflow_stage=CapabilityWorkflowStage.AUTHORING,
        target_context=CapabilityTargetContext.FUNCTION_REGISTRY,
        visibility=CapabilityVisibility.BEGINNER,
    )


class ConfigDraftCapability(AgentCapabilityDeclaration):
    """Capability that works with typed configuration draft state."""

    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.PIPELINE_AUTHORING,
        workflow_stage=CapabilityWorkflowStage.AUTHORING,
        target_context=CapabilityTargetContext.CONFIG_DRAFT,
        visibility=CapabilityVisibility.STANDARD,
    )


class PipelineDraftCapability(AgentCapabilityDeclaration):
    """Capability that works with pipeline draft or source planning state."""

    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.PIPELINE_AUTHORING,
        workflow_stage=CapabilityWorkflowStage.AUTHORING,
        target_context=CapabilityTargetContext.PIPELINE_DRAFT,
        visibility=CapabilityVisibility.STANDARD,
    )


class PlatePathCapability(AgentCapabilityDeclaration):
    """Capability that works from an explicit local plate path."""

    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.PLATE_DATA,
        workflow_stage=CapabilityWorkflowStage.DATA_PREPARATION,
        target_context=CapabilityTargetContext.PLATE_PATH,
        visibility=CapabilityVisibility.BEGINNER,
    )


class HeadlessExecutionCapability(AgentCapabilityDeclaration):
    """Capability that works with headless execution sessions or jobs."""

    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.HEADLESS_EXECUTION,
        workflow_stage=CapabilityWorkflowStage.EXECUTION,
        target_context=CapabilityTargetContext.HEADLESS_SESSION,
        visibility=CapabilityVisibility.STANDARD,
    )


class SubmittedJobCapability(HeadlessExecutionCapability):
    """Capability that observes a submitted compile or execution job."""

    exposition = HeadlessExecutionCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.STATUS,
        target_context=CapabilityTargetContext.SUBMITTED_JOB,
        role=CapabilityRole.DIAGNOSTIC,
    )


class UiBridgeCapability(AgentCapabilityDeclaration):
    """Capability that targets the running PyQt UI bridge."""

    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.UI_CONTROL,
        workflow_stage=CapabilityWorkflowStage.CONTROL,
        target_context=CapabilityTargetContext.UI_BRIDGE,
        visibility=CapabilityVisibility.STANDARD,
    )


class UiBridgeCliConnectionCapability(UiBridgeCapability):
    """Capability whose CLI command accepts UI bridge connection options."""

    cli_connection_profile = CapabilityCliConnectionProfile.UI_BRIDGE


class UiSelectedPlateCapability(UiBridgeCliConnectionCapability):
    """Capability that uses the current PlateManager selection as its plate."""

    exposition = UiBridgeCliConnectionCapability.exposition.refine(
        workflow_group=CapabilityWorkflowGroup.UI_SELECTED_PLATE,
        workflow_stage=CapabilityWorkflowStage.DATA_PREPARATION,
        target_context=CapabilityTargetContext.UI_SELECTED_PLATE,
        role=CapabilityRole.MODE_VARIANT,
    )


class UiWindowCapability(UiBridgeCliConnectionCapability):
    """Capability that targets a visible or focusable PyQt UI window."""

    exposition = UiBridgeCliConnectionCapability.exposition.refine(
        target_context=CapabilityTargetContext.UI_WINDOW,
    )


class UiSemanticActionCapability(UiBridgeCliConnectionCapability):
    """Capability that invokes declared semantic UI actions."""

    exposition = UiBridgeCliConnectionCapability.exposition


class UiWidgetFallbackCapability(UiWindowCapability):
    """Capability that uses generic widget projection as a fallback control."""

    exposition = UiWindowCapability.exposition.refine(
        visibility=CapabilityVisibility.EXPERT,
        role=CapabilityRole.FALLBACK,
    )


class UiCodeDocumentCapability(UiBridgeCliConnectionCapability):
    """Capability that targets UI-owned pycodified code documents."""

    exposition = UiBridgeCliConnectionCapability.exposition.refine(
        workflow_group=CapabilityWorkflowGroup.UI_STATE_EDITING,
        workflow_stage=CapabilityWorkflowStage.STATE_EDITING,
        target_context=CapabilityTargetContext.UI_CODE_DOCUMENT,
    )


class UiObjectStateCapability(UiBridgeCliConnectionCapability):
    """Capability that targets typed ObjectState scopes and fields."""

    exposition = UiBridgeCliConnectionCapability.exposition.refine(
        workflow_group=CapabilityWorkflowGroup.UI_STATE_EDITING,
        workflow_stage=CapabilityWorkflowStage.STATE_EDITING,
        target_context=CapabilityTargetContext.UI_OBJECT_STATE,
        visibility=CapabilityVisibility.EXPERT,
    )


class UiSnapshotCapability(UiObjectStateCapability):
    """Capability that targets ObjectState snapshot or branch time travel."""

    exposition = UiObjectStateCapability.exposition.refine(
        role=CapabilityRole.EXPERT,
    )


class ViewerWindowCliConnectionCapability(AgentCapabilityDeclaration):
    """Capability whose CLI command accepts viewer-window connection options."""

    cli_connection_profile = CapabilityCliConnectionProfile.VIEWER_WINDOW
    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.VIEWER_REVIEW,
        workflow_stage=CapabilityWorkflowStage.INSPECTION,
        target_context=CapabilityTargetContext.VIEWER_WINDOW,
        visibility=CapabilityVisibility.STANDARD,
    )


class RuntimeServerCliConnectionCapability(AgentCapabilityDeclaration):
    """Capability whose CLI command accepts runtime-server connection options."""

    cli_connection_profile = CapabilityCliConnectionProfile.RUNTIME_SERVER
    exposition = AgentCapabilityExposition(
        workflow_group=CapabilityWorkflowGroup.RUNTIME_DIAGNOSTICS,
        workflow_stage=CapabilityWorkflowStage.DIAGNOSTIC,
        target_context=CapabilityTargetContext.RUNTIME_SERVER,
        visibility=CapabilityVisibility.EXPERT,
        role=CapabilityRole.DIAGNOSTIC,
    )


@dataclass(frozen=True, slots=True)
class AgentCapabilityGroup:
    workflow_group: CapabilityWorkflowGroup
    capability_names: tuple[str, ...]
    tool_count: int
    resource_count: int

    @property
    def title(self) -> str:
        return self.workflow_group.title


@dataclass(frozen=True, slots=True)
class AgentCapabilityRegistry:
    schema_version: str
    capabilities: tuple[AgentCapabilitySpec, ...]
    groups: tuple[AgentCapabilityGroup, ...] = ()
    surface_profile: str = FullLocalCapabilitySurfaceProfile.name

    @property
    def non_read_only_tools(self) -> tuple[AgentCapabilitySpec, ...]:
        """Return tools whose declarations permit mutation or side effects."""
        return tuple(
            capability
            for capability in self.capabilities
            if capability.kind is CapabilityKind.TOOL and not capability.read_only
        )


class AgentCapabilityNamespace:
    """Attribute namespace generated from declared capability ABI names."""

    def __init__(self, capabilities: tuple[AgentCapabilitySpec, ...]) -> None:
        object.__setattr__(self, "_capabilities", capabilities)
        for capability in capabilities:
            object.__setattr__(
                self,
                _capability_attribute_name(capability.name),
                capability,
            )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable.")


@to_jsonable.register(AgentCapabilitySpec)
def _jsonable_agent_capability_spec(value: AgentCapabilitySpec) -> dict[str, object]:
    return value.as_jsonable()


@to_jsonable.register(AgentCapabilityGroup)
def _jsonable_agent_capability_group(value: AgentCapabilityGroup) -> dict[str, object]:
    return {
        "workflow_group": value.workflow_group.value,
        "title": value.title,
        "capability_names": list(value.capability_names),
        "tool_count": value.tool_count,
        "resource_count": value.resource_count,
    }


@to_jsonable.register(AgentCapabilityRegistry)
def _jsonable_agent_capability_registry(
    value: AgentCapabilityRegistry,
) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "surface_profile": value.surface_profile,
        "capabilities": [to_jsonable(capability) for capability in value.capabilities],
        "groups": [to_jsonable(group) for group in value.groups],
    }


TOPIC_ID_INPUT = AgentScalarInputContract("topic_id", default_value="pipeline_model")
SYMBOL_ID_INPUT = AgentScalarInputContract("symbol_id")
OPERATION_ID_INPUT = AgentScalarInputContract("operation_id")


class CapabilitiesResourceCapability(
    HostedTransportCapabilityMixin,
    DiscoveryCapability,
):
    name = "openhcs://capabilities"
    kind = CapabilityKind.RESOURCE
    title = "OpenHCS agent capability registry"
    description = (
        "Lists the resources, tools, side effects, and extras exposed by this server."
    )
    service = "capability_registry"
    output_contract = AgentCapabilityRegistry
    no_argument_invocation = AgentNoArgumentFunctionInvocation(
        function=lambda: get_capability_registry(),
    )


class HealthCheckCapability(DiscoveryCapability):
    name = "openhcs_health_check"
    cli_command = "health"
    kind = CapabilityKind.TOOL
    title = "Health check"
    description = (
        "Reports OpenHCS MCP health, installed OpenHCS version, packaged-resource "
        "readiness, server process identity, source freshness, installation-generation "
        "freshness, and the client-owned reconnect contract for stale-process "
        "diagnostics."
    )
    service = "capability_registry"
    exposition = DiscoveryCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.DIAGNOSTIC,
        role=CapabilityRole.DIAGNOSTIC,
    )
    data_exposure = (
        "installed_openhcs_version",
        "packaged_resource_readiness",
        "packaged_resource_paths",
        "mcp_process_identity",
        "mcp_source_freshness",
        "mcp_installation_generation",
    )
    output_contract = McpServerHealthResult


class ListCapabilitiesCapability(
    HostedTransportCapabilityMixin,
    DiscoveryCapability,
):
    name = "openhcs_list_capabilities"
    kind = CapabilityKind.TOOL
    title = "List capabilities"
    description = "Returns the canonical agent capability registry."
    service = "capability_registry"
    output_contract = AgentCapabilityRegistry
    no_argument_invocation = AgentNoArgumentFunctionInvocation(
        function=lambda: get_capability_registry(),
    )


class SearchFunctionsCapability(
    HostedTransportCapabilityMixin,
    FunctionCatalogCapability,
):
    name = "openhcs_search_functions"
    cli_command = "functions"
    kind = CapabilityKind.TOOL
    title = "Search processing functions"
    description = (
        "Searches the OpenHCS function registry by name, module, library, tag, "
        "or doc text. The library selector accepts either the registry library "
        "or an exact declaration-owned backend tag."
    )
    service = "function_catalog"
    input_contract = FunctionSearchRequest
    output_contract = FunctionCatalogPage
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.function_catalog,
        method=lambda service, request: service.search(
            query=request.query,
            library=request.library,
            limit=request.limit,
            compact_signatures=request.compact_signatures,
        ),
    )


class DescribeFunctionCapability(
    HostedTransportCapabilityMixin,
    FunctionCatalogCapability,
):
    name = "openhcs_describe_function"
    cli_command = "function"
    kind = CapabilityKind.TOOL
    title = "Describe processing function"
    description = (
        "Returns signature, parameter, and bounded documentation details "
        "for one registry function."
    )
    service = "function_catalog"
    input_contract = FunctionDetailRequest
    output_contract = FunctionDetail
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.function_catalog,
        method=lambda service, request: service.get(
            request.function_id,
            max_doc_chars=request.max_doc_chars,
            compact_signature=request.compact_signature,
        ),
    )


class RegisterCustomFunctionCapability(FunctionCatalogCapability):
    name = "openhcs_register_custom_function"
    cli_command = "register-custom-function"
    kind = CapabilityKind.TOOL
    title = "Register custom function"
    description = (
        "Validates, registers, and optionally persists custom function Python "
        "source through CustomFunctionManager, then returns registry function_id "
        "values for MCP pipeline authoring."
    )
    service = "function_catalog"
    exposition = FunctionCatalogCapability.exposition.refine(
        visibility=CapabilityVisibility.EXPERT,
        role=CapabilityRole.EXPERT,
    )
    mutating = True
    side_effects = ("writes_custom_function_file", "updates_function_registry")
    input_contract = CustomFunctionRegistrationRequest
    output_contract = CustomFunctionRegistrationResult
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.function_catalog,
        method=lambda service, request: service.register_custom_function(request),
    )


class GetAuthoringContextCapability(KnowledgeCapability):
    name = "openhcs_get_authoring_context"
    cli_command = "authoring-context"
    kind = CapabilityKind.TOOL
    title = "Get authoring context"
    description = (
        "Returns bounded prompt/context text for agents authoring OpenHCS code. "
        "Agents that do not already know OpenHCS should request kind='first_use' "
        "for a compact orientation and intent router before choosing tools, then "
        "request only the task-specific context it recommends."
    )
    service = "llm_context"
    input_contract = AuthoringContextRequest
    output_contract = AuthoringContext
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.authoring_context_service,
        method=lambda service, request: service.get_bounded_authoring_context(request),
    )


class KnowledgeResourceCapability(
    HostedTransportCapabilityMixin,
    KnowledgeCapability,
):
    name = "openhcs://knowledge"
    kind = CapabilityKind.RESOURCE
    title = "OpenHCS agent knowledge base"
    description = "Lists source-backed OpenHCS documentation available to agents."
    service = "knowledge_base"
    data_exposure = ("local_documentation_paths",)
    output_contract = KnowledgeBaseCatalog
    no_argument_invocation = AgentNoArgumentServiceInvocation(
        service=lambda context: context.knowledge_base_service,
        method=lambda service: service.list_documents(),
    )


class ListKnowledgeDocumentsCapability(
    HostedTransportCapabilityMixin,
    KnowledgeCapability,
):
    name = "openhcs_list_knowledge_documents"
    cli_command = "knowledge"
    kind = CapabilityKind.TOOL
    title = "List knowledge documents"
    description = "Lists source-backed OpenHCS documentation available through the MCP knowledge base."
    service = "knowledge_base"
    data_exposure = ("local_documentation_paths",)
    output_contract = KnowledgeBaseCatalog
    no_argument_invocation = AgentNoArgumentServiceInvocation(
        service=lambda context: context.knowledge_base_service,
        method=lambda service: service.list_documents(),
    )


class GetKnowledgeDocumentCapability(
    HostedTransportCapabilityMixin,
    KnowledgeCapability,
):
    name = "openhcs_get_knowledge_document"
    cli_command = "knowledge-document"
    kind = CapabilityKind.TOOL
    title = "Get knowledge document"
    description = (
        "Returns one bounded allowlisted OpenHCS documentation document or section."
    )
    service = "knowledge_base"
    data_exposure = ("local_documentation_paths", "documentation_content")
    input_contract = KnowledgeBaseDocumentRequest
    output_contract = KnowledgeBaseDocument
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.knowledge_base_service,
        method=lambda service, request: service.get_document(request),
    )


class SearchKnowledgeCapability(
    HostedTransportCapabilityMixin,
    KnowledgeCapability,
):
    name = "openhcs_search_knowledge"
    cli_command = "knowledge-search"
    kind = CapabilityKind.TOOL
    title = "Search knowledge base"
    description = "Searches the allowlisted OpenHCS documentation knowledge base."
    service = "knowledge_base"
    data_exposure = ("local_documentation_paths", "documentation_content_snippets")
    input_contract = KnowledgeBaseSearchRequest
    output_contract = KnowledgeBaseSearchResult
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.knowledge_base_service,
        method=lambda service, request: service.search(request),
    )


class GenerateSyntheticPlateCapability(PlatePathCapability):
    name = "openhcs_generate_synthetic_plate"
    cli_command = "generate-synthetic-plate"
    cli_aliases = ("synthetic-plate",)
    kind = CapabilityKind.TOOL
    title = "Generate synthetic plate"
    description = (
        "Generates a bounded synthetic microscopy plate using the same "
        "SyntheticMicroscopyGenerator surfaced by the UI generator window. "
        "Use it to create small multi-channel, overlapping-site fixtures "
        "before inspecting them with openhcs_inspect_plate_path."
    )
    service = "synthetic_plate_generation"
    mutating = True
    side_effects = ("writes_local_plate_files",)
    data_exposure = ("local_output_path", "generated_image_file_names")
    security_requirements = ("AgentPathPolicy writable root",)
    input_contract = SyntheticPlateGenerationRequest
    output_contract = SyntheticPlateGenerationResult
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.synthetic_plate_service,
        method=lambda service, request: service.generate(request),
    )


class InspectPlatePathCapability(PlatePathCapability):
    name = "openhcs_inspect_plate_path"
    cli_command = "inspect-plate"
    kind = CapabilityKind.TOOL
    title = "Inspect plate path"
    description = (
        "Diagnostic-only, read-only inspection of a local plate folder: microscope handler "
        "detection, microscope metadata, image-file samples, filename parse "
        "coverage, registry-derived format-specific candidate evidence, "
        "workspace-preparation advice, and structured workflow routing. "
        "It does not configure a running UI or make a handler override the setup "
        "route; use the PlateManager code document plus selected-plate init when "
        "the result must remain visible in the desktop."
    )
    service = "selected_plate"
    data_exposure = (
        "local_plate_path",
        "microscope_metadata",
        "image_file_names",
        "result_artifact_names",
        "filename_parse_summaries",
    )
    security_requirements = ("AgentPathPolicy readable root",)
    input_contract = PlatePathInspectionRequest
    output_contract = PlatePathInspectionResult
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.plate_inspection_service,
        method=lambda service, request: service.inspect(request),
    )


class QueryPlateFilesCapability(PlatePathCapability):
    name = "openhcs_query_plate_files"
    cli_command = "query-plate-files"
    kind = CapabilityKind.TOOL
    title = "Query plate files"
    description = (
        "Read-only query of image/result file records exposed "
        "by a local plate inventory. Returns virtual image names, source "
        "paths, result artifact paths, and metadata from the same inventory "
        "API used by the Image Browser."
    )
    service = "plate_inspection"
    data_exposure = (
        "local_plate_path",
        "plate_virtual_image_path",
        "plate_source_image_path",
        "result_artifact_names",
        "file_metadata",
    )
    security_requirements = ("AgentPathPolicy readable root",)
    input_contract = PlateFileQueryRequest
    output_contract = PlateFileQueryResult
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.plate_inspection_service,
        method=lambda service, request: service.query_files(request),
    )


class SamplePlateImageCapability(PlatePathCapability):
    name = "openhcs_sample_plate_image"
    cli_command = "sample-plate-image"
    kind = CapabilityKind.TOOL
    title = "Sample plate image"
    description = (
        "Resolves a plate image by virtual/source path, full virtual path, "
        "or unique basename, then reads bounded pixels from a native-resolution "
        "region and returns its statistics scope, source/resolution shapes, selected "
        "resolution, and downsampling provenance. Omit resolution_index for safe "
        "automatic selection or pass 0 for exact full-resolution pixels."
    )
    service = "plate_inspection"
    data_exposure = (
        "local_plate_path",
        "plate_virtual_image_path",
        "plate_source_image_path",
        "bounded_image_pixels",
    )
    security_requirements = ("AgentPathPolicy readable root",)
    input_contract = PlateImageSampleRequest
    output_contract = PlateImageSampleResult
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.plate_inspection_service,
        method=lambda service, request: service.sample_image(request),
    )


class StreamPlateFilesToViewerCapability(PlatePathCapability):
    name = "openhcs_stream_plate_files_to_viewer"
    cli_command = "stream-plate-files"
    kind = CapabilityKind.TOOL
    title = "Stream plate files to viewer"
    description = (
        "Resolves image or ROI result records by virtual path, source path, "
        "result path, basename, or bounded inventory query, then streams them "
        "to a managed viewer through the same core service used by the Image Browser."
    )
    service = "plate_streaming"
    data_exposure = (
        "local_plate_path",
        "plate_virtual_image_path",
        "plate_source_image_path",
        "result_artifact_names",
        "viewer_connection",
    )
    runtime_requirements = ("napari_or_fiji_viewer_runtime",)
    security_requirements = ("AgentPathPolicy readable root",)
    input_contract = PlateFileStreamRequest
    output_contract = PlateFileStreamResult
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.plate_streaming_service,
        method=lambda service, request: service.stream_files(request),
    )


class UiInspectSelectedPlateImagesCapability(UiSelectedPlateCapability):
    name = "openhcs_ui_inspect_selected_plate_images"
    cli_command = "selected-plate-images"
    kind = CapabilityKind.TOOL
    title = "Inspect selected plate images"
    description = (
        "Reads the current PlateManager selection from the running UI bridge, "
        "requires exactly one selected plate, resolves the selected, source, "
        "or output plate target, then returns the same read-only image inventory "
        "and microscope metadata produced by openhcs_inspect_plate_path."
    )
    service = "plate_inspection"
    data_exposure = (
        "ui_selected_plate_path",
        "microscope_metadata",
        "image_file_names",
        "result_artifact_names",
        "filename_parse_summaries",
    )
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token", "AgentPathPolicy readable root")
    input_contract = SelectedPlateImageInspectionRequest
    output_contract = SelectedPlateImageInspectionResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.selected_plate_service,
        method=lambda service, request, connection: service.inspect_images(
            request,
            connection,
        ),
    )


class UiQuerySelectedPlateFilesCapability(UiSelectedPlateCapability):
    name = "openhcs_ui_query_selected_plate_files"
    cli_command = "selected-plate-files"
    kind = CapabilityKind.TOOL
    title = "Query selected plate files"
    description = (
        "Reads the current PlateManager selection from the running UI bridge, "
        "requires exactly one selected plate, then returns the same image/result "
        "file records produced by openhcs_query_plate_files."
    )
    service = "selected_plate"
    data_exposure = (
        "ui_selected_plate_path",
        "plate_virtual_image_path",
        "plate_source_image_path",
        "result_artifact_names",
        "file_metadata",
    )
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token", "AgentPathPolicy readable root")
    input_contract = SelectedPlateFileQueryRequest
    output_contract = SelectedPlateFileQueryResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.selected_plate_service,
        method=lambda service, request, connection: service.query_files(
            request,
            connection,
        ),
    )


class UiSampleSelectedPlateImageCapability(UiSelectedPlateCapability):
    name = "openhcs_ui_sample_selected_plate_image"
    cli_command = "selected-plate-sample"
    kind = CapabilityKind.TOOL
    title = "Sample selected plate image"
    description = (
        "Sample a selected-plate image after reading the current PlateManager "
        "selection from the running UI bridge, "
        "then reads a bounded native-resolution region from a selected/source/output "
        "plate image by virtual/source path. Omit resolution_index for safe automatic "
        "selection or pass 0 for exact full-resolution pixels. If no "
        "image_path is supplied, it deterministically samples the first image "
        "reported by openhcs_inspect_plate_path."
    )
    service = "selected_plate"
    data_exposure = (
        "ui_selected_plate_path",
        "plate_virtual_image_path",
        "plate_source_image_path",
        "bounded_image_pixels",
    )
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token", "AgentPathPolicy readable root")
    input_contract = SelectedPlateImageSampleRequest
    output_contract = SelectedPlateImageSampleResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.selected_plate_service,
        method=lambda service, request, connection: service.sample_image(
            request,
            connection,
        ),
    )


class UiStreamSelectedPlateFilesToViewerCapability(UiSelectedPlateCapability):
    name = "openhcs_ui_stream_selected_plate_files_to_viewer"
    cli_command = "selected-plate-stream"
    kind = CapabilityKind.TOOL
    title = "Stream selected plate files to viewer"
    description = (
        "Reads the current PlateManager selection from the running UI bridge, "
        "resolves selected/source/output image or ROI records through the same "
        "inventory API as openhcs_ui_query_selected_plate_files, then streams "
        "them to a managed viewer."
    )
    service = "selected_plate"
    data_exposure = (
        "ui_selected_plate_path",
        "plate_virtual_image_path",
        "plate_source_image_path",
        "result_artifact_names",
        "viewer_connection",
    )
    runtime_requirements = (
        "running_openhcs_ui_bridge",
        "napari_or_fiji_viewer_runtime",
    )
    security_requirements = ("ui_bridge_auth_token", "AgentPathPolicy readable root")
    input_contract = SelectedPlateFileStreamRequest
    output_contract = SelectedPlateFileStreamResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.selected_plate_service,
        method=lambda service, request, connection: service.stream_files(
            request,
            connection,
        ),
    )


class ArchitectureTopicsResourceCapability(ArchitectureCapability):
    name = "openhcs://architecture/topics"
    kind = CapabilityKind.RESOURCE
    title = "Architecture topics"
    description = (
        "Lists read-only architecture topics backed by real OpenHCS internal symbols."
    )
    service = "architecture_projection"
    output_contract = ArchitectureTopicPage
    no_argument_invocation = AgentNoArgumentServiceInvocation(
        service=lambda context: context.architecture_service,
        method=lambda service: service.list_topics(),
    )


class ListArchitectureTopicsCapability(ArchitectureCapability):
    name = "openhcs_list_architecture_topics"
    cli_command = "architecture"
    kind = CapabilityKind.TOOL
    title = "List architecture topics"
    description = "Lists architecture topics available to agents."
    service = "architecture_projection"
    output_contract = ArchitectureTopicPage
    no_argument_invocation = AgentNoArgumentServiceInvocation(
        service=lambda context: context.architecture_service,
        method=lambda service: service.list_topics(),
    )


class ExplainArchitectureCapability(ArchitectureCapability):
    name = "openhcs_explain_architecture"
    cli_command = "architecture-topic"
    cli_aliases = ("explain-architecture",)
    kind = CapabilityKind.TOOL
    title = "Explain architecture topic"
    description = "Explains one OpenHCS architecture topic using source-backed internal API symbols."
    service = "architecture_projection"
    input_contract = TOPIC_ID_INPUT
    output_contract = ArchitectureTopic
    scalar_invocation = AgentScalarServiceInvocation(
        service=lambda context: context.architecture_service,
        method=lambda service, value: service.explain_topic(value),
    )


class DescribeInternalSymbolCapability(ArchitectureCapability):
    name = "openhcs_describe_internal_symbol"
    cli_command = "internal-symbol"
    cli_aliases = ("architecture-symbol",)
    kind = CapabilityKind.TOOL
    title = "Describe internal symbol"
    description = "Returns read-only signature/doc/source-location facts for one internal OpenHCS symbol."
    service = "architecture_projection"
    input_contract = SYMBOL_ID_INPUT
    output_contract = InternalApiSymbol
    scalar_invocation = AgentScalarServiceInvocation(
        service=lambda context: context.architecture_service,
        method=lambda service, value: service.describe_internal_symbol(value),
    )


class DescribeConfigSchemaCapability(
    HostedTransportCapabilityMixin,
    ConfigDraftCapability,
):
    name = "openhcs_describe_config_schema"
    cli_command = "config-schema"
    kind = CapabilityKind.TOOL
    title = "Describe configuration schema"
    description = (
        "Reflects GlobalPipelineConfig, PipelineConfig, or the FunctionStep config "
        "override surface without materializing lazy values. With no path_prefix it "
        "returns the top-level owner-derived map; pass a returned nested_schema_path "
        "to retrieve that subtree. Use config_type='step' for exact "
        "FunctionStepAddRequest.step_config_overrides keys."
    )
    service = "config"
    input_contract = ConfigSchemaRequest
    output_contract = ConfigSchema
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.config_service,
        method=lambda service, request: service.describe_schema_request(request),
    )


class CreateConfigCapability(ConfigDraftCapability):
    name = "openhcs_create_config"
    kind = CapabilityKind.TOOL
    title = "Create configuration"
    description = "Creates a draft config reference from a typed config patch."
    service = "config"
    mutating = True
    side_effects = ("creates_in_memory_config_ref",)
    input_contract = ConfigPatch
    output_contract = ConfigRef
    request_invocation = AgentConfigPatchServiceInvocation(
        service=lambda context: context.config_service,
        method=lambda service, request: service.create(
            request.config_type,
            request,
        ),
    )


class ValidateConfigPatchCapability(ConfigDraftCapability):
    name = "openhcs_validate_config_patch"
    kind = CapabilityKind.TOOL
    title = "Validate configuration patch"
    description = (
        "Validates that a config patch can instantiate the target OpenHCS config class."
    )
    service = "config"
    exposition = ConfigDraftCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.VALIDATION,
    )
    input_contract = ConfigPatch
    output_contract = ConfigValidationResult
    request_invocation = AgentConfigPatchServiceInvocation(
        service=lambda context: context.config_service,
        method=lambda service, request: service.validate_patch(
            request.config_type,
            request,
        ),
    )


class RenderConfigSourceCapability(ConfigDraftCapability):
    name = "openhcs_render_config_source"
    kind = CapabilityKind.TOOL
    title = "Render configuration source"
    description = "Renders a draft config reference as Python source using OpenHCS pycodify formatters."
    service = "config"
    input_contract = ConfigSourceRenderRequest
    output_contract = RenderedSource
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.config_service,
        method=lambda service, request: service.render_source(
            request.config_id,
            clean=request.clean,
        ),
    )


class CreatePipelineCapability(PipelineDraftCapability):
    name = "openhcs_create_pipeline"
    kind = CapabilityKind.TOOL
    title = "Create draft pipeline"
    description = (
        "Creates an in-memory agent-authored OpenHCS pipeline document, using "
        "the referenced PipelineConfig or a new default PipelineConfig."
    )
    service = "pipeline_authoring"
    mutating = True
    side_effects = ("creates_in_memory_pipeline_ref",)
    output_contract = PipelineRef
    input_contract = CreatePipelineRequest
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.pipeline_service,
        method=lambda service, request: service.create_pipeline_from_request(request),
    )


class AddFunctionStepCapability(PipelineDraftCapability):
    name = "openhcs_add_function_step"
    kind = CapabilityKind.TOOL
    title = "Add FunctionStep"
    description = (
        "Adds a FunctionStepSpec resolved through the OpenHCS function registry."
    )
    service = "pipeline_authoring"
    mutating = True
    side_effects = ("mutates_in_memory_pipeline_ref",)
    input_contract = FunctionStepAddRequest
    output_contract = PipelineSpec
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.pipeline_service,
        method=lambda service, request: service.add_function_step_from_request(request),
    )


class ValidatePipelineCapability(PipelineDraftCapability):
    name = "openhcs_validate_pipeline"
    kind = CapabilityKind.TOOL
    title = "Validate draft pipeline"
    description = (
        "Validates function references and constructs the complete OpenHCS "
        "PipelineDocument owned by the draft."
    )
    service = "pipeline_authoring"
    exposition = PipelineDraftCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.VALIDATION,
    )
    input_contract = PipelineValidationRequest
    output_contract = PipelineValidationResult
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.pipeline_service,
        method=lambda service, request: service.validate(request.pipeline_id),
    )


class RenderPipelineSourceCapability(PipelineDraftCapability):
    name = "openhcs_render_pipeline_source"
    kind = CapabilityKind.TOOL
    title = "Render pipeline source"
    description = (
        "Renders an authored PipelineDocument as Python source containing its "
        "PipelineConfig and FunctionStep declarations."
    )
    service = "pipeline_authoring"
    input_contract = PipelineSourceRenderRequest
    output_contract = RenderedSource
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.pipeline_service,
        method=lambda service, request: service.render_source(
            request.pipeline_id,
            clean=request.clean,
        ),
    )


class CreateOrchestratorSessionCapability(HeadlessExecutionCapability):
    name = "openhcs_create_orchestrator_session"
    kind = CapabilityKind.TOOL
    title = "Create orchestrator session"
    description = (
        "Creates an opaque headless execution session from a plate path and "
        "the complete PipelineDocument owned by a pipeline draft. Use the UI "
        "PlateManager code document and "
        "selected-plate workflow instead when an open UI should show the work."
    )
    service = "execution_session"
    mutating = True
    side_effects = ("creates_in_memory_execution_session",)
    input_contract = OrchestratorSessionCreationRequest
    output_contract = OrchestratorSessionRef
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.execution_service,
        method=lambda service, request: service.create_session_from_request(request),
    )


class CreateOrchestratorSessionFromPipelineSourceCapability(
    HeadlessExecutionCapability
):
    name = "openhcs_create_orchestrator_session_from_pipeline_source"
    kind = CapabilityKind.TOOL
    title = "Create source-backed orchestrator session"
    description = (
        "Creates an opaque headless execution session from an exact pycodified "
        "PipelineDocument containing pipeline_config and pipeline_steps, such "
        "as Pipeline Editor code-mode content. A PlateManager document is a "
        "multi-plate aggregate, not pipeline source; use the UI selected-plate "
        "workflow when an open UI should show rows, snapshots, and output auto-add."
    )
    service = "execution_session"
    mutating = True
    side_effects = ("creates_in_memory_execution_session",)
    progress_heartbeat_seconds = 10.0
    progress_worker_thread_safe = False
    input_contract = PipelineSourceOrchestratorSessionRequest
    output_contract = OrchestratorSessionRef
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.execution_service,
        method=lambda service, request: (
            service.create_session_from_pipeline_source_request(request)
        ),
    )


class GetOrchestratorSessionCapability(HeadlessExecutionCapability):
    name = "openhcs_get_orchestrator_session"
    kind = CapabilityKind.TOOL
    title = "Get orchestrator session"
    description = "Returns the stored plate, pipeline, config, and ZMQ connection identity for a session."
    service = "execution_session"
    input_contract = OrchestratorSessionRequest
    output_contract = OrchestratorSession
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.execution_service,
        method=lambda service, request: service.get_session_from_request(request),
    )


class InspectPipelineSourceArtifactPlanCapability(PipelineDraftCapability):
    name = "openhcs_inspect_pipeline_source_artifact_plan"
    cli_command = "artifact-plan"
    kind = CapabilityKind.TOOL
    title = "Inspect source artifact plan"
    description = (
        "Compiles a complete pycodified PipelineDocument with an explicit progress queue "
        "and returns bounded axis, step, group-key, virtual source-workspace, "
        "path, main-flow checkpoint, viewer-streaming, and artifact-output plans."
    )
    service = "execution_session"
    exposition = PipelineDraftCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.VALIDATION,
    )
    input_contract = PipelineSourceArtifactPlanInspectionRequest
    output_contract = ArtifactPlanInspection
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.execution_service,
        method=lambda service, request: (
            service.inspect_pipeline_source_artifact_plan_request(request)
        ),
    )


class SubmitCompileCapability(HeadlessExecutionCapability):
    name = "openhcs_submit_compile"
    kind = CapabilityKind.TOOL
    title = "Submit compile job"
    description = (
        "Submits a compile-only ZMQ execution job for an execution session. "
        "Use wait=False for normal agent workflows, then poll status by job_id; "
        "submit is bounded by submit_timeout_ms and wait=True is bounded by "
        "wait_timeout_ms."
    )
    service = "execution_session"
    mutating = True
    side_effects = ("submits_zmq_compile_job",)
    input_contract = CompileSubmissionRequest
    output_contract = ExecutionJobRef
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.execution_service,
        method=lambda service, request: service.submit_compile(
            request.session_id,
            wait=request.wait,
            submit_timeout_ms=request.submit_timeout_ms,
            wait_timeout_ms=request.wait_timeout_ms,
        ),
    )


class SubmitPipelineExecutionCapability(HeadlessExecutionCapability):
    name = "openhcs_submit_pipeline_execution"
    kind = CapabilityKind.TOOL
    title = "Submit pipeline execution"
    description = (
        "Submits a headless ZMQ pipeline execution job for an execution session. "
        "Use wait=False for normal agent workflows, then poll status by job_id; "
        "submit is bounded by submit_timeout_ms and wait=True is bounded by "
        "wait_timeout_ms. This path does not update the running UI PlateManager; "
        "use openhcs_ui_selected_plate_workflow for user-visible UI runs."
    )
    service = "execution_session"
    mutating = True
    side_effects = ("submits_zmq_execution_job",)
    input_contract = PipelineExecutionSubmissionRequest
    output_contract = ExecutionJobRef
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.execution_service,
        method=lambda service, request: service.submit_execution(
            request.session_id,
            compile_artifact_id=request.compile_artifact_id,
            wait=request.wait,
            submit_timeout_ms=request.submit_timeout_ms,
            wait_timeout_ms=request.wait_timeout_ms,
        ),
    )


class GetExecutionStatusCapability(SubmittedJobCapability):
    name = "openhcs_get_execution_status"
    kind = CapabilityKind.TOOL
    title = "Get execution status"
    description = "Polls the ZMQ server for one submitted compile or execution job."
    service = "execution_session"
    input_contract = ExecutionStatusRequest
    output_contract = ExecutionJobStatus
    request_invocation = AgentDataclassRequestServiceInvocation(
        service=lambda context: context.execution_service,
        method=lambda service, request: service.get_job_status(
            request.job_id,
            timeout_ms=request.timeout_ms,
        ),
    )


class ScanRuntimeServersCapability(RuntimeServerCliConnectionCapability):
    name = "openhcs_scan_runtime_servers"
    cli_command = "runtime-scan"
    kind = CapabilityKind.TOOL
    title = "Scan runtime servers"
    description = "Scans candidate ports for running OpenHCS ZMQ execution servers."
    service = "runtime_server"
    input_contract = RuntimeServerScanRequest
    output_contract = RuntimeServerScanResult
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.runtime_server_service,
        method=lambda service, request: service.scan_from_request(request),
    )


class GetRuntimeServerInfoCapability(RuntimeServerCliConnectionCapability):
    name = "openhcs_get_runtime_server_info"
    cli_command = "runtime-info"
    kind = CapabilityKind.TOOL
    title = "Get runtime server info"
    description = "Returns a read-only server snapshot from a running OpenHCS ZMQ execution server."
    service = "runtime_server"
    input_contract = RuntimeServerInfoRequest
    output_contract = RuntimeServerInfo
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.runtime_server_service,
        method=lambda service, request: service.server_info_from_request(request),
    )


class GetRuntimeServerExecutionStatusCapability(RuntimeServerCliConnectionCapability):
    name = "openhcs_get_runtime_server_execution_status"
    cli_command = "runtime-status"
    kind = CapabilityKind.TOOL
    title = "Get runtime execution status"
    description = "Returns a bounded execution-status projection from a running OpenHCS runtime server."
    service = "runtime_server"
    input_contract = RuntimeServerExecutionStatusRequest
    output_contract = RuntimeExecutionStatus
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.runtime_server_service,
        method=lambda service, request: service.execution_status_from_request(request),
    )


class InspectDebugRuntimeValuesCapability(RuntimeServerCliConnectionCapability):
    name = "openhcs_inspect_debug_runtime_values"
    cli_command = "runtime-debug-values"
    kind = CapabilityKind.TOOL
    title = "Inspect paused runtime values"
    description = (
        "Returns the renderer-independent artifact keys, storage locations, and "
        "value types visible in one paused OpenHCS debug worker."
    )
    service = "runtime_server"
    runtime_requirements = (
        "running_openhcs_execution_server",
        "paused_debug_session",
    )
    data_exposure = (
        "runtime_artifact_keys",
        "runtime_artifact_storage_locations",
        "runtime_value_types",
    )
    input_contract = RuntimeDebugInspectionRequest
    output_contract = RuntimeDebugInspectionResult
    request_invocation = AgentFromFieldsServiceInvocation(
        service=lambda context: context.runtime_server_service,
        method=lambda service, request: service.runtime_debug_inspection_from_request(
            request
        ),
    )


class ViewerSnapshotWindowCapability(ViewerWindowCliConnectionCapability):
    name = "openhcs_viewer_snapshot_window"
    cli_command = "snapshot-viewer"
    kind = CapabilityKind.TOOL
    title = "Snapshot viewer window"
    description = "Captures a running OpenHCS viewer window, such as Napari, through its ZMQ control socket."
    service = "viewer_window"
    mutating = True
    side_effects = ("writes_agent_output_file",)
    runtime_requirements = ("running_openhcs_viewer_server",)
    data_exposure = ("viewer_screenshot", "local_output_path")
    security_requirements = ("agent_path_policy",)
    input_contract = ViewerWindowSnapshotRequest
    output_contract = ViewerWindowSnapshotResult
    request_invocation = AgentViewerWindowRequestServiceInvocation(
        service=lambda context: context.viewer_window_service,
        method=lambda service, request: service.snapshot_window(request),
    )


class GetViewerWindowStateCapability(ViewerWindowCliConnectionCapability):
    name = "openhcs_get_viewer_window_state"
    cli_command = "viewer-state"
    kind = CapabilityKind.TOOL
    title = "Get viewer window state"
    description = (
        "Returns bounded structured layer, component, axis, payload-summary, and "
        "shape-bound state from a running OpenHCS viewer through its ZMQ "
        "control socket."
    )
    service = "viewer_window"
    runtime_requirements = ("running_openhcs_viewer_server",)
    data_exposure = (
        "viewer_layer_state",
        "viewer_axis_state",
        "viewer_payload_summaries",
        "viewer_shape_bounds",
    )
    input_contract = ViewerWindowStateRequest
    output_contract = ViewerWindowStateResult
    request_invocation = AgentViewerWindowRequestServiceInvocation(
        service=lambda context: context.viewer_window_service,
        method=lambda service, request: service.window_state(request),
    )


class GetViewerWindowPayloadsCapability(ViewerWindowCliConnectionCapability):
    name = "openhcs_get_viewer_window_payloads"
    cli_command = "viewer-payloads"
    kind = CapabilityKind.TOOL
    title = "Get viewer window payloads"
    description = (
        "Returns bounded per-layer, per-axis image and shape payload records, "
        "including exact optional arrays and shapes, from a running viewer "
        "control endpoint. Array values are omitted by default: explicitly set "
        "include_array_values=true with a sufficient max_array_elements, or use "
        "the image-sampling capability for bounded tiles."
    )
    service = "viewer_window"
    runtime_requirements = ("running_openhcs_viewer_server",)
    data_exposure = (
        "viewer_payload_records",
        "viewer_axis_coordinates",
        "viewer_shape_payloads",
        "viewer_array_values",
    )
    input_contract = ViewerWindowPayloadRequest
    output_contract = ViewerWindowPayloadResult
    request_invocation = AgentViewerWindowRequestServiceInvocation(
        service=lambda context: context.viewer_window_service,
        method=lambda service, request: service.window_payloads(request),
    )


class SampleViewerWindowImageCapability(ViewerWindowCliConnectionCapability):
    name = "openhcs_sample_viewer_window_image"
    cli_command = "sample-viewer-image"
    kind = CapabilityKind.TOOL
    title = "Sample viewer image payload"
    description = (
        "Returns native-resolution bounded image records and bounded pixel samples "
        "for routed image payloads from a running viewer control endpoint. Pixel "
        "values are omitted by default: set include_array_values=true and keep "
        "height*width within max_array_elements; tile a field when exact pixels "
        "are needed beyond that bound."
    )
    service = "viewer_window"
    runtime_requirements = ("running_openhcs_viewer_server",)
    data_exposure = (
        "viewer_payload_records",
        "viewer_axis_coordinates",
        "viewer_array_values",
    )
    input_contract = ViewerWindowImageSampleRequest
    output_contract = ViewerWindowImageSampleResult
    request_invocation = AgentViewerWindowRequestServiceInvocation(
        service=lambda context: context.viewer_window_service,
        method=lambda service, request: service.sample_image(request),
    )


class SummarizeViewerWindowRoisCapability(ViewerWindowCliConnectionCapability):
    name = "openhcs_summarize_viewer_window_rois"
    cli_command = "viewer-rois"
    kind = CapabilityKind.TOOL
    title = "Summarize viewer ROI payload"
    description = (
        "Returns compact ROI counts, bounds, area statistics, and examples "
        "for shape payloads from a running viewer control endpoint, optionally "
        "filtered to one route."
    )
    service = "viewer_window"
    runtime_requirements = ("running_openhcs_viewer_server",)
    data_exposure = (
        "viewer_shape_payloads",
        "viewer_shape_bounds",
        "viewer_roi_statistics",
    )
    input_contract = ViewerWindowRoiSummaryRequest
    output_contract = ViewerWindowRoiSummaryResult
    request_invocation = AgentViewerWindowRequestServiceInvocation(
        service=lambda context: context.viewer_window_service,
        method=lambda service, request: service.summarize_rois(request),
    )


class NavigateViewerWindowCapability(ViewerWindowCliConnectionCapability):
    name = "openhcs_navigate_viewer_window"
    cli_command = "navigate-viewer"
    kind = CapabilityKind.TOOL
    title = "Navigate viewer window"
    description = (
        "Sets a viewer layer visible or selected, moves zero-based route-local "
        "axis indices, and can select one zero-based data_index on a native "
        "feature-bearing result layer. The result reports feature_row_count and "
        "selected_data_indices so agents can verify the visible overlay and "
        "Napari feature-table selection are linked."
    )
    service = "viewer_window"
    mutating = True
    side_effects = ("mutates_viewer_window_state",)
    runtime_requirements = ("running_openhcs_viewer_server",)
    data_exposure = (
        "viewer_layer_state",
        "viewer_axis_state",
        "viewer_feature_row_count",
        "viewer_selected_data_indices",
    )
    input_contract = ViewerWindowNavigationRequest
    output_contract = ViewerWindowNavigationResult
    request_invocation = AgentViewerWindowRequestServiceInvocation(
        service=lambda context: context.viewer_window_service,
        method=lambda service, request: service.navigate_window(request),
        timeout_profile=CapabilityViewerControlTimeoutProfile.COMMAND,
    )


class IsolateViewerWindowLayersCapability(ViewerWindowCliConnectionCapability):
    name = "openhcs_isolate_viewer_window_layers"
    cli_command = "isolate-viewer"
    kind = CapabilityKind.TOOL
    title = "Isolate viewer layers"
    description = (
        "Shows only selected viewer layers, hides all non-selected viewer "
        "layers, selects one layer, and applies route-local axis indices."
    )
    service = "viewer_window"
    mutating = True
    side_effects = ("mutates_viewer_window_state",)
    runtime_requirements = ("running_openhcs_viewer_server",)
    data_exposure = ("viewer_layer_state", "viewer_axis_state")
    input_contract = ViewerWindowLayerIsolationRequest
    output_contract = ViewerWindowLayerIsolationResult
    request_invocation = AgentViewerWindowRequestServiceInvocation(
        service=lambda context: context.viewer_window_service,
        method=lambda service, request: service.isolate_layers(request),
        timeout_profile=CapabilityViewerControlTimeoutProfile.COMMAND,
    )


class ProbeViewerWindowCapability(ViewerWindowCliConnectionCapability):
    name = "openhcs_probe_viewer_window"
    cli_command = "probe-viewer"
    kind = CapabilityKind.TOOL
    title = "Probe viewer window"
    description = "Quickly reports whether a running OpenHCS viewer control endpoint is reachable."
    service = "viewer_window"
    exposition = ViewerWindowCliConnectionCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.DIAGNOSTIC,
        role=CapabilityRole.DIAGNOSTIC,
    )
    runtime_requirements = ("running_openhcs_viewer_server",)
    data_exposure = ("viewer_identity", "viewer_layer_counts")
    input_contract = ViewerWindowStateRequest
    output_contract = ViewerWindowProbeResult


class ValidateViewerWindowStateCapability(ViewerWindowCliConnectionCapability):
    name = "openhcs_validate_viewer_window_state"
    cli_command = "validate-viewer"
    kind = CapabilityKind.TOOL
    title = "Validate viewer window state"
    description = (
        "Validates mounted layers, expected axis labels, payload nonzero "
        "metadata, routed coordinate coverage, duplicate/missing payload "
        "coordinates, and payload spatial compatibility for a running "
        "OpenHCS viewer."
    )
    service = "viewer_window"
    runtime_requirements = ("running_openhcs_viewer_server",)
    data_exposure = (
        "viewer_layer_state",
        "viewer_axis_state",
        "viewer_payload_summaries",
        "viewer_coordinate_coverage",
        "viewer_payload_spatial_compatibility",
    )
    input_contract = ViewerWindowValidationRequest
    output_contract = ViewerWindowValidationSummaryResult
    request_invocation = AgentViewerWindowRequestServiceInvocation(
        service=lambda context: context.viewer_window_service,
        method=lambda service, request: service.validation_summary(request),
    )


class UiListBridgesCapability(UiBridgeCapability):
    name = "openhcs_ui_list_bridges"
    kind = CapabilityKind.TOOL
    title = "List UI bridges"
    description = (
        "Lists local OpenHCS PyQt UI bridge descriptor summaries visible to this user."
    )
    service = "ui_bridge"
    exposition = UiBridgeCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.DIAGNOSTIC,
        role=CapabilityRole.DIAGNOSTIC,
    )
    data_exposure = ("local_ui_bridge_descriptor_paths",)
    output_contract = UiBridgeCatalog
    no_argument_invocation = AgentNoArgumentServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service: service.list_bridges(),
    )


class UiBridgeStatusCapability(UiBridgeCliConnectionCapability):
    name = "openhcs_ui_bridge_status"
    cli_command = "ui-status"
    kind = CapabilityKind.TOOL
    title = "Get UI bridge status"
    description = "Reports whether a local running OpenHCS PyQt UI bridge is reachable."
    service = "ui_bridge"
    exposition = UiBridgeCliConnectionCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.DIAGNOSTIC,
        role=CapabilityRole.DIAGNOSTIC,
    )
    runtime_requirements = ("running_openhcs_ui_bridge",)
    output_contract = UiBridgeStatus
    connection_invocation = AgentConnectionServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, connection: service.status(connection),
    )


class UiListCodeDocumentsCapability(UiCodeDocumentCapability):
    name = "openhcs_ui_list_code_documents"
    cli_command = "code-documents"
    kind = CapabilityKind.TOOL
    title = "List UI code documents"
    description = (
        "Lists UI code documents with flat document_id values for follow-up calls."
    )
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    output_contract = UiCodeDocumentCatalog


class UiListStateSurfacesCapability(UiSelectedPlateCapability):
    name = "openhcs_ui_list_state_surfaces"
    cli_command = "state-surfaces"
    kind = CapabilityKind.TOOL
    title = "List UI state surfaces"
    description = (
        "Lists pollable domain state surfaces, including workflow status and live "
        "measurement results, with flat surface_id values for follow-up reads."
    )
    service = "ui_bridge"
    exposition = UiSelectedPlateCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.STATUS,
        role=CapabilityRole.PRIMARY,
    )
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    output_contract = UiStateSurfaceCatalog


class UiGetStateSurfaceCapability(UiSelectedPlateCapability):
    name = "openhcs_ui_get_state_surface"
    cli_command = "state-surface"
    kind = CapabilityKind.TOOL
    title = "Get UI state surface"
    description = (
        "Reads or polls one typed UI domain state surface such as plate-manager "
        "status rows or bounded live measurement tables."
    )
    service = "ui_bridge"
    exposition = UiSelectedPlateCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.STATUS,
        role=CapabilityRole.PRIMARY,
    )
    runtime_requirements = ("running_openhcs_ui_bridge",)
    data_exposure = ("local_paths",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiStateSurfaceRequest
    output_contract = UiStateSurfaceDocument
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.get_state_surface(
            request,
            connection,
        ),
    )


class UiListActionsCapability(UiSemanticActionCapability):
    name = "openhcs_ui_list_actions"
    cli_command = "actions"
    kind = CapabilityKind.TOOL
    title = "List UI actions"
    description = "Lists invokable UI actions with flat widget_id/action_id values."
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    output_contract = UiActionCatalog


class UiInvokeActionCapability(UiSemanticActionCapability):
    name = "openhcs_ui_invoke_action"
    cli_command = "invoke-action"
    kind = CapabilityKind.TOOL
    title = "Invoke UI action"
    description = "Dispatches one running-UI action using the selection_revision_token from openhcs_ui_list_actions; workflow progress is polled through related state surfaces."
    service = "ui_bridge"
    mutating = True
    side_effects = ("may_mutate_running_ui_state", "may_start_ui_workflow")
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiActionInvokeRequest
    output_contract = UiActionInvokeResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.invoke_action(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiSelectedPlateWorkflowCapability(UiSelectedPlateCapability):
    name = "openhcs_ui_selected_plate_workflow"
    cli_command = "selected-workflow"
    kind = CapabilityKind.TOOL
    title = "Selected plate workflow"
    description = (
        "Dispatches init, compile, or run for the current PlateManager "
        "selection through the UI bridge, preserving user-visible plate rows, "
        "ObjectState snapshots, selected state, and output-plate auto-add."
    )
    service = "ui_bridge"
    exposition = UiSelectedPlateCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.EXECUTION,
        role=CapabilityRole.PRIMARY,
    )
    mutating = True
    side_effects = ("may_mutate_running_ui_state", "may_start_ui_workflow")
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiSelectedPlateWorkflowRequest
    output_contract = UiSelectedPlateWorkflowResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.selected_plate_workflow(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiListWindowsCapability(UiWindowCapability):
    name = "openhcs_ui_list_windows"
    cli_command = "windows"
    kind = CapabilityKind.TOOL
    title = "List UI windows"
    description = "Lists visible/focusable UI windows with flat window_id values."
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    output_contract = UiWindowCatalog


class UiFocusWindowCapability(UiWindowCapability):
    name = "openhcs_ui_focus_window"
    kind = CapabilityKind.TOOL
    title = "Focus UI window"
    description = "Focuses one running UI window by stable window id or open ObjectState scope id."
    service = "ui_bridge"
    mutating = True
    side_effects = ("changes_running_ui_focus",)
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiWindowFocusRequest
    output_contract = UiWindowFocusResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.focus_window(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiNavigateWindowCapability(UiWindowCapability):
    name = "openhcs_ui_navigate_window"
    kind = CapabilityKind.TOOL
    title = "Navigate UI window"
    description = "Opens or focuses one ObjectState-backed UI window scope and reveals an optional field path or item id."
    service = "ui_bridge"
    mutating = True
    side_effects = ("changes_running_ui_focus", "may_open_running_ui_window")
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiWindowNavigateRequest
    output_contract = UiWindowNavigateResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.navigate_window(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiCloseWindowCapability(UiWindowCapability):
    name = "openhcs_ui_close_window"
    kind = CapabilityKind.TOOL
    title = "Close UI window"
    description = (
        "Requests a normal close for one visible UI bridge window by stable window id."
    )
    service = "ui_bridge"
    mutating = True
    side_effects = ("closes_running_ui_window",)
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiWindowCloseRequest
    output_contract = UiWindowCloseResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.close_window(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiSnapshotWindowCapability(UiWindowCapability):
    name = "openhcs_ui_snapshot_window"
    cli_command = "window-snapshot"
    kind = CapabilityKind.TOOL
    title = "Snapshot UI window"
    description = "Captures one running UI window or visible Qt top-level dialog to a PNG resource path."
    service = "ui_bridge"
    mutating = True
    side_effects = ("writes_agent_output_file",)
    runtime_requirements = ("running_openhcs_ui_bridge",)
    data_exposure = ("ui_screenshot", "local_output_path")
    security_requirements = ("ui_bridge_auth_token", "agent_path_policy")
    input_contract = UiWindowSnapshotRequest
    output_contract = UiWindowSnapshotResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.snapshot_window(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiGetWidgetTreeCapability(UiWidgetFallbackCapability):
    name = "openhcs_ui_get_widget_tree"
    cli_command = "widget-tree"
    kind = CapabilityKind.TOOL
    title = "Get UI widget tree"
    description = (
        "Returns a generic window-manager widget projection for one running "
        "UI window, including visible text, enabled state, clickable "
        "geometry, and action kinds for blind interaction."
    )
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    data_exposure = (
        "ui_widget_tree",
        "ui_clickable_geometry",
        "ui_visible_text",
        "ui_widget_enabled_state",
        "ui_action_kinds",
        "object_state_resolved_value_previews",
    )
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiWidgetTreeRequest
    output_contract = UiWidgetTreeResult


class UiInvokeWidgetActionCapability(UiWidgetFallbackCapability):
    name = "openhcs_ui_invoke_widget_action"
    cli_command = "invoke-widget-action"
    kind = CapabilityKind.TOOL
    title = "Invoke UI widget action"
    description = (
        "Invokes one generic projected Qt widget action by window id, "
        "widget-tree path id, and action kind."
    )
    service = "ui_bridge"
    mutating = True
    side_effects = ("mutates_running_ui",)
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiWidgetActionInvokeRequest
    output_contract = UiWidgetActionInvokeResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.invoke_widget_action(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiListObjectStateScopesCapability(UiObjectStateCapability):
    name = "openhcs_ui_list_object_state_scopes"
    cli_command = "object-state-scopes"
    kind = CapabilityKind.TOOL
    title = "List ObjectState scopes"
    description = (
        "Lists ObjectState scopes visible to the running OpenHCS UI bridge. "
        "Set scope_visibility.include_system_scopes=true to include global "
        "configuration and root system scopes."
    )
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    data_exposure = (
        "object_state_scope_ids",
        "object_type_names",
        "object_state_field_markers",
        "object_state_resolved_value_previews",
    )
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiObjectStateScopeListRequest
    output_contract = UiObjectStateScopeCatalog
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.list_object_state_scopes(
            request,
            connection,
        ),
    )


class UiGetObjectStateFieldsCapability(UiObjectStateCapability):
    name = "openhcs_ui_get_object_state_fields"
    cli_command = "object-state-fields"
    kind = CapabilityKind.TOOL
    title = "Get ObjectState fields"
    description = (
        "Returns compact ObjectState field rows with raw/resolved previews, "
        "dirty/default markers, inheritance flags, and provenance."
    )
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    data_exposure = (
        "object_state_scope_ids",
        "object_state_field_markers",
        "object_state_resolved_value_previews",
        "object_state_field_provenance",
    )
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiObjectStateFieldListQuery
    output_contract = UiObjectStateFieldListResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.get_object_state_fields(
            request,
            connection,
        ),
    )


class UiDescribeObjectStateFieldCapability(UiObjectStateCapability):
    name = "openhcs_ui_describe_object_state_field"
    cli_command = "object-state-field-help"
    cli_aliases = ("object-state-help", "field-help")
    kind = CapabilityKind.TOOL
    title = "Describe ObjectState field"
    description = (
        "Returns Python-introspected docs for one ObjectState field using "
        "its dotted field_path; object_state_scope_id is optional only "
        "when the field path uniquely identifies one live ObjectState field."
    )
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    data_exposure = (
        "object_state_scope_ids",
        "object_state_field_paths",
        "docstrings",
        "parameter_descriptions",
        "object_state_resolved_value_previews",
    )
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiObjectStateFieldHelpQuery
    output_contract = UiObjectStateFieldHelpResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.object_state_field_help_service,
        method=lambda service, request, connection: service.describe_query(
            request,
            connection,
        ),
    )


class UiMutateObjectStateFieldCapability(UiObjectStateCapability):
    name = "openhcs_ui_mutate_object_state_field"
    cli_command = "object-state-set"
    cli_aliases = ("object-state-edit", "object-state-mutate")
    kind = CapabilityKind.TOOL
    title = "Mutate ObjectState field"
    description = (
        "Applies an unsaved ObjectState field update or reset through the "
        "running UI. Save/commit remains explicit through managed-window "
        "save actions so agents can observe dirty/default feedback first."
    )
    service = "ui_bridge"
    mutating = True
    runtime_requirements = ("running_openhcs_ui_bridge",)
    side_effects = ("mutates_object_state", "records_object_state_snapshot")
    data_exposure = (
        "object_state_scope_ids",
        "object_state_field_paths",
        "object_state_field_markers",
        "object_state_raw_value_previews",
        "object_state_resolved_value_previews",
    )
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiObjectStateFieldMutationRequest
    output_contract = UiObjectStateFieldMutationResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.mutate_object_state_field(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiGetCodeDocumentCapability(UiCodeDocumentCapability):
    name = "openhcs_ui_get_code_document"
    cli_command = "code-document"
    cli_aliases = ("get-code-document",)
    kind = CapabilityKind.TOOL
    title = "Get UI code document"
    description = (
        "Reads a bounded UI-owned code document. clean=True returns sparse "
        "clean source; clean=False returns the full resolved pycodified "
        "object including defaults and inherited values."
    )
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    data_exposure = ("local_paths_in_source",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiCodeDocumentRequest
    output_contract = UiCodeDocument
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.get_document(
            request,
            connection,
        ),
    )


class UiValidateCodeDocumentCapability(UiCodeDocumentCapability):
    name = "openhcs_ui_validate_code_document"
    cli_command = "validate-code-document"
    kind = CapabilityKind.TOOL
    title = "Validate UI code document"
    description = "Validates an edited UI code document through the bridge source policy without mutating UI state."
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    data_exposure = ("local_paths_in_source",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiCodeDocumentValidationRequest
    output_contract = UiCodeDocumentValidationResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.validate_document(
            request,
            connection,
        ),
    )


class UiApplyCodeDocumentCapability(UiCodeDocumentCapability):
    name = "openhcs_ui_apply_code_document"
    cli_command = "apply-code-document"
    kind = CapabilityKind.TOOL
    title = "Apply UI code document"
    description = (
        "Applies an edited UI code document through the running PyQt workflow "
        "with revision protection, returning the resulting ObjectState snapshot, "
        "undo snapshot, and revision tokens."
    )
    service = "ui_bridge"
    mutating = True
    side_effects = ("mutates_running_ui_state",)
    runtime_requirements = ("running_openhcs_ui_bridge",)
    data_exposure = (
        "local_paths_in_source",
        "ui_revision_tokens",
        "object_state_snapshot_refs",
        "object_state_undo_targets",
    )
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiCodeDocumentApplyRequest
    output_contract = UiCodeDocumentApplyResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.apply_document(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiListSnapshotsCapability(UiSnapshotCapability):
    name = "openhcs_ui_list_snapshots"
    kind = CapabilityKind.TOOL
    title = "List UI snapshots"
    description = "Lists ObjectState snapshots visible to the running UI bridge."
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiSnapshotListRequest
    output_contract = UiSnapshotCatalog
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.list_snapshots(
            request,
            connection,
        ),
    )


class UiRestoreSnapshotCapability(UiSnapshotCapability):
    name = "openhcs_ui_restore_snapshot"
    kind = CapabilityKind.TOOL
    title = "Restore UI snapshot"
    description = (
        "Restores the running UI to a selected ObjectState snapshot through the bridge."
    )
    service = "ui_bridge"
    mutating = True
    side_effects = ("mutates_running_ui_state", "time_travels_ui_state")
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiSnapshotRestoreRequest
    output_contract = UiSnapshotRestoreResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.restore_snapshot(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiTimeTravelHeadCapability(UiSnapshotCapability):
    name = "openhcs_ui_time_travel_head"
    kind = CapabilityKind.TOOL
    title = "Return UI to current head"
    description = "Returns the running UI from ObjectState time travel to the current branch head."
    service = "ui_bridge"
    mutating = True
    side_effects = ("mutates_running_ui_state", "time_travels_ui_state")
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiTimeTravelHeadRequest
    output_contract = UiSnapshotRestoreResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.time_travel_head(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiListBranchesCapability(UiSnapshotCapability):
    name = "openhcs_ui_list_branches"
    kind = CapabilityKind.TOOL
    title = "List UI snapshot branches"
    description = "Lists ObjectState branches visible to the running UI bridge."
    service = "ui_bridge"
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    output_contract = UiBranchCatalog
    connection_invocation = AgentConnectionServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, connection: service.list_branches(connection),
    )


class UiSwitchBranchCapability(UiSnapshotCapability):
    name = "openhcs_ui_switch_branch"
    kind = CapabilityKind.TOOL
    title = "Switch UI snapshot branch"
    description = (
        "Switches the running UI to another ObjectState branch through the bridge."
    )
    service = "ui_bridge"
    mutating = True
    side_effects = ("mutates_running_ui_state", "time_travels_ui_state")
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiBranchSwitchRequest
    output_contract = UiSnapshotRestoreResult
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.switch_branch(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


class UiGetOperationStatusCapability(UiBridgeCliConnectionCapability):
    name = "openhcs_ui_get_operation_status"
    kind = CapabilityKind.TOOL
    title = "Get UI bridge operation status"
    description = "Returns status for an active or recent running-UI bridge operation."
    service = "ui_bridge"
    exposition = UiBridgeCliConnectionCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.STATUS,
        role=CapabilityRole.DIAGNOSTIC,
    )
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = OPERATION_ID_INPUT
    output_contract = UiBridgeOperationRef
    connection_scalar_invocation = AgentConnectionScalarServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, value, connection: service.get_operation_status(
            value,
            connection,
        ),
    )


class UiWaitForOperationCapability(UiBridgeCliConnectionCapability):
    name = "openhcs_ui_wait_for_operation"
    kind = CapabilityKind.TOOL
    title = "Wait for UI bridge operation"
    description = (
        "Waits once for an accepted running-UI bridge operation to reach its exact "
        "terminal status, preserving outcome, timestamps, errors, and warnings."
    )
    service = "ui_bridge"
    exposition = UiBridgeCliConnectionCapability.exposition.refine(
        workflow_stage=CapabilityWorkflowStage.STATUS,
        role=CapabilityRole.PRIMARY,
    )
    runtime_requirements = ("running_openhcs_ui_bridge",)
    security_requirements = ("ui_bridge_auth_token",)
    input_contract = UiBridgeOperationWaitRequest
    output_contract = UiBridgeOperationRef
    connection_request_invocation = AgentConnectionRequestServiceInvocation(
        service=lambda context: context.ui_bridge_service,
        method=lambda service, request, connection: service.wait_for_operation(
            request,
            connection,
        ),
        timeout_profile=CapabilityUiBridgeTimeoutProfile.COMMAND,
    )


def agent_capability_declarations() -> tuple[type[AgentCapabilityDeclaration], ...]:
    return tuple(AgentCapabilityDeclaration.__registry__.values())


def _declared_capabilities() -> tuple[AgentCapabilitySpec, ...]:
    return tuple(
        declaration.to_spec() for declaration in agent_capability_declarations()
    )


def _capability_groups(
    capabilities: tuple[AgentCapabilitySpec, ...],
) -> tuple[AgentCapabilityGroup, ...]:
    groups: list[AgentCapabilityGroup] = []
    for workflow_group in CapabilityWorkflowGroup:
        grouped_capabilities = tuple(
            capability
            for capability in capabilities
            if capability.workflow_group is workflow_group
        )
        if not grouped_capabilities:
            continue
        groups.append(
            AgentCapabilityGroup(
                workflow_group=workflow_group,
                capability_names=tuple(
                    capability.name for capability in grouped_capabilities
                ),
                tool_count=sum(
                    1
                    for capability in grouped_capabilities
                    if capability.kind is CapabilityKind.TOOL
                ),
                resource_count=sum(
                    1
                    for capability in grouped_capabilities
                    if capability.kind is CapabilityKind.RESOURCE
                ),
            )
        )
    return tuple(groups)


CAPABILITIES: tuple[AgentCapabilitySpec, ...] = _declared_capabilities()


def _capability_attribute_name(name: str) -> str:
    """Return a Python attribute generated from one final capability ABI name."""
    if name.startswith("openhcs://"):
        name = name.removeprefix("openhcs://")
    elif name.startswith("openhcs_"):
        name = name.removeprefix("openhcs_")
    return name.replace("/", "_").replace("-", "_").replace(":", "_")


agent_capabilities = AgentCapabilityNamespace(CAPABILITIES)


def get_capability_registry(
    capability_transport: CapabilityTransport | None = None,
    capability_surface_profile: LocalCapabilitySurfaceProfile | None = None,
) -> AgentCapabilityRegistry:
    """Return the canonical registry projected through transport and visibility."""
    validate_capability_registry(CAPABILITIES)
    selection = AgentCapabilitySurfaceSelection(
        transport=capability_transport,
        local_profile=(
            FullLocalCapabilitySurfaceProfile()
            if capability_surface_profile is None
            else capability_surface_profile
        ),
    )
    capabilities = tuple(
        capability for capability in CAPABILITIES if selection.includes(capability)
    )
    return AgentCapabilityRegistry(
        schema_version=SCHEMA_VERSION,
        capabilities=capabilities,
        groups=_capability_groups(capabilities),
        surface_profile=selection.local_profile.name,
    )


def get_agent_capability(name: str) -> AgentCapabilitySpec:
    """Return the declared capability for a final MCP/resource ABI name."""
    return get_agent_capability_declaration(name).to_spec()


def get_agent_capability_declaration(
    name: str,
) -> type[AgentCapabilityDeclaration]:
    """Return the declaration that owns one final MCP/resource ABI name."""
    try:
        return AgentCapabilityDeclaration.__registry__[name]
    except KeyError as exc:
        raise KeyError(f"Unknown OpenHCS agent capability: {name}") from exc


def validate_capability_registry(
    capabilities: tuple[AgentCapabilitySpec, ...] = CAPABILITIES,
) -> None:
    """Assert static capability metadata is complete enough for policy checks."""
    seen: set[str] = set()
    for capability in capabilities:
        if capability.name in seen:
            raise ValueError(f"Duplicate OpenHCS agent capability: {capability.name}")
        seen.add(capability.name)
        if not capability.transport_availability:
            raise ValueError(
                f"Capability {capability.name!r} must declare transport availability."
            )
        if len(capability.transport_availability) != len(
            set(capability.transport_availability)
        ):
            raise ValueError(
                f"Capability {capability.name!r} declares duplicate transports."
            )
        if capability.kind is CapabilityKind.TOOL and capability.mutating:
            if not capability.side_effects:
                raise ValueError(
                    f"Mutating tool {capability.name!r} must declare side_effects."
                )
        if capability.side_effects and not capability.mutating:
            raise ValueError(
                f"Capability {capability.name!r} declares side_effects but is not "
                "marked mutating."
            )
        if capability.kind is CapabilityKind.TOOL and capability.exposition is None:
            raise ValueError(
                f"Tool {capability.name!r} must declare "
                f"{AgentCapabilityExposition.__name__}."
            )
