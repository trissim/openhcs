"""DTO exports for the headless OpenHCS agent API.

This package intentionally keeps the re-export surface lazy.  Importing one
concrete DTO submodule, such as ``openhcs.agent.dto.common``, must not import
the full agent DTO graph.
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
import sys
from types import ModuleType

__all__ = (
    "SCHEMA_VERSION",
    "AgentError",
    "AgentResourceRef",
    "AgentWarning",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "RenderedSource",
    "McpServerHealthResult",
    "WindowSnapshotCaptureScope",
    "WindowSnapshotCaptureSpec",
    "AuthoringContext",
    "AuthoringContextRequest",
    "ArchitectureTopic",
    "ArchitectureTopicPage",
    "ArchitectureTopicSummary",
    "InternalApiSymbol",
    "ConfigFieldSchema",
    "ConfigPatch",
    "ConfigRegisteredType",
    "ConfigRef",
    "ConfigRegistrySchema",
    "ConfigSchema",
    "ConfigSchemaRequest",
    "ConfigSourceRenderRequest",
    "ConfigTypeSchema",
    "ConfigValidationResult",
    "CellProfilerArtifactBindingSummary",
    "CellProfilerModuleDeclarationSummary",
    "CustomFunctionRegistrationRequest",
    "CustomFunctionRegistrationResult",
    "FunctionDetailRequest",
    "FunctionArtifactSpec",
    "FunctionCatalogEntry",
    "FunctionCatalogPage",
    "FunctionDetail",
    "FunctionSearchRequest",
    "FunctionParameterSource",
    "FunctionParameterSpec",
    "FunctionRuntimeContractSummary",
    "KnowledgeBaseCatalog",
    "KnowledgeBaseDocument",
    "KnowledgeBaseDocumentRequest",
    "KnowledgeBaseDocumentSourceProjection",
    "KnowledgeBaseDocumentSummary",
    "KnowledgeBaseSearchHit",
    "KnowledgeBaseSearchRequest",
    "KnowledgeBaseSearchResult",
    "KnowledgeBaseSectionSummary",
    "ArtifactInputPlanSummary",
    "ArtifactMaterializationPathSummary",
    "ArtifactMaterializationPlanSummary",
    "ArtifactPlanInspection",
    "ArtifactPlanSummary",
    "ArtifactStoragePlanSummary",
    "CompileSubmissionRequest",
    "CompiledStepPlanSummary",
    "ExecutionConnectionSpec",
    "ExecutionJobRef",
    "ExecutionJobStatus",
    "ExecutionStatusRequest",
    "MainFlowMaterializationPlanSummary",
    "OrchestratorSession",
    "OrchestratorSessionCreationRequest",
    "OrchestratorSessionRequest",
    "OrchestratorSessionRef",
    "PipelineSourceArtifactPlanInspectionRequest",
    "PipelineSourceOrchestratorSessionRequest",
    "PipelineExecutionSubmissionRequest",
    "RuntimeDebugInspectionRequest",
    "RuntimeDebugInspectionResult",
    "RuntimeExecutionStatus",
    "RuntimeServerExecutionStatusRequest",
    "RuntimeServerInfo",
    "RuntimeServerInfoRequest",
    "RuntimeServerScanRequest",
    "RuntimeServerScanResult",
    "SourceWorkspaceFileRecord",
    "SourceWorkspaceSummary",
    "ViewerStreamingPlanSummary",
    "CreatePipelineRequest",
    "FunctionSpecRef",
    "FunctionStepAddRequest",
    "FunctionStepSpec",
    "PipelineRef",
    "PipelineSourceRenderRequest",
    "PipelineSpec",
    "PipelineValidationRequest",
    "PipelineValidationResult",
    "PlateFileQueryRecordSummary",
    "PlateFileQueryRequest",
    "PlateFileQueryResult",
    "PlateFileStreamRequest",
    "PlateFileStreamResult",
    "PlateInspectionBounds",
    "PlateInspectionComponentSummary",
    "PlateInspectionComponentValue",
    "PlateInspectionConfidence",
    "PlateInspectionHandlerCandidate",
    "PlateInspectionImageFileSummary",
    "PlateInspectionImageRecordSummary",
    "PlateInspectionIngestionRoute",
    "PlateInspectionIssueCode",
    "PlateInspectionParseFailure",
    "PlateInspectionParseSummary",
    "PlateInspectionResultFileRecordSummary",
    "PlateInspectionResultFileSummary",
    "PlateInspectionStatus",
    "PlateInspectionSourceBindingRole",
    "PlateInspectionValueSource",
    "PlateInspectionWorkflowAdvice",
    "PlateInspectionWorkflowScope",
    "PlateInspectionWorkspacePreparation",
    "PlateImageSampleRequest",
    "PlateImageSampleResult",
    "PlatePathInspectionRequest",
    "PlatePathInspectionResult",
    "PlateWorkspacePreparationOperation",
    "SelectedPlateFileQueryTarget",
    "SelectedPlateFileQueryRequest",
    "SelectedPlateFileQueryResult",
    "SelectedPlateFileStreamRequest",
    "SelectedPlateFileStreamResult",
    "SelectedPlateImageInspectionRequest",
    "SelectedPlateImageInspectionResult",
    "SelectedPlateImageSampleRequest",
    "SelectedPlateImageSampleResult",
    "SyntheticPlateGenerationRequest",
    "SyntheticPlateGenerationResult",
    "ViewerWindowDescriptor",
    "ViewerWindowLayerPayloads",
    "ViewerWindowLayerState",
    "ViewerWindowLayerValidationSummary",
    "ViewerWindowLayerIsolationRequest",
    "ViewerWindowLayerIsolationResult",
    "ViewerWindowLayerVisibilityRecord",
    "ViewerWindowImageSampleRequest",
    "ViewerWindowImageSampleResult",
    "ViewerWindowNavigationRequest",
    "ViewerWindowNavigationResult",
    "ViewerWindowPayloadRecord",
    "ViewerWindowPayloadRequest",
    "ViewerWindowPayloadResult",
    "ViewerWindowProbeResult",
    "ViewerWindowRoiSummaryRequest",
    "ViewerWindowRoiSummaryResult",
    "ViewerWindowSnapshotRequest",
    "ViewerWindowSnapshotResult",
    "ViewerWindowStateRequest",
    "ViewerWindowStateResult",
    "ViewerWindowValidationPolicy",
    "ViewerWindowValidationRequest",
    "ViewerWindowValidationSummaryResult",
    "UiActionCatalog",
    "UiActionIdentity",
    "UiActionInvocationStatus",
    "UiActionInvokeRequest",
    "UiActionInvokeResult",
    "UiActionSummary",
    "UiBranchCatalog",
    "UiBranchRef",
    "UiBranchSwitchRequest",
    "UiBridgeConfirmationRequirement",
    "UiBridgeConfirmationRequirementCarrier",
    "UiBridgeConnectionSpec",
    "UiBridgeCatalog",
    "UiBridgeDescriptorEnvelope",
    "UiBridgeDescriptorFile",
    "UiBridgeDescriptorSummary",
    "UiBridgeDescriptorWirePayload",
    "UiBridgeOperationIdentity",
    "UiBridgeOperationRef",
    "UiBridgeOperationStatusRequest",
    "UiBridgeOperationWaitRequest",
    "UiBridgeRequestEnvelope",
    "UiBridgeResponseEnvelope",
    "UiBridgeStatus",
    "UiCodeDocument",
    "UiCodeDocumentApplyRequest",
    "UiCodeDocumentApplyResult",
    "UiCodeDocumentCatalog",
    "UiCodeDocumentBaseRevision",
    "UiCodeDocumentCurrentRevision",
    "UiCodeDocumentId",
    "UiCodeDocumentOptionalBaseRevision",
    "UiCodeDocumentRequest",
    "UiCodeDocumentSelectionMode",
    "UiCodeDocumentSummary",
    "UiCodeDocumentValidationRequest",
    "UiCodeDocumentValidationResult",
    "UiMutationReceipt",
    "UiMutationRequestToken",
    "UiMutationRequestTokenCarrier",
    "UiLiveOverviewItem",
    "UiLiveOverviewMetric",
    "UiLiveOverviewSection",
    "UiLiveOverviewSeverity",
    "UiLiveOverviewState",
    "UiObjectStateFieldListQuery",
    "UiObjectStateFieldListOptions",
    "UiObjectStateFieldListResult",
    "UiObjectStateFieldPathIndex",
    "UiObjectStateFieldProjection",
    "UiObjectStateFieldProvenance",
    "UiObjectStateFieldScopeProjection",
    "UiObjectStateFieldHelpQuery",
    "UiObjectStateFieldHelpRequest",
    "UiObjectStateFieldHelpResult",
    "UiObjectStateFieldMutationRequest",
    "UiObjectStateFieldMutationResult",
    "UiObjectStateFieldSummary",
    "UiObjectStateValuePreview",
    "UiObjectStateScopeCatalog",
    "UiObjectStateScopeIdentity",
    "UiObjectStateScopeListRequest",
    "UiObjectStateScopeSummary",
    "UiObjectStateScopeVisibility",
    "UiPlateManagerRowState",
    "UiPlateManagerState",
    "UiSelectedPlateWorkflowKind",
    "UiSelectedPlateWorkflowRequest",
    "UiSelectedPlateWorkflowResult",
    "UiStateSurfaceDocument",
    "UiStateSurfaceEnvelope",
    "UiStateSurfaceCatalog",
    "UiStateSurfaceId",
    "UiStateSurfaceIdentity",
    "UiStateSurfaceRequest",
    "UiStateSurfaceSummary",
    "UiSemanticAddress",
    "UiSnapshotCatalog",
    "UiSnapshotListRequest",
    "UiSnapshotRef",
    "UiSnapshotRestoreRequest",
    "UiSnapshotRestoreResult",
    "UiTimeTravelRuntimeState",
    "UiTimeTravelHeadRequest",
    "UiWidgetIdentity",
    "UiWidgetActionInvokeRequest",
    "UiWidgetActionInvokeResult",
    "UiWidgetActionSummary",
    "UiWidgetRect",
    "UiWidgetTreeNode",
    "UiWidgetTreeRequest",
    "UiWidgetTreeResult",
    "UiWindowCatalog",
    "UiWindowCloseRequest",
    "UiWindowCloseResult",
    "UiWindowFocusRequest",
    "UiWindowFocusResult",
    "UiWindowIdentity",
    "UiWindowManagerScope",
    "UiWindowSemanticMarker",
    "UiWindowNavigateRequest",
    "UiWindowNavigateResult",
    "UiWindowOpenPolicy",
    "UiWindowOperationRequest",
    "UiWindowSnapshotRequest",
    "UiWindowSnapshotResult",
    "UiWindowSummary",
    "MainWindowWidgetIdentity",
    "ManagedWindowWidgetIdentity",
)

_EXPORT_NAMES = frozenset(__all__)
_DTO_EXPORT_PLACEHOLDER = object()
_MISSING_EXPORT = object()


def _dto_export_module_names() -> tuple[str, ...]:
    """Return DTO package modules from the package source authority."""
    return tuple(
        f"{__name__}.{module_info.name}"
        for module_info in iter_modules(__path__)
        if not module_info.ispkg
    )


def _external_export_modules() -> tuple[object, ...]:
    """Return non-DTO modules intentionally re-exported by this package."""
    from openhcs.agent import ui_bridge_identities
    from openhcs.runtime import window_snapshot

    return (window_snapshot, ui_bridge_identities)


def resolve_agent_dto_export(name: str):
    """Resolve one public DTO export from its owning module."""
    if name not in _EXPORT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    existing = globals().get(name, _MISSING_EXPORT)
    if existing is not _MISSING_EXPORT and existing is not _DTO_EXPORT_PLACEHOLDER:
        return existing
    for module in _external_export_modules():
        module_namespace = vars(module)
        if name in module_namespace:
            value = module_namespace[name]
            globals()[name] = value
            return value
    for module_name in _dto_export_module_names():
        module_namespace = import_module(module_name).__dict__
        if name in module_namespace:
            value = module_namespace[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str):
    """Resolve public DTO re-exports from their owning modules on demand."""
    return resolve_agent_dto_export(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORT_NAMES)


class _AgentDtoModule(ModuleType):
    """Module type that turns export placeholders into resolved DTO objects."""

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if name in _EXPORT_NAMES and value is _DTO_EXPORT_PLACEHOLDER:
            return resolve_agent_dto_export(name)
        return value


for _export_name in __all__:
    globals().setdefault(_export_name, _DTO_EXPORT_PLACEHOLDER)

sys.modules[__name__].__class__ = _AgentDtoModule
