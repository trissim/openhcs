"""Shared context for the OpenHCS MCP adapter."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import NoneType, UnionType
from typing import Union, get_args, get_origin, get_type_hints

from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.services import (
    AgentAuthoringContextService,
    ArchitectureProjectionService,
    ConfigService,
    ExecutionSessionService,
    FunctionCatalogService,
    KnowledgeBaseService,
    ObjectStateFieldHelpService,
    PlateInspectionService,
    PlateStreamingService,
    PipelineAuthoringService,
    RuntimeServerService,
    SelectedPlateService,
    SyntheticPlateGenerationService,
    UiBridgeService,
    ViewerWindowService,
)


@dataclass(slots=True)
class OpenHCSAgentContext:
    path_policy: AgentPathPolicy = field(default_factory=AgentPathPolicy.from_environment)
    function_catalog: FunctionCatalogService = field(default_factory=FunctionCatalogService)
    config_service: ConfigService = field(default_factory=ConfigService)
    architecture_service: ArchitectureProjectionService = field(
        default_factory=ArchitectureProjectionService
    )
    knowledge_base_service: KnowledgeBaseService | None = None
    plate_inspection_service: PlateInspectionService | None = None
    plate_streaming_service: PlateStreamingService | None = None
    synthetic_plate_service: SyntheticPlateGenerationService | None = None
    pipeline_service: PipelineAuthoringService | None = None
    authoring_context_service: AgentAuthoringContextService | None = None
    execution_service: ExecutionSessionService | None = None
    runtime_server_service: RuntimeServerService | None = None
    ui_bridge_service: UiBridgeService | None = None
    selected_plate_service: SelectedPlateService | None = None
    object_state_field_help_service: ObjectStateFieldHelpService | None = None
    viewer_window_service: ViewerWindowService | None = None

    def __post_init__(self) -> None:
        if self.pipeline_service is None:
            self.pipeline_service = PipelineAuthoringService(self.function_catalog)
        if self.authoring_context_service is None:
            self.authoring_context_service = AgentAuthoringContextService(
                self.function_catalog,
                self.config_service,
            )
        if self.knowledge_base_service is None:
            self.knowledge_base_service = KnowledgeBaseService.from_path_policy(
                self.path_policy
            )
        if self.plate_inspection_service is None:
            self.plate_inspection_service = PlateInspectionService(self.path_policy)
        if self.plate_streaming_service is None:
            self.plate_streaming_service = PlateStreamingService(
                self.plate_inspection_service
            )
        if self.synthetic_plate_service is None:
            self.synthetic_plate_service = SyntheticPlateGenerationService(
                self.path_policy,
                self.plate_inspection_service,
            )
        if self.execution_service is None:
            self.execution_service = ExecutionSessionService(
                path_policy=self.path_policy,
                pipeline_service=self.pipeline_service,
                config_service=self.config_service,
            )
        if self.runtime_server_service is None:
            self.runtime_server_service = RuntimeServerService()
        if self.ui_bridge_service is None:
            self.ui_bridge_service = UiBridgeService(path_policy=self.path_policy)
        if self.selected_plate_service is None:
            self.selected_plate_service = SelectedPlateService(
                self.ui_bridge_service,
                self.plate_inspection_service,
                self.plate_streaming_service,
            )
        if self.object_state_field_help_service is None:
            self.object_state_field_help_service = ObjectStateFieldHelpService(
                self.ui_bridge_service,
                self.function_catalog,
            )
        if self.viewer_window_service is None:
            self.viewer_window_service = ViewerWindowService(
                path_policy=self.path_policy
            )


def create_agent_context() -> OpenHCSAgentContext:
    return OpenHCSAgentContext()


def openhcs_agent_context_source_types() -> tuple[type, ...]:
    """Return source-watch owner types declared by OpenHCSAgentContext fields."""
    source_types: list[type] = []
    seen: set[type] = set()
    type_hints = get_type_hints(OpenHCSAgentContext)
    for context_field in fields(OpenHCSAgentContext):
        for source_type in _source_types_from_annotation(type_hints[context_field.name]):
            if source_type in seen:
                continue
            seen.add(source_type)
            source_types.append(source_type)
    return tuple(source_types)


def _source_types_from_annotation(annotation: object) -> tuple[type, ...]:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return tuple(
            source_type
            for item in get_args(annotation)
            for source_type in _source_types_from_annotation(item)
        )
    if isinstance(annotation, type) and annotation is not NoneType:
        return (annotation,)
    return ()
