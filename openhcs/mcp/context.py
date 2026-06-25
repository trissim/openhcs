"""Shared context for the OpenHCS MCP adapter."""

from __future__ import annotations

from dataclasses import dataclass, field

from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.services import (
    AgentAuthoringContextService,
    ArchitectureProjectionService,
    ConfigService,
    ExecutionSessionService,
    FunctionCatalogService,
    PipelineAuthoringService,
    RuntimeServerService,
    UiBridgeService,
    ViewerWindowService,
)


OPENHCS_AGENT_CONTEXT_SOURCE_TYPES = (
    AgentPathPolicy,
    FunctionCatalogService,
    ConfigService,
    ArchitectureProjectionService,
    PipelineAuthoringService,
    AgentAuthoringContextService,
    ExecutionSessionService,
    RuntimeServerService,
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
    pipeline_service: PipelineAuthoringService | None = None
    authoring_context_service: AgentAuthoringContextService | None = None
    execution_service: ExecutionSessionService | None = None
    runtime_server_service: RuntimeServerService | None = None
    ui_bridge_service: UiBridgeService | None = None
    viewer_window_service: ViewerWindowService | None = None

    def __post_init__(self) -> None:
        if self.pipeline_service is None:
            self.pipeline_service = PipelineAuthoringService(self.function_catalog)
        if self.authoring_context_service is None:
            self.authoring_context_service = AgentAuthoringContextService(
                self.function_catalog,
                self.config_service,
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
            self.ui_bridge_service = UiBridgeService()
        if self.viewer_window_service is None:
            self.viewer_window_service = ViewerWindowService()


def create_agent_context() -> OpenHCSAgentContext:
    return OpenHCSAgentContext()
