"""Shared context for the OpenHCS MCP adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openhcs.agent.path_policy import AgentPathPolicy

if TYPE_CHECKING:
    from openhcs.agent.services.architecture_projection_service import (
        ArchitectureProjectionService,
    )
    from openhcs.agent.services.config_service import ConfigService
    from openhcs.agent.services.execution_session_service import ExecutionSessionService
    from openhcs.agent.services.function_catalog_service import FunctionCatalogService
    from openhcs.agent.services.knowledge_base_service import KnowledgeBaseService
    from openhcs.agent.services.llm_context_service import AgentAuthoringContextService
    from openhcs.agent.services.object_state_field_help_service import (
        ObjectStateFieldHelpService,
    )
    from openhcs.agent.services.plate_inspection_service import PlateInspectionService
    from openhcs.agent.services.plate_streaming_service import PlateStreamingService
    from openhcs.agent.services.pipeline_authoring_service import (
        PipelineAuthoringService,
    )
    from openhcs.agent.services.runtime_server_service import RuntimeServerService
    from openhcs.agent.services.selected_plate_service import SelectedPlateService
    from openhcs.agent.services.synthetic_plate_service import (
        SyntheticPlateGenerationService,
    )
    from openhcs.agent.services.ui_bridge_service import UiBridgeService
    from openhcs.agent.services.viewer_window_service import ViewerWindowService


class OpenHCSAgentContext:
    """Lazy service container for MCP tool execution."""

    __slots__ = (
        "path_policy",
        "_architecture_service",
        "_authoring_context_service",
        "_config_service",
        "_execution_service",
        "_function_catalog",
        "_knowledge_base_service",
        "_object_state_field_help_service",
        "_pipeline_service",
        "_plate_inspection_service",
        "_plate_streaming_service",
        "_runtime_server_service",
        "_selected_plate_service",
        "_synthetic_plate_service",
        "_ui_bridge_service",
        "_viewer_window_service",
    )

    def __init__(
        self,
        *,
        path_policy: AgentPathPolicy | None = None,
        architecture_service: "ArchitectureProjectionService | None" = None,
        authoring_context_service: "AgentAuthoringContextService | None" = None,
        config_service: "ConfigService | None" = None,
        execution_service: "ExecutionSessionService | None" = None,
        function_catalog: "FunctionCatalogService | None" = None,
        knowledge_base_service: "KnowledgeBaseService | None" = None,
        object_state_field_help_service: "ObjectStateFieldHelpService | None" = None,
        pipeline_service: "PipelineAuthoringService | None" = None,
        plate_inspection_service: "PlateInspectionService | None" = None,
        plate_streaming_service: "PlateStreamingService | None" = None,
        runtime_server_service: "RuntimeServerService | None" = None,
        selected_plate_service: "SelectedPlateService | None" = None,
        synthetic_plate_service: "SyntheticPlateGenerationService | None" = None,
        ui_bridge_service: "UiBridgeService | None" = None,
        viewer_window_service: "ViewerWindowService | None" = None,
    ) -> None:
        self.path_policy = path_policy or AgentPathPolicy.from_environment()
        self._architecture_service = architecture_service
        self._authoring_context_service = authoring_context_service
        self._config_service = config_service
        self._execution_service = execution_service
        self._function_catalog = function_catalog
        self._knowledge_base_service = knowledge_base_service
        self._object_state_field_help_service = object_state_field_help_service
        self._pipeline_service = pipeline_service
        self._plate_inspection_service = plate_inspection_service
        self._plate_streaming_service = plate_streaming_service
        self._runtime_server_service = runtime_server_service
        self._selected_plate_service = selected_plate_service
        self._synthetic_plate_service = synthetic_plate_service
        self._ui_bridge_service = ui_bridge_service
        self._viewer_window_service = viewer_window_service

    @property
    def function_catalog(self) -> "FunctionCatalogService":
        if self._function_catalog is None:
            from openhcs.agent.services.function_catalog_service import (
                FunctionCatalogService,
            )

            self._function_catalog = FunctionCatalogService()
        return self._function_catalog

    @property
    def config_service(self) -> "ConfigService":
        if self._config_service is None:
            from openhcs.agent.services.config_service import ConfigService

            self._config_service = ConfigService()
        return self._config_service

    @property
    def architecture_service(self) -> "ArchitectureProjectionService":
        if self._architecture_service is None:
            from openhcs.agent.services.architecture_projection_service import (
                ArchitectureProjectionService,
            )

            self._architecture_service = ArchitectureProjectionService()
        return self._architecture_service

    @property
    def pipeline_service(self) -> "PipelineAuthoringService":
        if self._pipeline_service is None:
            from openhcs.agent.services.pipeline_authoring_service import (
                PipelineAuthoringService,
            )

            self._pipeline_service = PipelineAuthoringService(
                self.function_catalog,
                self.config_service,
            )
        return self._pipeline_service

    @property
    def authoring_context_service(self) -> "AgentAuthoringContextService":
        if self._authoring_context_service is None:
            from openhcs.agent.services.llm_context_service import (
                AgentAuthoringContextService,
            )

            self._authoring_context_service = AgentAuthoringContextService(
                self.function_catalog,
                self.config_service,
            )
        return self._authoring_context_service

    @property
    def knowledge_base_service(self) -> "KnowledgeBaseService":
        if self._knowledge_base_service is None:
            from openhcs.agent.services.knowledge_base_service import (
                KnowledgeBaseService,
            )

            self._knowledge_base_service = KnowledgeBaseService.from_path_policy(
                self.path_policy
            )
        return self._knowledge_base_service

    @property
    def plate_inspection_service(self) -> "PlateInspectionService":
        if self._plate_inspection_service is None:
            from openhcs.agent.services.plate_inspection_service import (
                PlateInspectionService,
            )

            self._plate_inspection_service = PlateInspectionService(self.path_policy)
        return self._plate_inspection_service

    @property
    def plate_streaming_service(self) -> "PlateStreamingService":
        if self._plate_streaming_service is None:
            from openhcs.agent.services.plate_streaming_service import (
                PlateStreamingService,
            )

            self._plate_streaming_service = PlateStreamingService(
                self.plate_inspection_service
            )
        return self._plate_streaming_service

    @property
    def synthetic_plate_service(self) -> "SyntheticPlateGenerationService":
        if self._synthetic_plate_service is None:
            from openhcs.agent.services.synthetic_plate_service import (
                SyntheticPlateGenerationService,
            )

            self._synthetic_plate_service = SyntheticPlateGenerationService(
                self.path_policy,
                self.plate_inspection_service,
            )
        return self._synthetic_plate_service

    @property
    def execution_service(self) -> "ExecutionSessionService":
        if self._execution_service is None:
            from openhcs.agent.services.execution_session_service import (
                ExecutionSessionService,
            )

            self._execution_service = ExecutionSessionService(
                path_policy=self.path_policy,
                pipeline_service=self.pipeline_service,
                config_service=self.config_service,
            )
        return self._execution_service

    @property
    def runtime_server_service(self) -> "RuntimeServerService":
        if self._runtime_server_service is None:
            from openhcs.agent.services.runtime_server_service import (
                RuntimeServerService,
            )

            self._runtime_server_service = RuntimeServerService()
        return self._runtime_server_service

    @property
    def ui_bridge_service(self) -> "UiBridgeService":
        if self._ui_bridge_service is None:
            from openhcs.agent.services.ui_bridge_service import UiBridgeService

            self._ui_bridge_service = UiBridgeService(path_policy=self.path_policy)
        return self._ui_bridge_service

    @property
    def selected_plate_service(self) -> "SelectedPlateService":
        if self._selected_plate_service is None:
            from openhcs.agent.services.selected_plate_service import (
                SelectedPlateService,
            )

            self._selected_plate_service = SelectedPlateService(
                self.ui_bridge_service,
                self.plate_inspection_service,
                self.plate_streaming_service,
            )
        return self._selected_plate_service

    @property
    def object_state_field_help_service(self) -> "ObjectStateFieldHelpService":
        if self._object_state_field_help_service is None:
            from openhcs.agent.services.object_state_field_help_service import (
                ObjectStateFieldHelpService,
            )

            self._object_state_field_help_service = ObjectStateFieldHelpService(
                self.ui_bridge_service,
                self.function_catalog,
            )
        return self._object_state_field_help_service

    @property
    def viewer_window_service(self) -> "ViewerWindowService":
        if self._viewer_window_service is None:
            from openhcs.agent.services.viewer_window_service import ViewerWindowService

            self._viewer_window_service = ViewerWindowService(
                path_policy=self.path_policy
            )
        return self._viewer_window_service


def create_agent_context() -> OpenHCSAgentContext:
    return OpenHCSAgentContext()
