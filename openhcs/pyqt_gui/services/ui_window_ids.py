"""Agent-visible OpenHCS UI window identifiers."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

from openhcs.agent.ui_bridge_identities import (
    AboutOpenHCSWindowIdentity,
    GlobalConfigWindowIdentity,
    ImageBrowserWindowIdentity,
    KnowledgeBaseWindowIdentity,
    LogViewerWindowIdentity,
    MainWindowWidgetIdentity,
    PipelineEditorWidgetIdentity,
    PlateManagerWidgetIdentity,
    SystemMonitorWindowIdentity,
    ZmqServerManagerWindowIdentity,
)


class OpenHCSUiWindowId:
    """Closed identifiers for stable windows exposed through the UI bridge."""

    main_window: ClassVar[str] = MainWindowWidgetIdentity.require_value()
    plate_manager: ClassVar[str] = PlateManagerWidgetIdentity.require_value()
    pipeline_editor: ClassVar[str] = PipelineEditorWidgetIdentity.require_value()
    system_monitor: ClassVar[str] = SystemMonitorWindowIdentity.require_value()
    zmq_server_manager: ClassVar[str] = ZmqServerManagerWindowIdentity.require_value()
    image_browser: ClassVar[str] = ImageBrowserWindowIdentity.require_value()
    log_viewer: ClassVar[str] = LogViewerWindowIdentity.require_value()
    global_config: ClassVar[str] = GlobalConfigWindowIdentity.require_value()
    knowledge_base: ClassVar[str] = KnowledgeBaseWindowIdentity.require_value()
    about: ClassVar[str] = AboutOpenHCSWindowIdentity.require_value()

    manager_scope_aliases: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "": global_config,
        }
    )

    @classmethod
    def agent_window_id_for_manager_scope(cls, manager_scope_id: str) -> str:
        if manager_scope_id in cls.manager_scope_aliases:
            return cls.manager_scope_aliases[manager_scope_id]
        return manager_scope_id

    @classmethod
    def manager_scopes_for_agent_window_id(
        cls, agent_window_id: str
    ) -> tuple[str, ...]:
        aliased_scopes = tuple(
            manager_scope_id
            for manager_scope_id, aliased_window_id in cls.manager_scope_aliases.items()
            if aliased_window_id == agent_window_id
            and manager_scope_id != agent_window_id
        )
        return (agent_window_id,) + aliased_scopes

    @classmethod
    def canonical_manager_scope_for_agent_window_id(cls, agent_window_id: str) -> str:
        for manager_scope_id, aliased_window_id in cls.manager_scope_aliases.items():
            if aliased_window_id == agent_window_id:
                return manager_scope_id
        return agent_window_id
