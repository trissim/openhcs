"""Registered identities for stable agent-facing UI bridge objects."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta


class UiBridgeIdentityDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Registered identity for stable agent-facing UI bridge objects."""

    __registry_key__ = "value"
    __skip_if_no_key__ = True

    value: ClassVar[str | None] = None
    enum_member_name: ClassVar[str | None] = None

    @classmethod
    def require_value(cls) -> str:
        if cls.value is None:
            raise ValueError(
                f"{cls.__name__} does not declare a bridge identity value."
            )
        return cls.value


class UiStableWindowIdentityDeclaration(UiBridgeIdentityDeclaration):
    """Registered identity and presentation for a stable public UI window."""

    title: ClassVar[str | None] = None

    @classmethod
    def require_title(cls) -> str:
        """Return the declaration-owned public window title."""

        if cls.title is None:
            raise ValueError(f"{cls.__name__} does not declare a window title.")
        return cls.title

    @classmethod
    def declaration_types(
        cls,
    ) -> tuple[type[UiStableWindowIdentityDeclaration], ...]:
        """Project the complete stable-window family from the identity registry."""

        return tuple(
            dict.fromkeys(
                declaration
                for declaration in cls.__registry__.values()
                if issubclass(declaration, cls)
                and declaration.value is not None
                and declaration.title is not None
            )
        )


class UiWidgetIdentityDeclaration(UiBridgeIdentityDeclaration):
    """Registered widget identity declaration."""

    @classmethod
    def action_provider_id(cls) -> str:
        """Return the stable action-provider id for this widget."""
        return f"{cls.require_value()}.actions"


class UiStableWidgetIdentityDeclaration(
    UiWidgetIdentityDeclaration,
    UiStableWindowIdentityDeclaration,
):
    """Stable public window whose content is also a registered UI widget."""


class UiOwnedByWidgetIdentityDeclaration(UiBridgeIdentityDeclaration):
    """Identity declaration for a UI bridge object owned by one widget."""

    widget_identity: ClassVar[type[UiWidgetIdentityDeclaration] | None] = None

    @classmethod
    def widget_id(cls) -> str:
        if cls.widget_identity is None:
            raise ValueError(f"{cls.__name__} does not declare a widget identity.")
        return cls.widget_identity.require_value()


class UiCodeDocumentIdentityDeclaration(UiOwnedByWidgetIdentityDeclaration):
    """Registered code document identity declaration."""


class UiStateSurfaceIdentityDeclarationBase(UiBridgeIdentityDeclaration):
    """Registered protocol identity for one UI state surface.

    Widget ownership and presentation metadata live on the owning widget class.
    """


class PlateManagerWidgetIdentity(UiStableWidgetIdentityDeclaration):
    value = "plate_manager"
    enum_member_name = "PLATE_MANAGER"
    title = "Plate Manager"


class PipelineEditorWidgetIdentity(UiStableWidgetIdentityDeclaration):
    value = "pipeline_editor"
    enum_member_name = "PIPELINE_EDITOR"
    title = "Pipeline Editor"


class PipelineDebugToolbarWidgetIdentity(UiWidgetIdentityDeclaration):
    value = "pipeline_debug_toolbar"
    enum_member_name = "PIPELINE_DEBUG_TOOLBAR"


class MainWindowWidgetIdentity(UiStableWidgetIdentityDeclaration):
    value = "main_window"
    enum_member_name = "MAIN_WINDOW"
    title = "OpenHCS"


class ManagedWindowWidgetIdentity(UiWidgetIdentityDeclaration):
    value = "managed_window"
    enum_member_name = "MANAGED_WINDOW"


class SystemMonitorWindowIdentity(UiStableWindowIdentityDeclaration):
    value = "system_monitor"
    title = "System Monitor"


class ZmqServerManagerWindowIdentity(UiStableWindowIdentityDeclaration):
    value = "zmq_server_manager"
    title = "ZMQ Server Manager"


class ImageBrowserWindowIdentity(UiStableWindowIdentityDeclaration):
    value = "image_browser"
    title = "Image Browser"


class LogViewerWindowIdentity(UiStableWindowIdentityDeclaration):
    value = "log_viewer"
    title = "Log Viewer"


class GlobalConfigWindowIdentity(UiStableWindowIdentityDeclaration):
    value = "global_config"
    title = "Configure OpenHCS"


class KnowledgeBaseWindowIdentity(UiStableWindowIdentityDeclaration):
    value = "knowledge_base"
    title = "OpenHCS Knowledge Base"


class AboutOpenHCSWindowIdentity(UiStableWindowIdentityDeclaration):
    value = "about_openhcs"
    title = "About OpenHCS"


class PlateManagerOrchestratorCodeDocumentIdentity(UiCodeDocumentIdentityDeclaration):
    value = "plate_manager.orchestrator_config"
    enum_member_name = "PLATE_MANAGER_ORCHESTRATOR"
    widget_identity = PlateManagerWidgetIdentity


class PlateManagerStateSurfaceIdentityDeclaration(
    UiStateSurfaceIdentityDeclarationBase
):
    value = "plate_manager.state"
    enum_member_name = "PLATE_MANAGER"


class PlateManagerLiveMeasurementsStateSurfaceIdentityDeclaration(
    UiStateSurfaceIdentityDeclarationBase
):
    value = "plate_manager.live_measurements"
    enum_member_name = "PLATE_MANAGER_LIVE_MEASUREMENTS"


class PipelineEditorStateSurfaceIdentityDeclaration(
    UiStateSurfaceIdentityDeclarationBase
):
    value = "pipeline_editor.state"
    enum_member_name = "PIPELINE_EDITOR"


class PipelineDebugSessionStateSurfaceIdentityDeclaration(
    UiStateSurfaceIdentityDeclarationBase
):
    value = "pipeline_debug_toolbar.session"
    enum_member_name = "PIPELINE_DEBUG_SESSION"


class UiLiveOverviewStateSurfaceIdentityDeclaration(
    UiStateSurfaceIdentityDeclarationBase
):
    value = "ui_live_overview.state"
    enum_member_name = "UI_LIVE_OVERVIEW"
