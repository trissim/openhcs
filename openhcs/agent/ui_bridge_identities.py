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
            raise ValueError(f"{cls.__name__} does not declare a bridge identity value.")
        return cls.value


class UiWidgetIdentityDeclaration(UiBridgeIdentityDeclaration):
    """Registered widget identity declaration."""

    @classmethod
    def action_provider_id(cls) -> str:
        """Return the stable action-provider id for this widget."""
        return f"{cls.require_value()}.actions"


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


class UiStateSurfaceIdentityDeclarationBase(UiOwnedByWidgetIdentityDeclaration):
    """Registered state surface identity declaration."""


class PlateManagerWidgetIdentity(UiWidgetIdentityDeclaration):
    value = "plate_manager"
    enum_member_name = "PLATE_MANAGER"


class PipelineEditorWidgetIdentity(UiWidgetIdentityDeclaration):
    value = "pipeline_editor"
    enum_member_name = "PIPELINE_EDITOR"


class PipelineDebugToolbarWidgetIdentity(UiWidgetIdentityDeclaration):
    value = "pipeline_debug_toolbar"
    enum_member_name = "PIPELINE_DEBUG_TOOLBAR"


class ManagedWindowWidgetIdentity(UiWidgetIdentityDeclaration):
    value = "managed_window"
    enum_member_name = "MANAGED_WINDOW"


class PlateManagerOrchestratorCodeDocumentIdentity(UiCodeDocumentIdentityDeclaration):
    value = "plate_manager.orchestrator_config"
    enum_member_name = "PLATE_MANAGER_ORCHESTRATOR"
    widget_identity = PlateManagerWidgetIdentity


class PlateManagerStateSurfaceIdentityDeclaration(UiStateSurfaceIdentityDeclarationBase):
    value = "plate_manager.state"
    enum_member_name = "PLATE_MANAGER"
    widget_identity = PlateManagerWidgetIdentity


class PipelineEditorStateSurfaceIdentityDeclaration(
    UiStateSurfaceIdentityDeclarationBase
):
    value = "pipeline_editor.state"
    enum_member_name = "PIPELINE_EDITOR"
    widget_identity = PipelineEditorWidgetIdentity
