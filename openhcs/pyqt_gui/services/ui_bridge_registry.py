"""Provider registry for the PyQt UI bridge surface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, TypeVar

from metaclass_registry import AutoRegisterMeta
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiBridgeRegistryKeyMixin,
    UiBridgeSnapshotProviderABC,
    UiActionProviderABC,
    UiCodeDocumentProviderABC,
    UiObjectStateScopeProviderABC,
    UiStateSurfaceProviderABC,
    UiWindowProviderABC,
)


UiBridgeProvider: TypeAlias = (
    UiCodeDocumentProviderABC
    | UiStateSurfaceProviderABC
    | UiActionProviderABC
    | UiWindowProviderABC
    | UiObjectStateScopeProviderABC
)
DynamicProviderT = TypeVar(
    "DynamicProviderT",
    UiCodeDocumentProviderABC,
    UiWindowProviderABC,
)


class UiRegisteredSurfaceKind(str, Enum):
    """Kinds of UI surfaces currently exposed through the bridge registry."""

    CODE_DOCUMENT = "code_document"
    STATE_SURFACE = "state_surface"
    ACTION = "action"
    WINDOW = "window"
    OBJECT_STATE_SCOPE = "object_state_scope"


@dataclass(frozen=True, slots=True)
class UiBridgeRegistrationContext:
    """Context passed to provider sets during bridge composition."""

    registry: "UiBridgeSurfaceRegistry"
    snapshot_provider: UiBridgeSnapshotProviderABC


class UiBridgeProviderSetABC(
    UiBridgeRegistryKeyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Composable collection of providers for one widget/domain."""

    @abstractmethod
    def register(self, context: UiBridgeRegistrationContext) -> None:
        raise NotImplementedError


class CompositeUiBridgeProviderSet(UiBridgeProviderSetABC):
    """Register several provider sets through one composition stage."""

    registry_key = "composite"

    def __init__(self, provider_sets: tuple[UiBridgeProviderSetABC, ...]) -> None:
        self._provider_sets = provider_sets

    def register(self, context: UiBridgeRegistrationContext) -> None:
        for provider_set in self._provider_sets:
            provider_set.register(context)


class UiBridgeSurfaceRegistry:
    """Nominal registry for UI bridge provider discovery and routing."""

    def __init__(self) -> None:
        self._code_document_providers: dict[str, UiCodeDocumentProviderABC] = {}
        self._state_surface_providers: dict[str, UiStateSurfaceProviderABC] = {}
        self._action_providers: dict[str, UiActionProviderABC] = {}
        self._window_providers: dict[str, UiWindowProviderABC] = {}
        self._object_state_scope_providers: dict[str, UiObjectStateScopeProviderABC] = {}

    def register_code_document_provider(
        self,
        provider: UiCodeDocumentProviderABC,
    ) -> None:
        document_id = provider.identity.document_id
        self._register_unique(
            self._code_document_providers,
            document_id,
            provider,
            UiRegisteredSurfaceKind.CODE_DOCUMENT,
        )

    def register_state_surface_provider(
        self,
        provider: UiStateSurfaceProviderABC,
    ) -> None:
        surface_id = provider.identity.surface_id
        self._register_unique(
            self._state_surface_providers,
            surface_id,
            provider,
            UiRegisteredSurfaceKind.STATE_SURFACE,
        )

    def register_action_provider(
        self,
        provider: UiActionProviderABC,
    ) -> None:
        provider_id = provider.identity.widget_id
        self._register_unique(
            self._action_providers,
            provider_id,
            provider,
            UiRegisteredSurfaceKind.ACTION,
        )

    def register_window_provider(
        self,
        provider: UiWindowProviderABC,
    ) -> None:
        provider_id = provider.identity.provider_id
        self._register_unique(
            self._window_providers,
            provider_id,
            provider,
            UiRegisteredSurfaceKind.WINDOW,
        )

    def register_object_state_scope_provider(
        self,
        provider: UiObjectStateScopeProviderABC,
    ) -> None:
        provider_id = provider.identity.provider_id
        self._register_unique(
            self._object_state_scope_providers,
            provider_id,
            provider,
            UiRegisteredSurfaceKind.OBJECT_STATE_SCOPE,
        )

    def code_document_provider(self, document_id: str) -> UiCodeDocumentProviderABC:
        if document_id in self._code_document_providers:
            return self._code_document_providers[document_id]
        return self._dynamic_provider(
            self._code_document_providers,
            document_id,
            unknown_message=f"Unknown UI code document: {document_id}",
        )

    def state_surface_provider(self, surface_id: str) -> UiStateSurfaceProviderABC:
        try:
            return self._state_surface_providers[surface_id]
        except KeyError as exc:
            raise KeyError(f"Unknown UI state surface: {surface_id}") from exc

    def action_provider(self, widget_id: str) -> UiActionProviderABC:
        try:
            return self._action_providers[widget_id]
        except KeyError as exc:
            raise KeyError(f"Unknown UI action provider: {widget_id}") from exc

    def window_provider(self, window_id: str) -> UiWindowProviderABC:
        return self._dynamic_provider(
            self._window_providers,
            window_id,
            unknown_message=f"Unknown UI window: {window_id}",
        )

    def code_document_providers(self) -> tuple[UiCodeDocumentProviderABC, ...]:
        return tuple(self._code_document_providers.values())

    def state_surface_providers(self) -> tuple[UiStateSurfaceProviderABC, ...]:
        return tuple(self._state_surface_providers.values())

    def action_providers(self) -> tuple[UiActionProviderABC, ...]:
        return tuple(self._action_providers.values())

    def window_providers(self) -> tuple[UiWindowProviderABC, ...]:
        return tuple(self._window_providers.values())

    def object_state_scope_providers(
        self,
    ) -> tuple[UiObjectStateScopeProviderABC, ...]:
        return tuple(self._object_state_scope_providers.values())

    @staticmethod
    def _dynamic_provider(
        providers: dict[str, DynamicProviderT],
        surface_id: str,
        *,
        unknown_message: str,
    ) -> DynamicProviderT:
        for provider in providers.values():
            if provider.handles(surface_id):
                return provider
        raise KeyError(unknown_message)

    @staticmethod
    def _register_unique(
        providers: dict[
            str,
            UiBridgeProvider,
        ],
        provider_id: str,
        provider: UiBridgeProvider,
        surface_kind: UiRegisteredSurfaceKind,
    ) -> None:
        if provider_id in providers:
            raise ValueError(
                f"Duplicate UI bridge {surface_kind.value} provider id: {provider_id}"
            )
        providers[provider_id] = provider
