"""Window projection providers for the PyQt UI bridge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from PyQt6.QtWidgets import QWidget
from pyqt_reactive.services.window_manager import WindowManager

from openhcs.agent.dto.common import AgentError, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiWindowCatalog,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowIdentity,
    UiWindowManagerScope,
    UiWindowSummary,
)
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiWindowProviderABC,
    UiWindowProviderIdentity,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.main import OpenHCSMainWindow


WindowRouteCollection: TypeAlias = (
    tuple["EmbeddedWindowRoute", ...] | tuple["ManagedWindowRoute", ...]
)
MAIN_WINDOW_PROVIDER_ID = "main_window.windows"
EMBEDDED_WINDOW_KIND = "embedded"
MANAGED_WINDOW_KIND = "managed"
DYNAMIC_SCOPE_WINDOW_KIND = "scope"


@dataclass(frozen=True, slots=True)
class FocusableWindowRouteMixin:
    """Shared focus algorithm for static focusable window routes."""

    focus_action: Callable[[], None]

    def summary(self) -> UiWindowSummary:
        raise NotImplementedError

    def focus(self) -> UiWindowSummary:
        self.focus_action()
        return self.summary()


@dataclass(frozen=True, slots=True)
class EmbeddedWindowRoute(FocusableWindowRouteMixin):
    """Focusable embedded widget route owned by the main window layout."""

    identity: UiWindowIdentity
    title: str
    widget_supplier: Callable[[], QWidget]

    def summary(self) -> UiWindowSummary:
        widget = self.widget_supplier()
        scope = UiWindowManagerScope.from_identity(self.identity)
        return UiWindowSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity,
            title=self.title,
            window_kind=EMBEDDED_WINDOW_KIND,
            visible=widget.isVisible(),
            focusable=True,
            manager_scope=scope,
        )


@dataclass(frozen=True, slots=True)
class ManagedWindowRoute(FocusableWindowRouteMixin):
    """Focusable WindowManager-backed static window route."""

    identity: UiWindowIdentity
    title: str

    def summary(self) -> UiWindowSummary:
        scope = UiWindowManagerScope.from_identity(self.identity)
        window = WindowManager.get_window(scope.value)
        visible = False
        if window is not None:
            visible = window.isVisible()
        return UiWindowSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity,
            title=self.title,
            window_kind=MANAGED_WINDOW_KIND,
            visible=visible,
            focusable=True,
            manager_scope=scope,
        )


@dataclass(frozen=True, slots=True)
class WindowIdentitySet:
    """Closed set of window identities for static route filtering."""

    identities: frozenset[UiWindowIdentity]

    @classmethod
    def from_routes(
        cls,
        routes: WindowRouteCollection,
    ) -> "WindowIdentitySet":
        return cls(frozenset(route.identity for route in routes))

    def contains(self, identity: UiWindowIdentity) -> bool:
        return identity in self.identities

    def union(self, other: "WindowIdentitySet") -> "WindowIdentitySet":
        return WindowIdentitySet(self.identities | other.identities)


@dataclass(frozen=True, slots=True)
class WindowRouteIndex:
    """Identity-indexed route lookup for static window routes."""

    embedded_routes: tuple[EmbeddedWindowRoute, ...]
    managed_routes: tuple[ManagedWindowRoute, ...]

    @property
    def static_identities(self) -> WindowIdentitySet:
        return WindowIdentitySet.from_routes(
            self.embedded_routes
        ).union(
            WindowIdentitySet.from_routes(self.managed_routes)
        )

    def embedded_route(self, identity: UiWindowIdentity) -> EmbeddedWindowRoute | None:
        return self._route_by_identity(self.embedded_routes, identity)

    def managed_route(self, identity: UiWindowIdentity) -> ManagedWindowRoute | None:
        return self._route_by_identity(self.managed_routes, identity)

    @staticmethod
    def _route_by_identity(
        routes: WindowRouteCollection,
        identity: UiWindowIdentity,
    ):
        for route in routes:
            if route.identity == identity:
                return route
        return None


class EmbeddedWindowProjection:
    """Project embedded main-window widgets."""

    def __init__(self, main_window: "OpenHCSMainWindow") -> None:
        self._main_window = main_window

    def routes(self) -> tuple[EmbeddedWindowRoute, ...]:
        embedded = self._main_window.embedded_widgets
        return (
            EmbeddedWindowRoute(
                identity=UiWindowIdentity(window_id="plate_manager"),
                title="Plate Manager",
                widget_supplier=embedded.require_plate_manager,
                focus_action=embedded.show_plate_manager,
            ),
            EmbeddedWindowRoute(
                identity=UiWindowIdentity(window_id="pipeline_editor"),
                title="Pipeline Editor",
                widget_supplier=embedded.require_pipeline_editor,
                focus_action=embedded.show_pipeline_editor,
            ),
            EmbeddedWindowRoute(
                identity=UiWindowIdentity(window_id="zmq_server_manager"),
                title="ZMQ Server Manager",
                widget_supplier=embedded.require_zmq_manager,
                focus_action=embedded.show_zmq_manager,
            ),
        )


class ManagedWindowProjection:
    """Project static WindowManager-managed windows."""

    def __init__(self, main_window: "OpenHCSMainWindow") -> None:
        self._main_window = main_window

    def routes(
        self,
        embedded_routes: tuple[EmbeddedWindowRoute, ...],
    ) -> tuple[ManagedWindowRoute, ...]:
        embedded_identities = WindowIdentitySet.from_routes(embedded_routes)
        return tuple(
            ManagedWindowRoute(
                identity=UiWindowIdentity(window_id=spec.window_id),
                title=spec.title,
                focus_action=self._focus_static_window_action(spec.window_id),
            )
            for spec in self._main_window.window_specs.values()
            if not embedded_identities.contains(
                UiWindowIdentity(window_id=spec.window_id)
            )
        )

    def _focus_static_window_action(self, window_id: str) -> Callable[[], None]:
        def focus() -> None:
            self._main_window.show_window(window_id, hide_if_startup=False)

        return focus


class DynamicScopeWindowProjection:
    """Project already-open dynamic ObjectState/scope windows."""

    def summaries(
        self,
        route_index: WindowRouteIndex,
    ) -> tuple[UiWindowSummary, ...]:
        return tuple(
            self.summary(UiWindowIdentity(window_id=scope_id))
            for scope_id in WindowManager.get_open_scopes()
            if not route_index.static_identities.contains(
                UiWindowIdentity(window_id=scope_id)
            )
        )

    def summary(self, identity: UiWindowIdentity) -> UiWindowSummary:
        scope = UiWindowManagerScope.from_identity(identity)
        window = WindowManager.get_window(scope.value)
        title = scope.value
        visible = False
        if window is not None:
            title = window.windowTitle()
            visible = window.isVisible()
        return UiWindowSummary(
            schema_version=SCHEMA_VERSION,
            identity=identity,
            title=title,
            window_kind=DYNAMIC_SCOPE_WINDOW_KIND,
            visible=visible,
            focusable=window is not None,
            manager_scope=scope,
        )


class UiWindowProjectionService:
    """Project the main-window/WindowManager window graph into agent DTOs."""

    def __init__(self, main_window: "OpenHCSMainWindow") -> None:
        self._embedded = EmbeddedWindowProjection(main_window)
        self._managed = ManagedWindowProjection(main_window)
        self._dynamic = DynamicScopeWindowProjection()

    def summaries(self) -> tuple[UiWindowSummary, ...]:
        route_index = self._route_index()
        return tuple(
            route.summary()
            for route in route_index.embedded_routes
        ) + tuple(
            route.summary()
            for route in route_index.managed_routes
        ) + self._dynamic.summaries(route_index)

    def handles(self, window_id: str) -> bool:
        identity = UiWindowIdentity(window_id=window_id)
        route_index = self._route_index()
        if route_index.embedded_route(identity) is not None:
            return True
        if route_index.managed_route(identity) is not None:
            return True
        return identity.window_id in WindowManager.get_open_scopes()

    def focus(self, request: UiWindowFocusRequest) -> UiWindowFocusResult:
        identity = request.as_window_identity()
        route_index = self._route_index()
        embedded_route = route_index.embedded_route(identity)
        if embedded_route is not None:
            return self._focused_result(request, embedded_route.focus())

        managed_route = route_index.managed_route(identity)
        if managed_route is not None and request.create_if_missing:
            return self._focused_result(request, managed_route.focus())

        scope = UiWindowManagerScope.from_identity(identity)
        if WindowManager.focus_and_navigate(scope.value):
            return self._focused_result(
                request,
                self._dynamic.summary(identity),
            )

        return UiWindowFocusResult(
            schema_version=SCHEMA_VERSION,
            window_id=identity.window_id,
            focused=False,
            errors=(
                AgentError(
                    code="unknown_ui_window",
                    message=f"Unknown or closed UI window: {scope.value!r}",
                ),
            ),
        )

    def _route_index(self) -> WindowRouteIndex:
        embedded_routes = self._embedded.routes()
        return WindowRouteIndex(
            embedded_routes=embedded_routes,
            managed_routes=self._managed.routes(embedded_routes),
        )

    @staticmethod
    def _focused_result(
        request: UiWindowFocusRequest,
        summary: UiWindowSummary,
    ) -> UiWindowFocusResult:
        return UiWindowFocusResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            focused=True,
            summary=summary,
        )


class MainWindowBridgeWindowProvider(UiWindowProviderABC):
    """Window provider backed by the main-window composition root."""

    identity = UiWindowProviderIdentity(
        provider_id=MAIN_WINDOW_PROVIDER_ID,
        title="Main window windows",
    )

    def __init__(self, main_window: "OpenHCSMainWindow") -> None:
        self._projection = UiWindowProjectionService(main_window)

    def catalog(self) -> UiWindowCatalog:
        return UiWindowCatalog(
            schema_version=SCHEMA_VERSION,
            windows=self._projection.summaries(),
        )

    def handles(self, window_id: str) -> bool:
        return self._projection.handles(window_id)

    def focus(self, request: UiWindowFocusRequest) -> UiWindowFocusResult:
        return self._projection.focus(request)


@dataclass(frozen=True, slots=True)
class MainWindowBridgeProviderSet(UiBridgeProviderSetABC):
    """Provider set for main-window generic UI projections."""

    main_window: "OpenHCSMainWindow"
    registry_key = MAIN_WINDOW_PROVIDER_ID

    def register(self, context: UiBridgeRegistrationContext) -> None:
        context.registry.register_window_provider(
            MainWindowBridgeWindowProvider(self.main_window)
        )
