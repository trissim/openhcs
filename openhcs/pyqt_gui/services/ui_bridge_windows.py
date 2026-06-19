"""Window projection providers for the PyQt UI bridge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta
from PyQt6.QtWidgets import QApplication, QWidget
from pyqt_reactive.services.scope_window_factory import ScopeWindowRegistry
from pyqt_reactive.services.scope_window_navigation import ScopeWindowNavigationService
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.services.window_navigation import WindowNavigationRequest
from pyqt_reactive.widgets.shared import (
    BaseFormDialog,
    ManagedWindowActionCapabilities,
)

from openhcs.agent.dto.common import AgentError, AgentResourceRef, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionIdentity,
    UiActionInvocationStatus,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiActionSummary,
    UiMutationReceipt,
    UiWindowCatalog,
    UiWindowCloseRequest,
    UiWindowCloseResult,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowIdentity,
    UiWindowManagerScope,
    UiWindowNavigateRequest,
    UiWindowNavigateResult,
    UiWindowSnapshotRequest,
    UiWindowSnapshotResult,
    UiWindowSummary,
)
from openhcs.runtime.qt_window_snapshot import (
    QtWindowSnapshotRequest,
    QtWindowSnapshotService,
)
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiActionProviderABC,
    UiActionProviderIdentity,
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
QT_TOP_LEVEL_PROVIDER_ID = "qt_top_level.windows"
EMBEDDED_WINDOW_KIND = "embedded"
MANAGED_WINDOW_KIND = "managed"
DYNAMIC_SCOPE_WINDOW_KIND = "scope"
QT_TOP_LEVEL_WINDOW_KIND = "qt_top_level"
QT_TOP_LEVEL_WINDOW_ID_PREFIX = "qt_top_level:"
MANAGED_WINDOW_ACTION_WIDGET_ID = "managed_window"
MANAGED_WINDOW_ACTIONS_TITLE = "Managed window actions"


class ManagedWindowAction(str, Enum):
    """Agent-visible actions shared by managed form windows."""

    SAVE_AND_CLOSE = "save_and_close"
    SAVE_WITHOUT_CLOSE = "save_without_close"
    DISCARD_AND_CLOSE = "discard_and_close"


MANAGED_WINDOW_ACTION_PROVIDER_IDENTITY = UiActionProviderIdentity(
    action_id="managed_window.actions",
    widget_id=MANAGED_WINDOW_ACTION_WIDGET_ID,
    title=MANAGED_WINDOW_ACTIONS_TITLE,
)


@dataclass(frozen=True, slots=True)
class ManagedWindowActionSpec:
    """Closed action semantics for one managed-window agent action."""

    action: ManagedWindowAction
    title: str
    side_effects: tuple[str, ...]
    is_supported: Callable[[ManagedWindowActionCapabilities], bool]
    dispatch: Callable[[BaseFormDialog], None]


def _supports_save_and_close(
    capabilities: ManagedWindowActionCapabilities,
) -> bool:
    return capabilities.save_and_close


def _supports_save_without_close(
    capabilities: ManagedWindowActionCapabilities,
) -> bool:
    return capabilities.save_without_close


def _supports_discard_and_close(
    capabilities: ManagedWindowActionCapabilities,
) -> bool:
    return capabilities.discard_and_close


def _dispatch_save_and_close(window: BaseFormDialog) -> None:
    window.agent_save_managed_window(close_window=True)


def _dispatch_save_without_close(window: BaseFormDialog) -> None:
    window.agent_save_managed_window(close_window=False)


def _dispatch_discard_and_close(window: BaseFormDialog) -> None:
    window.agent_discard_and_close_managed_window()


MANAGED_WINDOW_ACTION_SPECS = (
    ManagedWindowActionSpec(
        action=ManagedWindowAction.SAVE_AND_CLOSE,
        title="Save and close",
        side_effects=("saves_window_state", "closes_window"),
        is_supported=_supports_save_and_close,
        dispatch=_dispatch_save_and_close,
    ),
    ManagedWindowActionSpec(
        action=ManagedWindowAction.SAVE_WITHOUT_CLOSE,
        title="Save without closing",
        side_effects=("saves_window_state",),
        is_supported=_supports_save_without_close,
        dispatch=_dispatch_save_without_close,
    ),
    ManagedWindowActionSpec(
        action=ManagedWindowAction.DISCARD_AND_CLOSE,
        title="Discard changes and close",
        side_effects=("discards_unsaved_window_state", "closes_window"),
        is_supported=_supports_discard_and_close,
        dispatch=_dispatch_discard_and_close,
    ),
)
MANAGED_WINDOW_ACTION_SPEC_BY_ACTION = {
    spec.action: spec for spec in MANAGED_WINDOW_ACTION_SPECS
}


@dataclass(frozen=True, slots=True)
class WindowProjectionTarget:
    """Resolved Qt widget plus its agent-facing summary."""

    widget: QWidget
    summary: UiWindowSummary


class WindowCloseResultBoundaryPolicy:
    """Single result boundary for UI window close operations."""

    @staticmethod
    def closed_summary(summary: UiWindowSummary) -> UiWindowSummary:
        return replace(summary, visible=False, focusable=False)

    @classmethod
    def closed(
        cls,
        request: UiWindowCloseRequest,
        summary: UiWindowSummary,
    ) -> UiWindowCloseResult:
        return UiWindowCloseResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            closed=True,
            summary=cls.closed_summary(summary),
        )

    @staticmethod
    def error(
        request: UiWindowCloseRequest,
        code: str,
        message: str,
        *,
        summary: UiWindowSummary | None = None,
    ) -> UiWindowCloseResult:
        return UiWindowCloseResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            closed=False,
            summary=summary,
            errors=(AgentError(code=code, message=message),),
        )


class MainWindowContextFactoryMixin:
    """Construct registry leaves from an optional OpenHCS main-window context."""

    @classmethod
    def create(cls, main_window: "OpenHCSMainWindow | None"):
        return cls(main_window)


class WindowCatalogProjectionABC(
    MainWindowContextFactoryMixin,
    UiWindowProviderABC,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Provider/projection authority for one catalog of UI windows."""

    __registry_key__ = "projection_id"
    __skip_if_no_key__ = True

    projection_id: ClassVar[str | None] = None
    identity: ClassVar[UiWindowProviderIdentity]

    @classmethod
    def registered_types(cls) -> tuple[type["WindowCatalogProjectionABC"], ...]:
        return tuple(cls.__registry__.values())

    @classmethod
    def for_projection_id(
        cls,
        projection_id: str,
        main_window: "OpenHCSMainWindow | None",
    ) -> "WindowCatalogProjectionABC":
        return cls.__registry__[projection_id].create(main_window)

    def catalog(self) -> UiWindowCatalog:
        return UiWindowCatalog(
            schema_version=SCHEMA_VERSION,
            windows=self.summaries(),
        )

    @abstractmethod
    def summaries(self) -> tuple[UiWindowSummary, ...]:
        """Return the windows currently visible through this projection."""
        raise NotImplementedError

    @abstractmethod
    def handles(self, window_id: str) -> bool:
        """Return whether this projection owns a window id."""
        raise NotImplementedError

    @abstractmethod
    def focus(self, request: UiWindowFocusRequest) -> UiWindowFocusResult:
        raise NotImplementedError

    @abstractmethod
    def navigate(self, request: UiWindowNavigateRequest) -> UiWindowNavigateResult:
        raise NotImplementedError

    @abstractmethod
    def close(self, request: UiWindowCloseRequest) -> UiWindowCloseResult:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, request: UiWindowSnapshotRequest) -> UiWindowSnapshotResult:
        raise NotImplementedError


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

    def widget(self) -> QWidget:
        return self.widget_supplier()

    def summary(self) -> UiWindowSummary:
        widget = self.widget()
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

    def widget(self, create_if_missing: bool) -> QWidget | None:
        scope = UiWindowManagerScope.from_identity(self.identity)
        window = WindowManager.get_window(scope.value)
        if window is not None:
            return window
        if not create_if_missing:
            return None
        self.focus_action()
        return WindowManager.get_window(scope.value)

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


class QtTopLevelWindowProjection(WindowCatalogProjectionABC):
    """Project visible Qt top-level windows that are not WindowManager scopes."""

    projection_id = QT_TOP_LEVEL_PROVIDER_ID
    identity = UiWindowProviderIdentity(
        provider_id=QT_TOP_LEVEL_PROVIDER_ID,
        title="Qt top-level windows",
    )

    def __init__(
        self,
        main_window: "OpenHCSMainWindow | None",
    ) -> None:
        self._main_window = main_window
        self._snapshot_results = UiWindowSnapshotResultFactory()

    def summaries(self) -> tuple[UiWindowSummary, ...]:
        excluded_widgets = self._excluded_widgets()
        return tuple(
            self.summary(widget)
            for widget in self._top_level_widgets()
            if widget not in excluded_widgets
        )

    def handles(self, window_id: str) -> bool:
        return self.target(UiWindowIdentity(window_id=window_id)) is not None

    def target(self, identity: UiWindowIdentity) -> WindowProjectionTarget | None:
        for widget in self._top_level_widgets():
            if self._identity(widget) == identity:
                return WindowProjectionTarget(widget=widget, summary=self.summary(widget))
        return None

    def focus(self, request: UiWindowFocusRequest) -> UiWindowFocusResult:
        target = self.target(request.as_window_identity())
        if target is None:
            return UiWindowFocusResult(
                schema_version=SCHEMA_VERSION,
                window_id=request.window_id,
                focused=False,
                errors=(
                    AgentError(
                        code="unknown_ui_window",
                        message=f"Unknown or closed UI window: {request.window_id!r}",
                    ),
                ),
            )
        target.widget.show()
        target.widget.raise_()
        target.widget.activateWindow()
        return UiWindowFocusResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            focused=True,
            summary=self.summary(target.widget),
        )

    def navigate(self, request: UiWindowNavigateRequest) -> UiWindowNavigateResult:
        focus_result = self.focus(
            UiWindowFocusRequest(
                window_id=request.window_id,
                open_policy=request.open_policy,
            )
        )
        return UiWindowNavigateResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            focused=focus_result.focused,
            navigated=False,
            created=False,
            summary=focus_result.summary,
            errors=focus_result.errors,
            warnings=focus_result.warnings,
        )

    def close(self, request: UiWindowCloseRequest) -> UiWindowCloseResult:
        target = self.target(request.as_window_identity())
        if target is None:
            return WindowCloseResultBoundaryPolicy.error(
                request,
                "unknown_ui_window",
                f"Unknown or closed UI window: {request.window_id!r}",
            )
        if target.widget is self._main_window:
            return WindowCloseResultBoundaryPolicy.error(
                request,
                "ui_window_close_unsupported",
                "The OpenHCS main window cannot be closed through this UI bridge operation.",
                summary=target.summary,
            )
        if target.widget.close():
            return WindowCloseResultBoundaryPolicy.closed(
                request,
                target.summary,
            )
        return WindowCloseResultBoundaryPolicy.error(
            request,
            "ui_window_close_rejected",
            f"Qt rejected the close request for UI window: {request.window_id!r}",
            summary=target.summary,
        )

    def snapshot(self, request: UiWindowSnapshotRequest) -> UiWindowSnapshotResult:
        target = self.target(request.as_window_identity())
        if target is None:
            return self._snapshot_results.error(
                request,
                AgentError(
                    code="unknown_ui_window",
                    message=f"Unknown or closed UI window: {request.window_id!r}",
                ),
            )
        return self._snapshot_results.capture(request, target)

    @classmethod
    def summary(cls, widget: QWidget) -> UiWindowSummary:
        identity = cls._identity(widget)
        return UiWindowSummary(
            schema_version=SCHEMA_VERSION,
            identity=identity,
            title=cls._title(widget),
            window_kind=QT_TOP_LEVEL_WINDOW_KIND,
            visible=widget.isVisible(),
            focusable=True,
            manager_scope=None,
        )

    @staticmethod
    def _top_level_widgets() -> tuple[QWidget, ...]:
        application = QApplication.instance()
        if application is None:
            return ()
        del application
        return tuple(
            widget
            for widget in QApplication.topLevelWidgets()
            if widget.isWindow() and widget.isVisible()
        )

    @staticmethod
    def _excluded_widgets() -> frozenset[QWidget]:
        widgets: set[QWidget] = set()
        for scope_id in WindowManager.get_open_scopes():
            widget = WindowManager.get_window(scope_id)
            if widget is not None:
                widgets.add(widget.window())
        return frozenset(widgets)

    @staticmethod
    def _identity(widget: QWidget) -> UiWindowIdentity:
        return UiWindowIdentity(
            window_id=f"{QT_TOP_LEVEL_WINDOW_ID_PREFIX}{int(widget.winId())}"
        )

    @staticmethod
    def _title(widget: QWidget) -> str:
        title = widget.windowTitle()
        if title:
            return title
        return type(widget).__name__


class UiWindowSnapshotResultFactory:
    """Build UI bridge screenshot results from resolved Qt window targets."""

    def __init__(self) -> None:
        self._snapshotter = QtWindowSnapshotService()

    def capture(
        self,
        request: UiWindowSnapshotRequest,
        target: WindowProjectionTarget,
    ) -> UiWindowSnapshotResult:
        try:
            snapshot = self._snapshotter.capture(
                QtWindowSnapshotRequest(
                    widget=target.widget,
                    capture=request.snapshot,
                    subject_id=request.window_id,
                    title=target.summary.title,
                )
            )
        except Exception as exc:
            return self.error(
                request,
                AgentError.from_exception("ui_window_snapshot_failed", exc),
                summary=target.summary,
            )
        return UiWindowSnapshotResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            captured=True,
            resource=AgentResourceRef(
                uri=snapshot.uri,
                title=snapshot.title,
                mime_type=snapshot.mime_type,
                path=snapshot.path,
                size_bytes=snapshot.size_bytes,
                sha256=snapshot.sha256,
            ),
            summary=target.summary,
            width=snapshot.width,
            height=snapshot.height,
            snapshot=snapshot.capture,
        )

    @staticmethod
    def error(
        request: UiWindowSnapshotRequest,
        error: AgentError,
        *,
        summary: UiWindowSummary | None = None,
    ) -> UiWindowSnapshotResult:
        return UiWindowSnapshotResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            captured=False,
            summary=summary,
            errors=(error,),
        )


class UiWindowProjectionService(WindowCatalogProjectionABC):
    """Project the main-window/WindowManager window graph into agent DTOs."""

    projection_id = MAIN_WINDOW_PROVIDER_ID
    identity = UiWindowProviderIdentity(
        provider_id=MAIN_WINDOW_PROVIDER_ID,
        title="Main window windows",
    )

    def __init__(self, main_window: "OpenHCSMainWindow") -> None:
        self._embedded = EmbeddedWindowProjection(main_window)
        self._managed = ManagedWindowProjection(main_window)
        self._dynamic = DynamicScopeWindowProjection()
        self._snapshot_results = UiWindowSnapshotResultFactory()

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
        if identity.window_id in WindowManager.get_open_scopes():
            return True
        return ScopeWindowRegistry.find_handler(identity.window_id) is not None

    def focus(self, request: UiWindowFocusRequest) -> UiWindowFocusResult:
        identity = request.as_window_identity()
        route_index = self._route_index()
        embedded_route = route_index.embedded_route(identity)
        if embedded_route is not None:
            return self._focused_result(request, embedded_route.focus())

        managed_route = route_index.managed_route(identity)
        if managed_route is not None and request.open_policy.create_if_missing:
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

    def navigate(
        self,
        request: UiWindowNavigateRequest,
    ) -> UiWindowNavigateResult:
        identity = request.as_window_identity()
        route_index = self._route_index()
        embedded_route = route_index.embedded_route(identity)
        if embedded_route is not None:
            return self._navigate_result(
                request,
                focused=True,
                created=False,
                navigated=False,
                summary=embedded_route.focus(),
            )

        managed_route = route_index.managed_route(identity)
        if managed_route is not None and request.open_policy.create_if_missing:
            return self._navigate_result(
                request,
                focused=True,
                created=False,
                navigated=False,
                summary=managed_route.focus(),
            )

        result = ScopeWindowNavigationService.navigate(
            WindowNavigationRequest(
                scope_id=identity.window_id,
                item_id=request.item_id,
                field_path=request.field_path,
                create_if_missing=request.open_policy.create_if_missing,
            )
        )
        if result.focused:
            return self._navigate_result(
                request,
                focused=True,
                created=result.created,
                navigated=result.navigated,
                summary=self._dynamic.summary(identity),
            )

        return UiWindowNavigateResult(
            schema_version=SCHEMA_VERSION,
            window_id=identity.window_id,
            focused=False,
            navigated=False,
            created=result.created,
            errors=(
                AgentError(
                    code="unknown_ui_window",
                    message=f"Unknown or closed UI window: {identity.window_id!r}",
                ),
            ),
        )

    def close(self, request: UiWindowCloseRequest) -> UiWindowCloseResult:
        identity = request.as_window_identity()
        route_index = self._route_index()
        embedded_route = route_index.embedded_route(identity)
        if embedded_route is not None:
            return WindowCloseResultBoundaryPolicy.error(
                request,
                "ui_window_close_unsupported",
                "Embedded main-window panes cannot be closed through this UI bridge operation.",
                summary=embedded_route.summary(),
            )

        managed_route = route_index.managed_route(identity)
        if managed_route is not None:
            return self._close_window_manager_scope(
                request,
                managed_route.summary(),
            )

        scope = UiWindowManagerScope.from_identity(identity)
        scope_widget = WindowManager.get_window(scope.value)
        if scope_widget is not None:
            return self._close_window_manager_scope(
                request,
                self._dynamic.summary(identity),
            )

        if ScopeWindowRegistry.find_handler(identity.window_id) is not None:
            return WindowCloseResultBoundaryPolicy.error(
                request,
                "ui_window_not_open",
                f"UI window scope is not currently open: {identity.window_id!r}",
            )

        return WindowCloseResultBoundaryPolicy.error(
            request,
            "unknown_ui_window",
            f"Unknown or closed UI window: {identity.window_id!r}",
        )

    def snapshot(self, request: UiWindowSnapshotRequest) -> UiWindowSnapshotResult:
        identity = request.as_window_identity()
        target = self._target(
            identity,
            create_if_missing=request.open_policy.create_if_missing,
        )
        if target is None:
            return self._snapshot_results.error(
                request,
                AgentError(
                    code="unknown_ui_window",
                    message=f"Unknown or closed UI window: {identity.window_id!r}",
                ),
            )
        return self._snapshot_results.capture(request, target)

    def _target(
        self,
        identity: UiWindowIdentity,
        *,
        create_if_missing: bool,
    ) -> WindowProjectionTarget | None:
        route_index = self._route_index()
        embedded_route = route_index.embedded_route(identity)
        if embedded_route is not None:
            return WindowProjectionTarget(
                widget=embedded_route.widget(),
                summary=embedded_route.summary(),
            )

        managed_route = route_index.managed_route(identity)
        if managed_route is not None:
            managed_widget = managed_route.widget(create_if_missing=create_if_missing)
            if managed_widget is not None:
                return WindowProjectionTarget(
                    widget=managed_widget,
                    summary=managed_route.summary(),
                )
            return None

        scope = UiWindowManagerScope.from_identity(identity)
        scope_widget = WindowManager.get_window(scope.value)
        if scope_widget is not None:
            return WindowProjectionTarget(
                widget=scope_widget,
                summary=self._dynamic.summary(identity),
            )

        return None

    @staticmethod
    def _close_window_manager_scope(
        request: UiWindowCloseRequest,
        summary: UiWindowSummary,
    ) -> UiWindowCloseResult:
        scope = UiWindowManagerScope.from_identity(request.as_window_identity())
        if WindowManager.close_window(scope.value):
            return WindowCloseResultBoundaryPolicy.closed(
                request,
                summary,
            )
        return WindowCloseResultBoundaryPolicy.error(
            request,
            "ui_window_close_rejected",
            f"WindowManager rejected the close request for scope: {scope.value!r}",
            summary=summary,
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

    @staticmethod
    def _navigate_result(
        request: UiWindowNavigateRequest,
        *,
        focused: bool,
        created: bool,
        navigated: bool,
        summary: UiWindowSummary,
    ) -> UiWindowNavigateResult:
        return UiWindowNavigateResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            focused=focused,
            created=created,
            navigated=navigated,
            summary=summary,
        )


class ManagedWindowActionProvider(UiActionProviderABC):
    """Action provider for generic WindowManager-managed form windows."""

    identity = MANAGED_WINDOW_ACTION_PROVIDER_IDENTITY

    def catalog(self) -> UiActionCatalog:
        return UiActionCatalog(
            schema_version=SCHEMA_VERSION,
            actions=tuple(self.summary(spec.action.value) for spec in MANAGED_WINDOW_ACTION_SPECS),
        )

    def summary(self, action_id: str) -> UiActionSummary:
        spec = self._spec(action_id)
        target_scope_ids = self._target_scope_ids(spec)
        return UiActionSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=spec.action.value,
            ),
            title=spec.title,
            enabled=bool(target_scope_ids),
            invocation_mode="sync",
            side_effects=spec.side_effects,
            confirmation_required=True,
            selection_mode="targeted",
            current_selection_count=len(target_scope_ids),
            target_scope_ids=target_scope_ids,
        )

    def invoke(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
        try:
            spec = self._spec(request.action_id)
            target_scope_id = self._single_target_scope_id(request)
            window = self._target_window(target_scope_id, spec)
        except Exception as exc:
            return self._invoke_error(
                request,
                AgentError.from_exception("managed_window_action_rejected", exc),
            )

        if request.confirmation_is_required():
            return self._invoke_error(
                request,
                AgentError(
                    code="confirmation_required",
                    message=(
                        "Managed-window actions save, discard, or close UI state; "
                        "set require_confirmation=False to dispatch one."
                    ),
                ),
            )

        try:
            spec.dispatch(window)
        except Exception as exc:
            return self._invoke_error(
                request,
                AgentError.from_exception("managed_window_action_failed", exc),
            )

        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=spec.action.value,
            ),
            status=UiActionInvocationStatus.ACCEPTED.value,
            receipt=UiMutationReceipt(
                request_token=request.request_token,
                accepted=True,
            ),
            target_scope_ids=request.selected_scope_ids,
        )

    @staticmethod
    def _spec(action_id: str) -> ManagedWindowActionSpec:
        action = ManagedWindowAction(action_id)
        return MANAGED_WINDOW_ACTION_SPEC_BY_ACTION[action]

    @staticmethod
    def _single_target_scope_id(request: UiActionInvokeRequest) -> str:
        target_scope_ids = request.selected_scope_ids
        if len(target_scope_ids) != 1:
            raise ValueError(
                "Managed-window actions require exactly one target_scope_ids entry."
            )
        return target_scope_ids[0]

    @staticmethod
    def _target_window(
        scope_id: str,
        spec: ManagedWindowActionSpec,
    ) -> BaseFormDialog:
        window = WindowManager.get_window(scope_id)
        if not isinstance(window, BaseFormDialog):
            raise ValueError(f"Window scope is not a managed form window: {scope_id!r}")
        capabilities = window.managed_window_action_capabilities()
        if not spec.is_supported(capabilities):
            raise ValueError(
                f"Window does not support {spec.action.value!r}: {scope_id!r}"
            )
        return window

    @classmethod
    def _target_scope_ids(
        cls,
        spec: ManagedWindowActionSpec,
    ) -> tuple[str, ...]:
        return tuple(
            scope_id
            for scope_id in WindowManager.get_open_scopes()
            if cls._window_supports_action(WindowManager.get_window(scope_id), spec)
        )

    @staticmethod
    def _window_supports_action(
        window: QWidget | None,
        spec: ManagedWindowActionSpec,
    ) -> bool:
        if not isinstance(window, BaseFormDialog):
            return False
        return spec.is_supported(window.managed_window_action_capabilities())

    def _invoke_error(
        self,
        request: UiActionInvokeRequest,
        error: AgentError,
    ) -> UiActionInvokeResult:
        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=request.widget_id,
                action_id=request.action_id,
            ),
            status=UiActionInvocationStatus.REJECTED.value,
            receipt=UiMutationReceipt(
                request_token=request.request_token,
                accepted=False,
            ),
            target_scope_ids=request.selected_scope_ids,
            errors=(error,),
        )


@dataclass(frozen=True, slots=True)
class MainWindowBridgeProviderSet(UiBridgeProviderSetABC):
    """Provider set for main-window generic UI projections."""

    main_window: "OpenHCSMainWindow"
    registry_key = MAIN_WINDOW_PROVIDER_ID

    def register(self, context: UiBridgeRegistrationContext) -> None:
        for provider_type in WindowCatalogProjectionABC.registered_types():
            context.registry.register_window_provider(
                provider_type.create(self.main_window)
            )
        context.registry.register_action_provider(ManagedWindowActionProvider())
