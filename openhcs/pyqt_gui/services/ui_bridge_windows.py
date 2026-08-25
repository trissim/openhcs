"""Window projection providers for the PyQt UI bridge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta
from objectstate import ObjectState
from PyQt6.QtCore import QItemSelectionModel, QModelIndex, Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QMessageBox,
    QWidget,
)
from pyqt_reactive.forms.parameter_form_constants import (
    CONSTANTS as PARAMETER_FORM_CONSTANTS,
)
from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager
from pyqt_reactive.services.scope_window_factory import ScopeWindowRegistry
from pyqt_reactive.services.scope_window_navigation import ScopeWindowNavigationService
from pyqt_reactive.services.widget_tree_projection import (
    DEFAULT_WIDGET_DESCRIPTOR_PROJECTOR_REGISTRY,
    ROOT_WIDGET_PATH_ID,
    WIDGET_PATH_SEPARATOR,
    WidgetActionKind,
    WidgetActionTargetInvalidError,
    WidgetActionUnsupportedError,
    WidgetDescriptor,
    WidgetRect,
    WidgetTreeProjection,
    WidgetTreeProjectionService,
)
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.services.window_navigation import WindowNavigationRequest
from pyqt_reactive.services.window_snapshot import (
    QtWindowSnapshotRequest,
    QtWindowSnapshotService,
)
from pyqt_reactive.widgets.shared import (
    BaseFormDialog,
    BaseManagedWindow,
    ManagedWindowAction,
    ManagedWindowActionCapabilities,
)
from pyqt_reactive.widgets.shared.abstract_manager_widget import AbstractManagerWidget
from pyqt_reactive.widgets.shared.list_item_delegate import (
    DIRTY_FIELDS_ROLE,
    OBJECT_STATE_PATH_ROLE,
    SIG_DIFF_FIELDS_ROLE,
)
from python_introspect import overlay_non_none_dataclass, project_dataclass

from openhcs.agent.dto.common import SCHEMA_VERSION, AgentError, AgentResourceRef
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionIdentity,
    UiActionInvocationStatus,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiActionSummary,
    UiLiveOverviewItem,
    UiLiveOverviewMetric,
    UiLiveOverviewSection,
    UiLiveOverviewSeverity,
    UiMutationReceipt,
    UiWidgetActionInvokeRequest,
    UiWidgetActionInvokeResult,
    UiWidgetActionIssueCode,
    UiWidgetActionSemanticCarrier,
    UiWidgetActionSummary,
    UiWidgetRect,
    UiWidgetTreeNode,
    UiWidgetTreeRequest,
    UiWidgetTreeResult,
    UiWindowCatalog,
    UiWindowCloseRequest,
    UiWindowCloseResult,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowIdentity,
    UiWindowManagerScope,
    UiWindowNavigateRequest,
    UiWindowNavigateResult,
    UiWindowOperationRequest,
    UiWindowSemanticMarker,
    UiWindowSemanticCarrier,
    UiWindowSnapshotRequest,
    UiWindowSnapshotResult,
    UiWindowSummary,
)
from openhcs.agent.ui_bridge_actions import MainWindowAction
from openhcs.agent.ui_bridge_identities import (
    MainWindowWidgetIdentity,
    ManagedWindowWidgetIdentity,
)
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiActionProviderABC,
    UiActionProviderIdentity,
    UiLiveOverviewWidget,
    UiWindowProviderABC,
    UiWindowProviderIdentity,
)
from openhcs.pyqt_gui.services.ui_bridge_object_state import (
    ObjectStateFieldSemanticProjection,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId

if TYPE_CHECKING:
    from openhcs.pyqt_gui.main import OpenHCSMainWindow
    from openhcs.pyqt_gui.services.main_window_workflows import MainWindowDockPane


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
MANAGED_WINDOW_ACTIONS_TITLE = "Managed window actions"
MAIN_WINDOW_ACTIONS_TITLE = "Main window actions"
FIELD_INPUT_ACTION_ROLE = "field_input"
FIELD_RESET_ACTION_ROLE = "field_reset"
ITEM_SELECT_ACTION_ROLE = "item_select"


def _agent_object_state_scope_id(scope_id: str | None) -> str | None:
    if scope_id is None:
        return None
    return OpenHCSUiWindowId.agent_window_id_for_manager_scope(scope_id)


MANAGED_WINDOW_ACTION_PROVIDER_IDENTITY = (
    UiActionProviderIdentity.from_widget_declaration(
        ManagedWindowWidgetIdentity,
        title=MANAGED_WINDOW_ACTIONS_TITLE,
    )
)
MAIN_WINDOW_ACTION_PROVIDER_IDENTITY = UiActionProviderIdentity.from_widget_declaration(
    MainWindowWidgetIdentity,
    title=MAIN_WINDOW_ACTIONS_TITLE,
)


class ManagedWindowSummaryProjection:
    """Project ObjectState-backed managed-window status into window summaries."""

    @classmethod
    def project(cls, window: QWidget | None) -> UiWindowSemanticCarrier:
        if not isinstance(window, BaseManagedWindow):
            return UiWindowSemanticCarrier()
        state = window.state
        capabilities = window.managed_window_action_capabilities()
        return UiWindowSemanticCarrier(
            object_state_scope_id=cls._object_state_scope_id(state),
            dirty=bool(state.dirty_fields) if state is not None else False,
            signature_diff=(
                bool(state.signature_diff_fields) if state is not None else False
            ),
            dirty_field_count=(len(state.dirty_fields) if state is not None else 0),
            signature_diff_field_count=(
                len(state.signature_diff_fields) if state is not None else 0
            ),
            semantic_markers=cls._semantic_markers(state),
            managed_action_ids=cls._managed_action_ids(capabilities),
        )

    @staticmethod
    def _object_state_scope_id(state: ObjectState | None) -> str | None:
        if state is None:
            return None
        return _agent_object_state_scope_id(state.scope_id)

    @staticmethod
    def _semantic_markers(state: ObjectState | None) -> tuple[str, ...]:
        if state is None:
            return ()
        markers = []
        if state.dirty_fields:
            markers.append("*")
        if state.signature_diff_fields:
            markers.append("_")
        return tuple(markers)

    @staticmethod
    def _managed_action_ids(
        capabilities: ManagedWindowActionCapabilities,
    ) -> tuple[str, ...]:
        return tuple(
            action.value
            for action in ManagedWindowAction
            if action.is_supported(capabilities)
        )


class EmbeddedManagerSummaryProjection:
    """Project shared AbstractManagerWidget row semantics into window summaries."""

    @classmethod
    def project(cls, widget: QWidget | None) -> UiWindowSemanticCarrier:
        if not isinstance(widget, AbstractManagerWidget):
            return UiWindowSemanticCarrier()
        item_list = widget.item_list
        if item_list is None:
            return UiWindowSemanticCarrier()
        model = item_list.model()
        if model is None:
            return UiWindowSemanticCarrier()

        dirty_field_count = 0
        signature_diff_field_count = 0
        for row_index in range(model.rowCount()):
            item_record = _WidgetItemSemanticRecord.from_index(
                model.index(row_index, 0)
            )
            dirty_field_count += len(item_record.dirty_fields)
            signature_diff_field_count += len(item_record.signature_diff_fields)

        return UiWindowSemanticCarrier(
            dirty=bool(dirty_field_count),
            signature_diff=bool(signature_diff_field_count),
            dirty_field_count=dirty_field_count,
            signature_diff_field_count=signature_diff_field_count,
            semantic_markers=cls._semantic_markers(
                dirty_field_count=dirty_field_count,
                signature_diff_field_count=signature_diff_field_count,
            ),
        )

    @staticmethod
    def _semantic_markers(
        *,
        dirty_field_count: int,
        signature_diff_field_count: int,
    ) -> tuple[str, ...]:
        markers = []
        if dirty_field_count:
            markers.append("*")
        if signature_diff_field_count:
            markers.append("_")
        return tuple(markers)


@dataclass(frozen=True, slots=True)
class WindowProjectionTarget:
    """Resolved Qt widget plus its agent-facing summary."""

    widget: QWidget
    summary: UiWindowSummary


class WindowProjectionResultAuthority:
    """Shared agent-result fragments for UI window projection operations."""

    @staticmethod
    def unknown_window(identity: UiWindowIdentity) -> AgentError:
        return AgentError(
            code="unknown_ui_window",
            message=f"Unknown or closed UI window: {identity.window_id!r}",
        )


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

    def overview_sections(self) -> tuple[UiLiveOverviewSection, ...]:
        windows = self.summaries()
        error_dialogs = tuple(
            window
            for window in windows
            if UiWindowSemanticMarker.ERROR_DIALOG.value in window.semantic_markers
        )
        return (
            UiLiveOverviewSection(
                section_id=self.identity.provider_id,
                title=self.identity.title,
                summary=f"{len(windows)} windows",
                metrics=(
                    UiLiveOverviewMetric(
                        key="windows",
                        label="windows",
                        value=str(len(windows)),
                    ),
                    UiLiveOverviewMetric(
                        key="visible",
                        label="visible",
                        value=str(sum(1 for window in windows if window.visible)),
                    ),
                    UiLiveOverviewMetric(
                        key="error_dialogs",
                        label="error dialogs",
                        value=str(len(error_dialogs)),
                    ),
                ),
                items=tuple(self._overview_window_item(window) for window in windows),
            ),
        )

    @staticmethod
    def _overview_window_item(window: UiWindowSummary) -> UiLiveOverviewItem:
        is_error_dialog = (
            UiWindowSemanticMarker.ERROR_DIALOG.value in window.semantic_markers
        )
        return UiLiveOverviewItem(
            label=window.title,
            status=window.window_kind,
            detail=f"visible={window.visible} focusable={window.focusable}",
            severity=(
                UiLiveOverviewSeverity.ERROR.value
                if is_error_dialog
                else UiLiveOverviewSeverity.INFO.value
            ),
            source_window_id=window.window_id,
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

    @abstractmethod
    def widget_tree(self, request: UiWidgetTreeRequest) -> UiWidgetTreeResult:
        raise NotImplementedError

    @abstractmethod
    def invoke_widget_action(
        self,
        request: UiWidgetActionInvokeRequest,
    ) -> UiWidgetActionInvokeResult:
        raise NotImplementedError


class WindowTargetOperationProjectionMixin(ABC):
    """Shared target-backed window operations for projection providers."""

    @abstractmethod
    def target_for_operation(
        self,
        request: UiWindowOperationRequest,
    ) -> WindowProjectionTarget | None:
        raise NotImplementedError

    def snapshot(self, request: UiWindowSnapshotRequest) -> UiWindowSnapshotResult:
        target = self.target_for_operation(request)
        snapshot_results = UiWindowSnapshotResultFactory()
        if target is None:
            return snapshot_results.error(
                request,
                WindowProjectionResultAuthority.unknown_window(request),
            )
        return snapshot_results.capture(request, target)

    def widget_tree(self, request: UiWidgetTreeRequest) -> UiWidgetTreeResult:
        target = self.target_for_operation(request)
        widget_tree_results = UiWidgetTreeResultFactory()
        if target is None:
            return widget_tree_results.error(
                request,
                WindowProjectionResultAuthority.unknown_window(request),
            )
        return widget_tree_results.project(request, target)

    def invoke_widget_action(
        self,
        request: UiWidgetActionInvokeRequest,
    ) -> UiWidgetActionInvokeResult:
        target = self.target_for_operation(request)
        invoke_results = UiWidgetActionInvokeResultFactory()
        if target is None:
            return invoke_results.error(
                request,
                WindowProjectionResultAuthority.unknown_window(request),
            )
        return invoke_results.invoke(request, target)


@dataclass(frozen=True, slots=True)
class FocusableWindowRouteMixin(ABC):
    """Shared focus algorithm for static focusable window routes."""

    focus_action: Callable[[], None]

    @abstractmethod
    def summary(self) -> UiWindowSummary:
        raise NotImplementedError

    @abstractmethod
    def target(self, *, create_if_missing: bool) -> WindowProjectionTarget | None:
        raise NotImplementedError

    def focus(self) -> UiWindowSummary:
        self.focus_action()
        return self.summary()


@dataclass(frozen=True, slots=True)
class EmbeddedWindowRoute:
    """Focusable embedded widget route owned by the main window layout."""

    pane: "MainWindowDockPane"

    @property
    def identity(self) -> UiWindowIdentity:
        return UiWindowIdentity(window_id=self.pane.window_id)

    @property
    def title(self) -> str:
        return self.pane.title

    def widget(self) -> QWidget:
        return self.pane.widget

    def focus(self) -> UiWindowSummary:
        self.pane.show()
        return self.summary()

    def overview_sections(self) -> tuple[UiLiveOverviewSection, ...]:
        widget = self.widget()
        if isinstance(widget, UiLiveOverviewWidget):
            return widget.overview_sections()
        return ()

    def target(self, *, create_if_missing: bool) -> WindowProjectionTarget | None:
        del create_if_missing
        return WindowProjectionTarget(
            widget=self.widget(),
            summary=self.summary(),
        )

    def summary(self) -> UiWindowSummary:
        widget = self.widget()
        scope = UiWindowManagerScope.from_identity(self.identity)
        return project_dataclass(
            UiWindowSummary,
            EmbeddedManagerSummaryProjection.project(widget),
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

    def target(self, *, create_if_missing: bool) -> WindowProjectionTarget | None:
        widget = self.widget(create_if_missing=create_if_missing)
        if widget is None:
            return None
        return WindowProjectionTarget(
            widget=widget,
            summary=self.summary(),
        )

    def summary(self) -> UiWindowSummary:
        scope = UiWindowManagerScope.from_identity(self.identity)
        window = WindowManager.get_window(scope.value)
        visible = False
        if window is not None:
            visible = window.isVisible()
        return project_dataclass(
            UiWindowSummary,
            ManagedWindowSummaryProjection.project(window),
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
        return WindowIdentitySet.from_routes(self.embedded_routes).union(
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


@dataclass(frozen=True, slots=True)
class WindowRouteResolution:
    """A static window route resolved for a requested agent-visible identity."""

    request_identity: UiWindowIdentity
    route_identity: UiWindowIdentity
    route: EmbeddedWindowRoute | ManagedWindowRoute

    def summary(self, target_summary: UiWindowSummary | None = None) -> UiWindowSummary:
        if target_summary is None:
            target_summary = self.route.summary()
        if self.route_identity == self.request_identity:
            return target_summary
        return replace(target_summary, identity=self.request_identity)

    def focus(self) -> UiWindowSummary:
        return self.summary(self.route.focus())

    def target(self, *, create_if_missing: bool) -> WindowProjectionTarget | None:
        target = self.route.target(create_if_missing=create_if_missing)
        if target is None:
            return None
        return WindowProjectionTarget(
            widget=target.widget,
            summary=self.summary(target.summary),
        )


class EmbeddedWindowProjection:
    """Project embedded main-window widgets."""

    def __init__(self, main_window: "OpenHCSMainWindow") -> None:
        self._main_window = main_window

    def routes(self) -> tuple[EmbeddedWindowRoute, ...]:
        embedded = self._main_window.embedded_widgets
        return tuple(EmbeddedWindowRoute(pane=pane) for pane in embedded.panes())


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
            self.summary(
                self._identity_for_manager_scope(scope_id),
                scope_id,
            )
            for scope_id in WindowManager.get_open_scopes()
            if not route_index.static_identities.contains(
                self._identity_for_manager_scope(scope_id)
            )
        )

    def summary(
        self,
        identity: UiWindowIdentity,
        manager_scope_id: str | None = None,
    ) -> UiWindowSummary:
        manager_identity = identity
        if manager_scope_id is not None:
            manager_identity = UiWindowIdentity(window_id=manager_scope_id)
        scope = UiWindowManagerScope.from_identity(manager_identity)
        window = WindowManager.get_window(scope.value)
        title = scope.value
        visible = False
        if window is not None:
            title = window.windowTitle()
            visible = window.isVisible()
        return project_dataclass(
            UiWindowSummary,
            ManagedWindowSummaryProjection.project(window),
            schema_version=SCHEMA_VERSION,
            identity=identity,
            title=title,
            window_kind=DYNAMIC_SCOPE_WINDOW_KIND,
            visible=visible,
            focusable=window is not None,
            manager_scope=scope,
        )

    @staticmethod
    def _identity_for_manager_scope(scope_id: str) -> UiWindowIdentity:
        return UiWindowIdentity(
            window_id=OpenHCSUiWindowId.agent_window_id_for_manager_scope(scope_id)
        )


@dataclass(frozen=True, slots=True)
class OpenWindowManagerScope:
    """Witness for an agent identity resolved to an open WindowManager scope."""

    identity: UiWindowIdentity
    scope_id: str
    widget: QWidget

    def target(
        self,
        dynamic_projection: DynamicScopeWindowProjection,
    ) -> WindowProjectionTarget:
        return WindowProjectionTarget(
            widget=self.widget,
            summary=dynamic_projection.summary(self.identity, self.scope_id),
        )


class ScopeWindowTargetOperationProjectionABC(
    WindowTargetOperationProjectionMixin,
    ABC,
):
    """Shared target lookup for WindowManager and scope-registry projections."""

    @abstractmethod
    def _route_index(self) -> WindowRouteIndex:
        raise NotImplementedError

    @abstractmethod
    def _dynamic_scope_projection(self) -> DynamicScopeWindowProjection:
        raise NotImplementedError

    def target_for_operation(
        self,
        request: UiWindowOperationRequest,
    ) -> WindowProjectionTarget | None:
        identity = request.as_identity()
        return self._target(
            identity,
            create_if_missing=request.open_policy.create_if_missing,
        )

    def _target(
        self,
        identity: UiWindowIdentity,
        *,
        create_if_missing: bool,
    ) -> WindowProjectionTarget | None:
        route_index = self._route_index()
        embedded_route = self._embedded_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=False,
        )
        if embedded_route is not None:
            return embedded_route.target(create_if_missing=create_if_missing)

        managed_route = self._managed_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=False,
        )
        if managed_route is not None:
            return managed_route.target(create_if_missing=create_if_missing)

        open_target = self._open_dynamic_target(identity)
        if open_target is not None:
            return open_target

        embedded_alias = self._embedded_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=True,
        )
        if embedded_alias is not None:
            return embedded_alias.target(create_if_missing=create_if_missing)

        managed_alias = self._managed_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=True,
        )
        if managed_alias is not None:
            return managed_alias.target(create_if_missing=create_if_missing)

        scope = UiWindowManagerScope.from_identity(
            self._resolved_window_identity(identity)
        )
        scope_widget = WindowManager.get_window(scope.value)
        if scope_widget is not None:
            return WindowProjectionTarget(
                widget=scope_widget,
                summary=self._dynamic_scope_projection().summary(
                    identity,
                    scope.value,
                ),
            )

        if (
            create_if_missing
            and ScopeWindowRegistry.find_handler(identity.window_id) is not None
        ):
            navigation = ScopeWindowNavigationService.navigate(
                WindowNavigationRequest(
                    scope_id=identity.window_id,
                    create_if_missing=True,
                )
            )
            if navigation.focused:
                target_scope_id = navigation.window_scope_id or scope.value
                scope_widget = WindowManager.get_window(target_scope_id)
                if scope_widget is not None:
                    return WindowProjectionTarget(
                        widget=scope_widget,
                        summary=self._dynamic_scope_projection().summary(
                            identity,
                            target_scope_id,
                        ),
                    )

        return None

    def _embedded_route_resolution(
        self,
        identity: UiWindowIdentity,
        route_index: WindowRouteIndex,
        *,
        resolve_scope_alias: bool,
    ) -> WindowRouteResolution | None:
        return self._static_route_resolution(
            identity,
            route_index.embedded_route,
            resolve_scope_alias=resolve_scope_alias,
        )

    def _managed_route_resolution(
        self,
        identity: UiWindowIdentity,
        route_index: WindowRouteIndex,
        *,
        resolve_scope_alias: bool,
    ) -> WindowRouteResolution | None:
        return self._static_route_resolution(
            identity,
            route_index.managed_route,
            resolve_scope_alias=resolve_scope_alias,
        )

    def _static_route_resolution(
        self,
        identity: UiWindowIdentity,
        route_for_identity: Callable[
            [UiWindowIdentity],
            EmbeddedWindowRoute | ManagedWindowRoute | None,
        ],
        *,
        resolve_scope_alias: bool,
    ) -> WindowRouteResolution | None:
        route_identity = identity
        if resolve_scope_alias:
            route_identity = self._resolved_window_identity(identity)
            if route_identity == identity:
                return None

        route = route_for_identity(route_identity)
        if route is None:
            return None
        return WindowRouteResolution(
            request_identity=identity,
            route_identity=route_identity,
            route=route,
        )

    def _open_dynamic_target(
        self,
        identity: UiWindowIdentity,
    ) -> WindowProjectionTarget | None:
        open_scope = self._open_window_manager_scope(identity)
        if open_scope is None:
            return None
        return open_scope.target(self._dynamic_scope_projection())

    @classmethod
    def _open_window_manager_scope_id(cls, identity: UiWindowIdentity) -> str | None:
        open_scope = cls._open_window_manager_scope(identity)
        if open_scope is None:
            return None
        return open_scope.scope_id

    @staticmethod
    def _open_window_manager_scope(
        identity: UiWindowIdentity,
    ) -> OpenWindowManagerScope | None:
        for scope_id in OpenHCSUiWindowId.manager_scopes_for_agent_window_id(
            identity.window_id
        ):
            widget = WindowManager.get_window(scope_id)
            if widget is not None:
                return OpenWindowManagerScope(
                    identity=identity,
                    scope_id=scope_id,
                    widget=widget,
                )
        return None

    @staticmethod
    def _resolved_window_identity(identity: UiWindowIdentity) -> UiWindowIdentity:
        route = ScopeWindowRegistry.find_handler(identity.window_id)
        if route is None:
            return identity
        target = route.navigation_target(identity.window_id)
        return UiWindowIdentity(window_id=target.window_scope_id)


class QtTopLevelWindowProjection(
    WindowTargetOperationProjectionMixin,
    WindowCatalogProjectionABC,
):
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

    def summaries(self) -> tuple[UiWindowSummary, ...]:
        return tuple(
            self.summary(widget) for widget in self._projected_top_level_widgets()
        )

    def handles(self, window_id: str) -> bool:
        return self.target(UiWindowIdentity(window_id=window_id)) is not None

    def target(self, identity: UiWindowIdentity) -> WindowProjectionTarget | None:
        identity = identity.as_identity()
        for widget in self._projected_top_level_widgets():
            if self._identity(widget) == identity:
                return WindowProjectionTarget(
                    widget=widget, summary=self.summary(widget)
                )
        return None

    def target_for_operation(
        self,
        request: UiWindowOperationRequest,
    ) -> WindowProjectionTarget | None:
        return self.target(request)

    def focus(self, request: UiWindowFocusRequest) -> UiWindowFocusResult:
        target = self.target(request)
        if target is None:
            return UiWindowFocusResult(
                schema_version=SCHEMA_VERSION,
                window_id=request.window_id,
                focused=False,
                errors=(WindowProjectionResultAuthority.unknown_window(request),),
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
        target = self.target(request)
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

    def summary(self, widget: QWidget) -> UiWindowSummary:
        identity = self._identity(widget)
        return UiWindowSummary(
            schema_version=SCHEMA_VERSION,
            identity=identity,
            title=self._title(widget),
            window_kind=QT_TOP_LEVEL_WINDOW_KIND,
            visible=widget.isVisible(),
            focusable=True,
            manager_scope=None,
            semantic_markers=self._semantic_markers(widget),
        )

    def _projected_top_level_widgets(self) -> tuple[QWidget, ...]:
        excluded_widgets = self._excluded_widgets()
        return tuple(
            widget
            for widget in WindowManager.visible_top_level_windows()
            if widget not in excluded_widgets
        )

    def _excluded_widgets(self) -> frozenset[QWidget]:
        widgets: set[QWidget] = set()
        for scope_id in WindowManager.get_open_scopes():
            widget = WindowManager.get_window(scope_id)
            if widget is not None:
                top_level_widget = widget.window()
                if top_level_widget is not self._main_window:
                    widgets.add(top_level_widget)
        if self._main_window is not None:
            widgets.update(
                pane.dock_widget
                for pane in self._main_window.embedded_widgets.panes()
                if pane.dock_widget.isFloating()
            )
        return frozenset(widgets)

    def _identity(self, widget: QWidget) -> UiWindowIdentity:
        if widget is self._main_window:
            return UiWindowIdentity(window_id=OpenHCSUiWindowId.main_window)
        return UiWindowIdentity(
            window_id=f"{QT_TOP_LEVEL_WINDOW_ID_PREFIX}{int(widget.winId())}"
        )

    @staticmethod
    def _title(widget: QWidget) -> str:
        title = widget.windowTitle()
        if title:
            return title
        return type(widget).__name__

    @staticmethod
    def _semantic_markers(widget: QWidget) -> tuple[str, ...]:
        if (
            isinstance(widget, QMessageBox)
            and widget.icon() is QMessageBox.Icon.Critical
        ):
            return (UiWindowSemanticMarker.ERROR_DIALOG.value,)
        return ()


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
                    capture=request,
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
            output_dir_path=request.output_dir_path,
            capture_scope=request.capture_scope,
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
            output_dir_path=request.output_dir_path,
            capture_scope=request.capture_scope,
            captured=False,
            summary=summary,
            errors=(error,),
        )


@dataclass(frozen=True, slots=True)
class _ItemViewIndexResolution:
    view: QAbstractItemView
    index: QModelIndex


class UiWidgetPathResolver:
    """Resolve widget-tree path ids against live Qt widgets and descriptors."""

    @classmethod
    def path_from_id(cls, path_id: str) -> tuple[int, ...]:
        if path_id == ROOT_WIDGET_PATH_ID:
            return ()
        if not path_id:
            raise ValueError("Widget path_id must not be empty.")
        parts = path_id.split(WIDGET_PATH_SEPARATOR)
        path: list[int] = []
        for part in parts:
            try:
                index = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid widget path component: {part!r}") from exc
            if index < 0:
                raise ValueError(
                    f"Widget path component must be non-negative: {part!r}"
                )
            path.append(index)
        return tuple(path)

    @classmethod
    def widget_at_path(cls, widget: QWidget, path: tuple[int, ...]) -> QWidget | None:
        current = widget
        for child_index in path:
            children = cls.direct_children(current)
            if child_index >= len(children):
                return None
            current = children[child_index]
        return current

    @classmethod
    def descriptor_at_path(
        cls,
        descriptor: WidgetDescriptor,
        path: tuple[int, ...],
    ) -> WidgetDescriptor | None:
        current = descriptor
        for child_index in path:
            if child_index >= len(current.children):
                return None
            current = current.children[child_index]
        return current

    @staticmethod
    def direct_children(widget: QWidget) -> tuple[QWidget, ...]:
        return tuple(
            widget.findChildren(
                QWidget,
                options=Qt.FindChildOption.FindDirectChildrenOnly,
            )
        )

    @classmethod
    def item_view_index_at_path(
        cls,
        widget: QWidget,
        path: tuple[int, ...],
    ) -> "_ItemViewIndexResolution | None":
        current_widget = widget
        current_index = QModelIndex()
        active_view: QAbstractItemView | None = None

        for child_index in path:
            if active_view is None:
                children = cls.direct_children(current_widget)
                if child_index < len(children):
                    current_widget = children[child_index]
                    continue
                if not isinstance(current_widget, QAbstractItemView):
                    return None
                active_view = current_widget
                model_row = child_index - len(children)
            else:
                model_row = child_index

            model = active_view.model()
            if model is None:
                return None
            current_index = model.index(model_row, 0, current_index)
            if not current_index.isValid():
                return None

        if active_view is None:
            return None
        return _ItemViewIndexResolution(active_view, current_index)


class UiWidgetActionInvokeResultFactory:
    """Invoke projected widget actions through live Qt widgets."""

    AUTO_ACTION_KIND = "auto"

    def invoke(
        self,
        request: UiWidgetActionInvokeRequest,
        target: WindowProjectionTarget,
    ) -> UiWidgetActionInvokeResult:
        try:
            projection = WidgetTreeProjectionService.project(target.widget)
            path = UiWidgetPathResolver.path_from_id(request.path_id)
        except Exception as exc:
            return self.error(
                request,
                AgentError.from_exception(
                    UiWidgetActionIssueCode.RESOLUTION_FAILED.value,
                    exc,
                ),
                summary=target.summary,
            )

        descriptor = UiWidgetPathResolver.descriptor_at_path(projection.root, path)
        if descriptor is None:
            return self.error(
                request,
                AgentError(
                    code=UiWidgetActionIssueCode.WIDGET_UNKNOWN.value,
                    message=(
                        f"Widget path_id {request.path_id!r} was not found in "
                        f"window {request.window_id!r}."
                    ),
                    hint="Refresh the widget tree and retry with a current path_id.",
                ),
                summary=target.summary,
            )

        action_summary = UiWidgetTreeResultFactory.action_summary(
            descriptor,
            field_semantics=_WidgetFieldSemanticContext.from_target(target),
            item_semantics=_WidgetItemSemanticContext.from_target(target),
        )

        action_kind = self._resolve_action_kind(request, descriptor)
        if isinstance(action_kind, AgentError):
            return self.error(request, action_kind, summary=action_summary)

        if action_kind == WidgetActionKind.ITEM_SELECT:
            return self._invoke_item_select(
                request,
                target,
                path,
                descriptor,
                action_summary,
                action_kind=action_kind,
            )

        widget = UiWidgetPathResolver.widget_at_path(target.widget, path)
        if widget is None:
            return self.error(
                request,
                AgentError(
                    code=UiWidgetActionIssueCode.WIDGET_UNKNOWN.value,
                    message=(
                        f"Widget path_id {request.path_id!r} was not found in "
                        f"window {request.window_id!r}."
                    ),
                    hint="Refresh the widget tree and retry with a current path_id.",
                ),
                summary=action_summary,
            )

        guard_error = self._guard_error(request, descriptor, widget, action_kind)
        if guard_error is not None:
            return self.error(
                request,
                guard_error,
                summary=action_summary,
                action_kind=action_kind.value,
            )

        return self._invoke_projector_action(
            request,
            action_summary,
            widget,
            action_kind=action_kind,
        )

    def _invoke_projector_action(
        self,
        request: UiWidgetActionInvokeRequest,
        action_summary: UiWidgetActionSummary,
        widget: QWidget,
        *,
        action_kind: WidgetActionKind,
    ) -> UiWidgetActionInvokeResult:
        try:
            DEFAULT_WIDGET_DESCRIPTOR_PROJECTOR_REGISTRY.projector_for(
                widget
            ).invoke_action(
                widget,
                action_kind,
                target_index=request.target_index,
            )
        except WidgetActionUnsupportedError:
            return self.error(
                request,
                AgentError(
                    code=UiWidgetActionIssueCode.ACTION_UNSUPPORTED.value,
                    message=(
                        f"Widget path_id {request.path_id!r} does not support "
                        f"action kind {action_kind.value!r}."
                    ),
                ),
                summary=action_summary,
                action_kind=action_kind.value,
            )
        except WidgetActionTargetInvalidError as error:
            return self.error(
                request,
                AgentError(
                    code=UiWidgetActionIssueCode.INDEX_INVALID.value,
                    message=(
                        f"Target index {request.target_index!r} is outside the "
                        f"available range for widget path_id {request.path_id!r}."
                    ),
                    hint=(
                        "Refresh the widget tree and choose target_index from 0 "
                        f"through {error.item_count - 1}."
                    ),
                ),
                summary=action_summary,
                action_kind=action_kind.value,
            )

        return UiWidgetActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            path_id=request.path_id,
            action_kind=action_kind.value,
            invoked=True,
            receipt=UiMutationReceipt.accepted_for(request.request_token),
            summary=action_summary,
        )

    def _invoke_item_select(
        self,
        request: UiWidgetActionInvokeRequest,
        target: WindowProjectionTarget,
        path: tuple[int, ...],
        descriptor: WidgetDescriptor,
        action_summary: UiWidgetActionSummary,
        *,
        action_kind: WidgetActionKind,
    ) -> UiWidgetActionInvokeResult:
        guard_error = self._descriptor_guard_error(request, descriptor, action_kind)
        if guard_error is not None:
            return self.error(
                request,
                guard_error,
                summary=action_summary,
                action_kind=action_kind.value,
            )

        resolution = UiWidgetPathResolver.item_view_index_at_path(target.widget, path)
        if resolution is None:
            return self.error(
                request,
                AgentError(
                    code=UiWidgetActionIssueCode.ACTION_UNSUPPORTED.value,
                    message=(
                        f"Widget path_id {request.path_id!r} does not resolve to "
                        "a selectable item-view row."
                    ),
                ),
                summary=action_summary,
                action_kind=action_kind.value,
            )

        QTimer.singleShot(
            0,
            lambda: self._select_item_index(resolution.view, resolution.index),
        )
        return UiWidgetActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            path_id=request.path_id,
            action_kind=action_kind.value,
            invoked=True,
            receipt=UiMutationReceipt.accepted_for(request.request_token),
            summary=action_summary,
        )

    @staticmethod
    def _select_item_index(view: QAbstractItemView, index: QModelIndex) -> None:
        view.setCurrentIndex(index)
        selection_model = view.selectionModel()
        if selection_model is not None:
            selection_model.select(
                index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        view.scrollTo(index)

    def _resolve_action_kind(
        self,
        request: UiWidgetActionInvokeRequest,
        descriptor: WidgetDescriptor,
    ) -> WidgetActionKind | AgentError:
        if request.action_kind == self.AUTO_ACTION_KIND:
            if len(descriptor.action_kinds) == 1:
                return next(iter(descriptor.action_kinds))
            return self._action_kind_unavailable_error(request, descriptor)
        try:
            action_kind = WidgetActionKind(request.action_kind)
        except ValueError:
            return self._action_kind_unavailable_error(request, descriptor)
        if action_kind not in descriptor.action_kinds:
            return self._action_kind_unavailable_error(request, descriptor)
        return action_kind

    def _guard_error(
        self,
        request: UiWidgetActionInvokeRequest,
        descriptor: WidgetDescriptor,
        widget: QWidget,
        action_kind: WidgetActionKind,
    ) -> AgentError | None:
        if not descriptor.visible or not widget.isVisible():
            return AgentError(
                code=UiWidgetActionIssueCode.NOT_VISIBLE.value,
                message=f"Widget path_id {request.path_id!r} is not visible.",
            )
        if not descriptor.enabled or not widget.isEnabled():
            return AgentError(
                code=UiWidgetActionIssueCode.DISABLED.value,
                message=f"Widget path_id {request.path_id!r} is disabled.",
            )
        if not descriptor.clickable:
            return AgentError(
                code=UiWidgetActionIssueCode.NOT_CLICKABLE.value,
                message=f"Widget path_id {request.path_id!r} is not clickable.",
            )
        return None

    def _descriptor_guard_error(
        self,
        request: UiWidgetActionInvokeRequest,
        descriptor: WidgetDescriptor,
        action_kind: WidgetActionKind,
    ) -> AgentError | None:
        if not descriptor.visible:
            return AgentError(
                code=UiWidgetActionIssueCode.NOT_VISIBLE.value,
                message=f"Widget path_id {request.path_id!r} is not visible.",
            )
        if not descriptor.enabled:
            return AgentError(
                code=UiWidgetActionIssueCode.DISABLED.value,
                message=f"Widget path_id {request.path_id!r} is disabled.",
            )
        if not descriptor.clickable:
            return AgentError(
                code=UiWidgetActionIssueCode.NOT_CLICKABLE.value,
                message=f"Widget path_id {request.path_id!r} is not clickable.",
            )
        return None

    @staticmethod
    def _action_kind_unavailable_error(
        request: UiWidgetActionInvokeRequest,
        descriptor: WidgetDescriptor,
    ) -> AgentError:
        action_kinds = tuple(kind.value for kind in descriptor.action_kinds)
        return AgentError(
            code=UiWidgetActionIssueCode.ACTION_KIND_UNAVAILABLE.value,
            message=(
                f"Widget path_id {request.path_id!r} does not expose "
                f"action kind {request.action_kind!r}."
            ),
            hint=f"Available action kinds: {', '.join(action_kinds) or 'none'}.",
        )

    @staticmethod
    def error(
        request: UiWidgetActionInvokeRequest,
        error: AgentError,
        *,
        summary: UiWindowSummary | UiWidgetActionSummary | None = None,
        action_kind: str | None = None,
    ) -> UiWidgetActionInvokeResult:
        action_summary = summary if isinstance(summary, UiWidgetActionSummary) else None
        return UiWidgetActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            path_id=request.path_id,
            action_kind=action_kind or request.action_kind,
            invoked=False,
            receipt=UiMutationReceipt.rejected_for(request.request_token),
            summary=action_summary,
            errors=(error,),
        )


@dataclass(frozen=True, slots=True)
class _WidgetGeometryKey:
    """Stable enough bridge key for matching live form widgets to descriptors."""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_qwidget(cls, widget: QWidget) -> "_WidgetGeometryKey":
        rect = widget.rect()
        top_left = widget.mapToGlobal(rect.topLeft())
        return cls(
            x=top_left.x(),
            y=top_left.y(),
            width=rect.width(),
            height=rect.height(),
        )

    @classmethod
    def from_descriptor(cls, descriptor: WidgetDescriptor) -> "_WidgetGeometryKey":
        rect = descriptor.global_geometry
        return cls(
            x=rect.x,
            y=rect.y,
            width=rect.width,
            height=rect.height,
        )


@dataclass(frozen=True, slots=True)
class _WidgetFieldSemanticRecord:
    """Field semantics for one live widget action."""

    action_role: str
    projection: ObjectStateFieldSemanticProjection

    def to_action_carrier(self, *, window_id: str) -> UiWidgetActionSemanticCarrier:
        return project_dataclass(
            UiWidgetActionSemanticCarrier,
            self.projection.to_semantic_carrier(include_values=True),
            action_role=self.action_role,
            semantic_address=self.projection.semantic_address(window_id=window_id),
            object_state_scope_id=self.projection.agent_scope_id,
            field_path=self.projection.field_path,
        )


@dataclass(frozen=True, slots=True)
class _WidgetFieldSemanticContext:
    """Form-manager-backed lookup for widget action ObjectState semantics."""

    window_id: str
    records_by_geometry: dict[_WidgetGeometryKey, _WidgetFieldSemanticRecord]

    @classmethod
    def empty(cls, window_id: str) -> "_WidgetFieldSemanticContext":
        return cls(window_id=window_id, records_by_geometry={})

    @classmethod
    def from_target(
        cls,
        target: WindowProjectionTarget,
    ) -> "_WidgetFieldSemanticContext":
        if not isinstance(target.widget, BaseManagedWindow):
            return cls.empty(target.summary.window_id)

        records: dict[_WidgetGeometryKey, _WidgetFieldSemanticRecord] = {}
        for form_manager in target.widget.form_managers():
            cls._collect_form_manager(records, form_manager)
        return cls(
            window_id=target.summary.window_id,
            records_by_geometry=records,
        )

    @classmethod
    def _collect_form_manager(
        cls,
        records: dict[_WidgetGeometryKey, _WidgetFieldSemanticRecord],
        form_manager: ParameterFormManager,
    ) -> None:
        state = form_manager.state
        if not isinstance(state, ObjectState):
            return

        for field_name, widget in form_manager.widgets.items():
            cls._add_field_widget(
                records,
                form_manager=form_manager,
                state=state,
                field_name=field_name,
                widget=widget,
                action_role=FIELD_INPUT_ACTION_ROLE,
            )
        for field_name, widget in form_manager.reset_buttons.items():
            cls._add_field_widget(
                records,
                form_manager=form_manager,
                state=state,
                field_name=field_name,
                widget=widget,
                action_role=FIELD_RESET_ACTION_ROLE,
            )
        for nested_manager in form_manager.nested_managers.values():
            cls._collect_form_manager(records, nested_manager)

    @classmethod
    def _add_field_widget(
        cls,
        records: dict[_WidgetGeometryKey, _WidgetFieldSemanticRecord],
        *,
        form_manager: ParameterFormManager,
        state: ObjectState,
        field_name: str,
        widget: QWidget,
        action_role: str,
    ) -> None:
        field_path = cls._field_path(form_manager, field_name)
        if field_path not in state.parameters:
            return
        records[_WidgetGeometryKey.from_qwidget(widget)] = _WidgetFieldSemanticRecord(
            action_role=action_role,
            projection=ObjectStateFieldSemanticProjection.from_state(
                state,
                field_path,
            ),
        )

    @staticmethod
    def _field_path(form_manager: ParameterFormManager, field_name: str) -> str:
        if form_manager.field_id:
            return f"{form_manager.field_id}.{field_name}"
        return field_name

    def for_descriptor(
        self,
        descriptor: WidgetDescriptor,
    ) -> _WidgetFieldSemanticRecord | None:
        return self.records_by_geometry.get(
            _WidgetGeometryKey.from_descriptor(descriptor)
        )


@dataclass(frozen=True, slots=True)
class _WidgetItemSemanticRecord:
    """ObjectState semantics already attached to one projected item-view row."""

    agent_scope_id: str | None
    dirty_fields: tuple[str, ...]
    signature_diff_fields: tuple[str, ...]

    @classmethod
    def from_index(cls, index: QModelIndex) -> "_WidgetItemSemanticRecord":
        scope_value = index.data(OBJECT_STATE_PATH_ROLE)
        agent_scope_id = None
        if isinstance(scope_value, str):
            agent_scope_id = _agent_object_state_scope_id(scope_value)
        return cls(
            agent_scope_id=agent_scope_id,
            dirty_fields=cls._field_tuple(index.data(DIRTY_FIELDS_ROLE)),
            signature_diff_fields=cls._field_tuple(index.data(SIG_DIFF_FIELDS_ROLE)),
        )

    @staticmethod
    def _field_tuple(value) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (frozenset, list, set, tuple)):
            return tuple(field for field in value if isinstance(field, str))
        return ()

    @property
    def has_semantics(self) -> bool:
        return bool(
            self.agent_scope_id or self.dirty_fields or self.signature_diff_fields
        )

    @property
    def dirty(self) -> bool:
        return bool(self.dirty_fields)

    @property
    def signature_diff(self) -> bool:
        return bool(self.signature_diff_fields)

    @property
    def semantic_markers(self) -> tuple[str, ...]:
        markers = []
        if self.dirty_fields:
            markers.append("*")
        if self.signature_diff_fields:
            markers.append("_")
        return tuple(markers)

    def to_action_carrier(self) -> UiWidgetActionSemanticCarrier:
        return UiWidgetActionSemanticCarrier(
            action_role=ITEM_SELECT_ACTION_ROLE,
            object_state_scope_id=self.agent_scope_id,
            dirty=self.dirty,
            signature_diff=self.signature_diff,
            semantic_markers=self.semantic_markers,
        )


@dataclass(frozen=True, slots=True)
class _WidgetItemSemanticContext:
    """Resolve shared manager-list item roles for projected item rows."""

    root_widget: QWidget | None

    @classmethod
    def empty(cls) -> "_WidgetItemSemanticContext":
        return cls(root_widget=None)

    @classmethod
    def from_target(
        cls,
        target: WindowProjectionTarget,
    ) -> "_WidgetItemSemanticContext":
        return cls(root_widget=target.widget)

    def for_descriptor(
        self,
        descriptor: WidgetDescriptor,
    ) -> _WidgetItemSemanticRecord | None:
        if self.root_widget is None:
            return None
        resolution = UiWidgetPathResolver.item_view_index_at_path(
            self.root_widget,
            descriptor.path,
        )
        if resolution is None:
            return None
        record = _WidgetItemSemanticRecord.from_index(resolution.index)
        if not record.has_semantics:
            return None
        return record


class UiWidgetTreeResultFactory:
    """Build UI bridge widget-tree results from resolved Qt window targets."""

    def project(
        self,
        request: UiWidgetTreeRequest,
        target: WindowProjectionTarget,
    ) -> UiWidgetTreeResult:
        try:
            projection = WidgetTreeProjectionService.project(
                target.widget,
                policy=request.as_projection_policy(),
            )
        except Exception as exc:
            return self.error(
                request,
                AgentError.from_exception("ui_widget_tree_projection_failed", exc),
                summary=target.summary,
            )
        return self.from_projection(
            request,
            target.summary,
            projection,
            field_semantics=_WidgetFieldSemanticContext.from_target(target),
            item_semantics=_WidgetItemSemanticContext.from_target(target),
        )

    @classmethod
    def from_projection(
        cls,
        request: UiWidgetTreeRequest,
        summary: UiWindowSummary,
        projection: WidgetTreeProjection,
        field_semantics: _WidgetFieldSemanticContext | None = None,
        item_semantics: _WidgetItemSemanticContext | None = None,
    ) -> UiWidgetTreeResult:
        if field_semantics is None:
            field_semantics = _WidgetFieldSemanticContext.empty(summary.window_id)
        if item_semantics is None:
            item_semantics = _WidgetItemSemanticContext.empty()
        tree_state = _WidgetTreeBoundState()
        draft = cls._draft_node(
            projection.root,
            request=request,
            state=tree_state,
            depth=0,
        )
        action_draft = cls._draft_node(
            projection.root,
            request=replace(request, actionable_only=False),
            state=_WidgetTreeBoundState(),
            depth=0,
        )
        action_state = _WidgetTreeActionListState()
        actionable_widgets: tuple[UiWidgetActionSummary, ...] = ()
        actionable_count = 0
        if action_draft is not None:
            actionable_count = cls.included_action_summary_count(action_draft)
            actionable_widgets = tuple(
                cls.action_summaries_from_draft(
                    action_draft,
                    request=request,
                    state=action_state,
                    field_semantics=field_semantics,
                    item_semantics=item_semantics,
                )
            )
        returned_state = _WidgetTreeBoundState()
        root = None
        if request.include_tree and draft is not None:
            root = cls.node_from_draft(
                draft,
                request=request,
                state=returned_state,
            )
        return UiWidgetTreeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            projected=True,
            root=root,
            actionable_widgets=actionable_widgets,
            summary=summary,
            widget_count=projection.widget_count,
            actionable_count=actionable_count,
            returned_widget_count=returned_state.returned_widget_count,
            returned_actionable_count=action_state.returned_actionable_count,
            tree_truncated=tree_state.truncated or returned_state.truncated,
            actionable_widgets_truncated=action_state.truncated,
            actionable_only=request.actionable_only,
            include_tree=request.include_tree,
            max_depth=request.max_depth,
            max_nodes=request.max_nodes,
        )

    @classmethod
    def _draft_node(
        cls,
        descriptor: WidgetDescriptor,
        *,
        request: UiWidgetTreeRequest,
        state: "_WidgetTreeBoundState",
        depth: int,
    ) -> "_WidgetTreeNodeDraft | None":
        children: list[_WidgetTreeNodeDraft] = []
        if request.max_depth is not None and depth >= request.max_depth:
            if descriptor.children:
                state.truncated = True
        else:
            for child in descriptor.children:
                child_draft = cls._draft_node(
                    child,
                    request=request,
                    state=state,
                    depth=depth + 1,
                )
                if child_draft is not None:
                    children.append(child_draft)

        if (
            depth > 0
            and request.actionable_only
            and not descriptor.actionable
            and not children
        ):
            return None
        return _WidgetTreeNodeDraft(descriptor, tuple(children))

    @classmethod
    def node_from_draft(
        cls,
        draft: "_WidgetTreeNodeDraft",
        *,
        request: UiWidgetTreeRequest,
        state: "_WidgetTreeBoundState",
    ) -> UiWidgetTreeNode | None:
        if (
            request.max_nodes is not None
            and state.returned_widget_count >= request.max_nodes
        ):
            state.truncated = True
            return None
        state.returned_widget_count += 1

        children: list[UiWidgetTreeNode] = []
        for child in sorted(
            draft.children,
            key=lambda candidate: not candidate.descriptor.visible,
        ):
            child_node = cls.node_from_draft(child, request=request, state=state)
            if child_node is not None:
                children.append(child_node)
            if (
                request.max_nodes is not None
                and state.returned_widget_count >= request.max_nodes
            ):
                if len(children) < len(draft.children):
                    state.truncated = True
                break
        return cls.node(draft.descriptor, children=tuple(children))

    @classmethod
    def included_action_summary_count(cls, draft: "_WidgetTreeNodeDraft") -> int:
        count = 0
        if draft.descriptor.actionable:
            summary = cls.action_summary(draft.descriptor)
            if UiWidgetActionSummaryPolicy.includes(summary):
                count += 1
        for child in draft.children:
            count += cls.included_action_summary_count(child)
        return count

    @classmethod
    def action_summaries_from_draft(
        cls,
        draft: "_WidgetTreeNodeDraft",
        *,
        request: UiWidgetTreeRequest,
        state: "_WidgetTreeActionListState",
        field_semantics: _WidgetFieldSemanticContext,
        item_semantics: _WidgetItemSemanticContext,
        ancestors: tuple["_WidgetTreeNodeDraft", ...] = (),
    ):
        if (
            draft.descriptor.actionable
            and request.max_nodes is not None
            and state.returned_actionable_count >= request.max_nodes
        ):
            state.truncated = True
            return

        if draft.descriptor.actionable:
            summary = cls.action_summary(
                draft.descriptor,
                context_label=cls.action_context_label(draft, ancestors),
                field_semantics=field_semantics,
                item_semantics=item_semantics,
            )
            if UiWidgetActionSummaryPolicy.includes(summary):
                state.returned_actionable_count += 1
                yield summary

        for child_index, child in enumerate(draft.children):
            yield from cls.action_summaries_from_draft(
                child,
                request=request,
                state=state,
                field_semantics=field_semantics,
                item_semantics=item_semantics,
                ancestors=(*ancestors, draft),
            )
            if (
                request.max_nodes is not None
                and state.returned_actionable_count >= request.max_nodes
            ):
                if child_index < len(draft.children) - 1:
                    state.truncated = True
                break

    @classmethod
    def action_summary(
        cls,
        descriptor: WidgetDescriptor,
        *,
        context_label: str | None = None,
        field_semantics: _WidgetFieldSemanticContext | None = None,
        item_semantics: _WidgetItemSemanticContext | None = None,
    ) -> UiWidgetActionSummary:
        label = cls.action_label(descriptor)
        action_kinds = tuple(kind.value for kind in descriptor.action_kinds)
        summary = project_dataclass(
            UiWidgetActionSummary,
            descriptor,
            label=label,
            geometry=cls.rect(descriptor.geometry),
            global_geometry=cls.rect(descriptor.global_geometry),
            action_kinds=action_kinds,
            context_label=context_label,
        )
        return overlay_non_none_dataclass(
            summary,
            cls.action_semantic_carrier(
                descriptor,
                label=label,
                action_kinds=action_kinds,
                field_semantics=field_semantics,
                item_semantics=item_semantics,
            ),
        )

    @staticmethod
    def action_semantic_carrier(
        descriptor: WidgetDescriptor,
        *,
        label: str | None,
        action_kinds: tuple[str, ...],
        field_semantics: _WidgetFieldSemanticContext | None,
        item_semantics: _WidgetItemSemanticContext | None,
    ) -> UiWidgetActionSemanticCarrier:
        if field_semantics is not None:
            field_record = field_semantics.for_descriptor(descriptor)
            if field_record is not None:
                return field_record.to_action_carrier(
                    window_id=field_semantics.window_id
                )
        if item_semantics is not None:
            item_record = item_semantics.for_descriptor(descriptor)
            if item_record is not None:
                return item_record.to_action_carrier()
        if FieldResetWidgetActionSummary.matches_fields(
            class_name=descriptor.class_name,
            label=label,
            object_name=descriptor.object_name,
            action_kinds=action_kinds,
        ):
            return UiWidgetActionSemanticCarrier(action_role=FIELD_RESET_ACTION_ROLE)
        return UiWidgetActionSemanticCarrier()

    @classmethod
    def action_context_label(
        cls,
        draft: "_WidgetTreeNodeDraft",
        ancestors: tuple["_WidgetTreeNodeDraft", ...],
    ) -> str | None:
        branch = draft
        for ancestor in reversed(ancestors):
            siblings = tuple(
                child
                for child in ancestor.children
                if child.descriptor.path != branch.descriptor.path
            )
            if siblings:
                label = cls._first_context_label(siblings)
                if label:
                    return label
            branch = ancestor
        return None

    @classmethod
    def _first_context_label(
        cls,
        drafts: tuple["_WidgetTreeNodeDraft", ...],
    ) -> str | None:
        for draft in drafts:
            descriptor = draft.descriptor
            if not descriptor.actionable and not descriptor.action_kinds:
                label = cls._descriptor_context_label(descriptor)
                if label:
                    return label
            child_label = cls._first_context_label(draft.children)
            if child_label:
                return child_label
        return None

    @staticmethod
    def _descriptor_context_label(descriptor: WidgetDescriptor) -> str | None:
        for label in (
            descriptor.text,
            descriptor.title,
            descriptor.accessible_name,
        ):
            if label:
                return label
        return None

    @staticmethod
    def action_label(descriptor: WidgetDescriptor) -> str | None:
        for label in (
            descriptor.text,
            descriptor.title,
            descriptor.accessible_name,
            descriptor.current_text,
            descriptor.object_name,
        ):
            if label:
                return label
        return None

    @classmethod
    def node(
        cls,
        descriptor: WidgetDescriptor,
        *,
        children: tuple[UiWidgetTreeNode, ...] | None = None,
    ) -> UiWidgetTreeNode:
        return UiWidgetTreeNode(
            path=descriptor.path,
            path_id=descriptor.path_id,
            child_index=descriptor.child_index,
            class_name=descriptor.class_name,
            object_name=descriptor.object_name,
            visible=descriptor.visible,
            enabled=descriptor.enabled,
            geometry=cls.rect(descriptor.geometry),
            global_geometry=cls.rect(descriptor.global_geometry),
            tool_tip=descriptor.tool_tip,
            status_tip=descriptor.status_tip,
            whats_this=descriptor.whats_this,
            window_title=descriptor.window_title,
            accessible_name=descriptor.accessible_name,
            accessible_description=descriptor.accessible_description,
            text=descriptor.text,
            text_truncated=descriptor.text_truncated,
            title=descriptor.title,
            action_kinds=tuple(kind.value for kind in descriptor.action_kinds),
            clickable=descriptor.clickable,
            actionable=descriptor.actionable,
            checkable=descriptor.checkable,
            checked=descriptor.checked,
            current_index=descriptor.current_index,
            current_text=descriptor.current_text,
            item_count=descriptor.item_count,
            item_texts=descriptor.item_texts,
            children=(
                tuple(cls.node(child) for child in descriptor.children)
                if children is None
                else children
            ),
        )

    @staticmethod
    def rect(rect: WidgetRect) -> UiWidgetRect:
        return UiWidgetRect(
            x=rect.x,
            y=rect.y,
            width=rect.width,
            height=rect.height,
        )

    @staticmethod
    def error(
        request: UiWidgetTreeRequest,
        error: AgentError,
        *,
        summary: UiWindowSummary | None = None,
    ) -> UiWidgetTreeResult:
        return UiWidgetTreeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            projected=False,
            summary=summary,
            errors=(error,),
            actionable_only=request.actionable_only,
            include_tree=request.include_tree,
            max_depth=request.max_depth,
            max_nodes=request.max_nodes,
        )


@dataclass(frozen=True, slots=True)
class _WidgetTreeNodeDraft:
    descriptor: WidgetDescriptor
    children: tuple["_WidgetTreeNodeDraft", ...]


@dataclass(slots=True)
class _WidgetTreeBoundState:
    returned_widget_count: int = 0
    truncated: bool = False


@dataclass(slots=True)
class _WidgetTreeActionListState:
    returned_actionable_count: int = 0
    truncated: bool = False


class FieldResetWidgetActionSummary:
    """Action-summary rule for per-field reset buttons."""

    BUTTON_CLASS_NAME = "QPushButton"
    LABEL = PARAMETER_FORM_CONSTANTS.RESET_BUTTON_TEXT
    OBJECT_NAME_SUFFIX = (
        f"{PARAMETER_FORM_CONSTANTS.FIELD_ID_SEPARATOR}"
        f"{PARAMETER_FORM_CONSTANTS.RESET_BUTTON_TEXT.lower()}"
    )

    @classmethod
    def matches(cls, summary: UiWidgetActionSummary) -> bool:
        return cls.matches_fields(
            class_name=summary.class_name,
            label=summary.label,
            object_name=summary.object_name,
            action_kinds=summary.action_kinds,
        )

    @classmethod
    def matches_fields(
        cls,
        *,
        class_name: str,
        label: str | None,
        object_name: str,
        action_kinds: tuple[str, ...],
    ) -> bool:
        return (
            class_name == cls.BUTTON_CLASS_NAME
            and label == cls.LABEL
            and object_name.endswith(cls.OBJECT_NAME_SUFFIX)
            and action_kinds == ("button",)
        )


class UiWidgetActionSummaryPolicy:
    """Inclusion policy for action summaries."""

    @staticmethod
    def includes(summary: UiWidgetActionSummary) -> bool:
        del summary
        return True


class UiWindowProjectionService(
    ScopeWindowTargetOperationProjectionABC,
    WindowCatalogProjectionABC,
):
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

    def summaries(self) -> tuple[UiWindowSummary, ...]:
        route_index = self._route_index()
        return (
            tuple(route.summary() for route in route_index.embedded_routes)
            + tuple(route.summary() for route in route_index.managed_routes)
            + self._dynamic.summaries(route_index)
        )

    def overview_sections(self) -> tuple[UiLiveOverviewSection, ...]:
        route_index = self._route_index()
        return WindowCatalogProjectionABC.overview_sections(self) + tuple(
            section
            for route in route_index.embedded_routes
            for section in route.overview_sections()
        )

    def handles(self, window_id: str) -> bool:
        identity = UiWindowIdentity(window_id=window_id)
        route_index = self._route_index()
        if route_index.embedded_route(identity) is not None:
            return True
        if route_index.managed_route(identity) is not None:
            return True
        if identity.window_id in WindowManager.get_open_scopes():
            return True
        if self._open_window_manager_scope_id(identity) is not None:
            return True
        return ScopeWindowRegistry.find_handler(identity.window_id) is not None

    def focus(self, request: UiWindowFocusRequest) -> UiWindowFocusResult:
        identity = request.as_identity()
        route_index = self._route_index()
        embedded_route = self._embedded_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=False,
        )
        if embedded_route is not None:
            return self._focused_result(request, embedded_route.focus())

        managed_route = self._managed_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=False,
        )
        if managed_route is not None and request.open_policy.create_if_missing:
            return self._focused_result(request, managed_route.focus())

        open_scope_id = self._open_window_manager_scope_id(identity)
        if open_scope_id is not None:
            if WindowManager.focus_and_navigate(open_scope_id):
                return self._focused_result(
                    request,
                    self._dynamic_scope_projection().summary(
                        identity,
                        open_scope_id,
                    ),
                )

        embedded_route = self._embedded_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=True,
        )
        if embedded_route is not None:
            return self._focused_result(request, embedded_route.focus())
        managed_route = self._managed_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=True,
        )
        if managed_route is not None and request.open_policy.create_if_missing:
            return self._focused_result(request, managed_route.focus())

        result = ScopeWindowNavigationService.navigate(
            WindowNavigationRequest(
                scope_id=identity.window_id,
                create_if_missing=request.open_policy.create_if_missing,
            )
        )
        if result.focused:
            return self._focused_result(
                request,
                self._dynamic_scope_projection().summary(
                    identity,
                    result.window_scope_id,
                ),
            )

        return UiWindowFocusResult(
            schema_version=SCHEMA_VERSION,
            window_id=identity.window_id,
            focused=False,
            errors=(WindowProjectionResultAuthority.unknown_window(identity),),
        )

    def navigate(
        self,
        request: UiWindowNavigateRequest,
    ) -> UiWindowNavigateResult:
        identity = request.as_identity()
        route_index = self._route_index()
        embedded_route = self._embedded_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=False,
        )
        if embedded_route is not None:
            return self._navigate_result(
                request,
                focused=True,
                created=False,
                navigated=False,
                summary=embedded_route.focus(),
            )

        managed_route = self._managed_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=False,
        )
        if managed_route is not None and request.open_policy.create_if_missing:
            return self._navigate_result(
                request,
                focused=True,
                created=False,
                navigated=False,
                summary=managed_route.focus(),
            )

        open_scope_id = self._open_window_manager_scope_id(identity)
        if open_scope_id is not None:
            focused = WindowManager.focus_and_navigate(
                open_scope_id,
                item_id=request.item_id,
                field_path=request.field_path,
            )
            if focused:
                return self._navigate_result(
                    request,
                    focused=True,
                    created=False,
                    navigated=request.item_id is not None
                    or request.field_path is not None,
                    summary=self._dynamic_scope_projection().summary(
                        identity,
                        open_scope_id,
                    ),
                )

        embedded_route = self._embedded_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=True,
        )
        if embedded_route is not None:
            return self._navigate_result(
                request,
                focused=True,
                created=False,
                navigated=request.item_id is not None or request.field_path is not None,
                summary=embedded_route.focus(),
            )
        managed_route = self._managed_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=True,
        )
        if managed_route is not None and request.open_policy.create_if_missing:
            return self._navigate_result(
                request,
                focused=True,
                created=False,
                navigated=request.item_id is not None or request.field_path is not None,
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
                summary=self._dynamic_scope_projection().summary(
                    identity,
                    result.window_scope_id,
                ),
            )

        return UiWindowNavigateResult(
            schema_version=SCHEMA_VERSION,
            window_id=identity.window_id,
            focused=False,
            navigated=False,
            created=result.created,
            errors=(WindowProjectionResultAuthority.unknown_window(identity),),
        )

    def close(self, request: UiWindowCloseRequest) -> UiWindowCloseResult:
        identity = request.as_identity()
        route_index = self._route_index()
        embedded_route = self._embedded_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=False,
        )
        if embedded_route is not None:
            return WindowCloseResultBoundaryPolicy.error(
                request,
                "ui_window_close_unsupported",
                "Embedded main-window panes cannot be closed through this UI bridge operation.",
                summary=embedded_route.summary(),
            )

        managed_route = self._managed_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=False,
        )
        if managed_route is not None:
            return self._close_window_manager_scope(
                request,
                managed_route.summary(),
            )

        open_scope_id = self._open_window_manager_scope_id(identity)
        if open_scope_id is not None:
            return self._close_window_manager_scope(
                request,
                self._dynamic_scope_projection().summary(
                    identity,
                    open_scope_id,
                ),
            )

        embedded_route = self._embedded_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=True,
        )
        if embedded_route is not None:
            return WindowCloseResultBoundaryPolicy.error(
                request,
                "ui_window_close_unsupported",
                "Embedded main-window panes cannot be closed through this UI bridge operation.",
                summary=embedded_route.summary(),
            )
        managed_route = self._managed_route_resolution(
            identity,
            route_index,
            resolve_scope_alias=True,
        )
        if managed_route is not None:
            return self._close_window_manager_scope(
                request,
                managed_route.summary(),
            )

        scope = UiWindowManagerScope.from_identity(
            self._resolved_window_identity(identity)
        )
        scope_widget = WindowManager.get_window(scope.value)
        if scope_widget is not None:
            return self._close_window_manager_scope(
                request,
                self._dynamic_scope_projection().summary(
                    identity,
                    scope.value,
                ),
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

    @staticmethod
    def _close_window_manager_scope(
        request: UiWindowCloseRequest,
        summary: UiWindowSummary,
    ) -> UiWindowCloseResult:
        scope = summary.manager_scope
        if scope is None:
            scope = UiWindowManagerScope.from_identity(request.as_identity())
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

    def _dynamic_scope_projection(self) -> DynamicScopeWindowProjection:
        return self._dynamic

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


class MainWindowActionProvider(UiActionProviderABC):
    """Action provider for main-window application commands."""

    identity = MAIN_WINDOW_ACTION_PROVIDER_IDENTITY

    def __init__(self, main_window: "OpenHCSMainWindow") -> None:
        self._main_window = main_window

    def catalog(self) -> UiActionCatalog:
        return UiActionCatalog(
            schema_version=SCHEMA_VERSION,
            actions=tuple(self.summary(action.value) for action in MainWindowAction),
        )

    def summary(self, action_id: str) -> UiActionSummary:
        action = self._action(action_id)
        enabled = self._main_window.check_for_updates_action.isEnabled()
        return UiActionSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=action.value,
            ),
            title=action.title,
            enabled=enabled,
            invocation_mode="async",
            side_effects=action.side_effects,
            confirmation_required=action.confirmation_required,
            selection_mode="global",
            current_selection_count=0,
            target_scope_ids=(),
            disabled_error=(
                None
                if enabled
                else AgentError(
                    code="update_check_in_progress",
                    message="An OpenHCS update check is already in progress.",
                )
            ),
        )

    def invoke(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
        try:
            action = self._action(request.action_id)
            self._main_window.check_for_updates()
        except Exception as exc:
            return UiActionInvokeResult(
                schema_version=SCHEMA_VERSION,
                identity=UiActionIdentity(
                    widget_id=request.widget_id,
                    action_id=request.action_id,
                ),
                status=UiActionInvocationStatus.REJECTED.value,
                receipt=UiMutationReceipt.rejected_for(request.request_token),
                errors=(AgentError.from_exception("main_window_action_failed", exc),),
            )
        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=action.value,
            ),
            status=UiActionInvocationStatus.ACCEPTED.value,
            receipt=UiMutationReceipt.accepted_for(request.request_token),
        )

    @staticmethod
    def _action(action_id: str) -> MainWindowAction:
        action = MainWindowAction(action_id)
        if action is not MainWindowAction.CHECK_FOR_UPDATES:
            raise ValueError(f"Main-window action has no route: {action_id!r}")
        return action


class ManagedWindowActionProvider(UiActionProviderABC):
    """Action provider for generic WindowManager-managed form windows."""

    identity = MANAGED_WINDOW_ACTION_PROVIDER_IDENTITY

    def catalog(self) -> UiActionCatalog:
        return UiActionCatalog(
            schema_version=SCHEMA_VERSION,
            actions=tuple(self.summary(action.value) for action in ManagedWindowAction),
        )

    def summary(self, action_id: str) -> UiActionSummary:
        action = self._action(action_id)
        target_scope_ids = self._target_scope_ids(action)
        return UiActionSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=action.value,
            ),
            title=action.title,
            enabled=bool(target_scope_ids),
            invocation_mode="async",
            side_effects=action.side_effects,
            confirmation_required=action.confirmation_required,
            selection_mode="targeted",
            required_target_count=1,
            current_selection_count=len(target_scope_ids),
            target_scope_ids=target_scope_ids,
        )

    def invoke(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
        try:
            action = self._action(request.action_id)
            target_scope_id = self._single_target_scope_id(request)
            window = self._target_window(target_scope_id, action)
        except Exception as exc:
            return self._invoke_error(
                request,
                AgentError.from_exception("managed_window_action_rejected", exc),
            )

        if action.confirmation_required and request.confirmation_is_required():
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
            action.dispatch(window)
        except Exception as exc:
            return self._invoke_error(
                request,
                AgentError.from_exception("managed_window_action_failed", exc),
            )

        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=action.value,
            ),
            status=UiActionInvocationStatus.ACCEPTED.value,
            receipt=UiMutationReceipt.accepted_for(request.request_token),
            target_scope_ids=request.selected_scope_ids,
        )

    @staticmethod
    def _action(action_id: str) -> ManagedWindowAction:
        return ManagedWindowAction(action_id)

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
        action: ManagedWindowAction,
    ) -> BaseFormDialog:
        window = WindowManager.get_window(scope_id)
        if not isinstance(window, BaseFormDialog):
            raise ValueError(f"Window scope is not a managed form window: {scope_id!r}")
        capabilities = window.managed_window_action_capabilities()
        if not action.is_supported(capabilities):
            raise ValueError(f"Window does not support {action.value!r}: {scope_id!r}")
        return window

    @classmethod
    def _target_scope_ids(
        cls,
        action: ManagedWindowAction,
    ) -> tuple[str, ...]:
        return tuple(
            scope_id
            for scope_id in WindowManager.get_open_scopes()
            if cls._window_supports_action(WindowManager.get_window(scope_id), action)
        )

    @staticmethod
    def _window_supports_action(
        window: QWidget | None,
        action: ManagedWindowAction,
    ) -> bool:
        if not isinstance(window, BaseFormDialog):
            return False
        return action.is_supported(window.managed_window_action_capabilities())

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
            receipt=UiMutationReceipt.rejected_for(request.request_token),
            target_scope_ids=request.selected_scope_ids,
            errors=(error,),
        )


@dataclass(frozen=True, slots=True)
class MainWindowBridgeProviderSet(UiBridgeProviderSetABC):
    """Provider set for main-window generic UI projections."""

    main_window: "OpenHCSMainWindow"
    registry_key = MAIN_WINDOW_PROVIDER_ID

    @classmethod
    def for_main_window(cls, main_window) -> "MainWindowBridgeProviderSet":
        return cls(main_window)

    def register(self, context: UiBridgeRegistrationContext) -> None:
        for provider_type in WindowCatalogProjectionABC.registered_types():
            context.registry.register_window_provider(
                provider_type.create(self.main_window)
            )
        context.registry.register_action_provider(
            MainWindowActionProvider(self.main_window)
        )
        context.registry.register_action_provider(ManagedWindowActionProvider())
