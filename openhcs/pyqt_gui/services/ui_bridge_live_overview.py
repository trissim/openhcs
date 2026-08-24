"""Live UI overview state-surface provider."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from zmqruntime.viewer_state import ViewerInstance, ViewerStateManager

from openhcs.agent.dto.common import AgentError, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiCodeDocumentSelectionMode,
    UiLiveOverviewItem,
    UiLiveOverviewMetric,
    UiLiveOverviewSection,
    UiLiveOverviewSeverity,
    UiLiveOverviewState,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
)
from openhcs.serialization.json import to_jsonable
from openhcs.agent.ui_bridge_identities import (
    UiLiveOverviewStateSurfaceIdentityDeclaration,
)
from openhcs.core.streaming_config_declarations import ViewerType
from objectstate.object_state import ObjectStateRegistry
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiBridgeSnapshotProviderABC,
    UiLiveOverviewContributorABC,
    UiLiveOverviewContributorIdentity,
    UiOwnedStateSurfaceDeclaration,
    UiStateSurfaceProviderABC,
    UiStateSurfaceProviderIdentity,
    state_surface_declaration_for_identity,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
    UiBridgeSurfaceRegistry,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId


class ViewerSessionLiveOverviewContributor(UiLiveOverviewContributorABC):
    """Project the process-wide viewer lifecycle authority into the overview."""

    overview_identity = UiLiveOverviewContributorIdentity(
        section_id="viewer-sessions",
        title="Viewer sessions",
    )

    def __init__(
        self,
        viewer_state_manager: ViewerStateManager | None = None,
    ) -> None:
        if viewer_state_manager is None:
            viewer_state_manager = ViewerStateManager.get_instance()
        self._viewer_state_manager = viewer_state_manager

    def overview_sections(self) -> tuple[UiLiveOverviewSection, ...]:
        viewers = tuple(
            sorted(
                self._viewer_state_manager.list_viewers(),
                key=lambda viewer: (viewer.viewer_type, viewer.port),
            )
        )
        return (
            UiLiveOverviewSection(
                section_id=self.overview_identity.section_id,
                title=self.overview_identity.title,
                summary=f"{len(viewers)} viewers",
                metrics=(
                    UiLiveOverviewMetric(
                        key="viewers",
                        label="viewers",
                        value=str(len(viewers)),
                    ),
                    UiLiveOverviewMetric(
                        key="healthy",
                        label="healthy",
                        value=str(sum(1 for viewer in viewers if viewer.is_healthy)),
                    ),
                    UiLiveOverviewMetric(
                        key="queued_images",
                        label="queued images",
                        value=str(sum(viewer.queued_images for viewer in viewers)),
                    ),
                ),
                items=tuple(self._overview_item(viewer) for viewer in viewers),
            ),
        )

    @staticmethod
    def _overview_item(viewer: ViewerInstance) -> UiLiveOverviewItem:
        if viewer.error_message:
            severity = UiLiveOverviewSeverity.ERROR.value
        elif not viewer.is_healthy:
            severity = UiLiveOverviewSeverity.WARNING.value
        else:
            severity = UiLiveOverviewSeverity.INFO.value
        detail_parts = (
            f"port={viewer.port}",
            f"queued_images={viewer.queued_images}",
            f"processed_images={viewer.processed_images}",
        )
        if viewer.error_message:
            detail_parts = (*detail_parts, f"error={viewer.error_message}")
        return UiLiveOverviewItem(
            label=(
                f"{ViewerType.from_wire_value(viewer.viewer_type).display_name} viewer"
            ),
            status=viewer.state.name.lower(),
            detail=" ".join(detail_parts),
            severity=severity,
            source_window_id=OpenHCSUiWindowId.zmq_server_manager,
        )


class LiveOverviewBridgeProviderSet(UiBridgeProviderSetABC):
    """Register the live overview surface after its source providers exist."""

    registry_key = UiLiveOverviewStateSurfaceIdentityDeclaration.require_value()

    def __init__(
        self,
        declaration: UiOwnedStateSurfaceDeclaration | None = None,
        *,
        widget_declaration=None,
    ) -> None:
        if declaration is None or widget_declaration is None:
            from openhcs.pyqt_gui.main import OpenHCSMainWindow

            declaration = state_surface_declaration_for_identity(
                OpenHCSMainWindow.UI_STATE_SURFACE_DECLARATIONS,
                UiLiveOverviewStateSurfaceIdentityDeclaration,
            )
            widget_declaration = OpenHCSMainWindow.UI_BRIDGE_WIDGET_IDENTITY
        self._declaration = declaration
        self._identity = UiStateSurfaceProviderIdentity.from_owner(
            declaration,
            widget_declaration=widget_declaration,
        )

    @classmethod
    def for_main_window(cls, main_window) -> "LiveOverviewBridgeProviderSet":
        del main_window
        return cls()

    def register(self, context: UiBridgeRegistrationContext) -> None:
        context.registry.register_live_overview_contributor(
            ViewerSessionLiveOverviewContributor()
        )
        context.registry.register_state_surface_provider(
            UiLiveOverviewStateSurfaceProvider(
                registry=context.registry,
                snapshot_provider=context.snapshot_provider,
                declaration=self._declaration,
                identity=self._identity,
            )
        )


class UiLiveOverviewStateSurfaceProvider(UiStateSurfaceProviderABC):
    """Pollable UI overview assembled from registered provider contributions."""

    def __init__(
        self,
        *,
        registry: UiBridgeSurfaceRegistry,
        snapshot_provider: UiBridgeSnapshotProviderABC,
        declaration: UiOwnedStateSurfaceDeclaration,
        identity: UiStateSurfaceProviderIdentity,
    ) -> None:
        self._registry = registry
        self._snapshot_provider = snapshot_provider
        self._declaration = declaration
        self.identity = identity

    def summary(self) -> UiStateSurfaceSummary:
        return UiStateSurfaceSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity.as_surface_identity(),
            title=self.identity.title,
            widget_id=self.identity.widget_id,
            readable=True,
            supported_selection_modes=(UiCodeDocumentSelectionMode.ALL.value,),
            current_selection_count=0,
            total_scope_count=1,
        )

    def read(self, request: UiStateSurfaceRequest) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(
            UiCodeDocumentSelectionMode.ALL
        )
        try:
            state = UiLiveOverviewState(
                schema_version=SCHEMA_VERSION,
                summary=self.summary(),
                object_state_token=ObjectStateRegistry.get_token(),
                sections=self._sections(),
                selected_scope_ids=(),
                current_revision_token=self._snapshot_provider.revision_token(
                    self.identity.revision_key
                ),
                current_snapshot=self._snapshot_provider.current_snapshot(),
            )
        except Exception as exc:
            return self._state_error(
                request,
                (AgentError.from_exception("ui_state_surface_read_failed", exc),),
            )
        revision_token = self._revision_token(state, selection_mode=selection_mode)
        state = replace(
            state,
            current_revision_token=revision_token,
            current_snapshot=self._snapshot_provider.current_snapshot(),
            unchanged=request.base_revision_token == revision_token,
        )
        return self._document_from_state(state, selection_mode=selection_mode)

    def _sections(self) -> tuple[UiLiveOverviewSection, ...]:
        sections: list[UiLiveOverviewSection] = []
        for provider in self._registry.overview_contributors():
            try:
                sections.extend(provider.overview_sections())
            except Exception as exc:
                sections.append(
                    self._provider_error_section(provider.overview_identity, exc)
                )
        return tuple(sections)

    @staticmethod
    def _provider_error_section(
        identity: UiLiveOverviewContributorIdentity,
        exc: Exception,
    ) -> UiLiveOverviewSection:
        error = AgentError.from_exception("ui_live_overview_provider_failed", exc)
        return UiLiveOverviewSection(
            section_id=identity.section_id,
            title=identity.title,
            summary="overview unavailable",
            items=(
                UiLiveOverviewItem(
                    label=error.code,
                    status=error.exception_type,
                    detail=error.message,
                    severity=UiLiveOverviewSeverity.ERROR.value,
                ),
            ),
        )

    def _revision_token(
        self,
        state: UiLiveOverviewState,
        *,
        selection_mode: str,
    ) -> str:
        parts = (
            self.identity.revision_key,
            str(state.object_state_token),
            self._snapshot_provider.current_branch_head_snapshot_id(),
            str(ObjectStateRegistry.get_current_snapshot_index()),
            selection_mode,
            state.sections,
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    def _state_error(
        self,
        request: UiStateSurfaceRequest,
        errors: tuple[AgentError, ...],
    ) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(
            UiCodeDocumentSelectionMode.ALL
        )
        state = UiLiveOverviewState(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            object_state_token=ObjectStateRegistry.get_token(),
            sections=(),
            selected_scope_ids=(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
            errors=errors,
        )
        return self._document_from_state(state, selection_mode=selection_mode)

    def _document_from_state(
        self,
        state: UiLiveOverviewState,
        *,
        selection_mode: str,
    ) -> UiStateSurfaceDocument:
        payload = to_jsonable(state)
        if not isinstance(payload, dict):
            raise TypeError("Live overview payload did not serialize to an object.")
        return UiStateSurfaceDocument(
            schema_version=state.schema_version,
            summary=state.summary,
            payload_schema=self._declaration.payload_schema,
            payload=payload,
            current_revision_token=state.current_revision_token,
            current_snapshot=state.current_snapshot,
            selection_mode=selection_mode,
            selected_scope_ids=state.selected_scope_ids,
            unchanged=state.unchanged,
            warnings=state.warnings,
            errors=state.errors,
        )
