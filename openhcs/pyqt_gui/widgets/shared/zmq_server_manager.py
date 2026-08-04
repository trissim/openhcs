"""OpenHCS thin wrapper over generic ZMQ server browser widget."""

from __future__ import annotations

import logging
from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QTreeWidgetItem
from pyqt_reactive.services.zmq_server_info import (
    BaseServerInfo,
    ExecutionServerInfo,
)
from pyqt_reactive.services.zmq_server_scan_service import (
    ZMQServerScanService,
)
from pyqt_reactive.theming import StyleSheetGenerator
from pyqt_reactive.widgets.shared import (
    KillOperationPlan,
    TreeSyncAdapter,
    ZMQServerBrowserWidgetABC,
)
from zmqruntime.messages import PongResponse
from zmqruntime.progress import EventRegistryMutation
from zmqruntime.viewer_state import ViewerStateManager

from openhcs.agent.dto.ui_bridge import (
    UiLiveOverviewItem,
    UiLiveOverviewMetric,
    UiLiveOverviewSection,
    UiLiveOverviewSeverity,
)
from openhcs.core.progress import ProgressEvent, registry
from openhcs.pyqt_gui.config import ProgressUIConfig
from openhcs.pyqt_gui.services.ui_bridge_contracts import UiLiveOverviewWidget
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.widgets.shared.server_browser import (
    ExecutionProgressProjection,
    ExecutionServerProgressRenderer,
    LaunchingViewerServerInfo,
    LiveServerTreeSync,
    ProgressTreeBuilder,
    ServerKillService,
    ServerRowPresenter,
)
from openhcs.runtime.zmq_config import OpenHCSZMQConfig

logger = logging.getLogger(__name__)


class ZMQServerManagerWidget(UiLiveOverviewWidget, ZMQServerBrowserWidgetABC):
    """OpenHCS adapter for generic ZMQ browser UI + OpenHCS progress semantics."""

    _progress_registry_changed = pyqtSignal()

    def __init__(
        self,
        ports_to_scan: List[int],
        config: OpenHCSZMQConfig,
        progress_config: ProgressUIConfig,
        title: str = "ZMQ Servers",
        style_generator: Optional[StyleSheetGenerator] = None,
        parent=None,
    ):
        if style_generator is None:
            raise RuntimeError("style_generator is required for ZMQServerManagerWidget")

        self._config = config
        scan_service = ZMQServerScanService(
            config=config,
            host=config.client_host,
            transport_mode=config.transport_mode,
            timeout_ms=config.server_scan_timeout_ms,
        )
        super().__init__(
            ports_to_scan=ports_to_scan,
            title=title,
            style_generator=style_generator,
            scan_service=scan_service,
            parent=parent,
        )

        def _manager_callback(_instance) -> None:
            try:
                from PyQt6.QtCore import QMetaObject, Qt

                QMetaObject.invokeMethod(
                    self,
                    "refresh_launching_viewers_only",
                    Qt.ConnectionType.QueuedConnection,
                )
            except Exception as error:
                logger.debug("Viewer state callback invocation failed: %s", error)

        mgr = ViewerStateManager.get_instance()
        mgr.register_state_callback(_manager_callback)
        self._viewer_state_callback = _manager_callback
        self._viewer_state_callback_registered = True

        self._progress_tracker = registry()
        self._registry_listener = self._on_registry_mutation
        self._progress_tracker.add_mutation_listener(self._registry_listener)
        self._registry_listener_registered = True
        self._progress_dirty = False
        self._progress_config = progress_config

        self._tree_sync_adapter = TreeSyncAdapter()

        self._progress_tree_builder = ProgressTreeBuilder()
        self._progress_projection = ExecutionProgressProjection(
            builder=self._progress_tree_builder,
        )
        self._progress_renderer = ExecutionServerProgressRenderer(
            tracker=self._progress_tracker,
            projection=self._progress_projection,
            tree_sync_adapter=self._tree_sync_adapter,
            tree_state_adapter=self._tree_state_adapter,
            tree_builder=self._progress_tree_builder,
        )
        self._server_kill_service = ServerKillService.openhcs_default(config)
        self._server_row_presenter = ServerRowPresenter(
            create_tree_item=self._create_tree_item,
            update_execution_server_item=self._progress_renderer.update_execution_server_item,
            log_warning=logger.warning,
        )
        self._missing_port_counts: dict[int, int] = {}

        self._live_tree_sync = LiveServerTreeSync(
            tree=self.server_tree,
            find_item_by_port=self._find_existing_server_item,
            sync_server_item=self._sync_server_item,
            progress_execution_ids=lambda: set(
                self._progress_tracker.get_execution_ids()
            ),
            last_known_servers=self._last_known_servers,
            missing_port_counts=self._missing_port_counts,
        )

        # Coalesce progress events into redraws instead of polling while idle.
        self._progress_timer = QTimer()
        self._progress_timer.setSingleShot(True)
        self._progress_timer.timeout.connect(self._update_from_progress)
        self._progress_registry_changed.connect(
            self._queue_progress_refresh,
            type=Qt.ConnectionType.QueuedConnection,
        )

    def set_zmq_config(
        self,
        config: OpenHCSZMQConfig,
        ports_to_scan: List[int],
    ) -> None:
        """Apply one resolved transport config to browser and progress clients."""

        self._config = config
        self.ports_to_scan = ports_to_scan
        self._scan_service = ZMQServerScanService(
            config=config,
            host=config.client_host,
            transport_mode=config.transport_mode,
            timeout_ms=config.server_scan_timeout_ms,
        )
        self._server_kill_service = ServerKillService.openhcs_default(config)

    def set_progress_config(self, config: ProgressUIConfig) -> None:
        """Apply the application progress coalescing rate."""

        self._progress_config = config

    def overview_sections(self) -> tuple[UiLiveOverviewSection, ...]:
        rows = tuple(
            self._overview_item_for_row(row_index)
            for row_index in range(self.server_tree.topLevelItemCount())
        )
        return (
            UiLiveOverviewSection(
                section_id=OpenHCSUiWindowId.zmq_server_manager,
                title="ZMQ Server Manager",
                summary=f"{len(rows)} servers",
                metrics=(
                    UiLiveOverviewMetric(
                        key="servers",
                        label="servers",
                        value=str(len(rows)),
                    ),
                    UiLiveOverviewMetric(
                        key="ready",
                        label="ready",
                        value=str(self._ready_server_count()),
                    ),
                ),
                items=rows,
            ),
        )

    def _overview_item_for_row(self, row_index: int) -> UiLiveOverviewItem:
        item = self.server_tree.topLevelItem(row_index)
        ready = self._server_row_ready(item)
        return UiLiveOverviewItem(
            label=item.text(0),
            status=item.text(1),
            detail=item.text(2),
            severity=(
                UiLiveOverviewSeverity.INFO.value
                if ready
                else UiLiveOverviewSeverity.WARNING.value
            ),
            source_window_id=OpenHCSUiWindowId.zmq_server_manager,
        )

    def _ready_server_count(self) -> int:
        return sum(
            1
            for row_index in range(self.server_tree.topLevelItemCount())
            if self._server_row_ready(self.server_tree.topLevelItem(row_index))
        )

    def _server_row_ready(self, item: QTreeWidgetItem) -> bool:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        return isinstance(data, BaseServerInfo) and data.ready

    def populate_tree(self, parsed_servers: List[BaseServerInfo]) -> None:
        """Populate tree with servers, avoiding duplicates since tree.clear() is bypassed."""
        self._live_tree_sync.populate_tree(parsed_servers)

    def _find_existing_server_item(self, port: int) -> Optional[QTreeWidgetItem]:
        """Find existing server item by port."""
        for idx in range(self.server_tree.topLevelItemCount()):
            item = self.server_tree.topLevelItem(idx)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, (BaseServerInfo, LaunchingViewerServerInfo)) and (
                data.port == port
            ):
                return item
        return None

    def _sync_server_item(self, server_info: BaseServerInfo) -> None:
        """Sync a server item - update existing or create new."""
        existing_item = self._find_existing_server_item(server_info.port)
        status_icon = "✅" if server_info.ready else "🚀"
        rendered_item = self._server_row_presenter.render_server(
            server_info, status_icon
        )

        if existing_item is not None:
            if rendered_item is not None:
                existing_item.setText(0, rendered_item.text(0))
                if not isinstance(server_info, ExecutionServerInfo):
                    existing_item.setText(1, rendered_item.text(1))
                    existing_item.setText(2, rendered_item.text(2))
            existing_item.setData(0, Qt.ItemDataRole.UserRole, server_info)
            self._server_row_presenter.populate_server_children(
                server_info, existing_item
            )
            return

        if rendered_item is None:
            return

        rendered_item.setData(0, Qt.ItemDataRole.UserRole, server_info)
        self.server_tree.addTopLevelItem(rendered_item)
        self._server_row_presenter.populate_server_children(server_info, rendered_item)

    @pyqtSlot(list)
    def _update_server_list(self, responses: list[PongResponse]) -> None:
        """Override to bypass TreeRebuildCoordinator's tree.clear() which causes flicker."""
        servers = [BaseServerInfo.from_response(response) for response in responses]
        self.servers = servers
        for server in servers:
            self._last_known_servers[server.port] = server

        # Direct call to populate_tree bypasses the rebuild coordinator
        self.populate_tree(servers)

    def periodic_domain_cleanup(self) -> None:
        removed = self._progress_tracker.cleanup_old_executions()
        if removed > 0:
            logger.info(f"Periodic cleanup: removed {removed} old completed executions")

    def kill_ports_with_plan(
        self,
        *,
        ports: List[int],
        plan: KillOperationPlan,
        on_server_killed,
    ) -> tuple[bool, str]:
        return self._server_kill_service.kill_ports(
            ports=ports,
            plan=plan,
            on_server_killed=on_server_killed,
            log_info=logger.info,
            log_warning=logger.warning,
            log_error=logger.error,
        )

    def on_browser_shown(self) -> None:
        return None

    def on_browser_hidden(self) -> None:
        return None

    def on_browser_cleanup(self) -> None:
        if self._viewer_state_callback_registered:
            mgr = ViewerStateManager.get_instance()
            if self._viewer_state_callback:
                mgr.unregister_state_callback(self._viewer_state_callback)
            self._viewer_state_callback_registered = False

        if self._registry_listener_registered:
            removed = self._progress_tracker.remove_mutation_listener(
                self._registry_listener
            )
            if not removed:
                raise RuntimeError(
                    "ZMQServerManagerWidget listener removal failed: listener not registered"
                )
            self._registry_listener_registered = False

        if self._progress_timer is not None:
            self._progress_timer.stop()
            self._progress_timer.deleteLater()
            self._progress_timer = None

    def _on_registry_mutation(
        self,
        _mutation: EventRegistryMutation[ProgressEvent],
    ) -> None:
        """Queue registry-driven refresh onto the widget's Qt thread."""

        self._progress_registry_changed.emit()

    @pyqtSlot()
    def _queue_progress_refresh(self) -> None:
        self._progress_dirty = True
        if self._progress_timer is not None and not self._progress_timer.isActive():
            self._progress_timer.start(self._progress_config.update_interval_ms)

    @pyqtSlot()
    def _update_from_progress(self) -> None:
        """Render the latest coalesced progress snapshot."""
        if not self._progress_dirty:
            return
        self._progress_dirty = False
        try:
            for index in range(self.server_tree.topLevelItemCount()):
                item = self.server_tree.topLevelItem(index)
                if item is None:
                    continue
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(data, ExecutionServerInfo):
                    self._progress_renderer.update_execution_server_item(item, data)
        except Exception as error:
            logger.exception("Error updating from progress: %s", error)

    def _create_tree_item(
        self, display: str, status: str, info: str, data: BaseServerInfo
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([display, status, info])
        item.setData(0, Qt.ItemDataRole.UserRole, data)
        return item

    @pyqtSlot()
    def refresh_launching_viewers_only(self) -> None:
        if self._lifecycle_state.is_cleaning_up():
            return
        self.populate_tree(self.servers)
