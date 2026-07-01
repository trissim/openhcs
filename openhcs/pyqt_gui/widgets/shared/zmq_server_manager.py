"""OpenHCS thin wrapper over generic ZMQ server browser widget."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QTimer, Qt, pyqtSlot
from PyQt6.QtWidgets import QTreeWidgetItem

from pyqt_reactive.services.zmq_server_info_parser import (
    BaseServerInfo,
    DefaultServerInfoParser,
    ExecutionServerInfo,
    ServerInfoParserABC,
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
from zmqruntime.viewer_state import ViewerStateManager

from openhcs.agent.dto.ui_bridge import (
    UiLiveOverviewItem,
    UiLiveOverviewMetric,
    UiLiveOverviewSection,
    UiLiveOverviewSeverity,
)
from openhcs.core.progress import ProgressEvent, registry
from openhcs.pyqt_gui.services.ui_bridge_contracts import UiLiveOverviewWidget
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.widgets.shared.server_browser import (
    ExecutionProgressProjection,
    ExecutionServerProgressRenderer,
    LiveServerTreeSync,
    ProgressTopologyState,
    ProgressTreeBuilder,
    ServerKillService,
    ServerRowPresenter,
)

logger = logging.getLogger(__name__)


class ZMQServerManagerWidget(UiLiveOverviewWidget, ZMQServerBrowserWidgetABC):
    """OpenHCS adapter for generic ZMQ browser UI + OpenHCS progress semantics."""

    def __init__(
        self,
        ports_to_scan: List[int],
        title: str = "ZMQ Servers",
        style_generator: Optional[StyleSheetGenerator] = None,
        server_info_parser: Optional[ServerInfoParserABC] = None,
        parent=None,
    ):
        if style_generator is None:
            raise RuntimeError("style_generator is required for ZMQServerManagerWidget")

        from openhcs.constants.constants import CONTROL_PORT_OFFSET
        from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG

        parser = server_info_parser or DefaultServerInfoParser()
        scan_service = ZMQServerScanService(
            control_port_offset=CONTROL_PORT_OFFSET,
            config=OPENHCS_ZMQ_CONFIG,
            host="localhost",
        )
        super().__init__(
            ports_to_scan=ports_to_scan,
            title=title,
            style_generator=style_generator,
            scan_service=scan_service,
            server_info_parser=parser,
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
        self._registry_listener = self._on_registry_event
        self._progress_tracker.add_listener(self._registry_listener)
        self._registry_listener_registered = True
        self._progress_dirty = False
        self._topology_state = ProgressTopologyState()
        self._known_wells = self._topology_state.known_wells
        self._worker_assignments = self._topology_state.worker_assignments
        self._seen_execution_ids = self._topology_state.seen_execution_ids

        self._zmq_client = None
        self._progress_client_port: Optional[int] = None

        self._tree_sync_adapter = TreeSyncAdapter()

        self._progress_tree_builder = ProgressTreeBuilder()
        self._progress_projection = ExecutionProgressProjection(
            builder=self._progress_tree_builder,
            topology_state=self._topology_state,
        )
        self._progress_renderer = ExecutionServerProgressRenderer(
            tracker=self._progress_tracker,
            parser=self._server_info_parser,
            projection=self._progress_projection,
            tree_sync_adapter=self._tree_sync_adapter,
            tree_builder=self._progress_tree_builder,
        )
        self._server_kill_service = ServerKillService.openhcs_default()
        self._server_row_presenter = ServerRowPresenter(
            create_tree_item=self._create_tree_item,
            update_execution_server_item=self._progress_renderer.update_execution_server_item,
            log_warning=logger.warning,
        )
        self._missing_port_counts: Dict[int, int] = {}
        self._live_tree_sync = LiveServerTreeSync(
            tree=self.server_tree,
            find_item_by_port=self._find_existing_server_item,
            sync_server_item=self._sync_server_item,
            progress_execution_ids=lambda: set(self._progress_tracker.get_execution_ids()),
            parse_server_info=self._server_info_parser.parse,
            last_known_servers=self._last_known_servers,
            missing_port_counts=self._missing_port_counts,
        )

        # Coalesce progress events into redraws instead of polling while idle.
        self._progress_timer = QTimer()
        self._progress_timer.setSingleShot(True)
        self._progress_timer.timeout.connect(self._update_from_progress)

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
        if not isinstance(data, dict):
            return False
        try:
            return self._server_info_parser.parse(data).ready
        except Exception:
            return False

    def populate_tree(self, parsed_servers: List[BaseServerInfo]) -> None:
        """Populate tree with servers, avoiding duplicates since tree.clear() is bypassed."""
        self._live_tree_sync.populate_tree(parsed_servers)

    def _find_existing_server_item(self, port: int) -> Optional[QTreeWidgetItem]:
        """Find existing server item by port."""
        for idx in range(self.server_tree.topLevelItemCount()):
            item = self.server_tree.topLevelItem(idx)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, dict) and data.get("port") == port:
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
            existing_item.setData(0, Qt.ItemDataRole.UserRole, server_info.raw)
            self._server_row_presenter.populate_server_children(
                server_info, existing_item
            )
            return

        if rendered_item is None:
            return

        rendered_item.setData(0, Qt.ItemDataRole.UserRole, server_info.raw)
        self.server_tree.addTopLevelItem(rendered_item)
        self._server_row_presenter.populate_server_children(server_info, rendered_item)

    @pyqtSlot(list)
    def _update_server_list(self, servers: List[Dict[str, Any]]) -> None:
        """Override to bypass TreeRebuildCoordinator's tree.clear() which causes flicker."""
        self.servers = servers
        parsed_servers = [self._server_info_parser.parse(server) for server in servers]
        self.sync_progress_client_connection(parsed_servers)

        for server in servers:
            port = server.get("port")
            if port:
                self._last_known_servers[port] = server

        # Direct call to populate_tree bypasses the rebuild coordinator
        self.populate_tree(parsed_servers)

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
        execution_server_port = self._current_execution_server_port()
        if execution_server_port is not None:
            self._setup_progress_client(execution_server_port)

    def on_browser_hidden(self) -> None:
        if self._zmq_client is not None:
            self._zmq_client.disconnect()
            self._zmq_client = None
            self._progress_client_port = None

    def on_browser_cleanup(self) -> None:
        if self._zmq_client is not None:
            try:
                self._zmq_client.disconnect()
            except Exception as error:
                logger.warning(
                    "Failed to disconnect ZMQ client during cleanup: %s", error
                )
            self._zmq_client = None
            self._progress_client_port = None

        if self._viewer_state_callback_registered:
            mgr = ViewerStateManager.get_instance()
            if self._viewer_state_callback:
                mgr.unregister_state_callback(self._viewer_state_callback)
            self._viewer_state_callback_registered = False

        if self._registry_listener_registered:
            removed = self._progress_tracker.remove_listener(self._registry_listener)
            if not removed:
                raise RuntimeError(
                    "ZMQServerManagerWidget listener removal failed: listener not registered"
                )
            self._registry_listener_registered = False

        for execution_id in list(self._seen_execution_ids):
            self._progress_tracker.clear_execution(execution_id)
            self._topology_state.clear_execution(execution_id)
        self._topology_state.clear_all()

        if self._progress_timer is not None:
            self._progress_timer.stop()
            self._progress_timer.deleteLater()
            self._progress_timer = None

    def _setup_progress_client(self, port: int) -> None:
        from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

        if self._zmq_client is not None:
            try:
                self._zmq_client.disconnect()
            except Exception as error:
                logger.warning("Failed to disconnect existing ZMQ client: %s", error)
            self._zmq_client = None
            self._progress_client_port = None

        try:
            logger.debug("_setup_progress_client: creating new ZMQExecutionClient")
            self._zmq_client = ZMQExecutionClient(
                port=port,
                persistent=True,
                progress_callback=self._on_progress,
            )
            connected = self._zmq_client.connect(timeout=1)
            if not connected:
                logger.warning("_setup_progress_client: failed to connect")
                self._zmq_client = None
                self._progress_client_port = None
                return
            self._progress_client_port = port
            logger.debug(
                "_setup_progress_client: connected, starting progress listener"
            )
            self._zmq_client._start_progress_listener()
        except Exception as error:
            logger.warning("Failed to connect to execution server: %s", error)
            self._zmq_client = None
            self._progress_client_port = None

    def _on_progress(self, message: dict) -> None:
        event = ProgressEvent.from_dict(message)
        logger.debug(
            f"_on_progress: exec={event.execution_id[:8] if event.execution_id else None}, phase={event.phase}, status={event.status}"
        )
        self._topology_state.register_event(event)
        self._progress_tracker.register_event(event.execution_id, event)
        logger.debug(
            f"_on_progress: tracker now has {len(self._progress_tracker.get_execution_ids())} executions"
        )

    def _on_registry_event(self, _execution_id: str, _event: ProgressEvent) -> None:
        """Mark progress dirty when registry changes - triggers timer update."""
        self._progress_dirty = True
        if self._progress_timer is not None and not self._progress_timer.isActive():
            self._progress_timer.start(100)

    @pyqtSlot()
    def _update_from_progress(self) -> None:
        """Real-time progress update - called every 100ms by timer."""
        if not self._progress_dirty:
            return
        self._progress_dirty = False
        try:
            for index in range(self.server_tree.topLevelItemCount()):
                item = self.server_tree.topLevelItem(index)
                if item is None:
                    continue
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if not isinstance(data, dict):
                    continue
                try:
                    parsed_server_info = self._server_info_parser.parse(data)
                except Exception:
                    continue
                if isinstance(parsed_server_info, ExecutionServerInfo):
                    self._progress_renderer.update_execution_server_item(item, data)
        except Exception as error:
            logger.exception("Error updating from progress: %s", error)

    def sync_progress_client_connection(
        self, parsed_servers: List[BaseServerInfo]
    ) -> None:
        """Keep the progress client connected while an execution server is present."""
        execution_servers = tuple(
            server
            for server in parsed_servers
            if isinstance(server, ExecutionServerInfo)
        )
        if execution_servers:
            execution_server_port = execution_servers[0].port
            if (
                self._zmq_client is None
                or not self._zmq_client.is_connected()
                or self._progress_client_port != execution_server_port
            ):
                self._setup_progress_client(execution_server_port)
            return

        if self._current_execution_server_port() is not None and set(
            self._progress_tracker.get_execution_ids()
        ):
            return

        if self._zmq_client is not None:
            self._zmq_client.disconnect()
            self._zmq_client = None
            self._progress_client_port = None

    def _current_execution_server_port(self) -> Optional[int]:
        for index in range(self.server_tree.topLevelItemCount()):
            item = self.server_tree.topLevelItem(index)
            if item is None:
                continue
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(data, dict):
                continue
            try:
                server_info = self._server_info_parser.parse(data)
            except Exception:
                continue
            if isinstance(server_info, ExecutionServerInfo):
                return server_info.port
        return None

    def _create_tree_item(
        self, display: str, status: str, info: str, data: dict
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([display, status, info])
        item.setData(0, Qt.ItemDataRole.UserRole, data)
        return item

    @pyqtSlot()
    def refresh_launching_viewers_only(self) -> None:
        if self._lifecycle_state.is_cleaning_up():
            return
        self._update_server_list(self.servers)
