"""OpenHCS thin wrapper over generic ZMQ server browser widget."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QTimer, Qt, pyqtSlot
from PyQt6.QtWidgets import QTreeWidgetItem

from pyqt_reactive.services import (
    BaseServerInfo,
    DefaultServerInfoParser,
    ExecutionServerInfo,
    ServerInfoParserABC,
    ZMQServerScanService,
)
from pyqt_reactive.theming import StyleSheetGenerator
from pyqt_reactive.widgets.shared import (
    KillOperationPlan,
    TreeSyncAdapter,
    ZMQServerBrowserWidgetABC,
)
from zmqruntime.viewer_state import ViewerStateManager

from openhcs.core.progress import ProgressEvent, registry
from openhcs.pyqt_gui.widgets.shared.server_browser import (
    ProgressNode,
    ProgressTopologyState,
    ProgressTreeBuilder,
    ServerKillService,
    ServerRowPresenter,
    summarize_execution_server,
)

logger = logging.getLogger(__name__)


class ZMQServerManagerWidget(ZMQServerBrowserWidgetABC):
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

        self._tree_sync_adapter = TreeSyncAdapter()

        self._progress_tree_builder = ProgressTreeBuilder()
        self._server_kill_service = ServerKillService()
        self._server_row_presenter = ServerRowPresenter(
            create_tree_item=self._create_tree_item,
            update_execution_server_item=self._update_execution_server_item,
            log_warning=logger.warning,
        )
        self._missing_port_counts: Dict[int, int] = {}

        # Real-time progress timer for smooth UI updates during execution
        self._progress_timer = QTimer()
        self._progress_timer.timeout.connect(self._update_from_progress)
        self._progress_timer.start(100)  # 100ms for smooth updates

    def populate_tree(self, parsed_servers: List[BaseServerInfo]) -> None:
        """Populate tree with servers, avoiding duplicates since tree.clear() is bypassed."""
        scanned_ports = {info.port for info in parsed_servers}
        for port in scanned_ports:
            self._missing_port_counts.pop(port, None)

        # Add launching viewers (being auto-started from streaming)
        self._sync_launching_viewers(scanned_ports)

        # Add/update servers from scan
        for server_info in parsed_servers:
            self._sync_server_item(server_info)

        # Remove servers only after consecutive misses to avoid transient flicker.
        for idx in range(self.server_tree.topLevelItemCount() - 1, -1, -1):
            item = self.server_tree.topLevelItem(idx)
            if item is None:
                continue
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(data, dict):
                continue
            port = data.get("port")
            if port is None:
                continue
            if port in scanned_ports:
                continue

            # Check if this is a launching viewer (being auto-started from streaming)
            if data.get("launching"):
                # Don't count launching viewers as missing - they persist until ready
                self._missing_port_counts.pop(port, None)
                continue

            # Check if server has active executions before removing
            last_known = self._last_known_servers.get(port, {})
            running_execs = last_known.get("running_executions", [])
            active_execution_ids = [
                str(exec_info.get("execution_id"))
                for exec_info in running_execs
                if exec_info.get("execution_id")
            ]
            tracker_exec_ids = set(self._progress_tracker.get_execution_ids())
            has_active_executions = any(
                exec_id in tracker_exec_ids for exec_id in active_execution_ids
            )

            if has_active_executions:
                # Server is busy with active executions, don't remove it
                self._missing_port_counts.pop(port, None)
                continue

            misses = self._missing_port_counts.get(port, 0) + 1
            self._missing_port_counts[port] = misses
            if misses < 2:
                continue

            self._missing_port_counts.pop(port, None)
            self.server_tree.takeTopLevelItem(idx)

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

    def _sync_launching_viewers(self, scanned_ports: set[int]) -> None:
        """Add launching viewer entries to tree for viewers being auto-started."""
        from zmqruntime.viewer_state import ViewerStateManager, ViewerState

        mgr = ViewerStateManager.get_instance()
        launching_viewers = {
            viewer.port: viewer
            for viewer in mgr.list_viewers()
            if viewer.state == ViewerState.LAUNCHING
        }

        for port, viewer in launching_viewers.items():
            # Skip if already scanned (viewer is now running)
            if port in scanned_ports:
                continue

            # Check if already in tree
            existing_item = self._find_existing_server_item(port)
            if existing_item is not None:
                # Update existing item
                viewer_type = viewer.viewer_type.capitalize()
                queued = viewer.queued_images
                info_text = f"{queued} images queued" if queued > 0 else "Starting..."
                existing_item.setText(0, f"Port {port} - {viewer_type} Viewer")
                existing_item.setText(1, "🚀 Launching")
                existing_item.setText(2, info_text)
                continue

            # Create new tree item for launching viewer
            viewer_type = viewer.viewer_type.capitalize()
            queued = viewer.queued_images
            info_text = f"{queued} images queued" if queued > 0 else "Starting..."

            item = QTreeWidgetItem()
            item.setText(0, f"Port {port} - {viewer_type} Viewer")
            item.setText(1, "🚀 Launching")
            item.setText(2, info_text)
            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {"port": port, "launching": True, "viewer_type": viewer.viewer_type},
            )
            self.server_tree.addTopLevelItem(item)

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
        self._setup_progress_client()

    def on_browser_hidden(self) -> None:
        if self._zmq_client is not None:
            self._zmq_client.disconnect()
            self._zmq_client = None

    def on_browser_cleanup(self) -> None:
        if self._zmq_client is not None:
            try:
                self._zmq_client.disconnect()
            except Exception as error:
                logger.warning(
                    "Failed to disconnect ZMQ client during cleanup: %s", error
                )
            self._zmq_client = None

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

    def _setup_progress_client(self) -> None:
        from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

        if self._zmq_client is not None:
            try:
                self._zmq_client.disconnect()
            except Exception as error:
                logger.warning("Failed to disconnect existing ZMQ client: %s", error)
            self._zmq_client = None

        try:
            logger.debug("_setup_progress_client: creating new ZMQExecutionClient")
            self._zmq_client = ZMQExecutionClient(
                port=7777,
                persistent=True,
                progress_callback=self._on_progress,
            )
            connected = self._zmq_client.connect(timeout=1)
            if not connected:
                logger.warning("_setup_progress_client: failed to connect")
                self._zmq_client = None
                return
            logger.debug(
                "_setup_progress_client: connected, starting progress listener"
            )
            self._zmq_client._start_progress_listener()
        except Exception as error:
            logger.warning("Failed to connect to execution server: %s", error)
            self._zmq_client = None

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
                    self._update_execution_server_item(item, data)
        except Exception as error:
            logger.exception("Error updating from progress: %s", error)

    def _get_plate_name(self, plate_id: str, exec_id: str | None = None) -> str:
        plate_leaf = Path(plate_id).name
        if exec_id:
            return f"{plate_leaf} ({exec_id[:8]})"
        return plate_leaf

    def _get_topology_state(self) -> ProgressTopologyState:
        """Return topology state, tolerating domain-only tests that bypass Qt init."""
        try:
            return self._topology_state
        except (AttributeError, RuntimeError):
            return ProgressTopologyState()

    def _build_progress_tree(self, executions: Dict[str, list]) -> List[ProgressNode]:
        topology_state = self._get_topology_state()
        return self._progress_tree_builder.build_progress_tree(
            executions=executions,
            worker_assignments=self._worker_assignments,
            known_wells=self._known_wells,
            step_names=topology_state.step_names,
            get_plate_name=self._get_plate_name,
        )

    def sync_progress_client_connection(
        self, parsed_servers: List[BaseServerInfo]
    ) -> None:
        """Keep the progress client connected while an execution server is present."""
        has_execution_server = any(
            isinstance(server, ExecutionServerInfo) for server in parsed_servers
        )
        if has_execution_server:
            if self._zmq_client is None or not self._zmq_client.is_connected():
                self._setup_progress_client()
            return

        if self._zmq_client is not None:
            self._zmq_client.disconnect()
            self._zmq_client = None

    def _update_execution_server_item(
        self, server_item: QTreeWidgetItem, server_data: dict
    ) -> None:
        try:
            executions = {
                execution_id: self._progress_tracker.get_events(execution_id)
                for execution_id in self._progress_tracker.get_execution_ids()
            }
            logger.debug(
                f"_update_exec_server: tracker has {len(executions)} executions, progress events: {list(executions.keys())}"
            )
            for exec_id, events in executions.items():
                logger.debug(
                    f"  exec={exec_id[:8] if exec_id else None}, events={len(events) if events else 0}"
                )

            parsed_server_info = self._server_info_parser.parse(server_data)
            if not isinstance(parsed_server_info, ExecutionServerInfo):
                raise ValueError(
                    "Expected ExecutionServerInfo for execution subtree update, "
                    f"got {type(parsed_server_info).__name__}"
                )

            nodes = self._build_progress_tree(executions) if executions else []
            for node in nodes:
                logger.debug(
                    f"  plate={node.node_id.split('/')[-1] if node.node_id else None}, status={node.status}, percent={node.percent}"
                )
            nodes = self._merge_server_snapshot_nodes(nodes, parsed_server_info)
            for node in nodes:
                logger.debug(
                    f"  MERGED plate={node.node_id.split('/')[-1] if node.node_id else None}, status={node.status}, percent={node.percent}"
                )

            summary = summarize_execution_server(nodes)
            logger.debug(
                f"  SUMMARY: status={summary.status_text}, info={summary.info_text}"
            )
            server_item.setText(1, summary.status_text)
            server_item.setText(2, summary.info_text)
            self._tree_sync_adapter.sync_children(
                server_item,
                self._progress_tree_builder.to_tree_nodes(nodes),
            )
        except Exception as error:
            logger.exception("Error updating execution server item: %s", error)

    def _merge_server_snapshot_nodes(
        self, nodes: List[ProgressNode], server_info: ExecutionServerInfo
    ) -> List[ProgressNode]:
        by_plate_id: Dict[str, ProgressNode] = {node.node_id: node for node in nodes}
        running_execution_ids = {
            running.execution_id for running in server_info.running_execution_entries
        }
        running_plate_ids = {
            running.plate_id for running in server_info.running_execution_entries
        }

        for running in server_info.running_execution_entries:
            plate_id = running.plate_id
            execution_id = running.execution_id
            running_status = "⏳ Compiling" if running.compile_only else "⚙️ Executing"
            existing = by_plate_id.get(plate_id)

            if existing is None:
                plate_name = self._get_plate_name(plate_id, execution_id)
                node = ProgressNode(
                    node_id=plate_id,
                    node_type="plate",
                    label=f"📋 {plate_name}",
                    status=running_status,
                    info="0.0%",
                    execution_id=execution_id,
                    percent=0.0,
                    children=[],
                )
                nodes.append(node)
                by_plate_id[plate_id] = node
                continue

            # Progress-derived nodes are authoritative when present.
            if not existing.children and existing.percent <= 0.0:
                existing.status = running_status
                existing.execution_id = execution_id
                if existing.percent <= 0.0:
                    existing.info = "0.0%"

        for queued in server_info.queued_execution_entries:
            plate_id = queued.plate_id
            execution_id = queued.execution_id
            queue_suffix = f" (q#{queued.queue_position})"

            # Running state is authoritative: do not regress active rows to queued.
            if (
                execution_id in running_execution_ids
                or plate_id in running_plate_ids
            ):
                continue

            existing = by_plate_id.get(plate_id)

            if existing is None:
                plate_name = self._get_plate_name(plate_id, execution_id)
                node = ProgressNode(
                    node_id=plate_id,
                    node_type="plate",
                    label=f"📋 {plate_name}",
                    status="⏳ Queued",
                    info=f"0.0%{queue_suffix}",
                    execution_id=execution_id,
                    percent=0.0,
                    children=[],
                )
                nodes.append(node)
                by_plate_id[plate_id] = node
                logger.debug(
                    f"_merge: created NEW queued node for {plate_id[:30]}..."
                )
                continue

            # Progress events are authoritative for the SAME execution.
            # For a NEW queued execution (different execution_id), queued overrides.
            is_same_execution = existing.execution_id == execution_id
            has_real_progress = existing.children or existing.percent > 0

            if is_same_execution and has_real_progress:
                # Same execution with progress - ping lag, keep progress status
                logger.debug(
                    f"_merge: KEEP progress for {plate_id[:30]}... status={existing.status}"
                )
                continue

            # Only update to queued if the existing status is not already executing/compiling.
            # Progress-derived active status should never be overridden by ping.
            if existing.status in ("⚙️ Executing", "⏳ Compiling"):
                logger.debug(
                    f"_merge: SKIP queued for {plate_id[:30]}... already {existing.status}"
                )
                continue

            # New queued execution or no progress yet - update to queued
            logger.debug(
                f"_merge: SET queued for {plate_id[:30]}... (same_exec={is_same_execution})"
            )
            existing.status = "⏳ Queued"
            existing.execution_id = execution_id
            existing.percent = 0.0
            existing.info = f"0.0%{queue_suffix}"
            if not is_same_execution:
                existing.children = []

        return nodes

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
