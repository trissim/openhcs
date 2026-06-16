"""Incremental live-server tree synchronization."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem
from pyqt_reactive.services.zmq_server_info_parser import BaseServerInfo, ExecutionServerInfo
from zmqruntime.viewer_state import ViewerState, ViewerStateManager


class LiveServerTreeSync:
    """Synchronize scanned server rows and launching-viewer pseudo-rows."""

    def __init__(
        self,
        *,
        tree: QTreeWidget,
        find_item_by_port: Callable[[int], Optional[QTreeWidgetItem]],
        sync_server_item: Callable[[BaseServerInfo], None],
        progress_execution_ids: Callable[[], set[str]],
        last_known_servers: Dict[int, dict],
        missing_port_counts: Dict[int, int],
    ) -> None:
        self._tree = tree
        self._find_item_by_port = find_item_by_port
        self._sync_server_item = sync_server_item
        self._progress_execution_ids = progress_execution_ids
        self._last_known_servers = last_known_servers
        self._missing_port_counts = missing_port_counts

    def populate_tree(self, parsed_servers: List[BaseServerInfo]) -> None:
        scanned_ports = {info.port for info in parsed_servers}
        for port in scanned_ports:
            self._missing_port_counts.pop(port, None)

        self._sync_launching_viewers(scanned_ports)
        for server_info in parsed_servers:
            self._sync_server_item(server_info)
        self._remove_missing_server_rows(scanned_ports)

    def _sync_launching_viewers(self, scanned_ports: set[int]) -> None:
        manager = ViewerStateManager.get_instance()
        launching_viewers = {
            viewer.port: viewer
            for viewer in manager.list_viewers()
            if viewer.state == ViewerState.LAUNCHING
        }

        for port, viewer in launching_viewers.items():
            if port in scanned_ports:
                continue

            existing_item = self._find_item_by_port(port)
            viewer_type = viewer.viewer_type.capitalize()
            queued = viewer.queued_images
            info_text = f"{queued} images queued" if queued > 0 else "Starting..."

            if existing_item is not None:
                existing_item.setText(0, f"Port {port} - {viewer_type} Viewer")
                existing_item.setText(1, "🚀 Launching")
                existing_item.setText(2, info_text)
                continue

            item = QTreeWidgetItem()
            item.setText(0, f"Port {port} - {viewer_type} Viewer")
            item.setText(1, "🚀 Launching")
            item.setText(2, info_text)
            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {"port": port, "launching": True, "viewer_type": viewer.viewer_type},
            )
            self._tree.addTopLevelItem(item)

    def _remove_missing_server_rows(self, scanned_ports: set[int]) -> None:
        for index in range(self._tree.topLevelItemCount() - 1, -1, -1):
            item = self._tree.topLevelItem(index)
            if item is None:
                continue
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(data, dict):
                continue
            port = data.get("port")
            if port is None or port in scanned_ports:
                continue
            if data.get("launching"):
                self._missing_port_counts.pop(port, None)
                continue
            if self._has_active_execution(port):
                self._missing_port_counts.pop(port, None)
                continue

            misses = self._missing_port_counts.get(port, 0) + 1
            self._missing_port_counts[port] = misses
            if misses < 2:
                continue

            self._missing_port_counts.pop(port, None)
            self._tree.takeTopLevelItem(index)

    def _has_active_execution(self, port: int) -> bool:
        last_known = self._last_known_servers.get(port, {})
        running_execs = last_known.get("running_executions", [])
        active_execution_ids = [
            str(exec_info.get("execution_id"))
            for exec_info in running_execs
            if exec_info.get("execution_id")
        ]
        tracker_exec_ids = self._progress_execution_ids()
        return any(exec_id in tracker_exec_ids for exec_id in active_execution_ids)
