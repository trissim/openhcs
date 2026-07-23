"""Populate ZMQ server browser tree from typed server snapshots."""

from __future__ import annotations

from typing import Callable, Dict, List, Set

from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem
from zmqruntime.viewer_state import ViewerStateManager, ViewerState

from pyqt_reactive.services.zmq_server_info_parser import (
    BaseServerInfo,
)


class ServerTreePopulation:
    """Builds top-level server rows and launching/busy viewer pseudo-rows."""

    def __init__(
        self,
        *,
        create_tree_item: Callable[[str, str, str, dict], QTreeWidgetItem],
        render_server: Callable[[BaseServerInfo, str], QTreeWidgetItem],
        populate_server_children: Callable[[BaseServerInfo, QTreeWidgetItem], bool],
        log_info: Callable[..., None],
    ) -> None:
        self._create_tree_item = create_tree_item
        self._render_server = render_server
        self._populate_server_children = populate_server_children
        self._log_info = log_info

    def populate_tree(
        self,
        *,
        tree: QTreeWidget,
        parsed_servers: List[BaseServerInfo],
    ) -> None:
        from zmqruntime.queue_tracker import GlobalQueueTrackerRegistry

        launching_viewers = self._get_launching_viewers()
        self._add_launching_viewers_to_tree(tree, launching_viewers)

        scanned_ports = {server_info.port for server_info in parsed_servers}
        registry = GlobalQueueTrackerRegistry()
        self._add_busy_viewers_to_tree(
            tree=tree,
            registry=registry,
            scanned_ports=scanned_ports,
            launching_viewer_ports=set(launching_viewers.keys()),
        )
        self._add_scanned_servers_to_tree(
            tree=tree,
            parsed_servers=parsed_servers,
            launching_viewer_ports=set(launching_viewers.keys()),
        )

    @staticmethod
    def _get_launching_viewers() -> Dict[int, Dict]:
        mgr = ViewerStateManager.get_instance()
        return {
            viewer.port: {
                "type": viewer.viewer_type,
                "queued_images": viewer.queued_images,
            }
            for viewer in mgr.list_viewers()
            if viewer.state == ViewerState.LAUNCHING
        }

    def _add_launching_viewers_to_tree(
        self, tree: QTreeWidget, launching_viewers: Dict[int, Dict]
    ) -> None:
        for port, info in launching_viewers.items():
            viewer_type = info["type"].capitalize()
            queued_images = info["queued_images"]
            info_text = (
                f"{queued_images} images queued" if queued_images > 0 else "Starting..."
            )
            item = self._create_tree_item(
                f"Port {port} - {viewer_type} Viewer",
                "🚀 Launching",
                info_text,
                {"port": port, "launching": True},
            )
            tree.addTopLevelItem(item)

    def _add_busy_viewers_to_tree(
        self,
        *,
        tree: QTreeWidget,
        registry,
        scanned_ports: Set[int],
        launching_viewer_ports: Set[int],
    ) -> None:
        for tracker_port, tracker in registry.get_all_trackers().items():
            if tracker_port in scanned_ports or tracker_port in launching_viewer_ports:
                continue
            pending = tracker.get_pending_count()
            if pending <= 0:
                continue

            processed, total = tracker.get_progress()
            viewer_type = tracker.viewer_type.capitalize()
            status_text = "⚙️"
            info_text = f"Processing: {processed}/{total} images"
            if tracker.has_stuck_images():
                status_text = "⚠️"
                stuck_images = tracker.get_stuck_images()
                info_text += f" (⚠️ {len(stuck_images)} stuck)"

            pseudo_server = {
                "port": tracker_port,
                "server": f"{viewer_type}ViewerServer",
                "ready": True,
                "busy": True,
            }
            item = self._create_tree_item(
                f"Port {tracker_port} - {viewer_type}ViewerServer",
                status_text,
                info_text,
                pseudo_server,
            )
            tree.addTopLevelItem(item)

    def _add_scanned_servers_to_tree(
        self,
        *,
        tree: QTreeWidget,
        parsed_servers: List[BaseServerInfo],
        launching_viewer_ports: Set[int],
    ) -> None:
        for server_info in parsed_servers:
            if server_info.port in launching_viewer_ports:
                continue
            status_icon = "✅" if server_info.ready else "🚀"
            server_item = self._render_server(server_info, status_icon)
            if server_item is None:
                continue
            tree.addTopLevelItem(server_item)
            self._log_info(
                "About to populate children for server type %s on port %s",
                type(server_info).__name__,
                server_info.port,
            )
            has_children = self._populate_server_children(server_info, server_item)
            self._log_info(
                "Populate children returned %s for port %s",
                has_children,
                server_info.port,
            )
