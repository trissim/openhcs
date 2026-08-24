"""Incremental live-server tree synchronization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem
from pyqt_reactive.services.tree_item_key import EndpointPortProviderABC
from pyqt_reactive.services.zmq_server_info import BaseServerInfo
from pyqt_reactive.services.zmq_server_scan_service import StartingEndpointObservation
from zmqruntime.viewer_state import ViewerState, ViewerStateManager

from openhcs.core.streaming_config_declarations import ViewerType


@dataclass(frozen=True, slots=True)
class LaunchingViewerServerInfo(EndpointPortProviderABC):
    """Typed Qt row payload for a viewer that has not answered PING yet."""

    endpoint_port: int
    viewer_type: ViewerType
    queued_images: int

    @property
    def port(self) -> int:
        """Return the viewer endpoint port."""
        return self.endpoint_port


class LiveServerTreeSync:
    """Synchronize scanned server rows and launching-viewer pseudo-rows."""

    def __init__(
        self,
        *,
        tree: QTreeWidget,
        find_item_by_port: Callable[[int], QTreeWidgetItem | None],
        sync_server_item: Callable[[BaseServerInfo], None],
        sync_startup_endpoint: Callable[[StartingEndpointObservation], None],
    ) -> None:
        self._tree = tree
        self._find_item_by_port = find_item_by_port
        self._sync_server_item = sync_server_item
        self._sync_startup_endpoint = sync_startup_endpoint

    def populate_tree(
        self,
        parsed_servers: list[BaseServerInfo],
        startup_observations: tuple[StartingEndpointObservation, ...] = (),
    ) -> None:
        visible_ports = {info.port for info in parsed_servers}
        visible_ports.update(observation.port for observation in startup_observations)
        visible_ports.update(self._sync_launching_viewers(visible_ports))
        for observation in startup_observations:
            self._sync_startup_endpoint(observation)
        for server_info in parsed_servers:
            self._sync_server_item(server_info)
        self._remove_missing_server_rows(visible_ports)

    def _sync_launching_viewers(self, scanned_ports: set[int]) -> set[int]:
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
            launching_info = LaunchingViewerServerInfo(
                endpoint_port=port,
                viewer_type=ViewerType.from_wire_value(viewer.viewer_type),
                queued_images=viewer.queued_images,
            )
            viewer_type = launching_info.viewer_type.display_name
            queued = launching_info.queued_images
            info_text = f"{queued} images queued" if queued > 0 else "Starting..."

            if existing_item is not None:
                existing_item.setText(0, f"Port {port} - {viewer_type} Viewer")
                existing_item.setText(1, "🚀 Launching")
                existing_item.setText(2, info_text)
                existing_item.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    launching_info,
                )
                continue

            item = QTreeWidgetItem()
            item.setText(0, f"Port {port} - {viewer_type} Viewer")
            item.setText(1, "🚀 Launching")
            item.setText(2, info_text)
            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                launching_info,
            )
            self._tree.addTopLevelItem(item)
        return set(launching_viewers)

    def _remove_missing_server_rows(self, scanned_ports: set[int]) -> None:
        """Remove every endpoint row absent from the authoritative PONG snapshot."""
        for index in range(self._tree.topLevelItemCount() - 1, -1, -1):
            item = self._tree.topLevelItem(index)
            if item is None:
                continue
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(data, EndpointPortProviderABC):
                continue
            port = data.port
            if port in scanned_ports:
                continue
            self._tree.takeTopLevelItem(index)
