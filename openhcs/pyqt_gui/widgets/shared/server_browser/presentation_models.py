"""Server-browser state + presentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Callable

from PyQt6.QtWidgets import QTreeWidgetItem
from pyqt_reactive.services.zmq_server_info import (
    BaseServerInfo,
    ExecutionServerInfo,
    GenericServerInfo,
    ViewerServerInfo,
)

from openhcs.core.progress.projection import ExecutionRuntimeProjection


@dataclass(frozen=True)
class ExecutionServerSummary:
    """Derived status for execution-server top-level row."""

    status_text: str
    info_text: str


def summarize_execution_server(
    projection: ExecutionRuntimeProjection,
) -> ExecutionServerSummary:
    plate_count = len(projection.by_plate_latest)
    if plate_count == 0:
        return ExecutionServerSummary(status_text="✅ Idle", info_text="")

    parts = projection.count_status_labels()
    status_text = ", ".join(parts) if parts else "✅ Idle"
    info_text = f"Avg: {projection.overall_percent:.1f}% | {plate_count} plates"
    return ExecutionServerSummary(status_text=status_text, info_text=info_text)


class ServerRowPresenter:
    """Type-dispatched server row rendering and child population."""

    def __init__(
        self,
        *,
        create_tree_item: Callable[
            [str, str, str, BaseServerInfo],
            QTreeWidgetItem,
        ],
        update_execution_server_item: Callable[
            [QTreeWidgetItem, ExecutionServerInfo],
            None,
        ],
        log_warning: Callable[..., None],
    ) -> None:
        self._create_tree_item = create_tree_item
        self._update_execution_server_item = update_execution_server_item
        self._log_warning = log_warning

    @singledispatchmethod
    def render_server(self, info: BaseServerInfo, status_icon: str) -> QTreeWidgetItem:
        raise NotImplementedError(f"No render for {type(info).__name__}")

    @render_server.register
    def _(self, info: ExecutionServerInfo, status_icon: str) -> QTreeWidgetItem:
        server_text = f"Port {info.port} - Execution Server"
        if not info.ready:
            return self._create_tree_item(server_text, "🚀 Starting", "", info)
        info_text = f"{len(info.workers)} active workers" if info.workers else ""
        return self._create_tree_item(server_text, "✅ Idle", info_text, info)

    @render_server.register
    def _(self, info: ViewerServerInfo, status_icon: str) -> QTreeWidgetItem:
        kind_name = info.viewer_name.title()
        display_text = f"Port {info.port} - {kind_name} Viewer"
        info_text = ""
        if info.memory_mb is not None:
            info_text = f"Mem: {info.memory_mb:.0f}MB"
            if info.cpu_percent is not None:
                info_text += f" | CPU: {info.cpu_percent:.1f}%"
        return self._create_tree_item(display_text, status_icon, info_text, info)

    @render_server.register
    def _(self, info: GenericServerInfo, status_icon: str) -> QTreeWidgetItem:
        return self._create_tree_item(
            f"Port {info.port} - {info.server_name}", status_icon, "", info
        )

    @singledispatchmethod
    def populate_server_children(
        self, info: BaseServerInfo, server_item: QTreeWidgetItem
    ) -> bool:
        self._log_warning(
            "_populate_server_children: No handler for type %s, using default (no children)",
            type(info).__name__,
        )
        return False

    @populate_server_children.register
    def _(self, info: ExecutionServerInfo, server_item: QTreeWidgetItem) -> bool:
        self._update_execution_server_item(server_item, info)
        return server_item.childCount() > 0

    @populate_server_children.register
    def _(self, info: ViewerServerInfo, server_item: QTreeWidgetItem) -> bool:
        return False

    @populate_server_children.register
    def _(self, info: GenericServerInfo, server_item: QTreeWidgetItem) -> bool:
        return False
