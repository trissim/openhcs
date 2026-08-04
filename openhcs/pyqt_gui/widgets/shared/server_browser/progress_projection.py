"""Execution progress projection for the ZMQ server manager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

from PyQt6.QtWidgets import QTreeWidgetItem
from pyqt_reactive.services.zmq_server_info import ExecutionServerInfo
from pyqt_reactive.widgets.shared import TreeStateAdapter, TreeSyncAdapter

from openhcs.core.progress import ProgressEvent
from openhcs.core.progress.runtime_tree import RuntimeTreeNode, RuntimeTreeProjection

from .presentation_models import summarize_execution_server
from .progress_tree_builder import ProgressTreeBuilder

logger = logging.getLogger(__name__)


class ExecutionProgressProjection:
    """Build the shared execution projection from tracker and server snapshots."""

    def __init__(
        self,
        *,
        builder: ProgressTreeBuilder,
    ) -> None:
        self._builder = builder

    @staticmethod
    def plate_name(plate_id: str, exec_id: str | None = None) -> str:
        plate_leaf = Path(plate_id).name
        if exec_id:
            return f"{plate_leaf} ({exec_id[:8]})"
        return plate_leaf

    def build_runtime_tree(
        self,
        executions: Dict[str, List[ProgressEvent]],
        server_info: ExecutionServerInfo,
    ) -> RuntimeTreeProjection:
        return self._builder.build_projection(
            executions=executions,
            running_executions=server_info.running_execution_entries,
            queued_executions=server_info.queued_execution_entries,
            get_plate_name=self.plate_name,
        )


class ExecutionServerProgressRenderer:
    """Render execution progress into an execution-server tree row."""

    def __init__(
        self,
        *,
        tracker,
        projection: ExecutionProgressProjection,
        tree_sync_adapter: TreeSyncAdapter,
        tree_state_adapter: TreeStateAdapter,
        tree_builder: ProgressTreeBuilder,
    ) -> None:
        self._tracker = tracker
        self._projection = projection
        self._tree_sync_adapter = tree_sync_adapter
        self._tree_state_adapter = tree_state_adapter
        self._tree_builder = tree_builder

    def update_execution_server_item(
        self,
        server_item: QTreeWidgetItem,
        server_info: ExecutionServerInfo,
    ) -> None:
        try:
            executions = {
                execution_id: self._tracker.get_events(execution_id)
                for execution_id in self._tracker.get_execution_ids()
            }
            logger.debug(
                "_update_exec_server: tracker has %d executions, progress events: %s",
                len(executions),
                list(executions.keys()),
            )

            tree_projection = self._projection.build_runtime_tree(
                executions,
                server_info,
            )
            nodes = list(tree_projection.roots)
            summary = summarize_execution_server(tree_projection.runtime)
            logger.debug(
                "SUMMARY: status=%s, info=%s", summary.status_text, summary.info_text
            )
            server_item.setText(1, summary.status_text)
            server_item.setText(2, summary.info_text)
            self._sync_progress_children(server_item, nodes)
        except Exception as error:
            logger.exception("Error updating execution server item: %s", error)

    def _sync_progress_children(
        self,
        server_item: QTreeWidgetItem,
        nodes: Sequence[RuntimeTreeNode],
    ) -> None:
        """Sync typed nodes while preserving explicit expansion choices."""

        previous_children = tuple(
            server_item.child(index) for index in range(server_item.childCount())
        )
        previous_expansion = (
            self._tree_state_adapter.capture_subtree_expansion_state(
                previous_children
            )
        )

        self._tree_sync_adapter.sync_children(
            server_item,
            self._tree_builder.node_converter.to_tree_nodes(nodes),
        )
        current_children = tuple(
            server_item.child(index) for index in range(server_item.childCount())
        )
        self._tree_state_adapter.restore_subtree_expansion_state(
            current_children,
            previous_expansion,
            default_expanded=True,
        )
        if not previous_children and current_children:
            server_item.setExpanded(True)
