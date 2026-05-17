"""Execution progress projection for the ZMQ server manager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from PyQt6.QtWidgets import QTreeWidgetItem
from pyqt_reactive.services import ExecutionServerInfo, ServerInfoParserABC
from pyqt_reactive.widgets.shared import TreeSyncAdapter

from openhcs.core.progress import ProgressEvent

from .presentation_models import ProgressTopologyState, summarize_execution_server
from .progress_tree_builder import ProgressNode, ProgressTreeBuilder

logger = logging.getLogger(__name__)


class ExecutionProgressProjection:
    """Build and merge execution progress nodes from tracker/server snapshots."""

    def __init__(
        self,
        *,
        builder: ProgressTreeBuilder,
        topology_state: ProgressTopologyState,
    ) -> None:
        self._builder = builder
        self._topology_state = topology_state

    @staticmethod
    def plate_name(plate_id: str, exec_id: str | None = None) -> str:
        plate_leaf = Path(plate_id).name
        if exec_id:
            return f"{plate_leaf} ({exec_id[:8]})"
        return plate_leaf

    def build_progress_tree(
        self, executions: Dict[str, List[ProgressEvent]]
    ) -> List[ProgressNode]:
        return self._builder.build_progress_tree(
            executions=executions,
            worker_assignments=self._topology_state.worker_assignments,
            known_wells=self._topology_state.known_wells,
            step_names=self._topology_state.step_names,
            get_plate_name=self.plate_name,
        )

    def merge_server_snapshot_nodes(
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
                node = ProgressNode(
                    node_id=plate_id,
                    node_type="plate",
                    label=f"📋 {self.plate_name(plate_id, execution_id)}",
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
            if execution_id in running_execution_ids or plate_id in running_plate_ids:
                continue

            existing = by_plate_id.get(plate_id)
            if existing is None:
                node = ProgressNode(
                    node_id=plate_id,
                    node_type="plate",
                    label=f"📋 {self.plate_name(plate_id, execution_id)}",
                    status="⏳ Queued",
                    info=f"0.0%{queue_suffix}",
                    execution_id=execution_id,
                    percent=0.0,
                    children=[],
                )
                nodes.append(node)
                by_plate_id[plate_id] = node
                logger.debug("_merge: created NEW queued node for %s...", plate_id[:30])
                continue

            # Progress events are authoritative for the SAME execution.
            is_same_execution = existing.execution_id == execution_id
            has_real_progress = existing.children or existing.percent > 0

            if is_same_execution and has_real_progress:
                logger.debug(
                    "_merge: KEEP progress for %s... status=%s",
                    plate_id[:30],
                    existing.status,
                )
                continue

            if existing.status in ("⚙️ Executing", "⏳ Compiling"):
                logger.debug(
                    "_merge: SKIP queued for %s... already %s",
                    plate_id[:30],
                    existing.status,
                )
                continue

            logger.debug(
                "_merge: SET queued for %s... (same_exec=%s)",
                plate_id[:30],
                is_same_execution,
            )
            existing.status = "⏳ Queued"
            existing.execution_id = execution_id
            existing.percent = 0.0
            existing.info = f"0.0%{queue_suffix}"
            if not is_same_execution:
                existing.children = []

        return nodes


class ExecutionServerProgressRenderer:
    """Render execution progress into an execution-server tree row."""

    def __init__(
        self,
        *,
        tracker,
        parser: ServerInfoParserABC,
        projection: ExecutionProgressProjection,
        tree_sync_adapter: TreeSyncAdapter,
        tree_builder: ProgressTreeBuilder,
    ) -> None:
        self._tracker = tracker
        self._parser = parser
        self._projection = projection
        self._tree_sync_adapter = tree_sync_adapter
        self._tree_builder = tree_builder

    def update_execution_server_item(
        self, server_item: QTreeWidgetItem, server_data: dict
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

            parsed_server_info = self._parser.parse(server_data)
            if not isinstance(parsed_server_info, ExecutionServerInfo):
                raise ValueError(
                    "Expected ExecutionServerInfo for execution subtree update, "
                    f"got {type(parsed_server_info).__name__}"
                )

            nodes = self._projection.build_progress_tree(executions) if executions else []
            nodes = self._projection.merge_server_snapshot_nodes(
                nodes, parsed_server_info
            )
            summary = summarize_execution_server(nodes)
            logger.debug(
                "SUMMARY: status=%s, info=%s", summary.status_text, summary.info_text
            )
            server_item.setText(1, summary.status_text)
            server_item.setText(2, summary.info_text)
            self._tree_sync_adapter.sync_children(
                server_item,
                self._tree_builder.to_tree_nodes(nodes),
            )
        except Exception as error:
            logger.exception("Error updating execution server item: %s", error)
