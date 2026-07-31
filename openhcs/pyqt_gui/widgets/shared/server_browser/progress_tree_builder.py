"""PyQt adapter for core OpenHCS runtime tree projections."""

from __future__ import annotations

from typing import Callable, List, Mapping, Sequence

from pyqt_reactive.widgets.shared import TreeNode
from zmqruntime.messages import QueuedExecutionInfo, RunningExecutionInfo

from openhcs.core.progress import ProgressEvent
from openhcs.core.progress.projection import build_execution_runtime_projection
from openhcs.core.progress.runtime_tree import (
    RuntimeTreeNode,
    RuntimeTreeProjection,
    RuntimeTreeProjectionBuilder,
)

class ProgressTreeNodeConverter:
    """Convert core runtime tree nodes to PyQt-reactive TreeNode records."""

    def to_tree_node(self, node: RuntimeTreeNode) -> TreeNode:
        return TreeNode(
            node_id=node.node_id,
            node_type=node.node_type,
            label=node.label,
            status=node.status,
            info=node.info,
            children=[self.to_tree_node(child) for child in node.children],
        )

    def to_tree_nodes(self, nodes: Sequence[RuntimeTreeNode]) -> List[TreeNode]:
        return [self.to_tree_node(node) for node in nodes]


class ProgressTreeBuilder:
    """PyQt-facing adapter over the authoritative core runtime tree projection."""

    def __init__(
        self,
        *,
        projection_builder: RuntimeTreeProjectionBuilder | None = None,
        node_converter: ProgressTreeNodeConverter | None = None,
    ) -> None:
        self.projection_builder = projection_builder or RuntimeTreeProjectionBuilder()
        self.node_converter = node_converter or ProgressTreeNodeConverter()

    def build_projection(
        self,
        *,
        executions: Mapping[str, Sequence[ProgressEvent]],
        running_executions: Sequence[RunningExecutionInfo],
        queued_executions: Sequence[QueuedExecutionInfo],
        get_plate_name: Callable[[str, str | None], str],
    ) -> RuntimeTreeProjection:
        events_snapshot = {
            execution_id: list(events) for execution_id, events in executions.items()
        }
        runtime_projection = build_execution_runtime_projection(
            events_snapshot,
            running_executions=running_executions,
            queued_executions=queued_executions,
        )
        return self.projection_builder.build(
            executions=executions,
            get_plate_name=get_plate_name,
            runtime_projection=runtime_projection,
        )


__all__ = (
    "ProgressTreeBuilder",
    "ProgressTreeNodeConverter",
)
