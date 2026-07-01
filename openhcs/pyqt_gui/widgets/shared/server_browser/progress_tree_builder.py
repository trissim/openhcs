"""PyQt adapter for core OpenHCS runtime tree projections."""

from __future__ import annotations

from typing import Callable, List, Mapping, Sequence

from pyqt_reactive.widgets.shared import TreeNode

from openhcs.core.progress import ProgressEvent
from openhcs.core.progress.runtime_tree import (
    RuntimeExecutionTopology,
    RuntimeTreeNode,
    RuntimeTreeProjectionBuilder,
)


ProgressNode = RuntimeTreeNode


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
    """Compatibility adapter over the core runtime tree projection builder."""

    def __init__(
        self,
        *,
        projection_builder: RuntimeTreeProjectionBuilder | None = None,
        node_converter: ProgressTreeNodeConverter | None = None,
    ) -> None:
        self.projection_builder = projection_builder or RuntimeTreeProjectionBuilder()
        self.node_converter = node_converter or ProgressTreeNodeConverter()

    def build_progress_tree(
        self,
        *,
        executions: Mapping[str, Sequence[ProgressEvent]],
        worker_assignments: Mapping[tuple[str, str], Mapping[str, Sequence[str]]],
        known_wells: Mapping[tuple[str, str], Sequence[str]],
        step_names: Mapping[tuple[str, str, str], Mapping[int, str]],
        get_plate_name: Callable[[str, str | None], str],
    ) -> List[ProgressNode]:
        projection = self.projection_builder.build(
            executions=executions,
            topology=RuntimeExecutionTopology(
                worker_assignments=worker_assignments,
                known_wells=known_wells,
                step_names=step_names,
            ),
            get_plate_name=get_plate_name,
        )
        return list(projection.roots)


__all__ = (
    "ProgressNode",
    "ProgressTreeBuilder",
    "ProgressTreeNodeConverter",
)
