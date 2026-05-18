"""Build execution/compilation progress trees for the server browser."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from pyqt_reactive.strategies import (
    ExplicitPercentTreeAggregationPolicy,
    MeanTreeAggregationPolicy,
    TreeAggregationPolicyRegistry,
)
from pyqt_reactive.widgets.shared import TreeNode

from openhcs.core.progress import (
    ProgressChannel,
    ProgressEvent,
    phase_channel,
    is_failure_event,
    is_success_terminal_event,
)
from openhcs.core.progress.types import ProgressChannelRole


MEAN_AGGREGATION_POLICY_ID = "mean"
EXPLICIT_AGGREGATION_POLICY_ID = "explicit"


class ProgressNodeType(str, Enum):
    """Node kinds rendered in the progress tree."""
    PLATE = "plate"
    WORKER = "worker"
    WELL = "well"
    STEP = "step"
    COMPILATION = "compilation"


@dataclass
class ProgressNode:
    """Pure tree node model for progress rendering."""

    node_id: str
    node_type: str
    label: str
    status: str
    info: str
    execution_id: str | None = None
    percent: float = 0.0
    aggregation_policy_id: str = "mean"
    children: List["ProgressNode"] = field(default_factory=list)


_NODE_AGGREGATION_POLICY_BY_TYPE: Dict[str, str] = {
    ProgressNodeType.PLATE.value: MEAN_AGGREGATION_POLICY_ID,
    ProgressNodeType.WORKER.value: MEAN_AGGREGATION_POLICY_ID,
    ProgressNodeType.WELL.value: EXPLICIT_AGGREGATION_POLICY_ID,
    ProgressNodeType.STEP.value: EXPLICIT_AGGREGATION_POLICY_ID,
    ProgressNodeType.COMPILATION.value: EXPLICIT_AGGREGATION_POLICY_ID,
}

_TREE_AGGREGATION_REGISTRY = TreeAggregationPolicyRegistry(
    policies={
        MEAN_AGGREGATION_POLICY_ID: MeanTreeAggregationPolicy(),
        EXPLICIT_AGGREGATION_POLICY_ID: ExplicitPercentTreeAggregationPolicy(),
    }
)


class ProgressTreeBuilder:
    """Transforms ProgressEvent snapshots into hierarchical progress nodes."""

    @staticmethod
    def _make_progress_node(
        *,
        node_id: str,
        node_type: ProgressNodeType,
        label: str,
        status: str,
        info: str,
        execution_id: str | None = None,
        percent: float = 0.0,
        children: Optional[List[ProgressNode]] = None,
    ) -> ProgressNode:
        return ProgressNode(
            node_id=node_id,
            node_type=node_type.value,
            label=label,
            status=status,
            info=info,
            execution_id=execution_id,
            percent=percent,
            aggregation_policy_id=_NODE_AGGREGATION_POLICY_BY_TYPE[node_type.value],
            children=children or [],
        )

    def _make_step_node(
        self,
        *,
        axis_id: str,
        step_idx: int,
        step_name: str,
        status: str,
        info: str,
        percent: float,
    ) -> ProgressNode:
        return self._make_progress_node(
            node_id=f"{axis_id}_step_{step_idx}",
            node_type=ProgressNodeType.STEP,
            label=f"🔧 {step_idx + 1} - {step_name}",
            status=status,
            info=info,
            percent=percent,
        )

    def _make_well_progress_node(
        self,
        *,
        axis_id: str,
        status: str,
        percent: float,
        children: List[ProgressNode],
    ) -> ProgressNode:
        return self._make_progress_node(
            node_id=axis_id,
            node_type=ProgressNodeType.WELL,
            label=f"[{axis_id}]",
            status=status,
            info="",
            percent=percent,
            children=children,
        )

    def build_progress_tree(
        self,
        *,
        executions: Dict[str, List[ProgressEvent]],
        worker_assignments: Dict[tuple[str, str], Dict[str, List[str]]],
        known_wells: Dict[tuple[str, str], List[str]],
        step_names: Dict[tuple[str, str, str], Dict[int, str]],
        get_plate_name: Callable[[str, str | None], str],
    ) -> List[ProgressNode]:
        plates: Dict[tuple[str, str], Dict[str, List[ProgressEvent]]] = {}
        for exec_id, events_list in executions.items():
            for event in events_list:
                key = (exec_id, event.plate_id)
                if key not in plates:
                    plates[key] = {"events": []}
                plates[key]["events"].append(event)

        nodes_by_plate: Dict[str, tuple[float, ProgressNode]] = {}
        for (exec_id, plate_id), pdata in plates.items():
            events = pdata["events"]
            if not events:
                continue
            latest_timestamp = max((event.timestamp for event in events), default=0.0)
            plate_name = get_plate_name(plate_id, exec_id)
            is_executing = self._is_execution_mode(
                execution_id=exec_id,
                plate_id=plate_id,
                events=events,
                worker_assignments=worker_assignments,
            )
            if is_executing:
                children = self._build_worker_children(
                    execution_id=exec_id,
                    plate_id=plate_id,
                    events=events,
                    worker_assignments=worker_assignments,
                    step_names=step_names,
                )
            else:
                children = self._build_compilation_children(
                    execution_id=exec_id,
                    plate_id=plate_id,
                    events=events,
                    known_wells=known_wells,
                )

            plate_node = self._make_progress_node(
                node_id=plate_id,
                node_type=ProgressNodeType.PLATE,
                label=f"📋 {plate_name}",
                status="⚙️ Executing" if is_executing else "⏳ Compiling",
                info="",
                execution_id=exec_id,
                children=children,
            )
            self._aggregate_percent_recursive(plate_node)
            if is_executing:
                if self._has_failed_descendant(plate_node):
                    plate_node.status = "❌ Failed"
                elif plate_node.percent >= 100.0:
                    plate_node.status = "✅ Complete"
                elif self._all_leaves_queued(plate_node):
                    plate_node.status = "⏳ Queued"
                else:
                    plate_node.status = "⚙️ Executing"
            else:
                if self._has_failed_descendant(plate_node):
                    plate_node.status = "❌ Compile Failed"
                else:
                    plate_node.status = (
                        "✅ Compiled" if plate_node.percent >= 100.0 else "⏳ Compiling"
                    )
            self._apply_node_percent_text(plate_node)
            existing = nodes_by_plate.get(plate_id)
            if existing is None or latest_timestamp > existing[0]:
                nodes_by_plate[plate_id] = (latest_timestamp, plate_node)

        # Promote compile-only plates to "✅ Complete" when their execution-
        # mode sibling was already cleaned up from the tracker.  The topology
        # state still holds worker_assignments for the removed execution, so
        # we can detect that an execution run existed.  Clear the stale
        # compilation children so the tree doesn't regress to "Compiling".
        # Only promote when the compile itself is finished ("✅ Compiled") —
        # a fresh compile still in progress must not be overridden.
        for plate_id, (_ts, node) in nodes_by_plate.items():
            if node.status == "✅ Compiled":
                had_execution_sibling = any(
                    exec_id not in executions
                    for (exec_id, p_id) in worker_assignments
                    if p_id == plate_id
                )
                if had_execution_sibling:
                    node.status = "✅ Complete"
                    node.children = []

        return sorted(
            (pair[1] for pair in nodes_by_plate.values()), key=lambda node: node.node_id
        )

    def _build_worker_children(
        self,
        *,
        execution_id: str,
        plate_id: str,
        events: List[ProgressEvent],
        worker_assignments: Dict[tuple[str, str], Dict[str, List[str]]],
        step_names: Dict[tuple[str, str, str], Dict[int, str]],
    ) -> List[ProgressNode]:
        assignments = worker_assignments.get((execution_id, plate_id))
        if assignments is None:
            raise ValueError(
                f"Missing worker assignments for execution plate '{plate_id}'"
            )

        channels = self._partition_events_by_channel(events)
        pipeline_by_axis: Dict[str, ProgressEvent] = {
            event.axis_id: event
            for event in channels[ProgressChannel.PIPELINE.value]
            if event.axis_id
        }
        step_by_axis: Dict[str, ProgressEvent] = {
            event.axis_id: event
            for event in channels[ProgressChannel.STEP.value]
            if event.axis_id
        }

        worker_nodes: List[ProgressNode] = []
        execution_started = any(
            phase_channel(event.phase) == ProgressChannel.INIT for event in events
        )
        for worker_slot, axis_ids in sorted(assignments.items()):
            well_nodes = [
                self._build_well_node(
                    axis_id=axis_id,
                    pipeline_event=pipeline_by_axis.get(axis_id),
                    step_event=step_by_axis.get(axis_id),
                    step_names=step_names.get((execution_id, plate_id, axis_id), {}),
                )
                for axis_id in axis_ids
            ]
            failed_count = sum(1 for node in well_nodes if node.status == "❌ Failed")
            complete_count = sum(
                1 for node in well_nodes if node.status == "✅ Complete"
            )
            queued_count = sum(1 for node in well_nodes if node.status == "⏳ Queued")
            active_count = (
                len(well_nodes) - failed_count - complete_count - queued_count
            )
            if failed_count > 0:
                worker_status = f"❌ {failed_count} failed"
            elif complete_count == len(well_nodes):
                worker_status = "✅ Complete"
            elif queued_count == len(well_nodes):
                worker_status = "⚙️ Starting" if execution_started else "⏳ Queued"
            else:
                worker_status = f"⚙️ {active_count} active"
            worker_nodes.append(
                self._make_progress_node(
                    node_id=worker_slot,
                    node_type=ProgressNodeType.WORKER,
                    label=f"Worker {worker_slot}",
                    status=worker_status,
                    info="",
                    children=well_nodes,
                )
            )
        return worker_nodes

    def _build_compilation_children(
        self,
        *,
        execution_id: str,
        plate_id: str,
        events: List[ProgressEvent],
        known_wells: Dict[tuple[str, str], List[str]],
    ) -> List[ProgressNode]:
        channels = self._partition_events_by_channel(events)
        compile_by_axis: Dict[str, ProgressEvent] = {
            event.axis_id: event
            for event in channels[ProgressChannel.COMPILE.value]
            if event.axis_id
        }
        known_axis_ids = list(known_wells.get((execution_id, plate_id), []))
        axis_ids = known_axis_ids if known_axis_ids else sorted(compile_by_axis.keys())
        extra_axis_ids = [
            axis_id for axis_id in compile_by_axis if axis_id not in axis_ids
        ]
        axis_ids.extend(sorted(extra_axis_ids))

        compilation_nodes: List[ProgressNode] = []
        for axis_id in axis_ids:
            compile_event = compile_by_axis.get(axis_id)
            if compile_event is None:
                status = "⏳ Compiling"
                percent = 0.0
            elif is_failure_event(compile_event):
                status = "❌ Failed"
                percent = compile_event.percent
            elif is_success_terminal_event(compile_event):
                status = "✅ Compiled"
                percent = compile_event.percent
            else:
                status = "⏳ Compiling"
                percent = compile_event.percent

            compilation_nodes.append(
                self._make_progress_node(
                    node_id=axis_id,
                    node_type=ProgressNodeType.COMPILATION,
                    label=f"[{axis_id}]",
                    status=status,
                    info="",
                    percent=percent,
                )
            )
        return compilation_nodes

    @staticmethod
    def _partition_events_by_channel(
        events: List[ProgressEvent],
    ) -> Dict[str, List[ProgressEvent]]:
        partitioned: Dict[str, List[ProgressEvent]] = {
            ProgressChannel.INIT.value: [],
            ProgressChannel.COMPILE.value: [],
            ProgressChannel.PIPELINE.value: [],
            ProgressChannel.STEP.value: [],
        }
        for event in events:
            partitioned[phase_channel(event.phase).value].append(event)
        return partitioned

    def _build_well_node(
        self,
        *,
        axis_id: str,
        pipeline_event: Optional[ProgressEvent],
        step_event: Optional[ProgressEvent],
        step_names: Dict[int, str],
    ) -> ProgressNode:
        if pipeline_event is None:
            status = "⏳ Queued"
            percent = 0.0
        elif is_failure_event(pipeline_event):
            status = "❌ Failed"
            percent = pipeline_event.percent
        elif is_success_terminal_event(pipeline_event):
            status = "✅ Complete"
            percent = pipeline_event.percent
        else:
            status = f"⚙️ {pipeline_event.step_name}"
            percent = pipeline_event.percent

        # Pipeline events own well progress. Step children are display detail only.
        children: List[ProgressNode] = []
        if pipeline_event is not None and pipeline_event.total > 0:
            current_step_idx = pipeline_event.completed
            total_steps = pipeline_event.total

            if current_step_idx < total_steps:
                if step_event is not None:
                    step_name = step_event.step_name
                    if is_failure_event(step_event):
                        step_status = "❌ Failed"
                        step_percent = step_event.percent
                    else:
                        step_status = (
                            f"{step_event.completed}/{step_event.total} groups"
                        )
                        step_percent = step_event.percent
                else:
                    step_name = step_names.get(
                        current_step_idx, f"Step {current_step_idx + 1}"
                    )
                    step_status = "⏳ Starting"
                    step_percent = 0.0

                children.append(
                    self._make_step_node(
                        axis_id=axis_id,
                        step_idx=current_step_idx,
                        step_name=step_name,
                        status=step_status,
                        info=f"{step_percent:.1f}%",
                        percent=step_percent,
                    )
                )

            for step_idx in range(current_step_idx):
                step_name = step_names.get(step_idx, f"Step {step_idx + 1}")
                children.append(
                    self._make_step_node(
                        axis_id=axis_id,
                        step_idx=step_idx,
                        step_name=step_name,
                        status="✅ Complete",
                        info="100.0%",
                        percent=100.0,
                    )
                )

            # Add future steps at 0% to ensure proper average calculation
            for step_idx in range(current_step_idx + 1, total_steps):
                step_name = step_names.get(step_idx, f"Step {step_idx + 1}")
                children.append(
                    self._make_step_node(
                        axis_id=axis_id,
                        step_idx=step_idx,
                        step_name=step_name,
                        status="⏳ Pending",
                        info="0.0%",
                        percent=0.0,
                    )
                )

        return self._make_well_progress_node(
            axis_id=axis_id,
            status=status,
            percent=percent,
            children=children,
        )

    def _aggregate_percent_recursive(self, node: ProgressNode) -> float:
        if not node.children:
            return node.percent
        child_values = [
            self._aggregate_percent_recursive(child) for child in node.children
        ]
        expected_policy = _NODE_AGGREGATION_POLICY_BY_TYPE.get(node.node_type)
        if expected_policy is None:
            raise ValueError(f"No aggregation policy for node_type '{node.node_type}'")
        if node.aggregation_policy_id != expected_policy:
            raise ValueError(
                f"Aggregation policy mismatch for node_type '{node.node_type}': "
                f"expected '{expected_policy}', got '{node.aggregation_policy_id}'"
            )
        node.percent = _TREE_AGGREGATION_REGISTRY.aggregate(
            node.aggregation_policy_id, node.percent, child_values
        )
        return node.percent

    def _apply_node_percent_text(self, node: ProgressNode) -> None:
        if node.node_type in {
            ProgressNodeType.PLATE.value,
            ProgressNodeType.WORKER.value,
            ProgressNodeType.WELL.value,
            ProgressNodeType.COMPILATION.value,
        }:
            node.info = f"{node.percent:.1f}%"
        elif node.node_type == ProgressNodeType.STEP.value and not node.info:
            node.info = f"{node.percent:.1f}%"
        for child in node.children:
            self._apply_node_percent_text(child)

    @staticmethod
    def _is_execution_mode(
        *,
        execution_id: str,
        plate_id: str,
        events: List[ProgressEvent],
        worker_assignments: Dict[tuple[str, str], Dict[str, List[str]]],
    ) -> bool:
        # Phase channels are authoritative for mode selection:
        # compile-channel presence means compilation view, unless we also
        # have real axis-scoped pipeline/step execution events.
        has_compile_events = any(
            phase_channel(event.phase) == ProgressChannel.COMPILE for event in events
        )
        has_axis_execution_events = any(
            phase_channel(event.phase).role is ProgressChannelRole.EXECUTION
            and bool(event.axis_id)
            for event in events
        )
        if has_compile_events and not has_axis_execution_events:
            return False

        topology_key = (execution_id, plate_id)
        if topology_key in worker_assignments:
            return True
        return has_axis_execution_events

    @staticmethod
    def _all_leaves_queued(node: ProgressNode) -> bool:
        """Return True only when *every* leaf descendant has '⏳ Queued' status."""
        if not node.children:
            return node.status == "⏳ Queued"
        return all(
            ProgressTreeBuilder._all_leaves_queued(child) for child in node.children
        )

    @staticmethod
    def _has_status_descendant(node: ProgressNode, status: str) -> bool:
        if node.status == status:
            return True
        return any(
            ProgressTreeBuilder._has_status_descendant(child, status)
            for child in node.children
        )

    @staticmethod
    def _has_failed_descendant(node: ProgressNode) -> bool:
        if node.status.startswith("❌"):
            return True
        return any(
            ProgressTreeBuilder._has_failed_descendant(child) for child in node.children
        )

    @staticmethod
    def to_tree_node(node: ProgressNode) -> TreeNode:
        return TreeNode(
            node_id=node.node_id,
            node_type=node.node_type,
            label=node.label,
            status=node.status,
            info=node.info,
            children=[
                ProgressTreeBuilder.to_tree_node(child) for child in node.children
            ],
        )

    @staticmethod
    def to_tree_nodes(nodes: List[ProgressNode]) -> List[TreeNode]:
        return [ProgressTreeBuilder.to_tree_node(node) for node in nodes]
