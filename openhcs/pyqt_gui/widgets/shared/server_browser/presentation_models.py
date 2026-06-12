"""Server-browser state + presentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Callable, Dict, Iterable, List

from PyQt6.QtWidgets import QTreeWidgetItem

from openhcs.core.progress import ProgressEvent, phase_channel
from openhcs.core.progress.types import ProgressChannelRole
from pyqt_reactive.services import (
    BaseServerInfo,
    ExecutionServerInfo,
    GenericServerInfo,
    ViewerServerInfo,
)

from .progress_tree_builder import ProgressNode


class ProgressTopologyState:
    """Tracks and validates execution topology claims from progress events."""

    def __init__(self) -> None:
        self.known_wells: Dict[tuple[str, str], List[str]] = {}
        self.worker_assignments: Dict[tuple[str, str], Dict[str, List[str]]] = {}
        self.seen_execution_ids: set[str] = set()
        # Track step names per well: {(exec_id, plate_id, axis_id): {step_index: step_name}}
        self.step_names: Dict[tuple[str, str, str], Dict[int, str]] = {}

    def register_event(self, event: ProgressEvent) -> None:
        execution_id = event.execution_id
        plate_id = event.plate_id
        topology_key = (execution_id, plate_id)

        self.seen_execution_ids.add(execution_id)

        if event.total_wells:
            self.known_wells[topology_key] = list(event.total_wells)
        if event.worker_assignments is not None:
            self.worker_assignments[topology_key] = event.worker_assignments

        # Capture step names from INIT event (emitted once per execution)
        if (
            event.step_name == ""
            and event.phase.value == "init"
            and event.axis_id == ""
        ):
            if event.step_names:
                # step_names applies to all wells in this execution
                for well_id in self.known_wells.get(topology_key, []):
                    well_key = (execution_id, plate_id, well_id)
                    self.step_names[well_key] = {
                        i: name for i, name in enumerate(event.step_names)
                    }

        event_channel = phase_channel(event.phase)
        if (
            event_channel.role is ProgressChannelRole.EXECUTION
            and event.axis_id
            and topology_key not in self.worker_assignments
        ):
            raise ValueError(
                f"Execution event arrived without worker_assignments for plate '{plate_id}'"
            )

        if event.worker_slot is not None:
            if topology_key not in self.worker_assignments:
                raise ValueError(
                    f"Worker event arrived before assignments for plate '{plate_id}'"
                )
            assignments = self.worker_assignments[topology_key]
            if event.worker_slot not in assignments:
                raise ValueError(
                    f"Unknown worker slot '{event.worker_slot}' for plate '{plate_id}'"
                )
            expected = sorted(assignments[event.worker_slot])
            actual = sorted(event.owned_wells or [])
            if actual != expected:
                raise ValueError(
                    f"Worker claim mismatch for slot '{event.worker_slot}': expected={expected}, got={actual}"
                )

    def clear_execution(self, execution_id: str) -> None:
        self.seen_execution_ids.discard(execution_id)
        keys_to_remove = [
            key for key in self.known_wells.keys() if key[0] == execution_id
        ]
        for key in keys_to_remove:
            self.known_wells.pop(key, None)
            self.worker_assignments.pop(key, None)

    def clear_all(self) -> None:
        self.known_wells.clear()
        self.worker_assignments.clear()
        self.seen_execution_ids.clear()


@dataclass(frozen=True)
class ExecutionServerSummary:
    """Derived status for execution-server top-level row."""

    status_text: str
    info_text: str


def summarize_execution_server(nodes: Iterable[ProgressNode]) -> ExecutionServerSummary:
    node_list = list(nodes)
    if not node_list:
        return ExecutionServerSummary(status_text="✅ Idle", info_text="")

    queued = sum(1 for node in node_list if node.status == "⏳ Queued")
    compiling = sum(1 for node in node_list if node.status == "⏳ Compiling")
    executing = sum(1 for node in node_list if node.status == "⚙️ Executing")
    compiled = sum(1 for node in node_list if node.status == "✅ Compiled")
    completed = sum(1 for node in node_list if node.status == "✅ Complete")
    failed = sum(1 for node in node_list if node.status.startswith("❌"))
    overall_pct = sum(node.percent for node in node_list) / len(node_list)

    parts: list[str] = []
    if queued > 0:
        parts.append(f"⏳ {queued} queued")
    if compiling > 0:
        parts.append(f"⏳ {compiling} compiling")
    if executing > 0:
        parts.append(f"⚙️ {executing} executing")
    if compiled > 0:
        parts.append(f"✅ {compiled} compiled")
    if completed > 0:
        parts.append(f"✅ {completed} complete")
    if failed > 0:
        parts.append(f"❌ {failed} failed")

    status_text = ", ".join(parts) if parts else "✅ Idle"
    info_text = f"Avg: {overall_pct:.1f}% | {len(node_list)} plates"
    return ExecutionServerSummary(status_text=status_text, info_text=info_text)


class ServerRowPresenter:
    """Type-dispatched server row rendering and child population."""

    def __init__(
        self,
        *,
        create_tree_item: Callable[[str, str, str, dict], QTreeWidgetItem],
        update_execution_server_item: Callable[[QTreeWidgetItem, dict], None],
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
            return self._create_tree_item(server_text, "🚀 Starting", "", info.raw)
        info_text = f"{len(info.workers)} active workers" if info.workers else ""
        return self._create_tree_item(server_text, "✅ Idle", info_text, info.raw)

    @render_server.register
    def _(self, info: ViewerServerInfo, status_icon: str) -> QTreeWidgetItem:
        kind_name = info.viewer_kind.name.title()
        display_text = f"Port {info.port} - {kind_name} Viewer"
        info_text = ""
        if info.memory_mb is not None:
            info_text = f"Mem: {info.memory_mb:.0f}MB"
            if info.cpu_percent is not None:
                info_text += f" | CPU: {info.cpu_percent:.1f}%"
        return self._create_tree_item(display_text, status_icon, info_text, info.raw)

    @render_server.register
    def _(self, info: GenericServerInfo, status_icon: str) -> QTreeWidgetItem:
        return self._create_tree_item(
            f"Port {info.port} - {info.server_name}", status_icon, "", info.raw
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
        self._update_execution_server_item(server_item, info.raw)
        return server_item.childCount() > 0

    @populate_server_children.register
    def _(self, info: ViewerServerInfo, server_item: QTreeWidgetItem) -> bool:
        return False

    @populate_server_children.register
    def _(self, info: GenericServerInfo, server_item: QTreeWidgetItem) -> bool:
        return False
