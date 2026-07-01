"""Core runtime tree projection for OpenHCS progress displays."""

from __future__ import annotations

from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, List, Mapping, Optional, Sequence

from metaclass_registry import AutoRegisterMeta

from openhcs.core.progress import (
    ProgressChannel,
    ProgressEvent,
    is_failure_event,
    is_success_terminal_event,
    phase_channel,
    progress_channel_role,
)
from openhcs.core.progress.types import ProgressChannelRole
from openhcs.core.progress.projection import (
    ExecutionRuntimeProjection,
    PlateRuntimeProjection,
    build_execution_runtime_projection,
)
from openhcs.core.registry_strategies import (
    AlwaysMatchesContextMixin,
    MostDerivedContextStrategyMixin,
)


@dataclass(frozen=True, slots=True)
class RuntimeTreeNodeIdentity:
    """Typed identity inputs used by runtime tree node declarations."""

    node_id: str
    plate_name: str | None = None
    worker_slot: str | None = None
    axis_id: str | None = None
    step_index: int | None = None
    step_name: str | None = None


class RuntimeTreeNodeDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Semantic declaration for one runtime/progress tree node kind."""

    __registry_key__ = "node_kind"
    __skip_if_no_key__ = True
    __registry__: ClassVar[dict[str, type["RuntimeTreeNodeDeclarationBase"]]] = {}

    node_kind: ClassVar[str | None] = None
    sort_order: ClassVar[int] = 0
    preserve_existing_info: ClassVar[bool] = False

    @classmethod
    def require_node_kind(cls) -> str:
        if cls.node_kind is None:
            raise TypeError(f"{cls.__name__} does not declare a runtime node kind.")
        return cls.node_kind

    @classmethod
    def for_node_kind(cls, node_kind: str) -> type["RuntimeTreeNodeDeclarationBase"]:
        return cls.__registry__[node_kind]

    @classmethod
    def aggregate_percent(
        cls,
        *,
        node_percent: float,
        child_percents: Sequence[float],
    ) -> float:
        return node_percent

    @classmethod
    def info_for(cls, node: "RuntimeTreeNode") -> str:
        if cls.preserve_existing_info and node.info:
            return node.info
        return f"{node.percent:.1f}%"

    @classmethod
    def label_for(cls, identity: RuntimeTreeNodeIdentity) -> str:
        return identity.node_id

    @classmethod
    def to_abi_node_type(cls) -> str:
        return cls.require_node_kind()


class MeanPercentRuntimeTreeNode:
    """Runtime tree node whose percent is the mean of child percents."""

    @classmethod
    def aggregate_percent(
        cls,
        *,
        node_percent: float,
        child_percents: Sequence[float],
    ) -> float:
        if not child_percents:
            return 0.0
        return sum(child_percents) / len(child_percents)


class ExplicitPercentRuntimeTreeNode:
    """Runtime tree node whose percent is carried directly by the node."""

    @classmethod
    def aggregate_percent(
        cls,
        *,
        node_percent: float,
        child_percents: Sequence[float],
    ) -> float:
        return node_percent


class PlateProgressTreeNode(MeanPercentRuntimeTreeNode, RuntimeTreeNodeDeclarationBase):
    node_kind = "plate"
    sort_order = 0

    @classmethod
    def label_for(cls, identity: RuntimeTreeNodeIdentity) -> str:
        return f"📋 {identity.plate_name or identity.node_id}"


class WorkerProgressTreeNode(MeanPercentRuntimeTreeNode, RuntimeTreeNodeDeclarationBase):
    node_kind = "worker"
    sort_order = 10

    @classmethod
    def label_for(cls, identity: RuntimeTreeNodeIdentity) -> str:
        return f"Worker {identity.worker_slot or identity.node_id}"


class WellProgressTreeNode(ExplicitPercentRuntimeTreeNode, RuntimeTreeNodeDeclarationBase):
    node_kind = "well"
    sort_order = 20

    @classmethod
    def label_for(cls, identity: RuntimeTreeNodeIdentity) -> str:
        return f"[{identity.axis_id or identity.node_id}]"


class StepProgressTreeNode(ExplicitPercentRuntimeTreeNode, RuntimeTreeNodeDeclarationBase):
    node_kind = "step"
    sort_order = 30
    preserve_existing_info = True

    @classmethod
    def label_for(cls, identity: RuntimeTreeNodeIdentity) -> str:
        if identity.step_index is None:
            return identity.step_name or identity.node_id
        step_name = identity.step_name or f"Step {identity.step_index + 1}"
        return f"🔧 {identity.step_index + 1} - {step_name}"


class CompilationProgressTreeNode(
    ExplicitPercentRuntimeTreeNode,
    RuntimeTreeNodeDeclarationBase,
):
    node_kind = "compilation"
    sort_order = 40

    @classmethod
    def label_for(cls, identity: RuntimeTreeNodeIdentity) -> str:
        return f"[{identity.axis_id or identity.node_id}]"


@dataclass
class RuntimeTreeNode:
    """Pure core runtime tree node."""

    node_id: str
    node_type: str
    label: str
    status: str
    info: str
    execution_id: str | None = None
    percent: float = 0.0
    children: List["RuntimeTreeNode"] = field(default_factory=list)
    declaration: type[RuntimeTreeNodeDeclarationBase] | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.declaration is None:
            self.declaration = RuntimeTreeNodeDeclarationBase.for_node_kind(
                self.node_type
            )
        elif self.node_type != self.declaration.require_node_kind():
            raise ValueError(
                f"Runtime node type mismatch: {self.node_type!r} does not match "
                f"{self.declaration.require_node_kind()!r}."
            )

    @classmethod
    def from_declaration(
        cls,
        *,
        declaration: type[RuntimeTreeNodeDeclarationBase],
        identity: RuntimeTreeNodeIdentity,
        status: str,
        info: str,
        execution_id: str | None = None,
        percent: float = 0.0,
        children: Optional[List["RuntimeTreeNode"]] = None,
    ) -> "RuntimeTreeNode":
        return cls(
            node_id=identity.node_id,
            node_type=declaration.to_abi_node_type(),
            label=declaration.label_for(identity),
            status=status,
            info=info,
            execution_id=execution_id,
            percent=percent,
            children=children or [],
            declaration=declaration,
        )

    @property
    def require_declaration(self) -> type[RuntimeTreeNodeDeclarationBase]:
        if self.declaration is None:
            raise ValueError(f"Runtime node {self.node_id!r} has no declaration.")
        return self.declaration


@dataclass(frozen=True, slots=True)
class RuntimeTreeProjection:
    """Core runtime tree projection."""

    roots: tuple[RuntimeTreeNode, ...]


@dataclass(frozen=True, slots=True)
class RuntimeExecutionTopology:
    """Typed topology inputs used by runtime tree projection."""

    worker_assignments: Mapping[tuple[str, str], Mapping[str, Sequence[str]]]
    known_wells: Mapping[tuple[str, str], Sequence[str]]
    step_names: Mapping[tuple[str, str, str], Mapping[int, str]]


@dataclass(frozen=True, slots=True)
class RuntimeTreeWorkerStatusContext:
    well_nodes: Sequence[RuntimeTreeNode]
    execution_started: bool

    @property
    def failed_count(self) -> int:
        return sum(1 for node in self.well_nodes if node.status == "❌ Failed")

    @property
    def complete_count(self) -> int:
        return sum(1 for node in self.well_nodes if node.status == "✅ Complete")

    @property
    def queued_count(self) -> int:
        return sum(1 for node in self.well_nodes if node.status == "⏳ Queued")

    @property
    def active_count(self) -> int:
        return (
            len(self.well_nodes)
            - self.failed_count
            - self.complete_count
            - self.queued_count
        )


class RuntimeTreeWorkerStatusStrategy(
    MostDerivedContextStrategyMixin[RuntimeTreeWorkerStatusContext],
):
    """Core strategy for projecting a worker runtime tree status."""

    def status(self, context: RuntimeTreeWorkerStatusContext) -> str:
        raise NotImplementedError


class ActiveWorkerStatus(AlwaysMatchesContextMixin[RuntimeTreeWorkerStatusContext], RuntimeTreeWorkerStatusStrategy):
    strategy_key = "active"

    def status(self, context: RuntimeTreeWorkerStatusContext) -> str:
        return f"⚙️ {context.active_count} active"


class FailedWorkerStatus(ActiveWorkerStatus):
    strategy_key = "failed"

    def matches(self, context: RuntimeTreeWorkerStatusContext) -> bool:
        return context.failed_count > 0

    def status(self, context: RuntimeTreeWorkerStatusContext) -> str:
        return f"❌ {context.failed_count} failed"


class CompleteWorkerStatus(ActiveWorkerStatus):
    strategy_key = "complete"

    def matches(self, context: RuntimeTreeWorkerStatusContext) -> bool:
        return context.complete_count == len(context.well_nodes)

    def status(self, context: RuntimeTreeWorkerStatusContext) -> str:
        del context
        return "✅ Complete"


class QueuedWorkerStatus(ActiveWorkerStatus):
    strategy_key = "queued"

    def matches(self, context: RuntimeTreeWorkerStatusContext) -> bool:
        return context.queued_count == len(context.well_nodes)

    def status(self, context: RuntimeTreeWorkerStatusContext) -> str:
        return "⚙️ Starting" if context.execution_started else "⏳ Queued"


@dataclass(frozen=True, slots=True)
class RuntimeTreeEventStatusContext:
    event: ProgressEvent | None
    missing_status: str
    active_status: str
    success_status: str


class RuntimeTreeEventStatusStrategy(
    MostDerivedContextStrategyMixin[RuntimeTreeEventStatusContext],
):
    """Core strategy for projecting a node status from a progress event."""

    def status_and_percent(self, context: RuntimeTreeEventStatusContext) -> tuple[str, float]:
        raise NotImplementedError


class ActiveEventStatus(AlwaysMatchesContextMixin[RuntimeTreeEventStatusContext], RuntimeTreeEventStatusStrategy):
    strategy_key = "active"

    def status_and_percent(self, context: RuntimeTreeEventStatusContext) -> tuple[str, float]:
        if context.event is None:
            raise ValueError("Active event status requires a progress event.")
        return context.active_status, context.event.percent


class MissingEventStatus(ActiveEventStatus):
    strategy_key = "missing"

    def matches(self, context: RuntimeTreeEventStatusContext) -> bool:
        return context.event is None

    def status_and_percent(self, context: RuntimeTreeEventStatusContext) -> tuple[str, float]:
        return context.missing_status, 0.0


class FailedEventStatus(ActiveEventStatus):
    strategy_key = "failed"

    def matches(self, context: RuntimeTreeEventStatusContext) -> bool:
        return context.event is not None and is_failure_event(context.event)

    def status_and_percent(self, context: RuntimeTreeEventStatusContext) -> tuple[str, float]:
        if context.event is None:
            raise ValueError("Failed event status requires a progress event.")
        return "❌ Failed", context.event.percent


class SuccessTerminalEventStatus(ActiveEventStatus):
    strategy_key = "success_terminal"

    def matches(self, context: RuntimeTreeEventStatusContext) -> bool:
        return context.event is not None and is_success_terminal_event(context.event)

    def status_and_percent(self, context: RuntimeTreeEventStatusContext) -> tuple[str, float]:
        if context.event is None:
            raise ValueError("Success event status requires a progress event.")
        return context.success_status, context.event.percent


@dataclass(frozen=True, slots=True)
class RuntimeTreeStepStatusContext:
    step_event: ProgressEvent | None


class RuntimeTreeStepStatusStrategy(
    MostDerivedContextStrategyMixin[RuntimeTreeStepStatusContext],
):
    """Core strategy for projecting the active step child status."""

    def step_name_status_percent(
        self,
        *,
        context: RuntimeTreeStepStatusContext,
        fallback_step_name: str,
    ) -> tuple[str, str, float]:
        raise NotImplementedError


class MissingStepStatus(AlwaysMatchesContextMixin[RuntimeTreeStepStatusContext], RuntimeTreeStepStatusStrategy):
    strategy_key = "missing"

    def step_name_status_percent(
        self,
        *,
        context: RuntimeTreeStepStatusContext,
        fallback_step_name: str,
    ) -> tuple[str, str, float]:
        del context
        return fallback_step_name, "⏳ Starting", 0.0


class ActiveStepStatus(MissingStepStatus):
    strategy_key = "active"

    def matches(self, context: RuntimeTreeStepStatusContext) -> bool:
        return context.step_event is not None

    def step_name_status_percent(
        self,
        *,
        context: RuntimeTreeStepStatusContext,
        fallback_step_name: str,
    ) -> tuple[str, str, float]:
        del fallback_step_name
        if context.step_event is None:
            raise ValueError("Active step status requires a progress event.")
        return (
            context.step_event.step_name,
            f"{context.step_event.completed}/{context.step_event.total} groups",
            context.step_event.percent,
        )


class FailedStepStatus(ActiveStepStatus):
    strategy_key = "failed"

    def matches(self, context: RuntimeTreeStepStatusContext) -> bool:
        return context.step_event is not None and is_failure_event(context.step_event)

    def step_name_status_percent(
        self,
        *,
        context: RuntimeTreeStepStatusContext,
        fallback_step_name: str,
    ) -> tuple[str, str, float]:
        del fallback_step_name
        if context.step_event is None:
            raise ValueError("Failed step status requires a progress event.")
        return context.step_event.step_name, "❌ Failed", context.step_event.percent


@dataclass(frozen=True, slots=True)
class RuntimeTreeExecutionModeContext:
    execution_id: str
    plate_id: str
    events: Sequence[ProgressEvent]
    worker_assignments: Mapping[tuple[str, str], Mapping[str, Sequence[str]]]

    @property
    def has_worker_topology(self) -> bool:
        return (self.execution_id, self.plate_id) in self.worker_assignments

    @property
    def has_compile_event(self) -> bool:
        return any(
            phase_channel(event.phase) is ProgressChannel.COMPILE
            for event in self.events
        )

    @property
    def has_execution_event(self) -> bool:
        return any(
            progress_channel_role(phase_channel(event.phase))
            is ProgressChannelRole.EXECUTION
            for event in self.events
        )


class RuntimeTreeExecutionModeStrategy(
    MostDerivedContextStrategyMixin[RuntimeTreeExecutionModeContext],
):
    """Core strategy for classifying a progress snapshot as execution-mode."""

    def is_execution_mode(self, context: RuntimeTreeExecutionModeContext) -> bool:
        raise NotImplementedError


class CompileModeSnapshot(AlwaysMatchesContextMixin[RuntimeTreeExecutionModeContext], RuntimeTreeExecutionModeStrategy):
    strategy_key = "compile"

    def is_execution_mode(self, context: RuntimeTreeExecutionModeContext) -> bool:
        del context
        return False


class ExecutionEventSnapshot(CompileModeSnapshot):
    strategy_key = "execution_event"

    def matches(self, context: RuntimeTreeExecutionModeContext) -> bool:
        return context.has_execution_event

    def is_execution_mode(self, context: RuntimeTreeExecutionModeContext) -> bool:
        del context
        return True


class InitOnlyExecutionSnapshot(CompileModeSnapshot):
    strategy_key = "init_only_execution"

    def matches(self, context: RuntimeTreeExecutionModeContext) -> bool:
        return (
            context.has_worker_topology
            and not context.has_compile_event
            and not context.has_execution_event
        )

    def is_execution_mode(self, context: RuntimeTreeExecutionModeContext) -> bool:
        del context
        return True


class RuntimeTreeStatusProjector:
    """Own percent aggregation and parent status projection."""

    def finalize_plate_node(
        self,
        node: RuntimeTreeNode,
        *,
        plate_projection: PlateRuntimeProjection | None,
    ) -> None:
        self.aggregate_percent_recursive(node)
        if plate_projection is not None and plate_projection.status_label:
            node.status = plate_projection.status_label
            node.percent = plate_projection.percent
        elif self.all_leaves_queued(node):
            node.status = "⏳ Queued"
        self.apply_node_info_text(node)

    def aggregate_percent_recursive(self, node: RuntimeTreeNode) -> float:
        if not node.children:
            return node.percent
        child_values = [
            self.aggregate_percent_recursive(child) for child in node.children
        ]
        node.percent = node.require_declaration.aggregate_percent(
            node_percent=node.percent,
            child_percents=child_values,
        )
        return node.percent

    def apply_node_info_text(self, node: RuntimeTreeNode) -> None:
        node.info = node.require_declaration.info_for(node)
        for child in node.children:
            self.apply_node_info_text(child)

    @staticmethod
    def all_leaves_queued(node: RuntimeTreeNode) -> bool:
        if not node.children:
            return node.status == "⏳ Queued"
        return all(
            RuntimeTreeStatusProjector.all_leaves_queued(child)
            for child in node.children
        )

    @staticmethod
    def has_failed_descendant(node: RuntimeTreeNode) -> bool:
        if node.status.startswith("❌"):
            return True
        return any(
            RuntimeTreeStatusProjector.has_failed_descendant(child)
            for child in node.children
        )


class RuntimeTreeNodeFactory:
    """Own construction of runtime tree model nodes."""

    def make_runtime_node(
        self,
        *,
        declaration: type[RuntimeTreeNodeDeclarationBase],
        identity: RuntimeTreeNodeIdentity,
        status: str,
        info: str,
        execution_id: str | None = None,
        percent: float = 0.0,
        children: Optional[List[RuntimeTreeNode]] = None,
    ) -> RuntimeTreeNode:
        return RuntimeTreeNode.from_declaration(
            declaration=declaration,
            identity=identity,
            status=status,
            info=info,
            execution_id=execution_id,
            percent=percent,
            children=children,
        )

    def make_step_node(
        self,
        *,
        axis_id: str,
        step_idx: int,
        step_name: str,
        status: str,
        info: str,
        percent: float,
    ) -> RuntimeTreeNode:
        return self.make_runtime_node(
            declaration=StepProgressTreeNode,
            identity=RuntimeTreeNodeIdentity(
                node_id=f"{axis_id}_step_{step_idx}",
                axis_id=axis_id,
                step_index=step_idx,
                step_name=step_name,
            ),
            status=status,
            info=info,
            percent=percent,
        )

    def make_well_progress_node(
        self,
        *,
        axis_id: str,
        status: str,
        percent: float,
        children: List[RuntimeTreeNode],
    ) -> RuntimeTreeNode:
        return self.make_runtime_node(
            declaration=WellProgressTreeNode,
            identity=RuntimeTreeNodeIdentity(node_id=axis_id, axis_id=axis_id),
            status=status,
            info="",
            percent=percent,
            children=children,
        )


class RuntimeTreeProjectionBuilder:
    """Transforms progress snapshots into hierarchical runtime tree nodes."""

    def __init__(
        self,
        status_projector: RuntimeTreeStatusProjector | None = None,
        node_factory: RuntimeTreeNodeFactory | None = None,
    ) -> None:
        self.status_projector = status_projector or RuntimeTreeStatusProjector()
        self.node_factory = node_factory or RuntimeTreeNodeFactory()

    def build(
        self,
        *,
        executions: Mapping[str, Sequence[ProgressEvent]],
        topology: RuntimeExecutionTopology,
        get_plate_name: Callable[[str, str | None], str],
        runtime_projection: ExecutionRuntimeProjection | None = None,
    ) -> RuntimeTreeProjection:
        effective_runtime_projection = (
            runtime_projection
            if runtime_projection is not None
            else build_execution_runtime_projection(
                {
                    execution_id: list(events)
                    for execution_id, events in executions.items()
                }
            )
        )
        return RuntimeTreeProjection(
            roots=tuple(
                self.build_progress_tree(
                    executions=executions,
                    runtime_projection=effective_runtime_projection,
                    worker_assignments=topology.worker_assignments,
                    known_wells=topology.known_wells,
                    step_names=topology.step_names,
                    get_plate_name=get_plate_name,
                )
            )
        )

    def build_progress_tree(
        self,
        *,
        executions: Mapping[str, Sequence[ProgressEvent]],
        runtime_projection: ExecutionRuntimeProjection,
        worker_assignments: Mapping[tuple[str, str], Mapping[str, Sequence[str]]],
        known_wells: Mapping[tuple[str, str], Sequence[str]],
        step_names: Mapping[tuple[str, str, str], Mapping[int, str]],
        get_plate_name: Callable[[str, str | None], str],
    ) -> List[RuntimeTreeNode]:
        events_by_plate: Dict[tuple[str, str], List[ProgressEvent]] = defaultdict(list)
        for exec_id, events_list in executions.items():
            for event in events_list:
                events_by_plate[(exec_id, event.plate_id)].append(event)

        nodes_by_plate: Dict[str, tuple[float, RuntimeTreeNode]] = {}
        for (exec_id, plate_id), events in events_by_plate.items():
            if not events:
                continue
            latest_timestamp = max((event.timestamp for event in events), default=0.0)
            plate_name = get_plate_name(plate_id, exec_id)
            plate_projection = runtime_projection.get_plate(plate_id, exec_id)
            is_executing = self._is_execution_mode(
                execution_id=exec_id,
                plate_id=plate_id,
                events=events,
                worker_assignments=worker_assignments,
            )
            missing_execution_topology = (
                is_executing
                and (exec_id, plate_id) not in worker_assignments
            )
            if missing_execution_topology:
                children = []
                plate_percent = max(
                    (
                        event.percent
                        for event in events
                        if progress_channel_role(phase_channel(event.phase))
                        is ProgressChannelRole.EXECUTION
                    ),
                    default=0.0,
                )
            elif is_executing:
                children = self._build_worker_children(
                    execution_id=exec_id,
                    plate_id=plate_id,
                    events=events,
                    worker_assignments=worker_assignments,
                    step_names=step_names,
                )
                plate_percent = 0.0
            else:
                children = self._build_compilation_children(
                    execution_id=exec_id,
                    plate_id=plate_id,
                    events=events,
                    known_wells=known_wells,
                )
                plate_percent = 0.0

            plate_node = self.node_factory.make_runtime_node(
                declaration=PlateProgressTreeNode,
                identity=RuntimeTreeNodeIdentity(
                    node_id=plate_id,
                    plate_name=plate_name,
                ),
                status="⚙️ Executing" if is_executing else "⏳ Compiling",
                info="",
                execution_id=exec_id,
                percent=plate_percent,
                children=children,
            )
            self.status_projector.finalize_plate_node(
                plate_node,
                plate_projection=plate_projection,
            )
            existing = nodes_by_plate.get(plate_id)
            if existing is None or latest_timestamp > existing[0]:
                nodes_by_plate[plate_id] = (latest_timestamp, plate_node)

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
        worker_assignments: Mapping[tuple[str, str], Mapping[str, Sequence[str]]],
        step_names: Mapping[tuple[str, str, str], Mapping[int, str]],
    ) -> List[RuntimeTreeNode]:
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

        worker_nodes: List[RuntimeTreeNode] = []
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
            status_context = RuntimeTreeWorkerStatusContext(
                well_nodes=well_nodes,
                execution_started=execution_started,
            )
            worker_status = RuntimeTreeWorkerStatusStrategy.for_context(
                status_context
            ).status(status_context)
            worker_nodes.append(
                self.node_factory.make_runtime_node(
                    declaration=WorkerProgressTreeNode,
                    identity=RuntimeTreeNodeIdentity(
                        node_id=worker_slot,
                        worker_slot=worker_slot,
                    ),
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
        known_wells: Mapping[tuple[str, str], Sequence[str]],
    ) -> List[RuntimeTreeNode]:
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

        compilation_nodes: List[RuntimeTreeNode] = []
        for axis_id in axis_ids:
            compile_event = compile_by_axis.get(axis_id)
            status_context = RuntimeTreeEventStatusContext(
                event=compile_event,
                missing_status="⏳ Compiling",
                active_status="⏳ Compiling",
                success_status="✅ Compiled",
            )
            status, percent = RuntimeTreeEventStatusStrategy.for_context(
                status_context
            ).status_and_percent(status_context)

            compilation_nodes.append(
                self.node_factory.make_runtime_node(
                    declaration=CompilationProgressTreeNode,
                    identity=RuntimeTreeNodeIdentity(
                        node_id=axis_id,
                        axis_id=axis_id,
                    ),
                    status=status,
                    info="",
                    percent=percent,
                )
            )
        return compilation_nodes

    @staticmethod
    def _partition_events_by_channel(
        events: Sequence[ProgressEvent],
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
        step_names: Mapping[int, str],
    ) -> RuntimeTreeNode:
        active_status = (
            "⚙️"
            if pipeline_event is None
            else f"⚙️ {pipeline_event.step_name}"
        )
        status_context = RuntimeTreeEventStatusContext(
            event=pipeline_event,
            missing_status="⏳ Queued",
            active_status=active_status,
            success_status="✅ Complete",
        )
        status, percent = RuntimeTreeEventStatusStrategy.for_context(
            status_context
        ).status_and_percent(status_context)

        children: List[RuntimeTreeNode] = []
        if pipeline_event is not None and pipeline_event.total > 0:
            current_step_idx = pipeline_event.completed
            total_steps = pipeline_event.total

            if current_step_idx < total_steps:
                fallback_step_name = step_names.get(
                    current_step_idx, f"Step {current_step_idx + 1}"
                )
                step_context = RuntimeTreeStepStatusContext(step_event=step_event)
                step_name, step_status, step_percent = (
                    RuntimeTreeStepStatusStrategy.for_context(
                        step_context
                    ).step_name_status_percent(
                        context=step_context,
                        fallback_step_name=fallback_step_name,
                    )
                )

                children.append(
                    self.node_factory.make_step_node(
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
                    self.node_factory.make_step_node(
                        axis_id=axis_id,
                        step_idx=step_idx,
                        step_name=step_name,
                        status="✅ Complete",
                        info="100.0%",
                        percent=100.0,
                    )
                )

            for step_idx in range(current_step_idx + 1, total_steps):
                step_name = step_names.get(step_idx, f"Step {step_idx + 1}")
                children.append(
                    self.node_factory.make_step_node(
                        axis_id=axis_id,
                        step_idx=step_idx,
                        step_name=step_name,
                        status="⏳ Pending",
                        info="0.0%",
                        percent=0.0,
                    )
                )

        return self.node_factory.make_well_progress_node(
            axis_id=axis_id,
            status=status,
            percent=percent,
            children=children,
        )

    @staticmethod
    def _is_execution_mode(
        *,
        execution_id: str,
        plate_id: str,
        events: Sequence[ProgressEvent],
        worker_assignments: Mapping[tuple[str, str], Mapping[str, Sequence[str]]],
    ) -> bool:
        context = RuntimeTreeExecutionModeContext(
            execution_id=execution_id,
            plate_id=plate_id,
            events=events,
            worker_assignments=worker_assignments,
        )
        return RuntimeTreeExecutionModeStrategy.for_context(
            context
        ).is_execution_mode(context)


__all__ = (
    "CompilationProgressTreeNode",
    "ExplicitPercentRuntimeTreeNode",
    "MeanPercentRuntimeTreeNode",
    "PlateProgressTreeNode",
    "RuntimeExecutionTopology",
    "RuntimeTreeNode",
    "RuntimeTreeNodeDeclarationBase",
    "RuntimeTreeNodeFactory",
    "RuntimeTreeNodeIdentity",
    "RuntimeTreeProjection",
    "RuntimeTreeProjectionBuilder",
    "RuntimeTreeStatusProjector",
    "StepProgressTreeNode",
    "WellProgressTreeNode",
    "WorkerProgressTreeNode",
)
