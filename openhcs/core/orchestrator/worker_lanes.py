"""Worker-lane identity and assignment planning for compiled executions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, TypeAlias

from openhcs.core.compiled_execution import CompiledExecutionBundle
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.debug import (
    DebugExecutionContext,
    DebugExecutionPolicy,
    DebugSinkInstallRequest,
)
from openhcs.core.orchestrator.execution_result import RuntimeObservationMode
from openhcs.core.progress import ProgressExecutionContext


TransportAxisContexts: TypeAlias = List[tuple[str, ProcessingContext]]
ForkAxisContextKeys: TypeAlias = List[str]
LaneAxisContextPayload: TypeAlias = TransportAxisContexts | ForkAxisContextKeys
WorkerLaneAxisContexts: TypeAlias = List[tuple[str, LaneAxisContextPayload]]
WorkerLaneContextMap: TypeAlias = Dict[str, WorkerLaneAxisContexts]


class ForkInheritedWorkerExecutionState:
    """Compiled execution state inherited by forked worker processes."""

    _current: CompiledExecutionBundle | None = None

    @classmethod
    def install(cls, execution_bundle: CompiledExecutionBundle) -> None:
        cls._current = execution_bundle

    @classmethod
    def clear(cls) -> None:
        cls._current = None

    @classmethod
    def require_current(cls) -> CompiledExecutionBundle:
        if cls._current is None:
            raise RuntimeError(
                "ForkInheritedWorkerExecutionState is not installed in this worker."
            )
        return cls._current

    @classmethod
    def resolve_lane_contexts(
        cls,
        lane_axis_context_keys: List[tuple[str, List[str]]],
    ) -> List[tuple[str, TransportAxisContexts]]:
        contexts = cls.require_current().runtime_contexts
        return [
            (
                axis_id,
                [(context_key, contexts[context_key]) for context_key in context_keys],
            )
            for axis_id, context_keys in lane_axis_context_keys
        ]


@dataclass(frozen=True, slots=True)
class WorkerLaneExecutionContext(ProgressExecutionContext):
    """Execution identity for one deterministic worker lane."""

    debug_execution_policy: DebugExecutionPolicy
    worker_slot: str
    owned_wells: tuple[str, ...]

    def install_debug_sink(self, processing_context: DebugExecutionContext) -> None:
        self.debug_execution_policy.install_context_sink(
            DebugSinkInstallRequest(
                context=processing_context,
                execution_id=self.execution_id,
                plate_id=self.plate_id,
                worker_slot=self.worker_slot,
                owned_wells=self.owned_wells,
            )
        )


@dataclass(frozen=True, slots=True)
class WorkerAssignmentPlan:
    """Validated axis ownership and lane context projection for workers."""

    worker_assignments: Dict[str, List[str]]
    lane_axis_contexts: WorkerLaneContextMap

    def active_lane_items(self) -> tuple[tuple[str, WorkerLaneAxisContexts], ...]:
        """Return worker lanes that own at least one axis context."""

        return tuple(
            (worker_slot, lane_contexts)
            for worker_slot, lane_contexts in self.lane_axis_contexts.items()
            if lane_contexts
        )

    def owned_wells(self, worker_slot: str) -> tuple[str, ...]:
        """Return the deterministic well ownership for one worker slot."""

        return tuple(self.worker_assignments[worker_slot])


@dataclass(frozen=True, slots=True)
class WorkerLaneExecutionPlan(ProgressExecutionContext):
    """Shared execution plan for deterministic worker-lane runners."""

    debug_execution_policy: DebugExecutionPolicy
    assignments: WorkerAssignmentPlan
    runtime_observation_mode: RuntimeObservationMode

    def active_lane_items(self) -> tuple[tuple[str, WorkerLaneAxisContexts], ...]:
        """Return worker lanes that own at least one axis context."""

        return self.assignments.active_lane_items()

    def lane_context(self, worker_slot: str) -> WorkerLaneExecutionContext:
        return WorkerLaneExecutionContext(
            execution_id=self.execution_id,
            plate_id=self.plate_id,
            debug_execution_policy=self.debug_execution_policy,
            worker_slot=worker_slot,
            owned_wells=self.assignments.owned_wells(worker_slot),
        )


class CompiledContextLanePlanner:
    """Project compiled context keys into deterministic worker lanes."""

    def __init__(
        self,
        *,
        actual_max_workers: int,
        fork_inherited_execution: bool,
    ) -> None:
        self._actual_max_workers = actual_max_workers
        self._fork_inherited_execution = fork_inherited_execution

    def plan(
        self,
        contexts_snapshot: Mapping[str, ProcessingContext],
        worker_assignments: Optional[Dict[str, List[str]]],
    ) -> WorkerAssignmentPlan:
        contexts_by_axis = self._contexts_by_axis(contexts_snapshot)
        resolved_assignments = self._resolved_assignments(
            contexts_by_axis,
            worker_assignments,
        )
        axis_to_worker = self._axis_to_worker(resolved_assignments)
        lane_axis_contexts = self._lane_axis_contexts(
            contexts_snapshot,
            contexts_by_axis,
            resolved_assignments,
            axis_to_worker,
        )
        return WorkerAssignmentPlan(
            worker_assignments=resolved_assignments,
            lane_axis_contexts=lane_axis_contexts,
        )

    def _contexts_by_axis(
        self,
        contexts_snapshot: Mapping[str, ProcessingContext],
    ) -> Dict[str, List[str]]:
        contexts_by_axis: dict[str, list[str]] = defaultdict(list)
        for context_key in contexts_snapshot:
            axis_id = context_key.split("__combo_")[0]
            contexts_by_axis[axis_id].append(context_key)
        return dict(contexts_by_axis)

    def _resolved_assignments(
        self,
        contexts_by_axis: Mapping[str, List[str]],
        worker_assignments: Optional[Dict[str, List[str]]],
    ) -> Dict[str, List[str]]:
        if worker_assignments is None:
            return self._default_assignments(contexts_by_axis)

        resolved_assignments = {
            worker_slot: list(owned)
            for worker_slot, owned in worker_assignments.items()
            if owned
        }
        self._validate_assignments(contexts_by_axis, resolved_assignments)
        return resolved_assignments

    def _default_assignments(
        self,
        contexts_by_axis: Mapping[str, List[str]],
    ) -> Dict[str, List[str]]:
        generated: Dict[str, List[str]] = {
            f"worker_{idx}": [] for idx in range(self._actual_max_workers)
        }
        for idx, axis_id in enumerate(sorted(contexts_by_axis.keys())):
            generated[f"worker_{idx % self._actual_max_workers}"].append(axis_id)

        resolved_assignments = {
            worker_slot: owned for worker_slot, owned in generated.items() if owned
        }
        self._validate_assignments(contexts_by_axis, resolved_assignments)
        return resolved_assignments

    def _validate_assignments(
        self,
        contexts_by_axis: Mapping[str, List[str]],
        worker_assignments: Mapping[str, List[str]],
    ) -> None:
        expected_axis_ids = set(contexts_by_axis.keys())
        all_assigned_axis_ids: list[str] = []
        for owned in worker_assignments.values():
            all_assigned_axis_ids.extend(owned)

        if len(all_assigned_axis_ids) != len(set(all_assigned_axis_ids)):
            raise RuntimeError(
                f"Duplicate axis ownership detected in worker_assignments: {dict(worker_assignments)}"
            )
        if set(all_assigned_axis_ids) != expected_axis_ids:
            raise RuntimeError(
                f"worker_assignments mismatch. expected={sorted(expected_axis_ids)}, got={sorted(all_assigned_axis_ids)}"
            )

    def _axis_to_worker(
        self,
        worker_assignments: Mapping[str, List[str]],
    ) -> Dict[str, str]:
        axis_to_worker: Dict[str, str] = {}
        for worker_slot, owned in worker_assignments.items():
            for axis_id in owned:
                axis_to_worker[axis_id] = worker_slot
        return axis_to_worker

    def _lane_axis_contexts(
        self,
        contexts_snapshot: Mapping[str, ProcessingContext],
        contexts_by_axis: Mapping[str, List[str]],
        worker_assignments: Mapping[str, List[str]],
        axis_to_worker: Mapping[str, str],
    ) -> WorkerLaneContextMap:
        lane_axis_contexts: WorkerLaneContextMap = {
            worker_slot: [] for worker_slot in worker_assignments.keys()
        }

        for axis_id, axis_context_keys in contexts_by_axis.items():
            worker_slot = axis_to_worker[axis_id]
            if self._fork_inherited_execution:
                lane_axis_contexts[worker_slot].append(
                    (axis_id, list(axis_context_keys))
                )
            else:
                lane_axis_contexts[worker_slot].append(
                    (
                        axis_id,
                        [
                            (context_key, contexts_snapshot[context_key])
                            for context_key in axis_context_keys
                        ],
                    )
                )
        return lane_axis_contexts
