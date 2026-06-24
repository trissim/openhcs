"""Compiled plate execution flow for :class:`PipelineOrchestrator`."""

from __future__ import annotations

import logging
import time
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, TYPE_CHECKING

from openhcs.constants.constants import OrchestratorState
from openhcs.core.compiled_execution import CompiledExecutionBundle
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.context.processing_context import ProcessingContext, RequiredVisualizer
from openhcs.core.debug import DebugExecutionPolicy
from openhcs.core.orchestrator.analysis_consolidation import AnalysisConsolidationPlan
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeObservationMode,
)
from openhcs.core.orchestrator.worker_execution import (
    WorkerExecutorFactory,
)
from openhcs.core.orchestrator.worker_lanes import (
    WorkerAssignmentPlan,
    WorkerLaneExecutionPlan,
)
from openhcs.core.progress import (
    ProgressExecutionContext,
    ProgressEventPayload,
    ProgressPhase,
    ProgressQueue,
    ProgressStatus,
    create_event,
    set_progress_queue,
)
from openhcs.core.steps.abstract import AbstractStep

if TYPE_CHECKING:
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator


logger = logging.getLogger(__name__)


class ExecutionVisualizer(Protocol):
    """Visualizer contract used by compiled plate execution."""

    port: int
    persistent: bool
    is_running: bool

    def clear_viewer_state(self) -> bool:
        """Clear all viewer state before a new execution."""

    def stop_viewer(self) -> None:
        """Stop the viewer process/window."""


@dataclass(frozen=True, slots=True)
class CompiledPlateExecutionRequest(ProgressExecutionContext):
    """Public execute-compiled-plate call normalized into one request record."""

    pipeline_definition: List[AbstractStep]
    compiled_contexts: Dict[str, ProcessingContext]
    max_workers: Optional[int]
    visualizer: ExecutionVisualizer | None
    log_file_base: Optional[str]
    progress_queue: ProgressQueue | None
    worker_assignments: Optional[Dict[str, List[str]]]
    execution_bundle: Optional[CompiledExecutionBundle]
    runtime_observation_mode: RuntimeObservationMode
    debug_execution_policy: DebugExecutionPolicy

    def execution_bundle_for(
        self,
        validated: "ValidatedCompiledPlateExecution",
    ) -> CompiledExecutionBundle:
        if self.execution_bundle is not None:
            return self.execution_bundle
        if self.worker_assignments is None:
            return CompiledExecutionBundle.from_unassigned_runtime_contexts(
                pipeline_definition=validated.pipeline_definition,
                runtime_contexts=validated.compiled_contexts,
            )
        return CompiledExecutionBundle.from_runtime_contexts(
            pipeline_definition=validated.pipeline_definition,
            runtime_contexts=validated.compiled_contexts,
            worker_assignments=self.worker_assignments,
        )

    def worker_assignments_for(
        self,
        execution_bundle: CompiledExecutionBundle,
    ) -> Optional[Dict[str, List[str]]]:
        if self.worker_assignments is not None:
            return self.worker_assignments
        if execution_bundle.worker_assignments:
            return dict(execution_bundle.worker_assignments)
        return None


@dataclass(frozen=True, slots=True)
class ValidatedCompiledPlateExecution(ProgressExecutionContext):
    """Validated execution inputs plus defaults derived from orchestrator state."""

    pipeline_definition: List[AbstractStep]
    compiled_contexts: Dict[str, ProcessingContext]
    actual_max_workers: int
    progress_queue: ProgressQueue

    def worker_lane_execution_plan(
        self,
        *,
        request: CompiledPlateExecutionRequest,
        worker_assignment_plan: WorkerAssignmentPlan,
    ) -> WorkerLaneExecutionPlan:
        return WorkerLaneExecutionPlan(
            execution_id=self.execution_id,
            plate_id=self.plate_id,
            debug_execution_policy=request.debug_execution_policy,
            assignments=worker_assignment_plan,
            runtime_observation_mode=request.runtime_observation_mode,
        )


def execute_compiled_plate_request(
    orchestrator: "PipelineOrchestrator",
    request: CompiledPlateExecutionRequest,
) -> Dict[str, ExecutionResult]:
    """Execute compiled plate contexts through the runtime worker lanes."""

    validated = validate_compiled_plate_execution(orchestrator, request)
    if validated is None:
        return {}

    visualizers = bootstrap_execution_visualizers(
        orchestrator=orchestrator,
        compiled_contexts=validated.compiled_contexts,
        visualizer=request.visualizer,
        progress_queue=validated.progress_queue,
        progress_context=validated,
    )

    set_progress_queue(validated.progress_queue)
    try:
        orchestrator._cancelled = False
        orchestrator._state = OrchestratorState.EXECUTING
        logger.info(
            f"Starting execution for {len(validated.compiled_contexts)} axis values "
            f"with max_workers={validated.actual_max_workers}."
        )

        effective_config: GlobalPipelineConfig = orchestrator.get_effective_config()
        executor_resources = WorkerExecutorFactory(
            log_file_base=request.log_file_base,
            progress_queue=validated.progress_queue,
            progress_context=validated,
        ).create(
            effective_config=effective_config,
            actual_max_workers=validated.actual_max_workers,
        )

        execution_bundle = request.execution_bundle_for(validated)
        worker_assignments = request.worker_assignments_for(execution_bundle)

        executor_resources.install_execution_bundle(execution_bundle)
        orchestrator._executor = executor_resources.executor
        execution_results: Dict[str, ExecutionResult] = {}
        try:
            with executor_resources.execution_context():
                worker_assignment_plan = executor_resources.plan_worker_lanes(
                    actual_max_workers=validated.actual_max_workers,
                    execution_bundle=execution_bundle,
                    worker_assignments=worker_assignments,
                )
                worker_lane_execution_plan = validated.worker_lane_execution_plan(
                    request=request,
                    worker_assignment_plan=worker_assignment_plan,
                )
                execution_results = executor_resources.run_worker_lanes(
                    pipeline_definition=validated.pipeline_definition,
                    worker_lane_execution_plan=worker_lane_execution_plan,
                    parent_contexts=validated.compiled_contexts,
                )
                executor_resources.shutdown_executor()
        except BrokenProcessPool as exc:
            logger.warning(
                "ORCHESTRATOR: Executor context exit failed due to broken process "
                f"pool (workers were killed externally): {exc}"
            )
            if not execution_results:
                raise
        finally:
            executor_resources.clear_execution_bundle()

        executor_resources.cleanup_parent_gpu()
        AnalysisConsolidationPlan(orchestrator.microscope_handler).run(
            validated.compiled_contexts
        )
        project_execution_state(orchestrator, execution_results)
        stop_execution_visualizers(visualizers)
        return execution_results
    except Exception as exc:
        orchestrator._state = OrchestratorState.EXEC_FAILED
        logger.error(f"Failed to execute compiled plate: {exc}")
        raise
    finally:
        set_progress_queue(None)


def validate_compiled_plate_execution(
    orchestrator: "PipelineOrchestrator",
    request: CompiledPlateExecutionRequest,
) -> ValidatedCompiledPlateExecution | None:
    """Validate execute-compiled-plate invariants before worker setup."""

    pipeline_definition = resolved_pipeline_definition(
        orchestrator,
        request.pipeline_definition,
    )
    if not orchestrator.is_initialized():
        raise RuntimeError("Orchestrator must be initialized before executing.")
    if not pipeline_definition:
        raise ValueError("A valid (stateless) pipeline definition must be provided.")
    if not request.compiled_contexts:
        logger.warning("No compiled contexts provided for execution.")
        return None
    if request.progress_queue is None:
        raise ValueError(
            "progress_queue is required for execute_compiled_plate invariant path"
        )
    return ValidatedCompiledPlateExecution(
        execution_id=request.execution_id,
        plate_id=request.plate_id,
        pipeline_definition=pipeline_definition,
        compiled_contexts=request.compiled_contexts,
        actual_max_workers=actual_max_workers(orchestrator, request.max_workers),
        progress_queue=request.progress_queue,
    )


def resolved_pipeline_definition(
    orchestrator: "PipelineOrchestrator",
    pipeline_definition: List[AbstractStep],
) -> List[AbstractStep]:
    """Return the runtime-resolved pipeline definition if one is installed."""

    resolved_pipeline = orchestrator.resolved_pipeline_definition
    return resolved_pipeline if resolved_pipeline is not None else pipeline_definition


def actual_max_workers(
    orchestrator: "PipelineOrchestrator",
    max_workers: Optional[int],
) -> int:
    """Resolve the worker count from call override or effective config."""

    configured_workers = orchestrator.get_effective_config().num_workers
    requested_workers = max_workers if max_workers is not None else configured_workers
    return max(requested_workers, 1)


def bootstrap_execution_visualizers(
    *,
    orchestrator: "PipelineOrchestrator",
    compiled_contexts: Dict[str, ProcessingContext],
    visualizer: ExecutionVisualizer | None,
    progress_queue: ProgressQueue,
    progress_context: ProgressExecutionContext,
) -> list[ExecutionVisualizer]:
    """Create and readiness-check streaming visualizers for one execution."""

    if visualizer is not None:
        return []

    visualizers = create_required_visualizers(
        orchestrator=orchestrator,
        compiled_contexts=compiled_contexts,
        progress_queue=progress_queue,
        progress_context=progress_context,
    )
    if visualizers:
        wait_until_visualizers_ready(
            orchestrator=orchestrator,
            visualizers=visualizers,
            progress_queue=progress_queue,
            progress_context=progress_context,
        )
        clear_viewer_state(visualizers)
    return visualizers


def create_required_visualizers(
    *,
    orchestrator: "PipelineOrchestrator",
    compiled_contexts: Dict[str, ProcessingContext],
    progress_queue: ProgressQueue,
    progress_context: ProgressExecutionContext,
) -> list[ExecutionVisualizer]:
    """Create one viewer for each distinct visualizer requirement."""

    unique_configs: dict[tuple[str, int], tuple[RequiredVisualizer, object]] = {}
    for ctx in compiled_contexts.values():
        for required_visualizer in ctx.required_visualizers:
            if required_visualizer.key not in unique_configs:
                unique_configs[required_visualizer.key] = (
                    required_visualizer,
                    ctx.visualizer_config,
                )

    visualizers: list[ExecutionVisualizer] = []
    for required_visualizer, vis_config in unique_configs.values():
        emit_launching_viewer(
            required_visualizer=required_visualizer,
            progress_queue=progress_queue,
            progress_context=progress_context,
        )
        visualizers.append(
            orchestrator.get_or_create_visualizer(
                required_visualizer.config,
                vis_config,
            )
        )
    return visualizers


def emit_launching_viewer(
    *,
    required_visualizer: RequiredVisualizer,
    progress_queue: ProgressQueue,
    progress_context: ProgressExecutionContext,
) -> None:
    """Publish progress for viewer startup."""

    progress_queue.put(
        create_event(
            ProgressEventPayload(
                identity=progress_context.identity_for_event(axis_id="", step_name=""),
                phase=ProgressPhase.INIT,
                status=ProgressStatus.STARTED,
                percent=0.0,
                message=required_visualizer.launch_message,
            )
        ).to_dict()
    )


def wait_until_visualizers_ready(
    *,
    orchestrator: "PipelineOrchestrator",
    visualizers: list[ExecutionVisualizer],
    progress_queue: ProgressQueue,
    progress_context: ProgressExecutionContext,
) -> None:
    """Wait for all streaming visualizers to report readiness."""

    max_wait = 30.0
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if orchestrator._cancelled:
            raise RuntimeError("Execution cancelled by user")
        if all(v.is_running for v in visualizers):
            progress_queue.put(
                create_event(
                    ProgressEventPayload(
                        identity=progress_context.identity_for_event(
                            axis_id="",
                            step_name="",
                        ),
                        phase=ProgressPhase.INIT,
                        status=ProgressStatus.RUNNING,
                        percent=0.0,
                        message="All streaming viewers ready",
                    )
                ).to_dict()
            )
            return
        time.sleep(0.2)

    not_ready = [v.port for v in visualizers if not v.is_running]
    logger.warning(
        f"🔬 ORCHESTRATOR: Timeout waiting for streaming viewers. Not ready: {not_ready}"
    )
    progress_queue.put(
        create_event(
            ProgressEventPayload(
                identity=progress_context.identity_for_event(axis_id="", step_name=""),
                phase=ProgressPhase.INIT,
                status=ProgressStatus.RUNNING,
                percent=0.0,
                message=f"Timeout waiting for streaming viewers. Not ready: {not_ready}",
            )
        ).to_dict()
    )


def clear_viewer_state(visualizers: list[ExecutionVisualizer]) -> None:
    """Clear viewer state before sending a new execution stream."""

    for vis in visualizers:
        success = vis.clear_viewer_state()
        if not success:
            logger.warning(
                f"🔬 ORCHESTRATOR: Failed to clear state for viewer on port {vis.port}"
            )


def stop_execution_visualizers(visualizers: list[ExecutionVisualizer]) -> None:
    """Stop auto-created non-persistent visualizers after execution."""

    for idx, vis in enumerate(visualizers):
        try:
            if not vis.persistent:
                vis.stop_viewer()
        except Exception as exc:
            logger.warning(
                f"🔬 ORCHESTRATOR: Failed to cleanup visualizer {idx + 1}: {exc}"
            )


def project_execution_state(
    orchestrator: "PipelineOrchestrator",
    execution_results: Mapping[str, ExecutionResult],
) -> None:
    """Project worker-lane results back into orchestrator lifecycle state."""

    if all(result.is_success() for result in execution_results.values()):
        orchestrator._state = OrchestratorState.COMPLETED
    else:
        orchestrator._state = OrchestratorState.EXEC_FAILED
