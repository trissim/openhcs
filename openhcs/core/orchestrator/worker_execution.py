"""Worker-lane execution runners for compiled plate execution."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from openhcs.core.compiled_execution import (
    CompiledExecutionBundle,
    CompiledGpuRegistryPlan,
    CompiledRuntimeEnvironmentPlan,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import MultiprocessingStartMethod
from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.native_threading import configure_native_thread_count
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeContextObservation,
    RuntimeExecutionObservation,
    RuntimeObservationMode,
)
from openhcs.core.orchestrator.worker_lanes import (
    CompiledContextLanePlanner,
    ForkInheritedWorkerExecutionState,
    TransportAxisContexts,
    WorkerAssignmentPlan,
    WorkerLaneAxisContexts,
    WorkerLaneExecutionContext,
    WorkerLaneExecutionPlan,
)
from openhcs.core.orchestrator.worker_profiling import CProfileWorkerProfilingPolicy
from openhcs.core.progress import emit, ProgressPhase, ProgressStatus
from openhcs.core.progress import ProgressExecutionContext, ProgressQueue
from openhcs.core.progress.live_measurements import (
    live_measurement_context_for_records,
)
from openhcs.core.progress.runtime_artifacts import (
    runtime_artifact_context_for_records,
)
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_artifact_materialization import (
    observed_materialized_artifact_locations_by_address,
)
from openhcs.utils.environment import OpenHCSProcessEnvironment


logger = logging.getLogger(__name__)
PIPELINE_PROGRESS_STEP_NAME = "pipeline"


def _runtime_observation_progress_context(
    records: tuple[StoredRuntimeValue, ...],
    *,
    plan: CompiledStepPlan,
    context: ProcessingContext,
) -> dict | None:
    """Project one RuntimeValueStore observation delta through owned payloads."""

    runtime_artifacts = runtime_artifact_context_for_records(records)
    live_measurements = live_measurement_context_for_records(
        records,
        materialized_locations_by_address=(
            observed_materialized_artifact_locations_by_address(
                plan,
                context,
                records,
            )
        ),
    )
    if runtime_artifacts is None:
        return live_measurements
    if live_measurements is None:
        return runtime_artifacts
    return {**runtime_artifacts, **live_measurements}


@dataclass(frozen=True, slots=True)
class WorkerExecutorResources(ABC):
    """Nominal worker execution resources for one execution mode."""

    multiprocessing_context: Any
    use_multiprocessing: bool

    @property
    def executor(self) -> concurrent.futures.Executor | None:
        return None

    def execution_context(self):
        return contextlib.nullcontext()

    def install_execution_bundle(self, execution_bundle: CompiledExecutionBundle) -> None:
        """Install any mode-specific inherited runtime state."""

    def clear_execution_bundle(self) -> None:
        """Clear mode-specific inherited runtime state."""

    def plan_worker_lanes(
        self,
        *,
        actual_max_workers: int,
        execution_bundle: CompiledExecutionBundle,
        worker_assignments: Dict[str, List[str]] | None,
    ) -> WorkerAssignmentPlan:
        contexts_snapshot = self.contexts_snapshot(execution_bundle)
        return CompiledContextLanePlanner(
            actual_max_workers=actual_max_workers,
            fork_inherited_execution=self.uses_fork_inherited_contexts,
        ).plan(contexts_snapshot, worker_assignments)

    def contexts_snapshot(
        self,
        execution_bundle: CompiledExecutionBundle,
    ) -> Dict[str, ProcessingContext]:
        raw_contexts = self.raw_contexts_snapshot(execution_bundle)
        return FunctionStepTransportAuthority.normalize_contexts(dict(raw_contexts))

    @property
    @abstractmethod
    def uses_fork_inherited_contexts(self) -> bool:
        """Whether lane planning receives fork-inherited runtime context keys."""

    @abstractmethod
    def raw_contexts_snapshot(
        self,
        execution_bundle: CompiledExecutionBundle,
    ) -> Mapping[str, ProcessingContext]:
        """Return the context map consumed by lane planning for this mode."""

    @abstractmethod
    def run_worker_lanes(
        self,
        *,
        pipeline_definition: List[AbstractStep],
        worker_lane_execution_plan: WorkerLaneExecutionPlan,
        parent_contexts: Mapping[str, ProcessingContext],
    ) -> Dict[str, ExecutionResult]:
        """Execute all lane work for this mode."""

    def shutdown_executor(self) -> None:
        """Shutdown owned executor resources."""

    def cleanup_parent_gpu(self) -> None:
        """Cleanup parent GPU state when this mode executes in-process."""

        try:
            from openhcs.core.memory import cleanup_all_gpu_frameworks

            if not self.use_multiprocessing:
                cleanup_all_gpu_frameworks()
        except Exception as cleanup_error:
            logger.warning(
                f"Failed to cleanup GPU memory after plate execution: {cleanup_error}"
            )


@dataclass(frozen=True, slots=True)
class InlineWorkerExecutorResources(WorkerExecutorResources):
    """In-process single-lane execution resources."""

    @property
    def uses_fork_inherited_contexts(self) -> bool:
        return False

    def raw_contexts_snapshot(
        self,
        execution_bundle: CompiledExecutionBundle,
    ) -> Mapping[str, ProcessingContext]:
        return execution_bundle.transport_contexts

    def run_worker_lanes(
        self,
        *,
        pipeline_definition: List[AbstractStep],
        worker_lane_execution_plan: WorkerLaneExecutionPlan,
        parent_contexts: Mapping[str, ProcessingContext],
    ) -> Dict[str, ExecutionResult]:
        lane_results = InlineWorkerLaneRunner().run(
            pipeline_definition,
            worker_lane_execution_plan,
        )
        if worker_lane_execution_plan.runtime_observation_mode.collects_records:
            for result in lane_results.values():
                result.runtime_observation.merge_into(parent_contexts)
        return lane_results


@dataclass(frozen=True, slots=True)
class ForkInheritedWorkerExecutorResources(WorkerExecutorResources):
    """Fork-inherited runtime context execution resources."""

    @property
    def uses_fork_inherited_contexts(self) -> bool:
        return True

    def raw_contexts_snapshot(
        self,
        execution_bundle: CompiledExecutionBundle,
    ) -> Mapping[str, ProcessingContext]:
        return execution_bundle.runtime_contexts

    def install_execution_bundle(self, execution_bundle: CompiledExecutionBundle) -> None:
        ForkInheritedWorkerExecutionState.install(execution_bundle)

    def clear_execution_bundle(self) -> None:
        ForkInheritedWorkerExecutionState.clear()

    def run_worker_lanes(
        self,
        *,
        pipeline_definition: List[AbstractStep],
        worker_lane_execution_plan: WorkerLaneExecutionPlan,
        parent_contexts: Mapping[str, ProcessingContext],
    ) -> Dict[str, ExecutionResult]:
        return ForkInheritedWorkerLaneRunner(self.multiprocessing_context).run(
            worker_lane_execution_plan
        )


@dataclass(frozen=True, slots=True)
class PooledWorkerExecutorResources(WorkerExecutorResources):
    """Thread/process pool execution resources."""

    _executor: concurrent.futures.Executor

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def uses_fork_inherited_contexts(self) -> bool:
        return False

    def execution_context(self):
        return self._executor

    def raw_contexts_snapshot(
        self,
        execution_bundle: CompiledExecutionBundle,
    ) -> Mapping[str, ProcessingContext]:
        return execution_bundle.transport_contexts

    def run_worker_lanes(
        self,
        *,
        pipeline_definition: List[AbstractStep],
        worker_lane_execution_plan: WorkerLaneExecutionPlan,
        parent_contexts: Mapping[str, ProcessingContext],
    ) -> Dict[str, ExecutionResult]:
        return PooledWorkerLaneRunner(self._executor).run(
            pipeline_definition,
            worker_lane_execution_plan,
            parent_contexts,
        )

    def shutdown_executor(self) -> None:
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        except concurrent.futures.process.BrokenProcessPool as exc:
            logger.warning(
                "ORCHESTRATOR: Executor shutdown failed due to broken process "
                f"pool (workers were killed externally): {exc}"
            )
        except Exception as exc:
            logger.warning(f"ORCHESTRATOR: Executor shutdown failed: {exc}")


class WorkerExecutorFactory:
    """Create the worker resources matching the effective runtime config."""

    def __init__(
        self,
        *,
        log_file_base: str | None,
        progress_queue: ProgressQueue,
        progress_context: ProgressExecutionContext,
    ) -> None:
        self._log_file_base = log_file_base
        self._progress_queue = progress_queue
        self._progress_context = progress_context

    def create(
        self,
        *,
        runtime_environment: CompiledRuntimeEnvironmentPlan,
        actual_max_workers: int,
    ) -> WorkerExecutorResources:
        multiprocessing_context = multiprocessing.get_context(
            runtime_environment.multiprocessing_start_method.value
        )
        if actual_max_workers == 1:
            configure_native_thread_count(1)
            return InlineWorkerExecutorResources(
                multiprocessing_context=multiprocessing_context,
                use_multiprocessing=not runtime_environment.use_threading,
            )
        if (
            not runtime_environment.use_threading
            and runtime_environment.multiprocessing_start_method
            is MultiprocessingStartMethod.FORK
        ):
            return ForkInheritedWorkerExecutorResources(
                multiprocessing_context=multiprocessing_context,
                use_multiprocessing=True,
            )
        if runtime_environment.use_threading:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=actual_max_workers
            )
        else:
            executor = self._process_pool_executor(
                multiprocessing_context,
                actual_max_workers,
                runtime_environment.gpu_registry,
            )
        return PooledWorkerExecutorResources(
            multiprocessing_context=multiprocessing_context,
            use_multiprocessing=not runtime_environment.use_threading,
            _executor=executor,
        )

    def _process_pool_executor(
        self,
        multiprocessing_context: Any,
        actual_max_workers: int,
        gpu_registry_plan: CompiledGpuRegistryPlan,
    ) -> concurrent.futures.ProcessPoolExecutor:
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=actual_max_workers,
            mp_context=multiprocessing_context,
            initializer=_configure_worker_process,
            initargs=(
                self._log_file_base,
                gpu_registry_plan,
                self._progress_queue,
                self._progress_context,
            ),
        )


def _configure_worker_logging(log_file_base: str) -> None:
    """Configure worker-process logging under the parent execution log prefix."""

    import logging
    import os
    import time

    worker_pid = os.getpid()
    worker_timestamp = int(time.time() * 1000000)
    worker_id = f"{worker_pid}_{worker_timestamp}"
    worker_log_file = f"{log_file_base}_worker_{worker_id}.log"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(worker_log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logging.getLogger("openhcs").setLevel(logging.INFO)


def _configure_worker_process(
    log_file_base: str | None,
    gpu_registry_plan: CompiledGpuRegistryPlan,
    progress_queue: ProgressQueue | None = None,
    progress_context: ProgressExecutionContext | None = None,
) -> None:
    """Prepare process-local registries, logging, and progress transport."""

    import logging
    import os

    worker_log_level_name = os.environ.get("OPENHCS_LOG_LEVEL", "INFO").upper()
    worker_log_levels = logging.getLevelNamesMapping()
    if worker_log_level_name not in worker_log_levels:
        raise ValueError(f"Unknown OPENHCS_LOG_LEVEL: {worker_log_level_name!r}")
    worker_log_level = worker_log_levels[worker_log_level_name]
    if not isinstance(worker_log_level, int):
        raise ValueError(f"Unknown OPENHCS_LOG_LEVEL: {worker_log_level_name!r}")

    if not OpenHCSProcessEnvironment.cpu_only_mode():
        os.environ.pop("OPENHCS_SUBPROCESS_NO_GPU", None)
        os.environ.pop("POLYSTORE_SUBPROCESS_NO_GPU", None)

    if log_file_base is not None:
        _configure_worker_logging(log_file_base)
    else:
        logging.basicConfig(level=worker_log_level)
    logging.getLogger().setLevel(worker_log_level)
    logging.getLogger("openhcs").setLevel(worker_log_level)

    configure_native_thread_count(1)

    if not OpenHCSProcessEnvironment.cpu_only_mode():
        gpu_registry_plan.setup_global_registry()

    from openhcs.processing.func_registry import initialize_registry

    initialize_registry()

    if progress_queue is not None and progress_context is not None:
        from openhcs.core.progress import set_progress_queue

        set_progress_queue(progress_queue)


def _execute_fork_inherited_worker_lane_process(
    result_connection: Any,
    lane_axis_context_keys: List[tuple[str, List[str]]],
    lane_context: WorkerLaneExecutionContext,
    runtime_observation_mode: RuntimeObservationMode,
) -> None:
    """Process entrypoint for fork-inherited worker lane execution."""

    profiling_policy = CProfileWorkerProfilingPolicy.from_environment()
    try:
        with profiling_policy.profile(
            execution_id=lane_context.execution_id,
            plate_id=lane_context.plate_id,
            worker_slot=lane_context.worker_slot,
            owned_wells=list(lane_context.owned_wells),
        ):
            result_connection.send(
                (
                    "result",
                    _execute_fork_inherited_worker_lane_static(
                        lane_axis_context_keys,
                        lane_context,
                        runtime_observation_mode,
                    ),
                )
            )
    except BaseException as exc:
        import traceback

        result_connection.send(("error", exc, traceback.format_exc()))
    finally:
        result_connection.close()


class ForkInheritedWorkerLaneRunner:
    """Runs fork-inherited worker lanes without executor serialization overhead."""

    def __init__(self, multiprocessing_context: Any) -> None:
        self._multiprocessing_context = multiprocessing_context

    def run(
        self,
        execution_plan: WorkerLaneExecutionPlan,
    ) -> Dict[str, ExecutionResult]:
        active_lanes = execution_plan.active_lane_items()
        if len(active_lanes) == 1:
            worker_slot, lane_contexts = active_lanes[0]
            return self.run_inline_single_lane(
                worker_slot,
                lane_contexts,
                execution_plan,
            )

        processes: list[tuple[str, List[str], Any, Any]] = []
        execution_results: Dict[str, ExecutionResult] = {}

        for worker_slot, lane_contexts in active_lanes:
            owned_wells = list(execution_plan.assignments.owned_wells(worker_slot))
            worker_lane_context = execution_plan.lane_context(worker_slot)
            result_reader, result_writer = self._multiprocessing_context.Pipe(
                duplex=False
            )
            process = self._multiprocessing_context.Process(
                target=_execute_fork_inherited_worker_lane_process,
                args=(
                    result_writer,
                    lane_contexts,
                    worker_lane_context,
                    execution_plan.runtime_observation_mode,
                ),
            )
            process.start()
            result_writer.close()
            processes.append((worker_slot, owned_wells, process, result_reader))

        for worker_slot, owned_wells, process, result_reader in processes:
            try:
                message_kind, payload, *rest = result_reader.recv()
            except EOFError as exc:
                process.join()
                raise RuntimeError(
                    f"Fork worker lane {worker_slot} exited without returning "
                    f"a result; exitcode={process.exitcode}."
                ) from exc
            finally:
                result_reader.close()

            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Fork worker lane {worker_slot} exited with "
                    f"exitcode={process.exitcode}."
                )
            if message_kind == "error":
                if not rest:
                    raise RuntimeError(
                        f"Fork worker lane {worker_slot} returned an error without traceback."
                    )
                traceback_text = rest[0]
                raise RuntimeError(
                    f"Fork worker lane {worker_slot} generated an exception: "
                    f"{payload}\n{traceback_text}"
                )
            if message_kind != "result":
                raise RuntimeError(
                    f"Fork worker lane {worker_slot} returned unknown message "
                    f"{message_kind!r}."
                )

            lane_results = payload
            execution_results.update(lane_results)
            if execution_plan.runtime_observation_mode.collects_records:
                for result in lane_results.values():
                    result.runtime_observation.merge_into(
                        ForkInheritedWorkerExecutionState.require_current().runtime_contexts
                    )

        return execution_results

    def run_inline_single_lane(
        self,
        worker_slot: str,
        lane_axis_context_keys: List[tuple[str, List[str]]],
        execution_plan: WorkerLaneExecutionPlan,
    ) -> Dict[str, ExecutionResult]:
        """Run a single fork-inherited lane without launching a child process."""

        execution_state = ForkInheritedWorkerExecutionState.require_current()
        lane_axis_contexts = ForkInheritedWorkerExecutionState.resolve_lane_contexts(
            lane_axis_context_keys
        )
        lane_results = execute_worker_lane(
            pipeline_definition=list(execution_state.pipeline_definition),
            lane_axis_contexts=lane_axis_contexts,
            lane_context=execution_plan.lane_context(worker_slot),
            runtime_observation_mode=execution_plan.runtime_observation_mode,
        )
        return lane_results


class InlineWorkerLaneRunner:
    """Runs a single deterministic worker lane in the orchestrator process."""

    def run(
        self,
        pipeline_definition: List[AbstractStep],
        execution_plan: WorkerLaneExecutionPlan,
    ) -> Dict[str, ExecutionResult]:
        active_lanes = execution_plan.active_lane_items()
        if len(active_lanes) != 1:
            raise RuntimeError(
                "Inline worker lane execution requires exactly one active lane, "
                f"got {len(active_lanes)}."
            )

        worker_slot, lane_contexts = active_lanes[0]
        lane_context = execution_plan.lane_context(worker_slot)
        profiling_policy = CProfileWorkerProfilingPolicy.from_environment()
        with profiling_policy.profile(
            execution_id=lane_context.execution_id,
            plate_id=lane_context.plate_id,
            worker_slot=lane_context.worker_slot,
            owned_wells=list(lane_context.owned_wells),
        ):
            return execute_worker_lane(
                pipeline_definition=pipeline_definition,
                lane_axis_contexts=lane_contexts,
                lane_context=lane_context,
                runtime_observation_mode=execution_plan.runtime_observation_mode,
            )


class PooledWorkerLaneRunner:
    """Runs deterministic worker lanes through a thread or process executor."""

    def __init__(self, executor: concurrent.futures.Executor) -> None:
        self._executor = executor

    def run(
        self,
        pipeline_definition: List[AbstractStep],
        execution_plan: WorkerLaneExecutionPlan,
        parent_contexts: Mapping[str, ProcessingContext],
    ) -> Dict[str, ExecutionResult]:
        future_to_worker_slot = self._submit_lanes(
            pipeline_definition,
            execution_plan,
        )
        return self._collect_results(
            future_to_worker_slot,
            pipeline_definition,
            execution_plan,
            parent_contexts,
        )

    def _submit_lanes(
        self,
        pipeline_definition: List[AbstractStep],
        execution_plan: WorkerLaneExecutionPlan,
    ) -> Dict[concurrent.futures.Future, tuple[str, List[str]]]:
        pipeline_definition = FunctionStepTransportAuthority.normalize_pipeline(
            pipeline_definition
        )
        future_to_worker_slot: Dict[concurrent.futures.Future, tuple[str, List[str]]] = {}
        for worker_slot, lane_contexts in execution_plan.assignments.lane_axis_contexts.items():
            if not lane_contexts:
                continue
            owned_wells = list(execution_plan.assignments.owned_wells(worker_slot))
            try:
                future = self._executor.submit(
                    execute_worker_lane,
                    pipeline_definition,
                    lane_contexts,
                    execution_plan.lane_context(worker_slot),
                    execution_plan.runtime_observation_mode,
                )
                future_to_worker_slot[future] = (worker_slot, owned_wells)
            except Exception as submit_error:
                logger.error(
                    f"🔥 ORCHESTRATOR ERROR: Failed to submit lane {worker_slot}: {submit_error}",
                    exc_info=True,
                )
                raise
        return future_to_worker_slot

    def _collect_results(
        self,
        future_to_worker_slot: Mapping[concurrent.futures.Future, tuple[str, List[str]]],
        pipeline_definition: List[AbstractStep],
        execution_plan: WorkerLaneExecutionPlan,
        parent_contexts: Mapping[str, ProcessingContext],
    ) -> Dict[str, ExecutionResult]:
        execution_results: Dict[str, ExecutionResult] = {}
        for future in concurrent.futures.as_completed(future_to_worker_slot):
            worker_slot, owned_wells = future_to_worker_slot[future]

            try:
                lane_results = future.result()
                execution_results.update(lane_results)
                if execution_plan.runtime_observation_mode.collects_records:
                    for result in lane_results.values():
                        result.runtime_observation.merge_into(parent_contexts)
            except Exception as exc:
                self._emit_lane_error(
                    exc,
                    worker_slot=worker_slot,
                    owned_wells=owned_wells,
                    pipeline_definition=pipeline_definition,
                    execution_plan=execution_plan,
                )
                raise
        return execution_results

    def _emit_lane_error(
        self,
        exc: Exception,
        *,
        worker_slot: str,
        owned_wells: List[str],
        pipeline_definition: List[AbstractStep],
        execution_plan: WorkerLaneExecutionPlan,
    ) -> None:
        import traceback

        if not owned_wells:
            raise RuntimeError(
                f"Worker lane {worker_slot} cannot emit an axis error without owned wells."
            ) from exc

        full_traceback = traceback.format_exc()
        error_msg = (
            f"Worker lane {worker_slot} generated an exception during execution: {exc}"
        )
        logger.error(f"🔥 ORCHESTRATOR ERROR: {error_msg}", exc_info=True)
        logger.error(
            f"🔥 ORCHESTRATOR FULL TRACEBACK for worker lane {worker_slot}:\n{full_traceback}"
        )
        emit(
            execution_id=execution_plan.execution_id,
            plate_id=execution_plan.plate_id,
            axis_id=owned_wells[0],
            step_name=PIPELINE_PROGRESS_STEP_NAME,
            phase=ProgressPhase.AXIS_ERROR,
            status=ProgressStatus.ERROR,
            completed=0,
            total=len(pipeline_definition),
            percent=0.0,
            error=str(exc),
            traceback=full_traceback,
            worker_slot=worker_slot,
            owned_wells=owned_wells,
        )


def _execute_axis_with_sequential_combinations(
    pipeline_definition: List[AbstractStep],
    axis_contexts: TransportAxisContexts,
    lane_context: WorkerLaneExecutionContext,
    runtime_observation_mode: RuntimeObservationMode,
) -> ExecutionResult:
    """Execute all sequential combinations for a single axis in order."""

    if not axis_contexts:
        raise ValueError(
            "axis_contexts cannot be empty - this indicates a bug in the caller"
        )

    _, first_context = axis_contexts[0]
    axis_id = first_context.axis_id
    total_steps = len(pipeline_definition)
    completed_axis_steps = _completed_axis_step_count(first_context, total_steps)

    emit(
        execution_id=lane_context.execution_id,
        plate_id=lane_context.plate_id,
        axis_id=axis_id,
        step_name=PIPELINE_PROGRESS_STEP_NAME,
        phase=ProgressPhase.AXIS_STARTED,
        status=ProgressStatus.STARTED,
        completed=0,
        total=total_steps,
        percent=0.0,
        worker_slot=lane_context.worker_slot,
        owned_wells=list(lane_context.owned_wells),
    )

    runtime_observations: list[RuntimeContextObservation] = []
    for context_key, frozen_context in axis_contexts:
        runtime_store = frozen_context.runtime_value_store
        execution_observation_cursor = runtime_store.observation_cursor()
        result = _execute_single_axis_static(
            pipeline_definition,
            frozen_context,
            lane_context,
        )
        if runtime_observation_mode.collects_records:
            runtime_observations.append(
                RuntimeContextObservation(
                    context_key=context_key,
                    records=runtime_store.observed_values_after(
                        execution_observation_cursor
                    ),
                )
            )
        elif runtime_observation_mode.releases_worker_records:
            frozen_context.runtime_value_store.clear()

        from polystore.base import reset_memory_backend
        from openhcs.core.memory import cleanup_all_gpu_frameworks

        reset_memory_backend()
        if cleanup_all_gpu_frameworks:
            cleanup_all_gpu_frameworks()
        if not result.is_success():
            logger.error(
                f"🔄 WORKER: Combination {context_key} failed for axis {axis_id}"
            )
            emit(
                execution_id=lane_context.execution_id,
                plate_id=lane_context.plate_id,
                axis_id=axis_id,
                step_name=PIPELINE_PROGRESS_STEP_NAME,
                phase=ProgressPhase.AXIS_ERROR,
                status=ProgressStatus.ERROR,
                completed=0,
                total=total_steps,
                percent=0.0,
                message=result.error_message,
                worker_slot=lane_context.worker_slot,
                owned_wells=list(lane_context.owned_wells),
            )
            return ExecutionResult.error(
                axis_id=axis_id,
                failed_combination=context_key,
                error_message=result.error_message,
            )

    emit(
        execution_id=lane_context.execution_id,
        plate_id=lane_context.plate_id,
        axis_id=axis_id,
        step_name=PIPELINE_PROGRESS_STEP_NAME,
        phase=ProgressPhase.AXIS_COMPLETED,
        status=ProgressStatus.SUCCESS,
        completed=completed_axis_steps,
        total=total_steps,
        percent=(completed_axis_steps / total_steps) * 100.0,
        worker_slot=lane_context.worker_slot,
        owned_wells=list(lane_context.owned_wells),
    )
    return ExecutionResult.success(
        axis_id=axis_id,
        runtime_observation=RuntimeExecutionObservation(
            contexts=tuple(runtime_observations),
        ),
    )


def _completed_axis_step_count(
    context: ProcessingContext,
    total_steps: int,
) -> int:
    """Return the pipeline position where terminal plate processing begins."""

    return min(
        (
            step_index
            for step_index, step_plan in context.step_plans.items()
            if step_plan.execution_scope is FunctionStepExecutionScope.PLATE
        ),
        default=total_steps,
    )


def _execute_single_axis_static(
    pipeline_definition: List[AbstractStep],
    frozen_context: ProcessingContext,
    lane_context: WorkerLaneExecutionContext,
) -> ExecutionResult:
    """Execute one frozen axis context against the compiled pipeline."""

    axis_id = frozen_context.axis_id
    total_steps = len(pipeline_definition)

    if not frozen_context.is_frozen():
        error_msg = f"Context for axis {axis_id} is not frozen before execution"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if not pipeline_definition:
        error_msg = f"Empty pipeline_definition for axis {axis_id}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    frozen_context.bind_execution_runtime(
        execution_id=lane_context.execution_id,
        plate_id=lane_context.plate_id,
        worker_slot=lane_context.worker_slot,
        owned_wells=lane_context.owned_wells,
    )
    lane_context.install_debug_sink(frozen_context)
    runtime_value_store = frozen_context.runtime_value_store

    for step_index, step in enumerate(pipeline_definition):
        step_plan = frozen_context.step_plans[step_index]
        compiled_pattern = step_plan.compiled_function_pattern
        if (
            compiled_pattern is not None
            and compiled_pattern.execution_scope
            is FunctionStepExecutionScope.PLATE
        ):
            continue
        step_name = step_plan.step_name
        if not lane_context.debug_execution_policy.should_execute_step(step_index):
            if lane_context.debug_execution_policy.should_reuse_step_outputs(step_index):
                observation_cursor = runtime_value_store.observation_cursor()
                lane_context.debug_execution_policy.prepare_reused_step_outputs(
                    step_index=step_index,
                    step_name=step_name,
                    step_scope_id=step_plan.step_scope_id,
                    context=frozen_context,
                    artifact_outputs=step_plan.artifact_outputs,
                )
                observed_records = runtime_value_store.observed_values_after(
                    observation_cursor
                )
                runtime_progress_context = _runtime_observation_progress_context(
                    observed_records,
                    plan=step_plan,
                    context=frozen_context,
                )
                emit(
                    execution_id=lane_context.execution_id,
                    plate_id=lane_context.plate_id,
                    axis_id=axis_id,
                    step_name=step_name,
                    phase=ProgressPhase.STEP_COMPLETED,
                    status=ProgressStatus.SUCCESS,
                    completed=step_index + 1,
                    total=total_steps,
                    percent=((step_index + 1) / total_steps) * 100.0,
                    worker_slot=lane_context.worker_slot,
                    owned_wells=list(lane_context.owned_wells),
                    message="Reused warm debug artifacts",
                    context=runtime_progress_context,
                )
            continue

        emit(
            execution_id=lane_context.execution_id,
            plate_id=lane_context.plate_id,
            axis_id=axis_id,
            step_name=step_name,
            phase=ProgressPhase.STEP_STARTED,
            status=ProgressStatus.STARTED,
            completed=step_index,
            total=total_steps,
            percent=(step_index / total_steps) * 100.0,
            worker_slot=lane_context.worker_slot,
            owned_wells=list(lane_context.owned_wells),
        )

        observation_cursor = runtime_value_store.observation_cursor()
        step.process(frozen_context, step_index)
        observed_records = runtime_value_store.observed_values_after(observation_cursor)
        runtime_progress_context = _runtime_observation_progress_context(
            observed_records,
            plan=step_plan,
            context=frozen_context,
        )

        emit(
            execution_id=lane_context.execution_id,
            plate_id=lane_context.plate_id,
            axis_id=axis_id,
            step_name=step_name,
            phase=ProgressPhase.STEP_COMPLETED,
            status=ProgressStatus.SUCCESS,
            completed=step_index + 1,
            total=total_steps,
            percent=((step_index + 1) / total_steps) * 100.0,
            worker_slot=lane_context.worker_slot,
            owned_wells=list(lane_context.owned_wells),
            context=runtime_progress_context,
        )
        if lane_context.debug_execution_policy.step_stop_strategy().should_stop_after_step(
            step_index=step_index,
            step_name=step_name,
        ):
            break

    return ExecutionResult.success(axis_id=axis_id)


def execute_worker_lane(
    pipeline_definition: List[AbstractStep],
    lane_axis_contexts: WorkerLaneAxisContexts,
    lane_context: WorkerLaneExecutionContext,
    runtime_observation_mode: RuntimeObservationMode,
) -> Dict[str, ExecutionResult]:
    """Execute a deterministic worker lane: wells sequentially within one slot."""

    lane_results: Dict[str, ExecutionResult] = {}
    for axis_id, axis_contexts in lane_axis_contexts:
        lane_results[axis_id] = _execute_axis_with_sequential_combinations(
            pipeline_definition=pipeline_definition,
            axis_contexts=axis_contexts,
            lane_context=lane_context,
            runtime_observation_mode=runtime_observation_mode,
        )
    return lane_results


def _execute_fork_inherited_worker_lane_static(
    lane_axis_context_keys: List[tuple[str, List[str]]],
    lane_context: WorkerLaneExecutionContext,
    runtime_observation_mode: RuntimeObservationMode,
) -> Dict[str, ExecutionResult]:
    """Execute a worker lane using fork-inherited compiled contexts."""

    execution_bundle = ForkInheritedWorkerExecutionState.require_current()
    return execute_worker_lane(
        pipeline_definition=list(execution_bundle.pipeline_definition),
        lane_axis_contexts=ForkInheritedWorkerExecutionState.resolve_lane_contexts(
            lane_axis_context_keys
        ),
        lane_context=lane_context,
        runtime_observation_mode=runtime_observation_mode,
    )
