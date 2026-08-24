"""Compiled plate execution flow for :class:`PipelineOrchestrator`."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
)

from openhcs.constants.constants import Backend, OrchestratorState
from openhcs.core.artifacts import (
    SpecialArtifactType,
)
from openhcs.core.callable_contract import (
    CallableContract,
    FunctionStepExecutionScope,
)
from openhcs.core.compiled_execution import (
    CompiledExecutionBundle,
    CompiledRuntimeEnvironmentPlan,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.context.processing_context import (
    ProcessingContext,
    RequiredVisualizer,
)
from openhcs.core.debug import DebugExecutionPolicy
from openhcs.core.execution_visualizer import ExecutionVisualizerABC
from openhcs.core.function_patterns import CompiledFunctionInvocation
from openhcs.core.orchestrator.analysis_consolidation import (
    consolidate_analysis_outputs,
)
from openhcs.core.orchestrator.cancellation import ExecutionCancelledError
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeContextObservation,
    RuntimeExecutionObservation,
    RuntimeObservationMode,
)
from openhcs.core.orchestrator.worker_execution import (
    WorkerExecutorFactory,
)
from openhcs.core.orchestrator.worker_lanes import (
    WorkerAssignmentPlan,
    WorkerLaneExecutionPlan,
)
from openhcs.core.pipeline.funcstep_contract_validator import (
    FuncStepContractValidator,
)
from openhcs.core.progress import (
    ProgressEventPayload,
    ProgressExecutionContext,
    ProgressPhase,
    ProgressQueue,
    ProgressStatus,
    create_event,
    set_progress_queue,
)
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_stores import (
    RuntimeArtifactBatch,
    RuntimeArtifactLocation,
    RuntimeArtifactQuery,
    StoredRuntimeValue,
    replace_runtime_artifact_payload,
)
from openhcs.core.source_bindings import CompiledSourceBindingPlan
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_outputs import (
    OpenHCSMetadataWriter,
    RuntimeArtifactMaterializationAuthority,
)

if TYPE_CHECKING:
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.source_matching import SourceImageSetIdentityPolicy
    from openhcs.runtime.viewer_protocol import (
        ViewerControlResponse,
        ViewerSettleProgress,
    )


logger = logging.getLogger(__name__)
PLATE_STEP_PROGRESS_HEARTBEAT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class CompiledPlateExecutionExtras:
    """Diagnostics captured at the parent-owned execution boundary."""

    EXECUTION_RECORD_KEY: ClassVar[str] = "compiled_plate_execution_extras"
    RESULTS_SUMMARY_KEY: ClassVar[str] = "viewer_states_by_port"

    viewer_states_by_port: Mapping[int, "ViewerControlResponse"]


class CompiledPlateExecutionResults(dict[str, ExecutionResult]):
    """Axis results plus parent-owned diagnostics from one compiled execution."""

    def __init__(
        self,
        results: Mapping[str, ExecutionResult] | None = None,
        *,
        extras: CompiledPlateExecutionExtras | None = None,
    ) -> None:
        super().__init__(results or {})
        self.extras = extras or CompiledPlateExecutionExtras(
            viewer_states_by_port=MappingProxyType({})
        )


@dataclass(frozen=True, slots=True)
class CompiledPlateExecutionRequest(ProgressExecutionContext):
    """Public execute-compiled-plate call normalized into one request record."""

    execution_bundle: CompiledExecutionBundle
    max_workers: Optional[int]
    visualizer: ExecutionVisualizerABC | None
    log_file_base: Optional[str]
    progress_queue: ProgressQueue | None
    runtime_observation_mode: RuntimeObservationMode
    debug_execution_policy: DebugExecutionPolicy

    @property
    def pipeline_definition(self) -> List[AbstractStep]:
        return list(self.execution_bundle.pipeline_definition)

    @property
    def compiled_contexts(self) -> Dict[str, ProcessingContext]:
        return dict(self.execution_bundle.runtime_contexts)

    def worker_assignments_for(self) -> Optional[Dict[str, List[str]]]:
        if not self.execution_bundle.worker_assignments:
            return None
        return dict(self.execution_bundle.worker_assignments)


@dataclass(frozen=True, slots=True)
class ValidatedCompiledPlateExecution(ProgressExecutionContext):
    """Validated execution inputs plus defaults derived from orchestrator state."""

    pipeline_definition: List[AbstractStep]
    compiled_contexts: Dict[str, ProcessingContext]
    actual_max_workers: int
    progress_queue: ProgressQueue
    runtime_environment: CompiledRuntimeEnvironmentPlan

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
) -> CompiledPlateExecutionResults:
    """Execute compiled plate contexts through the runtime worker lanes."""

    validated = validate_compiled_plate_execution(orchestrator, request)
    if validated is None:
        return CompiledPlateExecutionResults()

    cancellation = orchestrator._execution_cancellation.begin()
    try:
        cancellation.raise_if_requested("before viewer bootstrap")
        visualizers = bootstrap_execution_visualizers(
            orchestrator=orchestrator,
            compiled_contexts=validated.compiled_contexts,
            visualizer=request.visualizer,
            progress_queue=validated.progress_queue,
            progress_context=validated,
        )
    except BaseException:
        orchestrator._execution_cancellation.finish(cancellation)
        raise

    set_progress_queue(validated.progress_queue)
    try:
        orchestrator._state = OrchestratorState.EXECUTING
        logger.info(
            f"Starting execution for {len(validated.compiled_contexts)} axis values "
            f"with max_workers={validated.actual_max_workers}."
        )

        executor_resources = WorkerExecutorFactory(
            log_file_base=request.log_file_base,
            progress_queue=validated.progress_queue,
            progress_context=validated,
            cancellation=cancellation,
        ).create(
            runtime_environment=validated.runtime_environment,
            actual_max_workers=validated.actual_max_workers,
        )

        execution_bundle = request.execution_bundle
        worker_assignments = request.worker_assignments_for()

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
                cancellation.raise_if_requested("after worker execution")
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
            executor_resources.release_parent_runtime_resources(execution_bundle)

        if all(result.is_success() for result in execution_results.values()):
            plate_runtime_observation = execute_plate_scoped_steps(
                validated.compiled_contexts,
                progress_queue=validated.progress_queue,
                progress_context=validated,
            )
            consolidate_analysis_outputs(
                validated.compiled_contexts,
                execution_results,
                plate_runtime_observation=plate_runtime_observation,
            )
            OpenHCSMetadataWriter.finalize_completed_plate(
                validated.compiled_contexts,
            )
            viewer_states_by_port = settle_viewer_state(
                visualizers,
                progress_queue=validated.progress_queue,
                progress_context=validated,
            )
        else:
            viewer_states_by_port = MappingProxyType({})
        project_execution_state(orchestrator, execution_results)
        if all(result.is_success() for result in execution_results.values()):
            _emit_execution_progress(
                progress_queue=validated.progress_queue,
                progress_context=validated,
                step_name="pipeline",
                phase=ProgressPhase.SUCCESS,
                status=ProgressStatus.SUCCESS,
                completed=len(validated.pipeline_definition),
                total=len(validated.pipeline_definition),
                percent=100.0,
            )
        return CompiledPlateExecutionResults(
            execution_results,
            extras=CompiledPlateExecutionExtras(
                viewer_states_by_port=viewer_states_by_port
            ),
        )
    except ExecutionCancelledError:
        orchestrator._state = OrchestratorState.READY
        logger.info("Compiled plate execution cancelled")
        raise
    except Exception as exc:
        orchestrator._state = OrchestratorState.EXEC_FAILED
        logger.error(f"Failed to execute compiled plate: {exc}")
        raise
    finally:
        try:
            stop_execution_visualizers(visualizers)
        finally:
            try:
                set_progress_queue(None)
            finally:
                orchestrator._execution_cancellation.finish(cancellation)


def validate_compiled_plate_execution(
    orchestrator: "PipelineOrchestrator",
    request: CompiledPlateExecutionRequest,
) -> ValidatedCompiledPlateExecution | None:
    """Validate execute-compiled-plate invariants before worker setup."""

    pipeline_definition = request.pipeline_definition
    compiled_contexts = request.compiled_contexts
    if not orchestrator.is_initialized():
        raise RuntimeError("Orchestrator must be initialized before executing.")
    if not pipeline_definition:
        raise ValueError("A valid (stateless) pipeline definition must be provided.")
    if not compiled_contexts:
        logger.warning("No compiled contexts provided for execution.")
        return None
    if request.progress_queue is None:
        raise ValueError(
            "progress_queue is required for execute_compiled_plate invariant path"
        )
    plate_step_indexes = validate_plate_scoped_contexts(compiled_contexts)
    if (
        plate_step_indexes
        and request.runtime_observation_mode is RuntimeObservationMode.OMIT
    ):
        raise ValueError(
            "Plate-scoped FunctionSteps require RuntimeObservationMode."
            "MERGE_INTO_PARENT so compiled artifact inputs are available after "
            "worker execution."
        )
    runtime_environment = request.execution_bundle.runtime_environment
    return ValidatedCompiledPlateExecution(
        execution_id=request.execution_id,
        plate_id=request.plate_id,
        pipeline_definition=pipeline_definition,
        compiled_contexts=compiled_contexts,
        actual_max_workers=actual_max_workers(runtime_environment, request.max_workers),
        progress_queue=request.progress_queue,
        runtime_environment=runtime_environment,
    )


def validate_plate_scoped_contexts(
    compiled_contexts: Mapping[str, ProcessingContext],
) -> tuple[int, ...]:
    """Validate plate lifecycle invariants shared by compilation contexts."""

    memberships = {
        context_key: tuple(
            step_index
            for step_index, step_plan in sorted(context.step_plans.items())
            if step_plan.execution_scope is FunctionStepExecutionScope.PLATE
        )
        for context_key, context in compiled_contexts.items()
    }
    distinct_memberships = frozenset(memberships.values())
    if len(distinct_memberships) != 1:
        raise ValueError(
            "Plate-scoped step membership drifted between compiled contexts: "
            f"{memberships!r}."
        )
    (plate_step_indexes,) = distinct_memberships
    if not plate_step_indexes:
        return ()

    first_plate_index = plate_step_indexes[0]
    axis_successors = {
        context_key: tuple(
            step_plan.step_name
            for step_index, step_plan in sorted(context.step_plans.items())
            if step_index > first_plate_index
            and step_plan.execution_scope is FunctionStepExecutionScope.AXIS
        )
        for context_key, context in compiled_contexts.items()
    }
    contexts_with_axis_successors = {
        context_key: successors
        for context_key, successors in axis_successors.items()
        if successors
    }
    if contexts_with_axis_successors:
        raise ValueError(
            "Plate-scoped FunctionSteps must be terminal; axis-scoped successors "
            "are not executable after post-plate processing: "
            f"{contexts_with_axis_successors!r}."
        )

    for step_index in plate_step_indexes:
        baseline_invocations: tuple[CompiledFunctionInvocation, ...] | None = None
        baseline_source_binding_plan: CompiledSourceBindingPlan | None = None
        metadata_owners = 0
        for context_key, context in compiled_contexts.items():
            step_plan = context.step_plans[step_index]
            _validate_plate_step_plan(step_plan)
            invocations = step_plan.compiled_function_pattern.default_group.invocations
            if baseline_invocations is None:
                baseline_invocations = invocations
                baseline_source_binding_plan = step_plan.source_binding_plan
            else:
                if invocations != baseline_invocations:
                    raise ValueError(
                        "Plate-scoped compiled invocations drifted between contexts "
                        f"for step {step_index} ({step_plan.step_name!r}); context "
                        f"{context_key!r} differs."
                    )
                if step_plan.source_binding_plan != baseline_source_binding_plan:
                    raise ValueError(
                        "Plate-scoped source-binding plan drifted between contexts "
                        f"for step {step_index} ({step_plan.step_name!r}); context "
                        f"{context_key!r} differs."
                    )
            if step_plan.create_openhcs_metadata:
                metadata_owners += 1
        if metadata_owners != 1:
            raise ValueError(
                f"Plate-scoped step {step_index} requires exactly one "
                f"metadata-writer output owner, got {metadata_owners}."
            )
    return plate_step_indexes


def execute_plate_scoped_steps(
    compiled_contexts: Mapping[str, ProcessingContext],
    *,
    progress_queue: ProgressQueue,
    progress_context: ProgressExecutionContext,
    heartbeat_interval_seconds: float = PLATE_STEP_PROGRESS_HEARTBEAT_SECONDS,
) -> RuntimeExecutionObservation:
    """Invoke plate-scoped FunctionSteps once from merged runtime records."""

    observation_cursors = {
        context_key: context.runtime_value_store.observation_cursor()
        for context_key, context in compiled_contexts.items()
    }
    plate_step_indexes = validate_plate_scoped_contexts(compiled_contexts)
    total_steps = max(next(iter(compiled_contexts.values())).step_plans) + 1
    records_by_axis = _runtime_record_snapshot(compiled_contexts)
    for step_index in plate_step_indexes:
        owner_context, owner_plan = _plate_output_owner(
            compiled_contexts,
            step_index,
        )
        with _PlateStepProgressHeartbeat(
            progress_queue=progress_queue,
            progress_context=progress_context,
            step_index=step_index,
            step_name=owner_plan.step_name,
            total_steps=total_steps,
            interval_seconds=heartbeat_interval_seconds,
        ):
            owner_invocations = (
                owner_plan.compiled_function_pattern.default_group.invocations
            )
            for invocation_position, owner_invocation in enumerate(owner_invocations):
                contract = owner_invocation.contract
                output_plans = tuple(
                    owner_invocation.select_outputs(
                        owner_plan.artifact_outputs
                    ).values()
                )
                (output_plan,) = output_plans
                resolved_output_plan = output_plan.for_group(
                    output_plan.require_single_group_key()
                )
                batch = _plate_artifact_batch(
                    compiled_contexts=compiled_contexts,
                    step_index=step_index,
                    invocation_position=invocation_position,
                    contract=contract,
                    records_by_axis=records_by_axis,
                    source_binding_plan=owner_plan.source_binding_plan,
                    source_image_set_identity_policy=(
                        owner_context.source_image_set_identity_policy
                    ),
                )
                kwargs = owner_invocation.kwargs_dict
                batch_parameter = RuntimeArtifactBatch.require_parameter_name()
                runtime_kwargs: dict[str, object] = {batch_parameter: batch}
                context_parameter = owner_invocation.contract.runtime_context_parameter
                if context_parameter is not None:
                    runtime_kwargs[context_parameter] = owner_context
                conflicting_parameters = kwargs.keys() & runtime_kwargs.keys()
                if conflicting_parameters:
                    raise ValueError(
                        f"Plate-scoped callable "
                        f"{owner_invocation.contract.function_name!r} cannot provide "
                        "runtime-owned kwargs "
                        f"{tuple(sorted(conflicting_parameters))!r}."
                    )
                result = owner_invocation.contract.resolve_runtime_callable()(
                    **kwargs,
                    **runtime_kwargs,
                )
                owner_axis_id = _require_context_axis_id(owner_context)
                runtime_value = RuntimeValue.normalize(
                    resolved_output_plan,
                    result,
                    axis_id=owner_axis_id,
                )
                location = RuntimeArtifactLocation(
                    path=resolved_output_plan.path,
                    backend=Backend.MEMORY.value,
                )
                record = owner_context.runtime_value_store.replace(
                    runtime_value,
                    path=location.path,
                    backend=location.backend,
                )
                replace_runtime_artifact_payload(
                    owner_context.filemanager,
                    runtime_value.data,
                    location,
                )
                records_by_axis = _records_with_output(records_by_axis, record)

            RuntimeArtifactMaterializationAuthority.materialize(
                owner_context,
                owner_plan.require_function_execution_ready(),
            )
        _emit_execution_progress(
            progress_queue=progress_queue,
            progress_context=progress_context,
            step_name=owner_plan.step_name,
            phase=ProgressPhase.STEP_COMPLETED,
            status=ProgressStatus.SUCCESS,
            completed=step_index + 1,
            total=total_steps,
            percent=((step_index + 1) / total_steps) * 100.0,
        )

    return RuntimeExecutionObservation(
        contexts=tuple(
            RuntimeContextObservation(
                context_key=context_key,
                records=records,
            )
            for context_key, context in compiled_contexts.items()
            if (
                records := context.runtime_value_store.observed_values_after(
                    observation_cursors[context_key]
                )
            )
        )
    )


@dataclass(slots=True)
class _PlateStepProgressHeartbeat:
    """Keep one compiler-owned plate step visible while it is executing."""

    progress_queue: ProgressQueue
    progress_context: ProgressExecutionContext
    step_index: int
    step_name: str
    total_steps: int
    interval_seconds: float = PLATE_STEP_PROGRESS_HEARTBEAT_SECONDS
    _stop_event: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()

    def __enter__(self) -> "_PlateStepProgressHeartbeat":
        self._emit(ProgressPhase.STEP_STARTED, ProgressStatus.STARTED)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._emit(ProgressPhase.RUNNING, ProgressStatus.RUNNING)

    def _emit(self, phase: ProgressPhase, status: ProgressStatus) -> None:
        _emit_execution_progress(
            progress_queue=self.progress_queue,
            progress_context=self.progress_context,
            step_name=self.step_name,
            phase=phase,
            status=status,
            completed=self.step_index,
            total=self.total_steps,
            percent=(self.step_index / self.total_steps) * 100.0,
        )


def _emit_execution_progress(
    *,
    progress_queue: ProgressQueue,
    progress_context: ProgressExecutionContext,
    step_name: str,
    phase: ProgressPhase,
    status: ProgressStatus,
    completed: int,
    total: int,
    percent: float,
    context: dict[str, object] | None = None,
) -> None:
    """Publish one parent-owned execution event without a worker claim."""

    progress_queue.put(
        create_event(
            ProgressEventPayload(
                identity=progress_context.identity_for_event(
                    axis_id="",
                    step_name=step_name,
                ),
                phase=phase,
                status=status,
                completed=completed,
                total=total,
                percent=percent,
                context=context,
            )
        ).to_dict()
    )


def _validate_plate_step_plan(step_plan: CompiledStepPlan) -> None:
    pattern = step_plan.compiled_function_pattern
    if pattern is None:
        raise ValueError(
            f"Plate-scoped step {step_plan.step_name!r} has no compiled pattern."
        )
    if pattern.is_grouped or len(pattern.groups) != 1:
        raise ValueError(
            f"Plate-scoped step {step_plan.step_name!r} must use one default group."
        )
    FuncStepContractValidator.validate_compiled_step_plan(step_plan)
    for invocation in pattern.default_group.invocations:
        _validate_plate_invocation(invocation, step_plan)


def _validate_plate_invocation(
    invocation: CompiledFunctionInvocation,
    step_plan: CompiledStepPlan,
) -> None:
    contract = invocation.contract
    FuncStepContractValidator.validate_plate_callable_contracts(
        (contract,),
        step_plan.step_name,
    )
    if invocation.runtime_parameter_bindings:
        raise ValueError(
            f"Plate-scoped callable {contract.function_name!r} cannot declare "
            "axis runtime parameter bindings."
        )
    declared_outputs = tuple(contract.artifact_outputs)
    if (
        len(declared_outputs) != 1
        or declared_outputs[0].artifact_type is not SpecialArtifactType
    ):
        raise ValueError(
            f"Plate-scoped callable {contract.function_name!r} must declare "
            "exactly one SpecialArtifactType output."
        )

    declared_input_refs = contract.artifact_inputs.ref_set()
    selected_input_refs = tuple(
        edge.spec.ref() for edge in invocation.artifact_input_edges
    )
    undeclared_input_refs = tuple(
        ref for ref in selected_input_refs if ref not in declared_input_refs
    )
    missing_required_input_refs = tuple(
        spec.ref()
        for spec in contract.artifact_inputs
        if spec.required and spec.ref() not in selected_input_refs
    )
    if undeclared_input_refs or missing_required_input_refs:
        raise ValueError(
            f"Plate-scoped invocation {invocation.key!r} artifact input refs drifted "
            f"from its module contract; undeclared={undeclared_input_refs!r}, "
            f"missing_required={missing_required_input_refs!r}."
        )

    invocation.select_inputs(step_plan.artifact_inputs)
    selected_output_refs = tuple(
        plan.ref() for plan in invocation.artifact_output_plans
    )
    expected_output_refs = tuple(spec.ref() for spec in declared_outputs)
    if selected_output_refs != expected_output_refs:
        raise ValueError(
            f"Plate-scoped invocation {invocation.key!r} compiled outputs drifted "
            f"from declarations; expected={expected_output_refs!r}, "
            f"actual={selected_output_refs!r}."
        )


def _runtime_record_snapshot(
    compiled_contexts: Mapping[str, ProcessingContext],
) -> Mapping[str, tuple[StoredRuntimeValue, ...]]:
    records: OrderedDict[
        str,
        OrderedDict[tuple[object, RuntimeArtifactLocation], StoredRuntimeValue],
    ] = OrderedDict()
    for context in compiled_contexts.values():
        for record in context.runtime_value_store.values():
            axis_id = record.key.scope.axis_id
            records.setdefault(axis_id, OrderedDict()).setdefault(
                (record.key, record.location),
                record,
            )
    return MappingProxyType(
        {
            axis_id: tuple(axis_records.values())
            for axis_id, axis_records in records.items()
        }
    )


def _plate_output_owner(
    compiled_contexts: Mapping[str, ProcessingContext],
    step_index: int,
) -> tuple[ProcessingContext, CompiledStepPlan]:
    owners = tuple(
        (context, context.step_plans[step_index])
        for context in compiled_contexts.values()
        if context.step_plans[step_index].owns_runtime_outputs
    )
    if len(owners) != 1:
        raise ValueError(
            f"Plate-scoped step {step_index} requires exactly one output owner, "
            f"got {len(owners)}."
        )
    return owners[0]


def _plate_artifact_batch(
    *,
    compiled_contexts: Mapping[str, ProcessingContext],
    step_index: int,
    invocation_position: int,
    contract: CallableContract,
    records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
    source_binding_plan: CompiledSourceBindingPlan,
    source_image_set_identity_policy: "SourceImageSetIdentityPolicy",
) -> RuntimeArtifactBatch:
    selected: OrderedDict[
        str,
        OrderedDict[tuple[object, RuntimeArtifactLocation], StoredRuntimeValue],
    ] = OrderedDict()
    for context in compiled_contexts.values():
        selected.setdefault(_require_context_axis_id(context), OrderedDict())

    for input_spec in contract.artifact_inputs:
        selected_for_spec = 0
        for context in compiled_contexts.values():
            axis_id = _require_context_axis_id(context)
            step_plan = context.step_plans[step_index]
            invocation = step_plan.compiled_function_pattern.default_group.invocations[
                invocation_position
            ]
            selected_input_edges = {
                edge.spec.ref(): edge
                for edge in invocation.select_inputs(step_plan.artifact_inputs).values()
                if edge.storage_plan is not None and edge.projection is not None
            }
            input_ref = input_spec.ref()
            if input_ref not in selected_input_edges:
                continue
            input_edge = selected_input_edges[input_ref]
            input_plan = input_edge.storage_plan
            projection = input_edge.projection
            if input_plan is None or projection is None:
                raise RuntimeError(
                    "Selected plate artifact input lost its storage plan."
                )
            for group_key in projection.producer_selection_scope.keys:
                query = RuntimeArtifactQuery.from_input_plan(
                    input_plan,
                    axis_id=axis_id,
                    backend=Backend.MEMORY.value,
                    group_key=group_key,
                )
                matches = tuple(
                    record
                    for record in records_by_axis.get(axis_id, ())
                    if query.matches(record)
                )
                for record in matches:
                    selected[axis_id].setdefault(
                        (record.key, record.location),
                        record,
                    )
                selected_for_spec += len(matches)
        if input_spec.required and selected_for_spec == 0:
            raise ValueError(
                f"Plate-scoped callable {contract.function_name!r} is missing required "
                f"input {input_spec.artifact_type.value}:"
                f"{input_spec.name!r}."
            )

    return RuntimeArtifactBatch(
        input_specs=contract.artifact_inputs,
        records_by_axis={
            axis_id: tuple(axis_records.values())
            for axis_id, axis_records in selected.items()
        },
        source_binding_plan=source_binding_plan,
        source_image_set_identity_policy=source_image_set_identity_policy,
    )


def _records_with_output(
    records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]],
    record: StoredRuntimeValue,
) -> Mapping[str, tuple[StoredRuntimeValue, ...]]:
    updated = OrderedDict(records_by_axis)
    axis_id = record.key.scope.axis_id
    existing = OrderedDict(
        ((item.key, item.location), item) for item in updated.get(axis_id, ())
    )
    existing[(record.key, record.location)] = record
    updated[axis_id] = tuple(existing.values())
    return MappingProxyType(dict(updated))


def _require_context_axis_id(context: ProcessingContext) -> str:
    if not context.axis_id:
        raise ValueError("Plate-scoped execution requires a compiled context axis_id.")
    return str(context.axis_id)


def actual_max_workers(
    runtime_environment: CompiledRuntimeEnvironmentPlan,
    max_workers: Optional[int],
) -> int:
    """Resolve the worker count from call override or compiled environment."""

    configured_workers = runtime_environment.configured_num_workers
    requested_workers = max_workers if max_workers is not None else configured_workers
    return max(requested_workers, 1)


def bootstrap_execution_visualizers(
    *,
    orchestrator: "PipelineOrchestrator",
    compiled_contexts: Dict[str, ProcessingContext],
    visualizer: ExecutionVisualizerABC | None,
    progress_queue: ProgressQueue,
    progress_context: ProgressExecutionContext,
) -> list[ExecutionVisualizerABC]:
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
) -> list[ExecutionVisualizerABC]:
    """Create one viewer for each distinct visualizer requirement."""

    unique_configs: dict[tuple[str, int], tuple[RequiredVisualizer, object]] = {}
    for ctx in compiled_contexts.values():
        for required_visualizer in ctx.required_visualizers:
            if required_visualizer.key not in unique_configs:
                unique_configs[required_visualizer.key] = (
                    required_visualizer,
                    ctx.visualizer_config,
                )

    visualizers: list[ExecutionVisualizerABC] = []
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
    visualizers: list[ExecutionVisualizerABC],
    progress_queue: ProgressQueue,
    progress_context: ProgressExecutionContext,
) -> None:
    """Wait for all streaming visualizers to report readiness."""

    max_wait = 30.0
    start_time = time.time()

    while time.time() - start_time < max_wait:
        orchestrator._execution_cancellation.raise_if_requested(
            "while waiting for streaming viewers"
        )
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
    message = f"Timeout waiting for streaming viewers. Not ready: {not_ready}"
    logger.error(f"🔬 ORCHESTRATOR: {message}")
    progress_queue.put(
        create_event(
            ProgressEventPayload(
                identity=progress_context.identity_for_event(axis_id="", step_name=""),
                phase=ProgressPhase.INIT,
                status=ProgressStatus.FAILED,
                percent=0.0,
                message=message,
            )
        ).to_dict()
    )
    raise TimeoutError(message)


def clear_viewer_state(visualizers: list[ExecutionVisualizerABC]) -> None:
    """Clear viewer state before sending a new execution stream."""

    for vis in visualizers:
        if not vis.clear_viewer_state():
            raise RuntimeError(f"Failed to clear state for viewer on port {vis.port}.")


@dataclass(slots=True)
class ViewerSettlementProgressObserver:
    """Project authoritative viewer progress into the execution event stream."""

    progress_queue: ProgressQueue
    progress_context: ProgressExecutionContext
    viewer_port: int
    heartbeat_interval_seconds: float = PLATE_STEP_PROGRESS_HEARTBEAT_SECONDS
    _last_progress: "ViewerSettleProgress | None" = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_emitted_at: float = field(
        default=float("-inf"),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("Viewer settlement heartbeat interval must be positive.")

    def __call__(self, progress: "ViewerSettleProgress") -> None:
        observed_at = time.monotonic()
        if progress == self._last_progress and (
            not progress.active_route_work_unit_active
            or observed_at - self._last_emitted_at < self.heartbeat_interval_seconds
        ):
            return

        self._last_progress = progress
        self._last_emitted_at = observed_at
        total = progress.total_update_count
        percent = (
            100.0 if total == 0 else (progress.completed_update_count / total) * 100.0
        )
        _emit_execution_progress(
            progress_queue=self.progress_queue,
            progress_context=self.progress_context,
            step_name=f"viewer_settlement_{self.viewer_port}",
            phase=ProgressPhase.VIEWER_SETTLEMENT,
            status=ProgressStatus.RUNNING,
            completed=progress.completed_update_count,
            total=total,
            percent=percent,
            context={
                "viewer_port": self.viewer_port,
                **progress.to_wire_mapping(),
            },
        )


def settle_viewer_state(
    visualizers: list[ExecutionVisualizerABC],
    *,
    progress_queue: ProgressQueue | None = None,
    progress_context: ProgressExecutionContext | None = None,
) -> Mapping[int, "ViewerControlResponse"]:
    """Drain queued updates and capture state before transient viewer shutdown."""

    if (progress_queue is None) is not (progress_context is None):
        raise ValueError(
            "Viewer settlement progress requires both queue and execution context."
        )

    for vis in visualizers:
        observer = (
            None
            if progress_queue is None or progress_context is None
            else ViewerSettlementProgressObserver(
                progress_queue,
                progress_context,
                vis.port,
            )
        )
        settled = (
            vis.settle_viewer_state()
            if observer is None
            else vis.settle_viewer_state(progress_callback=observer)
        )
        if not settled:
            raise RuntimeError(
                f"Failed to settle streamed updates for viewer on port {vis.port}."
            )

    viewer_states: dict[int, ViewerControlResponse] = {}
    for vis in visualizers:
        if vis.persistent:
            continue
        if vis.port in viewer_states:
            raise RuntimeError(
                f"Multiple execution viewers declared the same port {vis.port}."
            )
        viewer_states[vis.port] = vis.read_viewer_state()
    return MappingProxyType(viewer_states)


def stop_execution_visualizers(visualizers: list[ExecutionVisualizerABC]) -> None:
    """Stop auto-created non-persistent visualizers after execution."""

    for vis in visualizers:
        if vis.persistent:
            continue
        vis.force_stop()
        if vis.is_running:
            raise RuntimeError(
                "Non-persistent viewer on port "
                f"{vis.port} remained active after cleanup."
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
