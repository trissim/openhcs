from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.core.compiled_execution import (
    CompiledExecutionBundle,
    CompiledRuntimeEnvironmentPlan,
    CompiledWorkerStartPlan,
)
from openhcs.core.config import MultiprocessingStartMethod
from openhcs.core.debug import NoOpDebugExecutionPolicy
from openhcs.core.orchestrator import (
    analysis_consolidation as analysis_consolidation_module,
)
from openhcs.core.orchestrator import (
    compiled_plate_execution as compiled_plate_execution_module,
)
from openhcs.core.orchestrator import orchestrator as orchestrator_module
from openhcs.core.orchestrator import worker_execution as worker_execution_module
from openhcs.core.orchestrator.analysis_consolidation import (
    consolidate_analysis_outputs,
)
from openhcs.core.orchestrator.cancellation import (
    ExecutionCancellationAuthority,
    ExecutionCancellationSignal,
    ExecutionCancelledError,
)
from openhcs.core.orchestrator.compiled_plate_execution import (
    CompiledPlateExecutionRequest,
    CompiledPlateExecutionResults,
    clear_viewer_state,
    execute_compiled_plate_request,
    project_execution_state,
    settle_viewer_state,
    stop_execution_visualizers,
    validate_compiled_plate_execution,
    wait_until_visualizers_ready,
)
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeExecutionObservation,
    RuntimeObservationMode,
)
from openhcs.core.orchestrator.worker_execution import (
    ForkInheritedWorkerExecutorResources,
    ForkInheritedWorkerLaneRunner,
    InlineWorkerExecutorResources,
    PooledWorkerExecutorResources,
    PooledWorkerLaneRunner,
    WorkerExecutorFactory,
)
from openhcs.core.orchestrator.worker_lanes import (
    CompiledContextLanePlanner,
    ForkInheritedWorkerExecutionState,
    WorkerAssignmentPlan,
    WorkerLaneExecutionPlan,
)
from openhcs.core.progress import ProgressEvent, ProgressExecutionContext, ProgressPhase
from openhcs.runtime.viewer_protocol import (
    ViewerControlResponse,
    ViewerSettlePhase,
    ViewerSettleProgress,
)

PROGRESS_CONTEXT = ProgressExecutionContext(
    execution_id="exec",
    plate_id="plate",
)


def _runtime_environment(
    *,
    use_threading: bool,
    start_method: MultiprocessingStartMethod,
    configured_num_workers: int = 4,
) -> CompiledRuntimeEnvironmentPlan:
    return CompiledRuntimeEnvironmentPlan(
        worker_start=CompiledWorkerStartPlan(
            requested=start_method,
            resolved=start_method,
            reason="test",
            gpu_enabled=False,
            server_mode=False,
        ),
        use_threading=use_threading,
        configured_num_workers=configured_num_workers,
    )


def _execute_with_visualizer(monkeypatch, visualizer, *, progress_queue=None):
    context = SimpleNamespace(step_plans={})
    runtime_environment = _runtime_environment(
        use_threading=True,
        start_method=MultiprocessingStartMethod.SPAWN,
        configured_num_workers=1,
    )
    bundle = CompiledExecutionBundle(
        pipeline_definition=("step",),
        runtime_contexts={"A01": context},
        transport_contexts={"A01": context},
        worker_assignments=None,
        runtime_environment=runtime_environment,
    )
    request = CompiledPlateExecutionRequest(
        execution_id="exec",
        plate_id="plate",
        execution_bundle=bundle,
        max_workers=1,
        visualizer=None,
        log_file_base=None,
        progress_queue=progress_queue or SimpleNamespace(put=lambda _event: None),
        runtime_observation_mode=RuntimeObservationMode.MERGE_INTO_PARENT,
        debug_execution_policy=NoOpDebugExecutionPolicy(),
    )

    class FakeExecutorResources:
        executor = None

        def install_execution_bundle(self, _bundle):
            return None

        def execution_context(self):
            return nullcontext()

        def plan_worker_lanes(self, **_kwargs):
            return WorkerAssignmentPlan(
                worker_assignments={"worker_0": ["A01"]},
                lane_axis_contexts={"worker_0": [("A01", [("A01", context)])]},
            )

        def run_worker_lanes(self, **_kwargs):
            return {"A01": ExecutionResult.success("A01")}

        def shutdown_executor(self):
            return None

        def clear_execution_bundle(self):
            return None

        def release_parent_runtime_resources(self, _bundle):
            return None

    resources = FakeExecutorResources()
    monkeypatch.setattr(
        compiled_plate_execution_module,
        "WorkerExecutorFactory",
        lambda **_kwargs: SimpleNamespace(create=lambda **_create_kwargs: resources),
    )
    monkeypatch.setattr(
        compiled_plate_execution_module,
        "bootstrap_execution_visualizers",
        lambda **_kwargs: [visualizer],
    )
    monkeypatch.setattr(
        compiled_plate_execution_module,
        "execute_plate_scoped_steps",
        lambda _contexts, **_kwargs: RuntimeExecutionObservation(),
    )
    monkeypatch.setattr(
        compiled_plate_execution_module,
        "consolidate_analysis_outputs",
        lambda _contexts, _results, **_kwargs: None,
    )
    orchestrator = SimpleNamespace(
        _execution_cancellation=ExecutionCancellationAuthority(),
        _executor=None,
        _state=None,
        is_initialized=lambda: True,
        microscope_handler=object(),
    )
    return execute_compiled_plate_request(orchestrator, request)


def test_lane_planner_generates_stable_default_assignments_and_groups_combos():
    planner = CompiledContextLanePlanner(
        actual_max_workers=2,
        fork_inherited_execution=False,
    )

    plan = planner.plan(
        {
            "B01": "b-context",
            "A01__combo_0": "a0-context",
            "A01__combo_1": "a1-context",
            "C01": "c-context",
        },
        worker_assignments=None,
    )

    assert plan.worker_assignments == {
        "worker_0": ["A01", "C01"],
        "worker_1": ["B01"],
    }
    assert plan.lane_axis_contexts == {
        "worker_0": [
            (
                "A01",
                [
                    ("A01__combo_0", "a0-context"),
                    ("A01__combo_1", "a1-context"),
                ],
            ),
            ("C01", [("C01", "c-context")]),
        ],
        "worker_1": [("B01", [("B01", "b-context")])],
    }


def test_worker_axis_completion_stops_before_terminal_plate_steps() -> None:
    context = SimpleNamespace(
        step_plans={
            **{
                index: SimpleNamespace(execution_scope=FunctionStepExecutionScope.AXIS)
                for index in range(32)
            },
            32: SimpleNamespace(execution_scope=FunctionStepExecutionScope.PLATE),
        }
    )

    assert worker_execution_module._completed_axis_step_count(context, 33) == 32


def test_compiled_plate_execution_request_uses_bundle_as_runtime_authority():
    runtime_environment = _runtime_environment(
        use_threading=True,
        start_method=MultiprocessingStartMethod.SPAWN,
        configured_num_workers=7,
    )
    context = SimpleNamespace(step_plans={})
    bundle = CompiledExecutionBundle(
        pipeline_definition=("step",),
        runtime_contexts={"A01": context},
        transport_contexts={"A01": context},
        worker_assignments={"worker_0": ["A01"]},
        runtime_environment=runtime_environment,
    )
    request = CompiledPlateExecutionRequest(
        execution_id="exec",
        plate_id="plate",
        execution_bundle=bundle,
        max_workers=None,
        visualizer=None,
        log_file_base=None,
        progress_queue="queue",
        runtime_observation_mode=RuntimeObservationMode.MERGE_INTO_PARENT,
        debug_execution_policy=NoOpDebugExecutionPolicy(),
    )
    orchestrator = SimpleNamespace(
        is_initialized=lambda: True,
    )

    validated = validate_compiled_plate_execution(orchestrator, request)

    assert validated is not None
    assert validated.pipeline_definition == ["step"]
    assert validated.compiled_contexts == {"A01": context}
    assert validated.runtime_environment is runtime_environment
    assert validated.actual_max_workers == 7


def test_plate_scope_rejects_omitted_runtime_observations(monkeypatch):
    runtime_environment = _runtime_environment(
        use_threading=True,
        start_method=MultiprocessingStartMethod.SPAWN,
    )
    context = SimpleNamespace(step_plans={})
    bundle = CompiledExecutionBundle(
        pipeline_definition=("plate-step",),
        runtime_contexts={"A01": context},
        transport_contexts={"A01": context},
        worker_assignments=None,
        runtime_environment=runtime_environment,
    )
    request = CompiledPlateExecutionRequest(
        execution_id="exec",
        plate_id="plate",
        execution_bundle=bundle,
        max_workers=None,
        visualizer=None,
        log_file_base=None,
        progress_queue="queue",
        runtime_observation_mode=RuntimeObservationMode.OMIT,
        debug_execution_policy=NoOpDebugExecutionPolicy(),
    )
    monkeypatch.setattr(
        compiled_plate_execution_module,
        "validate_plate_scoped_contexts",
        lambda _contexts: (1,),
    )

    with pytest.raises(ValueError, match="MERGE_INTO_PARENT"):
        validate_compiled_plate_execution(
            SimpleNamespace(is_initialized=lambda: True),
            request,
        )


def test_lane_planner_preserves_fork_lane_payload_as_context_keys():
    planner = CompiledContextLanePlanner(
        actual_max_workers=2,
        fork_inherited_execution=True,
    )

    plan = planner.plan(
        {
            "A01__combo_0": "runtime-a0",
            "A01__combo_1": "runtime-a1",
            "B01": "runtime-b",
        },
        worker_assignments={"worker_0": ["A01"], "worker_1": ["B01"]},
    )

    assert plan.lane_axis_contexts == {
        "worker_0": [("A01", ["A01__combo_0", "A01__combo_1"])],
        "worker_1": [("B01", ["B01"])],
    }


def test_lane_planner_rejects_duplicate_axis_ownership():
    planner = CompiledContextLanePlanner(
        actual_max_workers=2,
        fork_inherited_execution=False,
    )

    with pytest.raises(RuntimeError, match="Duplicate axis ownership"):
        planner.plan(
            {"A01": "a-context", "B01": "b-context"},
            worker_assignments={"worker_0": ["A01"], "worker_1": ["A01", "B01"]},
        )


def test_lane_planner_rejects_missing_axis_ownership():
    planner = CompiledContextLanePlanner(
        actual_max_workers=2,
        fork_inherited_execution=False,
    )

    with pytest.raises(RuntimeError, match="worker_assignments mismatch"):
        planner.plan(
            {"A01": "a-context", "B01": "b-context"},
            worker_assignments={"worker_0": ["A01"]},
        )


def test_executor_factory_uses_inline_lane_for_single_threaded_worker(monkeypatch):
    context = object()
    monkeypatch.setattr(
        worker_execution_module.multiprocessing,
        "get_context",
        lambda method: context,
    )

    resources = WorkerExecutorFactory(
        log_file_base=None,
        progress_queue="queue",
        progress_context=PROGRESS_CONTEXT,
        cancellation=ExecutionCancellationSignal(),
    ).create(
        runtime_environment=_runtime_environment(
            use_threading=True,
            start_method=MultiprocessingStartMethod.SPAWN,
        ),
        actual_max_workers=1,
    )

    assert resources.executor is None
    assert resources.multiprocessing_context is context
    assert isinstance(resources, InlineWorkerExecutorResources)
    assert resources.uses_fork_inherited_contexts is False
    assert resources.use_multiprocessing is False


def test_executor_factory_uses_inline_lane_for_single_fork_worker(monkeypatch):
    context = object()
    monkeypatch.setattr(
        worker_execution_module.multiprocessing,
        "get_context",
        lambda method: context,
    )

    resources = WorkerExecutorFactory(
        log_file_base="/tmp/worker",
        progress_queue="queue",
        progress_context=PROGRESS_CONTEXT,
        cancellation=ExecutionCancellationSignal(),
    ).create(
        runtime_environment=_runtime_environment(
            use_threading=False,
            start_method=MultiprocessingStartMethod.FORK,
        ),
        actual_max_workers=1,
    )

    assert resources.executor is None
    assert resources.multiprocessing_context is context
    assert isinstance(resources, InlineWorkerExecutorResources)
    assert resources.uses_fork_inherited_contexts is False
    assert resources.use_multiprocessing is True


def test_executor_factory_creates_thread_pool_for_multi_worker_threading(monkeypatch):
    created = {}
    context = object()

    class FakeThreadPoolExecutor:
        def __init__(self, *, max_workers):
            created["max_workers"] = max_workers

    monkeypatch.setattr(
        worker_execution_module.multiprocessing,
        "get_context",
        lambda method: context,
    )
    monkeypatch.setattr(
        worker_execution_module.concurrent.futures,
        "ThreadPoolExecutor",
        FakeThreadPoolExecutor,
    )

    resources = WorkerExecutorFactory(
        log_file_base=None,
        progress_queue="queue",
        progress_context=PROGRESS_CONTEXT,
        cancellation=ExecutionCancellationSignal(),
    ).create(
        runtime_environment=_runtime_environment(
            use_threading=True,
            start_method=MultiprocessingStartMethod.SPAWN,
        ),
        actual_max_workers=3,
    )

    assert isinstance(resources.executor, FakeThreadPoolExecutor)
    assert created == {"max_workers": 3}
    assert isinstance(resources, PooledWorkerExecutorResources)
    assert resources.uses_fork_inherited_contexts is False
    assert resources.use_multiprocessing is False


def test_executor_factory_uses_fork_inherited_lane_without_pool(monkeypatch):
    context = object()
    monkeypatch.setattr(
        worker_execution_module.multiprocessing,
        "get_context",
        lambda method: context,
    )

    resources = WorkerExecutorFactory(
        log_file_base="/tmp/worker",
        progress_queue="queue",
        progress_context=PROGRESS_CONTEXT,
        cancellation=ExecutionCancellationSignal(),
    ).create(
        runtime_environment=_runtime_environment(
            use_threading=False,
            start_method=MultiprocessingStartMethod.FORK,
        ),
        actual_max_workers=2,
    )

    assert resources.executor is None
    assert resources.multiprocessing_context is context
    assert isinstance(resources, ForkInheritedWorkerExecutorResources)
    assert resources.uses_fork_inherited_contexts is True
    assert resources.use_multiprocessing is True


def test_executor_factory_creates_process_pool_with_worker_initializer(monkeypatch):
    created = {}
    context = object()

    class FakeProcessPoolExecutor:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(
        worker_execution_module.multiprocessing,
        "get_context",
        lambda method: context,
    )
    monkeypatch.setattr(
        worker_execution_module.concurrent.futures,
        "ProcessPoolExecutor",
        FakeProcessPoolExecutor,
    )
    runtime_environment = _runtime_environment(
        use_threading=False,
        start_method=MultiprocessingStartMethod.SPAWN,
    )

    resources = WorkerExecutorFactory(
        log_file_base="/tmp/worker-log",
        progress_queue="queue",
        progress_context=PROGRESS_CONTEXT,
        cancellation=ExecutionCancellationSignal(),
    ).create(
        runtime_environment=runtime_environment,
        actual_max_workers=4,
    )

    assert isinstance(resources.executor, FakeProcessPoolExecutor)
    assert created["max_workers"] == 4
    assert created["mp_context"] is context
    assert created["initializer"] is worker_execution_module._configure_worker_process
    assert created["initargs"] == (
        "/tmp/worker-log",
        "queue",
        PROGRESS_CONTEXT,
    )
    assert isinstance(resources, PooledWorkerExecutorResources)
    assert resources.uses_fork_inherited_contexts is False
    assert resources.use_multiprocessing is True


def test_worker_process_initializer_does_not_prepare_global_function_registry(
    monkeypatch,
) -> None:
    from openhcs.processing import func_registry

    initialized = []
    monkeypatch.setenv("OPENHCS_CPU_ONLY", "true")
    monkeypatch.setattr(
        worker_execution_module,
        "configure_native_thread_count",
        lambda _count: None,
    )
    monkeypatch.setattr(
        func_registry,
        "initialize_registry",
        lambda: initialized.append(True),
    )

    worker_execution_module._configure_worker_process(
        None,
    )

    assert initialized == []


def test_pooled_worker_lane_runner_submits_and_collects_lane_results(monkeypatch):
    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def submit(self, fn, *args):
            return FakeFuture(fn(*args))

    def fake_execute_worker_lane(
        pipeline_definition,
        lane_axis_contexts,
        lane_context,
        runtime_observation_mode,
        cancellation,
        release_axis_resources,
    ):
        assert release_axis_resources is True
        axis_id = lane_context.owned_wells[0]
        return {axis_id: ExecutionResult.success(axis_id)}

    monkeypatch.setattr(
        worker_execution_module.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        worker_execution_module,
        "execute_worker_lane",
        fake_execute_worker_lane,
    )

    execution_plan = WorkerLaneExecutionPlan(
        execution_id=PROGRESS_CONTEXT.execution_id,
        plate_id=PROGRESS_CONTEXT.plate_id,
        debug_execution_policy=NoOpDebugExecutionPolicy(),
        assignments=WorkerAssignmentPlan(
            worker_assignments={"worker_0": ["A01"]},
            lane_axis_contexts={"worker_0": [("A01", [("A01", object())])]},
        ),
        runtime_observation_mode=RuntimeObservationMode.OMIT,
    )

    results = PooledWorkerLaneRunner(FakeExecutor(), cancellation=None).run(
        pipeline_definition=[],
        execution_plan=execution_plan,
        parent_contexts={},
    )

    assert results == {"A01": ExecutionResult.success("A01")}


def test_worker_lane_honours_cancellation_before_next_axis(monkeypatch):
    cancellation = ExecutionCancellationSignal()
    visited = []

    def execute_axis(
        pipeline_definition,
        axis_contexts,
        lane_context,
        runtime_observation_mode,
        cancellation,
        release_axis_resources,
    ):
        assert release_axis_resources is True
        axis_id = axis_contexts[0][1].axis_id
        visited.append(axis_id)
        cancellation.request()
        return ExecutionResult.success(axis_id)

    monkeypatch.setattr(
        worker_execution_module,
        "_execute_axis_with_sequential_combinations",
        execute_axis,
    )
    lane_context = SimpleNamespace()
    lane_axis_contexts = [
        ("A01", [("A01", SimpleNamespace(axis_id="A01"))]),
        ("B01", [("B01", SimpleNamespace(axis_id="B01"))]),
    ]

    with pytest.raises(ExecutionCancelledError, match="before axis B01"):
        worker_execution_module.execute_worker_lane(
            pipeline_definition=[],
            lane_axis_contexts=lane_axis_contexts,
            lane_context=lane_context,
            runtime_observation_mode=RuntimeObservationMode.OMIT,
            cancellation=cancellation,
        )

    assert visited == ["A01"]


def test_cancellation_authority_preserves_pre_entry_request_for_exact_scope():
    authority = ExecutionCancellationAuthority()
    authority.request()

    first_scope = authority.begin()
    with pytest.raises(ExecutionCancelledError, match="first scope"):
        first_scope.raise_if_requested("in first scope")
    authority.finish(first_scope)

    second_scope = authority.begin()
    second_scope.raise_if_requested("in second scope")
    authority.finish(second_scope)


def test_pooled_worker_lane_runner_submits_stripped_pipeline_shells(monkeypatch):
    from openhcs.core.steps.function_step import FunctionStep

    submitted_pipeline = []
    stripped_step = FunctionStep(func=lambda image: image, name="Stripped")
    for field_name in tuple(vars(stripped_step)):
        delattr(stripped_step, field_name)

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def submit(self, fn, *args):
            return FakeFuture(fn(*args))

    def fake_execute_worker_lane(
        pipeline_definition,
        lane_axis_contexts,
        lane_context,
        runtime_observation_mode,
        cancellation,
        release_axis_resources,
    ):
        assert release_axis_resources is True
        submitted_pipeline.extend(pipeline_definition)
        return {"A01": ExecutionResult.success("A01")}

    monkeypatch.setattr(
        worker_execution_module.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        worker_execution_module,
        "execute_worker_lane",
        fake_execute_worker_lane,
    )

    execution_plan = WorkerLaneExecutionPlan(
        execution_id=PROGRESS_CONTEXT.execution_id,
        plate_id=PROGRESS_CONTEXT.plate_id,
        debug_execution_policy=NoOpDebugExecutionPolicy(),
        assignments=WorkerAssignmentPlan(
            worker_assignments={"worker_0": ["A01"]},
            lane_axis_contexts={"worker_0": [("A01", [("A01", object())])]},
        ),
        runtime_observation_mode=RuntimeObservationMode.OMIT,
    )

    PooledWorkerLaneRunner(FakeExecutor(), cancellation=None).run(
        pipeline_definition=[stripped_step],
        execution_plan=execution_plan,
        parent_contexts={},
    )

    assert submitted_pipeline == [stripped_step]


def test_fork_inherited_runner_executes_single_lane_without_self_merge(monkeypatch):
    executed = []

    class MergeForbiddenObservation:
        def merge_into(self, _contexts):
            raise AssertionError(
                "fork-inherited inline execution already wrote into parent contexts"
            )

    class ForbiddenMultiprocessingContext:
        def Pipe(self, *, duplex):
            raise AssertionError("single active lane must not open a pipe")

        def Process(self, **kwargs):
            raise AssertionError("single active lane must not launch a process")

    def fake_execute_worker_lane(
        pipeline_definition,
        lane_axis_contexts,
        lane_context,
        runtime_observation_mode,
    ):
        executed.append((pipeline_definition, lane_axis_contexts))
        return {
            "A01": ExecutionResult.success(
                "A01",
                runtime_observation=MergeForbiddenObservation(),
            )
        }

    monkeypatch.setattr(
        worker_execution_module,
        "execute_worker_lane",
        fake_execute_worker_lane,
    )
    ForkInheritedWorkerExecutionState.install(
        CompiledExecutionBundle(
            pipeline_definition=["step"],
            runtime_contexts={"A01": "runtime-context"},
            transport_contexts={},
            worker_assignments={"worker_0": ["A01"]},
            runtime_environment=_runtime_environment(
                use_threading=False,
                start_method=MultiprocessingStartMethod.FORK,
            ),
        )
    )
    try:
        execution_plan = WorkerLaneExecutionPlan(
            execution_id=PROGRESS_CONTEXT.execution_id,
            plate_id=PROGRESS_CONTEXT.plate_id,
            debug_execution_policy=NoOpDebugExecutionPolicy(),
            assignments=WorkerAssignmentPlan(
                worker_assignments={"worker_0": ["A01"]},
                lane_axis_contexts={"worker_0": [("A01", ["A01"])]},
            ),
            runtime_observation_mode=RuntimeObservationMode.MERGE_INTO_PARENT,
        )

        results = ForkInheritedWorkerLaneRunner(ForbiddenMultiprocessingContext()).run(
            execution_plan
        )
    finally:
        ForkInheritedWorkerExecutionState.clear()

    assert results["A01"].is_success()
    assert executed == [(["step"], [("A01", [("A01", "runtime-context")])])]


def test_pooled_worker_lane_runner_emits_error_before_reraising(monkeypatch):
    emitted = []

    class FakeFuture:
        def result(self):
            raise RuntimeError("lane exploded")

    class FakeExecutor:
        def submit(self, fn, *args):
            return FakeFuture()

    monkeypatch.setattr(
        worker_execution_module.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        worker_execution_module,
        "emit",
        lambda **kwargs: emitted.append(kwargs),
    )

    execution_plan = WorkerLaneExecutionPlan(
        execution_id=PROGRESS_CONTEXT.execution_id,
        plate_id=PROGRESS_CONTEXT.plate_id,
        debug_execution_policy=NoOpDebugExecutionPolicy(),
        assignments=WorkerAssignmentPlan(
            worker_assignments={"worker_0": ["A01"]},
            lane_axis_contexts={"worker_0": [("A01", [("A01", object())])]},
        ),
        runtime_observation_mode=RuntimeObservationMode.OMIT,
    )

    with pytest.raises(RuntimeError, match="lane exploded"):
        PooledWorkerLaneRunner(FakeExecutor(), cancellation=None).run(
            pipeline_definition=[],
            execution_plan=execution_plan,
            parent_contexts={},
        )

    assert emitted[0]["execution_id"] == "exec"
    assert emitted[0]["plate_id"] == "plate"
    assert emitted[0]["axis_id"] == "A01"
    assert emitted[0]["worker_slot"] == "worker_0"
    assert emitted[0]["owned_wells"] == ["A01"]


def test_executor_shutdown_plan_swallows_broken_pool_errors(caplog):
    class BrokenExecutor:
        def shutdown(self, **kwargs):
            raise worker_execution_module.concurrent.futures.process.BrokenProcessPool(
                "dead"
            )

    PooledWorkerExecutorResources(
        multiprocessing_context=object(),
        use_multiprocessing=True,
        _executor=BrokenExecutor(),
        cancellation=None,
    ).shutdown_executor()

    assert "broken process pool" in caplog.text


def test_analysis_consolidation_skips_disabled_config():
    context = SimpleNamespace(
        analysis_consolidation_config=SimpleNamespace(enabled=False),
        step_plans={},
    )

    consolidate_analysis_outputs(
        {"A01": context},
        {},
        plate_runtime_observation=RuntimeExecutionObservation(),
    )


def test_analysis_consolidation_propagates_runtime_failures(monkeypatch):
    context = SimpleNamespace(
        analysis_consolidation_config=SimpleNamespace(enabled=True),
        plate_metadata_config=object(),
        plate_path="/plate",
        filemanager=object(),
    )
    inputs = analysis_consolidation_module.RuntimeAnalysisConsolidationInputs(
        outputs_by_directory={},
        destination=analysis_consolidation_module.RuntimeAnalysisSummaryDestination(
            backend="memory",
            images_dir="/plate/images",
        ),
    )
    monkeypatch.setattr(
        analysis_consolidation_module,
        "execution_analysis_outputs",
        lambda *args: inputs,
    )
    monkeypatch.setattr(
        analysis_consolidation_module,
        "consolidate_runtime_analysis_table_output_groups",
        lambda **kwargs: ([], [("results", "invalid CSV")]),
    )

    with pytest.raises(RuntimeError, match="invalid CSV"):
        consolidate_analysis_outputs(
            {"A01": context},
            {},
            plate_runtime_observation=RuntimeExecutionObservation(),
        )


def test_execution_state_projector_maps_success_and_failure():
    orchestrator = SimpleNamespace(_state=None)

    project_execution_state(orchestrator, {"A01": ExecutionResult.success("A01")})
    assert orchestrator._state is orchestrator_module.OrchestratorState.COMPLETED

    project_execution_state(orchestrator, {"A01": ExecutionResult.error("A01")})
    assert orchestrator._state is orchestrator_module.OrchestratorState.EXEC_FAILED


def test_execution_visualizer_cleanup_stops_only_non_persistent_visualizers():
    stopped = []
    persistent = SimpleNamespace(
        persistent=True, force_stop=lambda: stopped.append("p")
    )
    transient = SimpleNamespace(
        persistent=False,
        port=5563,
        is_running=False,
        force_stop=lambda: stopped.append("t"),
    )

    stop_execution_visualizers([persistent, transient])

    assert stopped == ["t"]


def test_execution_visualizer_cleanup_rejects_active_non_persistent_viewer():
    transient = SimpleNamespace(
        persistent=False,
        port=5563,
        is_running=True,
        force_stop=lambda: None,
    )

    with pytest.raises(RuntimeError, match="remained active"):
        stop_execution_visualizers([transient])


def test_execution_visualizer_state_clear_failure_is_fatal():
    visualizer = SimpleNamespace(port=5563, clear_viewer_state=lambda: False)

    with pytest.raises(RuntimeError, match="Failed to clear state"):
        clear_viewer_state([visualizer])


def test_execution_visualizer_settle_failure_is_fatal():
    visualizer = SimpleNamespace(
        port=5563,
        persistent=False,
        settle_viewer_state=lambda: False,
    )

    with pytest.raises(RuntimeError, match="Failed to settle streamed updates"):
        settle_viewer_state([visualizer])


def test_compiled_execution_returns_settled_nonpersistent_viewer_state_before_cleanup(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        compiled_plate_execution_module.OpenHCSMetadataWriter,
        "finalize_completed_plate",
        lambda _contexts: events.append("metadata"),
    )
    response = ViewerControlResponse(
        payload={
            "status": "success",
            "layers": (
                {
                    "route_key": "step-2:objects",
                    "data_types": ("roi",),
                    "item_count": 2,
                    "payload_summary_count": 2,
                    "component_values": ({"channel": "DAPI", "site": 0, "z_index": 0},),
                },
            ),
        }
    )

    class TransientViewer:
        port = 5563
        persistent = False

        def __init__(self):
            self.running = True

        @property
        def is_running(self):
            return self.running

        def settle_viewer_state(self, *, progress_callback=None):
            events.append("settle")
            if progress_callback is not None:
                progress_callback(
                    ViewerSettleProgress(
                        ViewerSettlePhase.RUNNING,
                        1,
                        2,
                        "large-rois",
                        4,
                        True,
                    )
                )
            return True

        def read_viewer_state(self):
            events.append("capture")
            return response

        def force_stop(self):
            events.append("stop")
            self.running = False

    progress_queue = SimpleNamespace(
        put=lambda raw_event: events.append(
            ProgressEvent.from_dict(raw_event).phase.value
        )
    )
    results = _execute_with_visualizer(
        monkeypatch,
        TransientViewer(),
        progress_queue=progress_queue,
    )

    assert isinstance(results, CompiledPlateExecutionResults)
    assert results == {"A01": ExecutionResult.success("A01")}
    assert results.extras.viewer_states_by_port == {5563: response}
    layer = results.extras.viewer_states_by_port[5563].payload["layers"][0]
    assert layer["route_key"] == "step-2:objects"
    assert layer["data_types"] == ("roi",)
    assert layer["item_count"] == layer["payload_summary_count"] == 2
    assert layer["component_values"] == ({"channel": "DAPI", "site": 0, "z_index": 0},)
    assert events == [
        "metadata",
        "settle",
        ProgressPhase.VIEWER_SETTLEMENT.value,
        "capture",
        ProgressPhase.SUCCESS.value,
        "stop",
    ]


def test_viewer_settlement_progress_throttles_unchanged_active_observations(
    monkeypatch,
):
    active = ViewerSettleProgress(
        ViewerSettlePhase.RUNNING,
        1,
        2,
        "large-rois",
        4,
        True,
    )
    complete = ViewerSettleProgress.complete(2)
    observed_events = []

    class Viewer:
        port = 5563
        persistent = False

        def settle_viewer_state(self, *, progress_callback=None):
            assert progress_callback is not None
            for progress in (active, active, active, complete):
                progress_callback(progress)
            return True

        def read_viewer_state(self):
            return ViewerControlResponse(payload={"status": "success"})

    monotonic_values = iter((0.0, 0.5, 2.1, 2.2))
    monkeypatch.setattr(
        compiled_plate_execution_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    states = settle_viewer_state(
        [Viewer()],
        progress_queue=SimpleNamespace(put=observed_events.append),
        progress_context=PROGRESS_CONTEXT,
    )

    assert states == {5563: ViewerControlResponse(payload={"status": "success"})}
    progress_events = [ProgressEvent.from_dict(event) for event in observed_events]
    assert len(progress_events) == 3
    assert [
        event.context["active_route_work_unit_active"] for event in progress_events
    ] == [True, True, False]


def test_compiled_execution_cleans_up_after_viewer_state_capture_failure(monkeypatch):
    events = []

    class FailingTransientViewer:
        port = 5563
        persistent = False

        def __init__(self):
            self.running = True

        @property
        def is_running(self):
            return self.running

        def settle_viewer_state(self, *, progress_callback=None):
            del progress_callback
            events.append("settle")
            return True

        def read_viewer_state(self):
            events.append("capture")
            raise RuntimeError("state unavailable")

        def force_stop(self):
            events.append("stop")
            self.running = False

    with pytest.raises(RuntimeError, match="state unavailable"):
        _execute_with_visualizer(monkeypatch, FailingTransientViewer())

    assert events == ["settle", "capture", "stop"]


def test_persistent_execution_viewer_is_settled_without_capture_or_cleanup():
    events = []
    persistent = SimpleNamespace(
        port=5564,
        persistent=True,
        settle_viewer_state=lambda: events.append("settle") or True,
        read_viewer_state=lambda: events.append("capture"),
        force_stop=lambda: events.append("stop"),
    )

    viewer_states = settle_viewer_state([persistent])
    stop_execution_visualizers([persistent])

    assert viewer_states == {}
    assert events == ["settle"]


def test_execution_visualizer_readiness_timeout_is_fatal(monkeypatch):
    timestamps = iter((0.0, 31.0))
    monkeypatch.setattr(
        compiled_plate_execution_module.time,
        "time",
        lambda: next(timestamps, 31.0),
    )
    events = []

    with pytest.raises(TimeoutError, match=r"Not ready: \[5563\]"):
        wait_until_visualizers_ready(
            orchestrator=SimpleNamespace(
                _execution_cancellation=ExecutionCancellationAuthority()
            ),
            visualizers=[SimpleNamespace(port=5563, is_running=False)],
            progress_queue=SimpleNamespace(put=events.append),
            progress_context=PROGRESS_CONTEXT,
        )

    assert events[-1]["status"] == "failed"
