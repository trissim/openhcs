import pytest
from types import SimpleNamespace

from openhcs.core.config import MultiprocessingStartMethod
from openhcs.core.debug import NoOpDebugExecutionPolicy
from openhcs.core.progress import ProgressExecutionContext
from openhcs.core.orchestrator import orchestrator as orchestrator_module
from openhcs.core.orchestrator import worker_execution as worker_execution_module
from openhcs.core.orchestrator.analysis_consolidation import AnalysisConsolidationPlan
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeObservationMode,
)
from openhcs.core.orchestrator.compiled_plate_execution import (
    CompiledPlateExecutionRequest,
    project_execution_state,
    stop_execution_visualizers,
    validate_compiled_plate_execution,
)
from openhcs.core.orchestrator.worker_execution import (
    ForkInheritedWorkerLaneRunner,
    ForkInheritedWorkerExecutorResources,
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
from openhcs.core.compiled_execution import (
    CompiledExecutionBundle,
    CompiledGpuRegistryPlan,
    CompiledRuntimeEnvironmentPlan,
    CompiledWorkerStartPlan,
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
        gpu_registry=CompiledGpuRegistryPlan(
            configured_num_workers=configured_num_workers
        ),
    )


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


def test_compiled_plate_execution_request_uses_bundle_as_runtime_authority():
    runtime_environment = _runtime_environment(
        use_threading=True,
        start_method=MultiprocessingStartMethod.SPAWN,
        configured_num_workers=7,
    )
    context = SimpleNamespace()
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
    ).create(
        runtime_environment=runtime_environment,
        actual_max_workers=4,
    )

    assert isinstance(resources.executor, FakeProcessPoolExecutor)
    assert created["max_workers"] == 4
    assert created["mp_context"] is context
    assert created["initializer"] is worker_execution_module._configure_worker_with_gpu
    assert created["initargs"] == (
        "/tmp/worker-log",
        runtime_environment.gpu_registry,
        "queue",
        PROGRESS_CONTEXT,
    )
    assert isinstance(resources, PooledWorkerExecutorResources)
    assert resources.uses_fork_inherited_contexts is False
    assert resources.use_multiprocessing is True


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
    ):
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

    results = PooledWorkerLaneRunner(FakeExecutor()).run(
        pipeline_definition=[],
        execution_plan=execution_plan,
        parent_contexts={},
    )

    assert results == {"A01": ExecutionResult.success("A01")}


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
    ):
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

    PooledWorkerLaneRunner(FakeExecutor()).run(
        pipeline_definition=[stripped_step],
        execution_plan=execution_plan,
        parent_contexts={},
    )

    assert submitted_pipeline == [stripped_step]


def test_fork_inherited_runner_executes_single_active_lane_inline(monkeypatch):
    executed = []

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
        return {"A01": ExecutionResult.success("A01")}

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
            runtime_observation_mode=RuntimeObservationMode.OMIT,
        )

        results = ForkInheritedWorkerLaneRunner(
            ForbiddenMultiprocessingContext()
        ).run(execution_plan)
    finally:
        ForkInheritedWorkerExecutionState.clear()

    assert results == {"A01": ExecutionResult.success("A01")}
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
        PooledWorkerLaneRunner(FakeExecutor()).run(
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
    ).shutdown_executor()

    assert "broken process pool" in caplog.text


def test_analysis_consolidation_plan_skips_disabled_config():
    context = SimpleNamespace(
        analysis_consolidation_config=SimpleNamespace(enabled=False),
        step_plans={},
    )

    AnalysisConsolidationPlan(microscope_handler=SimpleNamespace(parser=object())).run(
        {"A01": context}
    )


def test_execution_state_projector_maps_success_and_failure():
    orchestrator = SimpleNamespace(_state=None)

    project_execution_state(orchestrator, {"A01": ExecutionResult.success("A01")})
    assert orchestrator._state is orchestrator_module.OrchestratorState.COMPLETED

    project_execution_state(orchestrator, {"A01": ExecutionResult.error("A01")})
    assert orchestrator._state is orchestrator_module.OrchestratorState.EXEC_FAILED


def test_execution_visualizer_cleanup_stops_only_non_persistent_visualizers():
    stopped = []
    persistent = SimpleNamespace(persistent=True, stop_viewer=lambda: stopped.append("p"))
    transient = SimpleNamespace(
        persistent=False,
        stop_viewer=lambda: stopped.append("t"),
    )

    stop_execution_visualizers([persistent, transient])

    assert stopped == ["t"]
