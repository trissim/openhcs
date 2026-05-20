import pytest
from types import SimpleNamespace

from openhcs.core.config import MultiprocessingStartMethod
from openhcs.core.debug import NoOpDebugExecutionPolicy
from openhcs.core.orchestrator import orchestrator as orchestrator_module
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeObservationMode,
)
from openhcs.core.orchestrator.orchestrator import (
    AnalysisConsolidationPlan,
    CompiledContextLanePlanner,
    ExecutionStateProjector,
    ExecutionVisualizerCleanup,
    ExecutorShutdownPlan,
    PooledWorkerLaneRunner,
    WorkerLaneExecutionIdentity,
    WorkerLaneExecutionPlan,
    WorkerExecutorFactory,
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
        orchestrator_module.multiprocessing,
        "get_context",
        lambda method: context,
    )

    resources = WorkerExecutorFactory(
        log_file_base=None,
        progress_queue="queue",
        progress_context={"execution_id": "exec", "plate_id": "plate"},
    ).create(
        effective_config=SimpleNamespace(
            use_threading=True,
            multiprocessing_start_method=MultiprocessingStartMethod.SPAWN,
        ),
        actual_max_workers=1,
    )

    assert resources.executor is None
    assert resources.multiprocessing_context is context
    assert resources.inline_worker_lane_execution is True
    assert resources.fork_inherited_execution is False
    assert resources.use_multiprocessing is False


def test_executor_factory_creates_thread_pool_for_multi_worker_threading(monkeypatch):
    created = {}
    context = object()

    class FakeThreadPoolExecutor:
        def __init__(self, *, max_workers):
            created["max_workers"] = max_workers

    monkeypatch.setattr(
        orchestrator_module.multiprocessing,
        "get_context",
        lambda method: context,
    )
    monkeypatch.setattr(
        orchestrator_module.concurrent.futures,
        "ThreadPoolExecutor",
        FakeThreadPoolExecutor,
    )

    resources = WorkerExecutorFactory(
        log_file_base=None,
        progress_queue="queue",
        progress_context={"execution_id": "exec", "plate_id": "plate"},
    ).create(
        effective_config=SimpleNamespace(
            use_threading=True,
            multiprocessing_start_method=MultiprocessingStartMethod.SPAWN,
        ),
        actual_max_workers=3,
    )

    assert isinstance(resources.executor, FakeThreadPoolExecutor)
    assert created == {"max_workers": 3}
    assert resources.inline_worker_lane_execution is False
    assert resources.fork_inherited_execution is False
    assert resources.use_multiprocessing is False


def test_executor_factory_uses_fork_inherited_lane_without_pool(monkeypatch):
    context = object()
    monkeypatch.setattr(
        orchestrator_module.multiprocessing,
        "get_context",
        lambda method: context,
    )

    resources = WorkerExecutorFactory(
        log_file_base="/tmp/worker",
        progress_queue="queue",
        progress_context={"execution_id": "exec", "plate_id": "plate"},
    ).create(
        effective_config=SimpleNamespace(
            use_threading=False,
            multiprocessing_start_method=MultiprocessingStartMethod.FORK,
        ),
        actual_max_workers=2,
    )

    assert resources.executor is None
    assert resources.multiprocessing_context is context
    assert resources.inline_worker_lane_execution is False
    assert resources.fork_inherited_execution is True
    assert resources.use_multiprocessing is True


def test_executor_factory_creates_process_pool_with_worker_initializer(monkeypatch):
    created = {}
    context = object()

    class FakeProcessPoolExecutor:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(
        orchestrator_module.multiprocessing,
        "get_context",
        lambda method: context,
    )
    monkeypatch.setattr(
        orchestrator_module.concurrent.futures,
        "ProcessPoolExecutor",
        FakeProcessPoolExecutor,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "get_current_global_config",
        lambda config_type: SimpleNamespace(alpha=1),
    )

    resources = WorkerExecutorFactory(
        log_file_base="/tmp/worker-log",
        progress_queue="queue",
        progress_context={"execution_id": "exec", "plate_id": "plate"},
    ).create(
        effective_config=SimpleNamespace(
            use_threading=False,
            multiprocessing_start_method=MultiprocessingStartMethod.SPAWN,
        ),
        actual_max_workers=4,
    )

    assert isinstance(resources.executor, FakeProcessPoolExecutor)
    assert created["max_workers"] == 4
    assert created["mp_context"] is context
    assert created["initializer"] is orchestrator_module._configure_worker_with_gpu
    assert created["initargs"] == (
        "/tmp/worker-log",
        {"alpha": 1},
        "queue",
        {"execution_id": "exec", "plate_id": "plate"},
    )
    assert resources.inline_worker_lane_execution is False
    assert resources.fork_inherited_execution is False
    assert resources.use_multiprocessing is True


def test_pooled_worker_lane_runner_submits_and_collects_lane_results(monkeypatch):
    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def submit(self, fn):
            return FakeFuture(fn())

    class FakeWorkerLaneExecutor:
        def __init__(self, *, lane_context, **kwargs):
            self._axis_id = lane_context.owned_wells[0]

        def execute(self):
            return {self._axis_id: ExecutionResult.success(self._axis_id)}

    monkeypatch.setattr(
        orchestrator_module.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "WorkerLaneExecutor",
        FakeWorkerLaneExecutor,
    )

    execution_plan = WorkerLaneExecutionPlan(
        identity=WorkerLaneExecutionIdentity(
            execution_id="exec",
            plate_id="plate",
            debug_execution_policy=NoOpDebugExecutionPolicy(),
        ),
        lane_axis_contexts={"worker_0": [("A01", [("A01", object())])]},
        worker_assignments={"worker_0": ["A01"]},
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
        def result(self):
            return {"A01": ExecutionResult.success("A01")}

    class FakeExecutor:
        def submit(self, fn):
            return FakeFuture()

    class FakeWorkerLaneExecutor:
        def __init__(self, *, pipeline_definition, **kwargs):
            submitted_pipeline.extend(pipeline_definition)

        def execute(self):
            return {}

    monkeypatch.setattr(
        orchestrator_module.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "WorkerLaneExecutor",
        FakeWorkerLaneExecutor,
    )

    execution_plan = WorkerLaneExecutionPlan(
        identity=WorkerLaneExecutionIdentity(
            execution_id="exec",
            plate_id="plate",
            debug_execution_policy=NoOpDebugExecutionPolicy(),
        ),
        lane_axis_contexts={"worker_0": [("A01", [("A01", object())])]},
        worker_assignments={"worker_0": ["A01"]},
        runtime_observation_mode=RuntimeObservationMode.OMIT,
    )

    PooledWorkerLaneRunner(FakeExecutor()).run(
        pipeline_definition=[stripped_step],
        execution_plan=execution_plan,
        parent_contexts={},
    )

    assert submitted_pipeline == [stripped_step]


def test_pooled_worker_lane_runner_emits_error_before_reraising(monkeypatch):
    emitted = []

    class FakeFuture:
        def result(self):
            raise RuntimeError("lane exploded")

    class FakeExecutor:
        def submit(self, fn):
            return FakeFuture()

    class FakeWorkerLaneExecutor:
        def __init__(self, **kwargs):
            pass

        def execute(self):
            return {}

    monkeypatch.setattr(
        orchestrator_module.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "WorkerLaneExecutor",
        FakeWorkerLaneExecutor,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "emit",
        lambda **kwargs: emitted.append(kwargs),
    )

    execution_plan = WorkerLaneExecutionPlan(
        identity=WorkerLaneExecutionIdentity(
            execution_id="exec",
            plate_id="plate",
            debug_execution_policy=NoOpDebugExecutionPolicy(),
        ),
        lane_axis_contexts={"worker_0": [("A01", [("A01", object())])]},
        worker_assignments={"worker_0": ["A01"]},
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
            raise orchestrator_module.concurrent.futures.process.BrokenProcessPool(
                "dead"
            )

    ExecutorShutdownPlan(BrokenExecutor()).run()

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
    projector = ExecutionStateProjector(orchestrator)

    projector.project({"A01": ExecutionResult.success("A01")})
    assert orchestrator._state is orchestrator_module.OrchestratorState.COMPLETED

    projector.project({"A01": ExecutionResult.error("A01")})
    assert orchestrator._state is orchestrator_module.OrchestratorState.EXEC_FAILED


def test_execution_visualizer_cleanup_stops_only_non_persistent_visualizers():
    stopped = []
    persistent = SimpleNamespace(persistent=True, stop_viewer=lambda: stopped.append("p"))
    transient = SimpleNamespace(
        persistent=False,
        stop_viewer=lambda: stopped.append("t"),
    )

    ExecutionVisualizerCleanup().run([persistent, transient])

    assert stopped == ["t"]
