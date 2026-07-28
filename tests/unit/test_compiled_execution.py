from types import SimpleNamespace

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactOutputPlan, MeasurementsArtifactType
from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.core.compiled_execution import (
    CompiledExecutionBundle,
    CompiledGpuRegistryPlan,
    CompiledRuntimeEnvironmentPlan,
    CompiledWorkerStartPlan,
)
from openhcs.core.config import MultiprocessingStartMethod
from openhcs.core.debug import NoOpDebugExecutionPolicy
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeContextObservation,
    RuntimeExecutionObservation,
    RuntimeObservationMode,
)
from openhcs.core.orchestrator import worker_execution
from openhcs.core.orchestrator.worker_lanes import WorkerLaneExecutionContext
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_artifact_values import RuntimeValue


def _runtime_environment() -> CompiledRuntimeEnvironmentPlan:
    return CompiledRuntimeEnvironmentPlan(
        worker_start=CompiledWorkerStartPlan(
            requested=MultiprocessingStartMethod.SPAWN,
            resolved=MultiprocessingStartMethod.SPAWN,
            reason="test",
            gpu_enabled=False,
            server_mode=False,
        ),
        use_threading=True,
        gpu_registry=CompiledGpuRegistryPlan(configured_num_workers=1),
    )


def test_compiled_execution_bundle_owns_transport_context_resolution(monkeypatch):
    runtime_context = object()
    transport_context = object()
    resolver_calls = []

    def resolve_for_transport(contexts):
        resolver_calls.append(contexts)
        return {"A01": transport_context}

    monkeypatch.setattr(
        "openhcs.core.compiled_execution.resolve_lazy_configurations_for_serialization",
        resolve_for_transport,
    )

    bundle = CompiledExecutionBundle.from_runtime_contexts(
        pipeline_definition=(),
        runtime_contexts={"A01": runtime_context},
        worker_assignments={"worker_0": ["A01"]},
        runtime_environment=_runtime_environment(),
    )

    assert resolver_calls == [{"A01": runtime_context}]
    assert bundle.runtime_contexts == {"A01": runtime_context}
    assert bundle.transport_contexts == {"A01": transport_context}
    assert bundle.worker_assignments == {"worker_0": ["A01"]}


def test_compiled_execution_bundle_exports_bundle_only_compilation_result():
    bundle = CompiledExecutionBundle(
        pipeline_definition=(),
        runtime_contexts={},
        transport_contexts={},
        worker_assignments={},
        runtime_environment=_runtime_environment(),
    )

    result = bundle.as_compilation_result()

    assert result["execution_bundle"] is bundle
    assert tuple(result) == ("execution_bundle",)


def test_runtime_execution_observation_merges_into_parent_contexts():
    worker_store = RuntimeValueStore()
    value = RuntimeValue.normalize(
        ArtifactOutputPlan(
            name="measurements",
            path="/memory/measurements.pkl",
            artifact_type=MeasurementsArtifactType,
            group_component=AllComponents.CHANNEL,
            group_keys=("DAPI",),
        ),
        MeasurementTable(
            name="measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                ({"object_id": 1},),
                fields=(FieldSpec("object_id", int),),
            ),
            subject=MeasurementSubject(
                MeasurementScope.ARTIFACT,
                "measurements",
            ),
        ),
        axis_id="A01",
    )
    record = worker_store.record(
        value,
        path="/memory/measurements.pkl",
        backend="memory",
    )
    parent_context = SimpleNamespace(runtime_value_store=RuntimeValueStore())

    RuntimeExecutionObservation(
        contexts=(
            RuntimeContextObservation(
                context_key="A01",
                records=worker_store.observed_values,
            ),
        ),
    ).merge_into({"A01": parent_context})

    assert parent_context.runtime_value_store.observed_values == (record,)


def test_worker_runtime_observation_excludes_inherited_store_history(monkeypatch):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_component=AllComponents.CHANNEL,
        group_keys=("DAPI",),
    )

    def measurement_value(object_id: int) -> RuntimeValue:
        return RuntimeValue.normalize(
            output_plan,
            MeasurementTable(
                name="measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"object_id": object_id},),
                    fields=(FieldSpec("object_id", int),),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                    "measurements",
                ),
            ),
            axis_id="A01",
        )

    runtime_store = RuntimeValueStore()
    runtime_store.record(
        measurement_value(1),
        path=output_plan.path,
        backend="memory",
    )
    current_records = []

    def execute_current_run(_pipeline, context, _lane_context):
        current_records.append(
            context.runtime_value_store.replace(
                measurement_value(2),
                path=output_plan.path,
                backend="memory",
            )
        )
        return ExecutionResult.success("A01")

    monkeypatch.setattr(
        worker_execution,
        "_execute_single_axis_static",
        execute_current_run,
    )
    monkeypatch.setattr(worker_execution, "emit", lambda **_kwargs: None)
    context = SimpleNamespace(
        axis_id="A01",
        runtime_value_store=runtime_store,
        step_plans={
            0: SimpleNamespace(execution_scope=FunctionStepExecutionScope.AXIS)
        },
    )

    result = worker_execution._execute_axis_with_sequential_combinations(
        pipeline_definition=[object()],
        axis_contexts=[("A01", context)],
        lane_context=WorkerLaneExecutionContext(
            execution_id="execution-1",
            plate_id="plate-1",
            debug_execution_policy=NoOpDebugExecutionPolicy(),
            worker_slot="worker-0",
            owned_wells=("A01",),
        ),
        runtime_observation_mode=RuntimeObservationMode.MERGE_INTO_PARENT,
    )

    assert result.runtime_observation.contexts == (
        RuntimeContextObservation(
            context_key="A01",
            records=tuple(current_records),
        ),
    )
