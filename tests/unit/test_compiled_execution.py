from types import SimpleNamespace

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.compiled_execution import (
    CompiledExecutionBundle,
    CompiledGpuRegistryPlan,
    CompiledRuntimeEnvironmentPlan,
    CompiledWorkerStartPlan,
)
from openhcs.core.config import MultiprocessingStartMethod
from openhcs.core.orchestrator.execution_result import (
    RuntimeContextObservation,
    RuntimeExecutionObservation,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import normalize_artifact_value


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
    value = normalize_artifact_value(
        ArtifactOutputPlan(
            name="measurements",
            path="/memory/measurements.pkl",
            kind=ArtifactKind.MEASUREMENTS,
            group_keys=("DAPI",),
        ),
        [{"object_id": 1}],
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
