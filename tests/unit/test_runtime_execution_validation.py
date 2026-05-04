from __future__ import annotations

from types import SimpleNamespace

from openhcs.core.artifacts import ArtifactKey, ArtifactKind, ArtifactScope
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionExpectation,
    RuntimeArtifactExecutionObservation,
    runtime_artifact_execution_failures,
)
from openhcs.core.runtime_exports import RuntimeExportExpectation
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import RuntimeValue, RuntimeValueSchema


def test_runtime_execution_validation_detects_missing_artifact_kind() -> None:
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=RuntimeValueStore())},
        output_root="/tmp/unused",
    )

    failures = runtime_artifact_execution_failures(
        RuntimeArtifactExecutionExpectation(
            artifact_kinds=frozenset((ArtifactKind.MEASUREMENTS,)),
            exports=RuntimeExportExpectation.from_flags(
                table_exports=False,
                image_exports=False,
            ),
        ),
        observation,
    )

    assert failures == (
        "axis 'A01' produced no runtime records for declared artifact kind "
        "'measurements'",
    )


def test_runtime_execution_observation_reads_context_stores() -> None:
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Measurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=(),
            schema=RuntimeValueSchema(kind=ArtifactKind.MEASUREMENTS),
        ),
        path="/memory/Measurements.pkl",
        backend="memory",
    )

    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": SimpleNamespace(runtime_value_store=store)},
        output_root="/tmp/unused",
    )

    assert observation.record_counts_by_axis["A01"][ArtifactKind.MEASUREMENTS] == 1


def test_runtime_execution_observation_uses_compiled_output_roots(tmp_path) -> None:
    actual_output_root = tmp_path / "actual_output"
    stale_output_root = tmp_path / "stale_output"
    image_dir = actual_output_root / "images"
    image_dir.mkdir(parents=True)
    table_output = actual_output_root / "A01_Measurements_step0.csv"
    image_output = image_dir / "A01_s001_w1_z001_t001.tif"
    table_output.write_text("ObjectNumber,Area\n1,42\n", encoding="utf-8")
    image_output.write_bytes(b"not decoded by export observation")
    stale_output_root.mkdir()

    store = RuntimeValueStore()
    context = SimpleNamespace(
        runtime_value_store=store,
        step_plans={
            0: SimpleNamespace(
                output_plate_root=str(actual_output_root),
                materialized_output=None,
            )
        },
    )

    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": context},
        output_root=stale_output_root,
    )

    assert observation.exports.table_outputs == (table_output,)
    assert observation.exports.image_outputs == (image_output,)
