from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager

from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract, FunctionStepExecutionScope
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.constants.constants import VariableComponents
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.config import NapariStreamingConfig
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    CompiledFunctionPattern,
    FunctionInvocationKey,
)
from openhcs.core.orchestrator.execution_result import ExecutionResult
from openhcs.core.pipeline.function_contracts import execution_scope
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_artifact_values import (
    ArtifactKey,
    RuntimeValue,
)
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionExpectation,
    RuntimeArtifactExecutionObservation,
    _runtime_artifact_viewer_output_payloads,
    runtime_artifact_execution_failures,
)
from openhcs.core.runtime_exports import RuntimeExportExpectation
from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser
from openhcs.processing.materialization import (
    CsvOptions,
    ImageFileOptions,
    MaterializationSpec,
    Output,
    roi_zip,
)
from openhcs.runtime.zmq_execution_observation import (
    ZMQRuntimeExecutionObservationExport,
)


def _compiled_pattern(
    scope: FunctionStepExecutionScope = FunctionStepExecutionScope.AXIS,
) -> CompiledFunctionPattern:
    @execution_scope(scope)
    def callable_(image):
        return image

    return CompiledFunctionPattern(
        groups=(
            CompiledFunctionGroup(
                group_key="default",
                invocations=(
                    CompiledFunctionInvocation(
                        key=FunctionInvocationKey(
                            function_name=callable_.__name__,
                            group_key="default",
                            position=0,
                        ),
                        contract=CallableContract.from_callable(callable_),
                    ),
                ),
            ),
        ),
        is_grouped=False,
    )


def _streaming_context(
    step_plans: dict[int, CompiledStepPlan],
) -> ProcessingContext:
    ensure_storage_registry()
    context = ProcessingContext(
        step_plans=step_plans,
        axis_id="A01",
        filemanager=FileManager(dict(storage_registry)),
    )
    context.microscope_handler = SimpleNamespace(
        parser=SourceSchemaFilenameParser(),
        microscope_type="runtime_validation",
    )
    context.plate_path = Path("/tmp/plate")
    return context


def test_runtime_execution_validation_detects_missing_artifact_kind() -> None:
    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": ProcessingContext(axis_id="A01")}
    )

    failures = runtime_artifact_execution_failures(
        RuntimeArtifactExecutionExpectation(
            artifact_kinds=frozenset((MeasurementsArtifactType,)),
            exports=RuntimeExportExpectation.from_output_specs(()),
        ),
        observation,
    )

    assert failures == (
        "axis 'A01' produced no runtime records for declared artifact kind "
        "'measurements'",
    )


def test_runtime_execution_observation_reads_context_stores() -> None:
    context = ProcessingContext(axis_id="A01")
    context.runtime_value_store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Measurements",
                artifact_type=MeasurementsArtifactType,
                scope=RuntimeExecutionAxisScope(axis_id="A01"),
            ),
            data=MeasurementTable(
                name="Measurements",
                rows=MeasurementSparseColumnarRows.from_rows((), fields=()),
                subject=MeasurementSubject(MeasurementScope.ARTIFACT, "Measurements"),
            ),
        ),
        path="/memory/Measurements.pkl",
        backend="memory",
    )

    observation = RuntimeArtifactExecutionObservation.from_contexts(
        {"A01": context}
    )

    assert observation.record_counts_by_axis["A01"][MeasurementsArtifactType] == 1


def test_runtime_execution_observation_ignores_uncontracted_files(
    tmp_path,
) -> None:
    actual_output_root = tmp_path / "actual_output"
    image_dir = actual_output_root / "images"
    image_dir.mkdir(parents=True)
    table_output = actual_output_root / "A01_Measurements_step0.csv"
    image_output = image_dir / "A01_s001_w1_z001_t001.tif"
    table_output.write_text("ObjectNumber,Area\n1,42\n", encoding="utf-8")
    image_output.write_bytes(b"not decoded by export observation")

    context = ProcessingContext(axis_id="A01")

    observation = RuntimeArtifactExecutionObservation.from_contexts({"A01": context})

    assert observation.exports.table_outputs == ()
    assert observation.exports.image_outputs == ()


def test_zmq_observation_exports_exact_compiler_owned_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    materialization = MaterializationSpec(CsvOptions(filename_suffix=".csv"))
    contracted_output = tmp_path / "results" / "Measurements.csv"
    contracted_output.parent.mkdir()
    contracted_output.write_text("ObjectNumber,Area\n1,42\n", encoding="utf-8")
    unrelated_output = tmp_path / "unrelated.csv"
    unrelated_output.write_text("stale\n1\n", encoding="utf-8")
    output = ArtifactOutputPlan(
        name="Measurements",
        path=str(contracted_output),
        artifact_type=MeasurementsArtifactType,
        materialization=materialization,
    )
    context = ProcessingContext(
        step_plans={
            0: CompiledStepPlan(
                step_index=0,
                step_name="Measure",
                step_type="FunctionStep",
                axis_id="A01",
                artifact_outputs=OrderedDict(((output.ref(), output),)),
                compiled_function_pattern=_compiled_pattern(),
            )
        },
        axis_id="A01",
    )
    monkeypatch.setattr(
        "openhcs.core.steps.function_artifact_materialization."
        "materialized_artifact_output_paths",
        lambda _plan, _context: (contracted_output,),
    )

    export = ZMQRuntimeExecutionObservationExport.from_execution(
        compiled_contexts={"A01": context},
        execution_results={"A01": ExecutionResult.success("A01")},
        output_roots=(tmp_path,),
    )

    assert export.expectation.artifact_kinds == frozenset((MeasurementsArtifactType,))
    assert export.expectation.exports.output_specs[0].name == "Measurements"
    assert export.expectation.exports.output_specs[0].materialization == materialization
    assert export.expectation.artifact_viewer == ()
    assert export.exports.output_files == (contracted_output,)
    assert unrelated_output not in export.exports.output_files


def test_compiled_artifact_viewer_expectations_preserve_full_producers() -> None:
    materialization = MaterializationSpec(
        ImageFileOptions(filename_suffix=".tif")
    )
    first_output = ArtifactOutputPlan(
        name="Repeated",
        path="/memory/first/Repeated.pkl",
        artifact_type=ImageArtifactType,
        materialization=materialization,
        producer_step_index=0,
        producer_step_scope_id="scope-first",
        producer_step_name="First",
    )
    second_output = ArtifactOutputPlan(
        name="Repeated",
        path="/memory/second/Repeated.pkl",
        artifact_type=ImageArtifactType,
        materialization=materialization,
        producer_step_index=1,
        producer_step_scope_id="scope-second",
        producer_step_name="Second",
    )
    streaming_config = NapariStreamingConfig(enabled=True, persistent=False)
    context = _streaming_context(
        {
            0: CompiledStepPlan(
                step_index=0,
                step_name="First",
                step_type="FunctionStep",
                axis_id="A01",
                step_scope_id="scope-first",
                    pipeline_position=1,
                    output_dir=Path("/tmp/output/first"),
                    analysis_results_dir=Path("/tmp/output/first/results"),
                artifact_outputs=OrderedDict(((first_output.ref(), first_output),)),
                compiled_function_pattern=_compiled_pattern(),
                streaming_configs={"napari_stream": streaming_config},
            ),
            1: CompiledStepPlan(
                step_index=1,
                step_name="Second",
                step_type="FunctionStep",
                axis_id="A01",
                step_scope_id="scope-second",
                    pipeline_position=2,
                    output_dir=Path("/tmp/output/second"),
                    analysis_results_dir=Path("/tmp/output/second/results"),
                artifact_outputs=OrderedDict(((second_output.ref(), second_output),)),
                compiled_function_pattern=_compiled_pattern(),
                streaming_configs={"napari_stream": streaming_config},
            ),
        }
    )
    first_metadata = {
        "site": "1",
        "channel": "1",
        "z_index": "1",
        "timepoint": "1",
        "well": "A01",
    }
    second_metadata = {**first_metadata, "site": "2", "channel": "2"}
    first_value = RuntimeValue.normalize(
        first_output,
        ImagePayloadMetadata(
            source_component_metadata=first_metadata,
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(0, 0),
                source_shape_yx=(2, 2),
            ),
        ).payload_with(np.zeros((2, 2), dtype=np.uint8)),
        axis_id="A01",
    )
    second_value = RuntimeValue.normalize(
        second_output,
        ImagePayloadMetadata(
            source_component_metadata=second_metadata,
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(1, 1),
                source_shape_yx=(4, 4),
            ),
        ).payload_with(np.ones((2, 2), dtype=np.uint8)),
        axis_id="A01",
    )
    context.runtime_value_store.record(
        first_value,
        path=first_output.path,
        backend="memory",
    )
    context.runtime_value_store.replace(
        second_value,
        path=second_output.path,
        backend="memory",
    )

    expectation = RuntimeArtifactExecutionExpectation.from_compiled_contexts(
        {"A01": context}
    )

    assert tuple(
        item.producer_identity.output_key for item in expectation.artifact_viewer
    ) == ("Repeated", "Repeated")
    assert tuple(
        item.producer_identity.step_scope_id for item in expectation.artifact_viewer
    ) == ("scope-first", "scope-second")
    assert tuple(
        tuple(payload.components for payload in item.payloads)
        for item in expectation.artifact_viewer
    ) == (
        (
            (
                ("site", "1"),
                ("channel", "1"),
                ("z_index", "1"),
                ("timepoint", "1"),
                ("well", "A01"),
            ),
        ),
        (
            (
                ("site", "2"),
                ("channel", "2"),
                ("z_index", "1"),
                ("timepoint", "1"),
                ("well", "A01"),
            ),
        ),
    )
    assert tuple(
        payload.source_spatial_domain
        for item in expectation.artifact_viewer
        for payload in item.payloads
    ) == (
        SourceSpatialDomain(origin_yx=(0, 0), source_shape_yx=(2, 2)),
        SourceSpatialDomain(origin_yx=(1, 1), source_shape_yx=(4, 4)),
    )


def test_artifact_viewer_expectation_preserves_exact_output_plane_components() -> None:
    metadata = ImagePayloadMetadata(
        source_component_metadata={
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
            "well": "A01",
        },
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(0, 0),
            source_shape_yx=(100, 100),
        ),
    )
    output = Output(
        path="/tmp/A01_w1_labels.roi.zip",
        content=(),
        metadata=metadata,
    ).with_variable_components((VariableComponents.SITE,))

    payloads = _runtime_artifact_viewer_output_payloads((output,))

    assert tuple(payload.components for payload in payloads) == (
        (
            ("site", "1"),
            ("channel", "1"),
            ("z_index", "1"),
            ("timepoint", "1"),
            ("well", "A01"),
        ),
    )
    assert payloads[0].source_spatial_domain == metadata.source_spatial_domain


def test_empty_roi_materialization_does_not_invent_viewer_layer() -> None:
    output = ArtifactOutputPlan(
        name="EmptyLabels",
        path="/memory/EmptyLabels.pkl",
        artifact_type=ObjectLabelsArtifactType,
        materialization=roi_zip(),
        producer_step_index=0,
        producer_step_scope_id="scope-empty-labels",
        producer_step_name="Segment",
    )
    streaming_config = NapariStreamingConfig(enabled=True, persistent=False)
    plan = CompiledStepPlan(
        step_index=0,
        step_name="Segment",
        step_type="FunctionStep",
        axis_id="A01",
        step_scope_id="scope-empty-labels",
        pipeline_position=0,
        output_dir=Path("/tmp/output/empty-labels"),
        analysis_results_dir=Path("/tmp/output/empty-labels-results"),
        variable_components=(),
        artifact_outputs=OrderedDict(((output.ref(), output),)),
        compiled_function_pattern=_compiled_pattern(),
        streaming_configs={"napari_stream": streaming_config},
    )
    context = _streaming_context({0: plan})
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((8, 8), dtype=np.int32)
        ),
        source_component_metadata={
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
            "well": "A01",
        },
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(8, 8)),
    )
    context.runtime_value_store.record(
        RuntimeValue.normalize(output, labels, axis_id="A01"),
        path=output.path,
        backend="memory",
    )

    expectation = RuntimeArtifactExecutionExpectation.from_compiled_contexts(
        {"A01": context}
    )

    assert expectation.artifact_viewer == ()


def test_runtime_execution_observation_reads_plate_export_from_exact_owner(
    tmp_path,
    monkeypatch,
) -> None:
    materialization = MaterializationSpec(CsvOptions(filename_suffix=".csv"))
    contracted_output = tmp_path / "results" / "PlateMeasurements.csv"
    contracted_output.parent.mkdir()
    contracted_output.write_text("ObjectNumber,Area\n1,42\n", encoding="utf-8")
    output = ArtifactOutputPlan(
        name="PlateMeasurements",
        path=str(contracted_output),
        artifact_type=MeasurementsArtifactType,
        materialization=materialization,
    )
    pattern = _compiled_pattern(FunctionStepExecutionScope.PLATE)
    contexts = {
        axis_id: ProcessingContext(
            step_plans={
                0: CompiledStepPlan(
                    step_index=0,
                    step_name="Export",
                    step_type="FunctionStep",
                    axis_id=axis_id,
                    artifact_outputs=OrderedDict(((output.ref(), output),)),
                    compiled_function_pattern=pattern,
                    create_openhcs_metadata=owns_output,
                )
            },
            axis_id=axis_id,
        )
        for axis_id, owns_output in (("A01", True), ("A02", False))
    }
    observed_axes: list[str] = []

    def output_paths(plan, _context):
        observed_axes.append(plan.axis_id)
        return (contracted_output,)

    monkeypatch.setattr(
        "openhcs.core.steps.function_artifact_materialization."
        "materialized_artifact_output_paths",
        output_paths,
    )

    observation = RuntimeArtifactExecutionObservation.from_contexts(contexts)

    assert observed_axes == ["A01"]
    assert observation.exports.table_outputs == (contracted_output,)


def test_zmq_observation_compresses_and_preserves_exact_runtime_records(
    tmp_path,
) -> None:
    context = ProcessingContext(axis_id="A01")
    context.runtime_value_store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Measurements",
                artifact_type=MeasurementsArtifactType,
                scope=RuntimeExecutionAxisScope(axis_id="A01"),
            ),
            data=MeasurementTable(
                name="Measurements",
                rows=MeasurementSparseColumnarRows.from_rows(
                    ({"Area": 42.0},),
                    fields=(FieldSpec("Area", float),),
                ),
                subject=MeasurementSubject(
                    MeasurementScope.ARTIFACT,
                    "Measurements",
                ),
            ),
        ),
        path="/memory/Measurements.pkl",
        backend="memory",
    )
    export = ZMQRuntimeExecutionObservationExport.from_execution(
        compiled_contexts={"A01": context},
        execution_results={"A01": ExecutionResult.success("A01")},
        output_roots=(tmp_path,),
    )
    path = tmp_path / "observation.pkl"

    export.write(path)
    restored = ZMQRuntimeExecutionObservationExport.read(path)

    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert restored.expectation == export.expectation
    assert restored.records_by_axis == export.records_by_axis
