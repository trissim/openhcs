from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.equivalence.outputs import RuntimeOutputSnapshot
from openhcs.core.equivalence.policy import RuntimeEquivalencePolicy
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_artifact_values import ArtifactKey, RuntimeValue
from openhcs.core.runtime_equivalence import runtime_reference_artifact_equivalence
from openhcs.core.runtime_execution_validation import RuntimeArtifactExecutionObservation
from openhcs.core.runtime_exports import RuntimeExportObservation
from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_DIALECT,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    MeasurementFeatureRecord,
    measurement_table_for_module,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    CellProfilerThresholdResult,
    ObjectThresholdResult,
    ThresholdModule,
)


def _recorded_threshold_table(output_name: str):
    result_rows = CellProfilerThresholdResult(
        final_threshold=0.25,
        original_threshold=0.2,
        weighted_variance=0.1,
        sum_of_entropies=-0.5,
        mask=np.ones((2, 2), dtype=bool),
    ).measurement_rows()
    image_input = ArtifactSpec.output("phase", ImageArtifactType)
    contract = ThresholdModule.callable_contract(
        module=ModuleBlock(
            name=str(ThresholdModule.module_name),
            module_num=1,
            setting_records=[
                ModuleSetting("Select the input image", image_input.name),
                ModuleSetting("Name the output image", output_name),
            ],
        ),
        invocation_key=FunctionInvocationKey(
            str(ThresholdModule.function_name),
            "default",
            0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_name=str(ThresholdModule.module_name),
            step_index=0,
            available_artifacts=ArtifactSpecCollection((image_input,)),
            available_artifact_producers=artifact_producers_for_outputs(
                (image_input,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey("fixture_producer", "default", 0),
                ),
            ),
        ),
    )
    _image_output, measurement_output = contract.artifact_outputs
    image_payload = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1.tif",
    ).payload_with(np.ones((2, 2), dtype=np.float32), None)
    request = SimpleNamespace(
        callable_contract=contract,
        spec=measurement_output,
        output_value=result_rows,
        artifact_output_value=lambda spec: image_payload,
    )
    return measurement_table_for_module(request)


def _declared_threshold_feature_names(output_name: str) -> tuple[str, ...]:
    annotations = get_type_hints(ObjectThresholdResult, include_extras=True)
    return tuple(
        ThresholdModule.measurement_feature_name(field.name, output_name)
        for field in fields(ObjectThresholdResult)
        if MeasurementFeatureRecord.axis_annotation(annotations[field.name]) is None
    )


def test_threshold_recording_derives_all_features_from_nominal_row_schema() -> None:
    output_name = "phaseThresh"
    table = _recorded_threshold_table(output_name)
    feature_names = _declared_threshold_feature_names(output_name)

    assert tuple(field.name for field in table.rows.fields) == (
        MeasurementRowAxisField.SLICE_INDEX.value,
        *feature_names,
    )
    assert tuple(table.rows.row_mappings()[0]) == (
        MeasurementRowAxisField.SLICE_INDEX.value,
        *feature_names,
    )


def test_threshold_recorded_features_survive_semantic_equivalence(
    tmp_path: Path,
) -> None:
    output_name = "phaseThresh"
    table = _recorded_threshold_table(output_name)
    row = table.rows.row_mappings()[0]
    feature_names = _declared_threshold_feature_names(output_name)
    reference_root = tmp_path / "native"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    (reference_root / "Image.csv").write_text(
        ",".join(("ImageNumber", *feature_names))
        + "\n"
        + ",".join(("1", *(str(row[name]) for name in feature_names)))
        + "\n",
        encoding="utf-8",
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name=table.name,
                artifact_type=MeasurementsArtifactType,
                scope=RuntimeExecutionAxisScope(axis_id="A01"),
            ),
            data=table,
        ),
        path=f"/memory/{table.name}.pkl",
        backend="memory",
    )
    observation = RuntimeArtifactExecutionObservation(
        {"A01": store.values()},
        RuntimeExportObservation.from_output_root(candidate_root),
    )

    report = runtime_reference_artifact_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_root),
        observation,
        policy=RuntimeEquivalencePolicy(
            measurement_dialect=CELLPROFILER_MEASUREMENT_DIALECT,
        ),
    )

    assert report.is_equivalent
