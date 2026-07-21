"""Measurement-source scope regressions for exact output recording plans."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecRef,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
)
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_image_values import image_payload_metadata
from openhcs.core.runtime_object_labels import ObjectLabelSet, ObjectLabelVariantData
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerImageRequest
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from tests.unit.cellprofiler_runtime_test_support import (
    runtime_adapter_request_for_test,
)


class _PayloadBackedOutputRecordRequest(CellProfilerOutputRecordRequest):
    artifact_values: ClassVar[dict[ArtifactSpecRef, ObjectLabelSet]] = {}

    def artifact_value(self, spec: ArtifactSpec) -> ObjectLabelSet:
        return self.artifact_values[spec.ref()]


def _source_labels(name: str, channel: str) -> ObjectLabelSet:
    return ObjectLabelSet(
        name=name,
        source_image_name=name,
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((1, 4, 4), dtype=np.int32),
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(f"/source/{name}.tif",),
            component_metadata=({"site": "1", "channel": channel},),
        ),
    )


def test_measurement_source_alignment_uses_selected_output_group_scope() -> None:
    first = ArtifactSpec.input("First", ObjectLabelsArtifactType)
    second = ArtifactSpec.input("Second", ObjectLabelsArtifactType)
    measurements = ArtifactSpec.output(
        "CombinedMeasurements",
        MeasurementsArtifactType,
    )

    def combine_objects(object_labels):
        return object_labels

    raw_contract = CallableContract.from_callable(combine_objects)
    contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_inputs=(first, second),
            artifact_outputs=(measurements,),
        ),
    )
    invocation_key = FunctionInvocationKey(
        contract.function_name,
        DEFAULT_GROUP_KEY,
        0,
    )
    edges = tuple(
        InvocationArtifactInputEdgePlan(
            key=InvocationArtifactInputProjectionKey(invocation_key, input_index),
            spec=spec,
            storage_plan=None,
            projection=None,
        )
        for input_index, spec in enumerate((first, second))
    )
    output_plan = ArtifactOutputPlan(
        name=measurements.name,
        path="/artifacts/CombinedMeasurements",
        artifact_type=measurements.artifact_type,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
    )
    adapter = CellProfilerRuntimeAdapter(
        runtime_adapter_request_for_test(
            runtime_value_store=RuntimeValueStore(),
            callable_contract=contract,
            artifact_inputs={edge.key: edge for edge in edges},
            artifact_outputs={output_plan.ref(): output_plan},
            group_key="1",
            axis_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=AllComponents.CHANNEL,
                value="1",
            ),
        )
    )
    first_value = _source_labels(first.name, "1")
    second_value = _source_labels(second.name, "2")
    _PayloadBackedOutputRecordRequest.artifact_values = {
        first.ref(): first_value,
        second.ref(): second_value,
    }
    request = _PayloadBackedOutputRecordRequest(
        callable_contract=contract,
        active_input_edges=edges,
        adapter=adapter,
        spec=measurements,
        output_plan=output_plan,
        output_value=MeasurementSparseColumnarRows.from_rows((), fields=()),
        source=CellProfilerImageRequest(
            payload=first_value,
            source_image_name=None,
            image_count=1,
        ),
        call_kwargs={},
        current_image=first_value,
    )

    metadata = request.measurement_source_metadata((first, second))

    assert metadata == image_payload_metadata(first_value)
