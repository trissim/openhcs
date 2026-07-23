"""Regressions for exact adapter-recorded image main-flow publication."""

from __future__ import annotations

from dataclasses import replace
from typing import cast
from unittest.mock import Mock

import numpy as np

from openhcs.core.aligned_image_payload import ImageOutputBundle
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleExecutor,
)
from openhcs.processing.backends.cellprofiler.color import color_to_gray


def _executor(*outputs: ArtifactSpec) -> CellProfilerModuleExecutor:
    contract = CallableContract.from_callable(color_to_gray)
    contract = replace(
        contract,
        metadata=replace(
            contract.metadata,
            artifact_outputs=outputs,
            runtime_adapter=CellProfilerRuntimeAdapter.runtime_adapter_spec(),
        ),
    )
    return CellProfilerModuleExecutor(
        raw_func=color_to_gray,
        callable_contract=contract,
    )


def _image_output(
    source_name: str,
    output_name: str,
) -> tuple[ArtifactSpec, ArtifactOutputPlan]:
    source = ArtifactSpec.input(source_name, ImageArtifactType)
    output = ArtifactSpec.output(
        output_name,
        ImageArtifactType,
        relations=(SourceStackLineageSourceRelation(source=source.ref()),),
    )
    return output, ArtifactOutputPlan(
        name=output.name,
        path=f"/memory/{output.name}.pkl",
        artifact_type=output.artifact_type,
        relations=output.relations,
    )


def test_independent_exact_invocations_accumulate_named_image_outputs() -> None:
    current: object = np.zeros((3, 2, 2), dtype=np.float32)
    declarations = (
        ("OrigRed", "IllumRed", 1.0),
        ("OrigBlue", "IllumBlue", 2.0),
        ("OrigGreen", "IllumGreen", 3.0),
    )

    for source_name, output_name, value in declarations:
        output, plan = _image_output(source_name, output_name)
        payload = np.full((2, 2), value, dtype=np.float32)
        adapter = Mock(spec=CellProfilerRuntimeAdapter)
        adapter.artifact_output_value.return_value = payload
        current = _executor(output)._published_active_main_flow_output(
            matched_outputs=((plan, output, payload),),
            declared_only_outputs={},
            adapter=cast(CellProfilerRuntimeAdapter, adapter),
            current_image=current,
            invocation_image=payload,
            plane_projection=None,
        )
        adapter.artifact_output_value.assert_called_once_with(plan)

    assert isinstance(current, ImageOutputBundle)
    assert tuple(context.output_key for context in current.slice_contexts) == (
        "IllumRed",
        "IllumBlue",
        "IllumGreen",
    )
    for index, expected in enumerate((1.0, 2.0, 3.0)):
        np.testing.assert_array_equal(
            current.slices[index],
            np.full((2, 2), expected, dtype=np.float32),
        )


def test_true_preserves_input_adapter_contract_keeps_main_flow_unchanged() -> None:
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    executor = _executor(measurements)
    plan = ArtifactOutputPlan(
        name=measurements.name,
        path="/memory/Measurements.pkl",
        artifact_type=measurements.artifact_type,
    )
    current = np.zeros((2, 2), dtype=np.float32)

    assert executor.callable_contract.preserves_input_main_flow()
    result = executor._published_active_main_flow_output(
        matched_outputs=((plan, measurements, object()),),
        declared_only_outputs={},
        adapter=cast(CellProfilerRuntimeAdapter, object()),
        current_image=current,
        invocation_image=current,
        plane_projection=None,
    )

    assert result is current


def test_mixed_outputs_publish_only_the_active_canonical_return_slot() -> None:
    image, image_plan = _image_output("Original", "Gray")
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    labels = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    measurement_plan = ArtifactOutputPlan(
        name=measurements.name,
        path="/memory/Measurements.pkl",
        artifact_type=measurements.artifact_type,
    )
    label_plan = ArtifactOutputPlan(
        name=labels.name,
        path="/memory/Cells.pkl",
        artifact_type=labels.artifact_type,
    )
    image_value = np.full((2, 2), 7, dtype=np.float32)
    adapter = Mock(spec=CellProfilerRuntimeAdapter)
    adapter.artifact_output_value.return_value = image_value
    executor = _executor(image, measurements, labels)

    assert executor.callable_contract.canonical_return_output_specs.names() == ("Gray",)
    result = executor._published_active_main_flow_output(
        matched_outputs=(
            (image_plan, image, image_value),
            (measurement_plan, measurements, object()),
            (label_plan, labels, object()),
        ),
        declared_only_outputs={},
        adapter=cast(CellProfilerRuntimeAdapter, adapter),
        current_image=np.zeros((2, 2), dtype=np.float32),
        invocation_image=image_value,
        plane_projection=None,
    )

    assert isinstance(result, ImageOutputBundle)
    assert tuple(context.output_key for context in result.slice_contexts) == ("Gray",)
    np.testing.assert_array_equal(result.slices[0], image_value)
    adapter.artifact_output_value.assert_called_once_with(image_plan)
