"""Focused MeasureImageSkeleton artifact and measurement contract tests."""

from __future__ import annotations

import numpy as np

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    ImageMeasurementSubjectRelation,
    MeasurementsArtifactType,
)
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.skeleton import (
    MeasureImageSkeletonModule,
    SkeletonMeasurement,
    measure_image_skeleton,
    measure_image_skeleton_3d,
)


def _compiled_contract():
    image = ArtifactSpec.output("SMI312", ImageArtifactType)
    invocation_key = FunctionInvocationKey(
        "measure_image_skeleton",
        "default",
        0,
    )
    module = ModuleBlock(
        name="MeasureImageSkeleton",
        module_num=1,
        setting_records=[
            ModuleSetting("Select an image to measure", image.name),
        ],
    )
    return MeasureImageSkeletonModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_name="Skeleton metrics",
            step_index=3,
            available_artifacts=ArtifactSpecCollection((image,)),
            main_flow_artifacts=ArtifactSpecCollection(
                (image.for_plan_type(ArtifactInputPlan),)
            ),
            available_artifact_producers=artifact_producers_for_outputs(
                (image,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey("fixture_producer", "default", 0),
                ),
            ),
        ),
    )


def test_measure_image_skeleton_compiles_one_image_measurement_output() -> None:
    contract = _compiled_contract()

    assert contract.artifact_inputs.names() == ("SMI312",)
    (measurements,) = contract.artifact_outputs.specs
    assert measurements.name == "Skeleton metrics_1_measurements"
    assert measurements.artifact_type is MeasurementsArtifactType
    assert ImageMeasurementSubjectRelation(
        source=ArtifactSpec.input("SMI312", ImageArtifactType).ref()
    ) in measurements.relations


def test_measure_image_skeleton_returns_schema_bearing_rows() -> None:
    image = np.asarray(
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        dtype=np.uint8,
    )

    output, rows = measure_image_skeleton(image)

    np.testing.assert_array_equal(output, image)
    assert isinstance(rows, ColumnarRows)
    assert rows.row_type is SkeletonMeasurement
    assert tuple(rows.iter_row_mappings()) == (
        {"slice_index": 0, "branches": 0, "endpoints": 2},
    )


def test_measure_image_skeleton_projects_exact_native_feature_names() -> None:
    _output, rows = measure_image_skeleton(
        np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    )

    projected = MeasureImageSkeletonModule.project_measurement_record_rows(
        rows,
        source_image_name="SMI312",
    )

    assert tuple(projected.columns) == (
        "slice_index",
        "Skeleton_Branches_SMI312",
        "Skeleton_Endpoints_SMI312",
    )
    assert tuple(projected.iter_row_mappings()) == (
        {
            "slice_index": 0,
            "Skeleton_Branches_SMI312": 0,
            "Skeleton_Endpoints_SMI312": 2,
        },
    )


def test_measure_image_skeleton_3d_uses_the_same_owned_measurement_contract() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, 1, 1] = 1

    output, rows = measure_image_skeleton_3d(image)

    np.testing.assert_array_equal(output, image)
    assert isinstance(rows, ColumnarRows)
    assert rows.row_type is SkeletonMeasurement
    assert MeasureImageSkeletonModule.declared_function_names() == (
        "measure_image_skeleton",
        "measure_image_skeleton_3d",
    )
