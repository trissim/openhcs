"""Contracts for uncovered CellProfiler image/result producers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    ImageMeasurementSubjectRelation,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.source_bindings import StepSourceBindingsConfig
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    measurement_table_for_module,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    FlipAndRotateModule,
    RotationResult,
    flip_and_rotate,
)
from openhcs.processing.backends.cellprofiler.maxima import (
    ExcludeMode,
    FindMaximaModule,
    MaximaResult,
    find_maxima,
    find_maxima_with_mask,
)


def _compiled_contract(
    module_type,
    function_name: str,
    output_name: str,
):
    image = ArtifactSpec.output("DNA", ImageArtifactType)
    invocation_key = FunctionInvocationKey(function_name, "default", 0)
    module = ModuleBlock(
        name=module_type.module_name,
        module_num=1,
        setting_records=[
            ModuleSetting("Select the input image", image.name),
            ModuleSetting("Name the output image", output_name),
        ],
    )
    return module_type.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_name=module.name,
            step_index=0,
            available_artifacts=ArtifactSpecCollection((image,)),
            available_artifact_producers=artifact_producers_for_outputs(
                (image,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey("fixture_producer", "default", 0),
                ),
            ),
        ),
    )


def _recorded_table(
    *,
    function_name: str,
    output_name: str,
    output_image: object,
    rows: ColumnarRows,
):
    image_spec = ArtifactSpec.output(output_name, ImageArtifactType)
    measurement_spec = ArtifactSpec.output(
        f"{output_name}_measurements",
        MeasurementsArtifactType,
    )
    request = SimpleNamespace(
        callable_contract=SimpleNamespace(
            function_name=function_name,
            artifact_outputs=ArtifactSpecCollection(
                (image_spec, measurement_spec),
            ),
        ),
        spec=measurement_spec,
        output_value=rows,
        artifact_output_value=lambda spec: output_image,
    )
    return measurement_table_for_module(request)


def test_uncovered_image_modules_compile_image_then_measurement_outputs() -> None:
    for module_type, function_name, output_name in (
        (FlipAndRotateModule, "flip_and_rotate", "RotatedDNA"),
        (FindMaximaModule, "find_maxima", "MaximaDNA"),
    ):
        contract = _compiled_contract(module_type, function_name, output_name)

        assert contract.artifact_inputs.names() == ("DNA",)
        assert tuple(spec.name for spec in contract.artifact_outputs) == (
            output_name,
            f"{module_type.module_name}_1_measurements",
        )
        assert tuple(spec.artifact_type for spec in contract.artifact_outputs) == (
            ImageArtifactType,
            MeasurementsArtifactType,
        )
        measurement = contract.artifact_outputs.specs[1]
        assert ImageMeasurementSubjectRelation(
            source=ArtifactSpec.output(output_name, ImageArtifactType).ref()
        ) in measurement.relations


def test_uncovered_image_callables_return_schema_bearing_measurement_rows() -> None:
    image = np.zeros((7, 7), dtype=np.float32)
    image[3, 3] = 1.0

    rotated, rotation_rows = flip_and_rotate(image)
    maxima, maxima_rows = find_maxima(
        image,
        min_distance=1,
        min_intensity=0.5,
    )

    np.testing.assert_array_equal(image_payload_data(rotated), image)
    assert isinstance(rotation_rows, ColumnarRows)
    assert rotation_rows.row_type is RotationResult
    assert tuple(rotation_rows.iter_row_mappings()) == (
        {"slice_index": 0, "rotation_angle": 0.0},
    )
    assert isinstance(maxima_rows, ColumnarRows)
    assert maxima_rows.row_type is MaximaResult
    assert tuple(maxima_rows.iter_row_mappings()) == (
        {
            "slice_index": 0,
            "maxima_count": 1,
            "min_distance_used": 1,
            "threshold_used": 0.5,
        },
    )
    assert int(np.count_nonzero(maxima)) == 1


def test_flip_rotation_uses_exact_native_output_qualified_feature_name() -> None:
    image = ImagePayloadMetadata(
        source_path="/input/DNA.tif"
    ).payload_with(np.ones((4, 4), dtype=np.float32), None)
    rotated, rows = flip_and_rotate(image, rotation_angle=12.5)

    table = _recorded_table(
        function_name="flip_and_rotate",
        output_name="RotatedDNA",
        output_image=rotated,
        rows=rows,
    )

    assert table.source_image_name == "RotatedDNA"
    assert tuple(table.rows.columns) == (
        "slice_index",
        "Rotation_RotatedDNA",
    )
    assert tuple(table.rows.iter_row_mappings()) == (
        {"slice_index": 0, "Rotation_RotatedDNA": 0.0},
    )


def test_find_maxima_retains_diagnostic_schema_without_inventing_cp_features() -> None:
    maxima, rows = find_maxima(
        np.eye(5, dtype=np.float32),
        min_distance=1,
    )
    maxima_payload = ImagePayloadMetadata(
        source_path="/input/DNA.tif"
    ).payload_with(maxima, None)

    table = _recorded_table(
        function_name="find_maxima",
        output_name="MaximaDNA",
        output_image=maxima_payload,
        rows=rows,
    )

    assert table.source_image_name == "MaximaDNA"
    assert tuple(table.rows.columns) == (
        "slice_index",
        "maxima_count",
        "min_distance_used",
        "threshold_used",
    )
    assert not any(column.startswith("FindMaxima_") for column in table.rows.columns)


def test_find_maxima_selects_one_stacked_invocation_for_masked_modes() -> None:
    available = (
        ArtifactSpec.output("DNA", ImageArtifactType),
        ArtifactSpec.output("Mask", ImageArtifactType),
        ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
    )
    context = ArtifactDeclarationStepContext(
        step_name="FindMaxima",
        step_index=0,
        available_artifacts=ArtifactSpecCollection(available),
        available_artifact_producers=artifact_producers_for_outputs(
            available,
            groups=(None, None, None),
            invocation_keys=tuple(
                FunctionInvocationKey(f"fixture_producer_{index}", "default", 0)
                for index in range(len(available))
            ),
        ),
    )
    cases = (
        ("Threshold", find_maxima, ("DNA",)),
        ("Mask", find_maxima_with_mask, ("DNA", "Mask")),
        ("Within Objects", find_maxima_with_mask, ("DNA", "Nuclei")),
    )
    for mode, expected_callable, input_names in cases:
        module = ModuleBlock(
            name="FindMaxima",
            module_num=1,
            setting_records=[
                ModuleSetting("Select the input image", "DNA"),
                ModuleSetting("Name the output image", "MaximaDNA"),
                ModuleSetting("Method for excluding background", mode),
                ModuleSetting("Select the image to use as a mask", "Mask"),
                ModuleSetting("Select the objects to search within", "Nuclei"),
            ],
        )
        contract = FindMaximaModule.callable_contract(
            module=module,
            invocation_key=FunctionInvocationKey("find_maxima", "default", 0),
            step_context=context,
        )

        assert contract.artifact_inputs.names() == input_names
        assert FindMaximaModule.resolve_function(
            module,
            contract=contract,
            source_bindings=StepSourceBindingsConfig(),
        ) is expected_callable

    assert FindMaximaModule._exclude_mode(
        ModuleBlock(
            name="FindMaxima",
            module_num=1,
            setting_records=[
                ModuleSetting(
                    "Method for excluding background",
                    "Within Objects",
                )
            ],
        )
    ) is ExcludeMode.OBJECTS


def test_synthetic_cppipe_import_and_public_transport_round_trip(
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "uncovered-image-measurements.cppipe"
    cppipe.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: https://cellprofiler.org",
                "NamesAndTypes:[module_num:1|enabled:True]",
                "    Assignments count:1",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                '    Select the rule criteria:and (file does contain "DNA")',
                "FlipAndRotate:[module_num:2|enabled:True]",
                "    Select the input image:DNA",
                "    Name the output image:RotatedDNA",
                "    Select method to flip image:Left to right",
                "    Select method to rotate image:Enter angle",
                "    Crop away the rotated edges?:Yes",
                "    Calculate rotation:Individually",
                "    Enter coordinates of the top or left pixel:0,0",
                "    Enter the coordinates of the bottom or right pixel:0,100",
                "    Select how the specified points should be aligned:horizontally",
                "    Enter angle of rotation:15",
                "FindMaxima:[module_num:3|enabled:True]",
                "    Select the input image:RotatedDNA",
                "    Name the output image:MaximaDNA",
                "    Individually label maxima?:Yes",
                "    Minimum distance between maxima:3",
                "    Method for excluding background:Threshold",
                "    Specify the minimum intensity of a peak:0.2",
                "    Select the image to use as a mask:None",
                "    Select the objects to search within:None",
            )
        ),
        encoding="utf-8",
    )

    steps, _pipeline_config = import_cellprofiler_pipeline(cppipe)
    source = FunctionStepTransportAuthority.source_from_pipeline(steps)
    namespace: dict[str, object] = {}
    exec(compile(source, "<uncovered-image-measurements>", "exec"), namespace)
    reconstructed = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )

    assert [step.name for step in steps] == ["FlipAndRotate", "FindMaxima"]
    assert "FlipMethod.LEFT_TO_RIGHT" in source
    assert "RotateMethod.ANGLE" in source
    assert "'rotation_angle': 15.0" in source
    assert "'min_distance': 3" in source
    assert "'min_intensity': 0.2" in source
    assert FunctionStepTransportAuthority.source_from_pipeline(reconstructed) == source
