"""Focused contracts for uncovered CellProfiler object-producing leaves."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
)
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
)
from openhcs.core.runtime_image_values import image_payload_metadata
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.interop.cellprofiler.database_column_dialect import (
    CellProfilerObjectCoreMeasurementFeature,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    CellProfilerMeasurementFeature,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    measurement_table_for_module,
)
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.object_images import (
    ConvertImageToObjectsModule,
    ConvertObjectsToImageModule,
    ObjectConversionStats,
    convert_image_to_objects,
)
from openhcs.processing.backends.cellprofiler.worms import (
    DeadWormAngleMeasurement,
    DeadWormStats,
    IdentifyDeadWormsModule,
    identify_dead_worms,
)

CONVERT_SETTINGS = (
    ("Select the input image", "Binary"),
    ("Name the output objects", "Cells"),
    ("Convert to boolean image", "Yes"),
    ("Preserve original labels", "No"),
    ("Background label", "0"),
    ("Connectivity", "1"),
)
DEAD_WORM_SETTINGS = (
    ("Select the input image", "CellMask"),
    ("Name the dead worm objects to be identified", "DeadWorms"),
    ("Worm width", "10"),
    ("Worm length", "100"),
    ("Number of angles", "32"),
    ("Automatically calculate distance parameters?", "Yes"),
    ("Spatial distance", "5"),
    ("Angular distance", "30"),
)


def _module(module_type, settings, *, module_num: int) -> ModuleBlock:
    return ModuleBlock(
        name=module_type.require_module_name(),
        module_num=module_num,
        setting_records=[ModuleSetting(name, value) for name, value in settings],
    )


def _contract(module_type, module: ModuleBlock, source: ArtifactSpec):
    invocation_key = FunctionInvocationKey(
        module_type.function_name,
        DEFAULT_GROUP_KEY,
        0,
    )
    producers = artifact_producers_for_outputs(
        (source,),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey("fixture_producer", DEFAULT_GROUP_KEY, 0),
        ),
    )
    contract = module_type.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_index=module.module_num - 1,
            available_artifacts=ArtifactSpecCollection((source,)),
            main_flow_artifacts=ArtifactSpecCollection(()),
            available_artifact_producers=producers,
        ),
    )
    module_type.validate_callable_artifact_abi(
        module_type.require_callable(),
        contract,
    )
    return invocation_key, producers, contract


@pytest.mark.parametrize(
    ("module_type", "settings", "source_name", "object_name", "expected_kwargs"),
    (
        (
            ConvertImageToObjectsModule,
            CONVERT_SETTINGS,
            "Binary",
            "Cells",
            {
                "select_the_input_image": "Binary",
                "name_the_output_objects": "Cells",
                "cast_to_bool": True,
                "preserve_label": False,
                "background": 0,
                "connectivity": 1,
            },
        ),
        (
            IdentifyDeadWormsModule,
            DEAD_WORM_SETTINGS,
            "CellMask",
            "DeadWorms",
            {
                "select_the_input_image": "CellMask",
                "name_the_dead_worm_objects_to_be_identified": "DeadWorms",
                "worm_width": 10,
                "worm_length": 100,
                "angle_count": 32,
                "auto_distance": True,
                "space_distance": 5.0,
                "angular_distance": 30.0,
            },
        ),
    ),
)
def test_uncovered_object_leaf_contracts_preserve_native_names_and_abi(
    module_type,
    settings,
    source_name: str,
    object_name: str,
    expected_kwargs: dict[str, object],
) -> None:
    module = _module(module_type, settings, module_num=2)
    _key, _producers, contract = _contract(
        module_type,
        module,
        ArtifactSpec.output(source_name, ImageArtifactType),
    )

    assert tuple(spec.artifact_type for spec in contract.artifact_outputs) == (
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
    )
    measurement, objects = contract.artifact_outputs
    assert objects.name == object_name
    assert any(
        type(relation) is ArtifactSpecRelation and relation.source == objects.ref()
        for relation in measurement.relations
    )
    assert (
        module_type.bind_settings(
            module,
            binder=SettingsBinder(),
        ).kwargs
        == expected_kwargs
    )


def test_object_leaf_callables_emit_schema_rows_then_nominal_labels() -> None:
    binary = np.zeros((12, 12), dtype=np.uint8)
    binary[2:5, 2:5] = 1
    binary[7:10, 7:10] = 1
    converted = inspect.unwrap(convert_image_to_objects)(
        binary,
        cast_to_bool=True,
        connectivity=1,
    )

    assert converted[0] is binary
    assert isinstance(converted[1], DataclassMeasurementColumnarRows)
    assert converted[1].row_type is ObjectConversionStats
    assert tuple(field.name for field in converted[1].fields) == (
        "slice_index",
        "object_count",
        "mean_area",
        "total_area",
    )
    assert isinstance(converted[2], ObjectLabelValue)
    assert set(np.unique(object_label_dense_array(converted[2]))) == {0, 1, 2}

    dead_worms = inspect.unwrap(identify_dead_worms)(
        np.zeros((12, 12), dtype=np.uint8),
        angle_count=2,
    )
    assert isinstance(dead_worms[1], ConcatenatedColumnarRows)
    assert dead_worms[1].row_count() == 1
    assert tuple(
        batch.row_type
        for batch in dead_worms[1].row_batches
        if isinstance(batch, DataclassMeasurementColumnarRows)
    ) == (DeadWormStats, DeadWormAngleMeasurement)
    assert isinstance(dead_worms[2], ObjectLabelValue)
    assert dead_worms[2].object_label_domain().declared_object_count == 0


def test_identify_dead_worms_projects_exact_native_measurement_features() -> None:
    module = _module(
        IdentifyDeadWormsModule,
        DEAD_WORM_SETTINGS,
        module_num=2,
    )
    _key, _producers, contract = _contract(
        IdentifyDeadWormsModule,
        module,
        ArtifactSpec.output("CellMask", ImageArtifactType),
    )
    source_rows = ConcatenatedColumnarRows(
        (
            DataclassMeasurementColumnarRows(
                (DeadWormStats(0, 2, 3.5, 3.5, 45.0),),
                row_type=DeadWormStats,
            ),
            DataclassMeasurementColumnarRows(
                (
                    DeadWormAngleMeasurement(
                        slice_index=0,
                        object_label=1,
                        angle=22.5,
                    ),
                    DeadWormAngleMeasurement(
                        slice_index=0,
                        object_label=2,
                        angle=67.5,
                    ),
                ),
                row_type=DeadWormAngleMeasurement,
            ),
        )
    )
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[5:7, 5:7] = 2
    label_payload = SourceImageObjectLabelBuildRequest(
        image=np.zeros((8, 8), dtype=np.float32),
        labels=labels,
        declared_object_count=2,
        declared_object_ids=(1, 2),
    ).payload()
    measurement, objects = contract.artifact_outputs
    table = measurement_table_for_module(
        SimpleNamespace(
            callable_contract=contract,
            spec=measurement,
            output_value=source_rows,
            artifact_output_value=lambda _spec: label_payload,
            object_label_output_domain_scope=lambda: None,
            adapter=SimpleNamespace(
                request=SimpleNamespace(
                    plane_projection=SimpleNamespace(plane_index=None)
                )
            ),
            measurement_source_metadata=lambda _specs: image_payload_metadata(
                label_payload
            ),
            source=SimpleNamespace(source_image_name="CellMask"),
        )
    )
    rows = tuple(table.rows.iter_row_mappings())

    assert objects.name == "DeadWorms"
    assert table.source_image_name == "CellMask"
    assert IdentifyDeadWormsModule.owns_measurement_feature_name("Worm_Angle")
    assert not IdentifyDeadWormsModule.owns_measurement_feature_name("Worm_Unknown")
    count_name = CellProfilerMeasurementFeature.object_count("DeadWorms").name
    assert tuple(row[count_name] for row in rows if count_name in row) == (2,)
    object_rows = tuple(
        row for row in rows if MeasurementRowAxisField.OBJECT_LABEL.value in row
    )
    assert {row[MeasurementRowAxisField.OBJECT_LABEL.value] for row in object_rows} == {
        1,
        2,
    }
    assert all(
        row[MeasurementRowAxisField.OBJECT_NAME.value] == "DeadWorms"
        for row in object_rows
    )
    features = {row[MeasurementRowAxisField.FEATURE_NAME.value] for row in object_rows}
    assert {
        CellProfilerObjectCoreMeasurementFeature.CENTER_X.value,
        CellProfilerObjectCoreMeasurementFeature.CENTER_Y.value,
        "Worm_Angle",
    }.issubset(features)
    angle_rows = tuple(
        row
        for row in object_rows
        if row[MeasurementRowAxisField.FEATURE_NAME.value] == "Worm_Angle"
    )
    assert tuple(
        row[MeasurementRowValueField.RESULT_VALUE.value] for row in angle_rows
    ) == (22.5, 67.5)
    assert CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER.value == (
        "Number_Object_Number"
    )


def test_convert_image_objects_are_consumable_by_downstream_object_binding() -> None:
    source = ArtifactSpec.output("Binary", ImageArtifactType)
    convert_module = _module(
        ConvertImageToObjectsModule,
        CONVERT_SETTINGS,
        module_num=2,
    )
    convert_key, source_producers, convert_contract = _contract(
        ConvertImageToObjectsModule,
        convert_module,
        source,
    )
    convert_producers = artifact_producers_for_outputs(
        convert_contract.artifact_outputs.specs,
        groups=(None,),
        invocation_keys=(convert_key,),
    )
    render_module = _module(
        ConvertObjectsToImageModule,
        (
            ("Select the input objects", "Cells"),
            ("Name the output image", "CellMask"),
            ("Select the color format", "Binary"),
            ("Select the colormap", "jet"),
        ),
        module_num=3,
    )
    render_contract = ConvertObjectsToImageModule.callable_contract(
        module=render_module,
        invocation_key=FunctionInvocationKey(
            "convert_objects_to_image",
            DEFAULT_GROUP_KEY,
            0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=2,
            available_artifacts=ArtifactSpecCollection(
                (source, *convert_contract.artifact_outputs.specs)
            ),
            main_flow_artifacts=ArtifactSpecCollection(()),
            available_artifact_producers=(*source_producers, *convert_producers),
        ),
    )

    assert render_contract.artifact_inputs.names_of_artifact_type(
        ObjectLabelsArtifactType
    ) == ("Cells",)


def test_object_leaf_cppipe_import_and_public_transport_round_trip(
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "uncovered-object-leaves.cppipe"
    cppipe.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:Binary
    Select the rule criteria:and (file does contain "Binary")
ConvertImageToObjects:[module_num:2|enabled:True]
    Select the input image:Binary
    Name the output objects:Cells
    Convert to boolean image:Yes
    Preserve original labels:No
    Background label:0
    Connectivity:1
ConvertObjectsToImage:[module_num:3|enabled:True]
    Select the input objects:Cells
    Name the output image:CellMask
    Select the color format:Binary
    Select the colormap:jet
IdentifyDeadWorms:[module_num:4|enabled:True]
    Select the input image:CellMask
    Name the dead worm objects to be identified:DeadWorms
    Worm width:10
    Worm length:100
    Number of angles:32
    Automatically calculate distance parameters?:Yes
    Spatial distance:5
    Angular distance:30
""",
        encoding="utf-8",
    )

    steps, _config = import_cellprofiler_pipeline(cppipe)
    assert [step.name for step in steps] == [
        "ConvertImageToObjects",
        "ConvertObjectsToImage",
        "IdentifyDeadWorms",
    ]
    invocations = tuple(
        next(normalize_function_pattern(step.func).iter_items()) for step in steps
    )
    assert tuple(item.key.function_name for item in invocations) == (
        "convert_image_to_objects",
        "convert_objects_to_image",
        "identify_dead_worms",
    )
    assert invocations[0].kwargs_dict == {
        "name_the_output_objects": "Cells",
        "cast_to_bool": True,
    }
    assert invocations[1].kwargs_dict["name_the_output_image"] == "CellMask"
    assert invocations[1].kwargs_dict["image_mode"].value == "binary"
    assert invocations[2].kwargs_dict == {
        "name_the_dead_worm_objects_to_be_identified": "DeadWorms",
    }

    source = FunctionStepTransportAuthority.source_from_pipeline(steps)
    namespace: dict[str, object] = {}
    exec(compile(source, "<uncovered-object-leaves>", "exec"), namespace)
    restored = FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)
    assert FunctionStepTransportAuthority.source_from_pipeline(restored) == source
