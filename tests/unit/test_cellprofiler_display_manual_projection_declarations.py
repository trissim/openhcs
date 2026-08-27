"""Focused declarations for CellProfiler display, manual-object, and projection modules."""

from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import object_label_dense_array
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
    FieldSpec,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.display_modules import (
    DisplayDensityPlotModule,
    DisplayHistogramModule,
    DisplayPlatemapModule,
    DisplayScatterPlotModule,
    ObjectOrImage,
    display_density_plot,
    display_histogram,
    display_platemap,
    display_scatter_plot,
)
from openhcs.processing.backends.cellprofiler.edit_objects import (
    EditObjectsManuallyModule,
    RenumberChoice,
    edit_objects_manually,
)
from openhcs.processing.backends.cellprofiler.manual_objects import (
    IdentifyObjectsManuallyModule,
    identify_objects_manually,
)
from openhcs.processing.backends.cellprofiler.projection import (
    MakeProjectionModule,
    ProjectionType,
    make_projection,
)
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
)


def _module(name: str, settings: dict[str, str]) -> ModuleBlock:
    return ModuleBlock(
        name=name,
        module_num=1,
        enabled=True,
        setting_records=[
            ModuleSetting(_setting_name, _setting_value)
            for (_setting_name, _setting_value) in settings.items()
        ],
    )


def _contract(
    module_type: type[CellProfilerModule],
    module: ModuleBlock,
    *,
    available: tuple[ArtifactSpec, ...],
    producers=(),
):
    return module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            str(module_type.function_name),
            DEFAULT_GROUP_KEY,
            0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_name=str(module_type.module_name),
            step_index=0,
            available_artifacts=ArtifactSpecCollection(available),
            main_flow_artifacts=ArtifactSpecCollection(()),
            available_artifact_producers=producers,
        ),
    )


def _measurement_fixture():
    objects = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)
    measurements = ArtifactSpec.output(
        "ObjectShapeMeasurements",
        MeasurementsArtifactType,
        relations=(
            GroupLineageSourceRelation(
                source=objects.for_plan_type(ArtifactInputPlan).ref()
            ),
        ),
        measurement_feature_owner=MeasureObjectSizeShapeModule,
    )
    (producer,) = artifact_producers_for_outputs(
        (measurements,),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey(
                "measure_object_size_shape",
                DEFAULT_GROUP_KEY,
                0,
            ),
        ),
    )
    return objects, measurements, producer


DISPLAY_CASES = (
    (
        DisplayDensityPlotModule,
        display_density_plot,
        {
            "Select the object to display on the X-axis": "Objects",
            "Select the object measurement to plot on the X-axis": "AreaShape_Area",
            "Select the object to display on the Y-axis": "Objects",
            "Select the object measurement to plot on the Y-axis": "AreaShape_Area",
            "Select the grid size": "100",
            "How should the X-axis be scaled?": "linear",
            "How should the Y-axis be scaled?": "linear",
            "How should the colorbar be scaled?": "linear",
            "Select the color map": "jet",
            "Enter a title for the plot, if desired": "",
        },
    ),
    (
        DisplayHistogramModule,
        display_histogram,
        {
            "Select the object whose measurements will be displayed": "Objects",
            "Select the object measurement to plot": "AreaShape_Area",
            "Number of bins": "100",
            "How should the X-axis be scaled?": "linear",
            "How should the Y-axis be scaled?": "linear",
            "Enter a title for the plot, if desired": "",
            "Specify min/max bounds for the X-axis?": "No",
            "Minimum/maximum values for the X-axis": "0.0,1.0",
        },
    ),
    (
        DisplayPlatemapModule,
        display_platemap,
        {
            "Display object or image measurements?": "Object",
            "Select the object whose measurements will be displayed": "Objects",
            "Select the measurement to plot": "AreaShape_Area",
            "Select your plate metadata": "AreaShape_Area",
            "Multiwell plate format": "96",
            "Select your well metadata": "AreaShape_Area",
            "Select your well row metadata": "AreaShape_Area",
            "Select your well column metadata": "AreaShape_Area",
            "How should the values be aggregated?": "avg",
            "Enter a title for the plot, if desired": "",
            "Well metadata format": "Well name",
        },
    ),
    (
        DisplayScatterPlotModule,
        display_scatter_plot,
        {
            "Type of measurement to plot on X-axis": "Object",
            "Select the object to plot on the X-axis": "Objects",
            "Select the measurement to plot on the X-axis": "AreaShape_Area",
            "Type of measurement to plot on Y-axis": "Object",
            "Select the object to plot on the Y-axis": "Objects",
            "Select the measurement to plot on the Y-axis": "AreaShape_Area",
            "How should the X-axis be scaled?": "linear",
            "How should the Y-axis be scaled?": "linear",
            "Enter a title for the plot, if desired": "",
        },
    ),
)


@pytest.mark.parametrize(
    ("module_type", "func", "settings"),
    DISPLAY_CASES,
    ids=lambda value: getattr(value, "module_name", None),
)
def test_display_declarations_consume_prior_measurements_and_emit_one_nominal_table(
    module_type,
    func,
    settings,
) -> None:
    objects, measurements, producer = _measurement_fixture()

    contract = _contract(
        module_type,
        _module(str(module_type.module_name), settings),
        available=(objects, measurements),
        producers=(producer,),
    )

    assert (
        CellProfilerModule.require_callable_contract_owner(
            CallableContract.from_callable(func)
        )
        is module_type
    )
    assert tuple(spec.artifact_type for spec in contract.artifact_inputs) == (
        MeasurementsArtifactType,
    )
    assert tuple(spec.artifact_type for spec in contract.artifact_outputs) == (
        MeasurementsArtifactType,
    )


def test_edit_and_projection_declarations_preserve_exact_output_order() -> None:
    image = ArtifactSpec.output("InputImage", ImageArtifactType)
    objects = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)
    producers = artifact_producers_for_outputs(
        (image, objects),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey("fixture_producer", DEFAULT_GROUP_KEY, 0),
        ),
    )

    edit_contract = _contract(
        EditObjectsManuallyModule,
        _module(
            "EditObjectsManually",
            {
                "Select the objects to be edited": "Objects",
                "Name the edited objects": "EditedObjects",
                "Numbering of the edited objects": "Renumber",
                "Display a guiding image?": "No",
                "Select the guiding image": "InputImage",
                "Allow overlapping objects?": "No",
            },
        ),
        available=(image, objects),
        producers=producers,
    )
    projection_contract = _contract(
        MakeProjectionModule,
        _module(
            "MakeProjection",
            {
                "Select the input image": "InputImage",
                "Type of projection": "Maximum",
                "Name the output image": "Projection",
                "Frequency": "6.0",
            },
        ),
        available=(image,),
        producers=producers[:1],
    )

    assert tuple(spec.artifact_type for spec in edit_contract.artifact_inputs) == (
        ObjectLabelsArtifactType,
    )
    assert tuple(spec.artifact_type for spec in edit_contract.artifact_outputs) == (
        MeasurementsArtifactType,
        ObjectLabelsArtifactType,
        ObjectLineageArtifactType,
    )
    assert tuple(
        spec.artifact_type for spec in projection_contract.artifact_inputs
    ) == (ImageArtifactType,)
    assert tuple(
        spec.artifact_type for spec in projection_contract.artifact_outputs
    ) == (
        ImageArtifactType,
        MeasurementsArtifactType,
    )


def test_interactive_manual_identification_rejects_headless_cppipe_import() -> None:
    with pytest.raises(ValueError, match="requires interactive desktop input"):
        IdentifyObjectsManuallyModule.validate_pipeline_import(
            _module(
                "IdentifyObjectsManually",
                {
                    "Select the input image": "InputImage",
                    "Name the objects to be identified": "ManualObjects",
                },
            )
        )


def test_histogram_float_range_reconstructs_as_one_cellprofiler_setting_row() -> None:
    invocation = next(
        normalize_function_pattern(
            (
                display_histogram,
                {
                    "object_name": "Objects",
                    "measurement_feature": "AreaShape_Area",
                    "x_bounds": (2.0, 8.0),
                },
            )
        ).iter_items()
    )
    blocks, consumed = DisplayHistogramModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=ArtifactDeclarationStepContext(
            step_name="DisplayHistogram",
            step_index=1,
        ),
    )

    assert consumed == ()
    assert len(blocks) == 1
    range_rows = tuple(
        record.value
        for record in blocks[0].setting_records
        if record.name == "Minimum/maximum values for the X-axis"
    )
    assert range_rows == ("2.0,8.0",)


def test_display_contract_rejects_unselected_measurement_feature() -> None:
    with pytest.raises(ValueError, match="selected prior measurement feature"):
        _contract(
            DisplayHistogramModule,
            _module(
                "DisplayHistogram",
                {
                    "Select the object whose measurements will be displayed": "None",
                    "Select the object measurement to plot": "None",
                },
            ),
            available=(),
        )


def test_assigned_public_function_steps_round_trip_without_private_state() -> None:
    steps = [
        FunctionStep(
            func=(
                display_density_plot,
                {
                    "x_object_name": "Objects",
                    "x_measurement_feature": "AreaShape_Area",
                    "y_object_name": "Objects",
                    "y_measurement_feature": "AreaShape_Perimeter",
                },
            ),
            name="DisplayDensityPlot",
        ),
        FunctionStep(
            func=(
                display_histogram,
                {
                    "object_name": "Objects",
                    "measurement_feature": "AreaShape_Area",
                    "x_bounds": (2.0, 8.0),
                },
            ),
            name="DisplayHistogram",
        ),
        FunctionStep(
            func=(
                display_platemap,
                {
                    "object_name": "Objects",
                    "measurement_feature": "AreaShape_Area",
                    "plate_metadata_feature": "Metadata_Plate",
                    "well_metadata_feature": "Metadata_Well",
                },
            ),
            name="DisplayPlatemap",
        ),
        FunctionStep(
            func=(
                display_scatter_plot,
                {
                    "x_object_name": "Objects",
                    "x_measurement_feature": "AreaShape_Area",
                    "y_object_name": "Objects",
                    "y_measurement_feature": "AreaShape_Perimeter",
                },
            ),
            name="DisplayScatterPlot",
        ),
        FunctionStep(
            func=(edit_objects_manually, {"renumber_choice": RenumberChoice.RETAIN}),
            name="EditObjectsManually",
        ),
        FunctionStep(
            func=identify_objects_manually,
            name="IdentifyObjectsManually",
        ),
        FunctionStep(
            func=(make_projection, {"projection_type": ProjectionType.MAXIMUM}),
            name="MakeProjection",
        ),
    ]

    source = FunctionStepTransportAuthority.source_from_pipeline(steps)
    namespace: dict[str, object] = {}
    exec(compile(source, "<assigned-cellprofiler-steps>", "exec"), namespace)
    reconstructed = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )

    assert [step.name for step in reconstructed] == [step.name for step in steps]
    assert FunctionStepTransportAuthority.source_from_pipeline(reconstructed) == source


def test_display_callables_resolve_values_from_declared_measurement_tables() -> None:
    object_table = MeasurementTable(
        name="ObjectMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "slice_index": 0,
                    "object_label": 1,
                    "AreaShape_Area": 2.0,
                    "AreaShape_Perimeter": 5.0,
                },
                {
                    "slice_index": 0,
                    "object_label": 2,
                    "AreaShape_Area": 4.0,
                    "AreaShape_Perimeter": 9.0,
                },
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("AreaShape_Area", float),
                FieldSpec("AreaShape_Perimeter", float),
            ),
        ),
        subject=MeasurementSubject(
            MeasurementScope.OBJECT,
            "Objects",
            "object_label",
        ),
    )
    image_table = MeasurementTable(
        name="ImageMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "slice_index": 0,
                    "Intensity_Mean": 3.0,
                    "Metadata_Plate": "Plate1",
                    "Metadata_Well": "A01",
                },
                {
                    "slice_index": 1,
                    "Intensity_Mean": 7.0,
                    "Metadata_Plate": "Plate1",
                    "Metadata_Well": "A02",
                },
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("Intensity_Mean", float),
                FieldSpec("Metadata_Plate", str),
                FieldSpec("Metadata_Well", str),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    object_plate_table = MeasurementTable(
        name="ObjectPlateMeasurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "object_label": 1, "AreaShape_Area": 2.0},
                {"slice_index": 0, "object_label": 2, "AreaShape_Area": 4.0},
                {"slice_index": 1, "object_label": 1, "AreaShape_Area": 6.0},
            ),
            fields=(
                FieldSpec("slice_index", int),
                FieldSpec("object_label", int),
                FieldSpec("AreaShape_Area", float),
            ),
        ),
        subject=MeasurementSubject(
            MeasurementScope.OBJECT,
            "Objects",
            "object_label",
        ),
    )
    image = np.zeros((2, 2), dtype=np.float32)
    tables = (object_table, image_table)

    _image, density = display_density_plot(
        image,
        x_object_name="Objects",
        x_measurement_feature="AreaShape_Area",
        y_object_name="Objects",
        y_measurement_feature="AreaShape_Perimeter",
        gridsize=4,
        measurement_tables=tables,
    )
    _image, histogram = display_histogram(
        image,
        object_name="Objects",
        measurement_feature="AreaShape_Area",
        num_bins=2,
        measurement_tables=tables,
    )
    _image, scatter = display_scatter_plot(
        image,
        x_object_name="Objects",
        x_measurement_feature="AreaShape_Area",
        y_object_name="Objects",
        y_measurement_feature="AreaShape_Perimeter",
        measurement_tables=tables,
    )
    _image, platemap = display_platemap(
        image,
        objects_or_image=ObjectOrImage.IMAGE,
        measurement_feature="Intensity_Mean",
        plate_metadata_feature="Metadata_Plate",
        well_metadata_feature="Metadata_Well",
        measurement_tables=tables,
    )
    _image, object_platemap = display_platemap(
        image,
        objects_or_image=ObjectOrImage.OBJECTS,
        object_name="Objects",
        measurement_feature="AreaShape_Area",
        plate_metadata_feature="Metadata_Plate",
        well_metadata_feature="Metadata_Well",
        measurement_tables=(object_plate_table, image_table),
    )

    assert density.row_mappings()[0]["num_points"] == 2
    assert histogram.row_mappings()[0]["total_count"] == 2
    assert scatter.row_mappings()[0]["point_count"] == 2
    assert {row["well"] for row in platemap.row_mappings() if "well" in row} == {
        "A01",
        "A02",
    }
    object_well_values = {
        row["well"]: row["value"]
        for row in object_platemap.row_mappings()
        if "well" in row
    }
    assert object_well_values == {"A01": 3.0, "A02": 6.0}


def test_manual_identification_rejects_missing_or_misaligned_interactive_labels() -> (
    None
):
    image = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(NotImplementedError, match="cannot silently synthesize"):
        identify_objects_manually(image)
    with pytest.raises(ValueError, match="must match the guiding image plane"):
        identify_objects_manually(image, np.zeros((3, 4), dtype=np.int32))


def test_manual_identification_preserves_actual_sparse_object_id_domain() -> None:
    image = np.zeros((3, 3), dtype=np.float32)
    labels = np.array(
        (
            (0, 2, 2),
            (0, 0, 0),
            (5, 5, 0),
        ),
        dtype=np.int32,
    )

    _image, rows, payload = identify_objects_manually(image, labels)

    assert rows.row_mappings()[0]["object_count"] == 2
    assert set(np.unique(object_label_dense_array(payload))) == {0, 2, 5}
    _image, edit_rows, retained, _relationship = edit_objects_manually(
        image,
        payload,
        renumber_choice=RenumberChoice.RETAIN,
    )
    assert edit_rows.row_mappings()[0]["edited_object_count"] == 2
    assert set(np.unique(object_label_dense_array(retained))) == {0, 2, 5}
    with pytest.raises(NotImplementedError, match="layered label payload"):
        edit_objects_manually(image, payload, allow_overlap=True)


def test_projection_special_output_is_nominal_columnar_rows() -> None:
    image = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

    projected, rows = make_projection(image, ProjectionType.MAXIMUM)

    np.testing.assert_array_equal(projected, np.max(image, axis=0))
    assert isinstance(rows, ColumnarRows)
    assert rows.row_mappings()[0]["projection_type"] == "Maximum"
