"""Executable CellProfiler ExportToSpreadsheet boundary tests."""

from __future__ import annotations

import csv
import inspect
import io

import pytest

from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ObjectLabelsArtifactType,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.callable_contract import CallableContract, FunctionStepExecutionScope
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    ObjectRelationshipDeclaration,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactBatch,
    RuntimeArtifactLocation,
    StoredRuntimeValue,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.source_metadata import ORIGINAL_SOURCE_METADATA_FIELD
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_settings import (
    ModuleSettingCoverageStatus,
)
from openhcs.interop.cellprofiler.worm_measurements import (
    WormControlPointAxis,
    WormControlPointMeasurementField,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.spreadsheet_export import (
    ExportToSpreadsheetModule,
    SpreadsheetColumnSelection,
    SpreadsheetDelimiter,
    SpreadsheetFileSelection,
    SpreadsheetNanRepresentation,
    export_to_spreadsheet,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureObjectIntensityModule,
)
from openhcs.processing.backends.cellprofiler.crop import CropModule
from openhcs.processing.backends.cellprofiler.tracking import TrackObjectsModule
from openhcs.processing.materialization import (
    FileBundleOptions,
    MaterializationSpec,
    WriteMode,
    materialize,
)


def test_export_to_spreadsheet_declares_exact_plate_callable_abi() -> None:
    contract = CallableContract.from_callable(export_to_spreadsheet)
    parameter = inspect.signature(export_to_spreadsheet).parameters["artifact_batch"]
    module_type = CellProfilerModule.require_module("ExportToSpreadsheet")

    assert ExportToSpreadsheetModule.emits_function_step()
    assert not ExportToSpreadsheetModule.uses_cellprofiler_runtime_adapter()
    assert module_type is ExportToSpreadsheetModule
    assert module_type.require_callable() is export_to_spreadsheet
    assert export_to_spreadsheet.__module__ == module_type.__module__
    assert module_type.__module__ == (
        "openhcs.processing.backends.cellprofiler.spreadsheet_export"
    )
    assert contract.execution_scope is FunctionStepExecutionScope.PLATE
    assert contract.processing_contract is None
    assert contract.runtime_adapter is None
    assert contract.runtime_bound_parameter_types == (RuntimeArtifactBatch,)
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_export_to_spreadsheet_contract_selects_ordered_tables_and_declares_bundle() -> (
    None
):
    available = ArtifactSpecCollection(
        (
            ArtifactSpec.output("measurements_a", MeasurementsArtifactType),
            ArtifactSpec.output("pixels", ImageArtifactType),
            ArtifactSpec.output("relationships", RelationshipsArtifactType),
            ArtifactSpec.output("measurements_b", MeasurementsArtifactType),
        )
    )
    contract = ExportToSpreadsheetModule.callable_contract(
        module=ModuleBlock(name="ExportToSpreadsheet", module_num=99),
        invocation_key=FunctionInvocationKey(
            function_name="export_to_spreadsheet",
            group_key="default",
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=4,
            available_artifacts=available,
            main_flow_artifacts=ArtifactSpecCollection(()),
            available_artifact_producers=artifact_producers_for_outputs(
                tuple(
                    spec for spec in available if spec.plan_type is ArtifactOutputPlan
                ),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey("fixture_producer", "default", 0),
                ),
            ),
        ),
    )

    runtime_inputs = contract.artifact_inputs
    declared_outputs = contract.artifact_outputs

    assert tuple(spec.name for spec in runtime_inputs) == (
        "measurements_a",
        "relationships",
        "measurements_b",
    )
    assert all(spec.plan_type is ArtifactInputPlan for spec in runtime_inputs)
    assert len(declared_outputs) == 1
    assert declared_outputs[0].name == "ExportToSpreadsheet_5_files"
    assert declared_outputs[0].artifact_type is SpecialArtifactType
    assert isinstance(declared_outputs[0].materialization, MaterializationSpec)
    assert declared_outputs[0].materialization.outputs == (FileBundleOptions(),)
    assert tuple(
        relation.source for relation in declared_outputs[0].relations
    ) == tuple(spec.ref() for spec in runtime_inputs)


def test_export_to_spreadsheet_binds_scalars_and_repeated_file_rows() -> None:
    module = _module(
        (
            ("Select the column delimiter", 'Comma (",")'),
            ("Add image metadata columns to your object data file?", "Yes"),
            ("Add image file and folder names to your object data file?", "No"),
            ("Select measurements to export", "Yes"),
            (
                "Calculate the per-image mean values for object measurements?",
                "Yes",
            ),
            (
                "Calculate the per-image median values for object measurements?",
                "No",
            ),
            (
                "Calculate the per-image standard deviation values for object measurements?",
                "No",
            ),
            ("Output file location", r"Default Output Folder sub-folder|\g<Run>"),
            ("Create a GenePattern GCT file?", "No"),
            ("Select source of sample row name", "Metadata"),
            ("Select the image to use as the identifier", "None"),
            ("Select the metadata to use as the identifier", "None"),
            ("Export all measurement types?", "No"),
            ("Press button to select measurements", "Image|Count,Cells|Area"),
            ("Representation of Nan/Inf", "Null"),
            ("Add a prefix to file names?", "Yes"),
            ("Filename prefix", "Plate_"),
            ("Overwrite existing files without warning?", "Yes"),
            ("Data to export", "Image"),
            (
                "Combine these object measurements with those of the previous object?",
                "No",
            ),
            ("File name", "image-data.csv"),
            ("Use the object name for the file name?", "No"),
            ("Data to export", "Cells"),
            (
                "Combine these object measurements with those of the previous object?",
                "No",
            ),
            ("File name", "DATA.csv"),
            ("Use the object name for the file name?", "Yes"),
            ("Data to export", "Cytoplasm"),
            (
                "Combine these object measurements with those of the previous object?",
                "Yes",
            ),
            ("File name", "unused.csv"),
            ("Use the object name for the file name?", "Yes"),
        )
    )

    bound = ExportToSpreadsheetModule.bind_settings(
        module,
        binder=SettingsBinder(),
    )

    assert bound.kwargs["delimiter"] is SpreadsheetDelimiter.COMMA
    assert bound.kwargs["selected_columns"] == (
        SpreadsheetColumnSelection("Image", "Count"),
        SpreadsheetColumnSelection("Cells", "Area"),
    )
    assert bound.kwargs["nan_representation"] is SpreadsheetNanRepresentation.NULL
    assert bound.kwargs["output_directory"] == "{Run}"
    assert bound.kwargs["file_selections"] == (
        SpreadsheetFileSelection(("Image",), "image-data.csv"),
        SpreadsheetFileSelection(("Cells", "Cytoplasm"), "Cells.csv"),
    )
    assert not bound.unmapped_kwargs
    assert {record.status for record in bound.setting_coverage} <= {
        ModuleSettingCoverageStatus.BOUND,
        ModuleSettingCoverageStatus.IGNORED,
    }


def test_export_to_spreadsheet_overwrite_setting_is_contract_owned() -> None:
    module = _module((("Overwrite existing files without warning?", "No"),))

    bound = ExportToSpreadsheetModule.bind_settings(
        module,
        binder=SettingsBinder(),
    )
    contract = ExportToSpreadsheetModule.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name="export_to_spreadsheet",
            group_key="default",
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            available_artifacts=ArtifactSpecCollection(()),
            main_flow_artifacts=ArtifactSpecCollection(()),
        ),
    )
    output = contract.artifact_outputs[0]

    assert bound.kwargs["overwrite_existing_files_without_warning"] is False
    assert not bound.unmapped_kwargs
    assert output.materialization.write_mode is WriteMode.ERROR


def test_export_to_spreadsheet_ignores_disabled_excel_size_limit() -> None:
    bound = ExportToSpreadsheetModule.bind_settings(
        _module((("Limit output to a size that is allowed in Excel", "No"),)),
        binder=SettingsBinder(),
    )

    assert not bound.unmapped_kwargs
    assert bound.kwargs == {"file_selections": ()}


def test_export_to_spreadsheet_rejects_enabled_excel_size_limit() -> None:
    with pytest.raises(ValueError, match="Excel row and column truncation"):
        ExportToSpreadsheetModule.bind_settings(
            _module((("Limit output to a size that is allowed in Excel", "Yes"),)),
            binder=SettingsBinder(),
        )


def test_export_to_spreadsheet_renders_only_declared_batch_records() -> None:
    measurements_a = _measurement_record(
        "measurements_a",
        axis_id="A01",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        rows=(
            {
                "slice_index": 0,
                "feature_name": "Count",
                "value": 2.0,
            },
            {
                "slice_index": 0,
                "feature_name": "BadValue",
                "value": float("nan"),
            },
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=(
                {
                    "site": "1",
                    "source_alias": "OrigColor",
                    ORIGINAL_SOURCE_METADATA_FIELD: {
                        "Run": "Run1",
                        "FrameNumber": "0",
                    },
                },
            )
        ),
    )
    cells = _measurement_record(
        "cells",
        axis_id="A01",
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells", "object_number"),
        rows=(
            {"slice_index": 0, "object_number": 1, "Area": 2.0, "Ignored": 9},
            {"slice_index": 0, "object_number": 2, "Area": 4.0, "Ignored": 8},
        ),
    )
    relationships = _relationship_record("relationships", axis_id="A01")
    undeclared = _measurement_record(
        "undeclared",
        axis_id="A01",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        rows=({"slice_index": 0, "Leaked": 999},),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("measurements_a", MeasurementsArtifactType),
            ArtifactSpec.input("cells", MeasurementsArtifactType),
            ArtifactSpec.input("relationships", RelationshipsArtifactType),
        ),
        records_by_axis={
            "A01": (undeclared, relationships, cells, measurements_a),
        },
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        select_measurements=True,
        selected_columns=(
            SpreadsheetColumnSelection("Image", "Count"),
            SpreadsheetColumnSelection("Image", "BadValue"),
            SpreadsheetColumnSelection("Image", "Metadata_FrameNumber"),
            SpreadsheetColumnSelection("Cells", "Area"),
        ),
        calculate_aggregate_means=True,
        output_directory="{Run}",
        export_all_measurement_types=False,
        file_selections=(
            SpreadsheetFileSelection(("Image",), "Image.csv"),
            SpreadsheetFileSelection(("Cells",), "Cells.csv"),
            SpreadsheetFileSelection(
                ("Object relationships",),
                "Relationships.csv",
            ),
        ),
        nan_representation=SpreadsheetNanRepresentation.NULL,
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert type(bundle) is dict
    assert tuple(bundle) == (
        "Run1/Image.csv",
        "Run1/Cells.csv",
        "Run1/Relationships.csv",
    )
    image_rows = tuple(csv.DictReader(io.StringIO(bundle["Run1/Image.csv"])))
    cell_rows = tuple(csv.DictReader(io.StringIO(bundle["Run1/Cells.csv"])))
    relationship_rows = tuple(
        csv.DictReader(io.StringIO(bundle["Run1/Relationships.csv"]))
    )
    assert image_rows == (
        {
            "image_number": "1",
            "Count": "2.0",
            "BadValue": "",
            "Metadata_FrameNumber": "0",
            "Mean_Cells_Area": "3.0",
        },
    )
    assert cell_rows == (
        {"image_number": "1", "object_label": "1", "Area": "2.0"},
        {"image_number": "1", "object_label": "2", "Area": "4.0"},
    )
    assert relationship_rows[0]["relationship_type"] == "related"
    assert relationship_rows[0]["source_role"] == "parent"
    assert relationship_rows[0]["target_role"] == "child"
    assert "Leaked" not in bundle["Run1/Image.csv"]
    assert "Ignored" not in bundle["Run1/Cells.csv"]


def test_export_to_spreadsheet_bundle_uses_generic_file_materialization() -> None:
    batch = RuntimeArtifactBatch(
        input_specs=(ArtifactSpec.input("measurements", MeasurementsArtifactType),),
        records_by_axis={
            "A01": (
                _measurement_record(
                    "measurements",
                    axis_id="A01",
                    subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
                    rows=({"slice_index": 0, "Count": 3},),
                ),
            )
        },
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )
    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )
    filemanager = FileManager({"memory": MemoryStorageBackend()})

    primary_path = materialize(
        MaterializationSpec(FileBundleOptions()),
        data=bundle,
        path="/analysis/ExportToSpreadsheet_1_files.pkl",
        filemanager=filemanager,
        backends=("memory",),
    )

    assert primary_path == "/analysis/Image.csv"
    assert filemanager.load(primary_path, "memory") == (b"image_number,Count\n1,3\n")


def test_export_to_spreadsheet_rejects_append_order_slice_synthesis() -> None:
    records = tuple(
        _measurement_record(
            "align_measurements",
            axis_id="A01",
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
            rows=(
                {
                    "slice_index": 0,
                    "source_image_name": "Stain2",
                    "Align_Xshift": shift,
                },
            ),
        )
        for shift in (-1.0, -2.0)
    )
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("align_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={"A01": records},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    with pytest.raises(ValueError, match="Conflicting sparse measurement values"):
        export_to_spreadsheet(
            add_filename_prefix=False,
            artifact_batch=batch,
        )


def test_export_to_spreadsheet_projects_site_group_scope_without_relabeling_stack() -> (
    None
):
    records = tuple(
        _measurement_record(
            "align_measurements",
            axis_id="A01",
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
            rows=(
                {
                    "slice_index": 0,
                    "source_image_name": "Stain2",
                    "feature_name": "Align_Xshift_Stain2",
                    "result_value": shift,
                },
            ),
            group_component=AllComponents.SITE,
            group_key=site,
        )
        for site, shift in (("1", -1.0), ("2", -2.0))
    )
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("align_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={"A01": records},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Image.csv"]))) == (
        {"image_number": "1", "Align_Xshift_Stain2": "-1.0"},
        {"image_number": "2", "Align_Xshift_Stain2": "-2.0"},
    )


def test_export_to_spreadsheet_uses_declared_image_set_identity_across_channels() -> (
    None
):
    image_record = _measurement_record(
        "image_measurements",
        axis_id="A01",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        rows=({"slice_index": 0, "Count": 2},),
        group_component=AllComponents.CHANNEL,
        group_key="1",
        variable_components=(AllComponents.SITE,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=({"site": "1", "channel": "1"},)
        ),
    )
    object_record = _measurement_record(
        "cell_measurements",
        axis_id="A01",
        subject=MeasurementSubject(
            MeasurementScope.OBJECT,
            "Cells",
            "object_number",
        ),
        rows=({"slice_index": 0, "object_number": 1, "Area": 4.0},),
        group_component=AllComponents.CHANNEL,
        group_key="2",
        variable_components=(AllComponents.SITE,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=({"site": "1", "channel": "2"},)
        ),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("image_measurements", MeasurementsArtifactType),
            ArtifactSpec.input("cell_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={"A01": (image_record, object_record)},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(
            frozenset((AllComponents.CHANNEL,))
        ),
    )

    bundle = export_to_spreadsheet(
        calculate_aggregate_means=True,
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Image.csv"]))) == (
        {
            "image_number": "1",
            "Count": "2",
            "Mean_Cells_Area": "4.0",
        },
    )
    assert tuple(csv.DictReader(io.StringIO(bundle["Cells.csv"]))) == (
        {"image_number": "1", "object_label": "1", "Area": "4.0"},
    )


def test_export_to_spreadsheet_nulls_metadata_that_differs_between_image_planes() -> (
    None
):
    records = tuple(
        _measurement_record(
            f"channel_{channel}_measurements",
            axis_id="A01",
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
            rows=({"slice_index": 0, f"Count_{channel}": int(channel)},),
            group_component=AllComponents.CHANNEL,
            group_key=channel,
            variable_components=(AllComponents.SITE,),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(
                    {
                        "site": "1",
                        "channel": channel,
                        ORIGINAL_SOURCE_METADATA_FIELD: {
                            "ChannelNumber": channel,
                            "Site": "1",
                        },
                    },
                )
            ),
        )
        for channel in ("1", "4")
    )
    batch = RuntimeArtifactBatch(
        input_specs=tuple(
            ArtifactSpec.input(record.key.name, MeasurementsArtifactType)
            for record in records
        ),
        records_by_axis={"A01": records},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(
            frozenset((AllComponents.CHANNEL,))
        ),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Image.csv"]))) == (
        {
            "image_number": "1",
            "Count_1": "1",
            "Count_4": "4",
            "Metadata_ChannelNumber": "",
            "Metadata_Site": "1",
        },
    )


def test_export_to_spreadsheet_merges_object_features_across_runtime_groups() -> None:
    provenance_by_channel = (
        SourceImageProvenancePlanes.from_components(
            component_metadata=({"site": "1", "channel": channel},)
        )
        for channel in ("1", "2")
    )
    records = tuple(
        _measurement_record(
            name,
            axis_id="A01",
            subject=MeasurementSubject(
                MeasurementScope.OBJECT,
                "Cells",
                "object_number",
            ),
            rows=(
                {
                    "slice_index": 0,
                    "object_number": 1,
                    feature_name: value,
                },
            ),
            group_component=AllComponents.CHANNEL,
            group_key=channel,
            variable_components=(AllComponents.SITE,),
            source_image_provenance_planes=provenance,
        )
        for name, channel, feature_name, value, provenance in zip(
            ("area_measurements", "perimeter_measurements"),
            ("1", "2"),
            ("Area", "Perimeter"),
            (4.0, 6.0),
            provenance_by_channel,
            strict=True,
        )
    )
    batch = RuntimeArtifactBatch(
        input_specs=tuple(
            ArtifactSpec.input(record.key.name, MeasurementsArtifactType)
            for record in records
        ),
        records_by_axis={"A01": records},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(
            frozenset((AllComponents.CHANNEL,))
        ),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Cells.csv"]))) == (
        {
            "image_number": "1",
            "object_label": "1",
            "Area": "4.0",
            "Perimeter": "6.0",
        },
    )


def test_export_to_spreadsheet_aggregates_mixed_producer_declared_rows() -> None:
    producer_rows = (
        {"slice_index": 0, "Count_Tissue": 2},
        {
            "slice_index": 0,
            "object_name": "Tissue",
            "object_label": 1,
            "Location_Center_X": 4.0,
        },
        {
            "slice_index": 0,
            "object_name": "Tissue",
            "object_label": 2,
            "Location_Center_X": 8.0,
        },
    )
    record = _measurement_record(
        "identify_primary_objects_measurements",
        axis_id="A01",
        subject=MeasurementSubject(
            MeasurementScope.IMAGE,
            MeasurementScope.IMAGE.value,
        ),
        rows=producer_rows,
    )
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input(
                "identify_primary_objects_measurements",
                MeasurementsArtifactType,
            ),
        ),
        records_by_axis={"A01": (record,)},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        calculate_aggregate_means=True,
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Image.csv"]))) == (
        {
            "image_number": "1",
            "Count_Tissue": "2",
            "Mean_Tissue_Location_Center_X": "6.0",
        },
    )


def test_export_to_spreadsheet_resolves_slice_indices_per_producer_table() -> None:
    records = tuple(
        _measurement_record(
            name,
            axis_id="A01",
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
            rows=({"slice_index": 0, feature: value},),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    component_metadata=({"well": "A01", "site": site, "channel": "1"},)
                )
            ),
        )
        for name, feature, value, site in (
            ("first_measurements", "First", 1, "1"),
            ("second_measurements", "Second", 2, "2"),
        )
    )
    batch = RuntimeArtifactBatch(
        input_specs=tuple(
            ArtifactSpec.input(name, MeasurementsArtifactType)
            for name in ("first_measurements", "second_measurements")
        ),
        records_by_axis={"A01": records},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Image.csv"]))) == (
        {"image_number": "1", "First": "1", "Second": ""},
        {"image_number": "2", "First": "", "Second": "2"},
    )


def test_export_to_spreadsheet_anchors_axisless_artifact_summary_to_stack() -> None:
    name = "neurite_outgrowth_summary"
    record = _measurement_record(
        name,
        axis_id="A01",
        subject=MeasurementSubject(MeasurementScope.ARTIFACT),
        rows=({"number_of_cells": 3, "total_outgrowth": 42.0},),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "4"},
                {"well": "A01", "site": "1", "channel": "1"},
            )
        ),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(ArtifactSpec.input(name, MeasurementsArtifactType),),
        records_by_axis={"A01": (record,)},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle[f"{name}.csv"]))) == (
        {
            "image_number": "1",
            "number_of_cells": "3",
            "total_outgrowth": "42.0",
        },
    )


def test_export_to_spreadsheet_rejects_axisless_artifact_without_source_identity() -> (
    None
):
    name = "unbound_artifact_summary"
    record = _measurement_record(
        name,
        axis_id="A01",
        subject=MeasurementSubject(MeasurementScope.ARTIFACT),
        rows=({"Count": 3},),
        source_image_provenance_planes=SourceImageProvenancePlanes(),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(ArtifactSpec.input(name, MeasurementsArtifactType),),
        records_by_axis={"A01": (record,)},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    with pytest.raises(
        ValueError,
        match="requires .*producer-declared source identity",
    ):
        export_to_spreadsheet(
            add_filename_prefix=False,
            artifact_batch=batch,
        )


def test_export_to_spreadsheet_rejects_axisless_image_rows_across_image_sets() -> None:
    name = "ambiguous_image_summary"
    record = _measurement_record(
        name,
        axis_id="A01",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        rows=({"Count": 3},),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "4"},
                {"well": "A01", "site": "1", "channel": "1"},
            )
        ),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(ArtifactSpec.input(name, MeasurementsArtifactType),),
        records_by_axis={"A01": (record,)},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    with pytest.raises(
        ValueError,
        match="cannot bind axisless rows.*image numbers \\(1, 2\\)",
    ):
        export_to_spreadsheet(
            add_filename_prefix=False,
            artifact_batch=batch,
        )


def test_export_to_spreadsheet_binds_payload_rows_to_exact_source_image_set() -> None:
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("object_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={
            "A01": (
                _measurement_record(
                    "object_measurements",
                    axis_id="A01",
                    subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
                    rows=(
                        {
                            "slice_index": 0,
                            "object_name": "Cells",
                            "object_label": 1,
                            "Area": 4.0,
                        },
                        {
                            "slice_index": 0,
                            "object_name": "Cells",
                            "object_label": 1,
                            "Children_Nuclei_Count": 1,
                        },
                    ),
                ),
            )
        },
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Cells.csv"]))) == (
        {
            "image_number": "1",
            "object_label": "1",
            "Area": "4.0",
            "Children_Nuclei_Count": "1",
        },
    )


def test_export_to_spreadsheet_aggregate_requires_declared_image_row() -> None:
    batch = RuntimeArtifactBatch(
        input_specs=(ArtifactSpec.input("cells", MeasurementsArtifactType),),
        records_by_axis={
            "A01": (
                _measurement_record(
                    "cells",
                    axis_id="A01",
                    subject=MeasurementSubject(
                        MeasurementScope.OBJECT,
                        "Cells",
                        "object_number",
                    ),
                    rows=({"slice_index": 0, "object_number": 1, "Area": 2.0},),
                ),
            )
        },
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    with pytest.raises(
        ValueError,
        match="producer-declared Image measurement row for image_number=1",
    ):
        export_to_spreadsheet(
            calculate_aggregate_means=True,
            add_filename_prefix=False,
            artifact_batch=batch,
        )


def test_export_to_spreadsheet_preserves_source_qualified_wide_features() -> None:
    source_values = (("BF_image", 82.5), ("Marker_image", 52.75))
    records = tuple(
        _measurement_record(
            "granularity_measurements",
            axis_id="A01",
            subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
            source_image_name=source_image_name,
            rows=(
                {
                    "slice_index": 0,
                    "object_id": 1,
                    f"Granularity_1_{source_image_name}": value,
                },
            ),
        )
        for source_image_name, value in source_values
    )
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("granularity_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={"A01": records},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    rows = tuple(csv.DictReader(io.StringIO(bundle["Cells.csv"])))
    assert rows == (
        {
            "image_number": "1",
            "object_label": "1",
            **{
                f"Granularity_1_{source_image_name}": str(value)
                for source_image_name, value in source_values
            },
        },
    )


def test_export_to_spreadsheet_keeps_crop_outputs_distinct_at_same_slice_index() -> (
    None
):
    crop_values = (("CropedWormsImage", 80, 100), ("CropBlue", 20, 50))
    records = tuple(
        _measurement_record(
            f"{source_image_name}_crop_measurements",
            axis_id="A01",
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
            source_image_name=source_image_name,
            rows=CropModule.prepare_measurement_record_rows(
                MeasurementSparseColumnarRows.from_rows(
                    (
                        {
                            "slice_index": 0,
                            "area_retained": area_retained,
                            "original_area": original_area,
                            "fraction_retained": area_retained / original_area,
                        },
                    ),
                    fields=(
                        FieldSpec("slice_index", int),
                        FieldSpec("area_retained", int),
                        FieldSpec("original_area", int),
                        FieldSpec("fraction_retained", float),
                    ),
                ),
                source_image_name=source_image_name,
            ),
        )
        for source_image_name, area_retained, original_area in crop_values
    )
    batch = RuntimeArtifactBatch(
        input_specs=tuple(
            ArtifactSpec.input(
                f"{source_image_name}_crop_measurements",
                MeasurementsArtifactType,
            )
            for source_image_name, _area_retained, _original_area in crop_values
        ),
        records_by_axis={"A01": records},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    rows = tuple(csv.DictReader(io.StringIO(bundle["Image.csv"])))
    assert rows == (
        {
            "image_number": "1",
            "Crop_AreaRetainedAfterCropping_CropedWormsImage": "80",
            "Crop_OriginalImageArea_CropedWormsImage": "100",
            "Crop_AreaRetainedAfterCropping_CropBlue": "20",
            "Crop_OriginalImageArea_CropBlue": "50",
        },
    )


def test_export_to_spreadsheet_preserves_declared_intensity_feature() -> None:
    feature_name = (
        "_".join(
            (
                *MeasureObjectIntensityModule.measurement_category_prefixes[0],
                MeasureObjectIntensityModule.MeasurementFeature.MEAN_INTENSITY.value,
            )
        )
        + "_DNA"
    )
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("intensity_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={
            "A01": (
                _measurement_record(
                    "intensity_measurements",
                    axis_id="A01",
                    subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
                    source_image_name="DNA",
                    rows=(
                        {
                            "slice_index": 0,
                            "object_id": 1,
                            feature_name: 12.5,
                        },
                    ),
                ),
            )
        },
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    rows = tuple(csv.DictReader(io.StringIO(bundle["Cells.csv"])))
    assert rows == (
        {
            "image_number": "1",
            "object_label": "1",
            feature_name: "12.5",
        },
    )


def test_export_to_spreadsheet_leaves_track_objects_features_unsuffixed() -> None:
    object_feature = TrackObjectsModule.measurement_feature_name("displacement")
    image_feature = TrackObjectsModule.measurement_feature_name(
        "new_object_count",
        "Cells",
    )
    records = (
        _measurement_record(
            "tracking_object_measurements",
            axis_id="A01",
            subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
            source_image_name="DNA",
            rows=(
                {
                    "slice_index": 0,
                    "object_id": 1,
                    object_feature: 4.25,
                },
            ),
        ),
        _measurement_record(
            "tracking_image_measurements",
            axis_id="A01",
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
            source_image_name="DNA",
            rows=({"slice_index": 0, image_feature: 3},),
        ),
    )
    batch = RuntimeArtifactBatch(
        input_specs=tuple(
            ArtifactSpec.input(name, MeasurementsArtifactType)
            for name in (
                "tracking_object_measurements",
                "tracking_image_measurements",
            )
        ),
        records_by_axis={"A01": records},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Cells.csv"]))) == (
        {
            "image_number": "1",
            "object_label": "1",
            object_feature: "4.25",
        },
    )
    assert tuple(csv.DictReader(io.StringIO(bundle["Image.csv"]))) == (
        {"image_number": "1", image_feature: "3"},
    )


def test_export_to_spreadsheet_leaves_worm_descriptor_fields_unsuffixed() -> None:
    descriptor_field = WormControlPointMeasurementField(
        WormControlPointAxis.COLUMN,
        1,
    ).name
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("worm_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={
            "A01": (
                _measurement_record(
                    "worm_measurements",
                    axis_id="A01",
                    subject=MeasurementSubject(MeasurementScope.OBJECT, "Worms"),
                    source_image_name="BinaryWorms",
                    rows=(
                        {
                            "slice_index": 0,
                            "object_number": 1,
                            descriptor_field: 17.5,
                        },
                    ),
                ),
            )
        },
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Worms.csv"]))) == (
        {
            "image_number": "1",
            "object_label": "1",
            descriptor_field: "17.5",
        },
    )


def test_export_to_spreadsheet_folds_descriptor_axes_before_coalescing() -> None:
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("texture_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={
            "A01": (
                _measurement_record(
                    "texture_measurements",
                    axis_id="A01",
                    subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
                    source_image_name="BF_image",
                    rows=tuple(
                        {
                            "slice_index": 0,
                            "object_id": 1,
                            "scale": 3,
                            "direction": direction,
                            "gray_levels": 256,
                            "axis": {
                                "slice_index": 0,
                                "scale": 3,
                                "direction": direction,
                                "gray_levels": 256,
                            },
                            "Texture_Contrast_BF_image": value,
                        }
                        for direction, value in ((0, 0.25), (1, 0.75))
                    ),
                ),
            )
        },
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    rows = tuple(csv.DictReader(io.StringIO(bundle["Cells.csv"])))
    assert len(rows) == 1
    assert rows[0]["Texture_Contrast_BF_image_3_00_256"] == "0.25"
    assert rows[0]["Texture_Contrast_BF_image_3_01_256"] == "0.75"
    assert "axis" not in rows[0]


def test_export_to_spreadsheet_folds_neighbor_scale_once() -> None:
    image_record = _measurement_record(
        "image_measurements",
        axis_id="A01",
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        rows=({"slice_index": 0, "Count_Cells": 1},),
    )
    neighbor_record = _measurement_record(
        "neighbor_measurements",
        axis_id="A01",
        subject=MeasurementSubject(MeasurementScope.OBJECT, "Cells", "object_id"),
        rows=(
            {
                "slice_index": 0,
                "object_id": 1,
                "scale": "expanded",
                "feature_name": "Neighbors_NumberOfNeighbors",
                "measurement_value": 2,
            },
        ),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("image_measurements", MeasurementsArtifactType),
            ArtifactSpec.input("neighbor_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={"A01": (image_record, neighbor_record)},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        calculate_aggregate_means=True,
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    assert tuple(csv.DictReader(io.StringIO(bundle["Cells.csv"]))) == (
        {
            "image_number": "1",
            "object_label": "1",
            "Neighbors_NumberOfNeighbors_expanded": "2",
        },
    )
    assert tuple(csv.DictReader(io.StringIO(bundle["Image.csv"]))) == (
        {
            "image_number": "1",
            "Count_Cells": "1",
            "Mean_Cells_Neighbors_NumberOfNeighbors_expanded": "2.0",
        },
    )


def test_export_to_spreadsheet_routes_row_owned_objects_and_normalizes_ids() -> None:
    batch = RuntimeArtifactBatch(
        input_specs=(
            ArtifactSpec.input("mixed_measurements", MeasurementsArtifactType),
        ),
        records_by_axis={
            "A01": (
                _measurement_record(
                    "mixed_measurements",
                    axis_id="A01",
                    subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
                    rows=(
                        {
                            "slice_index": 0,
                            "object_id": 1,
                            "object_name": "Cells",
                            "openhcs_object_row_identity": "row_ordinal",
                            "Area": 2.0,
                        },
                        {
                            "slice_index": 0,
                            "object_label": 1,
                            "object_name": "Cells",
                            "Perimeter": 3.0,
                        },
                    ),
                ),
            )
        },
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = export_to_spreadsheet(
        add_filename_prefix=False,
        artifact_batch=batch,
    )

    rows = tuple(csv.DictReader(io.StringIO(bundle["Cells.csv"])))
    assert rows == (
        {
            "image_number": "1",
            "object_label": "1",
            "Area": "2.0",
            "Perimeter": "3.0",
        },
    )


def _module(rows: tuple[tuple[str, str], ...]) -> ModuleBlock:
    records = [ModuleSetting(name, value) for name, value in rows]
    return ModuleBlock(
        name="ExportToSpreadsheet",
        module_num=7,
        setting_records=records,
    )


def _measurement_record(
    name: str,
    *,
    axis_id: str,
    subject: MeasurementSubject,
    rows: tuple[dict[str, object], ...] | ColumnarRows,
    source_image_name: str | None = None,
    source_image_provenance_planes: SourceImageProvenancePlanes | None = None,
    group_component: AllComponents | None = None,
    group_key: str | None = None,
    variable_components: tuple[AllComponents, ...] = (),
) -> StoredRuntimeValue:
    if not isinstance(rows, ColumnarRows):
        field_names = tuple(
            dict.fromkeys(field_name for row in rows for field_name in row)
        )
        rows = MeasurementSparseColumnarRows.from_rows(
            rows,
            fields=tuple(
                FieldSpec(
                    field_name,
                    _fixture_field_dtype(rows, field_name),
                )
                for field_name in field_names
            ),
        )
    if source_image_provenance_planes is None:
        slice_indices = tuple(
            dict.fromkeys(
                int(row["slice_index"])
                for row in rows.iter_row_mappings()
                if "slice_index" in row
            )
        )
        source_image_provenance_planes = SourceImageProvenancePlanes.from_components(
            component_metadata=tuple(
                {
                    **(
                        {group_component.value: group_key}
                        if group_component is not None and group_key is not None
                        else {}
                    ),
                    **(
                        {AllComponents.SITE.value: str(slice_index + 1)}
                        if group_component is not AllComponents.SITE
                        else {}
                    ),
                }
                for slice_index in slice_indices
            )
        )
    output_plan = ArtifactOutputPlan(
        name=name,
        path=f"/memory/{name}.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=(group_key,),
        group_component=group_component,
        variable_components=variable_components,
    )
    value = RuntimeValue.normalize(
        output_plan,
        MeasurementTable(
            name=name,
            rows=rows,
            subject=subject,
            source_image_name=source_image_name,
            source_image_provenance_planes=source_image_provenance_planes,
        ),
        axis_id=axis_id,
    )
    return StoredRuntimeValue(
        value,
        RuntimeArtifactLocation(path=output_plan.path, backend="memory"),
    )


def _fixture_field_dtype(
    rows: tuple[dict[str, object], ...],
    field_name: str,
) -> type[object]:
    field_types = tuple(
        dict.fromkeys(type(row[field_name]) for row in rows if field_name in row)
    )
    if len(field_types) != 1:
        raise TypeError(
            f"Fixture field {field_name!r} requires one exact scalar type, "
            f"got {field_types!r}."
        )
    return field_types[0]


def _relationship_record(name: str, *, axis_id: str) -> StoredRuntimeValue:
    output_plan = ArtifactOutputPlan(
        name=name,
        path=f"/memory/{name}.pkl",
        artifact_type=RelationshipsArtifactType,
    )
    value = RuntimeValue.normalize(
        output_plan,
        ObjectRelationship(
            name=name,
            declaration=ObjectRelationshipDeclaration(
                source=ArtifactSpec.output("Parents", ObjectLabelsArtifactType).ref(),
                target=ArtifactSpec.output("Children", ObjectLabelsArtifactType).ref(),
                relationship_type="related",
                source_role="parent",
                target_role="child",
                source_id_field="parent_number",
                target_id_field="child_number",
                producer_module_number=1,
                source_runtime_slice_offset=0,
                target_runtime_slice_offset=0,
            ),
            payload=DirectedObjectRelationshipPayload(
                source_ids=(1,), target_ids=(2,), slice_indices=(), slice_count=None
            ),
        ),
        axis_id=axis_id,
    )
    return StoredRuntimeValue(
        value,
        RuntimeArtifactLocation(path=output_plan.path, backend="memory"),
    )
