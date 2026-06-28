from pathlib import Path

import pytest

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_semantics import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectRelationship,
    RelationshipEndpoint,
    normalize_artifact_value,
)
from openhcs.interop.cellprofiler.analyst_export import (
    CPAImageChannelSpec,
    CPAPropertiesRenderer,
    CellProfilerAnalystExportRequest,
    CellProfilerAnalystProjectionBuilder,
    CellProfilerDatabaseExportSettings,
    CellProfilerExecutionExportContext,
    CellProfilerObjectTableMode,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.export_to_database import (
    ExportToDatabaseModule,
)


AXIS_ID = "A01_s1"


def test_cellprofiler_analyst_projection_uses_runtime_measurements_and_relationships():
    store = RuntimeValueStore()
    _record_measurements(
        store,
        axis_id=AXIS_ID,
        table=MeasurementTable(
            name="ImageMeasurements",
            rows=({"ImageNumber": 1, "Count_Nuclei": 2},),
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        ),
    )
    _record_measurements(
        store,
        axis_id=AXIS_ID,
        table=MeasurementTable(
            name="NucleiMeasurements",
            rows=(
                {"ImageNumber": 1, "ObjectNumber": 1, "AreaShape_Area": 24.0},
                {"ImageNumber": 1, "ObjectNumber": 2, "AreaShape_Area": 36.0},
            ),
            object_name="Nuclei",
            object_id_field="ObjectNumber",
        ),
    )
    _record_relationship(
        store,
        axis_id=AXIS_ID,
        relationship=ObjectRelationship(
            name="Cells_Nuclei_relationships",
            source=RelationshipEndpoint("Cells", role="parent", id_field="parent_id"),
            target=RelationshipEndpoint("Nuclei", role="child", id_field="child_id"),
            source_ids=(10, 10),
            target_ids=(1, 2),
            relationship_type="parent_child",
        ),
    )

    projection = CellProfilerAnalystProjectionBuilder().build(
        _request({AXIS_ID: store})
    )

    assert projection.image_table_name == "CPA_Per_Image"
    assert projection.image_rows[0].image_number == 1
    assert projection.image_rows[0].measurements["Count_Nuclei"] == 2
    assert len(projection.object_tables) == 1
    assert projection.object_tables[0].object_name == "Nuclei"
    assert projection.object_tables[0].table_name == "CPA_Per_Nuclei"
    assert [row["ObjectNumber"] for row in projection.object_tables[0].rows] == [1, 2]
    assert projection.relationship_tables[0].table_name == (
        "CPA_Cells_Nuclei_relationships"
    )
    assert [row["child_id"] for row in projection.relationship_tables[0].rows] == [1, 2]

    properties_file = CPAPropertiesRenderer().render(
        _request({AXIS_ID: store}),
        projection,
    )[0]

    assert properties_file.file_name == "DefaultDB_CPA_Nuclei.properties"
    assert properties_file.properties["db_type"] == "sqlite"
    assert properties_file.properties["image_table"] == "CPA_Per_Image"
    assert properties_file.properties["object_table"] == "CPA_Per_Nuclei"
    assert properties_file.properties["object_id"] == "Nuclei_Number_Object_Number"
    assert properties_file.properties["image_path_cols"] == "Image_PathName_DNA"
    assert properties_file.properties["image_file_cols"] == "Image_FileName_DNA"
    assert properties_file.properties["cell_x_loc"] == "Nuclei_Location_Center_X"
    assert properties_file.properties["channels_per_image"] == "1"


def test_cellprofiler_analyst_projection_requires_cpa_identity_columns():
    store = RuntimeValueStore()
    _record_measurements(
        store,
        axis_id=AXIS_ID,
        table=MeasurementTable(
            name="ImageMeasurements",
            rows=({"Count_Nuclei": 2},),
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        ),
    )

    with pytest.raises(ValueError, match="requires field 'ImageNumber'"):
        CellProfilerAnalystProjectionBuilder().build(_request({AXIS_ID: store}))


def test_export_to_database_declaration_parses_module_block_without_stringly_callers():
    module = ModuleBlock(
        name="ExportToDatabase",
        module_num=9,
        setting_records=[
            ModuleSetting(
                ExportToDatabaseModule.database_type_setting.canonical,
                "SQLite",
            ),
            ModuleSetting(
                ExportToDatabaseModule.sqlite_file_setting.canonical,
                "analysis.db",
            ),
            ModuleSetting(
                ExportToDatabaseModule.experiment_name_setting.canonical,
                "AdvancedSegmentation",
            ),
            ModuleSetting(
                ExportToDatabaseModule.want_table_prefix_setting.canonical,
                "Yes",
            ),
            ModuleSetting(
                ExportToDatabaseModule.table_prefix_setting.canonical,
                "Adv_",
            ),
            ModuleSetting(
                ExportToDatabaseModule.save_cpa_properties_setting.canonical,
                "Yes",
            ),
            ModuleSetting(
                ExportToDatabaseModule.objects_choice_setting.canonical,
                "Select...",
            ),
            ModuleSetting(
                ExportToDatabaseModule.objects_list_setting.canonical,
                "Nuclei, Cells",
            ),
            ModuleSetting(
                ExportToDatabaseModule.relationship_table_setting.canonical,
                "No",
            ),
            ModuleSetting(
                ExportToDatabaseModule.object_table_mode_setting.canonical,
                "One table per object type",
            ),
        ],
    )

    settings = ExportToDatabaseModule.database_export_settings(module)

    assert settings.sqlite_file == "analysis.db"
    assert settings.experiment_name == "AdvancedSegmentation"
    assert settings.table_prefix == "Adv_"
    assert settings.object_table_mode is CellProfilerObjectTableMode.PER_OBJECT
    assert settings.selected_objects == ("Nuclei", "Cells")
    assert settings.wants_properties_file is True
    assert settings.wants_relationship_tables is False


def _request(
    runtime_stores_by_axis: dict[str, RuntimeValueStore],
) -> CellProfilerAnalystExportRequest:
    return CellProfilerAnalystExportRequest(
        settings=CellProfilerDatabaseExportSettings(
            database_type="sqlite",
            sqlite_file="DefaultDB.db",
            experiment_name="Experiment",
            table_prefix="CPA_",
            object_table_mode=CellProfilerObjectTableMode.PER_OBJECT,
            selected_objects=None,
            wants_properties_file=True,
            wants_relationship_tables=True,
        ),
        context=CellProfilerExecutionExportContext(
            prepared=object(),
            execution=object(),
            runtime_stores_by_axis=runtime_stores_by_axis,
            output_roots=(Path("/tmp/openhcs-output"),),
            source_workspace_root=Path("/tmp/openhcs-source"),
            export_root=Path("/tmp/openhcs-cpa"),
        ),
        image_channels=(
            CPAImageChannelSpec(
                alias="DNA",
                image_name="DNA",
                channel_color="blue",
            ),
        ),
    )


def _record_measurements(
    store: RuntimeValueStore,
    *,
    axis_id: str,
    table: MeasurementTable,
) -> None:
    output_plan = ArtifactOutputPlan(
        name=table.name,
        path=f"/memory/{table.name}.pkl",
        kind=ArtifactKind.MEASUREMENTS,
    )
    store.record(
        normalize_artifact_value(output_plan, table, axis_id=axis_id),
        path=output_plan.path,
        backend="memory",
    )


def _record_relationship(
    store: RuntimeValueStore,
    *,
    axis_id: str,
    relationship: ObjectRelationship,
) -> None:
    output_plan = ArtifactOutputPlan(
        name=relationship.name,
        path=f"/memory/{relationship.name}.pkl",
        kind=ArtifactKind.RELATIONSHIPS,
    )
    store.record(
        normalize_artifact_value(output_plan, relationship, axis_id=axis_id),
        path=output_plan.path,
        backend="memory",
    )
