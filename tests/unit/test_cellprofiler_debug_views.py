from openhcs.core.artifacts import (
    ArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.debug import DebugArtifactRef, DebugCursor, DebugSnapshot
from openhcs.core.debug_views import DebugViewModel, DebugViewSectionKind
from openhcs.interop.cellprofiler.debug_views import (
    CellProfilerDebugView,
    DefaultCellProfilerDebugView,
)


class CellProfilerDebugViewFixture:
    SNAPSHOT_ID = "snap"
    MEASURE_IMAGE_INTENSITY = "MeasureImageIntensity"

    @staticmethod
    def cursor(
        *,
        step_index: int,
        invocation_name: str,
    ) -> DebugCursor:
        return DebugCursor(
            step_index=step_index,
            step_scope_id="scope",
            group_key="default",
            invocation_key=f"default:0:{invocation_name}",
        )

    @classmethod
    def artifact_ref(
        cls,
        *,
        kind: ArtifactType,
        name: str,
        cursor: DebugCursor,
        extension: str,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
    ) -> DebugArtifactRef:
        return DebugArtifactRef(
            kind=kind,
            name=name,
            cursor=cursor,
            storage_ref=f"debug/{cls.SNAPSHOT_ID}/{name}.{extension}",
            shape=shape,
            dtype=dtype,
        )


def section_by_kind(
    view: DebugViewModel,
    kind: DebugViewSectionKind,
):
    return next(section for section in view.sections if section.kind is kind)


def table_row_mapping(section):
    assert section.table is not None
    return dict(zip(section.table.columns, section.table.rows[0], strict=True))


def test_cellprofiler_debug_view_uses_generic_snapshot_sections():
    cursor = CellProfilerDebugViewFixture.cursor(
        step_index=0,
        invocation_name=CellProfilerDebugViewFixture.MEASURE_IMAGE_INTENSITY,
    )
    measurements_ref = CellProfilerDebugViewFixture.artifact_ref(
        kind=MeasurementsArtifactType,
        name="ImageMeasurements",
        cursor=cursor,
        extension="csv",
        dtype="csv",
    )
    snapshot = DebugSnapshot(
        snapshot_id=CellProfilerDebugViewFixture.SNAPSHOT_ID,
        cursor=cursor,
        step_name="measure",
        callable_name=CellProfilerDebugViewFixture.MEASURE_IMAGE_INTENSITY,
        axis_id="A01",
        source_paths=("A01_s1_w1.tif",),
        output_artifact_refs=(measurements_ref,),
        measurement_refs=(measurements_ref,),
        timing_seconds=0.125,
    )

    renderer = CellProfilerDebugView.for_module(
        CellProfilerDebugViewFixture.MEASURE_IMAGE_INTENSITY
    )
    view = renderer.build_view_model(snapshot)

    assert isinstance(renderer, DefaultCellProfilerDebugView)
    assert view.title == CellProfilerDebugViewFixture.MEASURE_IMAGE_INTENSITY
    assert tuple(section.kind for section in view.sections) == (
        DebugViewSectionKind.SUMMARY,
        DebugViewSectionKind.SOURCES,
        DebugViewSectionKind.OUTPUT_ARTIFACTS,
        DebugViewSectionKind.MEASUREMENTS,
        DebugViewSectionKind.TIMING,
    )
    assert "axis: A01" in section_by_kind(view, DebugViewSectionKind.SUMMARY).text
    assert section_by_kind(view, DebugViewSectionKind.SOURCES).text == "A01_s1_w1.tif"
    output_row = table_row_mapping(
        section_by_kind(view, DebugViewSectionKind.OUTPUT_ARTIFACTS)
    )
    assert output_row["kind"] == MeasurementsArtifactType.value
    assert output_row["name"] == "ImageMeasurements"
    assert output_row["storage_ref"] == "debug/snap/ImageMeasurements.csv"
    assert output_row["dtype"] == "csv"
    measurement_row = table_row_mapping(
        section_by_kind(view, DebugViewSectionKind.MEASUREMENTS)
    )
    assert measurement_row == output_row
    assert section_by_kind(view, DebugViewSectionKind.TIMING).text == "0.125000s"


def test_snapshot_sections_are_derived_from_present_debug_payloads():
    cursor = CellProfilerDebugViewFixture.cursor(
        step_index=1,
        invocation_name="RelateObjects",
    )
    input_ref = CellProfilerDebugViewFixture.artifact_ref(
        kind=ObjectLabelsArtifactType,
        name="Nuclei",
        cursor=cursor,
        extension="zarr",
        shape=(1, 64, 64),
        dtype="uint16",
    )
    relationship_ref = CellProfilerDebugViewFixture.artifact_ref(
        kind=RelationshipsArtifactType,
        name="ParentChild",
        cursor=cursor,
        extension="csv",
    )
    snapshot = DebugSnapshot(
        snapshot_id=CellProfilerDebugViewFixture.SNAPSHOT_ID,
        cursor=cursor,
        step_name="relate",
        callable_name="RelateObjects",
        input_artifact_refs=(input_ref,),
        relationship_refs=(relationship_ref,),
        exception="boom",
    )

    view = CellProfilerDebugView.for_module("RelateObjects").build_view_model(snapshot)

    assert tuple(section.kind for section in view.sections) == (
        DebugViewSectionKind.SUMMARY,
        DebugViewSectionKind.INPUT_ARTIFACTS,
        DebugViewSectionKind.RELATIONSHIPS,
        DebugViewSectionKind.ERROR,
    )
    input_row = table_row_mapping(
        section_by_kind(view, DebugViewSectionKind.INPUT_ARTIFACTS)
    )
    assert input_row["kind"] == ObjectLabelsArtifactType.value
    assert input_row["name"] == "Nuclei"
    assert input_row["shape"] == "1, 64, 64"
    relationship_row = table_row_mapping(
        section_by_kind(view, DebugViewSectionKind.RELATIONSHIPS)
    )
    assert relationship_row["kind"] == RelationshipsArtifactType.value
    assert relationship_row["name"] == "ParentChild"
    assert section_by_kind(view, DebugViewSectionKind.ERROR).text == "boom"
