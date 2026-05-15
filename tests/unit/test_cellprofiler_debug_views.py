from openhcs.core.artifacts import ArtifactKind
from openhcs.core.debug import DebugArtifactRef, DebugCursor, DebugSnapshot
from openhcs.core.debug_views import DebugViewModel
from openhcs.interop.cellprofiler.debug_views import (
    CellProfilerDebugView,
    DefaultCellProfilerDebugView,
    IdentifyPrimaryObjectsDebugView,
    MeasureImageIntensityDebugView,
    RelateObjectsDebugView,
    TableDrivenCellProfilerDebugView,
)


class CellProfilerDebugViewFixture:
    """Nominal authority for CellProfiler debug-view test products."""

    SNAPSHOT_ID = "snap"
    MEASURE_IMAGE_INTENSITY = "MeasureImageIntensity"
    IDENTIFY_PRIMARY_OBJECTS = "IdentifyPrimaryObjects"

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
        kind: ArtifactKind,
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


def test_cellprofiler_debug_view_registry_returns_default_renderer():
    cursor = CellProfilerDebugViewFixture.cursor(
        step_index=0,
        invocation_name=CellProfilerDebugViewFixture.MEASURE_IMAGE_INTENSITY,
    )
    measurements_ref = CellProfilerDebugViewFixture.artifact_ref(
        kind=ArtifactKind.MEASUREMENTS,
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

    assert isinstance(renderer, MeasureImageIntensityDebugView)
    assert isinstance(view, DebugViewModel)
    assert view.title == "MeasureImageIntensity"
    assert view.sections[0].table is not None
    assert ("axis", "A01") in view.sections[0].table.rows
    assert view.sections[1].table is not None
    assert view.sections[1].table.rows == (
        (
            "ImageMeasurements",
            "measurements",
            "debug/snap/ImageMeasurements.csv",
            "",
            "csv",
        ),
    )


def test_identify_primary_objects_debug_view_lists_output_artifacts():
    cursor = CellProfilerDebugViewFixture.cursor(
        step_index=1,
        invocation_name=CellProfilerDebugViewFixture.IDENTIFY_PRIMARY_OBJECTS,
    )
    snapshot = DebugSnapshot(
        snapshot_id=CellProfilerDebugViewFixture.SNAPSHOT_ID,
        cursor=cursor,
        step_name="identify",
        callable_name=CellProfilerDebugViewFixture.IDENTIFY_PRIMARY_OBJECTS,
        output_artifact_refs=(
            CellProfilerDebugViewFixture.artifact_ref(
                kind=ArtifactKind.OBJECT_LABELS,
                name="Nuclei",
                cursor=cursor,
                extension="zarr",
                shape=(1, 64, 64),
                dtype="uint16",
            ),
        ),
        timing_seconds=0.25,
    )

    renderer = CellProfilerDebugView.for_module(
        CellProfilerDebugViewFixture.IDENTIFY_PRIMARY_OBJECTS
    )
    view = renderer.build_view_model(snapshot)

    assert isinstance(renderer, IdentifyPrimaryObjectsDebugView)
    assert view.title == CellProfilerDebugViewFixture.IDENTIFY_PRIMARY_OBJECTS
    assert view.sections[0].table is not None
    assert view.sections[0].table.rows == (
        ("Nuclei", "debug/snap/Nuclei.zarr", "1x64x64", "uint16"),
    )
    assert view.sections[3].text == "0.250000s"


def test_measurement_debug_view_prioritizes_measurement_outputs():
    cursor = CellProfilerDebugViewFixture.cursor(
        step_index=2,
        invocation_name=CellProfilerDebugViewFixture.MEASURE_IMAGE_INTENSITY,
    )
    measurements_ref = CellProfilerDebugViewFixture.artifact_ref(
        kind=ArtifactKind.MEASUREMENTS,
        name="Intensity",
        cursor=cursor,
        extension="csv",
    )
    snapshot = DebugSnapshot(
        snapshot_id=CellProfilerDebugViewFixture.SNAPSHOT_ID,
        cursor=cursor,
        step_name="measure",
        callable_name=CellProfilerDebugViewFixture.MEASURE_IMAGE_INTENSITY,
        measurement_refs=(measurements_ref,),
    )

    renderer = CellProfilerDebugView.for_module(
        CellProfilerDebugViewFixture.MEASURE_IMAGE_INTENSITY
    )
    view = renderer.build_view_model(snapshot)

    assert isinstance(renderer, MeasureImageIntensityDebugView)
    assert view.sections[1].title == "Measurement Outputs"
    assert view.sections[1].table is not None
    assert view.sections[1].table.rows[0][0] == "Intensity"


def test_relationship_debug_view_prioritizes_relationship_outputs():
    cursor = CellProfilerDebugViewFixture.cursor(
        step_index=3,
        invocation_name="RelateObjects",
    )
    relationship_ref = CellProfilerDebugViewFixture.artifact_ref(
        kind=ArtifactKind.RELATIONSHIPS,
        name="ParentChild",
        cursor=cursor,
        extension="csv",
    )
    snapshot = DebugSnapshot(
        snapshot_id=CellProfilerDebugViewFixture.SNAPSHOT_ID,
        cursor=cursor,
        step_name="relate",
        callable_name="RelateObjects",
        relationship_refs=(relationship_ref,),
    )

    renderer = CellProfilerDebugView.for_module("RelateObjects")
    view = renderer.build_view_model(snapshot)

    assert isinstance(renderer, RelateObjectsDebugView)
    assert view.sections[1].title == "Relationship Outputs"
    assert view.sections[1].table is not None
    assert view.sections[1].table.rows[0][0] == "ParentChild"


def test_table_driven_debug_view_covers_major_cellprofiler_module_families():
    cursor = CellProfilerDebugViewFixture.cursor(
        step_index=4,
        invocation_name="MeasureObjectIntensity",
    )
    measurement_ref = CellProfilerDebugViewFixture.artifact_ref(
        kind=ArtifactKind.MEASUREMENTS,
        name="ObjectIntensity",
        cursor=cursor,
        extension="csv",
    )
    snapshot = DebugSnapshot(
        snapshot_id=CellProfilerDebugViewFixture.SNAPSHOT_ID,
        cursor=cursor,
        step_name="measure_objects",
        callable_name="MeasureObjectIntensity",
        measurement_refs=(measurement_ref,),
    )

    renderer = CellProfilerDebugView.for_module("MeasureObjectIntensity")
    view = renderer.build_view_model(snapshot)

    assert isinstance(renderer, TableDrivenCellProfilerDebugView)
    assert view.title == "MeasureObjectIntensity"
    assert view.sections[1].title == "Measurement Outputs"


def test_display_export_debug_view_includes_artifact_overview():
    cursor = CellProfilerDebugViewFixture.cursor(
        step_index=5,
        invocation_name="SaveImages",
    )
    image_ref = CellProfilerDebugViewFixture.artifact_ref(
        kind=ArtifactKind.IMAGE,
        name="SavedImage",
        cursor=cursor,
        extension="tif",
    )
    snapshot = DebugSnapshot(
        snapshot_id=CellProfilerDebugViewFixture.SNAPSHOT_ID,
        cursor=cursor,
        step_name="save",
        callable_name="SaveImages",
        input_artifact_refs=(image_ref,),
    )

    renderer = CellProfilerDebugView.for_module("SaveImages")
    view = renderer.build_view_model(snapshot)

    assert isinstance(renderer, TableDrivenCellProfilerDebugView)
    assert view.sections[1].title == "Artifact Overview"
    assert ("inputs", "1") in view.sections[1].table.rows
