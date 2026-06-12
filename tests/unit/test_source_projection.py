from __future__ import annotations

import pytest

from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourcePixelRef,
    SourcePlaneProjection,
    SourceProjectionSet,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


def test_source_projection_serializes_canonical_virtual_filename() -> None:
    projection_set = SourceProjectionSet(
        (
            SourcePlaneProjection(
                address=OpenHCSPlaneAddress(
                    well="A01",
                    site="1",
                    channel="2",
                    z_index="3",
                    timepoint="4",
                ),
                ref=SourcePixelRef(
                    backend="bioformats",
                    source_path="stack.ome.tif",
                    reader="bioformats",
                    series_index=5,
                    plane_index=6,
                    source_channel=2,
                    source_z_index=3,
                    source_timepoint=4,
                ),
                component_labels={
                    "channel": "DAPI",
                    "well": "A01",
                    "site": "Site 1",
                    "z_index": "Z3",
                    "timepoint": "T4",
                },
            ),
        )
    )

    metadata = projection_set.metadata_dict(
        parser=SourceSchemaFilenameParser(),
        microscope_handler_name="bioformats",
        source_filename_parser_name="SourceSchemaFilenameParser",
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )

    assert metadata["image_files"] == ["A01_s001_w2_z003_t004.tif"]
    assert metadata["channels"] == {"2": "DAPI"}
    assert metadata["z_indexes"] == {"3": "Z3"}
    assert metadata["workspace_mapping"]["A01_s001_w2_z003_t004.tif"] == {
        "backend": "bioformats",
        "reader": "bioformats",
        "source_path": "stack.ome.tif",
        "series_index": 5,
        "plane_index": 6,
        "c": 2,
        "z": 3,
        "t": 4,
    }
    assert metadata["source_projection"][0]["address"] == {
        "well": "A01",
        "site": "1",
        "channel": "2",
        "z_index": "3",
        "timepoint": "4",
    }


def test_source_projection_rejects_metadata_component_conflict() -> None:
    projection_set = SourceProjectionSet(
        (
            SourcePlaneProjection(
                address=OpenHCSPlaneAddress(
                    well="A01",
                    site="1",
                    channel="2",
                    z_index="3",
                    timepoint="4",
                ),
                ref=SourcePixelRef(
                    backend="disk",
                    source_path="image.tif",
                ),
                source_metadata={"z_index": "99"},
            ),
        )
    )

    with pytest.raises(ValueError, match="conflicts with canonical z_index"):
        projection_set.metadata_dict(
            parser=SourceSchemaFilenameParser(),
            microscope_handler_name="openhcs",
            source_filename_parser_name="SourceSchemaFilenameParser",
            grid_dimensions=[1, 1],
            pixel_size=1.0,
        )


def test_source_projection_rejects_duplicate_addresses() -> None:
    address = OpenHCSPlaneAddress(
        well="A01",
        site="1",
        channel="1",
        z_index="1",
        timepoint="1",
    )

    with pytest.raises(ValueError, match="Duplicate source projection address"):
        SourceProjectionSet(
            (
                SourcePlaneProjection(
                    address=address,
                    ref=SourcePixelRef("disk", "a.tif"),
                ),
                SourcePlaneProjection(
                    address=address,
                    ref=SourcePixelRef("disk", "b.tif"),
                ),
            )
        )
